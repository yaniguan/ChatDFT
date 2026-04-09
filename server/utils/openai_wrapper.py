# server/utils/openai_wrapper.py
# -*- coding: utf-8 -*-
"""
LLM call wrapper with:
- Multi-provider routing via ``server.utils.llm_providers.LLMRouter``
  (OpenAI, vLLM, pluggable). Each provider has its own client, concurrency
  cap, and retry budget. The router tries the primary provider first and
  falls back to a secondary on failure, so vLLM being down never breaks
  the dev workflow.
- Token usage extraction, JSON-mode schema-validity tracking.
- Automatic AgentLog instrumentation for the monitoring dashboard.

Every call written through ``chatgpt_call`` records one ``agent_log`` row
with the ``provider:model`` tuple encoded in the ``model`` column
(e.g. ``"vllm_local:Qwen/Qwen2.5-7B-Instruct"``). The dashboard parses that
to produce a per-provider breakdown without a schema migration.

The signature of ``chatgpt_call`` is preserved exactly — existing callers
(intent, hypothesis, plan, analyze, knowledge, qa, records, post_analysis)
get vLLM routing for free once they are listed in ``server/llm.yaml``.

``call_type`` values
--------------------
* ``llm_json``         — JSON mode, response parsed as valid JSON
* ``llm_json_invalid`` — JSON mode, response did not parse (schema failure)
* ``llm_text``         — text mode
* ``llm_error``        — provider-level failure (success=False)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router is the single source of truth for which provider handles what.
# Kept as a lazy import so tests can monkey-patch before first call.
# ---------------------------------------------------------------------------
from server.utils.llm_providers import LLMCallResult, get_llm_router

# Backwards-compat re-exports kept for any call sites that imported them.
try:
    from openai import APIStatusError, APITimeoutError, RateLimitError  # noqa: F401
except ImportError:  # pragma: no cover
    RateLimitError = Exception       # type: ignore
    APITimeoutError = Exception      # type: ignore
    APIStatusError = Exception       # type: ignore


# ---------------------------------------------------------------------------
# Caller-module inference (for automatic AgentLog tagging)
# ---------------------------------------------------------------------------

def _infer_agent_name() -> str:
    """
    Walk up the call stack and return the first frame whose module is an
    agent file (``*_agent.py`` under ``server/chat`` or ``server/execution``).
    Falls back to the immediate caller's module basename.
    """
    try:
        frame = inspect.currentframe()
        # Skip this function and the wrapper that called it.
        for _ in range(2):
            if frame is None:
                break
            frame = frame.f_back
        while frame is not None:
            mod = inspect.getmodule(frame)
            if mod and getattr(mod, "__file__", None):
                base = os.path.basename(mod.__file__).removesuffix(".py")
                if base.endswith("_agent") or base in {"rag_utils", "knowledge_agent"}:
                    return base
            frame = frame.f_back
    except Exception:
        pass
    return "unknown_agent"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def chatgpt_call(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    json_mode: bool = True,
    retries: int = 3,
    *,
    agent_name: Optional[str] = None,
    session_id: Optional[int] = None,
    log_to_agentlog: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Async LLM call routed through the configured provider.

    Parameters
    ----------
    messages, model, temperature, max_tokens, json_mode, retries :
        Standard OpenAI chat.completions arguments. ``model`` is treated as a
        *hint* — ``LLMRouter`` may substitute the provider's own default model
        if the hint belongs to a different backend (e.g. you pass ``gpt-4o``
        but the routed provider is vLLM serving Qwen).
    agent_name :
        Explicit agent identifier for routing + logging. If ``None``, inferred
        from the caller module.
    session_id :
        Optional chat session id so AgentLog rows can be correlated with
        workflow tasks.
    log_to_agentlog :
        Write an AgentLog row on completion. Default True.

    Returns
    -------
    The openai-style response dict. On total failure returns
    ``{"error": "...", "choices": []}``.
    """
    resolved_agent = agent_name or _infer_agent_name()
    input_preview = ""
    try:
        input_preview = json.dumps(messages, ensure_ascii=False)[:500]
    except Exception:
        pass

    router = get_llm_router()

    t0 = time.time()
    try:
        result: LLMCallResult = await router.chat_completion(
            agent_name=resolved_agent,
            messages=messages,
            model_hint=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            retries=retries,
            **kwargs,
        )
    except Exception as e:  # pragma: no cover — router is defensive
        log.exception("router.chat_completion crashed: %s", e)
        latency_ms = int((time.time() - t0) * 1000)
        _async_log_safe(
            resolved_agent, "llm_error", model, 0, 0, latency_ms,
            success=False, error_msg=str(e),
            input_preview=input_preview, output_preview="",
            session_id=session_id, log_to_agentlog=log_to_agentlog,
        )
        return {"error": str(e), "choices": []}

    latency_ms = result.latency_ms or int((time.time() - t0) * 1000)

    # ── Determine call_type / schema validity ──────────────────────────────
    call_type = "llm_json" if json_mode else "llm_text"
    schema_valid = True
    output_text = result.text()

    if not result.success:
        call_type = "llm_error"
        schema_valid = False
        success = False
        error_msg = result.error or "unknown error"
    else:
        success = True
        error_msg = None
        if json_mode and output_text:
            try:
                json.loads(output_text)
            except Exception:
                schema_valid = False
                call_type = "llm_json_invalid"
                error_msg = "schema_invalid"
        elif json_mode and not output_text:
            # Empty JSON body is always invalid in JSON mode
            schema_valid = False
            call_type = "llm_json_invalid"
            error_msg = "empty_response"

        result.raw.setdefault("_chatdft_meta", {}).update({
            "agent_name": resolved_agent,
            "provider": result.provider,
            "model": result.model,
            "latency_ms": latency_ms,
            "schema_valid": schema_valid,
            "retries_used": result.retries_used,
        })

    tokens = result.tokens() if result.success else {}
    tagged_model = f"{result.provider}:{result.model}" if result.provider and result.provider != "none" else (result.model or model)

    _async_log_safe(
        resolved_agent, call_type, tagged_model,
        tokens.get("prompt_tokens", 0),
        tokens.get("completion_tokens", 0),
        latency_ms,
        success=success and schema_valid,
        error_msg=error_msg,
        input_preview=input_preview,
        output_preview=output_text[:500],
        session_id=session_id,
        log_to_agentlog=log_to_agentlog,
        retries_used=result.retries_used,
    )

    if not result.success:
        return {"error": result.error, "choices": []}
    return result.raw


def _async_log_safe(
    agent_name: str,
    call_type: str,
    model: str,
    in_tok: int,
    out_tok: int,
    latency_ms: int,
    *,
    success: bool,
    error_msg: Optional[str],
    input_preview: str,
    output_preview: str,
    session_id: Optional[int],
    log_to_agentlog: bool,
    retries_used: int = 0,
) -> None:
    """Fire-and-forget AgentLog write. Never raises."""
    if not log_to_agentlog:
        return
    try:
        from server.utils.rag_utils import log_agent_call
        asyncio.create_task(log_agent_call(
            agent_name=agent_name,
            call_type=call_type,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            success=success,
            error_msg=error_msg,
            input_preview=input_preview,
            output_preview=output_preview,
            session_id=session_id,
            full_input={"retries_used": retries_used} if retries_used else None,
        ))
    except Exception as _e:
        log.debug("AgentLog write skipped: %s", _e)


# ---------------------------------------------------------------------------
# Helpers kept for backwards compat
# ---------------------------------------------------------------------------

def extract_text(response: Dict[str, Any]) -> str:
    """Extract the assistant message text from a chatgpt_call response."""
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def extract_usage(response: Dict[str, Any]) -> Dict[str, int]:
    """Extract token usage from a chatgpt_call response."""
    usage = response.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


async def embed_texts(
    texts: List[str],
    model: str = "text-embedding-3-small",
    *,
    agent_name: Optional[str] = None,
) -> List[List[float]]:
    """
    Embed a list of strings via the configured router.

    Routes through the same provider as the caller agent (from
    ``llm.yaml`` routing), so local vLLM embedding deployments can be
    plumbed in without touching agent code. Raises on failure.
    """
    router = get_llm_router()
    resolved = agent_name or _infer_agent_name()
    return await router.embed(agent_name=resolved, texts=texts, model_hint=model)
