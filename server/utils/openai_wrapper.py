# server/utils/openai_wrapper.py
# -*- coding: utf-8 -*-
"""
Async OpenAI wrapper with:
- Exponential-backoff retry (3 attempts)
- Token usage extraction
- JSON mode enforcement (with parse-validity tracking)
- Embedding endpoint
- Automatic AgentLog instrumentation for the monitoring dashboard

Every call produced by ``chatgpt_call`` writes one row to the ``agent_log``
table via ``server.utils.rag_utils.log_agent_call``. The ``agent_name`` can be
provided explicitly; otherwise it is inferred from the caller module
(e.g. ``server/chat/intent_agent.py`` → ``intent_agent``). This gives us
per-agent latency / token / success / schema-valid data for the dashboard
without having to modify every agent file.

``call_type`` encodes the outcome so the dashboard can compute
schema-valid rate for JSON-producing agents:

* ``llm_json``         — JSON mode, response parsed as valid JSON
* ``llm_json_invalid`` — JSON mode, response did not parse (schema failure)
* ``llm_text``         — text mode
* ``llm_error``        — API error (success=False)
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

try:
    from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIStatusError
    _client = AsyncOpenAI()
    _OK = True
except ImportError:
    _client = None  # type: ignore
    _OK = False
    RateLimitError = Exception
    APITimeoutError = Exception
    APIStatusError = Exception


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
        # Skip this function and ``chatgpt_call``
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
    **kwargs,
) -> Dict[str, Any]:
    """
    Async LLM call with retry and automatic AgentLog instrumentation.

    Parameters
    ----------
    messages, model, temperature, max_tokens, json_mode, retries :
        Standard OpenAI chat.completions arguments.
    agent_name :
        Explicit agent identifier for the monitoring dashboard. If ``None``,
        inferred from the caller module (see ``_infer_agent_name``).
    session_id :
        Optional chat session id so AgentLog rows can be correlated with
        workflow tasks.
    log_to_agentlog :
        Write an AgentLog row on completion. Default True; set False for
        internal/test calls.

    Returns
    -------
    The full response dict (``choices``, ``usage``, ``model`` ...).  On
    complete failure returns ``{"error": "...", "choices": []}``.
    """
    if not _OK or _client is None:
        return {"error": "OpenAI client not available", "choices": []}

    response_format = {"type": "json_object"} if json_mode else {"type": "text"}

    # Resolve caller for logging
    resolved_agent = agent_name or _infer_agent_name()
    t0 = time.time()
    input_preview = ""
    try:
        input_preview = json.dumps(messages, ensure_ascii=False)[:500]
    except Exception:
        pass

    last_err: Optional[Exception] = None
    response_dict: Optional[Dict[str, Any]] = None
    retries_used = 0

    for attempt in range(retries):
        try:
            resp = await _client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **kwargs,
            )
            response_dict = resp.model_dump() if hasattr(resp, "model_dump") else resp.to_dict()
            break

        except RateLimitError as e:
            wait = 2 ** (attempt + 1)
            log.warning("Rate limit (attempt %d/%d), retrying in %ds", attempt + 1, retries, wait)
            await asyncio.sleep(wait)
            last_err = e
            retries_used = attempt + 1

        except APITimeoutError as e:
            wait = 2 ** attempt
            log.warning("Timeout (attempt %d/%d), retrying in %ds", attempt + 1, retries, wait)
            await asyncio.sleep(wait)
            last_err = e
            retries_used = attempt + 1

        except Exception as e:
            log.error("chatgpt_call failed: %s", e, exc_info=True)
            last_err = e
            retries_used = attempt + 1
            break   # non-retryable

    latency_ms = int((time.time() - t0) * 1000)

    # ── Determine call_type, schema validity, success ──────────────────────
    call_type = "llm_json" if json_mode else "llm_text"
    schema_valid = True
    output_text = ""

    if response_dict is None:
        # API failure
        call_type = "llm_error"
        schema_valid = False
        success = False
        error_msg = str(last_err) if last_err else "unknown error"
    else:
        success = True
        error_msg = None
        try:
            output_text = (
                response_dict.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "") or ""
            )
        except Exception:
            output_text = ""

        if json_mode:
            try:
                json.loads(output_text)
                schema_valid = True
            except Exception:
                schema_valid = False
                call_type = "llm_json_invalid"
                # Keep success=True at the transport level but flag schema failure
                error_msg = "schema_invalid"

        # Build the return dict
        response_dict.setdefault("_chatdft_meta", {}).update({
            "agent_name": resolved_agent,
            "latency_ms": latency_ms,
            "schema_valid": schema_valid,
            "retries_used": retries_used,
        })

    # ── Fire-and-forget AgentLog write ─────────────────────────────────────
    if log_to_agentlog:
        usage = (response_dict or {}).get("usage") or {}
        try:
            from server.utils.rag_utils import log_agent_call
            asyncio.create_task(log_agent_call(
                agent_name=resolved_agent,
                call_type=call_type,
                model=model,
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                latency_ms=latency_ms,
                success=success and schema_valid,  # schema failure counts as unsuccessful
                error_msg=error_msg,
                input_preview=input_preview,
                output_preview=output_text[:500],
                session_id=session_id,
                full_input={"retries_used": retries_used} if retries_used else None,
            ))
        except Exception as _e:
            log.debug("AgentLog write skipped: %s", _e)

    if response_dict is None:
        return {"error": str(last_err), "choices": []}
    return response_dict


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
) -> List[List[float]]:
    """
    Embed a list of strings. Returns a list of float vectors.
    Raises on failure (caller should handle).
    """
    if not _OK or _client is None:
        raise RuntimeError("OpenAI client not available")

    resp = await _client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in resp.data]
