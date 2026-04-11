# server/utils/llm_providers.py
# -*- coding: utf-8 -*-
"""
LLM provider abstraction — OpenAI, vLLM, and future backends.

Design goals
------------
1. **One code path for every backend.** vLLM exposes an OpenAI-compatible
   HTTP API, so both providers wrap ``openai.AsyncOpenAI`` and share the
   exact same call shape. The difference is ``base_url`` plus a client-side
   concurrency cap for vLLM (server-side continuous batching still does the
   real throughput work).

2. **Per-agent routing + graceful fallback.** ``LLMRouter.select`` picks the
   provider for an agent from ``LLMConfig.routing`` and returns a sequence
   so callers can try a fallback on HTTP / timeout errors without rewriting
   control flow.

3. **Observability without schema changes.** Each provider stamps its
   response with ``_chatdft_meta`` containing provider name, resolved model,
   and latency. ``openai_wrapper.chatgpt_call`` then serialises the provider
   into the existing ``AgentLog.model`` column as ``"{provider}:{model}"``
   — the dashboard parses this without needing an Alembic migration.

4. **Health checks are optional and non-blocking.** On first use the vLLM
   provider pings its ``/health`` endpoint; failure marks the provider
   unhealthy and the router will skip it in favour of the fallback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from server.utils.llm_config import (
    LLMConfig,
    ProviderConfig,
    get_llm_config,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy openai import
# ---------------------------------------------------------------------------
try:
    from openai import AsyncOpenAI, APITimeoutError, APIStatusError, RateLimitError
    _OPENAI_OK = True
except ImportError:  # pragma: no cover — tests monkey-patch
    AsyncOpenAI = None  # type: ignore
    APITimeoutError = Exception  # type: ignore
    APIStatusError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore
    _OPENAI_OK = False


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------

@dataclass
class LLMCallResult:
    """Normalised response returned by every provider."""
    provider: str
    model: str
    raw: Dict[str, Any]           # full openai-style response dict
    latency_ms: int
    success: bool
    error: Optional[str] = None
    retries_used: int = 0

    def text(self) -> str:
        try:
            return self.raw["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def tokens(self) -> Dict[str, int]:
        usage = self.raw.get("usage") or {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }


# ---------------------------------------------------------------------------
# Provider base class
# ---------------------------------------------------------------------------

class LLMProvider:
    """Abstract provider. Subclasses implement ``_raw_chat_completion``."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name
        self._semaphore = asyncio.Semaphore(max(1, config.max_concurrent))
        self._healthy: Optional[bool] = None   # None → unknown, True/False set on first check
        self._client: Optional[AsyncOpenAI] = None
        self._init_client()

    # -- hooks ------------------------------------------------------------

    def _init_client(self) -> None:
        if not _OPENAI_OK or AsyncOpenAI is None:
            self._client = None
            return
        kwargs: Dict[str, Any] = {}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        key = self.config.resolved_api_key()
        if key:
            kwargs["api_key"] = key
        try:
            self._client = AsyncOpenAI(**kwargs)
        except Exception as e:  # pragma: no cover — bad config
            log.warning("provider %s: client init failed: %s", self.name, e)
            self._client = None

    async def health_check(self) -> bool:
        """
        Default health check = True if the client initialised. Subclasses
        may override (vLLM pings /health, OpenAI assumes reachable).
        The result is cached per provider instance.
        """
        if self._healthy is not None:
            return self._healthy
        self._healthy = self._client is not None
        return self._healthy

    def mark_unhealthy(self) -> None:
        self._healthy = False

    # -- public API -------------------------------------------------------

    async def chat_completion(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        json_mode: bool = True,
        retries: int = 3,
        **extra: Any,
    ) -> LLMCallResult:
        """
        Execute a chat completion through this provider with retry and
        concurrency control. Returns a normalised ``LLMCallResult``.
        """
        if self._client is None:
            return LLMCallResult(
                provider=self.name,
                model=model or self.config.model or "",
                raw={"error": "client unavailable", "choices": []},
                latency_ms=0,
                success=False,
                error="client unavailable",
            )

        resolved_model = model or self.config.model or "gpt-4o-mini"
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}

        t0 = time.time()
        last_err: Optional[Exception] = None
        raw: Optional[Dict[str, Any]] = None
        retries_used = 0

        async with self._semaphore:
            for attempt in range(retries):
                try:
                    resp = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            model=resolved_model,
                            messages=list(messages),
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format=response_format,
                            **extra,
                        ),
                        timeout=self.config.timeout_s,
                    )
                    raw = (
                        resp.model_dump() if hasattr(resp, "model_dump") else resp.to_dict()
                    )
                    break

                except RateLimitError as e:
                    wait = 2 ** (attempt + 1)
                    log.warning(
                        "%s rate-limited (attempt %d/%d), sleeping %ds",
                        self.name, attempt + 1, retries, wait,
                    )
                    await asyncio.sleep(wait)
                    last_err = e
                    retries_used = attempt + 1

                except (APITimeoutError, asyncio.TimeoutError) as e:
                    wait = 2 ** attempt
                    log.warning(
                        "%s timeout (attempt %d/%d), sleeping %ds",
                        self.name, attempt + 1, retries, wait,
                    )
                    await asyncio.sleep(wait)
                    last_err = e
                    retries_used = attempt + 1

                except Exception as e:
                    log.error("%s chat_completion failed: %s", self.name, e)
                    last_err = e
                    retries_used = attempt + 1
                    # Mark provider unhealthy on non-retryable errors (e.g.
                    # connection refused) so router can skip subsequent calls.
                    self.mark_unhealthy()
                    break

        latency_ms = int((time.time() - t0) * 1000)

        if raw is None:
            return LLMCallResult(
                provider=self.name,
                model=resolved_model,
                raw={"error": str(last_err) if last_err else "unknown", "choices": []},
                latency_ms=latency_ms,
                success=False,
                error=str(last_err) if last_err else "unknown",
                retries_used=retries_used,
            )

        # Attach provenance metadata that openai_wrapper lifts into AgentLog
        raw.setdefault("_chatdft_meta", {}).update({
            "provider": self.name,
            "model": resolved_model,
            "latency_ms": latency_ms,
            "retries_used": retries_used,
        })
        return LLMCallResult(
            provider=self.name,
            model=resolved_model,
            raw=raw,
            latency_ms=latency_ms,
            success=True,
            retries_used=retries_used,
        )

    async def embed(self, texts: Sequence[str], model: Optional[str] = None) -> List[List[float]]:
        """Embed a list of strings. Raises on failure (caller handles)."""
        if self._client is None:
            raise RuntimeError(f"{self.name}: client unavailable")
        async with self._semaphore:
            resp = await asyncio.wait_for(
                self._client.embeddings.create(
                    input=list(texts),
                    model=model or self.config.model or "text-embedding-3-small",
                ),
                timeout=self.config.timeout_s,
            )
            return [item.embedding for item in resp.data]


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """Thin wrapper — the default SDK already talks to api.openai.com."""
    pass


class VLLMProvider(LLMProvider):
    """
    OpenAI-compatible vLLM HTTP server.

    Differences from OpenAIProvider:
    * Base URL is required and points at the local / remote vLLM server.
    * ``api_key`` defaults to ``"EMPTY"`` — vLLM accepts any non-empty string.
    * Health check pings the ``/health`` endpoint via ``httpx`` if provided;
      otherwise falls back to ``True`` once the client initialises.
    """

    async def health_check(self) -> bool:
        if self._healthy is not None:
            return self._healthy
        if self._client is None:
            self._healthy = False
            return False
        url = self.config.health_check_url
        if not url:
            self._healthy = True
            return True
        try:
            import httpx  # optional dependency — already in FastAPI stack
            async with httpx.AsyncClient(timeout=5.0) as hc:
                r = await hc.get(url)
                self._healthy = r.status_code == 200
        except Exception as e:
            log.warning("vllm health check at %s failed: %s", url, e)
            self._healthy = False
        return self._healthy


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class LLMRouter:
    """
    Picks providers for agents and executes calls with fallback.

    Call sites should do::

        router = get_llm_router()
        result = await router.chat_completion(
            agent_name="intent_agent",
            messages=messages,
            model_hint="gpt-4o-mini",
        )
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_llm_config()
        self._providers: Dict[str, LLMProvider] = {}
        self._build_providers()

    def _build_providers(self) -> None:
        for name, pcfg in self.config.providers.items():
            if pcfg.type == "vllm":
                self._providers[name] = VLLMProvider(pcfg)
            else:
                self._providers[name] = OpenAIProvider(pcfg)

    def provider(self, name: str) -> Optional[LLMProvider]:
        return self._providers.get(name)

    async def _ordered_providers(self, agent_name: str) -> List[LLMProvider]:
        """
        Return providers to try for an agent in priority order, filtering out
        unhealthy ones. Always includes the fallback if it's distinct and
        healthy.
        """
        names = self.config.providers_in_priority(agent_name)
        out: List[LLMProvider] = []
        for name in names:
            p = self._providers.get(name)
            if p is None:
                continue
            if await p.health_check():
                out.append(p)
        return out

    async def chat_completion(
        self,
        *,
        agent_name: str,
        messages: Sequence[Dict[str, Any]],
        model_hint: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMCallResult:
        """
        Execute a chat completion. Tries the routed provider first, then the
        fallback provider on failure. ``model_hint`` overrides the provider's
        default model if supplied.
        """
        providers = await self._ordered_providers(agent_name)
        if not providers:
            return LLMCallResult(
                provider="none",
                model=model_hint or "",
                raw={"error": "no healthy providers", "choices": []},
                latency_ms=0,
                success=False,
                error="no healthy providers",
            )

        last: Optional[LLMCallResult] = None
        for prov in providers:
            # Only pass model_hint if the provider has been explicitly
            # configured for the same model family. vLLM cannot serve
            # gpt-4o-mini; OpenAI cannot serve Qwen/Qwen2.5-7B. So we honour
            # the provider's own default model when the hint is a known
            # OpenAI name and the provider is vLLM (or vice versa).
            effective_model = _pick_model(prov, model_hint)
            result = await prov.chat_completion(
                messages=messages,
                model=effective_model,
                **kwargs,
            )
            if result.success:
                return result
            log.warning(
                "router: provider=%s agent=%s failed (%s); trying next",
                prov.name, agent_name, result.error,
            )
            last = result
            prov.mark_unhealthy()
        # All providers failed — return the last error envelope
        assert last is not None
        return last

    async def embed(
        self,
        *,
        agent_name: str,
        texts: Sequence[str],
        model_hint: Optional[str] = None,
    ) -> List[List[float]]:
        """Embed texts via the routed provider, fallback on failure."""
        providers = await self._ordered_providers(agent_name)
        if not providers:
            raise RuntimeError("no healthy providers for embeddings")
        last_err: Optional[Exception] = None
        for prov in providers:
            try:
                return await prov.embed(texts, model=_pick_model(prov, model_hint))
            except Exception as e:
                log.warning(
                    "router.embed: provider=%s agent=%s failed (%s); trying next",
                    prov.name, agent_name, e,
                )
                prov.mark_unhealthy()
                last_err = e
        raise last_err or RuntimeError("embed failed")


def _pick_model(provider: LLMProvider, hint: Optional[str]) -> Optional[str]:
    """
    Return the model string to use for ``provider``.

    If the caller provides a hint that looks like it belongs to a different
    backend (e.g. ``gpt-4o`` on a vLLM provider), we discard the hint and
    fall back to the provider's default model.
    """
    if not hint:
        return provider.config.model
    is_openai_name = hint.startswith("gpt-") or hint.startswith("text-embedding-")
    if provider.config.type == "vllm" and is_openai_name:
        return provider.config.model
    if provider.config.type == "openai" and not is_openai_name:
        # Hint is probably a local model name; OpenAI can't serve it.
        return provider.config.model
    return hint


# ---------------------------------------------------------------------------
# Module-level singleton (with reset for tests)
# ---------------------------------------------------------------------------

_ROUTER: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = LLMRouter()
    return _ROUTER


def reset_llm_router() -> None:
    """Tests call this to re-read config after mutating env vars."""
    global _ROUTER
    _ROUTER = None
