# server/utils/openai_wrapper.py
# -*- coding: utf-8 -*-
"""
Async OpenAI wrapper with:
- Exponential-backoff retry (3 attempts)
- Token usage extraction
- JSON mode enforcement
- Embedding endpoint
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIStatusError
    _client = AsyncOpenAI()
    _OK = True
except Exception:
    _client = None  # type: ignore
    _OK = False
    RateLimitError = Exception
    APITimeoutError = Exception
    APIStatusError = Exception


async def chatgpt_call(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    json_mode: bool = True,
    retries: int = 3,
    **kwargs,
) -> Dict[str, Any]:
    """
    Async LLM call with retry.
    Returns the full response dict (choices, usage, model, ...).
    On complete failure returns {"error": "...", "choices": []}.
    """
    if not _OK or _client is None:
        return {"error": "OpenAI client not available", "choices": []}

    response_format = {"type": "json_object"} if json_mode else {"type": "text"}

    last_err: Optional[Exception] = None
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
            return resp.model_dump() if hasattr(resp, "model_dump") else resp.to_dict()

        except RateLimitError as e:
            wait = 2 ** (attempt + 1)
            log.warning("Rate limit (attempt %d/%d), retrying in %ds", attempt + 1, retries, wait)
            await asyncio.sleep(wait)
            last_err = e

        except APITimeoutError as e:
            wait = 2 ** attempt
            log.warning("Timeout (attempt %d/%d), retrying in %ds", attempt + 1, retries, wait)
            await asyncio.sleep(wait)
            last_err = e

        except Exception as e:
            log.error("chatgpt_call failed: %s", e, exc_info=True)
            last_err = e
            break   # non-retryable

    return {"error": str(last_err), "choices": []}


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
