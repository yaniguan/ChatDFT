# server/utils/perplexity_client.py
# -*- coding: utf-8 -*-
"""
Perplexity API client for real-time web/literature search.

Requires:
  PERPLEXITY_API_KEY env var
  pip install httpx

Usage:
  results = await search("CO2 reduction on Cu formate pathway DFT")
  for r in results:
      print(r["title"], r["url"], r["snippet"])
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
_BASE_URL = "https://api.perplexity.ai"

# Sonar models: "sonar" (128k), "sonar-pro" (200k, higher quality)
_DEFAULT_MODEL = "sonar"


async def search(
    query: str,
    *,
    model: str = _DEFAULT_MODEL,
    max_results: int = 5,
    search_domain_filter: Optional[List[str]] = None,
    return_citations: bool = True,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Query Perplexity's online LLM and return structured results.

    Returns list of dicts:
      {"title": str, "url": str, "snippet": str, "source": "perplexity"}
    """
    if not PERPLEXITY_API_KEY:
        return [{"title": "Perplexity unavailable", "url": "",
                 "snippet": "Set PERPLEXITY_API_KEY to enable real-time search.",
                 "source": "perplexity"}]
    try:
        import httpx
    except ImportError:
        return [{"title": "httpx not installed", "url": "",
                 "snippet": "pip install httpx", "source": "perplexity"}]

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "You are a scientific literature assistant. "
                "Return concise, accurate information with citations. "
                "Focus on peer-reviewed DFT and computational catalysis literature."
            )},
            {"role": "user", "content": query},
        ],
        "return_citations": return_citations,
        "search_recency_filter": "month",
    }
    if search_domain_filter:
        payload["search_domain_filter"] = search_domain_filter

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{_BASE_URL}/chat/completions",
                                  json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
    except (json.JSONDecodeError, ValueError) as e:
        return [{"title": f"Perplexity error: {e}", "url": "", "snippet": str(e), "source": "perplexity"}]

    answer = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    citations = data.get("citations") or []

    results = []
    # Return citations as individual entries
    for c in citations[:max_results]:
        if isinstance(c, str):
            results.append({"title": c, "url": c, "snippet": "", "source": "perplexity"})
        elif isinstance(c, dict):
            results.append({
                "title": c.get("title", c.get("url", "")),
                "url":   c.get("url", ""),
                "snippet": c.get("snippet", ""),
                "source": "perplexity",
            })

    # If no structured citations, include the answer text as one result
    if not results and answer:
        results.append({
            "title": f"Perplexity: {query[:60]}",
            "url": "",
            "snippet": answer[:800],
            "source": "perplexity",
        })

    return results


async def search_dft_literature(
    reaction: str,
    surface: str,
    topic: str = "reaction mechanism",
) -> List[Dict[str, Any]]:
    """Convenience wrapper for DFT-specific queries."""
    q = f"DFT computational study {reaction} on {surface} {topic} mechanism adsorption energy"
    return await search(q, max_results=5)
