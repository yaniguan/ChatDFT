# server/utils/zotero_client.py
# -*- coding: utf-8 -*-
"""
Zotero Web API client for personal library integration.

Requires:
  ZOTERO_API_KEY  — from zotero.org/settings/keys
  ZOTERO_USER_ID  — numeric user ID (or group ID with ZOTERO_GROUP_ID)
  pip install httpx

Usage:
  items = await search_library("CO2 reduction copper")
  for item in items:
      print(item["title"], item["doi"], item["abstract"][:200])
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

ZOTERO_API_KEY  = os.environ.get("ZOTERO_API_KEY", "")
ZOTERO_USER_ID  = os.environ.get("ZOTERO_USER_ID", "")
ZOTERO_GROUP_ID = os.environ.get("ZOTERO_GROUP_ID", "")   # alternative to user library
_BASE = "https://api.zotero.org"


def _library_path() -> str:
    if ZOTERO_GROUP_ID:
        return f"groups/{ZOTERO_GROUP_ID}"
    return f"users/{ZOTERO_USER_ID}"


def _headers() -> Dict[str, str]:
    return {
        "Zotero-API-Key": ZOTERO_API_KEY,
        "Zotero-API-Version": "3",
    }


def _parse_item(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten Zotero item data to a standard dict."""
    d = raw.get("data") or {}
    creators = d.get("creators") or []
    authors = [
        f"{c.get('lastName', '')} {c.get('firstName', '')}".strip()
        for c in creators if c.get("creatorType") == "author"
    ]
    return {
        "key":      raw.get("key", ""),
        "title":    d.get("title", ""),
        "abstract": d.get("abstractNote", ""),
        "doi":      d.get("DOI", ""),
        "url":      d.get("url", ""),
        "year":     d.get("date", "")[:4] if d.get("date") else "",
        "journal":  d.get("publicationTitle", "") or d.get("journalAbbreviation", ""),
        "authors":  authors,
        "tags":     [t.get("tag", "") for t in (d.get("tags") or [])],
        "source":   "zotero",
    }


async def search_library(
    query: str,
    *,
    limit: int = 10,
    item_type: str = "journalArticle",
    timeout: float = 20.0,
) -> List[Dict[str, Any]]:
    """Full-text search across the Zotero library."""
    if not ZOTERO_API_KEY or not (ZOTERO_USER_ID or ZOTERO_GROUP_ID):
        return [{
            "key": "", "title": "Zotero unavailable",
            "abstract": "Set ZOTERO_API_KEY and ZOTERO_USER_ID to enable library search.",
            "doi": "", "url": "", "year": "", "journal": "", "authors": [], "tags": [],
            "source": "zotero",
        }]
    try:
        import httpx
    except ImportError:
        return []

    params: Dict[str, Any] = {
        "q": query,
        "limit": limit,
        "format": "json",
        "include": "data",
    }
    if item_type:
        params["itemType"] = item_type

    url = f"{_BASE}/{_library_path()}/items"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, headers=_headers(), params=params)
            r.raise_for_status()
            items = r.json()
    except (ValueError, KeyError, TypeError) as e:
        return [{"key": "", "title": f"Zotero error: {e}", "abstract": str(e),
                 "doi": "", "url": "", "year": "", "journal": "",
                 "authors": [], "tags": [], "source": "zotero"}]

    return [_parse_item(it) for it in items]


async def get_item(key: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """Fetch a single item by its Zotero key."""
    if not ZOTERO_API_KEY:
        return None
    try:
        import httpx
        url = f"{_BASE}/{_library_path()}/items/{key}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, headers=_headers(), params={"format": "json"})
            r.raise_for_status()
            return _parse_item(r.json())
    except (json.JSONDecodeError, ValueError):
        return None


async def get_collection_items(
    collection_key: str,
    limit: int = 25,
    timeout: float = 20.0,
) -> List[Dict[str, Any]]:
    """Fetch all items from a specific Zotero collection."""
    if not ZOTERO_API_KEY:
        return []
    try:
        import httpx
        url = f"{_BASE}/{_library_path()}/collections/{collection_key}/items"
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, headers=_headers(),
                                 params={"format": "json", "limit": limit})
            r.raise_for_status()
            return [_parse_item(it) for it in r.json()]
    except (json.JSONDecodeError, ValueError):
        return []
