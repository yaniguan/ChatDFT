# server/utils/rag_utils.py
# -*- coding: utf-8 -*-
"""
Unified RAG utils (async)
- Works with either async DB session (AsyncSessionLocal) or sync session (SessionLocal) automatically.
- Provides:
    * retrieve_from_history(session_id, top_k)
    * retrieve_from_knowledge(query, top_k)   # placeholder for your vector DB / pgvector
    * rag_context(query, session_id, top_k)   # concurrent gather of history + knowledge
    * rag_search(query, top_k)                # simple wrapper returning list[str]
"""

from __future__ import annotations
from typing import List, Optional
import asyncio

# ----- Try async DB first; fall back to sync DB -----
AsyncSessionLocal = None
SessionLocal = None
ChatMessage = None

try:
    # Optional async DB (if you have server/db_last.py)
    from server.db_last import AsyncSessionLocal as _AsyncSessionLocal, ChatMessage as _AsyncChatMessage  # type: ignore
    AsyncSessionLocal = _AsyncSessionLocal
    ChatMessage = _AsyncChatMessage
except Exception:
    pass

if ChatMessage is None:
    # Fallback to sync DB (server/db.py)
    try:
        from server.db import SessionLocal as _SessionLocal, ChatMessage as _ChatMessage  # type: ignore
        SessionLocal = _SessionLocal
        ChatMessage = _ChatMessage
    except Exception:
        pass

# ---------------- History retrieval ----------------
async def retrieve_from_history(session_id: Optional[int], top_k: int = 5) -> List[str]:
    """Fetch recent messages for a session (assistant/user mixed), newest first; return in chronological order."""
    if not session_id or ChatMessage is None:
        return []

    # Prefer async session if available
    if AsyncSessionLocal is not None:
        try:
            from sqlalchemy import select, desc  # type: ignore
            async with AsyncSessionLocal() as s:  # type: ignore
                stmt = (
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(desc(ChatMessage.created_at))
                    .limit(top_k)
                )
                res = await s.execute(stmt)
                rows = list(res.scalars().all())
                rows.reverse()  # chronological
                return [str(m.content or "") for m in rows]
        except Exception:
            pass  # fall through to sync

    # Sync path (run in thread)
    if SessionLocal is not None:
        def _pull_sync() -> List[str]:
            s = SessionLocal()
            try:
                msgs = (
                    s.query(ChatMessage)
                    .filter(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at.desc())
                    .limit(top_k)
                    .all()
                )
                msgs.reverse()
                return [str(m.content or "") for m in msgs]
            finally:
                s.close()
        return await asyncio.to_thread(_pull_sync)

    # No DB available
    return []

# ---------------- Knowledge retrieval (placeholder) ----------------
async def retrieve_from_knowledge(query: str, top_k: int = 5) -> List[str]:
    """
    TODO:
      - Plug in FAISS / Milvus / pgvector or your papers table.
      - For now returns empty to keep behavior deterministic.
    """
    # Example stub: return []
    return []

# ---------------- RAG context builder ----------------
async def rag_context(query: str, session_id: Optional[int] = None, top_k: int = 5) -> str:
    """Assemble textual context from history + knowledge concurrently."""
    hist_coro = retrieve_from_history(session_id, top_k) if session_id else asyncio.sleep(0, result=[])
    know_coro = retrieve_from_knowledge(query, top_k)
    history_msgs, knowledge_snips = await asyncio.gather(hist_coro, know_coro)

    parts: List[str] = []
    if history_msgs:
        parts.append("[History]\n" + "\n".join(history_msgs))
    if knowledge_snips:
        parts.append("[Knowledge]\n" + "\n".join(knowledge_snips))
    return "\n\n".join(parts)

# ---------------- Convenience search API ----------------
async def rag_search(query: str, top_k: int = 8) -> List[str]:
    """
    A thin wrapper used by hypothesis/knowledge/plan agents.
    Right now returns knowledge snippets only (can be extended).
    """
    return await retrieve_from_knowledge(query, top_k)