# server/utils/rag_utils.py
# -*- coding: utf-8 -*-
"""
RAG utilities for ChatDFT.

Layers
------
1. embed_text(text)           → list[float]  (OpenAI text-embedding-3-small)
2. ingest_paper(doc, chunks)  → persists KnowledgeDoc + KnowledgeChunk rows
3. semantic_search(query, k)  → list[KnowledgeChunk] via pgvector cosine sim
4. keyword_search(query, k)   → list[KnowledgeChunk] via SQL ILIKE
5. hybrid_search(query, k)    → merge of semantic + keyword (RRF fusion)
6. retrieve_from_history(...)  → recent session messages (unchanged)
7. rag_context(...)           → assembles final context string for LLM prompt
8. log_agent_call(...)        → writes to AgentLog table

pgvector path
-------------
If pgvector is installed AND the PostgreSQL extension is enabled, similarity
search runs in-database (fast).  Otherwise it falls back to Python-side
cosine similarity over JSON embeddings (slow but correct).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB imports
# ---------------------------------------------------------------------------
try:
    from server.db import (
        AsyncSessionLocal,
        AgentLog,
        ChatMessage,
        KnowledgeDoc,
        KnowledgeChunk,
        HAS_PGVECTOR,
        VECTOR_DIM,
    )
    _DB_OK = True
except Exception as e:
    log.warning("rag_utils: DB import failed (%s) — running in stub mode", e)
    _DB_OK = False
    AsyncSessionLocal = None
    HAS_PGVECTOR = False
    VECTOR_DIM = 1536

# ---------------------------------------------------------------------------
# OpenAI client (embeddings + chat)
# ---------------------------------------------------------------------------
try:
    from openai import AsyncOpenAI  # type: ignore
    _oa = AsyncOpenAI()
    _OA_OK = True
except ImportError:
    _oa = None
    _OA_OK = False


# ===========================================================================
# 1. Embedding
# ===========================================================================

_EMBED_CACHE: Dict[str, List[float]] = {}   # in-process cache (session lifetime)
_EMBED_LOCK = asyncio.Lock()                 # prevent race conditions on cache writes


async def embed_text(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Return the embedding vector for *text*. Thread-safe in-memory cache."""
    text = text.strip()
    if not text:
        return [0.0] * VECTOR_DIM

    key = hashlib.md5(text.encode()).hexdigest()
    # Read without lock (dict reads are thread-safe in CPython)
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]

    if not _OA_OK or _oa is None:
        return [0.0] * VECTOR_DIM

    try:
        resp = await _oa.embeddings.create(input=text[:8000], model=model)
        vec = resp.data[0].embedding
        async with _EMBED_LOCK:
            _EMBED_CACHE[key] = vec
            # Evict oldest entries if cache grows too large
            if len(_EMBED_CACHE) > 10000:
                to_remove = list(_EMBED_CACHE.keys())[:2000]
                for k in to_remove:
                    _EMBED_CACHE.pop(k, None)
        return vec
    except Exception as e:
        log.warning("embed_text failed: %s", e)
        return [0.0] * VECTOR_DIM


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


# ===========================================================================
# 2. Ingestion pipeline
# ===========================================================================

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> List[str]:
    """
    Split *text* into overlapping word-based chunks.
    chunk_size / overlap are in words (not tokens) for simplicity.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


async def ingest_paper(
    title: str,
    abstract: str,
    full_text: Optional[str] = None,
    source_type: str = "arxiv",
    source_id: Optional[str] = None,
    url: Optional[str] = None,
    doi: Optional[str] = None,
    authors: Optional[List[str]] = None,
    year: Optional[int] = None,
    journal: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Optional[int]:
    """
    Upsert a paper and create/update its KnowledgeChunks with embeddings.
    Returns the KnowledgeDoc.id, or None on failure.
    """
    if not _DB_OK or AsyncSessionLocal is None:
        return None

    from sqlalchemy import select

    # Decide what text to chunk
    body = full_text or abstract or title

    try:
        async with AsyncSessionLocal() as s:
            # --- upsert KnowledgeDoc ---
            stmt = select(KnowledgeDoc).where(
                KnowledgeDoc.source_type == source_type,
                KnowledgeDoc.source_id == source_id,
            )
            res = await s.execute(stmt)
            doc = res.scalar_one_or_none()

            if doc is None:
                doc = KnowledgeDoc(
                    title=title,
                    abstract=abstract,
                    full_text=full_text,
                    source_type=source_type,
                    source_id=source_id,
                    url=url,
                    doi=doi,
                    authors=authors or [],
                    year=year,
                    journal=journal,
                    tags=tags or [],
                )
                s.add(doc)
                await s.flush()  # get doc.id
            else:
                # Update fields that may have improved
                if abstract and not doc.abstract:
                    doc.abstract = abstract
                if full_text and not doc.full_text:
                    doc.full_text = full_text

            doc_id = doc.id

            # --- create chunks if none exist yet ---
            existing_stmt = select(KnowledgeChunk).where(KnowledgeChunk.doc_id == doc_id)
            existing_res = await s.execute(existing_stmt)
            if not existing_res.scalars().all():
                # Use semantic_chunk() for section-aware splitting;
                # abstract is always prepended as chunk 0.
                abstract_text = abstract or title
                body_chunks = semantic_chunk(body)  # [{text, section}, ...]

                # Build final chunk list: abstract anchor + deduped body chunks
                deduped = [c for c in body_chunks if c["text"].strip() != abstract_text.strip()]
                all_chunks = [{"text": abstract_text, "section": "abstract"}] + deduped

                for idx, ch in enumerate(all_chunks):
                    chunk_text_raw = ch["text"]
                    section        = ch["section"]
                    if not chunk_text_raw.strip():
                        continue
                    vec = await embed_text(chunk_text_raw)
                    chunk = KnowledgeChunk(
                        doc_id=doc_id,
                        chunk_idx=idx,
                        text=chunk_text_raw[:4000],
                        token_count=len(chunk_text_raw.split()),
                        embedding=vec if HAS_PGVECTOR else vec,
                        section=section,
                    )
                    s.add(chunk)

            await s.commit()
            return doc_id
    except Exception as e:
        log.error("ingest_paper failed: %s", e, exc_info=True)
        return None


# ===========================================================================
# 3. Semantic search (pgvector or Python fallback)
# ===========================================================================

async def semantic_search(
    query: str,
    top_k: int = 8,
    tags_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Return top_k chunk dicts sorted by cosine similarity to *query*.
    Each dict: {chunk_id, doc_id, title, text, section, score, year, url}
    """
    if not _DB_OK or AsyncSessionLocal is None:
        return []

    query_vec = await embed_text(query)

    from sqlalchemy import select, text as sa_text

    try:
        async with AsyncSessionLocal() as s:
            if HAS_PGVECTOR:
                # In-database cosine distance via pgvector <=> operator
                raw = sa_text("""
                    SELECT
                        kc.id           AS chunk_id,
                        kc.doc_id,
                        kc.text,
                        kc.section,
                        kd.title,
                        kd.year,
                        kd.url,
                        kd.tags,
                        1 - (kc.embedding <=> CAST(:vec AS vector)) AS score
                    FROM knowledge_chunk kc
                    JOIN knowledge_doc kd ON kd.id = kc.doc_id
                    WHERE kc.embedding IS NOT NULL
                    ORDER BY kc.embedding <=> CAST(:vec AS vector)
                    LIMIT :k
                """)
                res = await s.execute(
                    raw,
                    {"vec": json.dumps(query_vec), "k": top_k * 3},
                )
                rows = res.fetchall()
            else:
                # Python-side cosine (slower, fine for small corpus)
                stmt = (
                    select(KnowledgeChunk, KnowledgeDoc.title, KnowledgeDoc.year,
                           KnowledgeDoc.url, KnowledgeDoc.tags)
                    .join(KnowledgeDoc, KnowledgeDoc.id == KnowledgeChunk.doc_id)
                    .where(KnowledgeChunk.embedding.isnot(None))
                    .limit(2000)
                )
                res = await s.execute(stmt)
                rows = res.fetchall()

                scored = []
                for row in rows:
                    chunk = row[0]
                    vec = chunk.embedding
                    if isinstance(vec, str):
                        vec = json.loads(vec)
                    if not vec:
                        continue
                    score = _cosine(query_vec, vec)
                    scored.append((score, chunk, row[1], row[2], row[3], row[4]))
                scored.sort(key=lambda x: x[0], reverse=True)
                rows = [
                    {
                        "chunk_id": r[1].id,
                        "doc_id": r[1].doc_id,
                        "text": r[1].text,
                        "section": r[1].section,
                        "title": r[2],
                        "year": r[3],
                        "url": r[4],
                        "tags": r[5],
                        "score": r[0],
                    }
                    for r in scored[:top_k]
                ]
                return _filter_and_format(rows, tags_filter, top_k)

            # Format pgvector results
            results = []
            for row in rows:
                tags = row.tags if hasattr(row, "tags") else []
                if tags_filter and not any(t in (tags or []) for t in tags_filter):
                    continue
                results.append({
                    "chunk_id": row.chunk_id,
                    "doc_id": row.doc_id,
                    "text": row.text,
                    "section": row.section,
                    "title": row.title,
                    "year": row.year,
                    "url": row.url,
                    "tags": tags,
                    "score": float(row.score),
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    except Exception as e:
        log.error("semantic_search failed: %s", e, exc_info=True)
        return []


def _filter_and_format(rows, tags_filter, top_k):
    if not tags_filter:
        return rows[:top_k]
    filtered = [r for r in rows if any(t in (r.get("tags") or []) for t in tags_filter)]
    return (filtered or rows)[:top_k]


# ===========================================================================
# 4. Keyword search (SQL ILIKE)
# ===========================================================================

async def keyword_search(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """Simple ILIKE-based fallback that works without embeddings."""
    if not _DB_OK or AsyncSessionLocal is None:
        return []

    from sqlalchemy import select, or_

    terms = [t for t in query.split() if len(t) > 2][:6]
    if not terms:
        return []

    try:
        async with AsyncSessionLocal() as s:
            conditions = [
                KnowledgeChunk.text.ilike(f"%{term}%") for term in terms
            ]
            stmt = (
                select(KnowledgeChunk, KnowledgeDoc.title, KnowledgeDoc.year,
                       KnowledgeDoc.url)
                .join(KnowledgeDoc, KnowledgeDoc.id == KnowledgeChunk.doc_id)
                .where(or_(*conditions))
                .limit(top_k * 2)
            )
            res = await s.execute(stmt)
            rows = res.fetchall()

            results = []
            for row in rows:
                chunk, title, year, url = row
                hit_count = sum(t.lower() in (chunk.text or "").lower() for t in terms)
                results.append({
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "section": chunk.section,
                    "title": title,
                    "year": year,
                    "url": url,
                    "score": hit_count / len(terms),
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
    except Exception as e:
        log.error("keyword_search failed: %s", e)
        return []


# ===========================================================================
# 5. Hybrid search (RRF fusion of semantic + keyword)
# ===========================================================================

async def hybrid_search(
    query: str,
    top_k: int = 8,
    semantic_weight: float = 0.7,
    tags_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion of semantic + keyword results.
    Returns deduplicated, re-ranked chunks.
    """
    sem_task = semantic_search(query, top_k=top_k * 2, tags_filter=tags_filter)
    kw_task  = keyword_search(query, top_k=top_k * 2)

    sem_results, kw_results = await asyncio.gather(sem_task, kw_task)

    rrf_scores: Dict[int, float] = {}
    chunk_map: Dict[int, Dict] = {}

    rrf_k = 60  # standard RRF constant

    for rank, r in enumerate(sem_results):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + semantic_weight / (rrf_k + rank + 1)
        chunk_map[cid] = r

    for rank, r in enumerate(kw_results):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + (1 - semantic_weight) / (rrf_k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = r

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for cid, score in ranked[:top_k]:
        item = chunk_map[cid].copy()
        item["rrf_score"] = score
        results.append(item)

    return results


# ===========================================================================
# RAG 2.0 — semantic chunking, cross-encoder reranking, agent context loader
# ===========================================================================

# ── 5a. Semantic / section-aware chunking ─────────────────────────────────

_SECTION_HEADERS = [
    "abstract", "introduction", "background", "related work",
    "methods", "methodology", "computational details", "calculation details",
    "results", "results and discussion", "discussion",
    "conclusions", "conclusion", "summary",
    "acknowledgements", "references",
]

def _detect_section(text: str) -> str:
    """Heuristic: return section label for the start of a paragraph."""
    first_line = text.strip().split("\n")[0].lower().strip()
    for s in _SECTION_HEADERS:
        if first_line.startswith(s):
            return s.replace(" ", "_")
    return "body"


def semantic_chunk(
    text: str,
    max_words: int = 350,
    overlap_words: int = 50,
) -> List[Dict[str, str]]:
    """
    Section-aware chunker.  Splits on section header patterns first, then
    falls back to word-count windowing within each section.

    Returns a list of dicts: [{text, section}, ...].
    This replaces the plain ``chunk_text()`` function for new ingestion.
    """
    import re as _re

    # Split on section header lines (e.g. "2. Methods", "Results and Discussion")
    header_re = _re.compile(
        r"(?:^|\n)("
        + "|".join(_re.escape(h) for h in _SECTION_HEADERS)
        + r")",
        _re.IGNORECASE,
    )
    parts = header_re.split(text)
    # parts alternates: [pre_text, header1, section1_body, header2, section2_body, ...]

    sections: List[tuple] = []  # (section_label, body_text)
    if len(parts) <= 1:
        # No headers found — treat whole doc as 'body'
        sections = [("body", text)]
    else:
        # Reconstruct sections
        sections.append(("intro", parts[0]))
        i = 1
        while i + 1 < len(parts):
            label = parts[i].lower().strip().replace(" ", "_")
            body  = parts[i + 1]
            sections.append((label, body))
            i += 2

    # Word-window within each section
    chunks: List[Dict[str, str]] = []
    for section_label, body in sections:
        words = body.split()
        if not words:
            continue
        i = 0
        while i < len(words):
            window = words[i: i + max_words]
            chunk_text_raw = " ".join(window)
            if chunk_text_raw.strip():
                chunks.append({"text": chunk_text_raw, "section": section_label})
            i += max_words - overlap_words

    return chunks


# ── 5b. Cross-encoder reranker ────────────────────────────────────────────

# Optional: sentence-transformers cross-encoder for reranking
try:
    from sentence_transformers.cross_encoder import CrossEncoder as _CrossEncoder
    _CE_MODEL = _CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    _HAS_CE = True
    log.info("rag_utils: cross-encoder loaded (cross-encoder/ms-marco-MiniLM-L-6-v2)")
except ImportError:
    _CE_MODEL = None
    _HAS_CE = False


async def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    Re-rank *candidates* (each with a 'text' field) using a cross-encoder.
    Falls back to the original RRF/semantic order if the model is unavailable.

    Parameters
    ----------
    query      : original user query
    candidates : list of chunk dicts (from hybrid_search — must have 'text')
    top_k      : number of chunks to keep after reranking

    Returns the top_k most relevant chunks with an added 'rerank_score' field.
    """
    if not candidates:
        return []

    if not _HAS_CE or _CE_MODEL is None:
        # No cross-encoder available — return as-is (already RRF-ranked)
        return candidates[:top_k]

    pairs = [(query, c["text"][:512]) for c in candidates]
    try:
        # Cross-encoder inference is CPU-bound — run in a thread
        loop = asyncio.get_running_loop()
        scores: list = await loop.run_in_executor(
            None, lambda: _CE_MODEL.predict(pairs).tolist()  # type: ignore[union-attr]
        )
        for chunk, score in zip(candidates, scores):
            chunk = dict(chunk)  # don't mutate caller's dict
            chunk["rerank_score"] = float(score)
        ranked = sorted(
            [dict(c, rerank_score=s) for c, s in zip(candidates, scores)],
            key=lambda x: x["rerank_score"],
            reverse=True,
        )
        return ranked[:top_k]
    except Exception as exc:
        log.warning("rerank: cross-encoder failed (%s), using original order", exc)
        return candidates[:top_k]


# ── 5c. get_context() — unified context loader for structure/parameter agents ─

async def get_context(
    query: str,
    *,
    session_id: Optional[int] = None,
    top_k: int = 6,
    tags_filter: Optional[List[str]] = None,
    include_structures: bool = True,
    include_dft_results: bool = True,
    use_reranker: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive context loader for structure and parameter agents.

    Queries three sources in parallel and returns a structured dict:
      - literature   : top-k chunks from RAG (semantic + keyword + reranker)
      - structures   : prior POSCAR / structure_t2s entries from this session
      - dft_results  : converged DFTResult rows from this session (for parameter hints)
      - history      : recent session messages (intent / plan)

    Usage::

        ctx = await get_context(
            "CO adsorption on Cu(111) hollow site",
            session_id=7,
            tags_filter=["CO2RR", "Cu"],
        )
        lit_text = ctx["literature_text"]   # formatted for LLM prompt
        poscar   = ctx["structures"][0]["poscar_content"]  # reuse existing slab
        e_ads    = ctx["dft_results"][0]["value"]           # converged E_ads

    Returns
    -------
    {
      "literature": [chunk_dict, ...],
      "literature_text": "## Literature Context\n...",
      "structures": [StructureT2S-like dicts, ...],
      "dft_results": [DFTResult-like dicts, ...],
      "history": [message_content_str, ...],
    }
    """
    from sqlalchemy import select, desc

    # ── Parallel fetch ────────────────────────────────────────────────────
    async def _fetch_literature():
        candidates = await hybrid_search(query, top_k=top_k * 3, tags_filter=tags_filter)
        if use_reranker and len(candidates) > top_k:
            return await rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]

    async def _fetch_structures():
        if not include_structures or not session_id or not _DB_OK:
            return []
        try:
            from server.db import StructureT2S
            async with AsyncSessionLocal() as s:
                stmt = (
                    select(StructureT2S)
                    .where(StructureT2S.session_id == session_id)
                    .order_by(desc(StructureT2S.created_at))
                    .limit(5)
                )
                res = await s.execute(stmt)
                rows = list(res.scalars().all())
                return [
                    {
                        "id": r.id,
                        "formula": r.formula,
                        "material": r.material,
                        "facet": r.facet,
                        "adsorbates": r.adsorbates,
                        "poscar_content": r.poscar_content,
                        "natural_language": r.natural_language,
                        "is_optimized": r.is_optimized,
                        "energy_eV": r.energy_eV,
                        "created_at": str(r.created_at),
                    }
                    for r in rows
                ]
        except Exception as exc:
            log.warning("get_context _fetch_structures failed: %s", exc)
            return []

    async def _fetch_dft_results():
        if not include_dft_results or not session_id or not _DB_OK:
            return []
        try:
            from server.db import DFTResult
            async with AsyncSessionLocal() as s:
                stmt = (
                    select(DFTResult)
                    .where(
                        DFTResult.session_id == session_id,
                        DFTResult.converged == True,  # noqa: E712
                    )
                    .order_by(desc(DFTResult.created_at))
                    .limit(10)
                )
                res = await s.execute(stmt)
                rows = list(res.scalars().all())
                return [
                    {
                        "id": r.id,
                        "result_type": r.result_type,
                        "species": r.species,
                        "surface": r.surface,
                        "site": r.site,
                        "value": r.value,
                        "unit": r.unit,
                        "extra": r.extra,
                        "converged": r.converged,
                    }
                    for r in rows
                ]
        except Exception as exc:
            log.warning("get_context _fetch_dft_results failed: %s", exc)
            return []

    literature, structures, dft_results, history = await asyncio.gather(
        _fetch_literature(),
        _fetch_structures(),
        _fetch_dft_results(),
        retrieve_from_history(session_id, top_k=6, msg_types=["intent", "plan"]),
    )

    # ── Format literature for LLM ─────────────────────────────────────────
    lit_lines = []
    for chunk in literature:
        title = chunk.get("title", "?")
        year  = chunk.get("year", "?")
        text  = chunk.get("text", "")[:500]
        score = chunk.get("rerank_score") or chunk.get("rrf_score") or chunk.get("score") or 0
        lit_lines.append(f"**{title} ({year})** [score={score:.3f}]\n{text}")
    literature_text = ("## Literature Context\n" + "\n\n".join(lit_lines)) if lit_lines else ""

    return {
        "literature": literature,
        "literature_text": literature_text,
        "structures": structures,
        "dft_results": dft_results,
        "history": history,
    }


# ===========================================================================
# 6. History retrieval (session context)
# ===========================================================================

async def retrieve_from_history(
    session_id: Optional[int],
    top_k: int = 10,
    msg_types: Optional[List[str]] = None,
) -> List[str]:
    """
    Fetch recent messages for a session, optionally filtered by msg_type.
    Returns content strings in chronological order.
    """
    if not session_id or not _DB_OK or AsyncSessionLocal is None:
        return []

    from sqlalchemy import select, desc

    try:
        async with AsyncSessionLocal() as s:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
            )
            if msg_types:
                stmt = stmt.where(ChatMessage.msg_type.in_(msg_types))
            stmt = stmt.order_by(desc(ChatMessage.created_at)).limit(top_k)
            res = await s.execute(stmt)
            rows = list(res.scalars().all())
            rows.reverse()
            return [str(m.content or "") for m in rows]
    except Exception as e:
        log.warning("retrieve_from_history failed: %s", e)
        return []


# ===========================================================================
# 7. Main RAG context builder
# ===========================================================================

async def retrieve_from_knowledge(query: str, top_k: int = 6) -> List[str]:
    """Convenience wrapper: returns text snippets from hybrid search."""
    results = await hybrid_search(query, top_k=top_k)
    snippets = []
    for r in results:
        header = f"[{r.get('title','?')} ({r.get('year','?')})]"
        snippets.append(f"{header}\n{r.get('text','')[:600]}")
    return snippets


async def rag_context(
    query: str,
    session_id: Optional[int] = None,
    top_k: int = 6,
    include_history: bool = True,
    tags_filter: Optional[List[str]] = None,
) -> str:
    """
    Assemble LLM-ready context string from session history + knowledge base.
    Called by intent / hypothesis / plan / analyze agents.
    """
    coros = []
    if include_history and session_id:
        coros.append(retrieve_from_history(session_id, top_k=8))
    else:
        coros.append(asyncio.sleep(0, result=[]))

    coros.append(hybrid_search(query, top_k=top_k, tags_filter=tags_filter))

    history_msgs, knowledge_chunks = await asyncio.gather(*coros)

    parts: List[str] = []

    if history_msgs:
        parts.append("## Session History (recent)\n" + "\n---\n".join(history_msgs[-6:]))

    if knowledge_chunks:
        lit_lines = []
        for chunk in knowledge_chunks:
            title = chunk.get("title", "?")
            year  = chunk.get("year", "?")
            text  = chunk.get("text", "")[:500]
            lit_lines.append(f"**{title} ({year})**\n{text}")
        parts.append("## Literature Context\n" + "\n\n".join(lit_lines))

    return "\n\n".join(parts)


async def rag_search(query: str, top_k: int = 8) -> List[str]:
    """Thin wrapper for backward compat."""
    return await retrieve_from_knowledge(query, top_k)


# ===========================================================================
# 8. Agent call logger
# ===========================================================================

async def log_agent_call(
    agent_name: str,
    call_type: str = "llm",
    model: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int = 0,
    success: bool = True,
    error_msg: Optional[str] = None,
    input_preview: Optional[str] = None,
    output_preview: Optional[str] = None,
    session_id: Optional[int] = None,
    full_input: Optional[Any] = None,
    full_output: Optional[Any] = None,
) -> None:
    """
    Fire-and-forget write to AgentLog.
    Call with:  asyncio.create_task(log_agent_call(...))
    or simply await it — it never raises.
    """
    if not _DB_OK or AsyncSessionLocal is None:
        return
    try:
        async with AsyncSessionLocal() as s:
            entry = AgentLog(
                session_id=session_id,
                agent_name=agent_name,
                call_type=call_type,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                success=success,
                error_msg=error_msg,
                input_preview=(input_preview or "")[:500],
                output_preview=(output_preview or "")[:500],
                full_input=full_input if not success else None,
                full_output=full_output if not success else None,
            )
            s.add(entry)
            await s.commit()
    except Exception as e:
        log.debug("log_agent_call silently failed: %s", e)
