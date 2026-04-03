# server/chat/knowledge_agent.py
# -*- coding: utf-8 -*-
"""
Knowledge Agent — unified RAG service for ChatDFT.

Responsibilities
----------------
1. Ingest arXiv papers (on-demand + daily scheduler).
2. Ingest user-uploaded PDFs (text + figure extraction).
3. Multi-modal embedding: text chunks (1536-d) + figure descriptions (GPT-4o vision).
4. Expose get_rag_context(query, stage) used by intent / hypothesis / plan agents.
5. REST endpoints for manual search and upload.

Endpoints
---------
POST /chat/knowledge              — search + ingest from arXiv on demand
POST /chat/knowledge/upload       — upload a PDF for ingestion
POST /chat/knowledge/daily        — trigger daily update manually
GET  /chat/knowledge/status       — DB stats (doc count, chunk count, last run)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import arxiv
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)
router = APIRouter()

# ── DB ───────────────────────────────────────────────────────────────────────
try:
    from server.db import (
        AsyncSessionLocal, ChatMessage, KnowledgeDoc, KnowledgeChunk,
        KnowledgeFigure, LiteratureUpdateLog,
    )
    _DB_OK = True
except ImportError as e:
    log.warning("knowledge_agent: DB import failed: %s", e)
    _DB_OK = False
    AsyncSessionLocal = None

# ── RAG utilities ─────────────────────────────────────────────────────────────
from server.utils.rag_utils import (
    ingest_paper, hybrid_search, rag_context as _rag_context,
    embed_text, log_agent_call,
)

# ── OpenAI ────────────────────────────────────────────────────────────────────
try:
    from openai import AsyncOpenAI
    _oa = AsyncOpenAI()
    _OA_OK = True
except ImportError:
    _oa = None
    _OA_OK = False

# ── Perplexity (real-time web search) ─────────────────────────────────────────
try:
    from server.utils.perplexity_client import search as _perplexity_search
    _PERPLEXITY_OK = True
except ImportError:
    _PERPLEXITY_OK = False
    async def _perplexity_search(*args, **kwargs):  # type: ignore
        return []

# ── Zotero (personal library) ─────────────────────────────────────────────────
try:
    from server.utils.zotero_client import search_library as _zotero_search
    _ZOTERO_OK = True
except ImportError:
    _ZOTERO_OK = False
    async def _zotero_search(*args, **kwargs):  # type: ignore
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Daily-update search queries (chemistry / materials / DFT focus)
# Add more to broaden the literature base.
# ─────────────────────────────────────────────────────────────────────────────
DAILY_QUERIES: List[Dict[str, Any]] = [
    # Core DFT & surface science
    {"q": "density functional theory surface catalysis reaction mechanism", "max": 20},
    {"q": "VASP DFT electrocatalysis adsorption energy", "max": 15},
    {"q": "first principles reaction pathway transition state", "max": 15},
    # Electrochemistry
    {"q": "electrochemical dehydrogenation alkane DFT platinum", "max": 10},
    {"q": "computational hydrogen electrode potential determining step", "max": 10},
    {"q": "grand canonical DFT electrode potential solvation", "max": 10},
    # Mechanism-specific
    {"q": "CO2 reduction copper mechanism DFT intermediates", "max": 10},
    {"q": "nitrogen reduction reaction NRR DFT mechanism", "max": 10},
    {"q": "oxygen evolution reaction OER DFT lattice oxygen", "max": 10},
    # General catalysis
    {"q": "volcano plot scaling relations heterogeneous catalysis", "max": 10},
    {"q": "microkinetic model DFT surface reaction kinetics", "max": 10},
]


# ─────────────────────────────────────────────────────────────────────────────
# arXiv fetch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_arxiv(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """Synchronous arXiv fetch (run in thread)."""
    search = arxiv.Search(
        query=query,
        max_results=min(max_results, 50),
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client(page_size=25, delay_seconds=0.5)
    results = []
    for r in client.results(search):
        year = r.published.year if r.published else None
        doi  = (r.doi or "").lower() or None
        results.append({
            "title":       r.title.strip(),
            "abstract":    r.summary.strip(),
            "source_type": "arxiv",
            "source_id":   r.entry_id.split("/")[-1],
            "url":         r.entry_id,
            "pdf_url":     getattr(r, "pdf_url", None),
            "doi":         doi,
            "authors":     [str(a) for a in r.authors[:8]],
            "year":        year,
            "journal":     "arXiv",
            "tags":        [r.primary_category] if hasattr(r, "primary_category") else [],
        })
    return results


async def _ingest_arxiv_results(records: List[Dict], extra_tags: List[str] = []) -> int:
    """Ingest a list of arXiv dicts into KnowledgeDoc + KnowledgeChunk."""
    n = 0
    for rec in records:
        tags = list(set((rec.get("tags") or []) + extra_tags))
        doc_id = await ingest_paper(
            title       = rec["title"],
            abstract    = rec["abstract"],
            source_type = rec["source_type"],
            source_id   = rec["source_id"],
            url         = rec.get("url"),
            doi         = rec.get("doi"),
            authors     = rec.get("authors", []),
            year        = rec.get("year"),
            journal     = rec.get("journal"),
            tags        = tags,
        )
        if doc_id:
            n += 1
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Figure extraction + multi-modal embedding
# ─────────────────────────────────────────────────────────────────────────────

async def _describe_figure_vision(image_bytes: bytes) -> str:
    """Use GPT-4o-vision to describe a figure in scientific context."""
    if not _OA_OK or _oa is None:
        return ""
    try:
        b64 = base64.b64encode(image_bytes).decode()
        resp = await _oa.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a computational chemistry expert. "
                            "Describe this figure from a scientific paper in 2-4 sentences. "
                            "Focus on: what type of plot it is (energy diagram, volcano plot, "
                            "DOS, reaction pathway, etc.), what the axes represent, "
                            "and the key scientific finding shown."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            max_tokens=300,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        log.warning("Figure vision description failed: %s", e)
        return ""


async def _extract_and_embed_figures(
    doc_id: int,
    pdf_bytes: bytes,
    figures_dir: Path,
) -> int:
    """Extract figures from PDF, describe via vision, embed, store in DB."""
    try:
        import fitz  # pymupdf
    except ImportError:
        log.info("pymupdf not installed — skipping figure extraction")
        return 0

    if not _DB_OK or AsyncSessionLocal is None:
        return 0

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_figs = 0
    figures_dir.mkdir(parents=True, exist_ok=True)

    from sqlalchemy import select
    for page_num, page in enumerate(doc):
        img_list = page.get_images(full=True)
        for fig_idx, img_info in enumerate(img_list):
            try:
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                img_bytes = base_img["image"]
                ext = base_img.get("ext", "png")

                # Skip tiny images (icons/logos)
                if len(img_bytes) < 5000:
                    continue

                # Save locally
                img_path = figures_dir / f"p{page_num}_f{fig_idx}.{ext}"
                img_path.write_bytes(img_bytes)

                # Get description via vision
                description = await _describe_figure_vision(img_bytes)
                if not description:
                    description = f"Figure {fig_idx+1} from page {page_num+1}"

                # Embed description
                vec = await embed_text(description)

                # Save to DB
                async with AsyncSessionLocal() as s:
                    # Check if already exists
                    from sqlalchemy import select
                    existing = await s.execute(
                        select(KnowledgeFigure).where(
                            KnowledgeFigure.doc_id == doc_id,
                            KnowledgeFigure.page_num == page_num,
                            KnowledgeFigure.figure_idx == fig_idx,
                        )
                    )
                    if existing.scalar_one_or_none() is None:
                        fig = KnowledgeFigure(
                            doc_id=doc_id,
                            page_num=page_num,
                            figure_idx=fig_idx,
                            description=description,
                            embedding=vec,
                            image_path=str(img_path),
                        )
                        s.add(fig)
                        await s.commit()
                        n_figs += 1
            except Exception as e:
                log.warning("Figure extraction error p%d f%d: %s", page_num, fig_idx, e)
    return n_figs


# ─────────────────────────────────────────────────────────────────────────────
# RAG context function — called by intent / hypothesis / plan agents
# ─────────────────────────────────────────────────────────────────────────────

async def get_rag_context(
    query: str,
    stage: str = "general",
    session_id: Optional[int] = None,
    top_k: int = 6,
    include_figures: bool = False,
) -> str:
    """
    Build context string for LLM prompt injection.
    Called by intent_agent, hypothesis_agent, plan_agent, analyze_agent.

    stage: "intent" | "hypothesis" | "plan" | "analysis" | "general"
    """
    # Semantic + keyword hybrid search
    chunks = await hybrid_search(query, top_k=top_k)

    parts: List[str] = []

    # Literature snippets
    if chunks:
        lit_lines = []
        for c in chunks:
            title = c.get("title", "Unknown")
            year  = c.get("year", "")
            text  = c.get("text", "")[:500]
            score = c.get("rrf_score") or c.get("score", 0)
            lit_lines.append(f"**[{title}, {year}]** (relevance: {score:.2f})\n{text}")
        parts.append("## Literature\n" + "\n\n".join(lit_lines))

    # Figure descriptions (for hypothesis stage — volcano plots, energy diagrams)
    if include_figures and _DB_OK and AsyncSessionLocal is not None:
        from sqlalchemy import select
        query_vec = await embed_text(query)
        fig_snippets = await _search_figures(query_vec, top_k=3)
        if fig_snippets:
            fig_lines = [f"- {s}" for s in fig_snippets]
            parts.append("## Related Figures\n" + "\n".join(fig_lines))

    # Past session context (if stage=plan, find similar completed workflows)
    if stage == "plan" and session_id and _DB_OK:
        past = await _get_past_plan_context(query, session_id)
        if past:
            parts.append("## Past Similar Calculations\n" + past)

    return "\n\n".join(parts)


async def _search_figures(query_vec: List[float], top_k: int = 3) -> List[str]:
    """Return figure description strings matching query."""
    if not _DB_OK or AsyncSessionLocal is None:
        return []
    try:
        from sqlalchemy import select
        async with AsyncSessionLocal() as s:
            stmt = (
                select(KnowledgeFigure, KnowledgeDoc.title)
                .join(KnowledgeDoc, KnowledgeDoc.id == KnowledgeFigure.doc_id)
                .where(KnowledgeFigure.embedding.isnot(None))
                .limit(500)
            )
            res = await s.execute(stmt)
            rows = res.fetchall()

        from server.utils.rag_utils import _cosine
        scored = []
        for row in rows:
            fig, title = row[0], row[1]
            vec = fig.embedding
            if isinstance(vec, str):
                vec = json.loads(vec)
            if not vec:
                continue
            score = _cosine(query_vec, vec)
            scored.append((score, fig, title))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, fig, title in scored[:top_k]:
            results.append(f"{fig.description} (from: {title}, score: {score:.2f})")
        return results
    except Exception as e:
        log.warning("_search_figures failed: %s", e)
        return []


async def _get_past_plan_context(query: str, current_session_id: int) -> str:
    """
    Find similar completed workflow tasks from other sessions.
    Returns a short markdown summary as hints for plan_agent.
    """
    if not _DB_OK or AsyncSessionLocal is None:
        return ""
    try:
        from sqlalchemy import select, func
        from server.db import WorkflowTask, ChatSession

        # Look for completed tasks with overlapping keywords
        keywords = [w for w in query.lower().split() if len(w) > 3][:6]

        async with AsyncSessionLocal() as s:
            stmt = (
                select(WorkflowTask)
                .where(
                    WorkflowTask.status == "done",
                    WorkflowTask.session_id != current_session_id,
                )
                .order_by(WorkflowTask.updated_at.desc())
                .limit(200)
            )
            res = await s.execute(stmt)
            tasks = res.scalars().all()

        scored = []
        for t in tasks:
            text = f"{t.name or ''} {t.description or ''} {t.agent or ''}".lower()
            hits = sum(1 for kw in keywords if kw in text)
            if hits >= 2:
                scored.append((hits, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return ""

        lines = ["Similar completed tasks from prior studies:"]
        for _, t in scored[:5]:
            inp = t.input_data or {}
            engine = t.engine or ""
            lines.append(f"- **{t.name}** ({engine}): "
                         f"agent={t.agent}, "
                         f"params={json.dumps(inp)[:120]}")
        return "\n".join(lines)
    except Exception as e:
        log.warning("_get_past_plan_context failed: %s", e)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Daily update runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_daily_update(trigger: str = "scheduler") -> Dict[str, Any]:
    """
    Fetch latest arXiv papers for all DAILY_QUERIES and ingest them.
    Called by the scheduler in main.py and by the manual endpoint.
    """
    t0 = time.time()
    total_new = 0
    queries_run = []
    error_msg = None

    log.info("Knowledge daily update starting (%s)", trigger)

    try:
        for q_cfg in DAILY_QUERIES:
            q    = q_cfg["q"]
            maxr = q_cfg.get("max", 15)
            try:
                records = await asyncio.to_thread(_fetch_arxiv, q, maxr)
                n = await _ingest_arxiv_results(records, extra_tags=["daily_update"])
                total_new += n
                queries_run.append({"q": q, "fetched": len(records), "new": n})
                log.info("  %s → %d fetched, %d new", q[:60], len(records), n)
            except Exception as e:
                log.warning("Query failed '%s': %s", q, e)
                queries_run.append({"q": q, "error": str(e)})

    except Exception as e:
        error_msg = str(e)
        log.error("Daily update failed: %s", e, exc_info=True)

    duration = round(time.time() - t0, 1)

    # Log the run
    if _DB_OK and AsyncSessionLocal is not None:
        try:
            async with AsyncSessionLocal() as s:
                run_log = LiteratureUpdateLog(
                    run_at=datetime.now(timezone.utc),
                    trigger=trigger,
                    queries_used=queries_run,
                    n_new_docs=total_new,
                    duration_s=duration,
                    error=error_msg,
                )
                s.add(run_log)
                await s.commit()
        except Exception as e:
            log.warning("Failed to log update run: %s", e)

    result = {
        "ok": error_msg is None,
        "trigger": trigger,
        "n_new_docs": total_new,
        "duration_s": duration,
        "queries": queries_run,
    }
    if error_msg:
        result["error"] = error_msg

    log.info("Knowledge daily update done: %d new docs in %.1fs", total_new, duration)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/chat/knowledge")
async def knowledge_search(request: Request) -> Any:
    """
    On-demand arXiv search + ingest + return relevant chunks.
    Also used by intent/hypothesis/plan agents via the API.
    """
    body = await request.json()
    query      = body.get("query") or ""
    intent     = body.get("intent") or {}
    session_id = body.get("session_id")
    limit      = int(body.get("limit") or 10)
    stage      = body.get("stage") or "general"

    # Build a richer search query from intent fields
    q_parts = [query]
    for k in ("task", "reaction", "domain", "substrate"):
        v = intent.get(k) or (intent.get("system") or {}).get(k, "")
        if v:
            q_parts.append(str(v))
    rich_q = " ".join(q_parts)

    # Fetch + ingest fresh arXiv papers
    t0 = time.time()
    try:
        records = await asyncio.to_thread(_fetch_arxiv, rich_q, limit * 2)
        n_new = await _ingest_arxiv_results(records)
    except Exception as e:
        log.warning("arXiv fetch failed: %s", e)
        records, n_new = [], 0

    # Retrieve best matching chunks from the full DB (newly ingested + existing)
    chunks = await hybrid_search(rich_q, top_k=limit)

    # ── Optional: Perplexity real-time search ──────────────────────────────────
    perplexity_results: List[Dict] = []
    use_perplexity = bool(body.get("use_perplexity", False)) or _PERPLEXITY_OK
    if use_perplexity and _PERPLEXITY_OK:
        try:
            perplexity_results = await _perplexity_search(rich_q, max_results=3)
        except (ValueError, KeyError, TypeError) as _pe:
            log.debug("Perplexity search failed: %s", _pe)

    # ── Optional: Zotero personal library search ───────────────────────────────
    zotero_results: List[Dict] = []
    use_zotero = bool(body.get("use_zotero", False)) or _ZOTERO_OK
    if use_zotero and _ZOTERO_OK:
        try:
            zotero_results = await _zotero_search(rich_q, limit=5)
        except (ValueError, KeyError, TypeError) as _ze:
            log.debug("Zotero search failed: %s", _ze)

    # Save a message to session
    if session_id and _DB_OK and AsyncSessionLocal is not None:
        try:
            async with AsyncSessionLocal() as s:
                msg = ChatMessage(
                    session_id=session_id,
                    role="assistant",
                    content=f"Retrieved {len(chunks)} relevant literature snippets.",
                    msg_type="knowledge",
                    references={"query": rich_q, "n_chunks": len(chunks)},
                )
                s.add(msg)
                await s.commit()
        except Exception:
            pass

    return JSONResponse({
        "ok": True,
        "records": [
            {
                "title":   c.get("title", ""),
                "text":    c.get("text", "")[:400],
                "year":    c.get("year"),
                "url":     c.get("url"),
                "score":   round(c.get("rrf_score") or c.get("score", 0), 3),
                "section": c.get("section"),
            }
            for c in chunks
        ],
        "perplexity_results": perplexity_results,
        "zotero_results": [
            {"title": z.get("title", ""), "doi": z.get("doi", ""),
             "year": z.get("year", ""), "journal": z.get("journal", ""),
             "abstract": z.get("abstract", "")[:300], "source": "zotero"}
            for z in zotero_results
        ],
        "n_fetched_from_arxiv": len(records),
        "n_new_ingested": n_new,
        "duration_s": round(time.time() - t0, 2),
    })


@router.post("/chat/knowledge/upload")
async def upload_paper(
    file: UploadFile = File(...),
    tags: str = Form(default=""),
    session_id: int = Form(default=0),
) -> Any:
    """
    Upload a PDF paper for ingestion into the knowledge base.
    Extracts text + figures, embeds, stores in DB.
    """
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"ok": False, "error": "Only PDF files are supported."}, status_code=400)

    pdf_bytes = await file.read()
    tag_list  = [t.strip() for t in tags.split(",") if t.strip()] + ["user_upload"]

    # Try to extract text with pdfplumber or pymupdf
    full_text = ""
    title = Path(file.filename).stem.replace("_", " ").replace("-", " ")

    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages[:30]:   # limit to 30 pages
                t = page.extract_text() or ""
                pages_text.append(t)
            full_text = "\n\n".join(pages_text)
            # Try to extract title from first page
            if pages_text:
                first_lines = pages_text[0].strip().split("\n")
                if first_lines:
                    title = first_lines[0][:200]
    except ImportError:
        log.info("pdfplumber not installed, using filename as title")
    except (ValueError, KeyError, TypeError) as e:
        log.warning("PDF text extraction failed: %s", e)

    # Ingest text
    doc_id = await ingest_paper(
        title=title,
        abstract=full_text[:1500] if full_text else f"Uploaded: {file.filename}",
        full_text=full_text or None,
        source_type="upload",
        source_id=f"upload_{int(time.time())}_{file.filename[:40]}",
        tags=tag_list,
    )

    # Ingest figures
    n_figs = 0
    if doc_id:
        figures_dir = Path("./data/figures") / str(doc_id)
        n_figs = await _extract_and_embed_figures(doc_id, pdf_bytes, figures_dir)

    return JSONResponse({
        "ok": True,
        "doc_id": doc_id,
        "filename": file.filename,
        "n_text_chars": len(full_text),
        "n_figures_extracted": n_figs,
        "tags": tag_list,
    })


@router.post("/chat/knowledge/daily")
async def trigger_daily_update(request: Request) -> Dict[str, Any]:
    """Manually trigger the daily literature update."""
    asyncio.create_task(run_daily_update(trigger="manual"))
    return JSONResponse({"ok": True, "message": "Daily update started in background."})


@router.get("/chat/knowledge/status")
async def knowledge_status() -> Dict[str, Any]:
    """Return knowledge-base statistics."""
    if not _DB_OK or AsyncSessionLocal is None:
        return JSONResponse({"ok": False, "error": "DB not available"})

    from sqlalchemy import select, func

    try:
        async with AsyncSessionLocal() as s:
            n_docs   = (await s.execute(select(func.count(KnowledgeDoc.id)))).scalar() or 0
            n_chunks = (await s.execute(select(func.count(KnowledgeChunk.id)))).scalar() or 0
            n_figs   = (await s.execute(select(func.count(KnowledgeFigure.id)))).scalar() or 0

            last_run = (await s.execute(
                select(LiteratureUpdateLog)
                .order_by(LiteratureUpdateLog.run_at.desc())
                .limit(1)
            )).scalar_one_or_none()

        return JSONResponse({
            "ok": True,
            "n_docs":   n_docs,
            "n_chunks": n_chunks,
            "n_figures": n_figs,
            "last_update": {
                "run_at":    str(last_run.run_at) if last_run else None,
                "trigger":   last_run.trigger if last_run else None,
                "n_new":     last_run.n_new_docs if last_run else 0,
                "duration_s": last_run.duration_s if last_run else 0,
            },
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
