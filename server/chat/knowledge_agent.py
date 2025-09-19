# -*- coding: utf-8 -*-
"""
Knowledge Agent — 只用 arXiv 抓文献（python-arxiv 库）
可被 /chat/knowledge 路由调用，也可被 PlanManager 直接调用（run_knowledge）
"""

from __future__ import annotations
from fastapi import APIRouter, Request, HTTPException
from typing import Any, Dict, List, Optional
import re, math

import arxiv  # pip install arxiv

# ---- DB models（按你项目里的路径）----
from server.db_last import AsyncSessionLocal, ChatSession, ChatMessage, Knowledge, WorkflowTask
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

router = APIRouter()

# --------------------- 轻量字符串工具 ---------------------
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()

def _mk_query(query: str, intent: Dict[str, Any]) -> str:
    parts = [query]
    f = intent or {}

    for k in ("reaction", "reaction_type", "domain"):
        v = f.get(k)
        if v: parts.append(str(v))

    sys = (f.get("system") or {}) if "system" in f else {}
    for k in ("material","catalyst","facet","molecule","defect"):
        v = sys.get(k) if isinstance(sys, dict) else f.get(k)
        if v: parts.append(str(v))

    cond = f.get("conditions") or {}
    for k in ("pH","potential","temperature","electrolyte","solvent"):
        v = cond.get(k)
        if v: parts.append(str(v))

    tps = f.get("target_properties") or f.get("targets") or []
    if isinstance(tps, list): parts += tps[:3]

    return " ".join(str(x) for x in parts if x)

def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())

def _dedup(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set(); out=[]
    for r in records:
        key = (r.get("doi") or "").lower() or _norm_title(r.get("title") or "")
        if key and key not in seen:
            seen.add(key); out.append(r)
    return out

def _score(record: Dict[str, Any], intent: Dict[str, Any], q: str) -> float:
    title = (record.get("title") or "").lower()
    abstr = (record.get("abstract") or "").lower()
    text = f"{title} {abstr}"
    hits = 0
    keys: List[str] = []
    f = intent or {}
    keys += [str(f.get("reaction") or ""), str(f.get("reaction_type") or ""), str(f.get("domain") or "")]
    sys = (f.get("system") or {}) if "system" in f else {}
    for k in ("material","catalyst","facet","molecule","defect"):
        v = sys.get(k) if isinstance(sys, dict) else f.get(k)
        if v: keys.append(str(v))
    for k in (f.get("target_properties") or []):
        keys.append(str(k))
    keys.append(q)
    for k in keys:
        k = (k or "").lower().strip()
        if k and k in text:
            hits += 1
    # arXiv没引用数，这里只用匹配得分
    return float(hits)

# --------------------- DB Helper ---------------------
async def _ensure_session_exists(session_id: int) -> None:
    async with AsyncSessionLocal() as s:
        row = (await s.execute(select(ChatSession).where(ChatSession.id == session_id))).scalars().first()
        if not row:
            raise HTTPException(404, "session not found")

async def _add_message(session_id: int, role: str, content: str, **extra) -> int:
    async with AsyncSessionLocal() as s:
        m = ChatMessage(session_id=session_id, role=role, content=content, **extra)
        s.add(m)
        await s.flush()
        mid = m.id
        await s.commit()
        return mid

async def _upsert_knowledge(rec: Dict[str, Any]) -> int:
    async with AsyncSessionLocal() as s:
        doi = (rec.get("doi") or "").lower() or None

        # by DOI
        if doi:
            row = (await s.execute(select(Knowledge).where(Knowledge.doi == doi))).scalars().first()
            if row:
                row.title   = row.title   or rec.get("title")
                row.content = row.content or rec.get("abstract")
                row.source_type = row.source_type or rec.get("source_type")
                row.source_id   = row.source_id   or rec.get("source_id")
                row.url     = row.url     or rec.get("url")
                row.tags    = row.tags    or rec.get("venue")
                await s.flush(); await s.commit()
                return row.id

        st, sid = rec.get("source_type"), rec.get("source_id")
        if st and sid:
            row = (await s.execute(
                select(Knowledge).where(Knowledge.source_type==st, Knowledge.source_id==sid)
            )).scalars().first()
            if row:
                row.doi    = row.doi    or doi
                row.title  = row.title  or rec.get("title")
                row.content= row.content or rec.get("abstract")
                row.url    = row.url    or rec.get("url")
                row.tags   = row.tags   or rec.get("venue")
                await s.flush(); await s.commit()
                return row.id

        title = rec.get("title")
        if title:
            row = (await s.execute(select(Knowledge).where(Knowledge.title == title))).scalars().first()
            if row:
                row.doi    = row.doi    or doi
                row.source_type = row.source_type or st
                row.source_id   = row.source_id   or sid
                row.content= row.content or rec.get("abstract")
                row.url    = row.url    or rec.get("url")
                row.tags   = row.tags   or rec.get("venue")
                await s.flush(); await s.commit()
                return row.id

        row = Knowledge(
            title   = title,
            content = rec.get("abstract"),
            source_type = st,
            source_id   = sid,
            url     = rec.get("url"),
            doi     = doi,
            tags    = rec.get("venue"),
        )
        s.add(row)
        try:
            await s.flush()
            kid = row.id
            await s.commit()
            return kid
        except IntegrityError:
            await s.rollback()
            if doi:
                row = (await s.execute(select(Knowledge).where(Knowledge.doi == doi))).scalars().first()
                if row: return row.id
            if st and sid:
                row = (await s.execute(select(Knowledge).where(Knowledge.source_type==st, Knowledge.source_id==sid))).scalars().first()
                if row: return row.id
            return -1

# --------------------- arXiv 抓取（使用 python-arxiv） ---------------------
def _fetch_arxiv_lib(q: str, limit: int) -> List[Dict[str, Any]]:
    """
    用 python-arxiv 同步拉取。arxiv.Client 默认就很好用，这里简单封装。
    """
    search = arxiv.Search(
        query=q,
        max_results=min(limit, 50),
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client(page_size=25, delay_seconds=0.5)  # 速率限制，避免被封
    records: List[Dict[str, Any]] = []
    for r in client.results(search):
        # r has: title, summary, entry_id(=abs url), published, pdf_url, authors, primary_category, doi (sometimes None)
        year = r.published.year if r.published else None
        doi = (r.doi or "").lower() if hasattr(r, "doi") else None
        records.append({
            "title": _norm(r.title),
            "venue": "arXiv",
            "year": year,
            "url": r.entry_id,                 # abs 链接
            "pdf_url": getattr(r, "pdf_url", None),
            "source_type": "arxiv",
            "source_id": r.entry_id.split("/")[-1],  # 近似 arxiv id
            "doi": doi,
            "abstract": _norm(r.summary),
            "citations": 0,  # arXiv没引用数
        })
    return records

# --------------------- 核心：供 PlanManager 直接调用 ---------------------
async def run_knowledge(
    query: str,
    intent: Dict[str, Any] | None = None,
    limit: int = 10,
    fast: bool = False,              # 兼容参数（无意义）
    session_id: int | None = None,
    return_bundle: bool = False,     # 兼容参数（这里不返回 bundle）
) -> Dict[str, Any]:
    intent = intent or {}
    q = _mk_query(_norm(query), intent)

    # 只用 arXiv
    try:
        records = _fetch_arxiv_lib(q, limit)
        stats = {"arxiv": len(records)}
        errors: List[str] = []
    except Exception as e:
        records, stats, errors = [], {}, [f"arxiv: {e}"]

    # 去重 + 打分 + 截断
    records = _dedup(records)
    for r in records:
        r["relevance"] = _score(r, intent, q)
    records.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    records = records[:limit]

    # 入库
    knowledge_ids: List[int] = []
    for r in records:
        kid = await _upsert_knowledge(r)
        if kid and kid > 0:
            knowledge_ids.append(kid)

    summary = f"Retrieved {len(records)} references from arXiv."

    msg_id = None
    if session_id is not None:
        msg_id = await _add_message(
            int(session_id), role="assistant", content=summary,
            msg_type="knowledge", intent_stage="knowledge",
            intent_area=(intent.get("domain") if isinstance(intent, dict) else None),
            references={"knowledge_ids": knowledge_ids, "records": records[:5]},
        )

    out = {
        "ok": True,
        "result": summary,
        "records": records,
        "knowledge_ids": knowledge_ids,
        "source_stats": stats,
        "errors": errors,
        "assistant_message_id": msg_id,
    }
    # 不返回 session bundle，保持轻量
    return out

# --------------------- FastAPI 路由 ---------------------
@router.post("/chat/knowledge")
async def knowledge_route(request: Request):
    body = await request.json()
    return await run_knowledge(
        query=body.get("query") or "",
        intent=body.get("intent") or {},
        limit=int(body.get("limit") or 10),
        fast=bool(body.get("fast") or False),
        session_id=body.get("session_id"),
        return_bundle=bool(body.get("return_bundle") or False),
    )