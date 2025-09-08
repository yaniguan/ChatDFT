# server/chat/session_agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from fastapi import APIRouter, Request, HTTPException
from typing import Any, Dict
from sqlalchemy import select, update, delete
from server.db_last import AsyncSessionLocal, ChatSession  # 确保 db.py 里有 ChatSession 模型
from server.db_last import AsyncSessionLocal, ChatSession, ChatMessage
from sqlalchemy import select, desc
import json

router = APIRouter(prefix="/chat/session")

def _row(s: ChatSession) -> Dict[str, Any]:
    return {
        "id": s.id,
        "name": s.name,
        "project": s.project,
        "tags": s.tags,
        "status": s.status,
        "pinned": s.pinned,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }
def _msg_row(m: ChatMessage) -> Dict[str, Any]:
    return {
        "id": m.id,
        "session_id": m.session_id,
        "msg_type": m.msg_type,        # "intent" | "hypothesis" | "plan" | "records" | ...
        "content": m.content,          # 建议存字符串（JSON 用 json.dumps 后的 str）
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }

@router.post("/messages")
async def messages(req: Request):
    """
    前端用它来“回灌”一个会话的历史数据。
    body = { "id": <session_id>, "limit": 500 }
    """
    body = await req.json()
    sid = body.get("id")
    if not sid:
        raise HTTPException(400, "id required")
    limit = int(body.get("limit", 200))
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == sid)
            .order_by(desc(ChatMessage.created_at))
            .limit(limit)
        )).scalars().all()
    return {"ok": True, "messages": [_msg_row(x) for x in rows]}
@router.post("/create")
async def create(req: Request):
    body = await req.json()
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name required")
    project = (body.get("project") or "").strip()
    tags = (body.get("tags") or "").strip()
    desc = (body.get("description") or "").strip()
    async with AsyncSessionLocal() as s:
        obj = ChatSession(name=name, project=project, tags=tags, description=desc, status="active", pinned=False)
        s.add(obj)
        await s.commit()
        await s.refresh(obj)
        return {"ok": True, "session_id": obj.id, "session": _row(obj)}

@router.post("/list")
async def list_sessions(req: Request):
    body = await req.json() if req.method == "POST" else {}
    limit = int(body.get("limit", 200))
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit))).scalars().all()
        return {"ok": True, "sessions": [_row(x) for x in rows]}

@router.post("/update")
async def update_session(req: Request):
    body = await req.json()
    sid = body.get("id")
    if not sid:
        raise HTTPException(400, "id required")
    fields = {}
    for k in ("name","project","tags","status","description","pinned"):
        if k in body: fields[k] = body[k]
    if not fields:
        return {"ok": True}  # no-op
    async with AsyncSessionLocal() as s:
        await s.execute(update(ChatSession).where(ChatSession.id==sid).values(**fields))
        await s.commit()
    return {"ok": True}

@router.post("/delete")
async def delete_session(req: Request):
    body = await req.json()
    sid = body.get("id")
    if not sid:
        raise HTTPException(400, "id required")
    async with AsyncSessionLocal() as s:
        await s.execute(delete(ChatSession).where(ChatSession.id==sid))
        await s.commit()
    return {"ok": True}

# 可选：用于前端 hydrate
# server/chat/session_agent.py  (替换原 /state)
from sqlalchemy import select, desc
from server.db_last import AsyncSessionLocal, ChatMessage

def _pick_latest(rows, mtype):
    for m in rows:
        if m.msg_type == mtype:
            return m
    return None

@router.post("/state")
async def state(req: Request):
    body = await req.json()
    sid = body.get("id")
    if not sid:
        raise HTTPException(400, "id required")

    async with AsyncSessionLocal() as s:
        rows = (await s.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == sid)
            .order_by(desc(ChatMessage.created_at))
            .limit(300)
        )).scalars().all()

    # 取各类最新一条
    intent_m   = _pick_latest(rows, "intent")
    hypo_md_m  = _pick_latest(rows, "hypothesis") or _pick_latest(rows, "hypothesis_md")
    plan_m     = _pick_latest(rows, "plan")
    rxn_m      = _pick_latest(rows, "rxn_network")
    wf_m       = _pick_latest(rows, "workflow_summary") or _pick_latest(rows, "records")
    hpc_m      = _pick_latest(rows, "hpc_jobs")

    def _parse(m):
        if not m: return None
        c = m.content or ""
        # DB content 可能是 JSON 字符串或纯文本
        try:
            import json
            return json.loads(c)
        except Exception:
            return c

    intent        = _parse(intent_m) or {}
    hypothesis_md = _parse(hypo_md_m) or ""
    plan_raw      = _parse(plan_m) or {}
    rxn           = _parse(rxn_m) or {}
    workflow_res  = _parse(wf_m) or {}
    hpc_jobs      = _parse(hpc_m)  # ← 反序列化 HPC 列表

    # 兼容前端字段
    plan_tasks = (plan_raw.get("tasks") if isinstance(plan_raw, dict) else None) or []
    snap = {
        "ok": True,
        "id": sid,
        "intent": intent,
        "hypothesis": hypothesis_md,
        "plan_raw": plan_raw if isinstance(plan_raw, dict) else {},
        "plan_tasks": plan_tasks,
        "rxn_net": rxn.get("elementary_steps") if isinstance(rxn, dict) else [],
        "intermediates": rxn.get("intermediates") if isinstance(rxn, dict) else [],
        "ts_candidates": rxn.get("ts_candidates") if isinstance(rxn, dict) else [],
        "coads_pairs": rxn.get("coads_pairs") if isinstance(rxn, dict) else [],
        "workflow_results": workflow_res.get("runs") if isinstance(workflow_res, dict) else [],
        "hpc_jobs": hpc_jobs,
    }
    return snap


@router.post("/hpc_jobs/save")
async def save_hpc_jobs(req: Request):

    # print("goes into the right position")

    body = await req.json()

    sid = body.get("id")
    jobs = body.get("jobs") or []

    if not sid:
        print("No id inside")
        raise HTTPException(400, "id required")
    try:
        print(jobs)
        payload = json.dumps(jobs, ensure_ascii=False)
    except Exception as e:
        print("json loading wrong")
        raise HTTPException(400, f"jobs not JSON-serializable: {e}")

    async with AsyncSessionLocal() as s:
        m = ChatMessage(session_id=int(sid), msg_type="hpc_jobs", content=payload, role='system')
        s.add(m)
        await s.commit()
    return {"ok": True, "count": len(jobs)}
