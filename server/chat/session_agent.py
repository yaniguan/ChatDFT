# server/chat/session_agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from fastapi import APIRouter, Request, HTTPException
from typing import Any, Dict
from sqlalchemy import select, update, delete
from server.db import AsyncSessionLocal, ChatSession  # 确保 db.py 里有 ChatSession 模型
from server.db import AsyncSessionLocal, ChatSession, ChatMessage
from sqlalchemy import select, desc

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
@router.post("/state")
async def state(req: Request):
    body = await req.json()
    sid = body.get("id")
    if not sid:
        raise HTTPException(400, "id required")
    # 这里先返回空壳，前端会 fallback；后面你可以接到 records/hypothesis 的缓存
    return {"ok": True, "id": sid}