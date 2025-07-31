# server/chat/session_agent.py
from fastapi import APIRouter, Request
from server.db import AsyncSessionLocal
from sqlalchemy import select
from server.db import ChatSession  # 确认db.py里有这个ORM类

router = APIRouter()

@router.post("/chat/session/list")
async def session_list(request: Request):
    async with AsyncSessionLocal() as session:
        rows = await session.execute(select(ChatSession))
        sessions = [{"id": r.id, "name": r.name} for r in rows.scalars()]
        return {"sessions": sessions}

@router.post("/chat/session/create")
async def session_create(request: Request):
    data = await request.json()
    name = data.get("name")
    async with AsyncSessionLocal() as session:
        obj = ChatSession(name=name)
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {"session_id": obj.id}