from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Optional
from server.db import AsyncSessionLocal, ChatMessage, ChatSession

router = APIRouter()

class MessageOut(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    intent: Optional[str]
    created_at: str

class HistoryResult(BaseModel):
    messages: List[MessageOut]

@router.post("/chat/history", response_model=HistoryResult)
async def chat_history(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    limit = data.get("limit", 20)
    if not session_id:
        return {"messages": []}
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            ChatMessage.__table__.select()
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())  # 顺序！
            .limit(limit)
        )
        rows = result.fetchall()
        messages = [
            MessageOut(
                id=row.id,
                session_id=row.session_id,
                role=row.role,
                content=row.content,
                intent=row.intent,
                created_at=row.created_at.isoformat()
            )
            for row in rows
        ]
    return {"messages": messages}


@router.post("/chat/message/create")
async def create_message(request: Request):
    data = await request.json()
    session_id = data["session_id"]
    role = data["role"]
    content = data["content"]
    async with AsyncSessionLocal() as session:
        msg = ChatMessage(session_id=session_id, role=role, content=content)
        session.add(msg)
        await session.commit()
    return {"ok": True}