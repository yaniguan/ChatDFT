from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Optional
from server.utils.openai_wrapper import chatgpt_call
from server.db import AsyncSessionLocal, Hypothesis
from datetime import datetime
import json, re

router = APIRouter()

class HypothesisResult(BaseModel):
    hypothesis: str
    confidence: float
    tags: Optional[List[str]] = None

HYPOTHESIS_SYSTEM_PROMPT = """
You are a scientific DFT copilot. 
Given a user's inquiry, parsed intent, and related scientific knowledge, 
propose a concise, actionable hypothesis or research objective for the next step in the workflow.

- Use clear, academic English.
- Focus on the user's material/object and the specified intent/task.
- If knowledge indicates a challenge or caveat, mention it.
- Output only structured JSON.

JSON format:
{
  "hypothesis": "...",
  "confidence": 0.0,
  "tags": ["dos", "bandgap"]
}
"""

async def call_gpt4o_hypothesis(query, intent=None, knowledge=None):
    context = ""
    if knowledge:
        context = f"\nRelated knowledge:\n{knowledge}"
    user_content = f"User inquiry: {query}\nIntent: {intent or 'unknown'}{context}"
    messages = [
        {"role": "system", "content": HYPOTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    text = await chatgpt_call(messages)
    text = text.strip()
    # 兼容 markdown code block
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    match = re.search(r'\{[\s\S]*?\}', text)
    if match:
        text = match.group()
    try:
        obj = json.loads(text)
        return obj
    except Exception:
        return {"hypothesis": query, "confidence": 0.2, "tags": []}

# 数据库存储
async def save_hypothesis(session_id, message_id, intent, hypothesis, confidence, tags, agent="gpt-4o"):
    async with AsyncSessionLocal() as session:
        entry = Hypothesis(
            session_id=session_id,
            message_id=message_id,
            intent=intent,
            hypothesis=hypothesis,
            confidence=confidence,
            tags=",".join(tags) if tags else None,
            agent=agent,
            created_at=datetime.utcnow()
        )
        session.add(entry)
        await session.commit()

@router.post("/chat/hypothesis", response_model=HypothesisResult)
async def chat_hypothesis(request: Request):
    """
    入参推荐结构（前端/后端调用时推荐携带）:
    {
        "query": "...",           # 用户输入
        "intent": "...",          # intent agent 输出
        "knowledge": "...",       # knowledge agent 输出
        "session_id": 1,          # 当前 session
        "message_id": 123         # 当前消息 id
    }
    """
    data = await request.json()
    user_query = data.get("query", "")
    intent = data.get("intent", None)
    knowledge = data.get("knowledge", "")
    session_id = data.get("session_id")
    message_id = data.get("message_id")
    # LLM
    hypo_obj = await call_gpt4o_hypothesis(user_query, intent, knowledge)
    # 写入数据库
    await save_hypothesis(
        session_id=session_id,
        message_id=message_id,
        intent=intent,
        hypothesis=hypo_obj.get("hypothesis"),
        confidence=hypo_obj.get("confidence", 0.7),
        tags=hypo_obj.get("tags", []),
        agent="gpt-4o"
    )
    return hypo_obj