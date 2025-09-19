# server/chat/records_agent.py
from fastapi import APIRouter, Request
from typing import Any, Dict
from server.utils.rag_utils import rag_context
from server.utils.openai_wrapper import chatgpt_call

router = APIRouter()

@router.post("/chat/records")
async def chat_records(request: Request) -> Dict[str, Any]:
    """
    Records Agent:
    从记录（jobs、计算结果、实验记录等）中检索上下文，
    并结合 RAG（可选知识库）生成分析结果或回答问题。
    """
    data = await request.json()
    query = data.get("query", "")
    session_id = data.get("session_id")  # 如果记录绑定了 session，可以用

    # 获取 RAG 上下文
    context = rag_context(query, session_id)

    # 调用 LLM
    answer = await chatgpt_call(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are the Records Agent. Use context and records to answer the query."},
            {"role": "user", "content": f"[Context]\n{context}\n\n[User Query]\n{query}"}
        ]
    )

    return {
        "session_id": session_id,
        "query": query,
        "answer": answer
    }