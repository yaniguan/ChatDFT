# server/chat/records_agent.py
"""Thin shim — /chat/records redirects to analyze_agent's /chat/analyze endpoint."""
from fastapi import APIRouter, Request
from typing import Any, Dict

router = APIRouter()


@router.post("/chat/records")
async def chat_records(request: Request) -> Dict[str, Any]:
    """
    Legacy endpoint kept for backward compatibility.
    Delegates to the analyze_agent pipeline which reads session DFT results,
    runs RAG, and returns structured analysis.
    """
    data = await request.json()
    session_id = data.get("session_id")
    focus = data.get("query") or data.get("focus") or "overall progress and next steps"

    # Build a synthetic request and call the analyze endpoint directly
    from fastapi import Request as _Request
    from starlette.datastructures import Headers
    import json as _json

    body = _json.dumps({"session_id": session_id, "focus": focus}).encode()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/chat/analyze",
        "headers": [(b"content-type", b"application/json")],
    }

    # Simpler: just import and call the core logic directly
    try:
        from server.chat.analyze_agent import _load_session_context, _build_analysis_prompt
        from server.utils.rag_utils import rag_context
        from server.utils.openai_wrapper import chatgpt_call

        ctx = await _load_session_context(session_id) if session_id else {}
        rag_text = await rag_context(focus, session_id=session_id, top_k=8)
        prompt = _build_analysis_prompt(ctx, focus, rag_text)

        answer = await chatgpt_call(
            [
                {"role": "system", "content": "You are a senior computational chemistry advisor."},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4o",
            temperature=0.3,
            max_tokens=1800,
        )
        return {"ok": True, "session_id": session_id, "focus": focus, "answer": answer}
    except Exception as e:
        return {"ok": False, "session_id": session_id, "detail": str(e)}
