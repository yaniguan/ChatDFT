# server/chat/history_agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
import json
import math

from sqlalchemy import select, or_, and_, desc
from server.db_last import (
    AsyncSessionLocal,
    ChatSession,
    ChatMessage,
    WorkflowTask,
)

router = APIRouter()

# -------------------------- utils --------------------------
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()

def _collect_keywords(intent: Dict[str, Any], query: str) -> List[str]:
    ks: List[str] = []
    f = intent or {}
    ks += [str(f.get("domain") or ""), str(f.get("reaction") or ""), str(f.get("reaction_type") or "")]
    sys = (f.get("system") or {})
    if isinstance(sys, dict):
        for k in ("material", "catalyst", "facet", "defect", "molecule"):
            v = sys.get(k)
            if v:
                ks.append(str(v))
    for t in (f.get("target_properties") or []):
        ks.append(str(t))
    if query:
        ks.append(query)
    # 去重、清洗
    out=[]
    seen=set()
    for k in ks:
        k=_norm(k).lower()
        if k and k not in seen:
            seen.add(k); out.append(k)
    return out

def _score_text(text: str, keys: List[str]) -> float:
    if not text:
        return 0.0
    t = text.lower()
    hits = sum(1 for k in keys if k and k in t)
    return hits

def _mk_snippet(m: ChatMessage, maxlen: int = 280) -> str:
    c = m.content or ""
    c = re.sub(r"\s+", " ", c).strip()
    return (c[: maxlen - 3] + "...") if len(c) > maxlen else c

def _mk_prompt_snippet(messages: List[ChatMessage], tasks: List[WorkflowTask]) -> str:
    lines = []
    if tasks:
        lines.append("## Prior workflow steps:")
        for t in tasks[:6]:
            lines.append(f"- [{t.status}] {t.step_order or ''} {t.name} (agent={t.agent}, engine={t.engine})")
    if messages:
        lines.append("## Prior notes:")
        for m in messages[:8]:
            lines.append(f"- {m.role}: {_mk_snippet(m)}")
    return "\n".join(lines) if lines else ""

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

# -------------------------- route --------------------------
@router.post("/chat/history")
async def history_route(request: Request):
    """
    Input:
      {
        "session_id": int?,              # 当前会话（如提供则把摘要写回）
        "query": str?,                   # 原始查询
        "intent": dict?,                 # 结构化 intent（用于检索和排序）
        "limit": int=50,                 # 返回的历史条数上限（消息+任务各自限制）
        "scope": "global"|"session",     # 仅当前会话 / 全库检索（默认 global）
        "return_session_bundle": bool?   # 是否回传当前会话最近消息/任务
      }

    Output:
      {
        "ok": True,
        "matches": {
          "messages": [ {id, session_id, role, content, created_at, score, intent_stage, ...}, ... ],
          "tasks":    [ {id, session_id, name, agent, status, step_order, score}, ... ]
        },
        "prompt_snippet": "markdown",
        "assistant_message_id": int|null,
        "session_bundle": {...}?   # 可选
      }
    """
    body = await request.json()
    sid   = body.get("session_id")
    query = _norm(body.get("query"))
    intent= body.get("intent") or {}
    limit = int(body.get("limit") or 50)
    scope = (body.get("scope") or "global").lower()
    return_bundle = bool(body.get("return_session_bundle"))

    if sid is not None:
        await _ensure_session_exists(int(sid))

    keys = _collect_keywords(intent, query)

    # ------------------- 查询历史 -------------------
    async with AsyncSessionLocal() as s:
        # 1) 消息检索
        msg_stmt = select(ChatMessage)
        if scope == "session" and sid is not None:
            msg_stmt = msg_stmt.where(ChatMessage.session_id == int(sid))
        # 只取对后续有帮助的消息类型：intent/knowledge/plan/execute/assistant/user
        msg_stmt = msg_stmt.order_by(desc(ChatMessage.created_at)).limit(500)
        msgs_all: List[ChatMessage] = (await s.execute(msg_stmt)).scalars().all()

        # 2) 任务检索
        task_stmt = select(WorkflowTask)
        if scope == "session" and sid is not None:
            task_stmt = task_stmt.where(WorkflowTask.session_id == int(sid))
        task_stmt = task_stmt.order_by(desc(WorkflowTask.created_at)).limit(500)
        tasks_all: List[WorkflowTask] = (await s.execute(task_stmt)).scalars().all()

    # ------------------- 打分排序 -------------------
    scored_msgs = []
    for m in msgs_all:
        sc = _score_text((m.content or "") + " " + json.dumps(m.references or {}), keys)
        if sc > 0 or not keys:   # 没有关键词时返回近期
            scored_msgs.append((sc, m))
    scored_msgs.sort(key=lambda x: (x[0], x[1].created_at or datetime.min), reverse=True)
    top_msgs = [x[1] for x in scored_msgs[:limit]]

    scored_tasks = []
    for t in tasks_all:
        blob = " ".join([
            t.name or "",
            t.description or "",
            t.agent or "",
            t.engine or "",
            json.dumps(t.input_data or {}),
            json.dumps(t.output_data or {}),
        ])
        sc = _score_text(blob, keys)
        if sc > 0 or not keys:
            scored_tasks.append((sc, t))
    scored_tasks.sort(key=lambda x: (x[0], x[1].created_at or datetime.min), reverse=True)
    top_tasks = [x[1] for x in scored_tasks[:limit]]

    # ------------------- 汇总输出 -------------------
    def _msg_out(m: ChatMessage, score: float) -> Dict[str, Any]:
        return {
            "id": m.id,
            "session_id": m.session_id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "score": score,
            "intent_stage": m.intent_stage,
            "intent_area": m.intent_area,
            "specific_intent": m.specific_intent,
            "confidence": m.confidence,
        }

    def _task_out(t: WorkflowTask, score: float) -> Dict[str, Any]:
        return {
            "id": t.id,
            "session_id": t.session_id,
            "name": t.name,
            "agent": t.agent,
            "engine": t.engine,
            "status": t.status,
            "step_order": t.step_order,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "score": score,
        }

    out_msgs = [_msg_out(m, sc) for sc, m in scored_msgs[:limit]]
    out_tasks= [_task_out(t, sc) for sc, t in scored_tasks[:limit]]

    prompt_snippet = _mk_prompt_snippet(top_msgs, top_tasks)  # 简明可直接拼进后续 prompt

    # ------------------- 写一条 assistant 消息（可选） -------------------
    assistant_msg_id = None
    if sid is not None:
        summary = (
            f"Found {len(out_msgs)} relevant notes and {len(out_tasks)} prior tasks.\n\n"
            + (prompt_snippet or "(no succinct prior context)")
        )
        assistant_msg_id = await _add_message(
            int(sid),
            role="assistant",
            content=summary,
            msg_type="history",
            intent_stage="history",
            intent_area=(intent.get("domain") if isinstance(intent, dict) else None),
            references={
                "message_ids": [m["id"] for m in out_msgs],
                "task_ids": [t["id"] for t in out_tasks],
                "keywords": keys,
                "scope": scope,
            },
        )

    result = {
        "ok": True,
        "matches": {
            "messages": out_msgs,
            "tasks": out_tasks,
        },
        "prompt_snippet": prompt_snippet,
        "assistant_message_id": assistant_msg_id,
    }

    # ------------------- 可选返回 session bundle -------------------
    if sid is not None and return_bundle:
        # 复用 knowledge/intent 里的“bundle”需求，这里轻量实现一下
        async with AsyncSessionLocal() as s:
            msgs = (await s.execute(
                select(ChatMessage).where(ChatMessage.session_id==int(sid))
                .order_by(ChatMessage.created_at.desc()).limit(50)
            )).scalars().all()
            tasks = (await s.execute(
                select(WorkflowTask).where(WorkflowTask.session_id==int(sid))
                .order_by(WorkflowTask.step_order.asc())
            )).scalars().all()

        def _m(m: ChatMessage):
            return {
                "id": m.id, "role": m.role, "content": m.content,
                "msg_type": m.msg_type, "intent_stage": m.intent_stage,
                "intent_area": m.intent_area, "specific_intent": m.specific_intent,
                "confidence": m.confidence, "created_at": m.created_at.isoformat() if m.created_at else None,
                "references": m.references
            }
        def _t(t: WorkflowTask):
            return {
                "id": t.id, "name": t.name, "agent": t.agent, "status": t.status,
                "step_order": t.step_order, "created_at": t.created_at.isoformat() if t.created_at else None
            }
        result["session_bundle"] = {
            "messages": [_m(x) for x in reversed(msgs)],
            "tasks": [_t(x) for x in tasks],
        }

    return result