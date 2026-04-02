# server/execution/utils/events.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import datetime

try:
    from server.db import AsyncSessionLocal, RunEvent as RunEventORM, LLMCall as LLMCallORM
except Exception:
    AsyncSessionLocal = None
    RunEventORM = LLMCallORM = None

async def post_event(evt: Dict[str, Any]) -> None:
    """把运行期的关键节点落库（可选也可以后扩 WebSocket 推送）。"""
    if AsyncSessionLocal is None or RunEventORM is None:
        return
    async with AsyncSessionLocal() as s:
        s.add(RunEventORM(
            run_id     = evt.get("run_id"),
            step_id    = evt.get("step_id"),
            phase      = evt.get("phase"),
            payload    = evt.get("payload") or {},
            created_at = datetime.utcnow(),
        ))
        await s.commit()

async def post_llm_call(model: str, prompt: Any, response: Any, meta: Optional[Dict[str, Any]] = None):
    if AsyncSessionLocal is None or LLMCallORM is None:
        return
    async with AsyncSessionLocal() as s:
        s.add(LLMCallORM(
            model=model,
            prompt=prompt,
            response=response,
            meta=meta or {},
            created_at=datetime.utcnow(),
        ))
        await s.commit()