# server/orchestrator/routes.py
"""
HTTP surface for the closed-loop orchestrator.

  POST /api/orchestrator/start    — launch a background loop, returns run_id
  POST /api/orchestrator/status   — poll run state + iteration trace
  POST /api/orchestrator/stop     — request cooperative cancel
  POST /api/orchestrator/runs     — list runs for a session
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from server.db import (
    AsyncSessionLocal,
    ChatMessage,
    OrchestrationRun,
    OrchestrationStep,
)
from server.orchestrator.loop import ChatDFTOrchestrator
from server.orchestrator.persistence import (
    create_run_row,
    finalize_run_row,
    persist_iteration,
)
from server.orchestrator.state import IterationRecord, OrchestrationState

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/orchestrator", tags=["orchestrator"])


# ---------------------------------------------------------------------------
# In-process registry of running loops (so /stop can find them)
# ---------------------------------------------------------------------------
# Maps run_id → orchestrator instance + asyncio task handle
_RUNNING: Dict[int, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    session_id: int = Field(..., description="ChatSession.id to drive the loop for.")
    max_iterations: int = Field(10, ge=1, le=50)
    confidence_threshold: float = Field(0.85, ge=0.0, le=1.0)
    no_new_actions_threshold: int = Field(2, ge=1, le=10)
    auto_submit: bool = Field(False, description="If False, only prepare jobs (don't submit to HPC).")
    cluster: str = Field("hoffman2")
    engine: str = Field("vasp")


class StatusRequest(BaseModel):
    run_id: int


class StopRequest(BaseModel):
    run_id: int


class ListRequest(BaseModel):
    session_id: int
    limit: int = Field(20, ge=1, le=200)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _load_session_context(session_id: int) -> Dict[str, Any]:
    """
    Pull the latest intent / hypothesis / plan messages for the session.
    Mirrors what session_agent.state() does, but only the slice we need.
    """
    import json as _json

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(desc(ChatMessage.created_at))
            .limit(300)
        )).scalars().all()

    def _pick(mtype: str) -> Optional[ChatMessage]:
        for m in rows:
            if m.msg_type == mtype:
                return m
        return None

    def _parse(m: Optional[ChatMessage]) -> Any:
        if m is None:
            return None
        try:
            return _json.loads(m.content or "")
        except (ValueError, TypeError):
            return m.content

    intent       = _parse(_pick("intent")) or {}
    hypothesis   = _parse(_pick("hypothesis")) or ""
    rxn_network  = _parse(_pick("rxn_network")) or {}
    plan_raw     = _parse(_pick("plan")) or {}

    plan_tasks = []
    if isinstance(plan_raw, dict):
        plan_tasks = list(plan_raw.get("tasks") or [])
    for t in plan_tasks:
        t.setdefault("status", "pending")

    if not isinstance(hypothesis, str):
        # Some sessions store as dict — coerce to MD-ish string
        hypothesis = _json.dumps(hypothesis, ensure_ascii=False)

    hypothesis_graph: Dict[str, Any] = {}
    if isinstance(rxn_network, dict):
        hypothesis_graph = {
            "intermediates":   rxn_network.get("intermediates") or [],
            "reaction_network": rxn_network.get("elementary_steps") or [],
            "ts_candidates":   rxn_network.get("ts_candidates") or [],
            "coads_pairs":     rxn_network.get("coads_pairs") or [],
            "predictions":     rxn_network.get("predictions") or [],
            "system":          rxn_network.get("system") or {},
        }

    return {
        "intent": intent if isinstance(intent, dict) else {},
        "hypothesis_md": hypothesis,
        "hypothesis_graph": hypothesis_graph,
        "plan_tasks": plan_tasks,
    }


def _make_emit_callback(run_id: int):
    async def _on_iter(state: OrchestrationState, rec: IterationRecord) -> None:
        try:
            await persist_iteration(state, rec)
        except Exception:        # pragma: no cover
            log.exception("persist_iteration failed for run=%d iter=%d",
                          run_id, rec.iteration)
    return _on_iter


async def _drive(orch: ChatDFTOrchestrator) -> None:
    """Top-level coroutine for a background run."""
    try:
        await orch.run()
    finally:
        try:
            await finalize_run_row(orch.state)
        except Exception:        # pragma: no cover
            log.exception("finalize_run_row failed for run=%d", orch.state.run_id)
        _RUNNING.pop(orch.state.run_id, None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/start")
async def start(req: StartRequest) -> Dict[str, Any]:
    ctx = await _load_session_context(req.session_id)
    if not ctx["intent"]:
        raise HTTPException(400, "session has no intent message yet — run /chat/intent first")
    if not ctx["plan_tasks"] and not ctx["hypothesis_graph"].get("reaction_network"):
        raise HTTPException(
            400,
            "session has no plan tasks and no hypothesis network — "
            "the orchestrator needs at least one to start. "
            "Run /chat/hypothesis + /chat/plan first.",
        )

    config = req.model_dump()
    run_id = await create_run_row(
        session_id=req.session_id,
        intent=ctx["intent"],
        config=config,
    )

    orch = ChatDFTOrchestrator(
        run_id=run_id,
        session_id=req.session_id,
        intent=ctx["intent"],
        hypothesis_md=ctx["hypothesis_md"],
        hypothesis_graph=ctx["hypothesis_graph"],
        initial_plan_tasks=ctx["plan_tasks"],
        max_iterations=req.max_iterations,
        confidence_threshold=req.confidence_threshold,
        no_new_actions_threshold=req.no_new_actions_threshold,
        auto_submit=req.auto_submit,
        cluster=req.cluster,
        engine=req.engine,
        on_iteration_complete=_make_emit_callback(run_id),
    )

    task = asyncio.create_task(_drive(orch))
    _RUNNING[run_id] = {"orch": orch, "task": task}

    return {
        "ok": True,
        "run_id": run_id,
        "session_id": req.session_id,
        "config": config,
        "n_initial_tasks": len(ctx["plan_tasks"]),
    }


@router.post("/stop")
async def stop(req: StopRequest) -> Dict[str, Any]:
    entry = _RUNNING.get(req.run_id)
    if entry is None:
        # Already finished — just return current state from DB
        async with AsyncSessionLocal() as db:
            row = (await db.execute(
                select(OrchestrationRun).where(OrchestrationRun.id == req.run_id)
            )).scalar_one_or_none()
        if row is None:
            raise HTTPException(404, f"run {req.run_id} not found")
        return {"ok": True, "run_id": req.run_id, "status": row.status, "already_finished": True}

    entry["orch"].request_stop()
    return {"ok": True, "run_id": req.run_id, "status": "stop_requested"}


@router.post("/status")
async def status(req: StatusRequest) -> Dict[str, Any]:
    # Live state if running; otherwise read from DB
    entry = _RUNNING.get(req.run_id)
    if entry is not None:
        state: OrchestrationState = entry["orch"].state
        return {
            "ok": True,
            "run_id": req.run_id,
            "live": True,
            "state": state.to_dict(),
            "trace_tail": [r.to_dict() for r in state.iteration_trace[-10:]],
        }

    async with AsyncSessionLocal() as db:
        run = (await db.execute(
            select(OrchestrationRun).where(OrchestrationRun.id == req.run_id)
        )).scalar_one_or_none()
        if run is None:
            raise HTTPException(404, f"run {req.run_id} not found")

        steps = (await db.execute(
            select(OrchestrationStep)
            .where(OrchestrationStep.run_id == req.run_id)
            .order_by(OrchestrationStep.iteration.desc())
            .limit(10)
        )).scalars().all()

    return {
        "ok": True,
        "run_id": req.run_id,
        "live": False,
        "state": run.final_state or {
            "status": run.status,
            "stop_reason": run.stop_reason,
            "iteration": run.iteration,
            "confidence": run.confidence,
            "reward_ema": run.reward_ema,
        },
        "trace_tail": [
            {
                "iteration": s.iteration,
                "executed_task_id": s.executed_task_id,
                "executed_agent": s.executed_agent,
                "success": s.success,
                "reward": s.reward,
                "confidence_after": s.confidence_after,
                "proposed_actions": s.proposed_actions or [],
                "rejected_actions": s.rejected_actions or [],
                "notes": s.notes,
            }
            for s in steps
        ],
    }


@router.post("/runs")
async def runs(req: ListRequest) -> Dict[str, Any]:
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            select(OrchestrationRun)
            .where(OrchestrationRun.session_id == req.session_id)
            .order_by(OrchestrationRun.id.desc())
            .limit(req.limit)
        )).scalars().all()

    return {
        "ok": True,
        "session_id": req.session_id,
        "runs": [
            {
                "id": r.id,
                "status": r.status,
                "stop_reason": r.stop_reason,
                "iteration": r.iteration,
                "confidence": r.confidence,
                "reward_ema": r.reward_ema,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "ended_at":   r.ended_at.isoformat()   if r.ended_at   else None,
            }
            for r in rows
        ],
    }


__all__ = ["router"]
