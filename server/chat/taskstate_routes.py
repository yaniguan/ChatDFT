# server/chat/taskstate_routes.py
"""
Persistent plan-task state: save & restore per-task step completions
so that browser refresh / server restart preserves everything.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import PlanTaskState, get_session

router = APIRouter(prefix="/chat/task_state", tags=["task_state"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TaskStateSave(BaseModel):
    session_id:      int
    task_plan_id:    int
    task_name:       Optional[str]   = None
    poscar:          Optional[str]   = None
    plot_png_b64:    Optional[str]   = None
    all_configs:     Optional[List]  = None
    selected_config: Optional[int]   = 0
    scripts:         Optional[Dict]  = None
    job_id:          Optional[str]   = None
    remote_path:     Optional[str]   = None
    results:         Optional[Dict]  = None
    energy_eV:       Optional[float] = None


class TaskStateOut(TaskStateSave):
    id: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/save")
async def save_task_state(
    body: TaskStateSave,
    db: AsyncSession = Depends(get_session),
) -> Any:
    """Upsert one task-state row (insert or update on conflict)."""
    # Build the values dict, excluding unset/None to avoid overwriting good data
    vals = body.model_dump(exclude_none=True)

    # Try to find existing row
    row = (await db.execute(
        select(PlanTaskState).where(
            PlanTaskState.session_id   == body.session_id,
            PlanTaskState.task_plan_id == body.task_plan_id,
        )
    )).scalars().first()

    if row is None:
        row = PlanTaskState(**vals)
        db.add(row)
    else:
        for k, v in vals.items():
            setattr(row, k, v)

    await db.commit()
    await db.refresh(row)
    return {"ok": True, "id": row.id}


@router.post("/list")
async def list_task_states(
    body: Dict[str, Any],
    db: AsyncSession = Depends(get_session),
) -> Any:
    """Return all task-state rows for a session."""
    session_id = body.get("session_id")
    if not session_id:
        return {"ok": False, "detail": "session_id required"}

    rows = (await db.execute(
        select(PlanTaskState)
        .where(PlanTaskState.session_id == session_id)
        .order_by(PlanTaskState.task_plan_id)
    )).scalars().all()

    out = []
    for r in rows:
        out.append({
            "id":              r.id,
            "session_id":      r.session_id,
            "task_plan_id":    r.task_plan_id,
            "task_name":       r.task_name,
            "poscar":          r.poscar,
            "plot_png_b64":    r.plot_png_b64,
            "all_configs":     r.all_configs,
            "selected_config": r.selected_config,
            "scripts":         r.scripts,
            "job_id":          r.job_id,
            "remote_path":     r.remote_path,
            "results":         r.results,
            "energy_eV":       r.energy_eV,
        })
    return {"ok": True, "states": out}
