# server/execution/utils/task_status.py
# -*- coding: utf-8 -*-
"""
Real-time workflow_task status transitions.

Usage (inside any agent)::

    from server.execution.utils.task_status import emit_task_status
    await emit_task_status(task_id, "running")
    ...
    await emit_task_status(task_id, "done", output_data={"poscar": ...})
    # or on failure:
    await emit_task_status(task_id, "failed", error_msg=str(exc))

The function is fire-and-forget when called from a sync context via
``emit_task_status_sync()``.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB layer — graceful no-op if unavailable
# ---------------------------------------------------------------------------
try:
    from sqlalchemy import select
    from server.db import AsyncSessionLocal, WorkflowTask
    _DB_OK = True
except ImportError as _e:
    log.warning("task_status: DB unavailable (%s) — status transitions disabled", _e)
    _DB_OK = False
    AsyncSessionLocal = None  # type: ignore
    WorkflowTask = None       # type: ignore


# ---------------------------------------------------------------------------
# Core async helper
# ---------------------------------------------------------------------------

async def emit_task_status(
    task_id: int,
    status: str,
    *,
    output_data: Optional[Dict[str, Any]] = None,
    error_msg: Optional[str] = None,
    run_time: Optional[float] = None,
) -> bool:
    """
    Update WorkflowTask.status (and optional fields) in PostgreSQL.

    Parameters
    ----------
    task_id     : WorkflowTask.id
    status      : "queued" | "running" | "done" | "failed"
    output_data : dict to store in WorkflowTask.output_data
    error_msg   : stored in WorkflowTask.error_msg on failure
    run_time    : wall-clock seconds for the task

    Returns True on success, False if DB unavailable or task not found.
    """
    if not _DB_OK or task_id is None:
        return False

    _VALID = {"queued", "running", "done", "failed", "idle"}
    if status not in _VALID:
        log.warning("emit_task_status: unknown status %r (must be one of %s)", status, _VALID)
        return False

    try:
        async with AsyncSessionLocal() as session:
            stmt = select(WorkflowTask).where(WorkflowTask.id == task_id)
            res = await session.execute(stmt)
            task = res.scalar_one_or_none()
            if task is None:
                log.warning("emit_task_status: task_id=%d not found", task_id)
                return False

            task.status = status
            task.updated_at = datetime.now(timezone.utc)
            if output_data is not None:
                task.output_data = output_data
            if error_msg is not None:
                task.error_msg = error_msg
            if run_time is not None:
                task.run_time = run_time

            await session.commit()
            log.debug("workflow_task %d → %s", task_id, status)
            return True

    except Exception as exc:
        log.warning("emit_task_status failed for task_id=%d: %s", task_id, exc)
        return False


# ---------------------------------------------------------------------------
# Sync convenience wrapper (for use inside sync agent methods)
# ---------------------------------------------------------------------------

def emit_task_status_sync(
    task_id: int,
    status: str,
    *,
    output_data: Optional[Dict[str, Any]] = None,
    error_msg: Optional[str] = None,
    run_time: Optional[float] = None,
) -> None:
    """
    Fire-and-forget version for sync callers.
    Schedules the coroutine if an event loop is running; otherwise runs it
    directly via ``asyncio.run()``.
    """
    coro = emit_task_status(
        task_id, status,
        output_data=output_data,
        error_msg=error_msg,
        run_time=run_time,
    )
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        try:
            asyncio.run(coro)
        except Exception:
            pass
