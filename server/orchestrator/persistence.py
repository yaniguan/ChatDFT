# server/orchestrator/persistence.py
"""
Persistence helpers for the orchestrator.

The loop itself is DB-agnostic (state is in memory).  Routes inject these
callables so the loop can be unit-tested without a database.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import update

from server.db import (
    AsyncSessionLocal,
    OrchestrationRun,
    OrchestrationStep,
    RewardSignalRow,
)
from server.orchestrator.state import IterationRecord, OrchestrationState

log = logging.getLogger(__name__)


async def create_run_row(
    *,
    session_id: int,
    intent: Dict[str, Any],
    config: Dict[str, Any],
) -> int:
    """Create a fresh OrchestrationRun row and return its id."""
    async with AsyncSessionLocal() as db:
        row = OrchestrationRun(
            session_id=session_id,
            status="running",
            iteration=0,
            confidence=0.5,
            reward_ema=0.0,
            config=config,
            intent=intent,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return int(row.id)


async def persist_iteration(state: OrchestrationState, rec: IterationRecord) -> None:
    """
    Insert one OrchestrationStep + any new reward signals for this iteration,
    and update the parent run's running counters.
    """
    async with AsyncSessionLocal() as db:
        # Step row
        step = OrchestrationStep(
            run_id=state.run_id,
            iteration=rec.iteration,
            executed_task_id=rec.executed_task_id,
            executed_agent=rec.executed_agent,
            success=rec.success,
            reward=rec.reward,
            confidence_after=rec.confidence_after,
            proposed_actions=rec.proposed_actions,
            rejected_actions=rec.rejected_actions,
            notes=rec.notes or None,
            started_at=datetime.utcfromtimestamp(rec.started_at),
            ended_at=datetime.utcfromtimestamp(rec.ended_at) if rec.ended_at else None,
        )
        db.add(step)

        # New reward signal (only the latest one — others were persisted
        # in earlier iterations).  Iteration N adds at most 1 signal.
        if state.reward_history:
            sig = state.reward_history[-1]
            db.add(RewardSignalRow(
                run_id=state.run_id,
                session_id=state.session_id,
                iteration=rec.iteration,
                species=sig.get("species"),
                surface=sig.get("surface"),
                reaction_type=sig.get("reaction_type"),
                predicted_trend=sig.get("predicted_trend"),
                dft_value=sig.get("dft_value"),
                reward=float(sig.get("reward", 0.0)),
                converged=bool(sig.get("converged", True)),
                details=sig.get("details"),
            ))

        # Update parent run counters
        await db.execute(
            update(OrchestrationRun)
            .where(OrchestrationRun.id == state.run_id)
            .values(
                iteration=state.iteration,
                confidence=state.confidence(),
                reward_ema=state.reward_ema(),
            )
        )
        await db.commit()


async def finalize_run_row(state: OrchestrationState) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(OrchestrationRun)
            .where(OrchestrationRun.id == state.run_id)
            .values(
                status=("error" if state.stop_reason == "error"
                        else "stopped" if state.stop_reason == "user_stopped"
                        else "done"),
                stop_reason=state.stop_reason,
                iteration=state.iteration,
                confidence=state.confidence(),
                reward_ema=state.reward_ema(),
                ended_at=(datetime.utcfromtimestamp(state.ended_at)
                          if state.ended_at else datetime.utcnow()),
                final_state=state.to_dict(),
            )
        )
        await db.commit()
