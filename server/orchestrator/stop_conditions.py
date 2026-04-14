# server/orchestrator/stop_conditions.py
"""
Stop policy for the orchestration loop.

Three independent triggers — first one fires:

  1. ``max_iterations``     hard wall on total iterations
  2. ``confidence``         reward-derived confidence ≥ threshold
  3. ``no_new_actions``     N consecutive refine rounds proposed nothing

Returns a string describing which one fired, or ``None`` if we keep going.
"""
from __future__ import annotations

import enum
from typing import Optional

from server.orchestrator.state import OrchestrationState


class StopReason(str, enum.Enum):
    MAX_ITERATIONS = "max_iterations_reached"
    CONFIDENCE = "confidence_threshold_reached"
    NO_NEW_ACTIONS = "no_new_actions_streak"
    BUDGET_EXHAUSTED = "all_agent_budgets_exhausted"
    USER_STOPPED = "user_stopped"
    ERROR = "error"


def check_stop(state: OrchestrationState) -> Optional[str]:
    """
    Evaluate global + per-agent stop conditions.

    Returns the matching :class:`StopReason` value, or ``None`` to continue.
    """
    if state.iteration >= state.max_iterations:
        return StopReason.MAX_ITERATIONS.value

    if state.confidence() >= state.confidence_threshold:
        return StopReason.CONFIDENCE.value

    if state.no_new_actions_streak >= state.no_new_actions_threshold:
        return StopReason.NO_NEW_ACTIONS.value

    # Per-agent: if every gated budget says "done", we have nothing left to do.
    # We only consider the gated agents (structure + parameter); hpc/post are
    # uncapped and don't count.
    gated = [b for b in state.budgets.values()
             if b.use_reward_gate or b.max_rounds is not None]
    if gated and all(b.is_exhausted(state.reward_ema()) for b in gated):
        return StopReason.BUDGET_EXHAUSTED.value

    return None
