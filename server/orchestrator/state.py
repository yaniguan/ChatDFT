# server/orchestrator/state.py
"""
Mutable state for one orchestration run.

Holds:
  - the live hypothesis (markdown + structured graph)
  - the growing list of plan tasks
  - completed results (DFTResult-like dicts)
  - reward history (RewardSignal list)
  - per-agent budget counters
  - iteration trace (for the UI / DB)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Per-agent budgets (mutable; checked by stop_conditions)
# ---------------------------------------------------------------------------

@dataclass
class AgentBudget:
    """
    Per-agent rate limit.

    For the structure agent we stop when reward EMA is high AND the LLM
    has stopped proposing new structure actions for K rounds.

    For the parameter agent we cap pure rounds (one round = one refine
    cycle that produced any parameter-related action).
    """
    name: str
    max_rounds: Optional[int] = None        # absolute cap (e.g. parameter=3)
    rounds_used: int = 0
    no_new_streak: int = 0                  # rounds in a row with no action of this kind
    streak_threshold: int = 2                # K
    reward_ema_threshold: float = 0.8       # used together with streak
    use_reward_gate: bool = False           # True only for structure

    def step(self, *, proposed_this_round: bool, current_reward_ema: float) -> None:
        if proposed_this_round:
            self.rounds_used += 1
            self.no_new_streak = 0
        else:
            self.no_new_streak += 1

    def is_exhausted(self, current_reward_ema: float) -> bool:
        if self.max_rounds is not None and self.rounds_used >= self.max_rounds:
            return True
        if self.use_reward_gate:
            if (current_reward_ema >= self.reward_ema_threshold
                    and self.no_new_streak >= self.streak_threshold):
                return True
        return False


def default_budgets() -> Dict[str, AgentBudget]:
    """
    Default per-agent budgets matching the user-approved policy:

      structure → reward_ema ≥ 0.8 AND no_new_streak ≥ 2
      parameter → max 3 rounds
      hpc, post → no per-agent cap (follow tasks)
    """
    return {
        "structure": AgentBudget(
            name="structure",
            use_reward_gate=True,
            reward_ema_threshold=0.8,
            streak_threshold=2,
        ),
        "parameter": AgentBudget(
            name="parameter",
            max_rounds=3,
        ),
        "hpc": AgentBudget(name="hpc"),       # uncapped
        "post": AgentBudget(name="post"),     # uncapped
    }


# ---------------------------------------------------------------------------
# Iteration trace (one entry per loop iteration, for the UI/DB)
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    iteration: int
    started_at: float
    ended_at: Optional[float] = None
    executed_task_id: Optional[int] = None
    executed_agent: Optional[str] = None
    success: Optional[bool] = None
    reward: Optional[float] = None
    confidence_after: Optional[float] = None
    proposed_actions: List[Dict[str, Any]] = field(default_factory=list)
    rejected_actions: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "executed_task_id": self.executed_task_id,
            "executed_agent": self.executed_agent,
            "success": self.success,
            "reward": self.reward,
            "confidence_after": self.confidence_after,
            "proposed_actions": self.proposed_actions,
            "rejected_actions": self.rejected_actions,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Top-level state
# ---------------------------------------------------------------------------

@dataclass
class OrchestrationState:
    run_id: int
    session_id: int
    intent: Dict[str, Any]
    hypothesis_md: str = ""
    hypothesis_graph: Dict[str, Any] = field(default_factory=dict)
    plan_tasks: List[Dict[str, Any]] = field(default_factory=list)   # WorkflowTask-like dicts
    completed_results: List[Dict[str, Any]] = field(default_factory=list)
    reward_history: List[Dict[str, Any]] = field(default_factory=list)
    budgets: Dict[str, AgentBudget] = field(default_factory=default_budgets)

    iteration: int = 0
    no_new_actions_streak: int = 0
    iteration_trace: List[IterationRecord] = field(default_factory=list)

    # Configuration knobs (set by orchestrator constructor)
    max_iterations: int = 10
    confidence_threshold: float = 0.85
    no_new_actions_threshold: int = 2
    auto_submit: bool = False         # default: prepare-only
    cluster: str = "hoffman2"
    engine: str = "vasp"

    # Status
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    stop_reason: Optional[str] = None     # populated when loop terminates

    def confidence(self) -> float:
        """Mean reward over the last 5 signals, mapped to [0,1]."""
        if not self.reward_history:
            return 0.5
        recent = self.reward_history[-5:]
        if len(recent) < 3:
            return 0.5
        mean_r = sum(s["reward"] for s in recent) / len(recent)
        return max(0.0, min(1.0, (mean_r + 1.0) / 2.0))

    def reward_ema(self, alpha: float = 0.3) -> float:
        """EMA of all reward signals, raw [-1, 1] scale."""
        ema = 0.0
        for i, sig in enumerate(self.reward_history):
            r = sig["reward"]
            ema = r if i == 0 else alpha * r + (1 - alpha) * ema
        return ema

    def pending_tasks(self) -> List[Dict[str, Any]]:
        return [t for t in self.plan_tasks if t.get("status") in (None, "pending", "idle")]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "iteration": self.iteration,
            "stop_reason": self.stop_reason,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "max_iterations": self.max_iterations,
            "confidence_threshold": self.confidence_threshold,
            "no_new_actions_threshold": self.no_new_actions_threshold,
            "no_new_actions_streak": self.no_new_actions_streak,
            "confidence": self.confidence(),
            "reward_ema": self.reward_ema(),
            "n_plan_tasks": len(self.plan_tasks),
            "n_pending_tasks": len(self.pending_tasks()),
            "n_completed_results": len(self.completed_results),
            "n_reward_signals": len(self.reward_history),
            "budgets": {
                name: {
                    "rounds_used": b.rounds_used,
                    "no_new_streak": b.no_new_streak,
                    "max_rounds": b.max_rounds,
                    "use_reward_gate": b.use_reward_gate,
                    "exhausted": b.is_exhausted(self.reward_ema()),
                }
                for name, b in self.budgets.items()
            },
            "auto_submit": self.auto_submit,
            "cluster": self.cluster,
            "engine": self.engine,
        }
