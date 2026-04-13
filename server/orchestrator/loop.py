# server/orchestrator/loop.py
"""
ChatDFT closed-loop orchestrator — the engine that closes:

    intent → hypothesis → plan → execute → result
                ▲                              │
                └────── refine + propose ──────┘

One iteration:

  1. If no pending plan tasks, ask the LLM refiner for new actions.
  2. Validate actions, convert to WorkflowTask dicts, append to plan.
     Decrement the budget for each affected agent kind.
  3. Pick the highest-priority pending task.
  4. Execute it via the existing ``_pipeline`` in agent_routes.py.
  5. Parse the result into a structured row.
  6. Score the result vs the hypothesis prediction → reward signal.
  7. Append result + reward to state, persist iteration trace.
  8. Check stop conditions; if any fire, set ``stop_reason`` and exit.

The loop is fully async and runs as a background asyncio task launched
from the FastAPI route handler (no thread, no subprocess).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from server.execution.agent_coordinator import RewardTracker
from server.orchestrator.action_to_tasks import action_to_tasks, primary_agent_for
from server.orchestrator.actions import ProposedAction
from server.orchestrator.refine import refine_hypothesis
from server.orchestrator.reward import compute_reward_for_result
from server.orchestrator.state import (
    AgentBudget,
    IterationRecord,
    OrchestrationState,
    default_budgets,
)
from server.orchestrator.stop_conditions import StopReason, check_stop

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline shim — call the existing executor
# ---------------------------------------------------------------------------

async def _execute_task_via_pipeline(
    task: Dict[str, Any],
    *,
    auto_submit: bool,
    cluster: str,
    engine: str,
) -> Dict[str, Any]:
    """
    Run a single WorkflowTask through the existing ``_pipeline()``.

    We import inside the function to avoid a circular-ish dependency at
    module-load time (agent_routes imports lots of stuff and we don't want
    that pulled in unless the orchestrator actually runs).
    """
    from server.execution.agent_routes import _pipeline

    agent_name = task.get("agent") or ""
    opts = {
        "engine": engine,
        "cluster": cluster,
        "submit": auto_submit,
        "wait": auto_submit,
        "fetch": auto_submit,
        "do_post": auto_submit,
        "run_id": task.get("meta", {}).get("run_id"),
    }
    ok, out = await _pipeline(agent_name, task, opts)
    out["_pipeline_ok"] = ok
    return out


def _pipeline_out_to_result_row(
    *,
    task: Dict[str, Any],
    pipeline_out: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Turn the heterogeneous ``_pipeline`` output into a normalized result row
    that ``compute_reward_for_result`` and the iteration trace can consume.

    Key fields we care about: agent, species, surface, value (eV),
    converged, result_type, raw status string for trace.
    """
    payload = (task.get("params") or {}).get("payload") or {}
    species = payload.get("adsorbate") or payload.get("species") or payload.get("species_a") or ""
    surface = payload.get("surface") or payload.get("base_surface") or payload.get("facet") or ""

    post = pipeline_out.get("post") or {}
    value = None
    converged = bool(pipeline_out.get("_pipeline_ok"))
    result_type = "unknown"

    # Try a handful of common shapes from post_analysis_agent
    if isinstance(post, dict):
        # Adsorption energy first
        for key in ("E_ads", "e_ads", "adsorption_energy"):
            if key in post and isinstance(post[key], (int, float)):
                value = float(post[key])
                result_type = "adsorption_energy"
                break
        if value is None:
            for key in ("barrier", "Ea", "activation_barrier"):
                if key in post and isinstance(post[key], (int, float)):
                    value = float(post[key])
                    result_type = "activation_barrier"
                    break
        if value is None:
            for key in ("E", "energy", "total_energy"):
                if key in post and isinstance(post[key], (int, float)):
                    value = float(post[key])
                    result_type = "total_energy"
                    break
        converged = bool(post.get("converged", converged))

    return {
        "task_id": task.get("id"),
        "agent": task.get("agent"),
        "species": species,
        "surface": surface,
        "value": value,
        "result_type": result_type,
        "converged": converged,
        "status": pipeline_out.get("status"),
        "job_dir": pipeline_out.get("job_dir"),
    }


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------

class ChatDFTOrchestrator:
    """
    Drives one closed-loop run.

    Construct, then call ``await orchestrator.run()``.  All state lives on
    ``self.state`` (an :class:`OrchestrationState`); persistence is delegated
    to a callback the caller injects (so tests can run without a DB).
    """

    def __init__(
        self,
        *,
        run_id: int,
        session_id: int,
        intent: Dict[str, Any],
        hypothesis_md: str,
        hypothesis_graph: Dict[str, Any],
        initial_plan_tasks: List[Dict[str, Any]],
        max_iterations: int = 10,
        confidence_threshold: float = 0.85,
        no_new_actions_threshold: int = 2,
        auto_submit: bool = False,
        cluster: str = "hoffman2",
        engine: str = "vasp",
        budgets: Optional[Dict[str, AgentBudget]] = None,
        on_iteration_complete: Optional[Any] = None,    # async callable (state, record) -> None
    ):
        self.state = OrchestrationState(
            run_id=run_id,
            session_id=session_id,
            intent=intent,
            hypothesis_md=hypothesis_md,
            hypothesis_graph=hypothesis_graph,
            plan_tasks=list(initial_plan_tasks),
            budgets=budgets or default_budgets(),
            max_iterations=max_iterations,
            confidence_threshold=confidence_threshold,
            no_new_actions_threshold=no_new_actions_threshold,
            auto_submit=auto_submit,
            cluster=cluster,
            engine=engine,
        )
        self._tracker = RewardTracker()
        self._on_iter = on_iteration_complete
        self._stop_requested = False

    # -- public control surface --------------------------------------------------

    def request_stop(self) -> None:
        """Cooperative cancel — picked up at next iteration boundary."""
        self._stop_requested = True

    async def run(self) -> OrchestrationState:
        log.info(
            "orchestrator: run start id=%d session=%d max_iter=%d auto_submit=%s",
            self.state.run_id, self.state.session_id,
            self.state.max_iterations, self.state.auto_submit,
        )
        try:
            while True:
                if self._stop_requested:
                    self.state.stop_reason = StopReason.USER_STOPPED.value
                    break

                self.state.iteration += 1
                rec = IterationRecord(
                    iteration=self.state.iteration,
                    started_at=time.time(),
                )

                try:
                    await self._one_iteration(rec)
                except Exception as exc:
                    log.exception("orchestrator: iteration crashed")
                    rec.success = False
                    rec.notes = f"iteration crashed: {exc}"
                    self.state.stop_reason = StopReason.ERROR.value
                    rec.ended_at = time.time()
                    self.state.iteration_trace.append(rec)
                    await self._maybe_emit(rec)
                    break

                rec.ended_at = time.time()
                rec.confidence_after = self.state.confidence()
                self.state.iteration_trace.append(rec)
                await self._maybe_emit(rec)

                stop = check_stop(self.state)
                if stop:
                    self.state.stop_reason = stop
                    log.info("orchestrator: stop_reason=%s", stop)
                    break
        finally:
            self.state.ended_at = time.time()
            log.info(
                "orchestrator: run end id=%d iters=%d stop=%s confidence=%.2f",
                self.state.run_id, self.state.iteration,
                self.state.stop_reason, self.state.confidence(),
            )
        return self.state

    # -- one iteration -----------------------------------------------------------

    async def _one_iteration(self, rec: IterationRecord) -> None:
        # 1. Refill the plan if empty
        if not self.state.pending_tasks():
            await self._refine_round(rec)

        pending = self.state.pending_tasks()
        if not pending:
            # Refiner declined to add anything → bump streak but still record
            self.state.no_new_actions_streak += 1
            rec.notes = (rec.notes + " | " if rec.notes else "") + "no pending tasks; refiner returned empty"
            return

        # 2. Pick highest-priority pending task (priority comes from action sort)
        task = pending[0]
        task["status"] = "running"
        rec.executed_task_id = task.get("id")
        rec.executed_agent = task.get("agent")

        # 3. Execute
        try:
            pipeline_out = await _execute_task_via_pipeline(
                task,
                auto_submit=self.state.auto_submit,
                cluster=self.state.cluster,
                engine=self.state.engine,
            )
            ok = bool(pipeline_out.get("_pipeline_ok"))
            task["status"] = "done" if ok else "failed"
            task["output"] = pipeline_out
            rec.success = ok
        except Exception as exc:
            task["status"] = "failed"
            task["error"] = str(exc)
            rec.success = False
            rec.notes = f"execute failed: {exc}"
            return

        # 4. Normalize → reward
        result_row = _pipeline_out_to_result_row(task=task, pipeline_out=pipeline_out)
        self.state.completed_results.append(result_row)

        signal = compute_reward_for_result(
            state=self.state, tracker=self._tracker, result=result_row,
        )
        if signal is not None:
            rec.reward = signal.get("reward")

    # -- refine round ------------------------------------------------------------

    async def _refine_round(self, rec: IterationRecord) -> None:
        """
        Ask the LLM refiner for new actions, validate them, convert to tasks,
        update budgets.
        """
        ema_now = self.state.reward_ema()

        updated_md, accepted, rejected, llm_stop_reason = await refine_hypothesis(self.state)

        if updated_md:
            # Append delta to hypothesis (don't overwrite — preserve provenance)
            sep = "\n\n---\n\n" if self.state.hypothesis_md else ""
            self.state.hypothesis_md += f"{sep}*Refinement (iter {self.state.iteration}):*\n{updated_md}"

        rec.proposed_actions = [a.to_dict() for a in accepted]
        rec.rejected_actions = list(rejected)

        # Reject actions whose primary budgeted agent is exhausted.
        # Budgets only get *stepped* by surviving (kept) actions — proposing
        # something that's rejected for budget reasons must not eat budget.
        keep: List[ProposedAction] = []
        for a in accepted:
            agent = primary_agent_for(a.kind)
            budget = self.state.budgets.get(agent)
            if budget is not None and budget.is_exhausted(ema_now):
                rec.rejected_actions.append({
                    "raw": a.to_dict(),
                    "errors": [f"budget exhausted for agent={agent}"],
                })
                continue
            keep.append(a)

        # 2. Convert kept actions to tasks
        new_tasks: List[Dict[str, Any]] = []
        for a in keep:
            spawned = action_to_tasks(
                a, existing_tasks=self.state.plan_tasks, session_id=self.state.session_id,
            )
            new_tasks.extend(spawned)

        self.state.plan_tasks.extend(new_tasks)

        # 3. Step budgets — only for kinds that survived rejection
        kinds_kept = {a.kind for a in keep}
        for agent_name, budget in self.state.budgets.items():
            primary_kinds = {k for k in kinds_kept
                             if primary_agent_for(k) == agent_name}
            budget.step(
                proposed_this_round=bool(primary_kinds),
                current_reward_ema=ema_now,
            )

        # 4. Track no-new-actions streak (only on actions actually appended)
        if not new_tasks:
            self.state.no_new_actions_streak += 1
            rec.notes = (rec.notes + " | " if rec.notes else "") + (
                f"refiner returned 0 actionable items (llm stop={llm_stop_reason})"
            )
        else:
            self.state.no_new_actions_streak = 0

    async def _maybe_emit(self, rec: IterationRecord) -> None:
        if self._on_iter is None:
            return
        try:
            await self._on_iter(self.state, rec)
        except Exception:        # pragma: no cover — never let UI emit crash the loop
            log.exception("orchestrator: on_iteration_complete callback crashed")
