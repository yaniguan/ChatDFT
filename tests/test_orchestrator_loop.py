"""
End-to-end tests for the closed-loop orchestrator.

The pipeline executor and the LLM refiner are mocked — these tests verify the
control flow:
  * results feed back into reward signals
  * reward influences whether we keep extending vs challenging
  * each documented stop condition fires correctly
  * per-agent budgets are respected
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.orchestrator.actions import ProposedAction  # noqa: E402
from server.orchestrator.loop import ChatDFTOrchestrator  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers — mock execute + mock refine
# ---------------------------------------------------------------------------


def _mock_execute_factory(value_seq: List[float], converged_seq: List[bool] | None = None):
    """
    Build an async pipeline mock that pops successive values for E_ads.
    """
    converged_seq = converged_seq or [True] * len(value_seq)
    counter = {"i": 0}

    async def _exec(task, *, auto_submit, cluster, engine):
        i = counter["i"]
        counter["i"] += 1
        v = value_seq[i % len(value_seq)]
        c = converged_seq[i % len(converged_seq)]
        return {
            "_pipeline_ok": True,
            "status": "done(test)",
            "post": {"E_ads": v, "converged": c},
            "job_dir": f"/tmp/test/{task.get('id', 0)}",
        }

    return _exec


def _mock_refine_factory(action_batches: List[List[ProposedAction]]):
    """
    Build an async refine mock that returns successive batches of actions.
    Each call pops the next batch (or empty if exhausted).
    """
    counter = {"i": 0}

    async def _refine(state):
        i = counter["i"]
        counter["i"] += 1
        if i < len(action_batches):
            batch = action_batches[i]
            stop = "" if batch else "test_exhausted"
            return ("test refinement", batch, [], stop)
        return ("", [], [], "test_exhausted")

    return _refine


def _intent() -> Dict[str, Any]:
    return {"problem_type": "ADSORPTION", "system": {"catalyst": "Pt", "facet": "111"}}


def _hypothesis_graph() -> Dict[str, Any]:
    return {
        "intermediates": ["CO*", "COOH*"],
        "reaction_network": ["CO2 + * -> CO2*", "CO2* + H -> COOH*"],
        "predictions": [
            {"species": "CO*", "surface": "Pt(111)", "trend": "exothermic", "range_lo": -1.5, "range_hi": -0.5},
        ],
    }


def _initial_task() -> Dict[str, Any]:
    return {
        "id": 1,
        "section": "execution",
        "name": "CO* on Pt(111)",
        "agent": "structure.relax_adsorbate",
        "description": "initial",
        "params": {"form": [], "payload": {"adsorbate": "CO*", "surface": "Pt(111)"}, "endpoint": ""},
        "depends_on": [],
        "status": "pending",
        "meta": {"run_id": 1},
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_iterations_stops_loop() -> None:
    exec_mock = _mock_execute_factory([-0.9])
    refine_mock = _mock_refine_factory(
        [
            [
                ProposedAction(
                    kind="extend",
                    subkind="intermediate",
                    target="OOH*",
                    params={"species": "OOH*", "surface": "Pt(111)"},
                    rationale="extend network",
                    priority=0.6,
                    cost_estimate=2,
                )
            ],
        ]
        * 50
    )  # always propose more

    with (
        patch("server.orchestrator.loop._execute_task_via_pipeline", exec_mock),
        patch("server.orchestrator.loop.refine_hypothesis", refine_mock),
    ):
        orch = ChatDFTOrchestrator(
            run_id=1,
            session_id=1,
            intent=_intent(),
            hypothesis_md="initial",
            hypothesis_graph=_hypothesis_graph(),
            initial_plan_tasks=[_initial_task()],
            max_iterations=3,
            confidence_threshold=0.99,  # never reached in test
            no_new_actions_threshold=99,  # never reached
        )
        state = await orch.run()

    assert state.iteration == 3
    assert state.stop_reason == "max_iterations_reached"


@pytest.mark.asyncio
async def test_no_new_actions_stops_loop() -> None:
    """When refiner returns no actions for N rounds and plan is empty, stop."""
    exec_mock = _mock_execute_factory([-0.9])
    # First refine returns nothing → streak 1
    # Second refine returns nothing → streak 2 → trigger stop
    refine_mock = _mock_refine_factory([])

    with (
        patch("server.orchestrator.loop._execute_task_via_pipeline", exec_mock),
        patch("server.orchestrator.loop.refine_hypothesis", refine_mock),
    ):
        orch = ChatDFTOrchestrator(
            run_id=2,
            session_id=1,
            intent=_intent(),
            hypothesis_md="initial",
            hypothesis_graph=_hypothesis_graph(),
            initial_plan_tasks=[_initial_task()],
            max_iterations=10,
            confidence_threshold=0.99,
            no_new_actions_threshold=2,
        )
        state = await orch.run()

    assert state.stop_reason == "no_new_actions_streak"


@pytest.mark.asyncio
async def test_results_feed_into_reward_signal() -> None:
    """A converged result that matches the predicted trend produces positive reward."""
    exec_mock = _mock_execute_factory([-0.9])  # within predicted [-1.5, -0.5]
    refine_mock = _mock_refine_factory([])  # stop after 1 task

    with (
        patch("server.orchestrator.loop._execute_task_via_pipeline", exec_mock),
        patch("server.orchestrator.loop.refine_hypothesis", refine_mock),
    ):
        orch = ChatDFTOrchestrator(
            run_id=3,
            session_id=1,
            intent=_intent(),
            hypothesis_md="initial",
            hypothesis_graph=_hypothesis_graph(),
            initial_plan_tasks=[_initial_task()],
            max_iterations=5,
            confidence_threshold=0.99,
            no_new_actions_threshold=2,
        )
        state = await orch.run()

    assert len(state.reward_history) == 1
    assert state.reward_history[0]["reward"] >= 0.5
    assert state.completed_results[0]["value"] == -0.9


@pytest.mark.asyncio
async def test_parameter_budget_caps_at_three_rounds() -> None:
    """
    parameter agent has max_rounds=3; verify-only refiner output should
    only be able to grow the plan three times.  After that, all further
    verify proposals are rejected and the loop stops via
    ``no_new_actions_streak`` once the threshold is met.
    """
    # Provide enough mock results for every executed task
    exec_mock = _mock_execute_factory([-0.9] * 50)
    verify_action = ProposedAction(
        kind="verify",
        subkind="reconverge",
        target="task#1",
        params={"target_task_id": 1},
        rationale="tighten convergence",
        priority=0.5,
        cost_estimate=1,
    )
    refine_mock = _mock_refine_factory([[verify_action]] * 50)

    with (
        patch("server.orchestrator.loop._execute_task_via_pipeline", exec_mock),
        patch("server.orchestrator.loop.refine_hypothesis", refine_mock),
    ):
        orch = ChatDFTOrchestrator(
            run_id=4,
            session_id=1,
            intent=_intent(),
            hypothesis_md="initial",
            hypothesis_graph=_hypothesis_graph(),
            initial_plan_tasks=[_initial_task()],
            max_iterations=20,
            confidence_threshold=0.999,  # never reached
            no_new_actions_threshold=2,
        )
        state = await orch.run()

    param_budget = state.budgets["parameter"]
    assert param_budget.rounds_used == 3, f"parameter budget should top out at 3 rounds, got {param_budget.rounds_used}"
    # After param exhaustion all further verifies are rejected → no_new_actions hits
    assert state.stop_reason == "no_new_actions_streak"


@pytest.mark.asyncio
async def test_user_stop_request_takes_effect_at_iteration_boundary() -> None:
    exec_mock = _mock_execute_factory([-0.9])
    extend_action = ProposedAction(
        kind="extend",
        subkind="intermediate",
        target="OOH*",
        params={"species": "OOH*", "surface": "Pt(111)"},
        rationale="extend",
        priority=0.6,
        cost_estimate=2,
    )
    refine_mock = _mock_refine_factory([[extend_action]] * 50)

    with (
        patch("server.orchestrator.loop._execute_task_via_pipeline", exec_mock),
        patch("server.orchestrator.loop.refine_hypothesis", refine_mock),
    ):
        orch = ChatDFTOrchestrator(
            run_id=5,
            session_id=1,
            intent=_intent(),
            hypothesis_md="initial",
            hypothesis_graph=_hypothesis_graph(),
            initial_plan_tasks=[_initial_task()],
            max_iterations=20,
        )
        # request stop before run starts; first iter check will pick it up
        orch.request_stop()
        state = await orch.run()

    assert state.stop_reason == "user_stopped"
    assert state.iteration <= 1
