"""Tests for action → WorkflowTask conversion."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.orchestrator.action_to_tasks import (  # noqa: E402
    action_to_tasks,
    primary_agent_for,
)
from server.orchestrator.actions import ProposedAction  # noqa: E402


def _act(**kw) -> ProposedAction:
    defaults = dict(target="x", rationale="r", priority=0.5, cost_estimate=1)
    defaults.update(kw)
    return ProposedAction(**defaults)


def test_extend_intermediate_emits_one_structure_task() -> None:
    a = _act(kind="extend", subkind="intermediate", params={"species": "COOH*", "surface": "Pt(111)"})
    tasks = action_to_tasks(a, existing_tasks=[], session_id=42)
    assert len(tasks) == 1
    t = tasks[0]
    assert t["agent"] == "structure.relax_adsorbate"
    assert t["status"] == "pending"
    assert t["params"]["payload"]["adsorbate"] == "COOH*"
    assert t["meta"]["source"] == "orchestrator"
    assert t["meta"]["action_kind"] == "extend"
    assert t["id"] == 1
    assert a.spawned_task_ids == [1]


def test_scan_coverage_emits_one_task_per_value() -> None:
    a = _act(
        kind="scan",
        subkind="coverage",
        params={"species": "H*", "surface": "Cu(111)", "values": [0.25, 0.5, 1.0]},
        cost_estimate=4,
    )
    tasks = action_to_tasks(a, existing_tasks=[{"id": 7}], session_id=1)
    assert len(tasks) == 3
    assert [t["id"] for t in tasks] == [8, 9, 10]
    assert all(t["agent"] == "adsorption.scan" for t in tasks)
    assert [t["params"]["payload"]["coverage"] for t in tasks] == [0.25, 0.5, 1.0]


def test_verify_reconverge_routes_to_run_dft_with_overrides() -> None:
    a = _act(kind="verify", subkind="reconverge", params={"target_task_id": 5, "delta_encut": 100, "kmesh_factor": 2.0})
    tasks = action_to_tasks(a, existing_tasks=[], session_id=1)
    assert len(tasks) == 1
    t = tasks[0]
    assert t["agent"] == "run_dft"
    assert t["params"]["payload"]["target_task_id"] == 5
    assert t["params"]["payload"]["incar_overrides"]["ENCUT"] == "+100.0"
    assert t["params"]["payload"]["kpoints_factor"] == 2.0


def test_challenge_alternative_step_is_neb() -> None:
    a = _act(
        kind="challenge", subkind="alternative_step", params={"original_step": "A->B", "alternative_step": "A->C->B"}
    )
    tasks = action_to_tasks(a, existing_tasks=[], session_id=1)
    assert len(tasks) == 1
    assert tasks[0]["agent"] == "neb.run"


def test_unknown_subkind_returns_empty() -> None:
    a = ProposedAction(kind="extend", subkind="nonsense", target="x", params={}, rationale="r")
    tasks = action_to_tasks(a, existing_tasks=[], session_id=1)
    assert tasks == []


def test_primary_agent_routing() -> None:
    assert primary_agent_for("verify") == "parameter"
    assert primary_agent_for("extend") == "structure"
    assert primary_agent_for("scan") == "structure"
    assert primary_agent_for("challenge") == "structure"
