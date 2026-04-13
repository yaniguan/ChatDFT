"""Tests for the action whitelist + validators."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.orchestrator.actions import (  # noqa: E402
    MAX_ACTIONS_PER_ROUND,
    MAX_TOTAL_COST_PER_ROUND,
    SUBKINDS,
    ProposedAction,
    validate_action,
    validate_action_batch,
)


def test_subkinds_complete() -> None:
    """Every kind has at least one subkind, and every subkind declares required + default_cost."""
    for kind, subs in SUBKINDS.items():
        assert subs, f"kind {kind} has no subkinds"
        for sub, spec in subs.items():
            assert "required" in spec, f"{kind}.{sub} missing 'required'"
            assert isinstance(spec["required"], list)
            assert spec["default_cost"] >= 1


def test_validate_minimal_verify() -> None:
    raw = {
        "kind": "verify",
        "subkind": "reconverge",
        "target": "CO* on Pt(111) — task#3",
        "params": {"target_task_id": 3},
        "rationale": "Energy oscillating in OSZICAR — tighten ENCUT.",
        "priority": 0.7,
    }
    action, errors = validate_action(raw)
    assert errors == []
    assert action is not None
    assert action.kind == "verify"
    assert action.subkind == "reconverge"
    assert action.priority == 0.7


def test_validate_unknown_kind_rejected() -> None:
    action, errors = validate_action({"kind": "explore"})  # not in whitelist
    assert action is None
    assert any("kind" in str(e) for e in errors)


def test_validate_missing_required_param_rejected() -> None:
    raw = {
        "kind": "extend",
        "subkind": "site",
        "target": "H* @ bridge",
        "params": {"species": "H*"},  # missing surface + site
        "rationale": "site sweep",
    }
    action, errors = validate_action(raw)
    assert action is None
    msgs = [str(e) for e in errors]
    assert any("params.surface" in m for m in msgs)
    assert any("params.site" in m for m in msgs)


def test_validate_missing_rationale_rejected() -> None:
    raw = {
        "kind": "verify",
        "subkind": "reconverge",
        "target": "x",
        "params": {"target_task_id": 1},
        # no rationale
    }
    action, errors = validate_action(raw)
    assert action is None
    assert any("rationale" in str(e) for e in errors)


def test_priority_clamped_to_unit_interval() -> None:
    raw = {
        "kind": "verify",
        "subkind": "reconverge",
        "target": "x",
        "params": {"target_task_id": 1},
        "rationale": "ok",
        "priority": 99.0,
    }
    action, _ = validate_action(raw)
    assert action is not None
    assert action.priority == 1.0


def test_batch_drops_duplicates() -> None:
    base = {
        "kind": "verify",
        "subkind": "reconverge",
        "target": "CO* on Pt(111)",
        "rationale": "dup",
        "params": {"target_task_id": 5},
        "priority": 0.4,
    }
    accepted, errors = validate_action_batch([base, dict(base)])
    assert len(accepted) == 1
    assert any("duplicate" in str(e) for e in errors)


def test_batch_enforces_max_actions() -> None:
    items = [
        {
            "kind": "verify",
            "subkind": "reconverge",
            "target": f"task#{i}",
            "rationale": f"r{i}",
            "params": {"target_task_id": i},
            "priority": 0.5,
            "cost_estimate": 1,
        }
        for i in range(MAX_ACTIONS_PER_ROUND + 3)
    ]
    accepted, errors = validate_action_batch(items)
    assert len(accepted) == MAX_ACTIONS_PER_ROUND
    assert any("MAX_ACTIONS_PER_ROUND" in str(e) for e in errors)


def test_batch_enforces_total_cost() -> None:
    items = [
        {
            "kind": "scan",
            "subkind": "coverage",
            "target": f"H/Cu coverage scan #{i}",
            "params": {"species": "H*", "surface": "Cu(111)", "values": [0.25, 0.5, 0.75, 1.0]},
            "rationale": "scan",
            "priority": 0.5,
            "cost_estimate": 4,
        }
        for i in range(5)
    ]
    accepted, errors = validate_action_batch(items)
    total = sum(a.cost_estimate for a in accepted)
    assert total <= MAX_TOTAL_COST_PER_ROUND
    assert any("MAX_TOTAL_COST_PER_ROUND" in str(e) for e in errors)


def test_batch_sorted_by_priority_desc() -> None:
    items = [
        {
            "kind": "verify",
            "subkind": "reconverge",
            "target": "low",
            "params": {"target_task_id": 1},
            "rationale": "low",
            "priority": 0.1,
        },
        {
            "kind": "verify",
            "subkind": "reconverge",
            "target": "high",
            "params": {"target_task_id": 2},
            "rationale": "high",
            "priority": 0.9,
        },
        {
            "kind": "verify",
            "subkind": "reconverge",
            "target": "mid",
            "params": {"target_task_id": 3},
            "rationale": "mid",
            "priority": 0.5,
        },
    ]
    accepted, _ = validate_action_batch(items)
    priorities = [a.priority for a in accepted]
    assert priorities == sorted(priorities, reverse=True)


def test_proposed_action_signature_dedup_key() -> None:
    a = ProposedAction(kind="verify", subkind="reconverge", target="x", params={"target_task_id": 1}, rationale="r")
    b = ProposedAction(kind="verify", subkind="reconverge", target="x", params={"target_task_id": 2}, rationale="r")
    assert a.signature == b.signature  # signature ignores params on purpose
