# tests/test_intent_eval_harness.py
# -*- coding: utf-8 -*-
"""
Unit tests for the intent eval harness scoring functions.

The harness's *live predictor* path is intentionally NOT exercised here —
it would require a running server and a real LLM. These tests cover the
pure scoring layer that grades a predicted dict against a sparse gold dict.
"""
from __future__ import annotations

import pytest

from tests.intent_eval.harness import (
    DEFAULT_EVAL_SET,
    _canonical_species,
    _chemical_equal,
    _split_species,
    aggregate,
    echo_predictor,
    load_eval_set,
    run,
    score_case,
)

# ---------------------------------------------------------------------------
# eval set integrity
# ---------------------------------------------------------------------------

def test_eval_set_loads_and_is_well_formed():
    cases = load_eval_set()
    assert len(cases) >= 20
    seen_areas = {c.gold.get("area") for c in cases if c.gold.get("area")}
    # All five canonical areas should be represented in the seed set.
    assert seen_areas >= {
        "electrochemistry",
        "thermal_catalysis",
        "photocatalysis",
        "heterogeneous_catalysis",
        "homogeneous_catalysis",
    }
    # Every case has an id, query, and gold.
    for c in cases:
        assert c.id and c.query and isinstance(c.gold, dict)
        assert "stage" in c.gold and "area" in c.gold


def test_eval_set_ids_are_unique():
    cases = load_eval_set()
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), "duplicate eval case ids"


# ---------------------------------------------------------------------------
# score_case primitives
# ---------------------------------------------------------------------------

def test_perfect_match_scores_all_correct():
    gold = {
        "stage": "catalysis",
        "area": "electrochemistry",
        "substrate": "Cu(111)",
        "conditions": {"pH": 7.0, "potential_V_vs_RHE": -0.8},
    }
    predicted = dict(gold)  # exact match
    field_scores, critical_em = score_case(predicted, gold)
    assert all(field_scores.values())
    assert critical_em is True
    # Nested keys are scored individually.
    assert "conditions.pH" in field_scores
    assert "conditions.potential_V_vs_RHE" in field_scores


def test_only_gold_keys_are_scored():
    gold = {"stage": "catalysis", "area": "electrochemistry"}
    predicted = {
        "stage": "catalysis",
        "area": "electrochemistry",
        "substrate": "Cu(111)",  # extra — should NOT be scored
        "tags": ["co2rr"],
    }
    field_scores, _ = score_case(predicted, gold)
    assert set(field_scores.keys()) == {"stage", "area"}


def test_critical_em_requires_substrate_when_present():
    gold = {"stage": "catalysis", "area": "electrochemistry", "substrate": "Cu(111)"}
    pred_wrong_substrate = {
        "stage": "catalysis",
        "area": "electrochemistry",
        "substrate": "Pt(111)",
    }
    _, em = score_case(pred_wrong_substrate, gold)
    assert em is False

    pred_correct = dict(gold)
    _, em = score_case(pred_correct, gold)
    assert em is True


def test_critical_em_ignores_substrate_when_absent_from_gold():
    gold = {"stage": "catalysis", "area": "homogeneous_catalysis"}
    pred = {
        "stage": "catalysis",
        "area": "homogeneous_catalysis",
        "substrate": None,
    }
    _, em = score_case(pred, gold)
    assert em is True


def test_string_comparison_is_case_and_whitespace_insensitive():
    gold = {"stage": "catalysis", "area": "electrochemistry", "substrate": "Cu(111)"}
    pred = {"stage": "Catalysis", "area": "ELECTROCHEMISTRY", "substrate": " cu(111) "}
    field_scores, em = score_case(pred, gold)
    assert all(field_scores.values())
    assert em is True


def test_numeric_tolerance_in_conditions():
    gold = {
        "stage": "catalysis",
        "area": "electrochemistry",
        "conditions": {"pH": 7.0, "potential_V_vs_RHE": -0.8},
    }
    pred = {
        "stage": "catalysis",
        "area": "electrochemistry",
        "conditions": {"pH": 7.0001, "potential_V_vs_RHE": -0.7999},
    }
    field_scores, _ = score_case(pred, gold)
    assert field_scores["conditions.pH"] is True
    assert field_scores["conditions.potential_V_vs_RHE"] is True


def test_missing_predicted_field_scores_false():
    gold = {"stage": "catalysis", "area": "electrochemistry", "substrate": "Cu(111)"}
    pred = {"stage": "catalysis", "area": "electrochemistry"}  # no substrate
    field_scores, em = score_case(pred, gold)
    assert field_scores["substrate"] is False
    assert em is False


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def test_aggregate_with_perfect_oracle():
    """A predictor that returns the gold dict verbatim should score 100%."""
    cases = load_eval_set()
    by_query = {c.query: c.gold for c in cases}

    def perfect(q):
        return dict(by_query[q])

    results = run(cases, perfect)
    report = aggregate(results)
    assert report.n_cases == len(cases)
    assert report.field_accuracy == 1.0
    assert report.critical_em_rate == 1.0
    assert report.failures == []


def test_aggregate_with_echo_predictor_scores_zero():
    cases = load_eval_set()
    results = run(cases, echo_predictor)
    report = aggregate(results)
    assert report.n_cases == len(cases)
    assert report.field_accuracy == 0.0
    assert report.critical_em_rate == 0.0
    # All cases land in failures.
    assert len(report.failures) == len(cases)
    # Confusion matrix points every gold area at "<missing>".
    for gold_area, row in report.area_confusion.items():
        assert row.get("<missing>", 0) > 0


def test_aggregate_with_area_only_predictor():
    """A predictor that only fills `stage` + `area` correctly should still earn partial credit."""
    cases = load_eval_set()
    by_q = {c.query: c.gold for c in cases}

    def stage_area_only(q):
        gold = by_q[q]
        return {"stage": gold["stage"], "area": gold["area"]}

    results = run(cases, stage_area_only)
    report = aggregate(results)
    assert report.per_field_accuracy["stage"] == 1.0
    assert report.per_field_accuracy["area"] == 1.0
    # substrate accuracy should be < 1 because most golds have substrate set
    if "substrate" in report.per_field_accuracy:
        assert report.per_field_accuracy["substrate"] < 1.0
    # critical_em fails on cases that have substrate in gold.
    assert report.critical_em_rate < 1.0


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------

def test_default_eval_set_path_exists():
    assert DEFAULT_EVAL_SET.exists()
    assert DEFAULT_EVAL_SET.suffix == ".jsonl"


# ---------------------------------------------------------------------------
# chemical-aware scoring (for reactant / product fields)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "gold,pred",
    [
        ("H2O", "water"),
        ("water", "H2O"),
        ("C2H4", "ethene"),
        ("C2H4", "ethylene"),
        ("CH3OH", "methanol"),
        ("NH3", "ammonia"),
        ("CO", "carbon monoxide"),
        ("CO2", "carbon dioxide"),
        ("CH4", "methane"),
        ("H2", "dihydrogen"),
    ],
)
def test_chemical_equal_formula_name_aliases(gold, pred):
    """The 0.562 baseline product accuracy had cases like C2H4 vs ethene
    counted as misses. That was a scoring bug, not a model failure."""
    assert _chemical_equal(gold, pred) is True
    # symmetry
    assert _chemical_equal(pred, gold) is True


@pytest.mark.parametrize(
    "gold,pred",
    [
        # Steam methane reforming: CH4 + H2O → CO + 3 H2 — both are primary products
        ("CO", "CO + H2"),
        ("H2", "CO + H2"),
        # Water splitting: H2O → H2 + O2 — model may return both
        ("H2", "H2 + O2"),
        ("O2", "H2 + O2"),
        # Comma separator
        ("CO", "CO, H2"),
        # "and" separator
        ("CO", "CO and H2"),
    ],
)
def test_chemical_equal_superset_match(gold, pred):
    """Gold species must be a subset of predicted species to accept."""
    assert _chemical_equal(gold, pred) is True


@pytest.mark.parametrize(
    "gold,pred",
    [
        # Distinct species remain distinct
        ("CO", "CO2"),
        ("H2", "O2"),
        ("N2", "NH3"),
        # Superset direction: gold richer than pred must NOT pass
        ("H2 + O2", "H2"),
        # None asymmetry
        (None, "H2"),
        ("H2", None),
        # Unknown species fall back to exact-match
        ("LaMnO3", "SrMnO3"),
    ],
)
def test_chemical_equal_rejects_different_species(gold, pred):
    assert _chemical_equal(gold, pred) is False


def test_chemical_equal_both_null_matches():
    assert _chemical_equal(None, None) is True


def test_canonical_species_normalizes_known_aliases():
    assert _canonical_species("water") == "h2o"
    assert _canonical_species("H2O") == "h2o"
    assert _canonical_species("H₂O") == "h2o"
    assert _canonical_species("ethene") == "c2h4"
    assert _canonical_species("ethylene") == "c2h4"


def test_canonical_species_passes_through_unknown():
    """Unknown species must fall through — don't silently collapse distinct species."""
    assert _canonical_species("LaMnO3") == "lamno3"
    assert _canonical_species("MoS2") == "mos2"


def test_split_species_handles_separators():
    assert _split_species("H2 + O2") == ["h2", "o2"]
    assert _split_species("CO, H2") == ["co", "h2"]
    assert _split_species("CO and H2") == ["co", "h2"]
    assert _split_species("H2O") == ["h2o"]  # no separator
    assert _split_species("") == []
    assert _split_species(None) == []


def test_score_case_uses_chemical_equal_for_product_field():
    """
    Real regression test: eval-019 (C2H4 vs 'ethene') must now pass.
    Without the chemical scorer this scores False and drags product
    accuracy below reality.
    """
    gold = {
        "stage": "catalysis",
        "area": "thermal_catalysis",
        "reactant": "CH3OH",
        "product": "C2H4",
    }
    predicted = {
        "stage": "catalysis",
        "area": "thermal_catalysis",
        "reactant": "methanol",
        "product": "ethene",
    }
    field_scores, _ = score_case(predicted, gold)
    assert field_scores["reactant"] is True
    assert field_scores["product"] is True


def test_score_case_chemistry_does_not_leak_to_non_species_fields():
    """
    ``substrate`` must NOT go through the chemical alias table — the
    species-vs-catalyst distinction is load-bearing. Confirm a substrate
    comparison is still strict.
    """
    gold = {"stage": "catalysis", "area": "electrochemistry", "substrate": "Cu(111)"}
    predicted = {"stage": "catalysis", "area": "electrochemistry", "substrate": "Pt(111)"}
    field_scores, _ = score_case(predicted, gold)
    assert field_scores["substrate"] is False
