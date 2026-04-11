# tests/test_intent_schema.py
# -*- coding: utf-8 -*-
"""
Unit tests for ``server.chat.intent_schema``.

These tests guard the contract between the intent agent prompt and the
downstream pipeline. They run in CI without any API access.
"""

from __future__ import annotations

import pytest

from server.chat.intent_schema import (
    AREA_VALUES,
    STAGE_VALUES,
    IntentSchema,
    format_validation_error,
    validate_intent,
)

# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_canonical_intent_validates():
    raw = {
        "stage": "catalysis",
        "area": "electrochemistry",
        "task": "study CO2RR on Cu(111)",
        "summary": "Investigate CO2 reduction to methanol on Cu(111).",
        "substrate": "Cu(111)",
        "facet": "111",
        "reactant": "CO2",
        "product": "CH3OH",
        "adsorbates": ["CO2*", "COOH*", "CO*"],
        "conditions": {"pH": 7.0, "potential_V_vs_RHE": -0.8},
        "reaction_network": {
            "intermediates": ["CO2*", "COOH*", "CO*"],
            "steps": ["CO2* + H+ + e- -> COOH*"],
        },
    }
    model, err = validate_intent(raw)
    assert err is None
    assert model is not None
    assert model.stage == "catalysis"
    assert model.area == "electrochemistry"
    assert model.substrate == "Cu(111)"
    assert model.conditions.pH == 7.0
    assert model.conditions.potential_V_vs_RHE == -0.8
    assert len(model.reaction_network.intermediates) == 3


def test_dump_roundtrip_preserves_extras():
    raw = {
        "stage": "catalysis",
        "area": "thermal_catalysis",
        "task": "x",
        "summary": "y",
        "_provider_metadata": {"foo": "bar"},  # extra
    }
    model, err = validate_intent(raw)
    assert err is None
    dumped = model.model_dump(mode="json")
    assert dumped["_provider_metadata"] == {"foo": "bar"}


# ---------------------------------------------------------------------------
# normalization — the LLM gets to be sloppy, the schema fixes it up
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_area,expected",
    [
        ("electro", "electrochemistry"),
        ("Electrocatalysis", "electrochemistry"),
        ("ELECTROCHEM", "electrochemistry"),
        ("thermal", "thermal_catalysis"),
        ("Thermocatalytic", "thermal_catalysis"),
        ("photo", "photocatalysis"),
        ("photothermal", "photocatalysis"),
        ("heterogeneous", "heterogeneous_catalysis"),
        ("homogeneous", "homogeneous_catalysis"),
        ("organometallic", "homogeneous_catalysis"),
        ("electrochemistry", "electrochemistry"),  # already canonical
    ],
)
def test_area_normalization(raw_area, expected):
    model, err = validate_intent(
        {
            "stage": "catalysis",
            "area": raw_area,
            "task": "x",
            "summary": "y",
        }
    )
    assert err is None, f"unexpected error: {err}"
    assert model.area == expected


@pytest.mark.parametrize(
    "raw_stage,expected",
    [
        ("catalysis", "catalysis"),
        ("Catalytic", "catalysis"),
        ("benchmark", "benchmarking"),
        ("analyze", "analysis"),
        ("structure", "structure_building"),
    ],
)
def test_stage_normalization(raw_stage, expected):
    model, err = validate_intent(
        {
            "stage": raw_stage,
            "area": "electrochemistry",
            "task": "x",
            "summary": "y",
        }
    )
    assert err is None
    assert model.stage == expected


# ---------------------------------------------------------------------------
# validation errors
# ---------------------------------------------------------------------------


def test_unknown_area_fails():
    model, err = validate_intent(
        {
            "stage": "catalysis",
            "area": "cosmic_rays",
            "task": "x",
            "summary": "y",
        }
    )
    assert model is None
    assert err is not None
    assert "area" in err
    assert "cosmic_rays" in err


def test_missing_required_fields_fails():
    model, err = validate_intent({"stage": "catalysis", "area": "electrochemistry"})
    assert model is None
    assert err is not None
    assert "task" in err
    assert "summary" in err


def test_empty_task_fails():
    model, err = validate_intent(
        {
            "stage": "catalysis",
            "area": "electrochemistry",
            "task": "",
            "summary": "y",
        }
    )
    assert model is None
    assert err is not None
    assert "task" in err


def test_non_dict_input_fails_gracefully():
    model, err = validate_intent(None)
    assert model is None
    assert err is not None

    model, err = validate_intent({})
    assert model is None
    assert err is not None


# ---------------------------------------------------------------------------
# coercion of permissive sub-fields
# ---------------------------------------------------------------------------


def test_metrics_string_list_is_coerced():
    model, err = validate_intent(
        {
            "stage": "catalysis",
            "area": "electrochemistry",
            "task": "x",
            "summary": "y",
            "metrics": ["limiting_potential", "selectivity"],
        }
    )
    assert err is None
    assert len(model.metrics) == 2
    assert model.metrics[0].name == "limiting_potential"
    assert model.metrics[1].name == "selectivity"


def test_constraints_string_becomes_notes():
    model, err = validate_intent(
        {
            "stage": "catalysis",
            "area": "electrochemistry",
            "task": "x",
            "summary": "y",
            "constraints": "must use SCAN functional",
        }
    )
    assert err is None
    assert model.constraints == {"notes": "must use SCAN functional"}


def test_molecule_scalar_is_listed():
    model, err = validate_intent(
        {
            "stage": "catalysis",
            "area": "electrochemistry",
            "task": "x",
            "summary": "y",
            "system": {"molecule": "CO2"},
        }
    )
    assert err is None
    assert model.system.molecule == ["CO2"]


def test_conditions_string_number_is_coerced():
    model, err = validate_intent(
        {
            "stage": "catalysis",
            "area": "electrochemistry",
            "task": "x",
            "summary": "y",
            "conditions": {"potential_V_vs_RHE": "-0.8"},
        }
    )
    assert err is None
    assert model.conditions.potential_V_vs_RHE == -0.8


# ---------------------------------------------------------------------------
# canonical value contracts (catch refactor accidents)
# ---------------------------------------------------------------------------


def test_canonical_enum_values_are_stable():
    """If you renamed an enum value, the prompt + downstream code likely break too."""
    assert STAGE_VALUES == (
        "catalysis",
        "screening",
        "benchmarking",
        "analysis",
        "structure_building",
    )
    assert AREA_VALUES == (
        "electrochemistry",
        "thermal_catalysis",
        "photocatalysis",
        "heterogeneous_catalysis",
        "homogeneous_catalysis",
    )


def test_format_validation_error_is_short_and_useful():
    from pydantic import ValidationError

    try:
        IntentSchema.model_validate({"stage": "x", "area": "y"})
    except ValidationError as exc:
        summary = format_validation_error(exc)
        # Multi-line, includes field names, no stack trace.
        assert "stage" in summary
        assert "area" in summary
        assert summary.count("\n") >= 1
    else:
        pytest.fail("expected ValidationError")
