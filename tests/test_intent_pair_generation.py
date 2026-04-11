# tests/test_intent_pair_generation.py
# -*- coding: utf-8 -*-
"""
Unit tests for ``scripts/generate_intent_pairs``.

These tests cover the pure-python layers — strata loading, response
parsing, validation gating, dedup. The Anthropic teacher and the
embedding/DB I/O paths are intentionally NOT exercised here; they
require live services.
"""
from __future__ import annotations

import json

import pytest

from scripts.generate_intent_pairs import (
    DEFAULT_STRATA_PATH,
    AnthropicTeacherClient,
    OpenAITeacherClient,
    RawPair,
    Stratum,
    TeacherClient,
    ValidatedPair,
    _chunk,
    _cosine,
    _split_n_pairs,
    build_history_prompt,
    build_stratum_prompt,
    build_teacher,
    build_teacher_system_prompt,
    dedup_by_embedding,
    load_strata,
    parse_teacher_response,
    validate_pairs,
)
from server.chat.intent_schema import AREA_VALUES, SCHEMA_VERSION

# ---------------------------------------------------------------------------
# strata loading
# ---------------------------------------------------------------------------

def test_default_strata_yaml_loads():
    strata = load_strata()
    assert len(strata) >= 20
    # all canonical areas covered
    assert {s.area for s in strata} >= set(AREA_VALUES)
    # all difficulties present
    assert {s.difficulty for s in strata} == {"simple", "medium", "hard"}
    # all ids unique
    ids = [s.id for s in strata]
    assert len(ids) == len(set(ids))


def test_strata_yaml_path_exists():
    assert DEFAULT_STRATA_PATH.exists()
    assert DEFAULT_STRATA_PATH.suffix == ".yaml"


def test_load_strata_rejects_unknown_area(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: 1\n"
        "strata:\n"
        "  - id: foo/bar/simple\n"
        "    area: cosmic_rays\n"
        "    family: bar\n"
        "    difficulty: simple\n"
        "    anchors: []\n"
    )
    with pytest.raises(ValueError, match="invalid area"):
        load_strata(bad)


def test_load_strata_rejects_duplicate_ids(tmp_path):
    bad = tmp_path / "dup.yaml"
    bad.write_text(
        "version: 1\n"
        "strata:\n"
        "  - id: same\n"
        "    area: electrochemistry\n"
        "    family: x\n"
        "    difficulty: simple\n"
        "    anchors: []\n"
        "  - id: same\n"
        "    area: thermal_catalysis\n"
        "    family: y\n"
        "    difficulty: medium\n"
        "    anchors: []\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_strata(bad)


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------

def test_build_teacher_system_prompt_includes_schema_and_addendum():
    prompt = build_teacher_system_prompt()
    # IntentSchema-derived fragments
    assert "stage" in prompt and "area" in prompt
    assert "electrochemistry" in prompt
    assert "thermal_catalysis" in prompt
    # teacher-role addendum
    assert "OUTPUT FORMAT" in prompt
    assert "pairs" in prompt
    assert "DIVERSITY RULES" in prompt


def test_build_stratum_prompt_embeds_anchors_and_guidance():
    s = Stratum(
        id="electrochemistry/CO2RR/simple",
        area="electrochemistry",
        family="CO2RR",
        difficulty="simple",
        anchors=["CO2 reduction to CO on Cu(111)", "CO2RR to formate on Sn"],
        guidance="One reactant, one product.",
    )
    prompt = build_stratum_prompt(s, n_pairs=4)
    assert "Generate 4" in prompt
    assert "area:       electrochemistry" in prompt
    assert "CO2 reduction to CO on Cu(111)" in prompt
    assert "CO2RR to formate on Sn" in prompt
    assert "One reactant, one product." in prompt
    assert "'electrochemistry'" in prompt


def test_build_stratum_prompt_handles_empty_anchors():
    s = Stratum(
        id="x", area="thermal_catalysis", family="x",
        difficulty="medium", anchors=[], guidance=None,
    )
    prompt = build_stratum_prompt(s, n_pairs=2)
    assert "(none)" in prompt
    assert "Stratum guidance" not in prompt


# ---------------------------------------------------------------------------
# parse_teacher_response
# ---------------------------------------------------------------------------

_GOOD_RESPONSE = json.dumps({
    "pairs": [
        {
            "query": "Compute CO2RR free energy diagram on Cu(111).",
            "intent": {
                "stage": "catalysis",
                "area": "electrochemistry",
                "task": "CO2RR free energy diagram on Cu(111)",
                "summary": "Free energy diagram for CO2RR on Cu(111).",
                "substrate": "Cu(111)",
            },
        },
        {
            "query": "Estimate the limiting potential for CO2 to CO on Sn.",
            "intent": {
                "stage": "catalysis",
                "area": "electrochemistry",
                "task": "CO2 to CO on Sn",
                "summary": "Limiting potential for CO2 to CO on Sn.",
                "substrate": "Sn",
            },
        },
    ]
})


def test_parse_clean_response():
    pairs = parse_teacher_response(_GOOD_RESPONSE)
    assert len(pairs) == 2
    assert all(isinstance(p, RawPair) for p in pairs)
    assert pairs[0].intent["substrate"] == "Cu(111)"


def test_parse_response_with_code_fences():
    fenced = "```json\n" + _GOOD_RESPONSE + "\n```"
    pairs = parse_teacher_response(fenced)
    assert len(pairs) == 2


def test_parse_response_with_leading_prose():
    text = "Sure, here are the pairs you requested:\n" + _GOOD_RESPONSE
    pairs = parse_teacher_response(text)
    assert len(pairs) == 2


def test_parse_response_skips_malformed_items():
    blob = json.dumps({
        "pairs": [
            {"query": "valid one", "intent": {"stage": "catalysis"}},
            {"query": ""},  # empty query — drop
            {"intent": {}},  # missing query — drop
            "not a dict",   # wrong type — drop
        ]
    })
    pairs = parse_teacher_response(blob)
    assert len(pairs) == 1
    assert pairs[0].query == "valid one"


def test_parse_response_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_teacher_response("definitely not json at all")


def test_parse_response_raises_on_missing_pairs_key():
    with pytest.raises(ValueError, match="pairs"):
        parse_teacher_response('{"not_pairs": []}')


# ---------------------------------------------------------------------------
# validate_pairs
# ---------------------------------------------------------------------------

def test_validate_pairs_separates_valid_from_invalid():
    raw = [
        # canonical valid
        RawPair(
            query="q1",
            intent={
                "stage": "catalysis",
                "area": "electrochemistry",
                "task": "t",
                "summary": "s",
            },
        ),
        # variant area spelling — should still validate (gets normalized)
        RawPair(
            query="q2",
            intent={
                "stage": "catalysis",
                "area": "electro",
                "task": "t",
                "summary": "s",
            },
        ),
        # invalid — unknown area
        RawPair(
            query="q3",
            intent={
                "stage": "catalysis",
                "area": "cosmic_rays",
                "task": "t",
                "summary": "s",
            },
        ),
        # invalid — missing required
        RawPair(query="q4", intent={"stage": "catalysis"}),
    ]
    valid, invalid = validate_pairs(raw, stratum_id="test/stratum")
    assert len(valid) == 2
    assert len(invalid) == 2
    # Both valid pairs ended up with canonical area spelling.
    assert all(p.intent["area"] == "electrochemistry" for p in valid)
    assert all(p.stratum_id == "test/stratum" for p in valid)


def test_validate_pairs_dump_is_json_serializable():
    raw = [
        RawPair(
            query="q",
            intent={
                "stage": "catalysis",
                "area": "electrochemistry",
                "task": "t",
                "summary": "s",
                "conditions": {"pH": 7.0, "potential_V_vs_RHE": -0.8},
            },
        )
    ]
    valid, _ = validate_pairs(raw, stratum_id="x")
    # Round-trip through json — confirms model_dump(mode="json") was used.
    blob = json.dumps(valid[0].intent)
    assert "electrochemistry" in blob
    assert "potential_V_vs_RHE" in blob


# ---------------------------------------------------------------------------
# dedup_by_embedding
# ---------------------------------------------------------------------------

def test_cosine_self_similarity_is_one():
    assert _cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_handles_zero_vector():
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert _cosine([], [1.0]) == 0.0


def test_dedup_drops_in_batch_duplicates():
    base = [1.0, 0.0, 0.0]
    near = [0.99, 0.01, 0.0]      # cosine ≈ 0.9999, should be dropped
    far = [0.0, 1.0, 0.0]
    candidates = [
        ValidatedPair(query="a", intent={}, stratum_id="x", embedding=base),
        ValidatedPair(query="a-rephrased", intent={}, stratum_id="x", embedding=near),
        ValidatedPair(query="b", intent={}, stratum_id="x", embedding=far),
    ]
    accepted, dropped = dedup_by_embedding(candidates, [], threshold=0.92)
    queries = [p.query for p in accepted]
    assert queries == ["a", "b"]
    assert dropped == 1


def test_dedup_drops_against_existing_embeddings():
    existing = [[1.0, 0.0, 0.0]]
    candidates = [
        ValidatedPair(query="dup", intent={}, stratum_id="x",
                      embedding=[0.999, 0.001, 0.0]),
        ValidatedPair(query="new", intent={}, stratum_id="x",
                      embedding=[0.0, 0.0, 1.0]),
    ]
    accepted, dropped = dedup_by_embedding(candidates, existing, threshold=0.92)
    assert [p.query for p in accepted] == ["new"]
    assert dropped == 1


def test_dedup_threshold_is_inclusive():
    """A pair right at the threshold should be dropped."""
    base = [1.0, 0.0]
    candidates = [
        ValidatedPair(query="a", intent={}, stratum_id="x", embedding=base),
        ValidatedPair(query="b", intent={}, stratum_id="x", embedding=list(base)),
    ]
    accepted, dropped = dedup_by_embedding(candidates, [], threshold=0.92)
    assert len(accepted) == 1
    assert dropped == 1


def test_dedup_raises_on_missing_embedding():
    candidates = [
        ValidatedPair(query="x", intent={}, stratum_id="x", embedding=None),
    ]
    with pytest.raises(ValueError, match="missing embedding"):
        dedup_by_embedding(candidates, [], threshold=0.9)


# ---------------------------------------------------------------------------
# schema version contract
# ---------------------------------------------------------------------------

def test_schema_version_is_exposed_and_positive():
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 1


# ---------------------------------------------------------------------------
# Phase 1.5: --from-history mode
# ---------------------------------------------------------------------------

def test_chunk_splits_evenly():
    assert list(_chunk([1, 2, 3, 4, 5, 6], 2)) == [[1, 2], [3, 4], [5, 6]]


def test_chunk_handles_partial_last():
    assert list(_chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunk_rejects_zero_size():
    with pytest.raises(ValueError):
        list(_chunk([1, 2], 0))


def test_chunk_empty_input():
    assert list(_chunk([], 5)) == []


def test_build_history_prompt_embeds_verbatim_queries_and_count():
    queries = [
        "Compute the d-band center of Pt(111) with CO adsorbed.",
        "I want to do CO2RR on Cu(100) at -0.6 V vs RHE.",
    ]
    prompt = build_history_prompt(queries)
    assert "Label each of the following 2 REAL user queries" in prompt
    # Both queries should appear verbatim with their numbered prefixes
    assert "1. Compute the d-band center of Pt(111) with CO adsorbed." in prompt
    assert "2. I want to do CO2RR on Cu(100) at -0.6 V vs RHE." in prompt
    # The schema constraint reference must be present
    assert "IntentSchema" in prompt
    # Strict count constraint must show up
    assert "exactly 2 entries" in prompt


def test_build_history_prompt_rejects_empty_list():
    with pytest.raises(ValueError, match="empty query list"):
        build_history_prompt([])


def test_build_history_prompt_uses_pairs_envelope():
    """The history mode reuses parse_teacher_response, which expects 'pairs'."""
    prompt = build_history_prompt(["q1"])
    assert '"pairs"' in prompt
    # parse_teacher_response should accept the matching response shape.
    fake_response = json.dumps({
        "pairs": [
            {
                "query": "q1",
                "intent": {
                    "stage": "catalysis",
                    "area": "electrochemistry",
                    "task": "t",
                    "summary": "s",
                },
            }
        ]
    })
    pairs = parse_teacher_response(fake_response)
    assert len(pairs) == 1
    assert pairs[0].query == "q1"


def test_history_validate_uses_history_stratum_id():
    """validate_pairs in history mode tags rows with history/<batch>."""
    raw = [
        RawPair(
            query="real user query",
            intent={
                "stage": "catalysis",
                "area": "electrochemistry",
                "task": "t",
                "summary": "s",
            },
        )
    ]
    valid, _ = validate_pairs(raw, stratum_id="history/0042")
    assert valid[0].stratum_id == "history/0042"


# ---------------------------------------------------------------------------
# Phase 2: build_teacher factory routing
# ---------------------------------------------------------------------------

@pytest.fixture
def clear_api_keys(monkeypatch):
    """
    Remove both API keys so client __init__ raises before it tries to
    actually connect. This lets us exercise the routing layer without
    touching any network.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# The routing tests don't care whether the actual SDK is installed in the
# test env. They care that build_teacher() reached the *correct* backend.
# Both backends raise RuntimeError at construction time whenever the API
# key OR the SDK package is missing, so we accept either message as proof
# of dispatch. (Base env has anthropic but not openai; llm-agent has both.)
_ANTHROPIC_MARKER = r"ANTHROPIC_API_KEY|anthropic SDK is required"
_OPENAI_MARKER = r"OPENAI_API_KEY|openai SDK is required"


@pytest.mark.parametrize(
    "model",
    ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
)
def test_build_teacher_routes_claude_to_anthropic(model, clear_api_keys):
    with pytest.raises(RuntimeError, match=_ANTHROPIC_MARKER):
        build_teacher(model)


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "o1-mini",
        "o3-mini",
        "chatgpt-4o-latest",
    ],
)
def test_build_teacher_routes_openai_prefixes(model, clear_api_keys):
    with pytest.raises(RuntimeError, match=_OPENAI_MARKER):
        build_teacher(model)


@pytest.mark.parametrize(
    "model",
    ["bogus-model", "llama-3-70b", "mistral-7b", ""],
)
def test_build_teacher_unknown_prefix_raises_value_error(model, clear_api_keys):
    with pytest.raises(ValueError, match="unknown model id prefix"):
        build_teacher(model)


def test_build_teacher_routing_is_case_insensitive(clear_api_keys):
    """Model ids in the wild come in mixed case — we normalize."""
    with pytest.raises(RuntimeError, match=_ANTHROPIC_MARKER):
        build_teacher("Claude-Opus-4-6")
    with pytest.raises(RuntimeError, match=_OPENAI_MARKER):
        build_teacher("GPT-4o")


def test_teacher_client_base_is_abstract():
    """The base class exists as a contract — don't instantiate it directly."""
    assert hasattr(TeacherClient, "ask_for_pairs")
    # Subclasses must exist and be importable.
    assert issubclass(AnthropicTeacherClient, TeacherClient)
    assert issubclass(OpenAITeacherClient, TeacherClient)


# ---------------------------------------------------------------------------
# Phase 2: per-call chunking (_split_n_pairs)
# ---------------------------------------------------------------------------

def test_split_n_pairs_total_preserved():
    """
    Sum of per-call sizes must equal requested n_pairs — no silent loss.
    This is the property that matters most: if the user asks for 60,
    they must get 60 requested across the batch.
    """
    for n in [1, 8, 15, 30, 45, 60, 100]:
        for m in [5, 10, 15, 20]:
            split = _split_n_pairs(n, m)
            assert sum(split) == n, f"n={n} max={m} → {split}"
            # Every batch size must respect the cap.
            assert all(s <= m for s in split)
            # No zero-size batches.
            assert all(s > 0 for s in split)


def test_split_n_pairs_exact_multiple():
    assert _split_n_pairs(60, 15) == [15, 15, 15, 15]
    assert _split_n_pairs(30, 15) == [15, 15]
    assert _split_n_pairs(15, 15) == [15]


def test_split_n_pairs_with_remainder():
    assert _split_n_pairs(50, 15) == [15, 15, 15, 5]
    assert _split_n_pairs(16, 15) == [15, 1]


def test_split_n_pairs_below_cap_stays_single_call():
    """Small requests should NOT be padded to max_per_call."""
    assert _split_n_pairs(8, 15) == [8]
    assert _split_n_pairs(1, 15) == [1]


def test_split_n_pairs_zero_returns_empty():
    assert _split_n_pairs(0, 15) == []


def test_split_n_pairs_rejects_zero_max():
    with pytest.raises(ValueError, match="max_per_call must be positive"):
        _split_n_pairs(10, 0)
    with pytest.raises(ValueError, match="max_per_call must be positive"):
        _split_n_pairs(10, -1)
