# tests/test_export_sft_dataset.py
# -*- coding: utf-8 -*-
"""
Unit tests for ``scripts/export_sft_dataset``.

These tests cover the pure layers (record formatting, stable split,
stats computation) and the file-writing pipeline. The DB fetch path
(``fetch_rows``) is not exercised here — it requires PostgreSQL with
real ``intent_pair`` rows. Run that path through an integration test or
a manual export against the dev DB.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from scripts.export_sft_dataset import (
    IntentPairRow,
    compute_stats,
    format_sft_record,
    format_user_payload,
    split_and_format,
    stable_split,
    write_jsonl,
)
from server.chat.intent_agent import _intent_system_prompt


def _row(
    *,
    id: int,
    query: str = "test query",
    area: str = "electrochemistry",
    stage: str = "catalysis",
    source: str = "claude_teacher",
    stratum: str = "electrochemistry/CO2RR/simple",
    quality: float = 1.0,
) -> IntentPairRow:
    return IntentPairRow(
        id=id,
        query=query,
        intent_json={
            "stage": stage,
            "area": area,
            "task": "task",
            "summary": "summary",
        },
        schema_version=1,
        source=source,
        teacher_model="claude-opus-4-6",
        quality_score=quality,
        stratum=stratum,
    )


# ---------------------------------------------------------------------------
# format_user_payload — must mirror production
# ---------------------------------------------------------------------------


def test_format_user_payload_shape_matches_production():
    """
    intent_agent._api_intent_impl builds:
        {"query": ..., "guided": ..., "fewshots_hint": [...], "rag_hint": ...}
    The training payload must use the same keys with empty defaults so
    the student model sees the same input shape it will see at inference.
    """
    payload = json.loads(format_user_payload("hello world"))
    assert set(payload.keys()) == {"query", "guided", "fewshots_hint", "rag_hint"}
    assert payload["query"] == "hello world"
    assert payload["guided"] == {}
    assert payload["fewshots_hint"] == []
    assert payload["rag_hint"] == ""


# ---------------------------------------------------------------------------
# format_sft_record
# ---------------------------------------------------------------------------


def test_format_sft_record_has_three_messages_in_correct_order():
    rec = format_sft_record(_row(id=1))
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]


def test_format_sft_record_system_message_is_canonical_prompt():
    rec = format_sft_record(_row(id=1))
    assert rec["messages"][0]["content"] == _intent_system_prompt()


def test_format_sft_record_assistant_is_intent_json():
    row = _row(id=1, area="thermal_catalysis")
    rec = format_sft_record(row)
    assistant = json.loads(rec["messages"][2]["content"])
    assert assistant["area"] == "thermal_catalysis"
    assert assistant["stage"] == "catalysis"


def test_format_sft_record_metadata_carries_provenance():
    row = _row(id=42, source="claude_teacher_history", stratum="history/0007", quality=0.95)
    rec = format_sft_record(row)
    md = rec["metadata"]
    assert md["intent_pair_id"] == 42
    assert md["source"] == "claude_teacher_history"
    assert md["stratum"] == "history/0007"
    assert md["quality_score"] == 0.95
    assert md["schema_version"] == 1
    assert md["teacher_model"] == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# stable_split
# ---------------------------------------------------------------------------


def test_stable_split_is_deterministic():
    """Same id, same val_fraction → same bucket every time."""
    assert stable_split(123, 0.1) == stable_split(123, 0.1)
    assert stable_split(99999, 0.2) == stable_split(99999, 0.2)


def test_stable_split_returns_only_train_or_val():
    for i in range(200):
        assert stable_split(i, 0.1) in ("train", "val")


def test_stable_split_distribution_matches_target():
    """Across many ids the val fraction should land within ±3% of target."""
    n = 5000
    counts = Counter(stable_split(i, 0.1) for i in range(n))
    val_frac = counts["val"] / n
    assert 0.07 <= val_frac <= 0.13, f"val_frac={val_frac}"


def test_stable_split_id_does_not_drift_when_dataset_grows():
    """
    The whole point of hashing is that an id assigned to val today still
    lands in val tomorrow, even if 1000 new rows have been added.
    """
    sample_ids = [7, 42, 1337, 99999]
    snapshot = {i: stable_split(i, 0.1) for i in sample_ids}
    # Different val_fraction would naturally re-bucket, so we re-test the
    # same fraction.
    again = {i: stable_split(i, 0.1) for i in sample_ids}
    assert snapshot == again


def test_stable_split_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        stable_split(1, 0.0)
    with pytest.raises(ValueError):
        stable_split(1, 1.0)
    with pytest.raises(ValueError):
        stable_split(1, -0.1)


# ---------------------------------------------------------------------------
# split_and_format
# ---------------------------------------------------------------------------


def test_split_and_format_partitions_all_rows():
    rows = [_row(id=i) for i in range(50)]
    splits = split_and_format(rows, val_fraction=0.2)
    assert "train" in splits and "val" in splits
    assert len(splits["train"]) + len(splits["val"]) == 50
    # Every record should be a valid 3-message SFT row.
    for rec in splits["train"] + splits["val"]:
        assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]


def test_split_and_format_preserves_per_row_content():
    rows = [
        _row(id=1, query="alpha"),
        _row(id=2, query="beta"),
        _row(id=3, query="gamma"),
    ]
    splits = split_and_format(rows, val_fraction=0.5)
    seen_queries = set()
    for rec in splits["train"] + splits["val"]:
        payload = json.loads(rec["messages"][1]["content"])
        seen_queries.add(payload["query"])
    assert seen_queries == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# write_jsonl
# ---------------------------------------------------------------------------


def test_write_jsonl_creates_directory_and_writes_lines(tmp_path):
    records = [{"messages": []}, {"messages": [{"role": "user", "content": "hi"}]}]
    out = tmp_path / "nested" / "subdir" / "train.jsonl"
    n = write_jsonl(records, out)
    assert n == 2
    assert out.exists()
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[1]) == {"messages": [{"role": "user", "content": "hi"}]}


def test_write_jsonl_handles_empty_list(tmp_path):
    out = tmp_path / "empty.jsonl"
    n = write_jsonl([], out)
    assert n == 0
    assert out.exists()
    assert out.read_text() == ""


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


def test_compute_stats_aggregates_sources_areas_strata():
    rows = [
        _row(id=1, source="claude_teacher", area="electrochemistry"),
        _row(id=2, source="claude_teacher", area="electrochemistry"),
        _row(id=3, source="claude_teacher_history", area="thermal_catalysis"),
        _row(id=4, source="human_labeled", area="photocatalysis"),
    ]
    stats = compute_stats(rows)
    assert stats["n_rows"] == 4
    assert stats["by_source"] == {
        "claude_teacher": 2,
        "claude_teacher_history": 1,
        "human_labeled": 1,
    }
    assert stats["by_area"]["electrochemistry"] == 2
    assert stats["by_area"]["thermal_catalysis"] == 1
    assert stats["by_area"]["photocatalysis"] == 1
    assert stats["by_stage"]["catalysis"] == 4


def test_compute_stats_handles_missing_intent_fields():
    row = IntentPairRow(
        id=1,
        query="x",
        intent_json={},
        schema_version=1,
        source="x",
        teacher_model=None,
        quality_score=None,
        stratum=None,
    )
    stats = compute_stats([row])
    assert stats["n_rows"] == 1
    assert stats["by_area"] == {}
    assert stats["by_stage"] == {}
    assert stats["by_stratum"] == {}


# ---------------------------------------------------------------------------
# end-to-end: split → write → reload
# ---------------------------------------------------------------------------


def test_e2e_split_write_reload_roundtrip(tmp_path):
    rows = [_row(id=i, query=f"q{i}") for i in range(40)]
    splits = split_and_format(rows, val_fraction=0.25)

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    n_train = write_jsonl(splits["train"], train_path)
    n_val = write_jsonl(splits["val"], val_path)

    assert n_train + n_val == 40

    # Reload and verify shape.
    train_loaded = [json.loads(line) for line in train_path.read_text().strip().split("\n") if line]
    val_loaded = [json.loads(line) for line in val_path.read_text().strip().split("\n") if line]
    assert len(train_loaded) == n_train
    assert len(val_loaded) == n_val
    for rec in train_loaded + val_loaded:
        assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
        # The user payload must be parseable JSON with our 4 keys.
        payload = json.loads(rec["messages"][1]["content"])
        assert set(payload.keys()) == {"query", "guided", "fewshots_hint", "rag_hint"}
        # The assistant content must be parseable JSON.
        assistant = json.loads(rec["messages"][2]["content"])
        assert assistant["area"] == "electrochemistry"
