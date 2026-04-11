#!/usr/bin/env python3
# scripts/verify_sft_dataset.py
# -*- coding: utf-8 -*-
"""
Local sanity-check for the SFT JSONL dataset exported by
``scripts/export_sft_dataset.py``.

Why this exists
---------------
The most common ways an SFT dataset breaks a LoRA run are **not** things
axolotl catches loudly. You typically waste an hour of GPU time and end
up with a model that emits garbage because:

* The assistant content is not valid JSON (the student never learns to
  close braces because a few rows in training had a trailing comma).
* The user payload shape drifted from what production sends at
  inference, so the student learns a superset schema.
* The role sequence is off by one (axolotl happily trains on
  system/user/user/assistant and you never notice).
* One record has a 40k-char content field because the teacher looped,
  blowing the sequence_len budget and silently truncating.

This script is pure-Python (no torch, no transformers) so it runs in any
env in ~half a second. Run it before ``train_qwen_lora.sh`` every time.

Usage
-----
::

    python -m scripts.verify_sft_dataset artifacts/sft_v1

Exit codes
----------
    0  — all checks passed
    1  — at least one structural error; won't train correctly
    2  — warnings only (e.g. very short dataset, low diversity)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from server.chat.intent_schema import validate_intent  # noqa: E402

# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------

@dataclass
class RecordError:
    line_no: int
    reason: str


@dataclass
class SplitReport:
    path: Path
    n_records: int = 0
    n_errors: int = 0
    errors: List[RecordError] = field(default_factory=list)
    # lengths in characters
    system_len: List[int] = field(default_factory=list)
    user_len: List[int] = field(default_factory=list)
    assistant_len: List[int] = field(default_factory=list)
    # distributions
    areas: Counter = field(default_factory=Counter)
    stages: Counter = field(default_factory=Counter)

    def record_error(self, line_no: int, reason: str) -> None:
        self.n_errors += 1
        self.errors.append(RecordError(line_no=line_no, reason=reason))

    def summary(self) -> Dict[str, Any]:
        def stats(xs: List[int]) -> Dict[str, float]:
            if not xs:
                return {}
            return {
                "min": min(xs), "p50": int(statistics.median(xs)),
                "max": max(xs), "mean": int(statistics.mean(xs)),
            }
        return {
            "path": str(self.path),
            "n_records": self.n_records,
            "n_errors": self.n_errors,
            "content_lengths_chars": {
                "system": stats(self.system_len),
                "user": stats(self.user_len),
                "assistant": stats(self.assistant_len),
            },
            "area_distribution": dict(self.areas),
            "stage_distribution": dict(self.stages),
        }


# ---------------------------------------------------------------------------
# record validation
# ---------------------------------------------------------------------------

_REQUIRED_USER_KEYS = {"query", "guided", "fewshots_hint", "rag_hint"}


def validate_record(record: Dict[str, Any]) -> Optional[str]:
    """
    Validate a single SFT record. Returns ``None`` on success or a short
    human-readable reason on failure.
    """
    messages = record.get("messages")
    if not isinstance(messages, list):
        return "'messages' is not a list"
    if len(messages) != 3:
        return f"expected exactly 3 messages, got {len(messages)}"

    # role sequence check
    expected_roles = ("system", "user", "assistant")
    for i, (msg, expected) in enumerate(zip(messages, expected_roles)):
        if not isinstance(msg, dict):
            return f"messages[{i}] is not a dict"
        if msg.get("role") != expected:
            return f"messages[{i}].role = {msg.get('role')!r}, expected {expected!r}"
        if not isinstance(msg.get("content"), str):
            return f"messages[{i}].content is not a string"
        if not msg["content"].strip():
            return f"messages[{i}].content is empty"

    # user payload must be a JSON object with the production-keys set
    try:
        user_payload = json.loads(messages[1]["content"])
    except json.JSONDecodeError as e:
        return f"user content is not valid JSON: {e}"
    if not isinstance(user_payload, dict):
        return "user content JSON is not an object"
    missing = _REQUIRED_USER_KEYS - set(user_payload.keys())
    if missing:
        return f"user payload missing keys: {sorted(missing)}"

    # assistant content must be JSON that passes IntentSchema
    try:
        intent = json.loads(messages[2]["content"])
    except json.JSONDecodeError as e:
        return f"assistant content is not valid JSON: {e}"
    if not isinstance(intent, dict):
        return "assistant content JSON is not an object"
    model, err = validate_intent(intent)
    if model is None:
        return f"assistant intent fails IntentSchema: {err}"

    return None


def _extract_distributions(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Pull (area, stage) from an already-valid record for the distribution report."""
    try:
        intent = json.loads(record["messages"][2]["content"])
        return intent.get("area"), intent.get("stage")
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# file-level verification
# ---------------------------------------------------------------------------

def verify_file(path: Path) -> SplitReport:
    """Verify one JSONL split file and return a SplitReport."""
    report = SplitReport(path=path)
    if not path.exists():
        report.record_error(0, f"file not found: {path}")
        return report

    with path.open() as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            report.n_records += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                report.record_error(line_no, f"line is not valid JSON: {e}")
                continue
            reason = validate_record(rec)
            if reason:
                report.record_error(line_no, reason)
                continue
            # success path — collect stats
            msgs = rec["messages"]
            report.system_len.append(len(msgs[0]["content"]))
            report.user_len.append(len(msgs[1]["content"]))
            report.assistant_len.append(len(msgs[2]["content"]))
            area, stage = _extract_distributions(rec)
            if area:
                report.areas[area] += 1
            if stage:
                report.stages[stage] += 1
    return report


def print_two_samples(path: Path, limit: int = 2) -> None:
    """Print ``limit`` full SFT records for visual inspection."""
    if not path.exists():
        return
    print(f"\n─── sample records from {path.name} ───")
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            rec = json.loads(line)
            user = json.loads(rec["messages"][1]["content"])
            asst = json.loads(rec["messages"][2]["content"])
            print(f"\n  [#{i}] intent_pair_id={rec.get('metadata', {}).get('intent_pair_id')}")
            print(f"  query:     {user['query']}")
            print(f"  stage:     {asst.get('stage')}")
            print(f"  area:      {asst.get('area')}")
            print(f"  substrate: {asst.get('substrate')}")
            print(f"  conds:     { {k: v for k, v in (asst.get('conditions') or {}).items() if v is not None} }")
            print(f"  tags:      {asst.get('tags')}")


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------

def verify_sft_directory(
    root: Path,
    *,
    warn_below: int = 500,
    warn_min_area_ratio: int = 5,
) -> int:
    """
    Run verification over ``root/train.jsonl`` and ``root/val.jsonl``.
    Returns the process exit code: 0 clean, 1 errors, 2 warnings only.
    """
    train = verify_file(root / "train.jsonl")
    val = verify_file(root / "val.jsonl")

    # ── errors section ────────────────────────────────────────────────────
    print("═" * 60)
    print(f"  SFT dataset verification: {root}")
    print("═" * 60)
    for report in (train, val):
        print(f"\n{report.path.name}: {report.n_records} records, "
              f"{report.n_errors} errors")
        if report.errors:
            print("  first 5 errors:")
            for err in report.errors[:5]:
                print(f"    line {err.line_no}: {err.reason}")

    # ── structural errors block the run ───────────────────────────────────
    total_errors = train.n_errors + val.n_errors
    if total_errors > 0:
        print(f"\nFAIL: {total_errors} structural errors across splits.")
        return 1

    # ── stats + sample records ────────────────────────────────────────────
    print("\n─── summary ───")
    print(json.dumps(train.summary(), indent=2))
    print(json.dumps(val.summary(), indent=2))

    print_two_samples(root / "train.jsonl", limit=2)
    print_two_samples(root / "val.jsonl", limit=1)

    # ── warnings (don't block the run) ────────────────────────────────────
    warnings: List[str] = []
    if train.n_records < warn_below:
        warnings.append(
            f"train split has only {train.n_records} records — "
            f"LoRA tends to underfit Qwen-7B below ~{warn_below} examples "
            f"on a structured-output task."
        )
    if train.areas:
        max_area = max(train.areas.values())
        min_area = min(train.areas.values())
        if min_area == 0:
            warnings.append("at least one area is absent from the train split")
        elif max_area / min_area > warn_min_area_ratio:
            top = train.areas.most_common(1)[0]
            bot = train.areas.most_common()[-1]
            warnings.append(
                f"area imbalance: {top[0]}={top[1]} vs {bot[0]}={bot[1]} "
                f"(ratio {max_area / min_area:.1f}× > {warn_min_area_ratio}×)"
            )
    if train.assistant_len and max(train.assistant_len) > 8000:
        warnings.append(
            f"longest assistant content = {max(train.assistant_len)} chars; "
            f"consider raising sequence_len above 4096 in qwen_lora.yaml"
        )

    if warnings:
        print("\n─── warnings ───")
        for w in warnings:
            print(f"  ⚠ {w}")
        print("\nPASS (with warnings).")
        return 2

    print("\nPASS. Dataset is ready to train.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sanity-check an SFT JSONL dataset before LoRA training.",
    )
    parser.add_argument("root", type=Path, nargs="?",
                        default=Path("artifacts/sft_v1"),
                        help="Directory containing train.jsonl + val.jsonl")
    parser.add_argument("--warn-below", type=int, default=500,
                        help="Warn if train has fewer than N records.")
    parser.add_argument("--warn-min-area-ratio", type=int, default=5,
                        help="Warn if max/min area count exceeds this ratio.")
    args = parser.parse_args(argv)
    return verify_sft_directory(
        args.root,
        warn_below=args.warn_below,
        warn_min_area_ratio=args.warn_min_area_ratio,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
