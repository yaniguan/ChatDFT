#!/usr/bin/env python3
# scripts/export_sft_dataset.py
# -*- coding: utf-8 -*-
"""
Export ``intent_pair`` rows to a supervised fine-tuning dataset.

Reads ``intent_pair`` from PostgreSQL, filters by schema version / source /
quality, formats each row as a 3-message conversation that EXACTLY mirrors
what production ``intent_agent`` sends to the LLM (so the student model
sees zero train/serve skew), does a stable hashed train/val split, and
writes ``train.jsonl`` + ``val.jsonl`` + ``stats.json`` to an output
directory.

The student we're targeting is ``Qwen/Qwen2.5-7B-Instruct`` served via
vLLM (see ``server/llm.yaml``). Both axolotl and llama-factory consume
JSONL where each line is ``{"messages": [{"role": "...", "content": ...}, ...]}``.

CLI
---
::

    # Default: SCHEMA_VERSION=current, all sources, quality >= 0.8, 90/10 split
    python -m scripts.export_sft_dataset --out artifacts/sft_v1

    # Only Claude-teacher rows, hard-cap at 2000 examples
    python -m scripts.export_sft_dataset \\
        --out artifacts/sft_v1 \\
        --sources claude_teacher,claude_teacher_history \\
        --max-rows 2000
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Allow ``python scripts/export_sft_dataset.py`` and ``-m`` invocation.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from server.chat.intent_agent import _intent_system_prompt  # noqa: E402
from server.chat.intent_schema import SCHEMA_VERSION  # noqa: E402

log = logging.getLogger("export_sft_dataset")


# ---------------------------------------------------------------------------
# Pure record formatting (covered by unit tests)
# ---------------------------------------------------------------------------

@dataclass
class IntentPairRow:
    """The subset of ``IntentPair`` columns this script consumes."""
    id: int
    query: str
    intent_json: Dict[str, Any]
    schema_version: int
    source: str
    teacher_model: Optional[str]
    quality_score: Optional[float]
    stratum: Optional[str]


def format_user_payload(query: str) -> str:
    """
    Build the user-message payload exactly as production ``intent_agent``
    constructs it (see ``_api_intent_impl``: ``user_payload = {query, guided,
    fewshots_hint, rag_hint}``). At training time we use empty values for
    everything except ``query`` so the student learns to operate from the
    raw query alone — the production agent's RAG/few-shot enrichment is
    handled separately and isn't reproducible from a stored row anyway.
    """
    payload = {
        "query": query,
        "guided": {},
        "fewshots_hint": [],
        "rag_hint": "",
    }
    return json.dumps(payload, ensure_ascii=False)


def format_sft_record(row: IntentPairRow) -> Dict[str, Any]:
    """
    Format one ``IntentPair`` row as an axolotl/sharegpt-style 3-message
    conversation. The system message is the canonical
    ``_intent_system_prompt()`` so the student trains against the EXACT
    contract production agents enforce.
    """
    return {
        "messages": [
            {"role": "system", "content": _intent_system_prompt()},
            {"role": "user", "content": format_user_payload(row.query)},
            {"role": "assistant", "content": json.dumps(row.intent_json, ensure_ascii=False)},
        ],
        "metadata": {
            "intent_pair_id": row.id,
            "source": row.source,
            "teacher_model": row.teacher_model,
            "stratum": row.stratum,
            "quality_score": row.quality_score,
            "schema_version": row.schema_version,
        },
    }


def stable_split(row_id: int, val_fraction: float) -> str:
    """
    Stable train/val assignment by hashing the row id. The same row always
    lands in the same split across runs, even if the row count changes —
    so retraining on the latest data does not silently leak val rows into
    train.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    digest = hashlib.sha256(f"intent_pair:{row_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2 ** 64
    return "val" if bucket < val_fraction else "train"


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------

async def fetch_rows(
    *,
    min_schema_version: int,
    sources: Optional[Sequence[str]],
    min_quality: float,
    max_rows: int,
) -> List[IntentPairRow]:
    """
    Pull rows matching the export filters. Ordered by id ASC so the stable
    split assignment stays deterministic across reruns.
    """
    from sqlalchemy import select

    from server.db import IntentPair, SessionLocal

    async with SessionLocal() as session:
        stmt = select(IntentPair).where(
            IntentPair.schema_version >= min_schema_version,
            IntentPair.quality_score >= min_quality,
        )
        if sources:
            stmt = stmt.where(IntentPair.source.in_(list(sources)))
        stmt = stmt.order_by(IntentPair.id.asc())
        if max_rows:
            stmt = stmt.limit(max_rows)
        rows = (await session.execute(stmt)).scalars().all()

    return [
        IntentPairRow(
            id=int(r.id),
            query=str(r.query),
            intent_json=dict(r.intent_json or {}),
            schema_version=int(r.schema_version or 0),
            source=str(r.source or ""),
            teacher_model=r.teacher_model,
            quality_score=float(r.quality_score) if r.quality_score is not None else None,
            stratum=r.stratum,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Export pipeline
# ---------------------------------------------------------------------------

def split_and_format(
    rows: Sequence[IntentPairRow],
    val_fraction: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """Apply ``stable_split`` and format every row. Pure function."""
    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    for row in rows:
        split = stable_split(row.id, val_fraction)
        out[split].append(format_sft_record(row))
    return out


def write_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def compute_stats(rows: Sequence[IntentPairRow]) -> Dict[str, Any]:
    by_source: Dict[str, int] = {}
    by_stratum: Dict[str, int] = {}
    by_area: Dict[str, int] = {}
    by_stage: Dict[str, int] = {}
    for r in rows:
        by_source[r.source] = by_source.get(r.source, 0) + 1
        if r.stratum:
            by_stratum[r.stratum] = by_stratum.get(r.stratum, 0) + 1
        area = (r.intent_json or {}).get("area")
        if isinstance(area, str):
            by_area[area] = by_area.get(area, 0) + 1
        stage = (r.intent_json or {}).get("stage")
        if isinstance(stage, str):
            by_stage[stage] = by_stage.get(stage, 0) + 1
    return {
        "n_rows": len(rows),
        "by_source": by_source,
        "by_area": by_area,
        "by_stage": by_stage,
        "by_stratum": dict(sorted(by_stratum.items(), key=lambda kv: -kv[1])),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sources = (
        [s.strip() for s in args.sources.split(",") if s.strip()]
        if args.sources
        else None
    )

    rows = await fetch_rows(
        min_schema_version=args.min_schema_version,
        sources=sources,
        min_quality=args.min_quality,
        max_rows=args.max_rows,
    )
    log.info(
        "fetched %d intent_pair rows (schema_version >= %d, quality >= %s, sources=%s)",
        len(rows),
        args.min_schema_version,
        args.min_quality,
        sources or "ALL",
    )
    if not rows:
        log.error("no rows matched the export filter — nothing to write")
        return 1

    splits = split_and_format(rows, val_fraction=args.val_fraction)
    out_dir = Path(args.out)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    stats_path = out_dir / "stats.json"

    n_train = write_jsonl(splits["train"], train_path)
    n_val = write_jsonl(splits["val"], val_path)

    stats = compute_stats(rows)
    stats.update({
        "n_train": n_train,
        "n_val": n_val,
        "val_fraction_target": args.val_fraction,
        "val_fraction_actual": round(n_val / max(1, n_train + n_val), 4),
        "schema_version": SCHEMA_VERSION,
        "min_schema_version": args.min_schema_version,
        "min_quality": args.min_quality,
        "sources": sources,
    })
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    log.info("wrote %d train + %d val rows to %s", n_train, n_val, out_dir)
    print(json.dumps({
        "n_train": n_train,
        "n_val": n_val,
        "out_dir": str(out_dir),
    }, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export intent_pair rows to an SFT JSONL dataset.",
    )
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory (train.jsonl, val.jsonl, stats.json land here).")
    p.add_argument("--min-schema-version", type=int, default=SCHEMA_VERSION,
                   help="Drop rows with schema_version below this (default: current).")
    p.add_argument("--sources", default="",
                   help="Comma-separated list of source tags to include (default: all). "
                        "e.g. 'claude_teacher,claude_teacher_history'.")
    p.add_argument("--min-quality", type=float, default=0.8,
                   help="Drop rows with quality_score below this.")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="Fraction of rows assigned to validation split.")
    p.add_argument("--max-rows", type=int, default=0,
                   help="Cap on rows pulled from DB (0 = no cap).")
    p.add_argument("--verbose", action="store_true", help="DEBUG logging.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
