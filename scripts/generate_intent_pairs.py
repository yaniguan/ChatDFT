#!/usr/bin/env python3
# scripts/generate_intent_pairs.py
# -*- coding: utf-8 -*-
"""
Claude-teacher pipeline for the intent_pair training set.

For each stratum in ``scripts/intent_pair_strata.yaml``, this script asks
an Anthropic teacher model (default: ``claude-opus-4-6``) for N
``(query, intent)`` pairs in a single structured response. Every pair is
validated against ``IntentSchema``; survivors are embedded with the
configured embedding model and deduplicated against (a) the in-memory
accepted set for the current run and (b) optionally the existing
``IntentPair`` rows in PostgreSQL. Accepted pairs are inserted with full
provenance metadata (source, teacher_model, stratum, schema_version).

Why this exists
---------------
Phase 1 of the intent agent improvement plan: replace the brittle inline
GPT-4o-mini call with a fine-tuned Qwen-7B trained on a high-quality
``(query, intent)`` corpus. This script is the corpus-building half.

Anthropic prompt caching is enabled on the (long) system prompt, which
cuts cost by ~5x when running across all strata in one invocation.

CLI
---
::

    # Tiny smoke test against one stratum, no DB writes:
    python -m scripts.generate_intent_pairs \\
        --limit-strata 1 --n-per-stratum 3 --dry-run

    # Full run, writes to PostgreSQL:
    export ANTHROPIC_API_KEY=sk-ant-...
    export DATABASE_URL=postgresql+asyncpg://yaniguan@localhost/chatdft_ase
    python -m scripts.generate_intent_pairs \\
        --n-per-stratum 8 \\
        --teacher-model claude-opus-4-6 \\
        --similarity-threshold 0.92

    # Smaller-budget Sonnet run:
    python -m scripts.generate_intent_pairs \\
        --teacher-model claude-sonnet-4-6 --max-pairs 500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Allow ``python scripts/generate_intent_pairs.py`` and
# ``python -m scripts.generate_intent_pairs`` to both work.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from server.chat.intent_agent import _intent_system_prompt  # noqa: E402
from server.chat.intent_schema import (  # noqa: E402
    AREA_VALUES,
    SCHEMA_VERSION,
    validate_intent,
)

log = logging.getLogger("generate_intent_pairs")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Stratum:
    id: str
    area: str
    family: str
    difficulty: str
    anchors: List[str] = field(default_factory=list)
    guidance: Optional[str] = None


@dataclass
class RawPair:
    """A teacher-emitted pair before validation."""
    query: str
    intent: Dict[str, Any]


@dataclass
class ValidatedPair:
    """A pair that passed IntentSchema validation."""
    query: str
    intent: Dict[str, Any]
    stratum_id: str

    # Filled in later by the embedding pass.
    embedding: Optional[List[float]] = None


@dataclass
class GenerationStats:
    requested: int = 0
    teacher_emitted: int = 0
    schema_invalid: int = 0
    duplicates_dropped: int = 0
    accepted: int = 0
    api_errors: int = 0

    def merge(self, other: "GenerationStats") -> None:
        self.requested += other.requested
        self.teacher_emitted += other.teacher_emitted
        self.schema_invalid += other.schema_invalid
        self.duplicates_dropped += other.duplicates_dropped
        self.accepted += other.accepted
        self.api_errors += other.api_errors

    def to_dict(self) -> Dict[str, int]:
        return {
            "requested": self.requested,
            "teacher_emitted": self.teacher_emitted,
            "schema_invalid": self.schema_invalid,
            "duplicates_dropped": self.duplicates_dropped,
            "accepted": self.accepted,
            "api_errors": self.api_errors,
        }


# ---------------------------------------------------------------------------
# Strata loading
# ---------------------------------------------------------------------------

DEFAULT_STRATA_PATH = _REPO_ROOT / "scripts" / "intent_pair_strata.yaml"


def load_strata(path: Optional[Path] = None) -> List[Stratum]:
    """Parse the strata YAML and return validated ``Stratum`` records."""
    import yaml  # local import keeps non-CLI imports fast

    target = Path(path) if path else DEFAULT_STRATA_PATH
    blob = yaml.safe_load(target.read_text())
    if not isinstance(blob, dict) or "strata" not in blob:
        raise ValueError(f"{target}: missing top-level `strata` key")

    out: List[Stratum] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(blob["strata"] or []):
        if not isinstance(raw, dict):
            raise ValueError(f"{target}: stratum #{i} is not a mapping")
        try:
            sid = str(raw["id"])
            area = str(raw["area"])
            family = str(raw["family"])
            difficulty = str(raw["difficulty"])
        except KeyError as e:
            raise ValueError(f"{target}: stratum #{i} missing key {e}") from None

        if area not in AREA_VALUES:
            raise ValueError(
                f"{target}: stratum {sid!r} has invalid area {area!r}; "
                f"must be one of {list(AREA_VALUES)}"
            )
        if difficulty not in ("simple", "medium", "hard"):
            raise ValueError(
                f"{target}: stratum {sid!r} has invalid difficulty {difficulty!r}"
            )
        if sid in seen_ids:
            raise ValueError(f"{target}: duplicate stratum id {sid!r}")
        seen_ids.add(sid)

        anchors = raw.get("anchors") or []
        if not isinstance(anchors, list):
            raise ValueError(f"{target}: stratum {sid!r} `anchors` must be a list")

        out.append(
            Stratum(
                id=sid,
                area=area,
                family=family,
                difficulty=difficulty,
                anchors=[str(a) for a in anchors],
                guidance=raw.get("guidance"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_TEACHER_ADDENDUM = """

ROLE
====
You are generating high-quality training data for the intent parser
described in the schema above. Each example you produce will be used as
supervised fine-tuning data for a smaller student model. Your job is to
make the (query, intent) pairs realistic, diverse, and chemically
meaningful — *not* to repeat anchor phrases verbatim.

OUTPUT FORMAT
=============
Return STRICT JSON only — no prose, no code fences. The JSON must have
exactly this shape:

  {"pairs": [
    {"query": "<a natural-language user query>",
     "intent": {<full IntentSchema JSON object>}},
    ...
  ]}

Each `intent` field must independently satisfy the IntentSchema described
above (correct enum values, non-empty `task` and `summary`, all required
keys, lists present even if empty).

DIVERSITY RULES
===============
- Vary surface phrasing across pairs in the same response. Use different
  sentence structures, different levels of detail, different units, etc.
- Mix research questions ("what is the rate-limiting step…") with
  task requests ("compute the free-energy diagram…").
- Some queries should be terse (one line), others should include
  explicit conditions and target metrics.
- Do NOT copy the anchor topics verbatim — use them as inspiration to
  generate adjacent but distinct queries.
"""


def build_teacher_system_prompt() -> str:
    """The cached system prompt: schema + teacher-role addendum."""
    return _intent_system_prompt() + _TEACHER_ADDENDUM


def build_stratum_prompt(stratum: Stratum, n_pairs: int) -> str:
    """The per-stratum user prompt."""
    anchor_block = "\n".join(f"  - {a}" for a in stratum.anchors) or "  (none)"
    guidance_block = f"\nStratum guidance: {stratum.guidance}" if stratum.guidance else ""
    return (
        f"Generate {n_pairs} distinct (query, intent) pairs for the following "
        f"stratum:\n"
        f"\n"
        f"  area:       {stratum.area}\n"
        f"  family:     {stratum.family}\n"
        f"  difficulty: {stratum.difficulty}\n"
        f"  topic anchors:\n{anchor_block}"
        f"{guidance_block}\n"
        f"\n"
        f"Every `intent.area` MUST be exactly {stratum.area!r}. "
        f"Vary the substrate, conditions, and phrasing across the {n_pairs} pairs. "
        f"Return STRICT JSON only as {{\"pairs\": [...]}}."
    )


def build_history_prompt(queries: Sequence[str]) -> str:
    """
    Per-batch user prompt for the ``--from-history`` mode. The teacher
    receives N real user queries and returns one labeled pair per query,
    in the same order. Reuses the canonical ``{"pairs": [...]}`` envelope
    so ``parse_teacher_response`` works without modification.
    """
    if not queries:
        raise ValueError("build_history_prompt: empty query list")
    numbered = "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(queries))
    return (
        f"Label each of the following {len(queries)} REAL user queries with a "
        f"canonical intent JSON. These are actual messages users have sent to "
        f"ChatDFT — do NOT rephrase them, do NOT skip any, do NOT add new ones.\n"
        f"\n"
        f"Queries:\n"
        f"{numbered}\n"
        f"\n"
        f"Return a STRICT JSON object of the form "
        f'{{"pairs": [{{"query": "<verbatim query 1>", "intent": {{...}}}}, ...]}}. '
        f"There must be exactly {len(queries)} entries, in input order, with each "
        f"`query` matching the input verbatim. Each `intent` must independently "
        f"satisfy the IntentSchema described in the system prompt."
    )


# ---------------------------------------------------------------------------
# Response parsing + validation
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        # ``` or ```json
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()


def parse_teacher_response(text: str) -> List[RawPair]:
    """
    Extract a list of ``RawPair`` from a teacher response. Tolerant of code
    fences and stray prose; raises ``ValueError`` only if no JSON object is
    recoverable at all.
    """
    if not text:
        return []
    cleaned = _strip_code_fences(text)
    # Find the outermost JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("teacher response contains no JSON object")
    try:
        blob = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"teacher response is not valid JSON: {e}") from e

    pairs_raw = blob.get("pairs")
    if not isinstance(pairs_raw, list):
        raise ValueError("teacher response missing top-level `pairs` list")

    out: List[RawPair] = []
    for item in pairs_raw:
        if not isinstance(item, dict):
            continue
        query = item.get("query")
        intent = item.get("intent")
        if not isinstance(query, str) or not query.strip():
            continue
        if not isinstance(intent, dict):
            continue
        out.append(RawPair(query=query.strip(), intent=intent))
    return out


def validate_pairs(
    raw_pairs: Sequence[RawPair],
    stratum_id: str,
) -> Tuple[List[ValidatedPair], List[Tuple[RawPair, str]]]:
    """
    Run every raw pair through ``IntentSchema``. Returns
    ``(valid_pairs, invalid_pairs_with_reason)``.
    """
    valid: List[ValidatedPair] = []
    invalid: List[Tuple[RawPair, str]] = []
    for pair in raw_pairs:
        model, err = validate_intent(pair.intent)
        if model is None:
            invalid.append((pair, err or "unknown validation error"))
            continue
        valid.append(
            ValidatedPair(
                query=pair.query,
                intent=model.model_dump(mode="json"),
                stratum_id=stratum_id,
            )
        )
    return valid, invalid


# ---------------------------------------------------------------------------
# Embedding + dedup
# ---------------------------------------------------------------------------

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def dedup_by_embedding(
    candidates: List[ValidatedPair],
    existing_embeddings: Sequence[Sequence[float]],
    threshold: float = 0.92,
) -> Tuple[List[ValidatedPair], int]:
    """
    Drop candidates whose embedding is within ``threshold`` cosine
    similarity of either (a) any prior accepted candidate in the same
    batch or (b) any vector in ``existing_embeddings``.

    Returns ``(accepted, n_dropped)``. Each accepted candidate must have
    ``embedding`` populated.
    """
    accepted: List[ValidatedPair] = []
    accepted_embs: List[Sequence[float]] = list(existing_embeddings)
    dropped = 0
    for cand in candidates:
        if cand.embedding is None:
            raise ValueError(f"candidate {cand.query!r} missing embedding")
        is_dup = any(
            _cosine(cand.embedding, prev) >= threshold for prev in accepted_embs
        )
        if is_dup:
            dropped += 1
            continue
        accepted.append(cand)
        accepted_embs.append(cand.embedding)
    return accepted, dropped


async def embed_queries(queries: List[str]) -> List[List[float]]:
    """Embed via the configured router (text-embedding-3-small by default)."""
    if not queries:
        return []
    from server.utils.openai_wrapper import embed_texts
    return await embed_texts(queries, agent_name="intent_pair_generator")


# ---------------------------------------------------------------------------
# Anthropic client + caching
# ---------------------------------------------------------------------------

class TeacherClient:
    """
    Abstract teacher-client interface.

    Concrete backends implement ``ask_for_pairs(user_prompt) -> str``
    returning the raw model response text. The generation pipeline
    remains backend-agnostic so the ``{"pairs": [...]}`` response parser
    and the IntentSchema validator are shared.

    Construct via :func:`build_teacher` — it auto-detects the right
    backend from the model id prefix.
    """

    model: str
    max_tokens: int

    async def ask_for_pairs(self, user_prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError


class AnthropicTeacherClient(TeacherClient):
    """
    Thin async wrapper around ``anthropic.AsyncAnthropic`` that pins the
    cached system prompt. Prompt caching cuts cost by ~5x across a full
    strata sweep because the ~3k-token system prompt is reused 28 times.
    """

    def __init__(self, model: str, max_tokens: int = 8000) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "anthropic SDK is required for the Anthropic teacher backend. "
                "Install with: pip install anthropic"
            ) from e

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is required."
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self._system_blocks = [
            {
                "type": "text",
                "text": build_teacher_system_prompt(),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    async def ask_for_pairs(self, user_prompt: str) -> str:
        """Send one stratum prompt and return the raw response text."""
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.7,
            system=self._system_blocks,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Concatenate all text blocks (the response is normally one block).
        return "".join(
            blk.text for blk in resp.content if getattr(blk, "type", None) == "text"
        )


class OpenAITeacherClient(TeacherClient):
    """
    Thin async wrapper around ``openai.AsyncOpenAI``.

    Uses ``response_format={"type": "json_object"}`` which GPT-4o family
    honours rigorously — we've seen zero malformed-JSON failures on
    60-pair batches where Claude Sonnet occasionally emits unclosed
    strings. No prompt caching (OpenAI doesn't expose it on the chat
    endpoint), so per-call cost is slightly higher than Anthropic Opus
    at the same output volume, but still cheaper end-to-end because
    gpt-4o is ~10x cheaper per output token than Opus.
    """

    def __init__(self, model: str, max_tokens: int = 8000) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "openai SDK is required for the OpenAI teacher backend. "
                "Install with: pip install openai"
            ) from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is required."
            )
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self._system_prompt = build_teacher_system_prompt()

    async def ask_for_pairs(self, user_prompt: str) -> str:
        """Send one stratum prompt and return the raw response text."""
        resp = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content
        return content or ""


def build_teacher(
    model: str,
    *,
    max_tokens: int = 8000,
) -> TeacherClient:
    """
    Factory: pick the right teacher backend for a model id.

    Routing rules (by prefix):
        * ``claude-*``       → :class:`AnthropicTeacherClient`
        * ``gpt-*`` / ``o1*``/ ``o3*`` / ``chatgpt-*``
                             → :class:`OpenAITeacherClient`

    Unknown prefixes raise ``ValueError`` — the caller must fix the
    model id rather than silently dispatch to an arbitrary backend.
    """
    lowered = model.lower()
    if lowered.startswith("claude-"):
        return AnthropicTeacherClient(model=model, max_tokens=max_tokens)
    if (
        lowered.startswith("gpt-")
        or lowered.startswith("o1")
        or lowered.startswith("o3")
        or lowered.startswith("chatgpt-")
    ):
        return OpenAITeacherClient(model=model, max_tokens=max_tokens)
    raise ValueError(
        f"build_teacher: unknown model id prefix {model!r}. "
        f"Expected claude-* or gpt-* / o1* / o3* / chatgpt-*."
    )


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------

async def fetch_existing_query_embeddings(limit: int = 5000) -> List[List[float]]:
    """
    Pull recent ``query_embedding`` vectors from ``intent_pair`` so the
    dedup pass can also catch cross-run duplicates. Limited because we
    only need a representative sample for cosine comparison.
    """
    from sqlalchemy import select

    from server.db import IntentPair, SessionLocal

    async with SessionLocal() as session:
        stmt = (
            select(IntentPair.query_embedding)
            .where(IntentPair.query_embedding.isnot(None))
            .order_by(IntentPair.created_at.desc())
            .limit(limit)
        )
        rows = (await session.execute(stmt)).scalars().all()

    out: List[List[float]] = []
    for r in rows:
        if r is None:
            continue
        # pgvector returns a numpy-like array; coerce to plain list of floats.
        try:
            out.append([float(x) for x in r])
        except (TypeError, ValueError):
            continue
    return out


async def load_user_history(
    *,
    limit: int = 200,
    min_chars: int = 12,
    max_chars: int = 2000,
) -> List[str]:
    """
    Pull recent ``role='user'`` queries out of ``chat_message`` for the
    ``--from-history`` mode. Drops queries that are too short to be
    meaningful or absurdly long, and de-duplicates verbatim repeats while
    preserving order.
    """
    from sqlalchemy import select

    from server.db import ChatMessage, SessionLocal

    async with SessionLocal() as session:
        stmt = (
            select(ChatMessage.content)
            .where(ChatMessage.role == "user")
            .order_by(ChatMessage.created_at.desc())
            .limit(limit * 3)  # over-pull, then filter
        )
        rows = (await session.execute(stmt)).scalars().all()

    seen: set[str] = set()
    out: List[str] = []
    for content in rows:
        if not content:
            continue
        text = str(content).strip()
        if len(text) < min_chars or len(text) > max_chars:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _chunk(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


async def relabel_history_batch(
    queries: List[str],
    *,
    batch_id: str,
    teacher: "TeacherClient",
    existing_embs: List[List[float]],
    similarity_threshold: float,
    dry_run: bool,
) -> Tuple[List[ValidatedPair], GenerationStats]:
    """
    Relabel one batch of real user queries via Claude. Mirrors
    ``generate_for_stratum`` but the user prompt is verbatim queries
    instead of a stratum description.
    """
    stats = GenerationStats(requested=len(queries))
    user_prompt = build_history_prompt(queries)

    try:
        raw_text = await teacher.ask_for_pairs(user_prompt)
    except Exception as e:
        log.warning("[history/%s] teacher API error: %s", batch_id, e)
        stats.api_errors += 1
        return [], stats

    try:
        raw_pairs = parse_teacher_response(raw_text)
    except ValueError as e:
        log.warning("[history/%s] could not parse teacher response: %s", batch_id, e)
        stats.api_errors += 1
        return [], stats

    stats.teacher_emitted = len(raw_pairs)
    valid, invalid = validate_pairs(raw_pairs, stratum_id=f"history/{batch_id}")
    stats.schema_invalid = len(invalid)

    if not valid:
        return [], stats

    if dry_run:
        for p in valid:
            p.embedding = [0.0] * 1536
    else:
        try:
            embs = await embed_queries([p.query for p in valid])
        except Exception as e:
            log.warning("[history/%s] embedding error: %s", batch_id, e)
            stats.api_errors += 1
            return [], stats
        for p, emb in zip(valid, embs):
            p.embedding = list(emb)

    accepted, dropped = dedup_by_embedding(
        valid, existing_embs, threshold=similarity_threshold
    )
    stats.duplicates_dropped = dropped
    stats.accepted = len(accepted)

    for p in accepted:
        if p.embedding is not None:
            existing_embs.append(p.embedding)

    return accepted, stats


async def insert_pairs(
    pairs: List[ValidatedPair],
    *,
    teacher_model: str,
    source: str,
    schema_version: int,
) -> int:
    """Bulk-insert accepted pairs and return the count actually written."""
    if not pairs:
        return 0
    from datetime import datetime

    from server.db import IntentPair, SessionLocal

    async with SessionLocal() as session:
        for p in pairs:
            session.add(
                IntentPair(
                    query=p.query,
                    query_embedding=p.embedding,
                    intent_json=p.intent,
                    schema_version=schema_version,
                    source=source,
                    teacher_model=teacher_model,
                    quality_score=1.0,
                    stratum=p.stratum_id,
                    notes=None,
                    created_at=datetime.utcnow(),
                )
            )
        await session.commit()
    return len(pairs)


# ---------------------------------------------------------------------------
# Per-stratum runner
# ---------------------------------------------------------------------------

def _split_n_pairs(n_pairs: int, max_per_call: int) -> List[int]:
    """
    Split ``n_pairs`` into a list of per-call batch sizes, each ≤
    ``max_per_call``. Preserves the total even when ``n_pairs`` is not a
    multiple of ``max_per_call``.

    Example::

        _split_n_pairs(60, 15)  → [15, 15, 15, 15]
        _split_n_pairs(50, 15)  → [15, 15, 15, 5]
        _split_n_pairs(8, 15)   → [8]
    """
    if n_pairs <= 0:
        return []
    if max_per_call <= 0:
        raise ValueError("max_per_call must be positive")
    full = n_pairs // max_per_call
    rest = n_pairs % max_per_call
    out = [max_per_call] * full
    if rest:
        out.append(rest)
    return out


async def _ask_teacher_once(
    stratum: Stratum,
    n_pairs: int,
    teacher: TeacherClient,
    stats: GenerationStats,
    label: str,
    *,
    max_attempts: int = 3,
) -> List[RawPair]:
    """
    One teacher call + response parse, with bounded retry.

    Retries cover two failure modes we see in practice:

    * **Transient 400/429/5xx from OpenAI or Anthropic.** OpenAI
      occasionally returns "We could not parse the JSON body of your
      request" under load even when the request is well-formed — retrying
      usually works. Rate limits and 5xxs are the obvious cases.
    * **Response-truncation parse errors.** Rare when the chunking layer
      has kept batch sizes sane, but still worth a retry — the model
      often generates a cleaner response on a second pass.

    On final failure bumps ``stats.api_errors`` and returns ``[]`` so
    the caller can continue with the surviving batches.
    """
    user_prompt = build_stratum_prompt(stratum, n_pairs)
    last_err: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw_text = await teacher.ask_for_pairs(user_prompt)
            return parse_teacher_response(raw_text)
        except ValueError as e:
            # Parse error — likely truncation. Retry once, then give up.
            last_err = f"parse: {e}"
        except Exception as e:
            last_err = f"api: {e}"
        if attempt < max_attempts:
            backoff = 2 ** attempt
            log.warning(
                "[%s] attempt %d/%d failed (%s); retrying in %ds",
                label, attempt, max_attempts, last_err, backoff,
            )
            await asyncio.sleep(backoff)
    log.warning("[%s] all %d attempts failed: %s", label, max_attempts, last_err)
    stats.api_errors += 1
    return []


async def generate_for_stratum(
    stratum: Stratum,
    *,
    n_pairs: int,
    teacher: TeacherClient,
    existing_embs: List[List[float]],
    similarity_threshold: float,
    dry_run: bool,
    max_per_call: int = 15,
) -> Tuple[List[ValidatedPair], GenerationStats]:
    """
    Generate pairs for one stratum.

    When ``n_pairs > max_per_call``, the request is split into multiple
    smaller teacher calls and the resulting :class:`RawPair` lists are
    concatenated before validation. This works around the fact that both
    Claude (Anthropic) and GPT-4o (OpenAI) truncate structured-JSON
    responses around ~32k output tokens, which is well below a 60-pair
    fully-populated IntentSchema batch.
    """
    stats = GenerationStats(requested=n_pairs)
    raw_pairs: List[RawPair] = []

    per_call_sizes = _split_n_pairs(n_pairs, max_per_call)
    for i, batch_n in enumerate(per_call_sizes):
        label = f"{stratum.id}#{i + 1}/{len(per_call_sizes)}" if len(per_call_sizes) > 1 else stratum.id
        chunk = await _ask_teacher_once(stratum, batch_n, teacher, stats, label)
        raw_pairs.extend(chunk)

    stats.teacher_emitted = len(raw_pairs)
    valid, invalid = validate_pairs(raw_pairs, stratum.id)
    stats.schema_invalid = len(invalid)
    for bad, reason in invalid:
        log.debug("[%s] invalid pair %r: %s", stratum.id, bad.query[:60], reason)

    if not valid:
        return [], stats

    # Embed accepted queries (skipped in dry-run to avoid burning credits).
    if dry_run:
        for p in valid:
            p.embedding = [0.0] * 1536
    else:
        try:
            embs = await embed_queries([p.query for p in valid])
        except Exception as e:
            log.warning("[%s] embedding error: %s", stratum.id, e)
            stats.api_errors += 1
            return [], stats
        for p, e in zip(valid, embs):
            p.embedding = list(e)

    accepted, dropped = dedup_by_embedding(
        valid, existing_embs, threshold=similarity_threshold
    )
    stats.duplicates_dropped = dropped
    stats.accepted = len(accepted)

    # Make this run's accepted embeddings visible to subsequent strata.
    for p in accepted:
        if p.embedding is not None:
            existing_embs.append(p.embedding)

    return accepted, stats


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

async def _run_strata_mode(
    args: argparse.Namespace,
    teacher: Optional[TeacherClient],
    existing_embs: List[List[float]],
) -> Tuple[GenerationStats, List[ValidatedPair]]:
    """Synthetic-anchor mode: walk the strata YAML."""
    strata = load_strata(args.strata)
    if args.limit_strata:
        strata = strata[: args.limit_strata]
    log.info("loaded %d strata", len(strata))

    overall = GenerationStats()
    all_accepted: List[ValidatedPair] = []

    for stratum in strata:
        if args.max_pairs and overall.accepted >= args.max_pairs:
            log.info("reached --max-pairs cap of %d, stopping", args.max_pairs)
            break

        if args.dry_run:
            print(f"=== {stratum.id} ===")
            print(build_stratum_prompt(stratum, args.n_per_stratum))
            print()
            overall.requested += args.n_per_stratum
            continue

        accepted, stats = await generate_for_stratum(
            stratum,
            n_pairs=args.n_per_stratum,
            teacher=teacher,  # type: ignore[arg-type]
            existing_embs=existing_embs,
            similarity_threshold=args.similarity_threshold,
            dry_run=False,
            max_per_call=args.max_per_call,
        )
        overall.merge(stats)
        all_accepted.extend(accepted)
        log.info(
            "[%s] requested=%d emitted=%d invalid=%d dropped=%d accepted=%d",
            stratum.id,
            stats.requested,
            stats.teacher_emitted,
            stats.schema_invalid,
            stats.duplicates_dropped,
            stats.accepted,
        )

    return overall, all_accepted


async def _run_history_mode(
    args: argparse.Namespace,
    teacher: Optional[TeacherClient],
    existing_embs: List[List[float]],
) -> Tuple[GenerationStats, List[ValidatedPair]]:
    """Real-history mode: relabel ``ChatMessage`` user queries."""
    if args.dry_run:
        # Use placeholder queries so the dry-run path is exercised end-to-end
        # without touching the database.
        history = [
            "I want to study CO2RR on Cu(111) at -0.8 V vs RHE.",
            "Compute the d-band center of Pt(111) before and after CO adsorption.",
            "Steam methane reforming on Ni(111): identify the rate-limiting step.",
        ]
        log.info("dry-run history: %d placeholder queries", len(history))
    else:
        history = await load_user_history(
            limit=args.history_limit,
            min_chars=args.history_min_chars,
            max_chars=args.history_max_chars,
        )
        log.info("loaded %d user-history queries", len(history))

    overall = GenerationStats()
    all_accepted: List[ValidatedPair] = []

    if not history:
        log.warning("no history queries found — nothing to do")
        return overall, all_accepted

    for batch_idx, batch in enumerate(_chunk(history, args.history_batch_size)):
        if args.max_pairs and overall.accepted >= args.max_pairs:
            log.info("reached --max-pairs cap of %d, stopping", args.max_pairs)
            break

        batch_id = f"{batch_idx:04d}"
        if args.dry_run:
            print(f"=== history/{batch_id} ===")
            print(build_history_prompt(batch))
            print()
            overall.requested += len(batch)
            continue

        accepted, stats = await relabel_history_batch(
            batch,
            batch_id=batch_id,
            teacher=teacher,  # type: ignore[arg-type]
            existing_embs=existing_embs,
            similarity_threshold=args.similarity_threshold,
            dry_run=False,
        )
        overall.merge(stats)
        all_accepted.extend(accepted)
        log.info(
            "[history/%s] queries=%d emitted=%d invalid=%d dropped=%d accepted=%d",
            batch_id,
            stats.requested,
            stats.teacher_emitted,
            stats.schema_invalid,
            stats.duplicates_dropped,
            stats.accepted,
        )

    return overall, all_accepted


async def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.dry_run:
        teacher: Optional[TeacherClient] = None
    else:
        teacher = build_teacher(
            model=args.teacher_model, max_tokens=args.max_tokens
        )

    if args.dry_run or args.no_db_dedup:
        existing_embs: List[List[float]] = []
    else:
        existing_embs = await fetch_existing_query_embeddings(
            limit=args.dedup_history
        )
        log.info("loaded %d existing query embeddings for dedup", len(existing_embs))

    if args.from_history:
        overall, all_accepted = await _run_history_mode(args, teacher, existing_embs)
        # Default source override for history mode if user didn't pass --source.
        source = (
            args.source if args.source != "claude_teacher" else "claude_teacher_history"
        )
    else:
        overall, all_accepted = await _run_strata_mode(args, teacher, existing_embs)
        source = args.source

    if args.dry_run:
        print(json.dumps(overall.to_dict(), indent=2))
        return 0

    # Bulk insert.
    written = await insert_pairs(
        all_accepted,
        teacher_model=args.teacher_model,
        source=source,
        schema_version=SCHEMA_VERSION,
    )
    log.info("wrote %d intent_pair rows (source=%s)", written, source)
    print(json.dumps(
        {**overall.to_dict(), "written_to_db": written, "source": source},
        indent=2,
    ))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate (query, intent) training pairs via Claude teacher.",
    )
    p.add_argument("--strata", type=Path, default=DEFAULT_STRATA_PATH,
                   help="Path to strata YAML.")
    p.add_argument("--n-per-stratum", type=int, default=8,
                   help="How many pairs to request per stratum.")
    p.add_argument("--teacher-model", default="claude-opus-4-6",
                   help="Anthropic model id (e.g. claude-opus-4-6, claude-sonnet-4-6).")
    p.add_argument("--max-tokens", type=int, default=12000,
                   help="Per-call max_tokens for the teacher backend. Default "
                        "sized for max-per-call=15 × ~500 tokens/pair + overhead.")
    p.add_argument("--max-per-call", type=int, default=15,
                   help="Max pairs requested in a single teacher call. Larger "
                        "n_per_stratum values are split into ceil(n/max_per_call) "
                        "calls. Both Claude and GPT-4o truncate at ~32k output "
                        "tokens, which is hit around 30+ fully-populated pairs; "
                        "15 leaves headroom.")
    p.add_argument("--similarity-threshold", type=float, default=0.92,
                   help="Cosine threshold above which queries are deduped.")
    p.add_argument("--dedup-history", type=int, default=5000,
                   help="How many existing intent_pair rows to load for dedup.")
    p.add_argument("--no-db-dedup", action="store_true",
                   help="Skip the cross-run DB dedup pass.")
    p.add_argument("--limit-strata", type=int, default=0,
                   help="Run only the first N strata (debugging).")
    p.add_argument("--max-pairs", type=int, default=0,
                   help="Stop once this many pairs have been accepted.")
    p.add_argument("--source", default="claude_teacher",
                   help="Provenance tag stored on every row. In --from-history "
                        "mode this defaults to 'claude_teacher_history' unless "
                        "explicitly overridden.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print prompts and stats without calling Claude or writing.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG logging.")
    # ── from-history mode ───────────────────────────────────────────────
    p.add_argument("--from-history", action="store_true",
                   help="Relabel real user queries from chat_message instead of "
                        "generating from strata anchors.")
    p.add_argument("--history-limit", type=int, default=200,
                   help="(--from-history) max user queries to pull from chat_message.")
    p.add_argument("--history-batch-size", type=int, default=10,
                   help="(--from-history) queries per Claude call. Higher amortizes "
                        "the cached system prompt better but risks max_tokens overflow.")
    p.add_argument("--history-min-chars", type=int, default=12,
                   help="(--from-history) drop queries shorter than this.")
    p.add_argument("--history-max-chars", type=int, default=2000,
                   help="(--from-history) drop queries longer than this.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
