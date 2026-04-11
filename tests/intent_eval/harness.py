# tests/intent_eval/harness.py
# -*- coding: utf-8 -*-
"""
Intent agent evaluation harness.

The harness has three layers, each independently usable:

1. ``load_eval_set(path)`` — read the JSONL eval set into a list of
   ``EvalCase`` records. Pure I/O, no LLM.
2. ``score_case(predicted, gold)`` and ``aggregate(results)`` — pure
   scoring functions that compare a predicted intent dict against a
   sparse gold dict. Used by the pytest regression tests; no API calls.
3. CLI entry point (``python -m tests.intent_eval.harness``) that runs a
   pluggable predictor against the full eval set and reports per-field
   accuracy + an area confusion matrix. The default predictor calls the
   live ``/chat/intent`` endpoint and is *opt-in* — CI must not run it.

Scoring rules
-------------
* Only the keys present in ``gold`` are scored. Predictions for keys
  absent from ``gold`` are ignored. This makes the eval set easy to
  extend incrementally — add only the fields you care about labelling.
* String comparison is case-insensitive and whitespace-normalized.
* Nested dicts (``conditions``) are scored recursively, key by key.
* Numeric conditions accept ±1e-3 tolerance.
* The aggregate report includes:
    - ``n_cases``               — number of cases evaluated
    - ``field_accuracy``        — total correct / total scored fields
    - ``per_field_accuracy``    — break down by field name
    - ``area_confusion``        — gold-area → predicted-area count map
    - ``critical_em``           — exact-match on (stage, area, substrate)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    id: str
    query: str
    gold: Dict[str, Any]
    notes: Optional[str] = None


@dataclass
class CaseResult:
    id: str
    query: str
    gold: Dict[str, Any]
    predicted: Dict[str, Any]
    field_scores: Dict[str, bool]      # flat dotted-path → bool
    critical_em: bool
    error: Optional[str] = None


@dataclass
class AggregateReport:
    n_cases: int
    field_accuracy: float
    per_field_accuracy: Dict[str, float]
    area_confusion: Dict[str, Dict[str, int]]
    critical_em_rate: float
    failures: List[CaseResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "field_accuracy": round(self.field_accuracy, 4),
            "critical_em_rate": round(self.critical_em_rate, 4),
            "per_field_accuracy": {
                k: round(v, 4) for k, v in self.per_field_accuracy.items()
            },
            "area_confusion": self.area_confusion,
            "n_failures": len(self.failures),
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

DEFAULT_EVAL_SET = Path(__file__).resolve().parent / "eval_set.jsonl"


def load_eval_set(path: Optional[Path] = None) -> List[EvalCase]:
    target = Path(path) if path else DEFAULT_EVAL_SET
    cases: List[EvalCase] = []
    with target.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{target}:{lineno}: invalid JSONL — {e}") from e
            cases.append(
                EvalCase(
                    id=str(obj["id"]),
                    query=str(obj["query"]),
                    gold=dict(obj.get("gold") or {}),
                    notes=obj.get("notes"),
                )
            )
    return cases


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

_NUMERIC_TOLERANCE = 1e-3

# Chemical-species alias table used by ``_chemical_equal``.
#
# Every entry maps a canonical token to a set of equivalent forms (lower-case,
# whitespace stripped). Both the gold and predicted strings are normalized
# through this table before comparison, so "C2H4" and "ethene" both become
# the same canonical form.
#
# Only the species that actually show up in the intent_eval set plus the most
# common DFT reactants/products are listed. Add more as needed — new entries
# should NEVER be destructive (mapping two distinct species to one canonical).
_CHEMICAL_ALIASES: Dict[str, set[str]] = {
    "h2":     {"h2", "dihydrogen", "hydrogen", "molecular hydrogen"},
    "o2":     {"o2", "dioxygen", "oxygen", "molecular oxygen"},
    "n2":     {"n2", "dinitrogen", "nitrogen"},
    "h2o":    {"h2o", "water", "h₂o"},
    "co":     {"co", "carbon monoxide"},
    "co2":    {"co2", "carbon dioxide", "co₂"},
    "ch4":    {"ch4", "methane"},
    "nh3":    {"nh3", "ammonia"},
    "ch3oh":  {"ch3oh", "methanol", "ch₃oh", "ch3 oh"},
    "c2h4":   {"c2h4", "ethene", "ethylene"},
    "c2h6":   {"c2h6", "ethane"},
    "c3h6":   {"c3h6", "propene", "propylene"},
    "c3h8":   {"c3h8", "propane"},
    "c4h8":   {"c4h8", "butene", "1-butene", "2-butene"},
    "c4h10":  {"c4h10", "butane", "n-butane"},
    "hcoo-":  {"hcoo-", "hcoo", "formate", "hcoo⁻", "formic acid"},
    "hcooh":  {"hcooh", "formic acid", "hcoo2h"},
    "ch3ch2oh": {"ch3ch2oh", "c2h5oh", "ethanol"},
    "no3-":   {"no3-", "no3", "nitrate", "no₃⁻"},
    "h+":     {"h+", "proton", "h⁺"},
}


def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip().lower().replace("  ", " ")


def _canonical_species(s: Any) -> str:
    """
    Map a species string to its canonical form via ``_CHEMICAL_ALIASES``.

    If the input matches no known alias, returns the normalized-lower-case
    original — so unknown species still participate in exact-match.
    """
    raw = _norm_str(s)
    if not raw:
        return ""
    for canon, aliases in _CHEMICAL_ALIASES.items():
        if raw in aliases:
            return canon
    return raw


def _split_species(s: Any) -> List[str]:
    """
    Split a species string on common separators (``+``, ``,``, ``/``) and
    canonicalize each token. ``"H2 + O2"`` → ``["h2", "o2"]``.
    """
    raw = _norm_str(s)
    if not raw:
        return []
    tokens: List[str] = []
    for sep in ("+", ",", "/", " and "):
        if sep in raw:
            tokens = [t.strip() for t in raw.split(sep) if t.strip()]
            break
    if not tokens:
        tokens = [raw]
    return [_canonical_species(t) for t in tokens]


def _chemical_equal(gold: Any, pred: Any) -> bool:
    """
    Chemistry-aware equality for ``reactant`` / ``product`` fields.

    Accepts three kinds of match:

    1. **Alias match** — ``"C2H4"`` == ``"ethene"`` via ``_CHEMICAL_ALIASES``.
    2. **Superset match** — if the gold species is present in the predicted
       species list (``gold={"CO"}`` and ``pred={"CO", "H2"}``), accept.
       This handles genuinely-multi-product reactions like steam methane
       reforming (CH₄+H₂O → CO+3H₂) where picking either primary product
       should be valid.
    3. **Empty-sides** — both null/empty → match (same as ``_equal``).
    """
    if gold is None and pred is None:
        return True
    gold_set = set(_split_species(gold))
    pred_set = set(_split_species(pred))
    if not gold_set and not pred_set:
        return True
    if not gold_set or not pred_set:
        return False
    # Exact canonical match on single-species values.
    if gold_set == pred_set:
        return True
    # Superset: every gold token must appear in the prediction.
    return gold_set.issubset(pred_set)


# Field paths that should go through ``_chemical_equal`` instead of ``_equal``.
# Kept as a set so callers can extend it at import time if needed.
_CHEMICAL_FIELDS = {"reactant", "product"}


def _equal(a: Any, b: Any) -> bool:
    """Compare two scalars with type-aware tolerance."""
    if a is None and b is None:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= _NUMERIC_TOLERANCE
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return [_norm_str(x) for x in a] == [_norm_str(x) for x in b]
    return _norm_str(a) == _norm_str(b)


def _score_dict(
    gold: Dict[str, Any],
    predicted: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, bool]:
    """
    Recursively score every key present in ``gold`` against ``predicted``.
    Returns a flat ``dotted.path → bool`` map.
    """
    out: Dict[str, bool] = {}
    for key, gold_val in gold.items():
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        pred_val = predicted.get(key) if isinstance(predicted, dict) else None
        if isinstance(gold_val, dict):
            sub_pred = pred_val if isinstance(pred_val, dict) else {}
            out.update(_score_dict(gold_val, sub_pred, prefix=path))
        else:
            # Chemistry-aware matching for species fields.
            if key in _CHEMICAL_FIELDS:
                out[path] = _chemical_equal(gold_val, pred_val)
            else:
                out[path] = _equal(gold_val, pred_val)
    return out


def score_case(predicted: Dict[str, Any], gold: Dict[str, Any]) -> Tuple[Dict[str, bool], bool]:
    """
    Score a single (predicted, gold) pair.

    Returns
    -------
    field_scores : dict of dotted-path → bool
    critical_em  : True iff (stage, area, substrate) all match.
                   ``substrate`` is ignored if absent from gold.
    """
    field_scores = _score_dict(gold, predicted or {})

    critical_keys = ["stage", "area"]
    if "substrate" in gold:
        critical_keys.append("substrate")
    critical_em = all(field_scores.get(k, False) for k in critical_keys)
    return field_scores, critical_em


def aggregate(results: Iterable[CaseResult]) -> AggregateReport:
    results_list = list(results)
    n = len(results_list)
    if n == 0:
        return AggregateReport(0, 0.0, {}, {}, 0.0)

    total_correct = 0
    total_scored = 0
    per_field_correct: Dict[str, int] = {}
    per_field_total: Dict[str, int] = {}
    area_confusion: Dict[str, Dict[str, int]] = {}
    critical_hits = 0
    failures: List[CaseResult] = []

    for r in results_list:
        for field_path, ok in r.field_scores.items():
            total_scored += 1
            total_correct += int(ok)
            per_field_total[field_path] = per_field_total.get(field_path, 0) + 1
            per_field_correct[field_path] = per_field_correct.get(field_path, 0) + int(ok)

        if r.critical_em:
            critical_hits += 1
        else:
            failures.append(r)

        gold_area = (r.gold or {}).get("area")
        pred_area = (r.predicted or {}).get("area")
        if gold_area:
            row = area_confusion.setdefault(gold_area, {})
            key = pred_area or "<missing>"
            row[key] = row.get(key, 0) + 1

    per_field_accuracy = {
        k: per_field_correct[k] / per_field_total[k]
        for k in per_field_total
    }
    field_accuracy = total_correct / total_scored if total_scored else 0.0
    critical_em_rate = critical_hits / n

    return AggregateReport(
        n_cases=n,
        field_accuracy=field_accuracy,
        per_field_accuracy=per_field_accuracy,
        area_confusion=area_confusion,
        critical_em_rate=critical_em_rate,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Predictors (pluggable)
# ---------------------------------------------------------------------------

Predictor = Callable[[str], Dict[str, Any]]


def echo_predictor(query: str) -> Dict[str, Any]:
    """Trivial predictor for tests — returns an empty intent."""
    return {}


def live_predictor(
    api_base: Optional[str] = None,
    *,
    session_name: str = "intent-eval",
) -> Predictor:
    """
    Build a predictor that calls the running ``/chat/intent`` endpoint.

    Requires the FastAPI server to be running and whatever LLM provider is
    routed to ``intent_agent`` in ``server/llm.yaml`` to have live credentials.
    Creates one throwaway chat session up front (not per-query) so the
    baseline run reuses a single session id — that keeps the
    ``chat_message`` audit trail readable afterwards.
    """
    import requests  # local import — keep harness import-time clean

    base = api_base or os.environ.get("CHATDFT_API_BASE", "http://localhost:8000")

    # Pre-create a single session for the whole eval run. If the endpoint
    # rejects the request, fall back to session_id=1 and hope a session
    # exists in the DB. We don't want to create one session per query.
    sid = 1
    try:
        r = requests.post(
            f"{base}/chat/session/create",
            json={"name": session_name, "description": "intent regression eval"},
            timeout=10,
        )
        if r.ok:
            sid = (r.json() or {}).get("session_id") or 1
    except Exception:
        pass

    def _predict(query: str) -> Dict[str, Any]:
        r = requests.post(
            f"{base}/chat/intent",
            json={"session_id": sid, "text": query},
            timeout=180,
        )
        if not r.ok:
            return {"_error": f"HTTP {r.status_code}: {r.text[:200]}"}
        return (r.json() or {}).get("intent") or {}

    return _predict


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(
    cases: List[EvalCase],
    predictor: Predictor,
) -> List[CaseResult]:
    results: List[CaseResult] = []
    for case in cases:
        try:
            pred = predictor(case.query)
            err = pred.get("_error") if isinstance(pred, dict) else None
        except Exception as e:  # pragma: no cover — predictor errors
            pred = {}
            err = f"{type(e).__name__}: {e}"
        field_scores, critical_em = score_case(pred or {}, case.gold)
        results.append(
            CaseResult(
                id=case.id,
                query=case.query,
                gold=case.gold,
                predicted=pred or {},
                field_scores=field_scores,
                critical_em=critical_em,
                error=err,
            )
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_argv(argv: List[str]) -> Dict[str, Any]:
    args: Dict[str, Any] = {
        "predictor": "echo",
        "eval_set": None,
        "min_field_accuracy": 0.0,
        "min_critical_em": 0.0,
        "show_failures": False,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--predictor" and i + 1 < len(argv):
            args["predictor"] = argv[i + 1]
            i += 2
        elif a == "--eval-set" and i + 1 < len(argv):
            args["eval_set"] = Path(argv[i + 1])
            i += 2
        elif a == "--min-field-accuracy" and i + 1 < len(argv):
            args["min_field_accuracy"] = float(argv[i + 1])
            i += 2
        elif a == "--min-critical-em" and i + 1 < len(argv):
            args["min_critical_em"] = float(argv[i + 1])
            i += 2
        elif a == "--show-failures":
            args["show_failures"] = True
            i += 1
        elif a in ("-h", "--help"):
            print(
                "Usage: python -m tests.intent_eval.harness "
                "[--predictor echo|live] [--eval-set PATH] "
                "[--min-field-accuracy 0.85] [--min-critical-em 0.90] "
                "[--show-failures]"
            )
            sys.exit(0)
        else:
            print(f"unknown arg: {a}", file=sys.stderr)
            sys.exit(2)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = _parse_argv(argv)

    cases = load_eval_set(args["eval_set"])
    if args["predictor"] == "live":
        pred = live_predictor()
    elif args["predictor"] == "echo":
        pred = echo_predictor
    else:
        print(f"unknown predictor: {args['predictor']}", file=sys.stderr)
        return 2

    results = run(cases, pred)
    report = aggregate(results)

    print(json.dumps(report.to_dict(), indent=2))

    if args["show_failures"]:
        print("\n--- failures ---")
        for f in report.failures:
            print(f"[{f.id}] gold={f.gold} predicted_area={f.predicted.get('area')!r}"
                  f" predicted_stage={f.predicted.get('stage')!r}")
            if f.error:
                print(f"  error: {f.error}")

    if report.field_accuracy < args["min_field_accuracy"]:
        print(
            f"FAIL: field_accuracy {report.field_accuracy:.3f} "
            f"< threshold {args['min_field_accuracy']}",
            file=sys.stderr,
        )
        return 1
    if report.critical_em_rate < args["min_critical_em"]:
        print(
            f"FAIL: critical_em_rate {report.critical_em_rate:.3f} "
            f"< threshold {args['min_critical_em']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
