# server/orchestrator/actions.py
"""
Action whitelist + validators for the orchestrator's refine step.

The LLM is only ever allowed to propose actions whose ``(kind, subkind)``
pair appears in :data:`SUBKINDS`.  Everything else is rejected before
it can grow the plan — this is the single chokepoint that prevents the
model from running away.

Action taxonomy
---------------
verify    — reproduce or tighten an existing result (low risk, low cost)
extend    — push hypothesis frontier (new intermediate, site, facet, coads)
challenge — actively try to falsify (alternative path, counterexample)
scan      — 1-D parametric sweep (coverage, facet, composition, field)
"""
from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


class ActionKind(str, enum.Enum):
    VERIFY = "verify"
    EXTEND = "extend"
    CHALLENGE = "challenge"
    SCAN = "scan"


# ---------------------------------------------------------------------------
# Whitelist: kind → {subkind: required_param_keys}
# ---------------------------------------------------------------------------
# Required params per subkind. Optional params are allowed but typed-checked
# downstream by ``action_to_tasks``. Cost defaults are advisory.

SUBKINDS: Dict[str, Dict[str, Dict[str, Any]]] = {
    ActionKind.VERIFY.value: {
        "reconverge": {
            "required": ["target_task_id"],
            "optional": ["delta_encut", "edens_factor", "kmesh_factor"],
            "default_cost": 1,
        },
        "replicate": {
            "required": ["target_task_id"],
            "optional": ["seed_jitter"],
            "default_cost": 1,
        },
        "functional": {
            "required": ["target_task_id", "new_xc"],
            "optional": ["plus_u", "vdw"],
            "default_cost": 1,
        },
    },
    ActionKind.EXTEND.value: {
        "intermediate": {
            "required": ["species"],
            "optional": ["site", "surface"],
            "default_cost": 2,
        },
        "site": {
            "required": ["species", "surface", "site"],
            "optional": [],
            "default_cost": 1,
        },
        "facet": {
            "required": ["species", "facet"],
            "optional": ["surface"],
            "default_cost": 2,
        },
        "coadsorbate": {
            "required": ["species_a", "species_b", "surface"],
            "optional": ["site_a", "site_b"],
            "default_cost": 2,
        },
    },
    ActionKind.CHALLENGE.value: {
        "alternative_step": {
            "required": ["original_step", "alternative_step"],
            "optional": ["surface"],
            "default_cost": 2,
        },
        "alternative_ts": {
            "required": ["step", "ts_variant"],
            "optional": ["surface"],
            "default_cost": 2,
        },
        "counterexample": {
            "required": ["species", "surface"],
            "optional": ["expected_value", "expected_trend"],
            "default_cost": 1,
        },
    },
    ActionKind.SCAN.value: {
        "coverage": {
            "required": ["species", "surface", "values"],  # values: list[float]
            "optional": [],
            "default_cost": 4,
        },
        "facet": {
            "required": ["species", "facets"],  # facets: list[str]
            "optional": ["surface_template"],
            "default_cost": 3,
        },
        "composition": {
            "required": ["base_surface", "compositions"],  # list[dict]
            "optional": ["species"],
            "default_cost": 3,
        },
        "field": {
            "required": ["species", "surface", "fields"],  # list[float] (V/Å or U vs RHE)
            "optional": ["units"],
            "default_cost": 4,
        },
    },
}

# Per-round caps to keep cost bounded
MAX_ACTIONS_PER_ROUND = 5
MAX_TOTAL_COST_PER_ROUND = 10
MAX_RATIONALE_CHARS = 500


# ---------------------------------------------------------------------------
# Dataclass + validator
# ---------------------------------------------------------------------------

@dataclass
class ProposedAction:
    """A single, validated action the LLM has proposed."""

    kind: str           # ActionKind value
    subkind: str        # one of SUBKINDS[kind]
    target: str         # short descriptor: "CO* on Pt(111)" / "task#7"
    params: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    priority: float = 0.5
    cost_estimate: int = 1

    # Filled in by orchestrator after action_to_tasks() runs
    spawned_task_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def signature(self) -> Tuple[str, str, str]:
        """Identity for de-duplication within a round."""
        return (self.kind, self.subkind, self.target)


@dataclass
class ValidationError:
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


def validate_action(raw: Mapping[str, Any]) -> Tuple[Optional[ProposedAction], List[ValidationError]]:
    """
    Validate a single action dict against the whitelist.

    Returns ``(ProposedAction, [])`` on success, or ``(None, errors)``
    if validation fails.  Never raises.
    """
    errors: List[ValidationError] = []

    kind = str(raw.get("kind", "")).strip().lower()
    if kind not in SUBKINDS:
        errors.append(ValidationError("kind", f"must be one of {sorted(SUBKINDS)}"))
        return None, errors

    subkind = str(raw.get("subkind", "")).strip().lower()
    if subkind not in SUBKINDS[kind]:
        errors.append(ValidationError(
            "subkind", f"for kind={kind} must be one of {sorted(SUBKINDS[kind])}",
        ))
        return None, errors

    target = str(raw.get("target", "")).strip()
    if not target:
        errors.append(ValidationError("target", "required, non-empty"))

    params = raw.get("params") or {}
    if not isinstance(params, dict):
        errors.append(ValidationError("params", "must be an object"))
        params = {}

    spec = SUBKINDS[kind][subkind]
    for required_key in spec["required"]:
        if required_key not in params or params[required_key] in (None, "", []):
            errors.append(ValidationError(
                f"params.{required_key}",
                f"required for {kind}.{subkind}",
            ))

    rationale = str(raw.get("rationale", "")).strip()
    if not rationale:
        errors.append(ValidationError("rationale", "required, non-empty"))
    elif len(rationale) > MAX_RATIONALE_CHARS:
        rationale = rationale[:MAX_RATIONALE_CHARS]  # truncate, don't reject

    try:
        priority = float(raw.get("priority", 0.5))
    except (TypeError, ValueError):
        errors.append(ValidationError("priority", "must be a float in [0, 1]"))
        priority = 0.5
    priority = max(0.0, min(1.0, priority))

    try:
        cost_estimate = int(raw.get("cost_estimate", spec["default_cost"]))
    except (TypeError, ValueError):
        errors.append(ValidationError("cost_estimate", "must be a positive int"))
        cost_estimate = spec["default_cost"]
    if cost_estimate < 1:
        errors.append(ValidationError("cost_estimate", "must be ≥ 1"))

    if errors:
        return None, errors

    return ProposedAction(
        kind=kind,
        subkind=subkind,
        target=target,
        params=dict(params),
        rationale=rationale,
        priority=priority,
        cost_estimate=cost_estimate,
    ), []


def validate_action_batch(
    raw_list: List[Mapping[str, Any]],
) -> Tuple[List[ProposedAction], List[ValidationError]]:
    """
    Validate a batch of actions and enforce per-round caps:
      - max MAX_ACTIONS_PER_ROUND actions
      - total cost ≤ MAX_TOTAL_COST_PER_ROUND
      - no duplicate (kind, subkind, target) signatures
    Returns ``(accepted, errors)``.
    """
    accepted: List[ProposedAction] = []
    errors: List[ValidationError] = []
    seen_signatures: set = set()
    total_cost = 0

    for i, raw in enumerate(raw_list):
        action, errs = validate_action(raw)
        for e in errs:
            errors.append(ValidationError(f"actions[{i}].{e.field}", e.reason))
        if action is None:
            continue

        if action.signature in seen_signatures:
            errors.append(ValidationError(
                f"actions[{i}]",
                f"duplicate signature {action.signature} in same round",
            ))
            continue

        if len(accepted) >= MAX_ACTIONS_PER_ROUND:
            errors.append(ValidationError(
                f"actions[{i}]",
                f"exceeds MAX_ACTIONS_PER_ROUND={MAX_ACTIONS_PER_ROUND}",
            ))
            continue

        if total_cost + action.cost_estimate > MAX_TOTAL_COST_PER_ROUND:
            errors.append(ValidationError(
                f"actions[{i}]",
                f"cumulative cost {total_cost + action.cost_estimate} "
                f"> MAX_TOTAL_COST_PER_ROUND={MAX_TOTAL_COST_PER_ROUND}",
            ))
            continue

        accepted.append(action)
        seen_signatures.add(action.signature)
        total_cost += action.cost_estimate

    # Sort by priority descending so high-priority actions execute first
    accepted.sort(key=lambda a: -a.priority)
    return accepted, errors


# ---------------------------------------------------------------------------
# JSON schema for prompting the LLM
# ---------------------------------------------------------------------------

def llm_action_schema_prompt() -> str:
    """
    Human-readable schema spec to embed in the refine prompt.
    Keep it small — the LLM only needs to know the shape and the whitelist.
    """
    lines = ["Allowed actions (kind.subkind → required params):"]
    for kind, subs in SUBKINDS.items():
        for subkind, spec in subs.items():
            req = ", ".join(spec["required"]) or "—"
            lines.append(f"  {kind}.{subkind}  required: [{req}]")
    lines.append("")
    lines.append(
        "Output strict JSON: "
        '{"actions": [{"kind": "...", "subkind": "...", "target": "...", '
        '"params": {...}, "rationale": "...", "priority": 0.0-1.0, '
        '"cost_estimate": int}], "stop_reason": null | string}'
    )
    lines.append(
        f"Hard limits per round: ≤ {MAX_ACTIONS_PER_ROUND} actions, "
        f"≤ {MAX_TOTAL_COST_PER_ROUND} total cost. "
        "If you have nothing useful to add, return an empty actions list "
        "and set stop_reason to a one-line explanation."
    )
    return "\n".join(lines)
