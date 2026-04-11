# server/chat/intent_schema.py
# -*- coding: utf-8 -*-
"""
Pydantic schema for the intent agent JSON output.

This module is the single source of truth for what the intent agent is
allowed to emit. Keep ``_intent_system_prompt`` in
``server/chat/intent_agent.py`` and this schema in lockstep — the prompt
*describes* the contract, this schema *enforces* it.

Design notes
------------
* The schema is intentionally lenient on optional sub-fields (lists default
  to empty, scalars to ``None``) so that minor model omissions do not blow
  up the pipeline. The strictness lives in two places:

  1. ``stage`` and ``area`` are constrained to canonical ``Literal`` values
     after a normalization pass that forgives common LLM spelling variants
     (``"electro"`` → ``"electrochemistry"``, ``"thermal"`` →
     ``"thermal_catalysis"``, etc.).
  2. ``task`` and ``summary`` must be non-empty strings.

* ``validate_intent`` is the public entry point used by ``intent_agent``.
  It returns ``(model, None)`` on success and ``(None, error_summary)`` on
  failure, where ``error_summary`` is a short string fit to be appended to
  the next LLM message in the retry loop.

* ``model_config`` allows extra keys, so any forward-compatible field the
  prompt grows in the future flows through unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------
#
# Bump SCHEMA_VERSION whenever the IntentSchema fields, enums, or coercion
# rules change in a way that would invalidate previously generated training
# pairs. ``IntentPair.schema_version`` records the version that produced
# each row, so the data flywheel and SFT pipeline can filter out stale
# samples without re-labeling everything.
SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Canonical enums
# ---------------------------------------------------------------------------

Stage = Literal[
    "catalysis",
    "screening",
    "benchmarking",
    "analysis",
    "structure_building",
]

Area = Literal[
    "electrochemistry",
    "thermal_catalysis",
    "photocatalysis",
    "heterogeneous_catalysis",
    "homogeneous_catalysis",
]

STAGE_VALUES: Tuple[str, ...] = (
    "catalysis",
    "screening",
    "benchmarking",
    "analysis",
    "structure_building",
)

AREA_VALUES: Tuple[str, ...] = (
    "electrochemistry",
    "thermal_catalysis",
    "photocatalysis",
    "heterogeneous_catalysis",
    "homogeneous_catalysis",
)

_STAGE_ALIASES: Dict[str, str] = {
    "catalytic": "catalysis",
    "catal": "catalysis",
    "catalysis_task": "catalysis",
    "screen": "screening",
    "screening_task": "screening",
    "benchmark": "benchmarking",
    "bench": "benchmarking",
    "analyze": "analysis",
    "analyse": "analysis",
    "analysis_task": "analysis",
    "structure": "structure_building",
    "build": "structure_building",
    "build_structure": "structure_building",
    "structure_build": "structure_building",
}

_AREA_ALIASES: Dict[str, str] = {
    # electrochemistry
    "electro": "electrochemistry",
    "electrocatalysis": "electrochemistry",
    "electrocatalytic": "electrochemistry",
    "electrochem": "electrochemistry",
    "ec": "electrochemistry",
    # thermal
    "thermal": "thermal_catalysis",
    "thermo": "thermal_catalysis",
    "thermocatalysis": "thermal_catalysis",
    "thermocatalytic": "thermal_catalysis",
    "thermal_cat": "thermal_catalysis",
    "thermalcatalysis": "thermal_catalysis",
    # photo
    "photo": "photocatalysis",
    "photocatalytic": "photocatalysis",
    "photothermal": "photocatalysis",
    "photo_thermal": "photocatalysis",
    # heterogeneous
    "heterogeneous": "heterogeneous_catalysis",
    "het": "heterogeneous_catalysis",
    "surface": "heterogeneous_catalysis",
    # homogeneous
    "homogeneous": "homogeneous_catalysis",
    "hom": "homogeneous_catalysis",
    "molecular": "homogeneous_catalysis",
    "organometallic": "homogeneous_catalysis",
}


def _canonicalize(value: Any, aliases: Dict[str, str], allowed: Tuple[str, ...]) -> Optional[str]:
    """
    Map a free-form LLM string to a canonical enum value.

    Returns ``None`` if no mapping is possible — the caller decides whether
    that constitutes a validation error.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    v = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not v:
        return None
    if v in allowed:
        return v
    if v in aliases:
        return aliases[v]
    # Substring fuzzing — handles things like "electro_catalysis (aqueous)".
    for key, target in aliases.items():
        if key in v:
            return target
    for canonical in allowed:
        if canonical in v:
            return canonical
    return None


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Conditions(BaseModel):
    model_config = ConfigDict(extra="allow")

    pH: Optional[float] = None
    potential_V_vs_RHE: Optional[float] = None
    solvent: Optional[str] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    electrolyte: Optional[str] = None

    @field_validator("pH", "potential_V_vs_RHE", "temperature", "pressure", mode="before")
    @classmethod
    def _coerce_number(cls, v: Any) -> Optional[float]:
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v.strip().rstrip("VKkPa%").strip())
            except ValueError:
                return None
        return None


class CatalystSystem(BaseModel):
    model_config = ConfigDict(extra="allow")

    catalyst: Optional[str] = None
    material: Optional[str] = None
    facet: Optional[str] = None
    molecule: List[str] = Field(default_factory=list)

    @field_validator("molecule", mode="before")
    @classmethod
    def _coerce_molecule(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        if isinstance(v, list):
            return [str(x) for x in v if x not in (None, "")]
        return [str(v)]


class ReactionNetwork(BaseModel):
    model_config = ConfigDict(extra="allow")

    intermediates: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    ts: List[Any] = Field(default_factory=list)
    coads: List[Any] = Field(default_factory=list)
    coads_pairs: List[str] = Field(default_factory=list)

    @field_validator("intermediates", "steps", "coads_pairs", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x not in (None, "")]
        return [str(v)] if v else []

    @field_validator("ts", "coads", mode="before")
    @classmethod
    def _coerce_any_list(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]


class Metric(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    unit: Optional[str] = None
    note: Optional[str] = None


class Deliverables(BaseModel):
    model_config = ConfigDict(extra="allow")

    target_products: List[str] = Field(default_factory=list)
    figures: List[str] = Field(default_factory=list)

    @field_validator("target_products", "figures", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x not in (None, "")]
        return [str(v)] if v else []


# ---------------------------------------------------------------------------
# Top-level schema
# ---------------------------------------------------------------------------

class IntentSchema(BaseModel):
    """Canonical intent output from the intent_agent LLM call."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Required, enum-constrained
    stage: Stage
    area: Area
    task: str = Field(min_length=1)
    summary: str = Field(min_length=1)

    # Optional structured fields
    system: CatalystSystem = Field(default_factory=CatalystSystem)
    substrate: Optional[str] = None
    facet: Optional[str] = None
    reactant: Optional[str] = None
    product: Optional[str] = None
    adsorbates: List[str] = Field(default_factory=list)
    conditions: Conditions = Field(default_factory=Conditions)
    metrics: List[Metric] = Field(default_factory=list)
    reaction_network: ReactionNetwork = Field(default_factory=ReactionNetwork)
    deliverables: Deliverables = Field(default_factory=Deliverables)
    hypothesis: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)

    # ---------- normalizers ----------

    @field_validator("stage", mode="before")
    @classmethod
    def _norm_stage(cls, v: Any) -> Any:
        canon = _canonicalize(v, _STAGE_ALIASES, STAGE_VALUES)
        if canon is None:
            raise ValueError(
                f"stage must be one of {list(STAGE_VALUES)}; got {v!r}"
            )
        return canon

    @field_validator("area", mode="before")
    @classmethod
    def _norm_area(cls, v: Any) -> Any:
        canon = _canonicalize(v, _AREA_ALIASES, AREA_VALUES)
        if canon is None:
            raise ValueError(
                f"area must be one of {list(AREA_VALUES)}; got {v!r}"
            )
        return canon

    @field_validator("adsorbates", "tags", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x not in (None, "")]
        return [str(v)] if v else []

    @field_validator("metrics", mode="before")
    @classmethod
    def _coerce_metrics(cls, v: Any) -> List[Dict[str, Any]]:
        if v is None:
            return []
        if not isinstance(v, list):
            v = [v]
        out: List[Dict[str, Any]] = []
        for item in v:
            if isinstance(item, dict):
                if "name" in item and item["name"]:
                    out.append(item)
            elif isinstance(item, str) and item:
                out.append({"name": item})
        return out

    @field_validator("constraints", mode="before")
    @classmethod
    def _coerce_constraints(cls, v: Any) -> Dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            return {"notes": v}
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_validation_error(exc: ValidationError, max_errors: int = 6) -> str:
    """
    Convert a ``ValidationError`` into a short, model-digestible error
    summary suitable for the retry prompt.
    """
    lines: List[str] = []
    for err in exc.errors()[:max_errors]:
        loc = ".".join(str(x) for x in err.get("loc") or [])
        msg = err.get("msg") or "invalid"
        lines.append(f"- {loc or '<root>'}: {msg}")
    extra = len(exc.errors()) - max_errors
    if extra > 0:
        lines.append(f"- (+{extra} more validation errors)")
    return "\n".join(lines)


def validate_intent(
    raw: Optional[Dict[str, Any]],
) -> Tuple[Optional[IntentSchema], Optional[str]]:
    """
    Validate a raw LLM JSON dict against ``IntentSchema``.

    Returns
    -------
    (model, None)              on success
    (None, error_summary)      on failure — the summary is a short multi-line
                               string ready to be sent back to the LLM in a
                               retry message.
    """
    if not isinstance(raw, dict) or not raw:
        return None, "- <root>: response was not a JSON object"
    try:
        return IntentSchema.model_validate(raw), None
    except ValidationError as exc:
        return None, format_validation_error(exc)
