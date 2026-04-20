# server/skills/base.py
"""
Core abstractions for the skill framework.

A skill declares its :class:`SkillSpec` (typed I/O, cost, domain) and
implements ``execute(ctx) -> SkillResult``.  The spec doubles as the
LLM function-calling schema — the registry can export all registered
skills as a JSON tool list that any model can reason over.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class Domain(str, Enum):
    STRUCTURE = "structure"
    PARAMETERS = "parameters"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    REASONING = "reasoning"


@dataclass
class FieldSpec:
    """Type descriptor for one input or output field."""

    type: str  # "str" | "int" | "float" | "bool" | "path" | "dict" | "list"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None

    def to_json_schema(self) -> Dict[str, Any]:
        TYPE_MAP = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "path": "string",
            "dict": "object",
            "list": "array",
        }
        schema: Dict[str, Any] = {"type": TYPE_MAP.get(self.type, "string")}
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class SkillSpec:
    """Immutable metadata describing a skill's contract."""

    name: str  # "structure.build_slab"
    description: str  # 1-sentence description for LLM
    domain: str  # Domain value
    inputs: Dict[str, FieldSpec]  # param name → spec
    outputs: Dict[str, FieldSpec]  # output field → spec
    cost: int = 0  # estimated HPC jobs (0 = local-only)
    side_effects: List[str] = field(default_factory=list)  # e.g. ["writes_poscar"]
    requires: List[str] = field(default_factory=list)  # prerequisite skill outputs

    def to_tool_schema(self) -> Dict[str, Any]:
        """Export as an OpenAI-style function tool schema."""
        properties: Dict[str, Any] = {}
        required_params: List[str] = []
        for name, fspec in self.inputs.items():
            properties[name] = fspec.to_json_schema()
            if fspec.required:
                required_params.append(name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                },
            },
        }


@dataclass
class SkillContext:
    """
    Runtime context passed to skill.execute().

    Carries shared state the skill might need (session id, working dir,
    config, prior results) without coupling to a specific DB or framework.
    """

    session_id: Optional[int] = None
    job_dir: Optional[Path] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    prior_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """Outcome of executing a skill."""

    ok: bool = True
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)  # file paths created

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Skill(ABC):
    """
    Base class for all skills.

    Subclass, define ``spec``, and implement ``execute``.
    Then decorate with ``@register_skill`` to wire into the registry.
    """

    spec: SkillSpec

    @abstractmethod
    async def execute(self, ctx: SkillContext) -> SkillResult: ...

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def domain(self) -> str:
        return self.spec.domain

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Return list of missing/invalid required fields."""
        errors = []
        for field_name, fspec in self.spec.inputs.items():
            if fspec.required and field_name not in inputs:
                errors.append(f"missing required input: {field_name}")
        return errors

    def __repr__(self) -> str:
        return f"<Skill {self.spec.name} domain={self.spec.domain} cost={self.spec.cost}>"
