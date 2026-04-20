# server/skills/registry.py
"""
Global skill registry — stores all registered skills and provides
discovery, validation, and LLM tool-schema export.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from server.skills.base import Skill

log = logging.getLogger(__name__)


class SkillRegistry:
    """
    Singleton registry of all available skills.

    Skills are added via :func:`register_skill` (class decorator) or
    ``registry.add(instance)``.  The orchestrator queries the registry
    at refine time to build the LLM tool list.
    """

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def add(self, skill: Skill) -> None:
        if skill.name in self._skills:
            log.warning("Overwriting skill %s", skill.name)
        self._skills[skill.name] = skill
        log.debug("Registered skill: %s (domain=%s, cost=%d)", skill.name, skill.domain, skill.spec.cost)

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def list_all(self) -> List[Skill]:
        return list(self._skills.values())

    def list_by_domain(self, domain: str) -> List[Skill]:
        return [s for s in self._skills.values() if s.domain == domain]

    def names(self) -> List[str]:
        return sorted(self._skills.keys())

    def to_tool_schemas(self, *, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export as OpenAI-compatible tool list for LLM prompts."""
        skills = self.list_by_domain(domain) if domain else self.list_all()
        return [s.spec.to_tool_schema() for s in skills]

    def to_prompt_block(self, *, domain: Optional[str] = None) -> str:
        """
        Human-readable skill listing for embedding in an LLM prompt.
        More compact than the full JSON schema — good for context-limited prompts.
        """
        skills = self.list_by_domain(domain) if domain else self.list_all()
        lines = ["Available skills:"]
        for s in sorted(skills, key=lambda x: x.name):
            inputs_str = ", ".join(
                f"{k}: {v.type}" + (" (opt)" if not v.required else "") for k, v in s.spec.inputs.items()
            )
            outputs_str = ", ".join(f"{k}: {v.type}" for k, v in s.spec.outputs.items())
            lines.append(f"  {s.name}({inputs_str}) -> {outputs_str}  [cost={s.spec.cost}] — {s.spec.description}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills


# Singleton instance
registry = SkillRegistry()


# ---------------------------------------------------------------------------
# Decorator factory
# ---------------------------------------------------------------------------


def register_skill(cls: Type[Skill]) -> Type[Skill]:
    """
    Class decorator that instantiates a Skill subclass and registers it.

    Usage::

        @register_skill
        class BuildSlab(Skill):
            spec = SkillSpec(name="structure.build_slab", ...)

            async def execute(self, ctx):
                ...
    """
    instance = cls()
    registry.add(instance)
    return cls
