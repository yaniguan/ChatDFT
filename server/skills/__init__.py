# server/skills/__init__.py
"""
ChatDFT Skill Framework
========================

A skill is an atomic, self-describing capability with typed I/O that the
orchestrator can discover and invoke.  Skills wrap existing agent code
behind a uniform interface so the LLM refiner can compose them without
knowing implementation details.

Public surface
--------------
- :class:`Skill`         — abstract base class; subclass and decorate
- :func:`register_skill` — decorator factory; registers into the global registry
- :data:`registry`       — the singleton :class:`SkillRegistry`
"""

from server.skills.base import FieldSpec, Skill, SkillContext, SkillResult, SkillSpec
from server.skills.registry import register_skill, registry

__all__ = [
    "FieldSpec",
    "Skill",
    "SkillContext",
    "SkillResult",
    "SkillSpec",
    "register_skill",
    "registry",
]
