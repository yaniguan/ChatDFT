"""Tests for the skill framework — registry, specs, and concrete skills."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.skills.base import FieldSpec, Skill, SkillResult, SkillSpec
from server.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_add_and_get() -> None:
    reg = SkillRegistry()

    class DummySkill(Skill):
        spec = SkillSpec(
            name="test.dummy",
            description="A dummy skill.",
            domain="test",
            inputs={"x": FieldSpec(type="int")},
            outputs={"y": FieldSpec(type="int")},
        )

        async def execute(self, ctx):
            return SkillResult(ok=True, outputs={"y": ctx.inputs["x"] * 2})

    dummy = DummySkill()
    reg.add(dummy)

    assert "test.dummy" in reg
    assert reg.get("test.dummy") is dummy
    assert len(reg) == 1
    assert reg.names() == ["test.dummy"]


def test_registry_to_tool_schemas() -> None:
    reg = SkillRegistry()

    class S(Skill):
        spec = SkillSpec(
            name="foo.bar",
            description="Does foo to bar.",
            domain="foo",
            inputs={
                "a": FieldSpec(type="str", description="input a"),
                "b": FieldSpec(type="int", default=5, required=False),
            },
            outputs={"c": FieldSpec(type="float")},
        )

        async def execute(self, ctx):
            return SkillResult()

    reg.add(S())
    schemas = reg.to_tool_schemas()
    assert len(schemas) == 1
    fn = schemas[0]["function"]
    assert fn["name"] == "foo.bar"
    assert "a" in fn["parameters"]["properties"]
    assert fn["parameters"]["required"] == ["a"]


def test_registry_to_prompt_block() -> None:
    reg = SkillRegistry()

    class S1(Skill):
        spec = SkillSpec(
            name="a.one",
            description="First.",
            domain="a",
            inputs={"x": FieldSpec(type="str")},
            outputs={"y": FieldSpec(type="str")},
        )

        async def execute(self, ctx):
            return SkillResult()

    class S2(Skill):
        spec = SkillSpec(
            name="b.two",
            description="Second.",
            domain="b",
            inputs={},
            outputs={"z": FieldSpec(type="int")},
        )

        async def execute(self, ctx):
            return SkillResult()

    reg.add(S1())
    reg.add(S2())
    block = reg.to_prompt_block()
    assert "a.one" in block
    assert "b.two" in block
    assert "First." in block


def test_list_by_domain() -> None:
    reg = SkillRegistry()

    class SA(Skill):
        spec = SkillSpec(name="d1.a", description="", domain="d1", inputs={}, outputs={})

        async def execute(self, ctx):
            return SkillResult()

    class SB(Skill):
        spec = SkillSpec(name="d2.b", description="", domain="d2", inputs={}, outputs={})

        async def execute(self, ctx):
            return SkillResult()

    reg.add(SA())
    reg.add(SB())
    assert len(reg.list_by_domain("d1")) == 1
    assert reg.list_by_domain("d1")[0].name == "d1.a"


# ---------------------------------------------------------------------------
# Concrete skill import tests (verify registration + spec validity)
# ---------------------------------------------------------------------------


def test_global_registry_has_builtin_skills() -> None:
    import server.skills.analysis  # noqa: F401
    import server.skills.parameters  # noqa: F401
    import server.skills.reasoning  # noqa: F401
    import server.skills.structure  # noqa: F401
    from server.skills import registry

    expected = [
        "structure.build_slab",
        "structure.place_adsorbate",
        "parameters.vasp_relaxation",
        "analysis.adsorption_energy",
        "reasoning.interpret_result",
    ]
    for name in expected:
        assert name in registry, f"Skill {name} not registered"
        skill = registry.get(name)
        assert skill is not None
        assert skill.spec.name == name
        assert skill.spec.description


def test_skill_spec_to_tool_schema_roundtrip() -> None:
    import server.skills.structure  # noqa: F401
    from server.skills import registry

    skill = registry.get("structure.build_slab")
    schema = skill.spec.to_tool_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "structure.build_slab"
    params = schema["function"]["parameters"]
    assert "element" in params["properties"]
    assert "element" in params["required"]
    assert "nx" not in params["required"]  # has default → not required


def test_validate_inputs() -> None:
    import server.skills.structure  # noqa: F401
    from server.skills import registry

    skill = registry.get("structure.build_slab")
    errors = skill.validate_inputs({"element": "Cu"})
    assert errors == []  # only element is required (facet has default)

    errors = skill.validate_inputs({})
    assert any("element" in e for e in errors)
