# server/skills/parameters/vasp_relaxation.py
"""Skill: generate VASP INCAR + KPOINTS for a geometry relaxation."""

from __future__ import annotations

from server.skills.base import Domain, FieldSpec, Skill, SkillContext, SkillResult, SkillSpec
from server.skills.registry import register_skill


@register_skill
class VaspRelaxation(Skill):
    spec = SkillSpec(
        name="parameters.vasp_relaxation",
        description="Generate VASP INCAR and KPOINTS files for a geometry relaxation.",
        domain=Domain.PARAMETERS,
        inputs={
            "system_kind": FieldSpec(
                type="str",
                description="System type",
                enum=["bulk", "slab", "adsorption", "molecule"],
                default="slab",
                required=False,
            ),
            "encut": FieldSpec(type="int", description="Planewave cutoff (eV)", default=400, required=False),
            "kppra": FieldSpec(type="int", description="k-points per reciprocal atom", default=1600, required=False),
            "ediffg": FieldSpec(
                type="float", description="Force convergence criterion (eV/A)", default=-0.03, required=False
            ),
            "overrides": FieldSpec(
                type="dict", description="Arbitrary INCAR key overrides", default={}, required=False
            ),
        },
        outputs={
            "incar": FieldSpec(type="str", description="INCAR file content"),
            "kpoints": FieldSpec(type="str", description="KPOINTS file content"),
        },
        cost=0,
        side_effects=["writes_incar", "writes_kpoints"],
        requires=["structure.build_slab"],
    )

    async def execute(self, ctx: SkillContext) -> SkillResult:
        from pathlib import Path

        from server.execution.parameters_agent import ParametersAgent

        inp = ctx.inputs
        system_kind = inp.get("system_kind", "slab")
        overrides = dict(inp.get("overrides") or {})
        if "encut" in inp and inp["encut"]:
            overrides.setdefault("ENCUT", int(inp["encut"]))
        if "ediffg" in inp and inp["ediffg"]:
            overrides.setdefault("EDIFFG", float(inp["ediffg"]))

        job_dir = ctx.job_dir or Path("/tmp/chatdft_skill_params")
        job_dir.mkdir(parents=True, exist_ok=True)

        task = {
            "id": None,
            "params": {
                "payload": {
                    "engine": "vasp",
                    "calc_type": "relax",
                    "system_kind": system_kind,
                    "kmesh": {"mode": "kppra", "value": int(inp.get("kppra", 1600))},
                    "overrides": overrides,
                },
            },
            "meta": {},
        }

        agent = ParametersAgent()
        agent.generate(task, job_dir)

        incar_path = job_dir / "INCAR"
        kpoints_path = job_dir / "KPOINTS"

        incar = incar_path.read_text() if incar_path.exists() else ""
        kpoints = kpoints_path.read_text() if kpoints_path.exists() else ""

        if not incar:
            return SkillResult(ok=False, error="INCAR not generated.")

        return SkillResult(
            ok=True,
            outputs={"incar": incar, "kpoints": kpoints},
            artifacts=[str(incar_path), str(kpoints_path)],
        )
