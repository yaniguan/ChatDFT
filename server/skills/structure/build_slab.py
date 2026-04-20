# server/skills/structure/build_slab.py
"""Skill: build a metal slab from element + Miller indices."""

from __future__ import annotations

from server.skills.base import Domain, FieldSpec, Skill, SkillContext, SkillResult, SkillSpec
from server.skills.registry import register_skill


@register_skill
class BuildSlab(Skill):
    spec = SkillSpec(
        name="structure.build_slab",
        description="Build a periodic metal slab from element, crystal system, and Miller indices.",
        domain=Domain.STRUCTURE,
        inputs={
            "element": FieldSpec(type="str", description="Chemical symbol, e.g. 'Pt', 'Cu'"),
            "facet": FieldSpec(
                type="str", description="Miller index as string, e.g. '111', '100'", default="111", required=False
            ),
            "nx": FieldSpec(type="int", description="Supercell x repetitions", default=2, required=False),
            "ny": FieldSpec(type="int", description="Supercell y repetitions", default=2, required=False),
            "nlayers": FieldSpec(type="int", description="Number of slab layers", default=4, required=False),
            "vacuum": FieldSpec(type="float", description="Vacuum thickness in Angstrom", default=15.0, required=False),
        },
        outputs={
            "poscar": FieldSpec(type="str", description="VASP POSCAR string"),
            "n_atoms": FieldSpec(type="int", description="Total atom count"),
            "formula": FieldSpec(type="str", description="Chemical formula"),
        },
        cost=0,
        side_effects=["writes_poscar"],
    )

    async def execute(self, ctx: SkillContext) -> SkillResult:
        from io import StringIO

        from ase.io import write as ase_write

        from server.execution.structure_agent import _build_slab, _crystal_system

        inp = ctx.inputs
        element = inp["element"]
        facet = inp.get("facet", "111")
        nx = int(inp.get("nx", 2))
        ny = int(inp.get("ny", 2))
        nlayers = int(inp.get("nlayers", 4))
        vacuum = float(inp.get("vacuum", 15.0))

        miller_h = int(facet[0])
        miller_k = int(facet[1])
        miller_l = int(facet[2]) if len(facet) >= 3 else 1
        cs = _crystal_system(element)

        atoms = _build_slab(element, cs, miller_h, miller_k, miller_l, nx, ny, nlayers, vacuum)

        buf = StringIO()
        ase_write(buf, atoms, format="vasp")
        poscar = buf.getvalue()

        if ctx.job_dir:
            (ctx.job_dir / "POSCAR").write_text(poscar)

        return SkillResult(
            ok=True,
            outputs={
                "poscar": poscar,
                "n_atoms": len(atoms),
                "formula": atoms.get_chemical_formula(),
            },
            artifacts=[str(ctx.job_dir / "POSCAR")] if ctx.job_dir else [],
        )
