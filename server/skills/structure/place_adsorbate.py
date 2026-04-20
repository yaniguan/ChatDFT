# server/skills/structure/place_adsorbate.py
"""Skill: place an adsorbate on a slab at a specified site."""

from __future__ import annotations

from server.skills.base import Domain, FieldSpec, Skill, SkillContext, SkillResult, SkillSpec
from server.skills.registry import register_skill


@register_skill
class PlaceAdsorbate(Skill):
    spec = SkillSpec(
        name="structure.place_adsorbate",
        description="Place an adsorbate molecule on a slab at a specified adsorption site.",
        domain=Domain.STRUCTURE,
        inputs={
            "slab_poscar": FieldSpec(type="str", description="POSCAR string of the slab"),
            "adsorbate": FieldSpec(type="str", description="Adsorbate formula, e.g. 'CO', 'OH', 'H'"),
            "site": FieldSpec(
                type="str",
                description="Site type",
                enum=["top", "bridge", "hollow", "fcc", "hcp", "auto"],
                default="auto",
                required=False,
            ),
        },
        outputs={
            "poscar": FieldSpec(type="str", description="POSCAR string with adsorbate placed"),
            "n_atoms": FieldSpec(type="int", description="Total atom count"),
            "formula": FieldSpec(type="str", description="Chemical formula"),
            "site_used": FieldSpec(type="str", description="Actual site type used"),
        },
        cost=0,
        side_effects=["writes_poscar"],
        requires=["structure.build_slab"],
    )

    async def execute(self, ctx: SkillContext) -> SkillResult:
        from io import StringIO

        from ase.io import read as ase_read
        from ase.io import write as ase_write

        from server.execution.structure_agent import (
            find_adsorption_sites_ase,
            place_adsorbate,
        )

        inp = ctx.inputs
        poscar_str = inp["slab_poscar"]
        adsorbate = inp["adsorbate"]
        site_pref = inp.get("site", "auto")

        slab = ase_read(StringIO(poscar_str), format="vasp")
        sites = find_adsorption_sites_ase(slab)
        if not sites:
            return SkillResult(ok=False, error="No adsorption sites found on slab.")

        # Pick site by preference
        if site_pref == "auto":
            chosen = sites[0]
        else:
            matched = [s for s in sites if s.get("type", "").startswith(site_pref)]
            chosen = matched[0] if matched else sites[0]

        combined = place_adsorbate(slab, chosen["position"], adsorbate)

        buf = StringIO()
        ase_write(buf, combined, format="vasp")
        poscar_out = buf.getvalue()

        if ctx.job_dir:
            (ctx.job_dir / "POSCAR").write_text(poscar_out)

        return SkillResult(
            ok=True,
            outputs={
                "poscar": poscar_out,
                "n_atoms": len(combined),
                "formula": combined.get_chemical_formula(),
                "site_used": chosen.get("type", "unknown"),
            },
            artifacts=[str(ctx.job_dir / "POSCAR")] if ctx.job_dir else [],
        )
