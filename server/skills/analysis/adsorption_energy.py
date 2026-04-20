# server/skills/analysis/adsorption_energy.py
"""Skill: parse OUTCAR/vasprun.xml and compute adsorption energy."""

from __future__ import annotations

from server.skills.base import Domain, FieldSpec, Skill, SkillContext, SkillResult, SkillSpec
from server.skills.registry import register_skill


@register_skill
class AdsorptionEnergy(Skill):
    spec = SkillSpec(
        name="analysis.adsorption_energy",
        description="Parse a completed VASP calculation and extract/compute adsorption energy.",
        domain=Domain.ANALYSIS,
        inputs={
            "e_slab_ads": FieldSpec(
                type="float", description="Total energy of slab+adsorbate system (eV)", required=False
            ),
            "e_slab": FieldSpec(type="float", description="Total energy of clean slab (eV)", required=False),
            "e_gas": FieldSpec(type="float", description="Total energy of gas-phase adsorbate (eV)", required=False),
        },
        outputs={
            "e_ads": FieldSpec(type="float", description="Adsorption energy (eV)"),
            "converged": FieldSpec(type="bool", description="Whether the calculation converged"),
            "total_energy": FieldSpec(type="float", description="Total energy from OUTCAR (eV)"),
        },
        cost=0,
        side_effects=[],
        requires=["execution.submit_and_wait"],
    )

    async def execute(self, ctx: SkillContext) -> SkillResult:
        from pathlib import Path

        job_dir = ctx.job_dir
        inp = ctx.inputs

        total_energy = None
        converged = False

        if job_dir:
            total_energy, converged = self._parse_outcar(Path(job_dir))

        # If explicit energies provided, compute E_ads directly
        e_slab_ads = inp.get("e_slab_ads") or total_energy
        e_slab = inp.get("e_slab")
        e_gas = inp.get("e_gas")

        e_ads = None
        if e_slab_ads is not None and e_slab is not None and e_gas is not None:
            e_ads = float(e_slab_ads) - float(e_slab) - float(e_gas)

        outputs = {
            "total_energy": total_energy,
            "converged": converged,
            "e_ads": e_ads,
        }
        if e_ads is None and total_energy is None:
            return SkillResult(ok=False, error="No OUTCAR found and no explicit energies provided.", outputs=outputs)

        return SkillResult(ok=True, outputs=outputs)

    @staticmethod
    def _parse_outcar(job_dir):
        """Extract final energy and convergence from OUTCAR."""
        outcar = job_dir / "OUTCAR"
        if not outcar.exists():
            return None, False

        energy = None
        converged = False
        try:
            text = outcar.read_text(errors="replace")
            for line in reversed(text.splitlines()):
                if "free  energy   TOTEN" in line:
                    parts = line.split()
                    energy = float(parts[-2])
                    break
            converged = "reached required accuracy" in text
        except Exception:
            pass
        return energy, converged
