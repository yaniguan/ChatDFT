# server/api/one_click.py
# -*- coding: utf-8 -*-
"""
One-Click DFT Workflow API
============================

The zero-friction endpoint: one natural-language sentence in,
ready-to-submit VASP calculations out.

Before (5 API calls + manual work):
    1. POST /chat/intent       → parse query
    2. POST /chat/hypothesis   → generate mechanism
    3. POST /chat/plan         → plan workflow
    4. POST /agent/structure/* → build slab, place adsorbate
    5. POST /agent/run         → generate INCAR/KPOINTS/script

After (1 API call):
    POST /api/one_click
    {"query": "CO adsorption energy on Cu(111) with DFT-D3"}
    →
    {
      "structure": {"poscar": "...", "formula": "CCuO...", "n_atoms": 38},
      "vasp_inputs": {
        "INCAR": "ENCUT = 400\nISMEAR = 1\nIVDW = 11\n...",
        "KPOINTS": "Automatic\n0\nGamma\n4 4 1\n0 0 0",
      },
      "workflow": [
        {"step": "relax_adsorbate", "incar": {...}},
        {"step": "static_energy", "incar": {...}},
      ],
      "validation": {"n_issues": 0, "all_clear": true},
      "hpc_script": "#!/bin/bash\n#SBATCH ...",
      "post_processing": "E_ads = E(slab+ads) - E(slab) - E(gas)",
    }
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["one-click"])


class OneClickRequest(BaseModel):
    """Single natural-language input for complete DFT workflow generation."""
    query: str = Field(..., description=(
        "Natural language description of what you want to compute. "
        "Examples: 'CO adsorption on Pt(111)', 'DOS of Cu(111)', "
        "'HER on MoS2 basal plane', 'Bader charge of O on Fe(110)'"
    ))
    accuracy: str = Field("normal", description="'fast', 'normal', or 'accurate'")
    cluster: str = Field("hoffman2", description="HPC cluster name from config")
    n_cores: int = Field(24, description="Number of CPU cores for HPC script")
    walltime: str = Field("24:00:00", description="Walltime for HPC job")
    generate_script: bool = Field(True, description="Include HPC submission script")


class OneClickResponse(BaseModel):
    ok: bool
    query: str
    # Parsed intent
    element: str = ""
    facet: str = ""
    adsorbate: str = ""
    calc_type: str = ""
    # Structure
    poscar: str = ""
    formula: str = ""
    n_atoms: int = 0
    # VASP inputs
    incar_string: str = ""
    incar_dict: Dict[str, Any] = {}
    kpoints: str = ""
    # Multi-step workflow
    workflow_steps: List[Dict[str, Any]] = []
    # Validation
    validation: Dict[str, Any] = {}
    # HPC
    hpc_script: str = ""
    # Guidance
    post_processing: str = ""
    notes: List[str] = []
    error: str = ""


@router.post("/one_click", response_model=OneClickResponse)
async def one_click(req: OneClickRequest):
    """
    Zero-friction DFT workflow: natural language → ready-to-submit VASP job.

    This endpoint does everything:
    1. Parse query → extract surface, adsorbate, calculation type
    2. Build structure → POSCAR with adsorbate placed at optimal site
    3. Generate INCAR → domain-aware parameters (spin, vdW, DFT+U, etc.)
    4. Resolve workflow → multi-step dependencies (DOS needs SCF, etc.)
    5. Validate → catch ENCUT/MAGMOM/ISPIN errors before submission
    6. Auto-fix → apply corrections automatically
    7. Generate HPC script → ready to submit

    The scientist types ONE sentence and gets everything.
    """
    notes: List[str] = []

    try:
        # ── 1. Parse query ───────────────────────────────────────────
        parsed = _parse_query(req.query)
        element = parsed["element"]
        facet = parsed["facet"]
        adsorbate = parsed.get("adsorbate", "")
        calc_type = parsed["calc_type"]
        notes.extend(parsed.get("notes", []))

        # ── 2. Build structure ───────────────────────────────────────
        structure_result = _build_structure(element, facet, adsorbate, parsed)
        poscar = structure_result["poscar"]
        formula = structure_result["formula"]
        n_atoms = structure_result["n_atoms"]
        elements_list = structure_result["elements"]
        notes.extend(structure_result.get("notes", []))

        # ── 3. Generate INCAR ────────────────────────────────────────
        incar_result = _generate_incar(
            calc_type, element, adsorbate, elements_list,
            req.query, req.accuracy,
        )
        incar_dict = incar_result["incar"]
        kpoints = incar_result["kpoints"]
        notes.extend(incar_result.get("notes", []))

        # ── 4. Resolve multi-step workflow ───────────────────────────
        workflow_steps = _resolve_steps(calc_type, incar_dict, elements_list)

        # ── 5. Validate + auto-fix ───────────────────────────────────
        from science.vasp.auto_remediation import validate_consistency, auto_fix
        issues = validate_consistency(
            incar=incar_dict,
            elements=elements_list,
            n_atoms=n_atoms,
            calc_type=calc_type,
            kpoints=kpoints,
        )
        fixed_incar, changes = auto_fix(incar_dict, issues)
        if changes:
            incar_dict = fixed_incar
            notes.extend([f"Auto-fixed: {c}" for c in changes])

        remaining_issues = [i for i in issues if not i.auto_fixable]
        validation = {
            "n_issues": len(issues),
            "n_auto_fixed": len(changes),
            "all_clear": len(remaining_issues) == 0,
            "remaining": [
                {"severity": i.severity, "message": i.message}
                for i in remaining_issues
            ],
        }

        # ── 6. Format INCAR string ───────────────────────────────────
        incar_string = _format_incar(incar_dict)

        # ── 7. Generate HPC script ──────────────────────────────────
        hpc_script = ""
        if req.generate_script:
            hpc_script = _generate_hpc_script(
                job_name=f"chatdft-{element}{facet}-{adsorbate or calc_type}",
                n_cores=req.n_cores,
                walltime=req.walltime,
            )

        # ── 8. Post-processing guidance ──────────────────────────────
        post_processing = _post_processing_guide(calc_type, adsorbate)

        return OneClickResponse(
            ok=True,
            query=req.query,
            element=element,
            facet=facet,
            adsorbate=adsorbate,
            calc_type=calc_type,
            poscar=poscar,
            formula=formula,
            n_atoms=n_atoms,
            incar_string=incar_string,
            incar_dict=incar_dict,
            kpoints=kpoints,
            workflow_steps=workflow_steps,
            validation=validation,
            hpc_script=hpc_script,
            post_processing=post_processing,
            notes=notes,
        )

    except Exception as e:
        return OneClickResponse(ok=False, query=req.query, error=str(e))


# ═══════════════════════════════════════════════════════════════════════
# Internal pipeline functions
# ═══════════════════════════════════════════════════════════════════════

def _parse_query(query: str) -> Dict[str, Any]:
    """Parse natural language query into structured intent."""
    q = query.lower()
    result: Dict[str, Any] = {
        "element": "Cu", "facet": "111", "calc_type": "static", "notes": [],
    }

    # Surface
    m = re.search(r'([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\((\d{3,4})\)', query)
    if m:
        result["element"] = m.group(1)
        result["facet"] = m.group(2)

    # Adsorbate (ordered by length to match longest first)
    for ads in ["CH3OH", "COOH", "CO2", "CH2O", "CHO", "OOH", "OH", "NH3", "CO", "N2", "H2O", "H2", "H", "O"]:
        if ads.lower() in q:
            result["adsorbate"] = ads
            break

    # Calc type
    calc_keywords = {
        "dos": ["dos", "density of states", "pdos", "d-band"],
        "band": ["band structure", "band gap"],
        "bader": ["bader", "charge analysis", "charge transfer"],
        "cohp": ["cohp", "coop", "lobster", "bond analysis"],
        "work_function": ["work function"],
        "elf": ["elf", "electron localization"],
        "cdd": ["charge density difference", "cdd"],
        "neb": ["neb", "transition state", "barrier", "nudged elastic"],
    }
    for calc, keywords in calc_keywords.items():
        if any(kw in q for kw in keywords):
            result["calc_type"] = calc
            break

    # Modifiers
    if any(kw in q for kw in ["relax", "optimize", "geometry opt"]):
        result["relax"] = True
    if any(kw in q for kw in ["d3", "dft-d3", "vdw", "van der waals", "dispersion"]):
        result["vdw"] = True
    if any(kw in q for kw in ["spin", "magnetic", "ispin"]):
        result["spin"] = True
    if any(kw in q for kw in ["solvation", "solvent", "vaspsol", "implicit"]):
        result["solvation"] = True
    if any(kw in q for kw in ["+u", "hubbard", "dft+u"]):
        result["dftu"] = True

    return result


def _build_structure(
    element: str, facet: str, adsorbate: str, parsed: Dict,
) -> Dict[str, Any]:
    """Build the atomic structure."""
    notes = []
    try:
        from server.execution.structure_agent import (
            _build_slab, _guess_crystal_system,
            find_adsorption_sites_ase, place_adsorbate,
            _atoms_to_poscar,
        )

        crystal = _guess_crystal_system(element, {})
        h, k, l = int(facet[0]), int(facet[1]), int(facet[2]) if len(facet) >= 3 else 1

        slab = _build_slab(element, crystal, h, k, l, nx=4, ny=4, nlayers=4, vacuum=15.0)

        # Fix bottom 2 layers
        from ase.constraints import FixAtoms
        positions = slab.get_positions()
        z_coords = positions[:, 2]
        z_sorted = sorted(set(round(z, 1) for z in z_coords))
        if len(z_sorted) >= 3:
            z_cutoff = z_sorted[1] + 0.5  # fix bottom 2 layers
            fix_indices = [i for i, z in enumerate(z_coords) if z < z_cutoff]
            slab.set_constraint(FixAtoms(indices=fix_indices))
            notes.append(f"Fixed bottom 2 layers ({len(fix_indices)} atoms)")

        if adsorbate:
            sites = find_adsorption_sites_ase(slab, height=2.0)
            if sites:
                slab = place_adsorbate(slab, sites[0]["position"], adsorbate)
                notes.append(f"Placed {adsorbate} at {sites[0].get('type', 'optimal')} site")

        poscar = _atoms_to_poscar(slab)
        symbols = slab.get_chemical_symbols()
        unique_elements = list(dict.fromkeys(symbols))

        return {
            "poscar": poscar,
            "formula": slab.get_chemical_formula(),
            "n_atoms": len(slab),
            "elements": unique_elements,
            "notes": notes,
        }
    except Exception as e:
        # Minimal fallback
        return {
            "poscar": f"# Structure generation failed: {e}\n# Build manually with ASE",
            "formula": f"{element}(slab)",
            "n_atoms": 36,
            "elements": [element],
            "notes": [f"Structure generation failed: {e}"],
        }


def _generate_incar(
    calc_type: str, element: str, adsorbate: str,
    elements: List[str], query: str, accuracy: str,
) -> Dict[str, Any]:
    """Generate INCAR parameters with all domain-specific adjustments."""
    from server.execution.vasp_incar import get_incar, suggested_kpoints
    from science.vasp.auto_remediation import MAGNETIC_ELEMENTS, POTCAR_ENMAX
    import math

    incar = get_incar(calc_type)
    notes = []

    # Accuracy
    if accuracy == "fast":
        incar["PREC"] = "Normal"
        incar["EDIFF"] = 1e-4
        kpoints = "4 4 1"
    elif accuracy == "accurate":
        incar["PREC"] = "Accurate"
        incar["EDIFF"] = 1e-6
        incar["ADDGRID"] = True
        kpoints = "8 8 1"
    else:
        kpoints = suggested_kpoints(calc_type)

    # ENCUT from elements
    max_enmax = max(POTCAR_ENMAX.get(e, 300) for e in elements)
    recommended = int(math.ceil(max_enmax * 1.3 / 10) * 10)
    if incar.get("ENCUT", 400) < recommended:
        incar["ENCUT"] = recommended
        notes.append(f"ENCUT={recommended} eV (1.3x ENMAX={max_enmax})")

    # Magnetic
    if any(e in MAGNETIC_ELEMENTS for e in elements):
        incar["ISPIN"] = 2
        notes.append("Spin-polarized (magnetic element detected)")

    # vdW
    q = query.lower()
    if any(kw in q for kw in ["d3", "dft-d3", "vdw", "dispersion"]):
        incar["IVDW"] = 11
        notes.append("DFT-D3 dispersion (IVDW=11)")

    # Solvation
    if any(kw in q for kw in ["solvation", "solvent", "vaspsol"]):
        incar["LSOL"] = True
        incar["EB_K"] = 78.4
        notes.append("VASPsol implicit solvation")

    # Relaxation
    if any(kw in q for kw in ["relax", "optim"]) or (adsorbate and calc_type == "static"):
        incar["IBRION"] = 2
        incar["NSW"] = 200
        incar["EDIFFG"] = -0.02
        incar["ISIF"] = 2
        notes.append("Geometry optimization enabled")

    return {"incar": incar, "kpoints": kpoints, "notes": notes}


def _resolve_steps(
    calc_type: str, incar: Dict, elements: List[str],
) -> List[Dict[str, Any]]:
    """Resolve multi-step workflow."""
    from science.vasp.auto_remediation import resolve_workflow
    from server.execution.vasp_incar import get_incar

    steps = resolve_workflow(calc_type)
    result = []
    for step in steps:
        base = get_incar(step.calc_type)
        base.update(step.incar_overrides)
        result.append({
            "step": step.name,
            "calc_type": step.calc_type,
            "incar": base,
            "depends_on": step.depends_on,
            "notes": step.notes,
        })
    return result


def _format_incar(params: Dict[str, Any]) -> str:
    """Format INCAR dict as VASP INCAR file string."""
    lines = ["# Generated by ChatDFT /api/one_click", ""]
    for key in sorted(params.keys()):
        val = params[key]
        if key.startswith("_"):
            continue
        if isinstance(val, bool):
            val_str = ".TRUE." if val else ".FALSE."
        elif isinstance(val, float):
            val_str = f"{val:.1e}" if abs(val) < 1e-3 else f"{val}"
        else:
            val_str = str(val)
        lines.append(f"  {key:<12} = {val_str}")
    return "\n".join(lines) + "\n"


def _generate_hpc_script(job_name: str, n_cores: int, walltime: str) -> str:
    """Generate a generic SLURM submission script."""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks={n_cores}
#SBATCH --time={walltime}
#SBATCH --output=stdout_%j.log
#SBATCH --error=stderr_%j.log

module load vasp/6.3.2
module load intel/2022.1

echo "Job started at $(date)"
echo "Running on $(hostname)"

mpirun -np $SLURM_NTASKS vasp_std > vasp.log 2>&1

echo "Job finished at $(date)"
"""


def _post_processing_guide(calc_type: str, adsorbate: str) -> str:
    """Return post-processing instructions for the calculation type."""
    guides = {
        "static": (
            "Extract total energy from OUTCAR: grep 'free  energy' OUTCAR | tail -1\n"
            + (f"E_ads = E(slab+{adsorbate}) - E(slab) - E({adsorbate}_gas)"
               if adsorbate else "")
        ),
        "dos": (
            "1. Extract DOSCAR with vaspkit (option 111) or pyprocar\n"
            "2. d-band center: epsilon_d = integral(E * n_d(E) dE) / integral(n_d(E) dE)\n"
            "3. Plot with: from pymatgen.io.vasp import Vasprun; v=Vasprun('vasprun.xml'); v.get_dos()"
        ),
        "band": (
            "1. Extract band structure from vasprun.xml\n"
            "2. Plot: from pymatgen.io.vasp import Vasprun; bs = Vasprun('vasprun.xml').get_band_structure()"
        ),
        "bader": (
            "1. Run: chgsum.pl AECCAR0 AECCAR2  (VTST scripts)\n"
            "2. Run: bader CHGCAR -ref CHGCAR_sum\n"
            "3. Read ACF.dat for per-atom charges: q_bader = Z_valence - q_ACF"
        ),
        "cohp": (
            "1. Run LOBSTER with lobsterin specifying atom pairs\n"
            "2. Plot COHPCAR.lobster: negative = bonding, positive = antibonding\n"
            "3. Integrated COHP (ICOHP) gives total bond strength"
        ),
        "work_function": (
            "1. phi = E_vacuum - E_Fermi\n"
            "2. E_Fermi: grep 'E-fermi' OUTCAR\n"
            "3. E_vacuum: planar average of LOCPOT (plateau far from slab)"
        ),
        "elf": (
            "1. Visualize ELFCAR in VESTA\n"
            "2. Isosurface at 0.75 shows lone pairs and covalent bonds\n"
            "3. ELF=1.0 fully localized, 0.5 electron gas, 0.0 depleted"
        ),
    }
    return guides.get(calc_type, "Extract results from OUTCAR / vasprun.xml")
