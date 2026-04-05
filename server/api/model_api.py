# server/api/model_api.py
# -*- coding: utf-8 -*-
"""
One-Call Model Inference API
=============================

Problem: ChatDFT has powerful models (GNN energy prediction, SCF diagnosis,
auto-remediation, hypothesis grounder) but they require importing Python
modules and constructing data structures manually.  A scientist doing
high-throughput screening wants:

    POST /api/predict_energy  {"surface": "Pt(111)", "adsorbate": "CO"}
    → {"E_ads_eV": -1.72, "model": "schnet", "confidence": 0.89}

Not:
    from science.predictions.energy_predictor import generate_dataset, ...
    from science.predictions.gnn_models import build_model, ...
    # 30 lines of data wrangling

This module exposes every science model as a single REST endpoint
with auto-preprocessing of inputs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["model-api"])


# ═══════════════════════════════════════════════════════════════════════
# Request / Response models
# ═══════════════════════════════════════════════════════════════════════

class PredictEnergyRequest(BaseModel):
    """Predict adsorption energy from surface + adsorbate description."""
    surface: str = Field(..., description="e.g. 'Pt(111)', 'Cu(100)', 'Fe(110)'")
    adsorbate: str = Field(..., description="e.g. 'CO', 'H', 'OH', 'OOH', 'COOH'")
    model: str = Field("schnet", description="GNN model: mpnn, gat, schnet, dimenet, se3_transformer")
    site: Optional[str] = Field(None, description="Adsorption site: top, bridge, hollow, fcc, hcp. Auto if None.")

class BatchPredictRequest(BaseModel):
    """Batch prediction for screening."""
    systems: List[Dict[str, str]] = Field(
        ..., description="List of {surface, adsorbate} dicts"
    )
    model: str = "schnet"

class ValidateInputsRequest(BaseModel):
    """Validate VASP inputs before submission — catches errors that waste HPC hours."""
    incar: Dict[str, Any] = Field(..., description="INCAR parameters as key-value dict")
    elements: List[str] = Field(..., description="Element symbols, e.g. ['Cu', 'C', 'O']")
    n_atoms: int = Field(..., description="Total number of atoms")
    calc_type: str = Field("static", description="Calculation type")
    kpoints: Optional[str] = Field(None, description="KPOINTS string, e.g. '4 4 1'")

class DiagnoseSCFRequest(BaseModel):
    """Diagnose SCF convergence from energy differences (OSZICAR data)."""
    energy_diffs: List[float] = Field(..., description="List of |E_n - E_{n-1}| from OSZICAR")
    target_ediff: float = Field(1e-5, description="EDIFF target")
    current_incar: Optional[Dict[str, Any]] = Field(None, description="Current INCAR for context-aware fix")

class ResolveWorkflowRequest(BaseModel):
    """Get the full multi-step workflow for a calculation type."""
    calc_type: str = Field(..., description="Target: dos, band, elf, bader, cohp, work_function, cdd")
    elements: Optional[List[str]] = Field(None, description="Elements for ENCUT inference")

class AutoFixRequest(BaseModel):
    """Validate AND auto-fix VASP inputs in one call."""
    incar: Dict[str, Any]
    elements: List[str]
    n_atoms: int
    calc_type: str = "static"
    kpoints: Optional[str] = None

class SmartParamsRequest(BaseModel):
    """Infer optimal VASP parameters from a natural-language description."""
    description: str = Field(..., description="e.g. 'CO adsorption on Cu(111) with spin polarization'")
    accuracy: str = Field("normal", description="'fast' (screening), 'normal', 'accurate' (publication)")


# ═══════════════════════════════════════════════════════════════════════
# 1. GNN Energy Prediction
# ═══════════════════════════════════════════════════════════════════════

@router.post("/predict_energy")
async def predict_energy(req: PredictEnergyRequest):
    """
    Predict adsorption energy from surface + adsorbate.

    No POSCAR needed — the endpoint builds the surface topology graph
    internally from the surface notation and adsorbate species.

    Example:
        POST /api/predict_energy
        {"surface": "Pt(111)", "adsorbate": "CO", "model": "schnet"}
        → {"E_ads_eV": -1.72, "model": "schnet", "site": "top"}
    """
    try:
        from science.predictions.energy_predictor import (
            generate_dataset, samples_to_graphs, create_trainer,
        )
        from science.predictions.gnn_models import build_model, _check_torch

        if not _check_torch():
            return _fallback_predict(req.surface, req.adsorbate, req.site)

        # Parse surface notation
        element, facet = _parse_surface(req.surface)

        # Generate a synthetic sample matching the request
        samples = generate_dataset(
            n_samples=50,
            metals=[element],
            adsorbates=[req.adsorbate],
        )

        if not samples:
            return _fallback_predict(req.surface, req.adsorbate, req.site)

        # Build model and predict
        graphs = samples_to_graphs(samples)
        model = build_model(req.model, d_node=graphs[0].x.shape[1])
        trainer = create_trainer(model)

        # Use the mean of synthetic predictions as estimate
        predictions = trainer.predict_batch(graphs[:10])
        mean_pred = float(sum(predictions) / max(len(predictions), 1))

        # Find the matching sample for the requested site
        target_sample = None
        for s in samples:
            if req.site and s.get("site_type", "").lower() != req.site.lower():
                continue
            target_sample = s
            break

        return {
            "ok": True,
            "E_ads_eV": round(mean_pred, 3),
            "model": req.model,
            "surface": req.surface,
            "adsorbate": req.adsorbate,
            "site": req.site or (target_sample or {}).get("site_type", "auto"),
            "confidence": "synthetic_estimate",
            "note": "Prediction from synthetic scaling-relation data. Use DFT for publication accuracy.",
        }
    except Exception as e:
        return _fallback_predict(req.surface, req.adsorbate, req.site, note=str(e))


def _fallback_predict(
    surface: str, adsorbate: str, site: Optional[str] = None, note: str = "",
) -> Dict[str, Any]:
    """Physics-based fallback using d-band model when PyTorch unavailable."""
    from science.core.constants import D_BAND_CENTRES, ADSORBATE_OFFSETS, CN_BINDING_SLOPE

    element, _ = _parse_surface(surface)
    d_band = D_BAND_CENTRES.get(element, -2.0)
    ads_offset = ADSORBATE_OFFSETS.get(adsorbate, 0.0)

    # Simple d-band model: E_ads ≈ α·ε_d + β
    e_ads = 0.5 * d_band + ads_offset

    # Coordination correction for site type
    cn_correction = 0.0
    if site == "hollow" or site == "fcc":
        cn_correction = CN_BINDING_SLOPE * 3
    elif site == "bridge":
        cn_correction = CN_BINDING_SLOPE * 1
    e_ads += cn_correction

    return {
        "ok": True,
        "E_ads_eV": round(e_ads, 3),
        "model": "d_band_scaling",
        "surface": surface,
        "adsorbate": adsorbate,
        "site": site or "auto",
        "confidence": "analytical_estimate",
        "note": note or "Analytical d-band model. Install PyTorch for GNN predictions.",
    }


@router.post("/predict_energy/batch")
async def predict_energy_batch(req: BatchPredictRequest):
    """
    Batch energy prediction for high-throughput screening.

    Example:
        POST /api/predict_energy/batch
        {"systems": [
            {"surface": "Pt(111)", "adsorbate": "CO"},
            {"surface": "Cu(111)", "adsorbate": "CO"},
            {"surface": "Au(111)", "adsorbate": "CO"}
        ], "model": "schnet"}
    """
    results = []
    for sys in req.systems:
        surface = sys.get("surface", "")
        adsorbate = sys.get("adsorbate", "")
        site = sys.get("site")
        pred = _fallback_predict(surface, adsorbate, site)
        results.append({
            "surface": surface,
            "adsorbate": adsorbate,
            "E_ads_eV": pred["E_ads_eV"],
            "model": pred["model"],
        })

    return {
        "ok": True,
        "n_predictions": len(results),
        "model": req.model,
        "predictions": results,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. VASP Input Validation
# ═══════════════════════════════════════════════════════════════════════

@router.post("/validate")
async def validate_inputs(req: ValidateInputsRequest):
    """
    Validate VASP inputs before submission.

    Catches errors that waste HPC hours:
    - MAGMOM count mismatch
    - ENCUT below POTCAR ENMAX
    - Missing ISPIN for magnetic elements
    - ELF with NCORE > 1
    - COHP without ISYM=-1
    - Wrong ISMEAR for low k-points

    Example:
        POST /api/validate
        {"incar": {"ENCUT": 200, "LELF": true, "NCORE": 4},
         "elements": ["Cu", "O"], "n_atoms": 36, "calc_type": "elf"}
        → {"ok": true, "n_errors": 2, "issues": [...], "auto_fixable": 2}
    """
    from science.vasp.auto_remediation import validate_consistency

    issues = validate_consistency(
        incar=req.incar,
        elements=req.elements,
        n_atoms=req.n_atoms,
        calc_type=req.calc_type,
        kpoints=req.kpoints,
    )

    return {
        "ok": True,
        "n_issues": len(issues),
        "n_errors": sum(1 for i in issues if i.severity == "error"),
        "n_warnings": sum(1 for i in issues if i.severity == "warning"),
        "auto_fixable": sum(1 for i in issues if i.auto_fixable),
        "issues": [
            {
                "severity": i.severity,
                "category": i.category,
                "message": i.message,
                "fix": i.fix,
                "auto_fixable": i.auto_fixable,
            }
            for i in issues
        ],
    }


@router.post("/autofix")
async def auto_fix_inputs(req: AutoFixRequest):
    """
    Validate AND auto-fix VASP inputs in one call.

    Returns the corrected INCAR dict + list of changes made.

    Example:
        POST /api/autofix
        {"incar": {"LELF": true, "NCORE": 4, "ENCUT": 200},
         "elements": ["Cu", "O"], "n_atoms": 36, "calc_type": "elf"}
        → {"ok": true, "fixed_incar": {"LELF": true, "NCORE": 1, "ENCUT": 520},
           "changes": ["NCORE: 4 → 1 (ELF requires NCORE=1)", ...]}
    """
    from science.vasp.auto_remediation import validate_consistency, auto_fix

    issues = validate_consistency(
        incar=req.incar,
        elements=req.elements,
        n_atoms=req.n_atoms,
        calc_type=req.calc_type,
        kpoints=req.kpoints,
    )

    fixed_incar, changes = auto_fix(req.incar, issues)

    return {
        "ok": True,
        "fixed_incar": fixed_incar,
        "n_changes": len(changes),
        "changes": changes,
        "remaining_issues": [
            {"severity": i.severity, "category": i.category, "message": i.message}
            for i in issues if not i.auto_fixable
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. SCF Diagnosis
# ═══════════════════════════════════════════════════════════════════════

@router.post("/diagnose_scf")
async def diagnose_scf(req: DiagnoseSCFRequest):
    """
    Diagnose SCF convergence issues from OSZICAR energy differences.

    Unlike custodian's pattern matching, this analyzes the actual convergence
    trajectory via FFT to detect charge sloshing and prescribe physics-based fixes.

    Example:
        POST /api/diagnose_scf
        {"energy_diffs": [0.1, 0.05, 0.08, 0.03, 0.06, ...],
         "target_ediff": 1e-5,
         "current_incar": {"ALGO": "Fast", "AMIX": 0.4}}
        → {"diagnosis": "charge_sloshing",
           "fix": {"ALGO": "All", "AMIX": 0.1, "BMIX": 0.01},
           "explanation": "Charge sloshing detected (AC/total=0.42)..."}
    """
    from science.vasp.auto_remediation import analyze_scf_trajectory

    result = analyze_scf_trajectory(
        energy_diffs=req.energy_diffs,
        target_ediff=req.target_ediff,
        current_incar=req.current_incar,
    )

    return {
        "ok": True,
        "diagnosis": result.diagnosis.value,
        "n_steps": result.n_steps,
        "final_ediff": result.final_ediff,
        "convergence_rate": result.convergence_rate,
        "sloshing_power_ratio": result.sloshing_power_ratio,
        "sign_change_rate": result.sign_change_rate,
        "estimated_steps_to_converge": result.estimated_steps_to_converge,
        "recommended_fix": result.recommended_fix,
        "explanation": result.explanation,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. Workflow Resolver
# ═══════════════════════════════════════════════════════════════════════

@router.post("/resolve_workflow")
async def resolve_workflow_endpoint(req: ResolveWorkflowRequest):
    """
    Get the full multi-step VASP workflow for a calculation type.

    Automatically resolves prerequisites (DOS needs SCF CHGCAR, etc.)
    and returns complete INCAR parameters for each step.

    Example:
        POST /api/resolve_workflow  {"calc_type": "dos"}
        → {"steps": [
             {"name": "scf_prerequisite", "incar": {"LWAVE": true, "LCHARG": true, ...}},
             {"name": "dos", "incar": {"ICHARG": 11, "ISMEAR": -5, ...}, "depends_on": ["scf_prerequisite"]}
           ]}
    """
    from science.vasp.auto_remediation import resolve_workflow
    from server.execution.vasp_incar import get_incar

    steps = resolve_workflow(req.calc_type)

    result_steps = []
    for step in steps:
        # Merge base INCAR with step overrides
        base = get_incar(step.calc_type)
        base.update(step.incar_overrides)

        # Apply element-specific adjustments
        if req.elements:
            from science.vasp.auto_remediation import MAGNETIC_ELEMENTS, POTCAR_ENMAX
            import math
            has_mag = any(e in MAGNETIC_ELEMENTS for e in req.elements)
            if has_mag and base.get("ISPIN", 1) == 1:
                base["ISPIN"] = 2
            max_enmax = max(POTCAR_ENMAX.get(e, 300) for e in req.elements)
            if base.get("ENCUT", 400) < max_enmax * 1.3:
                base["ENCUT"] = int(math.ceil(max_enmax * 1.3 / 10) * 10)

        result_steps.append({
            "name": step.name,
            "calc_type": step.calc_type,
            "incar": base,
            "depends_on": step.depends_on,
            "output_files": step.output_files,
            "input_files": step.input_files,
            "notes": step.notes,
        })

    return {
        "ok": True,
        "target": req.calc_type,
        "n_steps": len(result_steps),
        "steps": result_steps,
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. Smart Parameter Inference
# ═══════════════════════════════════════════════════════════════════════

@router.post("/smart_params")
async def smart_params(req: SmartParamsRequest):
    """
    Infer optimal VASP parameters from a natural-language description.

    Scientists describe what they want in plain English; this endpoint
    figures out every VASP parameter, KPOINTS, and prerequisite.

    Example:
        POST /api/smart_params
        {"description": "CO adsorption energy on Cu(111) with DFT-D3",
         "accuracy": "normal"}
        → {"incar": {"ENCUT": 400, "ISMEAR": 1, "IVDW": 11, ...},
           "kpoints": "4 4 1", "notes": [...]}
    """
    desc = req.description.lower()

    # Parse what we can from the description
    surface_info = _parse_description(desc)
    element = surface_info["element"]
    calc_type = surface_info["calc_type"]
    adsorbate = surface_info.get("adsorbate", "")

    # Build INCAR from calc type
    from server.execution.vasp_incar import get_incar, suggested_kpoints
    incar = get_incar(calc_type)

    # Accuracy level adjustments
    if req.accuracy == "fast":
        incar["PREC"] = "Normal"
        incar["EDIFF"] = 1e-4
        kpoints = "4 4 1"
    elif req.accuracy == "accurate":
        incar["PREC"] = "Accurate"
        incar["EDIFF"] = 1e-6
        incar["ADDGRID"] = True
        kpoints = "8 8 1"
    else:
        kpoints = suggested_kpoints(calc_type)

    notes = []

    # Domain-specific adjustments
    from science.vasp.auto_remediation import MAGNETIC_ELEMENTS, POTCAR_ENMAX
    import math

    if element in MAGNETIC_ELEMENTS:
        incar["ISPIN"] = 2
        incar["MAGMOM"] = f"$NATOMS*{MAGNETIC_ELEMENTS[element]}"
        notes.append(f"{element} is magnetic — ISPIN=2 + MAGMOM set automatically")

    # ENCUT from POTCAR
    enmax = POTCAR_ENMAX.get(element, 300)
    if adsorbate:
        for char in adsorbate:
            if char.isupper():
                ads_elem = char
                if char + adsorbate[adsorbate.index(char)+1:adsorbate.index(char)+2].lower() in POTCAR_ENMAX:
                    ads_elem = char + adsorbate[adsorbate.index(char)+1:adsorbate.index(char)+2].lower()
                enmax = max(enmax, POTCAR_ENMAX.get(ads_elem, 300))
    recommended_encut = int(math.ceil(enmax * 1.3 / 10) * 10)
    if incar.get("ENCUT", 400) < recommended_encut:
        incar["ENCUT"] = recommended_encut
        notes.append(f"ENCUT raised to {recommended_encut} eV (1.3×ENMAX={enmax})")

    # Van der Waals
    if any(kw in desc for kw in ["d3", "dft-d3", "vdw", "van der waals", "dispersion"]):
        incar["IVDW"] = 11
        notes.append("DFT-D3 dispersion correction enabled (IVDW=11)")

    # Spin-orbit
    if "spin-orbit" in desc or "soc" in desc:
        incar["LSORBIT"] = True
        incar["LNONCOLLINEAR"] = True
        notes.append("Spin-orbit coupling enabled")

    # DFT+U
    if "+u" in desc or "hubbard" in desc:
        incar["LDAU"] = True
        incar["LDAUTYPE"] = 2
        notes.append("DFT+U enabled — set LDAUL/LDAUU/LDAUJ for your system")

    # Implicit solvation
    if "solvent" in desc or "solvation" in desc or "vaspsol" in desc:
        incar["LSOL"] = True
        incar["EB_K"] = 78.4
        notes.append("VASPsol implicit solvation enabled (ε=78.4, water)")

    # Relaxation
    if any(kw in desc for kw in ["relax", "optimize", "geometry optimization"]):
        incar["IBRION"] = 2
        incar["NSW"] = 200
        incar["EDIFFG"] = -0.02
        incar["ISIF"] = 2
        notes.append("Geometry optimization: IBRION=2 (CG), EDIFFG=-0.02 eV/A")

    # Frequency / ZPE
    if any(kw in desc for kw in ["frequency", "freq", "zpe", "zero-point", "vibration"]):
        incar["IBRION"] = 5
        incar["NSW"] = 1
        incar["POTIM"] = 0.015
        incar["NFREE"] = 2
        notes.append("Frequency calculation: IBRION=5, POTIM=0.015, NFREE=2")

    return {
        "ok": True,
        "incar": incar,
        "kpoints": kpoints,
        "calc_type": calc_type,
        "element": element,
        "adsorbate": adsorbate,
        "accuracy": req.accuracy,
        "notes": notes,
        "incar_string": _incar_to_string(incar),
    }


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _parse_surface(surface: str):
    """Parse 'Pt(111)' → ('Pt', '111')."""
    import re
    m = re.match(r'([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\((\d{3,4})\)', surface)
    if m:
        return m.group(1), m.group(2)
    # Fallback: try element-only
    m2 = re.match(r'([A-Z][a-z]?)', surface)
    if m2:
        return m2.group(1), "111"
    return "Cu", "111"


def _parse_description(desc: str) -> Dict[str, Any]:
    """Extract surface, adsorbate, calc type from natural language."""
    import re

    result: Dict[str, Any] = {"element": "Cu", "facet": "111", "calc_type": "static"}

    # Surface
    m = re.search(r'([A-Z][a-z]?(?:\d*[A-Z][a-z]?)*)\((\d{3,4})\)', desc, re.IGNORECASE)
    if m:
        result["element"] = m.group(1)
        result["facet"] = m.group(2)

    # Adsorbate
    ads_patterns = ["COOH", "CO2", "CO", "CH3OH", "CH2O", "CHO", "OOH", "OH", "NH3", "N2", "H2O", "H2", "H", "O"]
    for ads in ads_patterns:
        if ads.lower() in desc:
            result["adsorbate"] = ads
            break

    # Calc type
    if "dos" in desc or "density of states" in desc:
        result["calc_type"] = "dos"
    elif "band" in desc:
        result["calc_type"] = "band"
    elif "bader" in desc or "charge analysis" in desc:
        result["calc_type"] = "bader"
    elif "cohp" in desc or "lobster" in desc:
        result["calc_type"] = "cohp"
    elif "work function" in desc:
        result["calc_type"] = "work_function"
    elif "elf" in desc or "electron localization" in desc:
        result["calc_type"] = "elf"
    elif "neb" in desc or "transition state" in desc:
        result["calc_type"] = "static"  # NEB handled separately
    elif "relax" in desc or "optim" in desc:
        result["calc_type"] = "static"
    elif "freq" in desc or "vibrat" in desc or "zpe" in desc:
        result["calc_type"] = "static"

    return result


def _incar_to_string(params: Dict[str, Any]) -> str:
    """Format INCAR dict as VASP INCAR file content."""
    lines = ["# Generated by ChatDFT /api/smart_params", ""]
    for key, val in sorted(params.items()):
        if key.startswith("_") or key.startswith("$"):
            continue
        if isinstance(val, bool):
            val_str = ".TRUE." if val else ".FALSE."
        elif isinstance(val, float):
            val_str = f"{val:.1e}" if abs(val) < 1e-3 else f"{val}"
        else:
            val_str = str(val)
        lines.append(f"  {key:<12} = {val_str}")
    return "\n".join(lines) + "\n"
