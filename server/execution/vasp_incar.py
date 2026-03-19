# server/execution/vasp_incar.py
# -*- coding: utf-8 -*-
"""
Comprehensive VASP INCAR parameter sets for each electronic structure calculation type.

Two-step workflow for non-SCF calcs (DOS, band, ELF via ICHARG=11):
  Step 1 — SCF:     ISTART=0, NSW=0, LWAVE=True, LCHARG=True  → writes WAVECAR + CHGCAR
  Step 2 — non-SCF: ISTART=1, ICHARG=11, dense k-mesh or band path

Special requirements:
  ELF     → NCORE=1 (hard requirement; VASP aborts otherwise)
  Bader   → LAECHG=True (all-electron AECCAR0/AECCAR2), PREC=Accurate, LREAL=False
  CDD     → three separate static calcs; ρ_diff = ρ_AB − ρ_A − ρ_B
  WF      → LVHAR=True (LOCPOT), LDIPOL=True, IDIPOL=3
  COHP    → ISYM=−1 (LOBSTER symmetry-breaks SCF), LORBIT=11, LWAVE=True
"""
from __future__ import annotations
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Base parameters shared by most surface calculations
# ---------------------------------------------------------------------------
_BASE_METAL: Dict[str, Any] = {
    "SYSTEM":  "ChatDFT",
    "PREC":    "Normal",
    "ENCUT":   400,
    "EDIFF":   1e-5,
    "ALGO":    "Fast",
    "ISMEAR":  1,       # Methfessel-Paxton smearing — metals
    "SIGMA":   0.2,
    "GGA":     "PE",    # PBE
    "LREAL":   "Auto",
    "NCORE":   4,
    "LWAVE":   False,
    "LCHARG":  False,
    "IBRION":  -1,
    "NSW":     0,
    "ISIF":    0,
}


def _merge(base: Dict, overrides: Dict) -> Dict:
    out = dict(base)
    out.update(overrides)
    return out


# ---------------------------------------------------------------------------
# 1. Static SCF — prerequisite for all non-SCF calculations
#    Writes WAVECAR + CHGCAR needed by DOS / band / ELF / WF / COHP
# ---------------------------------------------------------------------------
STATIC_SCF: Dict[str, Any] = _merge(_BASE_METAL, {
    "ISTART":  0,
    "ICHARG":  2,
    "LWAVE":   True,    # MUST be True — DOS/band/ELF read WAVECAR
    "LCHARG":  True,    # MUST be True — DOS/band/ELF read CHGCAR
    "EDIFF":   1e-6,    # Tighter for electronic props
    "NELM":    200,
    # Recommended k-mesh for SCF: 8x8x1 (surfaces) — denser for non-SCF
    # ISMEAR=1, SIGMA=0.2 stays (metals)
    "_comment": (
        "Step 1 of 2: SCF with LWAVE+LCHARG=True. "
        "Follow with ICHARG=11 + dense k-mesh for DOS/band."
    ),
})


# ---------------------------------------------------------------------------
# 2. DOS — density of states (non-SCF, ICHARG=11)
#    Use tetrahedron method (ISMEAR=-5) for accurate peak positions.
#    Requires prior SCF run in same directory (CHGCAR present).
# ---------------------------------------------------------------------------
DOS: Dict[str, Any] = _merge(_BASE_METAL, {
    "ISTART":  1,
    "ICHARG":  11,      # Non-SCF: reads CHGCAR, does NOT update charge
    "ISMEAR":  -5,      # Tetrahedron: no smearing artifacts in DOS
    "SIGMA":   0.05,
    "LORBIT":  11,      # lm-decomposed PDOS (s,p,d per atom)
    "NEDOS":   2000,    # Number of DOS grid points
    "EMIN":    -15.0,   # Energy window start (eV, relative to E_F)
    "EMAX":    10.0,    # Energy window end
    "LWAVE":   False,
    "LCHARG":  False,
    "NCORE":   4,
    "_comment": (
        "Step 2 of 2 (non-SCF). Requires CHGCAR from SCF run. "
        "Use ISMEAR=-5 + dense k-mesh (12×12×1 minimum for surfaces). "
        "LORBIT=11 gives lm-decomposed PDOS for d-band center analysis."
    ),
})


# ---------------------------------------------------------------------------
# 3. PDOS — projected DOS (same as DOS but emphasising orbital projections)
#    Separate entry for plan-agent dispatch; parameters identical to DOS.
# ---------------------------------------------------------------------------
PDOS: Dict[str, Any] = dict(DOS)
PDOS["_comment"] = (
    "Projected DOS: same INCAR as DOS. "
    "Extract d-band center: ε_d = ∫ε·n_d(ε)dε / ∫n_d(ε)dε. "
    "Post-process DOSCAR with vaspkit (option 111) or pyprocar."
)


# ---------------------------------------------------------------------------
# 4. Band structure (non-SCF, ICHARG=11, explicit k-path)
#    Generate KPOINTS with high-symmetry path (ASE bandpath or vaspkit).
# ---------------------------------------------------------------------------
BAND: Dict[str, Any] = _merge(_BASE_METAL, {
    "ISTART":  1,
    "ICHARG":  11,
    "ISMEAR":  0,       # Gaussian smearing for band plots (no tetrahedron on line-mode)
    "SIGMA":   0.05,
    "LORBIT":  11,
    "LWAVE":   False,
    "LCHARG":  False,
    "NCORE":   4,
    "_comment": (
        "Step 2 of 2 (non-SCF). KPOINTS must be in line-mode with high-symmetry path. "
        "ASE: lat.bandpath(npoints=60). "
        "ISMEAR=0 (Gaussian) — tetrahedron does not work on explicit k-line paths. "
        "Post-process: bs = calc.band_structure(); bs.plot()"
    ),
})


# ---------------------------------------------------------------------------
# 5. ELF — electron localization function
#    CRITICAL: NCORE=1 required — VASP aborts with NCORE>1 + LELF=True.
#    Must be a full SCF run (cannot use ICHARG=11 for LELF).
# ---------------------------------------------------------------------------
ELF: Dict[str, Any] = _merge(_BASE_METAL, {
    "ISTART":  1,       # Read WAVECAR from prior SCF
    "ICHARG":  0,       # SCF (charge from WAVECAR) — NOT ICHARG=11
    "LELF":    True,    # Write ELFCAR
    "LWAVE":   True,
    "LCHARG":  True,
    "NCORE":   1,       # *** MANDATORY — VASP aborts if NCORE > 1 with LELF ***
    "PREC":    "Accurate",
    "EDIFF":   1e-6,
    "_comment": (
        "NCORE=1 is mandatory with LELF=True (VASP hard requirement). "
        "Visualize ELFCAR in VESTA: isosurface at 0.75 shows lone pairs & bonds. "
        "ELF=1.0 → fully localized (lone pairs), ELF=0.5 → electron gas, ELF=0 → depleted."
    ),
})


# ---------------------------------------------------------------------------
# 6. Bader charge analysis
#    LAECHG=True: write all-electron partial charge AECCAR0 (core) + AECCAR2 (valence).
#    Post-process: chgsum.pl AECCAR0 AECCAR2 → CHGCAR_sum; bader CHGCAR -ref CHGCAR_sum
#    Requires bader binary (Henkelman group: theory.cm.utexas.edu/bader/).
# ---------------------------------------------------------------------------
BADER: Dict[str, Any] = _merge(_BASE_METAL, {
    "ISTART":  0,
    "ICHARG":  2,
    "ENCUT":   520,     # Higher ENCUT for accurate charge density
    "PREC":    "Accurate",
    "EDIFF":   1e-6,
    "LREAL":   False,   # Reciprocal-space projection — required for Bader
    "LAECHG":  True,    # Write AECCAR0 + AECCAR2 (all-electron densities)
    "LWAVE":   False,
    "LCHARG":  True,
    "NCORE":   4,
    "_comment": (
        "After SCF: run chgsum.pl AECCAR0 AECCAR2 (VTST scripts) → CHGCAR_sum. "
        "Then: bader CHGCAR -ref CHGCAR_sum → ACF.dat (charges per atom). "
        "Bader charge = Z_ion - q_bader. "
        "Download bader: theory.cm.utexas.edu/bader/"
    ),
})


# ---------------------------------------------------------------------------
# 7. Charge density difference (CDD)
#    Requires THREE separate static calculations:
#      calc_AB → adsorbate + slab (relaxed geometry)
#      calc_A  → slab only (same geometry as AB, atoms removed)
#      calc_B  → adsorbate only (same geometry as AB, atoms removed)
#    ρ_CDD = ρ_AB − ρ_A − ρ_B
# ---------------------------------------------------------------------------
CDD: Dict[str, Any] = _merge(_BASE_METAL, {
    "ISTART":  0,
    "ICHARG":  2,
    "EDIFF":   1e-6,
    "LWAVE":   False,
    "LCHARG":  True,    # All three calcs need CHGCAR
    "LAECHG":  False,   # Not needed for CDD
    "NCORE":   4,
    "_comment": (
        "Run identical INCAR for three systems: AB (combined), A (slab), B (adsorbate). "
        "Use the RELAXED geometry for all three — do NOT re-relax A or B separately. "
        "Post-process: Δρ = CHGCAR_AB - CHGCAR_A - CHGCAR_B "
        "(use vaspkit option 31X or custom Python: ase.io.vasp.read_vasp_chgcar). "
        "Yellow/blue isosurfaces show charge accumulation/depletion at interface."
    ),
})


# ---------------------------------------------------------------------------
# 8. Work function (LVHAR → LOCPOT)
#    LDIPOL=True + IDIPOL=3 corrects artificial dipole from asymmetric slabs.
#    WF = E_vacuum − E_Fermi (read LOCPOT planar average + OUTCAR E_Fermi).
# ---------------------------------------------------------------------------
WORK_FUNCTION: Dict[str, Any] = _merge(_BASE_METAL, {
    "ISTART":  0,
    "ICHARG":  2,
    "EDIFF":   1e-6,
    "LVHAR":   True,    # Write LOCPOT (total electrostatic potential)
    "LDIPOL":  True,    # Dipole correction — critical for asymmetric slabs
    "IDIPOL":  3,       # Correct along z-axis (slab normal)
    "LWAVE":   False,
    "LCHARG":  False,
    "NCORE":   4,
    "_comment": (
        "Work function φ = E_vacuum − E_Fermi. "
        "E_Fermi from OUTCAR ('E-fermi' line). "
        "E_vacuum = plateau of LOCPOT planar average (z far from slab). "
        "Post-process: python -c 'from ase.io.vasp import read_vasp_out; ...' "
        "or vaspkit option 426. "
        "LDIPOL+IDIPOL=3: corrects the artificial electric field for asymmetric slabs."
    ),
})


# ---------------------------------------------------------------------------
# 9. COHP / COOP — Crystal Orbital Hamilton Population (requires LOBSTER)
#    VASP: ISYM=-1 (break symmetry so LOBSTER gets all k-points),
#          LORBIT=11 for DOSCAR, LWAVE=True for WAVECAR.
#    Then run LOBSTER with mainInput specifying atom pairs.
# ---------------------------------------------------------------------------
COHP: Dict[str, Any] = _merge(_BASE_METAL, {
    "ISTART":  0,
    "ICHARG":  2,
    "ISYM":    -1,      # *** CRITICAL: LOBSTER needs all k-points unfolded ***
    "LWAVE":   True,    # LOBSTER reads WAVECAR
    "LCHARG":  True,
    "LORBIT":  11,      # DOSCAR needed by LOBSTER for orbital populations
    "EDIFF":   1e-6,
    "NCORE":   4,
    "NBANDS":  None,    # Increase if LOBSTER complains (nbands > 2 × NELECT typical)
    "_comment": (
        "ISYM=-1 is mandatory for LOBSTER (all k-points must be explicitly computed). "
        "After VASP: run lobster with mainInput: "
        "  COHPstartEnergy -20.0 / COHPendEnergy 5.0 / cohpGenerator from 1 to N type X type Y "
        "Bonding: -ICOHP > 0 = bonding; antibonding: < 0. "
        "LOBSTER download: schmeling.ac.at/lobster/"
    ),
})


# ---------------------------------------------------------------------------
# Dispatch table: calc_type → INCAR dict + recommended settings
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 10. Single-point for NNP training datasets
#     NSW=0, IBRION=-1 (no relaxation) — just forces + stress for each frame.
#     ISIF=2: compute stress tensor (needed for NNP).
#     LWAVE/LCHARG=False: skip heavy files — only OUTCAR needed.
# ---------------------------------------------------------------------------
SINGLEPOINT_NNP: Dict[str, Any] = _merge(_BASE_METAL, {
    "ISTART":  0,
    "ICHARG":  2,
    "PREC":    "Accurate",
    "ENCUT":   450,
    "EDIFF":   1e-6,
    "ISIF":    2,        # compute stress tensor — essential for NNP training
    "LWAVE":   False,    # no WAVECAR — not needed for NNP labels
    "LCHARG":  False,    # no CHGCAR — not needed for NNP labels
    "LREAL":   "Auto",
    "NELM":    200,
    "_comment": (
        "NNP training single-point. "
        "Keep ENCUT consistent (450 eV) across ALL structures in the dataset. "
        "ISIF=2 gives stress tensor as additional training label. "
        "Parse energy/forces/stress from OUTCAR; export as extXYZ for NNP training."
    ),
})


INCAR_PRESETS: Dict[str, Dict[str, Any]] = {
    "static":        STATIC_SCF,
    "static_scf":    STATIC_SCF,
    "scf":           STATIC_SCF,
    "dos":           DOS,
    "pdos":          PDOS,
    "band":          BAND,
    "bands":         BAND,
    "elf":           ELF,
    "bader":         BADER,
    "cdd":           CDD,
    "charge_density_difference": CDD,
    "work_function": WORK_FUNCTION,
    "wf":            WORK_FUNCTION,
    "cohp":          COHP,
    "coop":          COHP,
    "lobster":       COHP,
    "singlepoint_nnp": SINGLEPOINT_NNP,
    "nnp":           SINGLEPOINT_NNP,
    "nnp_sp":        SINGLEPOINT_NNP,
}


def get_incar(calc_type: str) -> Dict[str, Any]:
    """Return the INCAR parameter dict for *calc_type* (copy, safe to mutate)."""
    key = (calc_type or "static").lower().strip()
    preset = INCAR_PRESETS.get(key, STATIC_SCF)
    # Return a copy without the internal _comment key
    return {k: v for k, v in preset.items() if not k.startswith("_") and v is not None}


def get_comment(calc_type: str) -> str:
    """Return the usage comment/notes for *calc_type*."""
    key = (calc_type or "static").lower().strip()
    preset = INCAR_PRESETS.get(key, STATIC_SCF)
    return preset.get("_comment", "")


def incar_to_string(params: Dict[str, Any]) -> str:
    """Format an INCAR parameter dict as a VASP INCAR file string."""
    lines = ["# Generated by ChatDFT vasp_incar.py", ""]
    for key, val in params.items():
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


# ---------------------------------------------------------------------------
# Recommended k-mesh for surface calculations (n_layers × vacuum ignored here)
# ---------------------------------------------------------------------------
KPOINTS_SUGGESTIONS: Dict[str, str] = {
    "scf":           "8 8 1",
    "static":        "8 8 1",
    "dos":           "12 12 1",
    "pdos":          "12 12 1",
    "band":          "bandpath",   # high-symmetry line mode
    "elf":           "8 8 1",
    "bader":         "8 8 1",
    "cdd":           "8 8 1",
    "work_function": "8 8 1",
    "wf":            "8 8 1",
    "cohp":          "8 8 1",
}


def suggested_kpoints(calc_type: str) -> str:
    key = (calc_type or "static").lower()
    return KPOINTS_SUGGESTIONS.get(key, "8 8 1")
