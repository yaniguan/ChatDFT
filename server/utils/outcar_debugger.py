# server/utils/outcar_debugger.py
"""
OUTCAR / OSZICAR / INCAR parser for VASP convergence diagnostics.

Given a job directory, detect the failure mode and produce actionable
INCAR recommendations — so ChatDFT can say something specific instead of
"check your INCAR".

Detects:
  - Electronic non-convergence (NELM reached)
  - Energy oscillation (charge sloshing)
  - Ionic non-convergence (forces don't reach EDIFFG)
  - Magnetic moment instability
  - Symmetry errors (SGRCON / POSMAP)
  - Memory / parallelisation issues
  - Subspace-matrix hermiticity failure (DAV)
  - ZBRENT / ZBRAK relaxation errors
  - POTIM-related issues
  - Negative force-constant / soft phonon mode in freq calc
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class DiagnosticIssue:
    code:        str            # machine-readable tag
    severity:    str            # "critical" | "warning" | "info"
    description: str            # human-readable summary
    evidence:    List[str]      = field(default_factory=list)   # quoted OUTCAR snippets
    fixes:       List[str]      = field(default_factory=list)   # INCAR change suggestions
    refs:        List[str]      = field(default_factory=list)   # docs / papers


@dataclass
class DebugReport:
    job_dir:   str
    converged: bool                              # ionic convergence reached?
    elec_conv: bool                              # electronic convergence reached?
    issues:    List[DiagnosticIssue]             = field(default_factory=list)
    incar_patch: Dict[str, Any]                  = field(default_factory=dict)
    summary:   str                               = ""
    raw_stats: Dict[str, Any]                    = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_dir":   self.job_dir,
            "converged": self.converged,
            "elec_conv": self.elec_conv,
            "issues":    [{"code": i.code, "severity": i.severity,
                           "description": i.description,
                           "fixes": i.fixes, "evidence": i.evidence[:3]}
                          for i in self.issues],
            "incar_patch":  self.incar_patch,
            "summary":      self.summary,
            "raw_stats":    self.raw_stats,
        }


# ── INCAR reader ─────────────────────────────────────────────────────────

def _read_incar(d: Path) -> Dict[str, str]:
    incar = d / "INCAR"
    if not incar.exists():
        return {}
    params: Dict[str, str] = {}
    for line in incar.read_text(errors="ignore").splitlines():
        line = line.split("#")[0].strip()
        if "=" in line:
            k, _, v = line.partition("=")
            params[k.strip().upper()] = v.strip()
    return params


def _read_file_tail(path: Path, chars: int = 200_000) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(errors="ignore")[-chars:]
    except Exception:
        return ""


def _read_file_full(path: Path, chars: int = 50_000) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(errors="ignore")[:chars]
    except Exception:
        return ""


# ── Parser helpers ────────────────────────────────────────────────────────

def _parse_oszicar_energies(d: Path) -> List[float]:
    """Extract E0 energy series from OSZICAR."""
    text = _read_file_tail(d / "OSZICAR", 200_000)
    return [float(m) for m in re.findall(r"E0=\s*([-\d\.Ee+]+)", text)]


def _parse_nelm(incar: Dict[str, str]) -> int:
    return int(incar.get("NELM", "60"))


def _check_oscillation(energies: List[float], window: int = 20) -> Tuple[bool, float]:
    """Detect energy oscillation in the last `window` electronic steps."""
    if len(energies) < 6:
        return False, 0.0
    tail = energies[-window:]
    avg  = sum(tail) / len(tail)
    amp  = max(tail) - min(tail)
    # Oscillating if alternates sign around mean
    diffs = [tail[i+1] - tail[i] for i in range(len(tail)-1)]
    sign_changes = sum(1 for i in range(len(diffs)-1)
                       if diffs[i]*diffs[i+1] < 0)
    oscillating = sign_changes >= len(diffs) * 0.6 and amp > 0.01
    return oscillating, amp


def _parse_incar_from_outcar(text: str) -> Dict[str, str]:
    """Extract INCAR parameters echoed in the OUTCAR header."""
    params: Dict[str, str] = {}
    for m in re.finditer(r"^\s{3}(\w+)\s*=\s*(.+)$", text[:30_000], re.M):
        params[m.group(1).upper()] = m.group(2).strip()
    return params


# ── Individual checkers ───────────────────────────────────────────────────

def _check_electronic_convergence(d: Path, incar: Dict[str, str], outcar_text: str) -> Optional[DiagnosticIssue]:
    nelm = _parse_nelm(incar)
    reached = f"NELM    =   {nelm}" in outcar_text or re.search(
        r"aborting loop EDIFF reached.*?(?:no|not)", outcar_text, re.I
    )
    if reached is not None and not "aborting loop EDIFF reached" in outcar_text:
        # NELM reached without convergence
        algo = incar.get("ALGO", "Fast").upper()
        fixes = []
        if algo in ("FAST", ""):
            fixes.append("ALGO = Normal   # more stable than Fast (RMM-DIIS)")
        if algo == "NORMAL":
            fixes.append("ALGO = All      # fully blocked Davidson – slowest but safest")
        fixes += [
            "NELM = 200      # give more electronic iterations",
            "AMIX = 0.2      # reduce charge mixing (default 0.4)",
            "BMIX = 0.00001  # reduce Kerker metric",
        ]
        return DiagnosticIssue(
            code="ELEC_NOT_CONV",
            severity="critical",
            description=f"Electronic steps did not converge within NELM={nelm}.",
            evidence=[f"NELM={nelm} reached; 'aborting loop EDIFF reached' not found in OUTCAR"],
            fixes=fixes,
            refs=["https://www.vasp.at/wiki/index.php/ALGO"],
        )
    return None


def _check_charge_sloshing(d: Path, incar: Dict[str, str]) -> Optional[DiagnosticIssue]:
    energies = _parse_oszicar_energies(d)
    oscillating, amp = _check_oscillation(energies)
    if oscillating:
        return DiagnosticIssue(
            code="CHARGE_SLOSHING",
            severity="critical",
            description=f"Energy oscillating with amplitude ~{amp:.3f} eV (charge sloshing).",
            evidence=[f"Last E0 values: {[round(e,4) for e in energies[-8:]]}"],
            fixes=[
                "ALGO = Damped   # or ALGO = All",
                "AMIX = 0.1      # halve the charge mixing",
                "BMIX = 0.0001",
                "TIME = 0.05     # step size for Damped algorithm",
                "# For metals with small gap: ISMEAR=1, SIGMA=0.05",
            ],
            refs=["https://www.vasp.at/wiki/index.php/AMIX"],
        )
    return None


def _check_ionic_convergence(outcar_text: str, incar: Dict[str, str]) -> Optional[DiagnosticIssue]:
    nsw = int(incar.get("NSW", "0"))
    reached = "reached required accuracy" in outcar_text.lower()
    if nsw > 0 and not reached:
        ibrion = incar.get("IBRION", "2")
        potim  = float(incar.get("POTIM", "0.5"))
        fixes  = []
        if potim >= 0.5:
            fixes.append(f"POTIM = {potim/2:.2f}  # halve the ionic step size")
        if ibrion == "2":
            fixes.append("IBRION = 1      # switch from CG to RFO — more stable for near-TS geometries")
        fixes += [
            "EDIFFG = -0.05  # relax criterion to -0.05 eV/Å first; tighten later",
            "NSW = 500       # allow more ionic steps",
        ]
        return DiagnosticIssue(
            code="IONIC_NOT_CONV",
            severity="warning",
            description=f"Ionic relaxation did not converge within NSW={nsw} steps.",
            evidence=["'reached required accuracy' not found in OUTCAR"],
            fixes=fixes,
        )
    return None


def _check_magmom_instability(outcar_text: str, incar: Dict[str, str]) -> Optional[DiagnosticIssue]:
    if incar.get("ISPIN", "1") != "2":
        return None
    # If mag moment listed in OUTCAR is wildly different from MAGMOM init
    mag_matches = re.findall(r"number of electron\s+([\d\.]+)\s+magnetization\s+([-\d\.]+)", outcar_text)
    if not mag_matches:
        return None
    mags = [float(m[1]) for m in mag_matches]
    if len(mags) >= 2 and abs(mags[-1] - mags[-2]) > 2.0:
        return DiagnosticIssue(
            code="MAGMOM_UNSTABLE",
            severity="warning",
            description="Magnetic moment is oscillating / unstable between ionic steps.",
            evidence=[f"Magnetisation history (last 5): {mags[-5:]}"],
            fixes=[
                "# Reinitialise MAGMOM closer to the expected value",
                "MAGMOM = <per-atom values>  # use LORBIT=11 to inspect site moments",
                "# Try ISPIN=1 if the system is non-magnetic",
                "NUPDOWN = 0  # fix total spin if needed",
            ],
        )
    return None


def _check_symmetry_error(outcar_text: str) -> Optional[DiagnosticIssue]:
    if re.search(r"SGRCON|POSMAP|inconsistent Bravais", outcar_text, re.I):
        return DiagnosticIssue(
            code="SYMMETRY_ERROR",
            severity="critical",
            description="VASP detected a symmetry inconsistency (SGRCON/POSMAP error).",
            evidence=["SGRCON or POSMAP keyword found in OUTCAR"],
            fixes=[
                "ISYM = 0   # disable symmetry entirely",
                "# OR: ISYM = 2  (use symmetry but recompute from scratch)",
                "# Check POSCAR for near-symmetric but slightly broken geometry",
            ],
        )
    return None


def _check_subspace_hermitian(outcar_text: str) -> Optional[DiagnosticIssue]:
    if "Sub-Space-Matrix is not hermitian" in outcar_text:
        return DiagnosticIssue(
            code="DAV_HERMITIAN",
            severity="critical",
            description="Davidson sub-space matrix is not hermitian — numerical instability in ALGO=Fast.",
            evidence=["'Sub-Space-Matrix is not hermitian' found in OUTCAR"],
            fixes=[
                "ALGO = Normal   # switches from RMM-DIIS to blocked Davidson",
                "ALGO = All      # fully blocked Davidson — most stable",
                "ENCUT = <try slightly different value, e.g. +10 eV>",
            ],
        )
    return None


def _check_zbrent(outcar_text: str) -> Optional[DiagnosticIssue]:
    if re.search(r"ZBRENT|ZBRAK", outcar_text, re.I):
        return DiagnosticIssue(
            code="ZBRENT_ERROR",
            severity="critical",
            description="ZBRENT/ZBRAK root-finding failure during ionic relaxation.",
            evidence=["ZBRENT/ZBRAK keyword found in OUTCAR"],
            fixes=[
                "IBRION = 1    # switch to RFO; ZBRENT only occurs with IBRION=1,2,3",
                "POTIM = 0.1   # reduce step size drastically",
                "# Check for extremely flat potential energy surface",
                "# Recheck POSCAR for overlapping atoms",
            ],
        )
    return None


def _check_memory(outcar_text: str) -> Optional[DiagnosticIssue]:
    if re.search(r"please increase ncore|out of memory|insufficient memory", outcar_text, re.I):
        return DiagnosticIssue(
            code="MEMORY_ERROR",
            severity="critical",
            description="VASP ran out of memory.",
            evidence=["Memory error string found in OUTCAR"],
            fixes=[
                "NCORE = 4     # reduce memory per core (try 2, 4, 8)",
                "# Reduce ENCUT or use smaller supercell for initial test",
                "LWAVE = .FALSE.; LCHARG = .FALSE.  # don't write large wavefunctions",
                "# Request more RAM or more nodes on your cluster",
            ],
        )
    return None


def _check_negative_freq(outcar_text: str) -> Optional[DiagnosticIssue]:
    """Detect unexpected imaginary modes in a frequency / TS validation run."""
    freqs = re.findall(r"(f/i|f)\s*=\s*([\d\.]+)\s*THz", outcar_text)
    imaginary = [float(f[1]) for f in freqs if f[0] == "f/i"]
    if imaginary:
        n = len(imaginary)
        if n > 1:
            return DiagnosticIssue(
                code="MULTIPLE_IMAGINARY_FREQ",
                severity="warning",
                description=f"{n} imaginary frequencies detected. Expect exactly 1 for a true TS.",
                evidence=[f"Imaginary modes (THz): {imaginary}"],
                fixes=[
                    "# Re-optimise structure: check POTIM, NSW, EDIFFG",
                    "# Rerun NEB with more images near the TS",
                    "IBRION = 5  # switch to finite-differences for freq calc",
                    "POTIM = 0.015  # smaller finite-difference displacement",
                ],
            )
    return None


def _check_functional_notes(incar: Dict[str, str], d: Path) -> List[DiagnosticIssue]:
    """Flag potentially inappropriate functional / parameter choices."""
    issues = []
    gga  = incar.get("GGA", "PE").upper()
    ldau = incar.get("LDAU", ".FALSE.").upper()
    poscar_text = _read_file_full(d / "POSCAR", 500)

    # DFT+U reminder for first-row TM oxides
    tm_oxides = re.compile(r"\b(Fe|Co|Ni|Mn|Cr|V|Cu|Ti|Mo|W)(?!\d)", re.I)
    if tm_oxides.search(poscar_text) and ldau not in (".TRUE.", "T", "TRUE"):
        issues.append(DiagnosticIssue(
            code="MISSING_DFTU",
            severity="info",
            description="First-row transition metal detected; consider DFT+U for d-electron localisation.",
            fixes=[
                "LDAU = .TRUE.",
                "LDAUTYPE = 2     # Dudarev simplified rotational invariant",
                "LDAUU = 3.3      # example U for Fe (adjust per species/property)",
                "LDAUJ = 0.0",
                "LDAUL = 2        # apply to d orbitals (l=2)",
                "LDAUPRINT = 1    # print occupation matrix",
            ],
            refs=["https://www.vasp.at/wiki/index.php/LDAU"],
        ))

    # vdW reminder for physisorption / organic molecules
    disp = incar.get("IVDW", "0")
    if gga == "PE" and disp == "0":
        issues.append(DiagnosticIssue(
            code="NO_DISPERSION",
            severity="info",
            description="PBE without dispersion correction — consider adding DFT-D3 for physisorption / organic adsorbates.",
            fixes=[
                "IVDW = 11   # DFT-D3(BJ) — recommended default",
                "# Or use GGA=BO (optB86b-vdW) for surface adsorption benchmarks",
            ],
            refs=["https://www.vasp.at/wiki/index.php/IVDW"],
        ))

    return issues


# ── ZPE extraction from OUTCAR freq run ──────────────────────────────────

def extract_zpe_from_outcar(outcar_path: Path) -> Optional[float]:
    """
    Parse a VASP IBRION=5/6 frequency calculation OUTCAR and return ZPE (eV).
    ZPE = ½ Σ hν  for all *real* (non-imaginary) modes.
    """
    text = _read_file_tail(outcar_path, 300_000)
    # VASP prints: "   f  =    12.345  THz ..."   or   "   f/i=    5.678  THz ..."
    freq_matches = re.findall(
        r"^\s+(f/i|f)\s*=\s*([\d\.]+)\s+THz",
        text, re.M
    )
    if not freq_matches:
        return None
    # Sum only real (non-imaginary) modes; ½hν in eV where h·THz = h_eV * 1e12
    h_planck = 4.135668e-15  # eV·s
    zpe = 0.0
    for kind, val in freq_matches:
        if kind == "f":
            freq_THz = float(val)
            zpe += 0.5 * h_planck * freq_THz * 1e12
    return round(zpe, 5) if zpe > 0 else None


# ── Surface stability knowledge base ─────────────────────────────────────

SURFACE_STABILITY: Dict[str, Dict] = {
    "Pt(100)": {
        "warning": "Pt(100) undergoes a hex (5×20) reconstruction under UHV; "
                   "this reconstruction lifts under CO or H₂ adsorbates.",
        "mitigation": "Use Pt(100)-1×1 with adsorbed CO/H as model, or simulate "
                      "the reconstructed hex surface for clean-surface properties.",
    },
    "Au(111)": {
        "warning": "Au(111) exhibits the 22×√3 herringbone reconstruction. "
                   "A standard (111) slab overestimates adsorption strengths.",
        "mitigation": "For quantitative adsorption, use a (111) slab with a dispersion "
                      "functional (optB86b-vdW or DFT-D3). Report as approximate.",
    },
    "Ir(100)": {
        "warning": "Ir(100) reconstructs to a (5×1) overlayer structure under UHV.",
        "mitigation": "Check recent literature for Ir(100) under your conditions.",
    },
    "Ni(100)": {
        "warning": "Ni(100) can form a c(2×2)-O surface oxide under oxidising conditions.",
        "mitigation": "If modelling oxidation catalysis, include O* coverage effects.",
    },
    "Cu(100)": {
        "warning": "Cu(100) undergoes a missing-row (√2×√2)R45° reconstruction in the presence of O*.",
        "mitigation": "Consider oxygen-induced restructuring in CO2RR / OER models.",
    },
}

FUNCTIONAL_ADVICE: Dict[str, Dict] = {
    "CO_Pt": {
        "system_pattern": r"\bCO\b.*\bPt\b|\bPt\b.*\bCO\b",
        "issue": "PBE overestimates CO binding energy on Pt surfaces by 0.2–0.5 eV "
                 "(the 'CO puzzle'). This affects ΔG estimates and site selectivity.",
        "recommendations": [
            "Use BEEF-vdW (GGA=BF in VASP; requires BEEF patch) for unbiased adsorbate energetics.",
            "Apply PBE+D3 with a +0.2 eV empirical correction to CO adsorption energies.",
            "Benchmark against: Schimka et al., Nature Mater. 9, 741 (2010).",
        ],
        "incar_note": "GGA = BF  # BEEF-vdW (needs patched VASP)",
    },
    "oxides_DFT+U": {
        "system_pattern": r"\b(Fe₂O₃|TiO₂|CeO₂|NiO|CoO|MnO₂|VO₂|CrO₂|MoO₃|WO₃)\b",
        "issue": "PBE strongly delocalises d electrons in transition-metal oxides, "
                 "leading to incorrect band gaps, wrong oxidation states, and large "
                 "adsorption energy errors.",
        "recommendations": [
            "Add Hubbard U correction (DFT+U, Dudarev scheme LDAUTYPE=2).",
            "Common U values: Fe 3.3 eV, Ti 3.5 eV, Ni 6.4 eV, Co 3.3 eV, Ce 4.5 eV.",
            "Screen U with DFPT or linear response (Cococcioni & de Gironcoli, PRB 71, 2005).",
        ],
    },
    "physisorption_vdW": {
        "system_pattern": r"\b(benzene|toluene|phenol|pyridine|porphyrin|MOF|physisorb)",
        "issue": "PBE misses van der Waals interactions, underestimating physisorption by 0.5–1 eV.",
        "recommendations": [
            "Use DFT-D3(BJ): IVDW=11 in INCAR.",
            "For benchmark accuracy: optB86b-vdW (GGA=BO) or rev-vdW-DF2.",
        ],
    },
    "bandgap_hybrid": {
        "system_pattern": r"\b(band.?gap|semiconductor|photocatalys|TiO₂|GaN|Si\()",
        "issue": "PBE underestimates semiconductor band gaps by 30–50%. This affects "
                 "band alignment, optical absorption, and charge-transfer barriers.",
        "recommendations": [
            "Use HSE06 for accurate gaps: LHFCALC=.TRUE. AEXX=0.25 HFSCREEN=0.2.",
            "For qualitative trends: PBE is still widely used with explicit gap correction.",
            "GLLB-SC or GW for highly accurate quasiparticle gaps.",
        ],
    },
    "electrochemistry_PCET": {
        "system_pattern": r"\b(CHE|PCET|potential|V vs RHE|electrocatalysis|electroch)",
        "issue": "Standard DFT cannot model explicit electrode potential. "
                 "CHE assumes linear scaling with electrode potential.",
        "recommendations": [
            "Use the Computational Hydrogen Electrode (CHE) for thermodynamics.",
            "For explicit potential: consider VASPsol (LSOL=.TRUE.) or grand-canonical DFT.",
            "PCET barriers: align free energy diagram at the limiting potential first.",
        ],
    },
}


def check_surface_stability(material: str, facet: str) -> Optional[Dict]:
    """Return stability warning for known reconstructing surfaces."""
    key = f"{material}({facet})"
    return SURFACE_STABILITY.get(key)


def recommend_functional(system_description: str) -> List[Dict]:
    """Return functional recommendations matching the system description."""
    recs = []
    for key, data in FUNCTIONAL_ADVICE.items():
        pat = data.get("system_pattern", "")
        if pat and re.search(pat, system_description, re.I):
            recs.append({
                "topic":           key,
                "issue":           data["issue"],
                "recommendations": data["recommendations"],
                "incar_note":      data.get("incar_note", ""),
            })
    return recs


# ── Main entry point ──────────────────────────────────────────────────────

def debug_job(job_dir: str | Path) -> DebugReport:
    """
    Analyse a VASP job directory and return a DebugReport with:
      - convergence status
      - list of DiagnosticIssue objects
      - suggested INCAR patch dict
      - human-readable summary
    """
    d = Path(job_dir)
    incar          = _read_incar(d)
    outcar_text    = _read_file_tail(d / "OUTCAR", 300_000)
    outcar_head    = _read_file_full(d / "OUTCAR", 30_000)

    # Convergence flags
    ionic_conv = "reached required accuracy" in outcar_text.lower()
    elec_conv  = "aborting loop because EDIFF is reached" in outcar_text

    issues: List[DiagnosticIssue] = []

    # --- run all checkers ---
    for checker in [
        lambda: _check_electronic_convergence(d, incar, outcar_text),
        lambda: _check_charge_sloshing(d, incar),
        lambda: _check_ionic_convergence(outcar_text, incar),
        lambda: _check_magmom_instability(outcar_text, incar),
        lambda: _check_symmetry_error(outcar_text),
        lambda: _check_subspace_hermitian(outcar_text),
        lambda: _check_zbrent(outcar_text),
        lambda: _check_memory(outcar_text),
        lambda: _check_negative_freq(outcar_text),
    ]:
        try:
            issue = checker()
            if issue:
                issues.append(issue)
        except Exception:
            pass

    # Functional / parameter notes
    try:
        issues += _check_functional_notes(incar, d)
    except Exception:
        pass

    # ZPE extraction (opportunistic)
    zpe = None
    try:
        zpe = extract_zpe_from_outcar(d / "OUTCAR")
    except Exception:
        pass

    # Build combined INCAR patch
    incar_patch: Dict[str, Any] = {}
    for issue in issues:
        for fix in issue.fixes:
            m = re.match(r"^([A-Z_]+)\s*=\s*(.+?)(?:\s*#.*)?$", fix.strip())
            if m:
                k, v = m.group(1), m.group(2).strip()
                incar_patch[k] = v

    # Human summary
    critical = [i for i in issues if i.severity == "critical"]
    warnings = [i for i in issues if i.severity == "warning"]
    infos    = [i for i in issues if i.severity == "info"]
    parts = []
    if not issues:
        parts.append("No issues detected — job appears healthy.")
    else:
        if critical:
            parts.append(f"{len(critical)} critical: {', '.join(i.code for i in critical)}.")
        if warnings:
            parts.append(f"{len(warnings)} warning(s): {', '.join(i.code for i in warnings)}.")
        if infos:
            parts.append(f"{len(infos)} info note(s).")
    summary = "  ".join(parts)

    return DebugReport(
        job_dir   = str(d),
        converged = ionic_conv,
        elec_conv = elec_conv,
        issues    = issues,
        incar_patch = incar_patch,
        summary   = summary,
        raw_stats = {
            "zpe_eV":       zpe,
            "incar_keys":   list(incar.keys()),
            "oszicar_len":  len(_parse_oszicar_energies(d)),
        },
    )
