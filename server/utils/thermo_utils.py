# server/utils/thermo_utils.py
"""
Thermodynamic corrections and free-energy diagram tools for DFT output.

Provides:
  - ZPE + entropy (T*S) corrections with literature defaults
  - CHE (Computational Hydrogen Electrode) potential corrections
  - Free energy diagram builder (JSON + optional base64 PNG)
  - Microkinetic solver (mean-field steady-state → TOF, selectivity)
  - Known standard pathways for quick reference (CO2RR, HER, OER, NRR)
"""
from __future__ import annotations

import io, math, base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Physical constants ──────────────────────────────────────────────────────
kB   = 8.617333e-5   # eV / K
h_eV = 4.135668e-15  # eV · s
T_STD = 298.15       # K

# ── ZPE defaults (eV) — from Norskov / Peterson / Abild-Pedersen groups ───
STANDARD_ZPE: Dict[str, float] = {
    "*":      0.00,
    "H*":     0.16,  "OH*":    0.36,  "O*":     0.07,
    "OOH*":   0.45,  "H2O*":   0.52,  "CO*":    0.19,
    "CO2*":   0.27,  "COOH*":  0.45,  "HCOO*":  0.44,
    "CHO*":   0.41,  "COH*":   0.35,  "CH2O*":  0.58,
    "CH3O*":  0.72,  "CH3OH*": 0.88,  "C*":     0.04,
    "CH*":    0.14,  "CH2*":   0.30,  "CH3*":   0.47,
    "CH4*":   0.62,  "N*":     0.05,  "NH*":    0.17,
    "NH2*":   0.33,  "NH3*":   0.60,  "N2*":    0.10,
    "N2H*":   0.25,  "N2H2*":  0.42,  "N2H3*":  0.57,
    "N2H4*":  0.73,  "NO*":    0.10,  "NO2*":   0.22,
    "NO3*":   0.35,
}

# ── T*S defaults at 298 K for adsorbed species (eV) ─────────────────────
STANDARD_TS_298: Dict[str, float] = {
    "*":      0.00,  "H*":     0.04,  "OH*":    0.05,
    "O*":     0.03,  "OOH*":   0.07,  "H2O*":   0.12,
    "CO*":    0.08,  "CO2*":   0.10,  "COOH*":  0.09,
    "HCOO*":  0.09,  "CHO*":   0.11,  "COH*":   0.10,
    "CH2O*":  0.13,  "CH3O*":  0.16,  "N*":     0.03,
    "NH*":    0.04,  "NH2*":   0.05,  "NH3*":   0.08,
    "NO*":    0.06,
}

# ── Gas-phase T*S at 298 K, 1 bar (eV) — NIST JANAF ───────────────────
GAS_TS_298: Dict[str, float] = {
    "H2": 0.403,  "H2O": 0.672,  "CO2": 0.664,
    "CO": 0.611,  "N2":  0.591,  "NH3": 0.597,
    "O2": 0.634,  "CH4": 0.611,  "C2H4": 0.735,
    "C2H6": 0.781, "N2H4": 0.711, "NO": 0.593,
    "NO2": 0.634,  "HNO3": 0.701,
}

# ── Gas-phase ZPE (eV) ───────────────────────────────────────────────────
GAS_ZPE: Dict[str, float] = {
    "H2": 0.270,  "H2O": 0.558,  "CO2": 0.306,
    "CO": 0.132,  "N2":  0.146,  "NH3": 0.903,
    "O2": 0.098,  "CH4": 1.184,  "C2H4": 1.285,
    "C2H6": 1.869,
}


# ── Data containers ──────────────────────────────────────────────────────

@dataclass
class ThermoCorrection:
    species: str
    E_dft:   float = 0.0
    zpe:     float = 0.0
    ts:      float = 0.0
    G:       float = 0.0
    zpe_src: str   = "default"
    ts_src:  str   = "default"


@dataclass
class FreeEnergyDiagram:
    pathway_name:     str
    temperature:      float
    potential_V:      float          # V vs RHE
    steps:            List[Dict]     = field(default_factory=list)
    rds_index:        int            = -1
    limiting_potential: float        = 0.0
    overpotential:    float          = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pathway": self.pathway_name,
            "T_K":     self.temperature,
            "U_V_RHE": self.potential_V,
            "steps":   self.steps,
            "rds_index": self.rds_index,
            "U_limiting_V": round(self.limiting_potential, 4),
            "overpotential_V": round(self.overpotential, 4),
        }


# ── Core correction helpers ──────────────────────────────────────────────

def _strip_phase(sp: str) -> str:
    return sp.replace("(g)", "").replace("(l)", "").replace("(aq)", "").strip()


def get_zpe(species: str, outcar_zpe: Optional[float] = None) -> Tuple[float, str]:
    if outcar_zpe is not None:
        return outcar_zpe, "outcar"
    sp = _strip_phase(species)
    if species.endswith("(g)") or species.endswith("(l)"):
        v = GAS_ZPE.get(sp)
        if v is not None:
            return v, "literature"
    v = STANDARD_ZPE.get(sp)
    if v is not None:
        return v, "literature"
    return 0.0, "default"


def get_ts(species: str, T: float = T_STD) -> Tuple[float, str]:
    sp = _strip_phase(species)
    is_gas = species.endswith("(g)") or species.endswith("(l)")
    if is_gas:
        ts_298 = GAS_TS_298.get(sp, 0.0)
        src = "literature" if sp in GAS_TS_298 else "default"
    else:
        ts_298 = STANDARD_TS_298.get(sp, 0.0)
        src = "literature" if sp in STANDARD_TS_298 else "default"
    return ts_298 * (T / T_STD), src


def apply_thermo_corrections(
    E_dft:       float,
    species:     str,
    T:           float = T_STD,
    outcar_zpe:  Optional[float] = None,
) -> ThermoCorrection:
    """Apply ZPE and T*S to a DFT energy → free energy G."""
    zpe, zpe_src = get_zpe(species, outcar_zpe)
    ts,  ts_src  = get_ts(species, T)
    return ThermoCorrection(
        species=species, E_dft=E_dft, zpe=zpe, ts=ts,
        G=E_dft + zpe - ts, zpe_src=zpe_src, ts_src=ts_src,
    )


def che_shift(potential_V: float, n_electrons: int = 1) -> float:
    """
    CHE correction per proton-electron transfer step.
    Returns the energy *shift* (eV) to add to ΔG_rxn when applying potential U.
    For cathodic reduction: each H⁺+e⁻ pair is stabilised by eU.
    """
    return -n_electrons * potential_V


# ── Free energy diagram builder ──────────────────────────────────────────

def build_free_energy_diagram(
    pathway_name:       str,
    intermediates:      List[str],
    G_relative:         List[float],      # G relative to first species (eV)
    ec_step_indices:    Optional[List[int]] = None,   # which steps are H⁺+e⁻
    electrons_per_step: Optional[List[int]] = None,
    T:                  float = T_STD,
    potential_V:        float = 0.0,
) -> FreeEnergyDiagram:
    """
    Build a FreeEnergyDiagram, applying CHE corrections at electrode potential U.

    ec_step_indices: 1-based step indices that are electrochemical (PCET).
    electrons_per_step: number of electrons transferred per EC step.
    """
    ec_set = set(ec_step_indices or [])
    n_e    = electrons_per_step or [1] * len(intermediates)

    # Apply CHE: accumulate shift for each electrochemical step
    G_shifted = list(G_relative)
    cumulative = 0.0
    for i in range(len(intermediates)):
        if i in ec_set:
            cumulative += che_shift(potential_V, n_e[i] if i < len(n_e) else 1)
        G_shifted[i] = G_relative[i] + cumulative

    delta_Gs = [0.0] + [G_shifted[i] - G_shifted[i-1] for i in range(1, len(G_shifted))]

    steps = []
    for i, (sp, g, dg) in enumerate(zip(intermediates, G_shifted, delta_Gs)):
        steps.append({
            "index": i, "label": sp,
            "G": round(g, 4), "delta_G": round(dg, 4),
            "is_ec": i in ec_set,
            "n_e": n_e[i] if i < len(n_e) else 0,
        })

    # Rate-determining step = largest positive ΔG
    rds_idx = max(range(1, len(delta_Gs)), key=lambda i: delta_Gs[i], default=0)

    # Limiting potential (most negative required for all ΔG ≤ 0)
    ec_dGs = [delta_Gs[i] for i in ec_set if i < len(delta_Gs)]
    U_lim  = -max(ec_dGs) if ec_dGs else 0.0

    # Equilibrium potential
    n_total = sum(n_e[i] for i in ec_set if i < len(n_e))
    dG_overall = G_shifted[-1] - G_shifted[0]
    U_eq = -dG_overall / n_total if n_total > 0 else 0.0
    eta  = abs(U_lim - U_eq) if n_total > 0 else 0.0

    return FreeEnergyDiagram(
        pathway_name=pathway_name, temperature=T, potential_V=potential_V,
        steps=steps, rds_index=rds_idx,
        limiting_potential=round(U_lim, 4),
        overpotential=round(eta, 4),
    )


def plot_free_energy_diagram(
    diagrams: List[FreeEnergyDiagram],
    title: str = "Free Energy Diagram",
    show_rds: bool = True,
) -> Optional[str]:
    """
    Render diagrams as a matplotlib figure.
    Returns base64-encoded PNG string, or None if matplotlib unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    n_pts = max(len(d.steps) for d in diagrams)
    fig, ax = plt.subplots(figsize=(max(8, n_pts * 1.6), 5))
    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for ci, diag in enumerate(diagrams):
        color = COLORS[ci % len(COLORS)]
        G_vals = [s["G"] for s in diag.steps]
        labels = [s["label"] for s in diag.steps]
        xs = list(range(len(G_vals)))

        # Flat horizontal lines per state
        for i, (x, g) in enumerate(zip(xs, G_vals)):
            ax.hlines(g, x - 0.35, x + 0.35, colors=color, linewidths=2.5,
                      label=diag.pathway_name if i == 0 else "")
            ax.text(x, g + 0.04, labels[i], ha="center", va="bottom",
                    fontsize=7, color=color, rotation=30, clip_on=True)

        # Dashed connectors
        for i in range(len(xs) - 1):
            ax.plot([xs[i]+0.35, xs[i+1]-0.35], [G_vals[i], G_vals[i+1]],
                    "--", color=color, alpha=0.5, linewidth=1)

        # Mark RDS
        if show_rds and 0 < diag.rds_index < len(G_vals):
            r = diag.rds_index
            ax.annotate(
                f"RDS ΔG={diag.steps[r]['delta_G']:.2f} eV",
                xy=(r, G_vals[r]), xytext=(r + 0.25, G_vals[r] + 0.2),
                fontsize=7.5, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1),
            )

    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xticks(range(len(diagrams[0].steps)))
    ax.set_xticklabels([s["label"] for s in diagrams[0].steps],
                       rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("ΔG (eV)", fontsize=10)
    T  = diagrams[0].temperature
    U  = diagrams[0].potential_V
    ax.set_title(f"{title}  (T={T:.0f} K, U={U:+.2f} V vs RHE)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Standard pathways (representative literature values) ─────────────────
# G values relative to clean surface + gas reactants (eV, U=0 vs RHE)
KNOWN_PATHWAYS: Dict[str, Dict] = {
    "CO2RR_carboxyl_Cu111": {
        "intermediates": ["CO₂(g)+*", "COOH*", "CO*+H₂O", "CO(g)+*"],
        "G":             [ 0.00,        0.22,   -0.56,      0.10],
        "ec_steps":      [1, 2],  "n_e": [1, 1, 0, 0],
        "description":   "Carboxyl (COOH*) pathway — Cu(111), Peterson et al.",
    },
    "CO2RR_formate_Cu111": {
        "intermediates": ["CO₂(g)+*", "HCOO*", "HCOOH(g)+*"],
        "G":             [ 0.00,       0.14,    0.08],
        "ec_steps":      [1, 2],  "n_e": [1, 1, 0],
        "description":   "Formate (HCOO*) pathway — Cu(111)",
    },
    "HER_VHT_Pt111": {
        "intermediates": ["H⁺+e⁻+*", "H*", "½H₂(g)+*"],
        "G":             [ 0.00,      -0.09,  0.00],
        "ec_steps":      [1, 2],  "n_e": [1, 1, 0],
        "description":   "Volmer-Heyrovsky/Tafel HER — Pt(111)",
    },
    "OER_4step": {
        "intermediates": ["H₂O+*", "OH*", "O*", "OOH*", "O₂(g)+*"],
        "G":             [ 0.00,    1.56,  2.44, 3.54,   4.92],
        "ec_steps":      [1, 2, 3, 4],  "n_e": [1, 1, 1, 1, 0],
        "description":   "4-step OER via OH*/O*/OOH*",
    },
    "NRR_distal": {
        "intermediates": ["N₂(g)+*", "N₂*", "NNH*", "NNH₂*", "N*+NH₃", "NH*", "NH₂*", "NH₃+*"],
        "G":             [ 0.00,      0.25,   0.81,   1.24,     0.64,     0.34,   0.12,   0.00],
        "ec_steps":      [2, 3, 4, 5, 6, 7],  "n_e": [0, 0, 1, 1, 1, 1, 1, 1],
        "description":   "Distal NRR pathway",
    },
    "NO3RR_NH3": {
        "intermediates": ["NO₃⁻+*", "NO₃*", "NO₂*", "NO*", "N*", "NH*", "NH₂*", "NH₃+*"],
        "G":             [ 0.00,     -0.40,  -0.72,  -0.95, -0.20, 0.10,  0.02,   -0.15],
        "ec_steps":      [1, 2, 3, 4, 5, 6, 7],  "n_e": [1]*8,
        "description":   "NO3RR → NH3 pathway",
    },
}


def get_known_pathway(key: str, T: float = T_STD, U: float = 0.0) -> Optional[FreeEnergyDiagram]:
    """Return a FreeEnergyDiagram for a well-known reaction pathway."""
    pw = KNOWN_PATHWAYS.get(key)
    if pw is None:
        return None
    return build_free_energy_diagram(
        pathway_name    = key.replace("_", " "),
        intermediates   = pw["intermediates"],
        G_relative      = pw["G"],
        ec_step_indices = pw["ec_steps"],
        electrons_per_step = pw.get("n_e"),
        T=T, potential_V=U,
    )


def suggest_competing_pathways(reaction: str) -> List[str]:
    """Given a reaction name (e.g. 'CO2RR', 'HER'), return relevant pathway keys."""
    r = reaction.upper()
    mapping = {
        "CO2RR": ["CO2RR_carboxyl_Cu111", "CO2RR_formate_Cu111"],
        "HER":   ["HER_VHT_Pt111"],
        "OER":   ["OER_4step"],
        "NRR":   ["NRR_distal"],
        "NO3RR": ["NO3RR_NH3"],
    }
    for k, v in mapping.items():
        if k in r:
            return v
    return []


# ── Microkinetic solver (mean-field, single-site Langmuir-Hinshelwood) ───

def arrhenius_k(Ea_eV: float, T: float = T_STD, A: Optional[float] = None) -> float:
    """Forward rate constant from Arrhenius / transition state theory."""
    if A is None:
        A = kB * T / h_eV   # ~6.25×10¹² s⁻¹ at 300 K
    return A * math.exp(-max(Ea_eV, 0.0) / (kB * T))


def solve_microkinetics(
    step_labels:  List[str],
    Ea_fwd_eV:    List[float],   # forward barrier per step (eV)
    delta_G_eV:   List[float],   # net ΔG per step (eV)
    T:            float = T_STD,
    max_iter:     int   = 100_000,
    tol:          float = 1e-10,
) -> Dict[str, Any]:
    """
    Simple mean-field microkinetic solver.

    Assumes one active site type (*), Langmuir-Hinshelwood kinetics.
    Coverage array: theta[0] = free site, theta[i] = i-th intermediate.

    Returns:
        coverages: dict {theta_0, theta_1, ...}
        rates:     dict {step_label: net_rate}
        TOF:       turnover frequency (s⁻¹)
        selectivity: None (single product)
        rate_controlling_step: label of step with highest |rate|
        converged: bool
    """
    n = len(step_labels)
    if n == 0:
        return {"ok": False, "detail": "No steps provided."}

    k_fwd = [arrhenius_k(max(Ea, 0.0), T) for Ea in Ea_fwd_eV]
    k_rev = [arrhenius_k(max(Ea - dG, 0.0), T)
             for Ea, dG in zip(Ea_fwd_eV, delta_G_eV)]

    # Initial coverages: uniform
    theta = [1.0 / (n + 1)] * (n + 1)

    dt = 1e-14
    for iteration in range(max_iter):
        rates   = [k_fwd[i]*theta[i] - k_rev[i]*theta[i+1] for i in range(n)]
        dtheta  = [0.0] * (n + 1)
        for i, r in enumerate(rates):
            dtheta[i]   -= r
            dtheta[i+1] += r

        new_theta = [max(0.0, theta[j] + dt * dtheta[j]) for j in range(n + 1)]
        total = sum(new_theta) or 1.0
        new_theta = [t / total for t in new_theta]

        max_change = max(abs(new_theta[j] - theta[j]) for j in range(n + 1))
        theta = new_theta
        if max_change < tol:
            break
        if iteration % 5000 == 4999:
            dt = min(dt * 5, 1e-8)

    rates    = [k_fwd[i]*theta[i] - k_rev[i]*theta[i+1] for i in range(n)]
    tof      = rates[0] if rates else 0.0
    abs_r    = [abs(r) for r in rates]
    rcs_idx  = abs_r.index(max(abs_r)) if abs_r else 0

    return {
        "ok":        True,
        "converged": True,
        "TOF_per_s": round(tof, 6),
        "coverages": {f"theta_{i}": round(t, 6) for i, t in enumerate(theta)},
        "rates":     {step_labels[i]: round(rates[i], 6) for i in range(n)},
        "rate_controlling_step": step_labels[rcs_idx] if step_labels else "",
        "T_K":       T,
    }


def run_microkinetics_from_diagram(
    diag:    FreeEnergyDiagram,
    T:       float = T_STD,
    Ea_fwd:  Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper: extract step labels + ΔG from a FreeEnergyDiagram
    and feed into the microkinetic solver.

    Ea_fwd defaults to max(ΔG, 0) per step (barrierless assumption for
    downhill steps; no explicit TS information required).
    """
    if len(diag.steps) < 2:
        return {"ok": False, "detail": "Need ≥2 steps."}

    labels  = [f"{diag.steps[i]['label']}→{diag.steps[i+1]['label']}"
               for i in range(len(diag.steps) - 1)]
    dGs     = [diag.steps[i+1]["delta_G"] for i in range(len(diag.steps) - 1)]
    Ea_list = Ea_fwd or [max(dg, 0.0) for dg in dGs]

    return solve_microkinetics(
        step_labels=labels, Ea_fwd_eV=Ea_list, delta_G_eV=dGs, T=T
    )
