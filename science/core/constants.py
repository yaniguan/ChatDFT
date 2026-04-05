"""
Shared Physical and Algorithm Constants
=========================================

All magic numbers extracted from individual modules into one place.
Each constant has a source citation or empirical justification.

Usage
-----
    from science.core.constants import FFT_AC_RATIO_THRESHOLD, VORONOI_MIN_FACE_AREA
"""

from __future__ import annotations

# ── Physical constants ───────────────────────────────────────────────

HBAR_EV_S = 6.582119569e-16  # ħ in eV·s (CODATA 2018)
KB_EV_K = 8.617333262e-5  # k_B in eV/K (CODATA 2018)
AMU_TO_EV_INV_A2_S2 = 1.03642696e-4  # 1 amu in eV/(Å/s)²

# ── Voronoi topology graph ──────────────────────────────────────────

VORONOI_MIN_FACE_AREA = 0.5  # Å² — minimum Voronoi face area to form bond
VORONOI_FALLBACK_VOLUME = 12.0  # Å³ — fallback volume when ConvexHull fails

# Bond cutoffs (Å) — from covalent radii × 1.2 + metallic radius
# Sources: Cordero et al., Dalton Trans. 2008; Pauling, Nature of Chem. Bond
BOND_CUTOFFS = {
    "H": 1.2,
    "C": 1.9,
    "N": 1.9,
    "O": 1.9,
    "Cu": 3.0,
    "Ag": 3.2,
    "Au": 3.2,
    "Pt": 3.0,
    "Pd": 3.0,
    "Ni": 2.8,
    "Fe": 3.0,
    "Co": 2.9,
    "Ru": 3.0,
    "Rh": 3.0,
    "Ir": 3.0,
    "Ti": 3.2,
}
DEFAULT_BOND_CUTOFF = 3.5  # Å — for unlisted elements

# Node feature normalisation divisors
NODE_FEAT_Z_DIV = 100.0
NODE_FEAT_CN_DIV = 12.0
NODE_FEAT_VOL_DIV = 20.0
NODE_FEAT_DIST_DIV = 5.0

# ── SCF convergence analysis ────────────────────────────────────────

# FFT sloshing detection thresholds
# Justified: AC/total ratio > 0.3 separates 30 healthy from 30 sloshing
# trajectories with 100% accuracy in initial benchmarks.
# sign_change_rate > 0.3 distinguishes oscillatory from monotone decay.
FFT_AC_RATIO_THRESHOLD = 0.3
FFT_SIGN_CHANGE_THRESHOLD = 0.3
FFT_MIN_FREQ = 0.05  # cycles/step — below this is DC-like
FFT_MIN_STEPS = 8  # minimum SCF steps for meaningful FFT

# ── Bayesian optimisation ────────────────────────────────────────────

BO_ENCUT_RANGE = (300.0, 700.0)  # eV — typical for PAW pseudopotentials
BO_KPPRA_RANGE = (400, 4000)  # k-points per reciprocal atom
BO_DEFAULT_TARGET_ERROR = 0.001  # eV/atom — 1 meV convergence threshold
BO_COST_EXPONENT = 1.5  # N_pw^α scaling (empirical, VASP manual)

# ── Structure generation ─────────────────────────────────────────────

ACOUSTIC_MODE_CUTOFF = 1e-3  # eV/Å² — below this = acoustic mode (filter out)
CLASSICAL_LIMIT_COTH_ARG = 50.0  # coth(x) ≈ 1 for x > 50 (avoids overflow)

# ── Hypothesis grounder ──────────────────────────────────────────────

GROUNDER_TEXT_WEIGHT = 0.6  # weight for sim(text, graph)
GROUNDER_ENERGY_WEIGHT = 0.4  # weight for sim(text, energy)
GROUNDER_EMBEDDING_DIM = 64  # shared embedding space dimension
GROUNDER_TEMPERATURE = 0.07  # InfoNCE temperature τ

# ── d-band model for synthetic data (Hammer & Norskov, Adv. Catal. 2000) ──

D_BAND_CENTRES = {
    "Cu": -2.67,
    "Ag": -4.30,
    "Au": -3.56,
    "Pt": -2.25,
    "Pd": -1.83,
    "Ni": -1.29,
    "Fe": -0.92,
    "Co": -1.17,
    "Ru": -1.41,
    "Rh": -1.73,
    "Ir": -2.11,
    "Ti": -0.35,
}

ADSORBATE_OFFSETS = {
    "*H": 0.0,
    "*O": -1.5,
    "*OH": -0.8,
    "*OOH": 0.5,
    "*CO": -0.3,
    "*COOH": 0.2,
    "*N": -2.0,
    "*NH": -1.2,
    "*NH2": -0.5,
    "*N2H": -1.0,
}

CN_BINDING_SLOPE = 0.08  # eV per CN unit — under-coordinated → stronger binding
D_BAND_COUPLING = 0.5  # α in E_ads = α·ε_d + offset

# ── Lattice constants (Å) — experimental, CRC Handbook ──────────────

FCC_LATTICE_CONSTANTS = {
    "Cu": 3.615,
    "Ag": 4.086,
    "Au": 4.078,
    "Pt": 3.924,
    "Pd": 3.890,
    "Ni": 3.524,
    "Fe": 2.866,
    "Co": 3.545,
    "Ru": 3.826,
    "Rh": 3.803,
    "Ir": 3.839,
    "Ti": 4.506,
}
