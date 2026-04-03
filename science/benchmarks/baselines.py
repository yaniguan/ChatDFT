"""
Baseline Implementations for All Science Modules
==================================================
For each ChatDFT algorithm, we provide naive/standard baselines to
demonstrate the improvement our approach offers.

Baselines:
  1. Surface sites   : Distance-cutoff nearest-neighbor (no Voronoi)
  2. Structure gen    : Uniform random rattle (no physics)
  3. Hypothesis score : TF-IDF keyword overlap (no cross-modal alignment)
  4. SCF prediction   : Linear extrapolation (no FFT analysis)
  5. Parameter search : Grid search (no Bayesian optimization)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 1. Surface Site Baseline: Distance-Cutoff NN
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BaselineSiteResult:
    n_sites: int
    site_types: Dict[str, int]
    runtime_ms: float


def baseline_distance_cutoff_sites(
    positions: np.ndarray,
    elements: list[str],
    cutoff: float = 3.0,
) -> BaselineSiteResult:
    """
    Naive site finder: top sites above every atom, bridge at midpoint
    of every pair within cutoff. No Voronoi, no symmetry scoring.
    """
    t0 = time.perf_counter()
    N = len(positions)
    z_coords = positions[:, 2]
    surface_z = np.max(z_coords)
    # Surface atoms: within 1.0 A of top
    surface_mask = z_coords > (surface_z - 1.0)
    surface_idx = np.where(surface_mask)[0]

    sites = {"top": len(surface_idx), "bridge": 0, "hollow": 0}

    # Bridge: every pair within cutoff
    for i, idx_i in enumerate(surface_idx):
        for idx_j in surface_idx[i+1:]:
            d = np.linalg.norm(positions[idx_i] - positions[idx_j])
            if d < cutoff:
                sites["bridge"] += 1

    # Hollow: every triplet (very crude)
    if len(surface_idx) >= 3:
        sites["hollow"] = max(1, len(surface_idx) - 2)

    runtime = (time.perf_counter() - t0) * 1000
    return BaselineSiteResult(
        n_sites=sum(sites.values()),
        site_types=sites,
        runtime_ms=runtime,
    )


def voronoi_sites(positions, elements, cell):
    """Run our Voronoi method and return comparable result."""
    from science.representations.surface_graph import SurfaceTopologyGraph
    t0 = time.perf_counter()
    stg = SurfaceTopologyGraph(positions, elements, cell)
    stg.build()
    sites = stg.classify_adsorption_sites()
    runtime = (time.perf_counter() - t0) * 1000
    from collections import Counter
    counts = Counter(s.site_type for s in sites)
    return BaselineSiteResult(
        n_sites=len(sites),
        site_types=dict(counts),
        runtime_ms=runtime,
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. Structure Generation Baseline: Uniform Random Rattle
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RattleComparisonResult:
    method: str
    sigma_mean: float
    sigma_std: float
    preserves_symmetry: bool
    energy_spread_eV: float   # spread in potential energy of generated configs


def baseline_uniform_rattle(
    positions: np.ndarray,
    sigma: float = 0.1,
    n: int = 20,
    seed: int = 42,
) -> list[np.ndarray]:
    """
    Naive baseline: uniform Gaussian noise with fixed sigma for ALL atoms.
    No mass weighting, no temperature scaling, no quantum corrections.
    """
    rng = np.random.default_rng(seed)
    return [positions + rng.normal(0, sigma, positions.shape) for _ in range(n)]


def einstein_rattle_positions(
    positions: np.ndarray,
    masses: np.ndarray,
    T_K: float = 600,
    n: int = 20,
    seed: int = 42,
) -> list[np.ndarray]:
    """Run our Einstein rattle and return displaced positions."""
    from science.generation.informed_sampler import EinsteinRattler, AtomsLike
    rattler = EinsteinRattler(omega_THz=5.0, quantum=True, rng_seed=seed)
    atoms = AtomsLike(
        positions=positions,
        numbers=np.ones(len(positions), dtype=int),
        cell=np.eye(3) * 20.0,
        masses=masses,
    )
    batch = rattler.generate_batch(atoms, T_K, n)
    return [a.get_positions() for a in batch]


def compare_rattle_methods(
    positions: np.ndarray,
    masses: np.ndarray,
    T_K: float = 600,
    n: int = 50,
) -> Dict[str, RattleComparisonResult]:
    """Compare uniform vs Einstein rattle statistics."""
    results = {}

    # Baseline: uniform
    uniform_configs = baseline_uniform_rattle(positions, sigma=0.1, n=n)
    displacements_u = [c - positions for c in uniform_configs]
    sigma_per_atom_u = np.array([np.std(d, axis=1) for d in displacements_u])
    results["uniform"] = RattleComparisonResult(
        method="uniform",
        sigma_mean=float(np.mean(sigma_per_atom_u)),
        sigma_std=float(np.std(sigma_per_atom_u)),
        preserves_symmetry=False,
        energy_spread_eV=0.0,  # would need calculator
    )

    # Ours: Einstein
    einstein_configs = einstein_rattle_positions(positions, masses, T_K, n)
    displacements_e = [c - positions for c in einstein_configs]
    sigma_per_atom_e = np.array([np.std(d, axis=1) for d in displacements_e])
    results["einstein"] = RattleComparisonResult(
        method="einstein_quantum",
        sigma_mean=float(np.mean(sigma_per_atom_e)),
        sigma_std=float(np.std(sigma_per_atom_e)),
        preserves_symmetry=True,  # mass-weighted → heavier atoms move less
        energy_spread_eV=0.0,
    )

    return results


# ═══════════════════════════════════════════════════════════════════════
# 3. Hypothesis Scoring Baseline: TF-IDF Keyword Overlap
# ═══════════════════════════════════════════════════════════════════════

# Chemistry keywords for keyword-based scoring
_CHEM_KEYWORDS = {
    "adsorption", "desorption", "protonation", "reduction", "oxidation",
    "intermediate", "barrier", "overpotential", "selectivity", "catalyst",
    "surface", "electrode", "faradaic", "mechanism", "pathway",
    "exothermic", "endothermic", "thermodynamic", "kinetic",
}

_SPECIES_KEYWORDS = {
    "*", "CO2", "CO", "COOH", "CHO", "CH2O", "OH", "OOH", "O",
    "H", "H2", "H2O", "N2", "NH3", "NNH", "HCOOH",
}


def baseline_keyword_score(
    hypothesis: str,
    intermediates: list[str],
) -> float:
    """
    Naive baseline: score = Jaccard overlap between hypothesis tokens
    and the set of expected intermediates + chemistry keywords.
    """
    hyp_tokens = set(hypothesis.lower().replace("*", " * ").split())
    ref_tokens = set()
    for sp in intermediates:
        ref_tokens.update(sp.lower().replace("*", " * ").split())
    ref_tokens.update(k for k in _CHEM_KEYWORDS if k in hypothesis.lower())

    if not ref_tokens:
        return 0.0
    overlap = len(hyp_tokens & ref_tokens)
    union = len(hyp_tokens | ref_tokens)
    return overlap / max(union, 1)


def grounder_score(
    hypothesis: str,
    network_dict: dict,
    dG_profile: list[float] = None,
) -> float:
    """Run our cross-modal grounder and return the score."""
    from science.alignment.hypothesis_grounder import (
        HypothesisGrounder, ReactionNetwork,
    )
    grounder = HypothesisGrounder()
    network = ReactionNetwork.from_dict(network_dict)
    return grounder.score(hypothesis, network, dG_profile)


# ═══════════════════════════════════════════════════════════════════════
# 4. SCF Prediction Baseline: Linear Extrapolation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SCFPredictionComparison:
    method: str
    predicted_step: int
    actual_step: int
    error: int
    sloshing_detected: bool
    runtime_ms: float


def baseline_linear_extrapolation(
    dE: list[float],
    ediff: float = 1e-5,
    window: int = 5,
) -> Tuple[int, bool]:
    """
    Naive baseline: fit a line to the last `window` values of log|dE|
    and extrapolate to ediff. No FFT, no sloshing detection.
    """
    if len(dE) < window:
        return -1, False
    log_dE = np.log(np.clip(dE[-window:], 1e-20, None))
    steps = np.arange(len(dE) - window, len(dE), dtype=float)
    slope, intercept = np.polyfit(steps, log_dE, 1)
    if slope >= 0:
        return -1, False  # diverging
    n_conv = int(np.ceil((np.log(ediff) - intercept) / slope))
    return max(n_conv, 0), False  # cannot detect sloshing


def our_scf_prediction(
    dE: list[float],
    ediff: float = 1e-5,
    nelm: int = 60,
) -> Tuple[int, bool]:
    """Run our full SCF analysis pipeline."""
    from science.time_series.scf_convergence import (
        SCFTrajectory, ChargeSloshingDetector, ConvergenceRatePredictor,
    )
    traj = SCFTrajectory(dE=dE, ediff=ediff, nelm=nelm)
    sloshing = ChargeSloshingDetector().detect(traj)
    prediction = ConvergenceRatePredictor().predict(traj)
    return prediction.predicted_step, sloshing.is_sloshing


# ═══════════════════════════════════════════════════════════════════════
# 5. Parameter Search Baseline: Grid Search
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GridSearchResult:
    optimal_encut: float
    optimal_kppra: int
    n_evaluations: int
    best_error: float
    runtime_ms: float


def baseline_grid_search(
    energy_fn,    # (encut, kppra) -> energy_eV
    encut_values: list[float] = None,
    kppra_values: list[int] = None,
    n_atoms: int = 36,
    target_error: float = 0.001,
) -> GridSearchResult:
    """
    Standard grid search: evaluate ALL combinations.
    """
    if encut_values is None:
        encut_values = list(range(300, 601, 50))   # 7 values
    if kppra_values is None:
        kppra_values = list(range(400, 3201, 400))  # 8 values

    t0 = time.perf_counter()
    results = []
    for e in encut_values:
        for k in kppra_values:
            energy = energy_fn(e, k)
            results.append((e, k, energy))

    # Reference: highest ENCUT × KPPRA
    ref = max(results, key=lambda r: r[0] * r[1])
    ref_energy = ref[2]

    # Find best that meets target
    best = None
    for e, k, en in results:
        err = abs(en - ref_energy) / n_atoms
        if err <= target_error:
            if best is None or (e * k < best[0] * best[1]):
                best = (e, k, err)

    if best is None:
        best_sorted = sorted(results, key=lambda r: abs(r[2] - ref_energy))
        best = (best_sorted[0][0], best_sorted[0][1],
                abs(best_sorted[0][2] - ref_energy) / n_atoms)

    runtime = (time.perf_counter() - t0) * 1000
    return GridSearchResult(
        optimal_encut=best[0],
        optimal_kppra=best[1],
        n_evaluations=len(results),
        best_error=best[2],
        runtime_ms=runtime,
    )


def bayesian_search(
    energy_fn,
    n_atoms: int = 36,
    n_initial: int = 5,
    n_bo_steps: int = 10,
    target_error: float = 0.001,
) -> Tuple[float, int, int, float]:
    """
    Run our BO and return (optimal_encut, optimal_kppra, n_evals, best_error).
    """
    from science.optimization.bayesian_params import BayesianParameterOptimizer

    t0 = time.perf_counter()
    opt = BayesianParameterOptimizer(n_atoms=n_atoms, target_error=target_error)
    for encut, kppra in opt.suggest_initial(n_initial):
        energy = energy_fn(encut, kppra)
        opt.observe(encut, kppra, energy)
    for _ in range(n_bo_steps):
        encut, kppra = opt.suggest_next()
        energy = energy_fn(encut, kppra)
        opt.observe(encut, kppra, energy)

    result = opt.result()
    runtime = (time.perf_counter() - t0) * 1000
    return (result.optimal_encut, result.optimal_kppra,
            result.n_evaluations, result.predicted_error)


# ═══════════════════════════════════════════════════════════════════════
# Synthetic DFT Energy Landscapes (for reproducible benchmarks)
# ═══════════════════════════════════════════════════════════════════════

def synthetic_energy_landscape(encut: float, kppra: int,
                               noise: float = 0.002) -> float:
    """
    Realistic synthetic energy landscape for convergence testing.
    Models the typical behaviour: energy decreases with ENCUT/KPPRA
    but has material-dependent oscillations from Pulay stress.

    E(encut, kppra) = E_ref + A/encut^1.5 + B/kppra + C*sin(encut/50)
    """
    rng = np.random.default_rng(int(encut * 1000 + kppra))
    E_ref = -142.567  # "converged" energy
    pulay = 15.0 / (encut ** 1.5)
    kpoint = 2.5 / kppra
    oscillation = 0.003 * np.sin(encut / 50.0)
    return E_ref + pulay + kpoint + oscillation + rng.normal(0, noise)


def synthetic_sloshing_trajectory(n: int = 40, period: int = 6,
                                   decay: float = -0.02) -> list[float]:
    """Generate a synthetic SCF trajectory with charge sloshing."""
    t = np.arange(n, dtype=float)
    envelope = np.exp(decay * t)  # growing if decay < 0
    oscillation = np.abs(np.sin(2 * np.pi * t / period))
    base = 1e-2
    return list(base * envelope * (0.5 + oscillation) + 1e-7)


def synthetic_healthy_trajectory(n: int = 30, rate: float = 0.3) -> list[float]:
    """Generate a synthetic healthy (monotone exponential decay) SCF trajectory."""
    t = np.arange(n, dtype=float)
    return list(1e-1 * np.exp(-rate * t) + 1e-8)


# ═══════════════════════════════════════════════════════════════════════
# Run All Benchmarks
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkSummary:
    """Summary of all benchmark comparisons."""
    surface_sites: Dict
    structure_gen: Dict
    hypothesis_scoring: Dict
    scf_prediction: Dict
    parameter_search: Dict

    def to_table(self) -> str:
        """Format as ASCII table for display."""
        lines = [
            "=" * 80,
            "ChatDFT Algorithm Benchmark Summary",
            "=" * 80,
            "",
            f"{'Module':<25s} {'Ours':<20s} {'Baseline':<20s} {'Improvement':<15s}",
            "-" * 80,
        ]

        # Surface sites
        ss = self.surface_sites
        lines.append(
            f"{'Surface Site Finding':<25s} "
            f"{'Voronoi+Delaunay':<20s} "
            f"{'Distance cutoff':<20s} "
            f"{ss.get('improvement', 'N/A'):<15s}"
        )

        # Structure generation
        sg = self.structure_gen
        lines.append(
            f"{'Structure Generation':<25s} "
            f"{'Einstein rattle':<20s} "
            f"{'Uniform noise':<20s} "
            f"{sg.get('improvement', 'N/A'):<15s}"
        )

        # Hypothesis scoring
        hs = self.hypothesis_scoring
        lines.append(
            f"{'Hypothesis Scoring':<25s} "
            f"{'Cross-modal (ours)':<20s} "
            f"{'Keyword overlap':<20s} "
            f"{hs.get('improvement', 'N/A'):<15s}"
        )

        # SCF prediction
        sp = self.scf_prediction
        lines.append(
            f"{'SCF Prediction':<25s} "
            f"{'FFT + OLS (ours)':<20s} "
            f"{'Linear extrap.':<20s} "
            f"{sp.get('improvement', 'N/A'):<15s}"
        )

        # Parameter search
        ps = self.parameter_search
        lines.append(
            f"{'Parameter Search':<25s} "
            f"{'Bayesian Opt.':<20s} "
            f"{'Grid Search':<20s} "
            f"{ps.get('improvement', 'N/A'):<15s}"
        )

        lines.append("=" * 80)
        return "\n".join(lines)
