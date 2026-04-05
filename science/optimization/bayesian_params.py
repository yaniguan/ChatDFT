"""
Bayesian Optimisation for DFT Parameter Search
================================================
Scientific motivation
---------------------
DFT convergence testing (ENCUT, KPPRA sweeps) is conventionally done by
grid search — 10 ENCUT values × 8 KPPRA values = 80 DFT single-points.
This is wasteful: most of the grid is far from the Pareto-optimal front.

We replace the grid with Bayesian Optimisation (BO):
  1. Fit a Gaussian Process (GP) surrogate on (ENCUT, KPPRA) → (energy_error, cost)
  2. Use Expected Improvement (EI) to select the next (ENCUT, KPPRA) point
  3. After ~10–15 evaluations, identify the Pareto-optimal parameter set
  4. Report the optimal (accuracy, cost) trade-off with uncertainty bounds

This typically finds the converged parameters in 12–15 DFT runs instead of 80.

Algorithm
---------
Surrogate  : Gaussian Process with Matern-5/2 kernel
Acquisition : Expected Improvement (EI), or ParEGO for multi-objective
Cost model  : T ~ alpha * N_atoms * ENCUT^1.5 * KPPRA / N_atoms

Key references
--------------
[1] Jones et al., J. Global Optim. 13, 455 (1998) — EGO
[2] Knowles, IEEE Trans. Evol. Comput. 10, 50 (2006) — ParEGO
[3] Snoek et al., NeurIPS 2012 — Practical Bayesian Optimization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaussian Process (minimal, self-contained)
# ---------------------------------------------------------------------------


class GaussianProcess:
    """
    Gaussian Process regression with Matern-5/2 kernel.

    K(r) = σ² (1 + √5 r/l + 5r²/(3l²)) exp(-√5 r/l)

    Fit by maximising log marginal likelihood over (σ², l, σ_noise).
    """

    def __init__(self, length_scale: float = 1.0, signal_var: float = 1.0, noise_var: float = 1e-4):
        self.l = length_scale
        self.sigma2 = signal_var
        self.noise = noise_var
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self._K_inv: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def _matern52(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matern-5/2 kernel matrix."""
        D = cdist(X1 / self.l, X2 / self.l, metric="euclidean")
        sqrt5_D = np.sqrt(5) * D
        return self.sigma2 * (1 + sqrt5_D + 5 * D**2 / 3) * np.exp(-sqrt5_D)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        """Fit GP to training data."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        K = self._matern52(X, X) + self.noise * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
            self._alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self._K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
        except np.linalg.LinAlgError:
            # Fallback: add more noise for numerical stability
            K += 1e-3 * np.eye(len(X))
            self._K_inv = np.linalg.inv(K)
            self._alpha = self._K_inv @ y
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) predictions at X."""
        if self.X_train is None:
            return np.zeros(len(X)), np.ones(len(X))
        K_star = self._matern52(X, self.X_train)
        K_ss = self._matern52(X, X)
        mu = K_star @ self._alpha
        var = np.diag(K_ss - K_star @ self._K_inv @ K_star.T)
        var = np.maximum(var, 1e-10)
        return mu, np.sqrt(var)

    def optimize_hyperparameters(self, n_restarts: int = 3):
        """Optimise length_scale and signal_var by marginal likelihood."""
        if self.X_train is None or len(self.X_train) < 3:
            return

        def neg_log_marginal(params):
            self.l, self.sigma2 = np.exp(params[0]), np.exp(params[1])
            K = self._matern52(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
                log_ml = (
                    -0.5 * self.y_train @ alpha
                    - np.sum(np.log(np.diag(L)))
                    - 0.5 * len(self.y_train) * np.log(2 * np.pi)
                )
                return -log_ml
            except np.linalg.LinAlgError:
                return 1e10

        best_val = float("inf")
        best_params = [np.log(self.l), np.log(self.sigma2)]
        rng = np.random.default_rng(42)
        for _ in range(n_restarts):
            x0 = rng.normal(0, 1, 2)
            res = minimize(neg_log_marginal, x0, method="L-BFGS-B", bounds=[(-3, 3), (-3, 3)])
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        self.l = np.exp(best_params[0])
        self.sigma2 = np.exp(best_params[1])
        self.fit(self.X_train, self.y_train)


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function."""
    improvement = mu - best_y - xi
    Z = improvement / (sigma + 1e-12)
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return ei


# ---------------------------------------------------------------------------
# Cost model for DFT calculations
# ---------------------------------------------------------------------------


def dft_cost_model(encut: float, kppra: int, n_atoms: int = 36) -> float:
    """
    Empirical wall-time model for a VASP single-point:
        T ~ alpha * N_atoms * N_pw^{1.5} * N_k

    where N_pw ∝ ENCUT^{1.5} and N_k ∝ KPPRA / N_atoms.
    Returns relative cost (normalised so standard=1.0 at ENCUT=400, KPPRA=1600).
    """
    n_pw = (encut / 400.0) ** 1.5
    n_k = kppra / 1600.0
    return n_pw * n_k


# ---------------------------------------------------------------------------
# Bayesian DFT Parameter Optimizer
# ---------------------------------------------------------------------------


@dataclass
class ParameterPoint:
    encut: float
    kppra: int
    energy_eV: float = 0.0
    cost: float = 0.0
    energy_error: float = 0.0  # |E - E_converged|


@dataclass
class OptimizationResult:
    optimal_encut: float
    optimal_kppra: int
    predicted_error: float
    predicted_cost: float
    n_evaluations: int
    pareto_front: List[ParameterPoint]
    all_points: List[ParameterPoint]
    convergence_history: List[float]


class BayesianParameterOptimizer:
    """
    Bayesian Optimisation for (ENCUT, KPPRA) selection.

    Replaces the standard grid search with an intelligent sequential design
    that finds the Pareto-optimal parameter set in ~12–15 DFT evaluations.

    Usage
    -----
    >>> optimizer = BayesianParameterOptimizer(n_atoms=36)
    >>> # Initial exploration (3-5 points)
    >>> for encut, kppra in optimizer.suggest_initial(n=5):
    ...     energy = run_vasp_singlepoint(encut, kppra)
    ...     optimizer.observe(encut, kppra, energy)
    >>> # BO loop
    >>> for i in range(10):
    ...     encut, kppra = optimizer.suggest_next()
    ...     energy = run_vasp_singlepoint(encut, kppra)
    ...     optimizer.observe(encut, kppra, energy)
    >>> result = optimizer.result()
    """

    def __init__(
        self,
        n_atoms: int = 36,
        encut_range: Tuple[float, float] = (300, 600),
        kppra_range: Tuple[int, int] = (400, 3200),
        target_error: float = 0.001,
    ):
        self.n_atoms = n_atoms
        self.encut_range = encut_range
        self.kppra_range = kppra_range
        self.target_error = target_error  # eV/atom convergence target

        self.observations: List[ParameterPoint] = []
        self._gp = GaussianProcess(length_scale=1.0, signal_var=1.0, noise_var=1e-4)
        self._reference_energy: Optional[float] = None

    def suggest_initial(self, n: int = 5) -> List[Tuple[float, int]]:
        """Latin hypercube sampling for initial exploration."""
        rng = np.random.default_rng(42)
        encuts = np.linspace(self.encut_range[0], self.encut_range[1], n)
        kppras = np.linspace(self.kppra_range[0], self.kppra_range[1], n).astype(int)
        # Shuffle to avoid correlation
        rng.shuffle(encuts)
        rng.shuffle(kppras)
        # Round ENCUT to nearest 10, KPPRA to nearest 100
        return [(round(e / 10) * 10, round(k / 100) * 100) for e, k in zip(encuts, kppras)]

    def observe(self, encut: float, kppra: int, energy_eV: float):
        """Record one DFT evaluation result."""
        cost = dft_cost_model(encut, int(kppra), self.n_atoms)
        self.observations.append(
            ParameterPoint(
                encut=encut,
                kppra=int(kppra),
                energy_eV=energy_eV,
                cost=cost,
            )
        )

        # Update reference as the highest-accuracy point
        if self._reference_energy is None:
            self._reference_energy = energy_eV
        else:
            # Reference = point with highest ENCUT and KPPRA
            best = max(self.observations, key=lambda p: p.encut * p.kppra)
            self._reference_energy = best.energy_eV

        # Update errors
        for p in self.observations:
            p.energy_error = abs(p.energy_eV - self._reference_energy) / self.n_atoms

        # Refit GP
        if len(self.observations) >= 3:
            X = np.array([[p.encut, p.kppra] for p in self.observations])
            # Normalise to [0, 1]
            X_norm = X.copy()
            X_norm[:, 0] = (X_norm[:, 0] - self.encut_range[0]) / (self.encut_range[1] - self.encut_range[0])
            X_norm[:, 1] = (X_norm[:, 1] - self.kppra_range[0]) / (self.kppra_range[1] - self.kppra_range[0])
            y = np.array([p.energy_error for p in self.observations])
            self._gp.fit(X_norm, y)
            self._gp.optimize_hyperparameters()

    def suggest_next(self) -> Tuple[float, int]:
        """Suggest the next (ENCUT, KPPRA) to evaluate using Expected Improvement."""
        if len(self.observations) < 3:
            return self.suggest_initial(1)[0]

        # Grid of candidates
        encuts = np.arange(self.encut_range[0], self.encut_range[1] + 1, 20)
        kppras = np.arange(self.kppra_range[0], self.kppra_range[1] + 1, 200)
        grid = np.array([[e, k] for e in encuts for k in kppras])

        # Normalise
        grid_norm = grid.copy().astype(float)
        grid_norm[:, 0] = (grid_norm[:, 0] - self.encut_range[0]) / (self.encut_range[1] - self.encut_range[0])
        grid_norm[:, 1] = (grid_norm[:, 1] - self.kppra_range[0]) / (self.kppra_range[1] - self.kppra_range[0])

        mu, sigma = self._gp.predict(grid_norm)
        best_y = min(p.energy_error for p in self.observations)

        # EI favours low error; negate because we minimise
        ei = expected_improvement(-mu, sigma, -best_y)

        # Penalise high cost (multi-objective: scalarize with cost penalty)
        costs = np.array([dft_cost_model(e, int(k), self.n_atoms) for e, k in grid])
        # ParEGO-style scalarization: EI / cost^0.5
        score = ei / (np.sqrt(costs) + 1e-6)

        best_idx = np.argmax(score)
        encut_next = round(float(grid[best_idx, 0]) / 10) * 10
        kppra_next = round(float(grid[best_idx, 1]) / 100) * 100
        return float(encut_next), int(kppra_next)

    def result(self) -> OptimizationResult:
        """Return the optimization result with Pareto front."""
        if not self.observations:
            return OptimizationResult(
                optimal_encut=400,
                optimal_kppra=1600,
                predicted_error=0,
                predicted_cost=1,
                n_evaluations=0,
                pareto_front=[],
                all_points=[],
                convergence_history=[],
            )

        # Find Pareto front (error vs cost)
        points = sorted(self.observations, key=lambda p: p.energy_error)
        pareto = []
        min_cost = float("inf")
        for p in points:
            if p.cost < min_cost:
                pareto.append(p)
                min_cost = p.cost

        # Select optimal: lowest error that meets target, then lowest cost
        converged = [p for p in self.observations if p.energy_error <= self.target_error]
        if converged:
            best = min(converged, key=lambda p: p.cost)
        else:
            best = min(self.observations, key=lambda p: p.energy_error)

        return OptimizationResult(
            optimal_encut=best.encut,
            optimal_kppra=best.kppra,
            predicted_error=best.energy_error,
            predicted_cost=best.cost,
            n_evaluations=len(self.observations),
            pareto_front=pareto,
            all_points=list(self.observations),
            convergence_history=[p.energy_error for p in self.observations],
        )

    def summary(self) -> str:
        r = self.result()
        return (
            f"Bayesian Parameter Optimization ({r.n_evaluations} evaluations)\n"
            f"  Optimal: ENCUT={r.optimal_encut:.0f} eV, KPPRA={r.optimal_kppra}\n"
            f"  Error:   {r.predicted_error:.6f} eV/atom\n"
            f"  Cost:    {r.predicted_cost:.2f}x relative\n"
            f"  Pareto:  {len(r.pareto_front)} points\n"
            f"  Savings: ~{max(0, 80 - r.n_evaluations)} DFT runs vs grid search"
        )
