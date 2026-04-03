"""
SCF Convergence Trajectory Analysis
=====================================
Scientific motivation
---------------------
The self-consistent-field (SCF) loop in DFT is an iterative fixed-point
solver for the Kohn-Sham equations.  The convergence trajectory — the
sequence of total-energy differences {ΔE_n} across SCF iterations — is
a *time series* that encodes the quality of the initial guess, the
dielectric response of the material, and numerical properties of the chosen
mixing algorithm.

By treating convergence as a dynamical system, we can:
  1. **Detect charge sloshing** in O(n) via spectral analysis before
     it wastes hundreds of SCF iterations.
  2. **Predict the step at which convergence will be reached** from the
     first M iterations — enabling smart early stopping and preemptive
     parameter adjustment.
  3. **Classify the convergence regime** and map it to the optimal VASP
     ALGO/mixing settings (Damped, Fast, All, etc.).

Physical background
--------------------
SCF mixing: the updated charge density ρ_{n+1} is a linear combination of
ρ_n and the output density ρ_out(ρ_n):
    ρ_{n+1} = (1-β) ρ_n + β ρ_out(ρ_n)

The residual error evolves as:
    ε_n ~ ρ^n               (simple mixing, spectral radius ρ < 1)

For metallic systems the Lindhard dielectric matrix has long-wavelength
singular modes → large ρ → slow/oscillatory convergence (charge sloshing).

Pulay/DIIS mixing exploits the history of {ρ_n} to extrapolate toward the
fixed point, achieving super-linear convergence when well-conditioned.

Key references
--------------
[1] Pulay, Chem. Phys. Lett. 73, 393 (1980)     — DIIS mixing
[2] Johnson, PRB 38, 12807 (1988)                — Broyden mixing
[3] Kresse & Furthmuller, CMS 6, 15 (1996)       — VASP SCF algorithms
[4] Anglade & Gonze, PRB 78, 045126 (2008)       — charge sloshing analysis
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from science.core.constants import (
    FFT_AC_RATIO_THRESHOLD, FFT_MIN_FREQ, FFT_MIN_STEPS,
    FFT_SIGN_CHANGE_THRESHOLD,
)
from science.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SCFTrajectory:
    """
    Parsed SCF iteration history from a VASP OUTCAR.

    Attributes
    ----------
    dE     : list of |ΔE| (eV) per SCF iteration
    rms_dV : list of RMS charge-density change (optional, VASP RMS(C))
    nelm   : max allowed SCF steps (NELM in INCAR)
    ediff  : convergence threshold (EDIFF in INCAR, eV)
    """
    dE:     List[float]
    rms_dV: Optional[List[float]] = None
    nelm:   int   = 60
    ediff:  float = 1e-5

    @classmethod
    def from_outcar_text(cls, outcar: str,
                         nelm: int = 60,
                         ediff: float = 1e-5) -> "SCFTrajectory":
        """
        Parse a VASP OUTCAR string.

        Extracts lines of the form:
            DAV:   1    -1.234567E+02    ...   0.12345E-01
        and collects the absolute energy change in column 3.
        """
        import re
        dE_list, rms_list = [], []
        # VASP DAV line: "  DAV:   N    E_total   dE   RMS"
        dav_re  = re.compile(
            r"DAV:\s+\d+\s+[-\d.E+]+\s+([-\d.E+]+)\s+([-\d.E+]+)"
        )
        # Alternate: "RMM-DIIS" convergence lines
        rmm_re  = re.compile(
            r"RMM:\s+\d+\s+[-\d.E+]+\s+([-\d.E+]+)\s+([-\d.E+]+)"
        )
        for line in outcar.splitlines():
            m = dav_re.search(line) or rmm_re.search(line)
            if m:
                try:
                    dE_list.append(abs(float(m.group(1))))
                    rms_list.append(abs(float(m.group(2))))
                except ValueError:
                    pass
        return cls(dE=dE_list, rms_dV=rms_list or None,
                   nelm=nelm, ediff=ediff)

    def is_converged(self) -> bool:
        return bool(self.dE) and self.dE[-1] < self.ediff

    def hit_nelm(self) -> bool:
        return len(self.dE) >= self.nelm


# ---------------------------------------------------------------------------
# 1. Charge sloshing detector
# ---------------------------------------------------------------------------

@dataclass
class SloshingResult:
    is_sloshing: bool
    dominant_frequency: float      # cycles per SCF step
    amplitude: float               # peak FFT magnitude
    decay_rate: float              # γ in e^{-γn} envelope; >0 = converging
    confidence: float              # 0–1
    remedy: str


class ChargeSloshingDetector:
    """
    Detect oscillatory divergence (charge sloshing) in the SCF trajectory.

    Algorithm
    ---------
    1. Apply a Hanning window to reduce spectral leakage.
    2. Compute the one-sided FFT of log(|ΔE_n|).
    3. Find the dominant non-DC frequency.
    4. A sloshing signature requires:
       (a) dominant non-DC frequency amplitude > dc_ratio * DC_amplitude, and
       (b) the frequency is in (f_low, 0.5) cycles/step.
    5. Estimate the oscillation envelope decay γ via linear regression on
       |ΔE_n| in log space.
    """

    def __init__(
        self,
        dc_ratio: float = FFT_AC_RATIO_THRESHOLD,
        f_low: float = FFT_MIN_FREQ,
        min_steps: int = FFT_MIN_STEPS,
    ):
        self.dc_ratio  = dc_ratio
        self.f_low     = f_low
        self.min_steps = min_steps

    def detect(self, traj: SCFTrajectory) -> SloshingResult:
        dE = np.array(traj.dE, dtype=np.float64)
        n  = len(dE)

        if n < self.min_steps:
            return SloshingResult(
                is_sloshing=False, dominant_frequency=0.0,
                amplitude=0.0, decay_rate=0.0,
                confidence=0.0, remedy="Not enough SCF steps to analyse."
            )

        # Log-transform (avoid log(0))
        log_dE = np.log(np.clip(dE, 1e-20, None))

        # Detrend: remove the linear trend (exponential decay in log space)
        # so that we're looking at oscillations *around* the decay envelope
        steps = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(steps, log_dE, 1)
        detrended = log_dE - (intercept + slope * steps)

        # Hanning window on detrended signal (NOT mean-subtracted raw)
        window = np.hanning(n)
        windowed = detrended * window

        # FFT of detrended signal
        fft_vals = np.fft.rfft(windowed)
        fft_mag  = np.abs(fft_vals)
        freqs    = np.fft.rfftfreq(n)

        # Total signal power (L2 norm of detrended)
        total_power = float(np.sum(detrended**2)) + 1e-20
        ac_mask  = freqs > self.f_low
        if not np.any(ac_mask):
            return SloshingResult(
                is_sloshing=False, dominant_frequency=0.0,
                amplitude=0.0, decay_rate=0.0,
                confidence=0.0, remedy="No AC component detected."
            )

        dom_ac_idx = np.argmax(fft_mag[ac_mask])
        dom_freq   = float(freqs[ac_mask][dom_ac_idx])
        dom_amp    = float(fft_mag[ac_mask][dom_ac_idx])

        # Oscillation ratio: compare dominant AC power to total detrended power
        # For pure exponential decay, detrended ≈ noise → low ratio
        # For sloshing, detrended has strong periodic component → high ratio
        ac_power = float(np.sum(fft_mag[ac_mask]**2))
        total_fft_power = float(np.sum(fft_mag**2)) + 1e-20
        ratio = ac_power / total_fft_power

        # Also check sign-change frequency in raw dE differences
        # True sloshing has regular sign changes; monotone decay does not
        dE_diff = np.diff(dE)
        sign_changes = np.sum(np.diff(np.sign(dE_diff)) != 0)
        sign_change_rate = sign_changes / max(n - 2, 1)

        # Envelope decay via OLS on log|ΔE|
        decay_rate = -float(slope)   # positive → converging

        # Sloshing requires BOTH:
        # 1. Strong AC component relative to total (ratio > threshold)
        # 2. Frequent sign changes (oscillatory, not monotone)
        is_sloshing = (ratio > self.dc_ratio) and (sign_change_rate > FFT_SIGN_CHANGE_THRESHOLD)
        confidence  = float(np.clip(ratio * sign_change_rate * 4, 0, 1))

        remedy = self._recommend(is_sloshing, decay_rate, dom_freq)
        return SloshingResult(
            is_sloshing=is_sloshing,
            dominant_frequency=dom_freq,
            amplitude=dom_amp,
            decay_rate=decay_rate,
            confidence=confidence,
            remedy=remedy,
        )

    def _recommend(self, sloshing: bool, decay: float, freq: float) -> str:
        if not sloshing:
            return "No action needed — convergence appears healthy."
        if decay < 0:   # diverging
            if freq > 0.3:
                return (
                    "High-frequency sloshing detected (likely metallic surface "
                    "with strong dielectric response). Recommend: "
                    "ALGO=Damped, AMIX=0.1, BMIX=0.01, AMIX_MAG=0.2."
                )
            else:
                return (
                    "Low-frequency charge oscillation (possible near-gap system). "
                    "Recommend: reduce AMIX to 0.1–0.2, increase NELM, "
                    "or switch to ALGO=All."
                )
        else:   # oscillating but converging
            return (
                "Mild sloshing — converging slowly. "
                "Consider AMIX=0.2, BMIX=0.01, ALGO=Fast for speedup."
            )


# ---------------------------------------------------------------------------
# 2. Convergence rate predictor
# ---------------------------------------------------------------------------

@dataclass
class ConvergencePrediction:
    predicted_step: int            # estimated SCF step to reach EDIFF
    convergence_rate: float        # λ in exp(-λ·n), steps^{-1}
    r_squared: float               # goodness of exponential fit
    will_converge: bool            # False if predicted > NELM
    confidence: str                # 'high' | 'medium' | 'low'


class ConvergenceRatePredictor:
    """
    Predict when (or if) SCF will converge from the early trajectory.

    Model
    -----
    Assume exponential convergence in log space:
        log|ΔE_n| ≈ log(A) - λ·n

    Fit via OLS on the first `window` steps.
    Extrapolate: n_conv = (log(A) - log(EDIFF)) / λ.

    If λ ≤ 0 (diverging) or n_conv > NELM, flag as non-convergent.

    Reliability
    -----------
    R² < 0.7   → 'low' confidence  (non-monotone early history)
    R² < 0.9   → 'medium' confidence
    R² ≥ 0.9   → 'high' confidence
    """

    def __init__(self, window: int = 10, min_window: int = 4):
        self.window     = window
        self.min_window = min_window

    def predict(self, traj: SCFTrajectory) -> ConvergencePrediction:
        dE   = np.array(traj.dE, dtype=np.float64)
        n_obs = min(len(dE), self.window)

        if n_obs < self.min_window:
            return ConvergencePrediction(
                predicted_step=-1, convergence_rate=0.0,
                r_squared=0.0, will_converge=False, confidence='low'
            )

        log_dE = np.log(np.clip(dE[:n_obs], 1e-20, None))
        steps  = np.arange(n_obs, dtype=float)

        # OLS: log_dE = a + b * steps
        slope, intercept = np.polyfit(steps, log_dE, 1)
        lam   = -slope
        log_A = intercept

        # R² of fit
        pred  = intercept + slope * steps
        ss_res = np.sum((log_dE - pred)**2)
        ss_tot = np.sum((log_dE - log_dE.mean())**2) + 1e-20
        r2    = float(1.0 - ss_res / ss_tot)

        if lam <= 0.0 or not np.isfinite(lam):
            return ConvergencePrediction(
                predicted_step=-1, convergence_rate=float(lam),
                r_squared=max(r2, 0.0), will_converge=False, confidence='low'
            )

        log_target  = np.log(traj.ediff)
        n_conv      = int(np.ceil((log_A - log_target) / lam))
        will_conv   = 0 < n_conv <= traj.nelm

        confidence = ('high' if r2 >= 0.9 else
                      'medium' if r2 >= 0.7 else 'low')

        return ConvergencePrediction(
            predicted_step=n_conv,
            convergence_rate=float(lam),
            r_squared=max(r2, 0.0),
            will_converge=will_conv,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# 3. Algorithm recommender
# ---------------------------------------------------------------------------

_ALGO_RULES: List[Tuple[str, str]] = [
    # (condition_description, VASP settings recommendation)
    ("metal_fast",
     "ALGO=Fast; ISMEAR=1; SIGMA=0.2; AMIX=0.4; BMIX=1.0"),
    ("metal_sloshing",
     "ALGO=Damped; AMIX=0.1; BMIX=0.01; AMIX_MAG=0.2; BMIX_MAG=0.001"),
    ("insulator_slow",
     "ALGO=All; ISMEAR=0; SIGMA=0.05; AMIX=0.2; NELM=100"),
    ("insulator_fast",
     "ALGO=Fast; ISMEAR=0; SIGMA=0.01; AMIX=0.3"),
    ("oxide_correlated",
     "ALGO=All; LDAU=True; LDAUTYPE=2; AMIX=0.2; BMIX=0.0001"),
]


@dataclass
class AlgoRecommendation:
    algo:      str
    settings:  str
    rationale: str


class AlgorithmRecommender:
    """
    Map a convergence signature (sloshing + prediction + material hints)
    to optimal VASP ALGO/mixing settings.

    Input features
    --------------
    sloshing       : SloshingResult
    prediction     : ConvergencePrediction
    is_metal       : bool (from band gap or ISMEAR hint)
    has_d_electrons: bool (3d/4d/5d transition metal)
    """

    def recommend(
        self,
        sloshing:        SloshingResult,
        prediction:      ConvergencePrediction,
        is_metal:        bool = True,
        has_d_electrons: bool = False,
    ) -> AlgoRecommendation:
        # Priority: sloshing in metal → Damped
        if sloshing.is_sloshing and is_metal and sloshing.decay_rate < 0:
            return AlgoRecommendation(
                algo="Damped",
                settings="ALGO=Damped; AMIX=0.1; BMIX=0.01; AMIX_MAG=0.2",
                rationale=(
                    f"Oscillatory divergence detected (f={sloshing.dominant_frequency:.3f} "
                    f"cyc/step, confidence={sloshing.confidence:.2f}). "
                    "Damped algorithm suppresses long-wavelength charge oscillations."
                ),
            )
        # Slow but monotone insulator with d-electrons → LDAU + All
        if not is_metal and has_d_electrons and not prediction.will_converge:
            return AlgoRecommendation(
                algo="All",
                settings="ALGO=All; LDAU=True; LDAUTYPE=2; AMIX=0.2; BMIX=0.0001",
                rationale=(
                    "Non-convergent insulator with d-electrons: DFT+U is likely "
                    "needed. All (blocked Davidson) is more robust than RMM-DIIS "
                    "for correlated systems."
                ),
            )
        # Fast converging metal
        if is_metal and prediction.will_converge and prediction.convergence_rate > 0.3:
            return AlgoRecommendation(
                algo="Fast",
                settings="ALGO=Fast; ISMEAR=1; SIGMA=0.2",
                rationale=(
                    f"Rapid convergence predicted (λ={prediction.convergence_rate:.3f}, "
                    f"n_conv≈{prediction.predicted_step}). "
                    "RMM-DIIS (Fast) is efficient."
                ),
            )
        # Default
        return AlgoRecommendation(
            algo="All",
            settings="ALGO=All; AMIX=0.2; NELM=80",
            rationale=(
                "Conservative default: blocked Davidson is robust for "
                "ambiguous convergence signature."
            ),
        )


# ---------------------------------------------------------------------------
# Convenience: full analysis pipeline
# ---------------------------------------------------------------------------

@dataclass
class SCFAnalysisReport:
    sloshing:    SloshingResult
    prediction:  ConvergencePrediction
    algo:        AlgoRecommendation
    summary:     str

    def __str__(self) -> str:
        return self.summary


def analyse_scf(
    traj: SCFTrajectory,
    is_metal: bool = True,
    has_d_electrons: bool = False,
) -> SCFAnalysisReport:
    """
    Run the full SCF analysis pipeline.

    Returns
    -------
    SCFAnalysisReport with sloshing detection, convergence prediction,
    and algorithm recommendation.
    """
    sloshing   = ChargeSloshingDetector().detect(traj)
    prediction = ConvergenceRatePredictor().predict(traj)
    algo       = AlgorithmRecommender().recommend(
        sloshing, prediction, is_metal, has_d_electrons
    )

    n = len(traj.dE)
    last_dE = traj.dE[-1] if traj.dE else float("nan")
    conv_str = "CONVERGED" if traj.is_converged() else (
        "HIT NELM (not converged)" if traj.hit_nelm() else "IN PROGRESS"
    )

    summary = (
        f"SCF Analysis ({n} steps, status: {conv_str})\n"
        f"  Last |ΔE|         : {last_dE:.2e} eV  (target: {traj.ediff:.0e})\n"
        f"  Sloshing          : {'YES' if sloshing.is_sloshing else 'no'}"
        + (f" (f={sloshing.dominant_frequency:.3f} cyc/step, "
           f"decay={sloshing.decay_rate:+.3f})" if sloshing.is_sloshing else "") + "\n"
        f"  Convergence rate  : λ = {prediction.convergence_rate:.3f} steps⁻¹"
        f"  (predicted n_conv = {prediction.predicted_step}, "
        f"R²={prediction.r_squared:.2f}, {prediction.confidence} confidence)\n"
        f"  Recommendation    : {algo.settings}\n"
        f"  Rationale         : {algo.rationale}"
    )
    return SCFAnalysisReport(
        sloshing=sloshing, prediction=prediction,
        algo=algo, summary=summary,
    )


# ---------------------------------------------------------------------------
# Multi-ionic-step trajectory (track convergence across geometry steps)
# ---------------------------------------------------------------------------

class IonicConvergenceTracker:
    """
    Track SCF convergence quality across ionic relaxation steps.

    At each ionic step, the SCF loop is re-run from the updated density.
    Plotting 'SCF steps to converge' vs ionic step reveals:
     - Sudden spikes: geometry is in a difficult region (saddle, distorted)
     - Monotone decrease: approaching a minimum efficiently
     - Plateau then spike: near a soft mode / flat landscape

    This is useful for:
     - Early termination of NEB images that are stuck
     - Detecting when POTIM (ionic step size) is too large (spike pattern)
     - Identifying when IBRION switch (CG → BFGS) would help
    """

    def __init__(self):
        self._ionic_steps: List[SCFTrajectory] = []

    def add_ionic_step(self, traj: SCFTrajectory):
        self._ionic_steps.append(traj)

    def scf_step_counts(self) -> List[int]:
        """Number of SCF iterations per ionic step."""
        return [len(t.dE) for t in self._ionic_steps]

    def convergence_quality_series(self) -> np.ndarray:
        """
        Per-ionic-step convergence quality score ∈ [0, 1]:
          1.0 = converged quickly (≤ 10 SCF steps)
          0.5 = converged, many steps
          0.0 = did not converge
        """
        scores = []
        for t in self._ionic_steps:
            if t.is_converged():
                scores.append(max(0.0, 1.0 - len(t.dE) / (2 * t.nelm)))
            else:
                scores.append(0.0)
        return np.array(scores)

    def detect_potim_too_large(self) -> bool:
        """
        Heuristic: if SCF step count oscillates with period ≤ 3 and
        average > 25, ionic step size is likely too large.
        """
        counts = np.array(self.scf_step_counts(), dtype=float)
        if len(counts) < 6:
            return False
        diffs = np.abs(np.diff(counts))
        return bool(np.mean(diffs) > 10 and np.mean(counts) > 25)

    def report(self) -> str:
        counts = self.scf_step_counts()
        quality = self.convergence_quality_series()
        n_ionic = len(counts)
        n_conv  = sum(t.is_converged() for t in self._ionic_steps)
        return (
            f"Ionic relaxation: {n_ionic} steps, "
            f"{n_conv}/{n_ionic} SCF-converged\n"
            f"  Mean SCF steps/ionic step : {np.mean(counts):.1f}\n"
            f"  Mean convergence quality  : {np.mean(quality):.3f}\n"
            f"  POTIM too large?          : {self.detect_potim_too_large()}"
        )


# ---------------------------------------------------------------------------
# Threshold validation via cross-validation
# ---------------------------------------------------------------------------

@dataclass
class ThresholdValidationResult:
    """Result of sloshing threshold optimisation."""
    best_ac_ratio: float
    best_sign_change: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_healthy: int
    n_sloshing: int
    roc_curve: list  # list of (fpr, tpr, threshold) tuples


def _generate_validation_trajectories(
    n_healthy: int = 50,
    n_sloshing: int = 50,
    seed: int = 42,
) -> tuple:
    """
    Generate diverse synthetic SCF trajectories for threshold validation.

    Healthy trajectories: exponential decay with varying rates and noise.
    Sloshing trajectories: oscillatory with varying frequencies and decay.

    Returns (trajectories, labels) where label=1 means sloshing.
    """
    rng = np.random.default_rng(seed)
    trajectories = []
    labels = []

    for _ in range(n_healthy):
        n = rng.integers(15, 60)
        A = rng.uniform(0.1, 2.0)
        lam = rng.uniform(0.1, 0.5)
        noise = rng.uniform(0.0005, 0.005)
        t = np.arange(n)
        dE = A * np.exp(-lam * t) + rng.normal(0, noise, n)
        trajectories.append(np.abs(dE).tolist())
        labels.append(0)

    for _ in range(n_sloshing):
        n = rng.integers(15, 60)
        A = rng.uniform(0.005, 0.1)
        decay = rng.uniform(-0.01, 0.05)
        freq = rng.uniform(0.15, 0.45)
        t = np.arange(n)
        oscillation = A * np.exp(-decay * t) * (
            0.3 + np.abs(np.sin(2 * np.pi * freq * t))
        )
        noise = rng.uniform(0.0001, 0.001)
        dE = oscillation + rng.normal(0, noise, n)
        trajectories.append(np.abs(dE).tolist())
        labels.append(1)

    return trajectories, np.array(labels)


def validate_thresholds(
    n_healthy: int = 50,
    n_sloshing: int = 50,
    seed: int = 42,
    n_grid: int = 20,
) -> ThresholdValidationResult:
    """
    Validate and optimise sloshing detection thresholds via grid search.

    Sweeps over (ac_ratio_threshold, sign_change_threshold) on synthetic
    trajectories with known labels, selecting the pair that maximises F1.

    This provides empirical justification for the default thresholds
    rather than relying on arbitrary values.

    Parameters
    ----------
    n_healthy : int
        Number of synthetic healthy trajectories.
    n_sloshing : int
        Number of synthetic sloshing trajectories.
    seed : int
        Random seed for reproducibility.
    n_grid : int
        Grid resolution per dimension.

    Returns
    -------
    ThresholdValidationResult
        Optimal thresholds and classification metrics.
    """
    trajectories, labels = _generate_validation_trajectories(n_healthy, n_sloshing, seed)

    # Grid of threshold candidates
    ac_ratios = np.linspace(0.1, 0.6, n_grid)
    sign_changes = np.linspace(0.1, 0.6, n_grid)

    best_f1 = -1.0
    best_ac = 0.3
    best_sc = 0.3
    best_metrics = {}

    for ac_thresh in ac_ratios:
        for sc_thresh in sign_changes:
            detector = ChargeSloshingDetector(dc_ratio=ac_thresh)
            preds = []
            for dE_list in trajectories:
                traj = SCFTrajectory(dE=dE_list, ediff=1e-5, nelm=60)
                result = detector.detect(traj)
                # Override sign-change threshold for this test
                # Recompute sloshing decision with current thresholds
                dE = np.array(dE_list)
                dE_diff = np.diff(dE)
                sign_chg = np.sum(np.diff(np.sign(dE_diff)) != 0)
                sign_chg_rate = sign_chg / max(len(dE) - 2, 1)
                is_slosh = result.is_sloshing if sc_thresh == 0.3 else (
                    (result.confidence > 0) and (sign_chg_rate > sc_thresh)
                    and result.amplitude > 0
                )
                preds.append(int(is_slosh))

            preds = np.array(preds)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            tn = np.sum((preds == 0) & (labels == 0))

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            accuracy = (tp + tn) / len(labels)

            if f1 > best_f1:
                best_f1 = f1
                best_ac = float(ac_thresh)
                best_sc = float(sc_thresh)
                best_metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }

    # Build ROC curve at optimal sign_change threshold
    roc_curve = []
    for ac_thresh in np.linspace(0.05, 0.8, 30):
        detector = ChargeSloshingDetector(dc_ratio=ac_thresh)
        preds = []
        for dE_list in trajectories:
            traj = SCFTrajectory(dE=dE_list, ediff=1e-5, nelm=60)
            r = detector.detect(traj)
            preds.append(int(r.is_sloshing))
        preds = np.array(preds)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fpr = fp / max(fp + tn, 1)
        tpr = tp / max(tp + fn, 1)
        roc_curve.append((float(fpr), float(tpr), float(ac_thresh)))

    logger.info("Threshold validation complete",
                extra={"best_ac": best_ac, "best_sc": best_sc,
                       "f1": best_metrics["f1"]})

    return ThresholdValidationResult(
        best_ac_ratio=best_ac,
        best_sign_change=best_sc,
        accuracy=best_metrics["accuracy"],
        precision=best_metrics["precision"],
        recall=best_metrics["recall"],
        f1=best_metrics["f1"],
        n_healthy=n_healthy,
        n_sloshing=n_sloshing,
        roc_curve=roc_curve,
    )
