"""
Applicability Domain — Out-of-Distribution Detection for QSAR Models
=====================================================================

A QSAR model's predictions are only reliable within its applicability domain
(AD) — the chemical space covered by its training data. Beyond the AD,
predictions are extrapolations with unknown error.

Three complementary AD methods:

1. **Tanimoto distance** — nearest-neighbour distance in fingerprint space.
   If a query molecule's max Tanimoto similarity to the training set is below
   a threshold, it's out of domain.
   Reference: Sahigara et al., Molecules 17, 4791 (2012)

2. **Mahalanobis distance** — distance from the training distribution centroid
   in descriptor space, accounting for feature correlations.
   Molecules beyond k*std_dev are flagged as OOD.

3. **Ensemble disagreement** — variance among predictions from an ensemble
   of models. High disagreement = high epistemic uncertainty = OOD.
   Reference: Scalia et al., J. Chem. Inf. Model. 60, 2697 (2020)

These methods address the interview question: "what happens when someone
submits a molecule outside your training distribution?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)


@dataclass
class ADResult:
    """Applicability domain assessment for a single molecule."""
    smiles: str
    in_domain: bool
    confidence: float             # 0.0 = definitely OOD, 1.0 = definitely in-domain
    tanimoto_score: float         # max similarity to training set
    mahalanobis_distance: float   # distance from training centroid
    ensemble_std: float           # prediction disagreement
    details: Dict[str, float] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


# ---------------------------------------------------------------------------
# 1. Tanimoto Distance AD
# ---------------------------------------------------------------------------

class TanimotoAD:
    """
    Applicability domain based on Tanimoto similarity to training set.

    A molecule is in-domain if its maximum Tanimoto similarity to any
    training molecule exceeds a threshold.

    Typical thresholds:
    - Conservative: 0.5 (high confidence, narrow domain)
    - Moderate: 0.3 (balanced)
    - Permissive: 0.2 (wider domain, more risk)
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.train_fps: Optional[np.ndarray] = None

    def fit(self, fingerprints: np.ndarray):
        """Store training set fingerprints."""
        self.train_fps = fingerprints.astype(np.float32)

    def score(self, query_fp: np.ndarray) -> Tuple[float, bool]:
        """
        Compute max Tanimoto similarity to training set.

        Returns (max_similarity, is_in_domain).
        """
        if self.train_fps is None:
            return 0.0, False

        query = query_fp.astype(np.float32).reshape(1, -1)

        # Tanimoto = intersection / union for binary fingerprints
        # For count vectors, use generalised Tanimoto (Jaccard)
        intersection = np.minimum(query, self.train_fps).sum(axis=1)
        union = np.maximum(query, self.train_fps).sum(axis=1)
        tanimoto = intersection / np.maximum(union, 1e-9)

        max_sim = float(tanimoto.max())
        return max_sim, max_sim >= self.threshold

    def batch_score(self, fingerprints: np.ndarray) -> List[Tuple[float, bool]]:
        """Score a batch of molecules."""
        return [self.score(fp) for fp in fingerprints]


# ---------------------------------------------------------------------------
# 2. Mahalanobis Distance AD
# ---------------------------------------------------------------------------

class MahalanobisAD:
    """
    Applicability domain based on Mahalanobis distance.

    Measures how far a query molecule is from the training distribution
    centroid, accounting for feature correlations via the covariance matrix.

    Threshold is typically set at the 95th or 99th percentile of
    training set distances.
    """

    def __init__(self, percentile_threshold: float = 95.0):
        self.percentile = percentile_threshold
        self.mean: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.threshold: float = 0.0

    def fit(self, descriptors: np.ndarray):
        """Fit the Mahalanobis distance model to training descriptors."""
        X = descriptors.astype(np.float64)
        self.mean = X.mean(axis=0)

        # Regularised covariance inverse (add small diagonal for stability)
        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            self.cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.cov_inv = np.linalg.pinv(cov)

        # Set threshold from training distribution
        train_dists = np.array([self._distance(x) for x in X])
        self.threshold = float(np.percentile(train_dists, self.percentile))

    def _distance(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance for a single sample."""
        diff = x - self.mean
        return float(np.sqrt(diff @ self.cov_inv @ diff))

    def score(self, descriptor: np.ndarray) -> Tuple[float, bool]:
        """
        Compute Mahalanobis distance and check if in-domain.

        Returns (distance, is_in_domain).
        """
        if self.mean is None:
            return 0.0, False
        d = self._distance(descriptor.astype(np.float64))
        return d, d <= self.threshold

    def batch_score(self, descriptors: np.ndarray) -> List[Tuple[float, bool]]:
        return [self.score(d) for d in descriptors]


# ---------------------------------------------------------------------------
# 3. Ensemble Disagreement AD
# ---------------------------------------------------------------------------

class EnsembleAD:
    """
    Applicability domain based on ensemble prediction disagreement.

    Given predictions from N models, the standard deviation measures
    epistemic uncertainty. High std = the models disagree = likely OOD.

    This is the most model-aware AD method — it captures uncertainty
    that Tanimoto/Mahalanobis might miss (e.g., molecules that are
    structurally similar but in a poorly-sampled property region).
    """

    def __init__(self, std_threshold: float = 0.2):
        self.std_threshold = std_threshold

    def score(
        self,
        predictions: np.ndarray,  # shape (n_models,) or (n_models, n_tasks)
    ) -> Tuple[float, bool]:
        """
        Compute ensemble disagreement.

        Parameters
        ----------
        predictions : np.ndarray
            Predictions from multiple models for a single molecule.

        Returns (std, is_in_domain).
        """
        std = float(np.std(predictions))
        return std, std <= self.std_threshold

    def batch_score(
        self,
        predictions: np.ndarray,  # shape (n_molecules, n_models)
    ) -> List[Tuple[float, bool]]:
        """Score a batch."""
        return [self.score(predictions[i]) for i in range(len(predictions))]


# ---------------------------------------------------------------------------
# Combined AD Assessor
# ---------------------------------------------------------------------------

class ApplicabilityDomainAssessor:
    """
    Combined applicability domain assessment using all three methods.

    A molecule is classified as in-domain only if it passes all three
    checks (conservative) or a majority vote (moderate).

    Usage
    -----
        ad = ApplicabilityDomainAssessor()
        ad.fit(train_fingerprints, train_descriptors)

        result = ad.assess(
            smiles="CCO",
            fingerprint=fp_array,
            descriptor=desc_array,
            ensemble_predictions=np.array([0.8, 0.7, 0.9, 0.75]),
        )
        print(result.in_domain, result.confidence)
    """

    def __init__(
        self,
        tanimoto_threshold: float = 0.3,
        mahalanobis_percentile: float = 95.0,
        ensemble_std_threshold: float = 0.2,
        voting: str = "majority",   # "majority" or "all"
    ):
        self.tanimoto = TanimotoAD(threshold=tanimoto_threshold)
        self.mahalanobis = MahalanobisAD(percentile_threshold=mahalanobis_percentile)
        self.ensemble = EnsembleAD(std_threshold=ensemble_std_threshold)
        self.voting = voting

    def fit(
        self,
        fingerprints: np.ndarray,
        descriptors: np.ndarray,
    ):
        """Fit AD models on training data."""
        self.tanimoto.fit(fingerprints)
        self.mahalanobis.fit(descriptors)

    def assess(
        self,
        smiles: str,
        fingerprint: np.ndarray,
        descriptor: np.ndarray,
        ensemble_predictions: Optional[np.ndarray] = None,
    ) -> ADResult:
        """
        Full applicability domain assessment.

        Returns ADResult with per-method scores and overall verdict.
        """
        tan_score, tan_in = self.tanimoto.score(fingerprint)
        mah_dist, mah_in = self.mahalanobis.score(descriptor)

        ens_std = 0.0
        ens_in = True
        if ensemble_predictions is not None:
            ens_std, ens_in = self.ensemble.score(ensemble_predictions)

        # Voting
        votes = [tan_in, mah_in, ens_in]
        if self.voting == "all":
            in_domain = all(votes)
        else:
            in_domain = sum(votes) >= 2

        # Confidence: weighted combination
        # Tanimoto: higher = more confident
        # Mahalanobis: lower = more confident (invert)
        # Ensemble: lower std = more confident
        conf_tan = min(tan_score / max(self.tanimoto.threshold, 1e-9), 2.0) / 2.0
        conf_mah = max(1.0 - mah_dist / max(self.mahalanobis.threshold * 2, 1e-9), 0.0)
        conf_ens = max(1.0 - ens_std / max(self.ensemble.std_threshold * 2, 1e-9), 0.0) if ensemble_predictions is not None else 0.5

        confidence = (conf_tan + conf_mah + conf_ens) / 3.0

        return ADResult(
            smiles=smiles,
            in_domain=in_domain,
            confidence=confidence,
            tanimoto_score=tan_score,
            mahalanobis_distance=mah_dist,
            ensemble_std=ens_std,
            details={
                "tanimoto_in_domain": float(tan_in),
                "mahalanobis_in_domain": float(mah_in),
                "ensemble_in_domain": float(ens_in),
                "tanimoto_threshold": self.tanimoto.threshold,
                "mahalanobis_threshold": self.mahalanobis.threshold,
                "ensemble_std_threshold": self.ensemble.std_threshold,
            },
        )

    def batch_assess(
        self,
        smiles_list: List[str],
        fingerprints: np.ndarray,
        descriptors: np.ndarray,
        ensemble_predictions: Optional[np.ndarray] = None,
    ) -> List[ADResult]:
        """Assess a batch of molecules."""
        results = []
        for i in range(len(smiles_list)):
            ens = ensemble_predictions[i] if ensemble_predictions is not None else None
            results.append(self.assess(
                smiles_list[i], fingerprints[i], descriptors[i], ens,
            ))
        return results
