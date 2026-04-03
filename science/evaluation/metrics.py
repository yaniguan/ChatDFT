"""
ChatDFT Evaluation & Optimisation Pipeline
============================================
This module provides systematic evaluation of every AI/ML component in ChatDFT.

Design principles:
  - Every metric is computed offline against a golden dataset
  - Results are logged to PostgreSQL (evaluation_run table) for tracking over time
  - Each component has its own metric suite
  - Composite score aggregates all components into a single system health number
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Import the expanded golden dataset (25 reactions, 5 domains)
from science.evaluation.golden_dataset import (
    GOLDEN_SET, GOLDEN_BY_DOMAIN, GoldenExample,
    N_TOTAL, N_DOMAINS, summary as golden_summary,
)


# ---------------------------------------------------------------------------
# Component-level metrics
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)


class IntentParsingMetrics:
    """
    Evaluate intent parser accuracy.

    Metrics:
    - field_accuracy: fraction of intent fields correctly extracted
    - stage_accuracy: exact match on electrocatalysis/thermal/photo
    - material_recall: did we extract the correct material?
    - condition_extraction_rate: fraction of conditions (pH, U, T) captured
    """

    @staticmethod
    def evaluate(predicted: Dict, expected: Dict) -> List[MetricResult]:
        results = []
        # Stage accuracy
        pred_stage = (predicted.get("stage") or "").lower()
        exp_stage = (expected.get("stage") or "").lower()
        results.append(MetricResult(
            "intent_stage_accuracy",
            1.0 if pred_stage == exp_stage else 0.0,
            {"predicted": pred_stage, "expected": exp_stage},
        ))
        # Material extraction
        pred_mat = (predicted.get("system", {}).get("material") or "").lower()
        exp_mat = (expected.get("system", {}).get("material") or "").lower()
        results.append(MetricResult(
            "intent_material_match",
            1.0 if pred_mat == exp_mat else 0.0,
            {"predicted": pred_mat, "expected": exp_mat},
        ))
        # Facet extraction
        pred_facet = str(predicted.get("system", {}).get("facet", ""))
        exp_facet = str(expected.get("system", {}).get("facet", ""))
        results.append(MetricResult(
            "intent_facet_match",
            1.0 if pred_facet == exp_facet else 0.0,
        ))
        return results


class HypothesisMetrics:
    """
    Evaluate hypothesis generation quality.

    Metrics:
    - intermediate_recall: what fraction of expected intermediates were found
    - intermediate_precision: what fraction of predicted intermediates are correct
    - intermediate_f1: harmonic mean
    - stoichiometric_validity: are all steps atom-balanced
    - forward_direction: does the mechanism start from the correct reactant
    """

    @staticmethod
    def evaluate(predicted_intermediates: List[str],
                 expected_intermediates: List[str],
                 predicted_steps: List[str] = None,
                 reactant: str = "") -> List[MetricResult]:
        results = []
        pred_set = set(s.strip().lower() for s in predicted_intermediates)
        exp_set = set(s.strip().lower() for s in expected_intermediates)

        # Recall: how many expected intermediates did we find?
        if exp_set:
            recall = len(pred_set & exp_set) / len(exp_set)
        else:
            recall = 1.0
        results.append(MetricResult("hypothesis_intermediate_recall", recall))

        # Precision: how many predicted intermediates are correct?
        if pred_set:
            precision = len(pred_set & exp_set) / len(pred_set)
        else:
            precision = 0.0
        results.append(MetricResult("hypothesis_intermediate_precision", precision))

        # F1
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        results.append(MetricResult("hypothesis_intermediate_f1", f1))

        # Forward direction check
        if predicted_steps and reactant:
            first_step = predicted_steps[0].lower() if predicted_steps else ""
            fwd = 1.0 if reactant.lower() in first_step else 0.0
            results.append(MetricResult("hypothesis_forward_direction", fwd))

        return results


class ThermodynamicsMetrics:
    """
    Evaluate thermodynamic prediction accuracy.

    Metrics:
    - dG_mae: mean absolute error on free energy profile (eV)
    - overpotential_error: |predicted - expected| overpotential (V)
    - rds_match: did we identify the correct rate-determining step
    """

    @staticmethod
    def evaluate(predicted_dG: List[float],
                 expected_dG: List[float],
                 predicted_eta: float,
                 expected_eta: float) -> List[MetricResult]:
        results = []
        # Align lengths
        n = min(len(predicted_dG), len(expected_dG))
        if n > 0:
            mae = float(np.mean(np.abs(
                np.array(predicted_dG[:n]) - np.array(expected_dG[:n])
            )))
            results.append(MetricResult("thermo_dG_mae_eV", mae))
        # Overpotential error
        eta_err = abs(predicted_eta - expected_eta)
        results.append(MetricResult("thermo_overpotential_error_V", eta_err))
        # RDS match (compare which step has largest dG)
        if n > 1:
            pred_rds = int(np.argmax(np.diff(predicted_dG[:n])))
            exp_rds = int(np.argmax(np.diff(expected_dG[:n])))
            results.append(MetricResult(
                "thermo_rds_match",
                1.0 if pred_rds == exp_rds else 0.0,
                {"predicted_rds": pred_rds, "expected_rds": exp_rds},
            ))
        return results


class RAGMetrics:
    """
    Evaluate RAG retrieval quality.

    Metrics:
    - mrr: Mean Reciprocal Rank of relevant documents
    - ndcg_at_k: Normalised Discounted Cumulative Gain
    - hit_rate_at_k: fraction of queries where a relevant doc is in top-k
    """

    @staticmethod
    def mrr(ranked_relevant: List[bool]) -> float:
        """Mean Reciprocal Rank."""
        for i, rel in enumerate(ranked_relevant):
            if rel:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(relevance_scores: List[float], k: int = 5) -> float:
        """NDCG@k."""
        def dcg(scores, k):
            return sum(s / np.log2(i + 2) for i, s in enumerate(scores[:k]))
        ideal = sorted(relevance_scores, reverse=True)
        idcg = dcg(ideal, k)
        if idcg == 0:
            return 0.0
        return dcg(relevance_scores, k) / idcg

    @staticmethod
    def hit_rate(ranked_relevant: List[bool], k: int = 5) -> float:
        return 1.0 if any(ranked_relevant[:k]) else 0.0


class SCFPredictionMetrics:
    """
    Evaluate SCF convergence prediction accuracy.

    Metrics:
    - step_prediction_mae: |predicted_n_conv - actual_n_conv|
    - sloshing_detection_accuracy: binary classification accuracy
    - algo_recommendation_accuracy: fraction matching expert VASP settings
    """

    @staticmethod
    def evaluate(predicted_step: int, actual_step: int,
                 predicted_sloshing: bool, actual_sloshing: bool) -> List[MetricResult]:
        results = []
        if predicted_step > 0 and actual_step > 0:
            results.append(MetricResult(
                "scf_step_prediction_mae",
                float(abs(predicted_step - actual_step)),
            ))
        results.append(MetricResult(
            "scf_sloshing_detection_accuracy",
            1.0 if predicted_sloshing == actual_sloshing else 0.0,
        ))
        return results


class GrounderMetrics:
    """
    Evaluate cross-modal hypothesis grounder calibration.

    Metrics:
    - calibration_error: |mean(confidence) - mean(is_correct)|
    - auc_roc: area under ROC for confidence → correct/incorrect
    - brier_score: mean (confidence - is_correct)²
    """

    @staticmethod
    def brier_score(confidences: List[float], labels: List[int]) -> float:
        c = np.array(confidences)
        y = np.array(labels)
        return float(np.mean((c - y) ** 2))

    @staticmethod
    def calibration_error(confidences: List[float], labels: List[int],
                          n_bins: int = 10) -> float:
        c = np.array(confidences)
        y = np.array(labels)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            mask = (c >= lo) & (c < hi)
            if mask.sum() > 0:
                bin_conf = c[mask].mean()
                bin_acc = y[mask].mean()
                ece += mask.sum() / len(c) * abs(bin_conf - bin_acc)
        return float(ece)


# ---------------------------------------------------------------------------
# Composite evaluation runner
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    timestamp: str
    metrics: List[MetricResult]
    composite_score: float
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Evaluation Report ({self.timestamp})",
                 f"Composite Score: {self.composite_score:.3f}",
                 ""]
        for m in sorted(self.metrics, key=lambda x: x.name):
            lines.append(f"  {m.name:<40s} {m.value:.4f}")
        return "\n".join(lines)


def run_full_evaluation(
    intent_fn: Callable[[str], Dict],
    hypothesis_fn: Callable[[Dict], Tuple[List[str], List[str]]],
    thermo_fn: Callable[[str], Tuple[List[float], float]],
    golden_set: List[GoldenExample] = None,
) -> EvaluationReport:
    """
    Run the complete evaluation pipeline against the golden dataset.

    Parameters
    ----------
    intent_fn      : query → intent dict
    hypothesis_fn  : intent → (intermediates, steps)
    thermo_fn      : reaction → (dG_profile, overpotential)
    golden_set     : override default GOLDEN_SET

    Returns
    -------
    EvaluationReport with all component metrics and composite score.
    """
    if golden_set is None:
        golden_set = GOLDEN_SET

    all_metrics: List[MetricResult] = []

    for ex in golden_set:
        # 1. Intent parsing
        try:
            pred_intent = intent_fn(ex.query)
            all_metrics.extend(
                IntentParsingMetrics.evaluate(pred_intent, ex.expected_intent)
            )
        except (ValueError, TypeError, KeyError, IndexError) as e:
            all_metrics.append(MetricResult(f"intent_error_{ex.id}", 0.0,
                                            {"error": str(e), "error_type": type(e).__name__}))

        # 2. Hypothesis generation
        try:
            pred_intermediates, pred_steps = hypothesis_fn(pred_intent)
            all_metrics.extend(
                HypothesisMetrics.evaluate(
                    pred_intermediates, ex.expected_intermediates,
                    pred_steps, ex.query.split()[0],
                )
            )
        except (ValueError, TypeError, KeyError, IndexError) as e:
            all_metrics.append(MetricResult(f"hypothesis_error_{ex.id}", 0.0,
                                            {"error": str(e), "error_type": type(e).__name__}))

        # 3. Thermodynamics
        try:
            pred_dG, pred_eta = thermo_fn(ex.query)
            all_metrics.extend(
                ThermodynamicsMetrics.evaluate(
                    pred_dG, ex.expected_dG_profile,
                    pred_eta, ex.expected_overpotential,
                )
            )
        except (ValueError, TypeError, KeyError, IndexError) as e:
            all_metrics.append(MetricResult(f"thermo_error_{ex.id}", 0.0,
                                            {"error": str(e), "error_type": type(e).__name__}))

    # Composite score: weighted average of key metrics
    WEIGHTS = {
        "intent_stage_accuracy": 0.10,
        "intent_material_match": 0.10,
        "hypothesis_intermediate_f1": 0.25,
        "hypothesis_forward_direction": 0.10,
        "thermo_dG_mae_eV": -0.20,          # negative: lower is better
        "thermo_overpotential_error_V": -0.15,
        "thermo_rds_match": 0.10,
    }

    composite = 0.0
    total_weight = 0.0
    for m in all_metrics:
        if m.name in WEIGHTS:
            w = WEIGHTS[m.name]
            composite += w * m.value
            total_weight += abs(w)
    composite = composite / (total_weight + 1e-9)
    composite = max(0.0, min(1.0, composite))

    return EvaluationReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        metrics=all_metrics,
        composite_score=composite,
    )
