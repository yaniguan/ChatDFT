"""
Evaluation — Proper ML Evaluation with Statistical Rigour
==========================================================

This module provides evaluation methodology that would survive
an ML interview:

1. **Scaffold-split evaluation** — not random split (inflated numbers)
2. **Bootstrap confidence intervals** — report AUROC 0.85 ± 0.03, not just 0.85
3. **Multiple metrics** — AUROC, AUPRC, MCC, F1 for classification;
   RMSE, MAE, R², Spearman ρ for regression
4. **Statistical significance testing** — paired t-test between models
5. **Per-task evaluation** — for multi-task datasets (Tox21 has 12 tasks)

Key metrics and why they matter:
- AUROC: overall ranking quality (but misleading on imbalanced data!)
- AUPRC: precision-recall balance — the RIGHT metric for Tox21
- MCC: balanced metric that works for any class ratio
- Spearman ρ: rank correlation (for regression, often more meaningful
  than RMSE when you care about ordering not absolute values)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under ROC Curve."""
    from sklearn.metrics import roc_auc_score

    valid = ~(np.isnan(y_true) | np.isnan(y_score))
    if valid.sum() < 5 or len(np.unique(y_true[valid])) < 2:
        return float("nan")
    return float(roc_auc_score(y_true[valid], y_score[valid]))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under Precision-Recall Curve (better for imbalanced data)."""
    from sklearn.metrics import average_precision_score

    valid = ~(np.isnan(y_true) | np.isnan(y_score))
    if valid.sum() < 5 or len(np.unique(y_true[valid])) < 2:
        return float("nan")
    return float(average_precision_score(y_true[valid], y_score[valid]))


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Matthews Correlation Coefficient (-1 to 1).

    The BEST single metric for imbalanced binary classification.
    MCC = 0 means random, MCC = 1 means perfect.
    Unlike accuracy, MCC is informative even with 95:5 class ratio.
    """
    from sklearn.metrics import matthews_corrcoef

    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 5:
        return float("nan")
    return float(matthews_corrcoef(y_true[valid], y_pred[valid]))


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 score."""
    from sklearn.metrics import f1_score

    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 5:
        return float("nan")
    return float(f1_score(y_true[valid], y_pred[valid], zero_division=0))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 2:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 2:
        return float("nan")
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid])))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination)."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 2:
        return float("nan")
    ss_res = np.sum((y_true[valid] - y_pred[valid]) ** 2)
    ss_tot = np.sum((y_true[valid] - y_true[valid].mean()) ** 2)
    return float(1 - ss_res / max(ss_tot, 1e-9))


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation."""
    from scipy.stats import spearmanr

    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 5:
        return float("nan")
    rho, _ = spearmanr(y_true[valid], y_pred[valid])
    return float(rho)


# ---------------------------------------------------------------------------
# Evaluation result containers
# ---------------------------------------------------------------------------


@dataclass
class TaskMetrics:
    """Metrics for a single prediction task."""

    task_name: str
    task_type: str  # "classification" or "regression"
    metrics: Dict[str, float]
    n_samples: int
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class ModelEvaluation:
    """Complete evaluation of a single model."""

    model_name: str
    dataset_name: str
    split_method: str  # "scaffold" or "random"
    task_results: List[TaskMetrics]
    aggregate_metrics: Dict[str, float]  # mean across tasks
    train_time_s: float = 0.0
    n_params: int = 0

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_name}",
            f"Dataset: {self.dataset_name} ({self.split_method} split)",
            f"Parameters: {self.n_params:,}",
            f"Train time: {self.train_time_s:.1f}s",
            "",
        ]
        # Aggregate
        for metric, value in self.aggregate_metrics.items():
            ci = ""
            # Find CI from first task that has it
            for tr in self.task_results:
                if metric in tr.confidence_intervals:
                    lo, hi = tr.confidence_intervals[metric]
                    ci = f" [{lo:.4f}, {hi:.4f}]"
                    break
            lines.append(f"  {metric}: {value:.4f}{ci}")

        if len(self.task_results) > 1:
            lines.append("\nPer-task results:")
            for tr in self.task_results:
                primary = list(tr.metrics.values())[0] if tr.metrics else 0
                lines.append(f"  {tr.task_name}: {primary:.4f}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute metric with bootstrap confidence interval.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    point = metric_fn(y_true, y_score)

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_score[idx])
            if not np.isnan(score):
                bootstrap_scores.append(score)
        except (ValueError, IndexError):
            continue

    if len(bootstrap_scores) < 10:
        return point, point, point

    alpha = (1 - ci) / 2
    lo = float(np.percentile(bootstrap_scores, alpha * 100))
    hi = float(np.percentile(bootstrap_scores, (1 - alpha) * 100))

    return point, lo, hi


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate_classification(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task_name: str = "task",
    n_bootstrap: int = 500,
) -> TaskMetrics:
    """
    Evaluate a binary classification task.

    Computes: AUROC, AUPRC, MCC, F1 with bootstrap CIs.
    """
    y_pred = (y_score > 0.5).astype(float)

    metrics = {
        "auroc": auroc(y_true, y_score),
        "auprc": auprc(y_true, y_score),
        "mcc": mcc(y_true, y_pred),
        "f1": f1(y_true, y_pred),
    }

    # Bootstrap CIs for primary metrics
    cis = {}
    for name, fn in [("auroc", auroc), ("auprc", auprc)]:
        _, lo, hi = bootstrap_ci(y_true, y_score, fn, n_bootstrap=n_bootstrap)
        cis[name] = (lo, hi)

    valid = ~np.isnan(y_true)
    return TaskMetrics(
        task_name=task_name,
        task_type="classification",
        metrics=metrics,
        n_samples=int(valid.sum()),
        confidence_intervals=cis,
    )


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_name: str = "task",
    n_bootstrap: int = 500,
) -> TaskMetrics:
    """
    Evaluate a regression task.

    Computes: RMSE, MAE, R², Spearman ρ with bootstrap CIs.
    """
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "spearman": spearman_rho(y_true, y_pred),
    }

    cis = {}
    for name, fn in [("rmse", rmse), ("r2", r_squared)]:
        _, lo, hi = bootstrap_ci(y_true, y_pred, fn, n_bootstrap=n_bootstrap)
        cis[name] = (lo, hi)

    valid = ~np.isnan(y_true)
    return TaskMetrics(
        task_name=task_name,
        task_type="regression",
        metrics=metrics,
        n_samples=int(valid.sum()),
        confidence_intervals=cis,
    )


# ---------------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------------


def paired_test(
    scores_a: List[float],
    scores_b: List[float],
    test: str = "wilcoxon",
) -> Tuple[float, float, str]:
    """
    Test if model A is significantly better than model B.

    Uses Wilcoxon signed-rank test (non-parametric, paired) on
    per-fold or per-task scores.

    Returns (statistic, p_value, conclusion).
    """
    from scipy.stats import ttest_rel, wilcoxon

    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = a - b

    if len(diff) < 5 or np.all(diff == 0):
        return 0.0, 1.0, "insufficient data"

    if test == "wilcoxon":
        stat, p = wilcoxon(a, b)
    else:
        stat, p = ttest_rel(a, b)

    if p < 0.01:
        conclusion = "significantly different (p < 0.01)"
    elif p < 0.05:
        conclusion = "significantly different (p < 0.05)"
    else:
        conclusion = "no significant difference"

    return float(stat), float(p), conclusion


# ---------------------------------------------------------------------------
# Benchmark comparison table
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Results from benchmarking multiple models on one dataset."""

    dataset_name: str
    split_method: str
    evaluations: List[ModelEvaluation]

    def leaderboard(self, metric: str = None) -> str:
        """Format as a markdown leaderboard table."""
        if not self.evaluations:
            return "No results."

        # Determine primary metric
        if metric is None:
            first_task = self.evaluations[0].task_results[0]
            metric = list(first_task.metrics.keys())[0]

        # Sort by aggregate metric (higher is better for most, lower for rmse/mae)
        reverse = metric not in ("rmse", "mae")
        sorted_evals = sorted(
            self.evaluations,
            key=lambda e: e.aggregate_metrics.get(metric, 0),
            reverse=reverse,
        )

        lines = [
            f"## {self.dataset_name} Benchmark ({self.split_method} split)\n",
            f"| Rank | Model | {metric.upper()} | 95% CI | Params | Time |",
            "|------|-------|" + "-" * (len(metric) + 2) + "|--------|--------|------|",
        ]

        for rank, ev in enumerate(sorted_evals, 1):
            val = ev.aggregate_metrics.get(metric, 0)
            ci_str = ""
            for tr in ev.task_results:
                if metric in tr.confidence_intervals:
                    lo, hi = tr.confidence_intervals[metric]
                    ci_str = f"[{lo:.3f}, {hi:.3f}]"
                    break

            lines.append(
                f"| {rank} | {ev.model_name} | {val:.4f} | {ci_str} | {ev.n_params:,} | {ev.train_time_s:.1f}s |"
            )

        return "\n".join(lines)
