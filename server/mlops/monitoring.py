"""
Production Monitoring
=====================
Real-time monitoring of AI/ML component health in ChatDFT.

Monitors:
- Embedding drift (cosine distance between recent and baseline distributions)
- Grounder confidence calibration (is confidence well-calibrated?)
- RAG retrieval quality (hit rate, MRR trends)
- LLM response quality (JSON parse success rate, latency SLA)
- Cost burn rate (daily API spend)

Usage
-----
    monitor = ProductionMonitor()
    monitor.record_embedding("query", vector)
    monitor.record_confidence("hypothesis_grounder", 0.85, was_correct=True)
    monitor.record_retrieval_hit("rag", hit=True, rank=3)
    alerts = monitor.check_all()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Alert:
    severity: str         # "warning" | "critical"
    component: str        # "embedding", "grounder", "rag", "llm", "cost"
    message: str
    value: float
    threshold: float
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def __str__(self):
        return f"[{self.severity.upper()}] {self.component}: {self.message}"


class EmbeddingDriftMonitor:
    """
    Detect distribution shift in embedding vectors.

    Method: track rolling mean of embedding L2 norms and pairwise
    cosine similarity. If recent embeddings deviate significantly
    from the baseline window, fire an alert.
    """

    def __init__(self, baseline_window: int = 500, recent_window: int = 50,
                 norm_threshold: float = 0.15, sim_threshold: float = 0.10):
        self.baseline: Deque[np.ndarray] = deque(maxlen=baseline_window)
        self.recent: Deque[np.ndarray] = deque(maxlen=recent_window)
        self.norm_threshold = norm_threshold
        self.sim_threshold = sim_threshold

    def record(self, vector: List[float]):
        v = np.array(vector, dtype=np.float32)
        self.baseline.append(v)
        self.recent.append(v)

    def check(self) -> Optional[Alert]:
        if len(self.baseline) < 100 or len(self.recent) < 20:
            return None

        # Compare mean norms
        baseline_norms = [np.linalg.norm(v) for v in list(self.baseline)[:-len(self.recent)]]
        recent_norms = [np.linalg.norm(v) for v in self.recent]

        if not baseline_norms:
            return None

        norm_shift = abs(np.mean(recent_norms) - np.mean(baseline_norms)) / (np.mean(baseline_norms) + 1e-9)

        if norm_shift > self.norm_threshold:
            return Alert(
                severity="warning",
                component="embedding_drift",
                message=f"Embedding norm drift: {norm_shift:.1%} (threshold: {self.norm_threshold:.1%})",
                value=norm_shift,
                threshold=self.norm_threshold,
            )

        # Compare mean pairwise similarity
        baseline_mean = np.mean(list(self.baseline)[:-len(self.recent)], axis=0)
        recent_mean = np.mean(list(self.recent), axis=0)
        cos_sim = float(np.dot(baseline_mean, recent_mean) /
                        (np.linalg.norm(baseline_mean) * np.linalg.norm(recent_mean) + 1e-9))
        drift = 1.0 - cos_sim

        if drift > self.sim_threshold:
            return Alert(
                severity="critical" if drift > 2 * self.sim_threshold else "warning",
                component="embedding_drift",
                message=f"Embedding direction drift: {drift:.3f} (threshold: {self.sim_threshold})",
                value=drift,
                threshold=self.sim_threshold,
            )
        return None


class ConfidenceCalibrationMonitor:
    """
    Track whether model confidence scores are well-calibrated.

    A well-calibrated model: P(correct | confidence=0.8) ≈ 0.8

    Uses Expected Calibration Error (ECE) over a rolling window.
    """

    def __init__(self, window: int = 200, n_bins: int = 10,
                 ece_threshold: float = 0.15):
        self.confidences: Deque[float] = deque(maxlen=window)
        self.outcomes: Deque[int] = deque(maxlen=window)
        self.n_bins = n_bins
        self.ece_threshold = ece_threshold

    def record(self, confidence: float, was_correct: bool):
        self.confidences.append(confidence)
        self.outcomes.append(int(was_correct))

    def compute_ece(self) -> float:
        if len(self.confidences) < 50:
            return 0.0
        c = np.array(self.confidences)
        y = np.array(self.outcomes)
        ece = 0.0
        for i in range(self.n_bins):
            lo, hi = i / self.n_bins, (i + 1) / self.n_bins
            mask = (c >= lo) & (c < hi)
            if mask.sum() > 0:
                bin_conf = c[mask].mean()
                bin_acc = y[mask].mean()
                ece += mask.sum() / len(c) * abs(bin_conf - bin_acc)
        return float(ece)

    def check(self) -> Optional[Alert]:
        ece = self.compute_ece()
        if ece > self.ece_threshold:
            return Alert(
                severity="warning",
                component="confidence_calibration",
                message=f"Confidence miscalibrated: ECE={ece:.3f} (threshold: {self.ece_threshold})",
                value=ece,
                threshold=self.ece_threshold,
            )
        return None


class RetrievalQualityMonitor:
    """Track RAG retrieval hit rate and mean reciprocal rank."""

    def __init__(self, window: int = 200, hit_rate_threshold: float = 0.5,
                 mrr_threshold: float = 0.3):
        self.hits: Deque[bool] = deque(maxlen=window)
        self.ranks: Deque[int] = deque(maxlen=window)
        self.hit_rate_threshold = hit_rate_threshold
        self.mrr_threshold = mrr_threshold

    def record(self, hit: bool, rank: int = 0):
        self.hits.append(hit)
        if hit and rank > 0:
            self.ranks.append(rank)

    def check(self) -> List[Alert]:
        alerts = []
        if len(self.hits) < 30:
            return alerts

        hit_rate = sum(self.hits) / len(self.hits)
        if hit_rate < self.hit_rate_threshold:
            alerts.append(Alert(
                severity="warning",
                component="rag_retrieval",
                message=f"RAG hit rate dropped: {hit_rate:.1%} (threshold: {self.hit_rate_threshold:.1%})",
                value=hit_rate,
                threshold=self.hit_rate_threshold,
            ))

        if self.ranks:
            mrr = float(np.mean([1.0 / r for r in self.ranks]))
            if mrr < self.mrr_threshold:
                alerts.append(Alert(
                    severity="warning",
                    component="rag_retrieval",
                    message=f"RAG MRR dropped: {mrr:.3f} (threshold: {self.mrr_threshold})",
                    value=mrr,
                    threshold=self.mrr_threshold,
                ))
        return alerts


class LLMHealthMonitor:
    """Track LLM API health: success rate, latency SLA, JSON parse rate."""

    def __init__(self, window: int = 100,
                 success_threshold: float = 0.95,
                 latency_p99_threshold_ms: int = 15000,
                 json_parse_threshold: float = 0.90):
        self.successes: Deque[bool] = deque(maxlen=window)
        self.latencies: Deque[int] = deque(maxlen=window)
        self.json_parses: Deque[bool] = deque(maxlen=window)
        self.success_threshold = success_threshold
        self.latency_threshold = latency_p99_threshold_ms
        self.json_threshold = json_parse_threshold

    def record(self, success: bool, latency_ms: int, json_parsed: bool):
        self.successes.append(success)
        self.latencies.append(latency_ms)
        self.json_parses.append(json_parsed)

    def check(self) -> List[Alert]:
        alerts = []
        if len(self.successes) < 20:
            return alerts

        success_rate = sum(self.successes) / len(self.successes)
        if success_rate < self.success_threshold:
            alerts.append(Alert(
                severity="critical",
                component="llm_health",
                message=f"LLM success rate: {success_rate:.1%} (threshold: {self.success_threshold:.1%})",
                value=success_rate,
                threshold=self.success_threshold,
            ))

        if self.latencies:
            p99 = float(np.percentile(list(self.latencies), 99))
            if p99 > self.latency_threshold:
                alerts.append(Alert(
                    severity="warning",
                    component="llm_latency",
                    message=f"LLM p99 latency: {p99:.0f}ms (threshold: {self.latency_threshold}ms)",
                    value=p99,
                    threshold=float(self.latency_threshold),
                ))

        json_rate = sum(self.json_parses) / max(len(self.json_parses), 1)
        if json_rate < self.json_threshold:
            alerts.append(Alert(
                severity="warning",
                component="llm_json_parse",
                message=f"LLM JSON parse rate: {json_rate:.1%} (threshold: {self.json_threshold:.1%})",
                value=json_rate,
                threshold=self.json_threshold,
            ))
        return alerts


class CostMonitor:
    """Track daily API spend and alert on burn rate."""

    def __init__(self, daily_budget_usd: float = 50.0):
        self.daily_budget = daily_budget_usd
        self._costs: Deque[Tuple[float, float]] = deque(maxlen=10000)  # (timestamp, usd)

    def record(self, cost_usd: float):
        self._costs.append((time.time(), cost_usd))

    def daily_spend(self) -> float:
        cutoff = time.time() - 86400
        return sum(c for t, c in self._costs if t >= cutoff)

    def check(self) -> Optional[Alert]:
        spend = self.daily_spend()
        if spend > self.daily_budget:
            return Alert(
                severity="critical",
                component="cost",
                message=f"Daily API spend ${spend:.2f} exceeds budget ${self.daily_budget:.2f}",
                value=spend,
                threshold=self.daily_budget,
            )
        if spend > 0.8 * self.daily_budget:
            return Alert(
                severity="warning",
                component="cost",
                message=f"Daily API spend ${spend:.2f} approaching budget ${self.daily_budget:.2f}",
                value=spend,
                threshold=self.daily_budget,
            )
        return None


class ProductionMonitor:
    """Unified monitoring hub for all ChatDFT AI/ML components."""

    def __init__(self):
        self.embedding = EmbeddingDriftMonitor()
        self.calibration = ConfidenceCalibrationMonitor()
        self.retrieval = RetrievalQualityMonitor()
        self.llm = LLMHealthMonitor()
        self.cost = CostMonitor()
        self._alert_history: List[Alert] = []

    def check_all(self) -> List[Alert]:
        """Run all monitors and return any active alerts."""
        alerts = []
        for check_fn in [
            self.embedding.check,
            self.calibration.check,
            self.cost.check,
        ]:
            alert = check_fn()
            if alert:
                alerts.append(alert)
        alerts.extend(self.retrieval.check())
        alerts.extend(self.llm.check())
        self._alert_history.extend(alerts)
        return alerts

    def health_status(self) -> Dict:
        """Return current health status for /health endpoint."""
        alerts = self.check_all()
        return {
            "healthy": len([a for a in alerts if a.severity == "critical"]) == 0,
            "alerts": [str(a) for a in alerts],
            "embedding_drift": self.embedding.check() is not None,
            "calibration_ece": self.calibration.compute_ece(),
            "daily_cost_usd": self.cost.daily_spend(),
            "llm_success_rate": (
                sum(self.llm.successes) / max(len(self.llm.successes), 1)
            ),
        }


# Singleton
production_monitor = ProductionMonitor()
