"""
Experiment Tracker
==================
Track every AI/ML experiment run in ChatDFT: hyperparameters, metrics,
artifacts, and lineage. Replaces the basic AgentLog with a proper
experiment tracking system.

Design
------
- ExperimentRun: one invocation of a model/agent (e.g., one hypothesis generation)
- Each run records: model version, input hash, output hash, metrics, latency, cost
- Runs are persisted to PostgreSQL (experiment_run table) for long-term analysis
- In-memory buffer for fast writes, async flush to DB

Usage
-----
    tracker = ExperimentTracker()
    run = tracker.start_run("hypothesis_agent", model_version="1.0.0")
    run.log_param("temperature", 0.1)
    run.log_input(query_text)
    # ... do work ...
    run.log_metric("confidence", 0.85)
    run.log_metric("intermediate_f1", 0.92)
    run.log_output(hypothesis_json)
    run.end()
    # async: await tracker.flush()
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class ExperimentRun:
    """One tracked execution of an AI/ML component."""
    run_id: str = ""
    experiment_name: str = ""          # "hypothesis_agent", "rag_retrieval", etc.
    model_name: str = ""               # from model registry
    model_version: str = ""
    session_id: Optional[int] = None

    # Hyperparameters snapshot
    params: Dict[str, Any] = field(default_factory=dict)

    # I/O hashes (for dedup and lineage)
    input_hash: str = ""
    input_preview: str = ""            # first 500 chars
    output_hash: str = ""
    output_preview: str = ""

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Cost / performance
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    # Status
    status: str = "running"            # running | completed | failed
    error_msg: str = ""

    # Timestamps
    started_at: str = ""
    ended_at: str = ""

    # Provenance
    git_sha: str = ""
    code_version: str = ""
    parent_run_id: str = ""            # for chained runs (intent → hypothesis → plan)

    def __post_init__(self):
        if not self.run_id:
            self.run_id = str(uuid.uuid4())[:12]
        if not self.started_at:
            self.started_at = datetime.utcnow().isoformat() + "Z"

    def log_param(self, key: str, value: Any):
        self.params[key] = value

    def log_metric(self, key: str, value: float):
        self.metrics[key] = value

    def log_input(self, text: str):
        self.input_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        self.input_preview = text[:500]

    def log_output(self, text: str):
        self.output_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        self.output_preview = text[:500]

    def log_tokens(self, input_tokens: int, output_tokens: int,
                   cost_per_1k_input: float = 0.005,
                   cost_per_1k_output: float = 0.015):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = (input_tokens * cost_per_1k_input +
                         output_tokens * cost_per_1k_output) / 1000.0

    def end(self, status: str = "completed", error: str = ""):
        self.ended_at = datetime.utcnow().isoformat() + "Z"
        self.status = status
        self.error_msg = error
        if self.started_at:
            start = datetime.fromisoformat(self.started_at.rstrip("Z"))
            end = datetime.fromisoformat(self.ended_at.rstrip("Z"))
            self.latency_ms = int((end - start).total_seconds() * 1000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "session_id": self.session_id,
            "params": self.params,
            "input_hash": self.input_hash,
            "input_preview": self.input_preview,
            "output_hash": self.output_hash,
            "output_preview": self.output_preview,
            "metrics": self.metrics,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "status": self.status,
            "error_msg": self.error_msg,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "parent_run_id": self.parent_run_id,
        }


class ExperimentTracker:
    """
    Central experiment tracking for all ChatDFT AI/ML components.

    Provides:
    - Run lifecycle management (start → log → end)
    - Metric aggregation across runs
    - Cost tracking (LLM API spend)
    - Performance regression detection
    - Run chaining for multi-agent pipelines
    """

    def __init__(self):
        self._runs: Dict[str, ExperimentRun] = {}
        self._completed: List[ExperimentRun] = []
        self._max_completed = 10000

    def start_run(self, experiment_name: str,
                  model_name: str = "",
                  model_version: str = "",
                  session_id: int = None,
                  parent_run_id: str = "") -> ExperimentRun:
        run = ExperimentRun(
            experiment_name=experiment_name,
            model_name=model_name,
            model_version=model_version,
            session_id=session_id,
            parent_run_id=parent_run_id,
        )
        self._runs[run.run_id] = run
        return run

    def end_run(self, run_id: str, status: str = "completed", error: str = ""):
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        run.end(status, error)
        self._completed.append(run)
        if len(self._completed) > self._max_completed:
            self._completed = self._completed[-self._max_completed:]

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        return self._runs.get(run_id) or next(
            (r for r in self._completed if r.run_id == run_id), None
        )

    # --- Aggregation ---------------------------------------------------------

    def metric_history(self, experiment_name: str,
                       metric_name: str,
                       last_n: int = 100) -> List[float]:
        """Return recent values of a metric for a given experiment."""
        values = []
        for run in reversed(self._completed):
            if run.experiment_name == experiment_name and metric_name in run.metrics:
                values.append(run.metrics[metric_name])
                if len(values) >= last_n:
                    break
        return list(reversed(values))

    def cost_summary(self, since_hours: int = 24) -> Dict[str, float]:
        """Total cost by experiment in the last N hours."""
        cutoff = time.time() - since_hours * 3600
        costs: Dict[str, float] = {}
        for run in self._completed:
            try:
                ts = datetime.fromisoformat(run.started_at.rstrip("Z")).timestamp()
            except (ValueError, AttributeError):
                continue
            if ts >= cutoff:
                costs[run.experiment_name] = costs.get(run.experiment_name, 0) + run.cost_usd
        return costs

    def latency_percentiles(self, experiment_name: str,
                             last_n: int = 100) -> Dict[str, float]:
        """p50, p90, p99 latency for recent runs."""
        import numpy as np
        latencies = []
        for run in reversed(self._completed):
            if run.experiment_name == experiment_name and run.latency_ms > 0:
                latencies.append(run.latency_ms)
                if len(latencies) >= last_n:
                    break
        if not latencies:
            return {"p50": 0, "p90": 0, "p99": 0}
        arr = np.array(latencies)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
        }

    def detect_regression(self, experiment_name: str,
                           metric_name: str,
                           window: int = 20,
                           threshold: float = 0.1) -> Optional[str]:
        """
        Detect if a metric has degraded significantly.

        Compares the mean of the last `window` values to the prior `window`.
        Returns a warning string if degradation exceeds threshold, else None.
        """
        import numpy as np
        history = self.metric_history(experiment_name, metric_name, last_n=2 * window)
        if len(history) < 2 * window:
            return None
        recent = np.mean(history[-window:])
        prior = np.mean(history[-2 * window:-window])
        if prior == 0:
            return None
        change = (recent - prior) / abs(prior)
        # For metrics where higher is better (accuracy, F1):
        if change < -threshold:
            return (
                f"REGRESSION: {experiment_name}.{metric_name} dropped "
                f"{abs(change):.1%} (prior={prior:.4f}, recent={recent:.4f})"
            )
        return None

    # --- Persistence (async, to PostgreSQL) ----------------------------------

    async def flush_to_db(self):
        """Persist completed runs to the experiment_run database table."""
        try:
            from server.db import AsyncSessionLocal
        except ImportError:
            return

        runs_to_flush = [r for r in self._completed if r.status in ("completed", "failed")]
        if not runs_to_flush:
            return

        try:
            async with AsyncSessionLocal() as session:
                from sqlalchemy import text
                for run in runs_to_flush[-100:]:
                    await session.execute(text("""
                        INSERT INTO experiment_run
                        (run_id, experiment_name, model_name, model_version,
                         session_id, params, metrics, input_hash, output_hash,
                         latency_ms, input_tokens, output_tokens, cost_usd,
                         status, error_msg, started_at, ended_at, parent_run_id)
                        VALUES (:run_id, :experiment_name, :model_name, :model_version,
                                :session_id, :params, :metrics, :input_hash, :output_hash,
                                :latency_ms, :input_tokens, :output_tokens, :cost_usd,
                                :status, :error_msg, :started_at, :ended_at, :parent_run_id)
                        ON CONFLICT (run_id) DO NOTHING
                    """), {
                        **run.to_dict(),
                        "params": json.dumps(run.params),
                        "metrics": json.dumps(run.metrics),
                    })
                await session.commit()
                log.info("Flushed %d experiment runs to DB", len(runs_to_flush))
        except Exception as e:
            log.warning("Failed to flush experiment runs: %s", e)

    def summary(self) -> str:
        lines = [f"Experiment Tracker: {len(self._completed)} completed, "
                 f"{len(self._runs)} active"]
        # Per-experiment stats
        from collections import Counter
        counts = Counter(r.experiment_name for r in self._completed)
        for name, count in counts.most_common(10):
            latency = self.latency_percentiles(name)
            lines.append(f"  {name}: {count} runs, p50={latency['p50']:.0f}ms")
        return "\n".join(lines)


# Singleton
experiment_tracker = ExperimentTracker()
