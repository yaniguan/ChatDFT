"""
Experiment Registry and Trustworthiness Gates
===============================================

Every scientific run gets tracked with full provenance:
  - Config hash, data hash, code commit hash
  - Environment fingerprint (Python, numpy, torch versions)
  - Artifact directory with results
  - Trustworthiness assessment before publishing

Usage
-----
    from science.core.experiment import ExperimentTracker, TrustworthinessGate

    with ExperimentTracker("gnn_benchmark") as exp:
        exp.log_config({"model": "schnet", "epochs": 100})
        exp.log_metric("test_mae", 0.45)
        exp.log_artifact("fig7.png", "figures/fig7_gnn_comparison.png")

    gate = TrustworthinessGate(exp)
    report = gate.check()
    if report.publishable:
        print("Ready for publication")
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from science.core.seeds import experiment_manifest, get_global_seed


@dataclass
class ExperimentRun:
    """A single experiment run with full provenance."""
    id: str
    name: str
    timestamp: str
    config: Dict[str, Any]
    config_hash: str
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    environment: Dict[str, Any]
    git_commit: Optional[str]
    seed: Optional[int]
    duration_s: float = 0.0
    status: str = "running"  # running, completed, failed
    notes: str = ""


class ExperimentTracker:
    """
    Track experiments with full provenance and reproducibility.

    Parameters
    ----------
    name : str
        Experiment name.
    output_dir : str
        Directory for experiment artifacts and manifests.
    """

    def __init__(self, name: str, output_dir: str = "results/experiments"):
        self.name = name
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = f"{name}_{ts}"
        self._start_time = time.time()

        self._config: Dict[str, Any] = {}
        self._metrics: Dict[str, float] = {}
        self._artifacts: Dict[str, str] = {}

        self._run = ExperimentRun(
            id=self._run_id,
            name=name,
            timestamp=ts,
            config={},
            config_hash="",
            metrics={},
            artifacts={},
            environment=experiment_manifest(),
            git_commit=self._get_git_commit(),
            seed=get_global_seed(),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._run.status = "failed"
            self._run.notes = str(exc_val)
        else:
            self._run.status = "completed"
        self._run.duration_s = time.time() - self._start_time
        self._save_manifest()
        return False

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        self._config.update(config)
        self._run.config = self._config
        self._run.config_hash = hashlib.sha256(
            json.dumps(self._config, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]

    def log_metric(self, name: str, value: float) -> None:
        """Log a metric value."""
        self._metrics[name] = value
        self._run.metrics = self._metrics

    def log_artifact(self, name: str, filepath: str) -> None:
        """Log an artifact (file path)."""
        self._artifacts[name] = filepath
        self._run.artifacts = self._artifacts

    def log_data_hash(self, data: Any) -> str:
        """Compute and log hash of training/test data."""
        if isinstance(data, np.ndarray):
            h = hashlib.sha256(data.tobytes()).hexdigest()[:12]
        elif isinstance(data, (list, tuple)):
            h = hashlib.sha256(str(data).encode()).hexdigest()[:12]
        else:
            h = hashlib.sha256(str(data).encode()).hexdigest()[:12]
        self.log_config({"data_hash": h})
        return h

    def _get_git_commit(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip()[:12] if result.returncode == 0 else None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _save_manifest(self) -> None:
        """Save experiment manifest to JSON."""
        manifest_path = self._output_dir / f"{self._run_id}.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "id": self._run.id,
                "name": self._run.name,
                "timestamp": self._run.timestamp,
                "status": self._run.status,
                "duration_s": self._run.duration_s,
                "config": self._run.config,
                "config_hash": self._run.config_hash,
                "metrics": self._run.metrics,
                "artifacts": self._run.artifacts,
                "environment": self._run.environment,
                "git_commit": self._run.git_commit,
                "seed": self._run.seed,
                "notes": self._run.notes,
            }, f, indent=2, default=str)

    @property
    def run(self) -> ExperimentRun:
        return self._run


@dataclass
class TrustworthinessReport:
    """Report from trustworthiness gate checks."""
    publishable: bool
    checks: Dict[str, bool]
    warnings: List[str]
    blocking_failures: List[str]

    def summary(self) -> str:
        lines = ["Trustworthiness Report", "=" * 40]
        for check, passed in self.checks.items():
            status = "PASS" if passed else "FAIL"
            lines.append(f"  [{status}] {check}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if self.blocking_failures:
            lines.append("\nBlocking failures:")
            for f in self.blocking_failures:
                lines.append(f"  - {f}")
        lines.append(f"\nPublishable: {'YES' if self.publishable else 'NO'}")
        return "\n".join(lines)


class TrustworthinessGate:
    """
    Check whether an experiment run meets publication standards.

    Checks
    ------
    1. Reproducibility: seed recorded and environment fingerprinted
    2. No silent exceptions: status is 'completed' (not 'failed')
    3. Baseline comparison: at least one baseline metric recorded
    4. Constants provenance: config_hash is non-empty
    5. Uncertainty reported: uncertainty metric exists
    """

    def __init__(self, tracker: ExperimentTracker):
        self._run = tracker.run

    def check(self) -> TrustworthinessReport:
        checks = {}
        warnings = []
        failures = []

        # 1. Reproducibility
        checks["seed_recorded"] = self._run.seed is not None
        if not checks["seed_recorded"]:
            failures.append("No global seed — results not reproducible")

        checks["environment_recorded"] = bool(self._run.environment)
        checks["git_commit_recorded"] = self._run.git_commit is not None
        if not checks["git_commit_recorded"]:
            warnings.append("No git commit — cannot trace code version")

        # 2. No silent failures
        checks["run_completed"] = self._run.status == "completed"
        if not checks["run_completed"]:
            failures.append(f"Run status: {self._run.status}")

        # 3. Metrics present
        checks["metrics_recorded"] = len(self._run.metrics) > 0
        if not checks["metrics_recorded"]:
            failures.append("No metrics recorded")

        # 4. Config provenance
        checks["config_hashed"] = bool(self._run.config_hash)

        # 5. Uncertainty
        has_uncertainty = any(
            "uncertainty" in k or "std" in k or "unc" in k
            for k in self._run.metrics.keys()
        )
        checks["uncertainty_reported"] = has_uncertainty
        if not has_uncertainty:
            warnings.append("No uncertainty metric — add confidence intervals")

        # 6. Baseline comparison
        has_baseline = any(
            "baseline" in k for k in self._run.metrics.keys()
        )
        checks["baseline_compared"] = has_baseline
        if not has_baseline:
            warnings.append("No baseline comparison metric")

        publishable = len(failures) == 0

        return TrustworthinessReport(
            publishable=publishable,
            checks=checks,
            warnings=warnings,
            blocking_failures=failures,
        )
