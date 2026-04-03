"""
Model Registry
==============
Track every ML/AI model used in ChatDFT with version, performance baseline,
and deployment status. Supports rollback and A/B routing.

Usage
-----
    registry = ModelRegistry()
    registry.register("hypothesis_grounder", version="1.0.0",
                       model_type="contrastive", framework="numpy",
                       metrics={"brier_score": 0.12, "calibration_error": 0.08})
    active = registry.get_active("hypothesis_grounder")

    # A/B test: route 10% of traffic to a new version
    registry.register("hypothesis_grounder", version="1.1.0", ...)
    registry.set_ab_test("hypothesis_grounder", "1.0.0", "1.1.0", split=0.1)
    version = registry.route("hypothesis_grounder")  # returns "1.0.0" 90%, "1.1.0" 10%
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class ModelStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    name: str                          # e.g. "hypothesis_grounder"
    version: str                       # semver: "1.0.0"
    model_type: str                    # "contrastive", "embedding", "llm", "rule_based"
    framework: str                     # "numpy", "pytorch", "openai_api", "sentence_transformers"
    stage: ModelStage = ModelStage.DEVELOPMENT
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    artifact_hash: str = ""            # SHA-256 of model weights / config
    created_at: str = ""
    promoted_at: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "framework": self.framework,
            "stage": self.stage.value,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "artifact_hash": self.artifact_hash,
            "created_at": self.created_at,
            "promoted_at": self.promoted_at,
            "description": self.description,
            "dependencies": self.dependencies,
        }


@dataclass
class ABTest:
    model_name: str
    control_version: str       # current production
    treatment_version: str     # candidate
    split: float               # fraction routed to treatment (0.0–1.0)
    started_at: str = ""
    control_metrics: Dict[str, float] = field(default_factory=dict)
    treatment_metrics: Dict[str, float] = field(default_factory=dict)
    total_control: int = 0
    total_treatment: int = 0

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat() + "Z"


class ModelRegistry:
    """
    In-memory model registry with persistence hooks.

    In production, back this with a PostgreSQL table (model_registry).
    For now, provides the API contract that the rest of the system uses.
    """

    def __init__(self):
        self._versions: Dict[str, ModelVersion] = {}   # key → ModelVersion
        self._active: Dict[str, str] = {}              # model_name → active version string
        self._ab_tests: Dict[str, ABTest] = {}         # model_name → ABTest
        self._register_defaults()

    def _register_defaults(self):
        """Register all current models in ChatDFT."""
        defaults = [
            ModelVersion(
                name="text_embedding",
                version="3.0.0",
                model_type="embedding",
                framework="openai_api",
                stage=ModelStage.PRODUCTION,
                metrics={"dimension": 1536},
                hyperparameters={"model_id": "text-embedding-3-small", "max_tokens": 8000},
                description="OpenAI text-embedding-3-small for RAG + structure search",
            ),
            ModelVersion(
                name="llm_agent",
                version="4.0.0",
                model_type="llm",
                framework="openai_api",
                stage=ModelStage.PRODUCTION,
                metrics={"cost_per_1k_tokens": 0.005},
                hyperparameters={
                    "model_id": "gpt-4o", "temperature": 0.1,
                    "json_mode": True, "max_tokens": 2000,
                },
                description="GPT-4o for intent/hypothesis/plan/analysis agents",
            ),
            ModelVersion(
                name="cross_encoder_reranker",
                version="1.0.0",
                model_type="reranker",
                framework="sentence_transformers",
                stage=ModelStage.PRODUCTION,
                hyperparameters={"model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
                description="Cross-encoder for RAG reranking",
            ),
            ModelVersion(
                name="hypothesis_grounder",
                version="1.0.0",
                model_type="contrastive",
                framework="numpy",
                stage=ModelStage.PRODUCTION,
                hyperparameters={
                    "d_embed": 64, "temperature": 0.07,
                    "text_encoder": "bow_chemistry",
                    "graph_encoder": "species_onehot_pool",
                    "property_encoder": "conv1d_hand_crafted",
                },
                description="InfoNCE cross-modal alignment (text + graph + energy)",
            ),
            ModelVersion(
                name="scf_convergence_predictor",
                version="1.0.0",
                model_type="time_series",
                framework="numpy",
                stage=ModelStage.PRODUCTION,
                hyperparameters={
                    "dc_ratio": 0.3, "f_low": 0.05,
                    "prediction_window": 10, "min_window": 4,
                },
                description="FFT sloshing detection + exponential convergence fit",
            ),
            ModelVersion(
                name="surface_topology_graph",
                version="1.0.0",
                model_type="representation",
                framework="scipy",
                stage=ModelStage.PRODUCTION,
                hyperparameters={
                    "min_voronoi_area": 0.5,
                    "node_features": 6,
                    "pbc_images": 9,
                },
                description="Voronoi-based surface graph with symmetry scoring",
            ),
            ModelVersion(
                name="rag_retriever",
                version="1.0.0",
                model_type="retrieval",
                framework="pgvector",
                stage=ModelStage.PRODUCTION,
                hyperparameters={
                    "semantic_weight": 0.7,
                    "rrf_k": 60,
                    "chunk_size_words": 350,
                    "chunk_overlap_words": 50,
                },
                description="Hybrid semantic+keyword search with RRF fusion",
            ),
        ]
        for mv in defaults:
            self._versions[mv.key] = mv
            self._active[mv.name] = mv.version

    def register(self, name: str, version: str, model_type: str = "unknown",
                 framework: str = "unknown", stage: ModelStage = ModelStage.DEVELOPMENT,
                 metrics: Dict[str, float] = None,
                 hyperparameters: Dict[str, Any] = None,
                 description: str = "",
                 artifact_hash: str = "") -> ModelVersion:
        mv = ModelVersion(
            name=name, version=version, model_type=model_type,
            framework=framework, stage=stage,
            metrics=metrics or {}, hyperparameters=hyperparameters or {},
            description=description, artifact_hash=artifact_hash,
        )
        self._versions[mv.key] = mv
        log.info("Registered model %s", mv.key)
        return mv

    def promote(self, name: str, version: str, to_stage: ModelStage) -> ModelVersion:
        key = f"{name}:{version}"
        mv = self._versions.get(key)
        if mv is None:
            raise KeyError(f"Model {key} not found")
        mv.stage = to_stage
        mv.promoted_at = datetime.now(timezone.utc).isoformat() + "Z"
        if to_stage == ModelStage.PRODUCTION:
            self._active[name] = version
        log.info("Promoted %s to %s", key, to_stage.value)
        return mv

    def get_active(self, name: str) -> Optional[ModelVersion]:
        ver = self._active.get(name)
        if ver is None:
            return None
        return self._versions.get(f"{name}:{ver}")

    def get_version(self, name: str, version: str) -> Optional[ModelVersion]:
        return self._versions.get(f"{name}:{version}")

    def list_versions(self, name: str) -> List[ModelVersion]:
        return [v for v in self._versions.values() if v.name == name]

    def list_all(self) -> List[ModelVersion]:
        return list(self._versions.values())

    # --- A/B Testing ---------------------------------------------------------

    def set_ab_test(self, model_name: str, control: str, treatment: str,
                    split: float = 0.1) -> ABTest:
        ab = ABTest(
            model_name=model_name,
            control_version=control,
            treatment_version=treatment,
            split=split,
        )
        self._ab_tests[model_name] = ab
        log.info("A/B test started: %s control=%s treatment=%s split=%.1f%%",
                 model_name, control, treatment, split * 100)
        return ab

    def route(self, model_name: str) -> str:
        """Return which version to use, respecting any active A/B test."""
        ab = self._ab_tests.get(model_name)
        if ab is not None:
            if random.random() < ab.split:
                ab.total_treatment += 1
                return ab.treatment_version
            ab.total_control += 1
            return ab.control_version
        return self._active.get(model_name, "unknown")

    def record_ab_metric(self, model_name: str, version: str,
                         metric_name: str, value: float):
        ab = self._ab_tests.get(model_name)
        if ab is None:
            return
        if version == ab.control_version:
            ab.control_metrics.setdefault(metric_name, [])
            ab.control_metrics[metric_name].append(value)
        elif version == ab.treatment_version:
            ab.treatment_metrics.setdefault(metric_name, [])
            ab.treatment_metrics[metric_name].append(value)

    def get_ab_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        ab = self._ab_tests.get(model_name)
        if ab is None:
            return None
        return {
            "model_name": ab.model_name,
            "control": ab.control_version,
            "treatment": ab.treatment_version,
            "split": ab.split,
            "control_n": ab.total_control,
            "treatment_n": ab.total_treatment,
            "control_metrics": ab.control_metrics,
            "treatment_metrics": ab.treatment_metrics,
        }

    def summary(self) -> str:
        lines = ["Model Registry Summary", "=" * 50]
        for mv in sorted(self._versions.values(), key=lambda x: x.name):
            active = " [ACTIVE]" if self._active.get(mv.name) == mv.version else ""
            lines.append(
                f"  {mv.name}:{mv.version}  stage={mv.stage.value}{active}  "
                f"type={mv.model_type}  framework={mv.framework}"
            )
        ab_tests = list(self._ab_tests.values())
        if ab_tests:
            lines.append("\nActive A/B Tests:")
            for ab in ab_tests:
                lines.append(
                    f"  {ab.model_name}: {ab.control_version} vs {ab.treatment_version} "
                    f"(split={ab.split:.0%}, n_ctrl={ab.total_control}, n_treat={ab.total_treatment})"
                )
        return "\n".join(lines)


# Singleton
model_registry = ModelRegistry()
