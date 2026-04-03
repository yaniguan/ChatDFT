"""
Feature Store
=============
Centralised feature computation, versioning, lineage tracking, and serving
for all ML/AI components in ChatDFT.

Architecture
------------
    FeatureDefinition  → declares what a feature is, how to compute it
    FeatureVersion     → immutable snapshot of computed features for an entity
    FeatureLineage     → tracks which inputs produced which features
    FeatureStore       → registry + compute + serve + drift detection

Feature types
-------------
    structure  : Voronoi graph, coordination numbers, site classifications
    scf        : convergence trajectory statistics (rate, sloshing, steps)
    thermo     : free energy profile descriptors (RDS, overpotential, barriers)
    embedding  : text/graph/property embeddings from hypothesis grounder
    mechanism  : reaction network graph features (n_steps, n_intermediates, ...)
    rag        : retrieval quality features (hit rate, rank, relevance)

Entity types
------------
    session    : one user research session
    structure  : one atomic structure (POSCAR)
    job        : one DFT calculation
    hypothesis : one generated reaction mechanism
    paper      : one literature document

Usage
-----
    store = FeatureStore()

    # Register a feature definition
    store.register_feature(FeatureDefinition(
        name="voronoi_coordination",
        entity_type="structure",
        dtype="float32",
        shape=(None, 6),
        version="1.0.0",
        compute_fn=compute_voronoi_features,
        description="Voronoi topology graph node features [Z, layer, CN, vol, dist, var]",
    ))

    # Compute and store
    features = store.compute("voronoi_coordination", entity_id="poscar_abc123",
                              raw_input=poscar_text)

    # Retrieve (online serving)
    features = store.get("voronoi_coordination", entity_id="poscar_abc123")

    # Check drift
    drift = store.check_drift("voronoi_coordination")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FeatureDefinition:
    """Declares a feature: what it is, how to compute it."""
    name: str                          # "voronoi_coordination", "scf_convergence_rate"
    entity_type: str                   # "structure", "job", "hypothesis", "session"
    dtype: str = "float32"             # numpy dtype
    shape: Tuple = ()                  # () for scalar, (6,) for vector, (None, 6) for variable-length
    version: str = "1.0.0"            # feature definition version (bump when compute_fn changes)
    compute_fn: Optional[Callable] = None  # function(raw_input) → numpy array
    description: str = ""
    dependencies: List[str] = field(default_factory=list)   # other features this depends on
    ttl_seconds: int = 0               # 0 = never expires, >0 = recompute after TTL


@dataclass
class FeatureValue:
    """One computed feature value for a specific entity."""
    feature_name: str
    feature_version: str
    entity_type: str
    entity_id: str                     # SHA of POSCAR, job_uid, session_id, etc.
    value: np.ndarray                  # the actual feature data
    computed_at: str = ""
    input_hash: str = ""               # SHA-256 of the raw input that produced this feature
    compute_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.computed_at:
            self.computed_at = datetime.now(timezone.utc).isoformat() + "Z"


@dataclass
class FeatureLineage:
    """Tracks provenance: what inputs produced what features."""
    feature_name: str
    entity_id: str
    input_hash: str                    # hash of raw input (POSCAR text, OUTCAR text, etc.)
    input_type: str                    # "poscar", "outcar", "query_text", "hypothesis_json"
    parent_features: List[str] = field(default_factory=list)  # features used as input
    model_version: str = ""            # which model version computed this
    code_version: str = ""             # git SHA or package version
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"


# ---------------------------------------------------------------------------
# Built-in feature compute functions
# ---------------------------------------------------------------------------

def compute_voronoi_features(raw_input: str) -> np.ndarray:
    """Compute Voronoi surface graph features from POSCAR text."""
    try:
        from ase.io import read
        from io import StringIO
        atoms = read(StringIO(raw_input), format="vasp")
        from science.representations.surface_graph import SurfaceTopologyGraph
        stg = SurfaceTopologyGraph(
            positions=atoms.get_positions(),
            elements=[atoms.get_chemical_symbols()[i] for i in range(len(atoms))],
            cell=atoms.get_cell()[:],
        )
        stg.build()
        return stg.node_feature_matrix()
    except Exception as e:
        log.warning("Voronoi feature computation failed: %s", e)
        return np.zeros((1, 6), dtype=np.float32)


def compute_scf_features(raw_input: str) -> np.ndarray:
    """Extract SCF convergence features from OUTCAR text."""
    try:
        from science.time_series.scf_convergence import SCFTrajectory, analyse_scf
        traj = SCFTrajectory.from_outcar_text(raw_input)
        report = analyse_scf(traj)
        return np.array([
            report.prediction.convergence_rate,
            report.prediction.predicted_step,
            report.prediction.r_squared,
            float(report.sloshing.is_sloshing),
            report.sloshing.dominant_frequency,
            report.sloshing.amplitude,
            report.sloshing.decay_rate,
            float(len(traj.dE)),
            float(traj.is_converged()),
        ], dtype=np.float32)
    except (ValueError, KeyError, TypeError) as e:
        log.warning("SCF feature computation failed: %s", e)
        return np.zeros(9, dtype=np.float32)


def compute_mechanism_features(raw_input: str) -> np.ndarray:
    """Extract reaction network graph features from hypothesis JSON."""
    try:
        data = json.loads(raw_input)
        intermediates = data.get("intermediates", [])
        steps = data.get("reaction_network", data.get("steps", []))
        ts_edges = data.get("ts_edges", [])
        coads = data.get("coads_pairs", data.get("coads", []))

        n_surface = sum(1 for s in intermediates if s.endswith("*"))
        n_gas = sum(1 for s in intermediates if "(g)" in s)
        n_ec = sum(1 for s in steps
                   if any(x in str(s) for x in ["H+", "e-"]))

        return np.array([
            float(len(intermediates)),
            float(len(steps)),
            float(len(ts_edges)),
            float(len(coads)),
            float(n_surface),
            float(n_gas),
            float(n_ec),
            float(n_ec) / max(len(steps), 1),   # fraction electrochemical
        ], dtype=np.float32)
    except (ValueError, KeyError, TypeError) as e:
        log.warning("Mechanism feature computation failed: %s", e)
        return np.zeros(8, dtype=np.float32)


def compute_thermo_features(raw_input: str) -> np.ndarray:
    """Extract thermodynamic features from free energy profile JSON."""
    try:
        data = json.loads(raw_input)
        dG = data.get("dG_profile", data.get("steps", []))
        if isinstance(dG, list) and dG and isinstance(dG[0], dict):
            dG = [s.get("G", 0) for s in dG]
        dG = np.array(dG, dtype=np.float32)

        if len(dG) < 2:
            return np.zeros(7, dtype=np.float32)

        deltas = np.diff(dG)
        rds_idx = int(np.argmax(deltas))

        return np.array([
            float(len(dG)),                    # n_steps
            float(dG.min()),                   # min dG
            float(dG.max()),                   # max dG
            float(deltas[rds_idx]),            # RDS barrier
            float(rds_idx),                    # RDS index
            float(data.get("overpotential_V", data.get("overpotential", 0))),
            float(data.get("U_limiting_V", data.get("limiting_potential", 0))),
        ], dtype=np.float32)
    except (ValueError, KeyError, TypeError) as e:
        log.warning("Thermo feature computation failed: %s", e)
        return np.zeros(7, dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature Store
# ---------------------------------------------------------------------------

class FeatureStore:
    """
    Central feature registry, compute engine, and serving layer.

    Responsibilities:
    1. Register feature definitions (schema + compute function)
    2. Compute features on demand with caching
    3. Serve features for online (inference) and offline (training) use
    4. Track lineage (which inputs → which features → which predictions)
    5. Detect feature drift (distribution shift over time)
    """

    def __init__(self):
        self._definitions: Dict[str, FeatureDefinition] = {}
        self._cache: Dict[str, FeatureValue] = {}        # (feature_name:entity_id) → value
        self._lineage: List[FeatureLineage] = []
        self._drift_baselines: Dict[str, np.ndarray] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register all built-in ChatDFT features."""
        defaults = [
            FeatureDefinition(
                name="voronoi_coordination",
                entity_type="structure",
                dtype="float32",
                shape=(None, 6),
                version="1.0.0",
                compute_fn=compute_voronoi_features,
                description="Voronoi topology graph node features [Z, layer, CN, vol, dist, var]",
            ),
            FeatureDefinition(
                name="scf_convergence",
                entity_type="job",
                dtype="float32",
                shape=(9,),
                version="1.0.0",
                compute_fn=compute_scf_features,
                description="SCF trajectory features [rate, predicted_step, R2, sloshing, freq, amp, decay, n_steps, converged]",
            ),
            FeatureDefinition(
                name="mechanism_graph",
                entity_type="hypothesis",
                dtype="float32",
                shape=(8,),
                version="1.0.0",
                compute_fn=compute_mechanism_features,
                description="Reaction network features [n_inter, n_steps, n_ts, n_coads, n_surf, n_gas, n_ec, frac_ec]",
            ),
            FeatureDefinition(
                name="thermo_profile",
                entity_type="session",
                dtype="float32",
                shape=(7,),
                version="1.0.0",
                compute_fn=compute_thermo_features,
                description="Free energy profile features [n_steps, min_dG, max_dG, rds_barrier, rds_idx, eta, U_lim]",
            ),
        ]
        for fd in defaults:
            self._definitions[fd.name] = fd

    # --- Registration --------------------------------------------------------

    def register_feature(self, definition: FeatureDefinition):
        self._definitions[definition.name] = definition
        log.info("Registered feature: %s v%s", definition.name, definition.version)

    def get_definition(self, name: str) -> Optional[FeatureDefinition]:
        return self._definitions.get(name)

    def list_features(self) -> List[FeatureDefinition]:
        return list(self._definitions.values())

    # --- Compute + Cache -----------------------------------------------------

    def compute(self, feature_name: str, entity_id: str,
                raw_input: str, metadata: Dict[str, Any] = None) -> np.ndarray:
        """Compute a feature, cache it, and record lineage."""
        defn = self._definitions.get(feature_name)
        if defn is None:
            raise KeyError(f"Feature '{feature_name}' not registered")
        if defn.compute_fn is None:
            raise ValueError(f"Feature '{feature_name}' has no compute function")

        # Check cache (with TTL)
        cache_key = f"{feature_name}:{entity_id}"
        cached = self._cache.get(cache_key)
        if cached is not None and defn.ttl_seconds > 0:
            age = time.time() - datetime.fromisoformat(
                cached.computed_at.rstrip("Z")
            ).timestamp()
            if age < defn.ttl_seconds:
                return cached.value

        # Compute
        t0 = time.time()
        value = defn.compute_fn(raw_input)
        elapsed_ms = int((time.time() - t0) * 1000)

        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float32)

        input_hash = hashlib.sha256(raw_input.encode()).hexdigest()[:16]

        # Cache
        fv = FeatureValue(
            feature_name=feature_name,
            feature_version=defn.version,
            entity_type=defn.entity_type,
            entity_id=entity_id,
            value=value,
            input_hash=input_hash,
            compute_time_ms=elapsed_ms,
            metadata=metadata or {},
        )
        self._cache[cache_key] = fv

        # Lineage
        self._lineage.append(FeatureLineage(
            feature_name=feature_name,
            entity_id=entity_id,
            input_hash=input_hash,
            input_type=defn.entity_type,
            model_version=defn.version,
        ))

        log.debug("Computed %s for %s in %dms", feature_name, entity_id, elapsed_ms)
        return value

    def get(self, feature_name: str, entity_id: str) -> Optional[np.ndarray]:
        """Retrieve a cached feature value."""
        cache_key = f"{feature_name}:{entity_id}"
        fv = self._cache.get(cache_key)
        return fv.value if fv is not None else None

    def get_batch(self, feature_name: str, entity_ids: List[str]) -> Dict[str, np.ndarray]:
        """Retrieve features for multiple entities."""
        return {
            eid: self.get(feature_name, eid)
            for eid in entity_ids
            if self.get(feature_name, eid) is not None
        }

    # --- Lineage -------------------------------------------------------------

    def get_lineage(self, entity_id: str) -> List[FeatureLineage]:
        """Return all lineage records for an entity."""
        return [l for l in self._lineage if l.entity_id == entity_id]

    def get_lineage_for_feature(self, feature_name: str,
                                 last_n: int = 100) -> List[FeatureLineage]:
        return [l for l in self._lineage if l.feature_name == feature_name][-last_n:]

    def trace_provenance(self, entity_id: str) -> Dict[str, Any]:
        """
        Full provenance chain for an entity:
        raw_input → features → model prediction → output
        """
        lineage = self.get_lineage(entity_id)
        features = {}
        for l in lineage:
            fv = self._cache.get(f"{l.feature_name}:{entity_id}")
            features[l.feature_name] = {
                "version": l.model_version,
                "input_hash": l.input_hash,
                "computed_at": l.timestamp,
                "shape": fv.value.shape if fv else None,
            }
        return {
            "entity_id": entity_id,
            "features": features,
            "n_lineage_records": len(lineage),
        }

    # --- Drift Detection -----------------------------------------------------

    def set_drift_baseline(self, feature_name: str, values: np.ndarray):
        """Set the baseline distribution for drift detection."""
        self._drift_baselines[feature_name] = values

    def check_drift(self, feature_name: str,
                     recent_n: int = 50,
                     threshold: float = 0.2) -> Optional[Dict[str, Any]]:
        """
        Compare recent feature values to baseline using KS test on each dimension.
        Returns drift report if any dimension exceeds threshold.
        """
        from scipy import stats

        baseline = self._drift_baselines.get(feature_name)
        if baseline is None:
            return None

        # Collect recent values
        recent_values = []
        for key, fv in self._cache.items():
            if fv.feature_name == feature_name:
                recent_values.append(fv.value.flatten())
        if len(recent_values) < recent_n // 2:
            return None

        recent = np.array(recent_values[-recent_n:])
        if recent.ndim == 1:
            recent = recent.reshape(-1, 1)
        if baseline.ndim == 1:
            baseline = baseline.reshape(-1, 1)

        n_dims = min(recent.shape[1], baseline.shape[1])
        drifted_dims = []
        for d in range(n_dims):
            ks_stat, p_value = stats.ks_2samp(baseline[:, d], recent[:, d])
            if ks_stat > threshold:
                drifted_dims.append({
                    "dimension": d,
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_value),
                })

        if drifted_dims:
            return {
                "feature_name": feature_name,
                "drifted": True,
                "n_drifted_dimensions": len(drifted_dims),
                "details": drifted_dims,
            }
        return None

    # --- Summary -------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Feature Store: {len(self._definitions)} features, "
            f"{len(self._cache)} cached values, {len(self._lineage)} lineage records"
        ]
        for defn in self._definitions.values():
            n_cached = sum(1 for k in self._cache if k.startswith(defn.name + ":"))
            lines.append(
                f"  {defn.name} v{defn.version} ({defn.entity_type}) "
                f"shape={defn.shape} — {n_cached} cached"
            )
        return "\n".join(lines)


# Singleton
feature_store = FeatureStore()
