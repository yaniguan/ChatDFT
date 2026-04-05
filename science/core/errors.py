"""
Error Taxonomy for ChatDFT
===========================

Three-tier exception hierarchy so callers can catch at the right granularity:

    ChatDFTError            ← catch-all for any ChatDFT failure
    ├── DataError           ← invalid input data (structures, trajectories)
    │   ├── InvalidStructure
    │   ├── InvalidTrajectory
    │   └── InvalidReactionNetwork
    ├── ModelError          ← ML model failures (training, inference)
    │   ├── TrainingError
    │   └── InferenceError
    └── PhysicsError        ← physically impossible results
        ├── ThermodynamicError
        └── ConvergenceError

Usage
-----
    from science.core.errors import InvalidStructure, PhysicsError

    if cell_volume < 1e-6:
        raise InvalidStructure(
            "Cell volume near zero — degenerate lattice",
            context={"volume": cell_volume, "cell": cell.tolist()},
        )
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ChatDFTError(Exception):
    """Base exception for all ChatDFT failures."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base} [{ctx}]"
        return base


# ── Data errors ──────────────────────────────────────────────────────


class DataError(ChatDFTError):
    """Invalid input data."""


class InvalidStructure(DataError):
    """Atomic structure is malformed (NaN positions, zero cell, etc.)."""


class InvalidTrajectory(DataError):
    """SCF trajectory is malformed (empty, NaN values, etc.)."""


class InvalidReactionNetwork(DataError):
    """Reaction network has structural problems."""


# ── Model errors ─────────────────────────────────────────────────────


class ModelError(ChatDFTError):
    """ML model failure during training or inference."""


class TrainingError(ModelError):
    """Model training diverged or failed."""


class InferenceError(ModelError):
    """Model inference produced invalid output."""


# ── Physics errors ───────────────────────────────────────────────────


class PhysicsError(ChatDFTError):
    """Result violates physical constraints."""


class ThermodynamicError(PhysicsError):
    """Thermodynamic impossibility (e.g., negative entropy)."""


class ConvergenceError(PhysicsError):
    """Numerical procedure failed to converge."""
