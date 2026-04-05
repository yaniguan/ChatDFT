"""
Global Seed Manager for Reproducibility
=========================================

Ensures deterministic results across all ChatDFT modules.

Usage
-----
    from science.core.seeds import set_global_seed, get_rng

    set_global_seed(42)          # call once at script entry
    rng = get_rng("surface_graph")  # per-module RNG (derived from global seed)

Design
------
Each module gets its own RNG derived from a hash of (global_seed, module_name),
so adding a new module doesn't change existing modules' random sequences.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

import numpy as np

_GLOBAL_SEED: Optional[int] = None
_MODULE_RNGS: Dict[str, np.random.Generator] = {}


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set the global seed for all ChatDFT modules.

    Parameters
    ----------
    seed : int
        Master seed.
    deterministic : bool
        If True, also set numpy's legacy RNG and attempt to set
        PyTorch deterministic mode.
    """
    global _GLOBAL_SEED, _MODULE_RNGS
    _GLOBAL_SEED = seed
    _MODULE_RNGS.clear()

    np.random.seed(seed)

    if deterministic:
        try:
            import torch

            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass


def get_global_seed() -> Optional[int]:
    """Return the current global seed, or None if not set."""
    return _GLOBAL_SEED


def get_rng(module_name: str) -> np.random.Generator:
    """
    Get a per-module RNG derived from the global seed.

    If no global seed is set, returns a non-deterministic RNG
    and logs a warning.

    Parameters
    ----------
    module_name : str
        Module identifier (e.g., ``"surface_graph"``).

    Returns
    -------
    np.random.Generator
        Module-specific RNG instance.
    """
    if module_name in _MODULE_RNGS:
        return _MODULE_RNGS[module_name]

    if _GLOBAL_SEED is not None:
        # Derive module-specific seed via hash
        h = hashlib.sha256(f"{_GLOBAL_SEED}:{module_name}".encode()).hexdigest()
        module_seed = int(h[:8], 16)
        rng = np.random.default_rng(module_seed)
    else:
        rng = np.random.default_rng()

    _MODULE_RNGS[module_name] = rng
    return rng


def experiment_manifest() -> dict:
    """
    Return a reproducibility manifest for the current experiment.

    Includes: global seed, numpy version, torch version (if available),
    active module RNGs.
    """
    manifest = {
        "global_seed": _GLOBAL_SEED,
        "numpy_version": np.__version__,
        "active_modules": list(_MODULE_RNGS.keys()),
    }
    try:
        import torch

        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        manifest["torch_version"] = None
    try:
        import scipy

        manifest["scipy_version"] = scipy.__version__
    except ImportError:
        pass
    return manifest
