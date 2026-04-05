"""
Unified Energy Predictor — Train, Compare, and Benchmark GNN Models
====================================================================

Scientific motivation
---------------------
Adsorption energy prediction is the central task in computational catalysis
screening.  Given a surface topology graph (from Voronoi tessellation),
we want to predict the binding energy E_ads in eV.

This module provides:
  1. A synthetic adsorption energy generator grounded in DFT scaling relations
  2. A unified train/evaluate API across all 6 architectures
  3. Head-to-head benchmark comparison on identical data splits

The synthetic data encodes known physical trends:
  - d-band centre model: E_ads correlates with metal identity
  - Coordination effect: under-coordinated sites bind more strongly
  - Adsorbate dependence: *O, *OH, *OOH follow scaling relations
  - Gaussian noise ~ 0.1 eV mimics DFT-level uncertainty

Key references
--------------
[1] Hammer & Norskov, Adv. Catal. 45, 71 (2000) — d-band model
[2] Calle-Vallejo et al., Chem. Sci. 6, 3218 (2015) — coordination-activity
[3] Man et al., ChemCatChem 3, 1159 (2011) — OER scaling relations
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from science.core.constants import (
    ADSORBATE_OFFSETS,
    CN_BINDING_SLOPE,
    D_BAND_CENTRES,
    D_BAND_COUPLING,
    FCC_LATTICE_CONSTANTS,
)
from science.core.logging import get_logger

logger = get_logger(__name__)
from science.predictions.gnn_models import (  # noqa: E402
    GraphData,
    _check_torch,
    build_model,
    list_models,
)
from science.representations.surface_graph import SurfaceTopologyGraph  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic adsorption energy dataset
# ---------------------------------------------------------------------------

# Sourced from science.core.constants (Hammer & Norskov 2000, CRC Handbook)
_D_BAND = D_BAND_CENTRES
_ADS_OFFSET = ADSORBATE_OFFSETS
_CN_SLOPE = CN_BINDING_SLOPE


def _generate_slab(
    element: str, n_atoms: int = 8, rng: np.random.Generator = None
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Generate a synthetic slab with realistic lattice parameters.
    Returns (positions, elements, cell).
    """
    if rng is None:
        rng = np.random.default_rng()

    a = FCC_LATTICE_CONSTANTS.get(element, 3.80) + rng.normal(0, 0.01)
    d = a / np.sqrt(2)

    # 2-layer slab with 4 atoms per layer
    n_per_layer = max(n_atoms // 2, 2)
    nx_atoms = int(np.ceil(np.sqrt(n_per_layer)))
    ny_atoms = int(np.ceil(n_per_layer / nx_atoms))

    positions = []
    for layer in range(2):
        z = 10.0 + layer * d * np.sqrt(2) / np.sqrt(3)
        offset = d / 2 * layer
        for ix in range(nx_atoms):
            for iy in range(ny_atoms):
                if len(positions) >= n_atoms:
                    break
                x = ix * d + offset + rng.normal(0, 0.02)
                y = iy * d * np.sqrt(3) / 2 + rng.normal(0, 0.02)
                positions.append([x, y, z])

    positions = np.array(positions[:n_atoms], dtype=np.float64)
    elements = [element] * n_atoms
    cell = np.array(
        [
            [nx_atoms * d, 0, 0],
            [d / 2, ny_atoms * d * np.sqrt(3) / 2, 0],
            [0, 0, 25.0],
        ],
        dtype=np.float64,
    )

    return positions, elements, cell


def synthetic_adsorption_energy(
    element: str,
    adsorbate: str = "*OH",
    cn: float = 9.0,
    noise_std: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Generate synthetic adsorption energy based on DFT scaling relations.

    E_ads = alpha * eps_d + beta_ads + gamma * (CN - 9) + noise

    Parameters
    ----------
    element : str
        Metal element (e.g. 'Cu', 'Pt').
    adsorbate : str
        Adsorbate species (e.g. '*OH', '*O').
    cn : float
        Average coordination number of the binding site.
    noise_std : float
        Gaussian noise standard deviation (eV).

    Returns
    -------
    float
        Adsorption energy in eV (negative = exothermic binding).
    """
    if rng is None:
        rng = np.random.default_rng()

    eps_d = _D_BAND.get(element, -2.0)
    ads_off = _ADS_OFFSET.get(adsorbate, 0.0)

    E = D_BAND_COUPLING * eps_d + ads_off + _CN_SLOPE * (cn - 9.0)
    E += rng.normal(0, noise_std)
    return float(E)


@dataclass
class AdsorptionSample:
    """Single training/test sample."""

    element: str
    adsorbate: str
    positions: np.ndarray  # (N, 3)
    elements: list  # [str] * N
    cell: np.ndarray  # (3, 3)
    energy: float  # eV
    cn_mean: float  # average coordination number


def generate_dataset(
    n_samples: int = 200,
    elements: Optional[list] = None,
    adsorbates: Optional[list] = None,
    seed: int = 42,
    n_atoms: int = 8,
) -> list[AdsorptionSample]:
    """
    Generate a synthetic adsorption energy dataset.

    Returns list of AdsorptionSample with Voronoi-ready structures
    and DFT-proxy energies.
    """
    rng = np.random.default_rng(seed)
    if elements is None:
        elements = ["Cu", "Pt", "Ag", "Au", "Ni", "Pd"]
    if adsorbates is None:
        adsorbates = ["*H", "*O", "*OH", "*CO"]

    samples = []
    for _ in range(n_samples):
        elem = rng.choice(elements)
        ads = rng.choice(adsorbates)
        pos, elems, cell = _generate_slab(elem, n_atoms, rng)

        # Build graph to get actual CN
        stg = SurfaceTopologyGraph(pos, elems, cell)
        stg.build()
        cn_values = [n.coordination for n in stg.nodes]
        cn_mean = float(np.mean(cn_values)) if cn_values else 9.0

        energy = synthetic_adsorption_energy(elem, ads, cn_mean, rng=rng)

        samples.append(
            AdsorptionSample(
                element=elem,
                adsorbate=ads,
                positions=pos,
                elements=elems,
                cell=cell,
                energy=energy,
                cn_mean=cn_mean,
            )
        )
    return samples


def samples_to_graphs(
    samples: list[AdsorptionSample],
) -> list[GraphData]:
    """Convert AdsorptionSamples to GraphData list."""
    _check_torch()
    graphs = []
    for i, s in enumerate(samples):
        stg = SurfaceTopologyGraph(s.positions, s.elements, s.cell)
        stg.build()
        X = stg.node_feature_matrix()
        ei, ea = stg.edge_index_and_attr()
        g = GraphData.from_numpy(X, ei, ea, s.positions, y=s.energy, batch_idx=0)
        graphs.append(g)
    return graphs


# ---------------------------------------------------------------------------
# Batching utility
# ---------------------------------------------------------------------------


def collate_graphs(graphs: list[GraphData]) -> GraphData:
    """Collate a list of single-graph GraphData into a batched GraphData."""
    xs, eis, eas, poss, batches, ys = [], [], [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        n = g.x.shape[0]
        xs.append(g.x)
        eis.append(g.edge_index + offset)
        eas.append(g.edge_attr)
        poss.append(g.pos)
        batches.append(torch.full((n,), i, dtype=torch.long))
        if g.y is not None:
            ys.append(g.y)
        offset += n

    return GraphData(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(eis, dim=1),
        edge_attr=torch.cat(eas, dim=0),
        pos=torch.cat(poss, dim=0),
        batch=torch.cat(batches, dim=0),
        y=torch.cat(ys, dim=0) if ys else None,
    )


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Result of a single model training run."""

    model_name: str
    train_mae: float  # eV
    val_mae: float  # eV
    test_mae: float  # eV
    train_time_s: float  # seconds
    n_params: int
    loss_curve: list[float] = field(default_factory=list)


def train_and_evaluate(
    model_name: str,
    train_graphs: list[GraphData],
    val_graphs: list[GraphData],
    test_graphs: list[GraphData],
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 16,
    verbose: bool = False,
    **model_kwargs,
) -> TrainResult:
    """
    Train a model and evaluate on val/test sets.

    Parameters
    ----------
    model_name : str
        One of: mlp, mpnn, gat, schnet, dimenet, se3_transformer
    train_graphs, val_graphs, test_graphs : list[GraphData]
        Pre-split datasets.
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    TrainResult
        MAE on train/val/test, timing, parameter count, loss curve.
    """
    _check_torch()

    model = build_model(model_name, **model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    loss_curve = []
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        # Shuffle and mini-batch
        indices = torch.randperm(len(train_graphs))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_graphs), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = collate_graphs([train_graphs[i] for i in batch_idx])

            pred = model(batch)
            loss = F.l1_loss(pred, batch.y) if hasattr(F, "l1_loss") else (pred - batch.y).abs().mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_curve.append(avg_loss)

        # Validation MAE for scheduler
        if epoch % 5 == 0:
            val_mae = _evaluate_mae(model, val_graphs, batch_size)
            scheduler.step(val_mae)
            if verbose and epoch % 20 == 0:
                print(f"  [{model_name}] epoch {epoch:3d}  train_loss={avg_loss:.4f}  val_mae={val_mae:.4f}")

    train_time = time.time() - t0

    train_mae = _evaluate_mae(model, train_graphs, batch_size)
    val_mae = _evaluate_mae(model, val_graphs, batch_size)
    test_mae = _evaluate_mae(model, test_graphs, batch_size)

    return TrainResult(
        model_name=model_name,
        train_mae=train_mae,
        val_mae=val_mae,
        test_mae=test_mae,
        train_time_s=train_time,
        n_params=sum(p.numel() for p in model.parameters()),
        loss_curve=loss_curve,
    )


def _evaluate_mae(model: nn.Module, graphs: list[GraphData], batch_size: int = 16) -> float:
    """Compute MAE over a set of graphs."""
    model.eval()
    total_ae = 0.0
    n = 0
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = collate_graphs(graphs[start : start + batch_size])
            pred = model(batch)
            total_ae += (pred - batch.y).abs().sum().item()
            n += pred.shape[0]
    return total_ae / max(n, 1)


# ---------------------------------------------------------------------------
# Full benchmark comparison
# ---------------------------------------------------------------------------


def benchmark_all_models(
    n_samples: int = 200,
    n_epochs: int = 100,
    seed: int = 42,
    verbose: bool = False,
) -> list[TrainResult]:
    """
    Train and evaluate all 6 models on the same synthetic dataset.

    Returns sorted list of TrainResult (best test_mae first).
    """
    _check_torch()

    print("Generating synthetic adsorption energy dataset...")
    samples = generate_dataset(n_samples=n_samples, seed=seed)

    # 70/15/15 split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_train = int(0.7 * len(samples))
    n_val = int(0.15 * len(samples))

    train_samples = [samples[i] for i in idx[:n_train]]
    val_samples = [samples[i] for i in idx[n_train : n_train + n_val]]
    test_samples = [samples[i] for i in idx[n_train + n_val :]]

    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    print("Converting to graph data...")
    train_graphs = samples_to_graphs(train_samples)
    val_graphs = samples_to_graphs(val_samples)
    test_graphs = samples_to_graphs(test_samples)

    results = []
    for name in list_models():
        print(f"\nTraining {name}...")
        try:
            r = train_and_evaluate(
                name,
                train_graphs,
                val_graphs,
                test_graphs,
                n_epochs=n_epochs,
                verbose=verbose,
            )
            print(f"  {name}: test_mae={r.test_mae:.4f} eV, params={r.n_params:,}, time={r.train_time_s:.1f}s")
            results.append(r)
        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError if hasattr(torch, "cuda") else RuntimeError) as e:
            print(f"  {name} FAILED: {type(e).__name__}: {e}")

    results.sort(key=lambda r: r.test_mae)
    return results


def format_results_table(results: list[TrainResult]) -> str:
    """Format benchmark results as a markdown table."""
    lines = [
        "| Model | Test MAE (eV) | Val MAE (eV) | Params | Train Time |",
        "|-------|--------------|-------------|--------|------------|",
    ]
    for r in results:
        lines.append(
            f"| {r.model_name:16s} | {r.test_mae:.4f} | {r.val_mae:.4f} | {r.n_params:>7,} | {r.train_time_s:>6.1f}s |"
        )
    return "\n".join(lines)
