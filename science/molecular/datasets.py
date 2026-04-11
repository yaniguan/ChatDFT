"""
Molecular Datasets — MoleculeNet Loaders, Scaffold Splitting, Class Balancing
=============================================================================

Provides standardised access to MoleculeNet benchmark datasets with proper
evaluation methodology:

1. **Scaffold splitting** — split by Bemis-Murcko scaffold to test genuine
   generalisation to unseen chemical series (not random splitting, which
   inflates metrics). Reference: Wu et al., Chem. Sci. 9, 513 (2018)

2. **Imbalanced data handling** — SMOTE oversampling, class-weighted sampling,
   and focal loss support for datasets like Tox21 where positive:negative
   ratios can be 1:20+.

3. **Built-in datasets** — BBBP, Tox21, ClinTox, BACE, ESOL, FreeSolv,
   Lipophilicity, HIV, SIDER, MUV from MoleculeNet/DeepChem.

Dataset statistics (for interview reference):
  BBBP:          2039 molecules, binary (blood-brain barrier penetration)
  Tox21:        ~8000 molecules, 12 binary tasks (nuclear receptor/stress response)
  ClinTox:       1478 molecules, 2 binary tasks (clinical trial toxicity)
  BACE:          1513 molecules, binary (BACE-1 inhibitor)
  ESOL:          1128 molecules, regression (aqueous solubility, logS)
  FreeSolv:       642 molecules, regression (hydration free energy, kcal/mol)
  Lipophilicity: 4200 molecules, regression (octanol/water partition, logD)
  HIV:          41127 molecules, binary (HIV replication inhibition)
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    """Metadata for a MoleculeNet dataset."""
    name: str
    task_type: str           # "classification" or "regression"
    n_tasks: int             # number of prediction targets
    task_names: List[str]
    metric: str              # primary evaluation metric
    url: str                 # DeepChem CSV URL
    description: str = ""
    n_molecules: int = 0     # populated after loading
    class_balance: Dict[str, float] = field(default_factory=dict)


DATASET_REGISTRY: Dict[str, DatasetInfo] = {
    "bbbp": DatasetInfo(
        name="bbbp",
        task_type="classification",
        n_tasks=1,
        task_names=["p_np"],
        metric="auroc",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        description="Blood-brain barrier penetration (Martins et al., 2012)",
    ),
    "tox21": DatasetInfo(
        name="tox21",
        task_type="classification",
        n_tasks=12,
        task_names=[
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
            "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
            "SR-HSE", "SR-MMP", "SR-p53",
        ],
        metric="auroc",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        description="Toxicology in the 21st century (Tox21 Challenge)",
    ),
    "clintox": DatasetInfo(
        name="clintox",
        task_type="classification",
        n_tasks=2,
        task_names=["FDA_APPROVED", "CT_TOX"],
        metric="auroc",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        description="Clinical trial toxicity (Gayvert et al., 2016)",
    ),
    "bace": DatasetInfo(
        name="bace",
        task_type="classification",
        n_tasks=1,
        task_names=["Class"],
        metric="auroc",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        description="BACE-1 inhibitors for Alzheimer's (Subramanian et al., 2016)",
    ),
    "esol": DatasetInfo(
        name="esol",
        task_type="regression",
        n_tasks=1,
        task_names=["measured log solubility in mols per litre"],
        metric="rmse",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        description="Aqueous solubility (Delaney, 2004)",
    ),
    "freesolv": DatasetInfo(
        name="freesolv",
        task_type="regression",
        n_tasks=1,
        task_names=["expt"],
        metric="rmse",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        description="Hydration free energy (Mobley & Guthrie, 2014)",
    ),
    "lipophilicity": DatasetInfo(
        name="lipophilicity",
        task_type="regression",
        n_tasks=1,
        task_names=["exp"],
        metric="rmse",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        description="Octanol/water distribution coefficient logD (Hersey, ChEMBL)",
    ),
    "hiv": DatasetInfo(
        name="hiv",
        task_type="classification",
        n_tasks=1,
        task_names=["HIV_active"],
        metric="auroc",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        description="HIV replication inhibition (AIDS Antiviral Screen)",
    ),
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MoleculeDatapoint:
    """Single molecule with SMILES and target(s)."""
    smiles: str
    targets: np.ndarray       # shape (n_tasks,) — NaN for missing labels
    weight: float = 1.0       # sample weight (for class balancing)

    @property
    def is_valid(self) -> bool:
        return len(self.smiles) > 0 and not all(np.isnan(self.targets))


@dataclass
class MoleculeDataset:
    """A dataset of molecules with metadata."""
    data: List[MoleculeDatapoint]
    info: DatasetInfo
    split: str = "full"       # "full", "train", "val", "test"

    @property
    def smiles(self) -> List[str]:
        return [d.smiles for d in self.data]

    @property
    def targets(self) -> np.ndarray:
        return np.array([d.targets for d in self.data], dtype=np.float32)

    @property
    def weights(self) -> np.ndarray:
        return np.array([d.weight for d in self.data], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def summary(self) -> str:
        n = len(self.data)
        t = self.targets
        lines = [
            f"Dataset: {self.info.name} ({self.split})",
            f"  Molecules: {n}",
            f"  Tasks: {self.info.n_tasks} ({self.info.task_type})",
            f"  Metric: {self.info.metric}",
        ]
        if self.info.task_type == "classification":
            for i, name in enumerate(self.info.task_names):
                valid = ~np.isnan(t[:, i])
                pos = np.nansum(t[:, i] == 1)
                neg = np.nansum(t[:, i] == 0)
                ratio = pos / max(neg, 1)
                lines.append(
                    f"  {name}: {int(pos)} pos / {int(neg)} neg "
                    f"(ratio={ratio:.3f}, missing={int(n - valid.sum())})"
                )
        else:
            for i, name in enumerate(self.info.task_names):
                valid = t[~np.isnan(t[:, i]), i]
                if len(valid) > 0:
                    lines.append(
                        f"  {name}: mean={valid.mean():.3f}, "
                        f"std={valid.std():.3f}, "
                        f"range=[{valid.min():.3f}, {valid.max():.3f}]"
                    )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_DATA_CACHE_DIR = Path(os.environ.get("CHATDFT_DATA_DIR", "./data/molecular"))


def _download_csv(url: str, name: str) -> str:
    """Download a CSV file, cache locally. Return file path."""
    cache_dir = _DATA_CACHE_DIR / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ext = ".csv.gz" if url.endswith(".gz") else ".csv"
    path = cache_dir / f"{name}{ext}"

    if path.exists():
        log.debug("Using cached %s", path)
    else:
        log.info("Downloading %s from %s", name, url)
        import urllib.request
        urllib.request.urlretrieve(url, str(path))
        log.info("Saved to %s", path)

    return str(path)


def _read_csv(path: str) -> List[Dict[str, str]]:
    """Read CSV (supports .gz)."""
    import gzip

    if path.endswith(".gz"):
        f = gzip.open(path, "rt", encoding="utf-8")
    else:
        f = open(path, "r", encoding="utf-8")

    with f:
        reader = csv.DictReader(f)
        return list(reader)


def _find_smiles_column(columns: List[str]) -> str:
    """Find the SMILES column in a CSV header."""
    for name in ["smiles", "SMILES", "Smiles", "mol", "canonical_smiles"]:
        if name in columns:
            return name
    # Fallback: first column that looks like SMILES
    return columns[0]


def load_dataset(
    name: str,
    data_dir: Optional[str] = None,
) -> MoleculeDataset:
    """
    Load a MoleculeNet dataset.

    Parameters
    ----------
    name : str
        Dataset name (bbbp, tox21, esol, etc.)
    data_dir : str, optional
        Override data directory.

    Returns
    -------
    MoleculeDataset
    """
    name = name.lower()
    info = DATASET_REGISTRY.get(name)
    if info is None:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    if data_dir:
        global _DATA_CACHE_DIR
        _DATA_CACHE_DIR = Path(data_dir)

    path = _download_csv(info.url, name)
    rows = _read_csv(path)

    if not rows:
        raise ValueError(f"Empty dataset: {name}")

    columns = list(rows[0].keys())
    smiles_col = _find_smiles_column(columns)

    data = []
    for row in rows:
        smi = row.get(smiles_col, "").strip()
        if not smi:
            continue

        targets = []
        for task_name in info.task_names:
            val = row.get(task_name, "").strip()
            if val == "" or val.lower() == "nan":
                targets.append(float("nan"))
            else:
                try:
                    targets.append(float(val))
                except ValueError:
                    targets.append(float("nan"))

        data.append(MoleculeDatapoint(
            smiles=smi,
            targets=np.array(targets, dtype=np.float32),
        ))

    info.n_molecules = len(data)

    # Compute class balance for classification
    if info.task_type == "classification":
        for i, task_name in enumerate(info.task_names):
            vals = [d.targets[i] for d in data if not np.isnan(d.targets[i])]
            if vals:
                pos = sum(1 for v in vals if v == 1)
                neg = sum(1 for v in vals if v == 0)
                info.class_balance[task_name] = pos / max(neg, 1)

    dataset = MoleculeDataset(data=data, info=info, split="full")
    log.info("Loaded %s: %d molecules", name, len(data))
    return dataset


# ---------------------------------------------------------------------------
# Scaffold splitting
# ---------------------------------------------------------------------------

def scaffold_split(
    dataset: MoleculeDataset,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    """
    Split dataset by Bemis-Murcko scaffold.

    Molecules sharing the same scaffold go into the same split.
    This tests generalisation to unseen chemical scaffolds — a much
    harder and more realistic evaluation than random splitting.

    Reference: Wu et al., Chem. Sci. 9, 513 (2018)

    Parameters
    ----------
    dataset : MoleculeDataset
    train_frac, val_frac, test_frac : float
        Target split fractions.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (train, val, test) MoleculeDataset tuple.
    """
    from science.molecular.representations import get_scaffold

    # Group molecules by scaffold
    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, dp in enumerate(dataset.data):
        scaf = get_scaffold(dp.smiles)
        if scaf is None:
            scaf = dp.smiles  # Use SMILES itself as fallback
        scaffold_to_indices[scaf].append(i)

    # Shuffle scaffold order with seed for reproducibility, then fill bins.
    # This ensures a more balanced class distribution across splits compared
    # to deterministic largest-first ordering.
    rng = np.random.default_rng(seed)
    scaffold_sets = list(scaffold_to_indices.values())
    rng.shuffle(scaffold_sets)

    n = len(dataset.data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx, val_idx, test_idx = [], [], []
    for s_set in scaffold_sets:
        if len(train_idx) + len(s_set) <= n_train:
            train_idx.extend(s_set)
        elif len(val_idx) + len(s_set) <= n_val:
            val_idx.extend(s_set)
        else:
            test_idx.extend(s_set)

    # Shuffle within each split
    rng = np.random.default_rng(seed)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    def _subset(indices, split_name):
        return MoleculeDataset(
            data=[dataset.data[i] for i in indices],
            info=dataset.info,
            split=split_name,
        )

    train = _subset(train_idx, "train")
    val = _subset(val_idx, "val")
    test = _subset(test_idx, "test")

    log.info(
        "Scaffold split: train=%d, val=%d, test=%d (scaffolds=%d)",
        len(train), len(val), len(test), len(scaffold_sets),
    )
    return train, val, test


def random_split(
    dataset: MoleculeDataset,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    """Random split (baseline comparison — use scaffold_split for evaluation)."""
    n = len(dataset.data)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    def _subset(indices, split_name):
        return MoleculeDataset(
            data=[dataset.data[i] for i in indices],
            info=dataset.info,
            split=split_name,
        )

    return (
        _subset(idx[:n_train], "train"),
        _subset(idx[n_train:n_train + n_val], "val"),
        _subset(idx[n_train + n_val:], "test"),
    )


# ---------------------------------------------------------------------------
# Class balancing for imbalanced datasets
# ---------------------------------------------------------------------------

def compute_class_weights(
    dataset: MoleculeDataset, task_idx: int = 0
) -> Dict[int, float]:
    """
    Compute inverse-frequency class weights.

    For Tox21 where positive:negative can be 1:20, this gives
    weight_pos = n_neg / n_total, weight_neg = n_pos / n_total.
    """
    targets = dataset.targets[:, task_idx]
    valid = targets[~np.isnan(targets)]
    counts = Counter(valid.astype(int).tolist())
    total = sum(counts.values())
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    return weights


def apply_class_weights(
    dataset: MoleculeDataset, task_idx: int = 0
) -> MoleculeDataset:
    """Apply inverse-frequency weights to each sample."""
    weights = compute_class_weights(dataset, task_idx)
    for dp in dataset.data:
        if not np.isnan(dp.targets[task_idx]):
            dp.weight = weights.get(int(dp.targets[task_idx]), 1.0)
    return dataset


def smote_oversample(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTE oversampling for imbalanced binary classification.

    Synthetic Minority Over-sampling Technique (Chawla et al., 2002).
    Generates synthetic minority samples by interpolating between
    nearest neighbours in feature space.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,) — binary labels {0, 1}
    k_neighbors : int
        Number of nearest neighbours for interpolation.
    seed : int

    Returns
    -------
    (X_resampled, y_resampled) with balanced classes.
    """
    rng = np.random.default_rng(seed)

    # Identify minority/majority
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return X, y

    minority_cls = classes[np.argmin(counts)]
    majority_cls = classes[np.argmax(counts)]
    n_majority = counts.max()
    n_minority = counts.min()
    n_synthetic = n_majority - n_minority

    if n_synthetic == 0:
        return X, y

    minority_X = X[y == minority_cls]
    k = min(k_neighbors, len(minority_X) - 1)
    if k < 1:
        return X, y

    # Find k nearest neighbours for each minority sample
    from scipy.spatial.distance import cdist
    dists = cdist(minority_X, minority_X, metric="euclidean")
    np.fill_diagonal(dists, np.inf)
    nn_indices = np.argsort(dists, axis=1)[:, :k]

    # Generate synthetic samples
    synthetic = []
    for _ in range(n_synthetic):
        idx = rng.integers(len(minority_X))
        nn_idx = rng.choice(nn_indices[idx])
        lam = rng.random()
        new_sample = minority_X[idx] + lam * (minority_X[nn_idx] - minority_X[idx])
        synthetic.append(new_sample)

    synthetic = np.array(synthetic)
    X_new = np.vstack([X, synthetic])
    y_new = np.concatenate([y, np.full(n_synthetic, minority_cls)])

    # Shuffle
    perm = rng.permutation(len(X_new))
    return X_new[perm], y_new[perm]


# ---------------------------------------------------------------------------
# Focal loss (for extremely imbalanced datasets like Tox21, MUV)
# ---------------------------------------------------------------------------

def focal_loss_weights(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> np.ndarray:
    """
    Compute focal loss sample weights.

    Focal loss down-weights well-classified examples and focuses
    on hard, misclassified ones. Critical for datasets like Tox21
    where 95%+ of samples are negative.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference: Lin et al., ICCV 2017 (originally for object detection,
    widely adopted for imbalanced QSAR).

    Returns per-sample weights to multiply with BCE loss.
    """
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    p_t = np.clip(p_t, 1e-7, 1 - 1e-7)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return focal_weight


# ---------------------------------------------------------------------------
# Quick dataset summary for benchmarking
# ---------------------------------------------------------------------------

def list_datasets() -> List[str]:
    """Return available dataset names."""
    return list(DATASET_REGISTRY.keys())


def dataset_summary() -> str:
    """Print a markdown table of all available datasets."""
    lines = [
        "| Dataset | Type | Tasks | Metric | Description |",
        "|---------|------|-------|--------|-------------|",
    ]
    for info in DATASET_REGISTRY.values():
        lines.append(
            f"| {info.name} | {info.task_type} | {info.n_tasks} | "
            f"{info.metric} | {info.description} |"
        )
    return "\n".join(lines)
