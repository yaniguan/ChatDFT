"""
Molecular Representations
=========================

Three complementary molecular representations for ADME/QSAR prediction:

1. **Morgan fingerprints** (ECFP-like) — fixed-length binary/count vectors
   encoding circular substructures. Standard baseline for QSAR.
   Reference: Rogers & Hahn, J. Chem. Inf. Model. 50, 742 (2010)

2. **RDKit physicochemical descriptors** — 200+ computed properties
   (MW, logP, TPSA, HBD/HBA, rotatable bonds, etc.). Interpretable features.

3. **Molecular graphs** — atoms as nodes (element, degree, charge, aromaticity,
   chirality), bonds as edges (type, conjugation, ring membership).
   Used by GNN/MPNN models (Chemprop-style).
   Reference: Yang et al., J. Chem. Inf. Model. 59, 3370 (2019)

Each representation includes validation and handles edge cases (invalid SMILES,
disconnected fragments, empty molecules).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy RDKit import (not everyone has it installed)
# ---------------------------------------------------------------------------

_RDKIT_OK = False
try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem,
        Descriptors,
        rdMolDescriptors,
        Fragments,
    )
    from rdkit.Chem.Scaffolds import MurckoScaffold

    _RDKIT_OK = True
except ImportError:
    log.info("RDKit not installed — using fallback representations")
    Chem = None


def _require_rdkit():
    if not _RDKIT_OK:
        raise ImportError(
            "RDKit is required for molecular representations. "
            "Install with: pip install rdkit or conda install -c conda-forge rdkit"
        )


# ---------------------------------------------------------------------------
# SMILES validation
# ---------------------------------------------------------------------------


def validate_smiles(smiles: str) -> bool:
    """Check if a SMILES string is chemically valid."""
    if not smiles or not smiles.strip():
        return False
    if not _RDKIT_OK:
        return len(smiles) > 0
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def canonicalize(smiles: str) -> Optional[str]:
    """Return canonical SMILES or None if invalid."""
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def get_scaffold(smiles: str) -> Optional[str]:
    """Extract Bemis-Murcko scaffold for scaffold splitting."""
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(
            MurckoScaffold.GetScaffoldForMol(mol)
        )
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return Chem.MolToSmiles(mol)


# ---------------------------------------------------------------------------
# 1. Morgan Fingerprints (ECFP)
# ---------------------------------------------------------------------------

# Element features for fallback (no RDKit)
_ELEMENT_FEATURES = {
    "C": 0, "N": 1, "O": 2, "S": 3, "F": 4, "Cl": 5, "Br": 6, "I": 7,
    "P": 8, "Si": 9, "B": 10, "Se": 11, "Te": 12, "H": 13,
}


def morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_counts: bool = False,
) -> np.ndarray:
    """
    Compute Morgan fingerprint (ECFP-like).

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    radius : int
        Fingerprint radius (2 = ECFP4, 3 = ECFP6).
    n_bits : int
        Length of the bit vector.
    use_counts : bool
        If True, return count vector instead of binary.

    Returns
    -------
    np.ndarray
        Shape (n_bits,), dtype float32.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    if use_counts:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits
        )
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

    arr = np.zeros(n_bits, dtype=np.float32)
    on_bits = list(fp.GetOnBits())
    arr[on_bits] = 1.0
    return arr


def batch_morgan_fingerprints(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan fingerprints for a batch of SMILES. Shape: (N, n_bits)."""
    return np.array(
        [morgan_fingerprint(s, radius, n_bits) for s in smiles_list],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# 2. RDKit Physicochemical Descriptors
# ---------------------------------------------------------------------------

# Core descriptor set — interpretable and commonly used in ADME
DESCRIPTOR_NAMES = [
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "NumSaturatedRings", "NumHeteroatoms", "NumValenceElectrons",
    "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount",
    "RingCount", "MolMR", "LabuteASA", "BalabanJ", "BertzCT",
    "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
    "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v",
    "MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex",
    "MinEStateIndex",
]


def rdkit_descriptors(smiles: str) -> np.ndarray:
    """
    Compute physicochemical descriptors.

    Returns
    -------
    np.ndarray
        Shape (len(DESCRIPTOR_NAMES),), dtype float32.
        NaN values are replaced with 0.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    n = len(DESCRIPTOR_NAMES)
    if mol is None:
        return np.zeros(n, dtype=np.float32)

    desc_map = {name: func for name, func in Descriptors.descList}
    values = []
    for name in DESCRIPTOR_NAMES:
        func = desc_map.get(name)
        if func is not None:
            try:
                v = func(mol)
                values.append(float(v) if v is not None else 0.0)
            except Exception:
                values.append(0.0)
        else:
            values.append(0.0)

    arr = np.array(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def batch_rdkit_descriptors(smiles_list: List[str]) -> np.ndarray:
    """Compute RDKit descriptors for a batch. Shape: (N, n_descriptors)."""
    return np.array(
        [rdkit_descriptors(s) for s in smiles_list], dtype=np.float32
    )


# ---------------------------------------------------------------------------
# 3. Molecular Graphs (for GNN/MPNN)
# ---------------------------------------------------------------------------

# Atom features: [atomic_num_onehot(14), degree_onehot(6), formal_charge_onehot(5),
#                  hybridization_onehot(4), is_aromatic, is_in_ring, num_Hs_onehot(5)]
# Total: 35 features per atom

_ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Se", "Te", "H"]
_DEGREES = [0, 1, 2, 3, 4, 5]
_CHARGES = [-2, -1, 0, 1, 2]
_HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D"]
_NUM_HS = [0, 1, 2, 3, 4]

ATOM_FEATURE_DIM = len(_ATOM_SYMBOLS) + len(_DEGREES) + len(_CHARGES) + \
                   len(_HYBRIDIZATIONS) + 2 + len(_NUM_HS)  # 35

# Bond features: [bond_type_onehot(4), is_conjugated, is_in_ring, stereo_onehot(4)]
# Total: 10 features per bond
BOND_FEATURE_DIM = 10


def _one_hot(val, allowed_set):
    """One-hot encode a value against an allowed set."""
    vec = [0.0] * len(allowed_set)
    try:
        idx = allowed_set.index(val)
        vec[idx] = 1.0
    except ValueError:
        pass  # unknown value → all zeros
    return vec


@dataclass
class MolGraph:
    """
    Molecular graph representation for GNN input.

    Attributes
    ----------
    x : np.ndarray
        Node (atom) feature matrix, shape (n_atoms, ATOM_FEATURE_DIM).
    edge_index : np.ndarray
        Edge connectivity, shape (2, n_edges). Bidirectional.
    edge_attr : np.ndarray
        Edge (bond) features, shape (n_edges, BOND_FEATURE_DIM).
    smiles : str
        Original SMILES string.
    n_atoms : int
        Number of atoms.
    """
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    smiles: str = ""
    n_atoms: int = 0
    y: Optional[float] = None

    def to_torch(self):
        """Convert to PyTorch tensors."""
        import torch
        return {
            "x": torch.tensor(self.x, dtype=torch.float32),
            "edge_index": torch.tensor(self.edge_index, dtype=torch.long),
            "edge_attr": torch.tensor(self.edge_attr, dtype=torch.float32),
            "y": torch.tensor([self.y], dtype=torch.float32) if self.y is not None else None,
        }


def smiles_to_graph(smiles: str, y: Optional[float] = None) -> Optional[MolGraph]:
    """
    Convert SMILES to a molecular graph (Chemprop-style featurization).

    Parameters
    ----------
    smiles : str
        Input SMILES.
    y : float, optional
        Target property value.

    Returns
    -------
    MolGraph or None if SMILES is invalid.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        feat = []
        feat.extend(_one_hot(atom.GetSymbol(), _ATOM_SYMBOLS))
        feat.extend(_one_hot(atom.GetDegree(), _DEGREES))
        feat.extend(_one_hot(atom.GetFormalCharge(), _CHARGES))
        hyb = str(atom.GetHybridization()).split(".")[-1]
        feat.extend(_one_hot(hyb, _HYBRIDIZATIONS))
        feat.append(float(atom.GetIsAromatic()))
        feat.append(float(atom.IsInRing()))
        feat.extend(_one_hot(atom.GetTotalNumHs(), _NUM_HS))
        atom_features.append(feat)

    n_atoms = len(atom_features)
    if n_atoms == 0:
        return None

    x = np.array(atom_features, dtype=np.float32)

    # Bond features (bidirectional edges)
    src_list, dst_list, edge_features = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feat = _bond_features(bond)
        # Add both directions
        src_list.extend([i, j])
        dst_list.extend([j, i])
        edge_features.extend([bond_feat, bond_feat])

    if edge_features:
        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_attr = np.array(edge_features, dtype=np.float32)
    else:
        # Single atom molecule
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)

    return MolGraph(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        smiles=smiles, n_atoms=n_atoms, y=y,
    )


def _bond_features(bond) -> List[float]:
    """Extract bond features."""
    bt = bond.GetBondType()
    bond_type = _one_hot(
        str(bt),
        ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    )
    feat = bond_type + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]
    stereo = _one_hot(
        str(bond.GetStereo()),
        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
    )
    feat.extend(stereo)
    return feat


def batch_smiles_to_graphs(
    smiles_list: List[str],
    y_list: Optional[List[float]] = None,
) -> List[MolGraph]:
    """Convert a batch of SMILES to molecular graphs, skipping invalid ones."""
    graphs = []
    for i, smi in enumerate(smiles_list):
        y = y_list[i] if y_list is not None else None
        g = smiles_to_graph(smi, y=y)
        if g is not None:
            graphs.append(g)
    return graphs


# ---------------------------------------------------------------------------
# 4. Combined featurization
# ---------------------------------------------------------------------------


@dataclass
class MolecularFeatures:
    """Combined molecular features for a single molecule."""
    smiles: str
    fingerprint: np.ndarray       # (2048,)
    descriptors: np.ndarray       # (n_desc,)
    graph: Optional[MolGraph]     # for GNN
    scaffold: Optional[str]       # Bemis-Murcko scaffold
    is_valid: bool = True


def featurize_molecule(
    smiles: str,
    fp_radius: int = 2,
    fp_bits: int = 2048,
) -> MolecularFeatures:
    """Compute all three representations for a molecule."""
    _require_rdkit()
    canon = canonicalize(smiles)
    if canon is None:
        return MolecularFeatures(
            smiles=smiles,
            fingerprint=np.zeros(fp_bits, dtype=np.float32),
            descriptors=np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32),
            graph=None,
            scaffold=None,
            is_valid=False,
        )

    return MolecularFeatures(
        smiles=canon,
        fingerprint=morgan_fingerprint(canon, fp_radius, fp_bits),
        descriptors=rdkit_descriptors(canon),
        graph=smiles_to_graph(canon),
        scaffold=get_scaffold(canon),
        is_valid=True,
    )


# ---------------------------------------------------------------------------
# SMILES tokenizer (for Transformer models)
# ---------------------------------------------------------------------------

# Character-level SMILES vocabulary
SMILES_CHARS = [
    "<pad>", "<sos>", "<eos>", "<unk>",
    "C", "c", "N", "n", "O", "o", "S", "s", "F", "I",
    "H", "B", "P", "K",
    "l",  # for Cl
    "r",  # for Br
    "(", ")", "[", "]",
    "=", "#", "+", "-", ".",
    "/", "\\",
    "@",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "%",
]

CHAR_TO_IDX = {c: i for i, c in enumerate(SMILES_CHARS)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}
VOCAB_SIZE = len(SMILES_CHARS)


def tokenize_smiles(smiles: str, max_len: int = 128) -> np.ndarray:
    """
    Tokenize SMILES string to integer indices.

    Returns
    -------
    np.ndarray
        Shape (max_len,), padded with <pad> token (0).
    """
    tokens = [CHAR_TO_IDX["<sos>"]]
    for ch in smiles:
        idx = CHAR_TO_IDX.get(ch, CHAR_TO_IDX["<unk>"])
        tokens.append(idx)
    tokens.append(CHAR_TO_IDX["<eos>"])

    # Pad or truncate
    if len(tokens) >= max_len:
        tokens = tokens[:max_len]
    else:
        tokens.extend([CHAR_TO_IDX["<pad>"]] * (max_len - len(tokens)))

    return np.array(tokens, dtype=np.int64)


def detokenize_smiles(tokens: np.ndarray) -> str:
    """Convert token indices back to SMILES string."""
    chars = []
    for idx in tokens:
        idx = int(idx)
        if idx == CHAR_TO_IDX["<eos>"]:
            break
        if idx in (CHAR_TO_IDX["<pad>"], CHAR_TO_IDX["<sos>"]):
            continue
        ch = IDX_TO_CHAR.get(idx, "")
        chars.append(ch)
    return "".join(chars)
