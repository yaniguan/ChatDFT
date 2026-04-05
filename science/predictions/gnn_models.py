"""
Graph Neural Network Architectures for Adsorption Energy Prediction
====================================================================

Scientific motivation
---------------------
Predicting adsorption energies from surface topology graphs requires
architectures that capture progressively richer geometric information:

  - **MLP baseline**: ignores graph structure entirely (mean-pool node features)
  - **MPNN** (Gilmer 2017): learns edge-conditioned messages between neighbours
  - **GAT** (Velickovic 2018): learns attention weights over neighbours
  - **SchNet** (Schutt 2018): continuous radial filters on interatomic distances
  - **DimeNet** (Gasteiger 2020): directional message passing with bond angles
  - **SE(3)-Transformer** (Fuchs 2020): equivariant attention on positions

All models consume the output of ``SurfaceTopologyGraph``:
  - ``x``:          (N, d_node)  node feature matrix
  - ``edge_index``: (2, 2E)     directed edge list
  - ``edge_attr``:  (2E, d_edge) edge feature matrix
  - ``pos``:        (N, 3)       Cartesian coordinates (for SchNet/DimeNet/SE3)
  - ``batch``:      (N,)         graph membership indices

Key references
--------------
[1] Gilmer et al., ICML 2017 — Neural Message Passing
[2] Velickovic et al., ICLR 2018 — Graph Attention Networks
[3] Schutt et al., J. Chem. Phys. 148, 241722 (2018) — SchNet
[4] Gasteiger et al., ICLR 2020 — DimeNet
[5] Fuchs et al., NeurIPS 2020 — SE(3)-Transformers
[6] Behler & Parrinello, PRL 98, 146401 (2007) — symmetry functions
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Utility: graph data container (PyG-compatible but standalone)
# ---------------------------------------------------------------------------


@dataclass
class GraphData:
    """
    Minimal graph container matching PyTorch Geometric conventions.

    Parameters
    ----------
    x : (N, d_node) node features
    edge_index : (2, 2E) source-target edge list
    edge_attr : (2E, d_edge) edge features
    pos : (N, 3) Cartesian positions
    batch : (N,) graph membership for batched graphs
    y : scalar target (e.g. adsorption energy in eV)
    """

    x: "torch.Tensor"
    edge_index: "torch.Tensor"
    edge_attr: "torch.Tensor"
    pos: "torch.Tensor"
    batch: "torch.Tensor"
    y: Optional["torch.Tensor"] = None

    @staticmethod
    def from_numpy(
        x: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        pos: np.ndarray,
        y: Optional[float] = None,
        batch_idx: int = 0,
    ) -> "GraphData":
        """Convert numpy arrays (from SurfaceTopologyGraph) to GraphData."""
        if not _HAS_TORCH:
            raise ImportError("PyTorch required for GNN models")
        n = x.shape[0]
        return GraphData(
            x=torch.from_numpy(x).float(),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_attr).float(),
            pos=torch.from_numpy(pos).float(),
            batch=torch.full((n,), batch_idx, dtype=torch.long),
            y=torch.tensor([y], dtype=torch.float32) if y is not None else None,
        )


def _check_torch():
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for GNN models. Install with: pip install torch")


# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Simple MLP with optional residual connection."""

    def __init__(self, dims: list[int], activation: str = "silu", residual: bool = False):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU() if activation == "silu" else nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.residual = residual and dims[0] == dims[-1]

    def forward(self, x):
        out = self.net(x)
        return out + x if self.residual else out


class _RBFExpansion(nn.Module):
    """
    Radial basis function expansion of distances.

    Expands scalar distance d into K Gaussian basis functions:
        phi_k(d) = exp(-gamma * (d - mu_k)^2)
    where mu_k are evenly spaced in [0, cutoff].
    """

    def __init__(self, n_rbf: int = 20, cutoff: float = 5.0):
        super().__init__()
        self.n_rbf = n_rbf
        offsets = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("offsets", offsets)
        self.register_buffer("gamma", torch.tensor([-0.5 / (cutoff / n_rbf) ** 2]))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """dist: (E,) -> (E, n_rbf)"""
        return torch.exp(self.gamma * (dist.unsqueeze(-1) - self.offsets) ** 2)


class _BesselBasis(nn.Module):
    """
    Bessel radial basis for DimeNet (smoother than Gaussians at cutoff).

    phi_n(d) = sqrt(2/c) * sin(n*pi*d/c) / d
    """

    def __init__(self, n_basis: int = 8, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff
        freq = torch.arange(1, n_basis + 1).float() * math.pi / cutoff
        self.register_buffer("freq", freq)
        self.register_buffer("norm", torch.tensor(math.sqrt(2.0 / cutoff)))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """dist: (E,) -> (E, n_basis)"""
        d = dist.unsqueeze(-1).clamp(min=1e-8)
        return self.norm * torch.sin(self.freq * d) / d


class _CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope."""

    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.cos(math.pi * dist / self.cutoff)).clamp(min=0.0)


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean-pool scatter: src (E, d) indexed by (E,) -> (dim_size, d)."""
    out = torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    count.scatter_add_(0, index.unsqueeze(-1), torch.ones_like(index.unsqueeze(-1).float()))
    return out / count.clamp(min=1)


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum scatter: src (E, d) indexed by (E,) -> (dim_size, d)."""
    out = torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    return out


def _scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Softmax per group defined by index."""
    max_val = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    max_val.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
    out = torch.exp(src - max_val[index])
    denom = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    denom.scatter_add_(0, index, out)
    return out / denom[index].clamp(min=1e-12)


# ---------------------------------------------------------------------------
# 0. MLP Baseline (no graph structure)
# ---------------------------------------------------------------------------


class MLPBaseline(nn.Module):
    """
    Simple MLP baseline that ignores graph structure entirely.

    Pools node features by mean, then maps through 2-layer MLP.
    This is the "no message passing" control.
    """

    def __init__(self, d_node: int = 6, d_hidden: int = 64):
        _check_torch()
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_node, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, data: GraphData) -> torch.Tensor:
        # Mean-pool node features per graph
        h = _scatter_mean(data.x, data.batch, dim_size=data.batch.max().item() + 1)
        return self.mlp(h).squeeze(-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 1. MPNN — Message Passing Neural Network (Gilmer et al., 2017)
# ---------------------------------------------------------------------------


class _MPNNLayer(nn.Module):
    """
    Single MPNN layer: edge-conditioned message + GRU update.

    Message:  m_ij = MLP([h_i || h_j || e_ij])
    Aggregate: M_i = sum_j m_ij
    Update:   h_i' = GRU(h_i, M_i)
    """

    def __init__(self, d_node: int, d_edge: int, d_msg: int):
        super().__init__()
        self.msg_mlp = _MLP([2 * d_node + d_edge, d_msg, d_msg])
        self.gru = nn.GRUCell(d_msg, d_node)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        msg_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        messages = self.msg_mlp(msg_input)
        agg = _scatter_add(messages, dst, dim_size=x.shape[0])
        return self.gru(agg, x)


class MPNN(nn.Module):
    """
    Message Passing Neural Network for adsorption energy prediction.

    Architecture: node embedding -> T message-passing layers -> global pool -> MLP readout.

    Parameters
    ----------
    d_node : int
        Input node feature dimension (default 6 from Voronoi).
    d_edge : int
        Input edge feature dimension (default 3 from Voronoi).
    d_hidden : int
        Hidden dimension for message passing.
    n_layers : int
        Number of message passing steps.
    """

    def __init__(self, d_node: int = 6, d_edge: int = 3, d_hidden: int = 64, n_layers: int = 3):
        _check_torch()
        super().__init__()
        self.node_emb = nn.Linear(d_node, d_hidden)
        self.layers = nn.ModuleList([_MPNNLayer(d_hidden, d_edge, d_hidden) for _ in range(n_layers)])
        self.readout = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, data: GraphData) -> torch.Tensor:
        h = self.node_emb(data.x)
        for layer in self.layers:
            h = layer(h, data.edge_index, data.edge_attr)
        # Global mean pool per graph
        out = _scatter_mean(h, data.batch, dim_size=data.batch.max().item() + 1)
        return self.readout(out).squeeze(-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 2. GAT — Graph Attention Network (Velickovic et al., 2018)
# ---------------------------------------------------------------------------


class _GATLayer(nn.Module):
    """
    Multi-head graph attention layer.

    Attention:  alpha_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j || e_ij]))
    Aggregate:  h_i' = sum_j alpha_ij * V h_j
    """

    def __init__(self, d_in: int, d_out: int, d_edge: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        assert d_out % n_heads == 0

        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
        self.W_e = nn.Linear(d_edge, n_heads, bias=False)
        self.scale = math.sqrt(self.d_head)
        self.proj = nn.Linear(d_out, d_out)

    def forward(self, x, edge_index, edge_attr):
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        q = self.W_q(x).view(N, self.n_heads, self.d_head)
        k = self.W_k(x).view(N, self.n_heads, self.d_head)
        v = self.W_v(x).view(N, self.n_heads, self.d_head)

        # Attention scores
        attn = (q[dst] * k[src]).sum(dim=-1) / self.scale  # (E, n_heads)
        attn = attn + self.W_e(edge_attr)  # edge bias
        attn = F.leaky_relu(attn, 0.2)

        # Softmax per destination node, per head
        E = edge_index.shape[1]
        attn_flat = attn.view(E * self.n_heads)
        idx_flat = dst.unsqueeze(-1).expand(E, self.n_heads).reshape(E * self.n_heads)
        alpha_flat = _scatter_softmax(attn_flat, idx_flat, N * self.n_heads)
        alpha = alpha_flat.view(E, self.n_heads, 1)

        # Weighted aggregation
        msg = alpha * v[src]  # (E, n_heads, d_head)
        msg_flat = msg.view(E, -1)  # (E, d_out)
        out = _scatter_add(msg_flat, dst, dim_size=N)
        return self.proj(out)


class GAT(nn.Module):
    """
    Graph Attention Network for adsorption energy prediction.

    Multi-head attention learns which neighbours are most informative
    for predicting binding strength.

    Parameters
    ----------
    d_node, d_edge : int
        Input feature dimensions from Voronoi graph.
    d_hidden : int
        Hidden dimension (must be divisible by n_heads).
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of GAT layers.
    """

    def __init__(self, d_node: int = 6, d_edge: int = 3, d_hidden: int = 64, n_heads: int = 4, n_layers: int = 3):
        _check_torch()
        super().__init__()
        self.node_emb = nn.Linear(d_node, d_hidden)
        self.layers = nn.ModuleList([_GATLayer(d_hidden, d_hidden, d_edge, n_heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.SiLU(), nn.Linear(d_hidden, 1))

    def forward(self, data: GraphData) -> torch.Tensor:
        h = self.node_emb(data.x)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(h, data.edge_index, data.edge_attr)  # residual
            h = norm(h)
            h = F.silu(h)
        out = _scatter_mean(h, data.batch, dim_size=data.batch.max().item() + 1)
        return self.readout(out).squeeze(-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 3. SchNet — Continuous-Filter Convolutional Network (Schutt et al., 2018)
# ---------------------------------------------------------------------------


class _SchNetInteraction(nn.Module):
    """
    SchNet interaction block.

    1. Expand interatomic distance into RBF basis
    2. Generate continuous filter weights via MLP on RBF
    3. Element-wise multiply with neighbour features
    4. Aggregate and update via dense layer
    """

    def __init__(self, d_hidden: int, n_rbf: int, cutoff: float):
        super().__init__()
        self.rbf = _RBFExpansion(n_rbf, cutoff)
        self.cutoff_fn = _CosineCutoff(cutoff)
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.dense = nn.Linear(d_hidden, d_hidden)

    def forward(self, x, edge_index, pos):
        src, dst = edge_index[0], edge_index[1]
        diff = pos[dst] - pos[src]
        dist = torch.norm(diff, dim=-1)

        rbf = self.rbf(dist)
        envelope = self.cutoff_fn(dist).unsqueeze(-1)
        W = self.filter_net(rbf) * envelope

        messages = x[src] * W
        agg = _scatter_add(messages, dst, dim_size=x.shape[0])
        return self.dense(agg)


class SchNet(nn.Module):
    """
    SchNet: continuous-filter convolutions on interatomic distances.

    Uses radial basis function expansion of distances to generate
    filter weights, capturing distance-dependent interactions without
    explicit angle information.

    Parameters
    ----------
    d_node : int
        Input node feature dimension.
    d_hidden : int
        Hidden feature dimension.
    n_interactions : int
        Number of interaction blocks.
    n_rbf : int
        Number of radial basis functions.
    cutoff : float
        Distance cutoff in Angstroms.
    """

    def __init__(
        self, d_node: int = 6, d_hidden: int = 64, n_interactions: int = 3, n_rbf: int = 20, cutoff: float = 5.0
    ):
        _check_torch()
        super().__init__()
        self.node_emb = nn.Linear(d_node, d_hidden)
        self.interactions = nn.ModuleList([_SchNetInteraction(d_hidden, n_rbf, cutoff) for _ in range(n_interactions)])
        self.readout = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, data: GraphData) -> torch.Tensor:
        h = self.node_emb(data.x)
        for interaction in self.interactions:
            h = h + interaction(h, data.edge_index, data.pos)  # residual
        out = _scatter_mean(h, data.batch, dim_size=data.batch.max().item() + 1)
        return self.readout(out).squeeze(-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 4. DimeNet — Directional Message Passing (Gasteiger et al., 2020)
# ---------------------------------------------------------------------------


class _SphericalBasis(nn.Module):
    """
    Spherical basis expansion of angles for DimeNet.

    Combines Bessel radial basis with Fourier angular basis:
        a_lk(d, theta) = phi_k(d) * Y_l(theta)
    where Y_l(theta) = sin((l+1)*theta) / sin(theta) for l >= 0.
    """

    def __init__(self, n_radial: int = 8, n_angular: int = 7, cutoff: float = 5.0):
        super().__init__()
        self.bessel = _BesselBasis(n_radial, cutoff)
        self.n_angular = n_angular

    def forward(self, dist: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """
        dist: (E_triplet,), angle: (E_triplet,) in radians
        Returns: (E_triplet, n_radial * n_angular)
        """
        rbf = self.bessel(dist)  # (E_triplet, n_radial)
        # Fourier angular basis
        l_vals = torch.arange(1, self.n_angular + 1, device=angle.device).float()
        sin_theta = torch.sin(angle).clamp(min=1e-8).unsqueeze(-1)
        abf = torch.sin(l_vals.unsqueeze(0) * angle.unsqueeze(-1)) / sin_theta
        # Outer product
        return (rbf.unsqueeze(-1) * abf.unsqueeze(-2)).reshape(dist.shape[0], -1)


class _DimeNetInteraction(nn.Module):
    """DimeNet interaction block with directional message passing."""

    def __init__(self, d_hidden: int, d_rbf: int, d_sbf: int):
        super().__init__()
        self.rbf_proj = nn.Linear(d_rbf, d_hidden, bias=False)
        self.sbf_proj = nn.Linear(d_sbf, d_hidden, bias=False)
        self.msg_mlp = _MLP([d_hidden, d_hidden, d_hidden])
        self.update_mlp = _MLP([2 * d_hidden, d_hidden, d_hidden])

    def forward(self, m, rbf, sbf, edge_index, triplet_idx):
        """
        m: (E, d_hidden) edge messages
        rbf: (E, d_rbf) radial basis on edges
        sbf: (T, d_sbf) spherical basis on triplets
        triplet_idx: (T, 3) [edge_kj, edge_ji, node_j]
        """
        # Directional message transform
        e_kj = triplet_idx[:, 0]
        e_ji = triplet_idx[:, 1]

        m_kj = m[e_kj]
        rbf_kj = self.rbf_proj(rbf[e_kj])
        sbf_kji = self.sbf_proj(sbf)

        # Directional message
        dir_msg = m_kj * rbf_kj * sbf_kji
        dir_msg = self.msg_mlp(dir_msg)

        # Aggregate over incoming triplets per edge
        agg = _scatter_add(dir_msg, e_ji, dim_size=m.shape[0])

        # Update
        m_new = self.update_mlp(torch.cat([m, agg], dim=-1))
        return m + m_new  # residual


class DimeNet(nn.Module):
    """
    DimeNet: directional message passing with bond angles.

    Extends SchNet by incorporating angular information via triplets
    of atoms (i, j, k), enabling it to distinguish configurations
    that are identical in distance but differ in geometry.

    Parameters
    ----------
    d_node : int
        Input node feature dimension.
    d_hidden : int
        Hidden dimension.
    n_interactions : int
        Number of interaction blocks.
    n_rbf : int
        Number of Bessel radial basis functions.
    n_angular : int
        Number of angular Fourier components.
    cutoff : float
        Distance cutoff in Angstroms.
    """

    def __init__(
        self,
        d_node: int = 6,
        d_hidden: int = 64,
        n_interactions: int = 3,
        n_rbf: int = 8,
        n_angular: int = 7,
        cutoff: float = 5.0,
    ):
        _check_torch()
        super().__init__()
        self.cutoff = cutoff
        d_sbf = n_rbf * n_angular

        self.node_emb = nn.Linear(d_node, d_hidden)
        self.bessel = _BesselBasis(n_rbf, cutoff)
        self.sbasis = _SphericalBasis(n_rbf, n_angular, cutoff)
        self.cutoff_fn = _CosineCutoff(cutoff)

        # Initial edge embedding
        self.edge_emb = nn.Sequential(
            nn.Linear(d_hidden + n_rbf, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )

        self.interactions = nn.ModuleList([_DimeNetInteraction(d_hidden, n_rbf, d_sbf) for _ in range(n_interactions)])

        # Output: aggregate edge messages to nodes, then pool
        self.node_proj = nn.Linear(d_hidden, d_hidden)
        self.readout = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.SiLU(), nn.Linear(d_hidden, 1))

    def _build_triplets(self, edge_index, pos):
        """
        Build angle triplets (k->j->i) from edge list.

        Returns
        -------
        triplet_idx : (T, 3) — [edge_kj_idx, edge_ji_idx, node_j]
        angles : (T,) — angle at j between k-j and j-i
        dist_kj : (T,) — distance for k-j edges in triplets
        """
        src, dst = edge_index[0], edge_index[1]  # src->dst
        E = edge_index.shape[1]

        # For each node j, find all incoming edges (k->j) and outgoing edges (j->i)
        # triplet: edge_kj and edge_ji share node j, where kj.dst == ji.src == j
        triplet_e_kj = []
        triplet_e_ji = []
        triplet_j = []

        # Group edges by destination node
        max(src.max(), dst.max()) + 1
        for e_ji in range(E):
            j = src[e_ji].item()
            i = dst[e_ji].item()
            # Find all edges k->j (where dst == j and src != i)
            mask = (dst == j) & (src != i)
            for e_kj in mask.nonzero(as_tuple=False).squeeze(-1):
                triplet_e_kj.append(e_kj.item())
                triplet_e_ji.append(e_ji)
                triplet_j.append(j)

        if len(triplet_e_kj) == 0:
            device = pos.device
            return (
                torch.zeros(0, 3, dtype=torch.long, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, device=device),
            )

        triplet_idx = torch.tensor(
            list(zip(triplet_e_kj, triplet_e_ji, triplet_j)),
            dtype=torch.long,
            device=pos.device,
        )

        # Compute angles
        e_kj_idx = triplet_idx[:, 0]
        e_ji_idx = triplet_idx[:, 1]

        k = src[e_kj_idx]
        j = triplet_idx[:, 2]
        i = dst[e_ji_idx]

        vec_jk = pos[k] - pos[j]
        vec_ji = pos[i] - pos[j]
        cos_angle = F.cosine_similarity(vec_jk, vec_ji, dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
        angles = torch.acos(cos_angle)

        dist_kj = torch.norm(vec_jk, dim=-1)

        return triplet_idx, angles, dist_kj

    def forward(self, data: GraphData) -> torch.Tensor:
        src, dst = data.edge_index[0], data.edge_index[1]
        diff = data.pos[dst] - data.pos[src]
        dist = torch.norm(diff, dim=-1)

        # Radial basis on edges
        rbf = self.bessel(dist) * self.cutoff_fn(dist).unsqueeze(-1)

        # Initial edge messages from node embeddings + RBF
        h = self.node_emb(data.x)
        m = self.edge_emb(torch.cat([h[src] + h[dst], rbf], dim=-1))

        # Build triplets and spherical basis
        triplet_idx, angles, dist_kj = self._build_triplets(data.edge_index, data.pos)

        if triplet_idx.shape[0] > 0:
            sbf = self.sbasis(dist_kj, angles)
        else:
            sbf = torch.zeros(0, self.interactions[0].sbf_proj.in_features, device=data.x.device)

        # Interaction blocks
        for interaction in self.interactions:
            m = interaction(m, rbf, sbf, data.edge_index, triplet_idx)

        # Aggregate edge messages to nodes
        node_msg = _scatter_add(m, dst, dim_size=data.x.shape[0])
        h_out = self.node_proj(node_msg)

        # Global pool
        out = _scatter_mean(h_out, data.batch, dim_size=data.batch.max().item() + 1)
        return self.readout(out).squeeze(-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 5. SE(3)-Transformer — Equivariant Attention (Fuchs et al., 2020)
# ---------------------------------------------------------------------------


class _SE3AttentionLayer(nn.Module):
    """
    SE(3)-equivariant attention layer.

    Operates on two feature types:
      - Type-0 (scalar): invariant features, shape (N, d_scalar)
      - Type-1 (vector): equivariant features, shape (N, d_vector, 3)

    Attention is computed from scalar features + distance.
    Messages carry both scalar and vector information.
    Vector messages are constructed from relative position vectors,
    ensuring SE(3) equivariance.
    """

    def __init__(self, d_scalar: int, d_vector: int, n_heads: int = 4, n_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_scalar = d_scalar
        self.d_vector = d_vector
        self.d_head = d_scalar // n_heads

        # Scalar attention
        self.W_q = nn.Linear(d_scalar, d_scalar, bias=False)
        self.W_k = nn.Linear(d_scalar, d_scalar, bias=False)
        self.W_v_scalar = nn.Linear(d_scalar, d_scalar, bias=False)

        # Distance bias
        self.rbf = _RBFExpansion(n_rbf, cutoff)
        self.dist_proj = nn.Linear(n_rbf, n_heads, bias=False)

        # Vector message
        self.W_v_vec = nn.Linear(d_scalar, d_vector, bias=False)

        # Scalar update from vector (norm)
        self.vec_to_scalar = nn.Linear(d_vector, d_scalar)

        # Output projections
        self.scalar_out = nn.Linear(d_scalar, d_scalar)
        # EQUIVARIANT vector gate: per-channel scalar weight from scalar features.
        # This preserves SE(3) equivariance: v_new = gate(s) * v
        # where gate produces invariant per-channel weights.
        self.vector_gate = nn.Sequential(
            nn.Linear(d_scalar, d_vector),
            nn.Sigmoid(),
        )

        self.scale = math.sqrt(self.d_head)
        self.norm = nn.LayerNorm(d_scalar)

    def forward(self, s, v, edge_index, pos):
        """
        s: (N, d_scalar) scalar features
        v: (N, d_vector, 3) vector features
        edge_index: (2, E)
        pos: (N, 3)
        """
        N = s.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Relative positions (equivariant)
        rel_pos = pos[src] - pos[dst]  # (E, 3)
        dist = torch.norm(rel_pos, dim=-1)  # (E,)
        rel_dir = rel_pos / dist.unsqueeze(-1).clamp(min=1e-8)  # unit vectors

        # Attention from scalar features + distance
        q = self.W_q(s).view(N, self.n_heads, self.d_head)
        k = self.W_k(s).view(N, self.n_heads, self.d_head)

        attn = (q[dst] * k[src]).sum(-1) / self.scale  # (E, n_heads)
        attn = attn + self.dist_proj(self.rbf(dist))  # distance bias
        attn = F.silu(attn)

        # Softmax per destination
        E = edge_index.shape[1]
        attn_flat = attn.reshape(E * self.n_heads)
        idx_flat = dst.unsqueeze(-1).expand(E, self.n_heads).reshape(E * self.n_heads)
        alpha_flat = _scatter_softmax(attn_flat, idx_flat, N * self.n_heads)
        alpha = alpha_flat.view(E, self.n_heads)

        # Scalar messages
        v_scalar = self.W_v_scalar(s).view(N, self.n_heads, self.d_head)
        msg_s = (alpha.unsqueeze(-1) * v_scalar[src]).view(E, -1)  # (E, d_scalar)
        agg_s = _scatter_add(msg_s, dst, dim_size=N)

        # Vector messages: weight * direction
        v_weight = self.W_v_vec(s[src])  # (E, d_vector)
        msg_v = (alpha.mean(dim=-1, keepdim=True) * v_weight).unsqueeze(-1) * rel_dir.unsqueeze(-2)
        # msg_v: (E, d_vector, 3)
        agg_v = torch.zeros(N, self.d_vector, 3, device=s.device)
        for d in range(3):
            agg_v[:, :, d].scatter_add_(0, dst.unsqueeze(-1).expand(E, self.d_vector), msg_v[:, :, d])

        # Update scalar with vector norm info
        v_sum = v + agg_v  # (N, d_vector, 3)
        vec_norm = torch.norm(v_sum, dim=-1)  # (N, d_vector)
        s_new = self.norm(s + self.scalar_out(agg_s) + self.vec_to_scalar(vec_norm))
        # Equivariant vector update: gate(scalar) * vector
        # gate is computed from scalar features (invariant),
        # then multiplied element-wise with vector (preserves equivariance).
        gate = self.vector_gate(s_new)  # (N, d_vector) — invariant scalars
        v_new = gate.unsqueeze(-1) * v_sum  # (N, d_vector, 3)

        return s_new, v_new


class SE3Transformer(nn.Module):
    """
    SE(3)-Transformer for adsorption energy prediction.

    Maintains both scalar (invariant) and vector (equivariant) features.
    Attention is computed from scalar features with distance bias.
    Vector features are updated using relative position vectors,
    guaranteeing SE(3) equivariance by construction.

    This is critical for catalysis: the binding energy must not change
    if we rotate or translate the entire slab+adsorbate system.

    Parameters
    ----------
    d_node : int
        Input node feature dimension.
    d_scalar : int
        Scalar (type-0) feature dimension.
    d_vector : int
        Vector (type-1) feature dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of SE(3) attention layers.
    cutoff : float
        Distance cutoff for attention.
    """

    def __init__(
        self,
        d_node: int = 6,
        d_scalar: int = 64,
        d_vector: int = 16,
        n_heads: int = 4,
        n_layers: int = 3,
        cutoff: float = 5.0,
    ):
        _check_torch()
        super().__init__()
        self.node_emb = nn.Linear(d_node, d_scalar)
        self.layers = nn.ModuleList(
            [_SE3AttentionLayer(d_scalar, d_vector, n_heads, cutoff=cutoff) for _ in range(n_layers)]
        )
        self.readout = nn.Sequential(nn.Linear(d_scalar, d_scalar), nn.SiLU(), nn.Linear(d_scalar, 1))
        self.d_vector = d_vector

    def forward(self, data: GraphData) -> torch.Tensor:
        N = data.x.shape[0]
        s = self.node_emb(data.x)  # (N, d_scalar)
        v = torch.zeros(N, self.d_vector, 3, device=data.x.device)  # (N, d_vector, 3)

        for layer in self.layers:
            s, v = layer(s, v, data.edge_index, data.pos)

        out = _scatter_mean(s, data.batch, dim_size=data.batch.max().item() + 1)
        return self.readout(out).squeeze(-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, type] = {
    "mlp": MLPBaseline,
    "mpnn": MPNN,
    "gat": GAT,
    "schnet": SchNet,
    "dimenet": DimeNet,
    "se3_transformer": SE3Transformer,
}


def list_models() -> list[str]:
    """Return available model names."""
    return list(MODEL_REGISTRY.keys())


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name."""
    _check_torch()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    return MODEL_REGISTRY[name](**kwargs)
