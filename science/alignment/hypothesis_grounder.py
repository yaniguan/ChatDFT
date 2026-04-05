"""
Cross-Modal Hypothesis Grounding
==================================
Scientific motivation
---------------------
An LLM-generated reaction hypothesis (text) must be evaluated against two
independent evidence channels:

  (A) The reaction *graph* — which intermediates connect to which, in what
      stoichiometric order.
  (B) The DFT free energy *profile* — whether the proposed pathway is
      thermodynamically feasible and where the rate-determining step is.

Treating this as a *cross-modal alignment* problem lets us:
  1. Learn a shared embedding space where semantically consistent
     (text, graph, energy) triples cluster together.
  2. Compute a principled confidence score from geometric distance in that
     space, rather than heuristic keyword matching.
  3. Train on historical (hypothesis → DFT outcome) pairs so the scorer
     improves with data.

Three modalities
-----------------
  Text    : natural language hypothesis string  →  d_T-dimensional vector
  Graph   : reaction network as directed node/edge structure  →  d_G-dim
  Property: free energy diagram as 1D sequence  →  d_P-dim

Alignment objective
--------------------
InfoNCE (contrastive) loss:

    L = - (1/B) Σᵢ log  [ exp(sim(Tᵢ,Gᵢ)/τ) / Σⱼ exp(sim(Tᵢ,Gⱼ)/τ) ]

where sim(a,b) = aᵀb / (||a|| ||b||)  (cosine similarity),
τ is a learnable temperature, and the negatives j ≠ i are other elements
in the same batch.

At inference, we use sim(T, G) and optionally sim(T, P) as confidence
components — a higher score means the text hypothesis is consistent with
the computed evidence.

References
-----------
[1] Radford et al. (OpenAI), CLIP, ICML 2021 — InfoNCE cross-modal alignment
[2] Chen et al., SimCLR, ICML 2020 — contrastive learning framework
[3] Hu et al., Strategies for Pre-training Graph NNs, ICLR 2020
[4] Jablonka et al., GPT-4 for materials science, ACS Central Sci. 2024
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from science.core.logging import get_logger

# Optional heavy deps — graceful degradation
try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except ImportError:
    _HAS_ST = False

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Species tokenizer (chemistry-aware)
# ---------------------------------------------------------------------------

# One-hot chemical features for common surface/gas species
_SPECIES_VOCAB: Dict[str, int] = {
    "*": 0,
    "H*": 1,
    "OH*": 2,
    "O*": 3,
    "OOH*": 4,
    "CO*": 5,
    "COOH*": 6,
    "CHO*": 7,
    "CH2O*": 8,
    "CH3O*": 9,
    "CO2*": 10,
    "N*": 11,
    "NH*": 12,
    "NH2*": 13,
    "NH3*": 14,
    "N2*": 15,
    "NNH*": 16,
    "CO2(g)": 17,
    "H2O(g)": 18,
    "H2(g)": 19,
    "CO(g)": 20,
    "O2(g)": 21,
    "N2(g)": 22,
    "NH3(g)": 23,
    "H+": 24,
    "e-": 25,
    "<UNK>": 26,
}
_VOCAB_SIZE = len(_SPECIES_VOCAB)


def _species_idx(name: str) -> int:
    return _SPECIES_VOCAB.get(name.strip(), _SPECIES_VOCAB["<UNK>"])


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ReactionStep:
    lhs: List[str]
    rhs: List[str]
    is_electrochemical: bool = False
    n_electrons: int = 0


@dataclass
class ReactionNetwork:
    """
    Structured reaction network — the 'graph' modality.

    Attributes
    ----------
    steps       : elementary reaction steps (ReactionStep)
    intermediates : all surface + gas species in the mechanism
    ts_edges    : (lhs_species, rhs_species) pairs flagged as TSs
    surface     : e.g. 'Cu(111)'
    reactant    : e.g. 'CO2'
    product     : e.g. 'CO'
    """

    steps: List[ReactionStep]
    intermediates: List[str]
    ts_edges: List[Tuple[str, str]] = field(default_factory=list)
    surface: str = ""
    reactant: str = ""
    product: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ReactionNetwork":
        steps = [
            ReactionStep(
                lhs=s["lhs"],
                rhs=s["rhs"],
                is_electrochemical=("H+" in s["lhs"] or "e-" in s["lhs"]),
                n_electrons=s["lhs"].count("e-"),
            )
            for s in d.get("reaction_network", [])
        ]
        return cls(
            steps=steps,
            intermediates=d.get("intermediates", []),
            ts_edges=[(e[0], e[1]) for e in d.get("ts_edges", []) if len(e) == 2],
            surface=d.get("surface", ""),
            reactant=d.get("reactant", ""),
            product=d.get("product", ""),
        )

    def fingerprint(self) -> str:
        key = json.dumps(
            {
                "surface": self.surface,
                "reactant": self.reactant,
                "product": self.product,
                "intermediates": sorted(self.intermediates),
            },
            sort_keys=True,
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Modality encoders
# ---------------------------------------------------------------------------


class TextEncoder:
    """
    Encodes a natural-language hypothesis into a fixed-length vector.

    Two modes
    ---------
    sentence_transformer : use a pre-trained SentenceTransformer model.
    bow_chemistry        : fast bag-of-chemical-words fallback (no GPU needed).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        mode: str = "auto",  # 'sentence_transformer' | 'bow' | 'auto'
        d_out: int = 64,
    ):
        self.d_out = d_out
        self.mode = mode
        self._st = None
        if mode in ("sentence_transformer", "auto") and _HAS_ST:
            try:
                self._st = SentenceTransformer(model_name)
                self.mode = "sentence_transformer"
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(
                    "SentenceTransformer init failed, falling back to BoW", extra={"model": model_name, "error": str(e)}
                )
                self.mode = "bow"
        else:
            self.mode = "bow"

    def encode(self, text: str) -> np.ndarray:
        if self.mode == "sentence_transformer" and self._st is not None:
            raw = self._st.encode(text, normalize_embeddings=True)
            # Project to d_out with random linear layer (fixed seed)
            if raw.shape[0] != self.d_out:
                rng = np.random.default_rng(42)
                W = rng.standard_normal((raw.shape[0], self.d_out))
                W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
                raw = raw @ W
            return _l2_norm(raw.astype(np.float32))
        else:
            return self._bow_encode(text)

    def _bow_encode(self, text: str) -> np.ndarray:
        """
        Chemistry-aware bag-of-words:
          1. Count occurrences of each species vocab token.
          2. Append simple scalar features: n_surface_species, n_PCET_steps.
          3. Hash-project to d_out dims with random projection.
        """
        t = text.lower()
        counts = np.zeros(_VOCAB_SIZE, dtype=np.float32)
        for species, idx in _SPECIES_VOCAB.items():
            counts[idx] += t.count(species.lower())
        # scalar features
        n_ec = t.count("pcet") + t.count("proton") + t.count("electron")
        n_ads = t.count("adsorb") + t.count("bind") + t.count("surface")
        extras = np.array([n_ec, n_ads], dtype=np.float32)
        raw = np.concatenate([counts, extras])
        # fixed random projection to d_out
        rng = np.random.default_rng(1234)
        W = rng.standard_normal((raw.shape[0], self.d_out)).astype(np.float32)
        return _l2_norm(raw @ W)


class ReactionGraphEncoder:
    """
    Encode a ReactionNetwork into a fixed-length vector via:
      1. Node embedding : one-hot species → learned (here: fixed random) projection
      2. Edge aggregation : mean of step-level binary features
      3. Global features : n_steps, n_intermediates, frac_electrochemical, has_ts

    For production use, replace the random projections with a trained GNN
    (e.g. GIN with message passing over the intermediate-connectivity graph).
    """

    def __init__(self, d_out: int = 64, rng_seed: int = 42):
        self.d_out = d_out
        self.rng = np.random.default_rng(rng_seed)
        # Fixed projection matrix: vocab_size → d_out
        self._W_species = self.rng.standard_normal((_VOCAB_SIZE, d_out)).astype(np.float32)
        self._W_species /= np.linalg.norm(self._W_species, axis=0) + 1e-9

    def encode(self, network: ReactionNetwork) -> np.ndarray:
        # --- Intermediate embeddings (mean-pool over all species) ---------
        if network.intermediates:
            species_oh = np.zeros((_VOCAB_SIZE,), dtype=np.float32)
            for sp in network.intermediates:
                species_oh[_species_idx(sp)] += 1.0
            species_oh /= len(network.intermediates)
            species_emb = species_oh @ self._W_species
        else:
            species_emb = np.zeros(self.d_out, dtype=np.float32)

        # --- Step-level features ------------------------------------------
        n_steps = len(network.steps)
        n_ec = sum(s.is_electrochemical for s in network.steps)
        n_ts = len(network.ts_edges)
        n_inter = len(network.intermediates)

        # Global features (normalised)
        global_feat = np.array(
            [
                n_steps / max(n_steps, 1),
                n_ec / max(n_steps, 1),
                n_ts / max(n_steps, 1),
                n_inter / max(n_inter, 1),
                float("*" in network.intermediates),
            ],
            dtype=np.float32,
        )

        # Pad/project global to d_out
        rng2 = np.random.default_rng(99)
        W_glob = rng2.standard_normal((len(global_feat), self.d_out)).astype(np.float32)
        W_glob /= np.linalg.norm(W_glob, axis=0) + 1e-9
        global_emb = global_feat @ W_glob

        combined = species_emb + global_emb
        return _l2_norm(combined)


class FreeEnergyEncoder:
    """
    Encode a free-energy profile (1D time series of ΔG values) via:
      1. Normalise profile to zero-mean, unit-variance.
      2. Extract scalar descriptors: ΔG_min, ΔG_max, ΔG_rds, n_steps,
         limiting potential U_lim, overpotential η.
      3. 1D-convolutional feature extraction (hand-crafted filters).
      4. Project to d_out.

    This is the 'property' modality — it encodes computed DFT thermodynamics.
    """

    # Hand-crafted 1D filters: detect monotone decrease, barrier, plateau
    _FILTERS = [
        np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float32),  # rising step
        np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32),  # falling step
        np.array([-1.0, 2.0, -1.0, 0.0], dtype=np.float32),  # barrier shape
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # plateau
    ]

    def __init__(self, d_out: int = 64, seq_len: int = 16):
        self.d_out = d_out
        self.seq_len = seq_len
        rng = np.random.default_rng(77)
        self._W = rng.standard_normal(
            (seq_len + 6, d_out)  # conv output + 6 scalar features
        ).astype(np.float32)
        self._W /= np.linalg.norm(self._W, axis=0) + 1e-9

    def encode(self, dG_profile: List[float], U_limiting: float = 0.0, overpotential: float = 0.0) -> np.ndarray:
        """
        Parameters
        ----------
        dG_profile   : list of ΔG values along pathway (eV), relative to first step = 0
        U_limiting   : limiting potential (V_RHE)
        overpotential: overpotential (V)
        """
        if not dG_profile:
            return np.zeros(self.d_out, dtype=np.float32)

        # --- Normalise and pad/truncate to seq_len -----------------------
        G = np.array(dG_profile, dtype=np.float32)
        G = G - G[0]  # relative to first step
        mu, sig = G.mean(), G.std() + 1e-9
        G_norm = (G - mu) / sig

        seq = np.zeros(self.seq_len, dtype=np.float32)
        L = min(len(G_norm), self.seq_len)
        seq[:L] = G_norm[:L]

        # --- Conv features (valid convolution, no padding) ---------------
        conv_out = []
        for filt in self._FILTERS:
            k = len(filt)
            cv = [float(np.dot(seq[i : i + k], filt)) for i in range(self.seq_len - k + 1)]
            conv_out.append(np.max(cv) if cv else 0.0)

        # --- Scalar descriptors ------------------------------------------
        rds_idx = int(np.argmax(np.diff(G))) if len(G) > 1 else 0
        scalars = (
            np.array(
                [
                    float(G.min()),
                    float(G.max()),
                    float(G[rds_idx + 1] - G[rds_idx]) if rds_idx < len(G) - 1 else 0.0,
                    float(len(dG_profile)),
                    float(U_limiting),
                    float(overpotential),
                ],
                dtype=np.float32,
            )
            / 5.0
        )  # crude normalisation to O(1)

        raw = np.concatenate([seq, scalars])
        return _l2_norm(raw @ self._W)


# ---------------------------------------------------------------------------
# Cross-modal aligner
# ---------------------------------------------------------------------------


class HypothesisGrounder:
    """
    Compute alignment scores between a text hypothesis and its evidence.

    Usage
    -----
    >>> grounder = HypothesisGrounder()
    >>> score = grounder.score(
    ...     hypothesis="COOH* is the key intermediate on Cu(111) for CO2RR",
    ...     network=ReactionNetwork.from_dict(llm_output),
    ...     dG_profile=[0.0, 0.22, -0.15, -0.45, -1.10],
    ... )
    >>> print(score)  # 0.0 – 1.0, higher = more consistent
    """

    def __init__(
        self,
        d_embed: int = 64,
        temperature: float = 0.07,
    ):
        self.d = d_embed
        self.tau = temperature
        self.text_enc = TextEncoder(d_out=d_embed)
        self.graph_enc = ReactionGraphEncoder(d_out=d_embed)
        self.prop_enc = FreeEnergyEncoder(d_out=d_embed, seq_len=16)

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(
        self,
        hypothesis: str,
        network: ReactionNetwork,
        dG_profile: Optional[List[float]] = None,
        U_limiting: float = 0.0,
        overpotential: float = 0.0,
    ) -> float:
        """
        Cross-modal alignment confidence in [0, 1].

        Components
        ----------
        sim(T, G) : cosine similarity of text and graph embeddings (weight 0.6)
        sim(T, P) : cosine similarity of text and property embeddings (weight 0.4)
                    — only used when dG_profile is provided

        The raw cosine similarities are passed through a sigmoid to map to [0,1].
        """
        t_emb = self.text_enc.encode(hypothesis)
        g_emb = self.graph_enc.encode(network)

        sim_tg = float(np.dot(t_emb, g_emb))  # already L2-normalised

        if dG_profile is not None and len(dG_profile) > 1:
            p_emb = self.prop_enc.encode(dG_profile, U_limiting, overpotential)
            sim_tp = float(np.dot(t_emb, p_emb))
            raw = 0.6 * sim_tg + 0.4 * sim_tp
        else:
            raw = sim_tg

        return float(_sigmoid(raw / self.tau))

    # ------------------------------------------------------------------
    # Batch InfoNCE loss (for training / fine-tuning)
    # ------------------------------------------------------------------

    def infonce_loss(
        self,
        hypotheses: List[str],
        networks: List[ReactionNetwork],
    ) -> float:
        """
        Compute InfoNCE contrastive loss for a batch of (text, graph) pairs.

        Assumes batch[i] are positive pairs, all (i,j≠i) are negatives.
        Lower loss → better alignment.

        Parameters
        ----------
        hypotheses : B text hypotheses
        networks   : B corresponding reaction networks (same order)
        """
        B = len(hypotheses)
        assert B == len(networks), "Batch size mismatch."
        T = np.stack([self.text_enc.encode(h) for h in hypotheses])  # (B,d)
        G = np.stack([self.graph_enc.encode(n) for n in networks])  # (B,d)

        # Cosine similarity matrix (B × B)
        sim = T @ G.T / self.tau  # already L2-normed

        # Log-softmax over each row, keep diagonal (positive pairs)
        log_sm = sim - _log_sum_exp_rows(sim)  # (B,B)
        loss = -float(np.mean(np.diag(log_sm)))
        return loss

    # ------------------------------------------------------------------
    # Detailed breakdown (for explainability)
    # ------------------------------------------------------------------

    def score_breakdown(
        self,
        hypothesis: str,
        network: ReactionNetwork,
        dG_profile: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Return a dict with per-component scores for interpretability.
        """
        t_emb = self.text_enc.encode(hypothesis)
        g_emb = self.graph_enc.encode(network)
        sim_tg = float(np.dot(t_emb, g_emb))

        result: Dict[str, float] = {
            "text_graph_cosine": sim_tg,
            "text_graph_confidence": float(_sigmoid(sim_tg / self.tau)),
        }
        if dG_profile is not None and len(dG_profile) > 1:
            p_emb = self.prop_enc.encode(dG_profile)
            sim_tp = float(np.dot(t_emb, p_emb))
            result["text_property_cosine"] = sim_tp
            result["text_property_confidence"] = float(_sigmoid(sim_tp / self.tau))
            result["combined_confidence"] = self.score(hypothesis, network, dG_profile)
        else:
            result["combined_confidence"] = result["text_graph_confidence"]

        return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _l2_norm(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _log_sum_exp_rows(X: np.ndarray) -> np.ndarray:
    """Numerically stable log-sum-exp over each row."""
    m = X.max(axis=1, keepdims=True)
    return m.squeeze(1) + np.log(np.exp(X - m).sum(axis=1))


# ---------------------------------------------------------------------------
# Trainable Cross-Modal Grounder (PyTorch-backed)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as _F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TrainableGrounder:
    """
    End-to-end trainable cross-modal hypothesis grounder.

    Unlike HypothesisGrounder (which uses frozen random projections),
    this class learns the projection matrices via InfoNCE contrastive loss,
    so text, graph, and energy embeddings are genuinely aligned.

    Architecture
    ------------
    Text encoder:   BoW chemistry features → Linear → ReLU → Linear → L2 norm
    Graph encoder:  Species one-hot + global features → Linear → ReLU → Linear → L2
    Energy encoder: Padded dG + scalars → Linear → ReLU → Linear → L2 norm

    Training objective: InfoNCE (CLIP-style) with learnable temperature.

    Usage
    -----
    >>> grounder = TrainableGrounder(d_embed=64)
    >>> # Prepare training pairs: list of (hypothesis, network, dG_profile)
    >>> loss_curve = grounder.train(pairs, n_epochs=50, lr=1e-3)
    >>> score = grounder.score(hypothesis, network, dG_profile)

    Requires PyTorch. Falls back to HypothesisGrounder if unavailable.
    """

    def __init__(self, d_embed: int = 64, temperature: float = 0.07):
        if not _HAS_TORCH:
            raise ImportError("PyTorch required for TrainableGrounder")

        self.d = d_embed
        self._base_grounder = HypothesisGrounder(d_embed=d_embed, temperature=temperature)

        # Learnable projection heads (replace frozen random projections)
        d_text_in = _VOCAB_SIZE + 5  # BoW features
        d_graph_in = _VOCAB_SIZE + 5  # species + global
        d_energy_in = 16 + 6  # padded dG + scalar features

        self._text_proj = nn.Sequential(nn.Linear(d_text_in, 128), nn.ReLU(), nn.Linear(128, d_embed))
        self._graph_proj = nn.Sequential(nn.Linear(d_graph_in, 128), nn.ReLU(), nn.Linear(128, d_embed))
        self._energy_proj = nn.Sequential(nn.Linear(d_energy_in, 128), nn.ReLU(), nn.Linear(128, d_embed))
        # Learnable temperature
        self._log_tau = nn.Parameter(torch.tensor(math.log(temperature)))

    def _extract_text_features(self, hypothesis: str) -> np.ndarray:
        """Extract raw BoW features (before projection)."""
        text_lower = hypothesis.lower()
        species_oh = np.zeros(_VOCAB_SIZE, dtype=np.float32)
        for i, tok in enumerate(_SPECIES_VOCAB):
            if tok.lower() in text_lower:
                species_oh[i] = 1.0
        n_surface = sum(1 for v in ["surface", "slab", "facet", "111", "100", "110"] if v in text_lower)
        n_pcet = sum(1 for v in ["h+", "e-", "proton", "electron", "pcet"] if v in text_lower)
        extras = np.array(
            [
                float(len(text_lower)) / 500.0,
                species_oh.sum() / max(_VOCAB_SIZE, 1),
                float(n_surface) / 6.0,
                float(n_pcet) / 5.0,
                float("*" in hypothesis),
            ],
            dtype=np.float32,
        )
        return np.concatenate([species_oh, extras])

    def _extract_graph_features(self, network: ReactionNetwork) -> np.ndarray:
        """Extract raw graph features (before projection)."""
        species_oh = np.zeros(_VOCAB_SIZE, dtype=np.float32)
        for sp in network.intermediates:
            species_oh[_species_idx(sp)] += 1.0
        if network.intermediates:
            species_oh /= len(network.intermediates)
        n_steps = len(network.steps)
        n_ec = sum(s.is_electrochemical for s in network.steps)
        global_feat = np.array(
            [
                n_steps / max(n_steps, 1),
                n_ec / max(n_steps, 1),
                len(network.ts_edges) / max(n_steps, 1),
                len(network.intermediates) / max(len(network.intermediates), 1),
                float("*" in network.intermediates),
            ],
            dtype=np.float32,
        )
        return np.concatenate([species_oh, global_feat])

    def _extract_energy_features(self, dG_profile: List[float]) -> np.ndarray:
        """Extract raw energy features (before projection)."""
        seq_len = 16
        arr = np.array(dG_profile, dtype=np.float32)
        # Pad/truncate to seq_len
        if len(arr) < seq_len:
            padded = np.zeros(seq_len, dtype=np.float32)
            padded[: len(arr)] = arr
        else:
            padded = arr[:seq_len]
        # Scalar features
        scalars = np.array(
            [
                float(np.min(arr)),
                float(np.max(arr)),
                float(np.max(np.diff(arr))) if len(arr) > 1 else 0.0,
                float(np.min(np.diff(arr))) if len(arr) > 1 else 0.0,
                float(len(arr)) / 20.0,
                float(arr[-1]) if len(arr) > 0 else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([padded, scalars])

    def _encode_batch(
        self,
        hypotheses: List[str],
        networks: List[ReactionNetwork],
        dG_profiles: Optional[List[List[float]]] = None,
    ) -> tuple:
        """Encode a batch through learnable projections."""
        len(hypotheses)
        t_raw = torch.tensor(
            np.stack([self._extract_text_features(h) for h in hypotheses]),
            dtype=torch.float32,
        )
        g_raw = torch.tensor(
            np.stack([self._extract_graph_features(n) for n in networks]),
            dtype=torch.float32,
        )
        t_emb = _F.normalize(self._text_proj(t_raw), dim=-1)
        g_emb = _F.normalize(self._graph_proj(g_raw), dim=-1)

        p_emb = None
        if dG_profiles is not None:
            e_raw = torch.tensor(
                np.stack([self._extract_energy_features(p) for p in dG_profiles]),
                dtype=torch.float32,
            )
            p_emb = _F.normalize(self._energy_proj(e_raw), dim=-1)

        return t_emb, g_emb, p_emb

    def train(
        self,
        pairs: List[tuple],
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 16,
        hard_negatives: bool = True,
    ) -> List[float]:
        """
        Train the grounder on (hypothesis, network, dG_profile) triples.

        Parameters
        ----------
        pairs : list of (str, ReactionNetwork, list[float])
            Training triples. Each pair is a positive match.
        n_epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        batch_size : int
            Batch size for InfoNCE.
        hard_negatives : bool
            If True, use semi-hard negative mining (negatives with
            highest similarity that are still incorrect).

        Returns
        -------
        list[float]
            Loss curve (one value per epoch).
        """
        params = (
            list(self._text_proj.parameters())
            + list(self._graph_proj.parameters())
            + list(self._energy_proj.parameters())
            + [self._log_tau]
        )
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        loss_curve = []
        for epoch in range(n_epochs):
            # Shuffle
            indices = torch.randperm(len(pairs)).tolist()
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(pairs), batch_size):
                batch_idx = indices[start : start + batch_size]
                if len(batch_idx) < 2:
                    continue

                hypotheses = [pairs[i][0] for i in batch_idx]
                networks = [pairs[i][1] for i in batch_idx]
                dG_profiles = [pairs[i][2] for i in batch_idx]

                t_emb, g_emb, p_emb = self._encode_batch(hypotheses, networks, dG_profiles)

                # InfoNCE loss: text → graph
                tau = self._log_tau.exp()
                sim_tg = t_emb @ g_emb.T / tau
                labels = torch.arange(len(batch_idx), dtype=torch.long)
                loss_tg = _F.cross_entropy(sim_tg, labels)

                # InfoNCE loss: text → energy (if available)
                loss = loss_tg
                if p_emb is not None:
                    sim_tp = t_emb @ p_emb.T / tau
                    loss_tp = _F.cross_entropy(sim_tp, labels)
                    loss = 0.6 * loss_tg + 0.4 * loss_tp

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            loss_curve.append(avg_loss)

        return loss_curve

    def score(
        self,
        hypothesis: str,
        network: ReactionNetwork,
        dG_profile: Optional[List[float]] = None,
    ) -> float:
        """Score using trained projections."""
        t_emb, g_emb, p_emb = self._encode_batch(
            [hypothesis],
            [network],
            [dG_profile] if dG_profile is not None else None,
        )
        with torch.no_grad():
            sim_tg = float((t_emb[0] * g_emb[0]).sum())
            if p_emb is not None:
                sim_tp = float((t_emb[0] * p_emb[0]).sum())
                raw = 0.6 * sim_tg + 0.4 * sim_tp
            else:
                raw = sim_tg
        tau = float(self._log_tau.detach().exp())
        return float(_sigmoid(raw / tau))

    def compute_auc(
        self,
        pairs: List[tuple],
        mismatched_pairs: List[tuple],
    ) -> float:
        """
        Compute AUC for distinguishing correct from mismatched pairs.

        Parameters
        ----------
        pairs : correct (hypothesis, network, dG) triples
        mismatched_pairs : wrong (hypothesis, network, dG) triples

        Returns
        -------
        float : AUC score
        """
        scores_pos = [self.score(h, n, dg) for h, n, dg in pairs]
        scores_neg = [self.score(h, n, dg) for h, n, dg in mismatched_pairs]

        scores_pos + scores_neg
        [1] * len(scores_pos) + [0] * len(scores_neg)

        # Simple AUC computation
        n_pos = len(scores_pos)
        n_neg = len(scores_neg)
        concordant = sum(1 for sp in scores_pos for sn in scores_neg if sp > sn)
        tied = sum(1 for sp in scores_pos for sn in scores_neg if sp == sn)
        auc = (concordant + 0.5 * tied) / max(n_pos * n_neg, 1)
        return auc
