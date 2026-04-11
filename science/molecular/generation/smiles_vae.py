"""
SMILES VAE — Variational Autoencoder for Molecular Generation
==============================================================

Architecture:
  Encoder: SMILES → GRU → μ, σ → z (latent)
  Decoder: z → GRU → SMILES (teacher-forced)
  Loss: Reconstruction (CE) + KL divergence (β-VAE)

During generation:
  1. Sample z ~ N(0, I) or from a conditioned prior
  2. Decode z → SMILES
  3. Filter: validity, novelty, diversity

For multi-objective generation, the latent space is optimised using
Bayesian optimisation or gradient-based methods to find z vectors
that decode to molecules satisfying multiple property constraints.

Key references:
[1] Gomez-Bombarelli et al., ACS Cent. Sci. 4, 268 (2018) — original SMILES VAE
[2] Kusner et al., ICML 2017 — Grammar VAE
[3] Higgins et al., ICLR 2017 — β-VAE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _check_torch():
    if not _HAS_TORCH:
        raise ImportError("PyTorch required: pip install torch")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class VAEConfig:
    """Configuration for SMILES VAE."""

    vocab_size: int = 45  # from representations.VOCAB_SIZE
    embed_dim: int = 128
    hidden_dim: int = 256
    latent_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    max_len: int = 128
    beta: float = 1.0  # KL weight (β-VAE)
    teacher_forcing: float = 1.0


class SMILESEncoder(nn.Module):
    """GRU encoder: SMILES tokens → latent distribution (μ, logσ²)."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            cfg.embed_dim,
            cfg.hidden_dim,
            cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0,
        )
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

    def forward(self, tokens):
        # tokens: (B, L)
        x = self.embed(tokens)
        _, h = self.gru(x)  # h: (n_layers, B, hidden)
        h = h[-1]  # last layer: (B, hidden)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class SMILESDecoder(nn.Module):
    """GRU decoder: latent z → SMILES tokens."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=0)
        self.z_to_h = nn.Linear(cfg.latent_dim, cfg.hidden_dim * cfg.n_layers)
        self.gru = nn.GRU(
            cfg.embed_dim,
            cfg.hidden_dim,
            cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0,
        )
        self.out = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, z, target_tokens=None):
        """
        Decode latent vector to SMILES.

        If target_tokens provided, use teacher forcing.
        Otherwise, autoregressive generation.
        """
        B = z.shape[0]
        h = self.z_to_h(z).view(self.cfg.n_layers, B, -1).contiguous()

        if target_tokens is not None:
            # Teacher forcing
            x = self.embed(target_tokens[:, :-1])  # shift right
            output, _ = self.gru(x, h)
            logits = self.out(output)  # (B, L-1, vocab)
            return logits

        # Autoregressive generation
        from science.molecular.representations import CHAR_TO_IDX

        sos = CHAR_TO_IDX["<sos>"]
        CHAR_TO_IDX["<eos>"]

        token = torch.full((B, 1), sos, dtype=torch.long, device=z.device)
        outputs = []

        for _ in range(self.cfg.max_len):
            x = self.embed(token)
            out, h = self.gru(x, h)
            logit = self.out(out[:, -1, :])  # (B, vocab)
            outputs.append(logit)

            # Sample or greedy
            token = logit.argmax(dim=-1, keepdim=True)

        return torch.stack(outputs, dim=1)  # (B, max_len, vocab)


class SMILESVAE(nn.Module):
    """
    Complete SMILES Variational Autoencoder.

    Training: encoder(tokens) → (μ, logσ²) → sample z → decoder(z, tokens) → loss
    Generation: sample z ~ N(0,I) → decoder(z) → SMILES
    """

    def __init__(self, cfg: VAEConfig = None):
        super().__init__()
        self.cfg = cfg or VAEConfig()
        self.encoder = SMILESEncoder(self.cfg)
        self.decoder = SMILESDecoder(self.cfg)

    def reparameterize(self, mu, logvar):
        """Reparameterisation trick: z = μ + ε·σ."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, tokens):
        mu, logvar = self.encoder(tokens)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, tokens)
        return logits, mu, logvar

    def generate(self, n: int = 1, temperature: float = 1.0) -> List[str]:
        """Generate SMILES by sampling from the prior."""
        from science.molecular.representations import detokenize_smiles

        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            z = torch.randn(n, self.cfg.latent_dim, device=device) * temperature
            logits = self.decoder(z)  # (n, max_len, vocab)

            # Sample from softmax
            probs = F.softmax(logits / max(temperature, 0.01), dim=-1)
            tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(n, -1)

        smiles = [detokenize_smiles(t.cpu().numpy()) for t in tokens]
        return smiles

    def encode(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode SMILES tokens to latent space."""
        return self.encoder(tokens)

    def decode_from_z(self, z: torch.Tensor) -> List[str]:
        """Decode latent vectors to SMILES."""
        from science.molecular.representations import detokenize_smiles

        self.eval()
        with torch.no_grad():
            logits = self.decoder(z)
            tokens = logits.argmax(dim=-1)

        return [detokenize_smiles(t.cpu().numpy()) for t in tokens]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class VAETrainResult:
    n_epochs: int
    final_recon_loss: float
    final_kl_loss: float
    final_total_loss: float
    valid_rate: float  # % of generated SMILES that are valid
    unique_rate: float  # % of valid SMILES that are unique
    train_time_s: float = 0.0
    loss_curve: List[float] = field(default_factory=list)


def train_vae(
    smiles_list: List[str],
    cfg: VAEConfig = None,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    kl_annealing: bool = True,
    verbose: bool = False,
) -> Tuple[SMILESVAE, VAETrainResult]:
    """
    Train SMILES VAE.

    Parameters
    ----------
    smiles_list : List[str]
        Training SMILES.
    cfg : VAEConfig
        Model configuration.
    n_epochs : int
    lr : float
    batch_size : int
    kl_annealing : bool
        Linearly anneal KL weight from 0 to β over first half of training.
        Prevents posterior collapse (a common VAE failure mode).

    Returns
    -------
    (model, result) tuple.
    """
    _check_torch()
    import time

    from science.molecular.representations import tokenize_smiles, validate_smiles

    if cfg is None:
        cfg = VAEConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SMILESVAE(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Tokenize
    tokens = np.array([tokenize_smiles(s, cfg.max_len) for s in smiles_list])
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)

    loss_curve = []
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(tokens_tensor))
        epoch_recon, epoch_kl, n_batches = 0.0, 0.0, 0

        # KL annealing: linear warmup
        if kl_annealing:
            kl_weight = min(cfg.beta, cfg.beta * (epoch / max(n_epochs // 2, 1)))
        else:
            kl_weight = cfg.beta

        for start in range(0, len(tokens_tensor), batch_size):
            batch = tokens_tensor[perm[start : start + batch_size]].to(device)

            logits, mu, logvar = model(batch)

            # Reconstruction loss (cross-entropy)
            target = batch[:, 1:]  # shift target
            recon_loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                target.reshape(-1),
                ignore_index=0,  # ignore padding
            )

            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.shape[0]

            loss = recon_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        avg_recon = epoch_recon / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)
        loss_curve.append(avg_recon + kl_weight * avg_kl)

        if verbose and epoch % 10 == 0:
            log.info("Epoch %d: recon=%.4f kl=%.4f kl_w=%.3f", epoch, avg_recon, avg_kl, kl_weight)

    # Evaluate generation quality
    generated = model.generate(n=100)
    valid = [s for s in generated if validate_smiles(s)]
    valid_rate = len(valid) / max(len(generated), 1)
    unique_rate = len(set(valid)) / max(len(valid), 1)

    result = VAETrainResult(
        n_epochs=n_epochs,
        final_recon_loss=avg_recon,
        final_kl_loss=avg_kl,
        final_total_loss=loss_curve[-1] if loss_curve else 0,
        valid_rate=valid_rate,
        unique_rate=unique_rate,
        train_time_s=time.time() - t0,
        loss_curve=loss_curve,
    )

    return model, result
