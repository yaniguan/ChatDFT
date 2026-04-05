"""
Physics-Informed Structure Generation
======================================
Scientific motivation
---------------------
Training a neural-network interatomic potential (NNP) requires a dataset
that (a) covers the relevant configuration space densely, (b) does not
oversample near-equilibrium structures, and (c) does not waste DFT budget
on configurations the NNP already handles well.

This module provides three complementary strategies:

1. **Harmonic (Einstein) rattle**      — temperature-scaled Gaussian noise
2. **Normal-mode sampling**            — phonon-eigenmode-weighted displacement
3. **Active-learning uncertainty sampling** — committee disagreement as a
                                          query criterion

The strategies are physically motivated:

Einstein rattle
  Each atom is an independent quantum harmonic oscillator at temperature T.
  The displacement standard deviation follows:

      σᵢ = sqrt( ħ / (2 mᵢ ωᵢ) · coth(ħωᵢ / 2kBT) )

  High-T (classical) limit: σᵢ → sqrt(kBT / (mᵢ ωᵢ²))
  Low-T (ZPE) limit:       σᵢ → sqrt(ħ / (2 mᵢ ωᵢ))

Normal-mode sampling
  For each phonon mode ν with eigenvalue λν:
      amplitude Aν = sqrt(kBT / λν)    [equipartition]
  Mode displacement:  Δq = Σν cν Aν φν  where cν ∈ {−1, +1} randomly.
  Acoustic modes (λ < 1e-3 eV/Å²) are excluded.

Active-learning criterion
  Given a committee of N models {f_k}, the uncertainty for a candidate x is:
      σ²(x) = (1/N) Σk (f_k(x) − <f(x)>)²   (per-atom energy variance)
  Structures with σ(x) / N_atoms > threshold are sent to DFT.

Key references
--------------
[1] Behler & Parrinello, PRL 98, 146401 (2007)   — NNP training concept
[2] Botu & Ramprasad, IJQC 115, 1074 (2015)      — active learning for NNPs
[3] Vandermause et al., npj Comput. Mater. 6, 20 (2020) — on-the-fly AL
[4] Eriksson et al., Adv. Theory Simul. 2, 1800184 (2019) — TDEP rattling
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

# Constants (SI-based, converted to eV·Å·amu units)
_KB_EV = 8.617333e-5  # eV / K
_HBAR = 0.06466  # eV · fs  (ħ in eV·fs)
_AMU_TO_EV_FS2_A2 = 1.0  # internal: energies in eV, distances in Å, mass in amu


# ---------------------------------------------------------------------------
# Minimal atoms container (ASE-compatible interface)
# ---------------------------------------------------------------------------


@dataclass
class AtomsLike:
    """
    Thin wrapper so this module works with or without ASE installed.
    If ASE is available, pass actual ase.Atoms objects — this class is
    only used for standalone testing.
    """

    positions: np.ndarray  # (N, 3) Å
    numbers: np.ndarray  # (N,)   atomic numbers
    cell: np.ndarray  # (3, 3) Å
    masses: np.ndarray  # (N,)   amu

    def copy(self) -> "AtomsLike":
        return AtomsLike(
            positions=self.positions.copy(),
            numbers=self.numbers.copy(),
            cell=self.cell.copy(),
            masses=self.masses.copy(),
        )

    def get_positions(self):
        return self.positions

    def set_positions(self, p):
        self.positions = np.array(p)

    def get_masses(self):
        return self.masses


# ---------------------------------------------------------------------------
# Atomic masses (amu) for common elements
# ---------------------------------------------------------------------------
_MASSES: dict = {
    1: 1.008,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    13: 26.982,
    14: 28.086,
    22: 47.867,
    23: 50.942,
    24: 51.996,
    25: 54.938,
    26: 55.845,
    27: 58.933,
    28: 58.693,
    29: 63.546,
    30: 65.38,
    44: 101.07,
    45: 102.906,
    46: 106.42,
    47: 107.868,
    74: 183.84,
    75: 186.207,
    76: 190.23,
    77: 192.217,
    78: 195.084,
    79: 196.967,
}


# ---------------------------------------------------------------------------
# 1. Einstein / harmonic rattle
# ---------------------------------------------------------------------------


class EinsteinRattler:
    """
    Temperature-scaled atomic perturbations based on the Einstein model.

    Parameters
    ----------
    omega_THz : float
        Einstein frequency in THz. Typical metals: 3–8 THz.
        Use the Debye frequency ωD as a first approximation.
    quantum : bool
        If True, use the full quantum coth expression (includes ZPE).
        If False, use the classical sqrt(kBT / mω²) limit.
    surface_only : bool
        Only rattle atoms in the top `n_surface_layers` layers.
    """

    def __init__(
        self,
        omega_THz: float = 5.0,
        quantum: bool = True,
        surface_only: bool = False,
        n_surface_layers: int = 2,
        rng_seed: Optional[int] = None,
    ):
        self.omega_THz = omega_THz
        self.quantum = quantum
        self.surface_only = surface_only
        self.n_surface_layers = n_surface_layers
        self.rng = np.random.default_rng(rng_seed)

    def _sigma(self, mass_amu: float, T_K: float) -> float:
        """Displacement sigma in Å for one atom."""
        omega_rad_per_fs = 2 * np.pi * self.omega_THz * 1e-3  # THz → rad/fs
        # mass in eV·fs²/Å² (conversion: 1 amu = 0.0103637 eV·fs²/Å²)
        m = mass_amu * 0.0103637
        if T_K < 1e-6:
            if self.quantum:
                return float(np.sqrt(_HBAR / (2 * m * omega_rad_per_fs)))
            return 0.0

        kBT = _KB_EV * T_K
        hw = _HBAR * omega_rad_per_fs

        if self.quantum:
            # σ² = ħ / (2mω) · coth(ħω / 2kBT)
            x = hw / (2 * kBT)
            coth_x = 1.0 / np.tanh(x) if x < 50 else 1.0
            sigma2 = hw / (2 * m * omega_rad_per_fs) * coth_x
        else:
            # Classical: σ² = kBT / (mω²)
            sigma2 = kBT / (m * omega_rad_per_fs**2)
        return float(np.sqrt(max(sigma2, 0.0)))

    def rattle(self, atoms, T_K: float) -> object:
        """
        Return a new atoms object with rattled positions.

        Parameters
        ----------
        atoms : ASE Atoms or AtomsLike
        T_K   : float — temperature in Kelvin
        """
        rattled = _copy_atoms(atoms)
        pos = np.array(rattled.get_positions())
        masses = np.array(rattled.get_masses())
        N = len(masses)

        # Determine which atoms to rattle
        if self.surface_only:
            z = pos[:, 2]
            z_sort = np.sort(np.unique(np.round(z, 1)))[::-1]
            top_zs = set(z_sort[: self.n_surface_layers])
            mask = np.array([round(z[i], 1) in top_zs for i in range(N)])
        else:
            mask = np.ones(N, dtype=bool)

        for i in np.where(mask)[0]:
            m = masses[i]
            sig = self._sigma(m, T_K)
            pos[i] += self.rng.normal(0, sig, size=3)

        rattled.set_positions(pos)
        return rattled

    def generate_batch(self, atoms, T_K: float, n: int) -> List:
        """Generate n independently rattled configurations."""
        return [self.rattle(atoms, T_K) for _ in range(n)]


# ---------------------------------------------------------------------------
# 2. Normal-mode (phonon) sampling
# ---------------------------------------------------------------------------


class NormalModeSampler:
    """
    Sample configurations by exciting phonon eigenmodes.

    Given a dynamical matrix (or Hessian) H in Cartesian coordinates:
        H_ij = ∂²E / ∂uᵢ ∂uⱼ     [eV/Å²]

    Mass-weighted: D_ij = H_ij / sqrt(mᵢ mⱼ)
    Diagonalize:  D = Φᵀ Λ Φ    → eigenvalues λν, eigenvectors φν

    Displacement of atom i in mode ν:
        Δrᵢ(ν) = φ_{νi} · Aν / sqrt(mᵢ)

    Amplitude (equipartition):
        Aν = cν · sqrt(kBT / λν)    cν ∈ {-1, +1}

    Acoustic modes (λ < λ_acoustic_cutoff) are excluded to prevent
    spurious rigid-body translation/rotation.
    """

    def __init__(
        self,
        lambda_acoustic_cutoff: float = 1e-3,  # eV/Å² — below this = acoustic
        include_zpe: bool = True,
        rng_seed: Optional[int] = None,
    ):
        self.lambda_cut = lambda_acoustic_cutoff
        self.include_zpe = include_zpe
        self.rng = np.random.default_rng(rng_seed)
        self._eigvals: Optional[np.ndarray] = None
        self._eigvecs: Optional[np.ndarray] = None
        self._masses: Optional[np.ndarray] = None

    def fit(self, hessian: np.ndarray, masses: np.ndarray) -> "NormalModeSampler":
        """
        Diagonalise the mass-weighted Hessian.

        Parameters
        ----------
        hessian : (3N, 3N) ndarray in eV/Å²
        masses  : (N,) ndarray in amu
        """
        N = len(masses)
        assert hessian.shape == (3 * N, 3 * N), "Hessian must be (3N, 3N)"
        self._masses = masses

        # Mass-weight: D = M^{-1/2} H M^{-1/2}
        m_rep = np.repeat(masses * 0.0103637, 3)  # (3N,) in eV·fs²/Å²
        m_sqrt = np.sqrt(m_rep)
        D = hessian / np.outer(m_sqrt, m_sqrt)
        D = 0.5 * (D + D.T)  # ensure symmetry

        eigvals, eigvecs = np.linalg.eigh(D)
        # Remove acoustic modes (should be ≈0 but can be slightly negative)
        optical_mask = eigvals > self.lambda_cut
        self._eigvals = eigvals[optical_mask]
        self._eigvecs = eigvecs[:, optical_mask]  # (3N, n_optical)
        self._m_sqrt = m_sqrt
        return self

    def sample(self, atoms, T_K: float) -> object:
        """Generate one configuration by random superposition of phonon modes."""
        assert self._eigvals is not None, "Call fit() first."
        sampled = _copy_atoms(atoms)
        pos = np.array(sampled.get_positions())
        N3 = 3 * len(pos)
        kBT = _KB_EV * T_K

        displacement = np.zeros(N3)
        for nu, (lam, phi) in enumerate(zip(self._eigvals, self._eigvecs.T)):
            # Amplitude: classical + optional ZPE correction
            if self.include_zpe:
                # sigma = sqrt( hbar/(2 m_eff sqrt(lam)) coth(...) )
                # simplified: A = sqrt(kBT / lam) * correction
                omega = np.sqrt(max(lam, 1e-20))
                hw = _HBAR * omega
                if T_K > 1e-6:
                    x = hw / (2 * kBT)
                    cth = 1.0 / np.tanh(x) if x < 50 else 1.0
                    A = np.sqrt(hw / (2 * lam) * cth)
                else:
                    A = np.sqrt(hw / (2 * lam))
            else:
                A = np.sqrt(kBT / max(lam, 1e-20))

            sign = self.rng.choice([-1.0, 1.0])
            displacement += sign * A * phi / self._m_sqrt  # mass-unweighted

        pos += displacement.reshape(-1, 3)
        sampled.set_positions(pos)
        return sampled

    def generate_batch(self, atoms, T_K: float, n: int) -> List:
        return [self.sample(atoms, T_K) for _ in range(n)]

    def mode_frequencies_THz(self) -> np.ndarray:
        """Return optical mode frequencies in THz."""
        assert self._eigvals is not None
        # λ in eV/Å², ħ in eV·fs, ω in rad/fs, ν in THz
        return np.sqrt(self._eigvals) / (2 * np.pi * 1e-3)


# ---------------------------------------------------------------------------
# 3. Active-learning uncertainty sampler
# ---------------------------------------------------------------------------


class CommitteeUncertaintySampler:
    """
    Active-learning query strategy: select configurations where a committee
    of models disagrees most.

    Usage
    -----
    >>> sampler = CommitteeUncertaintySampler(committee=[model_a, model_b, model_c])
    >>> query = sampler.select(candidates, n_query=10)
    >>> # send query to DFT, add to training set

    Committee models must implement:
        model.predict_energy(atoms) → float   [eV]

    The uncertainty metric is the *per-atom standard deviation* across
    committee members, which is scale-invariant across system sizes:
        σ(x) = sqrt( Var_k[E_k(x)] ) / N_atoms
    """

    def __init__(
        self,
        committee: List[Callable],
        threshold: float = 0.005,  # eV/atom — typical literature value
    ):
        self.committee = committee
        self.threshold = threshold

    def uncertainty(self, atoms) -> float:
        """
        Compute per-atom energy uncertainty for a single configuration.
        Returns eV/atom.
        """
        N = len(atoms.get_positions())
        Es = [m(atoms) for m in self.committee]
        return float(np.std(Es) / N)

    def select(self, candidates: List, n_query: int = 10, return_scores: bool = False) -> List:
        """
        Rank candidates by uncertainty, return top-n_query.

        Parameters
        ----------
        candidates    : list of atoms objects
        n_query       : how many to send to DFT
        return_scores : if True, return (selected, scores) tuple
        """
        scores = np.array([self.uncertainty(a) for a in candidates])
        idx = np.argsort(scores)[::-1][:n_query]
        selected = [candidates[i] for i in idx]
        if return_scores:
            return selected, scores[idx]
        return selected

    def filter_above_threshold(self, candidates: List) -> Tuple[List, List]:
        """
        Split candidates into (needs_DFT, well_covered).
        needs_DFT : uncertainty > threshold  → send to DFT
        well_covered : uncertainty ≤ threshold → skip DFT
        """
        needs_dft, covered = [], []
        for a in candidates:
            (needs_dft if self.uncertainty(a) > self.threshold else covered).append(a)
        return needs_dft, covered


# ---------------------------------------------------------------------------
# Strain sampling
# ---------------------------------------------------------------------------


def strain_sample(
    atoms, strain_max: float = 0.06, n: int = 10, mode: str = "isotropic", rng_seed: Optional[int] = None
) -> List:
    """
    Generate structures by deforming the cell.

    Modes
    -----
    isotropic   : scale all cell vectors uniformly (EOS sampling)
    deviatoric  : random traceless strain tensor (shear)
    combined    : isotropic + deviatoric

    The strain tensor for isotropic mode:
        ε_ij = δ_ij · ε    ε ∈ [-strain_max, +strain_max]
    """
    rng = np.random.default_rng(rng_seed)
    result = []
    for _ in range(n):
        strained = _copy_atoms(atoms)
        cell = np.array(strained.cell if hasattr(strained, "cell") else strained.cell)
        pos = np.array(strained.get_positions())

        if mode in ("isotropic", "combined"):
            eps = rng.uniform(-strain_max, strain_max)
            F_iso = (1.0 + eps) * np.eye(3)
        else:
            F_iso = np.eye(3)

        if mode in ("deviatoric", "combined"):
            e = rng.uniform(-strain_max / 2, strain_max / 2, 6)
            F_dev = np.eye(3)
            F_dev[0, 1] += e[0]
            F_dev[0, 2] += e[1]
            F_dev[1, 0] += e[2]
            F_dev[1, 2] += e[3]
            F_dev[2, 0] += e[4]
            F_dev[2, 1] += e[5]
        else:
            F_dev = np.eye(3)

        F = F_iso @ F_dev
        new_cell = cell @ F.T
        new_pos = pos @ F.T

        if hasattr(strained, "set_cell"):
            strained.set_cell(new_cell, scale_atoms=False)
        strained.set_positions(new_pos)
        result.append(strained)
    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _copy_atoms(atoms):
    """Deep-copy an ASE Atoms or AtomsLike object."""
    if hasattr(atoms, "copy"):
        return atoms.copy()
    return copy.deepcopy(atoms)
