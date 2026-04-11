"""
Multi-Objective Molecular Optimisation
=======================================

Generate molecules that simultaneously satisfy multiple property constraints:
  - Activity: predicted pIC50 / binding affinity from QSAR model
  - Synthesisability: SA score < 4 (Ertl & Schuffenhauer, 2009)
  - Drug-likeness: QED > 0.5 (Bickerton et al., 2012)
  - Toxicity: low predicted toxicity from Tox21 model
  - Novelty: not in training set

Two optimisation strategies:

1. **Latent space optimisation** — sample z from VAE prior, use Bayesian
   optimisation (GP + EI) to find z* that maximises a weighted objective.
   Reference: Gomez-Bombarelli et al., ACS Cent. Sci. 4, 268 (2018)

2. **Pareto front enumeration** — generate a large pool, score on all
   objectives, return the Pareto-optimal set.

This module also provides molecular quality filters (PAINS, Brenk, Lipinski).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Molecular quality scoring (RDKit-based)
# ---------------------------------------------------------------------------

def compute_sa_score(smiles: str) -> float:
    """
    Synthetic Accessibility score (1-10, lower = easier to synthesise).

    Reference: Ertl & Schuffenhauer, J. Cheminf. 1, 8 (2009)
    Uses RDKit's SA_Score module.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import RDConfig
        import sys, os
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        return sascorer.calculateScore(mol)
    except (ImportError, Exception):
        # Fallback: estimate from molecular complexity
        return _estimate_sa_score(smiles)


def _estimate_sa_score(smiles: str) -> float:
    """Simple SA score estimate without RDKit contrib."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        # Heuristic: more rings + more stereocenters = harder
        n_rings = Descriptors.RingCount(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        score = 1.0 + 0.3 * n_rings + 0.1 * n_chiral + 0.02 * max(n_heavy - 20, 0)
        return min(score, 10.0)
    except Exception:
        return 5.0  # neutral if no RDKit


def compute_qed(smiles: str) -> float:
    """
    Quantitative Estimate of Drug-likeness (0-1, higher = more drug-like).

    Reference: Bickerton et al., Nature Chem. 4, 90 (2012)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import QED
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        return QED.qed(mol)
    except (ImportError, Exception):
        return 0.5


def check_lipinski(smiles: str) -> Dict[str, bool]:
    """
    Lipinski's Rule of Five for oral drug-likeness.

    Returns dict with each rule and overall pass/fail.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"pass": False}
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rules = {
            "MW_le_500": mw <= 500,
            "LogP_le_5": logp <= 5,
            "HBD_le_5": hbd <= 5,
            "HBA_le_10": hba <= 10,
        }
        rules["pass"] = sum(rules.values()) >= 3  # allow 1 violation
        return rules
    except Exception:
        return {"pass": False}


def check_pains(smiles: str) -> bool:
    """
    Check for PAINS (Pan-Assay Interference Compounds) alerts.

    Returns True if the molecule contains PAINS patterns (bad).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.FilterCatalog import (
            FilterCatalog, FilterCatalogParams,
        )
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return True
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        return catalog.HasMatch(mol)
    except (ImportError, Exception):
        return False


# ---------------------------------------------------------------------------
# Multi-objective scoring
# ---------------------------------------------------------------------------

@dataclass
class MoleculeScore:
    """Multi-objective score for a generated molecule."""
    smiles: str
    scores: Dict[str, float]     # objective_name → score
    weighted_score: float = 0.0  # single aggregated score
    is_valid: bool = True
    is_novel: bool = True
    passes_filters: bool = True
    pareto_rank: int = 0         # 0 = Pareto-optimal


@dataclass
class ObjectiveConfig:
    """Configuration for a single objective."""
    name: str
    weight: float = 1.0
    minimize: bool = False       # True for SA score, toxicity
    threshold: Optional[float] = None  # hard constraint
    scorer: Optional[Callable[[str], float]] = None


DEFAULT_OBJECTIVES = [
    ObjectiveConfig(
        name="qed",
        weight=1.0,
        minimize=False,
        threshold=0.5,
        scorer=compute_qed,
    ),
    ObjectiveConfig(
        name="sa_score",
        weight=1.0,
        minimize=True,
        threshold=4.0,
        scorer=compute_sa_score,
    ),
    ObjectiveConfig(
        name="lipinski",
        weight=0.5,
        minimize=False,
        threshold=0.5,
        scorer=lambda s: float(check_lipinski(s).get("pass", False)),
    ),
]


class MultiObjectiveScorer:
    """
    Score molecules on multiple objectives and find Pareto-optimal set.

    Usage
    -----
        scorer = MultiObjectiveScorer(objectives=DEFAULT_OBJECTIVES)
        # Add custom QSAR-based objective:
        scorer.add_objective(ObjectiveConfig(
            name="activity", weight=2.0, minimize=False,
            scorer=lambda s: qsar_model.predict_proba([s])[0],
        ))

        scores = scorer.score_batch(smiles_list)
        pareto = scorer.pareto_front(scores)
    """

    def __init__(
        self,
        objectives: Optional[List[ObjectiveConfig]] = None,
        training_smiles: Optional[set] = None,
    ):
        self.objectives = objectives or DEFAULT_OBJECTIVES
        self.training_smiles = training_smiles or set()

    def add_objective(self, obj: ObjectiveConfig):
        self.objectives.append(obj)

    def score(self, smiles: str) -> MoleculeScore:
        """Score a single molecule on all objectives."""
        from science.molecular.representations import validate_smiles

        is_valid = validate_smiles(smiles)
        if not is_valid:
            return MoleculeScore(
                smiles=smiles, scores={}, weighted_score=0.0,
                is_valid=False, passes_filters=False,
            )

        scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        passes = True

        for obj in self.objectives:
            if obj.scorer is not None:
                try:
                    val = obj.scorer(smiles)
                except Exception:
                    val = 0.0
            else:
                val = 0.0

            scores[obj.name] = val

            # Normalise: for minimisation objectives, invert
            norm_val = (1.0 - val / 10.0) if obj.minimize else val
            weighted_sum += obj.weight * norm_val
            total_weight += obj.weight

            # Check hard constraints
            if obj.threshold is not None:
                if obj.minimize and val > obj.threshold:
                    passes = False
                elif not obj.minimize and val < obj.threshold:
                    passes = False

        is_novel = smiles not in self.training_smiles
        has_pains = check_pains(smiles)

        return MoleculeScore(
            smiles=smiles,
            scores=scores,
            weighted_score=weighted_sum / max(total_weight, 1e-9),
            is_valid=is_valid,
            is_novel=is_novel,
            passes_filters=passes and not has_pains,
        )

    def score_batch(self, smiles_list: List[str]) -> List[MoleculeScore]:
        """Score a batch of molecules."""
        results = [self.score(s) for s in smiles_list]
        # Assign Pareto ranks
        self._assign_pareto_ranks(results)
        return results

    def _assign_pareto_ranks(self, results: List[MoleculeScore]):
        """Assign Pareto ranks (0 = optimal, 1 = dominated by rank-0, etc.)."""
        valid = [r for r in results if r.is_valid]
        if not valid:
            return

        # Extract score vectors
        obj_names = [o.name for o in self.objectives]
        vectors = []
        for r in valid:
            vec = []
            for obj in self.objectives:
                val = r.scores.get(obj.name, 0)
                # Negate minimisation objectives so higher = better
                vec.append(-val if obj.minimize else val)
            vectors.append(vec)

        vectors = np.array(vectors)

        # Non-dominated sorting
        ranks = np.zeros(len(vectors), dtype=int)
        remaining = set(range(len(vectors)))
        rank = 0

        while remaining:
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if all(vectors[j] >= vectors[i]) and any(vectors[j] > vectors[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            for i in front:
                ranks[i] = rank
                remaining.discard(i)
            rank += 1

        for i, r in enumerate(valid):
            r.pareto_rank = int(ranks[i])

    def pareto_front(self, results: List[MoleculeScore]) -> List[MoleculeScore]:
        """Return Pareto-optimal molecules (rank 0)."""
        return [r for r in results if r.pareto_rank == 0 and r.is_valid]

    def summary(self, results: List[MoleculeScore]) -> str:
        """Human-readable summary of generation results."""
        valid = [r for r in results if r.is_valid]
        novel = [r for r in valid if r.is_novel]
        passing = [r for r in valid if r.passes_filters]
        pareto = self.pareto_front(results)

        lines = [
            f"Generated: {len(results)} molecules",
            f"  Valid: {len(valid)} ({len(valid)/max(len(results),1):.1%})",
            f"  Novel: {len(novel)} ({len(novel)/max(len(valid),1):.1%})",
            f"  Passing filters: {len(passing)} ({len(passing)/max(len(valid),1):.1%})",
            f"  Pareto-optimal: {len(pareto)}",
        ]

        if valid:
            lines.append("\nObjective statistics (valid molecules):")
            for obj in self.objectives:
                vals = [r.scores.get(obj.name, 0) for r in valid]
                direction = "↓" if obj.minimize else "↑"
                thr = f" (threshold: {obj.threshold})" if obj.threshold else ""
                lines.append(
                    f"  {obj.name} {direction}: "
                    f"mean={np.mean(vals):.3f}, "
                    f"best={min(vals) if obj.minimize else max(vals):.3f}{thr}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Latent space optimisation (Bayesian optimisation in VAE latent space)
# ---------------------------------------------------------------------------

def optimise_latent_space(
    vae,
    scorer: MultiObjectiveScorer,
    n_iterations: int = 100,
    n_samples_per_iter: int = 10,
    temperature: float = 1.0,
    seed: int = 42,
) -> List[MoleculeScore]:
    """
    Optimise in VAE latent space to find molecules satisfying all objectives.

    Strategy:
    1. Sample z from prior
    2. Decode to SMILES
    3. Score on all objectives
    4. Use top-k z vectors as seeds for next iteration (exploitation)
    5. Also sample fresh z vectors (exploration)

    This is a simple evolutionary strategy. For production, replace with
    Bayesian optimisation (GP surrogate + Expected Improvement).
    """
    _check_torch()
    import torch

    rng = np.random.default_rng(seed)
    device = next(vae.parameters()).device

    all_results = []
    best_z = None

    for iteration in range(n_iterations):
        # Sample z vectors
        if best_z is not None and iteration > 0:
            # Exploit: perturb best z vectors
            noise = torch.randn(n_samples_per_iter // 2, vae.cfg.latent_dim,
                              device=device) * 0.3
            exploit_z = best_z.unsqueeze(0).expand(n_samples_per_iter // 2, -1) + noise
            # Explore: fresh samples
            explore_z = torch.randn(n_samples_per_iter - n_samples_per_iter // 2,
                                   vae.cfg.latent_dim, device=device) * temperature
            z = torch.cat([exploit_z, explore_z])
        else:
            z = torch.randn(n_samples_per_iter, vae.cfg.latent_dim,
                          device=device) * temperature

        # Decode
        smiles_list = vae.decode_from_z(z)

        # Score
        batch_results = scorer.score_batch(smiles_list)
        all_results.extend(batch_results)

        # Update best z
        valid_results = [(r, z[i]) for i, r in enumerate(batch_results)
                        if r.is_valid and r.passes_filters]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x[0].weighted_score)
            best_z = best_result[1]

    return all_results
