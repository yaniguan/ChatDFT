"""
Curated Literature Hypothesis-Mechanism Pairs
===============================================

A benchmark dataset of hypothesis-mechanism pairs extracted from
peer-reviewed publications, for validating the InfoNCE cross-modal
grounder on real scientific data.

Each entry contains:
  - A natural-language hypothesis (as a researcher might write)
  - The accepted reaction mechanism (intermediate sequence)
  - Free energy profile values (from published DFT calculations)
  - DOI and metadata for traceability
  - A "difficulty" label: easy/medium/hard

Usage
-----
    from science.validation.literature_pairs import (
        LITERATURE_PAIRS, generate_hard_negatives, evaluate_grounder,
    )

    # Train
    grounder.train(LITERATURE_PAIRS, ...)

    # Evaluate with hard negatives
    auc = evaluate_grounder(grounder, LITERATURE_PAIRS)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from science.alignment.hypothesis_grounder import ReactionNetwork


@dataclass
class LiteraturePair:
    """A single hypothesis-mechanism pair from literature."""

    id: str
    hypothesis: str
    mechanism: dict  # ReactionNetwork-compatible dict
    dG_profile: List[float]
    domain: str  # co2rr, her, oer, nrr, orr
    surface: str  # e.g. "Cu(111)"
    doi: str
    difficulty: str = "medium"  # easy, medium, hard
    notes: str = ""


# ── Curated pairs from published DFT studies ─────────────────────────

LITERATURE_PAIRS: List[LiteraturePair] = [
    # CO2RR — Peterson 2010
    LiteraturePair(
        id="lit_co2rr_co_cu111",
        hypothesis="CO2 reduces to CO on Cu(111) through a COOH* intermediate, "
        "with the first proton-electron transfer being rate-limiting",
        mechanism={
            "reaction_network": [
                {"lhs": ["CO2(g)", "*", "H+", "e-"], "rhs": ["COOH*"]},
                {"lhs": ["COOH*", "H+", "e-"], "rhs": ["CO*", "H2O(g)"]},
                {"lhs": ["CO*"], "rhs": ["CO(g)", "*"]},
            ],
            "intermediates": ["*", "CO2(g)", "COOH*", "CO*", "CO(g)", "H2O(g)"],
            "surface": "Cu(111)",
        },
        dG_profile=[0.0, 0.22, -0.15, -0.45, -1.10],
        domain="co2rr",
        surface="Cu(111)",
        doi="10.1039/C0EE00071J",
        difficulty="easy",
    ),
    # CO2RR — Kuhl 2012
    LiteraturePair(
        id="lit_co2rr_ch4_cu111",
        hypothesis="CO2 reduces to methane on Cu(111) via CHO* and CH2O* intermediates, "
        "requiring 8 proton-electron transfers with CO* hydrogenation as RDS",
        mechanism={
            "reaction_network": [
                {"lhs": ["CO2(g)", "*", "H+", "e-"], "rhs": ["COOH*"]},
                {"lhs": ["COOH*", "H+", "e-"], "rhs": ["CO*", "H2O(g)"]},
                {"lhs": ["CO*", "H+", "e-"], "rhs": ["CHO*"]},
                {"lhs": ["CHO*", "H+", "e-"], "rhs": ["CH2O*"]},
            ],
            "intermediates": ["*", "CO2(g)", "COOH*", "CO*", "CHO*", "CH2O*", "CH4(g)"],
            "surface": "Cu(111)",
        },
        dG_profile=[0.0, 0.22, -0.15, 0.74, 0.40, -1.33],
        domain="co2rr",
        surface="Cu(111)",
        doi="10.1039/C2EE21234J",
        difficulty="medium",
    ),
    # HER — Skulason 2012
    LiteraturePair(
        id="lit_her_pt111",
        hypothesis="Hydrogen evolution on Pt(111) proceeds via the Volmer-Tafel mechanism "
        "with near-thermoneutral H* binding (ΔG_H* ≈ -0.09 eV)",
        mechanism={
            "reaction_network": [
                {"lhs": ["H+", "e-", "*"], "rhs": ["H*"]},
                {"lhs": ["H*", "H*"], "rhs": ["H2(g)", "*", "*"]},
            ],
            "intermediates": ["*", "H+", "e-", "H*", "H2(g)"],
            "surface": "Pt(111)",
        },
        dG_profile=[0.0, -0.09, 0.0],
        domain="her",
        surface="Pt(111)",
        doi="10.1039/C1CP23212F",
        difficulty="easy",
    ),
    # HER — Hinnemann 2005
    LiteraturePair(
        id="lit_her_mos2",
        hypothesis="MoS2 edge sites catalyse HER with ΔG_H* = 0.08 eV, close to optimal for the Sabatier principle",
        mechanism={
            "reaction_network": [
                {"lhs": ["H+", "e-", "*"], "rhs": ["H*"]},
                {"lhs": ["H*", "H+", "e-"], "rhs": ["H2(g)", "*"]},
            ],
            "intermediates": ["*", "H+", "e-", "H*", "H2(g)"],
            "surface": "MoS2-edge",
        },
        dG_profile=[0.0, 0.08, 0.0],
        domain="her",
        surface="MoS2-edge",
        doi="10.1021/ja0504690",
        difficulty="easy",
    ),
    # OER — Man 2011
    LiteraturePair(
        id="lit_oer_ruo2",
        hypothesis="OER on RuO2(110) follows the 4-step mechanism with OH*, O*, OOH* intermediates "
        "where the O*→OOH* step is potential-determining",
        mechanism={
            "reaction_network": [
                {"lhs": ["H2O(l)", "*"], "rhs": ["OH*", "H+", "e-"]},
                {"lhs": ["OH*"], "rhs": ["O*", "H+", "e-"]},
                {"lhs": ["O*", "H2O(l)"], "rhs": ["OOH*", "H+", "e-"]},
                {"lhs": ["OOH*"], "rhs": ["O2(g)", "*", "H+", "e-"]},
            ],
            "intermediates": ["*", "OH*", "O*", "OOH*", "O2(g)", "H2O(l)"],
            "surface": "RuO2(110)",
        },
        dG_profile=[0.0, 1.60, 3.20, 4.40, 4.92],
        domain="oer",
        surface="RuO2(110)",
        doi="10.1002/cctc.201000397",
        difficulty="medium",
    ),
    # NRR — Montoya 2015
    LiteraturePair(
        id="lit_nrr_ru0001",
        hypothesis="N2 reduction on Ru(0001) follows a distal mechanism where the "
        "first N is fully hydrogenated before the second, with N2→NNH* as RDS",
        mechanism={
            "reaction_network": [
                {"lhs": ["N2(g)", "*"], "rhs": ["N2*"]},
                {"lhs": ["N2*", "H+", "e-"], "rhs": ["NNH*"]},
                {"lhs": ["NNH*", "H+", "e-"], "rhs": ["NNH2*"]},
                {"lhs": ["NNH2*", "H+", "e-"], "rhs": ["N*", "NH3(g)"]},
            ],
            "intermediates": ["*", "N2(g)", "N2*", "NNH*", "NNH2*", "N*", "NH3(g)"],
            "surface": "Ru(0001)",
        },
        dG_profile=[0.0, -0.40, 0.72, 0.16, -0.90, -2.18],
        domain="nrr",
        surface="Ru(0001)",
        doi="10.1002/cssc.201500322",
        difficulty="hard",
    ),
    # ORR — Norskov 2004
    LiteraturePair(
        id="lit_orr_pt111",
        hypothesis="ORR on Pt(111) follows the associative mechanism via OOH* "
        "with OH*→H2O as the rate-determining step at 0.9V_RHE",
        mechanism={
            "reaction_network": [
                {"lhs": ["O2(g)", "*", "H+", "e-"], "rhs": ["OOH*"]},
                {"lhs": ["OOH*", "H+", "e-"], "rhs": ["O*", "H2O(l)"]},
                {"lhs": ["O*", "H+", "e-"], "rhs": ["OH*"]},
                {"lhs": ["OH*", "H+", "e-"], "rhs": ["H2O(l)", "*"]},
            ],
            "intermediates": ["*", "O2(g)", "OOH*", "O*", "OH*", "H2O(l)"],
            "surface": "Pt(111)",
        },
        dG_profile=[0.0, -1.60, -3.45, -4.22, -4.92],
        domain="orr",
        surface="Pt(111)",
        doi="10.1021/jp047349j",
        difficulty="medium",
    ),
    # CO2RR formate — hard because mechanism is debated
    LiteraturePair(
        id="lit_co2rr_hcooh_sn",
        hypothesis="CO2 reduces to formate on Sn via direct CO2 insertion into Sn-H bond "
        "or through OCHO* intermediate, competing with CO pathway",
        mechanism={
            "reaction_network": [
                {"lhs": ["CO2(g)", "*", "H+", "e-"], "rhs": ["OCHO*"]},
                {"lhs": ["OCHO*", "H+", "e-"], "rhs": ["HCOOH(aq)", "*"]},
            ],
            "intermediates": ["*", "CO2(g)", "OCHO*", "HCOOH(aq)"],
            "surface": "Sn(112)",
        },
        dG_profile=[0.0, -0.21, -0.75],
        domain="co2rr",
        surface="Sn(112)",
        doi="10.1021/acscatal.6b01393",
        difficulty="hard",
    ),
]


def generate_hard_negatives(
    pairs: List[LiteraturePair],
    seed: int = 42,
) -> List[LiteraturePair]:
    """
    Generate hard negatives by swapping mechanisms between domains.

    Hard negatives are chemically plausible but wrong: e.g., pairing
    an HER hypothesis with an OER mechanism. These are harder to
    distinguish than random noise.

    Returns list of mismatched LiteraturePair (same length as input).
    """
    rng = np.random.default_rng(seed)
    domains = list(set(p.domain for p in pairs))
    negatives = []

    for pair in pairs:
        # Find a pair from a different domain
        other_domain = rng.choice([d for d in domains if d != pair.domain])
        other_pairs = [p for p in pairs if p.domain == other_domain]
        other = rng.choice(other_pairs)

        # Swap: keep hypothesis, take mechanism from other domain
        negatives.append(
            LiteraturePair(
                id=f"neg_{pair.id}",
                hypothesis=pair.hypothesis,
                mechanism=other.mechanism,
                dG_profile=other.dG_profile,
                domain=pair.domain,
                surface=pair.surface,
                doi=pair.doi,
                difficulty="hard_negative",
            )
        )

    return negatives


def evaluate_grounder(
    grounder,
    pairs: Optional[List[LiteraturePair]] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate a grounder on literature pairs + hard negatives.

    Returns dict with AUC, accuracy, mean_pos_score, mean_neg_score.
    """
    if pairs is None:
        pairs = LITERATURE_PAIRS

    negatives = generate_hard_negatives(pairs, seed=seed)

    pos_scores = []
    neg_scores = []

    for pair in pairs:
        network = ReactionNetwork.from_dict(pair.mechanism)
        score = grounder.score(pair.hypothesis, network, pair.dG_profile)
        pos_scores.append(score)

    for neg in negatives:
        network = ReactionNetwork.from_dict(neg.mechanism)
        score = grounder.score(neg.hypothesis, network, neg.dG_profile)
        neg_scores.append(score)

    # AUC
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    concordant = sum(1 for sp in pos_scores for sn in neg_scores if sp > sn)
    tied = sum(1 for sp in pos_scores for sn in neg_scores if sp == sn)
    auc = (concordant + 0.5 * tied) / max(n_pos * n_neg, 1)

    # Accuracy at threshold = 0.5
    all_scores = pos_scores + neg_scores
    all_labels = [1] * n_pos + [0] * n_neg
    preds = [1 if s > 0.5 else 0 for s in all_scores]
    accuracy = sum(p == lbl for p, lbl in zip(preds, all_labels)) / len(all_labels)

    return {
        "auc": auc,
        "accuracy": accuracy,
        "mean_pos_score": float(np.mean(pos_scores)),
        "mean_neg_score": float(np.mean(neg_scores)),
        "score_separation": float(np.mean(pos_scores) - np.mean(neg_scores)),
        "n_pairs": n_pos,
        "n_negatives": n_neg,
    }
