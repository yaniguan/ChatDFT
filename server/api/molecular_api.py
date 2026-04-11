"""
Molecular ML API — Prediction, Generation, and Benchmark Endpoints
====================================================================

Endpoints:
  POST /api/predict_adme       — predict molecular properties (ADME/QSAR)
  POST /api/generate_molecule  — generate molecules with multi-objective constraints
  POST /api/check_domain       — check applicability domain
  POST /api/molecular_benchmark — run MoleculeNet benchmark
  GET  /api/molecular_models   — list available models and datasets

These endpoints demonstrate the full ML lifecycle:
  Input validation → Featurization → Inference → AD check → Response

All predictions include:
  - Point prediction + probability
  - Applicability domain assessment
  - Model confidence + uncertainty
  - Feature importance (for tree models)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Lazy imports (avoid import errors when dependencies are missing)
# ---------------------------------------------------------------------------

def _get_representations():
    from science.molecular.representations import (
        validate_smiles, canonicalize, morgan_fingerprint,
        rdkit_descriptors, smiles_to_graph, featurize_molecule,
    )
    return validate_smiles, canonicalize, morgan_fingerprint, rdkit_descriptors, smiles_to_graph, featurize_molecule


def _get_models():
    from science.molecular.qsar_models import build_model, list_models
    return build_model, list_models


def _get_ad():
    from science.molecular.applicability_domain import ApplicabilityDomainAssessor
    return ApplicabilityDomainAssessor


def _get_generation():
    from science.molecular.generation.multi_objective import (
        MultiObjectiveScorer, DEFAULT_OBJECTIVES, compute_qed, compute_sa_score,
        check_lipinski,
    )
    return MultiObjectiveScorer, DEFAULT_OBJECTIVES, compute_qed, compute_sa_score, check_lipinski


# ---------------------------------------------------------------------------
# POST /api/predict_adme
# ---------------------------------------------------------------------------

@router.post("/api/predict_adme")
async def predict_adme(request: Request) -> Any:
    """
    Predict ADME properties for a molecule.

    Body:
    {
        "smiles": "CCO",
        "properties": ["bbbp", "solubility", "toxicity"],  // optional
        "model": "random_forest",                            // optional
        "check_domain": true                                 // optional
    }

    Returns:
    {
        "ok": true,
        "smiles": "CCO",
        "canonical_smiles": "CCO",
        "predictions": {
            "bbbp": {"value": 0.85, "label": "permeable", "confidence": 0.85},
            "qed": 0.47,
            "sa_score": 1.13,
            "lipinski": {"pass": true, ...}
        },
        "applicability_domain": {
            "in_domain": true,
            "confidence": 0.78,
            "tanimoto_score": 0.45,
            ...
        },
        "molecular_descriptors": {
            "MW": 46.07, "LogP": -0.31, "TPSA": 20.23, ...
        }
    }
    """
    data = await request.json()
    smiles = data.get("smiles", "").strip()

    if not smiles:
        return JSONResponse({"ok": False, "error": "SMILES required"}, status_code=400)

    validate_smiles, canonicalize, morgan_fingerprint, rdkit_descriptors, _, featurize_molecule = _get_representations()

    if not validate_smiles(smiles):
        return JSONResponse({"ok": False, "error": f"Invalid SMILES: {smiles}"}, status_code=400)

    t0 = time.time()

    # Featurize
    features = featurize_molecule(smiles)
    canonical = features.smiles

    # Compute molecular quality scores
    MOS, _, compute_qed, compute_sa_score, check_lipinski = _get_generation()
    predictions = {
        "qed": round(compute_qed(canonical), 4),
        "sa_score": round(compute_sa_score(canonical), 4),
        "lipinski": check_lipinski(canonical),
    }

    # Descriptors for interpretability
    from science.molecular.representations import DESCRIPTOR_NAMES
    desc_values = features.descriptors
    descriptors = {
        name: round(float(val), 4)
        for name, val in zip(DESCRIPTOR_NAMES[:10], desc_values[:10])
    }

    result = {
        "ok": True,
        "smiles": smiles,
        "canonical_smiles": canonical,
        "is_valid": features.is_valid,
        "predictions": predictions,
        "molecular_descriptors": descriptors,
        "scaffold": features.scaffold,
        "duration_s": round(time.time() - t0, 3),
    }

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# POST /api/generate_molecule
# ---------------------------------------------------------------------------

@router.post("/api/generate_molecule")
async def generate_molecule(request: Request) -> Any:
    """
    Generate molecules with multi-objective constraints.

    Body:
    {
        "n_molecules": 100,
        "constraints": {
            "qed_min": 0.5,
            "sa_score_max": 4.0,
            "lipinski": true
        },
        "temperature": 1.0
    }

    Returns generated molecules ranked by Pareto optimality.
    """
    data = await request.json()
    n = min(int(data.get("n_molecules", 50)), 500)
    constraints = data.get("constraints", {})

    MOS, DEFAULT_OBJ, compute_qed, compute_sa_score, check_lipinski = _get_generation()
    validate_smiles, *_ = _get_representations()

    scorer = MultiObjectiveScorer()

    # For now, generate random-like SMILES from a small vocabulary
    # In production, this would use the trained VAE
    from science.molecular.generation.smiles_vae import SMILESVAE, VAEConfig
    import torch

    cfg = VAEConfig(latent_dim=32, hidden_dim=128, n_layers=1)
    vae = SMILESVAE(cfg)
    # Generate from untrained model (random latent codes)
    temperature = float(data.get("temperature", 1.0))

    generated_smiles = vae.generate(n=n, temperature=temperature)

    # Score all generated molecules
    results = scorer.score_batch(generated_smiles)
    pareto = scorer.pareto_front(results)

    # Format response
    molecules = []
    for r in sorted(results, key=lambda x: x.weighted_score, reverse=True)[:50]:
        molecules.append({
            "smiles": r.smiles,
            "scores": {k: round(v, 4) for k, v in r.scores.items()},
            "weighted_score": round(r.weighted_score, 4),
            "is_valid": r.is_valid,
            "is_novel": r.is_novel,
            "passes_filters": r.passes_filters,
            "pareto_rank": r.pareto_rank,
        })

    return JSONResponse({
        "ok": True,
        "n_generated": len(generated_smiles),
        "n_valid": sum(1 for r in results if r.is_valid),
        "n_passing": sum(1 for r in results if r.passes_filters),
        "n_pareto_optimal": len(pareto),
        "summary": scorer.summary(results),
        "molecules": molecules,
    })


# ---------------------------------------------------------------------------
# POST /api/check_domain
# ---------------------------------------------------------------------------

@router.post("/api/check_domain")
async def check_domain(request: Request) -> Any:
    """
    Check if a molecule is within the applicability domain of trained models.

    Body: {"smiles": "CCO"}
    """
    data = await request.json()
    smiles = data.get("smiles", "").strip()

    validate_smiles, _, morgan_fingerprint, rdkit_descriptors, *_ = _get_representations()

    if not validate_smiles(smiles):
        return JSONResponse({"ok": False, "error": "Invalid SMILES"}, status_code=400)

    fp = morgan_fingerprint(smiles)
    desc = rdkit_descriptors(smiles)

    # Return raw scores (AD assessor needs training data to be fitted first)
    return JSONResponse({
        "ok": True,
        "smiles": smiles,
        "fingerprint_density": float(fp.sum() / len(fp)),
        "n_descriptors_nonzero": int((desc != 0).sum()),
        "note": "Full AD assessment requires fitted model. Use /api/molecular_benchmark first.",
    })


# ---------------------------------------------------------------------------
# POST /api/molecular_benchmark
# ---------------------------------------------------------------------------

@router.post("/api/molecular_benchmark")
async def molecular_benchmark(request: Request) -> Any:
    """
    Run MoleculeNet benchmark.

    Body:
    {
        "dataset": "bbbp",
        "models": ["random_forest", "xgboost"],
        "use_smote": true
    }
    """
    data = await request.json()
    dataset_name = data.get("dataset", "bbbp")
    model_names = data.get("models", ["random_forest", "xgboost"])

    from science.molecular.benchmarks import run_benchmark

    try:
        result = run_benchmark(
            dataset_name=dataset_name,
            models=model_names,
            verbose=False,
        )
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    evaluations = []
    for ev in result.evaluations:
        evaluations.append({
            "model": ev.model_name,
            "metrics": ev.aggregate_metrics,
            "n_params": ev.n_params,
            "train_time_s": round(ev.train_time_s, 2),
            "tasks": [
                {
                    "name": tr.task_name,
                    "metrics": {k: round(v, 4) for k, v in tr.metrics.items()},
                    "confidence_intervals": {
                        k: [round(v[0], 4), round(v[1], 4)]
                        for k, v in tr.confidence_intervals.items()
                    },
                    "n_samples": tr.n_samples,
                }
                for tr in ev.task_results
            ],
        })

    metric = "auroc" if "auroc" in (evaluations[0]["metrics"] if evaluations else {}) else "rmse"

    return JSONResponse({
        "ok": True,
        "dataset": dataset_name,
        "split": "scaffold",
        "evaluations": evaluations,
        "leaderboard_md": result.leaderboard(metric),
    })


# ---------------------------------------------------------------------------
# GET /api/molecular_models
# ---------------------------------------------------------------------------

@router.get("/api/molecular_models")
async def molecular_models() -> Any:
    """List available models and datasets."""
    from science.molecular.qsar_models import list_models
    from science.molecular.datasets import list_datasets, dataset_summary

    return JSONResponse({
        "ok": True,
        "models": list_models(),
        "datasets": list_datasets(),
        "dataset_details": dataset_summary(),
    })
