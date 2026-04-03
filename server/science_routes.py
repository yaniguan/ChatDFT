"""
Science API Routes — expose science/ algorithms as REST endpoints.

This bridges the gap between the science/ module (offline algorithms)
and the server/ module (online API). Every science algorithm is now
callable from the Streamlit UI or any HTTP client.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import json
import numpy as np

router = APIRouter(prefix="/science", tags=["science"])


# ─── Request / Response models ───────────────────────────────────────────

class SurfaceGraphRequest(BaseModel):
    poscar: str
    min_voronoi_area: float = 0.5
    ads_height: float = 1.8

class SCFAnalysisRequest(BaseModel):
    dE: List[float]
    nelm: int = 60
    ediff: float = 1e-5
    is_metal: bool = True
    has_d_electrons: bool = False

class GrounderRequest(BaseModel):
    hypothesis: str
    network: Dict[str, Any]
    dG_profile: Optional[List[float]] = None

class BORequest(BaseModel):
    observations: List[Dict[str, float]]  # [{encut, kppra, energy}]
    n_atoms: int = 36
    target_error: float = 0.001

class RattleRequest(BaseModel):
    poscar: str
    T_K: float = 600
    n_structures: int = 10
    omega_THz: float = 5.0
    strategy: str = "einstein"  # einstein | strain | combined


# ─── Surface Graph ───────────────────────────────────────────────────────

@router.post("/surface_graph")
async def surface_graph(req: SurfaceGraphRequest):
    """Build Voronoi topology graph and classify adsorption sites."""
    try:
        from ase.io import read
        from io import StringIO
        atoms = read(StringIO(req.poscar), format="vasp")

        from science.representations.surface_graph import SurfaceTopologyGraph
        stg = SurfaceTopologyGraph(
            positions=atoms.get_positions(),
            elements=atoms.get_chemical_symbols(),
            cell=atoms.get_cell()[:],
        )
        stg.build(min_voronoi_area=req.min_voronoi_area)
        sites = stg.classify_adsorption_sites(ads_height=req.ads_height)
        X = stg.node_feature_matrix()

        return {
            "ok": True,
            "n_atoms": len(stg.nodes),
            "n_edges": len(stg.edges),
            "n_sites": len(sites),
            "sites": [
                {"type": s.site_type, "position": s.position.tolist(),
                 "symmetry": s.symmetry_rank, "atoms": s.coordinating_atoms}
                for s in sites
            ],
            "node_features_shape": list(X.shape),
            "summary": stg.summary(),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─── SCF Analysis ────────────────────────────────────────────────────────

@router.post("/scf_analysis")
async def scf_analysis(req: SCFAnalysisRequest):
    """Analyse SCF convergence trajectory: sloshing detection + prediction."""
    try:
        from science.time_series.scf_convergence import SCFTrajectory, analyse_scf

        traj = SCFTrajectory(dE=req.dE, nelm=req.nelm, ediff=req.ediff)
        report = analyse_scf(traj, is_metal=req.is_metal,
                             has_d_electrons=req.has_d_electrons)

        return {
            "ok": True,
            "converged": traj.is_converged(),
            "n_steps": len(req.dE),
            "sloshing": {
                "detected": report.sloshing.is_sloshing,
                "frequency": report.sloshing.dominant_frequency,
                "amplitude": report.sloshing.amplitude,
                "decay_rate": report.sloshing.decay_rate,
                "confidence": report.sloshing.confidence,
                "remedy": report.sloshing.remedy,
            },
            "prediction": {
                "predicted_step": report.prediction.predicted_step,
                "convergence_rate": report.prediction.convergence_rate,
                "r_squared": report.prediction.r_squared,
                "will_converge": report.prediction.will_converge,
                "confidence": report.prediction.confidence,
            },
            "recommendation": {
                "algo": report.algo.algo,
                "settings": report.algo.settings,
                "rationale": report.algo.rationale,
            },
            "summary": report.summary,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─── Hypothesis Grounder ─────────────────────────────────────────────────

@router.post("/grounder")
async def grounder_score(req: GrounderRequest):
    """Score cross-modal alignment between hypothesis and reaction network."""
    try:
        from science.alignment.hypothesis_grounder import HypothesisGrounder, ReactionNetwork

        g = HypothesisGrounder()
        net = ReactionNetwork.from_dict(req.network)
        score = g.score(req.hypothesis, net, req.dG_profile)
        breakdown = g.score_breakdown(req.hypothesis, net, req.dG_profile)

        return {
            "ok": True,
            "score": score,
            "breakdown": breakdown,
            "network_fingerprint": net.fingerprint(),
        }
    except (ValueError, KeyError, TypeError) as e:
        return {"ok": False, "error": str(e)}


# ─── Bayesian Optimisation ───────────────────────────────────────────────

@router.post("/bo_suggest")
async def bo_suggest(req: BORequest):
    """Run Bayesian Optimisation for DFT parameter search."""
    try:
        from science.optimization.bayesian_params import BayesianParameterOptimizer

        opt = BayesianParameterOptimizer(
            n_atoms=req.n_atoms, target_error=req.target_error
        )
        for obs in req.observations:
            opt.observe(obs["encut"], int(obs["kppra"]), obs["energy"])

        next_encut, next_kppra = opt.suggest_next()
        result = opt.result()

        return {
            "ok": True,
            "next_suggestion": {"encut": next_encut, "kppra": next_kppra},
            "optimal": {
                "encut": result.optimal_encut,
                "kppra": result.optimal_kppra,
                "error": result.predicted_error,
                "cost": result.predicted_cost,
            },
            "n_evaluations": result.n_evaluations,
            "pareto_front": [
                {"encut": p.encut, "kppra": p.kppra,
                 "error": p.energy_error, "cost": p.cost}
                for p in result.pareto_front
            ],
            "summary": opt.summary(),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─── Structure Generation ────────────────────────────────────────────────

@router.post("/generate_structures")
async def generate_structures(req: RattleRequest):
    """Generate diverse structures for NNP training."""
    try:
        from ase.io import read, write
        from io import StringIO

        atoms = read(StringIO(req.poscar), format="vasp")

        if req.strategy == "einstein":
            from science.generation.informed_sampler import EinsteinRattler
            sampler = EinsteinRattler(omega_THz=req.omega_THz, quantum=True, rng_seed=42)
            configs = sampler.generate_batch(atoms, T_K=req.T_K, n=req.n_structures)
        elif req.strategy == "strain":
            from science.generation.informed_sampler import strain_sample
            configs = strain_sample(atoms, strain_max=0.06, n=req.n_structures, rng_seed=42)
        else:
            from science.generation.informed_sampler import EinsteinRattler, strain_sample
            rattler = EinsteinRattler(omega_THz=req.omega_THz, quantum=True, rng_seed=42)
            n_rattle = req.n_structures // 2
            n_strain = req.n_structures - n_rattle
            configs = (rattler.generate_batch(atoms, T_K=req.T_K, n=n_rattle) +
                       strain_sample(atoms, strain_max=0.06, n=n_strain, rng_seed=42))

        # Convert to POSCARs
        poscars = []
        for i, c in enumerate(configs):
            buf = StringIO()
            write(buf, c, format="vasp")
            poscars.append(buf.getvalue())

        return {
            "ok": True,
            "n_generated": len(poscars),
            "strategy": req.strategy,
            "T_K": req.T_K,
            "structures": poscars[:20],  # limit response size
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─── MLOps Status ────────────────────────────────────────────────────────

@router.get("/mlops/status")
async def mlops_status():
    """Return full MLOps system status."""
    try:
        from server.mlops.model_registry import model_registry
        from server.mlops.experiment_tracker import experiment_tracker
        from server.mlops.monitoring import production_monitor
        from server.feature_store.store import feature_store

        return {
            "ok": True,
            "model_registry": model_registry.summary(),
            "experiment_tracker": experiment_tracker.summary(),
            "feature_store": feature_store.summary(),
            "monitoring": production_monitor.health_status(),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
