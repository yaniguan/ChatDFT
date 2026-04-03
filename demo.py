#!/usr/bin/env python3
"""
ChatDFT Interactive Demo
=========================
Run all 6 scientific algorithms on built-in data with zero configuration.
No database, no API key, no VASP license needed.

Usage:
    python demo.py              # Run everything
    python demo.py --quick      # Skip figures (text output only)
    python demo.py --module 3   # Run only module 3 (hypothesis grounding)

Modules:
    1. Surface Topology Graph     — Voronoi tessellation for catalyst surfaces
    2. Physics-Informed Rattle    — Quantum harmonic oscillator structure gen
    3. Hypothesis Grounding       — Cross-modal alignment (text ↔ graph ↔ energy)
    4. SCF Convergence Analysis   — FFT sloshing detection + rate prediction
    5. Bayesian Parameter Search  — GP + EI for DFT convergence testing
    6. GNN Energy Prediction      — 5 architectures for E_ads (MPNN/GAT/SchNet/DimeNet/SE3)
"""

import argparse
import sys
import time

import numpy as np


def header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ─── Module 1: Surface Topology Graph ───────────────────────────────

def demo_surface_graph():
    header("Module 1: Surface Topology Graph")
    from ase.build import fcc111
    from science.representations.surface_graph import SurfaceTopologyGraph

    # Build a Cu(111) slab
    slab = fcc111("Cu", size=(3, 3, 4), vacuum=10.0, a=3.615)
    pos = slab.get_positions()
    elems = slab.get_chemical_symbols()
    cell = np.array(slab.get_cell())

    print("Input: Cu(111) 3x3 slab, 4 layers, 36 atoms")
    print(f"  Positions shape: {pos.shape}")

    # Build graph
    t0 = time.perf_counter()
    stg = SurfaceTopologyGraph(pos, elems, cell)
    stg.build()
    sites = stg.classify_adsorption_sites()
    t_ms = (time.perf_counter() - t0) * 1000

    print(f"\nGraph built in {t_ms:.1f} ms:")
    print(f"  Nodes: {len(stg.nodes)} atoms")
    print(f"  Edges: {len(stg.edges)} bonds (Voronoi-determined)")

    X = stg.node_feature_matrix()
    print(f"  Node features: {X.shape} (Z/100, layer, CN/12, V_vor/20, d_surf/5, angle_var)")

    from collections import Counter
    counts = Counter(s.site_type for s in sites)
    print(f"\nAdsorption sites ({len(sites)} total):")
    for stype, count in sorted(counts.items()):
        print(f"  {stype:12s}: {count}")

    # Show symmetry scores
    sym_scores = [s.symmetry_rank for s in sites]
    print(f"\nSymmetry score range: {min(sym_scores):.3f} – {max(sym_scores):.3f}")
    print("  (1.0 = perfectly symmetric, 0.0 = highly asymmetric)")

    ei, ea = stg.edge_index_and_attr()
    print(f"\nPyTorch Geometric compatible:")
    print(f"  edge_index: {ei.shape} (2 x 2E, bidirectional)")
    print(f"  edge_attr:  {ea.shape} (length, angle, voronoi_area)")


# ─── Module 2: Physics-Informed Structure Generation ────────────────

def demo_structure_generation():
    header("Module 2: Physics-Informed Structure Generation")
    from science.generation.informed_sampler import EinsteinRattler, AtomsLike

    # Create a Cu slab
    N = 12
    positions = np.zeros((N, 3))
    for i in range(N):
        positions[i] = [i % 3 * 2.55, (i // 3) % 4 * 2.55, (i // 9) * 2.0]
    masses = np.full(N, 63.546)  # Cu

    atoms = AtomsLike(
        positions=positions,
        numbers=np.full(N, 29),
        cell=np.eye(3) * 10.0,
        masses=masses,
    )

    # Einstein rattle at different temperatures
    rattler = EinsteinRattler(omega_THz=5.0, quantum=True, rng_seed=42)

    print("Einstein Quantum Harmonic Oscillator Rattle")
    print("  σᵢ = sqrt( ħ/(2mᵢω) · coth(ħω/2kBT) )")
    print()

    for T in [0.001, 100, 300, 600, 1000]:
        sigma = rattler._sigma(63.546, T)
        rattled = rattler.rattle(atoms, T)
        disp = np.linalg.norm(rattled.get_positions() - positions, axis=1)
        print(f"  T = {T:7.1f} K:  σ = {sigma:.4f} A,  "
              f"mean |Δr| = {np.mean(disp):.4f} A,  max = {np.max(disp):.4f} A")

    # Mass dependence
    print(f"\nMass dependence at T = 600 K:")
    for name, mass in [("H", 1.008), ("C", 12.011), ("Cu", 63.546), ("Pt", 195.084)]:
        sigma = rattler._sigma(mass, 600)
        print(f"  {name:3s} (m={mass:7.3f} amu): σ = {sigma:.4f} A")

    # Generate batch
    batch = rattler.generate_batch(atoms, T_K=600, n=50)
    print(f"\nGenerated {len(batch)} configurations at 600 K")
    all_disp = [np.mean(np.linalg.norm(a.get_positions() - positions, axis=1)) for a in batch]
    print(f"  Mean displacement: {np.mean(all_disp):.4f} ± {np.std(all_disp):.4f} A")


# ─── Module 3: Cross-Modal Hypothesis Grounding ────────────────────

def demo_hypothesis_grounding():
    header("Module 3: Cross-Modal Hypothesis Grounding")
    from science.alignment.hypothesis_grounder import (
        HypothesisGrounder, ReactionNetwork,
    )

    grounder = HypothesisGrounder()

    # Test with CO2RR on Cu(111)
    hypothesis = "CO2 reduction proceeds via COOH* intermediate on Cu(111), with CO* as the key product-forming species"
    network_dict = {
        "reaction_network": [
            {"lhs": ["CO2(g)", "*", "H+", "e-"], "rhs": ["COOH*"]},
            {"lhs": ["COOH*", "H+", "e-"], "rhs": ["CO*", "H2O(g)"]},
        ],
        "intermediates": ["*", "CO2(g)", "COOH*", "CO*", "CO(g)", "H2O(g)"],
        "surface": "Cu(111)",
    }
    dG = [0.0, 0.22, -0.15, -0.45, -1.10]

    network = ReactionNetwork.from_dict(network_dict)
    score = grounder.score(hypothesis, network, dG)
    breakdown = grounder.score_breakdown(hypothesis, network, dG)

    print("Hypothesis:")
    print(f'  "{hypothesis}"')
    print(f"\nReaction network: {len(network.steps)} steps, "
          f"{len(network.intermediates)} intermediates")
    print(f"Free energy profile: {dG}")
    print(f"\nAlignment score: {score:.4f}")
    print("Score breakdown:")
    for k, v in breakdown.items():
        print(f"  {k:30s}: {v:.4f}")

    # InfoNCE loss on a batch
    hypotheses = [
        "CO2 reduction to CO on Cu(111)",
        "Hydrogen evolution on Pt(111)",
        "OER on IrO2 via OH* intermediate",
    ]
    networks = [
        ReactionNetwork.from_dict({"reaction_network": [{"lhs": ["CO2(g)", "*"], "rhs": ["COOH*"]}],
                                    "intermediates": ["*", "CO2(g)", "COOH*", "CO*"]}),
        ReactionNetwork.from_dict({"reaction_network": [{"lhs": ["H+", "e-", "*"], "rhs": ["H*"]}],
                                    "intermediates": ["*", "H+", "e-", "H*", "H2(g)"]}),
        ReactionNetwork.from_dict({"reaction_network": [{"lhs": ["*", "H2O(g)"], "rhs": ["OH*"]}],
                                    "intermediates": ["*", "OH*", "O*", "OOH*", "O2(g)"]}),
    ]
    loss = grounder.infonce_loss(hypotheses, networks)
    print(f"\nInfoNCE contrastive loss (batch=3): {loss:.4f}")
    print("  (Lower = better text-graph alignment)")


# ─── Module 4: SCF Convergence Analysis ────────────────────────────

def demo_scf_analysis():
    header("Module 4: SCF Convergence Analysis")
    from science.time_series.scf_convergence import (
        SCFTrajectory, analyse_scf,
    )

    # Healthy convergence trajectory
    print("Example 1: Healthy exponential convergence")
    t = np.arange(30)
    dE_healthy = list(0.5 * np.exp(-0.35 * t) + 1e-8)
    traj = SCFTrajectory(dE=dE_healthy, ediff=1e-5, nelm=60)
    report = analyse_scf(traj, is_metal=True)
    print(report.summary)

    # Sloshing trajectory
    print("\n\nExample 2: Charge sloshing (metallic surface)")
    dE_slosh = list(0.01 * np.exp(-0.02 * t) * (0.5 + np.abs(np.sin(2 * np.pi * t / 5))) + 1e-7)
    traj2 = SCFTrajectory(dE=dE_slosh, ediff=1e-5, nelm=60)
    report2 = analyse_scf(traj2, is_metal=True)
    print(report2.summary)

    # Slow convergence
    print("\n\nExample 3: Slow convergence (insulator)")
    dE_slow = list(0.1 * np.exp(-0.05 * t) + 1e-6)
    traj3 = SCFTrajectory(dE=dE_slow, ediff=1e-5, nelm=60)
    report3 = analyse_scf(traj3, is_metal=False, has_d_electrons=True)
    print(report3.summary)


# ─── Module 5: Bayesian Parameter Search ───────────────────────────

def demo_bayesian_search():
    header("Module 5: Bayesian Parameter Optimisation")
    from science.optimization.bayesian_params import BayesianParameterOptimizer
    from science.benchmarks.baselines import synthetic_energy_landscape

    opt = BayesianParameterOptimizer(n_atoms=36, target_error=0.001)

    print("Searching (ENCUT, KPPRA) space for Cu(111) slab (36 atoms)...")
    print("Target: energy error < 1 meV/atom\n")

    # Initial exploration
    print("Phase 1: Latin Hypercube Exploration (5 points)")
    for encut, kppra in opt.suggest_initial(5):
        energy = synthetic_energy_landscape(encut, kppra)
        opt.observe(encut, kppra, energy)
        r = opt.result()
        print(f"  ENCUT={encut:5.0f}, KPPRA={kppra:4d} → E={energy:.6f} eV "
              f"(error={r.predicted_error:.6f})")

    # BO loop
    print("\nPhase 2: Bayesian Optimisation (10 iterations)")
    for i in range(10):
        encut, kppra = opt.suggest_next()
        energy = synthetic_energy_landscape(encut, kppra)
        opt.observe(encut, kppra, energy)
        r = opt.result()
        converged = "CONVERGED" if r.predicted_error <= 0.001 else ""
        print(f"  [{i+1:2d}] ENCUT={encut:5.0f}, KPPRA={kppra:4d} → "
              f"error={r.predicted_error:.6f} eV/atom {converged}")

    result = opt.result()
    print(f"\nResult:")
    print(opt.summary())
    print(f"\n  Grid search would require: ~56-80 evaluations")
    print(f"  We used: {result.n_evaluations} evaluations ({(1 - result.n_evaluations/56)*100:.0f}% savings)")


# ─── Module 6: GNN Energy Prediction ──────────────────────────────

def demo_gnn_prediction():
    header("Module 6: GNN Adsorption Energy Prediction")
    try:
        import torch
    except ImportError:
        print("  PyTorch not installed — skipping GNN demo")
        print("  Install with: pip install torch")
        return

    from science.predictions.gnn_models import build_model, list_models
    from science.predictions.energy_predictor import (
        generate_dataset, samples_to_graphs, train_and_evaluate,
        format_results_table,
    )

    print("5 GNN architectures + MLP baseline for E_ads prediction")
    print("Synthetic dataset: d-band model + coordination + adsorbate scaling\n")

    # Generate data
    samples = generate_dataset(n_samples=60, seed=42, n_atoms=8)
    graphs = samples_to_graphs(samples)
    train_g, val_g, test_g = graphs[:40], graphs[40:50], graphs[50:]

    print(f"Dataset: {len(samples)} samples ({len(train_g)}/{len(val_g)}/{len(test_g)} split)")
    energies = [s.energy for s in samples]
    print(f"Energy range: [{min(energies):.2f}, {max(energies):.2f}] eV\n")

    # Train all models (quick — 40 epochs)
    results = []
    for name in list_models():
        try:
            r = train_and_evaluate(name, train_g, val_g, test_g,
                                   n_epochs=40, batch_size=8)
            results.append(r)
            print(f"  {name:16s}: test_mae={r.test_mae:.4f} eV, "
                  f"params={r.n_params:>7,}, time={r.train_time_s:.1f}s")
        except Exception as e:
            print(f"  {name:16s}: FAILED — {e}")

    if results:
        results.sort(key=lambda r: r.test_mae)
        print(f"\nBest model: {results[0].model_name} "
              f"(MAE = {results[0].test_mae:.4f} eV)")
        print(f"\n{format_results_table(results)}")


# ─── Bonus: Golden Dataset Overview ────────────────────────────────

def demo_golden_dataset():
    header("Bonus: Golden Benchmark Dataset")
    from science.evaluation.golden_dataset import summary, GOLDEN_SET, GOLDEN_BY_DOMAIN

    print(summary())
    print()

    # Show one example per domain
    for domain, examples in GOLDEN_BY_DOMAIN.items():
        ex = examples[0]
        print(f"  {domain.upper()}: {ex.query}")
        print(f"    η = {ex.expected_overpotential:.2f} V, "
              f"intermediates: {', '.join(ex.expected_intermediates[:4])}...")
        print(f"    DOI: {ex.doi}")
        print()


# ─── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ChatDFT Interactive Demo")
    parser.add_argument("--module", type=int, default=0,
                        help="Run specific module (1-5, 0=all)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark suite with figures")
    args = parser.parse_args()

    print("=" * 60)
    print("  ChatDFT — Autonomous Reaction Pathway Discovery")
    print("  Interactive Demo (no database or API key needed)")
    print("=" * 60)

    modules = {
        1: demo_surface_graph,
        2: demo_structure_generation,
        3: demo_hypothesis_grounding,
        4: demo_scf_analysis,
        5: demo_bayesian_search,
        6: demo_gnn_prediction,
        7: demo_golden_dataset,
    }

    if args.benchmark:
        from science.benchmarks.run_benchmarks import main as run_benchmarks
        run_benchmarks()
        return

    if args.module > 0:
        if args.module in modules:
            modules[args.module]()
        else:
            print(f"Unknown module {args.module}. Available: 1-6")
            sys.exit(1)
    else:
        for mod_fn in modules.values():
            mod_fn()

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print()
    print("  Next steps:")
    print("    python demo.py --benchmark     # Full benchmark with figures")
    print("    streamlit run client/app.py     # Launch the web UI")
    print("    pytest tests/ -v                # Run all tests")
    print("=" * 60)


if __name__ == "__main__":
    main()
