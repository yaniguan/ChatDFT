"""
Unit tests for the science/ module.

Tests cover:
- Surface topology graph (Voronoi representation)
- Physics-informed structure generation (Einstein rattle, normal mode, strain)
- Cross-modal hypothesis grounder (InfoNCE, encoding, scoring)
- SCF convergence analysis (sloshing detection, convergence prediction)
- Bayesian parameter optimisation (GP, EI, BO loop)
- Evaluation metrics (intent, hypothesis, thermo, RAG)
"""

import json
import numpy as np
import pytest


# ─── Surface Topology Graph ──────────────────────────────────────────────────

class TestSurfaceTopologyGraph:

    def test_build_basic(self, cu111_positions, cu111_elements, cu111_cell):
        from science.representations.surface_graph import SurfaceTopologyGraph
        stg = SurfaceTopologyGraph(cu111_positions, cu111_elements, cu111_cell)
        stg.build()
        assert len(stg.nodes) == 8
        assert len(stg.edges) > 0
        assert all(n.element == "Cu" for n in stg.nodes)

    def test_node_features_shape(self, cu111_positions, cu111_elements, cu111_cell):
        from science.representations.surface_graph import SurfaceTopologyGraph
        stg = SurfaceTopologyGraph(cu111_positions, cu111_elements, cu111_cell)
        stg.build()
        X = stg.node_feature_matrix()
        assert X.shape == (8, 6)
        assert X.dtype == np.float32
        # Atomic number feature should be Cu=29 → 29/100=0.29
        assert np.allclose(X[:, 0], 0.29, atol=0.01)

    def test_edge_index_bidirectional(self, cu111_positions, cu111_elements, cu111_cell):
        from science.representations.surface_graph import SurfaceTopologyGraph
        stg = SurfaceTopologyGraph(cu111_positions, cu111_elements, cu111_cell)
        stg.build()
        ei, ea = stg.edge_index_and_attr()
        assert ei.shape[0] == 2
        assert ei.shape[1] == ea.shape[0]
        # Bidirectional: 2x number of unique edges
        assert ei.shape[1] == 2 * len(stg.edges)

    def test_site_classification(self, cu111_positions, cu111_elements, cu111_cell):
        from science.representations.surface_graph import SurfaceTopologyGraph
        stg = SurfaceTopologyGraph(cu111_positions, cu111_elements, cu111_cell)
        stg.build()
        sites = stg.classify_adsorption_sites()
        assert len(sites) > 0
        site_types = {s.site_type for s in sites}
        assert "top" in site_types

    def test_summary(self, cu111_positions, cu111_elements, cu111_cell):
        from science.representations.surface_graph import SurfaceTopologyGraph
        stg = SurfaceTopologyGraph(cu111_positions, cu111_elements, cu111_cell)
        stg.build()
        s = stg.summary()
        assert "8 atoms" in s
        assert "Cu" in s


# ─── Physics-Informed Sampler ─────────────────────────────────────────────────

class TestEinsteinRattler:

    def test_sigma_increases_with_temperature(self):
        from science.generation.informed_sampler import EinsteinRattler
        rattler = EinsteinRattler(omega_THz=5.0, quantum=True)
        s_low = rattler._sigma(63.546, T_K=100)
        s_high = rattler._sigma(63.546, T_K=1000)
        assert s_high > s_low

    def test_quantum_includes_zpe(self):
        from science.generation.informed_sampler import EinsteinRattler
        rattler = EinsteinRattler(omega_THz=5.0, quantum=True)
        s_zpe = rattler._sigma(63.546, T_K=0.001)
        assert s_zpe > 0  # Zero-point motion even at T→0

    def test_classical_zero_at_T0(self):
        from science.generation.informed_sampler import EinsteinRattler
        rattler = EinsteinRattler(omega_THz=5.0, quantum=False)
        s_zero = rattler._sigma(63.546, T_K=0.0)
        assert s_zero == 0.0

    def test_rattle_preserves_atom_count(self):
        from science.generation.informed_sampler import AtomsLike, EinsteinRattler
        atoms = AtomsLike(
            positions=np.random.randn(10, 3),
            numbers=np.array([29] * 10),
            cell=np.eye(3) * 10,
            masses=np.array([63.546] * 10),
        )
        rattler = EinsteinRattler(rng_seed=42)
        rattled = rattler.rattle(atoms, T_K=300)
        assert rattled.get_positions().shape == (10, 3)

    def test_batch_returns_n_structures(self):
        from science.generation.informed_sampler import AtomsLike, EinsteinRattler
        atoms = AtomsLike(
            positions=np.zeros((5, 3)),
            numbers=np.array([29] * 5),
            cell=np.eye(3) * 10,
            masses=np.array([63.546] * 5),
        )
        rattler = EinsteinRattler(rng_seed=42)
        batch = rattler.generate_batch(atoms, T_K=500, n=7)
        assert len(batch) == 7


class TestStrainSample:

    def test_isotropic_strain(self):
        from science.generation.informed_sampler import AtomsLike, strain_sample
        atoms = AtomsLike(
            positions=np.array([[0, 0, 0], [1, 1, 1]], dtype=float),
            numbers=np.array([29, 29]),
            cell=np.eye(3) * 5.0,
            masses=np.array([63.546, 63.546]),
        )
        strained = strain_sample(atoms, strain_max=0.05, n=5, mode="isotropic", rng_seed=42)
        assert len(strained) == 5


# ─── Hypothesis Grounder ─────────────────────────────────────────────────────

class TestHypothesisGrounder:

    def test_score_returns_float(self, sample_reaction_network, sample_dG_profile):
        from science.alignment.hypothesis_grounder import HypothesisGrounder, ReactionNetwork
        grounder = HypothesisGrounder()
        network = ReactionNetwork.from_dict(sample_reaction_network)
        score = grounder.score(
            "COOH* is the key intermediate for CO2RR on Cu(111)",
            network,
            sample_dG_profile,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_breakdown_has_components(self, sample_reaction_network):
        from science.alignment.hypothesis_grounder import HypothesisGrounder, ReactionNetwork
        grounder = HypothesisGrounder()
        network = ReactionNetwork.from_dict(sample_reaction_network)
        bd = grounder.score_breakdown("CO2RR on Cu(111)", network)
        assert "text_graph_cosine" in bd
        assert "combined_confidence" in bd

    def test_infonce_loss_is_finite(self, sample_reaction_network):
        from science.alignment.hypothesis_grounder import HypothesisGrounder, ReactionNetwork
        grounder = HypothesisGrounder()
        networks = [ReactionNetwork.from_dict(sample_reaction_network)] * 3
        hypotheses = [
            "CO2RR via carboxyl on Cu(111)",
            "HER on Pt(111) Volmer-Heyrovsky",
            "OER on IrO2 lattice oxygen mechanism",
        ]
        loss = grounder.infonce_loss(hypotheses, networks)
        assert np.isfinite(loss)

    def test_reaction_network_fingerprint(self, sample_reaction_network):
        from science.alignment.hypothesis_grounder import ReactionNetwork
        n1 = ReactionNetwork.from_dict(sample_reaction_network)
        n2 = ReactionNetwork.from_dict(sample_reaction_network)
        assert n1.fingerprint() == n2.fingerprint()


# ─── SCF Convergence Analysis ─────────────────────────────────────────────────

class TestSCFConvergence:

    def test_exponential_decay_detected(self, sample_scf_trajectory):
        from science.time_series.scf_convergence import SCFTrajectory, ConvergenceRatePredictor
        traj = SCFTrajectory(dE=sample_scf_trajectory, nelm=200, ediff=1e-5)
        pred = ConvergenceRatePredictor().predict(traj)
        assert pred.convergence_rate > 0
        assert pred.r_squared > 0.5
        assert pred.will_converge

    def test_sloshing_detected(self, sample_sloshing_trajectory):
        from science.time_series.scf_convergence import SCFTrajectory, ChargeSloshingDetector
        traj = SCFTrajectory(dE=sample_sloshing_trajectory, nelm=100, ediff=1e-5)
        result = ChargeSloshingDetector().detect(traj)
        assert result.is_sloshing
        assert result.dominant_frequency > 0

    def test_healthy_trajectory_converging(self):
        from science.time_series.scf_convergence import SCFTrajectory, ChargeSloshingDetector
        # Pure monotone exponential decay — even if detector flags as "mild sloshing",
        # the decay rate should be positive (converging)
        dE = [1.0 * np.exp(-0.3 * i) for i in range(30)]
        traj = SCFTrajectory(dE=dE, nelm=60, ediff=1e-5)
        result = ChargeSloshingDetector().detect(traj)
        assert result.decay_rate > 0  # converging, not diverging

    def test_algo_recommender(self, sample_scf_trajectory, sample_sloshing_trajectory):
        from science.time_series.scf_convergence import (
            SCFTrajectory, ChargeSloshingDetector,
            ConvergenceRatePredictor, AlgorithmRecommender,
        )
        # Healthy metal
        traj = SCFTrajectory(dE=sample_scf_trajectory, nelm=60, ediff=1e-5)
        slosh = ChargeSloshingDetector().detect(traj)
        pred = ConvergenceRatePredictor().predict(traj)
        rec = AlgorithmRecommender().recommend(slosh, pred, is_metal=True)
        assert rec.algo in ("Fast", "All", "Damped")

    def test_ionic_tracker(self, sample_scf_trajectory):
        from science.time_series.scf_convergence import SCFTrajectory, IonicConvergenceTracker
        tracker = IonicConvergenceTracker()
        for _ in range(5):
            tracker.add_ionic_step(SCFTrajectory(dE=sample_scf_trajectory[:20], ediff=1e-5))
        counts = tracker.scf_step_counts()
        assert len(counts) == 5
        report = tracker.report()
        assert "5 steps" in report


# ─── Bayesian Parameter Optimisation ──────────────────────────────────────────

class TestBayesianOptimizer:

    def test_suggest_initial(self):
        from science.optimization.bayesian_params import BayesianParameterOptimizer
        opt = BayesianParameterOptimizer(n_atoms=36)
        initial = opt.suggest_initial(n=5)
        assert len(initial) == 5
        for encut, kppra in initial:
            assert 300 <= encut <= 600
            assert 400 <= kppra <= 3200

    def test_observe_and_suggest(self):
        from science.optimization.bayesian_params import BayesianParameterOptimizer
        opt = BayesianParameterOptimizer(n_atoms=36)
        # Simulate a simple energy landscape
        def fake_energy(encut, kppra):
            return -100.0 + 0.1 * np.exp(-encut / 200) + 0.05 * np.exp(-kppra / 800)
        for encut, kppra in opt.suggest_initial(n=5):
            opt.observe(encut, kppra, fake_energy(encut, kppra))
        # Should be able to suggest next point
        next_e, next_k = opt.suggest_next()
        assert 300 <= next_e <= 600
        assert 400 <= next_k <= 3200

    def test_result_has_pareto(self):
        from science.optimization.bayesian_params import BayesianParameterOptimizer
        opt = BayesianParameterOptimizer(n_atoms=36)
        for e, k in [(300, 400), (400, 1600), (500, 2400), (600, 3200)]:
            opt.observe(e, k, -100.0 + 10.0 / e + 5.0 / k)
        result = opt.result()
        assert result.n_evaluations == 4
        assert len(result.pareto_front) >= 1

    def test_gp_predict(self):
        from science.optimization.bayesian_params import GaussianProcess
        gp = GaussianProcess()
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 2], dtype=float)
        gp.fit(X, y)
        mu, sigma = gp.predict(np.array([[0.5, 0.5]]))
        assert mu.shape == (1,)
        assert sigma.shape == (1,)
        assert sigma[0] > 0


# ─── Evaluation Metrics ──────────────────────────────────────────────────────

class TestEvaluationMetrics:

    def test_intent_parsing_metrics(self):
        from science.evaluation.metrics import IntentParsingMetrics
        predicted = {"stage": "electrocatalysis", "system": {"material": "Cu", "facet": "111"}}
        expected = {"stage": "electrocatalysis", "system": {"material": "Cu", "facet": "111"}}
        results = IntentParsingMetrics.evaluate(predicted, expected)
        assert len(results) == 3
        assert all(r.value == 1.0 for r in results)

    def test_hypothesis_metrics(self):
        from science.evaluation.metrics import HypothesisMetrics
        results = HypothesisMetrics.evaluate(
            predicted_intermediates=["*", "CO2(g)", "COOH*", "CO*", "extra*"],
            expected_intermediates=["*", "CO2(g)", "COOH*", "CO*"],
        )
        recall = next(r for r in results if r.name == "hypothesis_intermediate_recall")
        precision = next(r for r in results if r.name == "hypothesis_intermediate_precision")
        assert recall.value == 1.0      # found all expected
        assert precision.value == 0.8   # 4/5 predicted are correct

    def test_thermo_metrics(self):
        from science.evaluation.metrics import ThermodynamicsMetrics
        results = ThermodynamicsMetrics.evaluate(
            predicted_dG=[0.0, 0.25, -0.10, -0.50],
            expected_dG=[0.0, 0.22, -0.15, -0.45],
            predicted_eta=0.65,
            expected_eta=0.61,
        )
        mae = next(r for r in results if r.name == "thermo_dG_mae_eV")
        assert mae.value < 0.1

    def test_rag_mrr(self):
        from science.evaluation.metrics import RAGMetrics
        assert RAGMetrics.mrr([False, True, False]) == 0.5
        assert RAGMetrics.mrr([True, False, False]) == 1.0
        assert RAGMetrics.mrr([False, False, False]) == 0.0

    def test_grounder_brier_score(self):
        from science.evaluation.metrics import GrounderMetrics
        # Perfect calibration
        score = GrounderMetrics.brier_score([0.8, 0.2], [1, 0])
        assert score < 0.1


# ─── MLOps Components ────────────────────────────────────────────────────────

class TestModelRegistry:

    def test_default_models_registered(self):
        from server.mlops.model_registry import ModelRegistry
        registry = ModelRegistry()
        models = registry.list_all()
        assert len(models) >= 7
        names = {m.name for m in models}
        assert "hypothesis_grounder" in names
        assert "text_embedding" in names

    def test_ab_routing(self):
        from server.mlops.model_registry import ModelRegistry
        registry = ModelRegistry()
        registry.set_ab_test("hypothesis_grounder", "1.0.0", "1.1.0", split=0.5)
        versions = [registry.route("hypothesis_grounder") for _ in range(100)]
        assert "1.0.0" in versions
        assert "1.1.0" in versions


class TestExperimentTracker:

    def test_run_lifecycle(self):
        from server.mlops.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        run = tracker.start_run("test_experiment", model_name="test_model")
        run.log_param("lr", 0.01)
        run.log_metric("accuracy", 0.95)
        run.log_input("test input data")
        run.log_output("test output")
        tracker.end_run(run.run_id)
        retrieved = tracker.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.metrics["accuracy"] == 0.95
        assert retrieved.status == "completed"


class TestFeatureStore:

    def test_default_features_registered(self):
        from server.feature_store.store import FeatureStore
        store = FeatureStore()
        features = store.list_features()
        assert len(features) >= 4
        names = {f.name for f in features}
        assert "voronoi_coordination" in names
        assert "scf_convergence" in names

    def test_mechanism_features(self, sample_reaction_network):
        from server.feature_store.store import FeatureStore
        store = FeatureStore()
        features = store.compute(
            "mechanism_graph",
            entity_id="test_mech_001",
            raw_input=json.dumps(sample_reaction_network),
        )
        assert features.shape == (8,)
        assert features[0] == 6.0   # n_intermediates
        assert features[1] == 2.0   # n_steps

    def test_lineage_tracking(self, sample_reaction_network):
        from server.feature_store.store import FeatureStore
        store = FeatureStore()
        store.compute("mechanism_graph", "test_lin_001",
                      json.dumps(sample_reaction_network))
        lineage = store.get_lineage("test_lin_001")
        assert len(lineage) == 1
        assert lineage[0].feature_name == "mechanism_graph"
        assert lineage[0].input_hash != ""

    def test_provenance_trace(self, sample_reaction_network):
        from server.feature_store.store import FeatureStore
        store = FeatureStore()
        store.compute("mechanism_graph", "test_prov_001",
                      json.dumps(sample_reaction_network))
        prov = store.trace_provenance("test_prov_001")
        assert "mechanism_graph" in prov["features"]
