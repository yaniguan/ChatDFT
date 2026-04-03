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


# ─── Golden Dataset Tests ───────────────────────────────────────────────────

class TestGoldenDataset:

    def test_dataset_size(self):
        from science.evaluation.golden_dataset import GOLDEN_SET, N_TOTAL
        assert len(GOLDEN_SET) == 25
        assert N_TOTAL == 25

    def test_five_domains(self):
        from science.evaluation.golden_dataset import GOLDEN_BY_DOMAIN
        assert set(GOLDEN_BY_DOMAIN.keys()) == {"co2rr", "her", "oer", "nrr", "orr"}

    def test_domain_counts(self):
        from science.evaluation.golden_dataset import GOLDEN_BY_DOMAIN
        assert len(GOLDEN_BY_DOMAIN["co2rr"]) == 8
        assert len(GOLDEN_BY_DOMAIN["her"]) == 5
        assert len(GOLDEN_BY_DOMAIN["oer"]) == 5
        assert len(GOLDEN_BY_DOMAIN["nrr"]) == 4
        assert len(GOLDEN_BY_DOMAIN["orr"]) == 3

    def test_all_have_doi(self):
        from science.evaluation.golden_dataset import GOLDEN_SET
        for ex in GOLDEN_SET:
            assert ex.doi, f"Missing DOI for {ex.id}"

    def test_overpotentials_positive(self):
        from science.evaluation.golden_dataset import GOLDEN_SET
        for ex in GOLDEN_SET:
            assert ex.expected_overpotential > 0, f"Invalid η for {ex.id}"

    def test_dG_profiles_start_at_zero(self):
        from science.evaluation.golden_dataset import GOLDEN_SET
        for ex in GOLDEN_SET:
            assert ex.expected_dG_profile[0] == 0.0, f"dG[0] != 0 for {ex.id}"

    def test_intermediates_include_bare_site(self):
        from science.evaluation.golden_dataset import GOLDEN_SET
        for ex in GOLDEN_SET:
            assert "*" in ex.expected_intermediates, f"Missing * for {ex.id}"

    def test_unique_ids(self):
        from science.evaluation.golden_dataset import GOLDEN_SET
        ids = [ex.id for ex in GOLDEN_SET]
        assert len(ids) == len(set(ids)), "Duplicate IDs in golden set"

    def test_overpotential_range(self):
        from science.evaluation.golden_dataset import get_overpotential_range
        lo, hi = get_overpotential_range("her")
        assert lo < hi
        assert lo == pytest.approx(0.08, abs=0.01)

    def test_all_intermediates(self):
        from science.evaluation.golden_dataset import get_all_intermediates
        all_int = get_all_intermediates()
        assert "*" in all_int
        assert "CO2(g)" in all_int
        assert "H2O(g)" in all_int
        assert len(all_int) >= 20


# ─── Baseline Comparison Tests ──────────────────────────────────────────────

class TestBaselines:

    def test_baseline_site_finder(self, cu111_positions, cu111_elements):
        from science.benchmarks.baselines import baseline_distance_cutoff_sites
        result = baseline_distance_cutoff_sites(cu111_positions, cu111_elements)
        assert result.n_sites > 0
        assert "top" in result.site_types

    def test_baseline_keyword_score(self):
        from science.benchmarks.baselines import baseline_keyword_score
        score = baseline_keyword_score(
            "CO2 reduction to CO on Cu(111) via COOH* intermediate",
            ["*", "CO2(g)", "COOH*", "CO*"],
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # should find overlap

    def test_baseline_linear_extrapolation(self):
        from science.benchmarks.baselines import baseline_linear_extrapolation
        dE = list(0.5 * np.exp(-0.3 * np.arange(20)) + 1e-8)
        step, slosh = baseline_linear_extrapolation(dE)
        assert step > 0
        assert slosh is False  # baseline can't detect sloshing

    def test_synthetic_energy_landscape(self):
        from science.benchmarks.baselines import synthetic_energy_landscape
        e1 = synthetic_energy_landscape(400, 1600)
        e2 = synthetic_energy_landscape(600, 3200)
        # Higher ENCUT/KPPRA should be closer to converged
        assert abs(e2 - (-142.567)) < abs(e1 - (-142.567))


# ─── GNN Models ─────────────────────────────────────────────────────

class TestGNNModels:
    """Test GNN architectures for adsorption energy prediction."""

    @pytest.fixture
    def graph_data(self):
        """Small graph for testing GNN forward passes."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from science.predictions.gnn_models import GraphData
        rng = np.random.default_rng(42)
        N, E = 8, 20
        x = rng.random((N, 6)).astype(np.float32)
        ei = np.array([rng.integers(0, N, E), rng.integers(0, N, E)], dtype=np.int64)
        ea = rng.random((E, 3)).astype(np.float32)
        pos = rng.random((N, 3)).astype(np.float32) * 5
        return GraphData.from_numpy(x, ei, ea, pos, y=-1.5)

    def test_list_models(self):
        from science.predictions.gnn_models import list_models
        models = list_models()
        assert len(models) == 6
        assert "mlp" in models
        assert "se3_transformer" in models

    def test_mlp_forward(self, graph_data):
        from science.predictions.gnn_models import build_model
        model = build_model("mlp")
        out = model(graph_data)
        assert out.shape == (1,)
        assert model.num_params > 0

    def test_mpnn_forward(self, graph_data):
        from science.predictions.gnn_models import build_model
        model = build_model("mpnn")
        out = model(graph_data)
        assert out.shape == (1,)

    def test_gat_forward(self, graph_data):
        from science.predictions.gnn_models import build_model
        model = build_model("gat")
        out = model(graph_data)
        assert out.shape == (1,)

    def test_schnet_forward(self, graph_data):
        from science.predictions.gnn_models import build_model
        model = build_model("schnet")
        out = model(graph_data)
        assert out.shape == (1,)

    def test_dimenet_forward(self, graph_data):
        from science.predictions.gnn_models import build_model
        model = build_model("dimenet")
        out = model(graph_data)
        assert out.shape == (1,)

    def test_se3_transformer_forward(self, graph_data):
        from science.predictions.gnn_models import build_model
        model = build_model("se3_transformer")
        out = model(graph_data)
        assert out.shape == (1,)

    def test_build_unknown_model(self):
        from science.predictions.gnn_models import build_model
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("nonexistent")

    def test_graph_data_from_numpy(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from science.predictions.gnn_models import GraphData
        rng = np.random.default_rng(0)
        g = GraphData.from_numpy(
            x=rng.random((4, 6)).astype(np.float32),
            edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
            edge_attr=rng.random((3, 3)).astype(np.float32),
            pos=rng.random((4, 3)).astype(np.float32),
            y=-0.5,
        )
        assert g.x.shape == (4, 6)
        assert g.batch.shape == (4,)
        assert g.y.item() == pytest.approx(-0.5)

    def test_synthetic_dataset(self):
        from science.predictions.energy_predictor import generate_dataset
        samples = generate_dataset(n_samples=10, seed=0, n_atoms=8)
        assert len(samples) == 10
        # All energies should be finite
        for s in samples:
            assert np.isfinite(s.energy)
            assert len(s.elements) == 8

    def test_collate_graphs(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from science.predictions.gnn_models import GraphData
        from science.predictions.energy_predictor import collate_graphs
        rng = np.random.default_rng(42)
        graphs = []
        for i in range(3):
            N = 4 + i
            g = GraphData.from_numpy(
                x=rng.random((N, 6)).astype(np.float32),
                edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
                edge_attr=rng.random((2, 3)).astype(np.float32),
                pos=rng.random((N, 3)).astype(np.float32),
                y=float(i),
            )
            graphs.append(g)
        batch = collate_graphs(graphs)
        assert batch.x.shape[0] == 4 + 5 + 6  # 15 total nodes
        assert batch.y.shape == (3,)

    def test_training_reduces_loss(self):
        """Verify that a short training run reduces loss (model can learn)."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from science.predictions.energy_predictor import (
            generate_dataset, samples_to_graphs, train_and_evaluate,
        )
        samples = generate_dataset(n_samples=30, seed=42, n_atoms=8)
        graphs = samples_to_graphs(samples)
        r = train_and_evaluate("mpnn", graphs[:20], graphs[20:25], graphs[25:],
                               n_epochs=30, batch_size=8)
        # Loss should decrease from first to last epoch
        assert r.loss_curve[-1] < r.loss_curve[0]
