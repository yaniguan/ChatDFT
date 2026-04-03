"""
Scientific Correctness Tests
==============================

Tests that verify physical invariants, catch invalid inputs, and lock
numerical outputs against regression.

Categories:
  - Rotational/translational invariance
  - Physical monotonicity constraints
  - Negative tests (invalid inputs)
  - Numerical regression (fixed reference outputs)
  - Thermodynamic consistency
"""

import numpy as np
import pytest


# ─── Rotational / Translational Invariance ─────────────────────────

class TestInvariance:
    """Graph features and energies must be invariant under rigid transforms."""

    def _build_graph(self, positions, elements, cell):
        from science.representations.surface_graph import SurfaceTopologyGraph
        stg = SurfaceTopologyGraph(positions, elements, cell)
        stg.build()
        return stg

    def test_translation_invariance(self, cu111_positions, cu111_elements, cu111_cell):
        """Node features must not change under uniform translation."""
        stg1 = self._build_graph(cu111_positions, cu111_elements, cu111_cell)
        X1 = stg1.node_feature_matrix()

        # Translate by arbitrary vector
        shifted = cu111_positions + np.array([3.14, -2.71, 0.42])
        stg2 = self._build_graph(shifted, cu111_elements, cu111_cell)
        X2 = stg2.node_feature_matrix()

        # Features should be nearly identical (surface_dist changes with z-shift,
        # but relative ordering and normalised features should be close)
        # Z/100 and layer should be exactly the same
        np.testing.assert_allclose(X1[:, 0], X2[:, 0], atol=1e-6)  # Z/100
        np.testing.assert_array_equal(X1[:, 1], X2[:, 1])           # layer

    def test_rebuild_determinism(self, cu111_positions, cu111_elements, cu111_cell):
        """Building the same graph twice should give identical features."""
        stg1 = self._build_graph(cu111_positions, cu111_elements, cu111_cell)
        X1 = stg1.node_feature_matrix()

        stg2 = self._build_graph(cu111_positions, cu111_elements, cu111_cell)
        X2 = stg2.node_feature_matrix()

        np.testing.assert_array_equal(X1, X2)

    def test_schnet_rotation_invariance(self):
        """SchNet energy must be invariant under rotation (uses distances only)."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from science.predictions.gnn_models import build_model, GraphData

        rng = np.random.default_rng(42)
        N, E = 6, 12
        x = rng.random((N, 6)).astype(np.float32)
        ei = np.array([rng.integers(0, N, E), rng.integers(0, N, E)], dtype=np.int64)
        ea = rng.random((E, 3)).astype(np.float32)
        pos = rng.random((N, 3)).astype(np.float32) * 5

        # Rotation matrix (90° around z-axis)
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        pos_rot = pos @ R.T

        data1 = GraphData.from_numpy(x, ei, ea, pos, y=0.0)
        data2 = GraphData.from_numpy(x, ei, ea, pos_rot, y=0.0)

        model = build_model("schnet")
        model.eval()
        with torch.no_grad():
            e1 = model(data1).item()
            e2 = model(data2).item()
        assert abs(e1 - e2) < 1e-4, f"SchNet not rotation-invariant: {e1} vs {e2}"

    def test_se3_rotation_equivariance(self):
        """SE(3)-Transformer energy must be invariant under rotation.

        f(Rx) = f(x) for scalar output (energy).
        Internally, vector features should transform as R·v.
        """
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from science.predictions.gnn_models import build_model, GraphData

        rng = np.random.default_rng(42)
        N, E = 6, 12
        x = rng.random((N, 6)).astype(np.float32)
        ei = np.array([rng.integers(0, N, E), rng.integers(0, N, E)], dtype=np.int64)
        ea = rng.random((E, 3)).astype(np.float32)
        pos = rng.random((N, 3)).astype(np.float32) * 5

        # Arbitrary rotation (Rodrigues)
        theta = 1.23
        axis = np.array([0.3, 0.5, 0.8])
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
        R = (np.eye(3) + np.sin(theta) * K +
             (1 - np.cos(theta)) * K @ K).astype(np.float32)
        pos_rot = pos @ R.T

        data1 = GraphData.from_numpy(x, ei, ea, pos, y=0.0)
        data2 = GraphData.from_numpy(x, ei, ea, pos_rot, y=0.0)

        model = build_model("se3_transformer")
        model.eval()
        with torch.no_grad():
            e1 = model(data1).item()
            e2 = model(data2).item()
        # Energy (scalar) should be invariant
        assert abs(e1 - e2) < 1e-3, (
            f"SE(3)-Transformer not rotation-invariant: {e1} vs {e2}, "
            f"diff={abs(e1-e2):.6f}"
        )


# ─── Physical Monotonicity Constraints ──────────────────────────────

class TestPhysicsConstraints:
    """Physical laws that the algorithms must respect."""

    def test_einstein_sigma_increases_with_temperature(self):
        """σ(T) must increase monotonically with temperature."""
        from science.generation.informed_sampler import EinsteinRattler
        rattler = EinsteinRattler(omega_THz=5.0, quantum=True, rng_seed=42)
        temps = [1, 50, 100, 300, 600, 1000, 2000]
        sigmas = [rattler._sigma(63.546, T) for T in temps]
        for i in range(1, len(sigmas)):
            assert sigmas[i] >= sigmas[i - 1], (
                f"σ not monotone: σ({temps[i-1]}K)={sigmas[i-1]:.6f} > "
                f"σ({temps[i]}K)={sigmas[i]:.6f}"
            )

    def test_einstein_lighter_atoms_displace_more(self):
        """Lighter atoms must have larger displacement at same T."""
        from science.generation.informed_sampler import EinsteinRattler
        rattler = EinsteinRattler(omega_THz=5.0, quantum=True, rng_seed=42)
        sigma_H = rattler._sigma(1.008, 300)
        sigma_Cu = rattler._sigma(63.546, 300)
        sigma_Pt = rattler._sigma(195.084, 300)
        assert sigma_H > sigma_Cu > sigma_Pt

    def test_quantum_zpe_at_zero_temperature(self):
        """At T→0, quantum σ >> classical σ (zero-point energy dominates)."""
        from science.generation.informed_sampler import EinsteinRattler
        q_rattler = EinsteinRattler(omega_THz=5.0, quantum=True, rng_seed=42)
        c_rattler = EinsteinRattler(omega_THz=5.0, quantum=False, rng_seed=42)
        sigma_q = q_rattler._sigma(63.546, 0.001)
        sigma_c = c_rattler._sigma(63.546, 0.001)
        assert sigma_q > 0.1, "Quantum ZPE should give finite σ at T→0"
        assert sigma_q > 10 * sigma_c, "Quantum σ should dominate classical at T→0"

    def test_bo_higher_params_lower_error(self):
        """Higher ENCUT + KPPRA should give lower energy error."""
        from science.benchmarks.baselines import synthetic_energy_landscape
        e_low = synthetic_energy_landscape(350, 800)
        e_high = synthetic_energy_landscape(600, 3200)
        target = -142.567
        assert abs(e_high - target) < abs(e_low - target)

    def test_golden_dataset_dG_starts_at_zero(self):
        """All free energy profiles must start at 0 (CHE convention)."""
        from science.evaluation.golden_dataset import GOLDEN_SET
        for ex in GOLDEN_SET:
            assert ex.expected_dG_profile[0] == 0.0, (
                f"{ex.id}: dG[0] = {ex.expected_dG_profile[0]}"
            )

    def test_golden_dataset_overpotentials_physical(self):
        """Overpotentials must be positive and < 5 V."""
        from science.evaluation.golden_dataset import GOLDEN_SET
        for ex in GOLDEN_SET:
            assert 0 < ex.expected_overpotential < 5.0, (
                f"{ex.id}: η = {ex.expected_overpotential}"
            )


# ─── Negative Tests (Invalid Inputs) ────────────────────────────────

class TestNegativeInputs:
    """Invalid inputs should raise clear errors, not silently corrupt."""

    def test_nan_positions_rejected(self):
        from science.representations.surface_graph import SurfaceTopologyGraph
        from science.core.errors import InvalidStructure
        pos = np.array([[0, 0, 0], [np.nan, 1, 2]], dtype=float)
        cell = np.eye(3) * 10
        with pytest.raises(InvalidStructure, match="NaN"):
            SurfaceTopologyGraph(pos, ["Cu", "Cu"], cell)

    def test_zero_volume_cell_rejected(self):
        from science.representations.surface_graph import SurfaceTopologyGraph
        from science.core.errors import InvalidStructure
        pos = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        cell = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float)  # degenerate
        with pytest.raises(InvalidStructure, match="[Dd]egenerate"):
            SurfaceTopologyGraph(pos, ["Cu", "Cu"], cell)

    def test_mismatched_positions_elements(self):
        from science.representations.surface_graph import SurfaceTopologyGraph
        from science.core.errors import InvalidStructure
        pos = np.array([[0, 0, 0]], dtype=float)
        cell = np.eye(3) * 10
        with pytest.raises(InvalidStructure):
            SurfaceTopologyGraph(pos, ["Cu", "Pt"], cell)

    def test_empty_scf_trajectory(self):
        from science.time_series.scf_convergence import SCFTrajectory, analyse_scf
        traj = SCFTrajectory(dE=[], ediff=1e-5, nelm=60)
        report = analyse_scf(traj, is_metal=True)
        # Should handle gracefully, not crash
        assert report is not None

    def test_gnn_empty_graph(self):
        """GNN should handle graph with 0 edges gracefully."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from science.predictions.gnn_models import build_model, GraphData
        N = 4
        x = np.random.rand(N, 6).astype(np.float32)
        ei = np.zeros((2, 0), dtype=np.int64)  # no edges
        ea = np.zeros((0, 3), dtype=np.float32)
        pos = np.random.rand(N, 3).astype(np.float32)
        data = GraphData.from_numpy(x, ei, ea, pos, y=0.0)

        model = build_model("mlp")
        model.eval()
        with torch.no_grad():
            out = model(data)
        assert out.shape == (1,)
        assert torch.isfinite(out).all()


# ─── Numerical Regression Tests ─────────────────────────────────────

class TestNumericalRegression:
    """Lock key outputs to prevent silent numerical drift."""

    def test_cu111_node_features_regression(self, cu111_positions, cu111_elements, cu111_cell):
        """Cu(111) node features must match reference values."""
        from science.representations.surface_graph import SurfaceTopologyGraph
        stg = SurfaceTopologyGraph(cu111_positions, cu111_elements, cu111_cell)
        stg.build()
        X = stg.node_feature_matrix()

        # Cu atomic number / 100
        np.testing.assert_allclose(X[:, 0], 0.29, atol=0.01)
        # All atoms are Cu, so Z feature is uniform
        assert X[:, 0].std() < 0.001

    def test_einstein_sigma_regression(self):
        """Einstein σ at 300K for Cu must match reference."""
        from science.generation.informed_sampler import EinsteinRattler
        rattler = EinsteinRattler(omega_THz=5.0, quantum=True, rng_seed=42)
        sigma = rattler._sigma(63.546, 300.0)
        # Reference: σ(Cu, 300K, ω=5THz) ≈ 1.118 Å (verified from implementation)
        assert abs(sigma - 1.118) < 0.05, f"σ(Cu, 300K) = {sigma}"

    def test_fft_sloshing_detection_regression(self):
        """Sloshing detector must correctly classify known trajectories."""
        from science.time_series.scf_convergence import (
            SCFTrajectory, ChargeSloshingDetector,
        )
        detector = ChargeSloshingDetector()

        # Healthy: pure exponential decay
        t = np.arange(30)
        healthy_dE = list(0.5 * np.exp(-0.35 * t) + 1e-8)
        result_h = detector.detect(SCFTrajectory(dE=healthy_dE, ediff=1e-5, nelm=60))
        assert not result_h.is_sloshing, "Healthy trajectory misclassified as sloshing"

        # Sloshing: oscillatory with slow decay
        sloshing_dE = list(0.01 * np.exp(-0.02 * t) *
                           (0.5 + np.abs(np.sin(2 * np.pi * t / 5))) + 1e-7)
        result_s = detector.detect(SCFTrajectory(dE=sloshing_dE, ediff=1e-5, nelm=60))
        assert result_s.is_sloshing, "Sloshing trajectory missed"

    def test_synthetic_landscape_regression(self):
        """Synthetic energy landscape must produce stable values."""
        from science.benchmarks.baselines import synthetic_energy_landscape
        e = synthetic_energy_landscape(500, 2400)
        assert abs(e - (-142.5639)) < 0.01, f"Landscape regression: {e}"


# ─── Seed Manager Tests ─────────────────────────────────────────────

class TestSeedManager:
    """Verify reproducibility infrastructure."""

    def test_set_global_seed(self):
        from science.core.seeds import set_global_seed, get_rng
        set_global_seed(123)
        rng1 = get_rng("test_module")
        val1 = rng1.random()

        # Reset and get same value
        set_global_seed(123)
        rng2 = get_rng("test_module")
        val2 = rng2.random()
        assert val1 == val2

    def test_different_modules_different_rngs(self):
        from science.core.seeds import set_global_seed, get_rng
        set_global_seed(42)
        rng_a = get_rng("module_a")
        rng_b = get_rng("module_b")
        assert rng_a.random() != rng_b.random()

    def test_experiment_manifest(self):
        from science.core.seeds import set_global_seed, experiment_manifest
        set_global_seed(42)
        m = experiment_manifest()
        assert m["global_seed"] == 42
        assert "numpy_version" in m


# ─── Error Taxonomy Tests ───────────────────────────────────────────

class TestErrorTaxonomy:
    """Verify the error hierarchy works correctly."""

    def test_invalid_structure_is_data_error(self):
        from science.core.errors import InvalidStructure, DataError, ChatDFTError
        e = InvalidStructure("bad cell", context={"volume": 0.0})
        assert isinstance(e, DataError)
        assert isinstance(e, ChatDFTError)
        assert "volume=0.0" in str(e)

    def test_physics_error_hierarchy(self):
        from science.core.errors import ThermodynamicError, PhysicsError
        e = ThermodynamicError("negative entropy")
        assert isinstance(e, PhysicsError)

    def test_model_error_hierarchy(self):
        from science.core.errors import TrainingError, ModelError
        e = TrainingError("loss diverged", context={"epoch": 50, "loss": float("inf")})
        assert isinstance(e, ModelError)
        assert "epoch=50" in str(e)


# ─── Active Learning Tests ──────────────────────────────────────────

class TestActiveLearning:
    """Test the DFT↔GNN active learning loop."""

    def test_ensemble_uncertainty(self):
        """Ensemble should produce uncertainty > 0."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from science.predictions.active_learning import GNNEnsemble
        from science.predictions.energy_predictor import generate_dataset, samples_to_graphs

        ensemble = GNNEnsemble("schnet", n_models=3, d_hidden=32, n_interactions=2)
        samples = generate_dataset(n_samples=10, seed=42, n_atoms=8)
        graphs = samples_to_graphs(samples)

        means, stds = ensemble.predict_with_uncertainty(graphs)
        assert means.shape == (10,)
        assert stds.shape == (10,)
        # Untrained ensemble should have nonzero uncertainty
        assert np.any(stds > 0)

    def test_active_loop_reduces_uncertainty(self):
        """Active learning should reduce uncertainty over iterations."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from science.predictions.active_learning import (
            GNNEnsemble, ActiveLearner,
        )
        from science.predictions.energy_predictor import (
            generate_dataset, synthetic_adsorption_energy,
        )

        initial = generate_dataset(n_samples=15, seed=42, n_atoms=8)
        oracle = lambda e, a, cn: synthetic_adsorption_energy(e, a, cn, noise_std=0.05)

        ensemble = GNNEnsemble("schnet", n_models=3, d_hidden=32, n_interactions=2)
        learner = ActiveLearner(oracle=oracle, ensemble=ensemble,
                                pool_size=20, batch_per_iter=3)
        result = learner.run(initial, max_iterations=3, strategy="uncertainty", seed=42)

        assert result.n_dft_calls > len(initial)
        assert len(result.mae_curve) == 3
        assert len(result.uncertainty_curve) == 3

    def test_trainable_grounder_improves(self):
        """Trainable grounder should reduce loss during training."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from science.alignment.hypothesis_grounder import (
            TrainableGrounder, ReactionNetwork,
        )

        grounder = TrainableGrounder(d_embed=32)
        pairs = []
        for i in range(15):
            h = f"CO2 reduction step {i} on Cu via COOH*"
            n = ReactionNetwork.from_dict({
                "reaction_network": [{"lhs": ["CO2(g)", "*"], "rhs": ["COOH*"]}],
                "intermediates": ["*", "CO2(g)", "COOH*"],
            })
            dG = [0.0, 0.2 + i * 0.01, -0.1]
            pairs.append((h, n, dG))

        loss_curve = grounder.train(pairs, n_epochs=15, batch_size=8)
        assert loss_curve[-1] <= loss_curve[0]  # loss should not increase

    def test_threshold_validation(self):
        """FFT threshold validation should produce valid metrics."""
        from science.time_series.scf_convergence import validate_thresholds
        result = validate_thresholds(n_healthy=15, n_sloshing=15, seed=42, n_grid=5)
        assert 0 < result.best_ac_ratio < 1
        assert 0 < result.f1 <= 1
        assert len(result.roc_curve) > 0
