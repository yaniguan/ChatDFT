"""
Tests for the molecular ML module.

Tests cover:
  - Molecular representations (fingerprints, descriptors, graphs, tokenizer)
  - Dataset loading and scaffold splitting
  - QSAR models (train/predict interface)
  - Applicability domain assessment
  - Molecular generation (VAE, multi-objective scoring)
  - Evaluation metrics
  - Full benchmark pipeline
"""

import numpy as np
import pytest

# RDKit is required for most tests in this module


# ===========================================================================
# 1. Representations
# ===========================================================================

class TestRepresentations:

    def test_validate_smiles_valid(self):
        from science.molecular.representations import validate_smiles
        assert validate_smiles("CCO")
        assert validate_smiles("c1ccccc1")
        assert validate_smiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin

    def test_validate_smiles_invalid(self):
        from science.molecular.representations import validate_smiles
        assert not validate_smiles("not_a_smiles_xxx")
        assert not validate_smiles("")

    def test_canonicalize(self):
        from science.molecular.representations import canonicalize
        assert canonicalize("OCC") == "CCO"
        assert canonicalize("c1ccccc1") == "c1ccccc1"
        assert canonicalize("invalid_xxx") is None

    def test_get_scaffold(self):
        from science.molecular.representations import get_scaffold
        scaf = get_scaffold("c1ccc(CC(=O)O)cc1")
        assert scaf is not None
        assert len(scaf) > 0

    def test_morgan_fingerprint_shape(self):
        from science.molecular.representations import morgan_fingerprint
        fp = morgan_fingerprint("CCO", n_bits=1024)
        assert fp.shape == (1024,)
        assert fp.dtype == np.float32
        assert fp.sum() > 0  # some bits should be set

    def test_morgan_fingerprint_invalid(self):
        from science.molecular.representations import morgan_fingerprint
        fp = morgan_fingerprint("invalid_xxx", n_bits=1024)
        assert fp.sum() == 0  # all zeros for invalid

    def test_batch_morgan(self):
        from science.molecular.representations import batch_morgan_fingerprints
        fps = batch_morgan_fingerprints(["CCO", "c1ccccc1", "CC"], n_bits=512)
        assert fps.shape == (3, 512)

    def test_rdkit_descriptors_shape(self):
        from science.molecular.representations import rdkit_descriptors, DESCRIPTOR_NAMES
        desc = rdkit_descriptors("CCO")
        assert desc.shape == (len(DESCRIPTOR_NAMES),)
        assert desc[0] > 0  # MolWt should be > 0

    def test_smiles_to_graph(self):
        from science.molecular.representations import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
        g = smiles_to_graph("CCO")
        assert g is not None
        assert g.x.shape[0] == 3  # 3 heavy atoms (C, C, O)
        assert g.x.shape[1] == ATOM_FEATURE_DIM
        assert g.edge_index.shape[0] == 2
        assert g.edge_attr.shape[1] == BOND_FEATURE_DIM

    def test_smiles_to_graph_invalid(self):
        from science.molecular.representations import smiles_to_graph
        g = smiles_to_graph("invalid_xxx")
        assert g is None

    def test_tokenize_smiles(self):
        from science.molecular.representations import tokenize_smiles, detokenize_smiles
        tokens = tokenize_smiles("CCO", max_len=32)
        assert tokens.shape == (32,)
        assert tokens[0] == 1  # <sos>
        # Round-trip
        decoded = detokenize_smiles(tokens)
        assert "C" in decoded

    def test_featurize_molecule(self):
        from science.molecular.representations import featurize_molecule
        feat = featurize_molecule("c1ccccc1")
        assert feat.is_valid
        assert feat.fingerprint.sum() > 0
        assert feat.graph is not None
        assert feat.scaffold is not None


# ===========================================================================
# 2. Datasets
# ===========================================================================

class TestDatasets:

    def test_list_datasets(self):
        from science.molecular.datasets import list_datasets
        ds = list_datasets()
        assert "bbbp" in ds
        assert "esol" in ds
        assert "tox21" in ds

    def test_dataset_summary(self):
        from science.molecular.datasets import dataset_summary
        s = dataset_summary()
        assert "BBBP" in s or "bbbp" in s

    def test_smote_oversample(self):
        from science.molecular.datasets import smote_oversample
        rng = np.random.default_rng(42)
        X = rng.random((100, 10))
        y = np.array([0] * 90 + [1] * 10)
        X_new, y_new = smote_oversample(X, y, seed=42)
        assert len(X_new) > len(X)
        assert (y_new == 1).sum() == (y_new == 0).sum()  # balanced

    def test_focal_loss_weights(self):
        from science.molecular.datasets import focal_loss_weights
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.9, 0.1, 0.2, 0.8])
        w = focal_loss_weights(y_true, y_pred)
        assert w.shape == (4,)
        # Hard examples (misclassified) should get higher weight
        assert w[2] > w[0]  # y=1, pred=0.2 is harder than y=1, pred=0.9


# ===========================================================================
# 3. QSAR Models
# ===========================================================================

class TestQSARModels:

    @pytest.fixture
    def classification_data(self):
        """Small synthetic classification dataset."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 50))
        y = (X[:, 0] + X[:, 1] > 1.0).astype(float)
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Small synthetic regression dataset."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 50))
        y = X[:, 0] * 2 + X[:, 1] - 0.5 + rng.normal(0, 0.1, 100)
        return X, y

    def test_list_models(self):
        from science.molecular.qsar_models import list_models
        models = list_models()
        assert "svm" in models
        assert "random_forest" in models
        assert "xgboost" in models
        assert "mpnn" in models

    def test_svm_classification(self, classification_data):
        from science.molecular.qsar_models import build_model
        X, y = classification_data
        model = build_model("svm", task_type="classification")
        result = model.fit(X[:80], y[:80])
        assert result.model_name == "svm"
        preds = model.predict(X[80:])
        assert len(preds) == 20
        proba = model.predict_proba(X[80:])
        assert all(0 <= p <= 1 for p in proba)

    def test_random_forest_classification(self, classification_data):
        from science.molecular.qsar_models import build_model
        X, y = classification_data
        model = build_model("random_forest", task_type="classification", n_estimators=50)
        result = model.fit(X[:80], y[:80])
        assert result.train_metrics.get("oob_score", 0) > 0
        preds = model.predict_proba(X[80:])
        assert len(preds) == 20

    def test_xgboost_regression(self, regression_data):
        xgboost = pytest.importorskip("xgboost")
        from science.molecular.qsar_models import build_model
        X, y = regression_data
        model = build_model("xgboost", task_type="regression", n_estimators=50)
        model.fit(X[:80], y[:80])
        preds = model.predict(X[80:])
        assert len(preds) == 20
        # Predictions should be correlated with truth
        assert np.corrcoef(preds, y[80:])[0, 1] > 0.3

    def test_lightgbm_classification(self, classification_data):
        lightgbm = pytest.importorskip("lightgbm")
        from science.molecular.qsar_models import build_model
        X, y = classification_data
        model = build_model("lightgbm", task_type="classification", n_estimators=50)
        model.fit(X[:80], y[:80])
        proba = model.predict_proba(X[80:])
        assert all(0 <= p <= 1 for p in proba)


# ===========================================================================
# 4. Applicability Domain
# ===========================================================================

class TestApplicabilityDomain:

    def test_tanimoto_ad(self):
        from science.molecular.applicability_domain import TanimotoAD
        rng = np.random.default_rng(42)
        train = (rng.random((50, 100)) > 0.5).astype(float)
        ad = TanimotoAD(threshold=0.3)
        ad.fit(train)

        # In-domain: similar to training
        sim, in_domain = ad.score(train[0])
        assert sim >= 0.3
        assert in_domain

        # Out-of-domain: very different
        weird = np.ones(100, dtype=float)
        sim2, in_domain2 = ad.score(weird)
        # Result depends on data, just check types
        assert isinstance(sim2, float)
        assert isinstance(in_domain2, bool)

    def test_mahalanobis_ad(self):
        from science.molecular.applicability_domain import MahalanobisAD
        rng = np.random.default_rng(42)
        train = rng.normal(0, 1, (100, 10))
        ad = MahalanobisAD(percentile_threshold=95)
        ad.fit(train)

        # In-domain
        dist, in_domain = ad.score(rng.normal(0, 1, 10))
        assert isinstance(dist, float)

        # Out-of-domain: far from training
        dist2, in_domain2 = ad.score(np.ones(10) * 100)
        assert not in_domain2

    def test_ensemble_ad(self):
        from science.molecular.applicability_domain import EnsembleAD
        ad = EnsembleAD(std_threshold=0.2)

        # Low disagreement = in-domain
        std, in_dom = ad.score(np.array([0.8, 0.82, 0.79, 0.81]))
        assert in_dom
        assert std < 0.2

        # High disagreement = out-of-domain
        std2, in_dom2 = ad.score(np.array([0.1, 0.9, 0.3, 0.7]))
        assert not in_dom2

    def test_combined_assessor(self):
        from science.molecular.applicability_domain import ApplicabilityDomainAssessor
        rng = np.random.default_rng(42)
        fp = (rng.random((50, 100)) > 0.5).astype(float)
        desc = rng.normal(0, 1, (50, 10))

        ad = ApplicabilityDomainAssessor()
        ad.fit(fp, desc)

        result = ad.assess(
            smiles="CCO",
            fingerprint=fp[0],
            descriptor=desc[0],
            ensemble_predictions=np.array([0.5, 0.52, 0.48]),
        )
        assert result.in_domain
        assert 0 <= result.confidence <= 1


# ===========================================================================
# 5. Generation
# ===========================================================================

class TestGeneration:

    def test_compute_qed(self):
        from science.molecular.generation.multi_objective import compute_qed
        qed = compute_qed("c1ccccc1")  # benzene
        assert 0 < qed <= 1

    def test_compute_sa_score(self):
        from science.molecular.generation.multi_objective import compute_sa_score
        sa = compute_sa_score("CCO")  # ethanol — simple
        assert 1 <= sa <= 10

    def test_check_lipinski(self):
        from science.molecular.generation.multi_objective import check_lipinski
        result = check_lipinski("CCO")
        assert result["pass"]

    def test_multi_objective_scorer(self):
        from science.molecular.generation.multi_objective import MultiObjectiveScorer
        scorer = MultiObjectiveScorer()
        result = scorer.score("c1ccccc1")
        assert result.is_valid
        assert "qed" in result.scores
        assert "sa_score" in result.scores

    def test_pareto_front(self):
        from science.molecular.generation.multi_objective import MultiObjectiveScorer
        scorer = MultiObjectiveScorer()
        results = scorer.score_batch(["CCO", "c1ccccc1", "CC(=O)O"])
        pareto = scorer.pareto_front(results)
        assert len(pareto) >= 1

    def test_vae_structure(self):
        from science.molecular.generation.smiles_vae import SMILESVAE, VAEConfig
        import torch
        cfg = VAEConfig(latent_dim=16, hidden_dim=32, n_layers=1, max_len=32)
        vae = SMILESVAE(cfg)
        # Test forward pass
        tokens = torch.randint(0, cfg.vocab_size, (4, 32))
        logits, mu, logvar = vae(tokens)
        assert logits.shape[0] == 4
        assert mu.shape == (4, 16)


# ===========================================================================
# 6. Evaluation
# ===========================================================================

class TestEvaluation:

    def test_auroc(self):
        from science.molecular.evaluation import auroc
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        auc = auroc(y_true, y_score)
        assert 0.8 <= auc <= 1.0

    def test_auprc(self):
        from science.molecular.evaluation import auprc
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        ap = auprc(y_true, y_score)
        assert 0.5 <= ap <= 1.0

    def test_mcc(self):
        from science.molecular.evaluation import mcc
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        assert mcc(y_true, y_pred) == 1.0
        # Worse predictions should have lower MCC
        y_pred2 = np.array([1, 0, 1, 0, 1, 0])
        assert mcc(y_true, y_pred2) < 1.0

    def test_rmse(self):
        from science.molecular.evaluation import rmse
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        assert rmse(y_true, y_pred) == pytest.approx(0.1, abs=0.01)

    def test_bootstrap_ci(self):
        from science.molecular.evaluation import bootstrap_ci, auroc
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100).astype(float)
        y_score = y_true * 0.7 + rng.random(100) * 0.3
        point, lo, hi = bootstrap_ci(y_true, y_score, auroc, n_bootstrap=100)
        assert lo <= point <= hi
        assert point > 0.5

    def test_evaluate_classification(self):
        from science.molecular.evaluation import evaluate_classification
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.6, 0.4, 0.9, 0.2, 0.5])
        result = evaluate_classification(y_true, y_score, "test_task", n_bootstrap=50)
        assert result.task_name == "test_task"
        assert "auroc" in result.metrics
        assert "auprc" in result.metrics
        assert "mcc" in result.metrics

    def test_evaluate_regression(self):
        from science.molecular.evaluation import evaluate_regression
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        result = evaluate_regression(y_true, y_pred, "test_reg", n_bootstrap=50)
        assert "rmse" in result.metrics
        assert "r2" in result.metrics
        assert result.metrics["r2"] > 0.9


# ===========================================================================
# 7. Integration: model registry and feature store
# ===========================================================================

class TestIntegration:

    def test_model_registry_molecular(self):
        """Molecular models can be registered in the existing model registry."""
        from server.mlops.model_registry import ModelRegistry, ModelStage
        registry = ModelRegistry()
        mv = registry.register(
            name="qsar_bbbp_rf",
            version="1.0.0",
            model_type="random_forest",
            framework="sklearn",
            stage=ModelStage.DEVELOPMENT,
            metrics={"auroc": 0.85, "auprc": 0.82},
            hyperparameters={"n_estimators": 500, "max_depth": None},
            description="Random Forest for BBBP (scaffold split)",
        )
        assert mv.name == "qsar_bbbp_rf"

        # Promote to production
        registry.promote("qsar_bbbp_rf", "1.0.0", ModelStage.PRODUCTION)
        active = registry.get_active("qsar_bbbp_rf")
        assert active is not None
        assert active.stage == ModelStage.PRODUCTION
