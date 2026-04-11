"""
Molecular Benchmarks — Head-to-Head Model Comparison on MoleculeNet
====================================================================

This module runs the complete benchmark pipeline:

  1. Load dataset (BBBP, Tox21, ESOL, etc.)
  2. Scaffold split (80/10/10)
  3. Featurize (fingerprints, descriptors, graphs)
  4. Train all 7 models
  5. Evaluate with bootstrap CIs
  6. Statistical significance tests
  7. Generate publication-ready leaderboard

Usage
-----
    # Quick benchmark (one dataset, all models)
    result = run_benchmark("bbbp", models=["random_forest", "xgboost", "mpnn"])

    # Full MoleculeNet benchmark
    results = run_full_moleculenet_benchmark()

    # Print leaderboard
    print(result.leaderboard("auroc"))

The numbers from this module are what you quote in interviews.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from science.molecular.datasets import (
    load_dataset,
    scaffold_split,
    smote_oversample,
)
from science.molecular.evaluation import (
    BenchmarkResult,
    ModelEvaluation,
    evaluate_classification,
    evaluate_regression,
    paired_test,
)
from science.molecular.qsar_models import (
    build_model,
)
from science.molecular.representations import (
    batch_morgan_fingerprints,
    batch_rdkit_descriptors,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single dataset benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    dataset_name: str,
    models: Optional[List[str]] = None,
    use_smote: bool = True,
    fp_radius: int = 2,
    fp_bits: int = 2048,
    seed: int = 42,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Run a complete benchmark on one dataset.

    Parameters
    ----------
    dataset_name : str
        MoleculeNet dataset name (bbbp, tox21, esol, etc.)
    models : List[str], optional
        Model names to benchmark. Default: all classical models.
    use_smote : bool
        Apply SMOTE oversampling for imbalanced classification.
    verbose : bool

    Returns
    -------
    BenchmarkResult with leaderboard.
    """
    if models is None:
        models = ["svm", "random_forest", "xgboost", "lightgbm"]

    # 1. Load dataset
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {dataset_name}")
        print(f"{'=' * 60}")

    # load_dataset here is ``science.molecular.datasets.load_dataset`` (imported above),
    # not HuggingFace's ``datasets.load_dataset``. Bandit's B615 rule matches by
    # function name and cannot distinguish — the call only reads from our own
    # local CSV cache, so revision pinning does not apply.
    dataset = load_dataset(dataset_name)  # nosec B615
    if verbose:
        print(dataset.summary())

    # 2. Scaffold split
    train, val, test = scaffold_split(dataset, seed=seed)
    if verbose:
        print(f"\nScaffold split: train={len(train)}, val={len(val)}, test={len(test)}")

    # 3. Featurize
    if verbose:
        print("Featurizing...")

    train_fp = batch_morgan_fingerprints(train.smiles, fp_radius, fp_bits)
    test_fp = batch_morgan_fingerprints(test.smiles, fp_radius, fp_bits)

    train_desc = batch_rdkit_descriptors(train.smiles)
    test_desc = batch_rdkit_descriptors(test.smiles)

    # Combined features for classical models. val split is not used at
    # benchmark time by the current model set — drop it from the featurizer
    # loop to avoid wasted rdkit work. Re-add when a model needs early
    # stopping against a val set.
    train_X = np.hstack([train_fp, train_desc])
    test_X = np.hstack([test_fp, test_desc])

    # NOTE: graph featurization for GNN models is scaffolded but not yet
    # wired into the benchmark loop; re-enable batch_smiles_to_graphs()
    # when a GNN model is added to the ``models`` list.

    task_type = dataset.info.task_type
    evaluations = []

    # 4. Train and evaluate each model
    for model_name in models:
        if verbose:
            print(f"\nTraining {model_name}...")

        try:
            model = build_model(model_name, task_type=task_type)
            task_results = []

            for task_idx, task_name in enumerate(dataset.info.task_names):
                train_y = train.targets[:, task_idx]
                test_y = test.targets[:, task_idx]

                # Remove NaN samples
                train_valid = ~np.isnan(train_y)
                test_valid = ~np.isnan(test_y)

                if train_valid.sum() < 10 or test_valid.sum() < 5:
                    continue

                cur_train_X = train_X[train_valid]
                cur_train_y = train_y[train_valid]
                cur_test_X = test_X[test_valid]
                cur_test_y = test_y[test_valid]

                # SMOTE for imbalanced classification
                if use_smote and task_type == "classification" and model_name not in ("mpnn", "gat", "transformer"):
                    cur_train_X, cur_train_y = smote_oversample(
                        cur_train_X,
                        cur_train_y,
                        seed=seed,
                    )

                # Train
                if model_name in ("mpnn", "gat"):
                    # Build graphs with aligned y values (skip invalid SMILES)
                    from science.molecular.representations import smiles_to_graph

                    cur_train_graphs, cur_train_y_g = [], []
                    for i in range(len(train.data)):
                        if not train_valid[i]:
                            continue
                        g = smiles_to_graph(train.data[i].smiles)
                        if g is not None:
                            cur_train_graphs.append(g)
                            cur_train_y_g.append(train_y[i])

                    cur_test_graphs, cur_test_y_g = [], []
                    for i in range(len(test.data)):
                        if not test_valid[i]:
                            continue
                        g = smiles_to_graph(test.data[i].smiles)
                        if g is not None:
                            cur_test_graphs.append(g)
                            cur_test_y_g.append(test_y[i])

                    if not cur_train_graphs or not cur_test_graphs:
                        continue
                    cur_train_y_g = np.array(cur_train_y_g)
                    cur_test_y = np.array(cur_test_y_g)
                    result = model.fit(cur_train_graphs, cur_train_y_g)
                    y_score = model.predict_proba(cur_test_graphs)
                elif model_name == "transformer":
                    cur_train_smiles = [s for i, s in enumerate(train.smiles) if train_valid[i]]
                    cur_test_smiles = [s for i, s in enumerate(test.smiles) if test_valid[i]]
                    result = model.fit(cur_train_smiles, cur_train_y[: len(cur_train_smiles)])
                    y_score = model.predict_proba(cur_test_smiles)
                else:
                    # No sample weights after SMOTE (SMOTE already balances)
                    result = model.fit(cur_train_X, cur_train_y)
                    y_score = model.predict_proba(cur_test_X)

                # Evaluate
                if task_type == "classification":
                    y_score = np.clip(y_score[: len(cur_test_y)], 0, 1)
                    task_result = evaluate_classification(
                        cur_test_y,
                        y_score,
                        task_name=task_name,
                    )
                else:
                    y_score = y_score[: len(cur_test_y)]
                    task_result = evaluate_regression(
                        cur_test_y,
                        y_score,
                        task_name=task_name,
                    )

                task_results.append(task_result)

                if verbose:
                    primary = list(task_result.metrics.values())[0]
                    print(f"  {task_name}: {list(task_result.metrics.keys())[0]}={primary:.4f}")

            if not task_results:
                continue

            # Aggregate metrics across tasks
            all_metrics = {}
            for key in task_results[0].metrics:
                vals = [tr.metrics[key] for tr in task_results if not np.isnan(tr.metrics.get(key, float("nan")))]
                if vals:
                    all_metrics[key] = float(np.mean(vals))

            evaluations.append(
                ModelEvaluation(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    split_method="scaffold",
                    task_results=task_results,
                    aggregate_metrics=all_metrics,
                    train_time_s=result.train_time_s,
                    n_params=result.n_params,
                )
            )

        except Exception as e:
            log.warning("Model %s failed: %s", model_name, e)
            if verbose:
                print(f"  FAILED: {e}")

    benchmark = BenchmarkResult(
        dataset_name=dataset_name,
        split_method="scaffold",
        evaluations=evaluations,
    )

    if verbose:
        metric = "auroc" if task_type == "classification" else "rmse"
        print(f"\n{benchmark.leaderboard(metric)}")

    return benchmark


# ---------------------------------------------------------------------------
# Full MoleculeNet benchmark (multiple datasets)
# ---------------------------------------------------------------------------


def run_full_moleculenet_benchmark(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, BenchmarkResult]:
    """
    Run benchmarks across multiple MoleculeNet datasets.

    Default datasets:
    - Classification: bbbp, bace, clintox
    - Regression: esol, freesolv, lipophilicity

    Returns dict mapping dataset_name → BenchmarkResult.
    """
    if datasets is None:
        datasets = ["bbbp", "bace", "esol", "freesolv", "lipophilicity"]
    if models is None:
        models = ["svm", "random_forest", "xgboost", "lightgbm"]

    results = {}
    for ds in datasets:
        try:
            results[ds] = run_benchmark(ds, models=models, verbose=verbose)
        except Exception as e:
            log.error("Benchmark %s failed: %s", ds, e)
            if verbose:
                print(f"\nBenchmark {ds} FAILED: {e}")

    if verbose:
        print("\n" + "=" * 60)
        print("FULL BENCHMARK SUMMARY")
        print("=" * 60)
        for ds, bm in results.items():
            if bm.evaluations:
                best = max(
                    bm.evaluations, key=lambda e: list(e.aggregate_metrics.values())[0] if e.aggregate_metrics else 0
                )
                primary_metric = list(best.aggregate_metrics.keys())[0]
                primary_val = list(best.aggregate_metrics.values())[0]
                print(f"  {ds}: best={best.model_name} ({primary_metric}={primary_val:.4f})")

    return results


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def compare_models(
    result: BenchmarkResult,
    model_a: str,
    model_b: str,
    metric: str = "auroc",
) -> str:
    """
    Statistical comparison of two models.

    Returns human-readable comparison with p-value.
    """
    eval_a = next((e for e in result.evaluations if e.model_name == model_a), None)
    eval_b = next((e for e in result.evaluations if e.model_name == model_b), None)

    if eval_a is None or eval_b is None:
        return f"Model not found: {model_a if eval_a is None else model_b}"

    scores_a = [tr.metrics.get(metric, 0) for tr in eval_a.task_results]
    scores_b = [tr.metrics.get(metric, 0) for tr in eval_b.task_results]

    if len(scores_a) < 2:
        mean_a = scores_a[0] if scores_a else 0
        mean_b = scores_b[0] if scores_b else 0
        return (
            f"{model_a}: {metric}={mean_a:.4f}\n"
            f"{model_b}: {metric}={mean_b:.4f}\n"
            f"(insufficient tasks for significance test)"
        )

    stat, p, conclusion = paired_test(scores_a, scores_b)
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)

    return (
        f"{model_a}: {metric}={mean_a:.4f}\n"
        f"{model_b}: {metric}={mean_b:.4f}\n"
        f"Δ = {mean_a - mean_b:+.4f}, p = {p:.4f}\n"
        f"Conclusion: {conclusion}"
    )


def format_publication_table(results: Dict[str, BenchmarkResult]) -> str:
    """
    Format results as a publication-ready LaTeX/markdown table.

    Output format matches standard MoleculeNet papers.
    """
    all_models = set()
    for bm in results.values():
        for ev in bm.evaluations:
            all_models.add(ev.model_name)
    all_models = sorted(all_models)

    # Header
    lines = [
        "| Model | " + " | ".join(results.keys()) + " |",
        "|-------|" + "|".join(["-------"] * len(results)) + "|",
    ]

    for model in all_models:
        row = [model]
        for ds, bm in results.items():
            ev = next((e for e in bm.evaluations if e.model_name == model), None)
            if ev and ev.aggregate_metrics:
                val = list(ev.aggregate_metrics.values())[0]
                # Get CI
                ci_str = ""
                for tr in ev.task_results:
                    primary = list(tr.metrics.keys())[0]
                    if primary in tr.confidence_intervals:
                        lo, hi = tr.confidence_intervals[primary]
                        ci_str = f" ±{(hi - lo) / 2:.3f}"
                        break
                row.append(f"{val:.3f}{ci_str}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
