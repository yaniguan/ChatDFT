"""
Molecular ML Module — ADME/QSAR Prediction, Generation, and Full ML Lifecycle
==============================================================================

This module provides production-grade molecular property prediction and
molecular generation capabilities, demonstrating the complete ML lifecycle:

  Data → Features → Train → Evaluate → Register → Serve → Monitor → Retrain

Submodules
----------
representations : Molecular featurization (fingerprints, descriptors, graphs)
datasets        : MoleculeNet/TDC loaders with scaffold splitting
qsar_models     : QSAR models (SVM, RF, XGBoost, LightGBM, GNN, Transformer)
applicability_domain : Out-of-distribution detection
generation      : Molecular generation (SMILES VAE, graph, diffusion)
evaluation      : Proper evaluation with bootstrap CI and scaffold splits
benchmarks      : Head-to-head comparison on MoleculeNet benchmarks
"""
