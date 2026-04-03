# ChatDFT

**Autonomous Reaction Pathway Discovery via LLM-Guided DFT**

[![Tests](https://img.shields.io/badge/tests-129%20passing-brightgreen)]()
[![Benchmarks](https://img.shields.io/badge/benchmarks-25%20reactions-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Abstract

Heterogeneous catalysis research requires exploring a combinatorial space of intermediates, surface configurations, and DFT parameters. ChatDFT attacks this at the **algorithmic level** — not by building a wrapper around VASP, but by developing six novel computational methods that reduce the human and computational cost of reaction pathway discovery:

| # | Algorithm | Key Result |
|---|---|---|
| 1 | **Voronoi topology graph** for surface representation | 4-class site classification with symmetry scoring |
| 2 | **Quantum harmonic oscillator rattle** for structure generation | Mass-weighted ZPE-aware sampling (σ_H/σ_Pt = 13.9x) |
| 3 | **InfoNCE cross-modal alignment** for hypothesis grounding | AUC = 1.00 vs 0.43 keyword baseline (trained) |
| 4 | **FFT charge sloshing detection** for SCF diagnostics | 100% accuracy vs 50% baseline on 60 trajectories |
| 5 | **Bayesian optimisation** for DFT parameter search | 73% fewer DFT evaluations vs grid search |
| 6 | **GNN energy prediction** (MPNN/GAT/SchNet/DimeNet/SE(3)-Tr.) | 5 architectures benchmarked on synthetic E_ads |

All algorithms are benchmarked against baselines on a **25-reaction golden dataset** spanning CO₂RR, HER, OER, NRR, and ORR with literature-validated free energy profiles.

---

## Quick Start (30 seconds)

```bash
pip install -e .                    # Install
python demo.py                      # Run all 5 algorithms — zero config needed
python demo.py --benchmark          # Generate publication figures
make test                           # Run 129 tests
```

No database, no API key, no VASP license needed for the demo.

---

## Scientific Contributions

### 1. Voronoi Topology Graph (`science/representations/`)

**Problem**: Predicting adsorbate binding sites requires a representation that is rotation-invariant, sensitive to local chemistry, and GNN-compatible.

**Method**: Voronoi tessellation with periodic boundary conditions encodes catalyst slabs as topology graphs. Nodes carry 6 normalised features: [Z/100, layer, CN/12, V_Voronoi/20, d_surface/5, angle_variance]. Adsorption sites are classified into 4 types (top, bridge, hollow-fcc, hollow-hcp) via Delaunay triangulation with subsurface atom detection. Site symmetry is scored via the eigenvalue ratio of the local coordination tensor.

**Result**: On 6 fcc surfaces (Cu, Pt, Ag, Au, Ni), the Voronoi method produces **4-class classification with symmetry scores** while the distance-cutoff baseline produces only 3 classes with no quality ranking. Output is directly compatible with PyTorch Geometric `(edge_index, edge_attr)` format.

```python
stg = SurfaceTopologyGraph(positions, elements, cell)
stg.build()
sites = stg.classify_adsorption_sites()   # top, bridge, hollow_fcc, hollow_hcp
X     = stg.node_feature_matrix()         # (N, 6) → GNN input
```

### 2. Physics-Informed Structure Generation (`science/generation/`)

**Problem**: Training neural network potentials (NNPs) requires diverse configurations, but uniform random noise over-samples irrelevant regions and ignores quantum effects.

**Method**: Three complementary strategies grounded in statistical mechanics:
- **Einstein rattle**: σ�� = √(ħ/(2mᵢω) · coth(ħω/2kBT)) — includes zero-point energy at T→0, transitions to classical kBT/mω² at high T
- **Normal-mode sampling**: phonon eigenmodes excited with equipartition amplitudes √(kBT/λν), filtering acoustic modes below 10⁻³ eV/A²
- **Active learning**: committee uncertainty σ(x)/N_atoms selects DFT-informative configurations

**Result**: At 600 K, hydrogen atoms (1 amu) displace **13.9x more** than platinum (195 amu) — physically correct mass scaling that uniform noise misses entirely. At T→0, quantum ZPE gives σ = 0.222 A for Cu vs 0.012 A in the classical limit.

### 3. Cross-Modal Hypothesis Grounding (`science/alignment/`)

**Problem**: An LLM proposes a reaction mechanism in natural language. DFT computes a free energy diagram. How do we score consistency?

**Method**: Three modalities — (text hypothesis, reaction graph, free energy profile) — are encoded into a shared 64-dim embedding space via independent encoders. The alignment objective is **InfoNCE contrastive loss** (CLIP framework). At inference, weighted cosine similarity (0.6 × sim(T,G) + 0.4 × sim(T,P)) serves as a principled confidence score.

**Result**: On 30 hypothesis-mechanism pairs (15 correct + 15 mismatched), the trained cross-modal scorer achieves **AUC = 1.00** vs **AUC = 0.43** for keyword overlap, with a score separation of 0.25 between correct and incorrect pairs.

### 4. SCF Convergence Time-Series Analysis (`science/time_series/`)

**Problem**: Charge sloshing in metallic systems wastes hundreds of SCF iterations. Post-hoc diagnosis doesn't help — you need real-time detection.

**Method**: Two-stage analysis of the SCF energy residual sequence:
1. **FFT sloshing detection**: detrend log|ΔEₙ| via linear regression, apply Hanning window, compute one-sided FFT. Flag if AC/total power ratio > 0.3 AND sign-change rate > 0.3.
2. **Convergence rate prediction**: fit log|ΔEₙ| ≈ log(A) − λn via OLS, extrapolate n_conv = (log(A) − log(EDIFF))/λ. R² quantifies confidence.

**Result**: On 60 synthetic trajectories (30 healthy + 30 sloshing), the FFT+sign-change detector achieves **100% accuracy** vs **50% for the linear baseline** (which cannot detect sloshing at all). Step prediction MAE: **3.6 vs 7.4 steps**.

### 5. Bayesian Optimisation for DFT Parameters (`science/optimization/`)

**Problem**: Convergence testing (ENCUT × KPPRA grid) requires ~56-80 DFT single-points per material. Most of this grid is far from the optimal front.

**Method**: Gaussian Process with Matern-5/2 kernel fits a surrogate on (ENCUT, KPPRA) → energy_error. Expected Improvement (EI) with ParEGO-style cost scalarisation (EI/√cost) selects the next evaluation point.

**Result**: Finds converged parameters in **15 evaluations** vs 56 for grid search — a **73% reduction** in DFT compute with equivalent accuracy (error < 1 meV/atom).

### 6. GNN Energy Prediction (`science/predictions/`)

**Problem**: Screening thousands of catalyst candidates requires fast energy predictions. Which GNN architecture best captures the physics of adsorption from a surface topology graph?

**Method**: Five GNN architectures implemented in pure PyTorch, consuming the Voronoi topology graph directly:
- **MPNN** (Gilmer 2017): edge-conditioned messages with GRU update — captures bond-level interactions
- **GAT** (Velickovic 2018): multi-head attention learns which neighbours matter most
- **SchNet** (Schutt 2018): continuous radial basis filters on interatomic distances
- **DimeNet** (Gasteiger 2020): directional message passing incorporating bond angles via spherical harmonics
- **SE(3)-Transformer** (Fuchs 2020): equivariant attention with scalar (type-0) and vector (type-1) features, guaranteeing rotational invariance by construction

All are benchmarked against an MLP baseline (mean-pooled node features, no graph structure) on a synthetic adsorption energy dataset encoding d-band centre correlations, coordination effects, and adsorbate scaling relations.

**Result**: Geometry-aware models (SchNet, DimeNet, SE(3)-Transformer) consistently outperform topology-only models (MPNN, GAT) which outperform the MLP baseline, confirming that **angular and equivariant information matters** for energy prediction on catalyst surfaces.

```python
from science.predictions.gnn_models import build_model
from science.predictions.energy_predictor import generate_dataset, samples_to_graphs

model = build_model("schnet")               # or: mpnn, gat, dimenet, se3_transformer
samples = generate_dataset(n_samples=200)
graphs = samples_to_graphs(samples)          # Voronoi graph → GNN input
```

---

## Benchmark Results

All results generated by `python demo.py --benchmark`. Figures in `figures/`.

| Algorithm | ChatDFT | Baseline | Metric |
|---|---|---|---|
| Surface site classification | 4-class + symmetry score | 3-class, no ranking | site type granularity |
| Structure generation | Mass-weighted σ(T) with ZPE | Fixed σ = 0.1 A | physical correctness |
| Hypothesis grounding | AUC = 1.00 | AUC = 0.43 | discrimination (30 pairs) |
| SCF sloshing detection | 100% accuracy | 50% accuracy | 60 trajectories |
| SCF step prediction | MAE = 3.6 steps | MAE = 7.4 steps | 30 healthy trajectories |
| Parameter search | 15 evaluations | 56 evaluations | same target error |
| GNN energy prediction | SE(3)/DimeNet best | MLP baseline | test MAE (eV) on 200 samples |

### Golden Benchmark Dataset

25 reactions across 5 domains with literature-validated DFT free energy profiles:

| Domain | Reactions | η range (V) | DOIs |
|---|---|---|---|
| CO₂RR | 8 | 0.24 – 0.82 | Peterson 2010, Kuhl 2012 |
| HER | 5 | 0.08 – 0.42 | Skulason 2012, Hinnemann 2005 |
| OER | 5 | 0.35 – 0.60 | Man 2011, Rossmeisl 2007 |
| NRR | 4 | 0.72 – 2.18 | Montoya 2015, Shi 2014 |
| ORR | 3 | 0.45 – 0.70 | Norskov 2004 |

---

## System Architecture

The scientific algorithms integrate into an end-to-end research assistant:

```
Natural language query
        │
        ▼
   Intent Agent          ← parse substrate, facet, pH, reaction type
        │
        ▼
 Hypothesis Agent        ← LLM + RAG + cross-modal grounder [Module 3]
   (reaction network)
        │
        ▼
    Plan Agent           ← generate DFT task graph
        │
   ┌────┴────┐
   ▼         ▼
Structure  Parameters    ← Voronoi sites [Module 1] + BO search [Module 5]
  Agent      Agent         Einstein rattle [Module 2]
   └────┬────┘
        ▼
   HPC Execution         ← SSH job submission (PBS/SGE/SLURM)
        │
        ▼
  Post-Analysis          ← SCF diagnostics [Module 4] + CHE thermodynamics
        │
        ▼
  Free Energy Diagram + Overpotential + Rate-Determining Step
```

**Stack**: Python 3.10+ · FastAPI · PostgreSQL + pgvector · ASE · VASP · Streamlit

---

## Repository Structure

```
science/                          # Novel algorithms (no external deps beyond numpy/scipy/torch)
├── representations/surface_graph.py    # Voronoi topology graph (525 lines)
├── generation/informed_sampler.py      # Einstein + normal-mode rattle (435 lines)
├── alignment/hypothesis_grounder.py    # InfoNCE cross-modal scorer (499 lines)
├── time_series/scf_convergence.py      # FFT sloshing + rate predictor (551 lines)
├── optimization/bayesian_params.py     # GP + EI parameter search (345 lines)
├── predictions/gnn_models.py           # MPNN, GAT, SchNet, DimeNet, SE(3)-Transformer
├── predictions/energy_predictor.py     # Unified train/eval + synthetic benchmark
├── evaluation/golden_dataset.py        # 25-reaction benchmark dataset
├── evaluation/metrics.py               # Component-level evaluation suite
└── benchmarks/                         # Baseline comparisons + figure generation
    ├── baselines.py                    # 5 naive baselines
    └── run_benchmarks.py               # Publication figure generator (7 figures)

server/                           # FastAPI backend
├── chat/                         # LLM agents (intent, hypothesis, plan, knowledge)
├── execution/                    # Structure building, parameter selection, HPC
├── mlops/                        # Model registry, experiment tracker, monitoring
├── feature_store/                # Feature lineage + drift detection
└── science_routes.py             # REST API for science modules

client/app.py                     # Streamlit UI (9 tabs)
notebooks/                        # 7 interactive Jupyter notebooks
tests/                            # 129 tests (science + ML system + server)
figures/                          # Publication-quality benchmark figures (PDF + PNG)
```

---

## How to Run

### Demo (no database needed)
```bash
pip install -e .
python demo.py                          # All 5 algorithms
python demo.py --module 1               # Just surface graph
python demo.py --benchmark              # Generate all figures
```

### Full System
```bash
# 1. Database
docker-compose up -d db
alembic upgrade head

# 2. API server
uvicorn server.main:app --reload --port 8000

# 3. Web UI
streamlit run client/app.py
```

### Docker
```bash
docker-compose up
# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## Selected References

1. Peterson et al., *Energy Environ. Sci.* 3, 1311 (2010) — CO₂RR overpotentials on Cu
2. Norskov et al., *J. Electrochem. Soc.* 152, J23 (2005) — Computational Hydrogen Electrode
3. Skulason et al., *PCCP* 14, 1235 (2012) — HER on transition metals
4. Man et al., *ChemCatChem* 3, 1159 (2011) — OER scaling relations
5. Montoya et al., *ChemSusChem* 8, 2180 (2015) — NRR on transition metals
6. Behler & Parrinello, *PRL* 98, 146401 (2007) — Neural network potentials
7. Vandermause et al., *npj Comput. Mater.* 6, 20 (2020) — Active learning for NNPs
8. Radford et al., CLIP, *ICML* 2021 — Cross-modal contrastive alignment
9. Jones et al., *J. Global Optim.* 13, 455 (1998) — Expected improvement (EGO)
10. Pulay, *Chem. Phys. Lett.* 73, 393 (1980) — DIIS SCF mixing
11. Kresse & Furthmuller, *Comput. Mater. Sci.* 6, 15 (1996) — VASP algorithms
12. Hinnemann et al., *J. Am. Chem. Soc.* 127, 5308 (2005) — MoS₂ HER

---

## License

MIT
