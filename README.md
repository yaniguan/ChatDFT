# ChatDFT

**Autonomous Reaction Pathway Discovery via LLM-Guided DFT**

> Starting from a scientific problem, not an engineering one.

---

## The Problem

Heterogeneous catalysis research faces a brutal combinatorial challenge: for a given reaction (e.g., CO₂ reduction), the number of plausible elementary steps, surface configurations, and computational parameter choices grows exponentially with system complexity. A computational chemist spends most of their time on:

1. Manually hypothesising which intermediates matter
2. Building slab models and placing adsorbates at the right sites
3. Tuning DFT parameters for each new material/calculation type
4. Diagnosing why a calculation diverged and how to fix it
5. Extracting thermodynamic quantities and assembling free energy diagrams

Each step requires deep domain knowledge and produces hours of boilerplate work. The scientific question — *why does Cu(111) favour CO₂RR over HER at −0.5 V?* — gets buried under the mechanics.

**ChatDFT attacks this at the algorithmic level, not just the workflow level.**

---

## Scientific Contributions

### 1. Multi-Scale Molecular Representation (`science/representations/`)

Predicting where an adsorbate binds on a surface requires a representation that is invariant to rigid motions, sensitive to the local chemical environment, and lightweight enough for high-throughput screening.

We developed a **Voronoi topology graph** encoding of catalytic slabs:
- Nodes carry Wigner-Seitz volume, coordination number, and surface-layer index
- Edges carry Voronoi face areas as bond-strength proxies
- Adsorption sites are classified geometrically (top, bridge, hollow-fcc, hollow-hcp) via the eigenvalue spectrum of the local coordination tensor

This representation is the input to GNN-based site prediction models and also drives the rule-based site selection in the structure agent — the same representation serves both learned and symbolic components.

```python
stg = SurfaceTopologyGraph(positions, elements, cell)
stg.build()
sites = stg.classify_adsorption_sites()   # top, bridge, hollow_fcc, hollow_hcp
X     = stg.node_feature_matrix()         # (N, 6) for GNN input
```

### 2. Physics-Informed Structure Generation (`science/generation/`)

Training a neural-network interatomic potential (NNP) requires a dataset that covers configuration space efficiently — not just near-equilibrium structures. Three strategies are implemented, each grounded in physical models:

**Einstein rattle** — temperature-scaled displacements from the quantum harmonic oscillator:

```
σᵢ = sqrt( ħ/(2mᵢω) · coth(ħω/2kBT) )
```

This correctly captures zero-point motion at low T and transitions to the classical √(kBT/mω²) limit at high T.

**Normal-mode sampling** — each phonon mode is excited with amplitude √(kBT/λν) (equipartition), producing structurally diverse configurations that densely sample the actual potential energy landscape rather than an isotropic ball around the minimum.

**Active-learning selection** — a committee of NNP models assigns uncertainty σ(x)/N_atoms to candidate structures. Only configurations above a threshold are sent to DFT, making the training loop data-efficient.

```python
sampler = EinsteinRattler(omega_THz=5.0, quantum=True)
configs = sampler.generate_batch(slab, T_K=800, n=50)

al = CommitteeUncertaintySampler(committee=[model_a, model_b, model_c])
dft_queue, cached = al.filter_above_threshold(configs)
```

### 3. Cross-Modal Hypothesis Grounding (`science/alignment/`)

An LLM generates a reaction hypothesis in natural language. DFT produces a free energy diagram. How do we score whether they are consistent?

We frame this as a **cross-modal alignment** problem: three modalities — (text hypothesis, reaction graph, free energy profile) — are encoded into a shared embedding space via independent encoders, and cross-modal cosine similarity serves as a principled confidence score.

The alignment objective is InfoNCE contrastive loss, the same framework used in CLIP:

```
L = -(1/B) Σᵢ log [ exp(sim(Tᵢ,Gᵢ)/τ) / Σⱼ exp(sim(Tᵢ,Gⱼ)/τ) ]
```

At inference, a higher sim(T, G) means the text hypothesis is structurally consistent with the proposed reaction network, and a higher sim(T, P) means it is thermodynamically consistent with the computed energetics.

```python
grounder = HypothesisGrounder()
score = grounder.score(
    hypothesis = "COOH* is the key intermediate on Cu(111) for CO2RR",
    network    = ReactionNetwork.from_dict(llm_output),
    dG_profile = [0.0, 0.22, -0.15, -0.45, -1.10],
)
breakdown = grounder.score_breakdown(hypothesis, network, dG_profile)
```

### 4. Multi-Task Thermodynamic Optimisation

A catalyst must be evaluated across competing tasks simultaneously:

- Minimise overpotential (most negative potential to make all ΔG ≤ 0)
- Maximise selectivity (suppress undesired pathways)
- Satisfy thermodynamic feasibility constraints

The free energy engine implements the **Computational Hydrogen Electrode (CHE)** framework with full ZPE + entropy corrections, building on the Nørskov group formalism. The microkinetic solver finds the coverage distribution at steady state under the Arrhenius mean-field approximation, enabling overpotential–selectivity Pareto analysis.

The DFT parameter selection is itself a multi-objective problem: minimise CPU cost O(N_pw^{3/2} · N_k) while keeping energy error below a target threshold. Convergence tests sweep (ENCUT, KPPRA) space and identify the Pareto-optimal front.

### 5. Time-Series Analysis of SCF Convergence (`science/time_series/`)

The SCF iteration history is a time series — and treating it as one unlocks predictive and diagnostic capabilities that static post-hoc analysis cannot.

**Charge sloshing detection** via spectral analysis: the FFT of log|ΔE_n| reveals dominant oscillation frequencies. A ratio of AC to DC amplitude above a threshold flags metallic sloshing *before* it wastes 40+ SCF steps, and automatically recommends the correct ALGO/AMIX fix.

**Convergence rate prediction**: fitting log|ΔE_n| = log A − λn in the early window gives a convergence rate λ and predicts the step n_conv = (log A − log EDIFF)/λ at which EDIFF will be reached. R² of the fit quantifies prediction confidence.

**Ionic-step tracking**: across a geometry relaxation, the per-ionic-step SCF count is itself a signal — sudden spikes indicate difficult PES regions, oscillating counts indicate POTIM is too large, monotone decrease indicates healthy convergence toward a minimum.

```python
traj   = SCFTrajectory.from_outcar_text(outcar_text)
report = analyse_scf(traj, is_metal=True)
print(report)
# SCF Analysis (47 steps, status: CONVERGED)
#   Sloshing          : no
#   Convergence rate  : λ = 0.421 steps⁻¹  (predicted n_conv = 52, R²=0.94)
#   Recommendation    : ALGO=Fast; ISMEAR=1; SIGMA=0.2
```

---

## System Architecture

The scientific algorithms above are integrated into an end-to-end research assistant:

```
Natural language query
        │
        ▼
   Intent Agent          ← parse substrate, facet, pH, reaction type
        │
        ▼
 Hypothesis Agent        ← LLM + RAG literature + cross-modal grounder
   (reaction network)
        │
        ▼
    Plan Agent           ← generate DFT task graph
        │
   ┌────┴────┐
   ▼         ▼
Structure  Parameters    ← surface graph topology + Pareto parameter search
  Agent      Agent
   └────┬────┘
        ▼
   HPC Execution         ← SSH job submission (PBS/SGE/SLURM)
        │
        ▼
  Post-Analysis          ← SCF time-series + CHE thermodynamics
        │
        ▼
  Free Energy Diagram + Overpotential + Rate-Determining Step
```

**Stack**: Python · FastAPI · PostgreSQL + pgvector · ASE · VASP · Streamlit

---

## Key Algorithms at a Glance

| Module | Algorithm | Scientific Basis |
|---|---|---|
| `surface_graph.py` | Voronoi topology graph | Wigner-Seitz cells, coordination tensors |
| `informed_sampler.py` | Normal-mode & Einstein rattle | Phonon equipartition, QHO |
| `informed_sampler.py` | Committee uncertainty sampling | Active learning, variance estimation |
| `hypothesis_grounder.py` | InfoNCE cross-modal alignment | Contrastive learning (CLIP-style) |
| `scf_convergence.py` | FFT sloshing detection | Lindhard dielectric, spectral analysis |
| `scf_convergence.py` | Exponential convergence predictor | OLS on SCF log-residual trajectory |
| `thermo_utils.py` | CHE free energy engine | Nørskov/Peterson electrode model |
| `thermo_utils.py` | Microkinetic solver | Arrhenius mean-field steady state |
| `outcar_debugger.py` | Convergence failure diagnosis | 12 VASP failure modes + fixes |
| `parameters_agent.py` | KPPRA / kdensity k-grids | Monkhorst-Pack reciprocal-space sampling |
| `hypothesis_agent.py` | Structured mechanism validation | Stoichiometric + graph completeness checks |

---

## How to Run

**Requirements**: Python 3.10+, PostgreSQL 14+ with pgvector, VASP license (for actual DFT runs)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the database
# (set DATABASE_URL in .env)
alembic upgrade head

# 3. Start the API server
uvicorn server.main:app --reload --port 8000

# 4. Start the Streamlit UI
streamlit run client/app.py
```

**Try the scientific algorithms standalone** (no VASP or database needed):

```python
from science.representations.surface_graph import SurfaceTopologyGraph
from science.generation.informed_sampler   import EinsteinRattler
from science.alignment.hypothesis_grounder import HypothesisGrounder, ReactionNetwork
from science.time_series.scf_convergence   import SCFTrajectory, analyse_scf
```

---

## Selected Literature Foundations

- Nørskov et al., *J. Electrochem. Soc.* 152, J23 (2005) — CHE framework
- Peterson et al., *Energy Environ. Sci.* 3, 1311 (2010) — CO₂RR overpotential
- Behler & Parrinello, *PRL* 98, 146401 (2007) — NNP training
- Vandermause et al., *npj Comput. Mater.* 6, 20 (2020) — on-the-fly active learning
- Radford et al. (OpenAI), CLIP, *ICML* 2021 — cross-modal contrastive alignment
- Pulay, *Chem. Phys. Lett.* 73, 393 (1980) — DIIS SCF mixing
- Kresse & Furthmüller, *Comput. Mater. Sci.* 6, 15 (1996) — VASP algorithms
