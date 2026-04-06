# ChatDFT

**Autonomous Reaction Pathway Discovery via LLM-Guided DFT**

[![CI](https://github.com/yaniguan/ChatDFT/actions/workflows/ci.yml/badge.svg)](https://github.com/yaniguan/ChatDFT/actions/workflows/ci.yml)
[![Deploy](https://github.com/yaniguan/ChatDFT/actions/workflows/deploy.yml/badge.svg)](https://github.com/yaniguan/ChatDFT/actions/workflows/deploy.yml)
[![Tests](https://img.shields.io/badge/tests-129%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## What is ChatDFT?

ChatDFT is an end-to-end research assistant that turns a **natural language query** into **production DFT calculations** on HPC clusters. Describe your reaction system in plain English, and ChatDFT handles everything: hypothesis generation, structure building, VASP input preparation, HPC job submission, result parsing, and thermodynamic analysis.

```
"H adsorption on Pt/Pd/Ni(111) surface"
        |
        v
  +------------------+     +-------------------+     +------------------+
  |   Intent Agent   | --> | Hypothesis Agent  | --> |   Plan Agent     |
  | Parse: metals,   |     | LLM + RAG + cross |     | Generate DFT     |
  | facet, adsorbate  |     | modal grounding   |     | task graph       |
  +------------------+     +-------------------+     +------------------+
                                                              |
                    +--------------------+--------------------+
                    |                    |                    |
                    v                    v                    v
            +-----------+        +-----------+        +-----------+
            | Structure |        | Parameter |        |    HPC    |
            |   Agent   |        |   Agent   |        | Execution |
            | ASE+pymat |        | BO search |        | PBS/SLURM |
            +-----------+        +-----------+        +-----------+
                    |                    |                    |
                    +--------------------+--------------------+
                                         |
                                         v
                              +---------------------+
                              |   Post-Analysis     |
                              | SCF diagnostics,    |
                              | E_ads, free energy  |
                              | diagram, Excel      |
                              +---------------------+
```

---

## Key Features

### Chat-Driven DFT Pipeline
Type a query, get results. The LLM agent chain parses your intent, generates reaction mechanisms, builds structures, and submits calculations.

### Batch HPC Submission
Submit systematic studies (e.g., H adsorption on 3 metals x 3 sites = 9+4 reference calculations) to PBS/SLURM clusters in one click. Live status monitoring and automatic result collection.

### Cloud Deployment (AWS EKS)
Production-ready Kubernetes deployment on AWS with Terraform IaC, auto-deploy CI/CD via GitHub Actions, managed PostgreSQL (RDS), and container registry (ECR).

### Six Novel Algorithms

| # | Algorithm | Key Result |
|---|---|---|
| 1 | **Voronoi topology graph** for surface representation | 4-class site classification with symmetry scoring |
| 2 | **Quantum harmonic oscillator rattle** for structure generation | Mass-weighted ZPE-aware sampling (sigma_H/sigma_Pt = 13.9x) |
| 3 | **InfoNCE cross-modal alignment** for hypothesis grounding | AUC = 1.00 vs 0.43 keyword baseline |
| 4 | **FFT charge sloshing detection** for SCF diagnostics | 100% accuracy vs 50% baseline |
| 5 | **Bayesian optimisation** for DFT parameter search | 73% fewer DFT evaluations vs grid search |
| 6 | **GNN energy prediction** (5 architectures) | MPNN/GAT/SchNet/DimeNet/SE(3)-Transformer |

---

## Quick Start

### Demo (no database, no API key needed)

```bash
pip install -e .
python demo.py                    # Run all algorithms
python demo.py --benchmark        # Generate publication figures
make test                         # Run 129 tests
```

### Full System

```bash
# 1. Start database
docker-compose up -d db
alembic upgrade head

# 2. Start API server
export OPENAI_API_KEY=sk-...
uvicorn server.main:app --reload --port 8000

# 3. Start Web UI
streamlit run client/app.py
```

### Docker Compose

```bash
docker-compose up
# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## Production Deployment (AWS)

ChatDFT deploys to AWS EKS with Terraform and auto-deploys via GitHub Actions on push to `main`.

### Architecture

```
Internet --> ALB (Application Load Balancer)
                |-- /        --> Streamlit UI (port 8501)
                |-- /v1/*    --> FastAPI Server (port 8000)
                                    |
                               RDS PostgreSQL 16 + pgvector
```

### Deploy

```bash
# 1. Configure AWS
aws configure
cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars

# 2. Deploy infrastructure + application
bash infra/scripts/deploy.sh

# 3. CI/CD: auto-deploys on git push to main
git push origin main  # triggers build -> push -> deploy
```

### Infrastructure

| Component | Service | Spec |
|-----------|---------|------|
| Kubernetes | Amazon EKS | Managed control plane + t3.small nodes |
| Database | Amazon RDS | PostgreSQL 16 + pgvector extension |
| Registry | Amazon ECR | Docker images with lifecycle policy |
| Load Balancer | AWS ALB | Path-based routing (API vs UI) |
| IaC | Terraform | VPC, subnets, security groups, IAM |
| CI/CD | GitHub Actions | Lint + test + build + deploy on push |

---

## Batch Adsorption Study (MVP)

The flagship use case: systematic adsorption energy calculations across metals and sites.

### API

```bash
# Submit batch
curl -X POST http://localhost:8000/api/batch_adsorption \
  -H "Content-Type: application/json" \
  -d '{
    "adsorbate": "H",
    "metals": ["Pt", "Pd", "Ni"],
    "facet": "111",
    "sites": ["ontop", "bridge", "hollow"],
    "server_name": "hoffman2"
  }'

# Check status
curl "http://localhost:8000/api/batch_status?batch_uid=<uid>"

# Download Excel results
curl -o results.xlsx "http://localhost:8000/api/batch_results_excel?batch_uid=<uid>"
```

### Streamlit UI

The batch study is integrated into the chat interface. Type a query like "H adsorption on Pt/Pd/Ni 111" and the HPC submission panel appears automatically with:
- Metal/adsorbate/site selection
- VASP parameter configuration
- One-click submission to PBS/SLURM queue
- Live job status monitoring
- Results table with E_ads calculation
- Excel export with adsorption energies

---

## System Architecture

```
chatdft/
|-- science/                    # Novel algorithms (numpy/scipy/torch)
|   |-- representations/        # Voronoi topology graph
|   |-- generation/             # Einstein rattle, normal-mode sampling
|   |-- alignment/              # InfoNCE cross-modal hypothesis grounding
|   |-- time_series/            # FFT sloshing detection, SCF analysis
|   |-- optimization/           # Bayesian parameter search (GP + EI)
|   |-- predictions/            # GNN models (MPNN, GAT, SchNet, DimeNet, SE(3))
|   |-- evaluation/             # Golden dataset (25 reactions), metrics
|   +-- benchmarks/             # Baseline comparisons, figure generation
|
|-- server/                     # FastAPI backend
|   |-- chat/                   # LLM agents (intent, hypothesis, plan, knowledge)
|   |-- execution/              # Structure building, VASP params, HPC submission
|   |-- api/                    # One-click API, batch adsorption, preprocessor
|   |-- mlops/                  # Model registry, experiment tracker
|   +-- feature_store/          # Feature lineage + drift detection
|
|-- client/                     # Streamlit UI
|   |-- app.py                  # Main app (chat pipeline, structure lab, thermo)
|   +-- pages/                  # Batch adsorption page
|
|-- infra/                      # AWS deployment
|   |-- terraform/              # EKS, RDS, ECR, VPC, IAM
|   |-- k8s/                    # Kubernetes manifests
|   +-- scripts/                # Deploy, build-push, setup-cluster
|
+-- tests/                      # 129 tests (science + ML + server)
```

**Stack**: Python 3.10+ | FastAPI | PostgreSQL + pgvector | ASE + VASP | Streamlit | AWS EKS | Terraform | GitHub Actions

---

## Benchmark Results

All results generated by `python demo.py --benchmark`. 36 figures in `figures/`.

| Algorithm | ChatDFT | Baseline | Metric |
|---|---|---|---|
| Surface site classification | 4-class + symmetry | 3-class, no ranking | site granularity |
| Structure generation | Mass-weighted with ZPE | Fixed noise | physical correctness |
| Hypothesis grounding | AUC = 1.00 | AUC = 0.43 | 30 pairs |
| SCF sloshing detection | 100% accuracy | 50% accuracy | 60 trajectories |
| SCF step prediction | MAE = 3.6 steps | MAE = 7.4 | 30 trajectories |
| Parameter search | 15 evaluations | 56 evaluations | same error target |
| GNN energy prediction | SE(3)/DimeNet best | MLP baseline | test MAE (eV) |

### Golden Benchmark Dataset

25 reactions across 5 domains with literature-validated free energy profiles:

| Domain | Reactions | Source |
|---|---|---|
| CO2RR | 8 | Peterson 2010, Kuhl 2012 |
| HER | 5 | Skulason 2012, Hinnemann 2005 |
| OER | 5 | Man 2011, Rossmeisl 2007 |
| NRR | 4 | Montoya 2015, Shi 2014 |
| ORR | 3 | Norskov 2004 |

---

## CI/CD Pipeline

Every push to `main` triggers:

| Job | What it does |
|-----|-------------|
| **Ruff lint** | Code quality + import sorting |
| **Ruff format** | Style consistency |
| **MyPy** | Type checking (non-blocking) |
| **Tests** | 129 tests on Python 3.10/3.11/3.12 |
| **Security** | Bandit scan + pip audit |
| **Docker build** | Verify container builds |
| **Benchmark regression** | Ensure algorithms don't regress |
| **Deploy to EKS** | Auto-deploy when server/client changes |

---

## References

1. Peterson et al., *Energy Environ. Sci.* 3, 1311 (2010)
2. Norskov et al., *J. Electrochem. Soc.* 152, J23 (2005)
3. Skulason et al., *PCCP* 14, 1235 (2012)
4. Man et al., *ChemCatChem* 3, 1159 (2011)
5. Montoya et al., *ChemSusChem* 8, 2180 (2015)
6. Behler & Parrinello, *PRL* 98, 146401 (2007)
7. Radford et al., CLIP, *ICML* 2021
8. Jones et al., *J. Global Optim.* 13, 455 (1998)
9. Kresse & Furthmuller, *Comput. Mater. Sci.* 6, 15 (1996)

---

## License

MIT
