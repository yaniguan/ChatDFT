# ChatDFT System Pipeline

Complete technical specification — every stage, every data format, every tech decision.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT  (Streamlit)                            │
│  app.py (118K)  ←→  utils/api.py  ←→  HTTP JSON  ←→  FastAPI :8000   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────┐
│                      SERVER  (FastAPI + asyncpg)                        │
│                                                                         │
│  ┌──────────┐  ┌───────────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Chat    │  │  Execution    │  │  Science │  │  Utils           │  │
│  │  Layer   │  │  Layer        │  │  Module  │  │                  │  │
│  │          │  │               │  │          │  │  openai_wrapper  │  │
│  │ intent   │  │ structure     │  │ surface  │  │  rag_utils       │  │
│  │ hypothe- │  │ parameters    │  │ graph    │  │  outcar_debugger │  │
│  │ sis      │  │ ase_scripts   │  │ sampler  │  │  thermo_utils    │  │
│  │ plan     │  │ hpc_agent     │  │ grounder │  │  perplexity      │  │
│  │ qa       │  │ htp_agent     │  │ scf_conv │  │  zotero          │  │
│  │ analyze  │  │ post_analysis │  │          │  │                  │  │
│  │ knowledge│  │ job_watcher   │  │          │  │                  │  │
│  │ history  │  │ engines/      │  │          │  │                  │  │
│  └──────────┘  └───────┬───────┘  └──────────┘  └──────────────────┘  │
│                         │                                               │
└─────────────────────────┼───────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌────────────┐  ┌────────────┐  ┌────────────┐
   │ PostgreSQL │  │ HPC Cluster│  │ OpenAI API │
   │ + pgvector │  │ (SSH/rsync)│  │ GPT-4o     │
   │ 29 tables  │  │ PBS/SLURM  │  │ Embeddings │
   └────────────┘  └────────────┘  └────────────┘
```

---

## Tech Stack Summary

| Layer | Technology | Version / Detail |
|-------|-----------|------------------|
| **Frontend** | Streamlit | Python-based reactive UI, `st.session_state` for client state |
| **API** | FastAPI | Async, Pydantic v2 validation, CORS allow_origins=["*"] |
| **ORM** | SQLAlchemy 2.x | Async via `asyncpg`, declarative ORM, 29 tables |
| **Database** | PostgreSQL 14+ | `postgresql+asyncpg://yaniguan@localhost:5432/chatdft_ase` |
| **Vector Store** | pgvector | 1536-dim embeddings, cosine distance (`<=>` operator), JSON fallback |
| **Migrations** | Alembic | 4 migration versions, async-compatible |
| **LLM** | OpenAI GPT-4o | `temperature=0.1`, `json_mode=True`, 3x exponential-backoff retry |
| **Embeddings** | text-embedding-3-small | 1536 dimensions, in-memory MD5 cache |
| **Web Search** | Perplexity Sonar | 128K context, citation-backed, DFT-specific queries |
| **Literature** | Zotero API + arXiv | Daily auto-ingest (12 chemistry queries), PDF figure extraction via GPT-4o vision |
| **DFT Engine** | VASP (primary) | 22 calc profiles, QE stub, CP2K stub, factory pattern |
| **Atomistic** | ASE + pymatgen | Structure I/O, site finders, phonon utilities |
| **HPC** | SSH + rsync | PBS (Hoffman2/SGE), SLURM (Beluga, Cedar), status polling |
| **Serialisation** | POSCAR (text) | Universal exchange format between all agents |

---

## Stage 0: Session Initialisation

```
User opens Streamlit UI
        │
        ▼
POST /chat/session/create
  Request:  { name: "CO2RR on Cu(111)", project: "electrocatalysis" }
  Response: { ok: true, id: 42, uid: "s_abc123" }
        │
        ▼
DB Write: chat_session row
  Columns: id=42, uid="s_abc123", name="CO2RR on Cu(111)",
           project="electrocatalysis", status="active"
```

**Tech**: Streamlit `st.session_state["session_id"]` stores the active session ID for all subsequent calls.

---

## Stage 1: Intent Parsing

```
User types: "Study CO2 reduction to CO on Cu(111) at -0.5V vs RHE in 0.1M KHCO3"
        │
        ▼
POST /chat/intent
```

### Input
```json
{
  "session_id": 42,
  "text": "Study CO2 reduction to CO on Cu(111) at -0.5V vs RHE in 0.1M KHCO3",
  "guided": {}
}
```

### Processing Pipeline

```
1. Quick heuristic extraction (regex)
   ├── substrate: "Cu"
   ├── facet: "111"
   ├── potential: -0.5
   ├── electrolyte: "KHCO3"
   └── reaction: "CO2 → CO"

2. Few-shot retrieval from IntentPhrase table
   SELECT * FROM intent_phrase
   WHERE intent_stage ILIKE '%electro%'
     AND intent_area  ILIKE '%catalysis%'
   ORDER BY confidence DESC LIMIT 5
   → Returns 5 similar past intents for in-context learning

3. RAG context retrieval
   rag_utils.rag_context(query, session_id=42, stage="intent", top_k=5)
   ├── embed_text(query) → 1536-dim vector
   ├── semantic_search(vector, top_k=5) → pgvector cosine
   ├── keyword_search(query, top_k=5) → SQL ILIKE
   └── hybrid_search via RRF fusion:
       score = 0.6/(k+rank_semantic) + 0.4/(k+rank_keyword)

4. LLM call (GPT-4o, json_mode=True, temperature=0.1)
   System: "You are a computational chemistry intent parser..."
   User: query + few-shots + RAG context
   → Structured JSON output

5. Electronic calc type detection
   Regex scan for: "dos", "pdos", "band", "elf", "bader",
                   "cdd", "work function", "cohp"
```

### Output
```json
{
  "ok": true,
  "intent": {
    "stage": "electrocatalysis",
    "area": "heterogeneous_catalysis",
    "task": "Investigate CO2 reduction to CO on Cu(111)",
    "system": {
      "catalyst": "Cu",
      "facet": "111",
      "material": "Cu",
      "molecule": "CO2"
    },
    "conditions": {
      "pH": 6.8,
      "potential": -0.5,
      "potential_ref": "RHE",
      "electrolyte": "0.1M KHCO3",
      "temperature": 298.15
    },
    "reaction_network": {
      "steps": ["CO2(g) + * + H+ + e- → COOH*", "COOH* + H+ + e- → CO* + H2O(g)"],
      "intermediates": ["*", "CO2(g)", "COOH*", "CO*", "H2O(g)"],
      "ts": ["CO2→COOH", "COOH→CO"],
      "coads": []
    },
    "electronic_calcs": [],
    "tags": ["CO2RR", "Cu", "electrocatalysis"],
    "deliverables": ["adsorption energies", "free energy diagram", "overpotential"],
    "confidence": 0.87
  },
  "message_id": 101,
  "hypothesis_id": null,
  "fewshots_used": 3,
  "rag_refs": ["arxiv:2301.12345", "doi:10.1021/jacs.2023"]
}
```

### DB Writes
```
chat_message: id=101, session_id=42, role="assistant", msg_type="intent",
              content=JSON(intent), confidence=0.87
agent_log:    agent_name="intent_agent", model="gpt-4o",
              input_tokens=1240, output_tokens=380, latency_ms=2100
```

---

## Stage 2: Hypothesis Generation

```
POST /chat/hypothesis
```

### Input
```json
{
  "session_id": 42,
  "intent": { /* full intent from Stage 1 */ }
}
```

### Processing Pipeline

```
1. Check mechanism cache
   SELECT * FROM reaction_system
   WHERE system_hash = SHA256("electrochemical|Cu(111)|CO2|CO")
   → Cache MISS (first time)

2. Template selection
   Domain "electrochemical" + reactant "CO2" + product "CO"
   → Template: "electrochemical_reduction"
   → Bond events: C-O cleavage, protonation
   → Key species: CO2*, COOH*, CO*, H2O(g)
   → DFT notes: VASPsol solvation, CHE correction

3. RAG literature fetch
   query: "electrochemical CO2 reduction CO Cu(111) mechanism DFT"
   → top_k=8 chunks with title/year/relevance

4. LLM call #1: Markdown hypothesis (GPT-4o)
   → Natural language description of proposed mechanism

5. LLM call #2: Structured graph (GPT-4o, json_mode)
   System prompt includes:
   - Forward-direction enforcement: "First step MUST start with CO2(g)"
   - Species notation rules: "* for surface, (g) for gas"
   - Template bond events and key species
   - RAG literature context
   → JSON reaction network

6. Graph validation & auto-repair
   _validate_graph(graph, intent):
   ├── Canonicalise species (* suffix for surface)
   ├── Check all intermediates appear in at least one step
   ├── Verify stoichiometric balance (atom count lhs == rhs + surface)
   ├── Deduplicate coadsorbate pairs
   └── Ensure reactant/product in intermediates list

7. Confidence scoring (4 dimensions)
   ├── Specificity (0–0.30): material+facet+reactant+product all present
   ├── Literature (0–0.25): RAG ref count × mean relevance
   ├── Completeness (0–0.30): n_intermediates, n_steps, has_ts
   └── Known family (0–0.15): template match in registry

8. Cache to DB
   INSERT reaction_system (domain, surface, reactant, product, system_hash)
   INSERT mechanism_graph (system_id, name, intermediates, steps, coads, ts)
```

### Output
```json
{
  "ok": true,
  "result_md": "## Hypothesis: CO₂RR via Carboxyl Pathway on Cu(111)\n\n...",
  "fields": {
    "conditions": "pH=6.8, U=-0.5V vs RHE, 0.1M KHCO3",
    "rationale": "Cu(111) favours carboxyl pathway...",
    "elementary_steps": ["CO2(g) + * + H+ + e- → COOH*", ...],
    "next_dft_tasks": ["Optimize COOH* on Cu(111)", ...]
  },
  "graph": {
    "reaction_network": [
      {"lhs": ["CO2(g)", "*", "H+", "e-"], "rhs": ["COOH*"]},
      {"lhs": ["COOH*", "H+", "e-"], "rhs": ["CO*", "H2O(g)"]}
    ],
    "intermediates": ["*", "CO2(g)", "COOH*", "CO*", "H2O(g)", "CO(g)"],
    "coads_pairs": [["CO*", "H*"]],
    "ts_edges": [["CO2*", "COOH*"], ["COOH*", "CO*"]]
  },
  "confidence": 0.78
}
```

### DB Writes
```
hypothesis:      id=5, session_id=42, hypothesis=JSON, confidence=0.78
chat_message:    id=102, msg_type="hypothesis"
reaction_system: system_hash="a1b2c3..."
mechanism_graph: system_id=1, intermediates=[...], steps=[...]
agent_log:       2 entries (markdown + graph LLM calls)
```

---

## Stage 3: Plan Generation

```
POST /chat/plan
```

### Input
```json
{
  "session_id": 42,
  "intent": { /* Stage 1 */ },
  "hypothesis": { /* Stage 2 fields */ },
  "graph": { /* Stage 2 graph */ },
  "use_seed_policy": true,
  "conf_threshold": 0.55,
  "limits": { "intermediates": 40, "coads": 80, "ts": 40 }
}
```

### Processing Pipeline

```
1. Mechanism key matching (regex)
   "CO2 reduction" + "CO" → ["co2rr_co_path", "co2rr_carboxyl"]

2. Seed extraction (if confidence < 0.55)
   Load registry template for "co2rr_carboxyl":
   ├── steps: [CO2→COOH*, COOH*→CO*+H2O]
   ├── intermediates: [*, CO2(g), COOH*, CO*, CO(g), H2O(g)]
   └── ts: [CO2→COOH, COOH→CO]
   Light-merge with LLM hypothesis graph (union, dedup)

3. RAG context (stage="plan")
   ├── Literature chunks (top_k=5)
   ├── Past plan context (similar sessions)
   └── History context (this session's messages)

4. LLM call (GPT-4o, json_mode)
   System: "You are a DFT workflow planner..."
   Input: intent + hypothesis + seed + RAG + limits
   → Structured task list

5. Task graph construction
   For each intermediate in mechanism:
   ├── Task: Build structure (surface + adsorbate)
   ├── Task: Geometry optimization (VASP relax)
   ├── Task: Frequency calculation (if ZPE needed)
   └── Task: Electronic structure (if requested)
   For each TS edge:
   └── Task: CI-NEB between adjacent intermediates
   Final:
   └── Task: Post-analysis (free energy diagram)
```

### Output
```json
{
  "ok": true,
  "tasks": [
    {
      "id": 1, "name": "Build Cu(111) slab",
      "agent": "structure.build_slab",
      "params": {"element": "Cu", "miller": "111", "layers": 4, "vacuum": 15},
      "depends_on": []
    },
    {
      "id": 2, "name": "Place COOH on Cu(111)",
      "agent": "structure.place_adsorbate",
      "params": {"adsorbate": "COOH", "site": "top"},
      "depends_on": [1]
    },
    {
      "id": 3, "name": "Relax COOH/Cu(111)",
      "agent": "structure.relax_adsorbate",
      "params": {"engine": "vasp", "profile": "relax_slab"},
      "depends_on": [2]
    },
    {
      "id": 4, "name": "Frequency COOH/Cu(111)",
      "agent": "electronic.freq",
      "params": {"engine": "vasp", "profile": "freq"},
      "depends_on": [3]
    },
    {
      "id": 5, "name": "CI-NEB CO2→COOH",
      "agent": "neb.ci_neb",
      "params": {"images": 7, "engine": "vasp", "profile": "cineb"},
      "depends_on": [3]
    },
    {
      "id": 6, "name": "Free energy analysis",
      "agent": "post.analysis",
      "params": {"reaction": "CO2RR", "T_K": 298.15, "U_V_RHE": -0.5},
      "depends_on": [3, 4, 5]
    }
  ],
  "workflow": {
    "parallel_groups": {
      "1": [3, 4],
      "2": [5]
    }
  },
  "summary": "6 tasks: 2 structure builds, 1 relax, 1 freq, 1 CI-NEB, 1 analysis",
  "confidence": 0.82
}
```

### DB Writes
```
chat_message:  id=103, msg_type="plan", content=JSON(tasks)
workflow_task:  6 rows (one per task), status="idle", depends_on=[...]
agent_log:     1 entry
```

---

## Stage 4: Structure Building

```
POST /structure/build_slab    (or via /agent/structure.build_slab)
POST /structure/find_sites
POST /structure/place_adsorbate
```

### 4a. Slab Construction

```
Input:  { element: "Cu", miller: "111", layers: 4, vacuum: 15.0 }

Processing:
  1. _normalize_element("Cu") → "Cu"
  2. _parse_miller("111") → (1, 1, 1)
  3. _guess_crystal_system("Cu") → "fcc"
  4. Lattice parameter lookup: Cu → a = 3.615 Å
  5. ASE fcc111(symbol="Cu", size=(3,3,4), vacuum=15.0, a=3.615)
  6. Fix bottom 2 layers: FixAtoms(indices=[...])
  7. Write POSCAR via ase.io.write()
  8. Generate PNG preview via matplotlib (top + side view)

Output: {
  ok: true,
  poscar: "Cu(111) slab\n1.0\n  7.668  0.000  0.000\n...",
  plot_png_b64: "iVBORw0KGgo...",
  ase_code: "from ase.build import fcc111\natoms = fcc111('Cu', ...)",
  viz: { atoms: [...], cell: [...], bonds: [...] }
}

DB: structure_library row (type="surface", label="Cu(111)-3x3x4")
```

### 4b. Adsorption Site Finding

```
Input: { poscar_content: "Cu(111) slab\n...", height: 1.8 }

Processing:
  1. Parse POSCAR → ASE Atoms
  2. Convert to pymatgen Structure
  3. AdsorbateSiteFinder(structure).find_adsorption_sites()
     → top, bridge, hollow_fcc, hollow_hcp positions
  4. Deduplicate by distance (tol=0.3 Å)
  5. Sort by site type

Output: {
  ok: true,
  sites: [
    { type: "top",        coord: [1.277, 0.737, 17.8], height: 1.8 },
    { type: "bridge",     coord: [2.555, 1.475, 17.8], height: 1.8 },
    { type: "hollow_fcc", coord: [1.277, 2.212, 17.8], height: 1.8 },
    { type: "hollow_hcp", coord: [2.555, 2.949, 17.8], height: 1.8 }
  ]
}
```

### 4c. Adsorbate Placement

```
Input: {
  poscar_content: "Cu(111) slab\n...",
  adsorbate: "COOH",
  site: { type: "top", coord: [1.277, 0.737, 17.8] },
  height: 1.8
}

Processing:
  1. Lookup adsorbate geometry from library:
     COOH = Atoms("COOH", positions=[[0,0,0],[1.25,0,0],[1.9,1.1,0],[2.8,0.8,0.5]])
  2. Place adsorbate at site position + height offset
  3. No overlap check (inter-atomic distance > 0.8 Å)
  4. Write combined POSCAR

Output: {
  ok: true,
  poscar: "COOH on Cu(111)\n1.0\n...",
  plot_png_b64: "...",
  ase_code: "..."
}
```

---

## Stage 5: Parameter Generation

```
Triggered by: _run_parameters(task, job_dir) inside _pipeline()
```

### Processing

```
Input: task = {
  params: {
    payload: {
      engine: "vasp",
      profile: "relax_slab",
      overrides: { ENCUT: 520 }
    }
  }
}

1. Load calc profile from calc_profiles.yaml:
   relax_slab → {
     PREC: "Normal", ENCUT: 400, EDIFF: 1e-5, ALGO: "Fast",
     ISMEAR: 1, SIGMA: 0.2, GGA: "PE", IBRION: 2, NSW: 200,
     ISIF: 2, EDIFFG: -0.03, NELM: 200
   }

2. Apply overrides: ENCUT: 400 → 520

3. Detect structure in job_dir (POSCAR/slab.POSCAR/ads.POSCAR)
   → pymatgen Structure → reciprocal lattice vectors

4. K-point generation:
   _monkhorst_from_kppra(struct, kppra=1600)
   a*, b*, c* = reciprocal lengths (Å⁻¹)
   k_i = max(1, round(kppra^(1/3) * a*_i / a*_mean))
   For slab (c* >> a*, b*): k_z = 1 (Gamma in z)
   → KPOINTS: "4 4 1"

5. Write files:
   INCAR:   tag = value (one per line, VASP format)
   KPOINTS: Automatic\n0\nMonkhorst-Pack\n4 4 1\n0 0 0
```

### Output Files
```
job_dir/
├── INCAR
│   SYSTEM = ChatDFT
│   PREC = Normal
│   ENCUT = 520
│   EDIFF = 1.0e-05
│   ALGO = Fast
│   ISMEAR = 1
│   SIGMA = 0.2
│   GGA = PE
│   IBRION = 2
│   NSW = 200
│   ISIF = 2
│   EDIFFG = -0.03
│   NELM = 200
│   LREAL = Auto
│   NCORE = 4
│   LWAVE = .FALSE.
│   LCHARG = .FALSE.
│
└── KPOINTS
    Automatic mesh
    0
    Monkhorst-Pack
    4 4 1
    0 0 0
```

---

## Stage 6: HPC Submission & Monitoring

```
Triggered by: _run_hpc(task, job_dir, cluster, ...) inside _pipeline()
```

### 6a. Script Preparation

```
HPCAgent(cluster="hoffman2")
  → Load from cluster_config.yaml:
    host: yaniguan@hoffman2.idre.ucla.edu
    scheduler: sge
    remote_base: /u/scratch/y/yaniguan/chatdft
    ntasks: 32
    vasp_bin: ~/vasp_std_vtst_sol

  → prepare_script(step_ctx, job_dir):
    Template (SGE):
      #!/bin/bash
      #$ -N co2rr_cooh_relax
      #$ -pe dc* 32
      #$ -l h_data=4G,h_vmem=16G,h_rt=24:00:00
      #$ -cwd -j y -V
      . /u/local/Modules/default/init/modules.sh
      module load intel/2020.4 intelmpi/2020.4
      mpirun -np $NSLOTS ~/vasp_std_vtst_sol > vasp.out 2>&1
```

### 6b. File Transfer & Submission

```
1. rsync UP
   rsync -avz job_dir/ yaniguan@hoffman2:~/scratch/chatdft/s42/cooh_relax/
   Files: POSCAR, INCAR, KPOINTS, POTCAR, job.sh

2. Submit
   ssh hoffman2 "cd ~/scratch/chatdft/s42/cooh_relax && qsub job.sh"
   → Parse output: "Your job 1234567 ..."
   → job_id = "1234567"

3. Write _remote.json
   { cluster: "hoffman2", job_id: "1234567", remote_dir: "...", submitted_at: "..." }
```

### 6c. Job Monitoring (job_watcher.py)

```
watch_job(task_id=3, job_id="1234567", ...)

Loop:
  sleep(poll_interval=60)
  ssh hoffman2 "qstat 1234567 2>/dev/null || echo DONE"
  → Parse: Q=queued, R=running, C=done, E=exiting

  Status transitions:
  ┌──────┐  qstat=Q  ┌─────────┐  qstat=R  ┌─────────┐  qstat=C  ┌──────┐
  │queued├──────────►│ pending ├──────────►│ running ├──────────►│ done │
  └──────┘           └─────────┘           └─────────┘           └──┬───┘
                                                                     │
                                                                     ▼
                                                              fetch + analyse

When done:
  1. rsync DOWN: OUTCAR, vasprun.xml, OSZICAR, CONTCAR → local job_dir/
  2. PostAnalysisAgent().analyze(job_dir)
  3. Insert DFTResult rows
  4. Emit task status: "done"
```

---

## Stage 7: Post-Analysis

```
Triggered by: job_watcher after fetch, or manually via POST /agent/post.analysis
```

### Processing

```
PostAnalysisAgent().analyze(job_dir)

1. Engine detection
   VaspAdapter.detect(job_dir):
     Check for: INCAR, OUTCAR, vasprun.xml
     → True

2. Calc type classification
   VaspAdapter.calc_type(job_dir):
     dirname contains "neb" → "neb"
     INCAR has LORBIT=11 + ICHARG=11 → "dos"
     INCAR has IBRION=5 → "freq"
     else → "relax" or "scf"

3. Result parsing
   VaspAdapter.parse_job(job_dir):
     _pmg_parse(job_dir) using pymatgen Vasprun + Outcar:
     ├── E_total = -245.67 eV
     ├── E_fermi = -4.32 eV
     ├── bandgap = 0.0 eV (metal)
     ├── spin_mag = 0.0 μ_B
     └── notes: ["converged in 47 SCF steps"]

4. Adsorption energy calculation
   _calc_adsorption_energies(records):
     E_ads = E(slab+COOH) - E(slab) - E(COOH_gas)
     = -245.67 - (-220.12) - (-26.33) = 0.78 eV

5. Convergence diagnostics
   outcar_debugger.debug_job(job_dir):
     ├── converged: True
     ├── elec_conv: True
     ├── issues: []
     └── incar_patch: {}

6. Write results.json to job_dir
```

### Output
```json
{
  "ok": true,
  "jobs": [
    {
      "role": "ads_COOH",
      "calc_type": "relax",
      "E_eV": -245.67,
      "fermi_eV": -4.32,
      "bandgap_eV": 0.0,
      "E_ads_eV": 0.78,
      "converged": true,
      "warnings": []
    }
  ],
  "summary_md": "COOH adsorption on Cu(111): E_ads = 0.78 eV..."
}
```

### DB Writes
```
dft_result: result_type="adsorption_energy", species="COOH",
            surface="Cu(111)", site="top", value=0.78, unit="eV",
            converged=true
```

---

## Stage 8: Thermodynamic Analysis

```
POST /chat/qa/free_energy
```

### Input
```json
{
  "session_id": 42,
  "reaction": "CO2RR",
  "T_K": 298.15,
  "U_V_RHE": -0.5,
  "use_known_pathway": false
}
```

### Processing (thermo_utils.py)

```
1. Collect DFT results from session
   SELECT * FROM dft_result WHERE session_id=42
   → E_ads(COOH) = 0.78 eV
   → E_ads(CO)   = -0.15 eV
   → E_barrier(CO2→COOH) = 0.92 eV  (from NEB)

2. Apply thermodynamic corrections for each intermediate:

   COOH*:
     ZPE = 0.45 eV  (literature default, Nørskov group)
     TS  = 0.10 eV  (at 298 K)
     G   = E_DFT + ZPE - TS = 0.78 + 0.45 - 0.10 = 1.13 eV

   CO*:
     ZPE = 0.19 eV
     TS  = 0.08 eV
     G   = -0.15 + 0.19 - 0.08 = -0.04 eV

   Gas references:
     CO2(g): ZPE=0.31, TS(298K)=0.664 → G_ref
     H2O(g): ZPE=0.56, TS(298K)=0.672 → G_ref

3. Build relative free energy profile:
   Step 0: CO2(g) + *           G_rel = 0.00 eV
   Step 1: COOH*                G_rel = 0.22 eV  (1 PCET)
   Step 2: CO* + H2O(g)         G_rel = -0.45 eV (2 PCET)
   Step 3: CO(g) + *            G_rel = -1.10 eV (desorption)

4. Apply CHE at U = -0.5 V:
   For each electrochemical step (PCET):
     G_shifted = G_rel - n_e × U
   Step 1: 0.22 - 1×(-0.5) = 0.72 eV
   Step 2: -0.45 - 2×(-0.5) = 0.55 eV

5. Identify rate-determining step:
   RDS = step with largest positive ΔG = Step 1 (ΔG = 0.72 eV)

6. Limiting potential:
   U_limiting = most negative U where all ΔG ≤ 0
   = -0.72 V vs RHE

7. Overpotential:
   η = |U_limiting - U_equilibrium|
   U_eq(CO2→CO) = -0.11 V vs RHE
   η = |-0.72 - (-0.11)| = 0.61 V

8. Generate matplotlib plot → base64 PNG
```

### Output
```json
{
  "ok": true,
  "diagrams": [
    {
      "pathway": "CO2RR_carboxyl_Cu111",
      "T_K": 298.15,
      "U_V_RHE": -0.5,
      "steps": [
        {"index": 0, "label": "CO2(g)+*",     "G": 0.00, "is_ec": false},
        {"index": 1, "label": "COOH*",         "G": 0.72, "is_ec": true, "n_e": 1},
        {"index": 2, "label": "CO*+H2O(g)",    "G": 0.55, "is_ec": true, "n_e": 1},
        {"index": 3, "label": "CO(g)+*",       "G": -1.10, "is_ec": false}
      ],
      "rds_index": 1,
      "U_limiting_V": -0.72,
      "overpotential_V": 0.61
    }
  ],
  "plot_png_b64": "iVBORw0KGgo...",
  "interpretation_md": "The carboxyl pathway on Cu(111) has an overpotential of 0.61 V..."
}
```

---

## Stage 9: Microkinetic Modelling

```
POST /chat/qa/microkinetics
```

### Processing (thermo_utils.py)

```
solve_microkinetics(step_labels, Ea_fwd_eV, delta_G_eV, T_K)

1. Rate constants (Arrhenius):
   k_fwd = (k_B T / h) × exp(-Ea_fwd / k_B T)
   k_rev = k_fwd × exp(ΔG / k_B T)

   Step 1 (CO2→COOH):
     Ea = 0.92 eV, ΔG = 0.72 eV
     k_fwd = 6.25e12 × exp(-0.92/0.0257) = 1.2e-3 s⁻¹
     k_rev = 1.2e-3 × exp(0.72/0.0257) = 8.1e9 s⁻¹

   Step 2 (COOH→CO):
     Ea = 0.45 eV, ΔG = -0.17 eV
     k_fwd = 6.25e12 × exp(-0.45/0.0257) = 1.6e5 s⁻¹
     k_rev = 1.6e5 × exp(-0.17/0.0257) = 2.1e2 s⁻¹

2. Mean-field steady-state:
   dθ_COOH/dt = k1_fwd × θ_* × P_CO2 - k1_rev × θ_COOH
              - k2_fwd × θ_COOH + k2_rev × θ_CO × P_H2O = 0
   dθ_CO/dt   = k2_fwd × θ_COOH - k2_rev × θ_CO × P_H2O
              - k_des × θ_CO = 0
   θ_* + θ_COOH + θ_CO = 1

3. Solve linear system → coverages:
   θ_* ≈ 0.998, θ_COOH ≈ 1.2e-6, θ_CO ≈ 3.8e-4

4. TOF = k2_fwd × θ_COOH = 0.19 s⁻¹ site⁻¹
```

### Output
```json
{
  "ok": true,
  "microkinetics": {
    "TOF": 0.19,
    "TOF_unit": "s⁻¹ site⁻¹",
    "coverages": {"*": 0.998, "COOH*": 1.2e-6, "CO*": 3.8e-4},
    "rds": "CO2 + H+ + e- → COOH*",
    "T_K": 298.15
  },
  "interpretation_md": "Rate-controlling step: CO2→COOH (Ea=0.92 eV). TOF = 0.19 /s..."
}
```

---

## Stage 10: Scientific Analysis & Reporting

```
POST /chat/analyze
```

### Processing (analyze_agent.py)

```
1. Load ALL session context:
   ├── intent (Stage 1)
   ├── hypothesis_md (Stage 2)
   ├── plan tasks (Stage 3)
   ├── DFT results (Stages 7–9)
   └── workflow task statuses

2. RAG literature (stage="analysis"):
   "CO2RR Cu(111) overpotential comparison DFT literature"
   → 5 relevant paper chunks

3. LLM call (GPT-4o):
   System: "You are a senior computational chemist reviewing results..."
   Input: full session context + DFT results + RAG
   Output schema: { conclusions[], gaps[], suggestions[], publication_checklist }
```

### Output
```json
{
  "ok": true,
  "summary_md": "## Analysis: CO₂RR on Cu(111)\n\n...",
  "conclusions": [
    {
      "finding": "Cu(111) shows η=0.61V for CO₂→CO, consistent with Peterson et al. (0.55V)",
      "confidence": 0.85,
      "evidence": "DFT E_ads(COOH)=0.78eV, NEB barrier=0.92eV"
    }
  ],
  "gaps": [
    "No solvation correction applied (VASPsol recommended)",
    "Missing competing HER pathway comparison"
  ],
  "suggestions": [
    {
      "action": "Run GC-DFT potential sweep",
      "priority": "high",
      "calc_type": "gcdft",
      "estimated_cost": "expensive"
    },
    {
      "action": "Calculate HER barrier for selectivity comparison",
      "priority": "critical",
      "calc_type": "neb"
    }
  ],
  "publication_checklist": {
    "status": "incomplete",
    "present": ["adsorption energies", "reaction barriers", "free energy diagram"],
    "missing": ["solvation effects", "competing pathways", "coverage-dependent barriers"],
    "nice_to_have": ["Bader charge analysis", "COHP bonding analysis"]
  }
}
```

---

## Database Schema (29 Tables)

```
                    ┌─────────────────┐
                    │  chat_session    │
                    │  (id, uid, name) │
                    └────────┬────────┘
                             │ 1:N
          ┌──────────────────┼──────────────────────────┐
          ▼                  ▼                           ▼
  ┌───────────────┐  ┌──────────────┐          ┌───────────────┐
  │ chat_message   │  │ workflow_task │          │ execution_task │
  │ (role, type,   │  │ (agent, type,│          │ (task_type,    │
  │  content)      │  │  status)     │          │  payload,      │
  └───────┬────────┘  └──────┬───────┘          │  status)       │
          │                  │                   └───────────────┘
          ▼                  ▼
  ┌───────────────┐  ┌──────────────┐
  │ hypothesis     │  │ dft_result   │
  │ (confidence)   │  │ (species,    │
  └───────────────┘  │  value, unit) │
                     └──────────────┘

  ┌─────────────────┐     ┌──────────────────┐
  │ reaction_system  │◄───│ mechanism_graph   │
  │ (domain, hash)   │ 1:N│ (steps, coads,   │
  └─────────────────┘     │  intermediates)   │
                          └──────────────────┘

  ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │ knowledge_doc    │◄───│ knowledge_chunk   │     │ knowledge_figure │
  │ (title, doi,     │ 1:N│ (text, embedding, │     │ (description,    │
  │  source_type)    │    │  section)         │     │  embedding)      │
  └─────────────────┘     └──────────────────┘     └──────────────────┘

  ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │ bulk_structure   │◄───│ slab_structure   │◄───│ adsorption_struct │
  │ (formula)        │ 1:N│ (miller, layers) │ 1:N│ (adsorbate, site)│
  └─────────────────┘     └──────────────────┘     └──────────────────┘

  ┌─────────────────┐     ┌──────────────────┐
  │ htp_run          │◄───│ htp_structure    │
  │ (strategy,       │ 1:N│ (energy, forces, │
  │  n_total)        │    │  converged)      │
  └─────────────────┘     └──────────────────┘

  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │ structure_library │  │ structure_t2s    │  │ plan_task_state  │
  │ (type, poscar,    │  │ (natural_lang,   │  │ (scripts, job_id │
  │  plot_png_b64)    │  │  poscar_content) │  │  results)        │
  └──────────────────┘  └──────────────────┘  └──────────────────┘

  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │ job               │  │ file_asset       │  │ result_row       │
  │ (pbs_id, status,  │  │ (kind, path)     │  │ (step, energy)   │
  │  remote_dir)      │  └──────────────────┘  └──────────────────┘
  └──────────────────┘

  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │ agent_log         │  │ execution_run    │  │ execution_step   │
  │ (model, tokens,   │  │ (workdir, tasks) │  │ (agent, status)  │
  │  latency_ms)      │  └──────────────────┘  └──────────────────┘
  └──────────────────┘

  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │ intent_phrase     │  │ modification_    │  │ literature_      │
  │ (stage, area,     │  │   structure      │  │   update_log     │
  │  phrase)          │  │ (parent, doping) │  │ (n_new_docs)     │
  └──────────────────┘  └──────────────────┘  └──────────────────┘

  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │ calculation_      │  │ job_schedule     │  │ post_analysis    │
  │   parameter       │  │ (scheduler,      │  │ (analysis_type,  │
  │ (incar_settings)  │  │  walltime)       │  │  extracted_data) │
  └──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## Data Flow Summary

```
                    Natural Language Query
                           │
                    ┌──────▼──────┐
                    │   INTENT    │ → IntentPhrase (few-shot)
                    │  GPT-4o    │ → KnowledgeChunk (RAG)
                    └──────┬──────┘
                           │ JSON intent
                    ┌──────▼──────┐
                    │ HYPOTHESIS  │ → ReactionSystem (cache)
                    │  GPT-4o    │ → MechanismGraph (template)
                    │ + validate │ → KnowledgeChunk (RAG)
                    └──────┬──────┘
                           │ reaction graph + confidence
                    ┌──────▼──────┐
                    │    PLAN     │ → mechanism registry (seed)
                    │  GPT-4o    │ → past sessions (history)
                    └──────┬──────┘
                           │ task DAG
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │STRUCTURE │ │STRUCTURE │ │STRUCTURE │
        │ slab     │ │ ads #1   │ │ ads #2   │
        │ ASE +    │ │ pymatgen │ │          │
        │ pymatgen │ │ site     │ │          │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │ POSCAR      │ POSCAR      │ POSCAR
        ┌────▼─────┐ ┌────▼─────┐ ┌────▼─────┐
        │PARAMETERS│ │PARAMETERS│ │PARAMETERS│
        │ INCAR    │ │ INCAR    │ │ INCAR    │
        │ KPOINTS  │ │ KPOINTS  │ │ KPOINTS  │
        │ profiles │ │          │ │          │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │             │
        ┌────▼─────────────▼─────────────▼────┐
        │         HPC SUBMISSION              │
        │  SSH → rsync UP → qsub/sbatch      │
        │  poll (60s) → rsync DOWN            │
        │  PBS (Hoffman2) / SLURM (Beluga)   │
        └────────────────┬────────────────────┘
                         │ OUTCAR, CONTCAR, vasprun.xml
                    ┌────▼──────┐
                    │   POST    │ → outcar_debugger (diagnostics)
                    │ ANALYSIS  │ → pymatgen Vasprun parser
                    │           │ → E_ads, barriers, DOS
                    └────┬──────┘
                         │ DFTResult rows
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        ┌──────────┐ ┌────────┐ ┌──────────┐
        │ FREE     │ │ MICRO- │ │ ANALYSIS │
        │ ENERGY   │ │KINETICS│ │ GPT-4o   │
        │ CHE + ZPE│ │Arrhenius│ │ senior   │
        │ diagram  │ │ TOF    │ │ chemist  │
        └──────────┘ └────────┘ └──────────┘
              │          │          │
              ▼          ▼          ▼
        Publication-Ready Results
        ├── Free energy diagram (PNG)
        ├── Overpotential + RDS
        ├── TOF + coverages
        ├── Conclusions + confidence
        ├── Gaps + suggestions
        └── Publication checklist
```

---

## External Service Dependencies

| Service | Purpose | Auth | Endpoint |
|---------|---------|------|----------|
| **OpenAI** | GPT-4o (reasoning), text-embedding-3-small (embeddings), GPT-4o-vision (figure extraction) | `OPENAI_API_KEY` | api.openai.com |
| **Perplexity** | Real-time web search for DFT literature | `PERPLEXITY_API_KEY` | api.perplexity.ai |
| **Zotero** | Personal literature library search | `ZOTERO_API_KEY` + `ZOTERO_USER_ID` | api.zotero.org/users/{id} |
| **arXiv** | Daily paper ingestion (12 chemistry queries) | None (public) | export.arxiv.org/api |
| **PubChem** | Molecule geometry from SMILES | None (public) | pubchem.ncbi.nlm.nih.gov |
| **Hoffman2** | HPC job execution (SGE) | SSH key | hoffman2.idre.ucla.edu |
| **Beluga/Cedar** | HPC job execution (SLURM) | SSH key | beluga.computecanada.ca |

---

## Deployment

```bash
# Prerequisites
PostgreSQL 14+ with pgvector extension
Python 3.10+
VASP license (for actual DFT runs)
SSH key access to HPC cluster

# Environment
cp .env.example .env
# Set: DATABASE_URL, OPENAI_API_KEY, HPC_HOST, HPC_USER

# Database
createdb chatdft_ase
psql chatdft_ase -c "CREATE EXTENSION IF NOT EXISTS vector"
alembic upgrade head

# Server
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

# Client
streamlit run client/app.py --server.port 8501

# Background: daily literature ingestion runs automatically on startup
# (15s delay, then every 24h)
```
