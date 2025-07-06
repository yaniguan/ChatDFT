# ChatDFT: Language-Guided Hypothesis, Simulation, and Analysis for Materials Discovery

> Talk to your DFT workflow. Let agents handle the rest.

ChatDFT is a modular, intelligent system powered by Large Language Models (LLMs) to assist and automate DFT (Density Functional Theory) workflows, starting with VASP. It enables hypothesis generation, input file creation, error diagnosis, job execution, post-analysis, and human-readable explanations — all in natural language.


From Scientific Problem to Computation

1. Define the Scientific Problem

Start from a real research question:
	•	Classification:
	•	Geometry optimization
	•	Reaction mechanism
	•	Material property
	•	Surface adsorption
	•	Electronic structure
	•	Property Type:
	•	Thermodynamics
	•	Kinetics
	•	Optical / Electronic
	•	Targeted Outputs:
	•	Energy, structure, band gap
	•	Transition state (TS)
	•	ELF, charge density difference
	•	Adsorption energy
	•	Diffusion barriers

2. Select Suitable Software

Currently implemented: VASP
	•	Future extension: CP2K, Quantum ESPRESSO, LAMMPS, etc.

3. Define Required Tasks
	•	Geometry optimization
	•	Self-consistent field (SCF) runs
	•	DOS, band structure, Bader charge, ELF, charge density difference
	•	Surface adsorption: ΔG, E_ads, transition state search

⸻
```
(base) yaniguan@Mac-209 llm_vasp_mcp % tree
.
├── agents
│   ├── __init__.py
│   ├── hypothesis_agent.py
│   ├── incar_agent.py
│   ├── kpoints_agent.py
│   ├── poscar_agent.py
│   ├── postprocess_agent.py
│   ├── slurm_agent.py
│   └── task_agent.py
├── config.yaml
├── core
│   ├── memory
│   │   └── memory.py
│   ├── prompts
│   │   ├── incar_builder_prompt.md
│   │   ├── kpoints_builder_prompt.md
│   │   ├── poscar_builder_prompt.md
│   │   ├── slurm_job_prompt.md
│   │   └── task_classifier_prompt.md
│   └── tools
│       ├── format_checker.py
│       ├── plotting_tool.py
│       ├── slurm_tool.py
│       └── vasp_io_tool.py
├── data
│   └── samples
│       └── material_example.json
├── interfaces
│   ├── api.py
│   └── web_ui.py
├── logs
│   └── debug.log
├── main.py
├── notebooks
│   └── exploration.ipynb
├── outputs
│   ├── plots
│   └── results_summary.md
├── README.md
└── requirements.txt
```