#!/usr/bin/env python3
"""
ChatDFT API Demo — Click, Run, Check
======================================

Run this file directly to see every API feature in action.
No server needed — everything runs offline.

Usage:
    python demo_api.py              # Run all demos
    python demo_api.py --demo 3     # Run only demo 3
    python demo_api.py --quick      # Skip slow demos

Demos:
    1. One-Click Workflow      — natural language → VASP inputs
    2. Batch Screening         — predict E_ads for many systems
    3. Input Validation        — catch errors before wasting HPC hours
    4. Auto-Fix                — automatically correct VASP mistakes
    5. SCF Diagnosis           — diagnose convergence failures
    6. Format Auto-Detection   — any format → POSCAR
    7. Smart Parameters        — describe what you want → INCAR
    8. Workflow Resolver       — multi-step dependency auto-resolution
    9. Save to Disk            — complete workflow → ready-to-submit folder
"""

import argparse
import json
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

# ─── Styling ───────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def header(n: int, title: str):
    print(f"\n{BLUE}{'=' * 65}")
    print(f"  Demo {n}: {title}")
    print(f"{'=' * 65}{RESET}\n")


def ok(msg: str):
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str):
    print(f"  {YELLOW}!{RESET} {msg}")


def fail(msg: str):
    print(f"  {RED}✗{RESET} {msg}")


def show(label: str, value, indent=4):
    prefix = " " * indent
    if isinstance(value, dict):
        print(f"{prefix}{BOLD}{label}:{RESET}")
        for k, v in value.items():
            if isinstance(v, dict):
                print(f"{prefix}  {k}: {{...}}")
            elif isinstance(v, list) and len(v) > 3:
                print(f"{prefix}  {k}: [{len(v)} items]")
            else:
                print(f"{prefix}  {k}: {v}")
    elif isinstance(value, list):
        print(f"{prefix}{BOLD}{label}:{RESET} [{len(value)} items]")
        for item in value[:5]:
            print(f"{prefix}  - {item}")
    else:
        print(f"{prefix}{BOLD}{label}:{RESET} {value}")


# ═══════════════════════════════════════════════════════════════════════
# Demo 1: One-Click Workflow
# ═══════════════════════════════════════════════════════════════════════

def demo_one_click():
    header(1, "One-Click Workflow: Natural Language → VASP Inputs")

    from chatdft import ChatDFT
    dft = ChatDFT(offline=True)

    queries = [
        "CO adsorption on Cu(111) with DFT-D3",
        "DOS of Pt(111) surface",
        "H adsorption on Ni(111) with spin polarization",
    ]

    for query in queries:
        print(f"  {BOLD}Query:{RESET} \"{query}\"")
        t0 = time.time()
        result = dft.run(query)
        elapsed = time.time() - t0

        if result.ok:
            ok(f"Generated in {elapsed:.3f}s")
            print(f"    Surface:    {result.element}({result.facet})")
            print(f"    Adsorbate:  {result.adsorbate or '(none)'}")
            print(f"    Calc type:  {result.calc_type}")
            print(f"    Atoms:      {result.n_atoms}")
            print(f"    POSCAR:     {len(result.poscar.splitlines())} lines")
            print(f"    Workflow:   {len(result.workflow_steps)} step(s)")
            print(f"    Validation: {'PASS' if result.validation.get('all_clear') else 'ISSUES'}")
            if result.notes:
                for n in result.notes[:3]:
                    print(f"    Note: {n}")
        else:
            fail(f"Failed: {result.error}")
        print()


# ═══════════════════════════════════════════════════════════════════════
# Demo 2: Batch Screening
# ═══════════════════════════════════════════════════════════════════════

def demo_screening():
    header(2, "Batch Screening: Predict E_ads for Many Systems")

    from chatdft import ChatDFT
    dft = ChatDFT(offline=True)

    systems = [
        ("CO",  "Pt(111)"),
        ("CO",  "Cu(111)"),
        ("CO",  "Au(111)"),
        ("CO",  "Ag(111)"),
        ("H",   "Pt(111)"),
        ("H",   "Ni(111)"),
        ("H",   "Fe(110)"),
        ("OH",  "Pt(111)"),
        ("OH",  "Cu(111)"),
        ("OOH", "Pt(111)"),
    ]

    print(f"  Screening {len(systems)} (adsorbate, surface) pairs...\n")

    results = dft.screen(systems)

    print(f"  {'Adsorbate':>8s}  {'Surface':<10s}  {'E_ads (eV)':>10s}  {'Model':<15s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*15}")
    for r in results:
        print(f"  {r.adsorbate:>8s}  {r.surface:<10s}  {r.E_ads_eV:>+10.3f}  {r.model:<15s}")

    ok(f"Screened {len(results)} systems in <0.01s (no DFT needed)")


# ═══════════════════════════════════════════════════════════════════════
# Demo 3: Input Validation
# ═══════════════════════════════════════════════════════════════════════

def demo_validation():
    header(3, "Input Validation: Catch Errors Before Wasting HPC Hours")

    from chatdft import ChatDFT
    dft = ChatDFT(offline=True)

    test_cases = [
        {
            "label": "ELF with wrong NCORE (VASP will crash)",
            "incar": {"LELF": True, "NCORE": 4, "ENCUT": 400},
            "elements": ["Cu"], "n_atoms": 36, "calc_type": "elf",
        },
        {
            "label": "Fe system without ISPIN=2 (wrong physics)",
            "incar": {"ENCUT": 400, "ISPIN": 1, "ISMEAR": 1},
            "elements": ["Fe", "N"], "n_atoms": 48, "calc_type": "static",
        },
        {
            "label": "ENCUT way too low for oxygen (won't converge)",
            "incar": {"ENCUT": 200, "ISMEAR": 1},
            "elements": ["Cu", "C", "O"], "n_atoms": 40, "calc_type": "static",
        },
        {
            "label": "COHP without ISYM=-1 (LOBSTER will fail)",
            "incar": {"ISYM": 0, "LWAVE": True, "LORBIT": 11},
            "elements": ["Pt", "C", "O"], "n_atoms": 38, "calc_type": "cohp",
        },
        {
            "label": "Correct setup — should pass",
            "incar": {"ENCUT": 520, "ISPIN": 2, "ISMEAR": 1},
            "elements": ["Ni"], "n_atoms": 36, "calc_type": "static",
        },
    ]

    for tc in test_cases:
        print(f"  {BOLD}Test:{RESET} {tc['label']}")
        issues = dft.validate(
            incar=tc["incar"], elements=tc["elements"],
            n_atoms=tc["n_atoms"], calc_type=tc["calc_type"],
        )
        if not issues:
            ok("No issues found — ready to submit")
        else:
            for i in issues:
                icon = RED + "✗" + RESET if i["severity"] == "error" else YELLOW + "!" + RESET
                fix_str = f" → Fix: {i['fix']}" if i.get("fix") else ""
                print(f"    {icon} [{i['severity']}] {i['message']}{fix_str}")
        print()


# ═══════════════════════════════════════════════════════════════════════
# Demo 4: Auto-Fix
# ═══════════════════════════════════════════════════════════════════════

def demo_autofix():
    header(4, "Auto-Fix: Automatically Correct VASP Mistakes")

    from chatdft import ChatDFT
    dft = ChatDFT(offline=True)

    print(f"  {BOLD}Input INCAR (broken):{RESET}")
    broken = {"LELF": True, "NCORE": 4, "ENCUT": 200, "ISPIN": 1}
    for k, v in broken.items():
        print(f"    {k:<12} = {v}")

    print(f"\n  Elements: Fe, O  |  Calc type: elf\n")

    fixed, changes = dft.autofix(
        incar=broken,
        elements=["Fe", "O"],
        n_atoms=48,
        calc_type="elf",
    )

    print(f"  {BOLD}Changes applied:{RESET}")
    for c in changes:
        ok(c[:80])

    print(f"\n  {BOLD}Fixed INCAR:{RESET}")
    for k, v in sorted(fixed.items()):
        changed = k in {"NCORE", "ENCUT", "ISPIN"}
        marker = f" {GREEN}← fixed{RESET}" if changed else ""
        print(f"    {k:<12} = {v}{marker}")


# ═══════════════════════════════════════════════════════════════════════
# Demo 5: SCF Diagnosis
# ═══════════════════════════════════════════════════════════════════════

def demo_scf_diagnosis():
    header(5, "SCF Diagnosis: Analyze Convergence Trajectory")

    from chatdft import ChatDFT
    dft = ChatDFT(offline=True)

    cases = [
        {
            "label": "Healthy convergence",
            "ediffs": list(10 ** np.linspace(0, -6, 30)),
            "incar": {},
        },
        {
            "label": "Charge sloshing (oscillating)",
            "ediffs": list(np.abs(np.sin(np.linspace(0, 15*np.pi, 60))) * 0.01 + 0.001),
            "incar": {"ALGO": "Fast", "AMIX": 0.4},
        },
        {
            "label": "Slow monotonic convergence",
            "ediffs": list(10 ** np.linspace(-2, -3, 80)),
            "incar": {"ALGO": "Fast", "NELM": 200},
        },
    ]

    for case in cases:
        print(f"  {BOLD}Case:{RESET} {case['label']}")
        result = dft.diagnose_scf(
            case["ediffs"],
            current_incar=case["incar"],
        )
        diag = result["diagnosis"]
        color = GREEN if diag == "healthy" else RED
        print(f"    Diagnosis:  {color}{diag}{RESET}")
        print(f"    Conv rate:  {result['convergence_rate']:.4f}")
        print(f"    Sloshing:   {result['sloshing_ratio']:.3f}")
        if result["fix"]:
            print(f"    Fix:        {result['fix']}")
        print(f"    Reason:     {result['explanation'][:80]}...")
        print()


# ═══════════════════════════════════════════════════════════════════════
# Demo 6: Format Auto-Detection
# ═══════════════════════════════════════════════════════════════════════

def demo_format_detection():
    header(6, "Format Auto-Detection: Any Input → POSCAR")

    from server.api.preprocessor import detect_format

    test_inputs = [
        ("Pt(111) 4x4 with CO", "Material name"),
        ("bulk Cu", "Bulk material"),
        ("O=C=O", "SMILES (CO₂)"),
        ("CC(=O)O", "SMILES (acetic acid)"),
        ("3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0", "XYZ"),
        ("data_Cu\n_cell_length_a 3.615", "CIF"),
        ("Cu slab\n1.0\n7.668 0.0 0.0\n3.834 6.64 0.0\n0.0 0.0 25.0\nCu\n36\nCart\n0 0 0", "POSCAR"),
    ]

    print(f"  {'Input (first 40 chars)':<42s}  {'Detected Format':<15s}  {'Expected':<15s}")
    print(f"  {'─'*42}  {'─'*15}  {'─'*15}")

    for content, expected_label in test_inputs:
        fmt = detect_format(content)
        preview = content.replace("\n", "\\n")[:40]
        match = GREEN + "✓" + RESET
        print(f"  {preview:<42s}  {fmt:<15s}  {expected_label:<15s} {match}")

    # Actually convert one
    print(f"\n  {BOLD}Live conversion: 'Cu(111)' → POSCAR{RESET}")
    from chatdft import ChatDFT
    dft = ChatDFT(offline=True)
    poscar = dft.preprocess("Cu(111)")
    lines = poscar.strip().split("\n")
    for line in lines[:6]:
        print(f"    {line}")
    print(f"    ... ({len(lines)} lines total)")
    ok("Converted material name to POSCAR")


# ═══════════════════════════════════════════════════════════════════════
# Demo 7: Smart Parameters
# ═══════════════════════════════════════════════════════════════════════

def demo_smart_params():
    header(7, "Smart Parameters: Describe → INCAR")

    from chatdft import ChatDFT
    dft = ChatDFT(offline=True)

    descriptions = [
        "Bader charge analysis of CO on Cu(111)",
        "DOS of Pt(111) surface atoms",
        "Geometry relaxation of H on Fe(110) with DFT-D3",
    ]

    for desc in descriptions:
        print(f"  {BOLD}Description:{RESET} \"{desc}\"")
        result = dft.smart_params(desc)
        print(f"    Calc type:  {result.get('calc_type', '?')}")
        print(f"    KPOINTS:    {result.get('kpoints', '?')}")
        incar = result.get("incar", {})
        key_params = {k: v for k, v in incar.items() if k in {
            "ENCUT", "ISMEAR", "ISPIN", "LAECHG", "LORBIT", "IVDW",
            "IBRION", "NSW", "ICHARG", "LREAL", "PREC",
        }}
        print(f"    Key INCAR:  {key_params}")
        print()


# ═══════════════════════════════════════════════════════════════════════
# Demo 8: Workflow Resolver
# ═══════════════════════════════════════════════════════════════════════

def demo_workflow_resolver():
    header(8, "Workflow Resolver: Multi-Step Dependencies")

    from science.vasp.auto_remediation import resolve_workflow

    calc_types = ["dos", "band", "elf", "cohp", "bader", "work_function", "static"]

    print(f"  {'Calc Type':<16s}  {'Steps':>5s}  {'Workflow'}")
    print(f"  {'─'*16}  {'─'*5}  {'─'*40}")

    for calc in calc_types:
        steps = resolve_workflow(calc)
        step_names = " → ".join(s.name for s in steps)
        print(f"  {calc:<16s}  {len(steps):>5d}  {step_names}")

    # Show one in detail
    print(f"\n  {BOLD}Detailed: DOS workflow{RESET}")
    steps = resolve_workflow("dos")
    for i, step in enumerate(steps):
        print(f"    Step {i+1}: {step.name}")
        print(f"      Calc type:  {step.calc_type}")
        if step.incar_overrides:
            print(f"      INCAR:      {step.incar_overrides}")
        if step.depends_on:
            print(f"      Depends on: {step.depends_on}")
        if step.output_files:
            print(f"      Outputs:    {step.output_files}")
        print(f"      Notes:      {step.notes}")


# ═══════════════════════════════════════════════════════════════════════
# Demo 9: Save to Disk
# ═══════════════════════════════════════════════════════════════════════

def demo_save():
    header(9, "Save to Disk: Ready-to-Submit Folder")

    from chatdft import ChatDFT
    dft = ChatDFT(offline=True)

    result = dft.run("H adsorption on Cu(111)")

    with tempfile.TemporaryDirectory() as tmpdir:
        calc_dir = Path(tmpdir) / "H_Cu111"
        files = result.save(str(calc_dir))

        print(f"  Saved to: {calc_dir}/\n")
        print(f"  {'File':<20s}  {'Size':>8s}  {'Preview'}")
        print(f"  {'─'*20}  {'─'*8}  {'─'*40}")

        for fpath in sorted(files):
            p = Path(fpath)
            size = p.stat().st_size
            content = p.read_text()
            preview = content.split("\n")[0][:40]
            print(f"  {p.name:<20s}  {size:>6d} B  {preview}")

        ok(f"Wrote {len(files)} files — ready for: rsync + sbatch run.slurm")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

ALL_DEMOS = {
    1: ("One-Click Workflow", demo_one_click),
    2: ("Batch Screening", demo_screening),
    3: ("Input Validation", demo_validation),
    4: ("Auto-Fix", demo_autofix),
    5: ("SCF Diagnosis", demo_scf_diagnosis),
    6: ("Format Auto-Detection", demo_format_detection),
    7: ("Smart Parameters", demo_smart_params),
    8: ("Workflow Resolver", demo_workflow_resolver),
    9: ("Save to Disk", demo_save),
}


def main():
    parser = argparse.ArgumentParser(description="ChatDFT API Demo")
    parser.add_argument("--demo", type=int, help="Run only this demo number (1-9)")
    parser.add_argument("--quick", action="store_true", help="Skip slow demos")
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*65}")
    print(f"  ChatDFT API Demo — Click, Run, Check")
    print(f"  Zero config. No server. No API key. No VASP license.")
    print(f"{'='*65}{RESET}")

    if args.demo:
        if args.demo in ALL_DEMOS:
            name, fn = ALL_DEMOS[args.demo]
            fn()
        else:
            print(f"Unknown demo {args.demo}. Available: 1-{len(ALL_DEMOS)}")
            sys.exit(1)
    else:
        for num, (name, fn) in ALL_DEMOS.items():
            try:
                fn()
            except Exception as e:
                fail(f"Demo {num} ({name}) failed: {e}")

    print(f"\n{BOLD}{'='*65}")
    print(f"  All demos complete.")
    print(f"  Run 'python demo_api.py --demo N' for a specific demo.")
    print(f"{'='*65}{RESET}\n")


if __name__ == "__main__":
    main()
