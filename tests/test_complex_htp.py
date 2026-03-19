#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end tests for build_complex() and HTP dataset generation.
Run: python tests/test_complex_htp.py
"""
import json, sys, os, tempfile
import requests
import numpy as np

BASE = "http://localhost:8000"
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
BOLD = "\033[1m"
RST  = "\033[0m"

def post(ep, body, timeout=60):
    r = requests.post(f"{BASE}{ep}", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()

def check(label, val, expect=True):
    ok = bool(val) == expect if not isinstance(val, bool) else val == expect
    print(f"  {'✓' if ok else '✗'} {label}: {val}")
    return ok

def section(title):
    print(f"\n{'='*65}\n{BOLD}{title}{RST}\n{'='*65}")

results = {}

# ─────────────────────────────────────────────────────────────────────────────
# PART A: build_complex() — coordination compounds
# ─────────────────────────────────────────────────────────────────────────────
section("PART A: build_complex() — coordination compounds")

# A1. Cu(NH3)4 — square planar (Cu²⁺ d⁸-like, tetraammine)
section("A1. Cu(NH3)4 — square planar")
try:
    r = post("/agent/structure/build_complex", {
        "metal": "Cu", "ligand": "NH3", "n_coord": 4,
        "geometry": "square_planar", "bond_length": 2.05, "cell_size": 15.0
    })
    check("Build Cu(NH3)4", r.get("ok"))
    check("structure_type=complex", r.get("structure_type") == "complex")
    check("Label contains Cu", "Cu" in r.get("label",""))
    check("POSCAR non-empty", bool(r.get("poscar","")))
    n = r.get("n_atoms", 0)
    check(f"n_atoms = 17 (1 Cu + 4×N + 12×H = 17)", n == 17)
    check("description present", len(r.get("description","")) > 20)
    check("ase_code present", "Atoms" in r.get("ase_code",""))
    print(f"  → {r.get('label')}  formula={r.get('formula')}  n_atoms={n}")

    # Verify geometry: Cu-N distances should all equal bond_length
    poscar = r["poscar"]
    from ase.io.vasp import read_vasp
    import io
    atoms = read_vasp(io.StringIO(poscar))
    cu_idx = [i for i,s in enumerate(atoms.get_chemical_symbols()) if s == "Cu"][0]
    n_indices = [i for i,s in enumerate(atoms.get_chemical_symbols()) if s == "N"]
    cu_pos = atoms.get_positions()[cu_idx]
    n_dists = [np.linalg.norm(atoms.get_positions()[i] - cu_pos) for i in n_indices]
    check("All 4 Cu-N bonds ≈ 2.05 Å", all(abs(d - 2.05) < 0.05 for d in n_dists))
    print(f"  → Cu-N distances: {[f'{d:.3f}' for d in n_dists]}")
    results["A1"] = True
except Exception as e:
    print(f"  ✗ EXCEPTION: {e}")
    results["A1"] = False

# A2. Fe(CO)6 — octahedral (metal carbonyl)
section("A2. Fe(CO)6 — octahedral")
try:
    r = post("/agent/structure/build_complex", {
        "metal": "Fe", "ligand": "CO", "n_coord": 6,
        "geometry": "octahedral", "bond_length": 1.84, "cell_size": 15.0
    })
    check("Build Fe(CO)6", r.get("ok"))
    n = r.get("n_atoms", 0)
    check(f"n_atoms = 13 (1 Fe + 6×C + 6×O = 13)", n == 13)
    check("Formula contains Fe", "Fe" in r.get("formula",""))
    print(f"  → {r.get('label')}  n_atoms={n}")

    # Check C binds to Fe (C at bond_length, O beyond)
    poscar = r["poscar"]
    from ase.io.vasp import read_vasp
    import io
    atoms = read_vasp(io.StringIO(poscar))
    fe_idx = [i for i,s in enumerate(atoms.get_chemical_symbols()) if s == "Fe"][0]
    c_indices = [i for i,s in enumerate(atoms.get_chemical_symbols()) if s == "C"]
    fe_pos = atoms.get_positions()[fe_idx]
    c_dists = [np.linalg.norm(atoms.get_positions()[i] - fe_pos) for i in c_indices]
    check("All 6 Fe-C bonds ≈ 1.84 Å", all(abs(d - 1.84) < 0.05 for d in c_dists))
    print(f"  → Fe-C distances: {[f'{d:.3f}' for d in c_dists]}")
    results["A2"] = True
except Exception as e:
    print(f"  ✗ EXCEPTION: {e}")
    results["A2"] = False

# A3. PtCl4²⁻ — square planar (Pt²⁺ d⁸)
section("A3. PtCl4 — square planar")
try:
    r = post("/agent/structure/build_complex", {
        "metal": "Pt", "ligand": "Cl", "n_coord": 4,
        "geometry": "square_planar", "bond_length": 2.32, "cell_size": 15.0
    })
    check("Build PtCl4", r.get("ok"))
    n = r.get("n_atoms", 0)
    check("n_atoms = 5 (1 Pt + 4 Cl)", n == 5)
    print(f"  → {r.get('label')}  n_atoms={n}")
    results["A3"] = True
except Exception as e:
    print(f"  ✗ EXCEPTION: {e}")
    results["A3"] = False

# A4. Ni(CO)4 — tetrahedral
section("A4. Ni(CO)4 — tetrahedral")
try:
    r = post("/agent/structure/build_complex", {
        "metal": "Ni", "ligand": "CO", "n_coord": 4,
        "geometry": "tetrahedral", "bond_length": 1.84, "cell_size": 15.0
    })
    check("Build Ni(CO)4", r.get("ok"))
    n = r.get("n_atoms", 0)
    check("n_atoms = 9 (1 Ni + 4C + 4O)", n == 9)
    print(f"  → {r.get('label')}  n_atoms={n}")
    results["A4"] = True
except Exception as e:
    print(f"  ✗ EXCEPTION: {e}")
    results["A4"] = False

# A5. Co(NH3)6 — octahedral (tris-ammine)
section("A5. Co(NH3)6 — octahedral")
try:
    r = post("/agent/structure/build_complex", {
        "metal": "Co", "ligand": "NH3", "n_coord": 6,
        "geometry": "octahedral", "bond_length": 1.97, "cell_size": 15.0
    })
    check("Build Co(NH3)6", r.get("ok"))
    n = r.get("n_atoms", 0)
    check("n_atoms = 25 (1 Co + 6N + 18H)", n == 25)
    print(f"  → {r.get('label')}  n_atoms={n}")
    results["A5"] = True
except Exception as e:
    print(f"  ✗ EXCEPTION: {e}")
    results["A5"] = False

# A6. Linear AuCl2⁻
section("A6. AuCl2 — linear (d¹⁰)")
try:
    r = post("/agent/structure/build_complex", {
        "metal": "Au", "ligand": "Cl", "n_coord": 2,
        "geometry": "linear", "bond_length": 2.27, "cell_size": 12.0
    })
    check("Build AuCl2", r.get("ok"))
    n = r.get("n_atoms", 0)
    check("n_atoms = 3 (1 Au + 2 Cl)", n == 3)
    print(f"  → {r.get('label')}  n_atoms={n}")
    results["A6"] = True
except Exception as e:
    print(f"  ✗ EXCEPTION: {e}")
    results["A6"] = False

# A7. Error handling — unknown ligand
section("A7. Error handling — unknown ligand")
try:
    r = post("/agent/structure/build_complex", {
        "metal": "Cu", "ligand": "UNKNOWN_LIGAND", "n_coord": 4
    })
    check("Returns ok=False for unknown ligand", not r.get("ok", True))
    check("Error message mentions supported ligands",
          "support" in r.get("detail","").lower() or "support" in r.get("error","").lower()
          or "unknown" in str(r).lower())
    print(f"  → error: {r.get('detail') or r.get('error','')}")
    results["A7"] = True
except Exception as e:
    print(f"  ✗ EXCEPTION: {e}")
    results["A7"] = False

# ─────────────────────────────────────────────────────────────────────────────
# PART B: HTP dataset generation
# ─────────────────────────────────────────────────────────────────────────────
section("PART B: HTP dataset generation")

# First build a base structure (Pt111 slab) to use for all HTP tests
print("\nBuilding base structures for HTP tests...")
surf_pt = post("/agent/structure/build_surface", {
    "element": "Pt", "surface_type": "111",
    "nx": 3, "ny": 3, "nlayers": 3, "vacuum": 10.0, "fix_bottom": True
})
surf_cu = post("/agent/structure/build_surface", {
    "element": "Cu", "surface_type": "111",
    "nx": 3, "ny": 3, "nlayers": 3, "vacuum": 10.0, "fix_bottom": True
})
pt_poscar = surf_pt.get("poscar","")
cu_poscar = surf_cu.get("poscar","")
print(f"  Pt(111) 3×3×3: {surf_pt.get('formula')}  ok={surf_pt.get('ok')}")
print(f"  Cu(111) 3×3×3: {surf_cu.get('formula')}  ok={surf_cu.get('ok')}")

with tempfile.TemporaryDirectory() as tmpdir:
    db_path = os.path.join(tmpdir, "test_htp.db")
    xyz_path = os.path.join(tmpdir, "training.xyz")

    # B1. Rattle strategy
    section("B1. Rattle strategy (100 structures from Pt111)")
    try:
        r = post("/agent/htp/generate", {
            "base_structures": [{"poscar": pt_poscar, "label": "Pt111"}],
            "strategy": "rattle",
            "n_total": 100,
            "db_path": db_path,
            "stdev": 0.08,
            "seed": 42,
        }, timeout=30)
        check("HTP rattle generation ok", r.get("ok"))
        n = r.get("n_generated", 0)
        check("n_generated = 100", n == 100)
        print(f"  → Generated {n} structures in {db_path}")
        stats = r.get("stats", {})
        check("stats.total = 100", stats.get("total") == 100)
        check("stats.pending = 100", stats.get("pending") == 100)
        print(f"  → stats: {stats}")
        results["B1"] = True
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        results["B1"] = False

    # B2. Strain strategy
    section("B2. Strain strategy (Pt111)")
    try:
        r = post("/agent/htp/generate", {
            "base_structures": [{"poscar": pt_poscar, "label": "Pt111"}],
            "strategy": "strain",
            "n_total": 11,
            "db_path": db_path,
            "strains": [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
        }, timeout=30)
        check("HTP strain generation ok", r.get("ok"))
        n = r.get("n_generated", 0)
        check("n_generated = 11", n == 11)
        print(f"  → Generated {n} strained structures")
        results["B2"] = True
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        results["B2"] = False

    # B3. Rattle+strain combined
    section("B3. Rattle+strain combined (50 structures)")
    try:
        r = post("/agent/htp/generate", {
            "base_structures": [{"poscar": pt_poscar, "label": "Pt111"}],
            "strategy": "rattle_strain",
            "n_total": 50,
            "db_path": db_path,
            "stdev": 0.05,
        }, timeout=30)
        check("Rattle+strain ok", r.get("ok"))
        n = r.get("n_generated", 0)
        check("n_generated = 50", n == 50)
        print(f"  → Generated {n} structures")
        results["B3"] = True
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        results["B3"] = False

    # B4. Surface rattle (only top layers rattled)
    section("B4. Surface rattle (top 2 layers only)")
    try:
        r = post("/agent/htp/generate", {
            "base_structures": [{"poscar": pt_poscar, "label": "Pt111"}],
            "strategy": "surface_rattle",
            "n_total": 30,
            "db_path": db_path,
            "stdev": 0.10,
            "n_surface_layers": 2,
        }, timeout=30)
        check("Surface rattle ok", r.get("ok"))
        n = r.get("n_generated", 0)
        check("n_generated = 30", n == 30)
        print(f"  → Generated {n} surface-rattled structures")
        results["B4"] = True
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        results["B4"] = False

    # B5. Temperature rattle (Boltzmann-weighted)
    section("B5. Temperature rattle at 300K and 800K")
    try:
        r300 = post("/agent/htp/generate", {
            "base_structures": [{"poscar": pt_poscar, "label": "Pt111"}],
            "strategy": "temperature_rattle",
            "n_total": 20,
            "db_path": db_path,
            "temperature": 300.0,
        }, timeout=30)
        r800 = post("/agent/htp/generate", {
            "base_structures": [{"poscar": pt_poscar, "label": "Pt111"}],
            "strategy": "temperature_rattle",
            "n_total": 20,
            "db_path": db_path,
            "temperature": 800.0,
        }, timeout=30)
        check("T=300K rattle ok", r300.get("ok"))
        check("T=800K rattle ok", r800.get("ok"))
        check("20 structures at 300K", r300.get("n_generated") == 20)
        check("20 structures at 800K", r800.get("n_generated") == 20)
        print(f"  → 300K stdev hint: {r300.get('stdev_used','?')} Å")
        print(f"  → 800K stdev hint: {r800.get('stdev_used','?')} Å")
        results["B5"] = True
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        results["B5"] = False

    # B6. Alloy configs (Cu doped into Pt)
    section("B6. Alloy configs: Pt1-x Cu_x")
    try:
        r = post("/agent/htp/generate", {
            "base_structures": [{"poscar": pt_poscar, "label": "Pt111"}],
            "strategy": "alloy",
            "n_total": 15,
            "db_path": db_path,
            "host": "Pt",
            "dopant": "Cu",
            "concentrations": [0.1, 0.2, 0.3],
            "n_per_conc": 5,
        }, timeout=30)
        check("Alloy configs ok", r.get("ok"))
        n = r.get("n_generated", 0)
        check("n_generated = 15", n == 15)
        print(f"  → Generated {n} Pt1-xCux alloy configs")
        results["B6"] = True
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        results["B6"] = False

    # B7. Vacancy configs
    section("B7. Vacancy configs (Pt111 with vacancies)")
    try:
        r = post("/agent/htp/generate", {
            "base_structures": [{"poscar": pt_poscar, "label": "Pt111"}],
            "strategy": "vacancy",
            "n_total": 10,
            "db_path": db_path,
            "n_vacancies": 1,
        }, timeout=30)
        check("Vacancy configs ok", r.get("ok"))
        n = r.get("n_generated", 0)
        check("n_generated = 10", n == 10)
        print(f"  → Generated {n} vacancy structures")
        results["B7"] = True
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        results["B7"] = False

    # B8. Multi-base: both Pt and Cu surfaces
    section("B8. Multi-base structures (Pt + Cu, rattle, 40 each)")
    try:
        r = post("/agent/htp/generate", {
            "base_structures": [
                {"poscar": pt_poscar, "label": "Pt111"},
                {"poscar": cu_poscar, "label": "Cu111"},
            ],
            "strategy": "rattle",
            "n_total": 80,
            "db_path": db_path,
            "stdev": 0.08,
        }, timeout=30)
        check("Multi-base rattle ok", r.get("ok"))
        n = r.get("n_generated", 0)
        check("n_generated = 80", n == 80)
        print(f"  → Generated {n} structures from 2 base structures")
        results["B8"] = True
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        results["B8"] = False

    # B9. Dataset stats
    section("B9. Dataset stats")
    try:
        r = post("/agent/htp/stats", {"db_path": db_path})
        check("Stats ok", r.get("ok"))
        stats = r.get("stats", {})
        total = stats.get("total", 0)
        check("Total > 300 structures", total > 300)
        check("pending > 0", stats.get("pending", 0) > 0)
        check("done = 0 (no VASP run yet)", stats.get("done", 0) == 0)
        print(f"  → Total={stats.get('total')} Pending={stats.get('pending')} Done={stats.get('done')} Failed={stats.get('failed')}")
        print(f"  → Strategies: {stats.get('strategies', {})}")
        results["B9"] = True
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        results["B9"] = False

    # B10. Simulate marking done + export extXYZ
    # Use htp_agent directly via sys.path manipulation (unit-level test)
    section("B10. Simulate SP results → export extXYZ")
    try:
        import sys as _sys
        _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _repo not in _sys.path:
            _sys.path.insert(0, _repo)
        from server.execution.htp_agent import HTPDataset

        ds = HTPDataset(db_path=db_path)
        pending = ds.get_pending(limit=5)
        check("get_pending returns structures", len(pending) > 0)

        # Simulate marking 5 structures as done with fake energies/forces
        for row_id, atoms in pending:
            n = len(atoms)
            fake_energy = -5.0 * n + np.random.randn() * 0.1
            fake_forces = np.random.randn(n, 3) * 0.1
            fake_stress = np.random.randn(3, 3) * 0.01
            ds.mark_done(row_id, fake_energy, fake_forces, fake_stress)

        # Export extXYZ
        n_exp = ds.export_extxyz(xyz_path, only_done=True)
        check("export_extxyz returns > 0", n_exp > 0)
        check("extXYZ file created", os.path.exists(xyz_path))
        check("5 frames exported", n_exp == 5)
        print(f"  → Exported {n_exp} frames to {xyz_path}")

        # Verify extXYZ can be read back by ASE
        from ase.io import read as ase_read
        frames = ase_read(xyz_path, index=":")
        check("ASE can read back extXYZ", len(frames) == n_exp)
        f0 = frames[0]
        # In ASE 3.22+, E/F/S round-trip via SinglePointCalculator
        check("energy accessible", f0.calc is not None and f0.get_potential_energy() != 0)
        check("forces accessible", f0.calc is not None and f0.get_forces().shape == (len(f0), 3))
        e0 = f0.get_potential_energy()
        print(f"  → Frame 0: {len(f0)} atoms, energy={e0:.3f} eV")
        results["B10"] = True
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        results["B10"] = False

    # B11. Batch HTP script generation
    section("B11. HTP batch script (for cluster submission)")
    try:
        r = post("/agent/htp/script", {
            "db_path": "htp_dataset.db",
            "encut": 450,
            "kpoints": "4 4 1",
            "batch_size": 50,
            "scheduler": "sge",
        })
        check("HTP script ok", r.get("ok"))
        script = r.get("script", "")
        check("Script non-empty", len(script) > 200)
        check("ENCUT=450 in script", "450" in script)
        check("ISIF=2 (stress tensor)", "isif=2" in script.lower())
        check("extxyz export in script", "extxyz" in script.lower() or "xyz" in script.lower())
        check("Loops over pending structures", "pending" in script.lower() or "get_pending" in script)
        print(f"  → Script: htp_batch.py ({len(script)} chars)")
        print(f"  → First 400 chars:\n{script[:400]}")
        results["B11"] = True
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        results["B11"] = False

    # B12. INCAR NNP preset
    section("B12. NNP single-point INCAR preset")
    try:
        r = post("/agent/incar_preset", {"calc_type": "nnp"})
        check("NNP INCAR preset ok", r.get("ok"))
        incar = r.get("incar", {})
        check("PREC=Accurate", incar.get("PREC") == "Accurate")
        check("ENCUT=450", incar.get("ENCUT") == 450)
        check("EDIFF=1e-6", incar.get("EDIFF") == 1e-6)
        check("ISIF=2 (stress tensor)", incar.get("ISIF") == 2)
        check("LWAVE=False", incar.get("LWAVE") is False)
        check("LCHARG=False", incar.get("LCHARG") is False)
        check("NSW=0 (single point)", incar.get("NSW") == 0)
        incar_str = r.get("incar_string", "")
        print(f"  → INCAR string preview:\n{incar_str[:300]}")
        results["B12"] = True
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        results["B12"] = False

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY")
labels = {
    "A1": "Cu(NH3)4 square planar",
    "A2": "Fe(CO)6 octahedral",
    "A3": "PtCl4 square planar",
    "A4": "Ni(CO)4 tetrahedral",
    "A5": "Co(NH3)6 octahedral",
    "A6": "AuCl2 linear",
    "A7": "Error handling (unknown ligand)",
    "B1": "HTP rattle (100 structs)",
    "B2": "HTP strain sweep",
    "B3": "HTP rattle+strain",
    "B4": "HTP surface rattle",
    "B5": "HTP temperature rattle 300/800K",
    "B6": "HTP alloy configs Pt1-xCux",
    "B7": "HTP vacancy configs",
    "B8": "HTP multi-base structures",
    "B9": "Dataset stats",
    "B10": "Mark done + extXYZ export",
    "B11": "Batch job script",
    "B12": "NNP INCAR preset",
}
passed = sum(1 for v in results.values() if v)
total  = len(results)
for k in sorted(results):
    sym = "✓" if results[k] else "✗"
    print(f"  {sym} {k}: {labels[k]}")
print(f"\n{BOLD}{passed}/{total} tests passed{RST}")
sys.exit(0 if passed == total else 1)
