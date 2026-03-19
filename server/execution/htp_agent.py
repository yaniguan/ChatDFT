# server/execution/htp_agent.py
# -*- coding: utf-8 -*-
"""
High-Throughput NNP Dataset Generation

Generates diverse structures for neural-network-potential (NNP) training datasets
and prepares single-point VASP calculations.

Strategies
----------
rattle              : random atomic displacements (Gaussian noise)
strain              : uniform volumetric strain sweep
rattle_strain       : combined rattle + strain (most diverse)
surface_rattle      : rattle only top-2-layer atoms; bulk fixed
temperature_rattle  : Boltzmann-weighted displacements via Einstein model
alloy_configs       : random binary-alloy occupancy configurations
vacancy_configs     : random vacancy structures

Dataset management via HTPDataset (backed by ASE db).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.db import connect as _ase_db_connect
from ase.io import write as _ase_write, read as _ase_read
from ase.io.vasp import read_vasp as _read_vasp

import io as _io


# ══════════════════════════════════════════════════════════════════════════════
# 1. Rattle
# ══════════════════════════════════════════════════════════════════════════════

def rattle_structures(
    atoms: Atoms,
    n_structures: int,
    stdev: float = 0.1,
    seed: int = 42,
) -> List[Atoms]:
    """Generate n_structures by randomly displacing atoms with Gaussian noise (stdev Å)."""
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_structures):
        a = atoms.copy()
        displacements = rng.normal(0, stdev, (len(a), 3))
        a.set_positions(a.get_positions() + displacements)
        a.info["htp_id"] = i
        a.info["strategy"] = "rattle"
        a.info["stdev"] = stdev
        results.append(a)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. Strain
# ══════════════════════════════════════════════════════════════════════════════

def strain_structures(
    atoms: Atoms,
    strains: Optional[List[float]] = None,
) -> List[Atoms]:
    """Apply strain values as uniform volumetric strain.

    For each strain e: new_cell = cell * (1+e)^(1/3), scale atoms accordingly.
    Default strains: [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05].
    """
    if strains is None:
        strains = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05]
    results = []
    cell0 = atoms.get_cell()
    pos0 = atoms.get_positions()
    for idx, e in enumerate(strains):
        a = atoms.copy()
        scale = (1.0 + e) ** (1.0 / 3.0)
        new_cell = cell0 * scale
        a.set_cell(new_cell, scale_atoms=True)
        a.info["htp_id"] = idx
        a.info["strategy"] = "strain"
        a.info["strain"] = float(e)
        results.append(a)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. Rattle + strain (combined)
# ══════════════════════════════════════════════════════════════════════════════

def rattle_strain_structures(
    atoms: Atoms,
    n_structures: int,
    stdev: float = 0.1,
    strains: Optional[List[float]] = None,
    seed: int = 42,
) -> List[Atoms]:
    """Combined rattle + strain — most diverse for NNP training."""
    if strains is None:
        strains = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05]
    rng = np.random.default_rng(seed)
    results = []
    cell0 = atoms.get_cell()
    for i in range(n_structures):
        a = atoms.copy()
        # Random strain from list
        e = float(rng.choice(strains))
        scale = (1.0 + e) ** (1.0 / 3.0)
        a.set_cell(cell0 * scale, scale_atoms=True)
        # Rattle
        displacements = rng.normal(0, stdev, (len(a), 3))
        a.set_positions(a.get_positions() + displacements)
        a.info["htp_id"] = i
        a.info["strategy"] = "rattle_strain"
        a.info["stdev"] = stdev
        a.info["strain"] = e
        results.append(a)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. Surface rattle
# ══════════════════════════════════════════════════════════════════════════════

def surface_rattle(
    atoms: Atoms,
    n_structures: int,
    stdev: float = 0.15,
    n_surface_layers: int = 2,
    seed: int = 42,
) -> List[Atoms]:
    """Rattle only surface atoms (top n_surface_layers), keep bulk fixed.

    Surface atoms are identified as the n_surface_layers*n_atoms_per_layer atoms
    with highest z-coordinate.
    """
    rng = np.random.default_rng(seed)
    positions = atoms.get_positions()
    z = positions[:, 2]
    z_sorted = np.sort(z)[::-1]

    # Estimate atoms per layer: use histogram of z to find layer spacing
    n_atoms = len(atoms)
    if n_atoms == 0:
        return []

    # Find surface atoms as those within top z range
    # Use a simple approach: atoms with z > (z_max - n_layers * layer_height)
    z_max = z.max()
    z_min = z.min()
    z_range = z_max - z_min
    if z_range < 0.5:
        # Flat structure or molecule — rattle all
        surface_mask = np.ones(n_atoms, dtype=bool)
    else:
        # Estimate layer height
        layer_height = z_range / max(1, n_atoms ** (1 / 3) - 1) if n_atoms > 1 else z_range
        # Take top n_surface_layers * layer_height
        threshold = z_max - n_surface_layers * layer_height * 1.5
        surface_mask = z > threshold

    results = []
    for i in range(n_structures):
        a = atoms.copy()
        pos = a.get_positions().copy()
        displacements = rng.normal(0, stdev, (n_atoms, 3))
        pos[surface_mask] += displacements[surface_mask]
        a.set_positions(pos)
        a.info["htp_id"] = i
        a.info["strategy"] = "surface_rattle"
        a.info["stdev"] = stdev
        a.info["n_surface_atoms"] = int(surface_mask.sum())
        results.append(a)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. Temperature rattle (Einstein model)
# ══════════════════════════════════════════════════════════════════════════════

def temperature_rattle(
    atoms: Atoms,
    T: float,
    n_structures: int,
    omega_thz: float = 5.0,
    seed: int = 42,
) -> List[Atoms]:
    """Boltzmann-weighted displacements at temperature T using Einstein model.

    Displacement stdev estimated from Einstein model:
        σ = sqrt(k_B * T / (m * ω²))
    Default ω = 5 THz is typical for metals.

    Parameters
    ----------
    atoms      : base structure
    T          : temperature in Kelvin
    n_structures : number of structures to generate
    omega_thz  : Einstein frequency in THz (default 5.0 for metals)
    seed       : random seed
    """
    from ase.data import atomic_masses

    k_B = 8.617333e-5   # eV/K
    # Convert omega from THz to rad/s then to eV/Å² mass units
    # omega [rad/s] = 2*pi*omega_thz*1e12
    # sigma² [Å²] = k_B*T / (m * omega²)
    # m in kg, omega in rad/s → sigma in m → convert to Å
    # Use ASE mass units: mass in amu, 1 amu = 1.66054e-27 kg
    omega_rad_per_s = 2 * math.pi * omega_thz * 1e12
    k_B_J = 1.380649e-23  # J/K
    amu_kg = 1.66054e-27

    rng = np.random.default_rng(seed)
    results = []
    masses = atoms.get_masses()  # amu per atom

    for i in range(n_structures):
        a = atoms.copy()
        pos = a.get_positions().copy()
        for j, m_amu in enumerate(masses):
            m_kg = m_amu * amu_kg
            variance = k_B_J * T / (m_kg * omega_rad_per_s ** 2)
            sigma_m = math.sqrt(max(0.0, variance))
            sigma_ang = sigma_m * 1e10  # convert m → Å
            disp = rng.normal(0.0, sigma_ang, 3)
            pos[j] += disp
        a.set_positions(pos)
        a.info["htp_id"] = i
        a.info["strategy"] = "temperature_rattle"
        a.info["temperature_K"] = T
        a.info["omega_thz"] = omega_thz
        results.append(a)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. Alloy configurations
# ══════════════════════════════════════════════════════════════════════════════

def alloy_configs(
    atoms: Atoms,
    host: str,
    dopant: str,
    concentrations: Optional[List[float]] = None,
    n_per_conc: int = 5,
    seed: int = 42,
) -> List[Atoms]:
    """For binary alloy A_x B_(1-x), generate random occupancy configurations.

    For each concentration in concentrations, randomly replace a fraction of
    host atoms with dopant atoms.

    Parameters
    ----------
    atoms          : base structure (all host atoms, or mixed)
    host           : host element symbol (e.g. 'Cu')
    dopant         : dopant element symbol (e.g. 'Ni')
    concentrations : list of dopant fractions, e.g. [0.1, 0.2, 0.3]
    n_per_conc     : number of random configs per concentration
    seed           : random seed
    """
    if concentrations is None:
        concentrations = [0.1, 0.2, 0.3]
    rng = np.random.default_rng(seed)

    # Find indices of host atoms
    host_indices = [i for i, sym in enumerate(atoms.get_chemical_symbols())
                    if sym == host]
    if not host_indices:
        raise ValueError(f"No atoms of element '{host}' found in structure.")

    results = []
    htp_id = 0
    for conc in concentrations:
        n_replace = max(1, round(conc * len(host_indices)))
        for _ in range(n_per_conc):
            a = atoms.copy()
            chosen = rng.choice(host_indices, size=min(n_replace, len(host_indices)),
                                replace=False)
            syms = list(a.get_chemical_symbols())
            for idx in chosen:
                syms[idx] = dopant
            a.set_chemical_symbols(syms)
            a.info["htp_id"] = htp_id
            a.info["strategy"] = "alloy_configs"
            a.info["host"] = host
            a.info["dopant"] = dopant
            a.info["concentration"] = float(conc)
            a.info["n_dopants"] = int(len(chosen))
            results.append(a)
            htp_id += 1
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7. Vacancy configurations
# ══════════════════════════════════════════════════════════════════════════════

def vacancy_configs(
    atoms: Atoms,
    n_vacancies: int = 1,
    n_structures: int = 10,
    seed: int = 42,
) -> List[Atoms]:
    """Randomly remove n_vacancies atoms from the structure.

    Parameters
    ----------
    atoms        : base structure
    n_vacancies  : number of atoms to remove per structure
    n_structures : number of vacancy configurations to generate
    seed         : random seed
    """
    rng = np.random.default_rng(seed)
    n_atoms = len(atoms)
    if n_vacancies >= n_atoms:
        raise ValueError(
            f"n_vacancies={n_vacancies} must be < n_atoms={n_atoms}."
        )
    results = []
    for i in range(n_structures):
        a = atoms.copy()
        vacancy_indices = sorted(
            rng.choice(n_atoms, size=n_vacancies, replace=False).tolist(),
            reverse=True,
        )
        for idx in vacancy_indices:
            del a[idx]
        a.info["htp_id"] = i
        a.info["strategy"] = "vacancy_configs"
        a.info["n_vacancies"] = n_vacancies
        a.info["vacancy_indices"] = vacancy_indices
        results.append(a)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# HTPDataset — manages the collection of structures
# ══════════════════════════════════════════════════════════════════════════════

class HTPDataset:
    """Manages a collection of structures for NNP training.

    Uses ASE database (ase.db) as persistent backend.
    Each row stores the structure plus metadata in row.data and row.key_value_pairs.

    Status values: 'pending' | 'done' | 'failed'
    """

    def __init__(self, db_path: str = "htp_dataset.db"):
        self.db_path = db_path

    def _db(self):
        return _ase_db_connect(self.db_path)

    # ------------------------------------------------------------------
    def add_structures(self, structures: List[Atoms], source: str = "") -> List[int]:
        """Add structures with status='pending'. Returns list of db row ids."""
        db = self._db()
        ids = []
        for atoms in structures:
            htp_id = atoms.info.get("htp_id", 0)
            strategy = atoms.info.get("strategy", source or "unknown")
            row_id = db.write(
                atoms,
                key_value_pairs={
                    "status": "pending",
                    "source": source,
                    "strategy": strategy,
                    "htp_id": htp_id,
                },
            )
            ids.append(row_id)
        return ids

    # ------------------------------------------------------------------
    def get_pending(self, limit: int = 100) -> List[Tuple[int, Atoms]]:
        """Return (id, atoms) pairs with status='pending'."""
        db = self._db()
        results = []
        for row in db.select(status="pending", limit=limit):
            results.append((row.id, row.toatoms()))
        return results

    # ------------------------------------------------------------------
    def mark_done(
        self,
        row_id: int,
        energy: float,
        forces: np.ndarray,
        stress: Optional[np.ndarray] = None,
    ) -> None:
        """Update row: status='done', store energy/forces/stress."""
        db = self._db()
        data: Dict[str, Any] = {
            "energy": float(energy),
            "forces": forces.tolist() if hasattr(forces, "tolist") else forces,
        }
        if stress is not None:
            data["stress"] = (
                stress.tolist() if hasattr(stress, "tolist") else list(stress)
            )
        db.update(
            row_id,
            status="done",
            data=data,
        )

    # ------------------------------------------------------------------
    def mark_failed(self, row_id: int, reason: str) -> None:
        """Update row: status='failed', store failure reason."""
        db = self._db()
        db.update(row_id, status="failed", data={"fail_reason": reason})

    # ------------------------------------------------------------------
    def export_extxyz(self, output_path: str, only_done: bool = True) -> int:
        """Export structures to extXYZ format for NNP training.

        Each frame includes atoms.info['energy'], atoms.arrays['forces'],
        and optionally atoms.info['stress'].

        Returns the number of structures exported.
        """
        db = self._db()
        selector = {"status": "done"} if only_done else {}
        frames = []
        for row in db.select(**selector):
            atoms = row.toatoms()
            data = row.data or {}
            # Use SinglePointCalculator — required for proper extxyz round-trip in ASE 3.22+
            calc_kwargs: Dict[str, Any] = {}
            if "energy" in data:
                calc_kwargs["energy"] = float(data["energy"])
            if "forces" in data:
                calc_kwargs["forces"] = np.array(data["forces"], dtype=float)
            if "stress" in data:
                # Voigt 6-vector required by ASE SinglePointCalculator
                s = np.array(data["stress"], dtype=float)
                if s.shape == (3, 3):
                    # Convert 3×3 → Voigt 6: [xx, yy, zz, yz, xz, xy]
                    s = np.array([s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]])
                calc_kwargs["stress"] = s
            if calc_kwargs:
                from ase.calculators.singlepoint import SinglePointCalculator
                atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
            frames.append(atoms)

        if not frames:
            return 0

        _ase_write(output_path, frames, format="extxyz")
        return len(frames)

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        db = self._db()
        all_rows = list(db.select())
        total = len(all_rows)
        pending = sum(1 for r in all_rows if r.get("status") == "pending")
        done = sum(1 for r in all_rows if r.get("status") == "done")
        failed = sum(1 for r in all_rows if r.get("status") == "failed")

        elements: set = set()
        energies = []
        for r in all_rows:
            atoms = r.toatoms()
            elements.update(atoms.get_chemical_symbols())
            data = r.data or {}
            if "energy" in data:
                energies.append(float(data["energy"]))

        result: Dict[str, Any] = {
            "total": total,
            "pending": pending,
            "done": done,
            "failed": failed,
            "n_elements": len(elements),
            "elements": sorted(elements),
            "db_path": self.db_path,
        }
        if energies:
            result["mean_energy"] = float(np.mean(energies))
            result["min_energy"] = float(np.min(energies))
            result["max_energy"] = float(np.max(energies))
            result["std_energy"] = float(np.std(energies))
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Main generation function
# ══════════════════════════════════════════════════════════════════════════════

def generate_htp_dataset(
    base_structures: List[Dict],
    strategy: str = "rattle",
    n_total: int = 1000,
    db_path: str = "htp_dataset.db",
    **strategy_kwargs: Any,
) -> Dict[str, Any]:
    """Generate n_total diverse structures from base_structures using strategy.

    Parameters
    ----------
    base_structures : list of dicts with key 'poscar' (POSCAR string) and
                      optional 'label'
    strategy        : one of 'rattle', 'strain', 'rattle_strain',
                      'surface_rattle', 'temperature_rattle', 'alloy', 'vacancy'
    n_total         : total number of structures to generate
    db_path         : path to ASE database file
    **strategy_kwargs : passed through to the strategy function

    Returns
    -------
    dict with keys: ok, n_generated, db_path, stats
    """
    if not base_structures:
        return {"ok": False, "error": "No base structures provided."}

    # Parse base structures from POSCAR strings
    parsed: List[Atoms] = []
    for entry in base_structures:
        poscar_str = entry.get("poscar", "")
        if not poscar_str:
            continue
        try:
            buf = _io.StringIO(poscar_str)
            atoms = _read_vasp(buf)
            atoms.info["label"] = entry.get("label", "")
            parsed.append(atoms)
        except Exception as exc:
            return {"ok": False, "error": f"Failed to parse POSCAR: {exc}"}

    if not parsed:
        return {"ok": False, "error": "No valid POSCAR structures found."}

    n_per_base = max(1, n_total // len(parsed))
    all_structures: List[Atoms] = []

    strat = strategy.lower().strip()

    for atoms in parsed:
        try:
            if strat == "rattle":
                structs = rattle_structures(
                    atoms,
                    n_structures=n_per_base,
                    stdev=strategy_kwargs.get("stdev", 0.1),
                    seed=strategy_kwargs.get("seed", 42),
                )
            elif strat == "strain":
                structs = strain_structures(
                    atoms,
                    strains=strategy_kwargs.get("strains", None),
                )
            elif strat == "rattle_strain":
                structs = rattle_strain_structures(
                    atoms,
                    n_structures=n_per_base,
                    stdev=strategy_kwargs.get("stdev", 0.1),
                    strains=strategy_kwargs.get("strains", None),
                    seed=strategy_kwargs.get("seed", 42),
                )
            elif strat == "surface_rattle":
                structs = surface_rattle(
                    atoms,
                    n_structures=n_per_base,
                    stdev=strategy_kwargs.get("stdev", 0.15),
                    n_surface_layers=strategy_kwargs.get("n_surface_layers", 2),
                    seed=strategy_kwargs.get("seed", 42),
                )
            elif strat == "temperature_rattle":
                structs = temperature_rattle(
                    atoms,
                    T=strategy_kwargs.get("temperature", strategy_kwargs.get("T", 300.0)),
                    n_structures=n_per_base,
                    omega_thz=strategy_kwargs.get("omega_thz", 5.0),
                    seed=strategy_kwargs.get("seed", 42),
                )
            elif strat in ("alloy", "alloy_configs"):
                structs = alloy_configs(
                    atoms,
                    host=strategy_kwargs["host"],
                    dopant=strategy_kwargs["dopant"],
                    concentrations=strategy_kwargs.get("concentrations", None),
                    n_per_conc=strategy_kwargs.get("n_per_conc", 5),
                    seed=strategy_kwargs.get("seed", 42),
                )
            elif strat in ("vacancy", "vacancy_configs"):
                structs = vacancy_configs(
                    atoms,
                    n_vacancies=strategy_kwargs.get("n_vacancies", 1),
                    n_structures=n_per_base,
                    seed=strategy_kwargs.get("seed", 42),
                )
            else:
                return {"ok": False, "error": (
                    f"Unknown strategy '{strategy}'. "
                    "Choose from: rattle, strain, rattle_strain, surface_rattle, "
                    "temperature_rattle, alloy, vacancy."
                )}
        except Exception as exc:
            return {"ok": False, "error": f"Strategy '{strategy}' failed: {exc}"}

        all_structures.extend(structs)

    # Trim to n_total if we over-generated
    all_structures = all_structures[:n_total]

    dataset = HTPDataset(db_path=db_path)
    ids = dataset.add_structures(all_structures, source=strategy)
    stats = dataset.stats()

    # ── Mirror to PostgreSQL so the LLM can see HTP history ───────────────
    pg_run_id = _sync_htp_to_postgres(
        base_structures=base_structures,
        all_structures=all_structures,
        strategy=strategy,
        db_path=db_path,
        stats=stats,
        strategy_kwargs=strategy_kwargs,
    )

    return {
        "ok": True,
        "n_generated": len(all_structures),
        "n_added": len(ids),
        "db_path": db_path,
        "stats": stats,
        "pg_run_id": pg_run_id,
    }


def _sync_htp_to_postgres(
    base_structures: List[Dict],
    all_structures: List[Atoms],
    strategy: str,
    db_path: str,
    stats: Dict[str, Any],
    strategy_kwargs: Dict[str, Any],
) -> Optional[int]:
    """
    Mirror the newly generated HTP run into PostgreSQL (HTPRun + HTPStructure).
    Returns the HTPRun.id or None if DB unavailable.

    Runs synchronously via asyncio.run() / event loop detection.
    """
    async def _async_sync() -> Optional[int]:
        try:
            from server.db import AsyncSessionLocal, HTPRun, HTPStructure
        except Exception:
            return None

        try:
            async with AsyncSessionLocal() as db:
                base_labels = [
                    {
                        "label": s.get("label", ""),
                        "formula": s.get("formula", ""),
                    }
                    for s in base_structures
                ]
                run = HTPRun(
                    db_path=db_path,
                    strategy=strategy,
                    n_total=len(all_structures),
                    n_done=stats.get("n_done", 0),
                    n_failed=stats.get("n_failed", 0),
                    encut=strategy_kwargs.get("encut", 450),
                    kpoints=str(strategy_kwargs.get("kpoints", "4 4 1")),
                    base_labels=base_labels,
                    extra_kwargs={
                        k: v for k, v in strategy_kwargs.items()
                        if k not in ("encut", "kpoints")
                    },
                    status="generating",
                )
                db.add(run)
                await db.flush()

                for atoms in all_structures[:500]:   # cap to avoid huge inserts
                    struct = HTPStructure(
                        run_id=run.id,
                        label=atoms.info.get("label", ""),
                        formula=atoms.get_chemical_formula(),
                        n_atoms=len(atoms),
                        strategy=atoms.info.get("strategy", strategy),
                        status="pending",
                    )
                    db.add(struct)

                run.status = "pending"
                await db.commit()
                return run.id
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("_sync_htp_to_postgres failed: %s", exc)
            return None

    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # Inside async context — schedule as task; return None immediately
        loop.create_task(_async_sync())
        return None
    except RuntimeError:
        try:
            return asyncio.run(_async_sync())
        except Exception:
            return None
