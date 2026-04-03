# server/execution/structure_agent.py
# -*- coding: utf-8 -*-
"""
StructureAgent — interactive ASE+pymatgen environment for building catalytic surfaces,
finding adsorption sites, placing adsorbates, and recording structures for T2S training.
"""
from __future__ import annotations

import json
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ase import Atoms
from ase.io import write, read
from ase.data import chemical_symbols, covalent_radii, atomic_numbers
from ase.build import (
    fcc111, fcc100, fcc110,
    bcc110, bcc100, bcc111,
    hcp0001, hcp10m10,
    bulk, surface,
    add_adsorbate,
)
import ase.constraints

# Common SMILES for gas-phase DFT molecules
_SMILES_MAP: Dict[str, str] = {
    "C4H10": "CCCC",       # n-butane
    "C4H8":  "C=CCC",      # 1-butene (default; also CC=CC for 2-butene)
    "C4H8_2": "CC=CC",     # 2-butene
    "C3H8":  "CCC",        # propane
    "C3H6":  "C=CCC",      # propene
    "C2H6":  "CC",         # ethane
    "C2H4":  "C=C",        # ethylene
    "CH4":   "C",          # methane
    "CO2":   "O=C=O",
    "CO":    "[C-]#[O+]",
    "H2O":   "O",
    "NH3":   "N",
    "N2":    "N#N",
    "H2":    "[HH]",
    "Acetone": "CC(=O)C",
    "Ethanol": "CCO",
    "Methanol": "CO",
}

try:
    from pymatgen.core import Structure as PmgStructure
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.io.ase import AseAtomsAdaptor
    _HAS_PMG = True
except ImportError:
    _HAS_PMG = False

# =========================================================================
# Constants — lattice parameters & crystal system guesses
# =========================================================================

_A0: Dict[str, Any] = {
    # FCC (a in Å)
    "Cu": 3.615, "Pt": 3.924, "Ni": 3.523, "Pd": 3.889,
    "Ag": 4.086, "Au": 4.078, "Al": 4.046, "Ir": 3.839,
    "Rh": 3.803,
    # BCC
    "Fe": 2.867, "W":  3.165, "Mo": 3.147, "Cr": 2.884,
    "V":  3.024, "Nb": 3.294, "Ta": 3.301,
    # HCP (a, c/a) tuple
    "Ru": (2.706, 1.582), "Os": (2.734, 1.579),
    "Ti": (2.951, 1.586), "Zn": (2.665, 1.856), "Mg": (3.209, 1.624),
}

_CRYSTAL_SYSTEM: Dict[str, str] = {
    "Cu": "fcc", "Pt": "fcc", "Ni": "fcc", "Pd": "fcc",
    "Ag": "fcc", "Au": "fcc", "Al": "fcc", "Ir": "fcc", "Rh": "fcc",
    "Fe": "bcc", "W":  "bcc", "Mo": "bcc", "Cr": "bcc",
    "V":  "bcc", "Nb": "bcc", "Ta": "bcc",
    "Ru": "hcp", "Os": "hcp", "Ti": "hcp", "Zn": "hcp", "Mg": "hcp",
}

# Common adsorbate definitions: symbol -> list of (element, [dx,dy,dz] from anchor)
_ADSORBATES: Dict[str, List[Tuple[str, List[float]]]] = {
    "H":    [("H", [0, 0, 0])],
    "C":    [("C", [0, 0, 0])],
    "N":    [("N", [0, 0, 0])],
    "O":    [("O", [0, 0, 0])],
    "S":    [("S", [0, 0, 0])],
    "CO":   [("C", [0, 0, 0]), ("O", [0, 0, 1.15])],
    "OH":   [("O", [0, 0, 0]), ("H", [0, 0, 0.97])],
    "NO":   [("N", [0, 0, 0]), ("O", [0, 0, 1.15])],
    "CO2":  [("C", [0, 0, 0]), ("O", [0, 0, 1.16]), ("O", [0, 0, -1.16])],
    "CHO":  [("C", [0, 0, 0]), ("H", [0, 1.09, 0.50]), ("O", [0, 0, 1.21])],
    "COOH": [("C", [0, 0, 0]), ("O", [0, 0, 1.21]), ("O", [0, 1.33, 0.50]), ("H", [0, 2.00, 0.90])],
    "NH3":  [("N", [0, 0, 0]), ("H", [0.94, 0, 0.33]), ("H", [-0.47, 0.82, 0.33]), ("H", [-0.47, -0.82, 0.33])],
    "N2":   [("N", [0, 0, 0]), ("N", [0, 0, 1.10])],
    "H2O":  [("O", [0, 0, 0]), ("H", [0.76, 0, 0.59]), ("H", [-0.76, 0, 0.59])],
    "OOH":  [("O", [0, 0, 0]), ("O", [0, 0, 1.32]), ("H", [0, 0.96, 1.63])],
    "CH":   [("C", [0, 0, 0]), ("H", [0, 0, 1.09])],
    "CH2":  [("C", [0, 0, 0]), ("H", [0.63, 0, 0.88]), ("H", [-0.63, 0, 0.88])],
    "CH3":  [("C", [0, 0, 0]), ("H", [0, 0, 1.09]), ("H", [0.71, 0.71, 0.36]), ("H", [-0.71, 0.71, 0.36])],
    "NH":   [("N", [0, 0, 0]), ("H", [0, 0, 1.02])],
    "NH2":  [("N", [0, 0, 0]), ("H", [0.50, 0, 0.86]), ("H", [-0.50, 0, 0.86])],
}

_VALID_ELEMS = set(chemical_symbols)

# Element colors for 3D visualization
_ELEM_COLORS: Dict[str, str] = {
    "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "S": "#FFFF30", "P": "#FF8000", "F": "#90E050", "Cl": "#1FF01F",
    "Cu": "#FF8C00", "Pt": "#808090", "Ni": "#50D050", "Pd": "#006985",
    "Au": "#FFD123", "Ag": "#C0C0C0", "Fe": "#E06633", "Co": "#F090A0",
    "Ru": "#248F8F", "Ir": "#175487", "Rh": "#0AABFB", "Mo": "#54B5B5",
    "W":  "#2194D6", "Cr": "#8A99C7", "default": "#AAAAAA",
}

# =========================================================================
# Parsing helpers
# =========================================================================

def _normalize_element(raw: str, default: str = "Cu") -> str:
    """Robustly parse element symbol from user input.

    Handles: "pt" → "Pt", "PT" → "Pt", "Pt111" → "Pt", "platinum" prefix → "Pt"
    """
    s = (raw or "").strip()
    if not s:
        return default

    # Strip trailing digits / surface notation like "111", "(111)", "111 surface"
    s_clean = re.sub(r"[\d()\[\]]+.*$", "", s).strip()
    if not s_clean:
        s_clean = s

    # Try title-case of cleaned string (pt→Pt, PT→Pt, ag→Ag)
    candidate = s_clean.title()
    if candidate in _VALID_ELEMS:
        return candidate

    # Try first 2 chars title-cased, then 1 char title-cased
    for n in (2, 1):
        if len(s_clean) >= n:
            c = s_clean[:n].title()
            if c in _VALID_ELEMS:
                return c

    # Regex fallback: find first [A-Z][a-z]? pattern in title-cased version
    m = re.search(r"[A-Z][a-z]?", s_clean.title())
    if m and m.group(0) in _VALID_ELEMS:
        return m.group(0)

    return default

def _parse_miller(payload: Dict[str, Any]) -> Tuple[int, int, int]:
    mi = payload.get("miller_index")
    if isinstance(mi, (list, tuple)) and len(mi) >= 3:
        return int(mi[0]), int(mi[1]), int(mi[2])
    txt = str(mi or payload.get("facet") or "").strip()
    txt = txt.replace("(", "").replace(")", "").replace(",", " ")
    parts = txt.split()
    if len(parts) == 3:
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            pass
    if len(parts) == 1 and len(parts[0]) == 3 and parts[0].isdigit():
        return int(parts[0][0]), int(parts[0][1]), int(parts[0][2])
    return (1, 1, 1)

def _parse_supercell(s: Optional[str]) -> Tuple[int, int, int]:
    txt = (s or "4x4x1").lower().replace("*", "x").replace("×", "x")
    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\s*$", txt)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m2 = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", txt)
    if m2:
        return int(m2.group(1)), int(m2.group(2)), 1
    return (4, 4, 1)

def _guess_crystal_system(element: str, payload: Dict[str, Any]) -> str:
    cs = (payload.get("crystal_system") or "").lower()
    if cs in ("fcc", "bcc", "hcp"):
        return cs
    return _CRYSTAL_SYSTEM.get(element, "fcc")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# =========================================================================
# Slab builders
# =========================================================================

def _build_slab(
    element: str, crystal_system: str,
    h: int, k: int, l: int,
    nx: int, ny: int, nlayers: int, vacuum: float,
) -> Atoms:
    facet = f"{h}{k}{l}"
    a0_val = _A0.get(element)

    if crystal_system == "fcc":
        a = a0_val if isinstance(a0_val, (int, float)) else 3.615
        kw = dict(size=(nx, ny, nlayers), a=a, vacuum=vacuum, orthogonal=True)
        if facet == "111":   return fcc111(element, **kw)
        if facet == "100":   return fcc100(element, **kw)
        if facet == "110":   return fcc110(element, **kw)
        b = bulk(element, crystalstructure="fcc", a=a, cubic=True)
        return _generic_slab(b, (h, k, l), nlayers, vacuum, nx, ny)

    elif crystal_system == "bcc":
        a = a0_val if isinstance(a0_val, (int, float)) else 2.87
        kw = dict(size=(nx, ny, nlayers), a=a, vacuum=vacuum, orthogonal=True)
        if facet == "110":   return bcc110(element, **kw)
        if facet == "100":   return bcc100(element, **kw)
        if facet == "111":   return bcc111(element, **kw)
        b = bulk(element, crystalstructure="bcc", a=a, cubic=True)
        return _generic_slab(b, (h, k, l), nlayers, vacuum, nx, ny)

    elif crystal_system == "hcp":
        if isinstance(a0_val, tuple):
            a, ca = a0_val
        else:
            a, ca = (a0_val or 2.71), 1.58
        c = a * ca
        kw = dict(size=(nx, ny, nlayers), a=a, c=c, vacuum=vacuum, orthogonal=True)
        if facet in ("0001", "001"):    return hcp0001(element, **kw)
        if facet in ("1010", "10m10"): return hcp10m10(element, **kw)
        b = bulk(element, crystalstructure="hcp", a=a, c=c)
        return _generic_slab(b, (h, k, l), nlayers, vacuum, nx, ny)

    else:
        b = bulk(element, cubic=True)
        return _generic_slab(b, (h, k, l), nlayers, vacuum, nx, ny)


def _generic_slab(bulk_atoms: Atoms, miller: Tuple[int, int, int],
                  nlayers: int, vacuum: float, nx: int, ny: int) -> Atoms:
    try:
        slab = surface(bulk_atoms, miller, nlayers, vacuum=vacuum)
        if nx > 1 or ny > 1:
            slab = slab.repeat([nx, ny, 1])
        return slab
    except Exception:
        return bulk_atoms.repeat([nx, ny, nlayers])

# =========================================================================
# Adsorption sites
# =========================================================================

def find_adsorption_sites_ase(atoms: Atoms, height: float = 2.0) -> List[Dict[str, Any]]:
    """Find all adsorption sites using pymatgen AdsorbateSiteFinder (or fallback top sites)."""
    if _HAS_PMG:
        try:
            adaptor = AseAtomsAdaptor()
            structure = adaptor.get_structure(atoms)
            asf = AdsorbateSiteFinder(structure)
            all_sites = asf.find_adsorption_sites(distance=0.1)
            sites = []
            for site_type, coords_list in all_sites.items():
                for coords in coords_list:
                    c = coords.tolist() if hasattr(coords, "tolist") else list(coords)
                    sites.append({"type": site_type, "position": c})
            return sites
        except Exception:
            pass

    # Fallback: top sites only
    pos = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    z_max = pos[:, 2].max()
    sites = []
    for i, (sym, p) in enumerate(zip(syms, pos)):
        if p[2] > z_max - 1.5:
            sites.append({
                "type": "top",
                "position": [float(p[0]), float(p[1]), float(p[2] + height)],
                "atom_index": i,
                "symbol": sym,
            })
    return sites


def place_adsorbate(slab: Atoms, site_position: List[float], adsorbate: str) -> Atoms:
    """Place an adsorbate molecule at the given 3D position."""
    mol_def = _ADSORBATES.get(adsorbate.upper()) or _ADSORBATES.get(adsorbate)
    if mol_def is None:
        # Try single atom
        ads = Atoms(adsorbate, positions=[[0, 0, 0]])
    else:
        syms = [a[0] for a in mol_def]
        offsets = [a[1] for a in mol_def]
        ads = Atoms(symbols=syms, positions=offsets)

    # Translate anchor atom to site position
    anchor = ads.get_positions()[0].copy()
    ads.translate([site_position[0] - anchor[0],
                   site_position[1] - anchor[1],
                   site_position[2] - anchor[2]])
    return slab.copy() + ads


def generate_configurations(
    slab: Atoms, adsorbate: str,
    sites: Optional[List[Dict]] = None,
    max_configs: int = 4,
) -> List[Dict[str, Any]]:
    """Generate multiple adsorption configurations (one per unique site type)."""
    if sites is None:
        sites = find_adsorption_sites_ase(slab)
    priority = {"top": 0, "bridge": 1, "hollow_fcc": 2, "hollow_hcp": 2,
                "hollow": 2, "hollow_h": 3}
    sites_sorted = sorted(sites, key=lambda s: priority.get(s.get("type", "top"), 5))
    configs, seen = [], set()
    for site in sites_sorted:
        if len(configs) >= max_configs:
            break
        stype = site.get("type", "top")
        if stype in seen and len(configs) >= 2:
            continue
        try:
            combined = place_adsorbate(slab, site["position"], adsorbate)
            configs.append({"site_type": stype, "position": site["position"], "atoms": combined})
            seen.add(stype)
        except Exception:
            continue
    return configs


def _collect_unique_sites(slab: Atoms, height: float) -> List[Dict]:
    """
    Collect one representative (x,y) per unique site type from the surface,
    using pymatgen AdsorbateSiteFinder when available, with ASE fallback.
    Always returns at least top / bridge / hollow entries.
    The z-coordinate is set to surface_z_max + height by the caller.
    """
    # ── pymatgen path ─────────────────────────────────────────────────────────
    if _HAS_PMG:
        try:
            adaptor = AseAtomsAdaptor()
            structure = adaptor.get_structure(slab)
            asf = AdsorbateSiteFinder(structure)
            # use distance=0 so pymatgen gives us raw (x,y,z_surface) positions
            all_sites = asf.find_adsorption_sites(distance=0.0)

            # pymatgen type names → canonical names  ('all' is a combined list, skip it)
            _PMG_TYPE_MAP = {
                "ontop": "top", "on_top": "top",
                "bridge": "bridge",
                "hollow": "hollow_fcc",
                "hollow_fcc": "hollow_fcc", "hollow_hcp": "hollow_hcp",
            }
            _PMG_SKIP = {"all", "all_sites"}
            priority = {"top": 0, "bridge": 1, "hollow_fcc": 2, "hollow_hcp": 3}
            unique: Dict[str, Dict] = {}
            for pmg_type, coords_list in all_sites.items():
                if pmg_type in _PMG_SKIP:
                    continue
                ctype = _PMG_TYPE_MAP.get(pmg_type, pmg_type)
                if ctype not in unique and coords_list:
                    c = coords_list[0]
                    xy = c.tolist()[:2] if hasattr(c, "tolist") else list(c)[:2]
                    unique[ctype] = {"type": ctype, "xy": xy}

            if unique:
                return sorted(unique.values(),
                              key=lambda s: priority.get(s["type"], 5))
        except (ValueError, KeyError, TypeError):
            pass

    # ── ASE fallback: extract top-layer atoms as "top" sites ──────────────────
    pos  = slab.get_positions()
    z_max = pos[:, 2].max()
    top_atoms = [{"type": "top", "xy": [float(p[0]), float(p[1])]}
                 for p in pos if p[2] > z_max - 1.2]

    # Also add mid-points between adjacent top atoms as "bridge"
    bridge = []
    for i in range(len(top_atoms)):
        for j in range(i + 1, len(top_atoms)):
            xi, yi = top_atoms[i]["xy"]
            xj, yj = top_atoms[j]["xy"]
            d = ((xi - xj)**2 + (yi - yj)**2) ** 0.5
            if d < 3.5:   # Å, typical Pt-Pt bond
                bridge.append({"type": "bridge",
                                "xy": [(xi+xj)/2, (yi+yj)/2]})
                break

    sites: List[Dict] = []
    if top_atoms:   sites.append(top_atoms[len(top_atoms)//4])  # representative
    if bridge:      sites.append(bridge[0])
    # hollow_fcc: centroid of 3 top atoms
    if len(top_atoms) >= 3:
        xs = [t["xy"][0] for t in top_atoms[:3]]
        ys = [t["xy"][1] for t in top_atoms[:3]]
        sites.append({"type": "hollow_fcc",
                      "xy": [sum(xs)/3, sum(ys)/3]})
    return sites


def generate_ads_from_poscars(
    surface_poscar: str,
    mol_poscar: str,
    max_configs: int = 4,
    height: float = 2.0,
) -> Dict[str, Any]:
    """
    Generate adsorption configurations from surface + molecule POSCAR strings.

    Placement strategy:
      - Anchor atom = atom with lowest z in molecule (closest to surface)
      - Center molecule so anchor is at origin
      - For each unique site type (top / bridge / hollow_fcc / hollow_hcp):
          place anchor at (site_x, site_y, surface_z_max + height)
          then rotate 0° / 90° / 180° around z for variety
      - height directly controls the gap → changing it visually moves the molecule
    """
    import tempfile, os

    tmp_files = []
    def _write_tmp(content: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as f:
            f.write(content)
            tmp_files.append(f.name)
            return f.name

    try:
        slab = read(_write_tmp(surface_poscar), format="vasp")
        mol  = read(_write_tmp(mol_poscar),     format="vasp")

        # Surface z_max — anchor will be placed at this + height
        surf_z_max = float(slab.get_positions()[:, 2].max())

        # Get unique site (x, y) positions
        sites = _collect_unique_sites(slab, height)
        if not sites:
            return {"ok": False, "error": "No adsorption sites found.", "configs": []}

        # Center molecule with anchor atom at origin
        mol_pos = mol.get_positions()
        anchor_idx  = int(mol_pos[:, 2].argmin())   # lowest z = closest to surface
        anchor_xyz  = mol_pos[anchor_idx].copy()
        mol_centered = mol.copy()
        mol_centered.translate(-anchor_xyz)          # anchor now at (0,0,0)

        rotations = [0, 90, 180, 45]
        results: List[Dict] = []

        for site in sites:
            if len(results) >= max_configs:
                break
            sx, sy = site["xy"]
            target_z = surf_z_max + height           # ← height controls z directly

            for rot_deg in rotations:
                if len(results) >= max_configs:
                    break
                try:
                    mol_copy = mol_centered.copy()
                    if rot_deg:
                        mol_copy.rotate(rot_deg, "z", center=(0, 0, 0))
                    mol_copy.translate([sx, sy, target_z])   # anchor → site + height

                    combined = slab.copy() + mol_copy

                    with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR",
                                                    delete=False) as f2:
                        fname2 = f2.name; tmp_files.append(fname2)
                    write(fname2, combined, format="vasp", direct=True, vasp5=True)
                    poscar_out = open(fname2).read()

                    site_type  = site["type"]
                    label      = site_type + (f" rot{rot_deg}" if rot_deg else "")
                    cformula   = combined.get_chemical_formula()
                    cn_atoms   = len(combined)
                    # Derive surface / mol labels from POSCAR comment lines
                    surf_lbl = (surface_poscar.splitlines()[0].strip() or "surface")
                    mol_lbl  = (mol_poscar.splitlines()[0].strip()     or "molecule")
                    results.append({
                        "config_id":    len(results),
                        "site_type":    site_type,
                        "rotation":     rot_deg,
                        "label":        label,
                        "height":       height,
                        "n_atoms":      cn_atoms,
                        "formula":      cformula,
                        "poscar":       poscar_out,
                        "plot_png_b64": atoms_to_plot_b64(combined, rotations="10z,-80x,0y"),
                        # T2S fields
                        "description": describe_adsorption(
                            mol_lbl, surf_lbl, site_type, rot_deg,
                            height, cn_atoms, cformula,
                        ),
                        "ase_code": ase_code_adsorption(
                            surf_lbl, mol_lbl, site_type, rot_deg,
                            height, site.get("xy"),
                        ),
                        "structure_type": "adsorption",
                    })
                except (ValueError, KeyError, TypeError):
                    continue
            if len(results) >= max_configs:
                break

        if not results:
            return {"ok": False, "error": "Failed to place molecule at any site.", "configs": []}

        return {"ok": True, "n_configs": len(results), "configs": results}

    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc(), "configs": []}
    finally:
        for f in tmp_files:
            try: os.unlink(f)
            except Exception: pass


# =========================================================================
# Molecule builder (PubChem SMILES)
# =========================================================================

def _build_slab_direct(elem: str, csys: str, facet: str,
                        nx: int, ny: int, nlayers: int, vacuum: float) -> Atoms:
    """
    Build slab using ase.build functions directly (no orthogonal=True),
    exactly as the user's pattern:
        slab = fcc111('Cu', size=(4,4,3), vacuum=10)
    """
    a0_val = _A0.get(elem)

    if csys == "fcc":
        a = a0_val if isinstance(a0_val, (int, float)) else 3.615
        kw = dict(size=(nx, ny, nlayers), a=a, vacuum=vacuum)
        if facet == "111":  return fcc111(elem, **kw)
        if facet == "100":  return fcc100(elem, **kw)
        if facet == "110":  return fcc110(elem, **kw)
        b = bulk(elem, crystalstructure="fcc", a=a, cubic=True)
        return _generic_slab(b, tuple(int(c) for c in facet), nlayers, vacuum, nx, ny)

    elif csys == "bcc":
        a = a0_val if isinstance(a0_val, (int, float)) else 2.87
        kw = dict(size=(nx, ny, nlayers), a=a, vacuum=vacuum)
        if facet == "110":  return bcc110(elem, **kw)
        if facet == "100":  return bcc100(elem, **kw)
        if facet == "111":  return bcc111(elem, **kw)
        b = bulk(elem, crystalstructure="bcc", a=a, cubic=True)
        return _generic_slab(b, tuple(int(c) for c in facet), nlayers, vacuum, nx, ny)

    elif csys == "hcp":
        if isinstance(a0_val, tuple):
            a, ca = a0_val
        else:
            a, ca = (a0_val or 2.71), 1.58
        c = a * ca
        kw = dict(size=(nx, ny, nlayers), a=a, c=c, vacuum=vacuum)
        if facet in ("0001", "001"):    return hcp0001(elem, **kw)
        if facet in ("1010", "10m10"): return hcp10m10(elem, **kw)
        b = bulk(elem, crystalstructure="hcp", a=a, c=c)
        return _generic_slab(b, tuple(int(c) for c in facet), nlayers, vacuum, nx, ny)

    else:
        b = bulk(elem, cubic=True)
        return _generic_slab(b, tuple(int(c) for c in facet), nlayers, vacuum, nx, ny)


def build_surface_ase(
    element: str = "Cu",
    surface_type: str = "111",
    nx: int = 4,
    ny: int = 4,
    nlayers: int = 3,
    vacuum: float = 10.0,
    fix_bottom: bool = True,
) -> Dict[str, Any]:
    """
    Build a clean metal surface using the user-provided ASE pattern:
      slab = fcc111('Cu', size=(4,4,3), vacuum=10)
    Bottom layer fixed with FixAtoms.

    Returns {"ok": True, "poscar": str, "plot_png_b64": str, ...}
    """
    elem = _normalize_element(element)
    csys = _CRYSTAL_SYSTEM.get(elem, "fcc")
    # Normalise facet string: "111", "100", "110", "0001" etc.
    facet = str(surface_type).strip().replace("(", "").replace(")", "").replace(" ", "")
    try:
        slab = _build_slab_direct(elem, csys, facet, nx, ny, nlayers, vacuum)
    except (ValueError, KeyError, TypeError) as e:
        return {"ok": False, "error": str(e)}

    # Fix bottom layer (user's pattern)
    if fix_bottom:
        base_z = np.array([atom.z for atom in slab]).min()
        base_idx = [atom.index for atom in slab if abs(base_z - atom.z) < 0.1]
        slab.set_constraint(ase.constraints.FixAtoms(indices=base_idx))

    # Write POSCAR string
    import io, tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as f:
        fname = f.name
    try:
        write(fname, slab, format="vasp", direct=True, vasp5=True)
        poscar = open(fname).read()
    finally:
        try: os.unlink(fname)
        except OSError: pass

    _label   = f"{elem}({facet})-{nx}x{ny}x{nlayers}"
    _formula = slab.get_chemical_formula()
    _n_atoms = len(slab)
    return {
        "ok": True,
        "element": elem,
        "crystal_system": csys,
        "surface_type": facet,
        "size": [nx, ny, nlayers],
        "vacuum": vacuum,
        "n_atoms": _n_atoms,
        "formula": _formula,
        "label": _label,
        "poscar": poscar,
        "viz": atoms_to_viz_json(slab),
        "plot_png_b64": atoms_to_plot_b64(slab, rotations="10z,-80x,0y"),
        "atoms": slab,
        # T2S fields
        "description": describe_surface(elem, facet, nx, ny, nlayers, vacuum,
                                        _n_atoms, _formula, csys),
        "ase_code": ase_code_surface(elem, facet, nx, ny, nlayers, vacuum, csys),
        "structure_type": "surface",
    }


def build_molecule_pubchem(
    smiles: str = "CCCC",
    cell_size: float = 20.0,
    label: str = "",
) -> Dict[str, Any]:
    """
    Build a gas-phase molecule from PubChem using SMILES string.
    Uses the user-provided pattern:
      mol = pubchem_atoms_search(smiles='CC(=O)C')
      mol.set_cell([20,20,20]); mol.center()

    Returns {"ok": True, "atoms": Atoms, "poscar": str, "viz": dict, ...}
    """
    try:
        from ase.data.pubchem import pubchem_atoms_search
    except ImportError:
        return {"ok": False, "error": "ase.data.pubchem not available — install ase>=3.22"}

    # Try SMILES lookup first; also check common name map
    actual_smiles = _SMILES_MAP.get(smiles, smiles)
    try:
        mol = pubchem_atoms_search(smiles=actual_smiles)
    except (ValueError, KeyError, TypeError) as e:
        return {"ok": False, "error": f"PubChem search failed for SMILES '{actual_smiles}': {e}"}

    if mol is None:
        return {"ok": False, "error": f"No molecule found for SMILES '{actual_smiles}'"}

    mol.set_cell([cell_size, cell_size, cell_size])
    mol.center()

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as f:
        fname = f.name
    try:
        write(fname, mol, format="vasp", direct=False, vasp5=True)
        poscar = open(fname).read()
    finally:
        try: os.unlink(fname)
        except OSError: pass

    formula   = mol.get_chemical_formula()
    _label    = label or formula
    _n_atoms  = len(mol)
    return {
        "ok": True,
        "smiles": actual_smiles,
        "label": _label,
        "formula": formula,
        "n_atoms": _n_atoms,
        "cell_size": cell_size,
        "poscar": poscar,
        "viz": atoms_to_viz_json(mol),
        "plot_png_b64": atoms_to_plot_b64(mol, rotations="0x,0y,0z"),
        "atoms": mol,
        # T2S fields
        "description": describe_molecule(_label, actual_smiles, formula, _n_atoms, cell_size),
        "ase_code": ase_code_molecule(actual_smiles, _label, cell_size),
        "structure_type": "molecule",
    }


# =========================================================================
# Slab manipulation utilities
# =========================================================================

def _poscar_to_atoms(poscar: str) -> Atoms:
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as f:
        f.write(poscar); fname = f.name
    try:
        return read(fname, format="vasp")
    finally:
        try: os.unlink(fname)
        except OSError: pass


def _atoms_to_poscar(atoms: Atoms) -> str:
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as f:
        fname = f.name
    try:
        write(fname, atoms, format="vasp", direct=True, vasp5=True)
        return open(fname).read()
    finally:
        try: os.unlink(fname)
        except OSError: pass


def _detect_layers(atoms: Atoms, tol: float = 0.5) -> List[List[int]]:
    """Group atom indices into layers by z-coordinate clustering (gap > tol Å)."""
    zs     = atoms.get_positions()[:, 2]
    order  = np.argsort(zs)
    layers: List[List[int]] = [[order[0]]]
    for idx in order[1:]:
        if zs[idx] - zs[layers[-1][-1]] > tol:
            layers.append([])
        layers[-1].append(idx)
    return layers


def _result(atoms: Atoms, **extra) -> Dict[str, Any]:
    """Build a standard result dict from an Atoms object."""
    return {
        "ok": True,
        "poscar": _atoms_to_poscar(atoms),
        "plot_png_b64": atoms_to_plot_b64(atoms, rotations="10z,-80x,0y"),
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        **extra,
    }


def slab_add_layer(poscar: str) -> Dict[str, Any]:
    """
    Add one atomic layer on top of the slab by copying the topmost layer
    and translating it by the average inter-layer spacing.

    ASE: detect layers, copy top layer atoms, translate by d_interlayer,
    extend cell[2][2] accordingly.
    """
    try:
        slab = _poscar_to_atoms(poscar)
        layers = _detect_layers(slab)
        if len(layers) < 2:
            return {"ok": False, "error": "Cannot detect layers (need ≥ 2)"}

        top_layer    = layers[-1]
        prev_layer   = layers[-2]
        z_top  = np.mean(slab.get_positions()[top_layer,  2])
        z_prev = np.mean(slab.get_positions()[prev_layer, 2])
        d      = z_top - z_prev           # inter-layer spacing

        new_layer = slab[top_layer].copy()
        new_layer.translate([0, 0, d])
        slab_new  = slab + new_layer

        # Extend cell to accommodate new layer + keep vacuum
        old_cell  = slab.get_cell().copy()
        new_cell  = old_cell.copy()
        new_cell[2][2] += d
        slab_new.set_cell(new_cell, scale_atoms=False)
        slab_new.center(axis=2)           # re-center vacuum

        return _result(slab_new, label=f"add_layer ({len(layers)+1} layers)")
    except Exception as e:
        return {"ok": False, "error": str(e)}


def slab_delete_layer(poscar: str) -> Dict[str, Any]:
    """
    Delete the topmost atomic layer.

    ASE: detect layers, remove top-layer atom indices, re-center vacuum.
    """
    try:
        slab   = _poscar_to_atoms(poscar)
        layers = _detect_layers(slab)
        if len(layers) < 2:
            return {"ok": False, "error": "Only one layer detected — cannot delete"}

        top_indices = set(layers[-1])
        keep  = [i for i in range(len(slab)) if i not in top_indices]
        slab_new = slab[keep]
        slab_new.set_cell(slab.get_cell(), scale_atoms=False)
        slab_new.center(axis=2)

        return _result(slab_new, label=f"del_layer ({len(layers)-1} layers)")
    except Exception as e:
        return {"ok": False, "error": str(e)}


def slab_set_vacuum(poscar: str, vacuum: float = 15.0) -> Dict[str, Any]:
    """
    Set (or increase) the vacuum gap.

    ASE: slab.center(vacuum=vacuum/2) adjusts the c-vector so there is
    `vacuum` Å of empty space above the top layer.
    """
    try:
        slab = _poscar_to_atoms(poscar)
        slab.center(vacuum=vacuum / 2, axis=2)
        return _result(slab, label=f"vacuum={vacuum:.0f}Å",
                       vacuum=vacuum, cell_c=float(slab.get_cell()[2, 2]))
    except (ValueError, KeyError, TypeError) as e:
        return {"ok": False, "error": str(e)}


def slab_dope(
    poscar: str,
    host_element: str = "Pt",
    dopant_element: str = "Co",
    n_dopants: int = 1,
    site: str = "surface",       # "surface" | "subsurface" | "bulk_random"
) -> Dict[str, Any]:
    """
    Substitute n_dopants host atoms with dopant atoms.

    site="surface"    → replace from the topmost layer (most common for catalysis)
    site="subsurface" → replace from the second-topmost layer
    site="bulk_random"→ replace from the bottom layers (not surface)

    ASE: change atom.symbol directly.
    pymatgen: Structure.replace() could also be used.
    """
    try:
        slab   = _poscar_to_atoms(poscar)
        syms   = np.array(slab.get_chemical_symbols())
        layers = _detect_layers(slab)
        if not layers:
            return {"ok": False, "error": "No layers detected"}

        # Pick candidate layer
        if site == "subsurface" and len(layers) >= 2:
            candidates = [i for i in layers[-2] if syms[i] == host_element]
        elif site == "bulk_random":
            candidates = [i for i in range(len(slab))
                          if syms[i] == host_element and i not in layers[-1]]
        else:  # surface
            candidates = [i for i in layers[-1] if syms[i] == host_element]

        if not candidates:
            return {"ok": False,
                    "error": f"No {host_element} atoms found at {site} site"}
        if n_dopants > len(candidates):
            n_dopants = len(candidates)

        # Pick atoms near center of slab xy for most representative site
        xy_center = slab.get_positions()[candidates, :2].mean(axis=0)
        dists = np.linalg.norm(
            slab.get_positions()[candidates, :2] - xy_center, axis=1
        )
        chosen = [candidates[i] for i in np.argsort(dists)[:n_dopants]]

        slab_new = slab.copy()
        for idx in chosen:
            slab_new[idx].symbol = dopant_element

        label = f"{dopant_element}{n_dopants}/{host_element}_slab"
        return _result(slab_new, label=label,
                       doped_indices=chosen, dopant=dopant_element, host=host_element)
    except Exception as e:
        return {"ok": False, "error": str(e)}


def slab_make_symmetric(poscar: str, vacuum: float = 20.0) -> Dict[str, Any]:
    """
    Build an inversion-symmetric slab for GC-DFT (electrochemical) calculations.

    Algorithm (faithful to the user's validated approach):
      1. Center slab along z.
      2. Identify the base layer = atoms in the lowest-z cluster (layer detection
         uses _detect_layers with 0.5 Å gap tolerance — robust to relaxed geometries).
      3. Compute base_layer_center (3-D mean of base-layer atom positions).
      4. Invert all NON-base atoms through that center:
             pos_new = 2 * base_layer_center - pos_old
         This is a 3-D point inversion, which correctly inverts the stacking
         sequence (ABC → CBA) so both surfaces share the same termination.
      5. Append the inverted atoms to the original slab.
      6. Clear any existing constraints; fix the base layer only.
      7. Set cell c = 2 * slab_thickness + vacuum, then re-center along z.

    Robustness improvements over the original script:
      - Layer detection uses cluster gaps (>0.5 Å) instead of a hard 0.1 Å
        tolerance, so relaxed / strained slabs are handled correctly.
      - `atoms.set_constraint(None)` replaces `del atoms.constraints` (no crash
        if the slab has no constraints).
      - Cell height is computed from actual slab thickness + the vacuum parameter
        instead of being hardcoded at 60 Å.
      - Vectorised numpy operations replace the per-atom loops.
    """
    try:
        from copy import deepcopy

        slab = _poscar_to_atoms(poscar)
        slab.center(axis=2)   # align slab in z before measuring

        # ── 1. Find base layer using robust layer clustering ──────────────
        layers = _detect_layers(slab)          # sorted bottom → top
        if not layers:
            return {"ok": False, "error": "Could not detect atomic layers"}
        base_layer_index = layers[0]           # indices of the bottommost layer

        base_layer_center = slab.get_positions()[base_layer_index].mean(axis=0)

        # ── 2. Build inverted copy (all atoms except base layer) ──────────
        non_base_mask = np.ones(len(slab), dtype=bool)
        non_base_mask[base_layer_index] = False
        non_base_indices = np.where(non_base_mask)[0]

        if len(non_base_indices) == 0:
            return {"ok": False, "error": "Slab has only one layer — cannot symmetrise"}

        inverted = slab[non_base_indices].copy()
        inv_pos  = inverted.get_positions()
        inv_pos  = 2.0 * base_layer_center - inv_pos   # 3-D inversion
        inverted.set_positions(inv_pos)

        # ── 3. Combine ────────────────────────────────────────────────────
        combined = slab + inverted

        # ── 4. Fix constraints: base layer only ───────────────────────────
        combined.set_constraint(None)          # clear safely (no AttributeError)
        combined.set_constraint(
            ase.constraints.FixAtoms(indices=base_layer_index)
        )

        # ── 5. Adjust cell height and re-center ───────────────────────────
        pos_z = combined.get_positions()[:, 2]
        slab_thickness = pos_z.max() - pos_z.min()
        cell  = combined.get_cell().copy()
        cell[2][2] = slab_thickness + vacuum   # vacuum on top+bottom after centering
        combined.set_cell(cell, scale_atoms=False)
        combined.center(axis=2)

        n_orig_layers = len(layers)
        n_sym_layers  = len(_detect_layers(combined))
        return _result(
            combined,
            label=f"symmetric_slab (GC-DFT ready, {n_sym_layers} layers)",
            n_layers_original=n_orig_layers,
            n_layers_symmetric=n_sym_layers,
            vacuum=vacuum,
            base_layer_indices=base_layer_index,
        )
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


def build_interface(
    poscar_a: str,
    poscar_b: str,
    vacuum: float = 15.0,
    interface_gap: float = 2.2,
    strain_a: bool = False,
) -> Dict[str, Any]:
    """
    Build a heterogeneous interface by stacking slab_b on top of slab_a.

    For a quick initial model (e.g. Pt thin film on Cu substrate):
      - If strain_a=True, rescale slab_a's xy to match slab_b's cell (coherent interface).
      - Otherwise just stack (user should verify lattice mismatch).

    For production use pymatgen CoherentInterfaceBuilder:
      from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
      cib = CoherentInterfaceBuilder(film_structure, substrate_structure, film_miller, sub_miller)
      interfaces = list(cib.get_interfaces())

    ASE: straightforward Atoms concatenation + cell adjustment.
    """
    try:
        slab_a = _poscar_to_atoms(poscar_a)
        slab_b = _poscar_to_atoms(poscar_b)

        if strain_a:
            # Rescale slab_a xy to match slab_b (coherent stacking)
            cell_b = slab_b.get_cell()
            cell_a = slab_a.get_cell().copy()
            scale_x = cell_b[0, 0] / cell_a[0, 0] if cell_a[0, 0] else 1.0
            scale_y = cell_b[1, 1] / cell_a[1, 1] if cell_a[1, 1] else 1.0
            pos_a = slab_a.get_positions()
            pos_a[:, 0] *= scale_x
            pos_a[:, 1] *= scale_y
            slab_a.set_positions(pos_a)
            cell_a[0, 0] = cell_b[0, 0]
            cell_a[1, 1] = cell_b[1, 1]
            slab_a.set_cell(cell_a, scale_atoms=False)

        z_top_a = slab_a.get_positions()[:, 2].max()
        z_bot_b = slab_b.get_positions()[:, 2].min()
        slab_b_copy = slab_b.copy()
        slab_b_copy.translate([0, 0, z_top_a + interface_gap - z_bot_b])

        combined = slab_a + slab_b_copy
        combined.set_cell(slab_a.get_cell(), scale_atoms=False)
        combined.center(vacuum=vacuum / 2, axis=2)

        fa = slab_a.get_chemical_formula()
        fb = slab_b.get_chemical_formula()
        return _result(combined, label=f"{fa}/{fb}_interface",
                       interface_gap=interface_gap, strain_applied=strain_a)
    except Exception as e:
        return {"ok": False, "error": str(e)}


def generate_neb_images(
    is_poscar: str,
    fs_poscar: str,
    n_images: int = 6,
    method: str = "linear",      # "linear" | "idpp"
) -> Dict[str, Any]:
    """
    Generate NEB intermediate images between initial state (IS) and final state (FS).

    Returns a list of POSCAR strings (IS + intermediates + FS = n_images + 2 total).

    ASE NEB:
      from ase.neb import NEB
      images = [is_atoms] + [is_atoms.copy() for _ in range(n_images)] + [fs_atoms]
      neb = NEB(images)
      neb.interpolate()            # linear interpolation
      # or
      neb.interpolate('idpp')      # image-dependent pair potential (smoother path)

    Constraints (FixAtoms) on IS are automatically copied to all images.
    """
    try:
        from ase.neb import NEB
        is_atoms = _poscar_to_atoms(is_poscar)
        fs_atoms = _poscar_to_atoms(fs_poscar)

        if len(is_atoms) != len(fs_atoms):
            return {"ok": False,
                    "error": f"IS has {len(is_atoms)} atoms, FS has {len(fs_atoms)} — must match"}

        images  = [is_atoms.copy()] + [is_atoms.copy() for _ in range(n_images)] + [fs_atoms.copy()]
        neb     = NEB(images)
        try:
            neb.interpolate(method)
        except Exception:
            neb.interpolate()      # fallback to linear if idpp fails

        poscars = [_atoms_to_poscar(img) for img in images]
        plots   = [atoms_to_plot_b64(img, rotations="10z,-80x,0y") for img in images]

        return {
            "ok":        True,
            "n_images":  len(images),
            "images": [
                {"index": i, "poscar": poscars[i], "plot_png_b64": plots[i],
                 "label": "IS" if i == 0 else ("FS" if i == len(images)-1 else f"img{i:02d}")}
                for i in range(len(images))
            ],
            "formula": is_atoms.get_chemical_formula(),
        }
    except ImportError:
        return {"ok": False, "error": "ase.neb not available — install ase>=3.22"}
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


# =========================================================================
# Text description & ASE code generators  (for T2S library)
# =========================================================================

_FACET_DESC: Dict[str, str] = {
    "111":  "close-packed hexagonal (111) — highest atomic density, lowest surface energy for FCC",
    "100":  "square (100) — moderate density, common for FCC metals",
    "110":  "stepped (110) — open surface with ridge-and-valley structure",
    "0001": "close-packed basal plane (0001) for HCP metals",
    "110":  "stepped (110) — open surface with ridge-and-valley structure",
}
_CSYS_DESC: Dict[str, str] = {
    "fcc": "face-centered cubic (FCC)",
    "bcc": "body-centered cubic (BCC)",
    "hcp": "hexagonal close-packed (HCP)",
}
_SITE_DESC: Dict[str, str] = {
    "top":        "on-top site — directly above a single surface metal atom",
    "bridge":     "bridge site — midpoint between two adjacent surface metal atoms",
    "hollow_fcc": "FCC hollow site — three-fold hollow above a subsurface atom in the FCC stacking",
    "hollow_hcp": "HCP hollow site — three-fold hollow directly above a subsurface atom",
    "hollow":     "hollow site — three-fold coordinated hollow",
}


def describe_surface(
    elem: str, facet: str, nx: int, ny: int, nlayers: int,
    vacuum: float, n_atoms: int, formula: str, csys: str,
) -> str:
    """Generate a rich natural-language description of a metal surface slab."""
    facet_str = _FACET_DESC.get(facet, f"({facet}) facet")
    csys_str  = _CSYS_DESC.get(csys, csys)
    return (
        f"A clean {elem}({facet}) {csys_str} metal surface slab. "
        f"The ({facet}) facet exposes a {facet_str} arrangement of {elem} atoms. "
        f"Slab geometry: {nx}×{ny} lateral supercell, {nlayers} atomic layers along the surface normal, "
        f"{vacuum:.0f} Å vacuum gap above the top layer. "
        f"Total {n_atoms} atoms ({formula}). "
        f"Bottom atomic layer is fixed with FixAtoms to mimic a bulk-like substrate. "
        f"Appropriate as a substrate for adsorption energy, transition state, "
        f"and reaction free-energy DFT calculations on {elem} catalysts."
    )


def ase_code_surface(
    elem: str, facet: str, nx: int, ny: int, nlayers: int,
    vacuum: float, csys: str, a0: Optional[float] = None,
) -> str:
    """Return reproducible Python/ASE code to build this surface."""
    builder_map = {
        ("fcc", "111"): "fcc111", ("fcc", "100"): "fcc100", ("fcc", "110"): "fcc110",
        ("bcc", "110"): "bcc110", ("bcc", "100"): "bcc100", ("bcc", "111"): "bcc111",
        ("hcp", "0001"): "hcp0001", ("hcp", "001"): "hcp0001",
    }
    builder = builder_map.get((csys, facet), "surface")
    a_val   = a0 or _A0.get(elem)
    a_arg   = f", a={a_val}" if isinstance(a_val, (int, float)) else ""

    if builder == "surface":
        return (
            f"from ase.build import bulk, surface\n"
            f"from ase.constraints import FixAtoms\n"
            f"import numpy as np\n\n"
            f"b = bulk('{elem}', crystalstructure='{csys}', cubic=True)\n"
            f"slab = surface(b, ({', '.join(facet)}), {nlayers}, vacuum={vacuum})\n"
            f"slab = slab.repeat([{nx}, {ny}, 1])\n"
            f"base_z = slab.get_positions()[:, 2].min()\n"
            f"fix_idx = [i for i, z in enumerate(slab.get_positions()[:, 2])\n"
            f"           if abs(z - base_z) < 0.1]\n"
            f"slab.set_constraint(FixAtoms(indices=fix_idx))\n"
        )
    return (
        f"from ase.build import {builder}\n"
        f"from ase.constraints import FixAtoms\n"
        f"import numpy as np\n\n"
        f"slab = {builder}('{elem}', size=({nx}, {ny}, {nlayers}){a_arg}, vacuum={vacuum})\n"
        f"base_z = slab.get_positions()[:, 2].min()\n"
        f"fix_idx = [i for i, z in enumerate(slab.get_positions()[:, 2])\n"
        f"           if abs(z - base_z) < 0.1]\n"
        f"slab.set_constraint(FixAtoms(indices=fix_idx))\n"
    )


def describe_molecule(
    label: str, smiles: str, formula: str, n_atoms: int, cell_size: float,
) -> str:
    """Generate a natural-language description of a gas-phase molecule."""
    return (
        f"{label} gas-phase molecule. "
        f"Chemical formula: {formula}  |  SMILES: {smiles}. "
        f"Contains {n_atoms} atoms in a {cell_size:.0f}×{cell_size:.0f}×{cell_size:.0f} Å cubic cell, "
        f"centered. "
        f"3-D geometry retrieved from PubChem via SMILES lookup. "
        f"Used as gas-phase reference for computing adsorption energies (E_ads = E_slab+mol − E_slab − E_mol) "
        f"and Gibbs free energies after frequency/ZPE calculations."
    )


def ase_code_molecule(smiles: str, label: str, cell_size: float) -> str:
    """Return reproducible Python/ASE code to build this molecule."""
    return (
        f"from ase.data.pubchem import pubchem_atoms_search\n\n"
        f"mol = pubchem_atoms_search(smiles='{smiles}')  # {label}\n"
        f"mol.set_cell([{cell_size}, {cell_size}, {cell_size}])\n"
        f"mol.center()\n"
    )


def describe_adsorption(
    mol_label: str, surf_label: str,
    site_type: str, rotation: int, height: float,
    n_atoms: int, formula: str,
) -> str:
    """Generate a natural-language description of an adsorption configuration."""
    site_str = _SITE_DESC.get(site_type, site_type)
    rot_str  = f", rotated {rotation}° around the surface normal" if rotation else ""
    return (
        f"Adsorption configuration: {mol_label} on {surf_label}. "
        f"Binding geometry: {site_str}{rot_str}. "
        f"Anchor atom (lowest-z atom of {mol_label}) placed {height:.1f} Å above the "
        f"topmost surface atom. "
        f"Combined slab+adsorbate system: {n_atoms} atoms ({formula}). "
        f"This is an unrelaxed initial geometry; intended as input for VASP geometry "
        f"optimization (NSW>0, IBRION=2) to find the relaxed adsorption structure and energy."
    )


def ase_code_adsorption(
    surf_label: str, mol_label: str,
    site_type: str, rotation: int, height: float,
    site_xy: Optional[List[float]] = None,
) -> str:
    """Return reproducible Python/ASE code to reproduce this adsorption config."""
    sx, sy = (site_xy or [0.0, 0.0])[:2]
    rot_line = (
        f"mol.rotate({rotation}, 'z', center=(0, 0, 0))  # orientation variety\n"
        if rotation else ""
    )
    return (
        f"from ase.io import read\n"
        f"import numpy as np\n\n"
        f"slab = read('{surf_label}.POSCAR', format='vasp')\n"
        f"mol  = read('{mol_label}.POSCAR',  format='vasp')\n\n"
        f"# Place anchor atom at origin, then translate to site\n"
        f"mol_pos    = mol.get_positions()\n"
        f"anchor_idx = mol_pos[:, 2].argmin()   # atom closest to surface\n"
        f"mol.translate(-mol_pos[anchor_idx])\n"
        f"{rot_line}"
        f"surf_z_max = slab.get_positions()[:, 2].max()\n"
        f"mol.translate([{sx:.4f}, {sy:.4f}, surf_z_max + {height}])  # {site_type} site\n\n"
        f"combined = slab + mol\n"
    )


# =========================================================================
# Serialization & file I/O
# =========================================================================

def atoms_to_plot_b64(atoms: Atoms, rotations: str = "10z,-80x,0y", figsize=(6, 5)) -> str:
    """Render ASE Atoms as a matplotlib PNG, return base64 string."""
    try:
        import io, base64
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ase.visualize.plot import plot_atoms
        fig, ax = plt.subplots(figsize=figsize)
        plot_atoms(atoms, ax, rotation=rotations, show_unit_cell=0)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def atoms_to_viz_json(atoms: Atoms) -> Dict[str, Any]:
    """Serialize ASE Atoms to JSON for Streamlit 3D visualization (py3Dmol/ngl)."""
    pos = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    bonds: List[List[Any]] = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            ri = covalent_radii[atomic_numbers[syms[i]]]
            rj = covalent_radii[atomic_numbers[syms[j]]]
            if d < (ri + rj) * 1.3:
                bonds.append([i, j, round(d, 3)])
    atoms_data = [
        {
            "index": i,
            "symbol": sym,
            "position": [round(x, 5) for x in pos[i].tolist()],
            "color": _ELEM_COLORS.get(sym, _ELEM_COLORS["default"]),
        }
        for i, sym in enumerate(syms)
    ]
    return {
        "atoms": atoms_data,
        "bonds": bonds,
        "cell": atoms.get_cell().tolist(),
        "pbc": atoms.get_pbc().tolist(),
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
    }


def write_structure_files(job_dir: Path, atoms: Atoms, name_prefix: str = "slab") -> List[str]:
    """Write POSCAR, CIF, and visualization JSON to job_dir."""
    _ensure_dir(job_dir)
    files = []
    for fname in (f"{name_prefix}.POSCAR", "POSCAR"):
        write(job_dir / fname, atoms, format="vasp", direct=True, vasp5=True)
        files.append(fname)
    try:
        write(job_dir / f"{name_prefix}.cif", atoms)
        files.append(f"{name_prefix}.cif")
    except OSError:
        pass
    viz_path = job_dir / f"{name_prefix}_viz.json"
    viz_path.write_text(json.dumps(atoms_to_viz_json(atoms), indent=2))
    files.append(viz_path.name)
    meta = {
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "symbols": atoms.get_chemical_symbols(),
        "cell": atoms.get_cell().tolist(),
        "pbc": atoms.get_pbc().tolist(),
    }
    (job_dir / "structure.json").write_text(json.dumps(meta, indent=2))
    files.append("structure.json")
    uniq: List[str] = []
    for s in atoms.get_chemical_symbols():
        if s not in uniq:
            uniq.append(s)
    (job_dir / "POTCAR.spec").write_text(json.dumps(uniq, indent=2))
    files.append("POTCAR.spec")
    return files

# =========================================================================
# PostgreSQL recording
# =========================================================================

async def record_structure_to_db(
    session_id: Optional[int],
    atoms: Atoms,
    metadata: Dict[str, Any],
    natural_language: str = "",
) -> Optional[int]:
    """
    Persist structure + natural-language description to structure_t2s table.
    Returns the new row ID, or None on failure.
    """
    try:
        from server.db import AsyncSessionLocal, StructureT2S
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".POSCAR", delete=False, mode="w") as f:
            fname = f.name
        write(fname, atoms, format="vasp", direct=True, vasp5=True)
        poscar_content = open(fname).read()
        os.unlink(fname)

        row = StructureT2S(
            session_id=session_id,
            formula=atoms.get_chemical_formula(),
            material=metadata.get("element", ""),
            facet=metadata.get("facet", ""),
            crystal_system=metadata.get("crystal_system", ""),
            adsorbates=metadata.get("adsorbates", []),
            natural_language=natural_language,
            poscar_content=poscar_content,
            atoms_json=atoms_to_viz_json(atoms),
            n_atoms=len(atoms),
            is_optimized=bool(metadata.get("is_optimized", False)),
            provenance=metadata.get("provenance", {}),
        )
        async with AsyncSessionLocal() as db:
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return row.id
    except Exception:
        return None

# =========================================================================
# Main agent class
# =========================================================================

class StructureAgent:
    """
    Interactive structure building agent.

    Entry points
    ------------
    build(task, job_dir)        — main pipeline hook (used by agent_routes)
    find_sites(poscar_str)      — return adsorption sites for a POSCAR string
    place_ads(poscar_str, ads, site_idx) — place one adsorbate, return new POSCAR
    generate_ads_configs(poscar_str, ads, n, job_dir) — generate N configs
    """

    # ------------------------------------------------------------------
    # Session-awareness: reuse prior structures before rebuilding
    # ------------------------------------------------------------------
    @staticmethod
    def _lookup_session_structure(session_id, label_hint: str):
        """
        Check structure_t2s for an existing POSCAR with a matching label/material
        in *session_id*.  Returns a POSCAR string if found, or None.
        """
        if not session_id:
            return None
        try:
            import asyncio
            from sqlalchemy import select
            from server.db import AsyncSessionLocal, StructureT2S

            async def _query():
                async with AsyncSessionLocal() as db:
                    # Match by label_hint substring (case-insensitive)
                    stmt = (
                        select(StructureT2S)
                        .where(
                            StructureT2S.session_id == session_id,
                            StructureT2S.poscar_content.isnot(None),
                        )
                        .order_by(StructureT2S.created_at.desc())
                        .limit(20)
                    )
                    res = await db.execute(stmt)
                    rows = list(res.scalars().all())
                    hint_lc = label_hint.lower()
                    for row in rows:
                        candidate = " ".join(filter(None, [
                            row.material, row.facet, row.formula,
                            row.natural_language or "",
                        ])).lower()
                        if hint_lc and any(tok in candidate for tok in hint_lc.split()[:3]):
                            return row.poscar_content
                    return None

            try:
                loop = asyncio.get_running_loop()
                # Can't call asyncio.run() inside a running loop; skip lookup
                return None
            except RuntimeError:
                return asyncio.run(_query())
        except Exception:
            return None

    # ------------------------------------------------------------------
    def build(self, task: Dict[str, Any], job_dir: Path) -> Dict[str, Any]:
        """Build a slab from task spec and write files to job_dir."""
        job_dir = Path(job_dir)
        _ensure_dir(job_dir)

        payload = ((task.get("params") or {}).get("payload") or {})
        form = ((task.get("params") or {}).get("form") or [])
        overrides = {f.get("key"): f.get("value") for f in form if isinstance(f, dict) and "key" in f}

        raw_elem = (payload.get("element") or payload.get("metal") or
                    payload.get("species") or task.get("name") or "Cu")
        element = _normalize_element(raw_elem)

        # ── Session awareness: reuse prior structure if available ──────────
        task_id = task.get("db_id") or task.get("task_id")
        session_id = task.get("session_id")
        if task_id:
            from server.execution.utils.task_status import emit_task_status_sync
            emit_task_status_sync(task_id, "running")

        h, k, l = _parse_miller(payload)
        facet = f"{h}{k}{l}"
        layers = int(overrides.get("layers") or payload.get("layers") or 4)
        vacuum = float(overrides.get("vacuum_thickness") or payload.get("vacuum") or 15.0)
        nx, ny, _ = _parse_supercell(overrides.get("supercell") or payload.get("supercell"))
        crystal_system = _guess_crystal_system(element, {**payload, **overrides})

        atoms = _build_slab(element, crystal_system, h, k, l, nx, ny, layers, vacuum)
        files = write_structure_files(job_dir, atoms, name_prefix="slab")

        inp = {
            "element": element, "crystal_system": crystal_system,
            "miller_index": [h, k, l], "facet": facet,
            "layers": layers, "vacuum": vacuum, "supercell": [nx, ny, 1],
        }
        (job_dir / "_inputs.json").write_text(json.dumps(inp, indent=2))
        (job_dir / "_param_log.jsonl").write_text(json.dumps(inp) + "\n")

        nl_desc = (f"{crystal_system.upper()} {element}({facet}) surface slab, "
                   f"{nx}x{ny} supercell, {layers} layers, {vacuum:.1f} Å vacuum, "
                   f"{len(atoms)} atoms total.")
        (job_dir / "nl_description.txt").write_text(nl_desc)

        result = {
            "ok": True,
            "label": f"{element}({facet}) {nx}x{ny} L={layers} vac={vacuum}Å",
            "element": element,
            "crystal_system": crystal_system,
            "facet": facet,
            "miller_index": [h, k, l],
            "layers": layers,
            "vacuum": vacuum,
            "supercell": [nx, ny, 1],
            "n_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "files": files,
            "viz": atoms_to_viz_json(atoms),
            "nl_description": nl_desc,
        }
        if task_id:
            from server.execution.utils.task_status import emit_task_status_sync
            emit_task_status_sync(task_id, "done", output_data={
                "poscar_path": files.get("poscar", ""),
                "n_atoms": len(atoms),
                "formula": atoms.get_chemical_formula(),
            })
        return result

    # ------------------------------------------------------------------
    def find_sites(self, poscar_content: str, height: float = 2.0) -> Dict[str, Any]:
        """Find adsorption sites from a POSCAR string."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as f:
            f.write(poscar_content); fname = f.name
        try:
            atoms = read(fname, format="vasp")
            sites = find_adsorption_sites_ase(atoms, height=height)
            return {
                "ok": True,
                "n_sites": len(sites),
                "sites": sites,
                "site_types": list({s["type"] for s in sites}),
                "viz": atoms_to_viz_json(atoms),
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "sites": []}
        finally:
            try: os.unlink(fname)
            except Exception: pass

    # ------------------------------------------------------------------
    def place_ads(self, poscar_content: str, adsorbate: str,
                  site_index: int = 0, height: float = 2.0) -> Dict[str, Any]:
        """Place one adsorbate at the specified site index."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as f:
            f.write(poscar_content); fname = f.name
        fname2 = None
        try:
            slab = read(fname, format="vasp")
            sites = find_adsorption_sites_ase(slab, height=height)
            if not sites:
                return {"ok": False, "error": "No adsorption sites found"}
            site = sites[min(site_index, len(sites) - 1)]
            combined = place_adsorbate(slab, site["position"], adsorbate)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as f2:
                fname2 = f2.name
            write(fname2, combined, format="vasp", direct=True, vasp5=True)
            poscar_out = open(fname2).read()
            return {
                "ok": True,
                "adsorbate": adsorbate,
                "site_type": site.get("type"),
                "site_position": site["position"],
                "n_atoms": len(combined),
                "formula": combined.get_chemical_formula(),
                "poscar": poscar_out,
                "viz": atoms_to_viz_json(combined),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            for fn in (fname, fname2):
                try:
                    if fn: os.unlink(fn)
                except Exception: pass

    # ------------------------------------------------------------------
    def generate_ads_configs(
        self,
        poscar_content: str,
        adsorbate: str,
        max_configs: int = 4,
        job_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate multiple adsorption configurations and optionally write to job_dir."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as f:
            f.write(poscar_content); fname = f.name
        try:
            slab = read(fname, format="vasp")
            sites = find_adsorption_sites_ase(slab)
            configs = generate_configurations(slab, adsorbate, sites, max_configs)
            results = []
            for i, cfg in enumerate(configs):
                atoms = cfg["atoms"]
                with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as f2:
                    fname2 = f2.name
                write(fname2, atoms, format="vasp", direct=True, vasp5=True)
                poscar_out = open(fname2).read()
                os.unlink(fname2)
                entry: Dict[str, Any] = {
                    "config_id": i,
                    "site_type": cfg["site_type"],
                    "position": cfg["position"],
                    "n_atoms": len(atoms),
                    "formula": atoms.get_chemical_formula(),
                    "poscar": poscar_out,
                    "viz": atoms_to_viz_json(atoms),
                }
                if job_dir:
                    cfg_dir = Path(job_dir) / f"config_{i:02d}_{cfg['site_type']}"
                    write_structure_files(cfg_dir, atoms, name_prefix=f"ads_{adsorbate}")
                    entry["job_dir"] = str(cfg_dir)
                results.append(entry)
            return {"ok": True, "adsorbate": adsorbate, "n_configs": len(results), "configs": results}
        except OSError as e:
            return {"ok": False, "error": str(e), "configs": []}
        finally:
            try: os.unlink(fname)
            except Exception: pass

    # ------------------------------------------------------------------
    def build_molecule(
        self,
        smiles: str = "CCCC",
        label: str = "",
        cell_size: float = 20.0,
        job_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Fetch a molecule from PubChem by SMILES and write POSCAR to job_dir.
        Uses: pubchem_atoms_search(smiles='...') → set_cell([20,20,20]) → center()
        """
        result = build_molecule_pubchem(smiles=smiles, cell_size=cell_size, label=label)
        if not result.get("ok"):
            return result
        mol = result.pop("atoms")  # remove non-serializable Atoms object
        if job_dir:
            job_dir = Path(job_dir)
            files = write_structure_files(job_dir, mol, name_prefix=label or result["formula"])
            result["files"] = files
            result["job_dir"] = str(job_dir)
        return result

    # ------------------------------------------------------------------
    def deprotonate_molecule(
        self,
        poscar: str,
        n_remove: int = 1,
        site: str = "surface",
    ) -> Dict[str, Any]:
        """
        Remove n_remove hydrogen atoms from a molecule POSCAR to generate
        a deprotonated intermediate (e.g. CH3OH → CH2OH for CO2RR pathways).

        site:
          'surface'  — remove H atoms closest to the geometric center of
                       H atoms (approximates β-H or most accessible H).
          'terminal' — remove H atoms furthest from the carbon backbone
                       centroid (e.g. O-H bond in alcohol).
          'random'   — remove the first n_remove H atoms.

        Returns the modified POSCAR as a string plus metadata.
        """
        try:
            from ase.io import read as ase_read, write as ase_write
            import io, copy

            # Parse POSCAR
            buf = io.StringIO(poscar)
            atoms = ase_read(buf, format="vasp")

            # Identify hydrogen indices
            h_indices = [i for i, sym in enumerate(atoms.get_chemical_symbols()) if sym == "H"]
            if not h_indices:
                return {"ok": False, "detail": "No hydrogen atoms found in structure."}
            if n_remove > len(h_indices):
                return {"ok": False, "detail": f"Requested {n_remove} H removals but only {len(h_indices)} H atoms present."}

            pos = atoms.get_positions()

            if site == "terminal":
                # Remove H atoms furthest from the heavy-atom centroid
                heavy_idx = [i for i in range(len(atoms)) if atoms[i].symbol != "H"]
                if heavy_idx:
                    centroid = pos[heavy_idx].mean(axis=0)
                    h_dists = [(i, float(((pos[i] - centroid)**2).sum()**0.5)) for i in h_indices]
                    h_dists.sort(key=lambda x: -x[1])  # furthest first
                    to_remove = [idx for idx, _ in h_dists[:n_remove]]
                else:
                    to_remove = h_indices[:n_remove]
            elif site == "surface":
                # Remove H atoms closest to their own centroid (most surface-accessible)
                h_centroid = pos[h_indices].mean(axis=0)
                h_dists = [(i, float(((pos[i] - h_centroid)**2).sum()**0.5)) for i in h_indices]
                h_dists.sort(key=lambda x: x[1])  # closest to H-centroid first
                to_remove = [idx for idx, _ in h_dists[:n_remove]]
            else:
                # random / default: just take the first n_remove
                to_remove = h_indices[:n_remove]

            # Delete atoms (high index first to avoid index shifting)
            new_atoms = atoms.copy()
            del new_atoms[[sorted(to_remove, reverse=True)[0]] if len(to_remove) == 1
                           else sorted(to_remove, reverse=True)]

            # Write modified POSCAR
            out_buf = io.StringIO()
            ase_write(out_buf, new_atoms, format="vasp", direct=True)
            new_poscar = out_buf.getvalue()

            formula_orig = atoms.get_chemical_formula()
            formula_new  = new_atoms.get_chemical_formula()

            return {
                "ok": True,
                "poscar": new_poscar,
                "formula_original": formula_orig,
                "formula_deprotonated": formula_new,
                "n_removed": n_remove,
                "removed_indices": to_remove,
                "n_atoms_new": len(new_atoms),
            }
        except Exception as e:
            return {"ok": False, "detail": str(e)}

    # ------------------------------------------------------------------
    def build_surface(
        self,
        element: str = "Cu",
        surface_type: str = "111",
        nx: int = 4,
        ny: int = 4,
        nlayers: int = 3,
        vacuum: float = 10.0,
        fix_bottom: bool = True,
        job_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Build a clean metal surface using ASE (user's fcc111 pattern).
        Asks: element, surface_type (111/100/110/443), nx×ny, nlayers, vacuum.
        """
        result = build_surface_ase(
            element=element, surface_type=surface_type,
            nx=nx, ny=ny, nlayers=nlayers, vacuum=vacuum, fix_bottom=fix_bottom,
        )
        if not result.get("ok"):
            return result
        atoms = result.pop("atoms")  # remove non-serializable Atoms object
        if job_dir:
            job_dir = Path(job_dir)
            files = write_structure_files(job_dir, atoms, name_prefix="slab")
            result["files"] = files
            result["job_dir"] = str(job_dir)
        return result

    # ------------------------------------------------------------------
    def build_complex(
        self,
        metal: str = "Cu",
        ligand: str = "NH3",
        n_coord: int = 4,
        geometry: str = "square_planar",
        bond_length: float = 2.0,
        cell_size: float = 15.0,
        job_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Build a coordination compound: metal center + monodentate ligands.
        Delegates to the standalone build_complex() function.
        """
        result = build_complex(
            metal=metal, ligand=ligand, n_coord=n_coord,
            geometry=geometry, bond_length=bond_length, cell_size=cell_size,
        )
        if not result.get("ok"):
            return result
        if job_dir:
            job_dir = Path(job_dir)
            # write POSCAR from string
            job_dir.mkdir(parents=True, exist_ok=True)
            poscar_path = job_dir / "POSCAR"
            poscar_path.write_text(result["poscar"])
            result["files"] = [str(poscar_path)]
            result["job_dir"] = str(job_dir)
        return result


# ======================================================================
# Standalone build_complex function
# ======================================================================

def _rot_z_to_v(v: np.ndarray) -> np.ndarray:
    """Return 3×3 rotation matrix that maps (0,0,1) → v."""
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(v, z):
        return np.eye(3)
    if np.allclose(v, -z):
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    axis = np.cross(z, v)
    axis /= np.linalg.norm(axis)
    cos_a = float(np.dot(z, v))
    sin_a = float(np.sqrt(max(0.0, 1 - cos_a ** 2)))
    K = np.array([
        [0,        -axis[2],  axis[1]],
        [axis[2],  0,        -axis[0]],
        [-axis[1], axis[0],  0],
    ])
    return np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)


# Ligand library: {name: (symbols, local_coords)}
# local_coords[0] is the binding atom at origin; rest along +z pointing away from metal.
def _ligand_library() -> Dict[str, Tuple[List[str], np.ndarray]]:
    """
    Internal ligand coordinate library.
    Convention: binding atom at origin (index 0); all other atoms positioned
    along the +z direction so that rotating +z→v places the ligand outward.
    """
    _lib: Dict[str, Tuple[List[str], np.ndarray]] = {}

    # NH3 — N binds; 3 H atoms in tetrahedral positions pointing away (+z direction)
    # N-H bond length ~1.01 Å; tetrahedral angle 107.8°
    nh_bond = 1.012
    # H atoms: tetrahedral around N, with N-metal axis along -z.
    # The three H atoms form a trigonal base above the N (in +z direction).
    # Cone half-angle from tetrahedral: cos(θ)=-1/3 → θ≈109.47°; H is at 109.47° from N-metal.
    # So H is 109.47° from -z → 180°-109.47°=70.53° from +z.
    import math as _math
    cone_half = _math.acos(-1.0 / 3.0)  # ≈109.47° — angle between N-H and N→metal
    # angle of H from +z = π - cone_half
    h_angle_from_z = _math.pi - cone_half
    h_r = nh_bond * _math.sin(h_angle_from_z)
    h_z = nh_bond * _math.cos(h_angle_from_z)
    nh3_coords = np.array([
        [0.0, 0.0, 0.0],  # N (binding)
        [h_r,             0.0,             h_z],
        [h_r * _math.cos(2 * _math.pi / 3), h_r * _math.sin(2 * _math.pi / 3), h_z],
        [h_r * _math.cos(4 * _math.pi / 3), h_r * _math.sin(4 * _math.pi / 3), h_z],
    ])
    _lib["NH3"] = (["N", "H", "H", "H"], nh3_coords)

    # H2O — O binds; 2 H at 104.5° HOH angle pointing away
    oh_bond = 0.957
    hoh_half = _math.radians(104.5 / 2.0)
    h2o_coords = np.array([
        [0.0, 0.0, 0.0],                        # O (binding)
        [ oh_bond * _math.sin(hoh_half), 0.0, oh_bond * _math.cos(hoh_half)],
        [-oh_bond * _math.sin(hoh_half), 0.0, oh_bond * _math.cos(hoh_half)],
    ])
    _lib["H2O"] = (["O", "H", "H"], h2o_coords)

    # CO — C binds (metal-C-O); O at 1.13 Å from C pointing outward (+z)
    _lib["CO"] = (["C", "O"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]]))

    # CN — C binds; N at 1.15 Å pointing outward
    _lib["CN"] = (["C", "N"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.153]]))

    # NO — N binds; O at 1.15 Å pointing outward
    _lib["NO"] = (["N", "O"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.151]]))

    # Cl — single atom
    _lib["Cl"] = (["Cl"], np.array([[0.0, 0.0, 0.0]]))

    # F — single atom
    _lib["F"] = (["F"], np.array([[0.0, 0.0, 0.0]]))

    # OH — O binds; H at 0.97 Å pointing away (+z)
    _lib["OH"] = (["O", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.970]]))

    # PH3 — P binds; 3 H in tetrahedral positions pointing away
    ph_bond = 1.415
    # Same tetrahedral geometry as NH3
    p_cone_half = _math.acos(-1.0 / 3.0)
    p_h_angle_from_z = _math.pi - p_cone_half
    ph_r = ph_bond * _math.sin(p_h_angle_from_z)
    ph_z = ph_bond * _math.cos(p_h_angle_from_z)
    ph3_coords = np.array([
        [0.0, 0.0, 0.0],
        [ph_r,             0.0,             ph_z],
        [ph_r * _math.cos(2 * _math.pi / 3), ph_r * _math.sin(2 * _math.pi / 3), ph_z],
        [ph_r * _math.cos(4 * _math.pi / 3), ph_r * _math.sin(4 * _math.pi / 3), ph_z],
    ])
    _lib["PH3"] = (["P", "H", "H", "H"], ph3_coords)

    # SCN — S binds; C at 1.63 Å, then N at additional 1.16 Å outward
    _lib["SCN"] = (
        ["S", "C", "N"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.630], [0.0, 0.0, 1.630 + 1.157]]),
    )

    return _lib


# Geometry unit vectors
def _geometry_vectors(geometry: str, n_coord: int) -> List[np.ndarray]:
    """Return list of outward unit vectors for the given coordination geometry."""
    import math as _math
    geom = geometry.lower().replace("-", "_").replace(" ", "_")

    if geom == "linear" or n_coord == 2:
        return [np.array([0, 0, 1.0]), np.array([0, 0, -1.0])]

    if geom == "trigonal_planar" or n_coord == 3:
        return [
            np.array([1.0, 0.0, 0.0]),
            np.array([-0.5,  _math.sqrt(3) / 2, 0.0]),
            np.array([-0.5, -_math.sqrt(3) / 2, 0.0]),
        ]

    if geom == "tetrahedral":
        s = 1.0 / _math.sqrt(3)
        return [
            np.array([ s,  s,  s]),
            np.array([ s, -s, -s]),
            np.array([-s,  s, -s]),
            np.array([-s, -s,  s]),
        ]

    if geom == "square_planar":
        return [
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        ]

    if geom == "trigonal_bipyramidal" or n_coord == 5:
        return [
            np.array([1.0, 0.0, 0.0]),
            np.array([-0.5,  _math.sqrt(3) / 2, 0.0]),
            np.array([-0.5, -_math.sqrt(3) / 2, 0.0]),
            np.array([0.0, 0.0,  1.0]),
            np.array([0.0, 0.0, -1.0]),
        ]

    if geom == "octahedral" or n_coord == 6:
        return [
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
        ]

    # Fallback: use n_coord to pick geometry
    if n_coord == 2:
        return _geometry_vectors("linear", 2)
    if n_coord == 3:
        return _geometry_vectors("trigonal_planar", 3)
    if n_coord == 4:
        return _geometry_vectors("square_planar", 4)
    if n_coord == 5:
        return _geometry_vectors("trigonal_bipyramidal", 5)
    if n_coord == 6:
        return _geometry_vectors("octahedral", 6)

    raise ValueError(f"Unknown geometry '{geometry}' for n_coord={n_coord}")


def _auto_geometry(n_coord: int) -> str:
    """Pick a sensible default geometry for n_coord."""
    return {
        1: "linear",
        2: "linear",
        3: "trigonal_planar",
        4: "square_planar",
        5: "trigonal_bipyramidal",
        6: "octahedral",
    }.get(n_coord, "octahedral")


def build_complex(
    metal: str = "Cu",
    ligand: str = "NH3",
    n_coord: int = 4,
    geometry: str = "square_planar",
    bond_length: float = 2.0,
    cell_size: float = 15.0,
) -> Dict[str, Any]:
    """
    Build a coordination compound: metal center + n_coord monodentate ligands.

    Parameters
    ----------
    metal      : element symbol for the central metal (e.g. 'Cu', 'Fe', 'Pt')
    ligand     : ligand name from the internal library (H2O, NH3, CO, Cl, F,
                 CN, NO, OH, PH3, SCN)
    n_coord    : coordination number
    geometry   : coordination geometry name (tetrahedral, square_planar,
                 octahedral, linear, trigonal_planar, trigonal_bipyramidal)
    bond_length: metal–binding-atom distance in Å
    cell_size  : cube cell edge length in Å

    Returns
    -------
    dict with keys: ok, poscar, label, formula, n_atoms, structure_type,
                    description, ase_code
    """
    import io as _io
    from ase.io.vasp import write_vasp as _write_vasp

    lib = _ligand_library()
    supported = sorted(lib.keys())
    ligand_key = ligand.upper() if ligand.upper() in lib else ligand
    if ligand_key not in lib:
        # try case-insensitive
        matches = [k for k in lib if k.lower() == ligand.lower()]
        if matches:
            ligand_key = matches[0]
        else:
            return {
                "ok": False,
                "error": (
                    f"Unknown ligand '{ligand}'. "
                    f"Supported ligands: {supported}"
                ),
            }

    # Auto-pick geometry if n_coord doesn't match provided geometry
    geom = geometry.lower().replace("-", "_").replace(" ", "_")
    geom_n = {
        "linear": 2, "trigonal_planar": 3, "tetrahedral": 4,
        "square_planar": 4, "trigonal_bipyramidal": 5, "octahedral": 6,
    }
    if geom in geom_n and geom_n[geom] != n_coord:
        # Auto-pick correct geometry for this n_coord
        geometry = _auto_geometry(n_coord)

    try:
        vectors = _geometry_vectors(geometry, n_coord)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    # Trim vectors to n_coord
    vectors = vectors[:n_coord]

    lig_symbols, lig_local = lib[ligand_key]

    # Build Atoms object
    symbols = [metal]
    positions = [np.array([0.0, 0.0, 0.0])]

    for v in vectors:
        v_unit = v / np.linalg.norm(v)
        R = _rot_z_to_v(v_unit)
        for i, (sym, lc) in enumerate(zip(lig_symbols, lig_local)):
            # Rotate local coords so +z → v
            rotated = R @ lc
            # Translate so binding atom lands at bond_length * v
            pos = bond_length * v_unit + rotated
            symbols.append(sym)
            positions.append(pos)

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_cell([cell_size, cell_size, cell_size])
    atoms.center()
    atoms.pbc = False

    # Write POSCAR to string
    buf = _io.StringIO()
    _write_vasp(buf, atoms, vasp5=True)
    poscar_str = buf.getvalue()

    formula = atoms.get_chemical_formula()
    n_atoms = len(atoms)
    label = f"{metal}_{ligand_key}_{n_coord}_{geometry}"
    description = (
        f"{metal} center with {n_coord} {ligand_key} ligands "
        f"({geometry}, M-L = {bond_length:.2f} Å)"
    )

    # Generate equivalent ASE code
    ase_code_lines = [
        "import numpy as np",
        "from ase import Atoms",
        "from ase.io.vasp import write_vasp",
        "from server.execution.structure_agent import build_complex",
        "",
        f"result = build_complex(",
        f"    metal='{metal}', ligand='{ligand_key}', n_coord={n_coord},",
        f"    geometry='{geometry}', bond_length={bond_length}, cell_size={cell_size}",
        ")",
        "# result['poscar'] contains the POSCAR string",
    ]
    ase_code = "\n".join(ase_code_lines)

    return {
        "ok": True,
        "poscar": poscar_str,
        "label": label,
        "formula": formula,
        "n_atoms": n_atoms,
        "structure_type": "complex",
        "description": description,
        "ase_code": ase_code,
    }
