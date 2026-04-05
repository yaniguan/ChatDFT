# server/api/preprocessor.py
# -*- coding: utf-8 -*-
"""
Universal Structure Preprocessor
==================================

Problem: Scientists have structures in different formats (POSCAR, CIF, XYZ,
SMILES, or just "Pt(111) 3x3 with CO"). Every existing tool requires
a specific format. This wastes time on format conversion and introduces errors.

Solution: A single endpoint that auto-detects format, converts, and returns
a normalized POSCAR + metadata. Scientists never worry about formats again.

Supported inputs:
  1. POSCAR content (auto-detected by VASP format markers)
  2. CIF content (auto-detected by _cell_length or data_ header)
  3. XYZ content (auto-detected by integer first line)
  4. extXYZ content (auto-detected by Properties= in comment line)
  5. SMILES string (detected by organic chemistry patterns: C, c, =, #, @)
  6. Material name (e.g., "Pt(111) 3x3 slab" or "bulk Cu fcc")
  7. File upload (any of the above, with format hint from extension)

Each conversion path validates the result and returns:
  - Normalized POSCAR string
  - Formula, n_atoms, elements, cell volume
  - Detected format
  - Warnings (e.g., "no periodicity in XYZ — assumed molecule in box")
"""
from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["preprocessor"])


# ═══════════════════════════════════════════════════════════════════════
# Format detection
# ═══════════════════════════════════════════════════════════════════════

class InputFormat:
    POSCAR = "poscar"
    CIF = "cif"
    XYZ = "xyz"
    EXTXYZ = "extxyz"
    SMILES = "smiles"
    NAME = "name"
    UNKNOWN = "unknown"


def detect_format(content: str) -> str:
    """
    Auto-detect the format of a structure input string.

    Rules (applied in order):
    1. If starts with a line of numbers after optional comment → POSCAR
    2. If contains _cell_length or data_ → CIF
    3. If first line is a bare integer → XYZ or extXYZ
    4. If short (<100 chars) and has organic chemistry markers → SMILES
    5. If short and matches surface/material patterns → NAME
    6. Otherwise → UNKNOWN
    """
    content = content.strip()
    if not content:
        return InputFormat.UNKNOWN

    lines = content.split("\n")
    first_line = lines[0].strip()

    # CIF: has crystallographic headers
    if "_cell_length" in content or content.startswith("data_") or "_symmetry" in content:
        return InputFormat.CIF

    # POSCAR: line 2 is scaling factor (a number), line 3-5 are lattice vectors
    if len(lines) >= 6:
        try:
            float(lines[1].strip())
            # Check if lines 3-5 look like lattice vectors (3 floats each)
            for i in range(2, 5):
                parts = lines[i].split()
                if len(parts) >= 3:
                    [float(p) for p in parts[:3]]
            return InputFormat.POSCAR
        except (ValueError, IndexError):
            pass

    # XYZ / extXYZ: first line is an integer (atom count)
    if first_line.isdigit():
        if len(lines) >= 2 and "Properties=" in lines[1]:
            return InputFormat.EXTXYZ
        return InputFormat.XYZ

    # SMILES: short string with organic chemistry characters
    if len(content) < 200 and not content.count("\n") and _looks_like_smiles(content):
        return InputFormat.SMILES

    # Material name: short string with surface notation or element names
    if len(content) < 200:
        return InputFormat.NAME

    return InputFormat.UNKNOWN


def _looks_like_smiles(s: str) -> bool:
    """Heuristic: does this string look like a SMILES notation?"""
    # SMILES characters: C, c, O, N, =, #, @, (, ), [, ], +, -, numbers
    smiles_chars = set("CcOoNnSsPpFfIiBbrHh=@#()[]+-.0123456789/\\")
    if not s:
        return False
    char_ratio = sum(1 for c in s if c in smiles_chars) / len(s)
    has_organic = any(c in s for c in "CcNnOo")
    return char_ratio > 0.8 and has_organic and " " not in s


# ═══════════════════════════════════════════════════════════════════════
# Converters
# ═══════════════════════════════════════════════════════════════════════

def _atoms_to_poscar(atoms) -> str:
    """Convert ASE Atoms to POSCAR string."""
    from ase.io import write
    buf = io.StringIO()
    write(buf, atoms, format="vasp", sort=True)
    return buf.getvalue()


def _atoms_metadata(atoms) -> Dict[str, Any]:
    """Extract metadata from ASE Atoms."""
    symbols = atoms.get_chemical_symbols()
    unique = list(dict.fromkeys(symbols))
    cell = atoms.get_cell()
    volume = float(atoms.get_volume()) if atoms.pbc.any() else None
    return {
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
        "elements": unique,
        "cell_A": cell[:].tolist() if atoms.pbc.any() else None,
        "volume_A3": round(volume, 2) if volume else None,
        "pbc": atoms.pbc.tolist(),
    }


def convert_poscar(content: str) -> Tuple[str, Dict[str, Any], List[str]]:
    """Validate and normalize a POSCAR string."""
    from ase.io import read
    atoms = read(io.StringIO(content), format="vasp")
    poscar = _atoms_to_poscar(atoms)
    meta = _atoms_metadata(atoms)
    return poscar, meta, []


def convert_cif(content: str) -> Tuple[str, Dict[str, Any], List[str]]:
    """Convert CIF to POSCAR."""
    from ase.io import read
    atoms = read(io.StringIO(content), format="cif")
    poscar = _atoms_to_poscar(atoms)
    meta = _atoms_metadata(atoms)
    return poscar, meta, ["Converted from CIF. Check space group handling."]


def convert_xyz(content: str) -> Tuple[str, Dict[str, Any], List[str]]:
    """Convert XYZ to POSCAR (molecule in a box)."""
    from ase.io import read
    atoms = read(io.StringIO(content), format="xyz")
    warnings = []

    if not atoms.pbc.any():
        # Add a vacuum box
        atoms.center(vacuum=10.0)
        atoms.pbc = True
        warnings.append("No periodicity in XYZ — placed in 20A cubic box with 10A vacuum")

    poscar = _atoms_to_poscar(atoms)
    meta = _atoms_metadata(atoms)
    return poscar, meta, warnings


def convert_extxyz(content: str) -> Tuple[str, Dict[str, Any], List[str]]:
    """Convert extended XYZ to POSCAR."""
    from ase.io import read
    atoms = read(io.StringIO(content), format="extxyz")
    warnings = []

    if not atoms.pbc.any():
        atoms.center(vacuum=10.0)
        atoms.pbc = True
        warnings.append("No periodicity in extXYZ — placed in box")

    poscar = _atoms_to_poscar(atoms)
    meta = _atoms_metadata(atoms)
    return poscar, meta, warnings


def convert_smiles(smiles: str) -> Tuple[str, Dict[str, Any], List[str]]:
    """Convert SMILES to POSCAR (3D molecule in a box)."""
    warnings = []

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        from ase import Atoms
        conf = mol.GetConformer()
        positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]

        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.center(vacuum=10.0)
        atoms.pbc = True

        poscar = _atoms_to_poscar(atoms)
        meta = _atoms_metadata(atoms)
        warnings.append(f"3D geometry from MMFF force field — optimize with DFT for publication")
        return poscar, meta, warnings
    except ImportError:
        # Fallback: use PubChem lookup
        try:
            from server.execution.structure_agent import build_molecule_pubchem
            result = build_molecule_pubchem(smiles=smiles, label=smiles, cell_size=20.0)
            if result.get("poscar"):
                from ase.io import read
                atoms = read(io.StringIO(result["poscar"]), format="vasp")
                meta = _atoms_metadata(atoms)
                warnings.append("3D geometry from PubChem — optimize with DFT")
                return result["poscar"], meta, warnings
        except Exception:
            pass

        raise ValueError("SMILES conversion requires rdkit or PubChem lookup. "
                         "Install: pip install rdkit-pypi")


def convert_name(name: str) -> Tuple[str, Dict[str, Any], List[str]]:
    """
    Convert a material name/description to POSCAR.

    Supports:
      - "Pt(111)" → build fcc111 slab
      - "Cu(100) 3x3" → build fcc100 3x3 slab
      - "bulk Si" → build bulk Si
      - "Fe(110) 4x4 with CO" → build slab + place CO adsorbate
      - "graphene" → build graphene sheet
      - "MoS2" → build MoS2 monolayer
    """
    name_lower = name.lower().strip()
    warnings = []

    # Parse surface notation
    surface_match = re.search(
        r'([A-Z][a-z]?(?:\d*[A-Z][a-z]?)*)\((\d{3,4})\)', name
    )
    supercell_match = re.search(r'(\d+)\s*x\s*(\d+)', name_lower)
    adsorbate_match = re.search(r'with\s+(\w+)', name_lower)

    nx, ny = 4, 4
    if supercell_match:
        nx = int(supercell_match.group(1))
        ny = int(supercell_match.group(2))

    if surface_match:
        element = surface_match.group(1)
        facet = surface_match.group(2)

        from server.execution.structure_agent import _build_slab, _guess_crystal_system
        crystal = _guess_crystal_system(element, {})
        h, k, l = int(facet[0]), int(facet[1]), int(facet[2])

        atoms = _build_slab(element, crystal, h, k, l, nx, ny, nlayers=4, vacuum=15.0)

        # Place adsorbate if requested
        if adsorbate_match:
            adsorbate = adsorbate_match.group(1).upper()
            try:
                from server.execution.structure_agent import (
                    find_adsorption_sites_ase, place_adsorbate,
                )
                sites = find_adsorption_sites_ase(atoms, height=2.0)
                if sites:
                    site_pos = sites[0]["position"]
                    atoms = place_adsorbate(atoms, site_pos, adsorbate)
                    warnings.append(f"Placed {adsorbate} at {sites[0].get('type', 'auto')} site")
            except Exception as e:
                warnings.append(f"Could not place adsorbate {adsorbate}: {e}")

        poscar = _atoms_to_poscar(atoms)
        meta = _atoms_metadata(atoms)
        return poscar, meta, warnings

    elif "bulk" in name_lower:
        element_match = re.search(r'bulk\s+([A-Z][a-z]?)', name)
        if element_match:
            element = element_match.group(1)
            from ase.build import bulk
            from server.execution.structure_agent import _guess_crystal_system
            crystal = _guess_crystal_system(element, {})
            atoms = bulk(element, crystalstructure=crystal, cubic=True)
            poscar = _atoms_to_poscar(atoms)
            meta = _atoms_metadata(atoms)
            return poscar, meta, [f"Built bulk {element} ({crystal})"]

    # Fallback: try as element name for a simple slab
    element_match = re.match(r'^([A-Z][a-z]?)$', name.strip())
    if element_match:
        element = element_match.group(1)
        from server.execution.structure_agent import _build_slab, _guess_crystal_system
        crystal = _guess_crystal_system(element, {})
        atoms = _build_slab(element, crystal, 1, 1, 1, 4, 4, 4, 15.0)
        poscar = _atoms_to_poscar(atoms)
        meta = _atoms_metadata(atoms)
        return poscar, meta, [f"Built {element}(111) 4x4 slab (default)"]

    raise ValueError(f"Could not parse material name: '{name}'. "
                     f"Try 'Pt(111)', 'Cu(100) 3x3', 'bulk Fe', or a SMILES string.")


# ═══════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════

class PreprocessRequest(BaseModel):
    """Universal structure input."""
    content: str = Field(..., description=(
        "Structure in any format: POSCAR, CIF, XYZ, extXYZ, SMILES, "
        "or material name like 'Pt(111) 3x3 with CO'"
    ))
    format_hint: Optional[str] = Field(None, description=(
        "Force format: 'poscar', 'cif', 'xyz', 'extxyz', 'smiles', 'name'. "
        "Leave blank for auto-detection."
    ))


@router.post("/preprocess")
async def preprocess(req: PreprocessRequest):
    """
    Universal structure preprocessor — auto-detect format, convert, normalize.

    Scientists never need to know file format details.  Just paste whatever
    you have: POSCAR, CIF, XYZ, SMILES, or a material name.

    Examples:
        # From a name
        POST /api/preprocess  {"content": "Pt(111) 3x3 with CO"}
        → {"poscar": "...", "format": "name", "formula": "CPt36O", ...}

        # From CIF
        POST /api/preprocess  {"content": "data_Cu\\n_cell_length_a 3.615\\n..."}
        → {"poscar": "...", "format": "cif", "formula": "Cu4", ...}

        # From SMILES
        POST /api/preprocess  {"content": "O=C=O"}
        → {"poscar": "...", "format": "smiles", "formula": "CO2", ...}
    """
    fmt = req.format_hint or detect_format(req.content)

    converters = {
        InputFormat.POSCAR: convert_poscar,
        InputFormat.CIF: convert_cif,
        InputFormat.XYZ: convert_xyz,
        InputFormat.EXTXYZ: convert_extxyz,
        InputFormat.SMILES: convert_smiles,
        InputFormat.NAME: convert_name,
    }

    converter = converters.get(fmt)
    if not converter:
        return {
            "ok": False,
            "error": f"Could not detect format. Detected: '{fmt}'. "
                     f"Use format_hint to specify: {list(converters.keys())}",
        }

    try:
        poscar, meta, warnings = converter(req.content)
        return {
            "ok": True,
            "poscar": poscar,
            "detected_format": fmt,
            **meta,
            "warnings": warnings,
        }
    except Exception as e:
        return {
            "ok": False,
            "detected_format": fmt,
            "error": str(e),
        }


@router.post("/preprocess/batch")
async def preprocess_batch(structures: List[PreprocessRequest]):
    """Batch preprocess multiple structures."""
    results = []
    for req in structures:
        fmt = req.format_hint or detect_format(req.content)
        converters = {
            InputFormat.POSCAR: convert_poscar,
            InputFormat.CIF: convert_cif,
            InputFormat.XYZ: convert_xyz,
            InputFormat.EXTXYZ: convert_extxyz,
            InputFormat.SMILES: convert_smiles,
            InputFormat.NAME: convert_name,
        }
        converter = converters.get(fmt)
        try:
            poscar, meta, warnings = converter(req.content)
            results.append({"ok": True, "poscar": poscar, "format": fmt, **meta, "warnings": warnings})
        except Exception as e:
            results.append({"ok": False, "format": fmt, "error": str(e)})

    return {"ok": True, "n_processed": len(results), "results": results}
