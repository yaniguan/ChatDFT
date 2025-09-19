from __future__ import annotations
import io
import asyncio
from server.utils.openai_wrapper import chatgpt_call
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, Tuple, List
import json, re

from ase.io import write
from ase import Atoms
from ase.data import chemical_symbols
from ase.build import fcc111, fcc100, fcc110

# 118 个元素符号（简化：常见金属+非金属即可，也可全量）
_VALID_ELEMS = {
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"
}

def _normalize_element_name(raw: str) -> str:
    """将 'Pt111' / 'Pt(111)' / 'pt-111' / ' Pt 111 ' 清洗为 'Pt'"""
    s = str(raw or "").strip()
    if not s:
        return ""
    # 去空格和多余符号
    s = s.replace("_", "-")
    # 典型写法：Pt111 / Pt(111) / pt-111 / Pt 111
    m = re.match(r"^\s*([A-Za-z]{1,2})\s*(?:\(?\d{1,3}\)?|-\d{1,3})?\s*$", s)
    sym = m.group(1).capitalize() if m else s.capitalize()
    if sym not in _VALID_ELEMS:
        # 再尝试从“Pt111(111)”这类字符串抽第一个元素符号
        m2 = re.search(r"([A-Z][a-z]?)", s)
        sym = m2.group(1) if (m2 and m2.group(1) in _VALID_ELEMS) else ""
    return sym

def _parse_miller(payload) -> tuple[int,int,int]:
    """从 'miller_index' / 'facet' / '111' / [1,1,1] 解析 (h,k,l)"""
    mi = (payload or {}).get("miller_index")
    if isinstance(mi, (list, tuple)) and len(mi) >= 3:
        return int(mi[0]), int(mi[1]), int(mi[2])
    # 字符串：'1 1 1' / '111' / '(111)'
    txt = str(mi or (payload or {}).get("facet") or "").strip()
    if txt:
        txt = txt.replace("(", "").replace(")", "").replace(",", " ").replace("-", " ")
        parts = [p for p in re.split(r"\s+", txt) if p]
        if len(parts) == 3 and all(p.isdigit() or (p and p[0] in "+-" and p[1:].isdigit()) for p in parts):
            return int(parts[0]), int(parts[1]), int(parts[2])
        if len(parts) == 1 and parts[0].isdigit() and len(parts[0]) == 3:
            return int(parts[0][0]), int(parts[0][1]), int(parts[0][2])
    # 名称里兜底，比如 "Build slab — Pt111(111)"
    name = str((payload or {}).get("_task_name") or "")
    m = re.search(r"\(?([+-]?\d)\s*[, ]\s*([+-]?\d)\s*[, ]\s*([+-]?\d)\)?", name)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m2 = re.search(r"\((\d)(\d)(\d)\)", name)
    if m2:
        return int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
    return (1,1,1)  # 兜底

def _extract_inputs(task: dict) -> dict:
    p = ((task.get("params") or {}).get("payload") or {}).copy()
    p["_task_name"] = task.get("name") or ""
    element = _normalize_element_name(p.get("element") or p.get("metal") or p.get("symbol") or p.get("element_symbol") or p.get("element_name") or p.get("_task_name"))
    if not element:
        raise ValueError("Cannot parse element from inputs. Please set element='Pt' (not 'Pt111').")
    
    # (s: str | None, facet: str | None)
    for temp_data in task.get('params')['form']:
        if temp_data['key'] == 'miller_index': 
            h,k,l = _parse_miller(s=temp_data['value'], facet=task.get('params')['payload']['facet'])
            break

    layers = int(float(p.get("layers") or 4))
    vac = float(p.get("vacuum") or p.get("vacuum_thickness") or 15.0)
    sc = str(p.get("supercell") or "4x4x1").lower()
    p_clean = {
        "element": element,
        "miller_index": [h,k,l],
        "layers": layers,
        "vacuum_thickness": vac,
        "supercell": sc,
        "engine": (p.get("engine") or "vasp").lower()
    }
    return p_clean
# ==== end helpers ====
def _normalize_element_name(raw: str | None, default: str = "Cu") -> str:
    """
    从各种写法里提取元素符号：
    'Pt111' 'Pt(111)' 'pt-111' 'Pt_111' -> 'Pt'
    如果提取不到，返回 default
    """
    s = (raw or "").strip()
    if not s:
        return default
    # 先找第一个像元素的 token（大写字母+可选小写）
    m = re.search(r"[A-Z][a-z]?", s)
    if m:
        return m.group(0)
    # 兜底：只保留字母，取前两个字符修正大小写
    letters = re.sub(r"[^A-Za-z]", "", s)
    if len(letters) >= 1:
        if len(letters) == 1:
            return letters[0].upper()
        return letters[0].upper() + letters[1].lower()
    return default

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _clean_symbol(x: str, fallback: str = "Cu") -> str:
    s = (x or "").strip()
    if not s: return fallback
    s = s[0].upper() + s[1:].lower()
    return s if s in chemical_symbols else fallback

def _parse_supercell(s: str | None) -> Tuple[int,int,int]:
    txt = (s or "4x4x1").lower().replace("*","x").replace("×","x")
    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\s*$", txt)
    if not m: return (4,4,1)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

def _parse_miller(s: str | None, facet: str | None) -> str:
    if s and s.strip(): return "".join(re.findall(r"\d", s))
    if facet and str(facet).strip(): return "".join(re.findall(r"\d", str(facet)))
    return "111"

# 一些常用金属的晶格常数（Å），仅用于 fcc 系列 slab；其余默认 Cu
_A0 = {
    "Cu": 3.615, "Pt": 3.924, "Ni": 3.523, "Pd": 3.889, "Ag": 4.086, "Au": 4.078, "Co": 3.544, "Rh": 3.803
}

def _build_fcc_slab(element: str, facet: str, nx: int, ny: int, nlayers: int, vacuum: float) -> Atoms:
    a0 = _A0.get(element, _A0["Cu"])
    facet = facet.strip()
    if facet == "111":
        slab = fcc111(element, size=(nx, ny, nlayers), a=a0, vacuum=vacuum, orthogonal=True)
    elif facet == "100":
        slab = fcc100(element, size=(nx, ny, nlayers), a=a0, vacuum=vacuum, orthogonal=True)
    elif facet == "110":
        slab = fcc110(element, size=(nx, ny, nlayers), a=a0, vacuum=vacuum, orthogonal=True)
    else:
        # 未知面，退化到 111
        slab = fcc111(element, size=(nx, ny, nlayers), a=a0, vacuum=vacuum, orthogonal=True)
    return slab

def _write_all(job_dir: Path, atoms: Atoms):
    # VASP：同时写 slab.POSCAR 和标准 POSCAR（VASP5/diret）
    write(job_dir / "slab.POSCAR", atoms, format="vasp", direct=True, vasp5=True)
    write(job_dir / "POSCAR",      atoms, format="vasp", direct=True, vasp5=True)
    # 兼容你之前的命名
    write(job_dir / "structure.POSCAR", atoms, format="vasp", direct=True, vasp5=True)
    # 其他格式
    write(job_dir / "structure.cif", atoms)
    # 元信息
    meta = {
        "n_atoms": len(atoms),
        "symbols": atoms.get_chemical_symbols(),
        "cell": atoms.cell.array.tolist(),
        "pbc": list(map(bool, atoms.get_pbc())),
    }
    (job_dir / "structure.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    # POTCAR.spec：按 POSCAR 中元素顺序（去重保持顺序）
    uniq = []
    for s in atoms.get_chemical_symbols():
        if s not in uniq: uniq.append(s)
    (job_dir / "POTCAR.spec").write_text(json.dumps(uniq, indent=2))

class StructureAgent:
    """
    StructureAgent: Unified interface for structure generation.
    Supports: bulk, slab, adsorption, co-adsorption, transition state, and structure modification (doping, alloying, symmetry, expand, increase layers).
    All steps fallback to ASE if LLM result is missing or invalid.
    """
    def build(self, task: dict, job_dir: Path) -> dict:
        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        inp = _extract_inputs(task)
        (job_dir / "_inputs.json").write_text(json.dumps(inp, indent=2))
        mode = (task.get("params", {}).get("payload", {}).get("structure_type") or "slab").lower()
        # 1. Bulk structure
        if mode == "bulk":
            atoms = self._build_bulk(inp)
        # 2. Slab model
        elif mode == "slab":
            atoms = self._build_slab(inp)
        # 3. Adsorption
        elif mode == "adsorption":
            atoms = self._build_adsorption(inp)
        # 4. Co-adsorption
        elif mode == "co-adsorption":
            atoms = self._build_coadsorption(inp)
        # 5. Transition state trajectory
        elif mode == "ts" or mode == "transition_state":
            atoms = self._build_ts(inp)
        # 6. Structure modification
        else:
            atoms = self._build_slab(inp)  # fallback
        # Structure modification (doping, alloying, symmetry, expand, increase layers)
        atoms = self._modify_structure(atoms, inp)
        _write_all(job_dir, atoms)
        label = f"{inp['element']}({inp['miller_index']}) {inp['supercell']} layers={inp['layers']} vac={inp['vacuum_thickness']}"
        (job_dir / "_param_log.jsonl").write_text(json.dumps(inp) + "\n")
        return {
            "ok": True,
            "label": label,
            "element": inp["element"],
            "facet": inp["miller_index"],
            "layers": inp["layers"],
            "vacuum": inp["vacuum_thickness"],
            "supercell": inp["supercell"],
            "files": ["POSCAR", "slab.POSCAR", "structure.POSCAR", "structure.cif", "structure.json", "POTCAR.spec"]
        }

    async def _build_bulk(self, inp: dict) -> Atoms:
        prompt = (
            f"Please find the appropriate crystal structure for {inp['element']} from the Materials Project database. "
            "Download the structure via API and keep the unit cell size unchanged. "
            "Return the structure in POSCAR format."
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            llm_result = await chatgpt_call(messages)
            poscar_str = llm_result.get("poscar") or llm_result.get("content")
            if poscar_str:
                from ase.io import read
                atoms = read(io.StringIO(poscar_str), format="vasp")
                return atoms
        except Exception:
            pass
        from ase.build import bulk
        element = inp["element"]
        a0 = _A0.get(element, _A0["Cu"])
        return bulk(element, cubic=True, a=a0)

    async def _build_slab(self, inp: dict) -> Atoms:
        prompt = (
            f"Given the bulk structure at path '{{bulk_path}}', generate a slab model with the following parameters: "
            f"crystal facet: {inp['miller_index']}, number of layers: {inp['layers']}, unit cell size: {inp['supercell']}, "
            f"vacuum thickness: {inp['vacuum_thickness']} angstrom. "
            "Return the structure in POSCAR format."
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            llm_result = await chatgpt_call(messages)
            poscar_str = llm_result.get("poscar") or llm_result.get("content")
            if poscar_str:
                from ase.io import read
                atoms = read(io.StringIO(poscar_str), format="vasp")
                return atoms
        except Exception:
            pass
        element = inp["element"]
        h, k, l = inp["miller_index"]
        nx, ny, _ = _parse_supercell(inp["supercell"])
        layers = inp["layers"]
        vacuum = inp["vacuum_thickness"]
        facet = "".join(str(x) for x in [h, k, l])
        return _build_fcc_slab(element, facet, nx, ny, layers, vacuum)

    async def _build_adsorption(self, inp: dict) -> Atoms:
        prompt = (
            f"Given the slab structure, add the adsorbate species {inp.get('adsorbate', 'H')} to the surface. "
            "Return the structure in POSCAR format."
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            llm_result = await chatgpt_call(messages)
            poscar_str = llm_result.get("poscar") or llm_result.get("content")
            if poscar_str:
                from ase.io import read
                atoms = read(io.StringIO(poscar_str), format="vasp")
                return atoms
        except Exception:
            pass
        return await self._build_slab(inp)

    async def _build_coadsorption(self, inp: dict) -> Atoms:
        prompt = (
            f"Given the slab structure, add the co-adsorbate species {inp.get('co_adsorbate', 'H+O')} to the surface. "
            "Return the structure in POSCAR format."
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            llm_result = await chatgpt_call(messages)
            poscar_str = llm_result.get("poscar") or llm_result.get("content")
            if poscar_str:
                from ase.io import read
                atoms = read(io.StringIO(poscar_str), format="vasp")
                return atoms
        except Exception:
            pass
        return await self._build_slab(inp)

    async def _build_ts(self, inp: dict) -> Atoms:
        prompt = (
            f"Given the initial and final structures, generate a transition state trajectory for the reaction. "
            "Return the transition state structure in POSCAR format."
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            llm_result = await chatgpt_call(messages)
            poscar_str = llm_result.get("poscar") or llm_result.get("content")
            if poscar_str:
                from ase.io import read
                atoms = read(io.StringIO(poscar_str), format="vasp")
                return atoms
        except Exception:
            pass
        return await self._build_slab(inp)

    async def _modify_structure(self, atoms: Atoms, inp: dict) -> Atoms:
        prompt = (
            f"Given the structure, apply the following modifications: "
            f"doping: {inp.get('doping', '')}, alloying: {inp.get('alloying', '')}, "
            f"symmetry: {inp.get('symmetry', '')}, expand: {inp.get('expand', '')}, "
            f"increase vacuum thickness: {inp.get('vacuum_thickness', '')}. "
            "Return the modified structure in POSCAR format."
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            llm_result = await chatgpt_call(messages)
            poscar_str = llm_result.get("poscar") or llm_result.get("content")
            if poscar_str:
                from ase.io import read
                atoms = read(io.StringIO(poscar_str), format="vasp")
                return atoms
        except Exception:
            pass
        return atoms