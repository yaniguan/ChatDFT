# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, List
import json, re

from ase.io import write
from ase import Atoms
from ase.data import chemical_symbols
from ase.build import fcc111, fcc100, fcc110
# 顶部 imports 下方加：
import re
# ==== add near the top of structure_agent.py ====
import re, json
from pathlib import Path

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
    h,k,l = _parse_miller(p)
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
    build(task, job_dir) 读取 task.params.payload 与 form 覆盖，生成 slab 结构与文件：
      - POSCAR / slab.POSCAR / structure.POSCAR
      - structure.cif / structure.json
      - POTCAR.spec （元素有序列表，供 HPC 端从 $VASP_PP_PATH 拼 POTCAR）
    """
def build(self, task: dict, job_dir: Path) -> dict:
        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        # —— 解析 & 校验 —— #
        inp = _extract_inputs(task)     # ← 用上面的函数
        (job_dir / "_inputs.json").write_text(json.dumps(inp, indent=2))  # 记录解析后的输入，便于调试

        elem = inp["element"]            # 确保是 'Pt'
        h,k,l = inp["miller_index"]
        layers = inp["layers"]
        vac = inp["vacuum_thickness"]
        sc = inp["supercell"]
        _ensure_dir(job_dir)
        payload = (task.get("params") or {}).get("payload") or {}
        form = (task.get("params") or {}).get("form") or []

        # 表单覆盖（前端右侧表单）
        overrides = {f.get("key"): f.get("value") for f in form if isinstance(f, dict) and "key" in f}

        element_raw = payload.get("element") or payload.get("metal") or payload.get("species")
        element = _normalize_element_name(element_raw, default="Cu")
        payload["element"] = element  # 回填，后面参数/模板也用这个
        facet = (payload.get("facet") or "111").strip()
        miller = payload.get("miller") or payload.get("miller_index") or "1 1 1"
        if isinstance(miller, str):
            try:
                h, k, l = [int(x) for x in re.findall(r"-?\d+", miller)[:3]] or [1, 1, 1]
            except Exception:
                h, k, l = 1, 1, 1
        else:
            h, k, l = (miller + [1, 1, 1])[:3]

        # 写 POTCAR.spec 供模板兜底组装
        try:
            (job_dir / "POTCAR.spec").write_text(json.dumps([element], indent=2))
        except Exception:
            pass
        layers  = int(overrides.get("layers") or payload.get("layers") or 4)
        vacuum  = float(overrides.get("vacuum_thickness") or payload.get("vacuum") or 15.0)
        scell   = _parse_supercell(overrides.get("supercell") or payload.get("supercell") or "4x4x1")
        nx, ny, _ = scell

        # 目前假设 fcc（Cu、Pt、Ni…）；如要 bcc/hcp 可加判断映射
        atoms = _build_fcc_slab(element, facet, nx, ny, layers, vacuum)
        _write_all(job_dir, atoms)

        # 记录友好命名（避免奇怪的 Rg）
        label = f"{element}({facet}) {nx}x{ny} layers={layers} vac={vacuum}"
        (job_dir / "_param_log.jsonl").write_text(
            json.dumps({
                "element": element, "facet": facet, "layers": layers, "vacuum": vacuum, "supercell": f"{nx}x{ny}x1"
            }) + "\n"
        )

        return {
            "ok": True,
            "label": label,
            "element": element,
            "facet": facet,
            "layers": layers,
            "vacuum": vacuum,
            "supercell": [nx, ny, 1],
            "files": ["POSCAR", "slab.POSCAR", "structure.POSCAR", "structure.cif", "structure.json", "POTCAR.spec"]
        }