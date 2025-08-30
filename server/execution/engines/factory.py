# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional
from .vasp import VaspEngine
from .qe import QEEngine
from .cp2k import CP2KEngine
from .base import BaseEngine

def engine_factory(name: str) -> BaseEngine:
    n = (name or "vasp").strip().lower()
    if n in {"vasp", "vasp_std", "vasp_gam", "vasp_ncl"}:
        return VaspEngine()
    if n in {"qe", "quantum-espresso", "pw.x", "pw"}:
        return QEEngine()
    if n in {"cp2k"}:
        return CP2KEngine()
    # 默认 VASP
    return VaspEngine()

# server/execution/engines/factory.py
from pathlib import Path

def get_engine(name: str):
    n = (name or "vasp").lower()
    if n == "vasp":
        from .vasp import VaspEngine
        return VaspEngine()
    elif n in ("qe", "quantum_espresso", "pwscf"):
        from .qe import QEEngine
        return QEEngine()
    elif n == "cp2k":
        from .cp2k import CP2KEngine
        return CP2KEngine()
    else:
        raise ValueError(f"Unknown engine: {name}")

# 统一渲染接口（你可以在 task_routes.render() 中直接调用）
def render_inputs(payload: dict, out_dir: str | Path):
    engine = get_engine((payload.get("calc") or {}).get("engine", "vasp"))
    return engine.render_inputs(payload, Path(out_dir))