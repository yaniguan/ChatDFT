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