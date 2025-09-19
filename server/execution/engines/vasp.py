# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from .base import BaseEngine

class VaspEngine(BaseEngine):
    name = "vasp"
    default_submit = "sbatch run.slurm"  # 也可从 cluster_config 覆盖
    default_filters: Sequence[str] = ("OUTCAR", "vasprun.xml", "OSZICAR", "EIGENVAL", "CONTCAR", "stdout", "stderr")
    slurm_fname = "run.slurm"
    pbs_fname = "job.sh"

    # 这里保留父类逻辑：prepare_script / submit / wait / fetch_outputs 都会记录与发事件
    # 如需特殊 job_id 解析，可覆写 _parse_job_id()

# server/execution/engines/vasp.py
from pathlib import Path

class VaspEngine:
    def render_inputs(self, payload: dict, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) 结构（slab/adsorption/bulk）—— 复用你现有的结构构造
        # from .vasp_structure import build_structure
        # poscar_str = build_structure(payload)
        # (out_dir / "POSCAR").write_text(poscar_str)

        # 2) 参数（INCAR/KPOINTS/POTCAR）—— 复用你现有的参数生成
        # from .vasp_parameters import make_incar, make_kpoints, ensure_potcar
        # (out_dir / "INCAR").write_text(make_incar(payload))
        # (out_dir / "KPOINTS").write_text(make_kpoints(payload))
        # ensure_potcar(payload, out_dir)

        # 最小兜底：占位，先打通任务流
        for name in ("POSCAR", "INCAR", "KPOINTS"):
            p = out_dir / name
            if not p.exists():
                p.write_text(f"# TODO: generate {name} by your vasp_* modules\n")
        return {
            "poscar": str(out_dir / "POSCAR"),
            "incar": str(out_dir / "INCAR"),
            "kpoints": str(out_dir / "KPOINTS")
        }