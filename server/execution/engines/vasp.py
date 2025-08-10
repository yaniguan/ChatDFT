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