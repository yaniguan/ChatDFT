# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from .base import BaseEngine

class QEEngine(BaseEngine):
    name = "qe"
    default_submit = "sbatch run.slurm"
    default_filters: Sequence[str] = ("pw.out", "dos.out", "projwfc.out", "stdout", "stderr")
    slurm_fname = "run.slurm"
    pbs_fname = "job.sh"

    # 可以根据需要重写 _parse_job_id / prepare_script 行为