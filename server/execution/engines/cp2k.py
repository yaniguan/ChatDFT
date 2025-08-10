# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from .base import BaseEngine

class CP2KEngine(BaseEngine):
    name = "cp2k"
    default_submit = "sbatch run.slurm"
    default_filters: Sequence[str] = ("cp2k.out", "stdout", "stderr")
    slurm_fname = "run.slurm"
    pbs_fname = "job.sh"