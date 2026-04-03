"""
VASP OUTCAR Parser for SCF Trajectory Extraction
==================================================

Parses real VASP OUTCARs to extract SCF convergence trajectories,
enabling validation of the FFT sloshing detector on real DFT data.

Usage
-----
    from science.validation.outcar_parser import parse_outcar, OUTCARDataset

    # Single file
    trajectories = parse_outcar("path/to/OUTCAR")

    # Build labeled dataset
    dataset = OUTCARDataset.from_directory(
        "outcar_collection/",
        labels_file="labels.csv",   # columns: filename, is_sloshing
    )
    results = dataset.validate_detector()
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from science.core.logging import get_logger
from science.time_series.scf_convergence import (
    ChargeSloshingDetector,
    ConvergenceRatePredictor,
    SCFTrajectory,
    analyse_scf,
)

logger = get_logger(__name__)

# OUTCAR regex patterns
_RE_NELM = re.compile(r"NELM\s*=\s*(\d+)")
_RE_EDIFF = re.compile(r"EDIFF\s*=\s*([\d.Ee+-]+)")
_RE_ENERGY = re.compile(
    r"free  energy\s+TOTEN\s*=\s*([\d.Ee+-]+)\s+eV"
)
_RE_DE = re.compile(
    r"total energy-loss\s*=\s*([\d.Ee+-]+)"
)
_RE_DAV = re.compile(
    r"DAV:\s+(\d+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)"
)
_RE_RMM = re.compile(
    r"RMM:\s+(\d+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)"
)
_RE_IONIC_STEP = re.compile(r"---+\s*Iteration\s+(\d+)\(")
_RE_ALGO = re.compile(r"IALGO\s*=\s*(\d+)")
_RE_ISPIN = re.compile(r"ISPIN\s*=\s*(\d+)")
_RE_ISMEAR = re.compile(r"ISMEAR\s*=\s*([-\d]+)")


@dataclass
class OUTCARMetadata:
    """Metadata extracted from OUTCAR header."""
    nelm: int = 60
    ediff: float = 1e-4
    ialgo: int = 38       # 38=Normal, 48=VeryFast, 68=Fast
    ispin: int = 1
    ismear: int = 1
    n_atoms: int = 0
    elements: List[str] = field(default_factory=list)
    is_metal: bool = False


@dataclass
class ParsedOUTCAR:
    """Parsed OUTCAR with SCF trajectories and metadata."""
    filepath: str
    metadata: OUTCARMetadata
    ionic_steps: List[SCFTrajectory]
    total_scf_steps: int
    converged: bool

    @property
    def n_ionic_steps(self) -> int:
        return len(self.ionic_steps)


def parse_outcar(filepath: str) -> ParsedOUTCAR:
    """
    Parse a VASP OUTCAR file and extract SCF trajectories.

    Parameters
    ----------
    filepath : str
        Path to OUTCAR file.

    Returns
    -------
    ParsedOUTCAR
        Parsed data with SCF trajectories per ionic step.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"OUTCAR not found: {filepath}")

    text = path.read_text(errors="replace")
    lines = text.splitlines()

    # Extract metadata from header
    meta = OUTCARMetadata()
    for line in lines[:200]:
        m = _RE_NELM.search(line)
        if m:
            meta.nelm = int(m.group(1))
        m = _RE_EDIFF.search(line)
        if m:
            meta.ediff = float(m.group(1))
        m = _RE_ALGO.search(line)
        if m:
            meta.ialgo = int(m.group(1))
        m = _RE_ISPIN.search(line)
        if m:
            meta.ispin = int(m.group(1))
        m = _RE_ISMEAR.search(line)
        if m:
            meta.ismear = int(m.group(1))
            meta.is_metal = int(m.group(1)) >= 0  # ISMEAR>=0 suggests metal

    # Extract SCF residuals per ionic step
    ionic_steps: List[SCFTrajectory] = []
    current_dE: List[float] = []
    current_rms: List[float] = []

    for line in lines:
        # New ionic step marker
        if _RE_IONIC_STEP.search(line):
            if current_dE:
                ionic_steps.append(SCFTrajectory(
                    dE=current_dE, rms_dV=current_rms or None,
                    nelm=meta.nelm, ediff=meta.ediff,
                ))
            current_dE = []
            current_rms = []

        # DAV or RMM SCF line
        for pattern in [_RE_DAV, _RE_RMM]:
            m = pattern.search(line)
            if m:
                try:
                    dE_val = abs(float(m.group(3)))
                    rms_val = abs(float(m.group(4)))
                    current_dE.append(dE_val)
                    current_rms.append(rms_val)
                except (ValueError, IndexError):
                    pass
                break

    # Final ionic step
    if current_dE:
        ionic_steps.append(SCFTrajectory(
            dE=current_dE, rms_dV=current_rms or None,
            nelm=meta.nelm, ediff=meta.ediff,
        ))

    total_scf = sum(len(t.dE) for t in ionic_steps)
    converged = ionic_steps[-1].is_converged() if ionic_steps else False

    logger.info(f"Parsed OUTCAR",
                extra={"file": str(path.name), "ionic_steps": len(ionic_steps),
                       "total_scf": total_scf, "converged": converged})

    return ParsedOUTCAR(
        filepath=filepath,
        metadata=meta,
        ionic_steps=ionic_steps,
        total_scf_steps=total_scf,
        converged=converged,
    )


@dataclass
class DetectorValidationResult:
    """Result of validating sloshing detector on labeled OUTCAR dataset."""
    n_files: int
    n_labeled_sloshing: int
    n_labeled_healthy: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    cohen_kappa: float
    per_file_results: List[Dict]


class OUTCARDataset:
    """
    Labeled dataset of VASP OUTCARs for sloshing detector validation.

    Directory structure:
        outcar_dir/
        ├── labels.csv          # filename,is_sloshing (1/0)
        ├── Cu_111_CO2RR.OUTCAR
        ├── Pt_111_HER.OUTCAR
        └── ...

    Usage
    -----
    >>> dataset = OUTCARDataset.from_directory("outcars/", "labels.csv")
    >>> result = dataset.validate_detector()
    >>> print(f"F1: {result.f1:.3f}, Accuracy: {result.accuracy:.3f}")
    """

    def __init__(self, parsed_files: List[ParsedOUTCAR],
                 labels: Dict[str, bool]):
        self.files = parsed_files
        self.labels = labels

    @classmethod
    def from_directory(cls, directory: str, labels_file: str) -> "OUTCARDataset":
        """Load OUTCARs and labels from a directory."""
        dirpath = Path(directory)

        # Load labels
        labels = {}
        labels_path = dirpath / labels_file
        if labels_path.exists():
            with open(labels_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    labels[row["filename"]] = row["is_sloshing"].strip() == "1"

        # Parse all OUTCARs
        parsed = []
        for outcar_path in sorted(dirpath.glob("*.OUTCAR")) + sorted(dirpath.glob("OUTCAR*")):
            try:
                p = parse_outcar(str(outcar_path))
                parsed.append(p)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Skipping {outcar_path.name}: {e}")

        return cls(parsed, labels)

    def validate_detector(
        self,
        detector: Optional[ChargeSloshingDetector] = None,
    ) -> DetectorValidationResult:
        """
        Run sloshing detector on all labeled OUTCARs and compute metrics.

        Parameters
        ----------
        detector : ChargeSloshingDetector, optional
            Detector to validate. Uses default thresholds if not provided.

        Returns
        -------
        DetectorValidationResult
            Classification metrics and per-file results.
        """
        if detector is None:
            detector = ChargeSloshingDetector()

        tp = fp = tn = fn = 0
        per_file = []

        for parsed in self.files:
            fname = Path(parsed.filepath).name
            true_label = self.labels.get(fname)
            if true_label is None:
                continue

            # Check sloshing on all ionic steps
            any_sloshing = False
            for traj in parsed.ionic_steps:
                result = detector.detect(traj)
                if result.is_sloshing:
                    any_sloshing = True
                    break

            pred = any_sloshing
            if pred and true_label:
                tp += 1
            elif pred and not true_label:
                fp += 1
            elif not pred and true_label:
                fn += 1
            else:
                tn += 1

            per_file.append({
                "filename": fname,
                "true_sloshing": true_label,
                "pred_sloshing": pred,
                "correct": pred == true_label,
                "n_ionic_steps": parsed.n_ionic_steps,
                "total_scf": parsed.total_scf_steps,
            })

        n = tp + fp + tn + fn
        accuracy = (tp + tn) / max(n, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        # Cohen's kappa
        p_e = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / max(n * n, 1)
        kappa = (accuracy - p_e) / max(1 - p_e, 1e-9)

        return DetectorValidationResult(
            n_files=n,
            n_labeled_sloshing=tp + fn,
            n_labeled_healthy=tn + fp,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            cohen_kappa=kappa,
            per_file_results=per_file,
        )
