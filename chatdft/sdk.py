# chatdft/sdk.py
# -*- coding: utf-8 -*-
"""
ChatDFT Python SDK — Zero-Friction DFT for Scientists
=======================================================

Usage:
    from chatdft import ChatDFT

    dft = ChatDFT()  # connects to local server, or runs offline

    # One-line DFT setup
    result = dft.run("CO adsorption on Pt(111) with DFT-D3")
    result.save("./my_calc/")     # writes POSCAR, INCAR, KPOINTS, run.slurm

    # Quick energy prediction (no VASP needed)
    e_ads = dft.predict("CO", "Pt(111)")   # → -1.72 eV (d-band model)

    # Batch screening
    results = dft.screen([
        ("CO", "Pt(111)"),
        ("CO", "Cu(111)"),
        ("CO", "Au(111)"),
    ])

    # Validate before submitting
    issues = dft.validate(incar={"ENCUT": 200}, elements=["Cu", "O"], n_atoms=36)

    # Diagnose a failed SCF
    diagnosis = dft.diagnose_scf([0.1, 0.05, 0.08, 0.03, 0.06, 0.04, ...])

    # Preprocess any format
    poscar = dft.preprocess("Pt(111) 3x3 with CO")  # from name
    poscar = dft.preprocess("O=C=O")                 # from SMILES
    poscar = dft.preprocess(open("my.cif").read())    # from CIF
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class DFTResult:
    """Result from a ChatDFT workflow run."""
    ok: bool = True
    query: str = ""
    # Structure
    poscar: str = ""
    formula: str = ""
    n_atoms: int = 0
    element: str = ""
    facet: str = ""
    adsorbate: str = ""
    # VASP inputs
    incar: str = ""
    incar_dict: Dict[str, Any] = field(default_factory=dict)
    kpoints: str = ""
    calc_type: str = ""
    # Workflow
    workflow_steps: List[Dict[str, Any]] = field(default_factory=list)
    # Validation
    validation: Dict[str, Any] = field(default_factory=dict)
    # HPC
    hpc_script: str = ""
    # Guidance
    post_processing: str = ""
    notes: List[str] = field(default_factory=list)
    error: str = ""

    def save(self, directory: str, overwrite: bool = False) -> List[str]:
        """
        Save all VASP input files to a directory.

        Creates:
          - POSCAR
          - INCAR
          - KPOINTS
          - run.slurm (if HPC script available)
          - notes.txt

        Returns list of written file paths.
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        written = []

        def _write(name: str, content: str):
            p = d / name
            if p.exists() and not overwrite:
                raise FileExistsError(f"{p} exists. Use overwrite=True to replace.")
            p.write_text(content)
            written.append(str(p))

        if self.poscar:
            _write("POSCAR", self.poscar)
        if self.incar:
            _write("INCAR", self.incar)
        if self.kpoints:
            kp_content = f"Automatic\n0\nGamma\n{self.kpoints}\n0 0 0\n"
            _write("KPOINTS", kp_content)
        if self.hpc_script:
            _write("run.slurm", self.hpc_script)
        if self.notes:
            _write("notes.txt", "\n".join(self.notes))
        if self.workflow_steps:
            _write("workflow.json", json.dumps(self.workflow_steps, indent=2))

        return written

    def summary(self) -> str:
        """Print a human-readable summary."""
        lines = [
            f"ChatDFT Result: {self.query}",
            f"  Surface:    {self.element}({self.facet})",
            f"  Adsorbate:  {self.adsorbate or 'none'}",
            f"  Calc type:  {self.calc_type}",
            f"  Formula:    {self.formula}",
            f"  Atoms:      {self.n_atoms}",
            f"  Workflow:   {len(self.workflow_steps)} step(s)",
            f"  Validation: {'PASS' if self.validation.get('all_clear') else 'ISSUES FOUND'}",
        ]
        if self.notes:
            lines.append(f"  Notes:")
            for n in self.notes:
                lines.append(f"    - {n}")
        return "\n".join(lines)


@dataclass
class PredictionResult:
    """Result from energy prediction."""
    E_ads_eV: float
    surface: str
    adsorbate: str
    model: str
    confidence: str
    note: str = ""


class ChatDFT:
    """
    Python SDK for ChatDFT.

    Modes:
    - **Server mode** (default): connects to a running ChatDFT server
    - **Offline mode**: runs science algorithms locally without a server

    Usage::

        dft = ChatDFT()                        # auto-detect
        dft = ChatDFT(server="http://localhost:8000")  # explicit server
        dft = ChatDFT(offline=True)            # force offline
    """

    def __init__(
        self,
        server: Optional[str] = None,
        offline: bool = False,
        timeout: int = 120,
    ):
        self._server = server or os.getenv("CHATDFT_BACKEND", "http://localhost:8000")
        self._offline = offline
        self._timeout = timeout

        if not offline:
            try:
                import requests
                r = requests.get(f"{self._server}/health", timeout=3)
                self._online = r.status_code == 200
            except Exception:
                self._online = False
        else:
            self._online = False

    def run(
        self,
        query: str,
        accuracy: str = "normal",
        cluster: str = "hoffman2",
        n_cores: int = 24,
    ) -> DFTResult:
        """
        One-line DFT workflow from natural language.

        Example::

            result = dft.run("CO adsorption on Pt(111) with DFT-D3")
            result.save("./co_pt111/")
        """
        if self._online:
            return self._run_server(query, accuracy, cluster, n_cores)
        return self._run_offline(query, accuracy)

    def predict(
        self,
        adsorbate: str,
        surface: str,
        model: str = "d_band_scaling",
    ) -> PredictionResult:
        """
        Quick energy prediction (no VASP needed).

        Example::

            e = dft.predict("CO", "Pt(111)")
            print(f"E_ads = {e.E_ads_eV:.2f} eV")
        """
        if self._online:
            data = self._post("/api/predict_energy", {
                "surface": surface, "adsorbate": adsorbate, "model": model,
            })
            return PredictionResult(
                E_ads_eV=data.get("E_ads_eV", 0),
                surface=surface, adsorbate=adsorbate,
                model=data.get("model", model),
                confidence=data.get("confidence", ""),
                note=data.get("note", ""),
            )
        # Offline fallback
        from server.api.model_api import _fallback_predict
        data = _fallback_predict(surface, adsorbate)
        return PredictionResult(
            E_ads_eV=data["E_ads_eV"],
            surface=surface, adsorbate=adsorbate,
            model=data["model"],
            confidence=data["confidence"],
            note=data.get("note", ""),
        )

    def screen(
        self,
        systems: List[Tuple[str, str]],
        model: str = "d_band_scaling",
    ) -> List[PredictionResult]:
        """
        Batch screening — predict energies for many (adsorbate, surface) pairs.

        Example::

            results = dft.screen([("CO", "Pt(111)"), ("CO", "Cu(111)"), ("CO", "Au(111)")])
            for r in results:
                print(f"{r.surface}: {r.E_ads_eV:.2f} eV")
        """
        return [self.predict(ads, surf, model) for ads, surf in systems]

    def validate(
        self,
        incar: Dict[str, Any],
        elements: List[str],
        n_atoms: int,
        calc_type: str = "static",
        kpoints: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate VASP inputs before submission.

        Returns list of issues found (empty = all clear).
        """
        if self._online:
            data = self._post("/api/validate", {
                "incar": incar, "elements": elements,
                "n_atoms": n_atoms, "calc_type": calc_type, "kpoints": kpoints,
            })
            return data.get("issues", [])

        from science.vasp.auto_remediation import validate_consistency
        issues = validate_consistency(incar, elements, n_atoms, calc_type, kpoints=kpoints)
        return [
            {"severity": i.severity, "category": i.category, "message": i.message,
             "fix": i.fix, "auto_fixable": i.auto_fixable}
            for i in issues
        ]

    def autofix(
        self,
        incar: Dict[str, Any],
        elements: List[str],
        n_atoms: int,
        calc_type: str = "static",
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate AND auto-fix VASP inputs.

        Returns (fixed_incar, list_of_changes).
        """
        from science.vasp.auto_remediation import validate_consistency, auto_fix
        issues = validate_consistency(incar, elements, n_atoms, calc_type)
        return auto_fix(incar, issues)

    def diagnose_scf(
        self,
        energy_diffs: List[float],
        target_ediff: float = 1e-5,
        current_incar: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Diagnose SCF convergence from OSZICAR energy differences.

        Returns diagnosis + recommended INCAR fix.
        """
        from science.vasp.auto_remediation import analyze_scf_trajectory
        result = analyze_scf_trajectory(energy_diffs, target_ediff, current_incar)
        return {
            "diagnosis": result.diagnosis.value,
            "explanation": result.explanation,
            "fix": result.recommended_fix,
            "convergence_rate": result.convergence_rate,
            "sloshing_ratio": result.sloshing_power_ratio,
        }

    def preprocess(self, content: str, format_hint: Optional[str] = None) -> str:
        """
        Convert any format to POSCAR.

        Accepts: POSCAR, CIF, XYZ, SMILES, material name.
        Returns: normalized POSCAR string.
        """
        if self._online:
            data = self._post("/api/preprocess", {
                "content": content, "format_hint": format_hint,
            })
            if data.get("ok"):
                return data["poscar"]
            raise ValueError(data.get("error", "Preprocessing failed"))

        from server.api.preprocessor import detect_format, InputFormat
        fmt = format_hint or detect_format(content)
        converters = {
            InputFormat.POSCAR: "convert_poscar",
            InputFormat.CIF: "convert_cif",
            InputFormat.XYZ: "convert_xyz",
            InputFormat.EXTXYZ: "convert_extxyz",
            InputFormat.SMILES: "convert_smiles",
            InputFormat.NAME: "convert_name",
        }
        from server.api import preprocessor
        converter = getattr(preprocessor, converters[fmt])
        poscar, _, _ = converter(content)
        return poscar

    def smart_params(self, description: str, accuracy: str = "normal") -> Dict[str, Any]:
        """
        Infer optimal VASP parameters from a description.

        Returns {"incar": {...}, "kpoints": "...", "notes": [...]}.
        """
        if self._online:
            return self._post("/api/smart_params", {
                "description": description, "accuracy": accuracy,
            })
        from server.api.model_api import _parse_description, _incar_to_string
        from server.execution.vasp_incar import get_incar, suggested_kpoints
        parsed = _parse_description(description)
        incar = get_incar(parsed["calc_type"])
        kpoints = suggested_kpoints(parsed["calc_type"])
        return {"incar": incar, "kpoints": kpoints, "calc_type": parsed["calc_type"]}

    # ── Internal helpers ─────────────────────────────────────────────

    def _post(self, endpoint: str, body: Dict) -> Dict:
        import requests
        r = requests.post(f"{self._server}{endpoint}", json=body, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def _run_server(self, query, accuracy, cluster, n_cores) -> DFTResult:
        data = self._post("/api/one_click", {
            "query": query, "accuracy": accuracy,
            "cluster": cluster, "n_cores": n_cores,
        })
        return DFTResult(
            ok=data.get("ok", False),
            query=query,
            poscar=data.get("poscar", ""),
            formula=data.get("formula", ""),
            n_atoms=data.get("n_atoms", 0),
            element=data.get("element", ""),
            facet=data.get("facet", ""),
            adsorbate=data.get("adsorbate", ""),
            incar=data.get("incar_string", ""),
            incar_dict=data.get("incar_dict", {}),
            kpoints=data.get("kpoints", ""),
            calc_type=data.get("calc_type", ""),
            workflow_steps=data.get("workflow_steps", []),
            validation=data.get("validation", {}),
            hpc_script=data.get("hpc_script", ""),
            post_processing=data.get("post_processing", ""),
            notes=data.get("notes", []),
            error=data.get("error", ""),
        )

    def _run_offline(self, query, accuracy) -> DFTResult:
        """Offline mode: run the pipeline locally."""
        from server.api.one_click import _parse_query, _build_structure, _generate_incar
        from server.api.one_click import _resolve_steps, _format_incar, _generate_hpc_script
        from server.api.one_click import _post_processing_guide
        from science.vasp.auto_remediation import validate_consistency, auto_fix

        parsed = _parse_query(query)
        element = parsed["element"]
        facet = parsed["facet"]
        adsorbate = parsed.get("adsorbate", "")
        calc_type = parsed["calc_type"]

        struct = _build_structure(element, facet, adsorbate, parsed)
        incar_result = _generate_incar(calc_type, element, adsorbate, struct["elements"], query, accuracy)
        incar_dict = incar_result["incar"]

        issues = validate_consistency(incar_dict, struct["elements"], struct["n_atoms"], calc_type)
        fixed, changes = auto_fix(incar_dict, issues)
        if changes:
            incar_dict = fixed

        notes = struct.get("notes", []) + incar_result.get("notes", [])
        notes.extend([f"Auto-fixed: {c}" for c in changes])

        remaining = [i for i in issues if not i.auto_fixable]

        return DFTResult(
            ok=True,
            query=query,
            poscar=struct["poscar"],
            formula=struct["formula"],
            n_atoms=struct["n_atoms"],
            element=element,
            facet=facet,
            adsorbate=adsorbate,
            incar=_format_incar(incar_dict),
            incar_dict=incar_dict,
            kpoints=incar_result["kpoints"],
            calc_type=calc_type,
            workflow_steps=_resolve_steps(calc_type, incar_dict, struct["elements"]),
            validation={"all_clear": len(remaining) == 0, "n_issues": len(issues)},
            hpc_script=_generate_hpc_script(f"chatdft-{element}{facet}", 24, "24:00:00"),
            post_processing=_post_processing_guide(calc_type, adsorbate),
            notes=notes,
        )
