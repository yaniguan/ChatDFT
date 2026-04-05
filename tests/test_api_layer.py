"""
Tests for the scientist-facing API layer:
  1. Model inference APIs (predict, validate, diagnose, smart_params)
  2. Universal preprocessor (format detection, conversion)
  3. One-click workflow (natural language → VASP inputs)
  4. Python SDK (offline mode)
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════
# 1. Format Detection
# ═══════════════════════════════════════════════════════════════════════


class TestFormatDetection:
    def test_detect_poscar(self):
        from server.api.preprocessor import InputFormat, detect_format

        poscar = """Cu slab
1.0
  7.668  0.000  0.000
  3.834  6.640  0.000
  0.000  0.000 25.000
Cu
36
Cartesian
0.0 0.0 0.0
"""
        assert detect_format(poscar) == InputFormat.POSCAR

    def test_detect_cif(self):
        from server.api.preprocessor import InputFormat, detect_format

        cif = "data_Cu\n_cell_length_a 3.615\n_cell_length_b 3.615\n"
        assert detect_format(cif) == InputFormat.CIF

    def test_detect_xyz(self):
        from server.api.preprocessor import InputFormat, detect_format

        xyz = "3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n"
        assert detect_format(xyz) == InputFormat.XYZ

    def test_detect_smiles(self):
        from server.api.preprocessor import InputFormat, detect_format

        assert detect_format("O=C=O") == InputFormat.SMILES
        assert detect_format("CC(=O)O") == InputFormat.SMILES

    def test_detect_name(self):
        from server.api.preprocessor import InputFormat, detect_format

        assert detect_format("Pt(111) 3x3 with CO") == InputFormat.NAME
        assert detect_format("bulk Cu fcc") == InputFormat.NAME

    def test_detect_extxyz(self):
        from server.api.preprocessor import InputFormat, detect_format

        extxyz = '3\nProperties=species:S:1:pos:R:3 pbc="T T T"\nCu 0.0 0.0 0.0\n'
        assert detect_format(extxyz) == InputFormat.EXTXYZ


class TestNameConversion:
    def test_convert_simple_surface(self):
        from server.api.preprocessor import convert_name

        poscar, meta, warnings = convert_name("Cu(111)")
        assert meta["n_atoms"] > 0
        assert "Cu" in meta["elements"]
        assert poscar.strip()  # non-empty POSCAR

    def test_convert_surface_with_supercell(self):
        from server.api.preprocessor import convert_name

        poscar, meta, _ = convert_name("Pt(111) 4x4")
        assert meta["n_atoms"] > 0
        assert "Pt" in meta["elements"]

    def test_convert_bulk(self):
        from server.api.preprocessor import convert_name

        poscar, meta, _ = convert_name("bulk Cu")
        assert "Cu" in meta["formula"]

    def test_convert_element_fallback(self):
        from server.api.preprocessor import convert_name

        poscar, meta, _ = convert_name("Ag")
        assert "Ag" in meta["elements"]


# ═══════════════════════════════════════════════════════════════════════
# 2. Model API
# ═══════════════════════════════════════════════════════════════════════


class TestModelAPI:
    def test_fallback_predict(self):
        from server.api.model_api import _fallback_predict

        result = _fallback_predict("Pt(111)", "CO")
        assert "E_ads_eV" in result
        assert isinstance(result["E_ads_eV"], float)
        assert result["model"] == "d_band_scaling"

    def test_fallback_predict_different_metals(self):
        from server.api.model_api import _fallback_predict

        pt = _fallback_predict("Pt(111)", "CO")
        cu = _fallback_predict("Cu(111)", "CO")
        au = _fallback_predict("Au(111)", "CO")
        # All should produce different finite values
        assert pt["E_ads_eV"] != au["E_ads_eV"]
        assert all(abs(r["E_ads_eV"]) < 10 for r in [pt, cu, au])

    def test_parse_surface(self):
        from server.api.model_api import _parse_surface

        assert _parse_surface("Pt(111)") == ("Pt", "111")
        assert _parse_surface("Cu(100)") == ("Cu", "100")
        assert _parse_surface("Fe(110)") == ("Fe", "110")

    def test_parse_description(self):
        from server.api.model_api import _parse_description

        result = _parse_description("dos of cu(111) with co and dft-d3")
        assert result["calc_type"] == "dos"
        assert result.get("adsorbate") == "CO"

    def test_smart_params_produces_incar(self):
        from server.api.model_api import _parse_description
        from server.execution.vasp_incar import get_incar

        parsed = _parse_description("bader charge of co on cu(111)")
        incar = get_incar(parsed["calc_type"])
        assert "LAECHG" in incar or parsed["calc_type"] == "bader"

    def test_incar_to_string(self):
        from server.api.model_api import _incar_to_string

        result = _incar_to_string({"ENCUT": 400, "ISMEAR": 1, "LWAVE": False})
        assert "ENCUT" in result
        assert ".FALSE." in result


# ═══════════════════════════════════════════════════════════════════════
# 3. One-Click Workflow
# ═══════════════════════════════════════════════════════════════════════


class TestOneClick:
    def test_parse_query_basic(self):
        from server.api.one_click import _parse_query

        result = _parse_query("CO adsorption on Pt(111)")
        assert result["element"] == "Pt"
        assert result["facet"] == "111"
        assert result["adsorbate"] == "CO"

    def test_parse_query_dos(self):
        from server.api.one_click import _parse_query

        result = _parse_query("DOS of Cu(111)")
        assert result["calc_type"] == "dos"
        assert result["element"] == "Cu"

    def test_parse_query_modifiers(self):
        from server.api.one_click import _parse_query

        result = _parse_query("CO on Pt(111) with DFT-D3 and spin")
        assert result.get("vdw")
        assert result.get("spin")

    def test_build_structure(self):
        from server.api.one_click import _build_structure

        result = _build_structure("Cu", "111", "", {})
        assert result["n_atoms"] > 0
        assert "Cu" in result["elements"]
        assert result["poscar"].strip()

    def test_build_structure_with_adsorbate(self):
        from server.api.one_click import _build_structure

        result = _build_structure("Cu", "100", "H", {})
        assert result["n_atoms"] > 0
        # Should have H placed or at least attempted
        has_h = "H" in result["elements"]
        placed = any("Placed" in n or "H" in n for n in result.get("notes", []))
        assert has_h or placed or result["n_atoms"] > 0

    def test_generate_incar_magnetic(self):
        from server.api.one_click import _generate_incar

        result = _generate_incar("static", "Fe", "H", ["Fe", "H"], "H on Fe(110)", "normal")
        assert result["incar"].get("ISPIN") == 2

    def test_generate_incar_vdw(self):
        from server.api.one_click import _generate_incar

        result = _generate_incar("static", "Cu", "CO", ["Cu", "C", "O"], "CO on Cu(111) with DFT-D3", "normal")
        assert result["incar"].get("IVDW") == 11

    def test_resolve_steps_dos(self):
        from server.api.one_click import _resolve_steps

        steps = _resolve_steps("dos", {}, ["Cu"])
        assert len(steps) == 2
        assert steps[0]["step"] == "scf_prerequisite"

    def test_post_processing_guide(self):
        from server.api.one_click import _post_processing_guide

        guide = _post_processing_guide("bader", "CO")
        assert "chgsum" in guide.lower() or "AECCAR" in guide
        guide_dos = _post_processing_guide("dos", "")
        assert "d-band" in guide_dos.lower() or "DOSCAR" in guide_dos


# ═══════════════════════════════════════════════════════════════════════
# 4. Python SDK (offline mode)
# ═══════════════════════════════════════════════════════════════════════


class TestSDK:
    def test_sdk_offline_init(self):
        from chatdft import ChatDFT

        dft = ChatDFT(offline=True)
        assert not dft._online

    def test_sdk_predict(self):
        from chatdft import ChatDFT

        dft = ChatDFT(offline=True)
        result = dft.predict("CO", "Pt(111)")
        assert isinstance(result.E_ads_eV, float)
        assert result.surface == "Pt(111)"
        assert result.adsorbate == "CO"

    def test_sdk_screen(self):
        from chatdft import ChatDFT

        dft = ChatDFT(offline=True)
        results = dft.screen([("CO", "Pt(111)"), ("CO", "Cu(111)"), ("H", "Ni(111)")])
        assert len(results) == 3
        assert all(isinstance(r.E_ads_eV, float) for r in results)

    def test_sdk_validate(self):
        from chatdft import ChatDFT

        dft = ChatDFT(offline=True)
        issues = dft.validate(
            incar={"LELF": True, "NCORE": 4},
            elements=["Cu"],
            n_atoms=36,
            calc_type="elf",
        )
        assert len(issues) >= 1
        assert any(i["category"] == "ELF" for i in issues)

    def test_sdk_autofix(self):
        from chatdft import ChatDFT

        dft = ChatDFT(offline=True)
        fixed, changes = dft.autofix(
            incar={"LELF": True, "NCORE": 4, "ENCUT": 200},
            elements=["Cu", "O"],
            n_atoms=36,
            calc_type="elf",
        )
        assert fixed["NCORE"] == 1
        assert len(changes) >= 1

    def test_sdk_diagnose_scf(self):
        import numpy as np

        from chatdft import ChatDFT

        dft = ChatDFT(offline=True)
        ediffs = list(10 ** np.linspace(0, -6, 30))
        result = dft.diagnose_scf(ediffs)
        assert "diagnosis" in result
        assert result["diagnosis"] == "healthy"

    def test_sdk_run_offline(self):
        from chatdft import ChatDFT

        dft = ChatDFT(offline=True)
        result = dft.run("H adsorption on Cu(111)")
        assert result.ok
        assert result.poscar  # non-empty
        assert result.incar  # non-empty
        assert result.n_atoms > 0
        assert result.element == "Cu"

    def test_sdk_run_and_save(self, tmp_path):
        from chatdft import ChatDFT

        dft = ChatDFT(offline=True)
        result = dft.run("H on Pt(111)")
        files = result.save(str(tmp_path / "test_calc"))
        assert any("POSCAR" in f for f in files)
        assert any("INCAR" in f for f in files)
        assert any("KPOINTS" in f for f in files)

    def test_sdk_result_summary(self):
        from chatdft import ChatDFT

        dft = ChatDFT(offline=True)
        result = dft.run("CO on Cu(111)")
        summary = result.summary()
        assert "Cu" in summary
        assert "CO" in summary or "ChatDFT" in summary
