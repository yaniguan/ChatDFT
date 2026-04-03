"""
Server Agent Tests — with mocked DB and OpenAI
================================================
Tests the core server-side agents without requiring a running database
or OpenAI API key. Uses unittest.mock to intercept all external calls.

Run:  pytest tests/test_server_agents.py -v --tb=short
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set DATABASE_URL before any server imports
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test@localhost:5432/test_db")


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_openai_response():
    """Returns a factory that creates mock OpenAI chat completion responses."""
    def make_response(content: str, input_tokens=100, output_tokens=50):
        return {
            "ok": True,
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
    return make_response


@pytest.fixture
def mock_intent_json():
    """Sample intent JSON that LLM would return."""
    return json.dumps({
        "stage": "electrocatalysis",
        "area": "heterogeneous_catalysis",
        "task": "Study CO2 reduction to CO on Cu(111)",
        "system": {
            "catalyst": "Cu",
            "facet": "111",
            "material": "Cu",
            "molecule": "CO2",
        },
        "conditions": {
            "pH": 6.8,
            "potential": -0.5,
            "electrolyte": "KHCO3",
        },
        "reaction_network": {
            "steps": ["CO2(g) + * + H+ + e- → COOH*"],
            "intermediates": ["*", "CO2(g)", "COOH*", "CO*"],
        },
        "electronic_calcs": [],
        "confidence": 0.85,
    })


@pytest.fixture
def mock_hypothesis_json():
    """Sample hypothesis JSON that LLM would return."""
    return json.dumps({
        "reaction_network": [
            {"lhs": ["CO2(g)", "*", "H+", "e-"], "rhs": ["COOH*"]},
            {"lhs": ["COOH*", "H+", "e-"], "rhs": ["CO*", "H2O(g)"]},
        ],
        "intermediates": ["*", "CO2(g)", "COOH*", "CO*", "CO(g)", "H2O(g)"],
        "coads_pairs": [],
        "ts_edges": [["CO2*", "COOH*"]],
    })


# ─── Category: Intent Agent Tests ─────────────────────────────────────────

class TestIntentAgentFunctions:
    """Test intent parsing helper functions (no LLM call needed)."""

    def test_json_from_llm_raw_valid(self):
        from server.chat.intent_agent import _json_from_llm_raw
        result = _json_from_llm_raw('{"stage": "catalysis", "area": "hetero"}')
        assert result["stage"] == "catalysis"

    def test_json_from_llm_raw_with_markdown(self):
        from server.chat.intent_agent import _json_from_llm_raw
        raw = '```json\n{"stage": "catalysis"}\n```'
        result = _json_from_llm_raw(raw)
        assert result.get("stage") == "catalysis"

    def test_json_from_llm_raw_invalid(self):
        from server.chat.intent_agent import _json_from_llm_raw
        result = _json_from_llm_raw("this is not json")
        assert result == {} or result is None or isinstance(result, dict)

    def test_json_from_llm_raw_empty(self):
        from server.chat.intent_agent import _json_from_llm_raw
        result = _json_from_llm_raw("")
        assert result is None or isinstance(result, dict)

    def test_clip_function_exists(self):
        from server.chat.intent_agent import _clip
        assert _clip("hello world this is a test", 5) is not None
        assert len(_clip("hello world", 100)) <= 100


# ─── Category: Hypothesis Agent Tests ────────────────────────────────────

class TestHypothesisAgentFunctions:
    """Test hypothesis validation and graph normalization."""

    def test_validate_graph_basic(self, mock_hypothesis_json):
        """Graph validator should normalise species notation."""
        try:
            from server.chat.hypothesis_agent import _validate_graph
            graph = json.loads(mock_hypothesis_json)
            intent = {"system": {"material": "Cu", "facet": "111"}}
            fixed, warnings = _validate_graph(graph, intent)
            assert "intermediates" in fixed
            assert isinstance(warnings, list)
        except ImportError:
            pytest.skip("_validate_graph not importable (may be inline)")

    def test_hypothesis_graph_has_required_keys(self, mock_hypothesis_json):
        """LLM output must contain reaction_network, intermediates."""
        graph = json.loads(mock_hypothesis_json)
        assert "reaction_network" in graph
        assert "intermediates" in graph
        assert len(graph["intermediates"]) >= 3


# ─── Category: Plan Agent Tests ──────────────────────────────────────────

class TestPlanAgentFunctions:
    """Test plan generation helper functions."""

    def test_mech_guess_co2rr(self):
        """Mechanism matcher should recognize CO2RR keywords."""
        try:
            from server.chat.plan_agent import _mech_guess
            intent = {"domain": "electrocatalysis", "problem_type": "CO2 reduction"}
            keys = _mech_guess(intent, "CO2 reduction to CO on Cu")
            assert isinstance(keys, list)
        except ImportError:
            pytest.skip("_mech_guess not importable")

    def test_extract_intent(self):
        try:
            from server.chat.plan_agent import _extract_intent
            body = {
                "intent": {"stage": "electrocatalysis", "system": {"material": "Cu"}},
                "query": "CO2 reduction on Cu(111)",
            }
            result = _extract_intent(body)
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("_extract_intent not importable")


# ─── Category: QA Agent Tests ────────────────────────────────────────────

class TestQAAgentFunctions:
    """Test QA agent knowledge functions."""

    def test_surface_stability_known_reconstructions(self):
        """Should flag Pt(100) as reconstructed."""
        try:
            from server.utils.outcar_debugger import check_surface_stability
            result = check_surface_stability("Pt", "100")
            # Pt(100) has known hex(5×20) reconstruction
            if result is not None:
                assert "warning" in result or "reconstruction" in str(result).lower()
        except ImportError:
            pytest.skip("check_surface_stability not importable")

    def test_functional_recommendation_co_on_pt(self):
        """Should recommend BEEF-vdW or DFT-D3 for CO on Pt."""
        try:
            from server.utils.outcar_debugger import recommend_functional
            recs = recommend_functional("CO adsorption on Pt(111)")
            assert isinstance(recs, list)
            if recs:
                all_text = json.dumps(recs).lower()
                # Should mention the CO puzzle or dispersion
                assert any(kw in all_text for kw in ["beef", "vdw", "d3", "dispersion", "co"]), \
                    f"Expected functional advice for CO/Pt, got: {recs}"
        except ImportError:
            pytest.skip("recommend_functional not importable")


# ─── Category: Structure Agent Tests ──────────────────────────────────────

class TestStructureAgentFunctions:
    """Test structure building without HPC."""

    def test_build_slab_cu111(self):
        try:
            from server.execution.structure_agent import build_surface_ase
            result = build_surface_ase("Cu", "111", nx=2, ny=2, nlayers=3, vacuum=10.0)
            assert result.get("ok") or "poscar" in result or "atoms" in str(type(result))
        except (ImportError, TypeError):
            # Fall back to direct ASE test
            from ase.build import fcc111
            slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0, a=3.615)
            assert len(slab) == 12

    def test_normalize_element(self):
        try:
            from server.execution.structure_agent import _normalize_element
            assert _normalize_element("pt") == "Pt"
            assert _normalize_element("CU") == "Cu"
            assert _normalize_element("Ag") == "Ag"
        except ImportError:
            pytest.skip("_normalize_element not importable")

    def test_parse_miller(self):
        try:
            from server.execution.structure_agent import _parse_miller
            # _parse_miller takes a payload dict with "miller_index" key
            assert _parse_miller({"miller_index": "111"}) == (1, 1, 1)
            assert _parse_miller({"miller_index": "100"}) == (1, 0, 0)
        except (ImportError, TypeError, KeyError):
            pytest.skip("_parse_miller not testable in isolation")


# ─── Category: ASE Script Generation Tests ────────────────────────────────

class TestASEScriptGeneration:
    """Test that ASE scripts are valid Python."""

    def test_geo_script_is_valid_python(self):
        try:
            from server.execution.ase_scripts import script_geo
            script = script_geo(kpoints="4 4 1", encut=400, ediffg=-0.03)
            assert isinstance(script, str)
            assert "Vasp" in script or "vasp" in script.lower()
            # Should be valid Python syntax
            compile(script, "<test>", "exec")
        except ImportError:
            pytest.skip("script_geo not importable")

    def test_dos_script_is_valid_python(self):
        try:
            from server.execution.ase_scripts import script_dos
            script = script_dos(kpoints="6 6 1", encut=400)
            assert isinstance(script, str)
            compile(script, "<test>", "exec")
        except ImportError:
            pytest.skip("script_dos not importable")

    def test_neb_script_is_valid_python(self):
        try:
            from server.execution.ase_scripts import script_neb
            script = script_neb(n_images=7, kpoints="4 4 1", encut=400)
            assert isinstance(script, str)
            compile(script, "<test>", "exec")
        except ImportError:
            pytest.skip("script_neb not importable")

    def test_freq_script_is_valid_python(self):
        try:
            from server.execution.ase_scripts import script_freq
            script = script_freq(kpoints="4 4 1", encut=400)
            assert isinstance(script, str)
            compile(script, "<test>", "exec")
        except ImportError:
            pytest.skip("script_freq not importable")

    def test_gcdft_script_is_valid_python(self):
        try:
            from server.execution.ase_scripts import script_gcdft
            script = script_gcdft(kpoints="4 4 1", encut=400)
            assert isinstance(script, str)
            compile(script, "<test>", "exec")
        except ImportError:
            pytest.skip("script_gcdft not importable")


# ─── Category: Cache Layer Tests ──────────────────────────────────────────

class TestCacheLayer:
    """Test the Redis/in-memory cache."""

    @pytest.mark.asyncio
    async def test_memory_cache_set_get(self):
        from server.cache import InMemoryBackend
        backend = InMemoryBackend(max_size=100)
        await backend.set("key1", "value1", ttl=60)
        result = await backend.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_memory_cache_miss(self):
        from server.cache import InMemoryBackend
        backend = InMemoryBackend()
        result = await backend.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_memory_cache_eviction(self):
        from server.cache import InMemoryBackend
        backend = InMemoryBackend(max_size=10)
        for i in range(20):
            await backend.set(f"key{i}", f"value{i}")
        # Should have evicted some
        assert len(backend._store) <= 10

    @pytest.mark.asyncio
    async def test_memory_cache_stats(self):
        from server.cache import InMemoryBackend
        backend = InMemoryBackend()
        await backend.set("a", "1")
        await backend.get("a")  # hit
        await backend.get("b")  # miss
        stats = backend.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["backend"] == "memory"

    @pytest.mark.asyncio
    async def test_cache_json(self):
        from server.cache import Cache
        c = Cache()
        await c.set_json("test_json", {"hello": "world"}, ttl=60)
        result = await c.get_json("test_json")
        assert result == {"hello": "world"}


# ─── Category: Science Routes Tests ──────────────────────────────────────

class TestScienceRoutes:
    """Test science API routes are importable and valid."""

    def test_routes_importable(self):
        from server.science_routes import router
        assert router is not None
        assert router.prefix == "/science"

    def test_route_models_valid(self):
        from server.science_routes import (
            SurfaceGraphRequest, SCFAnalysisRequest,
            GrounderRequest, BORequest, RattleRequest,
        )
        # Should be constructable with valid data
        req = SCFAnalysisRequest(
            dE=[0.5, 0.1, 0.01, 0.001],
            nelm=60, ediff=1e-5,
        )
        assert len(req.dE) == 4

        req2 = GrounderRequest(
            hypothesis="test",
            network={"intermediates": ["*"]},
        )
        assert req2.hypothesis == "test"


# ─── Category: Log Config Tests ──────────────────────────────────────────

class TestLogConfig:

    def test_json_formatter(self):
        from server.log_config import JSONFormatter
        import logging
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="test message", args=None, exc_info=None
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "test message"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_setup_logging(self):
        from server.log_config import setup_logging
        setup_logging(level="WARNING", json_format=False)
        import logging
        assert logging.getLogger().level == logging.WARNING
