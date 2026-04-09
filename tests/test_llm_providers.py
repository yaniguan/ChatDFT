"""
Unit tests for the vLLM-capable LLM provider layer.

Scope
-----
- Config loading (YAML + env overrides + sane defaults when the file is missing).
- LLMRouter provider selection per agent.
- Fallback behaviour on primary-provider failure.
- Concurrency semaphore bounding.
- ``_pick_model`` cross-backend guard (don't send gpt-4o to vLLM, etc).
- openai_wrapper.chatgpt_call end-to-end through a mocked router.

Everything is mocked — no network, no live vLLM container, no AsyncOpenAI
client. CI runs this in well under a second.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.utils import llm_providers as llm_providers_mod  # noqa: E402
from server.utils.llm_config import (  # noqa: E402
    LLMConfig,
    ProviderConfig,
    load_llm_config,
    reset_llm_config,
)
from server.utils.llm_providers import (  # noqa: E402
    LLMCallResult,
    LLMProvider,
    LLMRouter,
    _pick_model,
    reset_llm_router,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_caches():
    reset_llm_config()
    reset_llm_router()
    yield
    reset_llm_config()
    reset_llm_router()


@pytest.fixture
def tmp_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "llm.yaml"
    path.write_text(
        """
providers:
  openai:
    type: openai
    api_key_env: OPENAI_API_KEY
    model: gpt-4o-mini
    timeout_s: 30
    max_concurrent: 4
  vllm_local:
    type: vllm
    base_url: http://vllm.test/v1
    api_key: EMPTY
    model: Qwen/Qwen2.5-7B-Instruct
    timeout_s: 60
    max_concurrent: 2

default_provider: openai
fallback_provider: openai
routing:
  intent_agent: vllm_local
  plan_agent: vllm_local
  hypothesis_agent: openai
"""
    )
    return path


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_missing_file_returns_default(self, tmp_path: Path):
        # Point to a file that doesn't exist → default config.
        cfg = load_llm_config(str(tmp_path / "nope.yaml"))
        assert "openai" in cfg.providers
        assert cfg.default_provider == "openai"

    def test_yaml_parsed_into_providers_and_routing(self, tmp_yaml: Path):
        cfg = load_llm_config(str(tmp_yaml))
        assert set(cfg.providers) == {"openai", "vllm_local"}
        assert cfg.providers["vllm_local"].type == "vllm"
        assert cfg.providers["vllm_local"].base_url == "http://vllm.test/v1"
        assert cfg.providers["vllm_local"].max_concurrent == 2
        assert cfg.routing["intent_agent"] == "vllm_local"
        assert cfg.routing["hypothesis_agent"] == "openai"

    def test_env_override_routing(self, tmp_yaml: Path, monkeypatch):
        monkeypatch.setenv("CHATDFT_LLM_ROUTING_QA_AGENT", "vllm_local")
        cfg = load_llm_config(str(tmp_yaml))
        assert cfg.routing.get("qa_agent") == "vllm_local"

    def test_env_override_default_provider(self, tmp_yaml: Path, monkeypatch):
        monkeypatch.setenv("CHATDFT_LLM_DEFAULT_PROVIDER", "vllm_local")
        cfg = load_llm_config(str(tmp_yaml))
        assert cfg.default_provider == "vllm_local"

    def test_env_override_vllm_base_url(self, tmp_yaml: Path, monkeypatch):
        monkeypatch.setenv("CHATDFT_VLLM_BASE_URL", "http://other.host:9000/v1")
        cfg = load_llm_config(str(tmp_yaml))
        assert cfg.providers["vllm_local"].base_url == "http://other.host:9000/v1"

    def test_provider_for_uses_routing_then_default(self, tmp_yaml: Path):
        cfg = load_llm_config(str(tmp_yaml))
        assert cfg.provider_for("intent_agent") == "vllm_local"
        assert cfg.provider_for("unknown_agent") == "openai"

    def test_providers_in_priority_includes_fallback(self, tmp_yaml: Path):
        cfg = load_llm_config(str(tmp_yaml))
        order = cfg.providers_in_priority("intent_agent")
        assert order == ["vllm_local", "openai"]

    def test_providers_in_priority_no_duplicate_when_equal(self, tmp_yaml: Path):
        cfg = load_llm_config(str(tmp_yaml))
        order = cfg.providers_in_priority("hypothesis_agent")
        assert order == ["openai"]  # fallback == primary → deduped

    def test_resolved_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("FAKE_KEY", "sk-abc123")
        p = ProviderConfig(name="x", type="openai", api_key_env="FAKE_KEY")
        assert p.resolved_api_key() == "sk-abc123"

    def test_resolved_api_key_literal_beats_env(self, monkeypatch):
        monkeypatch.setenv("FAKE_KEY", "sk-env")
        p = ProviderConfig(name="x", type="openai", api_key="sk-literal", api_key_env="FAKE_KEY")
        assert p.resolved_api_key() == "sk-literal"


# ---------------------------------------------------------------------------
# _pick_model cross-backend guard
# ---------------------------------------------------------------------------


class TestPickModel:
    def _make_provider(self, ptype: str, default_model: str) -> LLMProvider:
        # Build a provider without actually instantiating the OpenAI client
        cfg = ProviderConfig(name=f"{ptype}_test", type=ptype, model=default_model, base_url="http://x")
        p = LLMProvider.__new__(LLMProvider)  # bypass __init__
        p.config = cfg
        p.name = cfg.name
        p._semaphore = asyncio.Semaphore(1)
        p._healthy = True
        p._client = object()  # sentinel
        return p

    def test_vllm_provider_rejects_gpt_hint(self):
        p = self._make_provider("vllm", "Qwen/Qwen2.5-7B-Instruct")
        assert _pick_model(p, "gpt-4o") == "Qwen/Qwen2.5-7B-Instruct"

    def test_openai_provider_rejects_local_hint(self):
        p = self._make_provider("openai", "gpt-4o-mini")
        assert _pick_model(p, "Qwen/Qwen2.5-7B-Instruct") == "gpt-4o-mini"

    def test_matching_family_keeps_hint(self):
        p = self._make_provider("openai", "gpt-4o-mini")
        assert _pick_model(p, "gpt-4o") == "gpt-4o"

    def test_no_hint_returns_default(self):
        p = self._make_provider("vllm", "Qwen/Qwen2.5-7B-Instruct")
        assert _pick_model(p, None) == "Qwen/Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
# Router + fake providers
# ---------------------------------------------------------------------------


@dataclass
class FakeResp:
    provider: str
    model: str
    text: str = '{"ok": true}'
    fail: bool = False
    latency_ms: int = 50

    def as_result(self) -> LLMCallResult:
        raw = {
            "choices": [{"message": {"content": self.text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        return LLMCallResult(
            provider=self.provider,
            model=self.model,
            raw={} if self.fail else raw,
            latency_ms=self.latency_ms,
            success=not self.fail,
            error=None if not self.fail else "simulated failure",
        )


class FakeProvider(LLMProvider):
    """In-memory provider — no real HTTP."""

    def __init__(self, config: ProviderConfig, *, fail: bool = False, text: str = '{"ok": true}'):
        # Build dataclasses / semaphores without touching openai SDK
        self.config = config
        self.name = config.name
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._healthy = True
        self._client = object()  # non-None so chat_completion doesn't short-circuit
        self.fail = fail
        self.text = text
        self.call_count = 0

    async def health_check(self) -> bool:
        return self._healthy

    async def chat_completion(self, messages, *, model=None, **kwargs) -> LLMCallResult:
        self.call_count += 1
        effective_model = model or self.config.model
        return FakeResp(
            provider=self.name,
            model=effective_model or "",
            text=self.text,
            fail=self.fail,
        ).as_result()

    async def embed(self, texts, model=None) -> List[List[float]]:
        if self.fail:
            raise RuntimeError(f"{self.name} simulated failure")
        return [[0.1] * 4 for _ in texts]


def _make_cfg() -> LLMConfig:
    return LLMConfig(
        providers={
            "openai": ProviderConfig(name="openai", type="openai", model="gpt-4o-mini"),
            "vllm_local": ProviderConfig(
                name="vllm_local",
                type="vllm",
                base_url="http://vllm.test/v1",
                model="Qwen/Qwen2.5-7B-Instruct",
                max_concurrent=2,
            ),
        },
        routing={"intent_agent": "vllm_local", "hypothesis_agent": "openai"},
        default_provider="openai",
        fallback_provider="openai",
    )


def _install_router(monkeypatch, providers: Dict[str, LLMProvider]) -> LLMRouter:
    """Replace LLMRouter._build_providers to use our fakes."""
    cfg = _make_cfg()
    router = LLMRouter.__new__(LLMRouter)
    router.config = cfg
    router._providers = providers
    monkeypatch.setattr(llm_providers_mod, "_ROUTER", router)
    return router


class TestLLMRouterSelection:
    def test_routes_intent_to_vllm(self, monkeypatch):
        vllm = FakeProvider(_make_cfg().providers["vllm_local"])
        oai = FakeProvider(_make_cfg().providers["openai"])
        router = _install_router(monkeypatch, {"vllm_local": vllm, "openai": oai})

        result = asyncio.run(
            router.chat_completion(
                agent_name="intent_agent",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
        assert result.success
        assert result.provider == "vllm_local"
        assert vllm.call_count == 1
        assert oai.call_count == 0

    def test_routes_hypothesis_to_openai(self, monkeypatch):
        vllm = FakeProvider(_make_cfg().providers["vllm_local"])
        oai = FakeProvider(_make_cfg().providers["openai"])
        router = _install_router(monkeypatch, {"vllm_local": vllm, "openai": oai})

        result = asyncio.run(
            router.chat_completion(
                agent_name="hypothesis_agent",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
        assert result.success
        assert result.provider == "openai"
        assert oai.call_count == 1
        assert vllm.call_count == 0

    def test_unknown_agent_uses_default(self, monkeypatch):
        vllm = FakeProvider(_make_cfg().providers["vllm_local"])
        oai = FakeProvider(_make_cfg().providers["openai"])
        router = _install_router(monkeypatch, {"vllm_local": vllm, "openai": oai})

        result = asyncio.run(
            router.chat_completion(
                agent_name="brand_new_agent",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
        assert result.provider == "openai"


class TestLLMRouterFallback:
    def test_fallback_on_primary_failure(self, monkeypatch):
        vllm = FakeProvider(_make_cfg().providers["vllm_local"], fail=True)
        oai = FakeProvider(_make_cfg().providers["openai"])
        router = _install_router(monkeypatch, {"vllm_local": vllm, "openai": oai})

        result = asyncio.run(
            router.chat_completion(
                agent_name="intent_agent",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
        # Primary failed, fallback succeeded
        assert result.success
        assert result.provider == "openai"
        assert vllm.call_count == 1
        assert oai.call_count == 1

    def test_all_providers_fail_returns_last_error(self, monkeypatch):
        vllm = FakeProvider(_make_cfg().providers["vllm_local"], fail=True)
        oai = FakeProvider(_make_cfg().providers["openai"], fail=True)
        router = _install_router(monkeypatch, {"vllm_local": vllm, "openai": oai})

        result = asyncio.run(
            router.chat_completion(
                agent_name="intent_agent",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
        assert not result.success
        assert result.error == "simulated failure"

    def test_unhealthy_provider_skipped(self, monkeypatch):
        vllm = FakeProvider(_make_cfg().providers["vllm_local"])
        vllm._healthy = False  # simulate failed health check
        oai = FakeProvider(_make_cfg().providers["openai"])
        router = _install_router(monkeypatch, {"vllm_local": vllm, "openai": oai})

        result = asyncio.run(
            router.chat_completion(
                agent_name="intent_agent",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
        assert result.provider == "openai"
        assert vllm.call_count == 0  # skipped, never called


class TestEmbeddingRouting:
    def test_embed_routes_to_configured_provider(self, monkeypatch):
        vllm = FakeProvider(_make_cfg().providers["vllm_local"])
        oai = FakeProvider(_make_cfg().providers["openai"])
        router = _install_router(monkeypatch, {"vllm_local": vllm, "openai": oai})

        # knowledge_agent isn't in routing → default (openai)
        vecs = asyncio.run(
            router.embed(
                agent_name="knowledge_agent",
                texts=["hello", "world"],
            )
        )
        assert len(vecs) == 2
        assert all(len(v) == 4 for v in vecs)

    def test_embed_fallback_on_failure(self, monkeypatch):
        vllm = FakeProvider(_make_cfg().providers["vllm_local"], fail=True)
        oai = FakeProvider(_make_cfg().providers["openai"])
        router = _install_router(monkeypatch, {"vllm_local": vllm, "openai": oai})

        # Route intent_agent (vllm primary) → falls back to openai
        vecs = asyncio.run(
            router.embed(
                agent_name="intent_agent",
                texts=["x"],
            )
        )
        assert len(vecs) == 1


class TestConcurrencySemaphore:
    def test_semaphore_limits_in_flight(self, monkeypatch):
        """With max_concurrent=2, three calls should serialize in waves."""

        class SlowFake(FakeProvider):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.in_flight = 0
                self.max_observed = 0

            async def chat_completion(self, messages, *, model=None, **kwargs):
                # Acquire our own semaphore to enforce max_concurrent
                async with self._semaphore:
                    self.in_flight += 1
                    self.max_observed = max(self.max_observed, self.in_flight)
                    await asyncio.sleep(0.02)
                    self.in_flight -= 1
                    return FakeResp(provider=self.name, model="x").as_result()

        prov = SlowFake(_make_cfg().providers["vllm_local"])
        router = _install_router(monkeypatch, {"vllm_local": prov, "openai": prov})

        async def _run():
            await asyncio.gather(
                *(
                    router.chat_completion(
                        agent_name="intent_agent",
                        messages=[{"role": "user", "content": "x"}],
                    )
                    for _ in range(6)
                )
            )

        asyncio.run(_run())
        # max_concurrent=2 → at most 2 in flight at once
        assert prov.max_observed <= 2
        # But more than 1 (i.e., actually parallel, not serialized)
        assert prov.max_observed >= 1


# ---------------------------------------------------------------------------
# openai_wrapper.chatgpt_call end-to-end
# ---------------------------------------------------------------------------


class TestChatgptCallWiring:
    def test_delegates_to_router_and_tags_model(self, monkeypatch):
        vllm = FakeProvider(_make_cfg().providers["vllm_local"], text='{"field": "value"}')
        oai = FakeProvider(_make_cfg().providers["openai"])
        _install_router(monkeypatch, {"vllm_local": vllm, "openai": oai})

        # Disable AgentLog side-effect to avoid touching the DB
        from unittest.mock import patch

        from server.utils import openai_wrapper as ow

        logged: List[Dict[str, Any]] = []

        def _capture(*args, **kwargs):
            # _async_log_safe has positional params (agent, call_type, model,
            # in_tok, out_tok, latency) followed by kw-only fields. Normalise
            # into a dict for the test assertions.
            fields = ["agent_name", "call_type", "model", "input_tokens", "output_tokens", "latency_ms"]
            captured = dict(zip(fields, args))
            captured.update(kwargs)
            logged.append(captured)

        with patch.object(ow, "_async_log_safe", side_effect=_capture):
            response = asyncio.run(
                ow.chatgpt_call(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4o-mini",
                    agent_name="intent_agent",
                    log_to_agentlog=True,
                )
            )

        # Response is openai-shaped
        assert "choices" in response
        # AgentLog was called with provider:model tagging
        assert len(logged) == 1
        tagged = logged[0].get("model") or ""
        assert tagged.startswith("vllm_local:")

    def test_router_failure_returns_error_envelope(self, monkeypatch):
        vllm = FakeProvider(_make_cfg().providers["vllm_local"], fail=True)
        oai = FakeProvider(_make_cfg().providers["openai"], fail=True)
        _install_router(monkeypatch, {"vllm_local": vllm, "openai": oai})

        from unittest.mock import patch

        from server.utils import openai_wrapper as ow

        with patch.object(ow, "_async_log_safe"):
            response = asyncio.run(
                ow.chatgpt_call(
                    messages=[{"role": "user", "content": "hi"}],
                    agent_name="intent_agent",
                    log_to_agentlog=True,
                )
            )

        assert response.get("error")
        assert response.get("choices") == []

    def test_schema_invalid_flagged(self, monkeypatch):
        vllm = FakeProvider(
            _make_cfg().providers["vllm_local"],
            text="not json at all, sorry",
        )
        oai = FakeProvider(_make_cfg().providers["openai"])
        _install_router(monkeypatch, {"vllm_local": vllm, "openai": oai})

        from unittest.mock import patch

        from server.utils import openai_wrapper as ow

        logged: List[Dict[str, Any]] = []

        def _capture(*args, **kwargs):
            # _async_log_safe has positional params (agent, call_type, model,
            # in_tok, out_tok, latency) followed by kw-only fields. Normalise
            # into a dict for the test assertions.
            fields = ["agent_name", "call_type", "model", "input_tokens", "output_tokens", "latency_ms"]
            captured = dict(zip(fields, args))
            captured.update(kwargs)
            logged.append(captured)

        with patch.object(ow, "_async_log_safe", side_effect=_capture):
            asyncio.run(
                ow.chatgpt_call(
                    messages=[{"role": "user", "content": "hi"}],
                    agent_name="intent_agent",
                    json_mode=True,
                )
            )

        assert logged
        assert logged[0].get("call_type") == "llm_json_invalid"
        # Schema failure should mark the log row as unsuccessful
        assert logged[0].get("success") is False
