# server/utils/llm_config.py
# -*- coding: utf-8 -*-
"""
LLM provider configuration loader.

Reads a YAML file (default ``server/llm.yaml``) describing the available
providers and the per-agent routing table, layered with environment-variable
overrides. Produces strongly-typed ``LLMConfig`` / ``ProviderConfig`` objects
that ``llm_providers.LLMRouter`` consumes.

Schema
------
::

    providers:
      <provider_name>:
        type: openai | vllm
        base_url: https://api.openai.com/v1      # optional for openai
        api_key_env: OPENAI_API_KEY              # env var to read the key from
        api_key: EMPTY                           # literal key (vllm local)
        model: gpt-4o-mini                       # default model for provider
        timeout_s: 60
        max_concurrent: 16
        health_check_url: http://localhost:8001/health   # vllm only (optional)

    default_provider: openai                     # used when no routing match
    fallback_provider: openai                    # used if primary raises
    routing:
      intent_agent: openai                       # agent_name → provider_name
      plan_agent: openai
      ...

Env overrides
-------------
* ``CHATDFT_LLM_CONFIG`` — path to YAML file (default ``server/llm.yaml``)
* ``CHATDFT_LLM_DEFAULT_PROVIDER`` — override ``default_provider``
* ``CHATDFT_LLM_ROUTING_<AGENT>`` — override routing for one agent, e.g.
  ``CHATDFT_LLM_ROUTING_INTENT_AGENT=vllm_local``
* ``CHATDFT_VLLM_BASE_URL`` — shortcut override for the ``vllm_local`` base_url

The loader is tolerant: a missing file yields a safe default that sends
everything through the standard OpenAI client.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    name: str
    type: str                                 # "openai" | "vllm"
    base_url: Optional[str] = None
    api_key: Optional[str] = None             # resolved literal key
    api_key_env: Optional[str] = None         # env var name (pre-resolution)
    model: Optional[str] = None               # default model for this provider
    timeout_s: float = 60.0
    max_concurrent: int = 16
    health_check_url: Optional[str] = None

    def resolved_api_key(self) -> Optional[str]:
        """Return the effective API key, reading the env var if needed."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass
class LLMConfig:
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    routing: Dict[str, str] = field(default_factory=dict)
    default_provider: str = "openai"
    fallback_provider: Optional[str] = "openai"

    def provider_for(self, agent_name: str) -> str:
        """Return the provider name that should handle ``agent_name``."""
        if agent_name in self.routing:
            return self.routing[agent_name]
        return self.default_provider

    def providers_in_priority(self, agent_name: str) -> List[str]:
        """
        Return the ordered list of providers to try for ``agent_name``:
        primary first, fallback second (if distinct).
        """
        primary = self.provider_for(agent_name)
        order = [primary]
        if self.fallback_provider and self.fallback_provider != primary:
            order.append(self.fallback_provider)
        return [p for p in order if p in self.providers]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def _default_config() -> LLMConfig:
    """Fallback when no YAML is present: OpenAI-only, sane timeouts."""
    openai = ProviderConfig(
        name="openai",
        type="openai",
        base_url=None,  # let the SDK pick its default
        api_key_env="OPENAI_API_KEY",
        model="gpt-4o",
        timeout_s=60.0,
        max_concurrent=16,
    )
    return LLMConfig(
        providers={"openai": openai},
        routing={},
        default_provider="openai",
        fallback_provider="openai",
    )


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _parse_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # local import — keeps the module importable without PyYAML
    except ImportError:
        log.warning("llm_config: PyYAML not installed, using defaults")
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception as e:  # pragma: no cover — bad YAML is a dev error
        log.warning("llm_config: failed to parse %s: %s", path, e)
        return {}


def _coerce_provider(name: str, blob: Dict[str, Any]) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        type=str(blob.get("type") or "openai").lower(),
        base_url=blob.get("base_url"),
        api_key=blob.get("api_key"),
        api_key_env=blob.get("api_key_env"),
        model=blob.get("model"),
        timeout_s=float(blob.get("timeout_s") or 60.0),
        max_concurrent=int(blob.get("max_concurrent") or 16),
        health_check_url=blob.get("health_check_url"),
    )


def _apply_env_overrides(cfg: LLMConfig) -> None:
    """Mutate ``cfg`` in place with env-var overrides."""
    if (override := os.environ.get("CHATDFT_LLM_DEFAULT_PROVIDER")):
        cfg.default_provider = override

    if (override := os.environ.get("CHATDFT_LLM_FALLBACK_PROVIDER")):
        cfg.fallback_provider = override

    # Per-agent routing via CHATDFT_LLM_ROUTING_<AGENT>=<provider>
    prefix = "CHATDFT_LLM_ROUTING_"
    for k, v in os.environ.items():
        if k.startswith(prefix) and v:
            agent = k[len(prefix):].lower()
            cfg.routing[agent] = v

    # vLLM shortcut overrides: make it trivial to re-point at a new host.
    vllm = cfg.providers.get("vllm_local")
    if vllm is not None:
        if (url := os.environ.get("CHATDFT_VLLM_BASE_URL")):
            vllm.base_url = url
        if (model := os.environ.get("CHATDFT_VLLM_MODEL")):
            vllm.model = model


def load_llm_config(path: Optional[str] = None) -> LLMConfig:
    """
    Load the LLM config. Precedence:
    1. Explicit ``path`` argument
    2. ``CHATDFT_LLM_CONFIG`` environment variable
    3. ``server/llm.yaml`` next to this file
    4. Built-in default (OpenAI only)

    Returns a fully populated ``LLMConfig``. Never raises on missing files.
    """
    resolved: Optional[Path] = None
    if path:
        resolved = Path(path)
    elif (env_path := os.environ.get("CHATDFT_LLM_CONFIG")):
        resolved = Path(env_path)
    else:
        candidate = Path(__file__).resolve().parent.parent / "llm.yaml"
        if candidate.exists():
            resolved = candidate

    if resolved is None or not resolved.is_file():
        cfg = _default_config()
        _apply_env_overrides(cfg)
        return cfg

    blob = _parse_yaml(resolved)
    providers_blob = blob.get("providers") or {}
    providers: Dict[str, ProviderConfig] = {
        name: _coerce_provider(name, prov or {})
        for name, prov in providers_blob.items()
    }
    if not providers:
        providers = _default_config().providers

    cfg = LLMConfig(
        providers=providers,
        routing=dict(blob.get("routing") or {}),
        default_provider=str(blob.get("default_provider") or "openai"),
        fallback_provider=blob.get("fallback_provider") or "openai",
    )
    _apply_env_overrides(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Module-level singleton (with cache-invalidation for tests)
# ---------------------------------------------------------------------------

_CACHED: Optional[LLMConfig] = None


def get_llm_config() -> LLMConfig:
    global _CACHED
    if _CACHED is None:
        _CACHED = load_llm_config()
    return _CACHED


def reset_llm_config() -> None:
    """Clear the cache — tests call this before and after mutating env vars."""
    global _CACHED
    _CACHED = None
