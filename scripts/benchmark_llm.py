#!/usr/bin/env python3
"""
benchmark_llm.py
================
Concurrent throughput + latency benchmark for ChatDFT LLM providers.

Runs the same prompt N times at a fixed concurrency level against each
provider listed on the command line and prints p50 / p95 / p99 latency,
wall-clock time, tokens/sec and USD cost. Use it to validate that vLLM is
actually faster than hitting api.openai.com for a given workload before
flipping routing rules in ``server/llm.yaml``.

Usage
-----
    python scripts/benchmark_llm.py \\
        --providers openai vllm_local \\
        --n 50 \\
        --concurrency 16 \\
        --agent intent_agent

Output
------
Human-readable table to stdout; a structured JSON dump to
``runs/llm_benchmark_<timestamp>.json`` for regression tracking.

Safety
------
The benchmark uses `log_to_agentlog=False` so it doesn't pollute AgentLog
and skew the monitoring dashboard with synthetic traffic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.utils.llm_config import get_llm_config  # noqa: E402
from server.utils.llm_providers import LLMProvider, get_llm_router  # noqa: E402

# Representative intent-parsing prompt — structured JSON, realistic length.
DEFAULT_PROMPT = [
    {
        "role": "system",
        "content": (
            "You are an intent parser for computational catalysis. "
            "Return STRICT JSON only with keys: stage, area, task, "
            "system(catalyst, facet, material), conditions(pH, potential, T). "
            "Do not include any prose or code fences."
        ),
    },
    {
        "role": "user",
        "content": (
            "CO2 reduction to methanol on Cu(111) at pH 7, potential -0.8 V "
            "vs RHE, room temperature. Return the parsed intent."
        ),
    },
]


@dataclass
class CallStats:
    provider: str
    latency_ms: int
    success: bool
    input_tokens: int
    output_tokens: int


@dataclass
class ProviderReport:
    provider: str
    requests: int
    successes: int
    wall_s: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    tokens_per_sec: float
    total_input_tokens: int
    total_output_tokens: int
    cost_usd: float
    errors: List[str] = field(default_factory=list)


def _percentile(values: List[int], pct: float) -> float:
    return float(np.percentile(values, pct)) if values else 0.0


def _cost_usd(provider_type: str, model: str, in_tok: int, out_tok: int) -> float:
    """Crude cost estimate. Local providers are free."""
    if provider_type == "vllm":
        return 0.0
    rates = {
        "gpt-4o": (0.005, 0.015),
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-3.5-turbo": (0.0005, 0.0015),
    }
    rate_in, rate_out = rates.get(model, (0.0, 0.0))
    return (in_tok * rate_in + out_tok * rate_out) / 1000.0


async def _one_call(
    provider: LLMProvider,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    json_mode: bool,
) -> CallStats:
    result = await provider.chat_completion(
        messages=messages,
        temperature=0.1,
        max_tokens=max_tokens,
        json_mode=json_mode,
        retries=1,
    )
    tokens = result.tokens()
    return CallStats(
        provider=provider.name,
        latency_ms=result.latency_ms,
        success=result.success,
        input_tokens=tokens.get("prompt_tokens", 0),
        output_tokens=tokens.get("completion_tokens", 0),
    )


async def _run_provider(
    provider: LLMProvider,
    n: int,
    concurrency: int,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    json_mode: bool,
) -> ProviderReport:
    print(f"\n⧗ benchmarking {provider.name} ({provider.config.model}) …", flush=True)
    sem = asyncio.Semaphore(concurrency)
    results: List[CallStats] = []
    errors: List[str] = []

    async def _wrapped():
        async with sem:
            try:
                stats = await _one_call(provider, messages, max_tokens, json_mode)
                results.append(stats)
                if not stats.success:
                    errors.append(f"failed call (latency={stats.latency_ms}ms)")
            except Exception as e:
                errors.append(str(e))

    t0 = time.time()
    await asyncio.gather(*(_wrapped() for _ in range(n)))
    wall = time.time() - t0

    latencies = [r.latency_ms for r in results if r.success and r.latency_ms > 0]
    in_tok = sum(r.input_tokens for r in results)
    out_tok = sum(r.output_tokens for r in results)
    successes = sum(1 for r in results if r.success)
    tps = (in_tok + out_tok) / wall if wall > 0 else 0.0

    return ProviderReport(
        provider=provider.name,
        requests=n,
        successes=successes,
        wall_s=round(wall, 3),
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        mean_ms=float(np.mean(latencies)) if latencies else 0.0,
        tokens_per_sec=round(tps, 1),
        total_input_tokens=in_tok,
        total_output_tokens=out_tok,
        cost_usd=round(
            _cost_usd(provider.config.type, provider.config.model or "", in_tok, out_tok),
            4,
        ),
        errors=errors[:5],  # keep only a few
    )


def _print_table(reports: List[ProviderReport]) -> None:
    print("\nbenchmark_llm")
    print("=============")
    header = f"{'provider':<14}{'req':>5}{'ok':>4}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'tok/sec':>10}{'wall s':>9}{'USD':>10}"
    print(header)
    print("-" * len(header))
    for r in reports:
        print(
            f"{r.provider:<14}{r.requests:>5}{r.successes:>4}"
            f"{r.p50_ms:>9.0f}{r.p95_ms:>9.0f}{r.p99_ms:>9.0f}"
            f"{r.tokens_per_sec:>10.1f}{r.wall_s:>9.2f}${r.cost_usd:>8.4f}"
        )

    if len(reports) >= 2:
        walls = {r.provider: r.wall_s for r in reports if r.wall_s > 0}
        if len(walls) >= 2:
            slowest = max(walls.values())
            print()
            for p, w in walls.items():
                if w > 0:
                    print(f"  {p:<14} speedup vs slowest: {slowest / w:.2f}x")

    # Surface errors if any
    for r in reports:
        if r.errors:
            print(f"\n  {r.provider} errors ({len(r.errors)}):")
            for e in r.errors:
                print(f"    - {e}")


def _save_json(reports: List[ProviderReport], args: argparse.Namespace) -> Path:
    runs_dir = Path(args.out_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = runs_dir / f"llm_benchmark_{ts}.json"
    path.write_text(json.dumps({
        "timestamp": ts,
        "args": vars(args),
        "reports": [asdict(r) for r in reports],
    }, indent=2))
    return path


async def main_async(args: argparse.Namespace) -> int:
    cfg = get_llm_config()
    router = get_llm_router()

    requested = list(args.providers) or list(cfg.providers.keys())
    unknown = [p for p in requested if p not in cfg.providers]
    if unknown:
        print(f"unknown providers: {unknown}", file=sys.stderr)
        return 2

    providers = [router.provider(p) for p in requested]
    providers = [p for p in providers if p is not None]
    if not providers:
        print("no providers available", file=sys.stderr)
        return 2

    # Pre-flight: health check each provider so we don't blast 50 calls at a
    # dead vLLM server.
    healthy: List[LLMProvider] = []
    for p in providers:
        ok = await p.health_check()
        print(f"  {p.name}: health = {'OK' if ok else 'FAIL'}")
        if ok:
            healthy.append(p)

    if not healthy:
        print("no healthy providers — aborting", file=sys.stderr)
        return 1

    reports: List[ProviderReport] = []
    for p in healthy:
        r = await _run_provider(
            provider=p,
            n=args.n,
            concurrency=args.concurrency,
            messages=DEFAULT_PROMPT,
            max_tokens=args.max_tokens,
            json_mode=args.json_mode,
        )
        reports.append(r)

    _print_table(reports)
    out = _save_json(reports, args)
    print(f"\nraw report: {out}")
    return 0


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ChatDFT LLM provider benchmark")
    p.add_argument("--providers", nargs="*", default=[],
                   help="Provider names from llm.yaml. Default: all configured.")
    p.add_argument("--n", type=int, default=50, help="Total requests per provider")
    p.add_argument("--concurrency", type=int, default=16, help="Max in-flight requests")
    p.add_argument("--max-tokens", type=int, default=400, dest="max_tokens")
    p.add_argument("--json-mode", action="store_true", default=True, dest="json_mode")
    p.add_argument("--text-mode", action="store_false", dest="json_mode")
    p.add_argument("--agent", default="intent_agent",
                   help="Agent name (informational; prompt is fixed for now)")
    p.add_argument("--out-dir", default="runs", dest="out_dir")
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
