"""
Run Tracing
===========
Lightweight, dependency-free observability for one pipeline run:
per-step wall time, every LLM call (model, tokens, latency, cache hit, estimated cost),
tool calls and outbound data-source requests.

The active trace lives in a ContextVar, so every coroutine spawned inside a run
(LangGraph nodes, asyncio.gather fan-outs) records into the same trace.

For full hosted traces, LangChain's built-in LangSmith integration also works:
set LANGSMITH_TRACING=true and LANGSMITH_API_KEY.
"""

from __future__ import annotations

import functools
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import settings


@dataclass
class Span:
    name: str
    kind: str                 # "node" | "llm" | "tool" | "source"
    start: float
    duration_ms: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunTrace:
    started: float = field(default_factory=time.perf_counter)
    spans: List[Span] = field(default_factory=list)

    # ── recording ──────────────────────────────────────────────
    def add(self, span: Span) -> None:
        self.spans.append(span)

    def record_llm(self, name: str, model: str, input_tokens: int, output_tokens: int,
                   duration_ms: float, cached: bool = False, attempts: int = 1) -> None:
        self.add(Span(name=name, kind="llm", start=time.perf_counter(), duration_ms=duration_ms, meta={
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached": cached,
            "attempts": attempts,
            "cost_usd": 0.0 if cached else estimate_cost(model, input_tokens, output_tokens),
        }))

    # ── reporting ──────────────────────────────────────────────
    def of_kind(self, kind: str) -> List[Span]:
        return [s for s in self.spans if s.kind == kind]

    def summary(self) -> Dict[str, Any]:
        llm = self.of_kind("llm")
        live = [s for s in llm if not s.meta.get("cached")]
        nodes: Dict[str, float] = {}
        for s in self.of_kind("node"):
            nodes[s.name] = round(nodes.get(s.name, 0.0) + s.duration_ms, 1)
        return {
            "total_ms": round((time.perf_counter() - self.started) * 1000, 1),
            "node_ms": nodes,
            "node_runs": [s.name for s in self.of_kind("node")],
            "llm_calls": len(live),
            "llm_cache_hits": len(llm) - len(live),
            "llm_retries": sum(s.meta.get("attempts", 1) - 1 for s in live),
            "input_tokens": sum(s.meta["input_tokens"] for s in live),
            "output_tokens": sum(s.meta["output_tokens"] for s in live),
            "est_cost_usd": round(sum(s.meta["cost_usd"] for s in live), 5),
            "tool_calls": len(self.of_kind("tool")),
            "source_requests": len(self.of_kind("source")),
        }


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = settings.LLM_PRICES_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


_current: ContextVar[Optional[RunTrace]] = ContextVar("ai_scout_trace", default=None)


def current_trace() -> RunTrace:
    """The active run's trace (a throwaway one when called outside a run)."""
    trace = _current.get()
    if trace is None:
        trace = RunTrace()
        _current.set(trace)
    return trace


def start_trace() -> RunTrace:
    trace = RunTrace()
    _current.set(trace)
    return trace


@asynccontextmanager
async def span(name: str, kind: str, **meta):
    s = Span(name=name, kind=kind, start=time.perf_counter(), meta=dict(meta))
    try:
        yield s
    finally:
        s.duration_ms = round((time.perf_counter() - s.start) * 1000, 1)
        current_trace().add(s)


def traced_node(name: str):
    """Decorator: time an async LangGraph node as a 'node' span."""
    def wrap(fn):
        @functools.wraps(fn)
        async def inner(state):
            async with span(name, "node"):
                return await fn(state)
        return inner
    return wrap
