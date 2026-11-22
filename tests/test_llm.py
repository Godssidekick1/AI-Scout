"""Tests for the LLM layer: retries, validation, caching, token and cost accounting."""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from config import settings
from core import llm
from core.tracing import start_trace
from models.llm_schemas import SearchConstraints
from tests.conftest import run

MSGS = [HumanMessage(content="winger under 25")]
GOOD = SearchConstraints(position="winger", max_age=24)


class FakeStructuredModel:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def with_structured_output(self, schema, include_raw=False):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        raw = AIMessage(content="", usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200})
        return {"raw": raw, "parsed": out, "parsing_error": None if out is not None else "bad json"}


@pytest.fixture
def fast_sleep(monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: real_sleep(0))


def _use(monkeypatch, model):
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "k")
    monkeypatch.setattr(llm, "get_chat_model", lambda *a, **k: model)


def test_retries_then_succeeds_and_records_usage(monkeypatch, fast_sleep):
    model = FakeStructuredModel([TimeoutError("slow"), None, GOOD])   # error, invalid output, success
    _use(monkeypatch, model)

    async def go():
        trace = start_trace()
        out = await llm.structured(SearchConstraints, MSGS, name="parse", model="gemini-2.5-flash")
        return out, trace.summary()

    out, summary = run(go())
    assert out == GOOD
    assert model.calls == 3
    assert summary["llm_calls"] == 1 and summary["llm_retries"] == 2
    assert summary["input_tokens"] == 1000 and summary["output_tokens"] == 200
    assert summary["est_cost_usd"] == pytest.approx((1000 * 0.30 + 200 * 2.50) / 1e6)


def test_gives_up_after_max_retries(monkeypatch, fast_sleep):
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 2)
    _use(monkeypatch, FakeStructuredModel([RuntimeError("429"), RuntimeError("429")]))
    with pytest.raises(llm.LLMError):
        run(llm.structured(SearchConstraints, MSGS, name="parse"))


def test_cache_hit_skips_the_model(monkeypatch):
    monkeypatch.setattr(settings, "LLM_CACHE", True)
    model = FakeStructuredModel([GOOD])
    _use(monkeypatch, model)

    async def go():
        trace = start_trace()
        a = await llm.structured(SearchConstraints, MSGS, name="parse")
        b = await llm.structured(SearchConstraints, MSGS, name="parse")
        return a, b, trace.summary()

    a, b, summary = run(go())
    assert a == b == GOOD
    assert model.calls == 1
    assert summary["llm_calls"] == 1 and summary["llm_cache_hits"] == 1
