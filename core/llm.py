"""
LLM Access Layer
================
Every Gemini call in the pipeline goes through this module, which gives us in one place:

- one shared client per (model, temperature) instead of a new client per call
- schema-first structured output (`with_structured_output` bound to a pydantic model),
  so there is no "please return JSON" prompting and no Markdown-fence stripping
- retries with exponential backoff + jitter on errors, timeouts and invalid output
- a response cache keyed by a hash of (model, schema, messages)
- token / latency / cost accounting into the run trace

Tests replace `structured` / `get_chat_model` with fakes, so no network is needed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from typing import Awaitable, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from config import settings
from core.tracing import current_trace
from data_sources.data_cache import DataCache

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


class LLMError(RuntimeError):
    """Raised when an LLM step fails after all retries."""


_models: Dict[Tuple[str, float, int], object] = {}
_cache: Optional[DataCache] = None


def get_chat_model(model: Optional[str] = None, temperature: float = 0.2):
    """Shared Gemini chat client. Retries are handled here, not by the SDK."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = model or settings.GEMINI_MODEL
    try:
        loop_id = id(asyncio.get_running_loop())   # async clients are bound to one event loop
    except RuntimeError:
        loop_id = 0
    key = (model, temperature, loop_id)
    if key not in _models:
        _models[key] = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
            timeout=settings.LLM_TIMEOUT_S,
            max_retries=0,
        )
    return _models[key]


def _llm_cache() -> DataCache:
    global _cache
    if _cache is None:
        _cache = DataCache(cache_dir=settings.CACHE_DIR, ttl_hours=settings.CACHE_TTL_HOURS)
    return _cache


async def with_retries(fn: Callable[[], Awaitable[R]], what: str,
                       attempts: Optional[int] = None) -> Tuple[R, int]:
    """Run `fn` with exponential backoff + jitter. Returns (result, attempts_used)."""
    attempts = attempts or settings.LLM_MAX_RETRIES
    last: Optional[BaseException] = None
    for i in range(1, attempts + 1):
        try:
            return await asyncio.wait_for(fn(), timeout=settings.LLM_TIMEOUT_S), i
        except Exception as e:  # noqa: BLE001 - every failure is retryable here
            last = e
            if i == attempts:
                break
            delay = min(8.0, 0.5 * 2 ** (i - 1)) + random.uniform(0, 0.25)
            logger.warning("%s failed (attempt %d/%d): %s; retrying in %.1fs", what, i, attempts, e, delay)
            await asyncio.sleep(delay)
    raise LLMError(f"{what} failed after {attempts} attempts: {last}") from last


def _usage(msg: Optional[BaseMessage]) -> Tuple[int, int]:
    meta = getattr(msg, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))


def _cache_key(model: str, schema: Type[BaseModel], messages: List[BaseMessage]) -> str:
    payload = json.dumps({
        "model": model,
        "schema": schema.model_json_schema(),
        "messages": [(m.type, m.content) for m in messages],
    }, sort_keys=True, default=str)
    return "llm:" + hashlib.sha256(payload.encode()).hexdigest()


async def structured(schema: Type[T], messages: List[BaseMessage], *, name: str,
                     model: Optional[str] = None, temperature: float = 0.2) -> T:
    """Call Gemini and get back a validated instance of `schema`."""
    model = model or settings.GEMINI_MODEL
    trace = current_trace()

    key = _cache_key(model, schema, messages)
    if settings.LLM_CACHE:
        hit = _llm_cache().get(key)
        if hit is not None:
            trace.record_llm(name, model, 0, 0, 0.0, cached=True)
            return schema.model_validate(hit)

    runnable = get_chat_model(model, temperature).with_structured_output(schema, include_raw=True)

    async def call():
        out = await runnable.ainvoke(messages)
        if out.get("parsing_error") or out.get("parsed") is None:
            raise ValueError(f"invalid structured output: {out.get('parsing_error')}")
        return out

    t0 = time.perf_counter()
    out, used = await with_retries(call, name)
    tin, tout = _usage(out.get("raw"))
    trace.record_llm(name, model, tin, tout, (time.perf_counter() - t0) * 1000, attempts=used)

    parsed = out["parsed"]
    result = parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
    if settings.LLM_CACHE:
        _llm_cache().set(key, result.model_dump(mode="json"))
    return result


async def invoke_with_tools(bound_model, messages: List[BaseMessage], *, name: str,
                            model: Optional[str] = None) -> AIMessage:
    """One turn of a tool-calling conversation, with retries and accounting."""
    model = model or settings.GEMINI_MODEL
    t0 = time.perf_counter()
    msg, used = await with_retries(lambda: bound_model.ainvoke(messages), name)
    tin, tout = _usage(msg)
    current_trace().record_llm(name, model, tin, tout, (time.perf_counter() - t0) * 1000, attempts=used)
    return msg
