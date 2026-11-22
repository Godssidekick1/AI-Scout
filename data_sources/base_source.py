"""
Base Data Source
================
Shared helpers for all football data sources: a retrying HTTP request helper
and safe numeric conversion.
"""

import asyncio
import logging
import random
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


async def http_request(method: str, url: str, *, headers: Optional[dict] = None,
                       params: Optional[dict] = None, json_body: Optional[dict] = None,
                       retries: Optional[int] = None) -> Optional[httpx.Response]:
    """
    GET/POST with a timeout and exponential backoff on transport errors,
    429 and 5xx. Returns the response on 2xx, None otherwise (callers degrade
    gracefully rather than crash the run).
    """
    retries = settings.HTTP_MAX_RETRIES if retries is None else retries
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_S, follow_redirects=True) as client:
                resp = await client.request(method, url, headers=headers, params=params, json=json_body)
            if resp.status_code < 400:
                return resp
            if resp.status_code not in RETRYABLE_STATUS:
                logger.warning("%s %s -> HTTP %d", method, url, resp.status_code)
                return None
            reason = f"HTTP {resp.status_code}"
        except httpx.HTTPError as e:
            reason = f"{type(e).__name__}: {e}"
        if attempt < retries:
            delay = 0.5 * 2 ** attempt + random.uniform(0, 0.25)
            logger.debug("%s %s failed (%s); retry in %.1fs", method, url, reason, delay)
            await asyncio.sleep(delay)
        else:
            logger.warning("%s %s failed after %d attempts (%s)", method, url, retries + 1, reason)
    return None


class BaseSource:
    """Common base for football data sources."""

    source_name: str = "base"

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            if value is None or value == "" or value == "-":
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            if value is None or value == "" or value == "-":
                return default
            return int(float(value))
        except (ValueError, TypeError):
            return default
