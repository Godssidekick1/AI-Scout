"""
Run Context
===========
Per-run services shared by every node: data sources, a concurrency limit and a
memo so the same lookup (e.g. one player's Transfermarkt profile) is fetched once
per run no matter how many agents ask for it, even when they ask at the same time.

Kept in a ContextVar rather than in the LangGraph state, because it holds live
objects (clients, tasks) that should not be serialised into state.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config import settings
from core.tracing import span
from data_sources.data_cache import DataCache
from data_sources.ea_fc_ratings import EAFCRatingsSource
from data_sources.news_search import NewsSearchSource
from data_sources.transfermarkt_source import TransfermarktSource
from data_sources.understat_source import UNDERSTAT_LEAGUES, UnderstatSource, best_match, filter_players

logger = logging.getLogger(__name__)


class RunContext:
    def __init__(self):
        cache = DataCache(cache_dir=settings.CACHE_DIR, ttl_hours=settings.CACHE_TTL_HOURS)
        market_cache = DataCache(cache_dir=settings.CACHE_DIR, ttl_hours=168)
        self.understat = UnderstatSource(cache=cache)
        self.transfermarkt = TransfermarktSource(cache=market_cache)
        self.ea_fc = EAFCRatingsSource(cache=cache)
        self.news = NewsSearchSource(cache=cache)
        self._sem: Optional[asyncio.Semaphore] = None
        self._memo: Dict[str, asyncio.Future] = {}
        self.errors: List[str] = []

    @property
    def sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(settings.MAX_CONCURRENCY)
        return self._sem

    def record_error(self, msg: str) -> None:
        logger.warning(msg)
        if msg not in self.errors:
            self.errors.append(msg)

    async def once(self, key: str, source: str, factory: Callable[[], Awaitable[Any]]) -> Any:
        """Run `factory` at most once per key per run; concurrent callers share the result.
        Failures are recorded and return None instead of crashing the step."""
        if key not in self._memo:
            async def guarded():
                async with self.sem:
                    async with span(source, "source", key=key):
                        try:
                            return await factory()
                        except Exception as e:  # noqa: BLE001 - a source failure must not kill the run
                            self.record_error(f"{source} lookup failed for {key}: {e}")
                            return None
            self._memo[key] = asyncio.ensure_future(guarded())
        return await self._memo[key]

    # ── memoised lookups ───────────────────────────────────────
    async def league(self, league: str) -> List[Dict]:
        rows = await self.once(f"understat:{league}", "understat", lambda: self.understat.league_players(league))
        return rows or []

    async def search(self, league: str, position: Optional[str] = None, min_minutes: int = 0,
                     sort_by: str = "xGChain_p90", limit: int = 20) -> List[Dict]:
        return filter_players(await self.league(league), position, min_minutes, sort_by, limit)

    async def find_player(self, name: str, team: Optional[str] = None) -> Optional[Dict]:
        """Resolve a name to a real Understat player in any of the five leagues."""
        leagues = await asyncio.gather(*(self.league(lg) for lg in UNDERSTAT_LEAGUES))
        return best_match(name, [p for rows in leagues for p in rows], team)

    async def forecast(self):
        """The fitted forecast service (python -m forecast.build), or None if not built or disabled."""
        if not settings.ENABLE_FORECAST:
            return None

        async def load():
            from forecast.service import ForecastService
            return await asyncio.to_thread(ForecastService.load)

        if "forecast" not in self._memo:
            async def guarded():
                try:
                    return await load()
                except Exception as e:  # noqa: BLE001 - the model is an enhancement; the run goes on without it
                    self.record_error(f"forecast model unavailable: {e}")
                    return None
            self._memo["forecast"] = asyncio.ensure_future(guarded())
        return await self._memo["forecast"]

    async def market_profile(self, name: str) -> Optional[Dict]:
        if not settings.ENABLE_TRANSFERMARKT:
            return None
        return await self.once(f"tm:{name.lower()}", "transfermarkt", lambda: self.transfermarkt.get_player_stats(name))

    async def ea_ratings(self, name: str) -> Optional[Dict]:
        if not settings.ENABLE_EA_FC:
            return None
        return await self.once(f"ea:{name.lower()}", "ea_fc", lambda: self.ea_fc.get_player_stats(name))

    async def news_summary(self, name: str) -> Optional[Dict]:
        if not settings.has_serper():
            return None
        return await self.once(f"news:{name.lower()}", "serper", lambda: self.news.get_sentiment_summary(name))


_current: ContextVar[Optional[RunContext]] = ContextVar("ai_scout_context", default=None)


def get_context() -> RunContext:
    ctx = _current.get()
    if ctx is None:
        ctx = RunContext()
        _current.set(ctx)
    return ctx


def start_context() -> RunContext:
    ctx = RunContext()
    _current.set(ctx)
    return ctx
