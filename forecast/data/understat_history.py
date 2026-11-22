"""
Understat history
=================
Player-season totals for the big five, one row per player per team per season,
from 2014/15 (Understat's first season) onward.

Past seasons never change, so they are kept for good (the Internet Archive rebuild lives
only in this cache); the current season uses the normal 24h cache, and falls back to
the last cached copy when understat.com is down.


"""

from __future__ import annotations

import asyncio
import html
import logging
from typing import Dict, Iterable, List

import pandas as pd

from config import current_understat_season, settings
from data_sources.data_cache import DataCache
from data_sources.understat_source import UnderstatSource
from forecast.config import LEAGUES, METRICS

logger = logging.getLogger(__name__)

FIRST_SEASON = 2014
COLUMNS = ["us_id", "name", "team", "league", "season", "understat_position", "minutes", "games",
           "goals", "assists", "xG", "npg", *METRICS]


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def rows_from_raw(raw: Iterable[Dict], league: str, season: int) -> List[Dict]:
    """Understat's raw player list -> our rows. A player who moved mid-season is
    listed once with both teams joined by a comma; we keep the last team."""
    out = []
    for p in raw:
        name = html.unescape(p.get("player_name") or p.get("name") or "")   # "N&#039;Goumou" -> "N'Goumou"
        if not name or p.get("id") is None:
            continue
        team = html.unescape((p.get("team_title") or "").split(",")[-1].strip())
        out.append({
            "us_id": str(p["id"]), "name": name, "team": team, "league": league, "season": season,
            "understat_position": p.get("position", ""),
            "minutes": _num(p.get("time")), "games": _num(p.get("games")),
            "goals": _num(p.get("goals")), "assists": _num(p.get("assists")),
            "xG": _num(p.get("xG")), "npg": _num(p.get("npg")),          # xG - npxG = penalties taken x 0.76
            **{m: _num(p.get(m)) for m in METRICS},
        })
    return out


async def load_seasons(seasons: Iterable[int], leagues: Iterable[str] = LEAGUES,
                       concurrency: int = 3) -> pd.DataFrame:
    """Fetch every (league, season) and return one DataFrame of player-season totals."""
    current = current_understat_season()
    history_cache = DataCache(cache_dir=str(settings.cache_path), ttl_hours=24 * 365 * 100)   # finished seasons never change
    live_cache = DataCache(cache_dir=str(settings.cache_path), ttl_hours=settings.CACHE_TTL_HOURS)
    sem = asyncio.Semaphore(concurrency)

    async def one(league: str, season: int) -> List[Dict]:
        source = UnderstatSource(cache=live_cache if season >= current else history_cache, season=season)
        async with sem:
            raw = await source._fetch_league_raw(league)
        if not raw:
            logger.warning("Understat: no data for %s %s", league, season)
        return rows_from_raw(raw, league, season)

    jobs = [one(lg, s) for s in seasons for lg in leagues]
    rows = [r for chunk in await asyncio.gather(*jobs) for r in chunk]
    return pd.DataFrame(rows, columns=COLUMNS)


def default_seasons() -> List[int]:
    return list(range(FIRST_SEASON, current_understat_season() + 1))
