"""
Serie A player stats (Opta), from legaseriea.it
===============================================
The public JSON API behind legaseriea.it's stats pages (no key, no login):

  /competitions/{serie a}/seasons                      every season id
  /seasons/{id}/stats/players?category=General&page=N  30 players per page, each with
        ~280 {statsId, statsValue} pairs, incl. "Total Tackles", "Interceptions",
        "Total Clearances", "Blocks", "Recoveries", "Aerial Duels won/lost",
        "Duels won/lost", "Time Played"

Older seasons carry only a few counters (minutes, tackles won); a season counts as
covered only when tackles and interceptions are present, otherwise the FBref copy
is used for it. About 20 pages per season, ~1 request a second, cached (past
seasons forever, the current one for a day).

legaseriea.it's terms restrict automated collection and redistribution: fine for
personal research, but don't publish the raw data.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import pandas as pd

from config import current_understat_season
from forecast.data.defensive import PoliteClient, SourceUnavailable, frame

logger = logging.getLogger(__name__)

API = "https://api-sdp.legaseriea.it/v1/serie-a/football"
COMPETITION = "serie-a::Football_Competition::ec93b94f74294dc98ab5bcfd67fc0d88"
HEADERS = {"Origin": "https://www.legaseriea.it", "Referer": "https://www.legaseriea.it/"}

# our total -> statsIds to try, in order (the feed mixes Title-case and kebab-case ids)
FIELDS: Dict[str, tuple] = {
    "minutes": ("Time Played", "minutes-played", "Minuti"),
    "tackles": ("Total Tackles",),
    "tackles_won": ("Tackles Won", "tackles-won"),
    "interceptions": ("Interceptions",),
    "blocks": ("Blocks",),
    "clearances": ("Total Clearances",),
    "aerials_won": ("Aerial Duels won",),
    "aerials_lost": ("Aerial Duels lost",),
    "recoveries": ("Recoveries",),
    "duels_won": ("Duels won",),
    "duels_lost": ("Duels lost",),
}
_client: Optional[PoliteClient] = None


def client() -> PoliteClient:
    global _client
    if _client is None:
        _client = PoliteClient("legaseriea", min_interval=1.0, headers=HEADERS)
    return _client


def _ttl(season: int) -> Optional[float]:
    return 1.0 if season >= current_understat_season() else None


def season_ids() -> Dict[int, str]:
    """{starting year: seasonId}."""
    data = client().json("seasons", f"{API}/competitions/{COMPETITION}/seasons", params={"locale": "en-GB"},
                         ttl_days=7)
    items = data.get("seasons") if isinstance(data, dict) else data
    if isinstance(data, dict) and items is None:
        items = next((v for v in data.values() if isinstance(v, list)), [])
    out = {}
    for s in items or []:
        name = str(s.get("seasonName") or s.get("name") or "")
        if len(name) >= 4 and name[:4].isdigit() and s.get("seasonId"):
            out[int(name[:4])] = s["seasonId"]
    return out


def _value(stats: Dict[str, float], keys: tuple):
    for k in keys:
        if k in stats and stats[k] is not None:
            try:
                return float(stats[k])
            except (TypeError, ValueError):
                continue
    return None


def rows_from_players(players: List[Dict], season: int) -> List[Dict]:
    rows = []
    for p in players:
        stats = {s.get("statsId"): s.get("statsValue") for s in p.get("stats") or []}
        vals = {k: _value(stats, keys) for k, keys in FIELDS.items()}
        if not vals["minutes"]:
            continue
        team = p.get("team") or {}
        name = " ".join(x for x in (p.get("mediaFirstName"), p.get("mediaLastName")) if x) or p.get("displayName") \
            or p.get("shortName") or ""
        rows.append({"source": "legaseriea.it", "league": "serie_a", "season": season,
                     "src_id": str(p.get("playerId") or p.get("providerId")), "name": name,
                     "team": team.get("mediaShortName") or team.get("shortName") or team.get("officialName") or "",
                     "birth_year": None, **vals, "prog_passes": None, "prog_carries": None})
    return rows


def players(season_id: str, season: int, max_pages: int = 60) -> List[Dict]:
    out, seen = [], set()
    for page in range(1, max_pages + 1):
        data = client().json(f"players_{season}_{page}", f"{API}/seasons/{season_id}/stats/players",
                             params={"category": "General", "page": page, "locale": "en-GB"}, ttl_days=_ttl(season))
        batch = data.get("players") or []
        ids = {p.get("playerId") for p in batch}
        if not batch or ids <= seen:          # empty or repeated page = past the end
            return out
        seen |= ids
        out += batch
    return out


def load(seasons: Iterable[int]) -> pd.DataFrame:
    ids = season_ids()
    rows: List[Dict] = []
    for s in sorted(set(seasons)):
        if s not in ids:
            continue
        try:
            got = rows_from_players(players(ids[s], s), s)
        except SourceUnavailable as e:
            logger.warning("legaseriea.it: %s %s unavailable (%s)", "serie_a", s, e)
            continue
        rows += got
    return frame(rows)


def check() -> str:
    return "ok" if season_ids() else "DOWN (no seasons in response)"
