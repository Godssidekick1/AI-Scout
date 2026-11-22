"""
Understat Data Source
=====================
The grounding source: real season stats (minutes, goals, assists, xG, xA, shots,
key passes, xGChain, xGBuildup, cards) for every player in the top five leagues.

One request per league per season, cached for 24h, then all lookups are local.
No API key needed.
"""

import json
import logging
import re
from typing import Dict, List, Optional

from thefuzz import fuzz

from config import settings
from data_sources.base_source import BaseSource, http_request
from data_sources.data_cache import DataCache

logger = logging.getLogger(__name__)

# Our league key -> Understat slug
UNDERSTAT_LEAGUES = {
    "premier_league": "EPL",
    "la_liga": "La_liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie_A",
    "ligue_1": "Ligue_1",
}

# Understat position codes: GK, D, M, F, S (sub). A player can carry several, e.g. "F M S".
POSITION_FAMILIES = {
    "goalkeeper": {"GK"},
    "defender": {"D"},
    "midfielder": {"M"},
    "forward": {"F"},
}

_FAMILY_KEYWORDS = [
    ("goalkeeper", ("goalkeeper", "keeper", "gk")),
    ("defender", ("back", "defender", "centre-back", "center-back", "cb", "wing-back", "stopper", "libero")),
    ("forward", ("striker", "forward", "winger", "no.9", "number 9", "wide", "attacker", "false 9")),
    ("midfielder", ("midfield", "cdm", "no.10", "number 10", "playmaker", "pivot", "box-to-box", "regista", "mezzala")),
]


def position_family(position: str) -> str:
    """Map a free-text position ("inverted winger") to a coarse family ("forward")."""
    p = (position or "").lower()
    for family, words in _FAMILY_KEYWORDS:
        if any(w in p for w in words):
            return family
    return "forward"


def allowed_codes(requested: str) -> set:
    """Understat codes that count as the requested role.
    Wingers and attacking midfielders are listed as F or M, so we accept both for them."""
    family = position_family(requested)
    allowed = set(POSITION_FAMILIES[family])
    p = (requested or "").lower()
    if family == "forward" and any(w in p for w in ("winger", "wide", "inside")):
        allowed |= {"M"}
    if family == "midfielder" and any(w in p for w in ("attacking", "no.10", "number 10", "playmaker")):
        allowed |= {"F"}
    return allowed


def position_matches(understat_position: str, requested: str) -> bool:
    """Does an Understat position string fit the requested role?"""
    return bool(set((understat_position or "").split()) & allowed_codes(requested))


class UnderstatSource(BaseSource):
    source_name = "understat"
    BASE_URL = "https://understat.com"

    def __init__(self, cache: Optional[DataCache] = None, season: Optional[int] = None):
        self.cache = cache or DataCache()
        self.season = season or settings.UNDERSTAT_SEASON
        self.headers = {
            "User-Agent": "ai-scout/2.0 (portfolio project)",
            "Accept": "application/json, text/html;q=0.9",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{self.BASE_URL}/",
        }

    # ── fetching ───────────────────────────────────────────────
    async def league_players(self, league: str) -> List[Dict]:
        """All players of one league this season, normalised to per-90 stats."""
        raw = await self._fetch_league_raw(league)
        return [p for p in (self._normalise(r, league) for r in raw) if p]

    async def _fetch_league_raw(self, league: str) -> List[Dict]:
        slug = UNDERSTAT_LEAGUES.get(league)
        if not slug:
            return []
        cache_key = f"understat:league:{slug}:{self.season}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached.get("players", [])

        players = await self._fetch_json_api(slug) or await self._fetch_html(slug)
        if players:
            self.cache.set(cache_key, {"players": players})
            return players
        stale = self.cache.get(cache_key, stale_ok=True)        # site down: the last good copy
        if stale is not None:
            logger.warning("Understat: %s %s unreachable, using the cached copy", slug, self.season)
            return stale.get("players", [])
        logger.warning("Understat: no data for %s %s", slug, self.season)
        return players

    async def _fetch_json_api(self, slug: str) -> List[Dict]:
        """Current site: a JSON endpoint returning {"players": [...], "teams": ..., "dates": ...}."""
        resp = await http_request("GET", f"{self.BASE_URL}/getLeagueData/{slug}/{self.season}", headers=self.headers)
        if resp is None:
            return []
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            return []
        return self.parse_league_payload(data)

    async def _fetch_html(self, slug: str) -> List[Dict]:
        """Older site: player data embedded as `var playersData = JSON.parse('...')`."""
        resp = await http_request("GET", f"{self.BASE_URL}/league/{slug}/{self.season}", headers=self.headers)
        if resp is None:
            return []
        if self.is_maintenance_page(resp.text):
            logger.warning("Understat is showing its maintenance page ('We will come back soon'); try again later")
            return []
        return self.parse_embedded_html(resp.text)

    @staticmethod
    def is_maintenance_page(html: str) -> bool:
        """Understat answers every URL with HTTP 200 and this page while it is down."""
        return "come back soon" in (html or "").lower() and "playersData" not in (html or "")

    @staticmethod
    def parse_league_payload(data) -> List[Dict]:
        if isinstance(data, dict):
            players = data.get("players")
            if isinstance(players, list):
                return players
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def parse_embedded_html(html: str) -> List[Dict]:
        match = re.search(r"var\s+playersData\s*=\s*JSON\.parse\('(.+?)'\)", html)
        if not match:
            return []
        try:
            return json.loads(match.group(1).encode("utf-8").decode("unicode_escape"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []

    # ── normalising ────────────────────────────────────────────
    def _normalise(self, p: Dict, league: str) -> Optional[Dict]:
        name = p.get("player_name") or p.get("name")
        if not name:
            return None
        minutes = self._safe_int(p.get("time"))
        n90 = max(minutes / 90, 1.0)

        def per90(key: str) -> float:
            return round(self._safe_float(p.get(key)) / n90, 3)

        return {
            "player_id": f"us:{p.get('id', name)}",
            "name": name,
            "team": p.get("team_title", ""),
            "league": league,
            "understat_position": p.get("position", ""),
            "minutes_played": minutes,
            "games": self._safe_int(p.get("games")),
            "goals": self._safe_int(p.get("goals")),
            "assists": self._safe_int(p.get("assists")),
            "goals_p90": per90("goals"),
            "assists_p90": per90("assists"),
            "xG_p90": per90("xG"),
            "xA_p90": per90("xA"),
            "npxG_p90": per90("npxG"),
            "shots_p90": per90("shots"),
            "key_passes_p90": per90("key_passes"),
            "xGChain_p90": per90("xGChain"),
            "xGBuildup_p90": per90("xGBuildup"),
            "yellow_cards": self._safe_int(p.get("yellow_cards")),
            "red_cards": self._safe_int(p.get("red_cards")),
            "season": f"{self.season}-{str(self.season + 1)[-2:]}",
        }

    # ── queries ────────────────────────────────────────────────
    async def search(self, league: str, position: Optional[str] = None, min_minutes: int = 0,
                     sort_by: str = "xGChain_p90", limit: int = 20) -> List[Dict]:
        return filter_players(await self.league_players(league), position, min_minutes, sort_by, limit)

    async def find_player(self, name: str, team: Optional[str] = None,
                          leagues: Optional[List[str]] = None) -> Optional[Dict]:
        rows: List[Dict] = []
        for league in leagues or list(UNDERSTAT_LEAGUES):
            rows += await self.league_players(league)
        return best_match(name, rows, team)


def filter_players(rows: List[Dict], position: Optional[str] = None, min_minutes: int = 0,
                   sort_by: str = "xGChain_p90", limit: int = 20) -> List[Dict]:
    """Players with enough minutes in the requested role, best first by `sort_by`."""
    out = [
        p for p in rows
        if p["minutes_played"] >= min_minutes
        and (not position or position_matches(p["understat_position"], position))
    ]
    out.sort(key=lambda p: p.get(sort_by, 0.0), reverse=True)
    return out[:limit]


def best_match(name: str, rows: List[Dict], team: Optional[str] = None, min_score: int = 85) -> Optional[Dict]:
    """Fuzzy-match a name against player rows; a matching club breaks ties between namesakes."""
    best, best_score = None, 0
    for p in rows:
        score = fuzz.token_sort_ratio(name.lower(), p["name"].lower())
        if team and p.get("team") and fuzz.partial_ratio(team.lower(), p["team"].lower()) >= 80:
            score += 5
        if score > best_score:
            best, best_score = p, score
    return best if best_score >= min_score else None
