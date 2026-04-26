"""
Understat Data Source
=====================
Scrapes Understat for xG, xA, and shot-level data.
Uses Understat's internal JSON API endpoints.
No API key needed.
"""

import re
import json
import httpx
from typing import Dict, List, Optional
import logging
from data_sources.base_source import BaseSource
from data_sources.data_cache import DataCache

logger = logging.getLogger(__name__)

# Understat league URL slugs
UNDERSTAT_LEAGUES = {
    "premier_league": "EPL",
    "la_liga": "La_liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie_A",
    "ligue_1": "Ligue_1",
}

# Understat's internal API uses lowercase league names
UNDERSTAT_API_LEAGUES = {
    "premier_league": "epl",
    "la_liga": "la_liga",
    "bundesliga": "bundesliga",
    "serie_a": "serie_a",
    "ligue_1": "ligue_1",
}


class UnderstatSource(BaseSource):
    """
    Fetches xG/xA data from Understat's internal JSON API.
    Understat moved to client-side rendering, so we use their
    internal API endpoints instead of page scraping.
    """

    source_name = "understat"
    BASE_URL = "https://understat.com"

    # Current season (2024 = 2024-25 season on Understat)
    CURRENT_SEASON = 2024

    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://understat.com/",
            "X-Requested-With": "XMLHttpRequest",
        }

    async def _fetch_league_data(self, league: str, season: int = None) -> List[Dict]:
        """
        Fetch league player data from Understat.
        Tries the internal API first, then falls back to HTML scraping.
        """
        season = season or self.CURRENT_SEASON
        league_slug = UNDERSTAT_API_LEAGUES.get(league.lower(), "epl")

        cache_key = f"understat:league:{league_slug}:{season}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached.get("players", [])

        # Try multiple approaches to get data
        players = await self._try_api_endpoint(league_slug, season)
        if not players:
            players = await self._try_html_scraping(league, season)

        if players:
            self.cache.set(cache_key, {"players": players})

        return players

    async def _try_api_endpoint(self, league_slug: str, season: int) -> List[Dict]:
        """Try Understat's internal API endpoints."""
        api_paths = [
            f"/main/getPlayersStats/{league_slug}/{season}",
            f"/api/stats/league/{league_slug}/{season}",
            f"/stats/league/{league_slug}/{season}",
        ]

        for path in api_paths:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.get(
                        f"{self.BASE_URL}{path}",
                        headers=self.headers,
                        follow_redirects=True,
                    )
                    if response.status_code != 200:
                        continue

                    data = response.json()
                    # Check various response structures
                    if isinstance(data, list) and data:
                        return data
                    if isinstance(data, dict):
                        if "error" in data and data["error"].get("error_code"):
                            continue
                        # Try common keys
                        for key in ["players", "response", "data", "players_stats"]:
                            if key in data and isinstance(data[key], list):
                                return data[key]

            except (httpx.HTTPError, json.JSONDecodeError) as e:
                logger.debug(f"Understat API attempt failed ({path}): {e}")
                continue

        return []

    async def _try_html_scraping(self, league: str, season: int) -> List[Dict]:
        """
        Fallback: try to extract data from the HTML page.
        Even if embedded JSON is gone, try all known patterns.
        """
        league_slug = UNDERSTAT_LEAGUES.get(league.lower(), "EPL")
        url = f"{self.BASE_URL}/league/{league_slug}/{season}"

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, headers=self.headers, follow_redirects=True)
                response.raise_for_status()
                html = response.text

            # Try multiple extraction patterns
            for var_name in ["playersData", "players_data", "PLAYERS"]:
                data = self._extract_json_var(html, var_name)
                if data:
                    return data

            # Try finding any large JSON array in script tags
            data = self._extract_any_player_json(html)
            if data:
                return data

            logger.warning(f"Understat: Could not extract player data from {league_slug}/{season}")
            return []

        except httpx.HTTPError as e:
            logger.error(f"Understat HTML fetch error: {e}")
            return []

    def _extract_json_var(self, html: str, var_name: str) -> Optional[List]:
        """Extract JSON data from embedded script variables."""
        # Pattern 1: var NAME = JSON.parse('...')
        pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)
        if match:
            try:
                decoded = match.group(1).encode("utf-8").decode("unicode_escape")
                return json.loads(decoded)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # Pattern 2: var NAME = [...]  or var NAME = {...}
        pattern = rf"var\s+{var_name}\s*=\s*(\[.+?\]);?"
        match = re.search(pattern, html, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return None

    def _extract_any_player_json(self, html: str) -> Optional[List]:
        """Try to find any JSON array that looks like player data."""
        pattern = r'JSON\.parse\(\'(.+?)\'\)'
        matches = re.findall(pattern, html)

        for match_str in matches:
            try:
                decoded = match_str.encode("utf-8").decode("unicode_escape")
                data = json.loads(decoded)
                if isinstance(data, list) and len(data) > 10:
                    # Check if it looks like player data
                    if data[0] and any(k in str(data[0]).lower() for k in ["player", "xg", "goals"]):
                        return data
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

        return None

    async def search_players(
        self,
        name: Optional[str] = None,
        position: Optional[str] = None,
        league: Optional[str] = None,
        max_age: Optional[int] = None,
        min_minutes: int = 900,
    ) -> List[Dict]:
        """Search for players in a league."""
        target_league = (league or "premier_league").lower()

        players_data = await self._fetch_league_data(target_league)
        if not players_data:
            logger.warning(f"Understat: No data available for {target_league}")
            return []

        results = []
        for p in players_data:
            # Handle different key naming conventions
            player_name = p.get("player_name", p.get("name", p.get("player", "")))

            # Filter by name if provided
            if name and name.lower() not in player_name.lower():
                continue

            minutes = self._safe_int(p.get("time", p.get("minutes", 0)))
            if minutes < min_minutes:
                continue

            p90 = max(minutes / 90, 1)
            xg = self._safe_float(p.get("xG", p.get("xg", 0)))
            xa = self._safe_float(p.get("xA", p.get("xa", 0)))
            goals = self._safe_int(p.get("goals", 0))
            assists = self._safe_int(p.get("assists", 0))

            player_dict = {
                "player_id": str(p.get("id", "")),
                "name": player_name,
                "team": p.get("team_title", p.get("team", "")),
                "position": p.get("position", ""),
                "minutes_played": minutes,
                "goals": goals,
                "assists": assists,
                "xG": xg,
                "xA": xa,
                "xG_p90": round(xg / p90, 3),
                "xA_p90": round(xa / p90, 3),
                "goals_p90": round(goals / p90, 3),
                "assists_p90": round(assists / p90, 3),
                "shots": self._safe_int(p.get("shots", 0)),
                "key_passes": self._safe_int(p.get("key_passes", 0)),
                "npg": self._safe_int(p.get("npg", 0)),
                "npxG": self._safe_float(p.get("npxG", p.get("npxg", 0))),
                "xGChain": self._safe_float(p.get("xGChain", p.get("xg_chain", 0))),
                "xGBuildup": self._safe_float(p.get("xGBuildup", p.get("xg_buildup", 0))),
                "source": self.source_name,
            }

            results.append(player_dict)

        logger.info(f"Understat: Found {len(results)} players in {target_league}")
        return results

    async def get_player_stats(self, player_name: str, season: str = "2025-26") -> Optional[Dict]:
        """Get xG/xA stats for a specific player across all tracked leagues."""
        for league in UNDERSTAT_LEAGUES:
            results = await self.search_players(name=player_name, league=league)
            if results:
                return results[0]
        return None

    async def get_player_shot_data(self, player_id: int) -> List[Dict]:
        """Get detailed shot-level data for a player."""
        cache_key = f"understat:shots:{player_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached.get("shots", [])

        # Try API endpoint
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.BASE_URL}/player/{player_id}",
                    headers=self.headers,
                    follow_redirects=True,
                )
                response.raise_for_status()
                html = response.text

            shots_data = self._extract_json_var(html, "shotsData")
            if not shots_data:
                shots_data = self._extract_json_var(html, "shots_data")
            if not shots_data:
                return []

            shots = [
                {
                    "x": self._safe_float(s.get("X")) * 105,
                    "y": self._safe_float(s.get("Y")) * 68,
                    "xG": self._safe_float(s.get("xG")),
                    "result": s.get("result", ""),
                    "situation": s.get("situation", ""),
                    "minute": self._safe_int(s.get("minute")),
                }
                for s in shots_data
            ]
            self.cache.set(cache_key, {"shots": shots})
            return shots

        except httpx.HTTPError as e:
            logger.error(f"Understat shot data fetch error: {e}")
            return []
