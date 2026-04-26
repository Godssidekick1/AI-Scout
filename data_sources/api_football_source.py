"""
API-Football Data Source
========================
Fetches player profiles and statistics from API-Football (RapidAPI).
Free tier: 100 requests/day.
"""

import httpx
from typing import Dict, List, Optional
import logging
from data_sources.base_source import BaseSource
from data_sources.data_cache import DataCache
from config import settings

logger = logging.getLogger(__name__)

# League IDs in API-Football
LEAGUE_IDS = {
    "premier_league": 39,
    "la_liga": 140,
    "bundesliga": 78,
    "serie_a": 135,
    "ligue_1": 61,
    "eredivisie": 88,
    "primeira_liga": 94,
    "championship": 40,
    "mls": 253,
    "saudi_pro_league": 307,
}


class APIFootballSource(BaseSource):
    """
    Fetches data from API-Football via api-sports.io.
    Free tier: 100 requests/day.
    Get your key at: https://dashboard.api-football.com (free registration)
    """

    source_name = "api_football"
    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(self, cache: Optional[DataCache] = None):
        self.api_key = settings.API_FOOTBALL_KEY
        self.cache = cache or DataCache()
        self.headers = {
            "x-apisports-key": self.api_key,
        }

    async def _request(self, endpoint: str, params: dict) -> Optional[dict]:
        """Make a cached API request."""
        cache_key = f"apifootball:{endpoint}:{str(sorted(params.items()))}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        if not self.api_key:
            logger.warning("API-Football key not configured. Skipping API call.")
            return None

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.BASE_URL}/{endpoint}",
                    headers=self.headers,
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                if data.get("errors"):
                    logger.error(f"API-Football error: {data['errors']}")
                    return None

                self.cache.set(cache_key, data)
                return data

        except httpx.HTTPError as e:
            logger.error(f"API-Football HTTP error: {e}")
            return None

    async def search_players(
        self,
        name: Optional[str] = None,
        position: Optional[str] = None,
        league: Optional[str] = None,
        max_age: Optional[int] = None,
        min_minutes: int = 900,
    ) -> List[Dict]:
        """Search for players by name and/or league."""
        params = {"season": 2025}

        if name:
            params["search"] = name
        if league and league.lower() in LEAGUE_IDS:
            params["league"] = LEAGUE_IDS[league.lower()]

        data = await self._request("players", params)
        if not data or "response" not in data:
            return []

        results = []
        for item in data["response"]:
            player = item.get("player", {})
            stats_list = item.get("statistics", [])

            if not stats_list:
                continue

            # Use the first (most recent) stats entry
            stats = stats_list[0]
            games = stats.get("games", {})
            goals_data = stats.get("goals", {})
            passes = stats.get("passes", {})
            tackles = stats.get("tackles", {})
            team_info = stats.get("team", {})
            league_info = stats.get("league", {})

            minutes = self._safe_int(games.get("minutes"))
            if minutes < min_minutes:
                continue

            # Filter by max age
            age = self._safe_int(player.get("age"))
            if max_age and age > max_age:
                continue

            p90 = max(minutes / 90, 1)

            player_dict = {
                "player_id": str(player.get("id", "")),
                "name": player.get("name", ""),
                "age": age,
                "nationality": player.get("nationality", ""),
                "team": team_info.get("name", ""),
                "league": league_info.get("name", ""),
                "position": games.get("position", ""),
                "minutes_played": minutes,
                "goals_p90": round(self._safe_float(goals_data.get("total")) / p90, 3),
                "assists_p90": round(self._safe_float(goals_data.get("assists")) / p90, 3),
                "tackles_p90": round(self._safe_float(tackles.get("total")) / p90, 3),
                "interceptions_p90": round(self._safe_float(tackles.get("interceptions")) / p90, 3),
                "yellow_cards": self._safe_int(stats.get("cards", {}).get("yellow")),
                "red_cards": self._safe_int(stats.get("cards", {}).get("red")),
                "source": self.source_name,
            }

            results.append(player_dict)

        logger.info(f"API-Football: Found {len(results)} players")
        return results

    async def get_player_stats(self, player_name: str, season: str = "2025-26") -> Optional[Dict]:
        """Get stats for a specific player by name."""
        results = await self.search_players(name=player_name)
        if results:
            return results[0]
        return None

    async def get_player_by_id(self, player_id: int, season: int = 2025) -> Optional[Dict]:
        """Get detailed stats for a player by their API-Football ID."""
        data = await self._request("players", {"id": player_id, "season": season})
        if data and data.get("response"):
            item = data["response"][0]
            return {
                "player": item.get("player", {}),
                "statistics": item.get("statistics", []),
            }
        return None
