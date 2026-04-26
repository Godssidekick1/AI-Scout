"""
EA FC 26 Live Ratings — FUTWIZ Source
=====================================
Scrapes FUTWIZ.com for current EA FC 26 player ratings and attributes.
These reflect live form updates throughout the season.
No API key needed.
"""

import re
import httpx
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from thefuzz import fuzz
import logging
from data_sources.base_source import BaseSource
from data_sources.data_cache import DataCache

logger = logging.getLogger(__name__)


class EAFCRatingsSource(BaseSource):
    """
    Scrapes FUTWIZ for EA FC 26 player ratings.
    Includes live form ratings that change during the season.
    """

    source_name = "ea_fc_26"
    BASE_URL = "https://www.futwiz.com"

    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache(ttl_hours=72)  # 3-day cache
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    async def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a FUTWIZ page."""
        cache_key = f"futwiz:page:{url}"
        cached = self.cache.get(cache_key)
        if cached:
            return BeautifulSoup(cached.get("html", ""), "lxml")

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                html = response.text
                self.cache.set(cache_key, {"html": html})
                return BeautifulSoup(html, "lxml")

        except httpx.HTTPError as e:
            logger.error(f"FUTWIZ fetch error for {url}: {e}")
            return None

    async def search_players(
        self,
        name: Optional[str] = None,
        position: Optional[str] = None,
        league: Optional[str] = None,
        max_age: Optional[int] = None,
        min_minutes: int = 0,
    ) -> List[Dict]:
        """Search for players on FUTWIZ."""
        if not name:
            return []

        search_url = f"{self.BASE_URL}/fc26/players?name={name.replace(' ', '+')}"
        soup = await self._fetch_page(search_url)
        if not soup:
            return []

        results = []
        # FUTWIZ player cards in search results
        player_cards = soup.select("tr.player-row, div.player-card, table.table-striped tbody tr")

        for card in player_cards[:5]:
            try:
                # Try to extract player link and name
                name_el = card.select_one("a.player-link, td a, a[href*='/player/']")
                if not name_el:
                    continue

                player_name = name_el.get_text(strip=True)
                player_url = name_el.get("href", "")

                # Rating
                rating_el = card.select_one("span.rating, td.rating, span.otherversion26-txt")
                overall = self._safe_int(rating_el.get_text(strip=True)) if rating_el else 0

                # Position
                pos_el = card.select_one("span.position, td.position, div.pcdisplay-pos")
                pos = pos_el.get_text(strip=True) if pos_el else ""

                results.append({
                    "name": player_name,
                    "overall_rating": overall,
                    "position": pos,
                    "profile_url": f"{self.BASE_URL}{player_url}" if player_url and not player_url.startswith("http") else player_url,
                    "source": self.source_name,
                })

            except Exception as e:
                logger.debug(f"Error parsing FUTWIZ card: {e}")
                continue

        return results

    async def get_player_stats(self, player_name: str, season: str = "2025-26") -> Optional[Dict]:
        """Get EA FC 26 ratings for a specific player."""
        results = await self.search_players(name=player_name)
        if not results:
            return None

        # Fuzzy match to find the best result
        best_match = None
        best_score = 0
        for r in results:
            score = fuzz.ratio(player_name.lower(), r["name"].lower())
            if score > best_score:
                best_score = score
                best_match = r

        if not best_match or best_score < 60:
            return None

        # Get detailed attributes from player page
        if best_match.get("profile_url"):
            detail = await self._get_player_attributes(best_match["profile_url"])
            if detail:
                best_match.update(detail)

        return best_match

    async def _get_player_attributes(self, profile_url: str) -> Optional[Dict]:
        """Extract detailed attributes from a player's FUTWIZ profile page."""
        soup = await self._fetch_page(profile_url)
        if not soup:
            return None

        attrs = {}

        # Extract the 6 core attribute stats
        stat_names = ["pace", "shooting", "passing", "dribbling", "defending", "physical"]
        stat_elements = soup.select("div.card-stat span.card-stat-val, span.pcdisplay-ovr")

        for i, stat_name in enumerate(stat_names):
            if i < len(stat_elements):
                attrs[stat_name] = self._safe_int(stat_elements[i].get_text(strip=True))

        # Try to get work rates
        wr_el = soup.select_one("div:-soup-contains('Work Rate'), span:-soup-contains('Work Rate')")
        if wr_el:
            wr_text = wr_el.get_text(strip=True)
            wr_parts = wr_text.replace("Work Rate", "").strip().split("/")
            if len(wr_parts) == 2:
                attrs["work_rate_attack"] = wr_parts[0].strip()
                attrs["work_rate_defense"] = wr_parts[1].strip()

        # Weak foot and skill moves
        wf_el = soup.select_one("span:-soup-contains('Weak Foot'), div:-soup-contains('Weak Foot')")
        if wf_el:
            wf_text = re.search(r"(\d)", wf_el.get_text())
            if wf_text:
                attrs["weak_foot"] = int(wf_text.group(1))

        sm_el = soup.select_one("span:-soup-contains('Skill Moves'), div:-soup-contains('Skill Moves')")
        if sm_el:
            sm_text = re.search(r"(\d)", sm_el.get_text())
            if sm_text:
                attrs["skill_moves"] = int(sm_text.group(1))

        return attrs if attrs else None
