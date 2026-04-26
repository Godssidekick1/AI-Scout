"""
Transfermarkt Data Source
=========================
Scrapes Transfermarkt for market values, contract info, and injury history.
No API key needed — uses browser-like headers to bypass basic protections.
"""

import re
import httpx
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import logging
from data_sources.base_source import BaseSource
from data_sources.data_cache import DataCache

logger = logging.getLogger(__name__)


class TransfermarktSource(BaseSource):
    """
    Scrapes Transfermarkt for financial and injury data.
    Uses browser-like headers to access the site.
    """

    source_name = "transfermarkt"
    BASE_URL = "https://www.transfermarkt.com"

    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache(ttl_hours=168)  # 7-day cache for market data
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.transfermarkt.com/",
        }

    async def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a Transfermarkt page."""
        cache_key = f"transfermarkt:page:{url}"
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
            logger.error(f"Transfermarkt fetch error for {url}: {e}")
            return None

    async def search_players(
        self,
        name: Optional[str] = None,
        position: Optional[str] = None,
        league: Optional[str] = None,
        max_age: Optional[int] = None,
        min_minutes: int = 900,
    ) -> List[Dict]:
        """Search for players on Transfermarkt by name."""
        if not name:
            return []

        search_url = f"{self.BASE_URL}/schnellsuche/ergebnis/schnellsuche?query={name.replace(' ', '+')}"
        soup = await self._fetch_page(search_url)
        if not soup:
            return []

        results = []
        # Find player search result rows
        player_rows = soup.select("table.items tbody tr")

        for row in player_rows[:5]:  # Limit to top 5 results
            try:
                name_cell = row.select_one("td.hauptlink a")
                if not name_cell:
                    continue

                player_name = name_cell.get_text(strip=True)
                player_url = name_cell.get("href", "")

                # Extract age, position, team, market value from the row
                cells = row.select("td")
                position_text = ""
                team_text = ""
                market_value = ""

                for cell in cells:
                    text = cell.get_text(strip=True)
                    if cell.get("class") and "posrela" in " ".join(cell.get("class", [])):
                        pos_el = cell.select_one("tr:nth-of-type(2) td")
                        if pos_el:
                            position_text = pos_el.get_text(strip=True)
                    if "rechts" in " ".join(cell.get("class", [])):
                        market_value = text

                results.append({
                    "name": player_name,
                    "position": position_text,
                    "team": team_text,
                    "market_value_raw": market_value,
                    "market_value_eur": self._parse_market_value(market_value),
                    "profile_url": f"{self.BASE_URL}{player_url}" if player_url else "",
                    "source": self.source_name,
                })

            except Exception as e:
                logger.debug(f"Error parsing Transfermarkt row: {e}")
                continue

        logger.info(f"Transfermarkt: Found {len(results)} players for '{name}'")
        return results

    async def get_player_stats(self, player_name: str, season: str = "2025-26") -> Optional[Dict]:
        """Get market value and contract data for a player."""
        players = await self.search_players(name=player_name)
        if not players:
            return None

        player = players[0]
        profile_url = player.get("profile_url")
        if not profile_url:
            return player

        # Get profile page for more details
        detail = await self._get_player_profile(profile_url)
        if detail:
            player.update(detail)

        # Get injury history
        injuries = await self._get_injury_history(profile_url)
        player["injury_history"] = injuries

        return player

    async def _get_player_profile(self, profile_url: str) -> Optional[Dict]:
        """Extract detailed profile info from a player's page."""
        soup = await self._fetch_page(profile_url)
        if not soup:
            return None

        data = {}

        # Extract info from player data table
        info_table = soup.select_one("div.info-table")
        if info_table:
            rows = info_table.select("span")
            for i in range(0, len(rows) - 1, 2):
                label = rows[i].get_text(strip=True).lower()
                value = rows[i + 1].get_text(strip=True)

                if "contract" in label and "expires" in label:
                    data["contract_end_date"] = value
                elif "agent" in label:
                    data["agent"] = value
                elif "market value" in label:
                    data["market_value_eur"] = self._parse_market_value(value)

        # Try to find contract end from different element
        contract_el = soup.select_one("span.data-header__content:-soup-contains('Contract expires')")
        if contract_el:
            data["contract_end_date"] = contract_el.get_text(strip=True)

        return data if data else None

    async def _get_injury_history(self, profile_url: str) -> List[Dict]:
        """Get injury history from a player's injury page."""
        # Convert profile URL to injury URL
        injury_url = profile_url.replace("/profil/", "/verletzungen/")
        soup = await self._fetch_page(injury_url)
        if not soup:
            return []

        injuries = []
        injury_table = soup.select_one("table.items")
        if not injury_table:
            return []

        rows = injury_table.select("tbody tr")
        for row in rows[:10]:  # Last 10 injuries
            cells = row.select("td")
            if len(cells) < 4:
                continue

            try:
                injuries.append({
                    "injury": cells[1].get_text(strip=True) if len(cells) > 1 else "",
                    "from_date": cells[2].get_text(strip=True) if len(cells) > 2 else "",
                    "to_date": cells[3].get_text(strip=True) if len(cells) > 3 else "",
                    "days_missed": self._safe_int(
                        cells[4].get_text(strip=True).replace(" days", "")
                    ) if len(cells) > 4 else 0,
                })
            except Exception:
                continue

        return injuries

    def _parse_market_value(self, value_str: str) -> float:
        """Parse Transfermarkt market value string to millions EUR."""
        if not value_str or value_str == "-":
            return 0.0

        value_str = value_str.replace("€", "").strip()

        try:
            if "bn" in value_str.lower():
                return float(re.sub(r"[^\d.]", "", value_str)) * 1000
            elif "m" in value_str.lower():
                return float(re.sub(r"[^\d.]", "", value_str))
            elif "k" in value_str.lower() or "Th." in value_str:
                return float(re.sub(r"[^\d.]", "", value_str)) / 1000
            else:
                return float(re.sub(r"[^\d.]", "", value_str))
        except (ValueError, TypeError):
            return 0.0
