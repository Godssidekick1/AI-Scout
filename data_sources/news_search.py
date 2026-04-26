"""
News Search Data Source
=======================
Uses Serper.dev to search the web for player news, sentiment, and character intelligence.
Free tier: 2,500 searches/month.
"""

import httpx
from typing import Dict, List, Optional
import logging
from data_sources.base_source import BaseSource
from data_sources.data_cache import DataCache
from config import settings

logger = logging.getLogger(__name__)


class NewsSearchSource(BaseSource):
    """
    Searches the web for player-related news using Serper.dev.
    Used by the Background Checker agent.
    """

    source_name = "news_search"
    API_URL = "https://google.serper.dev/search"

    def __init__(self, cache: Optional[DataCache] = None):
        self.api_key = settings.SERPER_API_KEY
        self.cache = cache or DataCache(ttl_hours=12)  # 12-hour cache for news

    async def search_players(self, name: Optional[str] = None, **kwargs) -> List[Dict]:
        """Not used for news — use search_news() instead."""
        return []

    async def get_player_stats(self, player_name: str, season: str = "2025-26") -> Optional[Dict]:
        """Not used for news — use search_news() instead."""
        return None

    async def search_news(
        self,
        player_name: str,
        query_type: str = "general",    # "general", "character", "injury", "transfer"
    ) -> List[Dict]:
        """
        Search for news about a player.

        Args:
            player_name: The player's name
            query_type: Type of news to search for

        Returns:
            List of news article dictionaries
        """
        query_templates = {
            "general": f"{player_name} football latest news 2026",
            "character": f"{player_name} football attitude discipline behavior controversy",
            "injury": f"{player_name} football injury update fitness 2025 2026",
            "transfer": f"{player_name} transfer news rumour contract 2026",
        }

        query = query_templates.get(query_type, query_templates["general"])

        cache_key = f"news:{player_name}:{query_type}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached.get("results", [])

        if not self.api_key:
            logger.warning("Serper API key not configured. Skipping news search.")
            return []

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    self.API_URL,
                    headers={
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "q": query,
                        "num": 5,
                    },
                )
                response.raise_for_status()
                data = response.json()

        except httpx.HTTPError as e:
            logger.error(f"Serper API error: {e}")
            return []

        # Parse results
        results = []
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", ""),
                "date": item.get("date", ""),
                "source": item.get("source", ""),
            })

        self.cache.set(cache_key, {"results": results})
        logger.info(f"News search: Found {len(results)} articles for '{player_name}' ({query_type})")
        return results

    async def get_sentiment_summary(self, player_name: str) -> Dict:
        """
        Get a combined sentiment overview for a player.
        Searches for character, injury, and transfer news.
        """
        character_news = await self.search_news(player_name, "character")
        injury_news = await self.search_news(player_name, "injury")
        transfer_news = await self.search_news(player_name, "transfer")

        return {
            "player_name": player_name,
            "character_articles": character_news,
            "injury_articles": injury_news,
            "transfer_articles": transfer_news,
            "total_articles": len(character_news) + len(injury_news) + len(transfer_news),
        }
