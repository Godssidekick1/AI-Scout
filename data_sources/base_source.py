"""
Base Data Source
================
Abstract interface that all data sources implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseSource(ABC):
    """
    Base class for all football data sources.
    Every source must implement search_players() and get_player_stats().
    """

    source_name: str = "base"

    @abstractmethod
    async def search_players(
        self,
        name: Optional[str] = None,
        position: Optional[str] = None,
        league: Optional[str] = None,
        max_age: Optional[int] = None,
        min_minutes: int = 900,
    ) -> List[Dict]:
        """
        Search for players matching the given filters.

        Returns:
            List of player dictionaries with at least 'name', 'team', 'position'.
        """
        pass

    @abstractmethod
    async def get_player_stats(self, player_name: str, season: str = "2025-26") -> Optional[Dict]:
        """
        Get detailed statistics for a specific player.

        Returns:
            Dictionary of player stats, or None if not found.
        """
        pass

    def _safe_float(self, value, default: float = 0.0) -> float:
        """Safely convert a value to float."""
        try:
            if value is None or value == "" or value == "-":
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    def _safe_int(self, value, default: int = 0) -> int:
        """Safely convert a value to int."""
        try:
            if value is None or value == "" or value == "-":
                return default
            return int(float(value))
        except (ValueError, TypeError):
            return default
