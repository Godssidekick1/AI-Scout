"""
Tests for Data Sources
======================
Tests parsing logic with mocked HTTP responses — no real API calls.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from data_sources.data_cache import DataCache
from data_sources.transfermarkt_source import TransfermarktSource
from data_sources.understat_source import UnderstatSource


# ─── Cache Tests ────────────────────────────────────────────────

class TestDataCache:
    def setup_method(self):
        import tempfile
        self.cache = DataCache(cache_dir=tempfile.mkdtemp(), ttl_hours=1)

    def test_set_and_get(self):
        self.cache.set("test_key", {"name": "Salah", "goals": 15})
        result = self.cache.get("test_key")
        assert result is not None
        assert result["name"] == "Salah"

    def test_missing_key_returns_none(self):
        assert self.cache.get("nonexistent") is None

    def test_clear(self):
        self.cache.set("key1", {"data": 1})
        self.cache.set("key2", {"data": 2})
        self.cache.clear()
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None

    def test_stats(self):
        self.cache.set("key1", {"data": 1})
        stats = self.cache.stats()
        assert stats["total_entries"] == 1
        assert stats["valid_entries"] == 1


# ─── Transfermarkt Parsing Tests ────────────────────────────────

class TestTransfermarktParsing:
    def test_parse_market_value_millions(self):
        source = TransfermarktSource()
        assert source._parse_market_value("€25.00m") == 25.0
        assert source._parse_market_value("€80.00m") == 80.0

    def test_parse_market_value_thousands(self):
        source = TransfermarktSource()
        assert source._parse_market_value("€500Th.") == 0.5

    def test_parse_market_value_billions(self):
        source = TransfermarktSource()
        assert source._parse_market_value("€1.20bn") == 1200.0

    def test_parse_market_value_empty(self):
        source = TransfermarktSource()
        assert source._parse_market_value("") == 0.0
        assert source._parse_market_value("-") == 0.0
        assert source._parse_market_value(None) == 0.0


# ─── Understat Parsing Tests ────────────────────────────────────

class TestUnderstatParsing:
    def test_safe_float_conversion(self):
        source = UnderstatSource()
        assert source._safe_float("0.45") == 0.45
        assert source._safe_float(None) == 0.0
        assert source._safe_float("-") == 0.0
        assert source._safe_float("") == 0.0

    def test_safe_int_conversion(self):
        source = UnderstatSource()
        assert source._safe_int("15") == 15
        assert source._safe_int("3.7") == 3
        assert source._safe_int(None) == 0
        assert source._safe_int("-") == 0


# ─── Base Source Tests ──────────────────────────────────────────

class TestBaseSource:
    def test_safe_conversions(self):
        """Test type conversion helpers from BaseSource."""
        from data_sources.base_source import BaseSource

        class DummySource(BaseSource):
            async def search_players(self, **kw): return []
            async def get_player_stats(self, name, season="2025-26"): return None

        source = DummySource()
        assert source._safe_float("1.5") == 1.5
        assert source._safe_float(None) == 0.0
        assert source._safe_int("10") == 10
        assert source._safe_int(None, default=5) == 5
