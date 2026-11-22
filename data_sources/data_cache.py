"""
Data Cache
==========
SQLite-backed caching layer to avoid wasting free API calls.
Stores responses with a time-to-live (TTL).
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DataCache:
    """
    Simple SQLite cache with TTL support.

    Usage:
        cache = DataCache(cache_dir=".cache", ttl_hours=24)
        data = cache.get("player:salah:stats")
        if data is None:
            data = fetch_from_api(...)
            cache.set("player:salah:stats", data)
    """

    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24):
        self.ttl_seconds = ttl_hours * 3600
        self.db_path = Path(cache_dir) / "scout_cache.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create the cache table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()

    def get(self, key: str, stale_ok: bool = False) -> Optional[dict]:
        """
        Get a cached value by key.
        Returns None if not found or expired. An expired entry is kept on disk, not deleted: when
        the source is down (Understat was, for weeks), `stale_ok=True` returns the last good copy.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value, timestamp FROM cache WHERE key = ?", (key,)
            )
            row = cursor.fetchone()

        if row is None:
            return None

        value_json, timestamp = row
        age = time.time() - timestamp

        if age > self.ttl_seconds and not stale_ok:
            logger.debug(f"Cache EXPIRED: {key} (age: {age/3600:.1f}h)")
            return None

        logger.debug(f"Cache HIT: {key} (age: {age/3600:.1f}h)")
        return json.loads(value_json)

    def set(self, key: str, value: dict):
        """Store a value in the cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, timestamp) VALUES (?, ?, ?)",
                (key, json.dumps(value, default=str), time.time()),
            )
            conn.commit()
        logger.debug(f"Cache SET: {key}")

    def delete(self, key: str):
        """Remove a key from the cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear(self):
        """Clear the entire cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
        logger.info("Cache cleared")

    def stats(self) -> dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            total = cursor.fetchone()[0]
            cursor = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE ? - timestamp < ?",
                (time.time(), self.ttl_seconds),
            )
            valid = cursor.fetchone()[0]
        return {"total_entries": total, "valid_entries": valid, "expired": total - valid}
