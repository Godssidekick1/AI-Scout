"""
Transfermarkt open dataset
==========================
dcaribou/transfermarkt-datasets (CC0): player profiles (birth date, position,
foot, contract), market-value history and transfers. Downloaded once as
gzipped CSVs and cached on disk. The dataset is frozen at 6 July 2026, which
is fine for fitting on past seasons; live values come from the scraper in
data_sources/transfermarkt_source.py.

We also use it to give Understat players what Understat lacks: an age and a
detailed position. `match_players` links the two by name and club.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import httpx
import pandas as pd
from thefuzz import fuzz

from config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = ("players", "player_valuations", "transfers", "clubs")


def data_dir() -> Path:
    return settings.cache_path / "transfermarkt"


def download(tables: Iterable[str] = TABLES, force: bool = False) -> Dict[str, Path]:
    """Download the CSVs we use (skips files already on disk)."""
    paths = {}
    for table in tables:
        path = data_dir() / f"{table}.csv.gz"
        if force or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s", table)
            with httpx.stream("GET", f"{BASE_URL}/{table}.csv.gz", timeout=120, follow_redirects=True) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
        paths[table] = path
    return paths


def load(table: str) -> pd.DataFrame:
    path = data_dir() / f"{table}.csv.gz"
    if not path.exists():
        download([table])
    return pd.read_csv(path, low_memory=False)


# ── name matching ──────────────────────────────────────────────

def normalise(name: str) -> str:
    """'Kylian Mbappé-Lottin' -> 'kylian mbappe lottin'."""
    text = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]+", " ", text.lower()).strip()


def _tokens(name: str) -> List[str]:
    return [t for t in normalise(name).split() if len(t) >= 3]


class PlayerMatcher:
    """Links Understat players to Transfermarkt players.

    Candidates share at least one name token (so we never fuzzy-match against all
    50,000 players). A candidate scores on name similarity, plus a bonus if his
    Transfermarkt club history includes the Understat team in that season."""

    def __init__(self, players: pd.DataFrame, valuations: Optional[pd.DataFrame] = None):
        self.players = players.set_index("player_id")
        self.index: Dict[str, set] = defaultdict(set)
        for pid, name in zip(players["player_id"], players["name"]):
            for tok in _tokens(name):
                self.index[tok].add(pid)
        # (player_id) -> list of (date, club name) from the value history
        self.clubs: Dict[int, List[tuple]] = defaultdict(list)
        if valuations is not None and len(valuations):
            for pid, date, club in zip(valuations["player_id"], valuations["date"], valuations["current_club_name"]):
                if isinstance(club, str):
                    self.clubs[pid].append((str(date), club))

    def _club_bonus(self, pid: int, team: str, season: int) -> int:
        lo, hi = f"{season}-07-01", f"{season + 1}-06-30"
        for date, club in self.clubs.get(pid, ()):
            if lo <= date <= hi and fuzz.token_set_ratio(team.lower(), club.lower()) >= 80:
                return 15
        return 0

    def match(self, name: str, team: str = "", season: Optional[int] = None, min_score: int = 85) -> Optional[int]:
        candidates = set().union(*(self.index.get(t, set()) for t in _tokens(name))) if _tokens(name) else set()
        best, best_score = None, 0
        target = normalise(name)
        for pid in candidates:
            other = normalise(self.players.at[pid, "name"])
            bonus = self._club_bonus(pid, team, season) if team and season is not None else 0
            score = fuzz.token_sort_ratio(target, other)
            # Understat sometimes lists only a known-as name ('Rodri', 'Kepa'): token_set covers that, but a
            # one-word name is a subset of thousands of others ('Alex', 'Max'), so only trust it when the
            # club history agrees.
            if bonus:
                score = max(score, fuzz.token_set_ratio(target, other) - 5)
            score += bonus
            if score > best_score:
                best, best_score = pid, score
        return best if best_score >= min_score else None


def match_players(understat: pd.DataFrame, players: pd.DataFrame,
                  valuations: Optional[pd.DataFrame] = None) -> Dict[str, Optional[int]]:
    """{understat id: transfermarkt player_id or None}, matching each player on his latest season."""
    matcher = PlayerMatcher(players, valuations)
    latest = understat.sort_values("season").groupby("us_id").tail(1)
    out = {}
    for us_id, name, team, season in zip(latest["us_id"], latest["name"], latest["team"], latest["season"]):
        out[us_id] = matcher.match(name, team, int(season))
    matched = sum(v is not None for v in out.values())
    logger.info("Matched %d of %d Understat players to Transfermarkt", matched, len(out))
    return out


def value_at(valuations: pd.DataFrame, when: str) -> Dict[int, float]:
    """Each player's most recent market value on or before a date, in euros."""
    v = valuations[valuations["date"].astype(str) <= when]
    last = v.sort_values("date").groupby("player_id").tail(1)
    return dict(zip(last["player_id"], last["market_value_in_eur"].astype(float)))
