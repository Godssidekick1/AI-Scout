"""
Transfermarkt open dataset
==========================
dcaribou/transfermarkt-datasets (CC0): player profiles (birth date, position,
foot, contract), market-value history and transfers. Downloaded once as
gzipped CSVs and cached on disk. The dataset is frozen at 6 July 2026, which
is fine for fitting on past seasons; live values come from the scraper in
data_sources/transfermarkt_source.py.

We also use it to give Understat players what Understat lacks: an age, a
detailed position, where each one actually lined up recently (`game_lineups`:
the position of every start), and his current club, which is newer than the
last Understat season. `match_players` links the two by name and club.
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
TABLES = ("players", "player_valuations", "transfers", "clubs", "game_lineups")
LINEUP_COLUMNS = ["date", "player_id", "type", "position"]


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


def load(table: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    path = data_dir() / f"{table}.csv.gz"
    if not path.exists():
        download([table])
    return pd.read_csv(path, low_memory=False, usecols=usecols)


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


# ── recent positions and current clubs ─────────────────────────

def position_shares(lineups: pd.DataFrame, since: str, min_starts: int = 5) -> Dict[int, Dict[str, float]]:
    """{player_id: {position: share of his starts}} for starts on or after `since`. Transfermarkt
    lists one main position per player; this is where he actually played. Players with fewer than
    `min_starts` starts are left out (their main position is used instead)."""
    from forecast.config import POSITION_SHORT
    d = lineups[(lineups["type"] == "starting_lineup") & (lineups["date"].astype(str) >= since)
                & lineups["position"].isin(POSITION_SHORT)]          # drops vague "Defender", "Attack", keepers
    counts = d.groupby(["player_id", "position"]).size()
    out: Dict[int, Dict[str, float]] = {}
    for pid, s in counts.groupby(level=0):
        total = int(s.sum())
        if total >= min_starts:
            out[int(pid)] = {pos: round(c / total, 3) for (_, pos), c in s.items()}
    return out


def team_club_ids(understat: pd.DataFrame, us_to_tm: Dict[str, Optional[int]], valuations: pd.DataFrame,
                  season: int) -> Dict[str, int]:
    """Understat team name -> Transfermarkt club id, by majority vote of that season's matched players'
    clubs in the value history (names differ: 'Bayern Munich' vs 'FC Bayern München')."""
    lo, hi = f"{season}-07-01", f"{season + 1}-06-30"
    v = valuations[(valuations["date"].astype(str) >= lo) & (valuations["date"].astype(str) <= hi)]
    club_of = v.groupby("player_id")["current_club_id"].agg(lambda x: x.mode().iloc[0])
    rows = understat[understat["season"] == season]
    votes: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for us_id, team in zip(rows["us_id"], rows["team"]):
        tm = us_to_tm.get(us_id)
        if tm is not None and tm in club_of.index:
            votes[team][int(club_of[tm])] += 1
    return {team: max(c, key=c.get) for team, c in votes.items() if c}


def player_profiles(ps: pd.DataFrame, players: pd.DataFrame, valuations: pd.DataFrame,
                    lineups: Optional[pd.DataFrame]) -> pd.DataFrame:
    """One row per Understat player: where he lined up recently and where he plays now.

    current_club / current_league  Transfermarkt's current club (newer than Understat's last season);
                                   the league key for big-five clubs, else the Transfermarkt competition id
    moved                          his current club is not the team of his last Understat season
    active                         he played in a competition Transfermarkt tracks in its latest season
                                   (False: retired, or moved somewhere the data does not cover)
    position_shares                {Transfermarkt position: share of his starts} over the last two seasons
    """
    from forecast.config import TM_COMPETITIONS
    tm_league = {v: k for k, v in TM_COMPETITIONS.items()}
    latest = ps.sort_values("season").groupby("us_id").tail(1)
    us_to_tm = {u: (int(t) if pd.notna(t) else None) for u, t in zip(latest["us_id"], latest["tm_id"])}
    clubs: Dict[str, int] = {}
    for season in sorted(latest["season"].unique()):
        clubs.update(team_club_ids(ps, us_to_tm, valuations, int(season)))
    team_of_club = {c: t for t, c in clubs.items()}
    prof = players.set_index("player_id")
    newest = int(pd.to_numeric(prof["last_season"], errors="coerce").max())
    shares = position_shares(lineups, f"{newest - 1}-07-01") if lineups is not None else {}
    rows = []
    for us_id, team, tm in zip(latest["us_id"], latest["team"], latest["tm_id"]):
        if pd.isna(tm) or int(tm) not in prof.index:
            continue
        p = prof.loc[int(tm)]
        club_id = int(p["current_club_id"]) if pd.notna(p["current_club_id"]) else None
        moved = club_id is not None and team in clubs and clubs[team] != club_id
        comp = p.get("current_club_domestic_competition_id")
        rows.append({"us_id": us_id,
                     "current_club": team_of_club.get(club_id, p["current_club_name"]) if moved else team,
                     "current_league": tm_league.get(comp, comp if isinstance(comp, str) else None),
                     "moved": bool(moved),
                     "active": bool(pd.notna(p["last_season"]) and int(p["last_season"]) >= newest),
                     "position_shares": shares.get(int(tm))})
    return pd.DataFrame(rows).set_index("us_id") if rows else pd.DataFrame()
