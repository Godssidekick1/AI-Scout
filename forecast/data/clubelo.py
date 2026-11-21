"""
ClubElo
=======
Team strength ratings from api.clubelo.com: one CSV snapshot per date with
every rated club (Rank, Club, Country, Level, Elo, From, To). We take a
snapshot at the start of each season, so the rating is known before any of
that season's matches (no look-ahead in the backtest).
"""

from __future__ import annotations

import csv
import io
import logging
import time
from typing import Dict, Iterable, Optional

import httpx
from thefuzz import fuzz

from config import settings

logger = logging.getLogger(__name__)

API = "http://api.clubelo.com"
COUNTRY = {"premier_league": "ENG", "la_liga": "ESP", "bundesliga": "GER", "serie_a": "ITA", "ligue_1": "FRA"}

# Understat team name -> ClubElo name, where fuzzy matching is not enough.
ALIASES = {
    "Manchester City": "Man City", "Manchester United": "Man United", "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Forest", "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR", "Sheffield United": "Sheffield United", "Brighton": "Brighton",
    "Atletico Madrid": "Atletico", "Athletic Club": "Bilbao", "Real Betis": "Betis", "Celta Vigo": "Celta",
    "Deportivo La Coruna": "Depor", "Sporting Gijon": "Gijon", "Rayo Vallecano": "Rayo Vallecano",
    "Bayern Munich": "Bayern", "Borussia Dortmund": "Dortmund", "Borussia M.Gladbach": "Gladbach",
    "Bayer Leverkusen": "Leverkusen", "RasenBallsport Leipzig": "RB Leipzig", "Eintracht Frankfurt": "Frankfurt",
    "FC Cologne": "Koeln", "Hertha Berlin": "Hertha", "Fortuna Duesseldorf": "Duesseldorf",
    "Paris Saint Germain": "Paris SG", "Saint-Etienne": "Saint-Etienne", "AC Milan": "Milan",
    "Parma Calcio 1913": "Parma", "SPAL 2013": "Spal",
}


def season_start(season: int) -> str:
    return f"{season}-08-01"


def parse_snapshot(text: str) -> Dict[str, Dict]:
    """CSV text -> {club: {"elo", "country", "level"}}."""
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            out[row["Club"]] = {"elo": float(row["Elo"]), "country": row["Country"], "level": int(row["Level"])}
        except (KeyError, ValueError):
            continue
    return out


class ClubEloUnavailable(RuntimeError):
    """api.clubelo.com could not be reached or kept returning errors."""


def fetch_snapshot(date: str, cache_dir: Optional[str] = None, retries: int = 3) -> Dict[str, Dict]:
    """One dated snapshot, cached on disk (past dates never change).
    Retries transport errors and 5xx with backoff; raises ClubEloUnavailable if they persist."""
    path = settings.cache_path / "clubelo" / f"{date}.csv"
    if path.exists():
        return parse_snapshot(path.read_text())
    reason = ""
    for attempt in range(retries + 1):
        try:
            resp = httpx.get(f"{API}/{date}", timeout=settings.HTTP_TIMEOUT_S, follow_redirects=True)
            if resp.status_code < 400:
                snapshot = parse_snapshot(resp.text)
                if snapshot:   # only cache a real CSV, never an error page
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(resp.text)
                    return snapshot
                reason = "response had no rated clubs"
            elif resp.status_code < 500 and resp.status_code != 429:
                raise ClubEloUnavailable(f"{API}/{date} -> HTTP {resp.status_code}")
            else:
                reason = f"HTTP {resp.status_code}"
        except httpx.HTTPError as e:
            reason = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(1.0 * 2 ** attempt)
    raise ClubEloUnavailable(f"{API}/{date} failed after {retries + 1} attempts ({reason})")


def match_team(team: str, league: str, snapshot: Dict[str, Dict], min_score: int = 80) -> Optional[str]:
    """Understat team name -> ClubElo club name in the same country, or None."""
    alias = ALIASES.get(team)
    if alias and alias in snapshot:
        return alias
    if team in snapshot:
        return team
    country = COUNTRY.get(league)
    best, best_score = None, 0
    for club, info in snapshot.items():
        if country and info["country"] != country:
            continue
        score = max(fuzz.token_set_ratio(team.lower(), club.lower()), fuzz.partial_ratio(team.lower(), club.lower()))
        if score > best_score:
            best, best_score = club, score
    return best if best_score >= min_score else None


def team_elo_table(teams: Iterable[tuple], snapshots: Dict[int, Dict[str, Dict]]) -> Dict[tuple, Optional[float]]:
    """{(team, league, season): elo or None} for every team-season, with unmatched teams logged."""
    out, missing = {}, set()
    for team, league, season in teams:
        snap = snapshots.get(season) or {}
        club = match_team(team, league, snap)
        out[(team, league, season)] = snap[club]["elo"] if club else None
        if club is None:
            missing.add((team, league))
    if missing:
        logger.warning("ClubElo: no rating matched for %d teams, e.g. %s", len(missing), sorted(missing)[:5])
    return out
