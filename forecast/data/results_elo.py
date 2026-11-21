"""
Team Elo from match results
===========================
Fallback for when api.clubelo.com is down: our own Elo ratings computed from
domestic league results (football-data.co.uk, free CSVs back to the 1990s).

Each league is rated on its own, so ratings are comparable within a league but
not across leagues; the model's league effects carry the between-league gap.
That makes dElo for a player who changes league less informative than with
ClubElo (which links leagues through European games), so prefer ClubElo when it
is up. Snapshots come out in the same shape as clubelo.parse_snapshot, so
clubelo.match_team and team_elo_table work on them unchanged.

Method: classic football Elo. K = 20, home advantage 65 points, a goal-difference
multiplier, ratings carried across seasons; a promoted team enters at the mean
rating of the teams it replaced.
"""

from __future__ import annotations

import io
import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import httpx
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

BASE = "https://football-data.co.uk/mmz4281/{code}/{div}.csv"
DIVISION = {"premier_league": "E0", "la_liga": "SP1", "bundesliga": "D1", "serie_a": "I1", "ligue_1": "F1"}
COUNTRY = {"premier_league": "ENG", "la_liga": "ESP", "bundesliga": "GER", "serie_a": "ITA", "ligue_1": "FRA"}
WARMUP_SEASONS = 5          # seasons rated before the first one we need, so early ratings have settled
K, HOME_ADV, START, PROMOTED_FIRST = 20.0, 65.0, 1500.0, 1400.0

# football-data name -> the name Understat uses, where fuzzy matching would miss it.
ALIASES = {
    "Man City": "Manchester City", "Man United": "Manchester United", "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers", "Newcastle": "Newcastle United", "West Brom": "West Bromwich Albion",
    "QPR": "Queens Park Rangers", "Sheffield United": "Sheffield United", "Spurs": "Tottenham",
    "Ath Madrid": "Atletico Madrid", "Ath Bilbao": "Athletic Club", "Betis": "Real Betis", "Celta": "Celta Vigo",
    "La Coruna": "Deportivo La Coruna", "Sp Gijon": "Sporting Gijon", "Vallecano": "Rayo Vallecano",
    "Sociedad": "Real Sociedad", "Espanol": "Espanyol", "Alaves": "Alaves", "Leganes": "Leganes",
    "Bayern Munich": "Bayern Munich", "Dortmund": "Borussia Dortmund", "M'gladbach": "Borussia M.Gladbach",
    "Leverkusen": "Bayer Leverkusen", "RB Leipzig": "RasenBallsport Leipzig", "Ein Frankfurt": "Eintracht Frankfurt",
    "FC Koln": "FC Cologne", "Hertha": "Hertha Berlin", "Fortuna Dusseldorf": "Fortuna Duesseldorf",
    "Paris SG": "Paris Saint Germain", "St Etienne": "Saint-Etienne", "Milan": "AC Milan", "Inter": "Inter",
    "Parma": "Parma Calcio 1913", "Spal": "SPAL 2013", "Mainz": "Mainz 05", "Schalke 04": "Schalke 04",
    "Hamburg": "Hamburger SV", "Greuther Furth": "Greuther Fuerth", "Nurnberg": "Nuernberg",
    "Hannover": "Hannover 96", "Bielefeld": "Arminia Bielefeld", "St Pauli": "St. Pauli",
    "Heidenheim": "FC Heidenheim", "Holstein Kiel": "Holstein Kiel", "Clermont": "Clermont Foot",
    "Ajaccio GFCO": "GFC Ajaccio",
}


def season_code(season: int) -> str:
    """2014 -> '1415'."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def load_results(league: str, season: int) -> pd.DataFrame:
    """One league-season of results (date, home, away, home goals, away goals), cached on disk."""
    div = DIVISION[league]
    path = settings.cache_path / "football_data" / f"{div}_{season_code(season)}.csv"
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        resp = httpx.get(BASE.format(code=season_code(season), div=div), timeout=settings.HTTP_TIMEOUT_S,
                         follow_redirects=True)
        resp.raise_for_status()
        text = resp.content.decode("utf-8-sig", errors="replace")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    df = pd.read_csv(io.StringIO(text), usecols=lambda c: c in {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"},
                     on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    return df.sort_values("Date", kind="stable")


def _gd_multiplier(gd: int) -> float:
    gd = abs(gd)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8


def rate_league(league: str, seasons: Iterable[int]) -> Dict[int, Dict[str, float]]:
    """{season: {team: rating before that season's first match}} for one league."""
    seasons = sorted(seasons)
    ratings: Dict[str, float] = {}
    out: Dict[int, Dict[str, float]] = {}
    prev_teams: set = set()
    for season in range(seasons[0] - WARMUP_SEASONS, seasons[-1] + 1):
        try:
            games = load_results(league, season)
        except httpx.HTTPError as e:
            logger.warning("football-data: %s %s unavailable (%s)", league, season, e)
            continue
        teams = set(games["HomeTeam"]) | set(games["AwayTeam"])
        if not teams:
            continue
        if prev_teams:
            # Promoted sides enter at the mean final rating of the sides they replaced.
            relegated = [ratings[t] for t in prev_teams - teams]
            entry = sum(relegated) / len(relegated) if relegated else PROMOTED_FIRST
            for team in teams - prev_teams:
                ratings[team] = entry
        else:
            for team in teams:
                ratings.setdefault(team, START)
        out[season] = {t: ratings[t] for t in teams}
        for h, a, hg, ag in zip(games["HomeTeam"], games["AwayTeam"], games["FTHG"], games["FTAG"]):
            hg, ag = int(hg), int(ag)
            exp_h = 1 / (1 + 10 ** ((ratings[a] - ratings[h] - HOME_ADV) / 400))
            res_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            delta = K * _gd_multiplier(hg - ag) * (res_h - exp_h)
            ratings[h] += delta
            ratings[a] -= delta
        prev_teams = teams
    return {s: r for s, r in out.items() if s in set(seasons)}


def snapshots(seasons: List[int], leagues: Iterable[str] = DIVISION) -> Dict[int, Dict[str, Dict]]:
    """{season: {understat-style team name: {"elo", "country", "level"}}} for every league and season."""
    out: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    for league in leagues:
        for season, table in rate_league(league, seasons).items():
            for team, elo in table.items():
                name = ALIASES.get(team, team)
                out[season][name] = {"elo": round(elo, 1), "country": COUNTRY[league], "level": 1}
    logger.info("Results Elo: rated %d team-seasons", sum(len(v) for v in out.values()))
    return dict(out)
