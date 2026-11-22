"""
Synthetic league history with known answers
===========================================
Generates Understat-shaped player-season totals from a model whose true
parameters we choose (league effects, team effect, usage effect, age curves,
noise that shrinks with minutes). Used by the tests to check that the fitting
code recovers what it should, and to run the whole pipeline offline.

Nothing here is real data. Real fits use forecast.data.*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from forecast.config import LEAGUES, METRICS

GROUP_SUB_POSITION = {"CB": "Centre-Back", "FB": "Left-Back", "CM": "Central Midfield",
                      "AM": "Left Winger", "ST": "Centre-Forward"}
GROUP_SUB_POSITIONS = {"CB": ["Centre-Back"], "FB": ["Left-Back", "Right-Back"],
                       "CM": ["Central Midfield", "Defensive Midfield"],
                       "AM": ["Left Winger", "Right Winger", "Attacking Midfield", "Left Midfield", "Right Midfield"],
                       "ST": ["Centre-Forward", "Second Striker"]}
GROUP_US_CODE = {"CB": "D", "FB": "D", "CM": "M", "AM": "F M", "ST": "F"}
# Typical per-90 levels by group (rough, for realism only).
BASE = {
    "npxG":       {"CB": 0.04, "FB": 0.05, "CM": 0.08, "AM": 0.25, "ST": 0.40},
    "xA":         {"CB": 0.02, "FB": 0.10, "CM": 0.10, "AM": 0.20, "ST": 0.12},
    "shots":      {"CB": 0.50, "FB": 0.50, "CM": 1.00, "AM": 2.20, "ST": 3.00},
    "key_passes": {"CB": 0.30, "FB": 1.00, "CM": 1.20, "AM": 1.80, "ST": 1.00},
    "xGChain":    {"CB": 0.15, "FB": 0.30, "CM": 0.35, "AM": 0.55, "ST": 0.55},
    "xGBuildup":  {"CB": 0.12, "FB": 0.18, "CM": 0.25, "AM": 0.15, "ST": 0.08},
}


@dataclass
class Truth:
    league: Dict[str, float] = field(default_factory=lambda: {
        "premier_league": 0.0, "la_liga": 0.06, "bundesliga": 0.08, "serie_a": 0.04, "ligue_1": 0.12})
    team_per_100_elo: float = 0.08     # relative uplift per 100 Elo
    usage: float = -0.6                # relative change per unit change in share
    peak_age: Dict[str, float] = field(default_factory=lambda: {"CB": 29.5, "FB": 27.5, "CM": 28.0, "AM": 26.5, "ST": 27.0})
    age_curvature: float = 0.004       # relative loss per squared year from peak


def generate(n_players: int = 1500, seasons: Tuple[int, int] = (2014, 2025), seed: int = 7,
             truth: Truth | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[tuple, float]]:
    """Returns (understat_rows, tm_players, us_to_tm, team_elo)."""
    truth = truth or Truth()
    rng = np.random.default_rng(seed)
    teams: List[Tuple[str, str]] = [(f"{lg[:3].title()} Club {i}", lg) for lg in LEAGUES for i in range(16)]
    elo = {t: 1500 + rng.normal(0, 120) for t, _ in teams}

    groups = rng.choice(list(GROUP_SUB_POSITION), size=n_players, p=[0.22, 0.18, 0.22, 0.2, 0.18])
    birth = rng.uniform(seasons[0] - 33, seasons[1] - 18, size=n_players)
    talent = rng.lognormal(0, 0.28, size=n_players)          # multiplicative talent
    team_of = [teams[i] for i in rng.integers(0, len(teams), size=n_players)]

    pen_rng = np.random.default_rng(seed + 2)     # separate streams: these don't shift the stats
    takes_pens = (pen_rng.random(n_players) < 0.25) & np.isin(groups, ["AM", "ST", "CM"])
    rows = []
    for s in range(seasons[0], seasons[1] + 1):
        for i in range(n_players):
            age = s + 0.6 - birth[i]
            if age < 17 or age > 37:
                continue
            g = groups[i]
            # moves: better players drift toward stronger clubs
            if rng.random() < 0.15:
                cands = [teams[j] for j in rng.integers(0, len(teams), size=6)]
                cands.sort(key=lambda t: abs((elo[t[0]] - 1500) / 120 - (np.log(talent[i]) / 0.28)))
                team_of[i] = cands[0]
            team, lg = team_of[i]
            minutes = float(np.clip(rng.normal(1200 + 900 * (talent[i] - 1), 700), 0, 3420))
            if minutes < 90:
                continue
            age_factor = 1 - truth.age_curvature * (age - truth.peak_age[g]) ** 2
            team_factor = 1 + truth.team_per_100_elo * (elo[team] - 1500) / 100
            rec = {"us_id": str(i), "name": f"Player {i}", "team": team, "league": lg, "season": s,
                   "understat_position": GROUP_US_CODE[g], "minutes": round(minutes), "games": round(minutes / 85)}
            for m in METRICS:
                rate = BASE[m][g] * talent[i] * age_factor * team_factor * (1 + truth.league[lg])
                noise_sd = rate * 0.9 / np.sqrt(max(minutes, 90) / 90)
                p90 = max(rate + rng.normal(0, noise_sd), 0.0)
                rec[m] = p90 * minutes / 90
            rec["goals"], rec["assists"] = round(rec["npxG"]), round(rec["xA"])
            pens = pen_rng.poisson(3) if takes_pens[i] and minutes > 900 else 0
            rec["xG"], rec["npg"] = rec["npxG"] + 0.76 * pens, rec["goals"]
            rows.append(rec)
    us = pd.DataFrame(rows)
    sub_rng = np.random.default_rng(seed + 1)      # separate stream: positions don't shift the stats
    tm_players = pd.DataFrame({
        "player_id": np.arange(n_players), "name": [f"Player {i}" for i in range(n_players)],
        "date_of_birth": [pd.Timestamp(year=int(b), month=1, day=1) + pd.Timedelta(days=int((b % 1) * 365)) for b in birth],
        "sub_position": [sub_rng.choice(GROUP_SUB_POSITIONS[g]) for g in groups]})
    us_to_tm = {str(i): i for i in range(n_players)}
    team_elo = {(t, lg, s): elo[t] for t, lg in teams for s in range(seasons[0], seasons[1] + 2)}
    return us, tm_players, us_to_tm, team_elo


def profiles(ps: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Stand-in for forecast.data.transfermarkt.player_profiles: most players start mainly at their
    listed position, some split time with a second one, a few have moved club or stopped playing."""
    rng = np.random.default_rng(seed + 3)
    latest = ps.sort_values("season").groupby("us_id").tail(1)
    teams = latest[["team", "league"]].drop_duplicates().to_numpy()
    rows = []
    for us_id, team, league, sub, group in zip(latest["us_id"], latest["team"], latest["league"],
                                                latest["tm_sub_position"], latest["group"]):
        others = [p for subs in GROUP_SUB_POSITIONS.values() for p in subs if p != sub]
        main = rng.choice([0.9, 0.6, 0.35])
        shares = {sub: main, str(rng.choice(others)): round(1 - main, 2)} if isinstance(sub, str) else None
        moved = rng.random() < 0.08
        new_team, new_league = teams[rng.integers(len(teams))] if moved else (team, league)
        rows.append({"us_id": us_id, "current_club": new_team, "current_league": new_league,
                     "moved": bool(moved and new_team != team), "active": bool(rng.random() > 0.03),
                     "position_shares": shares})
    return pd.DataFrame(rows).set_index("us_id")


# ── real names on synthetic players, to test reading briefs offline ─────────────

REAL_CLUBS = {
    "premier_league": ["Arsenal", "Aston Villa", "Brighton", "Chelsea", "Everton", "Fulham", "Liverpool",
                       "Manchester City", "Manchester United", "Newcastle United", "Nottingham Forest", "Tottenham",
                       "West Ham", "Wolverhampton Wanderers", "Brentford", "Crystal Palace"],
    "la_liga": ["Athletic Club", "Atletico Madrid", "Barcelona", "Celta Vigo", "Girona", "Real Betis", "Real Madrid",
                "Real Sociedad", "Sevilla", "Valencia", "Villarreal", "Osasuna", "Getafe", "Mallorca", "Rayo Vallecano",
                "Alaves"],
    "bundesliga": ["Bayer Leverkusen", "Bayern Munich", "Borussia Dortmund", "Eintracht Frankfurt", "RasenBallsport Leipzig",
                   "VfB Stuttgart", "Wolfsburg", "Freiburg", "Hoffenheim", "Mainz 05", "Union Berlin", "Werder Bremen",
                   "Borussia M.Gladbach", "Augsburg", "FC Heidenheim", "St. Pauli"],
    "serie_a": ["AC Milan", "Atalanta", "Bologna", "Fiorentina", "Inter", "Juventus", "Lazio", "Napoli", "Roma",
                "Torino", "Udinese", "Genoa", "Como", "Lecce", "Cagliari", "Parma Calcio 1913"],
    "ligue_1": ["Lens", "Lille", "Lyon", "Marseille", "Monaco", "Nice", "Paris Saint Germain", "Rennes", "Strasbourg",
                "Brest", "Toulouse", "Nantes", "Reims", "Montpellier", "Auxerre", "Angers"],
}
REAL_PLAYERS = [   # (name, group, club, Transfermarkt position): the players the eval briefs name
    ("Wesley Fofana", "CB", "Chelsea", "Centre-Back"), ("Pedro Neto", "AM", "Chelsea", "Right Winger"),
    ("Noni Madueke", "AM", "Chelsea", "Right Winger"), ("Bukayo Saka", "AM", "Arsenal", "Right Winger"),
    ("Rodri", "CM", "Manchester City", "Defensive Midfield"), ("John Stones", "CB", "Manchester City", "Centre-Back"),
    ("Mohamed Salah", "AM", "Liverpool", "Right Winger"), ("Alexander Isak", "ST", "Newcastle United", "Centre-Forward"),
    ("Harry Kane", "ST", "Bayern Munich", "Centre-Forward"),
    ("Trent Alexander-Arnold", "FB", "Liverpool", "Right-Back"), ("Pedri", "CM", "Barcelona", "Central Midfield"),
    ("Jamal Musiala", "AM", "Bayern Munich", "Attacking Midfield"), ("William Saliba", "CB", "Arsenal", "Centre-Back"),
    ("Cole Palmer", "AM", "Chelsea", "Attacking Midfield"), ("Ousmane Dembélé", "AM", "Paris Saint Germain", "Right Winger"),
    ("Alejandro Grimaldo", "FB", "Bayer Leverkusen", "Left-Back"), ("Erling Haaland", "ST", "Manchester City", "Centre-Forward"),
    ("Khvicha Kvaratskhelia", "AM", "Napoli", "Left Winger"),
    ("Arda Güler", "AM", "Real Madrid", "Attacking Midfield"), ("Declan Rice", "CM", "Arsenal", "Defensive Midfield"),
]


def with_real_names(ps: pd.DataFrame, profiles: pd.DataFrame | None = None):
    """Rename synthetic clubs to real ones and give real names to a few synthetic players, so the
    brief reader can be tested offline on what scouts actually type. The numbers stay synthetic."""
    ps = ps.copy()
    team_map = {}
    for lg, names in REAL_CLUBS.items():
        fake = sorted(ps.loc[ps["league"] == lg, "team"].unique(), key=lambda t: int(t.split()[-1]))
        team_map.update(dict(zip(fake, names)))
    ps["team"] = ps["team"].map(team_map).fillna(ps["team"])
    latest = ps[ps["season"] == ps["season"].max()]
    used = set()
    for name, group, club, sub in REAL_PLAYERS:
        league = next(lg for lg, names in REAL_CLUBS.items() if club in names)
        pool = latest[(latest["group"] == group) & (latest["minutes"] >= 1200) & ~latest["us_id"].isin(used)
                      & (latest["league"] == league)]
        us_id = pool.sort_values("minutes", ascending=False)["us_id"].iloc[0]
        used.add(us_id)
        rows = ps["us_id"] == us_id
        ps.loc[rows, "name"] = name
        ps.loc[rows & (ps["season"] == ps["season"].max()), "team"] = club
        ps.loc[rows, "tm_sub_position"] = sub
    if profiles is not None and not profiles.empty:
        profiles = profiles.copy()
        profiles["current_club"] = profiles["current_club"].map(team_map).fillna(profiles["current_club"])
        last_team = ps.sort_values("season").groupby("us_id")["team"].last()
        named = ps.loc[ps["us_id"].isin(used), "us_id"].unique()
        profiles.loc[profiles.index.isin(named), "current_club"] = last_team.reindex(named).to_numpy()
        profiles.loc[profiles.index.isin(named), "moved"] = False
        profiles.loc[profiles.index.isin(named), "active"] = True
        subs = ps.drop_duplicates("us_id").set_index("us_id")["tm_sub_position"]
        for u in named:
            if u in profiles.index:
                profiles.at[u, "position_shares"] = {subs[u]: 1.0}
    return ps, profiles
