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
