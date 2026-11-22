"""
Player-season table and season pairs
====================================
`player_seasons` turns raw Understat totals into one row per player per
season, with per-90 rates, team context (team totals, the player's share of
them, ClubElo strength) and, where Transfermarkt matched, age and position.

`season_pairs` joins each season to the same player's next season. Those pairs
are the training rows: everything on the "from" side is known before the next
season starts; the "next" side is what the model must predict.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from forecast.config import METRICS, position_group

MIN_HISTORY_MINUTES = 90  # rows below this are dropped as noise
PENALTY_XG = 0.76         # Understat's xG for a penalty


def per90(total, minutes):
    return np.where(minutes > 0, total / np.maximum(minutes, 1) * 90, 0.0)


def player_seasons(us: pd.DataFrame, tm_players: Optional[pd.DataFrame] = None,
                   us_to_tm: Optional[Dict[str, Optional[int]]] = None,
                   team_elo: Optional[Dict[tuple, Optional[float]]] = None) -> pd.DataFrame:
    """One row per (us_id, season). A player who played for two big-five teams in one
    season keeps the team where he played most minutes, with totals summed."""
    us = us[us["minutes"] >= 0].copy()

    # Team totals per team-season, from every player row (before aggregating movers).
    team_tot = us.groupby(["team", "league", "season"])[list(METRICS) + ["minutes"]].sum()
    team_tot.columns = [f"team_{m}" for m in METRICS] + ["team_minutes"]

    us = us.sort_values("minutes", ascending=False)
    agg = {"name": "first", "team": "first", "league": "first", "understat_position": "first",
           "minutes": "sum", "games": "sum", "goals": "sum", "assists": "sum", **{m: "sum" for m in METRICS}}
    agg.update({c: "sum" for c in ("xG", "npg") if c in us})
    ps = us.groupby(["us_id", "season"], as_index=False).agg(agg)
    # Penalties taken: Understat gives every penalty 0.76 xG and leaves it out of npxG.
    # Builds made before xG was kept have no penalty data (NaN).
    ps["pens"] = (np.round(np.maximum(ps["xG"] - ps["npxG"], 0) / PENALTY_XG) if "xG" in ps else np.nan)
    ps = ps.join(team_tot, on=["team", "league", "season"])

    matches = np.maximum(ps["team_minutes"].to_numpy(float) / 990, 1.0)   # 11 players x 90 minutes a match
    for m in METRICS:
        ps[f"{m}_p90"] = per90(ps[m].to_numpy(float), ps["minutes"].to_numpy(float))
        ps[f"team_level_{m}"] = ps[f"team_{m}"].to_numpy(float) / matches   # team output per match: its style
        team_total = ps[f"team_{m}"].to_numpy(float)
        ps[f"share_{m}"] = np.where(team_total > 0, ps[m] / np.maximum(team_total, 1e-9), 0.0)

    # Transfermarkt profile: age at 1 August of the season, detailed position.
    ps["tm_id"] = ps["us_id"].map(us_to_tm or {})
    ps["age"] = np.nan
    ps["tm_sub_position"] = None
    if tm_players is not None and us_to_tm:
        prof = tm_players.set_index("player_id")
        dob = pd.to_datetime(ps["tm_id"].map(prof["date_of_birth"]), errors="coerce")
        start = pd.to_datetime(ps["season"].astype(str) + "-08-01")
        ps["age"] = ((start - dob).dt.days / 365.25).round(2)
        ps["tm_sub_position"] = ps["tm_id"].map(prof["sub_position"])
    ps["group"] = [position_group(sub if isinstance(sub, str) else None, pos)
                   for sub, pos in zip(ps["tm_sub_position"], ps["understat_position"])]

    # Team strength. Missing ratings fall back to that league's mean for the season.
    ps["elo"] = [(team_elo or {}).get((t, lg, s)) for t, lg, s in zip(ps["team"], ps["league"], ps["season"])]
    ps["elo"] = pd.to_numeric(ps["elo"], errors="coerce")
    ps["elo"] = ps["elo"].fillna(ps.groupby(["league", "season"])["elo"].transform("mean")).fillna(1500.0)

    ps = ps[ps["group"].notna() & (ps["minutes"] >= MIN_HISTORY_MINUTES)]
    return ps.reset_index(drop=True)


def team_totals(ps: pd.DataFrame) -> pd.DataFrame:
    """Team totals per (team, league, season), as stored on the player rows."""
    cols = [f"team_{m}" for m in METRICS] + [f"team_level_{m}" for m in METRICS if f"team_level_{m}" in ps] + ["elo"]
    cols = [c for c in cols if c in ps]
    return ps.groupby(["team", "league", "season"])[cols].first()


def expected_share(player_total: float, new_team_total: float) -> float:
    """His share of the new team's output if he keeps producing at his current level:
    he adds his own output to theirs. (Who leaves the new team is unknown in advance.)"""
    denom = new_team_total + player_total
    return player_total / denom if denom > 0 else 0.0


def season_pairs(ps: pd.DataFrame) -> pd.DataFrame:
    """Join each player-season t to the same player's season t+1."""
    nxt = ps[["us_id", "season", "team", "league", "minutes", "elo", *[f"{m}_p90" for m in METRICS]]].copy()
    nxt["season"] = nxt["season"] - 1
    nxt = nxt.rename(columns={"team": "team_next", "league": "league_next", "minutes": "minutes_next",
                              "elo": "elo_next", **{f"{m}_p90": f"y_{m}" for m in METRICS}})
    pairs = ps.merge(nxt, on=["us_id", "season"], how="inner")
    pairs["mover"] = (pairs["team"] != pairs["team_next"]) | (pairs["league"] != pairs["league_next"])
    pairs["league_change"] = pairs["league"] != pairs["league_next"]

    # Expected change in usage share at the next club (pre-move information only).
    totals = team_totals(ps)
    league_median = ps.groupby(["league", "season"])[[f"team_{m}" for m in METRICS]].median()
    for m in METRICS:
        new_tot = []
        for team, league, season, mover in zip(pairs["team_next"], pairs["league_next"], pairs["season"], pairs["mover"]):
            key = (team, league, season)
            if key in totals.index:
                new_tot.append(float(totals.at[key, f"team_{m}"]))
            elif (league, season) in league_median.index:   # promoted club: no big-five season yet
                new_tot.append(float(league_median.at[(league, season), f"team_{m}"]))
            else:
                new_tot.append(np.nan)
        new_tot = np.array(new_tot)
        exp = np.array([expected_share(p, t) if not np.isnan(t) else s
                        for p, t, s in zip(pairs[m], new_tot, pairs[f"share_{m}"])])
        pairs[f"dusage_{m}"] = np.where(pairs["mover"], exp - pairs[f"share_{m}"], 0.0)
        pairs[f"dstyle_{m}"] = style_change(pairs, totals, league_level_median(ps), m)
    pairs["target_season"] = pairs["season"] + 1
    return pairs.reset_index(drop=True)


def league_level_median(ps: pd.DataFrame) -> pd.DataFrame:
    cols = [f"team_level_{m}" for m in METRICS if f"team_level_{m}" in ps]
    return ps.groupby(["league", "season"])[cols].median()


def style_change(pairs: pd.DataFrame, totals: pd.DataFrame, league_median: pd.DataFrame, m: str) -> np.ndarray:
    """How much more (or less) of metric m the new team produces per match than his current one,
    both measured in season t: a centre-back leaving a possession side for a direct one should
    expect less build-up involvement. 0 for players who stay. Clipped to [-0.6, +1.5]."""
    col = f"team_level_{m}"
    if col not in pairs:
        return np.zeros(len(pairs))
    out = np.zeros(len(pairs))
    for i, (team, league, season, mover, now) in enumerate(zip(pairs["team_next"], pairs["league_next"], pairs["season"],
                                                              pairs["mover"], pairs[col])):
        if not mover or not now or now <= 0:
            continue
        key = (team, league, season)
        if key in totals.index and col in totals:
            new = float(totals.at[key, col])
        elif (league, season) in league_median.index:
            new = float(league_median.at[(league, season), col])
        else:
            continue
        out[i] = new / now - 1
    return np.clip(out, -0.6, 1.5)
