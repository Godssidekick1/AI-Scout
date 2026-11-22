"""
FBref big-five tables (Opta), from the worldfootballR_data release
==================================================================
FBref carried Opta's advanced player tables for the big five from 2017-18 until
January 2026. fbref.com itself sits behind a bot challenge (we don't get past
those), but JaseZiv's worldfootballR_data repository publishes FBref's own
big-five season tables as GitHub release assets (R .rds files, read with
pyreadr). We use four of them:

  defense     Tkl, TklW, Int, Blocks, Clr                  2017-18 to 2023-24 (+ first ~9 games of 2024-25)
  misc        aerials won/lost, ball recoveries            same
  possession  progressive carries                          same
  passing     progressive passes                           2017-18 to 2024-25 (partial)

Checked against premierleague.com for the 2022-23 Premier League (both Opta):
tackles, interceptions, clearances, aerials and recoveries agree to within a few
per cent player by player; FBref's "blocks" counts shots and passes blocked, so
blocks are not mixed across the two (see tests/test_defensive.py).
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List

import pandas as pd

from forecast.data.defensive import PoliteClient, frame

logger = logging.getLogger(__name__)

RELEASE = "https://github.com/JaseZiv/worldfootballR_data/releases/download/fb_big5_advanced_season_stats"
COMPS = {"Premier League": "premier_league", "La Liga": "la_liga", "Bundesliga": "bundesliga",
         "Serie A": "serie_a", "Ligue 1": "ligue_1"}
TABLES: Dict[str, Dict[str, str]] = {
    # "blocks" = shots blocked (Sh_Blocks): that is what Opta/premierleague.com call blocks.
    # FBref's Blocks_Blocks adds passes blocked, about 3x larger.
    "defense": {"Tkl_Tackles": "tackles", "TklW_Tackles": "tackles_won", "Int": "interceptions",
                "Sh_Blocks": "blocks", "Clr": "clearances"},
    "misc": {"Won_Aerial": "aerials_won", "Lost_Aerial": "aerials_lost", "Recov": "recoveries"},
    "possession": {"PrgC_Carries": "prog_carries"},
    "passing": {"PrgP": "prog_passes"},
}
KEYS = ["fbref_id", "Squad", "Comp", "Season_End_Year"]
_client = None


def client() -> PoliteClient:
    global _client
    if _client is None:
        _client = PoliteClient("fbref_release", min_interval=1.0, timeout=180)
    return _client


def read_table(name: str, ttl_days: float = 30) -> pd.DataFrame:
    import pyreadr
    path = client().file(f"big5_player_{name}.rds", f"{RELEASE}/big5_player_{name}.rds", ttl_days=ttl_days)
    df = pyreadr.read_r(str(path))[None]
    df["fbref_id"] = df["Url"].astype(str).str.extract(r"/players/([0-9a-f]+)/")[0]
    return df


def load(seasons: Iterable[int]) -> pd.DataFrame:
    """One row per player per club per league-season (season = starting year)."""
    wanted = {s + 1 for s in seasons}        # FBref labels seasons by their end year
    # The tables were scraped at different times (2024-25 defense covers ~9 games, passing the
    # whole season), so each table's totals are rescaled to the defense table's minutes.
    merged = None
    for name, cols in TABLES.items():
        df = read_table(name)
        df = df[df["Season_End_Year"].isin(wanted) & df["Comp"].isin(COMPS)]
        have = [c for c in cols if c in df]
        g = df.groupby(KEYS, as_index=False).agg(
            **({"name": ("Player", "first"), "born": ("Born", "first")} if merged is None else {}),
            n90=("Mins_Per_90", "sum"), **{cols[c]: (c, lambda x: x.sum(min_count=1)) for c in have})
        if merged is None:
            g = g.rename(columns={"n90": "def_n90"})
        else:
            scale = merged[KEYS].merge(g, on=KEYS, how="left")
            factor = (merged["def_n90"].to_numpy() / scale["n90"].where(scale["n90"] > 0).to_numpy())
            g = scale.drop(columns="n90")
            for c in have:
                g[cols[c]] = g[cols[c]] * factor
        merged = g if merged is None else merged.merge(g, on=KEYS, how="left")
    if merged is not None:
        merged["minutes"] = merged["def_n90"] * 90
    if merged is None or merged.empty:
        return frame([])
    merged = merged[merged["tackles"].notna() | merged["aerials_won"].notna()]
    rows: List[Dict] = []
    for r in merged.to_dict("records"):
        rows.append({"source": "fbref", "league": COMPS[r["Comp"]], "season": int(r["Season_End_Year"]) - 1,
                     "src_id": r["fbref_id"], "name": r["name"], "team": r["Squad"],
                     "birth_year": int(r["born"]) if pd.notna(r.get("born")) else None, "minutes": r["minutes"],
                     "duels_won": None, "duels_lost": None,
                     **{c: r.get(c) for c in ("tackles", "tackles_won", "interceptions", "blocks", "clearances",
                                              "aerials_won", "aerials_lost", "recoveries", "prog_passes",
                                              "prog_carries")}})
    return frame(rows)


def check() -> str:
    r = client()._get(f"{RELEASE}/big5_player_defense.rds")
    return "ok" if r.content[:2] == b"\x1f\x8b" or len(r.content) > 1_000_000 else "DOWN (unexpected content)"
