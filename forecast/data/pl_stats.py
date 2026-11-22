"""
Premier League player stats (Opta), from premierleague.com
==========================================================
The JSON API premierleague.com's own stats pages call (Pulselive "SDP"):

  /api/v3/competitions/8/seasons/{season}/players/stats/leaderboard   every player with
        his season totals (~85 Opta counters: tackles, interceptions, clearances, blocks,
        duels, aerials, recoveries...), 100 per page, cursor-paged
  /api/v1/competitions/8/seasons/{season}/teams                       the 20 clubs
  /api/v2/competitions/8/seasons/{season}/teams/{id}/squad            each club's squad,
        for the player's club(s) that season and birth date

No key or login. The API allows 300 requests a minute; we stay near one a
second and cache everything (past seasons forever, the current one for a day),
so a full rebuild is ~30 requests per season once and none after.

Fallback when the API is down: imadeddine-belkat/Premier-League-Stats on GitHub
(MIT), a per-club CSV export of the same API. Spot-checked identical to the live
API (e.g. Haaland 2024-25: 2741 min, 11 tackles, 5 interceptions, 22 clearances,
57/51 aerials, 34 recoveries in both).

premierleague.com's terms restrict automated collection and redistribution:
fine for personal research, but don't publish the raw data.
"""

from __future__ import annotations

import io
import logging
from typing import Dict, Iterable, List, Optional

import pandas as pd

from config import current_understat_season
from forecast.data.defensive import PoliteClient, SourceUnavailable, frame

logger = logging.getLogger(__name__)

API = "https://sdp-prem-prod.premier-league-prod.pulselive.com/api"
FIRST_SEASON = 2008            # earliest season the stats API returns
HEADERS = {"Origin": "https://www.premierleague.com", "Referer": "https://www.premierleague.com/"}
MIRROR = "https://raw.githubusercontent.com/imadeddine-belkat/Premier-League-Stats/main/pl_stats"
# club folders in the mirror (name_teamId), every club with a Premier League season since 2008-09
MIRROR_CLUBS = ("Arsenal_3 Birmingham_41 Blackburn_5 Blackpool_92 Bolton_30 Bournemouth_91 Brentford_94 Brighton_36 "
                "Burnley_90 Cardiff_97 Chelsea_8 Coventry_9 Everton_11 Forest_17 Fulham_54 Huddersfield_38 Hull_88 "
                "Ipswich_40 Leeds_2 Leicester_13 Liverpool_14 Luton_102 Man_City_43 Man_Utd_1 Middlesbrough_25 "
                "Newcastle_4 Norwich_45 Palace_31 Portsmouth_47 QPR_52 Reading_108 Sheffield_Utd_49 Southampton_20 "
                "Spurs_6 Stoke_110 Sunderland_56 Swansea_80 Villa_7 Watford_57 West_Brom_35 West_Ham_21 Wigan_111 "
                "Wolves_39").split()

# Opta keys -> our totals
FIELDS = {"tackles": "totalTackles", "tackles_won": "tacklesWon", "interceptions": "interceptions",
          "blocks": "blocks", "clearances": "totalClearances", "aerials_won": "aerialDuelsWon",
          "aerials_lost": "aerialDuelsLost", "recoveries": "recoveries", "duels_won": "duelsWon",
          "duels_lost": "duelsLost"}

_client: Optional[PoliteClient] = None


def client() -> PoliteClient:
    global _client
    if _client is None:
        _client = PoliteClient("premierleague", min_interval=0.8, headers=HEADERS)
    return _client


def _ttl(season: int) -> Optional[float]:
    return 1.0 if season >= current_understat_season() else None


def leaderboard(season: int) -> List[Dict]:
    """Every player's season totals: [{playerMetadata, stats}]."""
    out, cursor, page = [], None, 0
    while True:
        params = {"_limit": 100, **({"_next": cursor} if cursor else {})}
        data = client().json(f"lb_{season}_{page}", f"{API}/v3/competitions/8/seasons/{season}/players/stats/leaderboard",
                             params=params, ttl_days=_ttl(season))
        out += data.get("data", [])
        cursor = (data.get("pagination") or {}).get("_next")
        page += 1
        if not cursor or page > 50:
            return out


def squads(season: int) -> Dict[str, Dict]:
    """{player id: {"teams": [club names], "birth": "YYYY-MM-DD"}} for the season."""
    teams = client().json(f"teams_{season}", f"{API}/v1/competitions/8/seasons/{season}/teams",
                          params={"_limit": 100}, ttl_days=_ttl(season)).get("data", [])
    out: Dict[str, Dict] = {}
    for t in teams:
        try:
            sq = client().json(f"squad_{season}_{t['id']}", f"{API}/v2/competitions/8/seasons/{season}/teams/{t['id']}/squad",
                               ttl_days=_ttl(season))
        except SourceUnavailable as e:
            logger.warning("premierleague.com: no squad for %s %s (%s)", t.get("name"), season, e)
            continue
        for p in sq.get("players", []):
            rec = out.setdefault(str(p["id"]), {"teams": [], "birth": (p.get("dates") or {}).get("birth")})
            rec["teams"].append(t["name"])
    return out


def rows_from_api(board: List[Dict], squad: Dict[str, Dict], season: int) -> List[Dict]:
    rows = []
    for item in board:
        meta, st = item.get("playerMetadata") or {}, item.get("stats") or {}
        mins = st.get("timePlayed")
        if not mins:
            continue
        pid = str(meta.get("id"))
        sq = squad.get(pid, {})
        birth = sq.get("birth")
        rows.append({"source": "premierleague.com", "league": "premier_league", "season": season, "src_id": pid,
                     "name": meta.get("name") or f"{meta.get('firstName', '')} {meta.get('lastName', '')}".strip(),
                     "team": " / ".join(sq.get("teams", [])) or (meta.get("currentTeam") or {}).get("name", ""),
                     "birth_year": int(birth[:4]) if birth else None, "minutes": mins,
                     **{k: st.get(v, 0.0) for k, v in FIELDS.items()},
                     "prog_passes": None, "prog_carries": None})
    return rows


def rows_from_mirror(season: int) -> List[Dict]:
    label = f"{season}-{str(season + 1)[-2:]}"
    parts = []
    for club in MIRROR_CLUBS:
        try:
            path = client().file(f"mirror_{club}_{label}.csv", f"{MIRROR}/{club}/players_stats/{label}_players_stats.csv",
                                 ttl_days=_ttl(season))
        except SourceUnavailable:
            continue           # club not in the league that season (404)
        parts.append(pd.read_csv(path))
    if not parts:
        return []
    df = pd.concat(parts, ignore_index=True)
    df = df[df["timePlayed"].fillna(0) > 0]
    # season totals are per player, repeated in each club file he appeared in
    teams = df.groupby("playerId")["team_name"].agg(lambda s: " / ".join(dict.fromkeys(s)))
    df = df.drop_duplicates("playerId")
    return [{"source": "premierleague.com", "league": "premier_league", "season": season, "src_id": str(r.playerId),
             "name": r.playerName, "team": teams[r.playerId], "birth_year": None, "minutes": r.timePlayed,
             **{k: getattr(r, v, 0.0) if pd.notna(getattr(r, v, None)) else 0.0 for k, v in FIELDS.items()},
             "prog_passes": None, "prog_carries": None} for r in df.itertuples()]


def load(seasons: Iterable[int]) -> pd.DataFrame:
    rows: List[Dict] = []
    api_ok = True
    for s in sorted(set(seasons)):
        if s < FIRST_SEASON:
            continue
        got: List[Dict] = []
        if api_ok:
            try:
                got = rows_from_api(leaderboard(s), squads(s), s)
            except SourceUnavailable as e:
                logger.warning("premierleague.com API unavailable (%s); using the GitHub export", e)
                api_ok = False
        if not got:
            got = rows_from_mirror(s)
        if not got:
            logger.warning("premierleague.com: no data for %s", s)
        rows += got
    return frame(rows)


def check() -> str:
    data = client()._get(f"{API}/v1/competitions/8/seasons/{current_understat_season() - 1}/teams").json()
    return "ok" if data.get("data") else "DOWN (no teams in response)"
