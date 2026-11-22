"""
Build the forecast model
========================
    python -m forecast.build              # real data (needs network access to the sources)
    python -m forecast.build --synthetic  # offline demo on generated data
    python -m forecast.build --check      # can this machine reach the three sources?

Steps: load history -> link Understat to Transfermarkt -> ClubElo ratings ->
player-season table and season pairs -> rolling backtest (ladder) -> final fit
on every completed season -> save the service the agent loads, plus a
markdown backtest report in evals/results/.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

from config import current_understat_season
from forecast import backtest
from forecast.config import LEAGUES, METRICS
from forecast.model import MinutesModel, ProjectionModel, prepare
from forecast.service import ForecastService
from forecast.table import player_seasons, season_pairs

logger = logging.getLogger("forecast.build")
RESULTS = Path(__file__).resolve().parent.parent / "evals" / "results"


DEF_REPORT: Dict[str, str] = {}


def add_defensive(ps: pd.DataFrame, us: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    """Join defensive/duel/progression rates (forecast/data/defensive.py) onto the player-season
    table. Any failure leaves the table as it was: the model doesn't depend on these columns."""
    from forecast.data import defensive
    try:
        rows, cov = defensive.load_all(seasons)
        agg = us.groupby(["us_id", "league", "season"], as_index=False).agg(
            name=("name", "first"), team=("team", lambda t: " / ".join(dict.fromkeys(t))), minutes=("minutes", "sum"))
        born = (ps["season"] - ps["age"].round()).groupby(ps["us_id"]).median()   # from Transfermarkt
        agg["birth_year"] = agg["us_id"].map(born)
        linked = defensive.link(rows, agg)
        ps = defensive.add_rates(ps, defensive.player_season_table(linked))
        try:      # on-the-ball rates: their own link, since FBref passing covers seasons its defense table doesn't
            from forecast.data import fbref_stats
            ball = defensive.link(fbref_stats.load_ball(seasons), agg)
            ps = defensive.add_ball(ps, defensive.ball_table(ball))
            logger.info("Passing/take-ons: linked %d of %d FBref rows", int(ball["us_id"].notna().sum()), len(ball))
        except Exception as e:  # noqa: BLE001
            logger.warning("Passing and take-on data skipped: %s", e)
    except Exception as e:  # noqa: BLE001
        logger.warning("Defensive data skipped: %s", e)
        DEF_REPORT["text"] = f"Defensive data: not loaded ({str(e)[:160]})."
        return ps
    table = defensive.coverage_table(cov, sorted(set(seasons)))
    linked_n = int(linked["us_id"].notna().sum())
    big = linked[linked["minutes"] >= 900]
    rate = float(big["us_id"].notna().mean()) if len(big) else 0.0
    lines = ["## Defensive data coverage", "",
             "PL = premierleague.com, SA = legaseriea.it, FB = FBref big-five copy, - = none. "
             f"Linked {linked_n} of {len(linked)} source rows to Understat ({rate:.0%} of players with 900+ minutes).",
             "", backtest.to_markdown(table.rename(columns=lambda c: str(c))), ""]
    bad = cov[cov["status"] != "ok"] if not cov.empty else cov
    lines += [f"- {r['source']}: {r['status']}" for _, r in bad.iterrows()]
    DEF_REPORT["text"] = "\n".join(lines)
    return ps


def load_real(seasons: List[int], elo_source: str = "auto",
              defensive: bool = True, offline: bool = False) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    from forecast.data import clubelo, transfermarkt
    from forecast.data.understat_history import load_seasons

    us = asyncio.run(load_seasons(seasons, offline=offline))
    if us.empty:
        raise SystemExit("No Understat data loaded. Is understat.com reachable from here?")
    logger.info("Understat: %d player-season rows", len(us))

    transfermarkt.download()
    players, valuations = transfermarkt.load("players"), transfermarkt.load("player_valuations")
    us_to_tm = transfermarkt.match_players(us, players, valuations)

    snapshots = team_ratings(seasons, elo_source)
    teams = set(zip(us["team"], us["league"], us["season"]))
    team_elo = clubelo.team_elo_table(teams, snapshots)

    ps = player_seasons(us, players, us_to_tm, team_elo)
    if defensive:
        ps = add_defensive(ps, us, seasons)
    latest_values = transfermarkt.value_at(valuations, dt.date.today().isoformat())
    values = {u: latest_values.get(t) for u, t in us_to_tm.items() if t is not None and t in latest_values}
    lineups = transfermarkt.load("game_lineups", usecols=transfermarkt.LINEUP_COLUMNS)
    profiles = transfermarkt.player_profiles(ps, players, valuations, lineups)
    logger.info("Transfermarkt profiles: %d players, %d with recent lineups, %d moved since %s",
                len(profiles), int(profiles["position_shares"].notna().sum()) if len(profiles) else 0,
                int(profiles["moved"].sum()) if len(profiles) else 0, int(ps["season"].max()))
    return ps, values, profiles


ELO_USED = {"source": "clubelo"}


def team_ratings(seasons: List[int], source: str = "auto") -> Dict[int, Dict[str, Dict]]:
    """Season-start team ratings: ClubElo, or our own Elo from results (forecast/data/results_elo.py)
    when ClubElo is down ('auto') or when asked for ('results')."""
    from forecast.data import clubelo, results_elo

    if source in ("auto", "clubelo"):
        try:
            snaps = {s: clubelo.fetch_snapshot(clubelo.season_start(s)) for s in seasons}
            ELO_USED["source"] = "clubelo"
            return snaps
        except clubelo.ClubEloUnavailable as e:
            if source == "clubelo":
                raise SystemExit(f"ClubElo ratings unavailable: {e}. Rerun later, or use --elo results.")
            logger.warning("ClubElo unavailable (%s); using Elo computed from match results instead", e)
    ELO_USED["source"] = "results"
    return results_elo.snapshots(seasons)


def check_sources() -> bool:
    """One request to each source; prints what works. Returns True if all three do."""
    from data_sources.understat_source import UnderstatSource
    from forecast.data import clubelo, transfermarkt

    def understat() -> str:
        r = httpx.get(f"{UnderstatSource.BASE_URL}/league/EPL", timeout=20, follow_redirects=True)
        if UnderstatSource.is_maintenance_page(r.text):
            return "DOWN (site is showing its maintenance page)"
        return "ok" if r.status_code == 200 else f"DOWN (HTTP {r.status_code})"

    def tm() -> str:
        r = httpx.head(f"{transfermarkt.BASE_URL}/players.csv.gz", timeout=20, follow_redirects=True)
        return "ok" if r.status_code == 200 else f"DOWN (HTTP {r.status_code})"

    def elo() -> str:
        r = httpx.get(f"{clubelo.API}/{clubelo.season_start(current_understat_season() - 1)}", timeout=20)
        if r.status_code != 200:
            return f"DOWN (HTTP {r.status_code})"
        return "ok" if clubelo.parse_snapshot(r.text) else "DOWN (no ratings in response)"

    all_ok = True
    for name, probe in (("understat.com", understat), ("Transfermarkt dataset", tm), ("api.clubelo.com", elo)):
        try:
            status = probe()
        except httpx.HTTPError as e:
            status = f"UNREACHABLE ({type(e).__name__})"
        all_ok &= status == "ok"
        print(f"{name:24s} {status}")
    print("\nDefensive data (optional; the build continues without any of these):")
    from forecast.data import defensive
    defensive.check()
    return all_ok


def load_synthetic(seed: int = 7) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    from forecast import synthetic
    us, tm, us_to_tm, team_elo = synthetic.generate(seed=seed)
    ps = player_seasons(us, tm, us_to_tm, team_elo)
    from forecast.data import defensive
    ps = defensive.add_rates(ps, defensive.synthetic_table(ps, seed))
    ps = defensive.add_ball(ps, defensive.synthetic_ball(ps, seed))
    rng = np.random.default_rng(seed)
    values = {u: float(np.exp(rng.normal(16, 1))) for u in ps["us_id"].unique()}
    return ps, values, synthetic.profiles(ps, seed)


def report(res: Dict[str, pd.DataFrame], model: ProjectionModel, synthetic: bool, test_seasons: List[int],
           ps: Optional[pd.DataFrame] = None) -> str:
    lines = [f"# Forecast backtest ({dt.date.today().isoformat()})", ""]
    if synthetic:
        lines += ["> Synthetic data: this checks the code, not football. Real numbers come from a real build.", ""]
    elif ELO_USED["source"] == "results":
        lines += ["> Team strength: Elo computed from domestic results (football-data.co.uk), not ClubElo, "
                  "so dElo is not comparable across leagues.", ""]
    if ps is not None and not synthetic:
        have = set(zip(ps["league"], ps["season"]))
        seasons = range(int(ps["season"].min()), int(ps["season"].max()) + 1)
        gaps = [f"{lg} {s}" for lg in LEAGUES for s in seasons if (lg, s) not in have]
        lines += [f"Data: {len(have)} league-seasons loaded" + (f"; missing: {', '.join(gaps)}." if gaps else "."), ""]
    lines += [f"Test seasons (target): {', '.join(map(str, test_seasons))}. Error = minutes-weighted RMSE of per-90 "
              "predictions; lower is better.", ""]
    for subset, title in (("all", "All players"), ("movers", "Players who changed club"),
                          ("league_change", "Players who changed league")):
        lines += [f"## {title}", "", backtest.to_markdown(backtest.ladder_table(res["scores"], subset)), ""]
    lines += ["## 80% interval coverage (full model)", "",
              backtest.to_markdown(res["coverage"].set_index("metric"), 3), "",
              *( [DEF_REPORT["text"], ""] if DEF_REPORT.get("text") else [] ),
              "## Minutes model", "", backtest.to_markdown(res["minutes"].set_index("season"), 1), "",
              "## Fitted coefficients (final model, relative effects)", ""]
    for m in METRICS:
        coef = {k: round(float(v), 3) for k, v in model.explain(m).items() if "|" not in k}
        lines.append(f"- **{m}**: " + ", ".join(f"{k} = {v}" for k, v in coef.items()))
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="use generated data (offline demo)")
    ap.add_argument("--check", action="store_true", help="only check that the data sources are reachable")
    ap.add_argument("--elo", choices=("auto", "clubelo", "results"), default="auto",
                    help="team ratings: ClubElo, Elo from match results, or ClubElo with results as fallback")
    ap.add_argument("--first-season", type=int, default=2014)
    ap.add_argument("--no-defensive", action="store_true", help="skip the defensive/duel data sources")
    ap.add_argument("--offline", action="store_true",
                    help="Understat from the local cache only (no requests to understat.com)")
    ap.add_argument("--test-seasons", type=int, default=4, help="how many recent completed seasons to test on")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    if args.check:
        raise SystemExit(0 if check_sources() else 1)

    current = current_understat_season()
    if args.synthetic:
        ps, values, profiles = load_synthetic()
    else:
        ps, values, profiles = load_real(list(range(args.first_season, current + 1)), args.elo, not args.no_defensive,
                                         args.offline)

    pairs = season_pairs(ps)
    completed = pairs[pairs["target_season"] < current] if not args.synthetic else pairs
    last = int(completed["target_season"].max())
    test_seasons = list(range(last - args.test_seasons + 1, last + 1))

    res = backtest.run(ps, completed, test_seasons)
    full, hist = prepare(ps, completed)
    model = ProjectionModel().fit(full, hist)
    minutes = MinutesModel().fit(full, ps)

    text = report(res, model, args.synthetic, test_seasons, ps)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"backtest_{'synthetic_' if args.synthetic else ''}{dt.date.today().isoformat()}.md"
    out.write_text(text)
    path = ForecastService(ps=ps, model=model, minutes=minutes, values=values, profiles=profiles,
                           backtest_summary=text, synthetic=args.synthetic).save()
    print(backtest.to_markdown(backtest.ladder_table(res["scores"])))
    print(f"\nReport: {out}\nModel:  {path}")


if __name__ == "__main__":
    main()
