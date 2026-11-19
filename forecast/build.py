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
from forecast.config import METRICS
from forecast.model import MinutesModel, ProjectionModel, prepare
from forecast.service import ForecastService
from forecast.table import player_seasons, season_pairs

logger = logging.getLogger("forecast.build")
RESULTS = Path(__file__).resolve().parent.parent / "evals" / "results"


def load_real(seasons: List[int]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    from forecast.data import clubelo, transfermarkt
    from forecast.data.understat_history import load_seasons

    us = asyncio.run(load_seasons(seasons))
    if us.empty:
        raise SystemExit("No Understat data loaded. Is understat.com reachable from here?")
    logger.info("Understat: %d player-season rows", len(us))

    transfermarkt.download()
    players, valuations = transfermarkt.load("players"), transfermarkt.load("player_valuations")
    us_to_tm = transfermarkt.match_players(us, players, valuations)

    try:
        snapshots = {s: clubelo.fetch_snapshot(clubelo.season_start(s)) for s in seasons}
    except clubelo.ClubEloUnavailable as e:
        raise SystemExit(f"ClubElo ratings unavailable: {e}. The API is sometimes down; "
                         "rerun later (Understat and Transfermarkt downloads stay cached).")
    teams = set(zip(us["team"], us["league"], us["season"]))
    team_elo = clubelo.team_elo_table(teams, snapshots)

    ps = player_seasons(us, players, us_to_tm, team_elo)
    latest_values = transfermarkt.value_at(valuations, dt.date.today().isoformat())
    values = {u: latest_values.get(t) for u, t in us_to_tm.items() if t is not None and t in latest_values}
    return ps, values


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
    return all_ok


def load_synthetic(seed: int = 7) -> Tuple[pd.DataFrame, Dict[str, float]]:
    from forecast import synthetic
    us, tm, us_to_tm, team_elo = synthetic.generate(seed=seed)
    ps = player_seasons(us, tm, us_to_tm, team_elo)
    rng = np.random.default_rng(seed)
    values = {u: float(np.exp(rng.normal(16, 1))) for u in ps["us_id"].unique()}
    return ps, values


def report(res: Dict[str, pd.DataFrame], model: ProjectionModel, synthetic: bool, test_seasons: List[int]) -> str:
    lines = [f"# Forecast backtest ({dt.date.today().isoformat()})", ""]
    if synthetic:
        lines += ["> Synthetic data: this checks the code, not football. Real numbers come from a real build.", ""]
    lines += [f"Test seasons (target): {', '.join(map(str, test_seasons))}. Error = minutes-weighted RMSE of per-90 "
              "predictions; lower is better.", ""]
    for subset, title in (("all", "All players"), ("movers", "Players who changed club"),
                          ("league_change", "Players who changed league")):
        lines += [f"## {title}", "", backtest.to_markdown(backtest.ladder_table(res["scores"], subset)), ""]
    lines += ["## 80% interval coverage (full model)", "",
              backtest.to_markdown(res["coverage"].set_index("metric"), 3), "",
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
    ap.add_argument("--first-season", type=int, default=2014)
    ap.add_argument("--test-seasons", type=int, default=4, help="how many recent completed seasons to test on")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    if args.check:
        raise SystemExit(0 if check_sources() else 1)

    current = current_understat_season()
    if args.synthetic:
        ps, values = load_synthetic()
    else:
        ps, values = load_real(list(range(args.first_season, current + 1)))

    pairs = season_pairs(ps)
    completed = pairs[pairs["target_season"] < current] if not args.synthetic else pairs
    last = int(completed["target_season"].max())
    test_seasons = list(range(last - args.test_seasons + 1, last + 1))

    res = backtest.run(ps, completed, test_seasons)
    full, hist = prepare(ps, completed)
    model = ProjectionModel().fit(full, hist)
    minutes = MinutesModel().fit(full, ps)

    text = report(res, model, args.synthetic, test_seasons)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"backtest_{'synthetic_' if args.synthetic else ''}{dt.date.today().isoformat()}.md"
    out.write_text(text)
    path = ForecastService(ps=ps, model=model, minutes=minutes, values=values,
                           backtest_summary=text, synthetic=args.synthetic).save()
    print(backtest.to_markdown(backtest.ladder_table(res["scores"])))
    print(f"\nReport: {out}\nModel:  {path}")


if __name__ == "__main__":
    main()
