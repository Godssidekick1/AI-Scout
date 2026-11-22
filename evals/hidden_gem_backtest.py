"""
Hidden-gem backtest: does the model find players before they break out?
=======================================================================
    python -m evals.hidden_gem_backtest                 # real build (run python -m forecast.build first)
    python -m evals.hidden_gem_backtest --synthetic     # checks the code on generated data

For each cutoff season C (2019-20 ... 2022-23 by default) the model is refitted on what was known
at the end of C: player-seasons up to C, season pairs whose target is C or earlier, and Transfermarkt
values on 1 July after C. No current clubs, positions or values leak in.

A "hidden gem" brief, for every role: age 23 or under, valued at most 15m (and a second brief: 26 or
under, at most 35m), 900+ minutes in C, not already at one of his league's top four clubs (by Elo).

Three ways to pick 10 players from that same pool:
  model          the product: ForecastService.candidates(role) ranked by projected role score
  last season    the 10 best observed role scores in C (what a scout with a stats site would do)
  priciest       the 10 most valuable (the market's own view)
and the pool's hit rate itself, which is what picking at random would give.

Outcomes over the next two seasons (C+1, C+2):
  top performer  a season with 1500+ minutes in the top quarter of his role (observed role score)
  big move       joined a top-four club of his league (by Elo) and played 900+ minutes there
  value x2       Transfermarkt value two years later at least double his value at the cutoff
  any            big move or value x2: the market caught up with him

An LLM-only shortlist can't be scored this way: a language model trained on later seasons already
knows who broke out, so its 'picks' from 2021 would be hindsight. That comparison is left out on purpose.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from forecast.config import METRICS
from forecast.model import MinutesModel, ProjectionModel, prepare
from forecast.roles import ROLES, role_scores
from forecast.service import ForecastService
from forecast.table import season_pairs

RESULTS = Path(__file__).resolve().parent / "results"
BRIEFS = {"U23, <=15m": (23, 15e6), "U26, <=35m": (26, 35e6)}
OUTCOMES = ["top performer", "big move", "value x2", "any"]
N_PICK = 10


def elite_teams(ps: pd.DataFrame, season: int, top: int = 4) -> set:
    """Each league's top `top` clubs by Elo in a season (within a league, so it holds for
    ClubElo and for Elo computed from domestic results alike)."""
    d = ps[ps["season"] == season].drop_duplicates(["league", "team"])
    return set(d.sort_values("elo", ascending=False).groupby("league").head(top)["team"])


def observed_scores(ps: pd.DataFrame, season: int, role, min_minutes: int = 0) -> pd.DataFrame:
    """Observed role score of every player in a season: the same formula as the projection,
    with his actual per-90s in place of the projected ones."""
    lite = ForecastService(ps=ps[ps["season"] <= season], model=None, minutes=None)
    rows = lite.with_defensive(lite.in_latest_season(lite.ps))
    rows = rows[rows["group"].isin(role.groups)]
    obs = rows.copy()
    for m in METRICS:
        obs[f"pred_{m}"] = obs[f"{m}_p90"]
    obs["obs_score"] = role_scores(obs, role, rows)
    return obs.loc[obs["minutes"] >= min_minutes, ["us_id", "team", "minutes", "obs_score"]]


def outcomes(ps: pd.DataFrame, cutoff: int, role, values_then: Dict[str, float],
             values_later: Dict[str, float]) -> pd.DataFrame:
    """One row per player in the cutoff season with the four outcome flags."""
    base = ps[ps["season"] == cutoff][["us_id", "team"]].drop_duplicates("us_id").set_index("us_id")
    out = pd.DataFrame(index=base.index)
    top = pd.Series(False, index=base.index)
    move = pd.Series(False, index=base.index)
    for s in (cutoff + 1, cutoff + 2):
        sc = observed_scores(ps, s, role)
        regulars = sc[sc["minutes"] >= 900]
        if len(regulars) >= 20:
            q75 = regulars["obs_score"].quantile(0.75)
            hit = set(sc.loc[(sc["minutes"] >= 1500) & (sc["obs_score"] >= q75), "us_id"])
            top |= base.index.isin(hit)
        elite = elite_teams(ps, s)
        later = ps[(ps["season"] == s) & ps["team"].isin(elite) & (ps["minutes"] >= 900)]
        later = later[later["us_id"].map(base["team"]).fillna("").to_numpy() != later["team"].to_numpy()]
        move |= base.index.isin(set(later["us_id"]))
    v0 = base.index.map(values_then).astype(float)
    v2 = base.index.map(values_later).astype(float)
    out["top performer"] = top.to_numpy()
    out["big move"] = move.to_numpy()
    out["value x2"] = np.where(np.isfinite(v0) & np.isfinite(v2) & (v0 > 0), v2 >= 2 * v0, False)
    out["any"] = out["big move"] | out["value x2"]
    return out


def cut_service(full: ForecastService, cutoff: int, values: Dict[str, float]) -> ForecastService:
    ps = full.ps[full.ps["season"] <= cutoff].copy()
    pairs = season_pairs(ps)
    fit_pairs, hist = prepare(ps, pairs)
    model = ProjectionModel().fit(fit_pairs, hist)
    minutes = MinutesModel().fit(fit_pairs, ps)
    return ForecastService(ps=ps, model=model, minutes=minutes, values=values, profiles=None,
                           synthetic=full.synthetic)


def tm_values(ps: pd.DataFrame, synthetic: bool, dates: List[str]) -> Dict[str, Dict[str, float]]:
    """us_id -> Transfermarkt value on each date."""
    if synthetic:
        return {d: synthetic_values(ps, int(d[:4]) - 1) for d in dates}
    from forecast.data import transfermarkt
    transfermarkt.download(["player_valuations"])
    val = transfermarkt.load("player_valuations")
    ids = ps.drop_duplicates("us_id").set_index("us_id")["tm_id"].dropna()
    out = {}
    for d in dates:
        at = transfermarkt.value_at(val, d)
        out[d] = {u: at[int(t)] for u, t in ids.items() if int(t) in at}
    return out


def synthetic_values(ps: pd.DataFrame, season: int, seed: int = 11) -> Dict[str, float]:
    """Generated data has no value history: price players on their last two seasons of output,
    age and club, with noise, so the code path can be tested (not the football)."""
    rng = np.random.default_rng(seed + season)
    d = ps[ps["season"].between(season - 1, season)]
    g = d.groupby("us_id").agg(out=("npxG_p90", "mean"), xa=("xA_p90", "mean"), chain=("xGChain_p90", "mean"),
                               age=("age", "last"), elo=("elo", "last"), mins=("minutes", "sum"))
    z = lambda s: (s - s.mean()) / (s.std() or 1)
    lv = 15.5 + 0.5 * z(g["out"] + g["xa"] + g["chain"]) + 0.4 * z(g["elo"]) + 0.3 * z(g["mins"]) \
        - 0.05 * (g["age"].fillna(26) - 25).clip(lower=0) ** 1.5 + rng.normal(0, 0.4, len(g))
    return dict(zip(g.index, np.exp(lv)))


def run(full: ForecastService, cutoffs: List[int]) -> pd.DataFrame:
    rows = []
    for c in cutoffs:
        then, later = f"{c + 1}-07-01", f"{c + 3}-07-01"
        vals = tm_values(full.ps, full.synthetic, [then, later])
        svc = cut_service(full, c, vals[then])
        elite = elite_teams(svc.ps, c)
        for key, role in ROLES.items():
            outc = outcomes(full.ps, c, role, vals[then], vals[later])
            obs = observed_scores(svc.ps, c, role).set_index("us_id")["obs_score"]
            for brief, (max_age, cap) in BRIEFS.items():
                kw = dict(max_age=max_age, max_value_eur=cap, min_minutes=900, exclude_teams=elite)
                pool = svc.candidates(key, n=100000, **kw)
                if len(pool) < 2 * N_PICK:
                    continue
                picks = {
                    "model": svc.candidates(key, n=N_PICK, **kw)["us_id"],
                    "last season": pool.assign(o=pool["us_id"].map(obs)).nlargest(N_PICK, "o")["us_id"],
                    "priciest": pool.nlargest(N_PICK, "market_value_eur")["us_id"],
                    "pool (random)": pool["us_id"],
                }
                for method, ids in picks.items():
                    o = outc.reindex(list(ids)).fillna(False)
                    rows.append({"cutoff": c, "role": key, "brief": brief, "method": method, "n": len(o),
                                 "pool": len(pool), **{k: float(o[k].mean()) for k in OUTCOMES},
                                 "ids": list(ids) if method != "pool (random)" else []})
        print(f"cutoff {c}: done")
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame, seed: int = 0, boots: int = 2000) -> pd.DataFrame:
    """Hit rate per method, averaged over (cutoff, role, brief) cells, with the lift over the pool
    and a bootstrap 90% interval for that lift (resampling cells)."""
    rng = np.random.default_rng(seed)
    cells = res.pivot_table(index=["cutoff", "role", "brief"], columns="method", values=OUTCOMES)
    out = []
    for method in ("model", "last season", "priciest", "pool (random)"):
        r = {"method": method, "picks": int(res.loc[res["method"] == method, "n"].sum())}
        for k in OUTCOMES:
            r[k] = cells[(k, method)].mean()
            if method != "pool (random)":
                lift = (cells[(k, method)] - cells[(k, "pool (random)")]).to_numpy()
                b = [lift[rng.integers(0, len(lift), len(lift))].mean() for _ in range(boots)]
                r[f"{k} lift"] = f"{lift.mean():+.1%} [{np.percentile(b, 5):+.1%}, {np.percentile(b, 95):+.1%}]"
        out.append(r)
    r = {"method": "model minus last season", "picks": None}     # the comparison that matters
    for k in OUTCOMES:
        d = (cells[(k, "model")] - cells[(k, "last season")]).to_numpy()
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(boots)]
        r[f"{k} lift"] = f"{d.mean():+.1%} [{np.percentile(b, 5):+.1%}, {np.percentile(b, 95):+.1%}]"
    out.append(r)
    return pd.DataFrame(out)


def report(res: pd.DataFrame, summary: pd.DataFrame, synthetic: bool, cutoffs: List[int]) -> str:
    from forecast.backtest import to_markdown
    lines = [f"# Hidden-gem backtest ({dt.date.today().isoformat()})", ""]
    if synthetic:
        lines += ["> Synthetic data and generated values: this checks the code, not football.", ""]
    lines += [f"Cutoffs: {', '.join(f'{c}-{str(c + 1)[2:]}' for c in cutoffs)}. Briefs: {', '.join(BRIEFS)}, every role, "
              f"not at a top-four club. {N_PICK} picks per method per cell; outcomes over the next two seasons.", "",
              "Hit rates (share of picks), averaged over cells. Lift = method minus the pool's own rate "
              "(random picking), with a bootstrap 90% interval.", ""]
    rates = summary[summary["picks"].notna()].set_index("method")[["picks", *OUTCOMES]].copy()
    rates["picks"] = rates["picks"].astype(int)
    for k in OUTCOMES:
        rates[k] = rates[k].map(lambda v: f"{v:.1%}")
    lines += [to_markdown(rates, index_name="method"), "", "Lift over random picks from the same pool, and the model "
              "against the last-season baseline (the bar it has to clear to be worth using):", ""]
    lifts = summary.set_index("method")[[f"{k} lift" for k in OUTCOMES]].dropna(how="all")
    lines += [to_markdown(lifts, index_name="method"), ""]
    by = res[res["method"] != "pool (random)"].groupby(["cutoff", "method"])[OUTCOMES].mean().unstack("method")
    lines += ["## By cutoff (any = big move or value x2)", "",
              to_markdown(by["any"].map(lambda v: f"{v:.0%}"), index_name="cutoff"), "",
              "## By role (top performer)", "",
              to_markdown(res.groupby(["role", "method"])["top performer"].mean().unstack("method")
                          .map(lambda v: f"{v:.0%}"), index_name="role"), ""]
    lines += ["Caveats: roles share players (a winger can be picked for two roles), so cells are not independent "
              "and the intervals are somewhat too narrow. Pools larger than 800 are cut to the 800 closest in "
              "style before ranking, as in the product. Elo-based 'top-four club' uses the Elo in the build "
              "(computed from domestic results when ClubElo is down).", "",
              "LLM-only picks are not scored: a model trained on later data knows the outcomes.", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="build and test on generated data")
    ap.add_argument("--cutoffs", default="2019,2020,2021,2022", help="cutoff seasons (start years)")
    args = ap.parse_args(argv)
    if args.synthetic:
        from forecast.build import load_synthetic
        ps, values, _ = load_synthetic()
        full = ForecastService(ps=ps, model=None, minutes=None, values=values, synthetic=True)
        last = int(ps["season"].max())
        cutoffs = [c for c in (last - 4, last - 3, last - 2)]
    else:
        full = ForecastService.load()
        if full is None:
            raise SystemExit("No saved model. Run python -m forecast.build first.")
        cutoffs = [int(c) for c in args.cutoffs.split(",")]
    have = int(full.ps["season"].max())
    cutoffs = [c for c in cutoffs if c + 2 <= have - (0 if args.synthetic else 1)]   # two finished seasons after
    res = run(full, cutoffs)
    summary = summarise(res)
    text = report(res, summary, full.synthetic, cutoffs)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"hidden_gems_{'synthetic_' if full.synthetic else ''}{dt.date.today().isoformat()}.md"
    out.write_text(text)
    res.drop(columns="ids").to_csv(out.with_suffix(".csv"), index=False)
    print(text)
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
