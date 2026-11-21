"""
Try the forecast model yourself
===============================
Works from the model that `python -m forecast.build` saved; no API keys needed.

    python -m forecast.predict player "Bukayo Saka"
    python -m forecast.predict player "Bukayo Saka" --to "Real Madrid"
    python -m forecast.predict player "Ollie Watkins" --to "Bayern Munich" --age 33 --minutes 1500
    python -m forecast.predict role "ball_playing_cb" --to "Manchester City" --max-age 24 --max-value 30
    python -m forecast.predict role "a Stones replacement" --leagues serie_a ligue_1 -n 15
    python -m forecast.predict roles          # the role profiles you can search for
    python -m forecast.predict clubs          # clubs you can move a player to, with team strength

`player` shows last season, the model's estimate of his underlying level
(ability), next season's projection with an 80% range, and how much each
context term moves it. `role` ranks the players whose style fits a role.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from forecast.config import LEAGUES, METRICS, season_label
from forecast.model import history_features
from forecast.roles import ROLES, role_from_text
from forecast.service import ForecastService, artifact_path
from forecast.table import team_totals

LEAGUE_NAMES = {"premier_league": "Premier League", "la_liga": "La Liga", "bundesliga": "Bundesliga",
                "serie_a": "Serie A", "ligue_1": "Ligue 1"}
TERMS = ("league", "team", "usage", "age", "trend", "drift")


def load(path: Optional[str]) -> ForecastService:
    p = Path(path) if path else artifact_path()
    service = ForecastService.load(p)
    if service is None:
        sys.exit(f"No saved model at {p}.\nBuild one first:  python -m forecast.build   "
                 "(or  python -m forecast.build --synthetic  for an offline demo)")
    if service.synthetic:
        print("Note: this model was built on SYNTHETIC data (--synthetic). Players and numbers are made up.\n")
    return service


def league(lg: str) -> str:
    return LEAGUE_NAMES.get(lg, lg)


def pct(x: float) -> str:
    return f"{x * 100:+.0f}%"


def resolve_club(service: ForecastService, club: Optional[str]) -> Optional[pd.Series]:
    if not club:
        return None
    row = service._team_row(club)
    if row is None:
        from thefuzz import process
        teams = clubs_table(service)["team"].tolist()
        close = [t for t, _ in process.extract(club, teams, limit=5)]
        sys.exit(f"No club matching '{club}' in the latest season. Closest: {', '.join(close)}\n"
                 "See every club with:  python -m forecast.predict clubs")
    return row


def clubs_table(service: ForecastService) -> pd.DataFrame:
    return service.in_latest_season(team_totals(service.ps).reset_index()).sort_values("elo", ascending=False)


# ── commands ───────────────────────────────────────────────────

def cmd_player(service: ForecastService, args) -> None:
    hit = service.find(args.name, args.team)
    if hit is None:
        sys.exit(f"No player matching '{args.name}'. Try the full name as Understat spells it, "
                 "and add --team to separate namesakes.")
    target = resolve_club(service, args.to)
    player = hit.to_frame().T.infer_objects()

    proj = service.project(player, args.to if target is not None else None, age=args.age, minutes=args.minutes)
    p = proj.iloc[0]

    # the same rows project() used, to break the projection into its terms
    rows = service._pair_rows(player, args.to if target is not None else None)
    if args.age is not None:
        rows["age"] = float(args.age)
    hist = history_features(service.ps, rows)
    for m in METRICS:
        rows[f"trend_{m}"] = hist[f"trend_{m}"].to_numpy()
    parts = service.model.contributions(rows, hist)

    age = f"age {p['age']:.0f}" if pd.notna(p["age"]) else "age unknown"
    print(f"{p['name']}  |  {hit['team']} ({league(hit['league'])})  |  {p['group']}  |  {age}  |  "
          f"{int(hit['minutes'])} min in {season_label(int(hit['season']))}")
    if target is not None and p["team_next"] != hit["team"]:
        delo = float(target["elo"]) - float(hit["elo"])
        print(f"Scenario: moves to {p['team_next']} ({league(p['league_next'])}), team strength {delo:+.0f} Elo")
    else:
        print("Scenario: stays at his club")
    mins_note = " (your input)" if args.minutes is not None else " (minutes model)"
    print(f"Expected minutes next season: {p['expected_minutes']:.0f}{mins_note}")
    value = p.get("market_value_eur")
    if pd.notna(value):
        print(f"Market value: EUR {value / 1e6:.1f}M")
    print()

    head = f"{'per 90':<11}{'last':>7}{'ability':>9}" + "".join(f"{t:>8}" for t in TERMS) + f"{'next':>8}  80% range"
    print(head)
    print("-" * len(head))
    for m in METRICS:
        c = parts[m].iloc[0]
        effects = "".join(f"{pct(float(c.get(t, 0.0))):>8}" for t in TERMS)
        print(f"{m:<11}{float(hit[f'{m}_p90']):>7.2f}{float(c['ability']):>9.2f}{effects}"
              f"{p[f'pred_{m}']:>8.2f}  {p[f'lo_{m}']:.2f}-{p[f'hi_{m}']:.2f}")
    print("\nnext = ability x (1 + sum of the term columns). ability = his recency-weighted level, pulled toward "
          "his position's average when he has few minutes.\nleague = league tax, team = change in team strength, "
          "usage = change in his share of the team's output, age = position age curve, trend = his own recent "
          "trajectory, drift = league-wide change year to year.")


def cmd_role(service: ForecastService, args) -> None:
    key = args.role if args.role in ROLES else role_from_text(args.role)
    if key is None:
        sys.exit(f"Unknown role '{args.role}'. See:  python -m forecast.predict roles")
    role = ROLES[key]
    target = resolve_club(service, args.to)
    leagues = args.leagues or None
    if leagues:
        bad = [lg for lg in leagues if lg not in LEAGUES]
        if bad:
            sys.exit(f"Unknown league(s) {bad}. Use: {', '.join(LEAGUES)}")
    df = service.candidates(key, n=args.n, to_team=args.to if target is not None else None, max_age=args.max_age,
                            max_value_eur=args.max_value * 1e6 if args.max_value else None,
                            min_minutes=args.min_minutes, leagues=leagues, exclude=args.exclude or ())
    print(f"Role: {role.label}" + (" (proxy: key skills not in free data)" if role.proxy else "")
          + f"  |  weights: " + ", ".join(f"{m} {w:g}" for m, w in role.weights.items()))
    if target is not None:
        print(f"Projected at: {target['team']} ({league(target['league'])})")
    if df.empty:
        print("\nNo players pass these filters. Loosen --max-age, --max-value or --min-minutes.")
        return
    shown = [m for m in role.weights][:3]
    head = (f"{'#':>3}  {'player':<24}{'club':<20}{'age':>4}{'min':>6}{'role':>6}{'gap':>6}{'EUR M':>7}"
            + "".join(f"{m[:9]:>10}" for m in shown))
    print("\n" + head)
    print("-" * len(head))
    for i, r in df.iterrows():
        age = f"{r['age']:.0f}" if pd.notna(r["age"]) else "?"
        val = f"{r['market_value_eur'] / 1e6:.1f}" if pd.notna(r["market_value_eur"]) else "?"
        gap = f"{r['value_gap']:+.1f}" if pd.notna(r["value_gap"]) else "?"
        print(f"{i + 1:>3}  {str(r['name'])[:23]:<24}{str(r['team'])[:19]:<20}{age:>4}{r['minutes']:>6.0f}"
              f"{r['role_score']:>6.2f}{gap:>6}{val:>7}" + "".join(f"{r[f'pred_{m}']:>10.2f}" for m in shown))
    print("\nrole = projected fit to the role next season (0 = average player in it, in standard deviations). "
          "gap = value gap: positive means cheap for that projected output (the hidden-talent signal).\n"
          f"Metric columns are projected per-90 next season. Details for one player:  "
          f"python -m forecast.predict player \"{df.iloc[0]['name']}\"" + (f" --to \"{args.to}\"" if args.to else ""))


def cmd_roles(service: ForecastService, args) -> None:
    for r in ROLES.values():
        flag = "  (proxy)" if r.proxy else ""
        print(f"{r.key:<20} {r.label}{flag}\n{'':<20} positions {', '.join(r.groups)}; weights "
              + ", ".join(f"{m} {w:g}" for m, w in r.weights.items()))
    print("\nYou can also describe the role in words, e.g.  role \"a Stones replacement\"  or  role \"inverted winger\".")


def cmd_clubs(service: ForecastService, args) -> None:
    tt = clubs_table(service)
    if args.league:
        tt = tt[tt["league"] == args.league]
    print("Clubs in each league's latest season, strongest first (Elo at season start)\n")
    for _, r in tt.iterrows():
        print(f"{r['elo']:>7.0f}  {r['team']:<28}{league(r['league'])} {season_label(int(r['season']))}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m forecast.predict", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="path to a saved model (default: the one forecast.build wrote)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("player", help="project one player, optionally at another club")
    p.add_argument("name")
    p.add_argument("--team", help="his current club, to tell namesakes apart")
    p.add_argument("--to", help="club he moves to (default: he stays)")
    p.add_argument("--age", type=float, help="what if he were this age")
    p.add_argument("--minutes", type=float, help="what if he played this many league minutes next season")

    r = sub.add_parser("role", help="rank players for a role")
    r.add_argument("role", help="a role key (see 'roles') or a description such as 'a Stones replacement'")
    r.add_argument("--to", help="club to project them at (sets league tax, team strength, usage)")
    r.add_argument("--max-age", type=float)
    r.add_argument("--max-value", type=float, help="max market value, EUR millions")
    r.add_argument("--min-minutes", type=int, default=900)
    r.add_argument("--leagues", nargs="+", metavar="LEAGUE", help=" ".join(LEAGUES))
    r.add_argument("--exclude", nargs="+", metavar="NAME", help="players to leave out (the obvious picks)")
    r.add_argument("-n", type=int, default=10)

    sub.add_parser("roles", help="list role profiles")
    c = sub.add_parser("clubs", help="list clubs and their strength")
    c.add_argument("--league", choices=LEAGUES)

    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")      # accented names on a Windows console
    service = load(args.model)
    {"player": cmd_player, "role": cmd_role, "roles": cmd_roles, "clubs": cmd_clubs}[args.cmd](service, args)


if __name__ == "__main__":
    main()
