"""
Try the forecast model yourself
===============================
Works from the model that `python -m forecast.build` saved; no API keys needed.

    python -m forecast.predict                # type requests in plain English (Gemini if GOOGLE_API_KEY is set)
    python -m forecast.predict find --replace "Rodri" "younger, progressive passer, under 30m"
    python -m forecast.predict find "replace Rodri, younger, under 30m, outside the Premier League"
    python -m forecast.predict find "creative winger under 23 for Arsenal, hidden gems"
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

from forecast.config import LEAGUES, METRICS, POSITION_SHORT, season_label
from forecast.model import history_features
from forecast.roles import DEF_KEYS, ROLES, role_from_text
from forecast.service import ForecastService, artifact_path
from forecast.table import team_totals

LEAGUE_NAMES = {"premier_league": "Premier League", "la_liga": "La Liga", "bundesliga": "Bundesliga",
                "serie_a": "Serie A", "ligue_1": "Ligue 1"}
TERMS = ("league", "team", "style", "usage", "age", "trend", "drift")


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
          "his position's average when he has few minutes.\nleague = league tax, team = change in team strength, style = new team's output per match vs his old team's, "
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
              f"{r['role_score']:>6.2f}{gap:>6}{val:>7}" + "".join(f"{metric_value(r, m):>10}" for m in shown))
    print("\nrole = projected fit to the role next season (0 = average player in it, in standard deviations). "
          "gap = value gap: positive means cheap for that projected output (the hidden-talent signal).\n"
          f"Metric columns are projected per-90 next season. Details for one player:  "
          f"python -m forecast.predict player \"{df.iloc[0]['name']}\"" + (f" --to \"{args.to}\"" if args.to else ""))


def metric_value(r, m: str) -> str:
    """Projected per-90 for a model metric; the recent observed rate for a defensive one."""
    if m in DEF_KEYS:
        from forecast.roles import def_column
        col, _ = def_column(m)
        if float(r.get("def_minutes_used", 0) or 0) <= 0 or pd.isna(r.get(col, np.nan)):
            return "-"
        v = float(r[col])
        return f"{v:.0%}" if m.endswith("_pct") else f"{v:.2f}"
    return f"{r[f'pred_{m}']:.2f}"


LEAGUE_SHORT = {"premier_league": "EPL", "la_liga": "LaLiga", "bundesliga": "Bund", "serie_a": "SerieA", "ligue_1": "L1"}
UNMEASURED_WORDS = {"tackl": "tackling", "intercept": "interceptions", "aerial": "aerials", "header": "heading",
                    "defensive": "defending", "defending": "defending", "duel": "duels", "pace": "pace",
                    "fast": "pace", "quick": "pace", "physical": "physicality", "tall": "height", "in the air": "aerials",
                    "track back": "tracking back", "tracks back": "tracking back", "tracking back": "tracking back",
                    "work rate": "work rate", "hard-working": "work rate", "hard working": "work rate",
                    "adaptab": "tactical versatility", "versatil": "tactical versatility",
                    "any formation": "tactical versatility", "formations": "tactical versatility",
                    "any system": "tactical versatility", "multiple positions": "tactical versatility",
                    "two-footed": "weaker foot", "leader": "leadership", "mentality": "mentality",
                    "attitude": "attitude (the agent's news check covers this)",
                    "pressing": "pressing", "work rate": "work rate", "saves": "shot-stopping",
                    "shot-stopp": "shot-stopping", "keeper": "goalkeeping", "goalkeeper": "goalkeeping"}


def run_brief(service: ForecastService, text: str, replace: Optional[str] = None, n: int = 10,
              overrides: Optional[dict] = None, exclude=()) -> None:
    """Parse a brief, apply any explicit overrides, print what was understood and the shortlist."""
    from forecast import brief as briefs
    b = briefs.understand(text, service, replace=replace)
    for k, v in (overrides or {}).items():
        if v is not None:
            setattr(b, k, v)
    if (overrides or {}).get("to_team"):
        b.to_team = resolve_club(service, overrides["to_team"])["team"]

    if b.interpretation:
        print(f"Reading ({b.parser}): {b.interpretation}")
    elif not b.parser.startswith("rules") or "failed" in b.parser:
        print(f"Reading: {b.parser}")
    print("Understood: " + " | ".join(b.describe()))
    if b.unmeasured:
        words = sorted({UNMEASURED_WORDS.get(u, u) for u in b.unmeasured})
        print(f"Not in the free data, so not used: {', '.join(words)} (needs event data)")
    print("Searching...", flush=True)
    try:
        df, him, prof = briefs.search(b, service, n=n, exclude=exclude)
    except ValueError as e:
        print(f"\n{e}")
        return
    print("Ranking on: " + ", ".join(f"{m} {w:.0%}" for m, w in sorted(prof.weights.items(), key=lambda x: -x[1]))
          + (f"  ({prof.note})" if prof.note else ""))
    if df.empty:
        extra = " Goal/assist targets are strict: 15 npxG in a season is elite." if (b.min_goals or b.min_assists) else ""
        print("\nNo players pass these filters. Loosen the age, value, league or minutes limits." + extra)
        return

    for name, m, bar in b.bars:
        print(f"Bar to beat: {name}'s projected {m} per 90 = {bar:.2f}")
    if b.note:
        print("\n" + b.note)
    shown = [m for m, _ in sorted(prof.weights.items(), key=lambda x: -x[1])][:3]
    extra = [c for c in ("proj_goals", "proj_assists") if c in df and getattr(b, "min_" + c[5:]) is not None]
    facts = [(c, h) for c, h, on in (("position_share", "at pos", "position_share" in df),
                                     ("also_share", "also", "also_share" in df),
                                     ("experience_minutes", "exp min", bool(b.experience_leagues)),
                                     ("pens", "pens", b.penalty_taker),
                                     ("def_pass_pct", "pass%", b.min_pass_pct is not None)) if on and c in df]
    vs = him is not None and b.anchor == "replacing"     # 'like X' is a style, not a bar to clear
    score_col = "vs him" if vs else "score"
    head = (f"{'#':>3}  {'player':<23}{'club':<19}{'lg':<7}{'pos':<4}{'age':>4}{'min':>6}{'style':>6}{score_col:>7}"
            f"{'gap':>6}{'EUR M':>7}" + "".join(f"{m[:9]:>10}" for m in shown)
            + "".join(f"{c[5:]:>8}" for c in extra) + "".join(f"{h:>8}" for _, h in facts))
    print("\n" + head)
    print("-" * len(head))

    def line(i, r, score):
        age = f"{r['age']:.0f}" if pd.notna(r["age"]) else "?"
        val = f"{r['market_value_eur'] / 1e6:.1f}" if pd.notna(r["market_value_eur"]) else "?"
        gap = f"{r['value_gap']:+.1f}" if pd.notna(r.get("value_gap", np.nan)) else ""
        sty = f"{r['style_similarity']:.0f}" if pd.notna(r.get("style_similarity", np.nan)) else ""
        pos = POSITION_SHORT.get(r.get("recent_position") or r.get("tm_sub_position"), r.get("group", ""))
        club = str(r.get("current_club") or r["team"])[:17] + ("*" if r.get("moved") else "")
        lg = r.get("current_league") or r["league"]

        def fact(c):
            v = r.get(c, np.nan)
            if pd.isna(v):
                return f"{'':>8}"
            return f"{v:>8.0%}" if c.endswith(("share", "_pct")) else f"{v:>8.0f}"
        return (f"{i:>3}  {str(r['name'])[:22]:<23}{club:<19}{LEAGUE_SHORT.get(lg, str(lg))[:6]:<7}"
                f"{pos:<4}{age:>4}{r['minutes']:>6.0f}{sty:>6}{score:>7}{gap:>6}{val:>7}"
                + "".join(f"{metric_value(r, m):>10}" for m in shown)
                + "".join(f"{r[c]:>8.1f}" if c in r and pd.notna(r[c]) else f"{'':>8}" for c in extra)
                + "".join(fact(c) for c, _ in facts))

    if him is not None:
        print(line("him" if vs else "ref", him, "0.00" if vs else f"{him['role_score']:.2f}"))
    for i, r in df.iterrows():
        score = r["role_score"] - him["role_score"] if vs else r["role_score"]
        print(line(i + 1, r, f"{score:+.2f}" if vs else f"{score:.2f}"))
    at = f" at {b.to_team}" if b.to_team else ""
    print(f"\nstyle = how alike the shape of their output is, each compared with his position's average "
          "(100 = same profile, 50 = unrelated, 0 = opposite). "
          + ("vs him = projected output in this profile next season" + at + ", minus his (0 = as good as him), "
             if vs else "score = projected output in this profile next season" + at
             + " (0 = average in the role), ")
          + "in standard deviations. gap = value gap: positive = cheap for that output. "
            "Metric columns are projected per-90 next season"
          + ("; defensive columns (" + ", ".join(m for m in shown if m in DEF_KEYS) + ") are his observed rates over "
             "the last three covered seasons, shrunk toward his position's average ('-' = no defensive data for him)"
             if any(m in DEF_KEYS for m in shown) else "") + ". pos = where he started most over the last two "
            "seasons (Transfermarkt lineups)."
          + (" * = moved since the last season in the data; club and league are his current ones (Transfermarkt), "
             "his stats are from his old club." if "moved" in df and df["moved"].any() else "")
          + (" at pos = share of his recent starts in the position asked for." if any(c == "position_share" for c, _ in facts) else "")
          + (" pass% = pass completion over his last two seasons with FBref data." if b.min_pass_pct is not None else "")
          + (f" ref = {him['name']}, the style reference, scored the same way." if him is not None and not vs else "")
          + (" exp min = career minutes in that league." if b.experience_leagues else "")
          + (" pens = penalties taken, last two seasons." if b.penalty_taker else "") + "\n"
          f"Drill into one:  python -m forecast.predict player \"{df.iloc[0]['name']}\""
          + (f" --to \"{b.to_team}\"" if b.to_team else ""))


def cmd_find(service: ForecastService, args) -> None:
    text = " ".join(args.brief or [])
    if not text and not args.replace:
        sys.exit("Give a player to replace (--replace) and/or a description, e.g.\n"
                 '  python -m forecast.predict find --replace "Rodri" "younger, progressive passer, under 30m"')
    run_brief(service, text, replace=args.replace, n=args.n, exclude=args.exclude or (),
              overrides={"to_team": args.to, "max_age": args.max_age, "min_age": args.min_age,
                         "max_value_eur": args.max_value * 1e6 if args.max_value else None,
                         "leagues": args.leagues, "min_minutes": args.min_minutes,
                         "sort": "value" if args.hidden else None})


def interactive(service: ForecastService) -> None:
    from config import settings
    mode = ("Gemini reads your request" if settings.has_gemini()
            else "no GOOGLE_API_KEY, so a rule-based reader is used: name a player as 'replace X' or 'like X'")
    print(f"AI Scout: describe the player you want in your own words ({mode}).\n"
          "Examples:  a cb for Chelsea to replace Fofana, good on the ball, under 26, max 40m\n"
          "           goalscoring winger aged 25-30 who can score 15 goals, not from the Premier League\n"
          "           hidden gems: creative midfielders under 23 in Ligue 1\n"
          "Press Enter on an empty line or Ctrl+C to quit.\n")
    while True:
        try:
            text = input("Scout> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            return
        print()
        try:
            run_brief(service, text)
        except KeyboardInterrupt:
            print("\n(stopped)")
        except Exception:  # noqa: BLE001 - keep the prompt alive, show what broke
            import traceback
            traceback.print_exc()
        print("\n" + "=" * 60 + "\n")


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
    ap.add_argument("--chroma", action="store_true",
                    help="retrieve with the ChromaDB index (default here: in-memory numpy, same results)")
    sub = ap.add_subparsers(dest="cmd")

    f = sub.add_parser("find", help="shortlist from a player to replace and/or a description")
    f.add_argument("brief", nargs="*", help="what you're looking for, e.g. 'younger, progressive passer, under 30m'")
    f.add_argument("--replace", help="the player you want to replace")
    f.add_argument("--to", help="club to project at (default: the replaced player's club)")
    f.add_argument("--max-age", type=float)
    f.add_argument("--min-age", type=float)
    f.add_argument("--max-value", type=float, help="EUR millions")
    f.add_argument("--leagues", nargs="+", metavar="LEAGUE", help=" ".join(LEAGUES))
    f.add_argument("--min-minutes", type=int)
    f.add_argument("--hidden", action="store_true", help="sort by value gap (cheap for the projected output)")
    f.add_argument("--exclude", nargs="+", metavar="NAME")
    f.add_argument("-n", type=int, default=10)

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
    try:
        import faulthandler
        faulthandler.enable()                         # a native crash prints where it happened
    except Exception:  # noqa: BLE001 - no real stderr (e.g. under pytest)
        pass
    from config import settings
    settings.ENABLE_VECTOR_INDEX = bool(args.chroma)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")      # accented names on a Windows console
    service = load(args.model)
    if args.cmd is None:
        interactive(service)
        return
    {"find": cmd_find, "player": cmd_player, "role": cmd_role, "roles": cmd_roles,
     "clubs": cmd_clubs}[args.cmd](service, args)


if __name__ == "__main__":
    main()
