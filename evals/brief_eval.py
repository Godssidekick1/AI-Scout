"""
Brief eval: does a scouting request turn into the right search, and do the results obey it?
===========================================================================================
    python -m evals.brief_eval              # Gemini reads each brief (needs GOOGLE_API_KEY)
    python -m evals.brief_eval --no-llm     # the rule-based reader
    python -m evals.brief_eval --only madueke-arsenal -v

Runs every brief in evals/briefs.json against the saved model (python -m forecast.build).
Two kinds of check, reported separately because they fail for different reasons:

reading   what the brief SHOULD mean (hand-written in briefs.json): positions, side, club,
          reference player, ages, budget, leagues, players to beat, squad role, asks the
          data cannot measure. Failures here are the reader's fault (Gemini or rules).
results   whether every returned player obeys the brief AS READ: position from Transfermarkt,
          side, age range, budget, leagues, not already at the club, minutes, beats the
          named player's projection. Failures here are the search code's fault, and should
          be zero whatever the reader did.

Writes evals/results/briefs_<date>.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import settings
from forecast import brief as briefs
from forecast.config import POSITIONS, SIDES
from forecast.service import ForecastService, _plain

HERE = Path(__file__).resolve().parent
Check = Tuple[str, bool, str]          # (name, passed, detail)


def reading_checks(b: briefs.Brief, expect: Dict) -> List[Check]:
    out: List[Check] = []

    def check(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    if "positions" in expect:
        check("positions", set(expect["positions"]) <= set(b.positions or briefs.ROLE_POSITIONS.get(b.role or "", [])),
              f"read {b.positions or b.role}")
    if "side" in expect:
        check("side", b.side == expect["side"], f"read {b.side}")
    if "side_strict" in expect:
        check("side_strict", b.side_strict == expect["side_strict"], f"read {b.side_strict}")
    if "to_team" in expect:
        check("club", b.to_team == expect["to_team"], f"read {b.to_team}")
    if "anchor" in expect:
        name = b.replace["name"] if b.replace is not None else None
        check("reference player", name is not None and _plain(expect["anchor"]) in _plain(name),
              f"read {name or b.replace_query}")
    if "anchor_kind" in expect:
        check("replace vs like", b.anchor == expect["anchor_kind"], f"read {b.anchor}")
    for key, attr in (("min_age", "min_age"), ("max_age", "max_age"), ("min_goals", "min_goals")):
        if key in expect:
            check(key, getattr(b, attr) == expect[key], f"read {getattr(b, attr)}")
    if "max_value_m" in expect:
        got = b.max_value_eur / 1e6 if b.max_value_eur else None
        check("budget", got == expect["max_value_m"], f"read {got}")
    if "leagues_include" in expect:
        check("leagues", set(b.leagues or []) == set(expect["leagues_include"]), f"read {b.leagues}")
    if "leagues_exclude" in expect:
        check("leagues excluded", b.leagues is not None and not set(b.leagues) & set(expect["leagues_exclude"]),
              f"read {b.leagues}")
    if "beat" in expect:
        got = [(p["name"], m) for p, m in b.beat]
        ok = all(any(_plain(n) in _plain(g) and m == gm for g, gm in got) for n, m in expect["beat"])
        check("players to beat", ok, f"read {got}")
    if "squad_role" in expect:
        check("squad role", b.squad_role == expect["squad_role"], f"read {b.squad_role}")
    if "sort" in expect:
        check("sort", b.sort == expect["sort"], f"read {b.sort}")
    if "min_minutes_at_least" in expect:
        check("minutes", b.min_minutes >= expect["min_minutes_at_least"], f"read {b.min_minutes}")
    if "emphasis" in expect:
        want = {briefs.EMPHASIS_KEYS[e] for e in expect["emphasis"]}
        check("emphasis", want <= set(b.emphasis_labels), f"read {b.emphasis_labels}")
    if "also_positions" in expect:
        check("also plays", set(expect["also_positions"]) <= set(b.also_positions), f"read {b.also_positions}")
    if "experience" in expect:
        check("league experience", set(b.experience_leagues) == set(expect["experience"]), f"read {b.experience_leagues}")
    if "penalty_taker" in expect:
        check("penalty taker", b.penalty_taker == expect["penalty_taker"], f"read {b.penalty_taker}")
    if "partner" in expect:
        got = b.partner["name"] if b.partner is not None else None
        check("partner", got is not None and _plain(expect["partner"]) in _plain(got), f"read {got or b.partner_query}")
    if "min_pass_pct" in expect:
        check("pass completion", b.min_pass_pct is not None and abs(b.min_pass_pct * 100 - expect["min_pass_pct"]) < 0.5,
              f"read {b.min_pass_pct}")
    if "positions_include" in expect:
        check("positions include", set(expect["positions_include"]) <= set(b.positions), f"read {b.positions}")
    if expect.get("max_age_none"):
        check("no age cap invented", b.max_age is None, f"read {b.max_age}")
    if "no_leagues_filter" in expect:
        check("no league filter", not b.leagues, f"read {b.leagues}")
    if "unmeasured_any" in expect:
        got = " ".join(b.unmeasured)
        check("flags unmeasurable asks", any(w in got for w in expect["unmeasured_any"]), f"read {b.unmeasured}")
    return out


def result_checks(b: briefs.Brief, df: pd.DataFrame, service=None) -> List[Check]:
    """Every returned player must satisfy the brief as read. Unknown values are not failures
    (no Transfermarkt match, no age, no value) but are counted. Position is where he started over
    the last two seasons when the build has lineups, else his Transfermarkt main position."""
    out: List[Check] = []
    if df.empty:
        return [("results returned", False, "empty list")]
    positions = b.positions or briefs.ROLE_POSITIONS.get(b.role or "", [])
    allowed = {s for p in positions for s in POSITIONS[p]}
    if b.side and b.side_strict:
        allowed = (allowed & set(SIDES[b.side])) or set(SIDES[b.side])
    sub = df["tm_sub_position"] if "tm_sub_position" in df else pd.Series(None, index=df.index)

    def rule(name, bad, unknown=0):
        detail = f"{int(bad.sum())} of {len(df)} break it" + (f"; {unknown} unknown" if unknown else "")
        out.append((name, not bad.any(), detail))

    if allowed:
        share = df["position_share"] if "position_share" in df else pd.Series(float("nan"), index=df.index)
        by_lineups = share.notna() & (share < 0.5)
        by_main = share.isna() & sub.notna() & ~sub.isin(allowed | {s for p in b.also_positions for s in POSITIONS[p]})
        rule("position (recent starts, else Transfermarkt)", by_lineups | by_main, int((share.isna() & sub.isna()).sum()))
    if b.also_positions and "also_share" in df:
        rule("also plays " + "/".join(b.also_positions), df["also_share"].notna() & (df["also_share"] < briefs.ALSO_SHARE))
    if b.experience_leagues and service is not None:
        rule(f"{b.experience_minutes}+ minutes in " + "/".join(b.experience_leagues),
             service.league_minutes(df["us_id"], b.experience_leagues) < b.experience_minutes)
    if b.min_pass_pct is not None and "def_pass_pct" in df:
        rule(f"pass completion >= {b.min_pass_pct:.0%}", df["def_pass_pct"].isna() | (df["def_pass_pct"] < b.min_pass_pct - 1e-9))
    if b.anchor == "like" and b.replace is not None and "style_similarity" in df:
        rule("style leans toward the reference (score > 50)", df["style_similarity"].notna() & (df["style_similarity"] <= 50))
    if b.penalty_taker and "pens" in df:
        rule("took 2+ penalties", df["pens"].notna() & (df["pens"] < 2))
    if b.min_age is not None:
        rule("min age (unknown fails)", df["age"].isna() | (df["age"] < b.min_age))
    if b.max_age is not None:
        rule("max age (unknown fails)", df["age"].isna() | (df["age"] > b.max_age))
    cap = b.max_value_eur if b.max_value_eur is not None else (briefs.HIDDEN_MAX_VALUE if b.sort == "value" else None)
    if cap is not None:
        v = df["market_value_eur"]
        rule("budget / hidden cap (unknown fails)", v.isna() | (v > cap))
    now_lg = df["current_league"] if "current_league" in df else df["league"]
    now_club = df["current_club"] if "current_club" in df else df["team"]
    if b.leagues:
        rule("leagues (current club)", ~now_lg.isin(b.leagues))
    if b.to_team and b.exclude_own_club:
        rule("not already at the club", (df["team"] == b.to_team) | (now_club == b.to_team))
    rule("minutes", df["minutes"] < b.min_minutes)
    for name, m, bar in b.bars:
        if b.note:            # nobody reached the bar; the list is the closest, by design
            out.append((f"beats {name} on {m}", False, "nobody reaches it; closest shown"))
        else:
            rule(f"beats {name} on {m}", df[f"pred_{m}"] < bar)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-llm", action="store_true", help="use the rule-based reader")
    ap.add_argument("--only", nargs="+", help="brief ids to run")
    ap.add_argument("--file", default="briefs.json",
                    help="brief set in evals/ (briefs_holdout.json: phrasings the reader was not tuned on)")
    ap.add_argument("--model", help="saved model path (default: the build's)")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("-v", "--verbose", action="store_true", help="print each shortlist")
    ap.add_argument("--synthetic-names", action="store_true",
                    help="offline check of the reader: a synthetic build with real club and player names")
    args = ap.parse_args(argv)

    service = ForecastService.load(Path(args.model) if args.model else None)
    if service is None:
        sys.exit("No saved model. Run python -m forecast.build first.")
    if service.synthetic and args.synthetic_names:
        from forecast import synthetic
        service.ps, service.profiles = synthetic.with_real_names(service.ps, getattr(service, "profiles", None))
        service.__dict__.pop("_def_profile", None)
        print("Synthetic model with real names: checks the reader and the filters, not football.\n")
    elif service.synthetic:
        print("Warning: SYNTHETIC model. Real names will not be found; build on real data for a meaningful eval "
              "(or pass --synthetic-names).\n")
    settings.ENABLE_VECTOR_INDEX = False
    if args.no_llm:
        settings.GOOGLE_API_KEY = ""
    reader = "rules" if not settings.has_gemini() else "Gemini"

    cases = json.loads((HERE / args.file).read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]

    rows, lines = [], []
    for c in cases:
        b = briefs.understand(c["text"], service)
        try:
            df, _, _ = briefs.search(b, service, n=args.n)
            err = ""
        except Exception as e:  # noqa: BLE001 - a crash is a failed case, not a failed eval
            df, err = pd.DataFrame(), f"{type(e).__name__}: {e}"
        read = reading_checks(b, c["expect"])
        if c["expect"].get("no_search"):      # e.g. goalkeepers: the right answer is to refuse
            res = [("declines what the data cannot do", bool(err), err or f"returned {len(df)} players")]
        else:
            res = result_checks(b, df, service) if not err else [("search ran", False, err)]
        rows.append({"id": c["id"], "read_ok": sum(ok for _, ok, _ in read), "read_n": len(read),
                     "res_ok": sum(ok for _, ok, _ in res), "res_n": len(res), "n": len(df)})
        lines += [f"### {c['id']}", "", f"> {c['text']}", "", f"Reading ({b.parser}): " + " | ".join(b.describe()), ""]
        for kind, checks in (("reading", read), ("results", res)):
            for name, ok, detail in checks:
                lines.append(f"- {'PASS' if ok else 'FAIL'} {kind}: {name} ({detail})")
        if not df.empty:
            lines.append("- top: " + ", ".join(f"{r['name']} ({r['team']})" for _, r in df.head(5).iterrows()))
        lines.append("")
        if args.verbose:
            print("\n".join(lines[-(len(read) + len(res) + 6):]))

    if briefs._GEMINI_BROKEN:
        reader = "rules (Gemini could not run)"
        print(f"Gemini could not run, so these results are for the rule-based reader:\n  {briefs._GEMINI_BROKEN}\n")
    tbl = pd.DataFrame(rows)
    read_rate = tbl["read_ok"].sum() / max(tbl["read_n"].sum(), 1)
    res_rate = tbl["res_ok"].sum() / max(tbl["res_n"].sum(), 1)
    clean = int(((tbl["read_ok"] == tbl["read_n"]) & (tbl["res_ok"] == tbl["res_n"])).sum())
    head = [f"# Brief eval ({dt.date.today().isoformat()}, reader: {reader}, "
            f"{'synthetic' if service.synthetic else 'real'} model)", "",
            f"- briefs fully correct: {clean} of {len(tbl)}",
            f"- reading checks passed: {read_rate:.0%} ({tbl['read_ok'].sum()} of {tbl['read_n'].sum()})",
            f"- result checks passed: {res_rate:.0%} ({tbl['res_ok'].sum()} of {tbl['res_n'].sum()})", "",
            "| brief | reading | results | players |", "| --- | --- | --- | --- |"]
    head += [f"| {r['id']} | {r['read_ok']}/{r['read_n']} | {r['res_ok']}/{r['res_n']} | {r['n']} |" for r in rows]
    text = "\n".join(head + ["", "## Detail", ""] + lines) + "\n"
    out = HERE / "results" / f"{Path(args.file).stem}_{dt.date.today().isoformat()}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print("\n".join(head[:6]))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
