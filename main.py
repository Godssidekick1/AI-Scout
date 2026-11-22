"""
AI Scout — command line
=======================

    python main.py "Find an inverted winger for Slot's Liverpool 4-3-3, under 25, budget 60M"
    python main.py --format json "Striker under 23 from La Liga or Bundesliga"
    python main.py --check        # which keys and sources are configured
    python main.py --graph        # print the LangGraph as Mermaid
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import settings  # noqa: E402

VERDICT_ICON = {"GO": "✅", "CAUTION": "⚠️", "NO-GO": "❌"}


def setup_logging(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def check_config() -> None:
    print("\nConfiguration")
    rows = [
        ("Gemini (agent + LLM steps)", settings.has_gemini(), "set GOOGLE_API_KEY; without it a deterministic baseline runs"),
        ("Serper (news)", settings.has_serper(), "set SERPER_API_KEY; character checks fall back to card counts"),
        ("Understat (stats, no key)", True, ""),
        ("Transfermarkt (value, no key)", settings.ENABLE_TRANSFERMARKT, "disabled"),
        ("EA FC / FUTWIZ (no key)", settings.ENABLE_EA_FC, "disabled"),
    ]
    for name, ok, hint in rows:
        print(f"  {'✅' if ok else '❌'} {name:<32} {'' if ok else hint}")
    print(f"  Model: {settings.GEMINI_MODEL} (bulk: {settings.GEMINI_FAST_MODEL}) | Understat season: {settings.UNDERSTAT_SEASON}")


def print_report_text(result: dict, top_n: int = 5) -> None:
    reports = result.get("reports", [])
    print("\n" + "=" * 80)
    print("AI SCOUT — SCOUTING REPORT")
    print("=" * 80)
    for msg in result.get("messages", []):
        print(f"  · {msg}")

    if not reports:
        print("\nNo verified candidates. Try a broader role, a higher budget or more leagues.")
    for i, r in enumerate(reports[:top_n], 1):
        fin = r.get("financial_assessment") or {}
        age = r.get("age") if r.get("age") is not None else "?"
        print(f"\n{'─' * 80}\n#{i}  {r['player_name']} ({age})  {r['team']}  |  {r['league']}  |  {r['season']}")
        print(f"{'─' * 80}")
        print(f"  Overall {r['overall_score']}/100  {VERDICT_ICON.get(r['verdict'], '')} {r['verdict']}")
        print(f"  Tactical fit {r['tactical_fit_score']}/100 (data coverage {int(r['tactical_fit_coverage'] * 100)}%)"
              f" | Stats {r['statistical_rating']}/100 | Character {r['character_score']}/10"
              f" | Value {r['financial_viability']}/100 | Injury risk {r['injury_risk']}/10")
        ks = r.get("key_stats", {})
        print(f"  {ks.get('minutes_played', 0):.0f} min, {ks.get('goals', 0):.0f} G, {ks.get('assists', 0):.0f} A | "
              f"npxG/90 {ks.get('npxG_p90', 0):.2f}  xA/90 {ks.get('xA_p90', 0):.2f}  "
              f"xGChain/90 {ks.get('xGChain_p90', 0):.2f}  (PL-adjusted)")
        pj = r.get("projection") or {}
        if pj:
            gap = r.get("value_gap")
            gap_txt = f" | value gap {gap:+.0%}" if gap is not None else ""
            print(f"  Next season (model): npxG/90 {pj.get('pred_npxG', 0):.2f} [{pj.get('lo_npxG', 0):.2f}-{pj.get('hi_npxG', 0):.2f}]"
                  f"  xA/90 {pj.get('pred_xA', 0):.2f} [{pj.get('lo_xA', 0):.2f}-{pj.get('hi_xA', 0):.2f}]"
                  f"  xGBuildup/90 {pj.get('pred_xGBuildup', 0):.2f}  ~{pj.get('expected_minutes', 0):.0f} min"
                  f" | role score {r.get('role_score')}{gap_txt}")
        for label, key in (("Strengths", "strengths"), ("Weaknesses", "weaknesses"), ("Risks", "risk_factors")):
            if r.get(key):
                print(f"  {label}: " + "; ".join(r[key]))
        if fin:
            est = " (estimate)" if fin.get("market_value_is_estimate") else ""
            print(f"  Value EUR {fin.get('current_market_value_eur')}M{est} | contract {fin.get('contract_end_date') or '?'}"
                  f" | feasibility {fin.get('transfer_feasibility')}")
        if r.get("similar_players"):
            print(f"  Similar in this pool: {', '.join(r['similar_players'])}")
        for name, sim in (r.get("benchmark_comparison") or {}).items():
            print(f"  Similarity to {name}: {sim}%")
        print(f"\n  {r.get('executive_summary', '')}")
        print(f"  Sources: {', '.join(r.get('data_sources', []))}")

    if result.get("rejected"):
        print(f"\nRejected by the verifier ({len(result['rejected'])}):")
        for rej in result["rejected"][:10]:
            print(f"  ✗ {rej['name']}: {rej['reason']}")

    t = result.get("trace", {})
    if t:
        print("\nRun trace")
        print(f"  Total {t['total_ms'] / 1000:.1f}s | finder attempts {result.get('finder_attempts')} | "
              f"LLM calls {t['llm_calls']} (+{t['llm_cache_hits']} cached, {t['llm_retries']} retries) | "
              f"tokens {t['input_tokens']} in / {t['output_tokens']} out | est. ${t['est_cost_usd']:.4f} | "
              f"tool calls {t['tool_calls']} | source requests {t['source_requests']}")
        print("  " + " | ".join(f"{k} {v / 1000:.1f}s" for k, v in t["node_ms"].items()))
    if result.get("errors"):
        print("\nDegraded steps:")
        for e in result["errors"]:
            print(f"  ! {e}")
    print()


def main():
    parser = argparse.ArgumentParser(description="AI Scout — agentic football scouting")
    parser.add_argument("query", nargs="?", help="Scouting query in natural language")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--check", action="store_true", help="Show configuration status")
    parser.add_argument("--graph", action="store_true", help="Print the workflow graph as Mermaid")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the LLM response cache")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.check:
        check_config()
        return

    from agents.scout_workflow import build_scout_workflow, run_scout

    if args.graph:
        print(build_scout_workflow().get_graph().draw_mermaid())
        return
    if not args.query:
        parser.print_help()
        return
    if args.no_cache:
        settings.LLM_CACHE = False

    result = run_scout(args.query)
    if args.format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report_text(result)


if __name__ == "__main__":
    main()
