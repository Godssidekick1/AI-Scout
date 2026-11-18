"""
AI Scout evals
==============
Measures the pipeline on a fixed set of golden queries (evals/golden_queries.json).

    python -m evals.run_evals --suite parse            # parser only: cheap, fast
    python -m evals.run_evals --suite pipeline         # full runs
    python -m evals.run_evals --suite pipeline --judge # + LLM-as-judge on summaries
    python -m evals.run_evals --suite pipeline --limit 5 --no-cache

Metrics
-------
parse     field accuracy for position family, max age, budget, system, leagues, benchmark
pipeline  per query and in aggregate:
          - schema_valid      every report validates against ScoutingReport
          - unique_rate       unique players / reports (1.0 = no duplication)
          - constraint_pass   reports with known age/value that meet max age and budget
          - acceptance_rate   proposals that survived verification (low = agent proposing
                              players that don't exist or don't fit)
          - finder_attempts, latency, LLM calls, tokens, estimated cost
judge     faithfulness (1-5) of each executive summary to the facts it was given

Results are written to evals/results/<timestamp>-<suite>.json and printed as a table.
Needs GOOGLE_API_KEY (and network access to the data sources).
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from agents.request_parser import parse_request  # noqa: E402
from agents.scout_workflow import arun_scout  # noqa: E402
from analysis.tactical_fit import system_key  # noqa: E402
from config import settings  # noqa: E402
from core import llm  # noqa: E402
from data_sources.understat_source import position_family  # noqa: E402
from models.llm_schemas import SummaryJudgement  # noqa: E402
from models.scouting_report import ScoutingReport  # noqa: E402

HERE = Path(__file__).parent
JUDGE_PROMPT = """You grade an executive summary written by a scouting system.
Given the FACTS it was allowed to use and the SUMMARY, rate faithfulness 1-5
(5 = every claim supported by the facts; 1 = invents stats, injuries or history)
and say whether it ends with a clear recommendation."""


def load_queries(limit: int = 0) -> List[Dict]:
    qs = json.loads((HERE / "golden_queries.json").read_text())
    return qs[:limit] if limit else qs


# ── parse suite ────────────────────────────────────────────────
async def eval_parse(q: Dict) -> Dict:
    out = await parse_request({"user_request": q["query"]})
    c, exp = out["constraints"], q["expect"]
    checks = {"family": position_family(c["position"]) == exp["family"]}
    if "max_age" in exp:
        checks["max_age"] = c["max_age"] == exp["max_age"]
    if "budget_max_m" in exp:
        checks["budget"] = (c["budget_max_m"] or None) == exp["budget_max_m"]
    if "system" in exp:
        checks["system"] = system_key(c["tactical_system"]) == exp["system"]
    if "leagues" in exp:
        checks["leagues"] = sorted(c["target_leagues"]) == sorted(exp["leagues"])
    if "benchmark" in exp:
        checks["benchmark"] = exp["benchmark"].lower() in (c["benchmark_player"] or "").lower()
    return {"id": q["id"], "checks": checks, "accuracy": sum(checks.values()) / len(checks), "parsed": c}


# ── pipeline suite ─────────────────────────────────────────────
async def eval_pipeline(q: Dict, judge: bool) -> Dict:
    t0 = time.perf_counter()
    res = await arun_scout(q["query"])
    latency = time.perf_counter() - t0
    reports, c = res["reports"], res["constraints"]

    valid = 0
    for r in reports:
        try:
            ScoutingReport.model_validate(r)
            valid += 1
        except Exception:  # noqa: BLE001
            pass

    checked = passed = 0
    for r in reports:
        value = (r.get("financial_assessment") or {}).get("current_market_value_eur")
        estimate = (r.get("financial_assessment") or {}).get("market_value_is_estimate")
        ok, known = True, False
        if c.get("max_age") and r.get("age") is not None:
            known, ok = True, ok and r["age"] <= c["max_age"]
        if c.get("budget_max_m") and value and not estimate:
            known, ok = True, ok and value <= c["budget_max_m"]
        checked += known
        passed += known and ok

    proposed = res["candidates_verified"] + len(res["rejected"])
    row = {
        "id": q["id"],
        "reports": len(reports),
        "schema_valid": valid / len(reports) if reports else None,
        "unique_rate": len({r["player_id"] for r in reports}) / len(reports) if reports else None,
        "constraint_pass": passed / checked if checked else None,
        "acceptance_rate": res["candidates_verified"] / proposed if proposed else None,
        "finder_attempts": res["finder_attempts"],
        "latency_s": round(latency, 2),
        "llm_calls": res["trace"]["llm_calls"],
        "tokens": res["trace"]["input_tokens"] + res["trace"]["output_tokens"],
        "cost_usd": res["trace"]["est_cost_usd"],
        "errors": res["errors"],
    }
    if judge and reports:
        row["faithfulness"] = await judge_summaries(reports[:3])
    return row


async def judge_summaries(reports: List[Dict]) -> float:
    scores = []
    for r in reports:
        facts = {k: r.get(k) for k in ("player_name", "age", "team", "league", "verdict", "overall_score",
                                       "tactical_fit_score", "key_stats", "strengths", "weaknesses", "risk_factors")}
        j = await llm.structured(
            SummaryJudgement,
            [SystemMessage(content=JUDGE_PROMPT),
             HumanMessage(content=f"FACTS:\n{json.dumps(facts, default=str)}\n\nSUMMARY:\n{r['executive_summary']}")],
            name="eval.judge", temperature=0.0)
        scores.append(j.faithfulness)
    return statistics.mean(scores)


# ── reporting ──────────────────────────────────────────────────
def mean(rows: List[Dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.mean(vals), 3) if vals else None


def summarise(suite: str, rows: List[Dict]) -> Dict:
    if suite == "parse":
        fields: Dict[str, List[bool]] = {}
        for r in rows:
            for k, v in r["checks"].items():
                fields.setdefault(k, []).append(v)
        return {"queries": len(rows), "overall_accuracy": mean(rows, "accuracy"),
                "field_accuracy": {k: round(sum(v) / len(v), 3) for k, v in fields.items()}}
    keys = ["schema_valid", "unique_rate", "constraint_pass", "acceptance_rate", "finder_attempts",
            "latency_s", "llm_calls", "tokens", "cost_usd", "faithfulness"]
    out = {"queries": len(rows), **{k: mean(rows, k) for k in keys}}
    lat = sorted(r["latency_s"] for r in rows)
    out["latency_p90_s"] = lat[int(0.9 * (len(lat) - 1))] if lat else None
    out["runs_with_errors"] = sum(1 for r in rows if r["errors"])
    return out


async def main_async(args) -> None:
    if not settings.has_gemini():
        sys.exit("Set GOOGLE_API_KEY to run evals.")
    if args.no_cache:
        settings.LLM_CACHE = False
    rows = []
    for q in load_queries(args.limit):
        row = await (eval_parse(q) if args.suite == "parse" else eval_pipeline(q, args.judge))
        rows.append(row)
        shown = {k: v for k, v in row.items() if k not in ("parsed", "errors")}
        print(json.dumps(shown, default=str))

    summary = summarise(args.suite, rows)
    print("\nSUMMARY\n" + "\n".join(f"  {k:<18} {v}" for k, v in summary.items()))
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.suite}.json"
    path.write_text(json.dumps({"suite": args.suite, "model": settings.GEMINI_MODEL, "summary": summary,
                                "rows": rows}, indent=2, default=str))
    print(f"\nSaved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["parse", "pipeline"], default="parse")
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge on summaries (pipeline suite)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
