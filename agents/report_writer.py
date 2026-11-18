"""
Step 5: Report Writer
=====================
Joins candidates with their character and financial results by player_id,
scores them deterministically, and asks Gemini for all executive summaries in
ONE batched structured call (instead of one call per player).
"""

import json
import logging
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from analysis.player_comparison import PlayerComparisonEngine, cosine_similarity
from analysis.tactical_fit import statistical_rating, tactical_fit
from config import settings
from core import llm
from core.context import get_context
from core.tracing import traced_node
from data_sources.understat_source import position_family
from models.llm_schemas import ExecutiveSummaries, SearchConstraints
from models.scouting_report import CharacterReport, FinancialAssessment, ScoutingReport

logger = logging.getLogger(__name__)

WEIGHTS = {"tactical": 0.30, "statistical": 0.25, "character": 0.20, "financial": 0.15, "fitness": 0.10}

SUMMARY_SYSTEM_PROMPT = """You are a Premier League chief scout.
For each player in the JSON, write a 3-4 sentence executive summary: tactical fit for the
stated system, the #1 strength, the #1 risk, and a clear recommendation consistent with
the verdict. Use only the facts provided; do not add stats, injuries or history.
Return one summary per player_id."""

KEY_STATS = ["minutes_played", "goals", "assists", "npxG_p90", "xA_p90", "key_passes_p90",
             "xGChain_p90", "xGBuildup_p90"]


@traced_node("report_writer")
async def write_reports(state: Dict) -> Dict:
    c = SearchConstraints.model_validate(state["constraints"])
    candidates = state.get("candidates", [])
    chars = state.get("character_reports", {})
    fins = state.get("financial_assessments", {})
    family = position_family(c.position)
    errors: List[str] = []

    engine = PlayerComparisonEngine()
    for cand in candidates:
        engine.add_player(cand["player_id"], cand["name"], cand)

    benchmark = await get_context().find_player(c.benchmark_player) if c.benchmark_player else None

    reports: List[ScoutingReport] = []
    for cand in candidates:
        pid = cand["player_id"]
        char = CharacterReport.model_validate(chars.get(pid) or {"player_id": pid, "player_name": cand["name"]})
        fin = FinancialAssessment.model_validate(fins.get(pid) or {"player_id": pid, "player_name": cand["name"]})

        fit, coverage = tactical_fit(cand, c.tactical_system)
        stat = statistical_rating(cand, family)
        overall = round(
            fit * WEIGHTS["tactical"]
            + stat * WEIGHTS["statistical"]
            + char.attitude_score * 10 * WEIGHTS["character"]
            + fin.value_for_money_score * WEIGHTS["financial"]
            + (10 - char.injury_risk_score) * 10 * WEIGHTS["fitness"], 1)

        reports.append(ScoutingReport(
            player_id=pid,
            player_name=cand["name"],
            position=c.position,
            understat_position=cand.get("understat_position", ""),
            age=cand.get("age"),
            team=cand.get("team", ""),
            league=cand.get("league", ""),
            season=cand.get("season", ""),
            tactical_fit_score=fit,
            tactical_fit_coverage=coverage,
            statistical_rating=stat,
            character_score=round(char.attitude_score, 1),
            financial_viability=fin.value_for_money_score,
            injury_risk=char.injury_risk_score,
            overall_score=overall,
            verdict=verdict(overall, char.injury_risk_score, fin.transfer_feasibility),
            strengths=strengths(cand),
            weaknesses=weaknesses(cand, family),
            risk_factors=risks(char, fin),
            similar_players=[p["name"] for p in engine.find_similar(cand, n=3, exclude_id=pid)],
            benchmark_comparison={benchmark["name"]: cosine_similarity(cand, benchmark)} if benchmark else {},
            key_stats={k: cand.get(k, 0) for k in KEY_STATS},
            character_report=char,
            financial_assessment=fin,
            data_sources=sorted(set(cand.get("sources", []) + char.sources)),
        ))

    reports.sort(key=lambda r: r.overall_score, reverse=True)

    summaries: Dict[str, str] = {}
    if reports and settings.has_gemini():
        try:
            summaries = await summarise(reports, c)
        except llm.LLMError as e:
            errors.append(f"report_writer: summaries failed, used templates ({e})")
    for r in reports:
        r.executive_summary = summaries.get(r.player_id) or template_summary(r)

    top = reports[0] if reports else None
    msg = (f"Wrote {len(reports)} reports. Top: {top.player_name} ({top.overall_score}/100, {top.verdict})"
           if top else "No verified candidates to report on.")
    return {"scouting_reports": [r.model_dump() for r in reports], "messages": [msg], "errors": errors}


def verdict(overall: float, injury_risk: float, feasibility: str) -> str:
    if feasibility == "Over budget":
        return "NO-GO"
    if overall >= 70 and injury_risk < 5:
        return "GO"
    if overall >= 50:
        return "CAUTION"
    return "NO-GO"


def strengths(s: Dict) -> List[str]:
    out = []
    if s.get("npxG_p90", 0) >= 0.4:
        out.append(f"High-quality chances: {s['npxG_p90']:.2f} npxG/90")
    if s.get("goals_p90", 0) > s.get("xG_p90", 0) * 1.15 and s.get("goals", 0) >= 5:
        out.append("Finishing above expectation (goals > xG)")
    if s.get("xA_p90", 0) >= 0.25:
        out.append(f"Chance creator: {s['xA_p90']:.2f} xA/90")
    if s.get("key_passes_p90", 0) >= 2.0:
        out.append(f"{s['key_passes_p90']:.1f} key passes/90")
    if s.get("xGChain_p90", 0) >= 0.6:
        out.append("Heavily involved in attacking moves (xGChain)")
    if s.get("xGBuildup_p90", 0) >= 0.35:
        out.append("Strong contributor to build-up play")
    if (s.get("ea_pace") or 0) >= 88:
        out.append(f"Elite pace (EA FC {s['ea_pace']})")
    return out[:5] or ["Balanced statistical profile"]


def weaknesses(s: Dict, family: str) -> List[str]:
    out = []
    if s.get("xG_p90", 0) >= 0.2 and s.get("goals_p90", 0) < s.get("xG_p90", 0) * 0.75:
        out.append("Finishing below expectation (goals < xG)")
    if family == "forward" and s.get("npxG_p90", 0) < 0.15 and s.get("xA_p90", 0) < 0.15:
        out.append("Low end product for a forward")
    if s.get("minutes_played", 0) < 900:
        out.append(f"Small sample: {s.get('minutes_played', 0)} minutes")
    if s.get("league_coefficient", 1.0) < 0.85:
        out.append("League adjustment applied: output discounted for PL equivalence")
    return out[:4] or ["No major statistical weaknesses"]


def risks(char: CharacterReport, fin: FinancialAssessment) -> List[str]:
    out = []
    if char.attitude_score < 6:
        out.append("Character concerns in news coverage")
    if char.discipline_issues:
        out.append(f"Discipline: {len(char.discipline_issues)} issue(s)")
    if char.injury_risk_score > 6:
        out.append("High injury risk from recent history")
    elif char.injury_risk_score > 4:
        out.append("Moderate injury concerns")
    if fin.years_remaining is not None and fin.years_remaining < 1.5:
        out.append("Contract near expiry")
    if fin.current_market_value_eur > 70:
        out.append("Premium fee (EUR 70M+)")
    if fin.market_value_is_estimate:
        out.append("Market value is a model estimate, not Transfermarkt")
    if fin.transfer_feasibility == "Over budget":
        out.append("Exceeds budget")
    return out or ["No major risk factors identified"]


async def summarise(reports: List[ScoutingReport], c: SearchConstraints) -> Dict[str, str]:
    facts = [{
        "player_id": r.player_id, "name": r.player_name, "age": r.age, "team": r.team, "league": r.league,
        "verdict": r.verdict, "overall": r.overall_score, "tactical_fit": r.tactical_fit_score,
        "key_stats": r.key_stats, "strengths": r.strengths, "weaknesses": r.weaknesses, "risks": r.risk_factors,
        "market_value_m": r.financial_assessment.current_market_value_eur if r.financial_assessment else None,
    } for r in reports]
    out = await llm.structured(
        ExecutiveSummaries,
        [SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
         HumanMessage(content=json.dumps({"system": c.tactical_system or "general", "position": c.position,
                                          "players": facts}, default=str))],
        name="report_writer.summaries",
        temperature=0.3,
    )
    return {s.player_id: s.summary for s in out.summaries}


def template_summary(r: ScoutingReport) -> str:
    return (f"{r.player_name} ({r.team}) scores {r.overall_score}/100 with a tactical fit of "
            f"{r.tactical_fit_score}/100. Main strength: {r.strengths[0]}. "
            f"Main risk: {r.risk_factors[0]}. Verdict: {r.verdict}.")
