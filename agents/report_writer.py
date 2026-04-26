"""
Agent 5: Report Writer ("The Scout")
=====================================
Assembles all data into final scouting reports with tactical fit scores,
risk assessment, and a GO / CAUTION / NO-GO verdict.
"""

import json
import logging
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from analysis.player_comparison import PlayerComparisonEngine
from config import settings

logger = logging.getLogger(__name__)

REPORT_PROMPT = """You are a Premier League chief scout writing a final scouting report.

Player: {player_name} ({age}, {position}) — {team} ({league})

Stats (adjusted for PL equivalence):
{stats_summary}

Character Assessment:
- Attitude: {attitude_score}/10
- Discipline issues: {discipline_issues}
- Injury risk: {injury_risk}/10 ({total_days_injured} days missed in 24 months)

Financial:
- Market value: €{market_value}M
- Contract remaining: {years_remaining} years
- Value-for-money: {vfm_score}/100
- Transfer feasibility: {feasibility}

Tactical system required: {tactical_system}

Write a 3-4 sentence executive summary that:
1. Evaluates tactical fit for the specified system
2. Highlights the #1 strength and #1 risk
3. Makes a clear recommendation

Respond with ONLY the summary text, no JSON or formatting."""


def write_reports(state: Dict) -> Dict:
    """
    Generate final scouting reports for all candidates.
    """
    candidates = state.get("candidate_pool", [])
    character_reports = state.get("character_reports", [])
    financial_assessments = state.get("financial_assessments", [])
    tactical_system = state.get("tactical_system", "General")
    benchmark_player = state.get("benchmark_player")

    comparison = PlayerComparisonEngine()

    # Add all candidates to comparison engine
    for i, c in enumerate(candidates):
        comparison.add_player(
            player_id=c.get("player_id", str(i)),
            name=c.get("name", "Unknown"),
            stats=c,
        )

    reports = []
    for i, candidate in enumerate(candidates):
        char = character_reports[i] if i < len(character_reports) else {}
        fin = financial_assessments[i] if i < len(financial_assessments) else {}

        # Tactical fit score
        tactical_fit = _calculate_tactical_fit(candidate, tactical_system)

        # Statistical rating
        stat_rating = _calculate_statistical_rating(candidate)

        # Character score
        char_score = char.get("attitude_score", 7.5)
        injury_risk = char.get("injury_risk_score", 2.0)

        # Financial viability
        fin_score = fin.get("value_for_money_score", 50.0)

        # Overall weighted score
        overall = (
            tactical_fit * 0.30 +
            stat_rating * 0.25 +
            (char_score * 10) * 0.20 +
            fin_score * 0.15 +
            ((10 - injury_risk) * 10) * 0.10
        )

        # Verdict
        if overall >= 75 and injury_risk < 5:
            verdict = "GO ✅"
        elif overall >= 55:
            verdict = "CAUTION ⚠️"
        else:
            verdict = "NO-GO ❌"

        # Strengths and weaknesses
        strengths = _identify_strengths(candidate)
        weaknesses = _identify_weaknesses(candidate)
        risks = _identify_risks(char, fin, injury_risk)

        # Similar players
        similar = comparison.find_similar(candidate, n=3, exclude_id=candidate.get("player_id", str(i)))

        # Executive summary via LLM
        summary = _generate_summary(candidate, char, fin, tactical_system)

        report = {
            "player_name": candidate.get("name", "Unknown"),
            "position": candidate.get("position", ""),
            "age": candidate.get("age", 0),
            "team": candidate.get("team", ""),
            "league": candidate.get("league", ""),
            "tactical_fit_score": round(tactical_fit, 1),
            "statistical_rating": round(stat_rating, 1),
            "character_score": round(char_score, 1),
            "financial_viability": round(fin_score, 1),
            "injury_risk": round(injury_risk, 1),
            "overall_score": round(overall, 1),
            "verdict": verdict,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "risk_factors": risks,
            "similar_players": [p["name"] for p in similar],
            "executive_summary": summary,
            "data_sources": candidate.get("sources", []),
            "character_report": char,
            "financial_assessment": fin,
        }

        reports.append(report)

    # Sort by overall score
    reports.sort(key=lambda r: r["overall_score"], reverse=True)

    state["scouting_reports"] = state.get("scouting_reports", []) + reports
    state["messages"] = [
        AIMessage(content=f"📋 Generated {len(reports)} scouting reports. "
                          f"Top candidate: {reports[0]['player_name']} "
                          f"({reports[0]['overall_score']}/100 — {reports[0]['verdict']})"
                          if reports else "No reports generated.")
    ]

    return state


def _calculate_tactical_fit(stats: Dict, tactical_system: str) -> float:
    """Calculate how well a player fits the tactical system (0-100)."""
    system = (tactical_system or "general").lower()
    score = 40.0

    if "slot" in system or "4-3-3" in system or "inverted" in system:
        # Arne Slot: progressive, technical, disciplined pressing
        score += stats.get("xT_p90", 0) * 25
        score += stats.get("progressive_carries_p90", 0) * 4
        score += stats.get("pressing_success_pct", 0) * 0.4
        score += stats.get("passes_into_penalty_area_p90", 0) * 5

    elif "guardiola" in system or "possession" in system or "city" in system:
        # Pep: passing, creativity, ball retention
        score += stats.get("pv_plus", 0) * 35
        score += stats.get("shot_creating_actions_p90", 0) * 6
        score += stats.get("progressive_passes_p90", 0) * 3
        score += stats.get("passes_into_penalty_area_p90", 0) * 4

    elif "arteta" in system or "arsenal" in system:
        # Arteta: positional play, versatility, high work rate
        score += stats.get("xT_p90", 0) * 20
        score += stats.get("progressive_carries_p90", 0) * 3
        score += stats.get("tackles_p90", 0) * 4
        score += stats.get("pressing_success_pct", 0) * 0.4

    elif "conte" in system or "3-5-2" in system or "3-4-3" in system:
        # Conte: intensity, physicality, defensive contribution
        score += stats.get("tackles_p90", 0) * 5
        score += stats.get("interceptions_p90", 0) * 4
        score += stats.get("high_intensity_sprints_p90", 0) * 1
        score += stats.get("goals_p90", 0) * 15

    else:
        # General fit
        score += (stats.get("goals_p90", 0) + stats.get("assists_p90", 0)) * 20
        score += stats.get("xT_p90", 0) * 15
        score += stats.get("tackles_p90", 0) * 3

    return min(score, 100.0)


def _calculate_statistical_rating(stats: Dict) -> float:
    """Calculate overall statistical quality (0-100)."""
    rating = (
        stats.get("goals_p90", 0) * 16 +
        stats.get("assists_p90", 0) * 14 +
        stats.get("xG_p90", 0) * 12 +
        stats.get("xA_p90", 0) * 10 +
        stats.get("xT_p90", 0) * 8 +
        stats.get("shot_creating_actions_p90", 0) * 3 +
        stats.get("progressive_carries_p90", 0) * 2 +
        stats.get("tackles_p90", 0) * 2 +
        stats.get("pressing_success_pct", 0) * 0.3
    )
    return min(rating, 100.0)


def _identify_strengths(stats: Dict) -> List[str]:
    """Identify top strengths from statistics."""
    strengths = []
    if stats.get("goals_p90", 0) > 0.5:
        strengths.append("Elite goal-scoring output")
    if stats.get("xG_p90", 0) > 0.4 and stats.get("goals_p90", 0) > stats.get("xG_p90", 0):
        strengths.append("Clinical finisher — outperforms xG")
    if stats.get("assists_p90", 0) > 0.35:
        strengths.append("High-level chance creation")
    if stats.get("xT_p90", 0) > 0.45:
        strengths.append("Exceptional progressive threat")
    if stats.get("progressive_carries_p90", 0) > 5.0:
        strengths.append("Elite ball-carrying ability")
    if stats.get("pressing_success_pct", 0) > 32:
        strengths.append("Effective high pressing")
    if stats.get("tackles_p90", 0) > 2.5:
        strengths.append("Strong defensive contribution")
    if stats.get("high_intensity_sprints_p90", 0) > 22:
        strengths.append("Outstanding physical output")
    if stats.get("ea_overall", 0) > 82:
        strengths.append(f"EA FC rating: {stats.get('ea_overall')} — elite tier")
    if not strengths:
        strengths.append("Balanced statistical profile")
    return strengths[:5]


def _identify_weaknesses(stats: Dict) -> List[str]:
    """Identify potential weaknesses."""
    weaknesses = []
    if stats.get("defensive_duel_win_rate", 0) > 0 and stats.get("defensive_duel_win_rate", 0) < 45:
        weaknesses.append("Below-average defensive duels")
    if stats.get("tackles_p90", 0) < 0.8 and "midfielder" in stats.get("position", "").lower():
        weaknesses.append("Limited defensive contribution for a midfielder")
    if stats.get("pressing_success_pct", 0) > 0 and stats.get("pressing_success_pct", 0) < 25:
        weaknesses.append("Pressing intensity needs improvement")
    if stats.get("goals_p90", 0) < stats.get("xG_p90", 0) * 0.75:
        weaknesses.append("Underperforming xG — finishing concerns")
    if stats.get("league_coefficient", 1.0) < 0.7:
        weaknesses.append("Playing in a weaker league — adaptation risk")
    if not weaknesses:
        weaknesses.append("No major weaknesses identified")
    return weaknesses[:4]


def _identify_risks(char: Dict, fin: Dict, injury_risk: float) -> List[str]:
    """Compile risk factors."""
    risks = []
    if char.get("attitude_score", 10) < 6:
        risks.append("Character concerns flagged by media analysis")
    if char.get("discipline_issues"):
        count = len(char["discipline_issues"])
        risks.append(f"Discipline history: {count} incident(s)")
    if injury_risk > 6:
        risks.append("HIGH injury risk — recurring issues in 24-month history")
    elif injury_risk > 4:
        risks.append("Moderate injury concerns")
    if fin.get("years_remaining", 5) < 1.5:
        risks.append("Contract near expiry — limited resale protection")
    if fin.get("current_market_value_eur", 0) > 70:
        risks.append("Premium fee required (€70M+)")
    if fin.get("transfer_feasibility") == "Over budget":
        risks.append("Exceeds specified budget")
    if not risks:
        risks.append("No major risk factors identified")
    return risks


def _generate_summary(candidate: Dict, char: Dict, fin: Dict, system: str) -> str:
    """Generate a narrative executive summary via Gemini."""
    if not settings.has_gemini():
        return f"{candidate.get('name', 'Player')} presents a {'strong' if fin.get('value_for_money_score', 0) > 60 else 'moderate'} transfer target option."

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.3,
        )

        stats_summary = (
            f"Goals/90: {candidate.get('goals_p90', 0)}, Assists/90: {candidate.get('assists_p90', 0)}, "
            f"xG/90: {candidate.get('xG_p90', 0)}, xA/90: {candidate.get('xA_p90', 0)}, "
            f"xT/90: {candidate.get('xT_p90', 0)}, Tackles/90: {candidate.get('tackles_p90', 0)}, "
            f"Pressing: {candidate.get('pressing_success_pct', 0)}%"
        )

        prompt = REPORT_PROMPT.format(
            player_name=candidate.get("name", "Unknown"),
            age=candidate.get("age", "?"),
            position=candidate.get("position", "?"),
            team=candidate.get("team", "?"),
            league=candidate.get("league", "?"),
            stats_summary=stats_summary,
            attitude_score=char.get("attitude_score", 7.5),
            discipline_issues=", ".join(char.get("discipline_issues", [])) or "None",
            injury_risk=char.get("injury_risk_score", 2),
            total_days_injured=char.get("total_days_injured_24mo", 0),
            market_value=fin.get("current_market_value_eur", 0),
            years_remaining=fin.get("years_remaining", "?"),
            vfm_score=fin.get("value_for_money_score", 50),
            feasibility=fin.get("transfer_feasibility", "Unknown"),
            tactical_system=system or "General",
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    except Exception as e:
        logger.debug(f"LLM summary generation failed: {e}")
        return f"{candidate.get('name', 'Player')} from {candidate.get('team', '?')} in the {candidate.get('league', '?')}."
