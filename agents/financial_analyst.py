"""
Step 4b: Financial Analyst (runs in parallel with the background checker)
=========================================================================
Pure Python over the Transfermarkt profile the verifier already fetched:
market value (estimated, and flagged as such, when missing), contract length,
wage estimate, value-for-money and feasibility against the budget.
"""

import asyncio
import logging
import re
from datetime import date, datetime
from typing import Dict, Optional

from core.context import get_context
from core.tracing import traced_node
from models.llm_schemas import SearchConstraints
from models.scouting_report import FinancialAssessment

logger = logging.getLogger(__name__)


@traced_node("financial_analyst")
async def analyze_finances(state: Dict) -> Dict:
    c = SearchConstraints.model_validate(state["constraints"])
    candidates = state.get("candidates", [])
    results = await asyncio.gather(*(assess(cand, c.budget_max_m) for cand in candidates))
    return {
        "financial_assessments": {a.player_id: a.model_dump() for a in results},
        "messages": [f"Financial analysis done for {len(results)} candidates."],
    }


async def assess(cand: Dict, budget: Optional[float]) -> FinancialAssessment:
    market = await get_context().market_profile(cand["name"]) or {}
    a = FinancialAssessment(player_id=cand["player_id"], player_name=cand["name"])
    age = cand.get("age") or 25

    a.current_market_value_eur = cand.get("market_value_eur_m") or market.get("market_value_eur") or 0.0
    if not a.current_market_value_eur:
        a.current_market_value_eur = estimate_market_value(cand, age)
        a.market_value_is_estimate = True
    a.contract_end_date = cand.get("contract_end_date") or market.get("contract_end_date") or ""
    a.agent = cand.get("agent") or market.get("agent") or ""
    a.years_remaining = years_remaining(a.contract_end_date)
    a.estimated_wage_weekly_eur = estimate_wages(a.current_market_value_eur, age)
    a.value_for_money_score = value_for_money(cand, a.current_market_value_eur, a.years_remaining, age)
    a.transfer_feasibility = feasibility(a.current_market_value_eur, a.years_remaining, budget)
    return a


def estimate_market_value(stats: Dict, age: int) -> float:
    """Rough fallback when Transfermarkt has no value. Always flagged as an estimate."""
    base = 10.0 + stats.get("npxG_p90", 0) * 40 + stats.get("xA_p90", 0) * 40 + stats.get("xGChain_p90", 0) * 20
    base *= 1.4 if age < 22 else 1.2 if age < 25 else 1.0 if age < 28 else 0.8 if age < 30 else 0.5
    base *= {"premier_league": 1.5, "la_liga": 1.2, "bundesliga": 1.2, "serie_a": 1.1, "ligue_1": 1.0}.get(
        stats.get("league", ""), 0.8)
    return round(base, 1)


def estimate_wages(market_value_m: float, age: int) -> float:
    factor = 2000 if age < 22 else 3000 if age < 26 else 3500 if age < 30 else 2500
    return round(market_value_m * factor)


_DATE_FORMATS = ("%b %d, %Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y")


def years_remaining(contract_end: str, today: Optional[date] = None) -> Optional[float]:
    if not contract_end:
        return None
    text = re.sub(r"\s+", " ", contract_end).strip()
    today = today or date.today()
    for fmt in _DATE_FORMATS:
        try:
            end = datetime.strptime(text, fmt).date()
            return round(max((end - today).days / 365, 0), 1)
        except ValueError:
            continue
    return None


def value_for_money(stats: Dict, value_m: float, years_left: Optional[float], age: int) -> float:
    output = stats.get("npxG_p90", 0) * 60 + stats.get("xA_p90", 0) * 60 + stats.get("xGChain_p90", 0) * 30
    age_factor = 1.3 if age < 23 else 1.1 if age < 26 else 1.0 if age < 29 else 0.75
    contract_factor = 1.0 if years_left is None else 1.3 if years_left < 1.5 else 0.85 if years_left > 3 else 1.0
    # Output per EUR 1M, squashed to 0-100: r / (r + 1) gives 50 at one output point per million
    # and approaches 100 only for exceptional output at a low price.
    r = output * age_factor * contract_factor / max(value_m, 1.0)
    return round(100 * r / (r + 1), 1)


def feasibility(value_m: float, years_left: Optional[float], budget: Optional[float]) -> str:
    if budget and value_m:
        if value_m <= budget * 0.6:
            return "Easy"
        if value_m <= budget * 0.9:
            return "Moderate"
        if value_m <= budget:
            return "Tight"
        return "Over budget"
    if years_left is not None and years_left < 1.5:
        return "Bargain: contract near expiry"
    return "Moderate"
