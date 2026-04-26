"""
Agent 4: Financial Analyst ("The Accountant")
=============================================
Analyzes market value, contract situations, and financial viability.
Uses Transfermarkt data + statistical output to calculate value-for-money.
"""

import asyncio
import logging
from typing import Dict, List

from langchain_core.messages import AIMessage

from data_sources.transfermarkt_source import TransfermarktSource
from data_sources.data_cache import DataCache
from config import settings

logger = logging.getLogger(__name__)


def analyze_finances(state: Dict) -> Dict:
    """
    Assess financial viability for all candidates.
    """
    candidates = state.get("candidate_pool", [])
    budget_max = state.get("budget_max")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                assessments = pool.submit(
                    asyncio.run, _async_analyze(candidates, budget_max)
                ).result()
        else:
            assessments = asyncio.run(_async_analyze(candidates, budget_max))
    except RuntimeError:
        assessments = asyncio.run(_async_analyze(candidates, budget_max))

    state["financial_assessments"] = state.get("financial_assessments", []) + assessments
    state["messages"] = [
        AIMessage(content=f"💰 Completed financial analysis for {len(assessments)} candidates. "
                          f"Assessed market values, contracts, and value-for-money.")
    ]

    return state


async def _async_analyze(candidates: List[Dict], budget_max: float = None) -> List[Dict]:
    """Fetch financial data and compute viability scores."""
    cache = DataCache(
        cache_dir=settings.CACHE_DIR,
        ttl_hours=168,  # 7-day cache for financial data
    )
    tmk = TransfermarktSource(cache=cache)

    assessments = []

    for candidate in candidates[:10]:
        name = candidate.get("name", "Unknown")
        age = candidate.get("age", 25)

        assessment = {
            "player_name": name,
            "current_market_value_eur": 0,
            "contract_end_date": "",
            "years_remaining": 0,
            "agent": "",
            "release_clause_eur": None,
            "estimated_wage_weekly_eur": 0,
            "value_for_money_score": 50.0,
            "transfer_feasibility": "Unknown",
        }

        # --- Fetch from Transfermarkt ---
        if settings.ENABLE_TRANSFERMARKT:
            try:
                tm_data = await tmk.get_player_stats(name)
                if tm_data:
                    assessment["current_market_value_eur"] = tm_data.get("market_value_eur", 0)
                    assessment["contract_end_date"] = tm_data.get("contract_end_date", "")
                    assessment["agent"] = tm_data.get("agent", "")

                    # Estimate years remaining
                    if assessment["contract_end_date"]:
                        assessment["years_remaining"] = _estimate_years_remaining(
                            assessment["contract_end_date"]
                        )
            except Exception as e:
                logger.debug(f"Transfermarkt financial fetch failed for {name}: {e}")

        # --- Estimate market value if not found ---
        if assessment["current_market_value_eur"] == 0:
            assessment["current_market_value_eur"] = _estimate_market_value(candidate, age)

        # --- Estimate weekly wages ---
        assessment["estimated_wage_weekly_eur"] = _estimate_wages(
            assessment["current_market_value_eur"], age
        )

        # --- Value-for-money score ---
        assessment["value_for_money_score"] = _calculate_value_for_money(
            candidate, assessment, age
        )

        # --- Transfer feasibility ---
        market_value = assessment["current_market_value_eur"]
        years_left = assessment["years_remaining"]

        if budget_max and market_value:
            if market_value <= budget_max * 0.6:
                assessment["transfer_feasibility"] = "Easy"
            elif market_value <= budget_max * 0.9:
                assessment["transfer_feasibility"] = "Moderate"
            elif market_value <= budget_max:
                assessment["transfer_feasibility"] = "Tight"
            else:
                assessment["transfer_feasibility"] = "Over budget"
        elif years_left and years_left < 1.5:
            assessment["transfer_feasibility"] = "bargain — contract near expiry"
        else:
            assessment["transfer_feasibility"] = "Moderate"

        assessments.append(assessment)

    return assessments


def _estimate_market_value(stats: Dict, age: int) -> float:
    """Estimate market value from stats and age when Transfermarkt data is missing."""
    base = 15.0  # Base value in millions

    # Output-based value
    base += stats.get("goals_p90", 0) * 25
    base += stats.get("assists_p90", 0) * 18
    base += stats.get("xG_p90", 0) * 12
    base += stats.get("xA_p90", 0) * 10

    # Age adjustment
    if age < 22:
        base *= 1.4    # High upside premium
    elif age < 25:
        base *= 1.2    # Peak market value age
    elif age < 28:
        base *= 1.0
    elif age < 30:
        base *= 0.8
    else:
        base *= 0.5     # Depreciating

    # League adjustment
    league_multipliers = {
        "premier_league": 1.5,
        "bundesliga": 1.2, "la_liga": 1.2,
        "serie_a": 1.1, "ligue_1": 1.0,
        "eredivisie": 0.8, "primeira_liga": 0.75,
    }
    league = stats.get("league", "").lower().replace(" ", "_")
    base *= league_multipliers.get(league, 0.7)

    return round(base, 1)


def _estimate_wages(market_value: float, age: int) -> float:
    """Estimate weekly wages from market value."""
    # Rough heuristic: weekly wage ≈ market_value * 2000-4000 EUR
    # Higher for older established players, lower for young prospects
    if age < 22:
        factor = 2000
    elif age < 26:
        factor = 3000
    elif age < 30:
        factor = 3500
    else:
        factor = 2500

    return round(market_value * factor)


def _estimate_years_remaining(contract_end: str) -> float:
    """Estimate years remaining from contract end date."""
    try:
        from datetime import datetime
        end_date = datetime.strptime(contract_end.strip(), "%b %d, %Y")
        now = datetime.now()
        delta = end_date - now
        return round(max(delta.days / 365, 0), 1)
    except (ValueError, TypeError):
        try:
            from datetime import datetime
            end_date = datetime.strptime(contract_end.strip(), "%Y-%m-%d")
            now = datetime.now()
            delta = end_date - now
            return round(max(delta.days / 365, 0), 1)
        except (ValueError, TypeError):
            return 2.0  # Default assumption


def _calculate_value_for_money(stats: Dict, financial: Dict, age: int) -> float:
    """
    Calculate value-for-money score (0-100).
    Combines statistical output with cost and contract length.
    """
    market_value = financial.get("current_market_value_eur", 1)
    if market_value <= 0:
        market_value = 1

    # Statistical output score
    output = (
        stats.get("goals_p90", 0) * 15 +
        stats.get("assists_p90", 0) * 12 +
        stats.get("xG_p90", 0) * 10 +
        stats.get("xA_p90", 0) * 8 +
        stats.get("xT_p90", 0) * 6
    )

    # Age factor (younger = better value for resale)
    if age < 23:
        age_factor = 1.3
    elif age < 26:
        age_factor = 1.1
    elif age < 29:
        age_factor = 1.0
    else:
        age_factor = 0.75

    # Contract factor (more years = selling club has leverage = higher price)
    years_left = financial.get("years_remaining", 2)
    contract_factor = 1.0
    if years_left < 1.5:
        contract_factor = 1.3  # Bargain potential
    elif years_left > 3:
        contract_factor = 0.85  # Expensive

    # Value ratio
    value_ratio = (output * age_factor * contract_factor) / (market_value / 10)
    score = min(round(value_ratio * 100, 1), 100)

    return max(score, 0)
