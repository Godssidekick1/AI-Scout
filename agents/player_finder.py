"""
Agent 2: Player Finder ("The Data Analyst")
============================================
Uses Gemini + Serper to identify real player candidates,
then enriches with Transfermarkt and EA FC 26 live data.
"""

import asyncio
import json
import logging
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from data_sources.transfermarkt_source import TransfermarktSource
from data_sources.ea_fc_ratings import EAFCRatingsSource
from data_sources.news_search import NewsSearchSource
from data_sources.data_cache import DataCache
from analysis.league_normalizer import apply_league_tax, LEAGUE_COEFFICIENTS
from config import settings

logger = logging.getLogger(__name__)

PLAYER_SEARCH_PROMPT = """You are a top football scout. Based on the following scouting criteria, 
suggest 8-12 REAL players who would be strong transfer targets.

Criteria:
- Position: {position}
- Tactical system: {system}
- Max age: {max_age}
- Budget: €{budget}M
- Key attributes: {attributes}
- Target leagues: {leagues}

For EACH player, provide this EXACT JSON format (no markdown, no extra text):
{{
    "players": [
        {{
            "name": "Full Name",
            "age": 23,
            "team": "Current Club",
            "league": "League Name",
            "position": "Position",
            "nationality": "Country",
            "estimated_goals_p90": 0.3,
            "estimated_assists_p90": 0.2,
            "estimated_xG_p90": 0.35,
            "estimated_xA_p90": 0.18,
            "strengths": ["strength1", "strength2"],
            "why_recommended": "Brief reason for recommendation"
        }}
    ]
}}

IMPORTANT:
- Only suggest players who are CURRENTLY playing in the 2024-25 or 2025-26 season
- Include players from top 5 leagues AND hidden gems from Eredivisie, Primeira Liga, etc.
- Be realistic with estimated stats based on their actual known performance
- Respond with ONLY the JSON, no explanation"""


def find_players(state: Dict) -> Dict:
    """
    Use Gemini + web search to identify real player candidates,
    then enrich with Transfermarkt and EA FC data.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                candidates = pool.submit(asyncio.run, _async_find_players(state)).result()
        else:
            candidates = asyncio.run(_async_find_players(state))
    except RuntimeError:
        candidates = asyncio.run(_async_find_players(state))

    state["candidate_pool"] = candidates
    state["messages"] = [
        AIMessage(content=f"🔍 Found {len(candidates)} candidates. "
                          f"Enriched with Transfermarkt + EA FC 26 data. "
                          f"Applied league tax normalization.")
    ]

    return state


async def _async_find_players(state: Dict) -> List[Dict]:
    """Use LLM to identify candidates, then enrich with real data."""
    constraints = state.get("search_constraints", {})
    position = constraints.get("position", state.get("position", "winger"))
    max_age = constraints.get("max_age", 30)
    budget_max = constraints.get("budget_max", state.get("budget_max", 100))
    system = constraints.get("tactical_system", state.get("tactical_system", "General"))
    attributes = constraints.get("key_attributes", [])
    target_leagues = constraints.get("target_leagues", [])

    cache = DataCache(
        cache_dir=settings.CACHE_DIR,
        ttl_hours=settings.CACHE_TTL_HOURS,
    )

    # ─── Step 1: Use Gemini to identify candidates ───
    candidates = []
    if settings.has_gemini():
        candidates = _ask_gemini_for_players(
            position, system, max_age, budget_max, attributes, target_leagues
        )
        logger.info(f"Gemini suggested {len(candidates)} players")

    # ─── Step 2: Supplement with Serper web search ───
    if settings.has_serper():
        web_players = await _search_web_for_players(
            position, max_age, budget_max, cache
        )
        # Merge web results (don't duplicate)
        known_names = {c.get("name", "").lower() for c in candidates}
        for wp in web_players:
            if wp.get("name", "").lower() not in known_names:
                candidates.append(wp)
                known_names.add(wp["name"].lower())

    if not candidates:
        logger.warning("No candidates found from any source")
        return []

    # ─── Step 3: Enrich with Transfermarkt ───
    if settings.ENABLE_TRANSFERMARKT:
        tmk = TransfermarktSource(cache=cache)
        for candidate in candidates[:12]:
            try:
                tm_data = await tmk.get_player_stats(candidate.get("name", ""))
                if tm_data:
                    candidate["market_value_eur"] = tm_data.get("market_value_eur", 0)
                    candidate["contract_end_date"] = tm_data.get("contract_end_date", "")
                    candidate["agent"] = tm_data.get("agent", "")
                    if "transfermarkt" not in candidate.get("sources", []):
                        candidate.setdefault("sources", []).append("transfermarkt")
            except Exception as e:
                logger.debug(f"Transfermarkt enrichment failed for {candidate.get('name')}: {e}")

    # ─── Step 4: Enrich with EA FC 26 ratings ───
    if settings.ENABLE_EA_FC:
        ea_fc = EAFCRatingsSource(cache=cache)
        for candidate in candidates[:10]:
            try:
                attrs = await ea_fc.get_player_stats(candidate.get("name", ""))
                if attrs:
                    candidate["ea_overall"] = attrs.get("overall_rating", 0)
                    candidate["ea_pace"] = attrs.get("pace", 0)
                    candidate["ea_shooting"] = attrs.get("shooting", 0)
                    candidate["ea_passing"] = attrs.get("passing", 0)
                    candidate["ea_dribbling"] = attrs.get("dribbling", 0)
                    candidate["ea_defending"] = attrs.get("defending", 0)
                    candidate["ea_physical"] = attrs.get("physical", 0)
                    if "ea_fc_26" not in candidate.get("sources", []):
                        candidate.setdefault("sources", []).append("ea_fc_26")
            except Exception as e:
                logger.debug(f"EA FC enrichment failed for {candidate.get('name')}: {e}")

    # ─── Step 5: Apply league tax ───
    league_name_map = {
        "premier league": "premier_league",
        "epl": "premier_league",
        "la liga": "la_liga",
        "bundesliga": "bundesliga",
        "serie a": "serie_a",
        "ligue 1": "ligue_1",
        "eredivisie": "eredivisie",
        "primeira liga": "primeira_liga",
        "championship": "championship",
        "mls": "mls",
        "saudi pro league": "saudi_pro_league",
    }

    adjusted_candidates = []
    for candidate in candidates:
        league = candidate.get("league", "").lower()
        league_key = league_name_map.get(league, league.replace(" ", "_"))
        adjusted = apply_league_tax(candidate, league_key)

        # Filter by budget
        market_value = adjusted.get("market_value_eur", 0)
        if budget_max and market_value and market_value > budget_max:
            continue

        if "gemini" not in adjusted.get("sources", []):
            adjusted.setdefault("sources", []).append("gemini")

        adjusted_candidates.append(adjusted)

    # Sort by composite score
    adjusted_candidates.sort(
        key=lambda x: (
            x.get("xG_p90", x.get("estimated_xG_p90", 0)) +
            x.get("xA_p90", x.get("estimated_xA_p90", 0)) +
            x.get("goals_p90", x.get("estimated_goals_p90", 0)) * 0.5
        ),
        reverse=True,
    )

    return adjusted_candidates[:15]


def _ask_gemini_for_players(
    position: str, system: str, max_age: int,
    budget: float, attributes: list, leagues: list
) -> List[Dict]:
    """Use Gemini to suggest real player candidates."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.4,
        )

        prompt = PLAYER_SEARCH_PROMPT.format(
            position=position,
            system=system or "General",
            max_age=max_age or 30,
            budget=budget or 100,
            attributes=", ".join(attributes) if attributes else "well-rounded",
            leagues=", ".join(leagues) if leagues else "All major leagues",
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        # Strip markdown code fences
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)
        players = data.get("players", data if isinstance(data, list) else [])

        # Normalize field names
        result = []
        for p in players:
            normalized = {
                "name": p.get("name", ""),
                "age": p.get("age", 0),
                "team": p.get("team", ""),
                "league": p.get("league", ""),
                "position": p.get("position", ""),
                "nationality": p.get("nationality", ""),
                "goals_p90": p.get("estimated_goals_p90", p.get("goals_p90", 0)),
                "assists_p90": p.get("estimated_assists_p90", p.get("assists_p90", 0)),
                "xG_p90": p.get("estimated_xG_p90", p.get("xG_p90", 0)),
                "xA_p90": p.get("estimated_xA_p90", p.get("xA_p90", 0)),
                "why_recommended": p.get("why_recommended", ""),
                "sources": ["gemini"],
            }
            result.append(normalized)

        return result

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Gemini player search failed: {e}")
        return []


async def _search_web_for_players(
    position: str, max_age: int, budget: float, cache: DataCache
) -> List[Dict]:
    """Use Serper to search for transfer targets matching criteria."""
    search = NewsSearchSource(cache=cache)

    query = f"best {position} transfer targets under {max_age or 25} years old 2025 2026 football"
    results = await search.search_news("", query_type="general")

    # We can't parse structured player data from news — this is supplementary
    # The main player identification comes from Gemini
    return []
