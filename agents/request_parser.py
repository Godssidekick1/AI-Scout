"""
Agent 1: Request Parser ("The Translator")
===========================================
Parses natural language scouting queries into structured search constraints.
Uses Google Gemini to extract position, system, budget, and key attributes.
"""

import json
import logging
from typing import Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import settings

logger = logging.getLogger(__name__)

PARSER_SYSTEM_PROMPT = """You are a Lead Football Scout at a Premier League club.
Parse the user's scouting request and extract structured information.

Return ONLY a valid JSON object with these keys:
{
    "position": "the playing position they want (e.g., 'inverted winger', 'box-to-box midfielder', 'right-back')",
    "tactical_system": "the tactical system mentioned (e.g., 'Arne Slot 4-3-3', 'Guardiola possession') or null",
    "budget_max": "maximum budget in millions EUR as a number, or null",
    "benchmark_player": "a reference player mentioned (e.g., 'like Mo Salah') or null",
    "key_attributes": ["list", "of", "key", "attributes", "emphasized"],
    "target_leagues": ["leagues to search in, or empty list for all"],
    "max_age": "maximum age as integer, or null",
    "min_minutes": "minimum minutes played (default 900)"
}

Be precise. If something isn't mentioned, use null or empty list.
Respond with ONLY the JSON — no markdown, no explanation."""


def parse_request(state: Dict) -> Dict:
    """
    Parse the user's scouting request into structured constraints.
    This is the first node in the LangGraph workflow.
    """
    user_request = state["user_request"]

    if not settings.has_gemini():
        logger.warning("No Gemini API key — using fallback parsing")
        parsed = _fallback_parse(user_request)
    else:
        parsed = _llm_parse(user_request)

    # Update state with parsed constraints
    state["search_constraints"] = parsed
    state["position"] = parsed.get("position", "winger")
    state["tactical_system"] = parsed.get("tactical_system")
    state["benchmark_player"] = parsed.get("benchmark_player")
    state["budget_max"] = parsed.get("budget_max")
    state["key_attributes"] = parsed.get("key_attributes", [])

    state["messages"] = [
        AIMessage(content=f"📋 Parsed request: Looking for a {parsed.get('position', 'player')} "
                          f"{'in ' + parsed.get('tactical_system') + ' system' if parsed.get('tactical_system') else ''} "
                          f"{'(budget: €' + str(parsed.get('budget_max')) + 'M)' if parsed.get('budget_max') else ''}")
    ]

    return state


def _llm_parse(user_request: str) -> Dict:
    """Use Gemini to parse the request."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
        )

        messages = [
            SystemMessage(content=PARSER_SYSTEM_PROMPT),
            HumanMessage(content=f"Scouting request: {user_request}"),
        ]

        response = llm.invoke(messages)
        content = response.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        parsed = json.loads(content)
        logger.info(f"LLM parsed request: {parsed}")
        return parsed

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"LLM parsing failed: {e}")
        return _fallback_parse(user_request)


def _fallback_parse(user_request: str) -> Dict:
    """Simple keyword-based fallback when LLM is unavailable."""
    req = user_request.lower()

    # Position detection
    position = "winger"
    position_keywords = {
        "striker": "striker", "forward": "striker", "no.9": "striker", "number 9": "striker",
        "winger": "winger", "inverted winger": "inverted winger", "wide forward": "winger",
        "midfielder": "midfielder", "box-to-box": "box-to-box midfielder",
        "attacking mid": "attacking midfielder", "no.10": "attacking midfielder",
        "defensive mid": "defensive midfielder", "cdm": "defensive midfielder",
        "centre-back": "centre-back", "cb": "centre-back", "defender": "centre-back",
        "right-back": "right-back", "left-back": "left-back", "full-back": "full-back",
        "goalkeeper": "goalkeeper", "keeper": "goalkeeper",
    }
    for kw, pos in position_keywords.items():
        if kw in req:
            position = pos
            break

    # Budget detection
    budget = None
    import re
    budget_match = re.search(r"(\d+)\s*m(?:illion)?", req)
    if budget_match:
        budget = float(budget_match.group(1))

    # Age detection
    max_age = None
    age_match = re.search(r"under\s+(\d+)", req)
    if age_match:
        max_age = int(age_match.group(1))

    # System detection
    system = None
    if "slot" in req or "4-3-3" in req:
        system = "Arne Slot 4-3-3"
    elif "guardiola" in req or "possession" in req:
        system = "Guardiola possession"
    elif "arteta" in req:
        system = "Arteta inverted full-backs"
    elif "conte" in req or "3-5-2" in req:
        system = "Conte 3-5-2"

    return {
        "position": position,
        "tactical_system": system,
        "budget_max": budget,
        "benchmark_player": None,
        "key_attributes": [],
        "target_leagues": [],
        "max_age": max_age,
        "min_minutes": 900,
    }
