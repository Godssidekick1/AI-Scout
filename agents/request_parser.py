"""
Step 1: Request Parser
======================
Natural-language query -> validated SearchConstraints (Gemini structured output).
Falls back to keyword rules when there is no API key or the call fails.
"""

import logging
import re
from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from core import llm
from core.tracing import traced_node
from models.llm_schemas import SearchConstraints

logger = logging.getLogger(__name__)

PARSER_SYSTEM_PROMPT = """You are the lead scout at a Premier League club.
Turn the user's scouting request into search constraints.
Only fill a field when the request states or clearly implies it; otherwise leave it null or empty.
"under 25" means max_age 24. Budgets are in millions of euros.
Supported leagues: premier_league, la_liga, bundesliga, serie_a, ligue_1."""


@traced_node("request_parser")
async def parse_request(state: Dict) -> Dict:
    request = state["user_request"]
    errors = []

    if settings.has_gemini():
        try:
            constraints = await llm.structured(
                SearchConstraints,
                [SystemMessage(content=PARSER_SYSTEM_PROMPT),
                 HumanMessage(content=f"<request>{request}</request>")],
                name="request_parser",
                temperature=0.0,
            )
        except llm.LLMError as e:
            errors.append(f"request_parser: LLM failed, used keyword fallback ({e})")
            constraints = fallback_parse(request)
    else:
        constraints = fallback_parse(request)

    c = constraints
    summary = f"Parsed request: {c.position}"
    if c.tactical_system:
        summary += f" for {c.tactical_system}"
    if c.max_age:
        summary += f", age <= {c.max_age}"
    if c.budget_max_m:
        summary += f", budget EUR {c.budget_max_m:g}M"

    return {"constraints": c.model_dump(), "messages": [summary], "errors": errors}


_POSITIONS = [  # longest phrases first so "inverted winger" beats "winger"
    ("inverted winger", "inverted winger"), ("box-to-box", "box-to-box midfielder"),
    ("attacking mid", "attacking midfielder"), ("defensive mid", "defensive midfielder"),
    ("centre-back", "centre-back"), ("center-back", "centre-back"), ("right-back", "right-back"),
    ("left-back", "left-back"), ("full-back", "full-back"), ("wing-back", "wing-back"),
    ("goalkeeper", "goalkeeper"), ("keeper", "goalkeeper"), ("striker", "striker"),
    ("number 9", "striker"), ("no.9", "striker"), ("no.10", "attacking midfielder"),
    ("winger", "winger"), ("forward", "striker"), ("midfielder", "midfielder"),
    ("cdm", "defensive midfielder"), ("defender", "centre-back"),
]

_LEAGUES = {
    "premier league": "premier_league", "epl": "premier_league", "la liga": "la_liga",
    "bundesliga": "bundesliga", "serie a": "serie_a", "ligue 1": "ligue_1",
}


def fallback_parse(request: str) -> SearchConstraints:
    """Keyword rules used when Gemini is unavailable."""
    req = request.lower()

    position = next((pos for kw, pos in _POSITIONS if re.search(rf"(?<![a-z]){re.escape(kw)}", req)), "winger")

    budget = None
    m = re.search(r"(?:budget|max|up to|€|eur)\s*€?\s*(\d+(?:\.\d+)?)\s*(m|million|mn)\b", req) \
        or re.search(r"\b(\d+(?:\.\d+)?)\s*(m|million|mn)\b", req)
    if m:
        budget = float(m.group(1))

    max_age = None
    m = re.search(r"under\s+(\d{2})", req)
    if m:
        max_age = int(m.group(1)) - 1
    else:
        m = re.search(r"(?:max(?:imum)? age|aged? up to|at most)\s+(\d{2})", req)
        if m:
            max_age = int(m.group(1))

    system = None
    if "slot" in req or "4-3-3" in req:
        system = "Arne Slot 4-3-3"
    elif "guardiola" in req or "possession" in req:
        system = "Guardiola possession"
    elif "arteta" in req:
        system = "Arteta positional play"
    elif "conte" in req or "3-5-2" in req:
        system = "Conte 3-5-2"

    leagues = [key for name, key in _LEAGUES.items() if name in req]

    return SearchConstraints(position=position, tactical_system=system, budget_max_m=budget,
                             max_age=max_age, target_leagues=sorted(set(leagues)))
