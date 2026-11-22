"""
Shared test fixtures: fictional Understat data, a fake Transfermarkt, and fake
Gemini models, so the whole graph runs offline and deterministically.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from config import settings  # noqa: E402
from core import llm  # noqa: E402
from core.tracing import current_trace  # noqa: E402
from data_sources.ea_fc_ratings import EAFCRatingsSource  # noqa: E402
from data_sources.transfermarkt_source import TransfermarktSource  # noqa: E402
from data_sources.understat_source import UnderstatSource  # noqa: E402
from models.llm_schemas import CharacterAnalysis, ExecutiveSummaries, PlayerSummary, SearchConstraints  # noqa: E402


def _row(pid, name, team, pos, minutes, goals, assists, xg, xa, kp, chain, buildup, yellow=1, red=0):
    """One player in Understat's raw format (numbers as strings, like the real API)."""
    return {"id": str(pid), "player_name": name, "team_title": team, "position": pos, "games": str(minutes // 85),
            "time": str(minutes), "goals": str(goals), "assists": str(assists), "xG": str(xg), "xA": str(xa),
            "npxG": str(round(xg * 0.9, 2)), "shots": str(goals * 3 + 10), "key_passes": str(kp),
            "xGChain": str(chain), "xGBuildup": str(buildup), "yellow_cards": str(yellow), "red_cards": str(red)}


# Fictional players. Ages and values live in the fake Transfermarkt below.
UNDERSTAT_FIXTURE: Dict[str, List[Dict]] = {
    "premier_league": [
        _row(1, "Aaron Swift", "Northside", "F M S", 1500, 9, 6, 8.1, 5.2, 40, 14.0, 4.0),
        _row(2, "Ben Carver", "Rivertown", "F S", 1400, 7, 4, 6.0, 3.1, 30, 11.0, 3.0),
        _row(3, "Cole Vance", "Eastport", "M F S", 1300, 4, 7, 3.5, 5.5, 45, 12.0, 6.0),
        _row(4, "Dan Mercer", "Westfield", "F", 1600, 12, 2, 10.5, 2.0, 20, 13.5, 2.5),
        _row(5, "Eli Brooks", "Northside", "D", 1700, 1, 2, 0.8, 1.5, 12, 6.0, 7.0),
        _row(6, "Finn Hale", "Harbour", "F M", 400, 2, 1, 1.5, 0.8, 8, 3.0, 1.0),  # too few minutes
    ],
    "la_liga": [
        _row(11, "Gabriel Ortega", "Sevilla Azul", "F M S", 1450, 8, 5, 7.0, 4.5, 38, 13.0, 4.5),
        _row(12, "Hugo Prado", "Valle CF", "F S", 1350, 6, 6, 5.5, 4.8, 36, 12.5, 4.2),
        _row(13, "Ivan Lucero", "Costa FC", "M S", 1500, 3, 5, 2.5, 4.0, 42, 11.0, 7.5),
    ],
    "bundesliga": [
        _row(21, "Jonas Keller", "Rhein 04", "F M", 1250, 7, 7, 6.2, 5.0, 41, 13.8, 5.0),
        _row(22, "Kai Brandt", "Nordhafen", "F", 1300, 10, 1, 8.8, 1.2, 15, 11.0, 2.0),
    ],
    "serie_a": [
        _row(31, "Luca Ferri", "Torre Calcio", "F M S", 1400, 6, 4, 5.0, 3.5, 33, 10.5, 3.8),
    ],
    "ligue_1": [
        _row(41, "Marc Dubois", "Olympique Nord", "F M", 1350, 5, 8, 4.4, 5.9, 47, 13.2, 5.5),
    ],
}

TRANSFERMARKT_FIXTURE: Dict[str, Dict] = {
    "Aaron Swift": {"age": 23, "market_value_eur": 55.0, "contract_end_date": "Jun 30, 2028",
                    "injury_history": [{"injury": "Hamstring injury", "days_missed": 21, "from_date": "Jan 2, 2026"}]},
    "Ben Carver": {"age": 27, "market_value_eur": 40.0, "contract_end_date": "Jun 30, 2027"},
    "Cole Vance": {"age": 21, "market_value_eur": 35.0, "contract_end_date": "Jun 30, 2029"},
    "Dan Mercer": {"age": 24, "market_value_eur": 90.0, "contract_end_date": "Jun 30, 2029"},  # over budget
    "Gabriel Ortega": {"age": 22, "market_value_eur": 45.0, "contract_end_date": "Jun 30, 2028"},
    "Hugo Prado": {"age": 24, "market_value_eur": 30.0, "contract_end_date": "Jun 30, 2027"},
    "Jonas Keller": {"age": 23, "market_value_eur": 50.0, "contract_end_date": "Jun 30, 2028"},
    "Kai Brandt": {"age": 29, "market_value_eur": 25.0, "contract_end_date": "Jun 30, 2027"},
    "Luca Ferri": {"age": 22, "market_value_eur": 20.0, "contract_end_date": "Jun 30, 2027"},
    "Marc Dubois": {"age": 24, "market_value_eur": 38.0, "contract_end_date": "Jun 30, 2028"},
}

REAL_WINGERS = ["Aaron Swift", "Cole Vance", "Gabriel Ortega", "Hugo Prado", "Jonas Keller", "Luca Ferri", "Marc Dubois"]


@pytest.fixture
def tm_calls() -> Dict[str, int]:
    return {}


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path, tm_calls):
    """Every test runs offline: fixture data, temp cache, no keys unless a test sets one."""
    monkeypatch.setattr(settings, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    monkeypatch.setattr(settings, "SERPER_API_KEY", "")
    monkeypatch.setattr(settings, "LLM_CACHE", False)
    monkeypatch.setattr(settings, "MAX_CANDIDATES", 10)
    monkeypatch.setattr(settings, "MIN_VERIFIED_CANDIDATES", 5)
    monkeypatch.setattr(settings, "FINDER_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(llm, "_cache", None)
    monkeypatch.setattr(llm, "_models", {})

    async def fake_league_raw(self, league):
        return UNDERSTAT_FIXTURE.get(league, [])

    async def fake_tm(self, name, season="2025-26"):
        tm_calls[name] = tm_calls.get(name, 0) + 1
        data = TRANSFERMARKT_FIXTURE.get(name)
        return {"name": name, **data} if data else None

    async def no_ea(self, name, season="2025-26"):
        return None

    monkeypatch.setattr(UnderstatSource, "_fetch_league_raw", fake_league_raw)
    monkeypatch.setattr(TransfermarktSource, "get_player_stats", fake_tm)
    monkeypatch.setattr(EAFCRatingsSource, "get_player_stats", no_ea)


# ── fake Gemini ────────────────────────────────────────────────
WINGER_QUERY = "Find an inverted winger for Slot's 4-3-3, under 25, budget 60M"
WINGER_CONSTRAINTS = SearchConstraints(position="inverted winger", tactical_system="Arne Slot 4-3-3",
                                       budget_max_m=60, max_age=24)


def tool_call(name: str, args: Dict, i: int = 0) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{i}"}],
                     usage_metadata={"input_tokens": 500, "output_tokens": 50, "total_tokens": 550})


def submit(names: List[str]) -> AIMessage:
    return tool_call("submit_candidates", {"candidates": [{"name": n, "reason": "fits"} for n in names]})


class FakeToolModel:
    """Stands in for a Gemini chat model with bound tools. `responder(messages, turn)` scripts each turn."""

    def __init__(self, responder: Callable):
        self.responder = responder
        self.turn = 0
        self.seen: List[List] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.seen.append(list(messages))
        self.turn += 1
        return self.responder(messages, self.turn)


@pytest.fixture
def fake_gemini(monkeypatch):
    """Turn on the LLM path with scripted answers. Returns a setter for the finder's script."""
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "test-key")

    async def fake_structured(schema, messages, *, name, model=None, temperature=0.2):
        current_trace().record_llm(name, model or settings.GEMINI_MODEL, 300, 80, 1.0)
        if schema is SearchConstraints:
            return WINGER_CONSTRAINTS
        if schema is ExecutiveSummaries:
            players = json.loads(messages[-1].content)["players"]
            return ExecutiveSummaries(summaries=[PlayerSummary(player_id=p["player_id"], summary=f"{p['name']} summary.")
                                                 for p in players])
        if schema is CharacterAnalysis:
            return CharacterAnalysis(attitude_score=8, maturity_assessment="Professional.", locker_room_presence="Positive")
        raise AssertionError(f"unexpected schema {schema}")

    monkeypatch.setattr(llm, "structured", fake_structured)

    holder = {}

    def set_finder(responder: Callable) -> FakeToolModel:
        holder["model"] = FakeToolModel(responder)
        monkeypatch.setattr(llm, "get_chat_model", lambda *a, **k: holder["model"])
        return holder["model"]

    return set_finder


def last_human(messages) -> str:
    return next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))


def run(coro):
    return asyncio.run(coro)
