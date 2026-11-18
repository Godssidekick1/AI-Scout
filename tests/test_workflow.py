"""
End-to-end graph tests with fake Gemini and fixture data.
These are the regression tests for the state-duplication bug and the agent loop.
"""

import asyncio
import json
import time

from agents.scout_workflow import arun_scout, build_scout_workflow
from config import settings
from core import llm
from data_sources.transfermarkt_source import TransfermarktSource
from tests.conftest import REAL_WINGERS, WINGER_QUERY, last_human, run, submit, tool_call


def _search_then_submit(names_by_attempt):
    """Agent script: search once, then submit the names for the current attempt."""
    state = {"attempt": 0}

    def responder(messages, turn):
        if "Call submit_candidates" in last_human(messages):
            return submit(names_by_attempt[min(state["attempt"], len(names_by_attempt) - 1)])
        if not any(getattr(m, "type", "") == "tool" for m in messages):
            return tool_call("search_league_players", {"league": "premier_league", "position": "winger"})
        names = names_by_attempt[min(state["attempt"], len(names_by_attempt) - 1)]
        state["attempt"] += 1
        return submit(names)

    return responder


def test_reports_are_not_duplicated_and_join_by_id(fake_gemini):
    fake_gemini(_search_then_submit([REAL_WINGERS]))
    result = run(arun_scout(WINGER_QUERY))

    ids = [r["player_id"] for r in result["reports"]]
    assert len(ids) == len(set(ids)), "each player must appear exactly once"
    assert len(ids) == result["candidates_verified"] == len(REAL_WINGERS)
    for r in result["reports"]:
        assert r["character_report"]["player_id"] == r["player_id"]
        assert r["financial_assessment"]["player_id"] == r["player_id"]
        assert r["executive_summary"] == f"{r['player_name']} summary."
    # 1 parser + 2 finder turns + 1 batched summary call (no per-player summary calls)
    assert result["trace"]["llm_calls"] == 4


def test_each_node_runs_once_and_checks_fan_out(fake_gemini):
    fake_gemini(_search_then_submit([REAL_WINGERS]))
    runs = run(arun_scout(WINGER_QUERY))["trace"]["node_runs"]
    for node in ("request_parser", "player_finder", "verifier", "background_checker",
                 "financial_analyst", "report_writer"):
        assert runs.count(node) == 1, node


def test_hallucinated_players_are_rejected_and_finder_retries(fake_gemini):
    first = ["Aaron Swift", "Cole Vance", "Imaginary Wonderkid", "Not A Player"]
    second = ["Gabriel Ortega", "Hugo Prado", "Jonas Keller"]
    model = fake_gemini(_search_then_submit([first, second]))
    result = run(arun_scout(WINGER_QUERY))

    rejected = {r["name"]: r["reason"] for r in result["rejected"]}
    assert "not found" in rejected["Imaginary Wonderkid"]
    assert "not found" in rejected["Not A Player"]
    assert result["finder_attempts"] == 2
    assert result["candidates_verified"] == 5
    # the second attempt was told what was rejected and what is already accepted
    retry_task = json.loads(model.seen[-1][1].content)
    assert {"name": "Not A Player", "reason": rejected["Not A Player"]} in retry_task["already_rejected"]
    assert "aaron swift" in retry_task["already_accepted"]


def test_loop_stops_after_max_attempts(fake_gemini):
    fake_gemini(_search_then_submit([["Ghost One", "Ghost Two"]]))
    result = run(arun_scout(WINGER_QUERY))
    assert result["finder_attempts"] == settings.FINDER_MAX_ATTEMPTS
    assert result["reports"] == []


def test_constraints_are_enforced(fake_gemini):
    everyone = REAL_WINGERS + ["Dan Mercer", "Ben Carver", "Finn Hale", "Eli Brooks"]
    fake_gemini(_search_then_submit([everyone]))
    result = run(arun_scout(WINGER_QUERY))
    names = {r["player_name"] for r in result["reports"]}
    reasons = {r["name"]: r["reason"] for r in result["rejected"]}
    assert "Dan Mercer" not in names and "budget" in reasons["Dan Mercer"]   # EUR 90M > 60M
    assert "Ben Carver" not in names and "age" in reasons["Ben Carver"]      # 27 > 24
    assert "Finn Hale" not in names and "minutes" in reasons["Finn Hale"]    # 400 min
    assert "Eli Brooks" not in names and "plays" in reasons["Eli Brooks"]    # a defender
    for r in result["reports"]:
        assert r["age"] <= 24
        assert r["financial_assessment"]["current_market_value_eur"] <= 60


def test_transfermarkt_fetched_once_per_player_across_agents(fake_gemini, tm_calls):
    fake_gemini(_search_then_submit([REAL_WINGERS]))
    run(arun_scout(WINGER_QUERY))
    assert tm_calls and max(tm_calls.values()) == 1


def test_agent_that_never_submits_falls_back_to_baseline(fake_gemini, monkeypatch):
    monkeypatch.setattr(settings, "FINDER_MAX_TOOL_STEPS", 3)
    fake_gemini(lambda messages, turn: tool_call("search_league_players",
                                                 {"league": "la_liga", "position": "winger"}, turn))
    result = run(arun_scout(WINGER_QUERY))
    assert any("agent failed" in e for e in result["errors"])
    assert result["reports"], "baseline search should still produce verified players"


def test_llm_failure_degrades_gracefully(fake_gemini, monkeypatch):
    fake_gemini(_search_then_submit([REAL_WINGERS]))

    async def broken(*a, **k):
        raise llm.LLMError("simulated outage")

    monkeypatch.setattr(llm, "structured", broken)
    result = run(arun_scout(WINGER_QUERY))
    assert any("request_parser" in e for e in result["errors"])
    assert any("summaries failed" in e for e in result["errors"])
    assert result["reports"] and all(r["executive_summary"] for r in result["reports"])


def test_runs_without_any_api_key():
    result = run(arun_scout(WINGER_QUERY))
    assert result["constraints"]["max_age"] == 24
    assert result["constraints"]["budget_max_m"] == 60
    assert result["reports"]
    assert result["trace"]["llm_calls"] == 0
    for r in result["reports"]:
        assert "understat" in r["data_sources"]


def test_per_player_work_is_concurrent(monkeypatch):
    async def slow_tm(self, name, season="2025-26"):
        await asyncio.sleep(0.2)
        return {"name": name, "age": 22, "market_value_eur": 30.0}

    monkeypatch.setattr(TransfermarktSource, "get_player_stats", slow_tm)
    monkeypatch.setattr(settings, "MAX_CONCURRENCY", 20)
    t0 = time.perf_counter()
    result = run(arun_scout(WINGER_QUERY))
    elapsed = time.perf_counter() - t0
    n = len(result["rejected"]) + result["candidates_verified"]
    assert n >= 6
    assert elapsed < 0.2 * n / 2, f"{n} lookups took {elapsed:.2f}s; they should overlap"


def test_graph_shape():
    g = build_scout_workflow().get_graph()
    edges = {(e.source, e.target) for e in g.edges}
    assert ("verifier", "player_finder") in edges
    assert ("verifier", "background_checker") in edges
    assert ("verifier", "financial_analyst") in edges
    assert ("background_checker", "report_writer") in edges
    assert ("financial_analyst", "report_writer") in edges
