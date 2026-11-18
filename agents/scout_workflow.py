"""
Scout Workflow (LangGraph)
==========================

    request_parser
         |
    player_finder  <-------------+   agent: Gemini + data tools
         |                       |
     verifier  --- too few? -----+   critic: deterministic checks, loops back with feedback
         |
    +----+--------------+            fan-out: both run in the same superstep
    |                   |
 background_checker  financial_analyst
    |                   |
    +----+--------------+            join: report_writer waits for both
         |
    report_writer -> END
"""

import asyncio
import logging
from typing import Dict

from langgraph.graph import END, START, StateGraph

from agents.background_checker import check_backgrounds
from agents.financial_analyst import analyze_finances
from agents.player_finder import find_players
from agents.report_writer import write_reports
from agents.request_parser import parse_request
from agents.verifier import route_after_verify, verify_candidates
from core.context import start_context
from core.tracing import start_trace
from models.workflow_state import ScoutingState

logger = logging.getLogger(__name__)


def build_scout_workflow():
    g = StateGraph(ScoutingState)
    g.add_node("request_parser", parse_request)
    g.add_node("player_finder", find_players)
    g.add_node("verifier", verify_candidates)
    g.add_node("background_checker", check_backgrounds)
    g.add_node("financial_analyst", analyze_finances)
    g.add_node("report_writer", write_reports)

    g.add_edge(START, "request_parser")
    g.add_edge("request_parser", "player_finder")
    g.add_edge("player_finder", "verifier")
    g.add_conditional_edges("verifier", route_after_verify,
                            ["player_finder", "background_checker", "financial_analyst"])
    g.add_edge(["background_checker", "financial_analyst"], "report_writer")
    g.add_edge("report_writer", END)
    return g.compile()


async def arun_scout(user_request: str) -> Dict:
    logger.info("Starting scout workflow: %r", user_request)
    trace = start_trace()
    ctx = start_context()

    final = await build_scout_workflow().ainvoke({"user_request": user_request}, {"recursion_limit": 40})

    reports = final.get("scouting_reports", [])
    logger.info("Workflow complete: %d reports", len(reports))
    return {
        "constraints": final.get("constraints", {}),
        "reports": reports,
        "candidates_verified": len(final.get("candidates", [])),
        "rejected": final.get("rejected", []),
        "finder_attempts": final.get("finder_attempts", 0),
        "messages": final.get("messages", []),
        "errors": final.get("errors", []) + ctx.errors,
        "trace": trace.summary(),
    }


def run_scout(user_request: str) -> Dict:
    """Synchronous entry point for the CLI."""
    return asyncio.run(arun_scout(user_request))
