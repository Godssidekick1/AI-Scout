"""
Scout Workflow — LangGraph DAG
==============================
Wires all 5 agents together into a sequential workflow:
Request Parser → Player Finder → [Background Checker + Financial Analyst] → Report Writer
"""

import logging
from typing import Dict

from langgraph.graph import StateGraph, END

from models.workflow_state import ScoutingState
from agents.request_parser import parse_request
from agents.player_finder import find_players
from agents.background_checker import check_backgrounds
from agents.financial_analyst import analyze_finances
from agents.report_writer import write_reports

logger = logging.getLogger(__name__)


def build_scout_workflow() -> StateGraph:
    """
    Build the LangGraph DAG for the multi-agent scouting system.

    Flow:
        request_parser → player_finder → background_checker → financial_analyst → report_writer → END
    
    Note: background_checker and financial_analyst run sequentially for now.
    They could run in parallel with LangGraph's fan-out, but keeping it
    simple for the initial version to avoid async complexity.
    """
    workflow = StateGraph(ScoutingState)

    # Add agent nodes
    workflow.add_node("request_parser", parse_request)
    workflow.add_node("player_finder", find_players)
    workflow.add_node("background_checker", check_backgrounds)
    workflow.add_node("financial_analyst", analyze_finances)
    workflow.add_node("report_writer", write_reports)

    # Define the flow
    workflow.add_edge("request_parser", "player_finder")
    workflow.add_edge("player_finder", "background_checker")
    workflow.add_edge("background_checker", "financial_analyst")
    workflow.add_edge("financial_analyst", "report_writer")
    workflow.add_edge("report_writer", END)

    # Entry point
    workflow.set_entry_point("request_parser")

    return workflow.compile()


def run_scout(user_request: str) -> Dict:
    """
    Run the complete scouting workflow.

    Args:
        user_request: Natural language scouting query.

    Returns:
        Dictionary with reports, candidate count, and workflow messages.
    """
    logger.info(f"Starting scout workflow: '{user_request}'")

    workflow = build_scout_workflow()

    # Initialize state
    initial_state = {
        "user_request": user_request,
        "tactical_system": None,
        "position": None,
        "budget_max": None,
        "search_constraints": None,
        "benchmark_player": None,
        "key_attributes": None,
        "candidate_pool": [],
        "character_reports": [],
        "financial_assessments": [],
        "scouting_reports": [],
        "messages": [],
        "errors": [],
    }

    # Execute
    final_state = workflow.invoke(initial_state)

    reports = final_state.get("scouting_reports", [])
    messages = final_state.get("messages", [])

    logger.info(f"Workflow complete. {len(reports)} reports generated.")

    return {
        "reports": reports,
        "candidates_analyzed": len(final_state.get("candidate_pool", [])),
        "messages": [m.content if hasattr(m, 'content') else str(m) for m in messages],
        "errors": final_state.get("errors", []),
    }
