"""
Workflow State
==============
Shared state passed between all LangGraph agent nodes.
"""

from typing import Dict, List, Optional, Annotated
from typing_extensions import TypedDict
import operator

import pandas as pd


class ScoutingState(TypedDict):
    """The shared state that flows through the LangGraph workflow."""

    # === User Input ===
    user_request: str
    tactical_system: Optional[str]
    position: Optional[str]
    budget_max: Optional[float]

    # === Request Parser Output ===
    search_constraints: Optional[Dict]
    benchmark_player: Optional[str]
    key_attributes: Optional[List[str]]

    # === Player Finder Output ===
    candidate_pool: Annotated[List[Dict], operator.add]

    # === Background Checker Output ===
    character_reports: Annotated[List[Dict], operator.add]

    # === Financial Analyst Output ===
    financial_assessments: Annotated[List[Dict], operator.add]

    # === Report Writer Output ===
    scouting_reports: Annotated[List[Dict], operator.add]

    # === Workflow Metadata ===
    messages: Annotated[List, operator.add]
    errors: Annotated[List[str], operator.add]
