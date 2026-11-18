"""
Workflow State
==============
Shared state for the LangGraph workflow.

How updates work: a node returns ONLY the keys it changed. For keys with a reducer
(the Annotated ones), LangGraph combines the node's value with the current value
using that reducer; other keys are simply overwritten.

- `messages`, `errors`, `rejected` append (operator.add), so parallel nodes can
  both write to them safely.
- `character_reports` and `financial_assessments` are dicts keyed by player_id and
  merged, so results are joined by id, never by list position.
"""

import operator
from typing import Annotated, Dict, List, Optional

from typing_extensions import TypedDict


def merge_dicts(left: Optional[Dict], right: Optional[Dict]) -> Dict:
    return {**(left or {}), **(right or {})}


class ScoutingState(TypedDict, total=False):
    # Input
    user_request: str

    # Request parser
    constraints: Dict                       # SearchConstraints.model_dump()

    # Player finder (agent) <-> verifier loop
    proposals: List[Dict]                   # this attempt's ProposedCandidate dicts (overwritten)
    finder_attempts: int
    rejected: Annotated[List[Dict], operator.add]   # {"name", "reason"}: fed back to the finder
    candidates: List[Dict]                  # verified Candidate dicts (overwritten by verifier)

    # Parallel checks, keyed by player_id
    character_reports: Annotated[Dict[str, Dict], merge_dicts]
    financial_assessments: Annotated[Dict[str, Dict], merge_dicts]

    # Report writer
    scouting_reports: List[Dict]

    # Metadata
    messages: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
