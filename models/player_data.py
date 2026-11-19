"""
Player Data Models
==================
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProposedCandidate(BaseModel):
    """A name the finder agent puts forward. Nothing here is trusted until verified."""

    name: str = Field(description="Full player name as commonly written.")
    team: Optional[str] = Field(default=None, description="Current club, if known.")
    claimed_age: Optional[int] = Field(default=None, description="Age if known; verified later where possible.")
    reason: str = Field(default="", description="One line on why this player fits.")


class Candidate(BaseModel):
    """A verified candidate: identity and stats come from Understat, money from Transfermarkt."""

    player_id: str
    name: str
    team: str = ""
    league: str = ""
    position: str = ""                 # role searched for
    understat_position: str = ""       # e.g. "F M S"
    season: str = ""
    age: Optional[int] = None
    age_source: str = "unknown"        # "transfermarkt" | "llm_claim" | "unknown"

    minutes_played: int = 0
    games: int = 0
    goals: int = 0
    assists: int = 0
    goals_p90: float = 0.0
    assists_p90: float = 0.0
    xG_p90: float = 0.0
    xA_p90: float = 0.0
    npxG_p90: float = 0.0
    shots_p90: float = 0.0
    key_passes_p90: float = 0.0
    xGChain_p90: float = 0.0
    xGBuildup_p90: float = 0.0
    yellow_cards: int = 0
    red_cards: int = 0

    # EA FC attributes (0-99) when FUTWIZ has the player
    ea_overall: Optional[int] = None
    ea_pace: Optional[int] = None
    ea_dribbling: Optional[int] = None
    ea_passing: Optional[int] = None
    ea_defending: Optional[int] = None
    ea_physical: Optional[int] = None

    market_value_eur_m: Optional[float] = None
    contract_end_date: str = ""
    agent: str = ""

    league_coefficient: float = 1.0
    adjusted_for_pl: bool = False
    why_recommended: str = ""
    sources: List[str] = Field(default_factory=list)

    # Forecast model (v3): next season's projection at the target club, when the model is built.
    projection: Dict[str, float] = Field(default_factory=dict)   # pred_/lo_/hi_<metric>, expected_minutes
    role_score: Optional[float] = None                            # 0 = average player in the role
    value_gap: Optional[float] = None                             # 0.5 = output suggests worth 50% more than price
