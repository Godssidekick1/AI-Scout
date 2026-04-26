"""
Player Data Models
==================
Data structures for player profiles, statistics, and game attributes.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PlayerProfile(BaseModel):
    """Basic biographical and career information about a player."""

    player_id: str = ""
    name: str
    age: int = 0
    nationality: str = ""
    team: str = ""
    league: str = ""
    position: str = ""
    secondary_positions: List[str] = []
    contract_end: str = ""        # e.g. "2027-06-30"
    market_value_eur: float = 0   # in millions
    agent: str = ""


class PlayerStats(BaseModel):
    """Per-90 performance statistics from real data sources."""

    player_id: str = ""
    name: str = ""
    season: str = ""               # e.g. "2025-26"
    minutes_played: int = 0

    # --- Attacking / Creative ---
    goals_p90: float = 0.0
    assists_p90: float = 0.0
    xG_p90: float = 0.0            # Expected Goals
    xA_p90: float = 0.0            # Expected Assists
    xT_p90: float = 0.0            # Expected Threat
    shot_creating_actions_p90: float = 0.0
    passes_into_penalty_area_p90: float = 0.0
    pv_plus: float = 0.0           # Possession Value Added

    # --- Ball Progression ---
    progressive_carries_p90: float = 0.0
    progressive_carries_distance: float = 0.0
    progressive_passes_p90: float = 0.0

    # --- Defensive ---
    tackles_p90: float = 0.0
    interceptions_p90: float = 0.0
    pressing_success_pct: float = 0.0
    defensive_duel_win_rate: float = 0.0
    blocks_p90: float = 0.0

    # --- Physical ---
    high_intensity_sprints_p90: float = 0.0

    # --- Discipline ---
    yellow_cards: int = 0
    red_cards: int = 0

    # --- Meta ---
    league_coefficient: float = 1.0     # After league-tax adjustment
    adjusted_for_pl: bool = False


class PlayerAttributes(BaseModel):
    """
    EA FC 26-style player attributes.
    Sourced from FUTWIZ (live form) to reflect current-season performance.
    """

    player_id: str = ""
    name: str = ""
    overall_rating: int = 0

    # --- Core 6 attributes (0-99 scale) ---
    pace: int = 0
    shooting: int = 0
    passing: int = 0
    dribbling: int = 0
    defending: int = 0
    physical: int = 0

    # --- Extras ---
    weak_foot: int = 0              # 1-5 stars
    skill_moves: int = 0            # 1-5 stars
    work_rate_attack: str = ""      # High / Medium / Low
    work_rate_defense: str = ""     # High / Medium / Low

    # --- Live form ---
    form_modifier: int = 0          # Live form adjustment (-3 to +3)
    is_in_form: bool = False


class MergedPlayerData(BaseModel):
    """
    All data about a single player merged from multiple sources.
    This is what agents work with internally.
    """

    profile: PlayerProfile = Field(default_factory=PlayerProfile)
    stats: PlayerStats = Field(default_factory=PlayerStats)
    attributes: Optional[PlayerAttributes] = None
    data_sources: List[str] = []    # Which sources contributed data
