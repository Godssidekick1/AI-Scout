"""
LLM I/O Schemas
===============
Every LLM call returns one of these pydantic models (via Gemini structured output),
so the field descriptions double as the prompt's output spec and the result is
validated before any agent touches it.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

LeagueKey = Literal["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]


RoleKey = Literal["poacher", "complete_forward", "inside_forward", "creative_winger", "playmaker",
                  "box_to_box", "deep_playmaker", "attacking_fullback", "ball_playing_cb"]


class SearchConstraints(BaseModel):
    """Structured version of the user's scouting request."""

    position: str = Field(description="Playing role wanted, e.g. 'inverted winger', 'box-to-box midfielder', 'right-back'.")
    tactical_system: Optional[str] = Field(default=None, description="Coach or system named, e.g. 'Arne Slot 4-3-3', 'Guardiola possession'. Null if none.")
    budget_max_m: Optional[float] = Field(default=None, description="Maximum transfer fee in millions of euros. Null if not stated.")
    max_age: Optional[int] = Field(default=None, description="Maximum age in years ('under 25' means 24). Null if not stated.")
    benchmark_player: Optional[str] = Field(default=None, description="A reference player the user compares to ('like Salah'). Null if none.")
    key_attributes: List[str] = Field(default_factory=list, description="Qualities the user emphasises, e.g. 'pressing', 'pace'.")
    target_leagues: List[LeagueKey] = Field(default_factory=list, description="Leagues to search; empty means all five.")
    min_minutes: int = Field(default=600, description="Minimum league minutes this season for a reliable sample.")
    role: Optional[RoleKey] = Field(default=None, description="Closest role profile to what the user wants, e.g. a "
                                    "John Stones replacement is ball_playing_cb, a 'fox in the box' is poacher.")
    target_team: Optional[str] = Field(default=None, description="The club the player would join, if stated "
                                       "('for Man City'). Null if not stated.")


class CharacterAnalysis(BaseModel):
    """What news coverage says about a player's character."""

    attitude_score: float = Field(ge=0, le=10, description="0-10; 7.5 means nothing notable either way.")
    discipline_issues: List[str] = Field(default_factory=list, description="Concrete incidents found in the articles only.")
    maturity_assessment: str = Field(description="One sentence on professional maturity, based only on the articles.")
    locker_room_presence: Literal["Positive", "Neutral", "Negative", "Unknown"]
    key_concerns: List[str] = Field(default_factory=list, description="Red flags a buying club should know about.")


class PlayerSummary(BaseModel):
    player_id: str
    summary: str = Field(description="3-4 sentences: tactical fit, the #1 strength, the #1 risk, a clear recommendation.")


class ExecutiveSummaries(BaseModel):
    summaries: List[PlayerSummary]


class SummaryJudgement(BaseModel):
    """LLM-as-judge verdict on one executive summary (used by evals)."""

    faithfulness: int = Field(ge=1, le=5, description="5 = every claim is supported by the facts given; 1 = invents facts.")
    has_clear_recommendation: bool
    reasoning: str = Field(description="One sentence.")
