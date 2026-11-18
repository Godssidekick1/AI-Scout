"""
Scouting Report Models
======================
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CharacterReport(BaseModel):
    player_id: str
    player_name: str = ""
    attitude_score: float = Field(default=7.5, ge=0, le=10)
    discipline_issues: List[str] = Field(default_factory=list)
    injury_history: List[str] = Field(default_factory=list)
    injury_risk_score: float = Field(default=2.0, ge=0, le=10)
    total_days_injured: int = 0
    maturity_assessment: str = "No data available"
    locker_room_presence: str = "Unknown"
    key_concerns: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class FinancialAssessment(BaseModel):
    player_id: str
    player_name: str = ""
    current_market_value_eur: float = 0             # millions
    market_value_is_estimate: bool = False
    contract_end_date: str = ""
    years_remaining: Optional[float] = None
    agent: str = ""
    estimated_wage_weekly_eur: float = 0
    value_for_money_score: float = Field(default=50.0, ge=0, le=100)
    transfer_feasibility: str = "Unknown"


class ScoutingReport(BaseModel):
    """Final report for one player. Every pipeline output is validated against this."""

    player_id: str
    player_name: str
    position: str = ""
    understat_position: str = ""
    age: Optional[int] = None
    team: str = ""
    league: str = ""
    season: str = ""

    tactical_fit_score: float = Field(ge=0, le=100)
    tactical_fit_coverage: float = Field(ge=0, le=1, description="Share of the fit model's features that had data.")
    statistical_rating: float = Field(ge=0, le=100)
    character_score: float = Field(ge=0, le=10)
    financial_viability: float = Field(ge=0, le=100)
    injury_risk: float = Field(ge=0, le=10)

    overall_score: float = Field(ge=0, le=100)
    verdict: str                                     # "GO" | "CAUTION" | "NO-GO"

    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)

    similar_players: List[str] = Field(default_factory=list)
    benchmark_comparison: Dict[str, float] = Field(default_factory=dict)

    key_stats: Dict[str, float] = Field(default_factory=dict)
    character_report: Optional[CharacterReport] = None
    financial_assessment: Optional[FinancialAssessment] = None
    executive_summary: str = ""
    data_sources: List[str] = Field(default_factory=list)
