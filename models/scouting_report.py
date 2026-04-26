"""
Scouting Report Models
======================
Data structures for character reports, financial assessments, and the final scouting report.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CharacterReport(BaseModel):
    """Character and behavioral assessment from news/media analysis."""

    player_name: str = ""
    attitude_score: float = Field(default=7.5, ge=0, le=10)
    discipline_issues: List[str] = []
    injury_history: List[str] = []
    injury_risk_score: float = Field(default=2.0, ge=0, le=10)
    total_days_injured_24mo: int = 0
    maturity_assessment: str = ""
    locker_room_presence: str = ""
    sources: List[str] = []


class FinancialAssessment(BaseModel):
    """Financial viability analysis from market data."""

    player_name: str = ""
    current_market_value_eur: float = 0       # in millions
    contract_end_date: str = ""
    years_remaining: float = 0
    agent: str = ""
    release_clause_eur: Optional[float] = None
    estimated_wage_weekly_eur: float = 0
    value_for_money_score: float = Field(default=50.0, ge=0, le=100)
    transfer_feasibility: str = ""            # "Easy", "Moderate", "Difficult"


class ScoutingReport(BaseModel):
    """Final comprehensive scouting report for a player."""

    # --- Player Info ---
    player_name: str = ""
    position: str = ""
    age: int = 0
    team: str = ""
    league: str = ""
    nationality: str = ""

    # --- Scores ---
    tactical_fit_score: float = Field(default=0, ge=0, le=100)
    statistical_rating: float = Field(default=0, ge=0, le=100)
    character_score: float = Field(default=7.5, ge=0, le=10)
    financial_viability: float = Field(default=50, ge=0, le=100)
    injury_risk: float = Field(default=2.0, ge=0, le=10)

    # --- Verdict ---
    overall_score: float = Field(default=0, ge=0, le=100)
    verdict: str = "CAUTION"       # "GO", "CAUTION", "NO-GO"

    # --- Analysis ---
    strengths: List[str] = []
    weaknesses: List[str] = []
    risk_factors: List[str] = []

    # --- Comparisons ---
    similar_players: List[str] = []
    benchmark_comparison: Dict[str, float] = {}

    # --- Detailed Data ---
    character_report: Optional[CharacterReport] = None
    financial_assessment: Optional[FinancialAssessment] = None

    # --- LLM narrative ---
    executive_summary: str = ""
