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


EmphasisKey = Literal["build_up", "chance_creation", "goal_threat", "shooting", "involvement"]
PositionKey = Literal["winger", "striker", "attacking_mid", "central_mid", "defensive_mid", "fullback", "centre_back"]


class BeatSpec(BaseModel):
    player: str = Field(description="Player to out-produce, as named.")
    quality: EmphasisKey = Field(description="What to beat him at: goal_threat for goals or scoring.")


class BriefSpec(BaseModel):
    """A scout's free-text brief, read into the forecast model's filters (forecast.brief)."""

    interpretation: str = Field(description="One plain sentence restating what the user is asking for.")
    replace_player: Optional[str] = Field(default=None, description="Player the user wants to REPLACE, as named "
                                          "('replace Fofana'). Null if none.")
    similar_to: Optional[str] = Field(default=None, description="Player named as a style reference: 'like Madueke', "
                                      "'a Madueke type', 'similar to Rodri'. Null if none.")
    must_beat: List[BeatSpec] = Field(default_factory=list, description="Players the target must out-produce: 'outdo "
                                      "Saka for goals' -> player 'Saka', quality goal_threat.")
    positions: List[PositionKey] = Field(default_factory=list, description="Positions wanted. A winger of any kind "
                                         "(inside, inverted, wide forward) is winger, never striker.")
    side: Optional[Literal["left", "right"]] = Field(default=None, description="Side mentioned ('can play on the "
                                                     "left', 'left winger').")
    side_only: bool = Field(default=False, description="True only if the player must play that side ('a left "
                                            "winger'); false for 'can play on the left too'.")
    squad_role: Optional[Literal["starter", "rotation", "backup"]] = Field(default=None, description="Squad role "
                                                                            "stated: 'backup', 'squad player', 'starter'.")
    upside: bool = Field(default=False, description="True when the user wants potential or room to grow.")
    target_club: Optional[str] = Field(default=None, description="Club the player would join ('for Chelsea', 'my team "
                                       "is Arsenal'). Null if not stated.")
    role: Optional[RoleKey] = Field(default=None, description="Closest role profile. A goalscoring or inside winger is "
                                    "inside_forward; a creative/crossing winger is creative_winger; a ball-playing or "
                                    "Stones-type centre-back is ball_playing_cb; a holding/deep passer is deep_playmaker.")
    emphasis: List[EmphasisKey] = Field(default_factory=list, description="Measurable qualities stressed: build_up "
                                        "(progressive passing, on the ball, distribution), chance_creation (creative, "
                                        "assists, key passes), goal_threat (goals, finishing), shooting (shot volume), "
                                        "involvement (link-up, combination play).")
    min_age: Optional[int] = Field(default=None, description="Minimum age in years. 'between 25-30' means 25.")
    max_age: Optional[int] = Field(default=None, description="Maximum age in years. 'under 25' means 24, 'u21' means 21.")
    younger_than_replaced: bool = Field(default=False, description="True when the user wants someone younger than the "
                                        "replaced player without giving an age.")
    max_value_eur_m: Optional[float] = Field(default=None, description="Maximum market value or budget in millions of "
                                             "euros (treat pounds as euros). Null if not stated.")
    include_leagues: List[LeagueKey] = Field(default_factory=list, description="Only these leagues ('from Serie A'). "
                                             "Empty means no restriction.")
    exclude_leagues: List[LeagueKey] = Field(default_factory=list, description="Leagues to leave out ('outside the "
                                             "Premier League').")
    min_minutes: Optional[int] = Field(default=None, description="Minimum league minutes last season: 1800 for "
                                       "'reliable', 'regular starter', 'proven'; 450 for a squad or rotation player.")
    min_goals: Optional[float] = Field(default=None, description="League goals per season the user requires ('15+ goals').")
    min_assists: Optional[float] = Field(default=None, description="League assists per season the user requires.")
    hidden_talent: bool = Field(default=False, description="True for hidden, unknown, undervalued, bargain or "
                                "'not the obvious names' requests.")
    unmeasured: List[str] = Field(default_factory=list, description="Asks the six stats cannot measure, as short "
                                  "phrases: defending, tackling, aerials, pace, tracking back, work rate, versatility, "
                                  "leadership, physicality, goalkeeping, character.")


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
