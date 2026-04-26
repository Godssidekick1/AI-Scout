"""
League Normalizer (League Tax)
==============================
Adjusts player stats across different leagues to Premier League equivalence.
A player scoring 0.5 goals/90 in the Eredivisie isn't directly comparable
to 0.5 goals/90 in the Premier League — this engine handles that.
"""

from typing import Dict, List


# League difficulty coefficients (1.0 = Premier League level)
# Based on UEFA club coefficients and competitive intensity analysis
LEAGUE_COEFFICIENTS = {
    # Top 5 leagues
    "premier_league": 1.00,
    "la_liga": 0.88,
    "bundesliga": 0.87,
    "serie_a": 0.85,
    "ligue_1": 0.78,
    # Strong leagues
    "eredivisie": 0.70,
    "primeira_liga": 0.68,
    "belgian_pro_league": 0.65,
    "championship": 0.65,
    "scottish_premiership": 0.55,
    # Other notable leagues
    "saudi_pro_league": 0.55,
    "mls": 0.60,
    "liga_mx": 0.58,
    "super_lig": 0.62,    # Turkish
    "austrian_bundesliga": 0.58,
    "swiss_super_league": 0.56,
    "j_league": 0.55,
    "k_league": 0.52,
    "a_league": 0.48,
}

# Offensive metrics that get "taxed" (scaled down for weaker leagues)
OFFENSIVE_METRICS = [
    "goals_p90", "assists_p90", "xG_p90", "xA_p90",
    "xT_p90", "shot_creating_actions_p90",
    "progressive_carries_p90", "passes_into_penalty_area_p90",
    "pv_plus", "key_passes_p90",
]

# Defensive metrics that scale up (defending in high-intensity leagues is harder)
DEFENSIVE_METRICS = [
    "tackles_p90", "interceptions_p90",
    "pressing_success_pct", "defensive_duel_win_rate",
    "blocks_p90",
]


def get_coefficient(league_name: str) -> float:
    """Get the league difficulty coefficient."""
    return LEAGUE_COEFFICIENTS.get(league_name.lower().replace(" ", "_"), 0.55)


def apply_league_tax(stats: Dict, league_name: str) -> Dict:
    """
    Adjust player stats to Premier League equivalence.

    Offensive stats get reduced (it's harder to score in the PL).
    Defensive stats get slightly boosted (defending in weaker leagues
    is less intense, so those stats are slightly inflated).

    Args:
        stats: Dictionary of player statistics
        league_name: Name of the player's league

    Returns:
        Adjusted statistics dictionary with league tax applied
    """
    coef = get_coefficient(league_name)
    adjusted = stats.copy()

    # Tax offensive metrics
    for metric in OFFENSIVE_METRICS:
        if metric in adjusted and adjusted[metric] is not None:
            adjusted[metric] = round(adjusted[metric] * coef, 3)

    # Scale up defensive metrics (defending in faster leagues is harder)
    defensive_scale = 1 + (1 - coef) / 3
    for metric in DEFENSIVE_METRICS:
        if metric in adjusted and adjusted[metric] is not None:
            adjusted[metric] = round(adjusted[metric] * defensive_scale, 3)

    adjusted["league_coefficient"] = coef
    adjusted["adjusted_for_pl"] = True

    return adjusted


def league_adaptability_score(
    current_league: str,
    age: int,
    previous_leagues: List[str] = None,
) -> float:
    """
    Predict how well a player will adapt to the Premier League.

    Considers:
        - Current league quality (closer to PL = easier adaptation)
        - Age (21-27 is peak adaptability)
        - Multi-league experience (exposure to different styles)

    Returns:
        Score from 0-100 (100 = very likely to adapt well)
    """
    previous_leagues = previous_leagues or []
    base_coef = get_coefficient(current_league)

    # Start from league quality (0-100)
    score = base_coef * 100

    # Age factor
    if 21 <= age <= 27:
        score += 10  # Peak adaptability window
    elif 18 <= age <= 20 or 28 <= age <= 29:
        score += 5   # Young enough or still adaptable
    # 30+ gets no age bonus

    # Multi-league experience
    if previous_leagues:
        top_league_count = sum(
            1 for lg in previous_leagues
            if get_coefficient(lg) >= 0.75
        )
        score += min(top_league_count * 5, 15)

    return min(round(score, 1), 100)
