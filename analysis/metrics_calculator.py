"""
Advanced Metrics Calculator
===========================
Computes xT (Expected Threat), possession-adjusted stats,
pressing intensity, and composite progression scores.
"""

from typing import List, Tuple
import numpy as np


# ──────────────────────────────────────────────
# Expected Threat (xT) Grid
# 12 columns (pitch length) × 8 rows (pitch width)
# Values represent the probability of a goal from each zone.
# ──────────────────────────────────────────────
XT_GRID = np.array([
    [0.000, 0.000, 0.000, 0.000, 0.001, 0.001, 0.001, 0.002, 0.010, 0.010, 0.020, 0.030],
    [0.000, 0.001, 0.001, 0.001, 0.001, 0.001, 0.010, 0.020, 0.020, 0.030, 0.040, 0.050],
    [0.000, 0.001, 0.010, 0.020, 0.020, 0.020, 0.030, 0.030, 0.040, 0.050, 0.080, 0.120],
    [0.001, 0.010, 0.020, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.120, 0.150, 0.200],
    [0.001, 0.010, 0.020, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.120, 0.150, 0.200],
    [0.000, 0.001, 0.010, 0.020, 0.020, 0.020, 0.030, 0.030, 0.040, 0.050, 0.080, 0.120],
    [0.000, 0.001, 0.001, 0.001, 0.001, 0.001, 0.010, 0.020, 0.020, 0.030, 0.040, 0.050],
    [0.000, 0.000, 0.000, 0.000, 0.001, 0.001, 0.001, 0.002, 0.010, 0.010, 0.020, 0.030],
])


def calculate_xT(
    pass_data: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> float:
    """
    Calculate Expected Threat from pass/carry coordinates.

    Args:
        pass_data: List of ((start_x, start_y), (end_x, end_y)) tuples.
                   Coordinates in meters (pitch = 105m × 68m).

    Returns:
        Total positive xT gained.
    """
    total_xT = 0.0

    for (start_x, start_y), (end_x, end_y) in pass_data:
        # Map to grid coordinates
        sx = int(min(max(start_x / 105 * 12, 0), 11))
        sy = int(min(max(start_y / 68 * 8, 0), 7))
        ex = int(min(max(end_x / 105 * 12, 0), 11))
        ey = int(min(max(end_y / 68 * 8, 0), 7))

        # xT = threat gained by moving the ball forward
        gain = XT_GRID[ey, ex] - XT_GRID[sy, sx]
        total_xT += max(gain, 0)  # Only count positive threat gains

    return round(total_xT, 3)


def possession_adjusted_tackles(tackles_p90: float, team_possession_pct: float) -> float:
    """
    Adjust tackles for team possession.
    Teams with less possession naturally make more tackles.

    Args:
        tackles_p90: Raw tackles per 90 minutes
        team_possession_pct: Average team possession (0-100)

    Returns:
        Possession-adjusted tackles per 90
    """
    if team_possession_pct >= 100:
        return tackles_p90

    # Normalize to a 50% possession baseline
    adjustment = 50 / (100 - team_possession_pct)
    return round(tackles_p90 * adjustment, 2)


def pressing_intensity(
    pressures_p90: float, successful_pressures_p90: float
) -> float:
    """
    Calculate a combined pressing intensity score.

    Returns:
        Score from 0-100
    """
    if pressures_p90 == 0:
        return 0.0

    success_rate = successful_pressures_p90 / pressures_p90
    # Combine volume and success rate
    intensity = (pressures_p90 * 2 + success_rate * 100) / 3
    return round(min(intensity, 100), 2)


def progression_score(
    progressive_carries_p90: float,
    progressive_passes_p90: float,
    passes_into_penalty_area_p90: float,
    progressive_carries_distance: float = 0,
) -> float:
    """
    Calculate a composite ball progression score.

    Returns:
        Score from 0-100
    """
    score = (
        progressive_carries_p90 * 5
        + progressive_passes_p90 * 4
        + passes_into_penalty_area_p90 * 8
        + progressive_carries_distance / 100 * 3
    )
    return round(min(score, 100), 2)


def creative_output_score(
    xA_p90: float,
    shot_creating_actions_p90: float,
    key_passes_p90: float = 0,
) -> float:
    """
    Calculate a creative output composite score.

    Returns:
        Score from 0 to 100
    """
    score = (
        xA_p90 * 30
        + shot_creating_actions_p90 * 5
        + key_passes_p90 * 8
    )
    return round(min(score, 100), 2)


def goal_threat_score(
    goals_p90: float,
    xG_p90: float,
    npxG_p90: float = 0,
    shots_p90: float = 0,
) -> float:
    """
    Calculate a goal threat composite score.
    Rewards clinical finishing (outperforming xG).

    Returns:
        Score from 0 to 100
    """
    finishing_quality = goals_p90 / max(xG_p90, 0.01)  # >1.0 means clinical
    finishing_bonus = max(0, (finishing_quality - 1.0) * 10)

    score = (
        goals_p90 * 20
        + xG_p90 * 15
        + finishing_bonus
        + shots_p90 * 1.5
    )
    return round(min(score, 100), 2)
