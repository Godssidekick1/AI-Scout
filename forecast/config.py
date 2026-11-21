"""
Shared constants for the forecast model: which metrics it predicts, which
leagues it covers, and how players are grouped by position.
"""

from typing import Dict, Optional, Tuple

# Season-total columns from Understat. The model predicts each one per 90.
METRICS: Tuple[str, ...] = ("npxG", "xA", "shots", "key_passes", "xGChain", "xGBuildup")

# v3 scope: the big five. Keys match data_sources.understat_source.UNDERSTAT_LEAGUES.
LEAGUES: Tuple[str, ...] = ("premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1")
REFERENCE_LEAGUE = "premier_league"   # league effects are measured relative to this one

# Transfermarkt competition ids for the same leagues (transfermarkt-datasets).
TM_COMPETITIONS: Dict[str, str] = {
    "premier_league": "GB1",
    "la_liga": "ES1",
    "bundesliga": "L1",
    "serie_a": "IT1",
    "ligue_1": "FR1",
}

# Position groups. The league tax and the age curve are fitted per group,
# because the research shows both differ by position.
POSITION_GROUPS: Tuple[str, ...] = ("CB", "FB", "CM", "AM", "ST")

_TM_SUB_POSITION_GROUP = {
    "Centre-Back": "CB",
    "Left-Back": "FB", "Right-Back": "FB",
    "Defensive Midfield": "CM", "Central Midfield": "CM",
    "Left Midfield": "AM", "Right Midfield": "AM", "Attacking Midfield": "AM",
    "Left Winger": "AM", "Right Winger": "AM",
    "Second Striker": "ST", "Centre-Forward": "ST",
}


def position_group(tm_sub_position: Optional[str] = None, understat_position: str = "") -> Optional[str]:
    """Map a Transfermarkt sub-position (preferred) or an Understat code string to a group.
    Goalkeepers return None: v3 does not model them until event data is available."""
    if tm_sub_position:
        if tm_sub_position == "Goalkeeper":
            return None
        group = _TM_SUB_POSITION_GROUP.get(tm_sub_position)
        if group:
            return group
    codes = set((understat_position or "").split())
    if "GK" in codes:
        return None
    if "F" in codes:
        return "ST" if "M" not in codes else "AM"
    if "M" in codes:
        return "CM"
    if "D" in codes:
        return "CB"
    return None


def season_label(season: int) -> str:
    """2024 -> '2024-25' (seasons are named by their starting year)."""
    return f"{season}-{str(season + 1)[-2:]}"
