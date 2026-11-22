"""
Shared constants for the forecast model: which metrics it predicts, which
leagues it covers, and how players are grouped by position.
"""

from typing import Dict, Optional, Tuple

# Season-total columns from Understat. The model predicts each one per 90.
METRICS: Tuple[str, ...] = ("npxG", "xA", "shots", "key_passes", "xGChain", "xGBuildup")

# Defensive / duel / progression metrics from forecast/data/defensive.py. These are observed
# per-90 rates (`{m}_p90`), not projected by the model: free coverage starts in 2017-18 (2008-09
# for the Premier League), too short for the season-pair model, so roles use recent observed
# rates shrunk toward the position mean.
DEF_METRICS: Tuple[str, ...] = ("tackles", "interceptions", "tkl_int", "blocks", "clearances", "aerials_won",
                                "aerials", "recoveries", "duels_won", "prog_passes", "prog_carries", "take_ons_won")
# success ratios: name -> (numerator, denominator) over the season totals
DEF_RATIOS: Dict[str, Tuple[str, str]] = {"aerial_win_pct": ("aerials_won", "aerials"),
                                          "tackle_win_pct": ("tackles_won", "tackles"),
                                          "duel_win_pct": ("duels_won", "duels"),
                                          "pass_pct": ("passes_completed", "passes_attempted"),
                                          "take_on_pct": ("take_ons_won", "take_ons")}
# On-the-ball totals (FBref passing and possession tables). Kept apart from the defensive totals: FBref's
# passing table covers seasons its defense table doesn't (all of 2024-25), so they are linked on their own.
BALL_TOTALS: Tuple[str, ...] = ("passes_completed", "passes_attempted", "take_ons_won", "take_ons")
BALL_RATES: Tuple[str, ...] = ("take_ons_won_p90", "pass_pct", "take_on_pct")
DEF_LABELS: Dict[str, str] = {
    "tackles": "tackles", "interceptions": "interceptions", "tkl_int": "tackles + interceptions",
    "blocks": "blocks", "clearances": "clearances", "aerials_won": "aerials won", "aerials": "aerial duels",
    "recoveries": "ball recoveries", "duels_won": "duels won", "prog_passes": "progressive passes",
    "prog_carries": "progressive carries", "aerial_win_pct": "aerial win %", "tackle_win_pct": "tackle win %",
    "duel_win_pct": "duel win %", "take_ons_won": "successful take-ons", "pass_pct": "pass completion %",
    "take_on_pct": "take-on success %"}

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


# Finer positions a brief can ask for (Transfermarkt main position; it lists one per player).
POSITIONS = {
    "winger": ("Left Winger", "Right Winger", "Left Midfield", "Right Midfield"),
    "striker": ("Centre-Forward", "Second Striker"),
    "attacking_mid": ("Attacking Midfield",),
    "central_mid": ("Central Midfield",),
    "defensive_mid": ("Defensive Midfield",),
    "fullback": ("Left-Back", "Right-Back"),
    "centre_back": ("Centre-Back",),
}
SIDES = {"left": ("Left Winger", "Left Midfield", "Left-Back"), "right": ("Right Winger", "Right Midfield", "Right-Back")}
POSITION_SHORT = {"Left Winger": "LW", "Right Winger": "RW", "Left Midfield": "LM", "Right Midfield": "RM",
                  "Centre-Forward": "CF", "Second Striker": "SS", "Attacking Midfield": "AM",
                  "Central Midfield": "CM", "Defensive Midfield": "DM", "Left-Back": "LB", "Right-Back": "RB",
                  "Centre-Back": "CB"}


def groups_for(positions) -> tuple:
    """Position groups that contain these finer positions (for players with no Transfermarkt match)."""
    return tuple(dict.fromkeys(_TM_SUB_POSITION_GROUP[s] for p in positions for s in POSITIONS[p]))


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
