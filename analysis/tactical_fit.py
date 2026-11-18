"""
Tactical Fit & Statistical Rating
=================================
Transparent, deterministic scoring. Each tactical system is a weight vector over
features we actually have data for (Understat per-90s + optional EA FC attributes).

    fit = 100 * sum(w_i * f_i) / sum(w_i)      over features with data

where f_i is the feature scaled to 0-1 against a fixed "elite" reference value.
Features without data are dropped and the weights renormalised, and the share of
weight that had data is reported as `coverage`, so a score built on 4 of 9
features says so instead of pretending.
"""

from typing import Dict, Optional, Tuple

# Feature -> value treated as elite (scores 1.0). Per-90 values are after league tax.
FEATURE_SCALES: Dict[str, float] = {
    "xG_p90": 0.70,
    "npxG_p90": 0.60,
    "xA_p90": 0.40,
    "key_passes_p90": 3.0,
    "shots_p90": 4.0,
    "xGChain_p90": 1.00,     # involvement in attacking possessions
    "xGBuildup_p90": 0.60,   # involvement in build-up, excluding shots and key passes
    "ea_pace": 99,
    "ea_dribbling": 99,
    "ea_passing": 99,
    "ea_defending": 99,
    "ea_physical": 99,
}

SYSTEM_PROFILES: Dict[str, Dict[str, float]] = {
    # Progressive, technical, pressing from the front
    "slot": {"xGChain_p90": 3, "xA_p90": 2, "key_passes_p90": 2, "xG_p90": 2,
             "ea_dribbling": 1, "ea_pace": 1, "ea_physical": 1},
    # Possession: build-up involvement and chance creation over shooting volume
    "guardiola": {"xGBuildup_p90": 3, "xGChain_p90": 3, "key_passes_p90": 2, "xA_p90": 2,
                  "ea_passing": 2, "ea_dribbling": 1},
    # Positional play with high work rate
    "arteta": {"xGChain_p90": 3, "xGBuildup_p90": 2, "xA_p90": 2, "npxG_p90": 2,
               "ea_defending": 1, "ea_physical": 1},
    # Intensity, directness, end product
    "conte": {"npxG_p90": 3, "shots_p90": 2, "xGChain_p90": 2,
              "ea_physical": 2, "ea_pace": 2, "ea_defending": 1},
    "general": {"xGChain_p90": 2, "xG_p90": 2, "xA_p90": 2, "key_passes_p90": 1,
                "xGBuildup_p90": 1, "ea_pace": 1, "ea_dribbling": 1},
}

_SYSTEM_KEYWORDS = [
    ("slot", ("slot", "liverpool", "4-3-3")),
    ("guardiola", ("guardiola", "pep", "possession", "city")),
    ("arteta", ("arteta", "arsenal")),
    ("conte", ("conte", "3-5-2", "3-4-3")),
]

# Overall statistical quality by position family
RATING_PROFILES: Dict[str, Dict[str, float]] = {
    "forward": {"npxG_p90": 3, "xA_p90": 2, "xGChain_p90": 2, "shots_p90": 1, "key_passes_p90": 1},
    "midfielder": {"xGChain_p90": 3, "xGBuildup_p90": 3, "xA_p90": 2, "key_passes_p90": 2, "npxG_p90": 1},
    "defender": {"xGBuildup_p90": 3, "xGChain_p90": 2, "xA_p90": 1, "ea_defending": 3, "ea_physical": 1},
    "goalkeeper": {"xGBuildup_p90": 1, "ea_defending": 1},
}


def system_key(tactical_system: Optional[str]) -> str:
    s = (tactical_system or "").lower()
    for key, words in _SYSTEM_KEYWORDS:
        if any(w in s for w in words):
            return key
    return "general"


def weighted_score(stats: Dict, weights: Dict[str, float]) -> Tuple[float, float]:
    """Returns (score 0-100, coverage 0-1)."""
    total_w = sum(weights.values())
    num, used_w = 0.0, 0.0
    for feat, w in weights.items():
        val = stats.get(feat)
        if val is None:
            continue
        num += w * min(max(float(val) / FEATURE_SCALES[feat], 0.0), 1.0)
        used_w += w
    if used_w == 0:
        return 0.0, 0.0
    return round(100 * num / used_w, 1), round(used_w / total_w, 2)


def tactical_fit(stats: Dict, tactical_system: Optional[str]) -> Tuple[float, float]:
    return weighted_score(stats, SYSTEM_PROFILES[system_key(tactical_system)])


def statistical_rating(stats: Dict, family: str) -> float:
    score, _ = weighted_score(stats, RATING_PROFILES.get(family, RATING_PROFILES["forward"]))
    return score
