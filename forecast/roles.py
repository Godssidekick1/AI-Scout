"""
Role profiles and style vectors
===============================
A brief ("a Stones replacement", "a poacher") becomes a role profile: weights
over the metrics the model projects. The same metrics give each player a style
vector, the *shape* of his output with level removed, used for "plays like"
retrieval in ChromaDB.

Weights can name two kinds of metric:
  - METRICS (npxG, xA, ...): projected per-90 next season by the model (`pred_*`).
  - DEF_METRICS / DEF_RATIOS (tackles + interceptions, aerials won, aerial win %,
    progressive passes...): the player's recent observed rates from
    forecast/data/defensive.py (`def_*`), shrunk toward his position's mean.
    Free coverage: Premier League 2008-09 onward, Serie A recent seasons, all
    big five 2017-18 to 2023-24. A player without coverage scores as average
    on these (z = 0), and `def_minutes_used` says so.
Keeper distribution and shot-stopping are still not covered (no keepers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecast.config import DEF_METRICS, DEF_RATIOS, METRICS

DEF_KEYS = frozenset(DEF_METRICS) | frozenset(DEF_RATIOS)


@dataclass(frozen=True)
class RoleProfile:
    key: str
    label: str
    groups: Tuple[str, ...]
    weights: Dict[str, float]
    proxy: bool = False          # True when the role's key skills are not in the free data
    note: str = ""


ROLES: Dict[str, RoleProfile] = {r.key: r for r in [
    RoleProfile("poacher", "Poacher", ("ST",), {"npxG": 0.6, "shots": 0.3, "xGChain": 0.1}),
    RoleProfile("complete_forward", "Complete forward", ("ST",),
                {"npxG": 0.4, "xA": 0.2, "xGChain": 0.25, "key_passes": 0.15}),
    RoleProfile("inside_forward", "Inside forward", ("AM",),
                {"npxG": 0.4, "shots": 0.2, "xA": 0.2, "xGChain": 0.2}),
    RoleProfile("creative_winger", "Creative winger", ("AM",),
                {"xA": 0.35, "key_passes": 0.3, "xGChain": 0.25, "npxG": 0.1}),
    RoleProfile("playmaker", "Attacking playmaker (no.10)", ("AM", "CM"),
                {"xA": 0.35, "key_passes": 0.35, "xGChain": 0.2, "xGBuildup": 0.1}),
    RoleProfile("box_to_box", "Box-to-box midfielder", ("CM",),
                {"xGChain": 0.25, "xGBuildup": 0.2, "npxG": 0.15, "xA": 0.15, "tkl_int": 0.15, "recoveries": 0.1}),
    RoleProfile("deep_playmaker", "Deep-lying playmaker", ("CM",),
                {"prog_passes": 0.35, "xGBuildup": 0.35, "key_passes": 0.1, "tkl_int": 0.1, "recoveries": 0.1},
                note="Progressive passes come from FBref (2022-23 onward); xGBuildup covers the rest."),
    RoleProfile("ball_winner", "Ball-winning midfielder", ("CM",),
                {"tkl_int": 0.35, "recoveries": 0.25, "tackle_win_pct": 0.1, "aerials_won": 0.1, "xGBuildup": 0.2}),
    RoleProfile("attacking_fullback", "Attacking full-back", ("FB",),
                {"xA": 0.3, "key_passes": 0.2, "xGBuildup": 0.2, "xGChain": 0.1, "prog_carries": 0.1, "tkl_int": 0.1}),
    RoleProfile("defensive_fullback", "Defensive full-back", ("FB",),
                {"tkl_int": 0.3, "tackle_win_pct": 0.15, "aerial_win_pct": 0.1, "recoveries": 0.15, "blocks": 0.1,
                 "xGBuildup": 0.2}),
    RoleProfile("ball_playing_cb", "Ball-playing centre-back", ("CB",),
                {"prog_passes": 0.3, "xGBuildup": 0.3, "prog_carries": 0.1, "aerial_win_pct": 0.1, "tkl_int": 0.1,
                 "clearances": 0.1},
                note="Progressive passes/carries come from FBref (2022-23 onward); xGBuildup covers the rest."),
    RoleProfile("stopper_cb", "Defensive centre-back (stopper)", ("CB",),
                {"aerials_won": 0.2, "aerial_win_pct": 0.2, "tkl_int": 0.2, "clearances": 0.15, "blocks": 0.1,
                 "tackle_win_pct": 0.15}),
    RoleProfile("target_forward", "Target forward", ("ST",),
                {"aerials_won": 0.3, "aerial_win_pct": 0.1, "npxG": 0.35, "xGChain": 0.25}),
    RoleProfile("pressing_forward", "Pressing forward", ("ST", "AM"),
                {"recoveries": 0.25, "tkl_int": 0.25, "npxG": 0.3, "xGChain": 0.2}),
]}

# Free-text cues -> role, used by the fallback parser and to sanity-check the LLM.
ROLE_CUES: List[Tuple[str, Tuple[str, ...]]] = [
    ("stopper_cb", ("stopper", "no-nonsense", "defensive centre-back", "defensive center-back", "commanding",
                    "dominant centre-back", "dominant center-back", "old-school")),
    ("ball_playing_cb", ("centre-back", "center-back", "centre back", "cb", "stones", "ball-playing")),
    ("defensive_fullback", ("defensive full-back", "defensive fullback", "defensive right-back", "defensive left-back",
                            "solid full-back", "solid fullback")),
    ("attacking_fullback", ("full-back", "fullback", "wing-back", "right-back", "left-back")),
    ("ball_winner", ("ball-winning", "ball winning", "ball-winner", "ball winner", "destroyer", "enforcer",
                     "defensive midfielder", "anchor")),
    ("deep_playmaker", ("deep-lying", "regista", "pivot", "number 6", "no.6", "holding")),
    ("box_to_box", ("box-to-box", "central midfielder", "cm", "number 8", "no.8")),
    ("playmaker", ("playmaker", "number 10", "no.10", "attacking midfielder")),
    ("creative_winger", ("creative winger", "winger", "wide")),
    ("inside_forward", ("inside forward", "inverted winger", "inverted", "goalscoring winger", "goal-scoring winger",
                        "scoring winger", "goalscoring wide", "wide forward", "winger who scores", "winger who can score")),
    ("poacher", ("poacher", "fox in the box", "goalscorer")),
    ("target_forward", ("target man", "target forward", "target striker", "aerial striker", "big man up front")),
    ("pressing_forward", ("pressing forward", "pressing striker", "presses from the front", "press from the front")),
    ("complete_forward", ("striker", "centre-forward", "center-forward", "number 9", "no.9", "forward")),
]


def _has(text: str, cue: str) -> bool:
    if len(cue) <= 3:                       # short codes like "cb" must be whole words
        return re.search(rf"\b{re.escape(cue)}\b", text) is not None
    return cue in text


def role_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    # most specific cues first: inverted/inside before winger
    for key in ("inside_forward", "poacher", "target_forward", "pressing_forward", "stopper_cb", "ball_winner",
                "deep_playmaker", "ball_playing_cb", "defensive_fullback", "attacking_fullback",
                "box_to_box", "playmaker", "creative_winger", "complete_forward"):
        cues = dict(ROLE_CUES)[key]
        if any(_has(t, c) for c in cues):
            return key
    return None


# ── style vectors ──────────────────────────────────────────────

def metric_scales(rows: pd.DataFrame) -> Dict[str, float]:
    """Population spread of each per-90 metric, so shots don't dominate xA."""
    return {m: float(rows[f"{m}_p90"].std() or 1.0) for m in METRICS}


def style_vector(per90: Dict[str, float], scales: Dict[str, float]) -> np.ndarray:
    v = np.array([float(per90.get(m, 0.0)) / scales[m] for m in METRICS])
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def target_vector(role: RoleProfile) -> np.ndarray:
    v = np.array([role.weights.get(m, 0.0) for m in METRICS])
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.ones(len(METRICS)) / np.sqrt(len(METRICS))


def uses_defensive(role: RoleProfile) -> bool:
    return any(m in DEF_KEYS for m in role.weights)


def def_column(m: str) -> Tuple[str, str]:
    """(column on candidate rows, column on the reference) for a defensive metric."""
    ref = m if m in DEF_RATIOS else f"{m}_p90"
    return f"def_{ref}", f"def_{ref}"


# ── scoring ────────────────────────────────────────────────────

def role_scores(proj: pd.DataFrame, role: RoleProfile, reference: pd.DataFrame) -> np.ndarray:
    """Weighted sum of the role's metrics, each as a z-score against the reference population
    (the role's position groups this season). 0 = an average player in the role.

    Projected metrics compare `pred_{m}` with the reference's `{m}_p90`. Defensive metrics compare
    the candidate's recent observed `def_*` rate with the same column on reference players who have
    defensive data; a candidate without coverage (NaN) counts as average on them."""
    ref = reference[reference["group"].isin(role.groups)] if "group" in reference else reference
    total = np.zeros(len(proj))
    for m, w in role.weights.items():
        if m in DEF_KEYS:
            col, rcol = def_column(m)
            if col not in proj or rcol not in ref:
                continue
            r = ref[ref["def_minutes_used"] > 0][rcol] if "def_minutes_used" in ref else ref[rcol]
            r = r.dropna()
            if len(r) < 10:
                continue
            mu, sd = r.mean(), r.std() or 1.0
            z = (proj[col].to_numpy(float) - mu) / sd
            if "def_minutes_used" in proj:
                z = np.where(proj["def_minutes_used"].to_numpy(float) > 0, z, 0.0)
            total += w * np.nan_to_num(z)
            continue
        mu, sd = ref[f"{m}_p90"].mean(), ref[f"{m}_p90"].std() or 1.0
        total += w * (proj[f"pred_{m}"].to_numpy(float) - mu) / sd
    return total


def value_gap(score: np.ndarray, value_eur: np.ndarray) -> np.ndarray:
    """How under- or over-priced each player is for his projected role score.
    Fits log(value) ~ score across the candidates; returns predicted / actual - 1
    (0.5 means the model's output suggests he is worth 50% more than his price)."""
    ok = np.isfinite(value_eur) & (value_eur > 0)
    out = np.full(len(score), np.nan)
    if ok.sum() < 3:
        return out
    A = np.column_stack([np.ones(ok.sum()), score[ok]])
    coef, *_ = np.linalg.lstsq(A, np.log(value_eur[ok]), rcond=None)
    pred = coef[0] + coef[1] * score
    out[ok] = np.exp(pred[ok] - np.log(value_eur[ok])) - 1
    return out
