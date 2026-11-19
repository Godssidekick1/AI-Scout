"""
Role profiles and style vectors
===============================
A brief ("a Stones replacement", "a poacher") becomes a role profile: weights
over the metrics the model projects. The same metrics give each player a style
vector, the *shape* of his output with level removed, used for "plays like"
retrieval in ChromaDB.

Limits of free data: defensive actions, progressive passes and carries, aerials
and keeper distribution are not in Understat. Roles that depend on them are
marked `proxy=True` and use xGBuildup/xGChain as the closest available signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecast.config import METRICS


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
    RoleProfile("inside_forward", "Inside forward", ("AM", "ST"),
                {"npxG": 0.4, "shots": 0.2, "xA": 0.2, "xGChain": 0.2}),
    RoleProfile("creative_winger", "Creative winger", ("AM",),
                {"xA": 0.35, "key_passes": 0.3, "xGChain": 0.25, "npxG": 0.1}),
    RoleProfile("playmaker", "Attacking playmaker (no.10)", ("AM", "CM"),
                {"xA": 0.35, "key_passes": 0.35, "xGChain": 0.2, "xGBuildup": 0.1}),
    RoleProfile("box_to_box", "Box-to-box midfielder", ("CM",),
                {"xGChain": 0.35, "xGBuildup": 0.3, "npxG": 0.15, "xA": 0.2}),
    RoleProfile("deep_playmaker", "Deep-lying playmaker", ("CM",),
                {"xGBuildup": 0.6, "xGChain": 0.25, "key_passes": 0.15}, proxy=True,
                note="Progressive passing is approximated by xGBuildup."),
    RoleProfile("attacking_fullback", "Attacking full-back", ("FB",),
                {"xA": 0.35, "key_passes": 0.25, "xGBuildup": 0.25, "xGChain": 0.15}),
    RoleProfile("ball_playing_cb", "Ball-playing centre-back", ("CB",),
                {"xGBuildup": 0.8, "xGChain": 0.2}, proxy=True,
                note="Progressive passes and carries need event data; xGBuildup is the proxy. "
                     "Defending and aerial ability are not measured."),
]}

# Free-text cues -> role, used by the fallback parser and to sanity-check the LLM.
ROLE_CUES: List[Tuple[str, Tuple[str, ...]]] = [
    ("ball_playing_cb", ("centre-back", "center-back", "centre back", "cb", "stones", "ball-playing")),
    ("attacking_fullback", ("full-back", "fullback", "wing-back", "right-back", "left-back")),
    ("deep_playmaker", ("deep-lying", "regista", "pivot", "number 6", "no.6", "holding")),
    ("box_to_box", ("box-to-box", "central midfielder", "cm", "number 8", "no.8")),
    ("playmaker", ("playmaker", "number 10", "no.10", "attacking midfielder")),
    ("creative_winger", ("creative winger", "winger", "wide")),
    ("inside_forward", ("inside forward", "inverted winger", "inverted")),
    ("poacher", ("poacher", "fox in the box", "goalscorer")),
    ("complete_forward", ("striker", "centre-forward", "center-forward", "number 9", "no.9", "forward")),
]


def _has(text: str, cue: str) -> bool:
    if len(cue) <= 3:                       # short codes like "cb" must be whole words
        return re.search(rf"\b{re.escape(cue)}\b", text) is not None
    return cue in text


def role_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    # most specific cues first: inverted/inside before winger
    for key in ("inside_forward", "poacher", "deep_playmaker", "ball_playing_cb", "attacking_fullback",
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
    return v / np.linalg.norm(v)


# ── scoring ────────────────────────────────────────────────────

def role_scores(proj: pd.DataFrame, role: RoleProfile, reference: pd.DataFrame) -> np.ndarray:
    """Weighted sum of projected metrics, each as a z-score against the reference population
    (the role's position groups this season). 0 = an average player in the role."""
    ref = reference[reference["group"].isin(role.groups)] if "group" in reference else reference
    total = np.zeros(len(proj))
    for m, w in role.weights.items():
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
