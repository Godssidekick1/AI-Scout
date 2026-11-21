"""
Scouting briefs without an LLM
==============================
Turns what a scout would type, a player to replace, a description of the
player wanted, or both, into a search over the forecast model:

    "replace Rodri: younger, progressive passer, under 30m, outside the Premier League"

The named player sets the style to match (the shape of his output) and the club
to project candidates at. The description adds a role, metric emphasis
("progressive", "creative", "goalscorer"), filters (age, value, leagues,
minutes) and the sort ("hidden", "bargain" ranks by value gap). Every parse is
printed back, so you can see what was understood and correct it with flags.

Runs offline from the saved model. The agent's LLM parser does the same job
for full natural language when GOOGLE_API_KEY is set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecast.config import LEAGUES, METRICS
from forecast.roles import ROLES, RoleProfile, metric_scales, role_from_text, style_vector

LEAGUE_WORDS: Dict[str, Tuple[str, ...]] = {
    "premier_league": ("premier league", "epl", "england", "english"),
    "la_liga": ("la liga", "laliga", "spain", "spanish"),
    "bundesliga": ("bundesliga", "germany", "german"),
    "serie_a": ("serie a", "italy", "italian"),
    "ligue_1": ("ligue 1", "ligue1", "france", "french"),
}

# description words -> metric multipliers
EMPHASIS: List[Tuple[str, Tuple[str, ...], Dict[str, float]]] = [
    ("build-up / progression", ("progressive", "progression", "passer", "passing", "build-up", "buildup",
                                "build up", "ball-playing", "ball playing", "distribut", "deep-lying", "carries",
                                "carrier", "on the ball"),
     {"xGBuildup": 2.0, "xGChain": 1.5}),
    ("chance creation", ("creative", "creator", "creat", "assist", "playmak", "vision", "key pass", "chances for"),
     {"xA": 2.0, "key_passes": 2.0}),
    ("goal threat", ("goals", "goal threat", "goalscor", "scorer", "finisher", "clinical", "box presence", "poach"),
     {"npxG": 2.0, "shots": 1.5}),
    ("shooting volume", ("shooter", "shots", "shoots"), {"shots": 2.0}),
    ("involvement", ("involved", "link-up", "link up", "combination", "combines"), {"xGChain": 2.0}),
]
UNMEASURED = ("track back", "tracks back", "tracking back", "work rate", "hard-working", "hard working", "adaptab",
              "versatil", "any formation", "formations", "any system", "multiple positions", "two-footed", "leader",
              "mentality", "attitude", "tackl", "intercept", "aerial", "header", "defensive", "defending", "duel",
              "pace", "fast",
              "quick", "physical", "tall", "in the air", "pressing", "work rate", "saves", "shot-stopp", "keeper")
HIDDEN = ("hidden", "unknown", "under the radar", "under-the-radar", "bargain", "undervalued", "underrated",
          "cheap", "value for money", "gem", "unexpected")
CLUB_ALIASES = {"man city": "Manchester City", "man utd": "Manchester United", "man united": "Manchester United",
                "spurs": "Tottenham", "psg": "Paris Saint Germain", "barca": "Barcelona", "bayern": "Bayern Munich",
                "dortmund": "Borussia Dortmund", "bvb": "Borussia Dortmund", "leverkusen": "Bayer Leverkusen",
                "atletico": "Atletico Madrid", "juve": "Juventus", "inter": "Inter", "milan": "AC Milan",
                "villa": "Aston Villa", "newcastle": "Newcastle United", "wolves": "Wolverhampton Wanderers",
                "forest": "Nottingham Forest", "leipzig": "RasenBallsport Leipzig", "rb leipzig": "RasenBallsport Leipzig"}
MONEY = re.compile(r"(?:[€£$]\s*(\d+(?:\.\d+)?)\s*(m|mn|million|k|bn)?\b)|(?:\b(\d+(?:\.\d+)?)\s*(m|mn|million|k)\b)")
NAME_PATTERNS = (
    r"replac(?:e|ing|ement for)\s+(?P<n>[^,.;:]+?)(?=\s+(?:with|who|that|but|for|at|from|in|under|younger|older|,)|[,.;:]|$)",
    r"(?:similar to|plays like|like|same as)\s+(?P<n>[^,.;:]+?)(?=\s+(?:with|who|that|but|for|at|from|in|under|younger|older|,)|[,.;:]|$)",
    r"(?:the next|a new|another)\s+(?P<n>[^,.;:]+?)(?=\s*[,.;:]|\s+(?:with|who|for|at|under)|$)",
    r"(?P<n>[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)*)\s+(?:replacement|type|clone|successor)\b",
)


@dataclass
class Brief:
    text: str = ""
    replace: Optional[pd.Series] = None       # the player row being replaced
    replace_query: Optional[str] = None       # what was typed, if no match was found
    role: Optional[str] = None
    emphasis: Dict[str, float] = field(default_factory=dict)
    emphasis_labels: List[str] = field(default_factory=list)
    to_team: Optional[str] = None
    max_age: Optional[float] = None
    min_age: Optional[float] = None
    max_value_eur: Optional[float] = None
    leagues: Optional[List[str]] = None
    min_minutes: int = 900
    min_goals: Optional[float] = None         # projected non-penalty xG over expected minutes
    min_assists: Optional[float] = None       # projected xA over expected minutes
    sort: str = "fit"
    note: str = ""                          # "fit" (projected output in the profile) or "value" (value gap)
    unmeasured: List[str] = field(default_factory=list)

    def describe(self) -> List[str]:
        out = []
        if self.replace is not None:
            r = self.replace
            age = f", {r['age']:.0f}" if pd.notna(r.get("age")) else ""
            out.append(f"replacing {r['name']} ({r['team']}, {r['group']}{age}): matching his style")
        elif self.replace_query:
            out.append(f"no player matched '{self.replace_query}' (ignored; try the full name)")
        if self.role:
            out.append(f"type: {ROLES[self.role].label}")
        if self.emphasis_labels:
            out.append("emphasis: " + ", ".join(self.emphasis_labels))
        if self.to_team:
            out.append(f"projected at {self.to_team}")
        if self.min_age is not None or self.max_age is not None:
            lo = f"{self.min_age:.0f}" if self.min_age is not None else ""
            hi = f"{self.max_age:.0f}" if self.max_age is not None else ""
            out.append(f"age {lo}-{hi}" if lo and hi else (f"age <= {hi}" if hi else f"age >= {lo}"))
        if self.max_value_eur is not None:
            out.append(f"value <= EUR {self.max_value_eur / 1e6:g}M")
        if self.leagues:
            out.append("leagues: " + ", ".join(self.leagues))
        out.append(f"min {self.min_minutes} minutes last season")
        if self.min_goals is not None:
            out.append(f"projected {self.min_goals:g}+ goals next season (npxG over his expected minutes)")
        if self.min_assists is not None:
            out.append(f"projected {self.min_assists:g}+ assists next season (xA over his expected minutes)")
        out.append("sorted by value gap (hidden talent)" if self.sort == "value" else "sorted by projected output")
        return out


def _has(t: str, cue: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(cue)}", t) is not None


def _club_names(service) -> List[str]:
    from forecast.table import team_totals
    return sorted(service.in_latest_season(team_totals(service.ps).reset_index())["team"].unique(), key=len, reverse=True)


def find_club(text: str, service, require_preposition: bool = True) -> Optional[str]:
    """A club named in the text, e.g. 'for Arsenal', 'at Man City'."""
    t = text.lower()
    clubs = _club_names(service)
    lower = {c.lower(): c for c in clubs}
    names = list(lower) + [a for a in CLUB_ALIASES if CLUB_ALIASES[a].lower() in lower]
    pre = r"\b(?:for|at|to|join|joining|into)\s+(?:the\s+)?" if require_preposition else r"\b"
    for name in sorted(names, key=len, reverse=True):
        if re.search(pre + re.escape(name) + r"\b", t):
            return lower.get(name) or CLUB_ALIASES[name]
    return None


def parse(text: str, service, replace: Optional[str] = None) -> Brief:
    """Parse a brief. `replace` is the player to replace, if given separately."""
    b = Brief(text=text or "")
    raw = (text or "").strip()
    t = raw.lower()

    # who to replace: explicit, else a pattern in the text
    name = replace
    if not name:
        for pat in NAME_PATTERNS:
            m = re.search(pat, raw, flags=re.IGNORECASE if "[A-Z]" not in pat else 0)
            if m and service.find(m.group("n").strip()) is not None:
                name = m.group("n").strip()
                break
    if name:
        hit = service.find(name)
        if hit is None:
            b.replace_query = name
        else:
            b.replace = hit
            t = t.replace(name.lower(), " ")

    # club to project at
    b.to_team = find_club(t, service)
    if b.to_team:
        t = t.replace(b.to_team.lower(), " ")
    elif b.replace is not None:
        b.to_team = b.replace["team"]

    # money first, so "under 30m" is not read as an age
    for m in MONEY.finditer(t):
        amount, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        mult = {"k": 1e3, "bn": 1e9}.get((unit or "m").lower(), 1e6)
        b.max_value_eur = float(amount) * mult
    t_age = MONEY.sub(" ", t)

    # age
    if (m := re.search(r"\b(?:aged?\s*|between\s*|from\s*)?(\d{2})\s*(?:-|\u2013|to|and)\s*(\d{2})\b(?!\s*\+?\s*(?:goals|assists|min))",
                       t_age)) and 15 <= int(m.group(1)) < int(m.group(2)) <= 40:
        b.min_age, b.max_age = float(m.group(1)), float(m.group(2))
    elif m := re.search(r"\b(?:under|below|younger than|less than|max(?:imum)? age|aged? under|u-?|<)\s*(\d{2})s?\b", t_age):
        b.max_age = float(m.group(1)) - (1 if not re.search(r"\bu-?\d{2}", m.group(0)) else 0)
    elif m := re.search(r"\b(\d{2})\s*(?:or|and)\s*(?:younger|under)\b", t_age):
        b.max_age = float(m.group(1))
    elif re.search(r"\byounger\b", t_age):
        b.max_age = float(b.replace["age"]) - 1 if b.replace is not None and pd.notna(b.replace.get("age")) else 24
    elif re.search(r"\b(young|youngster|prospect|wonderkid|teenager)\b", t_age):
        b.max_age = 23
    if m := re.search(r"\b(?:over|older than|at least)\s*(\d{2})\b", t_age):
        b.min_age = float(m.group(1))
    elif re.search(r"\b(experienced|veteran|proven)\b", t_age):
        b.min_age = 27

    # leagues
    excluded = [lg for lg, words in LEAGUE_WORDS.items()
                if any(re.search(rf"\b(?:outside|not|except|excluding|no)\s+(?:(?:the|in|from|of)\s+)*{re.escape(w)}\b", t)
                       for w in words)]
    included = [lg for lg, words in LEAGUE_WORDS.items() if lg not in excluded
                and any(re.search(rf"\b(?:from|in|playing in|based in)\s+(?:the\s+)?{re.escape(w)}\b", t) for w in words)]
    if included:
        b.leagues = included
    elif excluded:
        b.leagues = [lg for lg in LEAGUES if lg not in excluded]

    # minutes
    if re.search(r"\b(regular|starter|nailed|proven|ever-present|reliable|reliability|consistent|durable|available)", t):
        b.min_minutes = 1800
    elif re.search(r"\b(rotation|squad player|backup|bench)\b", t):
        b.min_minutes = 450

    # output targets: "15+ goals", "10 assists a season" -> projected npxG / xA over his expected minutes
    if m := re.search(r"\b(\d{1,2})\s*\+?\s*(?:or more\s+)?(?:league\s+)?goals\b", t):
        b.min_goals = float(m.group(1))
    if m := re.search(r"\b(\d{1,2})\s*\+?\s*(?:or more\s+)?(?:league\s+)?assists\b", t):
        b.min_assists = float(m.group(1))

    # role, emphasis, unmeasured asks, sort
    b.role = role_from_text(re.sub(r"forward (?:pass|run|carr)\w*", " ", t))
    for label, cues, mult in EMPHASIS:
        if any(_has(t, c) for c in cues):
            b.emphasis_labels.append(label)
            for k, v in mult.items():
                b.emphasis[k] = max(b.emphasis.get(k, 1.0), v)
    b.unmeasured = sorted({c for c in UNMEASURED if _has(t, c)})
    if any(_has(t, c) for c in HIDDEN):
        b.sort = "value"
    return b


def profile(b: Brief, service) -> Tuple[RoleProfile, Optional[np.ndarray]]:
    """The metric weights to rank by and the style vector to retrieve with.

    Replacing a player: his own style vector, reweighted by any emphasis, is both the
    retrieval target and (blended with a named role) the ranking weights, so 'output
    like his' means more of what he does most. Otherwise the role's weights, reweighted."""
    scales = metric_scales(service.in_latest_season(service.ps))
    mult = np.array([b.emphasis.get(m, 1.0) for m in METRICS])
    role = ROLES[b.role] if b.role else None

    if b.replace is not None:
        own = style_vector({m: float(b.replace[f"{m}_p90"]) for m in METRICS}, scales)
        weights = np.clip(own, 0, None)
        if role is not None:
            rw = np.array([role.weights.get(m, 0.0) for m in METRICS])
            weights = 0.5 * weights / max(weights.sum(), 1e-9) + 0.5 * rw / rw.sum()
        weights = weights * mult
        target = own * mult
        groups = tuple(dict.fromkeys((b.replace["group"],) + (role.groups if role is not None else ())))
        label = f"{b.replace['name']}-like" + (f" {role.label.lower()}" if role else "")
    else:
        if role is None:
            raise ValueError("Name a player to replace, or describe the type of player (e.g. 'deep-lying playmaker').")
        weights = np.array([role.weights.get(m, 0.0) for m in METRICS]) * mult
        target = None if not b.emphasis else weights
        groups, label = role.groups, role.label
    weights = weights / max(weights.sum(), 1e-9)
    if target is not None:
        target = target / max(np.linalg.norm(target), 1e-9)
    prof = RoleProfile("custom", label, tuple(groups),
                       {m: round(float(w), 3) for m, w in zip(METRICS, weights) if w > 0.02},
                       proxy=bool(role and role.proxy), note=role.note if role else "")
    return prof, target


def search(b: Brief, service, n: int = 10, exclude: Tuple[str, ...] = ()) -> Tuple[pd.DataFrame, Optional[pd.Series], RoleProfile]:
    """Ranked shortlist for a brief, plus the replaced player's own projection (as a yardstick)."""
    from forecast.roles import role_scores
    prof, target = profile(b, service)
    thresholds = b.min_goals is not None or b.min_assists is not None
    pool_n = max(n * 6, 120) if thresholds else (max(n, 25) if b.sort == "value" else n)
    df = service.candidates(prof, n=pool_n, to_team=b.to_team, max_age=b.max_age,
                            min_age=b.min_age, max_value_eur=b.max_value_eur, min_minutes=b.min_minutes,
                            leagues=b.leagues, exclude=exclude,
                            exclude_ids=[b.replace["us_id"]] if b.replace is not None else (), target=target)
    him = None
    if b.replace is not None:
        him = service.project(b.replace.to_frame().T.infer_objects(), b.to_team).iloc[0].copy()
        him["role_score"] = round(float(role_scores(him.to_frame().T.infer_objects(), prof,
                                                    service.in_latest_season(service.ps))[0]), 2)
        him["proj_goals"] = round(float(him["pred_npxG"] * him["expected_minutes"] / 90), 1)
        him["proj_assists"] = round(float(him["pred_xA"] * him["expected_minutes"] / 90), 1)
    if not df.empty and thresholds:
        season = df["expected_minutes"] / 90
        keep = np.ones(len(df), bool)
        if b.min_goals is not None:
            keep &= (df["pred_npxG"] * season >= b.min_goals).to_numpy()
        if b.min_assists is not None:
            keep &= (df["pred_xA"] * season >= b.min_assists).to_numpy()
        df = df.assign(proj_goals=(df["pred_npxG"] * season).round(1), proj_assists=(df["pred_xA"] * season).round(1))
        if keep.any():
            df = df[keep].reset_index(drop=True)
        else:   # nobody reaches the target: show the closest instead of an empty list
            b.note = "Nobody is projected to reach that target; showing the closest."
            df = df.sort_values("proj_goals" if b.min_goals is not None else "proj_assists",
                                ascending=False).reset_index(drop=True)
    if not df.empty and b.sort == "value":
        df = df.sort_values("value_gap", ascending=False, na_position="last")
    return df.head(n).reset_index(drop=True), him, prof
