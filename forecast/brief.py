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

from forecast.config import LEAGUES, METRICS, POSITIONS, SIDES, groups_for
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
                                "build up", "ball-playing", "progress", "possession", "steps into", "ball playing", "distribut", "deep-lying", "carries",
                                "carrier", "on the ball"),
     {"xGBuildup": 2.0, "xGChain": 1.5}),
    ("chance creation", ("creative", "creator", "creat", "assist", "playmak", "vision", "key pass", "chances for"),
     {"xA": 2.0, "key_passes": 2.0}),
    ("goal threat", ("goals", "goal threat", "goalscor", "scorer", "scores", "into the box", "scoring", "finisher", "clinical", "box presence", "poach"),
     {"npxG": 2.0, "shots": 1.5}),
    ("shooting volume", ("shooter", "shots", "shoots"), {"shots": 2.0}),
    ("involvement", ("involved", "link-up", "link up", "combination", "combines"), {"xGChain": 2.0}),
]
UNMEASURED = ("track back", "tracks back", "tracking back", "work rate", "hard-working", "hard working", "adaptab",
              "versatil", "any formation", "formations", "any system", "multiple positions", "two-footed", "leader",
              "mentality", "attitude", "tackl", "intercept", "aerial", "header", "defensive", "defending", "duel",
              "pace", "fast",
              "quick", "physical", "tall", "in the air", "pressing", "work rate", "saves", "shot-stopp", "keeper", "goalkeeper")
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
    note: str = ""
    parser: str = "rules"
    interpretation: str = ""
    unmeasured: List[str] = field(default_factory=list)
    anchor: str = "replacing"                 # "replacing" him, or a style reference ("like")
    positions: List[str] = field(default_factory=list)    # keys of forecast.config.POSITIONS
    side: Optional[str] = None                # "left" / "right"
    side_strict: bool = False                 # only that side, vs "can play there too"
    beat: List[Tuple[pd.Series, str]] = field(default_factory=list)   # (player, metric) to out-produce per 90
    beat_query: List[str] = field(default_factory=list)  # names that matched nobody
    squad_role: Optional[str] = None          # starter / rotation / backup
    exclude_own_club: bool = True             # leave the target club's own players out
    bars: List[Tuple[str, str, float]] = field(default_factory=list)  # filled by search(): (name, metric, per 90)

    def describe(self) -> List[str]:
        out = []
        if self.replace is not None:
            r = self.replace
            age = f", {r['age']:.0f}" if pd.notna(r.get("age")) else ""
            what = "replacing" if self.anchor == "replacing" else "style like"
            out.append(f"{what} {r['name']} ({r['team']}, {r['group']}{age})")
        elif self.replace_query:
            out.append(f"no player matched '{self.replace_query}' (ignored; try the full name)")
        if self.positions:
            out.append("position: " + ", ".join(p.replace("_", " ") for p in self.positions))
        if self.side:
            out.append(f"{self.side} side " + ("only" if self.side_strict else "preferred (shown in pos column)"))
        if self.role:
            out.append(f"type: {ROLES[self.role].label}")
        bars = {(n, m): v for n, m, v in self.bars}
        for p, m in self.beat:
            v = bars.get((p["name"], m))
            out.append(f"must beat {p['name']}'s projected {m}/90" + (f" ({v:.2f})" if v is not None else ""))
        for q in self.beat_query:
            out.append(f"no player matched '{q}' to compare with")
        if self.squad_role:
            out.append(f"squad role: {self.squad_role}")
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
        if self.to_team and self.exclude_own_club:
            out.append(f"excluding {self.to_team}'s own players")
        out.append("sorted by value gap (cheap for the output)" if self.sort == "value" else "sorted by projected output")
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
        for i, pat in enumerate(NAME_PATTERNS):
            m = re.search(pat, raw, flags=re.IGNORECASE if "[A-Z]" not in pat else 0)
            if m and service.find(m.group("n").strip()) is not None:
                name = m.group("n").strip()
                b.anchor = "like" if i == 1 else "replacing"
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
    elif m := re.search(r"\b(?:under|below|younger than|less than|max(?:imum)? age(?: of)?|aged? under|u-?|<)\s*(\d{2})s?\b", t_age):
        strict = re.match(r"(under|below|younger than|less than|aged? under|<)", m.group(0))
        b.max_age = float(m.group(1)) - (1 if strict else 0)
    elif m := re.search(r"\b(\d{2})\s*(?:or|and)\s*(?:younger|under)\b", t_age):
        b.max_age = float(m.group(1))
    elif re.search(r"\byounger\b", t_age):
        b.max_age = float(b.replace["age"]) - 1 if b.replace is not None and pd.notna(b.replace.get("age")) else 24
    elif re.search(r"\b(young|youngster|prospect|wonderkid|teenager)\b", t_age):
        b.max_age = 23
    if (m := re.search(r"\b(?:over|older than|at least|min(?:imum)? age(?: of)?)\s*(\d{2})\b(?!\s*\+?\s*(?:goals|assists|min|m\b))",
                       t_age)) and 15 <= int(m.group(1)) <= 40:
        b.min_age = float(m.group(1))
    elif b.min_age is None and re.search(r"\b(experienced|veteran)\b", t_age):
        b.min_age = 27

    # leagues
    excluded = [lg for lg, words in LEAGUE_WORDS.items()
                if any(re.search(rf"\b(?:outside|not|except|excluding|no|but)\s+(?:(?:the|in|from|of)\s+)*{re.escape(w)}\b", t)
                       for w in words)]
    included = [lg for lg, words in LEAGUE_WORDS.items() if lg not in excluded
                and any(re.search(rf"\b(?:from|in|playing in|based in)\s+(?:the\s+)?{re.escape(w)}\b", t) for w in words)]
    if included:   # "from Serie A or the Bundesliga": every league named alongside counts
        named = [lg for lg, words in LEAGUE_WORDS.items() if lg not in excluded and any(_has(t, w) for w in words)]
        b.leagues = named
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

    # positions and side
    for key, cues in POSITION_CUES:
        if any(_has(t, c) for c in cues) and key not in b.positions:
            b.positions.append(key)
    if "striker" in b.positions and re.search(r"inside forward|wide forward", t):
        b.positions.remove("striker")
    for side in ("left", "right"):
        if re.search(rf"\b{side}(?:\s+wing(?:er)?|-sided|\s+side|\s+flank)|on the {side}|\b{side}[- ]?(?:back|mid)", t):
            b.side = b.side or side
            b.side_strict = not re.search(r"\b(too|also|as well|either|both)\b", t)

    # out-produce a named player ("outdo Saka for goal scoring")
    for m in re.finditer(r"(?:outdo|outscore|out-score|outperform|out-produce|better than|beat|more goals than|more "
                         r"assists than)\s+(?P<n>[a-z][\w'\-]*(?:\s+(?!(?:for|at|in|on|and|who|with)\b)[a-z][\w'\-]*)?)(?:\s+(?:for|at|in|on)\s+(?P<q>[a-z\- ]+))?",
                         raw.lower()):
        who = m.group("n").strip()
        hit = service.find(who)
        if hit is None:
            hit = service.find(who.split()[0])
        q = (m.group("q") or "") + " " + m.group(0)
        metric = ("xA" if re.search(r"assist|creat|chance", q) else "shots" if "shot" in q
                  else "xGBuildup" if re.search(r"pass|build", q) else "npxG")
        if hit is not None and all(hit["us_id"] != p["us_id"] for p, _ in b.beat):
            b.beat.append((hit, metric))

    # squad role
    if re.search(r"\brotation\b", t):
        b.squad_role = "rotation"
    elif re.search(r"\b(back-?up|understudy|deputy|squad player|cover)\b", t):
        b.squad_role = "backup"
    elif re.search(r"\b(starter|first[- ]choice|nailed)\b", t):
        b.squad_role = "starter"
    apply_squad_role(b, upside=bool(re.search(r"potential|upside|develop|grow into|room to grow", t)))
    return b


POSITION_CUES = [
    ("winger", ("winger", "wide player", "inside forward", "wide forward", "wide attacker", " rw", " lw")),
    ("striker", ("striker", "centre-forward", "center-forward", "centre forward", "number 9", "no.9", "no. 9")),
    ("attacking_mid", ("attacking midfielder", "number 10", "no.10", "no. 10", "playmaker")),
    ("defensive_mid", ("defensive midfielder", "holding midfielder", "number 6", "no.6", "anchor", "regista")),
    ("central_mid", ("central midfielder", "box-to-box", "box to box", "number 8", "no.8")),
    ("fullback", ("full-back", "fullback", "full back", "wing-back", "wingback", "right-back", "left-back")),
    ("centre_back", ("centre-back", "center-back", "centre back", "center back", "cb")),
]
POSITION_ROLE = {"winger": "creative_winger", "striker": "complete_forward", "attacking_mid": "playmaker",
                 "central_mid": "box_to_box", "defensive_mid": "deep_playmaker", "fullback": "attacking_fullback",
                 "centre_back": "ball_playing_cb"}
ROLE_POSITIONS = {"creative_winger": ["winger"], "inside_forward": ["winger"], "poacher": ["striker"],
                  "complete_forward": ["striker"], "attacking_fullback": ["fullback"], "ball_playing_cb": ["centre_back"]}


def apply_squad_role(b: Brief, upside: bool = False) -> None:
    """A starter needs a proven sample; a backup only a usable one, and should come cheap for his
    output, so backups are sorted by value gap unless a budget already caps the price."""
    if b.squad_role == "starter":
        b.min_minutes = max(b.min_minutes, 1800)
    elif b.squad_role in ("backup", "rotation"):
        b.min_minutes = min(b.min_minutes, 450)
        if b.max_value_eur is None:
            b.sort = "value"
    if upside and b.max_age is None:
        b.max_age = 24


EMPHASIS_KEYS = {"build_up": "build-up / progression", "chance_creation": "chance creation",
                 "goal_threat": "goal threat", "shooting": "shooting volume", "involvement": "involvement"}

LLM_PROMPT = """You turn a football scout's request into search filters for a model of the big five leagues.
The model measures only six per-90 stats: npxG, xA, shots, key passes, xGChain, xGBuildup. Anything else the
scout asks for (defending, aerials, pace, work rate, versatility, character...) goes in `unmeasured`.
Fill only what the request states or clearly implies; leave the rest empty. Never invent a budget, age or club."""


def understand(text: str, service, replace: Optional[str] = None) -> Brief:
    """Read a brief in any wording with Gemini when a key is set; the rule parser otherwise,
    or if the call fails. The result says which parser read it."""
    from config import settings
    if settings.has_gemini() and (text or "").strip():
        try:
            return from_llm(text, service, replace)
        except Exception as e:  # noqa: BLE001 - fall back to rules, say so
            b = parse(text, service, replace=replace)
            b.parser = f"rules (Gemini failed: {str(e)[:80]})"
            return b
    return parse(text, service, replace=replace)


def from_llm(text: str, service, replace: Optional[str] = None) -> Brief:
    import asyncio
    from langchain_core.messages import HumanMessage, SystemMessage
    from core import llm
    from models.llm_schemas import BriefSpec
    msg = text if not replace else f"Player to replace: {replace}\nRequest: {text}"
    spec = asyncio.run(llm.structured(BriefSpec, [SystemMessage(content=LLM_PROMPT), HumanMessage(content=msg)],
                                      name="forecast.brief", temperature=0.0))
    return from_spec(spec, service, text, replace)


def from_spec(spec, service, text: str = "", replace: Optional[str] = None) -> Brief:
    """Check an LLM reading against the data: the player and club must exist, roles and
    leagues come from fixed lists, and an age 'younger than him' is computed here."""
    b = Brief(text=text, parser="Gemini", interpretation=spec.interpretation)
    name = replace or spec.replace_player or spec.similar_to
    b.anchor = "replacing" if (replace or spec.replace_player) else "like"
    if name:
        hit = service.find(name)
        if hit is None:
            b.replace_query = name
        else:
            b.replace = hit
    if spec.target_club:
        row = service._team_row(spec.target_club)
        club = row["team"] if row is not None else find_club(spec.target_club, service, require_preposition=False)
        b.to_team = club
    if b.to_team is None and b.replace is not None:
        b.to_team = b.replace["team"]
    b.role = spec.role
    for key in spec.emphasis:
        label = EMPHASIS_KEYS[key]
        b.emphasis_labels.append(label)
        for lbl, _, mult in EMPHASIS:
            if lbl == label:
                for k, v in mult.items():
                    b.emphasis[k] = max(b.emphasis.get(k, 1.0), v)
    b.min_age = float(spec.min_age) if spec.min_age else None
    b.max_age = float(spec.max_age) if spec.max_age else None
    if spec.younger_than_replaced and b.max_age is None and b.replace is not None and pd.notna(b.replace.get("age")):
        b.max_age = float(b.replace["age"]) - 1
    b.max_value_eur = spec.max_value_eur_m * 1e6 if spec.max_value_eur_m else None
    if spec.include_leagues:
        b.leagues = [lg for lg in spec.include_leagues if lg not in spec.exclude_leagues]
    elif spec.exclude_leagues:
        b.leagues = [lg for lg in LEAGUES if lg not in spec.exclude_leagues]
    if spec.min_minutes:
        b.min_minutes = int(spec.min_minutes)
    b.min_goals, b.min_assists = spec.min_goals, spec.min_assists
    b.sort = "value" if spec.hidden_talent else "fit"
    b.unmeasured = list(dict.fromkeys(u.strip().lower() for u in spec.unmeasured if u.strip()))
    b.positions = [p for p in dict.fromkeys(spec.positions) if p in POSITIONS]
    b.side, b.side_strict = spec.side, bool(spec.side and spec.side_only)
    for beat in spec.must_beat:
        hit = service.find(beat.player)
        if hit is None:
            b.beat_query.append(beat.player)
        else:
            b.beat.append((hit, BEAT_METRIC[beat.quality]))
    b.squad_role = spec.squad_role
    apply_squad_role(b, upside=spec.upside)
    if spec.min_minutes:                       # an explicit minutes ask wins over the squad-role default
        b.min_minutes = int(spec.min_minutes)
    return b


BEAT_METRIC = {"goal_threat": "npxG", "chance_creation": "xA", "shooting": "shots", "build_up": "xGBuildup",
               "involvement": "xGChain"}


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
        if role is None and (b.positions or b.emphasis):
            # a position without a role: its default role, so the search always has a profile
            role = ROLES[POSITION_ROLE.get(b.positions[0] if b.positions else "", "complete_forward")]
            if b.positions and b.positions[0] == "winger" and b.emphasis.get("npxG", 1) > 1:
                role = ROLES["inside_forward"]
        if role is None:
            raise ValueError("Say what kind of player: a position (winger, centre-back...), a style, or a player "
                             "to replace or compare with. Goalkeepers are not covered by the free data.")
        weights = np.array([role.weights.get(m, 0.0) for m in METRICS]) * mult
        target = None if not b.emphasis else weights
        groups, label = role.groups, role.label
    positions = b.positions or ROLE_POSITIONS.get(b.role or "", [])
    if positions:
        groups = groups_for(positions)
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
    thresholds = b.min_goals is not None or b.min_assists is not None or bool(b.beat)
    pool_n = max(n * 6, 120) if thresholds else (max(n, 25) if b.sort == "value" else n)
    positions = b.positions or ROLE_POSITIONS.get(b.role or "", [])
    subs = {s for p in positions for s in POSITIONS[p]}
    if b.side and b.side_strict:
        subs = (subs & set(SIDES[b.side])) or set(SIDES[b.side])
    excl_ids = ([b.replace["us_id"]] if b.replace is not None else []) + [p["us_id"] for p, _ in b.beat]
    df = service.candidates(prof, n=pool_n, to_team=b.to_team, max_age=b.max_age,
                            min_age=b.min_age, max_value_eur=b.max_value_eur, min_minutes=b.min_minutes,
                            leagues=b.leagues, exclude=exclude, exclude_ids=excl_ids, target=target,
                            sub_positions=subs or None,
                            exclude_teams=[b.to_team] if b.to_team and b.exclude_own_club else ())
    bars = []                                  # (label, metric, per-90 bar) from players to out-produce
    for p, m in b.beat:
        ref = service.project(p.to_frame().T.infer_objects(), b.to_team if p["team"] != b.to_team else None).iloc[0]
        bars.append((p["name"], m, float(ref[f"pred_{m}"])))
    b.bars = bars
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
        for _, m, bar in bars:
            keep &= (df[f"pred_{m}"] >= bar).to_numpy()
        df = df.assign(proj_goals=(df["pred_npxG"] * season).round(1), proj_assists=(df["pred_xA"] * season).round(1))
        if keep.any():
            df = df[keep].reset_index(drop=True)
        else:   # nobody reaches the target: show the closest instead of an empty list
            b.note = "Nobody is projected to reach that target; showing the closest."
            key = (f"pred_{bars[0][1]}" if bars else "proj_goals" if b.min_goals is not None else "proj_assists")
            df = df.sort_values(key, ascending=False).reset_index(drop=True)
    if not df.empty and b.sort == "value":
        df = df.sort_values("value_gap", ascending=False, na_position="last")
    return df.head(n).reset_index(drop=True), him, prof
