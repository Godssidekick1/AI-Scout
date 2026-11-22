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
from forecast.roles import DEF_KEYS, ROLES, RoleProfile, group_centres, metric_scales, role_from_text, style_vector

LEAGUE_WORDS: Dict[str, Tuple[str, ...]] = {
    "premier_league": ("premier league", "epl", "england", "english"),
    "la_liga": ("la liga", "laliga", "spain", "spanish"),
    "bundesliga": ("bundesliga", "germany", "german"),
    "serie_a": ("serie a", "italy", "italian"),
    "ligue_1": ("ligue 1", "ligue1", "france", "french"),
}

# description words -> what that quality is made of (model metrics and observed defensive rates, DEF_KEYS).
# The qualities a brief stresses share EMPHASIS_SHARE of the ranking equally, taken from the role's weights,
# so an emphasis works even when the role puts no weight on it (a holding midfielder who scores).
EMPHASIS: List[Tuple[str, Tuple[str, ...], Dict[str, float]]] = [
    ("build-up / progression", ("progressive", "progression", "passer", "passing", "build-up", "buildup",
                                "build up", "ball-playing", "progress", "possession", "steps into", "ball playing",
                                "distribut", "deep-lying", "carries", "carrier", "on the ball", "calm", "tempo",
                                "dictat", "control the game", "controls the game", "keeps the ball", "keep the ball",
                                "retain"),
     {"xGBuildup": 0.6, "xGChain": 0.4, "prog_passes": 0.6, "prog_carries": 0.4}),
    ("chance creation", ("creative", "creator", "creat", "assist", "playmak", "vision", "key pass", "chances for"),
     {"xA": 0.5, "key_passes": 0.5}),
    ("goal threat", ("goals", "goal threat", "goalscor", "goal scor", "goal-scor", "scorer", "scores", "into the box",
                     "scoring", "finisher", "finishing", "clinical", "box presence", "poach"),
     {"npxG": 0.7, "shots": 0.3}),
    ("shooting volume", ("shooter", "shots", "shoots", "shooting", "long range", "long-range", "from distance"),
     {"shots": 0.7, "npxG": 0.3}),
    ("involvement", ("involved", "link-up", "link up", "combination", "combines"), {"xGChain": 1.0}),
    ("defending", ("defensive", "defending", "defend", "tackl", "intercept", "track back", "tracks back",
                   "tracking back", "hold his ground", "holds his ground", "solid at the back", "stopper", "blocks",
                   "clearances", "duel", "1v1 defend", "one-on-one defend"),
     {"tkl_int": 0.45, "tackle_win_pct": 0.2, "blocks": 0.15, "clearances": 0.2}),
    ("aerial ability", ("aerial", "in the air", "header", "heading", "headers", "dominant in the air"),
     {"aerials_won": 0.55, "aerial_win_pct": 0.45}),
    ("dribbling / take-ons", ("dribbl", "take on", "takes on", "take-on", "beat a man", "beat his man",
                              "beats his man", "beat players", "beats players", "1v1", "one-on-one", "one v one",
                              "ball carrier", "carry the ball", "carries the ball"),
     {"take_ons_won": 0.6, "take_on_pct": 0.2, "prog_carries": 0.2}),
    ("ball security / technique", ("technical", "technique", "press resistant", "press-resistant", "secure in "
                                   "possession", "rarely loses", "tidy", "pass accuracy", "pass completion",
                                   "pass success", "composed", "composure", "keeps it simple"),
     {"pass_pct": 0.7, "take_on_pct": 0.3}),
    ("ball winning / work rate", ("pressing", "presser", "presses", "ball-winn", "ball winn", "wins the ball",
                                  "wins it back", "recover", "work rate", "hard-working", "hard working", "workhorse",
                                  "tenacious", "combative", "engine"),
     {"recoveries": 0.5, "tkl_int": 0.35, "duels_won": 0.15}),
]
EMPHASIS_SHARE = 0.5
UNMEASURED = ("adaptab", "versatil", "any formation", "formations", "any system", "multiple positions", "two-footed",
              "leader", "mentality", "attitude", "pace", "pacey", "fast", "quick", "physical", "tall", "saves",
              "shot-stopp", "keeper", "goalkeeper")
HIDDEN = ("hidden", "unknown", "under the radar", "under-the-radar", "bargain", "undervalued", "underrated",
          "cheap", "value for money", "gem", "unexpected", "underexplored", "under-explored", "unexplored",
          "overlooked", "off the radar", "lesser known", "lesser-known", "not well known", "under the radar",
          "unheralded", "undiscovered", "not the obvious")
CLUB_ALIASES = {"man city": "Manchester City", "man utd": "Manchester United", "man united": "Manchester United",
                "spurs": "Tottenham", "psg": "Paris Saint Germain", "barca": "Barcelona", "bayern": "Bayern Munich",
                "dortmund": "Borussia Dortmund", "bvb": "Borussia Dortmund", "leverkusen": "Bayer Leverkusen",
                "atletico": "Atletico Madrid", "juve": "Juventus", "inter": "Inter", "milan": "AC Milan",
                "villa": "Aston Villa", "newcastle": "Newcastle United", "wolves": "Wolverhampton Wanderers",
                "forest": "Nottingham Forest", "leipzig": "RasenBallsport Leipzig", "rb leipzig": "RasenBallsport Leipzig"}
MONEY = re.compile(r"(?:[€£$]\s*(\d+(?:\.\d+)?)\s*(m|mn|million|k|bn)?\b)|(?:\b(\d+(?:\.\d+)?)\s*(m|mn|million|k)\b)")
NAME_PATTERNS = (
    r"replac(?:e|ing|ement for)\s+(?P<n>[^,.;:]+?)(?=\s+(?:(?:with|who|that|but|for|at|from|in|under|younger|older)\b|,)|[,.;:]|$)",
    r"(?:similar to|plays like|like|same as)\s+(?P<n>[^,.;:]+?)(?=\s+(?:(?:with|who|that|but|for|at|from|in|under|younger|older|to|and)\b|,)|[,.;:]|$)",
    r"(?:the next|a new|another)\s+(?P<n>[^,.;:]+?)(?=\s*[,.;:]|\s+(?:with|who|for|at|under)\b|$)",
    r"(?P<n>[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)*)\s+(?:type|clone|style|-style|lookalike)\b",
    r"(?P<n>[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)*)\s+(?:replacement|successor)\b",
    r"(?:back-?up|cover|understudy|deputy|successor|alternative|replacement|competition|heir)\s+(?:for|to|of)\s+"
    r"(?P<n>[^,.;:]+?)(?=\s+(?:(?:with|who|that|but|for|at|from|in|under|younger|older|a|an)\b|,)|[,.;:]|$)",
    r"\b(?:a|an)\s+(?:younger|cheaper|better|faster|new|budget|poor man's|discount)\s+(?:version of\s+)?"
    r"(?P<n>[^,.;:]+?)(?=\s+(?:(?:with|who|that|but|for|at|from|in|under)\b|,)|[,.;:]|$)",
)
NAME_PATTERNS = NAME_PATTERNS + (
    r"(?:style|flair|qualities|profile|ability|abilities|traits|game|skills?|vision|technique|instincts|mould|mold)"
    r"\s+(?:\w+\s+){0,3}?of\s+(?:a\s+(?:young|modern)\s+)?(?P<n>[^,.;:]+?)(?=\s+(?:(?:in|with|who|that|but|for|at|and|to)\b|,)|[,.;:]|$)",
    r"\b(?:a|the)\s+(?:modern|modern-day|new|next|young)\s+(?P<n>[^,.;:]+?)(?=\s+(?:(?:in|with|who|that|but|for|at|and|to)\b|,)|[,.;:]|$)",
)
LIKE_PATTERNS = {1, 3, 7, 8}  # NAME_PATTERNS that name a style reference rather than the player to replace
REPLACE_WORDS = r"replac|successor|cover|back-?up|understudy|heir|instead of"
HIDDEN_MAX_VALUE = 35e6       # 'hidden / underexplored' without a budget: leave out the famous, expensive names

# Famous players the data does not have (retired, or before 2014): what they were known for, so 'the flair
# of Kaka' still means something. (positions, qualities = EMPHASIS labels)
ARCHETYPES: Dict[str, Tuple[List[str], List[str]]] = {
    "kaka": (["attacking_mid"], ["goal threat", "dribbling / take-ons", "build-up / progression"]),
    "zidane": (["attacking_mid"], ["ball security / technique", "dribbling / take-ons", "chance creation"]),
    "ronaldinho": (["attacking_mid", "winger"], ["dribbling / take-ons", "chance creation", "goal threat"]),
    "riquelme": (["attacking_mid"], ["chance creation", "ball security / technique"]),
    "ozil": (["attacking_mid"], ["chance creation", "ball security / technique"]),
    "totti": (["attacking_mid", "striker"], ["goal threat", "chance creation"]),
    "del piero": (["striker", "attacking_mid"], ["goal threat", "chance creation"]),
    "pirlo": (["defensive_mid"], ["build-up / progression", "chance creation", "ball security / technique"]),
    "xavi": (["central_mid"], ["ball security / technique", "build-up / progression", "chance creation"]),
    "iniesta": (["central_mid", "attacking_mid"], ["dribbling / take-ons", "ball security / technique", "chance creation"]),
    "busquets": (["defensive_mid"], ["ball security / technique", "build-up / progression", "defending"]),
    "makelele": (["defensive_mid"], ["ball winning / work rate", "defending", "ball security / technique"]),
    "vieira": (["central_mid", "defensive_mid"], ["ball winning / work rate", "defending", "build-up / progression"]),
    "gerrard": (["central_mid"], ["goal threat", "shooting volume", "build-up / progression"]),
    "lampard": (["central_mid", "attacking_mid"], ["goal threat", "shooting volume"]),
    "scholes": (["central_mid"], ["build-up / progression", "ball security / technique", "shooting volume"]),
    "schweinsteiger": (["central_mid"], ["build-up / progression", "ball winning / work rate"]),
    "fabregas": (["central_mid", "attacking_mid"], ["chance creation", "build-up / progression"]),
    "henry": (["striker", "winger"], ["goal threat", "dribbling / take-ons", "chance creation"]),
    "drogba": (["striker"], ["goal threat", "aerial ability", "shooting volume"]),
    "ronaldo nazario": (["striker"], ["goal threat", "dribbling / take-ons", "shooting volume"]),
    "van persie": (["striker"], ["goal threat", "shooting volume", "chance creation"]),
    "robben": (["winger"], ["goal threat", "dribbling / take-ons", "shooting volume"]),
    "ribery": (["winger"], ["dribbling / take-ons", "chance creation"]),
    "beckham": (["winger"], ["chance creation"]),
    "maldini": (["centre_back", "fullback"], ["defending", "ball security / technique"]),
    "nesta": (["centre_back"], ["defending", "ball security / technique"]),
    "terry": (["centre_back"], ["defending", "aerial ability"]),
    "puyol": (["centre_back"], ["defending", "aerial ability"]),
    "cafu": (["fullback"], ["chance creation", "dribbling / take-ons"]),
    "roberto carlos": (["fullback"], ["chance creation", "shooting volume"]),
}
ARCHETYPE_ALIASES = {"ricardo kaka": "kaka", "zizou": "zidane", "r9": "ronaldo nazario", "il fenomeno": "ronaldo nazario",
                     "thierry henry": "henry", "andrea pirlo": "pirlo", "xavi hernandez": "xavi", "andres iniesta": "iniesta",
                     "steven gerrard": "gerrard", "frank lampard": "lampard", "paul scholes": "scholes",
                     "claude makelele": "makelele", "patrick vieira": "vieira", "didier drogba": "drogba",
                     "arjen robben": "robben", "franck ribery": "ribery", "paolo maldini": "maldini", "john terry": "terry",
                     "carles puyol": "puyol", "david beckham": "beckham", "mesut ozil": "ozil", "cesc fabregas": "fabregas",
                     "zinedine zidane": "zidane", "robin van persie": "van persie", "alessandro nesta": "nesta",
                     "bastian schweinsteiger": "schweinsteiger", "sergio busquets": "busquets"}


def archetype(name: str) -> Optional[str]:
    """The ARCHETYPES key for a famous player the data lacks ('kaka', 'Kaká', 'Ricardo Kaka')."""
    from forecast.service import _plain
    n = _plain(name).strip()
    n = ARCHETYPE_ALIASES.get(n, n)
    return n if n in ARCHETYPES else None


def apply_archetype(b: "Brief", key: str, display: str) -> None:
    positions, qualities = ARCHETYPES[key]
    b.archetype = display
    for q in qualities:
        add_emphasis(b, q)
    if not b.positions:
        b.positions = list(positions)


EXPERIENCE_MINUTES = 1800      # "EPL experience": 20 full matches' worth, so a few cameos don't count
ALSO_SHARE = 0.10              # "can also play X": at least this share of his recent starts there


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
    also_positions: List[str] = field(default_factory=list)  # must ALSO have played these ("can play holding mid too")
    experience_leagues: List[str] = field(default_factory=list)   # career minutes needed in one of these leagues
    experience_minutes: int = EXPERIENCE_MINUTES
    penalty_taker: bool = False               # took 2+ penalties over the last two seasons
    partner: Optional[pd.Series] = None       # player he would play alongside: rank on what complements him
    partner_query: Optional[str] = None
    min_pass_pct: Optional[float] = None      # pass completion, 0-1 (FBref, last two seasons)
    positions_generic: bool = False           # only 'midfielder' was said: a style reference may widen it
    style_ref: Optional[pd.Series] = None     # style reference when there is also a player to replace
    archetype: Optional[str] = None           # a famous player the data lacks, read as his known qualities

    def describe(self) -> List[str]:
        out = []
        if self.replace is not None:
            r = self.replace
            age = f", {r['age']:.0f}" if pd.notna(r.get("age")) else ""
            what = "replacing" if self.anchor == "replacing" else "style like (only players leaning his way)"
            out.append(f"{what} {r['name']} ({r['team']}, {r['group']}{age})")
        elif self.replace_query:
            out.append(f"no player matched '{self.replace_query}' (ignored; try the full name)")
        if self.style_ref is not None:
            out.append(f"style like {self.style_ref['name']} ({self.style_ref['team']}, {self.style_ref['group']}; "
                       "only players leaning his way)")
        if self.archetype:
            out.append(f"{self.archetype} is not in the data (retired or before 2014): read as his known qualities")
        if self.positions:
            out.append("position: " + ", ".join(p.replace("_", " ") for p in self.positions)
                       + " (half or more of his starts over the last two seasons)")
        if self.also_positions:
            out.append("must also play: " + ", ".join(p.replace("_", " ") for p in self.also_positions)
                       + f" ({ALSO_SHARE:.0%}+ of recent starts)")
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
        if self.experience_leagues:
            out.append(f"experience: {self.experience_minutes}+ career minutes in "
                       + " or ".join(LEAGUE_LABEL.get(lg, lg) for lg in self.experience_leagues))
        if self.penalty_taker:
            out.append("penalty taker: 2+ penalties taken over the last two seasons")
        if self.partner is not None:
            out.append(f"to play alongside {self.partner['name']}: extra weight on what he does less of")
        elif self.partner_query:
            out.append(f"no player matched '{self.partner_query}' to partner")
        if self.min_pass_pct is not None:
            out.append(f"pass completion >= {self.min_pass_pct:.0%} (last two seasons with passing data)")
        if self.min_goals is not None:
            out.append(f"projected {self.min_goals:g}+ goals next season (npxG over his expected minutes)")
        if self.min_assists is not None:
            out.append(f"projected {self.min_assists:g}+ assists next season (xA over his expected minutes)")
        if self.to_team and self.exclude_own_club:
            out.append(f"excluding {self.to_team}'s own players")
        if self.sort == "value":
            out.append("hidden talent: " + (f"value <= EUR {HIDDEN_MAX_VALUE / 1e6:g}M (no famous names), "
                                            if self.max_value_eur is None else "")
                       + "ranked on projected output and style")
        else:
            out.append("ranked on projected output" + (" and style" if self.replace is not None or self.style_ref is not None else ""))
        return out


LEAGUE_PROPER = {"premier_league": ("premier league", "epl"), "la_liga": ("la liga", "laliga"),
                 "bundesliga": ("bundesliga",), "serie_a": ("serie a",), "ligue_1": ("ligue 1", "ligue1")}
LEAGUE_LABEL = {"premier_league": "the Premier League", "la_liga": "La Liga", "bundesliga": "the Bundesliga",
                "serie_a": "Serie A", "ligue_1": "Ligue 1"}


def _has(t: str, cue: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(cue)}", t) is not None


STOP_NAMES = {"premier", "league", "liga", "serie", "ligue", "bundesliga", "europe", "european", "english", "spanish",
              "german", "italian", "french", "i", "we", "my", "our", "the", "a", "an", "he", "someone", "find",
              "need", "want", "looking", "hidden", "young", "cheap", "who", "under", "max", "budget"}


def _plain_name(name: str) -> str:
    from forecast.service import _plain
    return _plain(name)


def _player_in(text: str, service) -> Optional[str]:
    """The player named at the end of a phrase: 'the underperforming cole palmer' -> 'cole palmer'.
    Leading words are dropped one at a time; every remaining word must be in the player's name."""
    words = text.split()
    for k in range(min(len(words), 4)):
        sub = " ".join(words[k:])
        if sub and _named(sub, service, first_word=False) is not None:
            return sub
    return None


def _named(who: str, service, first_word: bool = True) -> Optional[pd.Series]:
    """A player for a typed name, only if every typed word is in his name ('rice' -> Declan Rice;
    'rice at' never matches someone else by fuzz). Tries the full text, then its first word."""
    from forecast.service import _plain
    for q in dict.fromkeys((who, who.split()[0] if who.split() and first_word else who)):
        h = service.find(q)
        if h is not None and set(_plain(q).split()) <= set(_plain(h["name"]).split()):
            return h
    return None


def capitalised_player(raw: str, service) -> Optional[str]:
    """A player named with capitals anywhere ('Bayern need a backup for Harry Kane'): every word of a
    capitalised run must appear in the matched player's name, so 'Young' never matches Ashley Young
    unless typed as a name. Club names and league words are skipped."""
    from forecast.service import _plain
    clubs = {c.lower() for c in _club_names(service)} | set(CLUB_ALIASES)
    for m in re.finditer(r"[A-ZÀ-Ý][\w'\-À-ÿ]+(?:\s+[A-ZÀ-Ý][\w'\-À-ÿ]+)*", raw):
        if re.search(r"(?:alongside|next to|partner(?:ing| for| to)?|pair(?:ed|ing)? with|complement(?:s|ing)?|"
                     r"play with|than|outscore\w*|outdo\w*)\s+$", raw[:m.start()].lower()):
            continue           # a partner or a player to beat, not the style reference
        run = re.sub(r"(?:'s|’s|-(?:style|type|like|esque|lite))\b", "", m.group(0))
        words = [w for w in run.split() if w.lower() not in STOP_NAMES]
        for k in range(len(words), 0, -1):            # longest sub-run first
            for i in range(len(words) - k + 1):
                cand = " ".join(words[i:i + k])
                if cand.lower() in clubs or any(cand.lower() in c.split() for c in clubs if len(cand) > 3 and " " not in cand and cand.lower() == c):
                    continue
                hit = service.find(cand)
                if hit is not None and set(_plain(cand).split()) <= set(_plain(hit["name"]).split()):
                    return cand
    return None


def _league_alt(lg: str) -> str:
    return "|".join(re.escape(w) for w in LEAGUE_WORDS[lg])


def experience_asks(t: str, home_league: Optional[str] = None) -> Tuple[List[str], str]:
    """Leagues the player must have experience in ("EPL experience", "proven in the Premier League",
    "knows the league" = the target club's league), and the text with those phrases removed so they
    are not also read as a league filter or a minutes ask."""
    found = []
    for lg in LEAGUE_WORDS:
        w = _league_alt(lg)
        pats = (rf"\b(?:{w})\s+(?:experience|experienced|pedigree|proven|background|minutes|games|appearances)\b",
                rf"\b(?:experience|experienced|proven|played|track record|pedigree|background|minutes|games|appearances)"
                rf"\s+(?:(?:of|in|at|from)\s+)?(?:the\s+)?(?:{w})\b",
                rf"\bknows?\s+(?:the\s+)?(?:{w})\b",
                rf"\b(?:used|adapted|accustomed)\s+to\s+(?:the\s+)?(?:{w})\b")
        for pat in pats:
            if re.search(pat, t):
                found.append(lg)
                t = re.sub(pat, " ", t)
    if home_league and re.search(r"\b(?:knows?|used to|adapted to|accustomed to|experience (?:in|of))\s+the\s+league\b"
                                 r"|\bleague experience\b", t):
        found.append(home_league)
        t = re.sub(r"\b(?:knows?|used to|adapted to|accustomed to|experience (?:in|of))\s+the\s+league\b"
                   r"|\bleague experience\b", " ", t)
    return list(dict.fromkeys(found)), t


PENALTY = re.compile(r"\b(?:penalt(?:y|ies)(?!\s+(?:box|area))|spot[- ]?kicks?|pens\b|from the spot)")


def _club_names(service) -> List[str]:
    from forecast.table import team_totals
    return sorted(service.in_latest_season(team_totals(service.ps).reset_index())["team"].unique(), key=len, reverse=True)


def find_club(text: str, service, require_preposition: bool = True, raw: Optional[str] = None) -> Optional[str]:
    """The club the player would join: 'for Arsenal', 'at Man City', 'Bayern need...', 'my club is
    Napoli', or (with `raw`, the text as typed) a capitalised club name anywhere that is not a
    source or a comparison ('from Lille', 'Brighton style')."""
    t = text.lower()
    clubs = _club_names(service)
    lower = {c.lower(): c for c in clubs}
    names = sorted(list(lower) + [a for a in CLUB_ALIASES if CLUB_ALIASES[a].lower() in lower], key=len, reverse=True)

    def club(name):
        return lower.get(name) or CLUB_ALIASES[name]
    if not require_preposition:
        return next((club(n) for n in names if re.search(rf"\b{re.escape(n)}\b", t)), None)
    for name in names:
        n = re.escape(name)
        if re.search(rf"\b(?:for|at|to|in|join|joining|into|sign for|signing for)\s+(?:the\s+)?{n}\b", t) or \
                re.search(rf"(?:^|[.;!?]\s*)(?:the\s+)?{n}\s+(?:need|needs|want|wants|are|is|must|should|require|"
                          rf"requires|looking|seek|seeks|lack|lacks|could use)\b", t) or \
                re.search(rf"\bmy\s+(?:team|club|side)\s+(?:is\s+)?{n}\b", t):
            return club(name)
    if raw:
        for name in names:
            disp = club(name)
            m = re.search(rf"\b{re.escape(disp if name in lower else name)}\b", raw, flags=0 if name in lower else re.I)
            if m is None and name not in lower:
                continue
            if m is None:
                continue
            before, after = raw[:m.start()].lower(), raw[m.end():].lower()
            if re.search(r"(?:from|than|like|ex-|former|ex|of)\s*$", before) or re.match(r"\s*(?:-?style|type|way|'s|’s)\b", after):
                continue
            return disp
    return None


def parse(text: str, service, replace: Optional[str] = None) -> Brief:
    """Parse a brief. `replace` is the player to replace, if given separately."""
    b = Brief(text=text or "")
    raw = (text or "").strip()
    t = raw.lower()

    # who to replace and who to play like: both can be named ("the flair of Kaka ... to replace Palmer")
    name, like_name, like_arch = replace, None, None
    for i, pat in enumerate(NAME_PATTERNS):
        for m in re.finditer(pat, raw, flags=re.IGNORECASE if "[A-Z]" not in pat else 0):
            cand = _player_in(m.group("n").strip(), service)     # 'the underperforming cole palmer' -> 'cole palmer'
            if i in LIKE_PATTERNS:
                if like_name or like_arch:
                    continue
                key = archetype(m.group("n").strip()) or archetype(m.group("n").strip().split()[0] if m.group("n").split() else "")
                if cand and (not name or _plain_name(cand) != _plain_name(name)):
                    like_name = cand
                elif key and not cand:
                    like_arch = (key, m.group("n").strip())
            elif name is None and cand:
                name = cand
    if name is None and like_name is not None:
        name, like_name, b.anchor = like_name, None, "like"
    elif name is not None and not replace:
        b.anchor = "replacing"
    if not name and not like_arch:
        name = capitalised_player(raw, service)
        if name:
            b.anchor = ("replacing" if re.search(REPLACE_WORDS, t)
                        and not re.search(re.escape(name.lower()) + r"[- ]?(?:style|type|like|esque)", t) else "like")
    if name:
        hit = service.find(name)
        if hit is None:
            b.replace_query = name
        else:
            b.replace = hit
            t = t.replace(name.lower(), " ")
    if like_name:
        b.style_ref = _named(like_name, service)
        t = t.replace(like_name.lower(), " ")
    if like_arch:
        apply_archetype(b, like_arch[0], like_arch[1].title())
        t = t.replace(like_arch[1].lower(), " ")

    # club to project at
    b.to_team = find_club(t, service, raw=raw.replace(name, " ") if name else raw)
    if b.to_team:
        t = t.replace(b.to_team.lower(), " ")
    elif b.replace is not None:
        b.to_team = b.replace["team"]

    # league experience, before league filters, minutes and age read the same words
    home = service._team_row(b.to_team) if b.to_team else None
    b.experience_leagues, t = experience_asks(t, home["league"] if home is not None else None)

    # pass completion: "at least 85% pass success", "pass accuracy over 88%"
    m = re.search(r"(\d{2}(?:\.\d)?)\s*%\s*\+?\s*(?:or more\s+|or above\s+)?(?:pass|passing|completion)", t) or \
        re.search(r"pass(?:ing)?\s*(?:success|completion|accuracy)?\s*(?:rate|ratio|%)?\s*(?:of\s+)?(?:at\s*least|over|above|"
                  r"min(?:imum)?|>=?)?\s*(\d{2}(?:\.\d)?)\s*%", t)
    if m and 50 <= float(m.group(1)) < 100:
        b.min_pass_pct = float(m.group(1)) / 100
        t = t.replace(m.group(0), " ")

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
    # a league's own name counts on its own ("serie a or la liga", "Bundesliga based"); a country or
    # adjective only after from/in ("a German striker" is a nationality, "from Germany" a league)
    included = [lg for lg, words in LEAGUE_WORDS.items() if lg not in excluded
                and (any(re.search(rf"\b(?:from|in|playing in|plays in|based in)\s+(?:the\s+)?{re.escape(w)}\b", t)
                         or re.search(rf"\b{re.escape(w)}[- ]based\b", t) for w in words)
                     or any(_has(t, w) and re.search(rf"\b{re.escape(w)}\b", t) for w in LEAGUE_PROPER[lg]))]
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

    # penalties are measurable (Understat's penalty xG), so they are a filter, not a ranking weight
    b.penalty_taker = bool(PENALTY.search(t))
    t_play = PENALTY.sub(" ", t)

    # role, emphasis, unmeasured asks, sort
    b.role = role_from_text(re.sub(r"forward (?:pass|run|carr)\w*", " ", t_play))
    for label, cues, _ in EMPHASIS:
        if any(_has(t_play, c) for c in cues):
            add_emphasis(b, label)
    b.unmeasured = sorted({c for c in UNMEASURED if _has(t, c)})
    if any(_has(t, c) for c in HIDDEN):
        b.sort = "value"

    # positions: what he is, what he must also play, and emergency cover (not measurable)
    b.positions, b.also_positions, cover, b.positions_generic = read_positions(t)
    if cover:
        b.unmeasured.append("emergency cover at " + ", ".join(p.replace("_", " ") for p in cover))
    if b.positions and (b.role is None or b.positions[0] not in ROLE_FITS.get(b.role, [])):
        # the role follows the main position: "a goalscoring midfielder who can play holding mid" is a
        # box-to-box player who also covers the 6, not a deep-lying playmaker
        b.role = (ROLES_FOR_SCORING.get(b.positions[0]) if b.emphasis.get("npxG") else None) \
            or POSITION_ROLE[b.positions[0]]
    for side in ("left", "right"):
        if re.search(rf"\b{side}(?:\s+wing(?:er)?|-sided|\s+side|\s+flank)|on the {side}|\b{side}[- ]?(?:back|mid)", t):
            b.side = b.side or side
            b.side_strict = not re.search(r"\b(too|also|as well|either|both)\b", t)

    # out-produce a named player ("outdo Saka for goal scoring")
    for m in re.finditer(r"(?:(?:outdo|outscore|out-score|outperform|out-produce|outshine|outcreate)(?:s|es)?|better than|"
                         r"beats?|more (?:[a-z]+\s+){0,2}than)\s+(?P<n>[a-z][\w'\-]*(?:\s+(?!(?:for|at|in|on|and|who|with)\b)[a-z][\w'\-]*)?)(?:\s+(?:for|at|in|on)\s+(?P<q>[a-z\- ]+))?",
                         raw.lower()):
        from forecast.service import _plain
        who = m.group("n").strip()
        hit = None
        for q_name in (who, who.split()[0]):        # 'haaland per 90' -> 'haaland'
            h = service.find(q_name)
            if h is not None and set(_plain(q_name).split()) <= set(_plain(h["name"]).split()):
                hit = h
                break
        q = (m.group("q") or "") + " " + m.group(0)
        metric = ("xA" if re.search(r"assist|creat|chance", q) else "shots" if "shot" in q
                  else "xGBuildup" if re.search(r"pass|build", q) else "npxG")
        if hit is not None and all(hit["us_id"] != p["us_id"] for p, _ in b.beat):
            b.beat.append((hit, metric))

    # a partner: "play alongside Rice", "partner for Saliba"; "him" = the player named as backup/replacement
    if m := re.search(r"(?:alongside|next to|partner(?:ing|ship with| for| to)?|pair(?:ed|ing)? with|complement(?:s|ing)?|"
                      r"play with|link up with)\s+(?P<n>[a-z][\w'\-]*(?:\s+(?!(?:at|in|for|and|or|who|with|,)\b)[a-z][\w'\-]*)?)",
                      raw.lower()):
        who = m.group("n").strip()
        if who.split()[0] in ("him", "them", "he"):
            ref = re.search(r"(?:back-?up|cover|understudy|deputy|replacement|successor)\s+(?:for|to|of)\s+"
                            r"(?P<n>[a-z][\w'\-]*(?:\s+(?!(?:at|in|for|and|or|who|with)\b)[a-z][\w'\-]*)?)", raw.lower())
            who = ref.group("n").strip() if ref else ""
        hit = _named(who, service) if who else None
        if hit is not None and (b.replace is None or hit["us_id"] != b.replace["us_id"]):
            b.partner = hit
        elif who and hit is None:
            b.partner_query = who

    # squad role
    if re.search(r"\brotation\b", t):
        b.squad_role = "rotation"
    elif re.search(r"\b(back-?up|understudy|deputy|squad player|cover)\b", t):
        b.squad_role = "backup"
    elif re.search(r"\b(starter|first[- ]choice|nailed)\b", t):
        b.squad_role = "starter"
    apply_squad_role(b, upside=bool(re.search(r"potential|upside|develop|grow into|room to grow", t)))
    positions_from_anchor(b, service)
    return b


POSITION_CUES = [   # (position, regex). Order matters only for overlapping words.
    ("winger", r"\bwingers?\b|\bwide (?:player|forward|attacker|man|midfielder)|inside forward|\b[rl]w\b|\b[rl]m\b"),
    ("striker", r"\bstrikers?\b|cent(?:re|er)[- ]forward|number 9\b|no\.? ?9\b|\bcf\b"),
    ("attacking_mid", r"attacking mid(?:field(?:er)?)?s?\b|number 10\b|no\.? ?10\b|\bcam\b|(?<!deep-lying )(?<!deep lying )\bplaymaker"
                      r"(?! winger)"),
    ("defensive_mid", r"(?:defensive|holding) mid(?:field(?:er)?)?s?\b|number 6\b|no\.? ?6\b|\bcdm\b|\bdm\b|\banchor(?:man)?\b"
                      r"|\bregista\b|deep[- ]lying|\bpivot\b|\bthe 6\b"),
    ("central_mid", r"central mid(?:field(?:er)?)?s?\b|box[- ]to[- ]box|number 8\b|no\.? ?8\b|\bcm\b|\bthe 8\b"),
    ("midfielder", r"(?<!defensive )(?<!holding )(?<!attacking )(?<!central )(?<!wide )\bmidfielders?\b"),  # CM or AM
    ("fullback", r"full[- ]?backs?\b|wing[- ]?backs?\b|(?:right|left)[- ]backs?\b|\b[rl]wb\b|\b[rl]b\b"),
    ("centre_back", r"cent(?:re|er)[- ]?backs?\b|\bcbs?\b|\bcentral defender|\bstopper\b"),
]
ALSO_CUE = re.compile(r"\b(?:can|could|able to|also|comfortable|happy to|willing to|capable of|option to)\b[^.;,]*"
                      r"\b(?:play|plays|playing|cover|covers|covering|drop|dropping|fill in|filling in)\b[^.;,]*$")
COVER_CUE = re.compile(r"\b(?:if forced|if needed|if required|at a push|in an emergency|in a pinch|when needed|fill in|"
                       r"fills in|filling in|deputi[sz]e|emergency)\b[^.;,]*$")
COVER_AFTER = re.compile(r"^[^.;,]{0,25}\b(?:if forced|if needed|if required|at a push|in an emergency|in a pinch|when "
                         r"needed|if necessary)\b")


def read_positions(t: str) -> Tuple[List[str], List[str], List[str], bool]:
    """(main positions, positions he must also play, emergency cover) in the order they are mentioned.
    'can also play as the holding mid' -> also; 'if forced wingback' -> cover (not checked: lineups
    cannot show what a player could do in an emergency). If every mention is an 'also', they are main.
    The last value is True when the only main position word was a generic one ('midfielder')."""
    hits = []
    for key, rx in POSITION_CUES:
        for m in re.finditer(rx, t):
            before = t[max(0, m.start() - 60):m.start()]
            after = t[m.end():m.end() + 40]
            kind = ("cover" if COVER_CUE.search(before) or COVER_AFTER.search(after)
                    else "also" if ALSO_CUE.search(before) else "main")
            hits.append((m.start(), key, kind))
    hits.sort()
    generic = bool(hits) and all(k == "midfielder" for _, k, kind in hits if kind == "main") and \
        any(kind == "main" for _, _, kind in hits)
    hits = [(i, k2, kind) for i, k, kind in hits
            for k2 in (("central_mid", "attacking_mid") if k == "midfielder" else (k,))]
    main = list(dict.fromkeys(k for _, k, kind in hits if kind == "main"))
    also = list(dict.fromkeys(k for _, k, kind in hits if kind == "also" and k not in main))
    cover = list(dict.fromkeys(k for _, k, kind in hits if kind == "cover" and k not in main and k not in also))
    if not main and also:
        main, also = also, []
    if "striker" in main and re.search(r"inside forward|wide forward", t):
        main.remove("striker")
    return main, also, cover, generic




POSITION_ROLE = {"winger": "creative_winger", "striker": "complete_forward", "attacking_mid": "playmaker",
                 "central_mid": "box_to_box", "defensive_mid": "deep_playmaker", "fullback": "attacking_fullback",
                 "centre_back": "ball_playing_cb"}
ROLE_POSITIONS = {"creative_winger": ["winger"], "inside_forward": ["winger"], "poacher": ["striker"],
                  "complete_forward": ["striker"], "attacking_fullback": ["fullback"], "ball_playing_cb": ["centre_back"],
                  "stopper_cb": ["centre_back"], "defensive_fullback": ["fullback"],
                  "ball_winner": ["defensive_mid", "central_mid"], "target_forward": ["striker"],
                  "pressing_forward": ["striker"]}
ROLE_FITS = {**ROLE_POSITIONS, "playmaker": ["attacking_mid"], "box_to_box": ["central_mid"],
             "deep_playmaker": ["defensive_mid"]}       # the main position each role is played from
ROLES_FOR_SCORING = {"winger": "inside_forward", "striker": "complete_forward"}   # a scoring ask changes the role


ZONES = {"centre_back": {"centre_back", "fullback", "defensive_mid"}, "fullback": {"fullback", "centre_back", "winger"},
         "defensive_mid": {"defensive_mid", "central_mid", "centre_back"},
         "central_mid": {"central_mid", "defensive_mid", "attacking_mid"},
         "attacking_mid": {"attacking_mid", "central_mid", "winger", "striker"},
         "winger": {"winger", "attacking_mid", "fullback"}, "striker": {"striker", "attacking_mid", "winger"}}


def positions_from_anchor(b: Brief, service=None) -> None:
    """No position stated but a player named ('a Stones replacement'): his position, from where he
    started most over the last two seasons (else his Transfermarkt main position)."""
    if b.replace is None:
        return
    sub = None
    if service is not None and getattr(service, "profiles", None) is not None:
        sub = service.recent_position(b.replace.to_frame().T).iloc[0]
    sub = sub or b.replace.get("tm_sub_position")
    key = next((k for k, subs in POSITIONS.items() if sub in subs), None)
    if not key:
        return
    if not b.positions and not (b.role and b.role in ROLE_POSITIONS):
        b.positions = [key]                   # both sides stay open: the brief did not ask for one
    elif (b.anchor == "like" and b.positions_generic and key not in b.positions
          and key in ZONES.get(b.positions[0], ())):
        # "a midfielder like Güler": his own position (a 10) counts too, as long as it neighbours the one asked for
        b.positions.append(key)


def apply_squad_role(b: Brief, upside: bool = False) -> None:
    """A starter needs a proven sample; a backup only a usable one (he may be a backup now). Price
    only changes the order when the brief asks for value ('cheap', 'bargain', a budget)."""
    if b.squad_role == "starter":
        b.min_minutes = max(b.min_minutes, 1800)
    elif b.squad_role in ("backup", "rotation"):
        b.min_minutes = min(b.min_minutes, 450)
    # 'potential / upside / bursting with talent' does not set an age cap: the user didn't give one


EMPHASIS_KEYS = {"build_up": "build-up / progression", "chance_creation": "chance creation",
                 "goal_threat": "goal threat", "shooting": "shooting volume", "involvement": "involvement",
                 "defending": "defending", "aerial": "aerial ability", "ball_winning": "ball winning / work rate",
                 "dribbling": "dribbling / take-ons", "ball_security": "ball security / technique"}


def add_emphasis(b: "Brief", label: str) -> None:
    """Record an emphasis: its metric shares are summed into b.emphasis (profile() uses the model-metric
    part as EMPHASIS_SHARE of the ranking, and adds the defensive part on top)."""
    if label in b.emphasis_labels:
        return
    b.emphasis_labels.append(label)
    for k, v in next(sh for lbl, _, sh in EMPHASIS if lbl == label).items():
        b.emphasis[k] = b.emphasis.get(k, 0.0) + v


LLM_PROMPT = """You turn a football scout's request into search filters for a model of the big five leagues.
It projects six per-90 stats (npxG, xA, shots, key passes, xGChain, xGBuildup) and also has observed defensive
data: tackles, interceptions, blocks, clearances, aerial duels, ball recoveries, duels, progressive passes and
carries, take-ons (dribbles) and pass completion %. Defending, tackling, tracking back or holding ground -> emphasis
defending; aerials, heading -> aerial; pressing, work rate, winning the ball back -> ball_winning; dribbling, taking
on his man -> dribbling; technical, press-resistant -> ball_security; a pass completion figure -> min_pass_completion_pct.
Only what none of these measure (pace, versatility, physicality, leadership, character, goalkeeping...) goes in
`unmeasured`. "Has played in La Liga" is experience_leagues, not include_leagues: he may play elsewhere now.
Fill only what the request states or clearly implies; leave the rest empty. Never invent a budget, age or club."""


def understand(text: str, service, replace: Optional[str] = None) -> Brief:
    """Read a brief in any wording with Gemini when a key is set; the rule parser otherwise,
    or if the call fails. The result says which parser read it."""
    global _GEMINI_BROKEN
    from config import settings
    if settings.has_gemini() and (text or "").strip() and not _GEMINI_BROKEN:
        try:
            return from_llm(text, service, replace)
        except Exception as e:  # noqa: BLE001 - fall back to rules, say so
            msg = str(e)
            if "cannot run" in msg or "daily quota" in msg or "every Gemini model" in msg:   # won't fix itself soon
                _GEMINI_BROKEN = msg
            b = parse(text, service, replace=replace)
            b.parser = f"rules (Gemini failed: {msg[:300]})"
            return b
    b = parse(text, service, replace=replace)
    if _GEMINI_BROKEN:
        b.parser = "rules (Gemini unavailable: see the first error)"
    return b


_GEMINI_BROKEN = ""


def from_llm(text: str, service, replace: Optional[str] = None) -> Brief:
    import asyncio
    from langchain_core.messages import HumanMessage, SystemMessage
    from core import llm
    from models.llm_schemas import BriefSpec
    from config import settings
    msg = text if not replace else f"Player to replace: {replace}\nRequest: {text}"
    models = [settings.GEMINI_MODEL] + [m.strip() for m in settings.GEMINI_BRIEF_FALLBACKS.split(",")
                                        if m.strip() and m.strip() != settings.GEMINI_MODEL]
    global _MODEL_START
    last: Optional[Exception] = None
    for i in range(_MODEL_START, len(models)):
        try:
            spec = asyncio.run(llm.structured(BriefSpec, [SystemMessage(content=LLM_PROMPT), HumanMessage(content=msg)],
                                              name="forecast.brief", model=models[i], temperature=0.0))
        except llm.LLMError as e:
            if "daily quota" not in str(e) and "not available" not in str(e):
                raise
            last, _MODEL_START = e, i + 1      # skip this model for the rest of the session
            continue
        b = from_spec(spec, service, text, replace)
        b.parser = f"Gemini ({models[i]})"
        return b
    raise llm.LLMError(f"every Gemini model's daily quota is used up or unavailable ({', '.join(models)}); "
                       f"last error: {str(last)[:200]}")


_MODEL_START = 0       # first model still worth trying this session


def from_spec(spec, service, text: str = "", replace: Optional[str] = None) -> Brief:
    """Check an LLM reading against the data: the player and club must exist, roles and
    leagues come from fixed lists, and an age 'younger than him' is computed here."""
    b = Brief(text=text, parser="Gemini", interpretation=spec.interpretation)
    rep = replace or spec.replace_player
    name = rep or spec.similar_to
    b.anchor = "replacing" if rep else "like"
    if name:
        hit = _named(name, service, first_word=False)
        hit = service.find(name) if hit is None else hit
        if hit is None and not rep and (archetype(name) or spec.similar_to_traits):
            pass                                  # a famous player the data lacks: his qualities, below
        elif hit is None:
            b.replace_query = name
        else:
            b.replace = hit
    if rep and spec.similar_to:                   # both: Palmer is replaced, Kaka sets the style
        ref = _named(spec.similar_to, service, first_word=False)
        if ref is not None and ref["us_id"] != (b.replace["us_id"] if b.replace is not None else None):
            b.style_ref = ref
    style_name = spec.similar_to if (spec.similar_to and b.style_ref is None
                                     and (rep or b.replace is None)) else None
    if style_name:
        key = archetype(style_name)
        if key:
            apply_archetype(b, key, style_name)
        elif spec.similar_to_traits:
            b.archetype = style_name
            for q in spec.similar_to_traits:
                add_emphasis(b, EMPHASIS_KEYS[q])
            if not spec.positions and spec.similar_to_position:
                b.positions = [spec.similar_to_position]
    if spec.target_club:
        row = service._team_row(spec.target_club)
        club = row["team"] if row is not None else find_club(spec.target_club, service, require_preposition=False)
        b.to_team = club
    if b.to_team is None and b.replace is not None:
        b.to_team = b.replace["team"]
    b.role = spec.role
    for key in spec.emphasis:
        add_emphasis(b, EMPHASIS_KEYS[key])
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
    b.also_positions = [p for p in dict.fromkeys(spec.also_positions) if p in POSITIONS and p not in b.positions]
    if not b.positions and b.also_positions:
        b.positions, b.also_positions = b.also_positions, []
    b.experience_leagues = [lg for lg in dict.fromkeys(spec.experience_leagues) if lg in LEAGUES]
    if spec.experience_min_minutes:
        b.experience_minutes = int(spec.experience_min_minutes)
    b.penalty_taker = bool(spec.penalty_taker)
    b.positions_generic = bool(spec.position_is_generic)
    if b.positions_generic and b.positions == ["central_mid"]:
        b.positions.append("attacking_mid")           # 'a midfielder': an 8 or a 10
    if spec.min_pass_completion_pct:
        b.min_pass_pct = float(spec.min_pass_completion_pct) / 100
    if spec.partner_player:
        hit = service.find(spec.partner_player)
        if hit is not None and (b.replace is None or hit["us_id"] != b.replace["us_id"]):
            b.partner = hit
        elif hit is None:
            b.partner_query = spec.partner_player
    b.side, b.side_strict = spec.side, bool(spec.side and spec.side_only)
    for beat in spec.must_beat:
        hit = service.find(beat.player)
        if hit is None:
            b.beat_query.append(beat.player)
        else:
            if beat.quality in BEAT_METRIC:
                b.beat.append((hit, BEAT_METRIC[beat.quality]))
            else:
                add_emphasis(b, EMPHASIS_KEYS[beat.quality])
    b.squad_role = spec.squad_role
    apply_squad_role(b, upside=spec.upside)
    if spec.min_minutes:                       # an explicit minutes ask wins over the squad-role default
        b.min_minutes = int(spec.min_minutes)
    positions_from_anchor(b, service)
    return b


BEAT_METRIC = {"goal_threat": "npxG", "chance_creation": "xA", "shooting": "shots", "build_up": "xGBuildup",
               "involvement": "xGChain"}    # defensive qualities aren't projected: they become an emphasis


def profile(b: Brief, service) -> Tuple[RoleProfile, Optional[np.ndarray]]:
    """The metric weights to rank by and the style vector to retrieve with.

    Base weights: replacing a player, what he does more than a typical player in his position
    (blended with a named role); otherwise the role's weights. Defensive metrics come from the role
    and from a replaced defender's own strengths. Then every quality the brief stresses gets an equal
    part of EMPHASIS_SHARE of the ranking, split over the metrics it is made of (model metrics and
    defensive ones alike), so 'goal scoring, great shooting, calm on the ball' ranks 2/3 on scoring."""
    scales = metric_scales(service.in_latest_season(service.ps))
    role = ROLES[b.role] if b.role else None
    dists = []                                  # one metric distribution per stressed quality
    for quality in b.emphasis_labels:
        sh = next((mm for lbl, _, mm in EMPHASIS if lbl == quality), {})
        tot = sum(sh.values())
        if tot > 0:
            dists.append({m: v / tot for m, v in sh.items()})
    if b.partner is not None:
        # what his partner does less than a typical player in that position: a Rice partner who adds
        # the creating or scoring Rice doesn't, rather than a second Rice
        ref_ps = service.in_latest_season(service.ps)
        pv = style_vector({m: float(b.partner[f"{m}_p90"]) for m in METRICS}, scales,
                          group_centres(ref_ps).get(b.partner["group"]))
        gap = np.clip(-pv, 0, None)
        if gap.sum() > 0:
            dists.append({m: float(v / gap.sum()) for m, v in zip(METRICS, gap) if v > 0})
    emph = np.array([sum(d.get(m, 0.0) for d in dists) for m in METRICS])
    emph = emph / np.linalg.norm(emph) if emph.sum() > 0 else None     # retrieval direction (model metrics)

    anchor = b.style_ref if b.style_ref is not None else b.replace
    if anchor is not None:
        ref = service.in_latest_season(service.ps)
        per90 = {m: float(anchor[f"{m}_p90"]) for m in METRICS}
        own = style_vector(per90, scales, group_centres(ref).get(anchor["group"]))
        # rank on what he does MORE than a typical player in his position; if he is below
        # average at everything, fall back to the plain shape of his output
        weights = np.clip(own, 0, None)
        if weights.sum() < 0.3:
            weights = np.clip(style_vector(per90, scales), 0, None)
        if role is not None:
            rw = np.array([role.weights.get(m, 0.0) for m in METRICS])
            weights = 0.5 * weights / max(weights.sum(), 1e-9) + 0.5 * rw / max(rw.sum(), 1e-9)
        target = own if emph is None else own + emph
        groups = tuple(dict.fromkeys((anchor["group"],) + (role.groups if role is not None else ())))
        label = f"{anchor['name']}-like" + (f" {role.label.lower()}" if role else "")
    else:
        if role is None and (b.positions or b.emphasis):
            # a position without a role: its default role, so the search always has a profile
            role = ROLES[POSITION_ROLE.get(b.positions[0] if b.positions else "", "complete_forward")]
            if b.positions and b.positions[0] == "winger" and b.emphasis.get("npxG"):
                role = ROLES["inside_forward"]
        if role is None:
            raise ValueError("Say what kind of player: a position (winger, centre-back...), a style, or a player "
                             "to replace or compare with. Goalkeepers are not covered by the free data.")
        weights = np.array([role.weights.get(m, 0.0) for m in METRICS])
        target = None if emph is None else emph
        groups, label = role.groups, role.label
    positions = b.positions or ROLE_POSITIONS.get(b.role or "", [])
    if positions:
        groups = groups_for(list(positions) + [p for p in b.also_positions if p not in positions])
    weights = weights / max(weights.sum(), 1e-9)
    if target is not None:
        target = target / max(np.linalg.norm(target), 1e-9)

    # Base: model metrics, the role's own defensive weights, and a replaced defender's strengths.
    role_def = {m: w for m, w in (role.weights.items() if role is not None else ()) if m in DEF_KEYS}
    share = sum(role_def.values()) * (0.5 if b.replace is not None else 1.0)
    base = {m: float(w) * (1 - share) for m, w in zip(METRICS, weights)}
    for m, w in role_def.items():
        base[m] = base.get(m, 0.0) + w * (0.5 if b.replace is not None else 1.0)
    if b.replace is not None:
        for m, w in own_defensive(b.replace, service).items():
            base[m] = base.get(m, 0.0) + w
    tot = sum(base.values()) or 1.0
    final = {m: w / tot for m, w in base.items()}
    if dists:
        final = {m: (1 - EMPHASIS_SHARE) * w for m, w in final.items()}
        for d in dists:
            for m, v in d.items():
                final[m] = final.get(m, 0.0) + EMPHASIS_SHARE * v / len(dists)
    total = sum(final.values()) or 1.0
    prof = RoleProfile("custom", label_of(label, b), tuple(groups),
                       {m: round(w / total, 3) for m, w in final.items() if w / total > 0.02},
                       proxy=bool(role and role.proxy), note=role.note if role else "")
    return prof, target


OWN_DEF_SHARE = 0.3                # weight of a replaced defender's own defensive strengths
OWN_DEF_METRICS = ("tkl_int", "aerials_won", "aerial_win_pct", "clearances", "blocks", "recoveries", "prog_passes")


def label_of(label: str, b: Brief) -> str:
    extra = [lbl for lbl in b.emphasis_labels if lbl in ("defending", "aerial ability", "ball winning / work rate")]
    return label + (f" ({', '.join(extra)})" if extra else "")


def own_defensive(player: pd.Series, service) -> Dict[str, float]:
    """For a replaced defender or midfielder: the defensive metrics he is above average at for his
    position, weighted by how far above (so 'replace Van Dijk' values aerials and clearances)."""
    if player.get("group") not in ("CB", "FB", "CM") or not hasattr(service, "defensive_profile"):
        return {}
    from forecast.roles import def_column
    prof = service.with_defensive(service.in_latest_season(service.ps))
    me = prof[prof["us_id"] == player["us_id"]]
    if me.empty or float(me["def_minutes_used"].iloc[0] if "def_minutes_used" in me else 0) <= 0:
        return {}
    ref = prof[(prof["group"] == player["group"]) & (prof.get("def_minutes_used", 0) > 0)]
    z = {}
    for m in OWN_DEF_METRICS:
        col, _ = def_column(m)
        if col in prof and ref[col].notna().sum() >= 10:
            v = (float(me[col].iloc[0]) - ref[col].mean()) / (ref[col].std() or 1.0)
            if np.isfinite(v) and v > 0.25:
                z[m] = v
    tot = sum(z.values())
    return {m: OWN_DEF_SHARE * v / tot for m, v in z.items()} if tot else {}


def search(b: Brief, service, n: int = 10, exclude: Tuple[str, ...] = ()) -> Tuple[pd.DataFrame, Optional[pd.Series], RoleProfile]:
    """Ranked shortlist for a brief, plus the replaced player's own projection (as a yardstick)."""
    from forecast.roles import role_scores
    prof, target = profile(b, service)
    thresholds = b.min_goals is not None or b.min_assists is not None or bool(b.beat)
    pool_n = max(n * 6, 120) if thresholds else n
    cap = b.max_value_eur if b.max_value_eur is not None else (HIDDEN_MAX_VALUE if b.sort == "value" else None)
    positions = b.positions or ROLE_POSITIONS.get(b.role or "", [])
    subs = {s for p in positions for s in POSITIONS[p]}
    if b.side and b.side_strict:
        subs = (subs & set(SIDES[b.side])) or set(SIDES[b.side])
    also = {s for p in b.also_positions for s in POSITIONS[p]}
    excl_ids = ([b.replace["us_id"]] if b.replace is not None else []) + [p["us_id"] for p, _ in b.beat] \
        + ([b.style_ref["us_id"]] if b.style_ref is not None else [])
    df = service.candidates(prof, n=pool_n, to_team=b.to_team, max_age=b.max_age,
                            min_age=b.min_age, max_value_eur=cap, min_minutes=b.min_minutes,
                            leagues=b.leagues, exclude=exclude, exclude_ids=excl_ids, target=target,
                            sub_positions=subs or None, also_positions=also or None,
                            min_also_share=ALSO_SHARE,
                            experience=(b.experience_leagues, b.experience_minutes) if b.experience_leagues else None,
                            penalty_taker=b.penalty_taker, min_pass_pct=b.min_pass_pct,
                            style_match=b.replace is not None or b.style_ref is not None,
                            exclude_teams=[b.to_team] if b.to_team and b.exclude_own_club else ())
    bars = []                                  # (label, metric, per-90 bar) from players to out-produce
    for p, m in b.beat:
        ref = service.project(p.to_frame().T.infer_objects(), b.to_team if p["team"] != b.to_team else None).iloc[0]
        bars.append((p["name"], m, float(ref[f"pred_{m}"])))
    b.bars = bars
    him = None
    if b.replace is not None:
        him = service.project(b.replace.to_frame().T.infer_objects(), b.to_team).iloc[0].copy()
        him_row = service.with_defensive(him.to_frame().T.infer_objects())
        him["role_score"] = round(float(role_scores(him_row, prof,
                                                    service.with_defensive(service.in_latest_season(service.ps)))[0]), 2)
        for c in him_row.columns:
            if c.startswith("def_") and c not in him:
                him[c] = him_row[c].iloc[0]
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
    # 'hidden / bargain' already capped the price; within that, rank on fit (output + style), since
    # a value-gap sort put off-profile players first
    return df.head(n).reset_index(drop=True), him, prof


# ── hybrid: a scout's (LLM's) own picks, checked by the data ───────────

SCOUT_PROMPT = (
    "You are an experienced football scout. Read the brief and name the players you would actually put on "
    "the shortlist, from your own knowledge of the game. Respect every hard requirement in the brief (age, "
    "budget, position, league, club). 'Hidden', 'underexplored' or 'under the radar' means players most fans "
    "would not know yet: no established stars. Give each player's current club. Do not invent players.")


def llm_picks(text: str) -> Tuple[List[dict], str]:
    """Gemini's own shortlist for a brief: [{name, club, why}], and the model used. Uses the same model
    chain as the brief reader and gives up quietly (empty list) when every model is out of quota."""
    import asyncio
    from langchain_core.messages import HumanMessage, SystemMessage
    from core import llm
    from models.llm_schemas import ScoutPicks
    from config import settings
    if not settings.has_gemini() or not (text or "").strip():
        return [], ""
    models = [settings.GEMINI_MODEL] + [m.strip() for m in settings.GEMINI_BRIEF_FALLBACKS.split(",")
                                        if m.strip() and m.strip() != settings.GEMINI_MODEL]
    global _MODEL_START
    for i in range(_MODEL_START, len(models)):
        try:
            out = asyncio.run(llm.structured(ScoutPicks, [SystemMessage(content=SCOUT_PROMPT), HumanMessage(content=text)],
                                             name="forecast.scout_picks", model=models[i], temperature=0.0))
        except llm.LLMError as e:
            if "daily quota" not in str(e) and "not available" not in str(e):
                return [], f"failed: {str(e)[:120]}"
            _MODEL_START = i + 1
            continue
        return [p.model_dump() for p in out.picks], models[i]
    return [], "every Gemini model is out of quota"


def check_picks(b: Brief, service, picks: List[dict], shortlist: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Score a scout's picks exactly like the model's own candidates, and flag what the data disagrees
    with: player not in the data, a different current club, over budget, outside the age range, wrong
    position for the profile, too few minutes. `pool_rank` = where he would rank among every player who
    passes the brief's filters (1 = the model's top pick); None when he fails a filter."""
    from forecast.roles import role_scores
    from forecast.service import STYLE_WEIGHT, style_score, _plain
    prof, target = profile(b, service)
    cap = b.max_value_eur if b.max_value_eur is not None else (HIDDEN_MAX_VALUE if b.sort == "value" else None)
    reference = service.with_defensive(service.in_latest_season(service.ps))
    rows = []
    for p in picks:
        hit = _named(p["name"], service, first_word=False)
        if hit is None:
            hit = service.find(p["name"], team=p.get("club"))
            if hit is not None and not (set(_plain(p["name"]).split()) & set(_plain(hit["name"]).split())):
                hit = None
        if hit is None:
            rows.append({"name": p["name"], "llm_club": p.get("club", ""), "why": p.get("why", ""),
                         "flags": "not in the data (outside the big five, or no Understat match)"})
            continue
        proj = service.with_defensive(service.project(hit.to_frame().T.infer_objects(), b.to_team))
        r = proj.iloc[0].to_dict()
        r["role_score"] = round(float(role_scores(proj, prof, reference)[0]), 2)
        cos = np.nan
        if target is not None and (b.replace is not None or b.style_ref is not None):   # as in search()
            v = service._style_vectors(hit.to_frame().T.infer_objects())[0]
            cos = float(np.dot(v, target))
        r["style_similarity"] = float(style_score(cos)) if np.isfinite(cos) else np.nan
        r["fit"] = r["role_score"] + (STYLE_WEIGHT * cos if np.isfinite(cos) else 0.0)
        flags = []
        data_club = r.get("current_club") or r["team"]
        if p.get("club") and _plain(p["club"]) not in _plain(str(data_club)) and _plain(str(data_club)) not in _plain(p["club"]):
            flags.append(f"club: data says {data_club}")
        val = r.get("market_value_eur")
        if cap is not None and (pd.isna(val) or val > cap):
            flags.append("value unknown" if pd.isna(val) else f"over budget ({val / 1e6:.0f}m)")
        age = r.get("age")
        if b.max_age is not None and (pd.isna(age) or age > b.max_age):
            flags.append("age unknown" if pd.isna(age) else f"too old ({age:.0f})")
        if b.min_age is not None and (pd.isna(age) or age < b.min_age):
            flags.append("age unknown" if pd.isna(age) else f"too young ({age:.0f})")
        lg_now = r.get("current_league") or r.get("league")
        if b.leagues and lg_now not in b.leagues:
            flags.append(f"plays in {lg_now}, brief asks for {', '.join(b.leagues)}")
        if b.experience_leagues and service.league_minutes(pd.Series([hit["us_id"]]), b.experience_leagues).iloc[0] \
                < b.experience_minutes:
            flags.append(f"under {b.experience_minutes} min in {', '.join(b.experience_leagues)}")
        if r.get("group") not in prof.groups:
            flags.append(f"plays {r.get('group')}, profile is {'/'.join(prof.groups)}")
        if r.get("minutes", 0) < b.min_minutes:
            flags.append(f"only {r.get('minutes', 0):.0f} min last season")
        if b.replace is not None and hit["us_id"] == b.replace["us_id"]:
            flags.append("is the player being replaced")
        r.update({"llm_club": p.get("club", ""), "why": p.get("why", ""), "flags": "; ".join(flags)})
        rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if "us_id" not in out:
        out["us_id"] = np.nan
    listed = set(shortlist["us_id"]) if shortlist is not None and not shortlist.empty else set()
    out["on_model_list"] = out["us_id"].isin(listed)
    out["pool_rank"] = None
    if "fit" in out:
        pool, _, _ = search(b, service, n=10 ** 6)
        if not pool.empty:
            fits = np.sort(pool["fit"].to_numpy(float))[::-1]
            ok = out["flags"].fillna("").str.len().eq(0) & out["fit"].notna()
            out.loc[ok, "pool_rank"] = [int((fits > f + 1e-3).sum()) + 1 for f in out.loc[ok, "fit"]]
            out.attrs["pool_size"] = len(pool)
    return out
