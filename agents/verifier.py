"""
Step 3: Verifier (the critic)
=============================
Deterministic checks on every name the finder proposed. A player only becomes a
candidate if they resolve to a real Understat record and pass the constraints:

    exists in top-5 league data -> enough minutes -> right role -> right league
    -> age <= max (Transfermarkt age, else the agent's claim) -> value <= budget

Everything rejected is recorded with a reason. If fewer than MIN_VERIFIED_CANDIDATES
survive, the graph routes back to the finder (with the rejections as feedback), up
to FINDER_MAX_ATTEMPTS times. This is the self-correcting loop of the pipeline.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from analysis.league_normalizer import apply_league_tax
from analysis.tactical_fit import statistical_rating
from config import settings
from core.context import get_context
from core.tracing import traced_node
from data_sources.understat_source import position_family, position_matches
from models.llm_schemas import SearchConstraints
from models.player_data import Candidate, ProposedCandidate

logger = logging.getLogger(__name__)


@traced_node("verifier")
async def verify_candidates(state: Dict) -> Dict:
    c = SearchConstraints.model_validate(state["constraints"])
    existing = [Candidate.model_validate(x) for x in state.get("candidates", [])]
    seen_ids = {x.player_id for x in existing}

    proposals, names = [], set()
    for raw in state.get("proposals", []):
        p = ProposedCandidate.model_validate(raw)
        if p.name.lower() not in names:
            names.add(p.name.lower())
            proposals.append(p)

    results = await asyncio.gather(*(check(p, c) for p in proposals))

    accepted: List[Candidate] = []
    rejected: List[Dict] = []
    for p, (cand, reason) in zip(proposals, results):
        if cand is None:
            rejected.append({"name": p.name, "reason": reason})
        elif cand.player_id in seen_ids:
            continue
        else:
            seen_ids.add(cand.player_id)
            accepted.append(cand)

    family = position_family(c.position)
    pool = existing + accepted
    pool.sort(key=lambda x: statistical_rating(x.model_dump(), family), reverse=True)
    pool = pool[: settings.MAX_CANDIDATES]

    return {
        "candidates": [x.model_dump() for x in pool],
        "rejected": rejected,
        "messages": [f"Verifier: {len(accepted)} of {len(proposals)} proposals verified; "
                     f"{len(pool)} candidates in pool."],
    }


async def check(p: ProposedCandidate, c: SearchConstraints) -> Tuple[Optional[Candidate], str]:
    ctx = get_context()
    stats = await ctx.find_player(p.name, p.team)
    if stats is None:
        return None, f"not found in top-5 league data for {settings.UNDERSTAT_SEASON}"
    if stats["minutes_played"] < c.min_minutes:
        return None, f"only {stats['minutes_played']} minutes (< {c.min_minutes})"
    if not position_matches(stats["understat_position"], c.position):
        return None, f"plays {stats['understat_position']!r}, not a {c.position}"
    if c.target_leagues and stats["league"] not in c.target_leagues:
        return None, f"plays in {stats['league']}, outside target leagues"

    market, ea = await asyncio.gather(ctx.market_profile(stats["name"]), ctx.ea_ratings(stats["name"]))
    market = market or {}

    age, age_source = market.get("age"), "transfermarkt"
    if age is None and p.claimed_age:
        age, age_source = p.claimed_age, "llm_claim"
    if age is None:
        age_source = "unknown"
    if c.max_age and age is not None and age > c.max_age:
        return None, f"age {age} > {c.max_age}"

    value = market.get("market_value_eur") or None
    if c.budget_max_m and value and value > c.budget_max_m:
        return None, f"market value EUR {value}M > budget {c.budget_max_m}M"

    sources = ["understat"] + (["transfermarkt"] if market else []) + (["ea_fc"] if ea else [])
    taxed = apply_league_tax({k: v for k, v in stats.items()}, stats["league"])
    cand = Candidate(
        **{k: v for k, v in taxed.items() if k in Candidate.model_fields},
        position=c.position,
        age=age,
        age_source=age_source,
        market_value_eur_m=value,
        contract_end_date=market.get("contract_end_date", "") or "",
        agent=market.get("agent", "") or "",
        why_recommended=p.reason,
        sources=sources,
        **_ea_fields(ea),
    )
    await attach_projection(cand, c)
    return cand, ""


async def attach_projection(cand: Candidate, c: SearchConstraints) -> None:
    """Add the forecast model's projection at the target club, when the model is built."""
    service = await get_context().forecast()
    if service is None:
        return
    row = service.find(cand.name, cand.team)
    if row is None:
        return
    import pandas as pd
    from agents.player_finder import projection_dict
    from forecast.roles import ROLES, role_scores
    proj = await asyncio.to_thread(service.project, row.to_frame().T, c.target_team)
    r = proj.iloc[0]
    cand.projection = projection_dict(r)
    if c.role:
        ref = service.in_latest_season(service.ps)
        cand.role_score = round(float(role_scores(proj, ROLES[c.role], ref)[0]), 2)
    if "forecast" not in cand.sources:
        cand.sources.append("forecast")


def _ea_fields(ea: Optional[Dict]) -> Dict:
    if not ea:
        return {}
    out = {"ea_overall": ea.get("overall_rating") or None}
    for attr in ("pace", "dribbling", "passing", "defending", "physical"):
        out[f"ea_{attr}"] = ea.get(attr) or None
    return out


def route_after_verify(state: Dict):
    """Loop back to the finder while too few candidates survived, else fan out to the checks."""
    enough = len(state.get("candidates", [])) >= settings.MIN_VERIFIED_CANDIDATES
    out_of_attempts = state.get("finder_attempts", 0) >= settings.FINDER_MAX_ATTEMPTS
    nothing_new = not state.get("proposals")
    if enough or out_of_attempts or nothing_new:
        return ["background_checker", "financial_analyst"]
    return "player_finder"
