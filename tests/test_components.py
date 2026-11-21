"""Unit tests: keyword parser, tactical fit model, Understat parsing and matching, finance helpers."""

from datetime import date

import pytest

from agents.financial_analyst import feasibility, years_remaining
from agents.request_parser import fallback_parse
from analysis.tactical_fit import system_key, tactical_fit, weighted_score
from config import current_understat_season
from data_sources.understat_source import (UnderstatSource, best_match, filter_players, position_family,
                                           position_matches)
from tests.conftest import UNDERSTAT_FIXTURE


# ── keyword parser ─────────────────────────────────────────────
@pytest.mark.parametrize("query,position,age,budget", [
    ("Find an inverted winger for Slot's 4-3-3, under 25, budget 60M", "inverted winger", 24, 60),
    ("Box-to-box midfielder for Arteta, max 50M", "box-to-box midfielder", None, 50),
    ("Striker under 23 from La Liga or Bundesliga", "striker", 22, None),
    ("A centre-back, budget €35m", "centre-back", None, 35),
])
def test_fallback_parse(query, position, age, budget):
    c = fallback_parse(query)
    assert c.position == position
    assert c.max_age == age
    assert c.budget_max_m == budget


def test_fallback_parse_leagues_and_no_false_budget():
    c = fallback_parse("Striker under 23 from La Liga or Bundesliga in a 4-3-3")
    assert c.target_leagues == ["bundesliga", "la_liga"]
    assert c.budget_max_m is None


# ── tactical fit ───────────────────────────────────────────────
def test_system_detection():
    assert system_key("Arne Slot 4-3-3") == "slot"
    assert system_key("Guardiola possession") == "guardiola"
    assert system_key(None) == "general"


def test_fit_is_bounded_and_rewards_output():
    elite = {"xGChain_p90": 1.2, "xA_p90": 0.5, "key_passes_p90": 3.5, "xG_p90": 0.8}
    weak = {"xGChain_p90": 0.1, "xA_p90": 0.02, "key_passes_p90": 0.3, "xG_p90": 0.05}
    assert tactical_fit(elite, "slot")[0] == 100.0
    assert tactical_fit(weak, "slot")[0] < 15


def test_missing_features_renormalise_and_lower_coverage():
    stats = {"xGChain_p90": 0.5, "xA_p90": 0.2, "key_passes_p90": 1.5, "xG_p90": 0.35}
    score_without_ea, cov = tactical_fit(stats, "slot")
    assert 0 < cov < 1                       # EA attributes missing
    score, full = weighted_score({**stats, "ea_dribbling": 50, "ea_pace": 50, "ea_physical": 50},
                                 {"xGChain_p90": 3, "xA_p90": 2, "key_passes_p90": 2, "xG_p90": 2,
                                  "ea_dribbling": 1, "ea_pace": 1, "ea_physical": 1})
    assert full == 1.0
    assert score_without_ea == pytest.approx(50.0)
    assert score == pytest.approx(50.1, abs=0.05)   # (9 x 0.5 + 3 x 50/99) / 12


# ── Understat ──────────────────────────────────────────────────
def test_season_rollover():
    assert current_understat_season(date(2026, 11, 18)) == 2026
    assert current_understat_season(date(2027, 3, 1)) == 2026
    assert current_understat_season(date(2026, 7, 31)) == 2025


def test_parse_json_payload_and_embedded_html():
    rows = UNDERSTAT_FIXTURE["premier_league"]
    assert UnderstatSource.parse_league_payload({"players": rows, "teams": {}}) == rows
    html = "<script>var playersData = JSON.parse('[{\\x22id\\x22:\\x221\\x22,\\x22player_name\\x22:\\x22A\\x22}]');</script>"
    assert UnderstatSource.parse_embedded_html(html) == [{"id": "1", "player_name": "A"}]


def test_normalise_per90():
    src = UnderstatSource(season=2026)
    p = src._normalise(UNDERSTAT_FIXTURE["premier_league"][0], "premier_league")
    assert p["player_id"] == "us:1"
    assert p["xG_p90"] == round(8.1 / (1500 / 90), 3)
    assert p["season"] == "2026-27"


def test_positions():
    assert position_family("inverted winger") == "forward"
    assert position_family("right-back") == "defender"
    assert position_family("box-to-box midfielder") == "midfielder"
    assert position_matches("M S", "inverted winger")       # wingers are often listed as M
    assert not position_matches("M S", "striker")
    assert not position_matches("D", "winger")


def test_matching_and_filtering():
    src = UnderstatSource(season=2026)
    rows = [src._normalise(r, "premier_league") for r in UNDERSTAT_FIXTURE["premier_league"]]
    assert best_match("aaron swift", rows)["name"] == "Aaron Swift"
    assert best_match("Completely Made Up", rows) is None
    top = filter_players(rows, "striker", min_minutes=600, sort_by="xG_p90", limit=2)
    assert [p["name"] for p in top] == ["Dan Mercer", "Aaron Swift"]


# ── finance helpers ────────────────────────────────────────────
def test_years_remaining_formats():
    today = date(2026, 6, 30)
    assert years_remaining("Jun 30, 2028", today) == 2.0
    assert years_remaining("30.06.2027", today) == 1.0
    assert years_remaining("", today) is None
    assert years_remaining("garbage", today) is None


def test_feasibility():
    assert feasibility(30, None, 60) == "Easy"
    assert feasibility(58, None, 60) == "Tight"
    assert feasibility(70, None, 60) == "Over budget"
    assert feasibility(30, 1.0, None).startswith("Bargain")
