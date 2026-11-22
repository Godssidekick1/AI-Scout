"""
Defensive / duel data: source parsing, linking to Understat, rates, and their use in roles and briefs.
All offline: payloads below mirror the shapes the real APIs return.
"""

import numpy as np
import pandas as pd
import pytest

from forecast import synthetic
from forecast.data import defensive, fbref_stats, pl_stats, seriea_stats
from forecast.model import MinutesModel, ProjectionModel, prepare
from forecast.roles import ROLES, role_from_text, role_scores
from forecast.service import ForecastService
from forecast.table import player_seasons, season_pairs


# ── sources ────────────────────────────────────────────────────

def test_premierleague_api_rows_use_season_squads_and_opta_keys():
    board = [{"playerMetadata": {"id": "223094", "name": "Erling Haaland", "currentTeam": {"name": "Manchester City"}},
              "stats": {"timePlayed": 2741.0, "totalTackles": 11.0, "tacklesWon": 6.0, "interceptions": 5.0,
                        "blocks": 1.0, "totalClearances": 22.0, "aerialDuelsWon": 57.0, "aerialDuelsLost": 51.0,
                        "recoveries": 34.0, "duelsWon": 94.0, "duelsLost": 160.0}},
             {"playerMetadata": {"id": "1", "name": "Unused Sub"}, "stats": {"yellowCards": 0.0}}]
    squad = {"223094": {"teams": ["Manchester City"], "birth": "2000-07-21"}}
    rows = pl_stats.rows_from_api(board, squad, 2024)
    assert len(rows) == 1                      # players without minutes are dropped
    r = rows[0]
    assert (r["minutes"], r["tackles"], r["interceptions"], r["clearances"]) == (2741.0, 11.0, 5.0, 22.0)
    assert (r["aerials_won"], r["aerials_lost"], r["birth_year"], r["team"]) == (57.0, 51.0, 2000, "Manchester City")


def test_premierleague_mirror_counts_a_mid_season_mover_once(monkeypatch, tmp_path):
    header = "playerId,playerName,team_name,timePlayed,totalTackles,interceptions,blocks,totalClearances," \
             "aerialDuelsWon,aerialDuelsLost,recoveries,duelsWon,duelsLost,tacklesWon\n"
    files = {"Arsenal_3": header + "7,Mover Man,Arsenal,1800,40,20,3,30,25,15,90,100,80,25\n",
             "Chelsea_8": header + "7,Mover Man,Chelsea,1800,40,20,3,30,25,15,90,100,80,25\n"}

    def fake_file(key, url, ttl_days=None):
        club = url.split("/pl_stats/")[1].split("/")[0]
        if club not in files:
            raise defensive.SourceUnavailable("404")
        p = tmp_path / key
        p.write_text(files[club])
        return p
    monkeypatch.setattr(pl_stats.client(), "file", fake_file)
    rows = pl_stats.rows_from_mirror(2020)
    assert len(rows) == 1 and rows[0]["team"] == "Arsenal / Chelsea" and rows[0]["tackles"] == 40


def test_seriea_rows_read_mixed_case_stat_ids():
    players = [{"playerId": "x1", "mediaFirstName": "Morten", "mediaLastName": "Thorsby",
                "team": {"mediaShortName": "Genoa"},
                "stats": [{"statsId": "Time Played", "statsValue": 1902}, {"statsId": "Total Tackles", "statsValue": 49},
                          {"statsId": "tackles-won", "statsValue": 30}, {"statsId": "Interceptions", "statsValue": 5},
                          {"statsId": "Total Clearances", "statsValue": 28}, {"statsId": "Blocks", "statsValue": 1},
                          {"statsId": "Recoveries", "statsValue": 51}, {"statsId": "Aerial Duels won", "statsValue": 107},
                          {"statsId": "Aerial Duels lost", "statsValue": 103}, {"statsId": "Duels won", "statsValue": 180},
                          {"statsId": "Duels lost", "statsValue": 197}]},
               {"playerId": "x2", "displayName": "Bench Guy", "team": {}, "stats": [{"statsId": "minutes-played",
                                                                                   "statsValue": 0}]}]
    rows = seriea_stats.rows_from_players(players, 2024)
    assert len(rows) == 1
    r = rows[0]
    assert r["name"] == "Morten Thorsby" and r["team"] == "Genoa" and r["minutes"] == 1902
    assert (r["tackles"], r["tackles_won"], r["aerials_won"], r["duels_lost"]) == (49, 30, 107, 197)


def test_old_serie_a_seasons_without_tackles_are_not_used():
    df = defensive.frame([{"source": "legaseriea.it", "league": "serie_a", "season": 2014, "src_id": str(i),
                           "name": f"P{i}", "team": "T", "minutes": 2500.0, "tackles_won": 10.0} for i in range(300)])
    assert not defensive.usable(df)
    df["tackles"], df["interceptions"] = 20.0, 10.0
    assert defensive.usable(df)


def test_fbref_rescales_tables_scraped_at_different_times(monkeypatch):
    """2024-25: the defense table covers 9 games, passing the whole season. Rates must use one base."""
    base = {"Season_End_Year": [2025], "Squad": ["Arsenal"], "Comp": ["Premier League"], "Player": ["Declan Rice"],
            "Born": [1999.0], "Url": ["https://fbref.com/en/players/1c7012b8/Declan-Rice"]}
    tables = {
        "defense": {**base, "Mins_Per_90": [9.0], "Tkl_Tackles": [18.0], "TklW_Tackles": [10.0], "Int": [9.0],
                    "Sh_Blocks": [2.0], "Clr": [9.0]},
        "misc": {**base, "Mins_Per_90": [9.0], "Won_Aerial": [9.0], "Lost_Aerial": [9.0], "Recov": [54.0]},
        "possession": {**base, "Mins_Per_90": [9.0], "PrgC_Carries": [9.0]},
        "passing": {**base, "Mins_Per_90": [36.0], "PrgP": [180.0]},     # whole season: 5 per 90
    }
    monkeypatch.setattr(fbref_stats, "read_table",
                        lambda name, ttl_days=30: pd.DataFrame(tables[name]).assign(
                            fbref_id=lambda d: d["Url"].str.extract(r"/players/([0-9a-f]+)/")[0]))
    df = fbref_stats.load([2024])
    r = df.iloc[0]
    assert r["season"] == 2024 and r["minutes"] == pytest.approx(810)
    assert r["prog_passes"] == pytest.approx(45)        # 5 per 90 over the 9 games the defense table covers
    assert r["tackles"] == 18 and r["blocks"] == 2       # blocks = shots blocked (Opta's definition)


def test_load_all_takes_the_best_source_per_league_season_and_survives_an_outage(monkeypatch):
    def rows(source, league, season, n=300, tackles=20.0):
        return defensive.frame([{"source": source, "league": league, "season": season, "src_id": str(i),
                                 "name": f"P{i}", "team": "T", "minutes": 2000.0, "tackles": tackles,
                                 "interceptions": 10.0} for i in range(n)])

    def broken(seasons):
        raise RuntimeError("HTTP 503")
    monkeypatch.setattr(defensive, "_sources", lambda: [
        ("premierleague.com", lambda s: rows("premierleague.com", "premier_league", 2022)),
        ("legaseriea.it", broken),
        ("fbref", lambda s: pd.concat([rows("fbref", "premier_league", 2022, tackles=99.0),
                                       rows("fbref", "la_liga", 2022)]))])
    out, cov = defensive.load_all([2022])
    epl = out[out["league"] == "premier_league"]
    assert set(epl["source"]) == {"premierleague.com"} and (epl["tackles"] == 20).all()
    assert set(out[out["league"] == "la_liga"]["source"]) == {"fbref"}
    assert "unavailable" in cov[cov["source"] == "legaseriea.it"]["status"].iloc[0]


# ── linking and rates ──────────────────────────────────────────

def test_link_matches_on_name_team_and_minutes_and_never_twice():
    us = pd.DataFrame({"us_id": ["10", "11", "12"], "name": ["Virgil van Dijk", "Rodrigo Hernández", "Ben White"],
                       "team": ["Liverpool", "Manchester City", "Arsenal"], "league": "premier_league",
                       "season": 2022, "minutes": [3060.0, 3100.0, 2400.0]})
    rows = defensive.frame([
        {"source": "x", "league": "premier_league", "season": 2022, "src_id": "a", "name": "Virgil van Dijk",
         "team": "Liverpool", "minutes": 3060.0},
        {"source": "x", "league": "premier_league", "season": 2022, "src_id": "b", "name": "Rodri",
         "team": "Manchester City", "minutes": 3098.0},
        {"source": "x", "league": "premier_league", "season": 2022, "src_id": "c", "name": "Ben Whiteley",
         "team": "Brentford", "minutes": 300.0},
        {"source": "x", "league": "la_liga", "season": 2022, "src_id": "d", "name": "Ben White",
         "team": "Arsenal", "minutes": 2400.0}])
    linked = defensive.link(rows, us)
    assert linked["us_id"].tolist() == ["10", "11", None, None]


def test_rates_ratios_and_shrinkage():
    ps = pd.DataFrame({"us_id": ["a", "b", "c"] * 2, "season": [2023] * 3 + [2024] * 3, "group": "CB",
                       "league": "premier_league"})
    table = pd.DataFrame({"us_id": ["a", "b", "a", "b", "c"], "season": [2023, 2023, 2024, 2024, 2024],
                          "def_minutes": [2700.0, 2700.0, 2700.0, 2700.0, 180.0], "def_source": "x",
                          **{c: np.nan for c in defensive.TOTALS}})
    table["tackles"], table["interceptions"] = [60, 30, 60, 30, 10], [30, 30, 30, 30, 0]
    table["aerials_won"], table["aerials_lost"] = [60, 20, 60, 20, 1], [40, 20, 40, 20, 1]
    ps = defensive.add_rates(ps, table)
    a24 = ps[(ps.us_id == "a") & (ps.season == 2024)].iloc[0]
    assert a24["tkl_int_p90"] == pytest.approx(3.0) and a24["aerial_win_pct"] == pytest.approx(0.6)
    assert np.isnan(ps[(ps.us_id == "c") & (ps.season == 2024)]["aerial_win_pct"].iloc[0])   # < 5 aerials
    prof = defensive.recent_rates(ps, ps[ps.season == 2024]).set_index("us_id")
    # c's 5.0 per 90 over 180 minutes is pulled most of the way to the CB mean (3.0)
    assert 3.0 < prof.at["c", "def_tkl_int_p90"] < 3.5 and prof.at["c", "def_minutes_used"] == 180
    assert prof.at["a", "def_tkl_int_p90"] > prof.at["b", "def_tkl_int_p90"]


# ── roles, service, briefs ─────────────────────────────────────

@pytest.fixture(scope="module")
def service():
    us, tm, us_to_tm, elo = synthetic.generate(n_players=600, seed=5)
    ps = player_seasons(us, tm, us_to_tm, elo)
    ps = defensive.add_rates(ps, defensive.synthetic_table(ps, 5))
    pairs = season_pairs(ps)
    full, hist = prepare(ps, pairs)
    return ForecastService(ps=ps, model=ProjectionModel().fit(full, hist), minutes=MinutesModel().fit(full, ps),
                           values={u: 1e7 for u in ps["us_id"].unique()}, synthetic=True)


@pytest.mark.parametrize("text,role", [
    ("a no-nonsense stopper", "stopper_cb"), ("a dominant centre-back", "stopper_cb"),
    ("ball-winning midfielder", "ball_winner"), ("a target man", "target_forward"),
    ("a Stones-type centre-back", "ball_playing_cb"), ("holding midfielder", "deep_playmaker"),
])
def test_new_roles_from_text(text, role):
    assert role_from_text(text) == role


def test_defensive_role_ranks_on_observed_rates(service, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "ENABLE_VECTOR_INDEX", False)
    df = service.candidates("stopper_cb", n=10, min_minutes=900)
    assert not df.empty and (df["group"] == "CB").all()
    assert df["role_score"].is_monotonic_decreasing
    ref = service.with_defensive(service.in_latest_season(service.ps))
    cb = ref[ref["group"] == "CB"]
    assert df["def_aerials_won_p90"].mean() > cb["def_aerials_won_p90"].mean()


def test_uncovered_player_counts_as_average_on_defensive_metrics(service):
    proj = pd.DataFrame({"def_tkl_int_p90": [9.0, np.nan], "def_minutes_used": [2000.0, 0.0],
                         **{f"pred_{m}": [0.0, 0.0] for m in ("npxG",)}})
    ref = service.with_defensive(service.in_latest_season(service.ps))
    from forecast.roles import RoleProfile
    s = role_scores(proj, RoleProfile("t", "t", ("CB",), {"tkl_int": 1.0}), ref)
    assert s[0] > 1 and s[1] == 0


def test_brief_turns_defending_and_aerials_into_weights(service, monkeypatch):
    from config import settings
    from forecast import brief as briefs
    monkeypatch.setattr(settings, "ENABLE_VECTOR_INDEX", False)
    b = briefs.parse("a centre-back who is reliable defensively and great in the air, two-footed", service)
    assert {"defending", "aerial ability"} <= set(b.emphasis_labels)
    assert "two-footed" in b.unmeasured and not any("aerial" in u or "defen" in u for u in b.unmeasured)
    prof, _ = briefs.profile(b, service)
    assert prof.weights.get("aerials_won", 0) > 0.05 and prof.weights.get("tkl_int", 0) > 0.05
    assert sum(prof.weights.values()) == pytest.approx(1, abs=0.02)
    df, _, _ = briefs.search(b, service, n=5)
    assert not df.empty


def test_llm_spec_with_a_defensive_must_beat_becomes_an_emphasis(service):
    from forecast import brief as briefs
    from models.llm_schemas import BeatSpec, BriefSpec
    star = service.in_latest_season(service.current()).iloc[0]["name"]
    spec = BriefSpec(interpretation="x", positions=["centre_back"], emphasis=["aerial"],
                     must_beat=[BeatSpec(player=star, quality="defending")])
    b = briefs.from_spec(spec, service)
    assert not b.beat and {"aerial ability", "defending"} <= set(b.emphasis_labels)


def test_old_model_without_defensive_columns_still_searches(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "ENABLE_VECTOR_INDEX", False)
    us, tm, us_to_tm, elo = synthetic.generate(n_players=400, seed=9)
    ps = player_seasons(us, tm, us_to_tm, elo)
    pairs = season_pairs(ps)
    full, hist = prepare(ps, pairs)
    svc = ForecastService(ps=ps, model=ProjectionModel().fit(full, hist), minutes=MinutesModel().fit(full, ps))
    assert not svc.candidates("stopper_cb", n=5).empty
