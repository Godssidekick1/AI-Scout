"""
Forecast model tests. The data is synthetic with known true parameters
(forecast/synthetic.py), so we can check that fitting recovers what it should
and that the backtest machinery and the agent wiring work end to end, offline.
"""

import asyncio
import json

import numpy as np
import pandas as pd
import pytest

from config import settings
from forecast import backtest, synthetic
from forecast.config import METRICS, position_group
from forecast.data import clubelo
from forecast.data.transfermarkt import PlayerMatcher, normalise
from forecast.data.understat_history import rows_from_raw
from forecast.model import MinutesModel, ProjectionModel, prepare
from forecast.roles import ROLES, role_from_text, style_vector, value_gap
from forecast.service import ForecastService
from forecast.table import expected_share, player_seasons, season_pairs


@pytest.fixture(scope="module")
def world():
    us, tm, us_to_tm, elo = synthetic.generate(n_players=900, seed=3)
    ps = player_seasons(us, tm, us_to_tm, elo)
    pairs = season_pairs(ps)
    full, hist = prepare(ps, pairs)
    return {"us": us, "tm": tm, "map": us_to_tm, "elo": elo, "ps": ps, "pairs": pairs, "full": full, "hist": hist}


@pytest.fixture(scope="module")
def fitted(world):
    return ProjectionModel().fit(world["full"], world["hist"])


@pytest.fixture(scope="module")
def service(world, fitted):
    rng = np.random.default_rng(0)
    values = {u: float(np.exp(rng.normal(16, 1))) for u in world["ps"]["us_id"].unique()}
    return ForecastService(ps=world["ps"], model=fitted, minutes=MinutesModel().fit(world["full"], world["ps"]),
                           values=values, synthetic=True)


# ── table ──────────────────────────────────────────────────────

def test_pairs_link_consecutive_seasons_and_flag_movers(world):
    pairs = world["pairs"]
    assert (pairs["target_season"] == pairs["season"] + 1).all()
    stayers = pairs[~pairs["mover"]]
    for m in METRICS:
        assert (stayers[f"dusage_{m}"] == 0).all()
    assert pairs["mover"].sum() > 100 and pairs["league_change"].sum() > 50


def test_expected_share_adds_the_player_to_the_new_team():
    assert expected_share(10, 30) == pytest.approx(0.25)
    assert expected_share(0, 0) == 0.0


def test_mid_season_mover_is_one_row_with_summed_minutes():
    base = {"understat_position": "F", "games": 10, "goals": 1, "assists": 1, **{m: 1.0 for m in METRICS}}
    us = pd.DataFrame([
        {"us_id": "1", "name": "A", "team": "X", "league": "premier_league", "season": 2020, "minutes": 900, **base},
        {"us_id": "1", "name": "A", "team": "Y", "league": "la_liga", "season": 2020, "minutes": 300, **base},
    ])
    ps = player_seasons(us)
    assert len(ps) == 1 and ps.iloc[0]["minutes"] == 1200 and ps.iloc[0]["team"] == "X"


def test_position_groups():
    assert position_group("Centre-Back") == "CB"
    assert position_group("Goalkeeper") is None
    assert position_group(None, "F M S") == "AM"
    assert position_group(None, "D S") == "CB"


# ── model ──────────────────────────────────────────────────────

def test_league_and_team_effects_have_the_true_order_and_sign(fitted):
    # truth: Ligue 1 +12%, Bundesliga +8%, La Liga +6% vs the Premier League; +8% per 100 Elo
    e = fitted.explain("shots")
    assert e["league[ligue_1]"] > e["league[la_liga]"] > 0
    assert 0.03 < e["team_elo_per_100"] < 0.15


def test_age_curve_rises_young_and_falls_old_for_strikers(fitted):
    curve = fitted.fits["npxG"].age_step["ST"]
    assert curve[21] > 0 > curve[33]
    # centre-backs peak later in the synthetic truth, so they decline less at 31
    assert fitted.fits["xGBuildup"].age_step["CB"][31] > fitted.fits["xGBuildup"].age_step["ST"][31]


def test_shrinkage_pulls_short_samples_toward_the_group_mean(fitted, world):
    assert 150 <= fitted.fits["npxG"].k <= 4500
    pairs, hist = world["full"], world["hist"]
    pred = fitted.predict(pairs.head(200), hist.head(200))
    assert (pred[[f"pred_{m}" for m in METRICS]] >= 0).all().all()
    assert (pred["lo_npxG"] <= pred["pred_npxG"]).all() and (pred["pred_npxG"] <= pred["hi_npxG"]).all()


def test_backtest_full_model_beats_both_baselines(world):
    res = backtest.run(world["ps"], world["pairs"], test_seasons=[2023, 2024, 2025])
    table = backtest.ladder_table(res["scores"])
    assert table.loc["+trend", "improvement_vs_last_season_%"] > 10
    for m in METRICS:
        assert table.loc["+trend", m] < table.loc["last_season", m]
        assert table.loc["+trend", m] < table.loc["handset", m]
    cov = res["coverage"]["coverage_80"]
    assert ((cov > 0.7) & (cov < 0.95)).all()
    assert (res["minutes"]["model_rmse"] < res["minutes"]["same_minutes_rmse"]).all()
    md = backtest.to_markdown(table)
    assert md.startswith("| model |") and "last_season" in md


def test_backtest_never_trains_on_the_test_season(world, monkeypatch):
    seen = []
    real_fit = ProjectionModel.fit

    def spy(self, pairs, hist):
        seen.append(int(pairs["target_season"].max()))
        return real_fit(self, pairs, hist)

    monkeypatch.setattr(ProjectionModel, "fit", spy)
    backtest.run(world["ps"], world["pairs"], test_seasons=[2025])
    assert seen and max(seen) <= 2024


# ── roles ──────────────────────────────────────────────────────

@pytest.mark.parametrize("text,role", [
    ("a replacement for John Stones", "ball_playing_cb"),
    ("inverted winger for a 4-3-3", "inside_forward"),
    ("a fox in the box", "poacher"),
    ("box-to-box midfielder", "box_to_box"),
    ("attacking right-back", "attacking_fullback"),
    ("a number 9 striker", "complete_forward"),
    ("a scientific approach", None),            # 'cb' etc. must not match inside words
])
def test_role_from_text(text, role):
    assert role_from_text(text) == role


def test_value_gap_flags_cheap_high_scorers():
    score = np.array([2.0, 1.0, 0.0, -1.0, 1.0])
    value = np.array([20e6, 20e6, 5e6, 2e6, 60e6])
    gap = value_gap(score, value)
    assert gap[0] > 0 > gap[4]


def test_style_vector_ignores_level():
    scales = {m: 1.0 for m in METRICS}
    a = style_vector({m: 0.2 for m in METRICS}, scales)
    b = style_vector({m: 0.4 for m in METRICS}, scales)
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1)


# ── service ────────────────────────────────────────────────────

@pytest.mark.parametrize("vector_index", [True, False])
def test_candidates_respect_filters_and_rank_by_role_score(service, monkeypatch, vector_index):
    monkeypatch.setattr(settings, "ENABLE_VECTOR_INDEX", vector_index)
    df = service.candidates("poacher", n=8, max_age=26, max_value_eur=30e6)
    assert 0 < len(df) <= 8
    assert set(df["group"]) <= set(ROLES["poacher"].groups)
    assert (df["age"].dropna() <= 26).all()
    assert (df["market_value_eur"].dropna() <= 30e6).all()
    assert list(df["role_score"]) == sorted(df["role_score"], reverse=True)
    assert {"pred_npxG", "lo_npxG", "hi_npxG", "value_gap", "style_similarity", "expected_minutes"} <= set(df.columns)


def test_project_to_a_new_club_sets_usage_and_league(service):
    player = service.current().query("league == 'ligue_1' and group == 'ST'").head(1)
    target = service.ps.query("league == 'premier_league'")["team"].iloc[0]
    out = service.project(player, target)
    assert out.iloc[0]["team_next"] == target and out.iloc[0]["league_next"] == "premier_league"


def test_service_round_trips_to_disk(service, tmp_path):
    path = service.save(tmp_path / "svc.pkl")
    again = ForecastService.load(path)
    assert again is not None and again.latest_season == service.latest_season


# ── loaders ────────────────────────────────────────────────────

def test_understat_rows_keep_last_team_and_parse_numbers():
    rows = rows_from_raw([{"id": "9", "player_name": "X", "team_title": "A,B", "position": "F", "time": "900",
                           "games": "10", "goals": "3", "assists": "1", "npxG": "2.5", "xA": "1.0", "shots": "20",
                           "key_passes": "10", "xGChain": "4", "xGBuildup": "1"}], "serie_a", 2019)
    assert rows[0]["team"] == "B" and rows[0]["minutes"] == 900.0 and rows[0]["npxG"] == 2.5


def test_clubelo_snapshot_and_aliases():
    snap = clubelo.parse_snapshot("Rank,Club,Country,Level,Elo,From,To\n1,Man City,ENG,1,2050.5,x,y\n"
                                  "2,Bayern,GER,1,1990,x,y\n")
    assert clubelo.match_team("Manchester City", "premier_league", snap) == "Man City"
    assert clubelo.match_team("Bayern Munich", "bundesliga", snap) == "Bayern"
    assert clubelo.match_team("Nowhere FC", "premier_league", snap) is None


def test_transfermarkt_matcher_uses_club_history_for_namesakes():
    players = pd.DataFrame({"player_id": [1, 2], "name": ["Rodrigo Hernandez", "Rodrigo Hernandez"]})
    vals = pd.DataFrame({"player_id": [1, 2], "date": ["2022-10-01", "2022-10-01"],
                         "current_club_name": ["Manchester City", "Villarreal"]})
    m = PlayerMatcher(players, vals)
    assert m.match("Rodrigo Hernández", "Manchester City", 2022) == 1
    assert normalise("Kylian Mbappé-Lottin") == "kylian mbappe lottin"


# ── agent wiring ───────────────────────────────────────────────

def test_finder_tools_expose_the_model(service):
    from agents.player_finder import build_tools
    tools = {t.name: t for t in build_tools(service, None)}
    assert {"find_role_candidates", "project_player"} <= set(tools)
    out = json.loads(asyncio.run(tools["find_role_candidates"].ainvoke({"role": "poacher", "limit": 5})))
    assert len(out) == 5 and "projection_next_season" in out[0] and "value_gap" in out[0]
    one = json.loads(asyncio.run(tools["project_player"].ainvoke({"name": out[0]["name"]})))
    assert one["name"] == out[0]["name"]


def test_no_model_tools_without_a_service():
    from agents.player_finder import build_tools
    assert "find_role_candidates" not in {t.name for t in build_tools(None)}


def test_verifier_attaches_projection_and_reports_compute_value_gap(service, monkeypatch):
    from agents import report_writer, verifier
    from core.context import start_context
    from models.llm_schemas import SearchConstraints
    from models.player_data import Candidate

    ctx = start_context()

    async def fake_forecast():
        return service
    monkeypatch.setattr(ctx, "forecast", fake_forecast)

    row = service.current().query("group == 'ST'").iloc[0]
    cand = Candidate(player_id="us:x", name=row["name"], team=row["team"], sources=["understat"])
    c = SearchConstraints(position="striker", role="poacher")
    asyncio.run(verifier.attach_projection(cand, c))
    assert "pred_npxG" in cand.projection and cand.role_score is not None and "forecast" in cand.sources

    cands = [{"player_id": str(i), "role_score": s, "market_value_eur_m": v}
             for i, (s, v) in enumerate([(1.5, 10), (1.0, 30), (0.0, 20), (-0.5, 5)])]
    gaps = report_writer.value_gaps(cands, {})
    assert gaps["0"] > gaps["1"]


def test_clubelo_retries_then_raises_on_server_errors(monkeypatch, tmp_path):
    import httpx
    monkeypatch.setattr(type(settings), "cache_path", property(lambda self: tmp_path))
    monkeypatch.setattr(clubelo.time, "sleep", lambda s: None)
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return httpx.Response(502, text="")

    monkeypatch.setattr(clubelo.httpx, "get", fake_get)
    with pytest.raises(clubelo.ClubEloUnavailable):
        clubelo.fetch_snapshot("2024-08-01", retries=2)
    assert len(calls) == 3
    assert not (tmp_path / "clubelo" / "2024-08-01.csv").exists()   # error pages are never cached


def test_understat_maintenance_page_detected():
    from data_sources.understat_source import UnderstatSource
    assert UnderstatSource.is_maintenance_page("<html><body><h1>We will come back soon</h1></body></html>")
    assert not UnderstatSource.is_maintenance_page("<script>var playersData = JSON.parse('[]')</script>")


def test_archive_validation_checks_season_and_completeness():
    from forecast.data import understat_archive as ua

    def page(dates, players=({"id": "1", "player_name": "A"},)):
        enc = lambda x: json.dumps(list(x)).replace("'", "\'")
        return (f"<script>var datesData = JSON.parse('{enc(dates)}');"
                f"var playersData = JSON.parse('{enc(players)}');</script>")

    done = [{"datetime": "2019-08-09 20:00:00", "isResult": True}, {"datetime": "2020-07-26 15:00:00", "isResult": True}]
    assert ua.validate(page(done), 2019)[1] == "ok"
    assert ua.validate(page(done), 2020)[1] == "page shows season 2019"
    unfinished = done + [{"datetime": "2020-07-27 15:00:00", "isResult": False}] * 3
    assert "not yet played" in ua.validate(page(unfinished), 2019)[1]
    cands = ua.candidates([["20200729000000", "https://understat.com/league/EPL/2019"],
                           ["20200301000000", "https://understat.com/league/EPL/2019"],   # mid-season: skipped
                           ["20160620000000", "https://understat.com/league/Ligue_1"],     # yearless, June: 2015
                           ["20161001000000", "https://understat.com/league/Ligue_1"]])    # yearless, autumn: skipped
    assert cands == {("EPL", 2019): [("20200729000000", "https://understat.com/league/EPL/2019")],
                     ("Ligue_1", 2015): [("20160620000000", "https://understat.com/league/Ligue_1")]}


def test_results_elo_rates_and_handles_promotion(monkeypatch):
    from forecast.data import results_elo

    def fake(league, season):
        teams = ["A", "B", "C"] if season < 2014 else ["A", "B", "D"]   # C relegated, D promoted in 2014
        rows = [(teams[0], teams[1], 3, 0), (teams[0], teams[2], 2, 0), (teams[1], teams[2], 1, 1)]
        return pd.DataFrame(rows, columns=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    monkeypatch.setattr(results_elo, "load_results", fake)
    table = results_elo.rate_league("premier_league", [2014])[2014]
    assert table["A"] > 1500 > table["D"]
    snap = results_elo.snapshots([2014], leagues=["premier_league"])[2014]
    assert snap["A"]["country"] == "ENG"
    assert clubelo.match_team("A", "premier_league", snap) == "A"


def test_transfermarkt_matcher_rejects_one_word_subset_without_club():
    players = pd.DataFrame({"player_id": [1, 2], "name": ["Alex", "Kepa Arrizabalaga"]})
    vals = pd.DataFrame({"player_id": [2], "date": ["2025-09-01"], "market_value_in_eur": [1e6],
                         "current_club_name": ["Arsenal FC"]})
    m = PlayerMatcher(players, vals)
    assert m.match("Alex Berenguer", "Athletic Club", 2025) is None      # 'Alex' alone is not him
    assert m.match("Kepa", "Arsenal", 2025) == 2                          # known-as name, club agrees


def test_predict_cli_player_and_role(service, tmp_path, capsys, monkeypatch):
    from forecast import predict
    monkeypatch.setattr(settings, "ENABLE_VECTOR_INDEX", False)
    path = service.save(tmp_path / "svc.pkl")
    cur = service.current()
    name = cur[cur["season"] == service.latest_season].iloc[0]["name"]
    club = cur[(cur["season"] == service.latest_season) & (cur["name"] != name)].iloc[0]["team"]

    predict.main(["--model", str(path), "player", name, "--to", club, "--age", "31", "--minutes", "1800"])
    out = capsys.readouterr().out
    assert f"moves to {club}" in out and "1800 (your input)" in out and "npxG" in out

    # the breakdown adds up: next = ability * (1 + terms)
    row = cur[cur["name"] == name]
    proj = service.project(row, club, age=31, minutes=1800).iloc[0]
    rows = service._pair_rows(row, club).assign(age=31.0)
    hist = predict.history_features(service.ps, rows)
    for m in METRICS:
        rows[f"trend_{m}"] = hist[f"trend_{m}"].to_numpy()
    c = service.model.contributions(rows, hist)["npxG"].iloc[0]
    assert abs(max(c["ability"] * (1 + c.drop("ability").sum()), 0) - proj["pred_npxG"]) < 1e-9

    predict.main(["--model", str(path), "role", "a Stones replacement", "-n", "3"])
    assert "Ball-playing centre-back" in capsys.readouterr().out


def test_a_league_missing_its_latest_season_stays_searchable(service, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_VECTOR_INDEX", False)
    ps = service.ps
    gappy = ForecastService(ps=ps[~((ps["league"] == "bundesliga") & (ps["season"] == ps["season"].max()))],
                            model=service.model, minutes=service.minutes, values=service.values, synthetic=True)
    club = gappy.ps[gappy.ps["league"] == "bundesliga"]["team"].iloc[-1]
    assert gappy._team_row(club) is not None
    df = gappy.candidates("poacher", n=200, min_minutes=0, leagues=["bundesliga"])
    assert not df.empty
