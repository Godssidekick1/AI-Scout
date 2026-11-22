"""
Forecast service
================
What the agent talks to at run time. Holds the player-season table and the
fitted models, and answers two questions:

    candidates(role, ...)        players whose style is closest to a role, with filters
    project(player, to_team)     next season's per-90 output at that club, with a range

Built and saved by `python -m forecast.build`; loaded from disk by the agent.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import settings
from forecast.config import METRICS
from forecast.model import MinutesModel, ProjectionModel, history_features
from forecast.roles import (ROLES, RoleProfile, group_centres, metric_scales, role_scores, style_vector,
                            target_vector, uses_defensive, value_gap)
from forecast.table import expected_share, team_totals

logger = logging.getLogger(__name__)


_FOLD = str.maketrans({"ø": "o", "æ": "ae", "ß": "ss", "ı": "i", "ł": "l", "đ": "d", "-": " ", "'": ""})


def _plain(text: str) -> str:
    """Lower case, accents removed: 'Ødegaard' -> 'odegaard'."""
    import unicodedata
    t = unicodedata.normalize("NFKD", str(text).lower().translate(_FOLD))
    return "".join(c for c in t if not unicodedata.combining(c)).strip()


STYLE_WEIGHT = 0.75     # ranking weight of style (cosine, -1..1) next to the output score (in SDs)


def style_score(cos) -> np.ndarray:
    """Cosine between style vectors (centred on the position's average, so -1..1) as 0-100:
    100 = same shape, 50 = unrelated, 0 = the opposite profile."""
    return np.round((np.asarray(cos, float) + 1) * 50).astype(float)


def artifact_path() -> Path:
    return settings.cache_path / "forecast" / "service.pkl"


@dataclass
class ForecastService:
    ps: pd.DataFrame                       # player-season table, all seasons
    model: ProjectionModel
    minutes: MinutesModel
    values: Dict[str, float] = field(default_factory=dict)   # us_id -> latest market value (EUR)
    backtest_summary: str = ""
    synthetic: bool = False
    # us_id -> current_club, current_league, moved, active, position_shares (forecast.data.transfermarkt
    # .player_profiles). None for builds made before it existed: those filters are then skipped.
    profiles: Optional[pd.DataFrame] = None

    # ── persistence ───────────────────────────────────────────
    def save(self, path: Optional[Path] = None) -> Path:
        path = path or artifact_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: Optional[Path] = None) -> Optional["ForecastService"]:
        path = path or artifact_path()
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── helpers ───────────────────────────────────────────────
    @property
    def latest_season(self) -> int:
        return int(self.ps["season"].max())

    def in_latest_season(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows from each league's most recent season in the data. Usually one season for all,
        but a league whose latest season failed to load would otherwise drop out entirely."""
        latest = self.ps.groupby("league")["season"].max()
        return df[df["season"] == df["league"].map(latest)]

    def current(self) -> pd.DataFrame:
        """Each player's most recent season in the table."""
        return self.ps.sort_values("season").groupby("us_id").tail(1)

    def find(self, name: str, team: Optional[str] = None) -> Optional[pd.Series]:
        """A player by name. Full names match fuzzily; a surname or part of a name ("Saka",
        "Odegaard") matches players whose name contains every typed word, preferring his
        club if given, then the most recent season and most minutes."""
        from data_sources.understat_source import best_match
        cur = self.current()
        rows = [{"name": n, "team": t, "us_id": i} for n, t, i in zip(cur["name"], cur["team"], cur["us_id"])]
        hit = best_match(name, rows, team)
        if hit:
            return cur[cur["us_id"] == hit["us_id"]].iloc[0]
        words = set(_plain(name).split())
        if not words:
            return None
        sub = cur[[words <= set(_plain(n).split()) for n in cur["name"]]]
        if team and not sub.empty:
            at = sub[[_plain(team) in _plain(t) for t in sub["team"]]]
            sub = at if not at.empty else sub
        return sub.sort_values(["season", "minutes"]).iloc[-1] if not sub.empty else None

    def _team_row(self, team: str) -> Optional[pd.Series]:
        tt = self.in_latest_season(team_totals(self.ps).reset_index())
        from thefuzz import fuzz
        best, score = None, 0
        for i, r in tt.iterrows():
            s = fuzz.token_set_ratio(team.lower(), r["team"].lower())
            if s > score:
                best, score = r, s
        return best if score >= 80 else None

    def _pair_rows(self, players: pd.DataFrame, to_team: Optional[str]) -> pd.DataFrame:
        """Build 'next season' rows: each player moving to `to_team` (or staying, if None)."""
        rows = players.copy()
        target = self._team_row(to_team) if to_team else None
        rows["team_next"] = target["team"] if target is not None else rows["team"]
        rows["league_next"] = target["league"] if target is not None else rows["league"]
        rows["elo_next"] = float(target["elo"]) if target is not None else rows["elo"]
        rows["mover"] = rows["team_next"] != rows["team"]
        for m in METRICS:
            if target is not None:
                exp = np.array([expected_share(p, float(target[f"team_{m}"])) for p in rows[m]])
                rows[f"dusage_{m}"] = np.where(rows["mover"], exp - rows[f"share_{m}"], 0.0)
                lvl = f"team_level_{m}"
                if lvl in rows and lvl in target and float(target[lvl]) > 0:
                    now = rows[lvl].to_numpy(float)
                    ch = np.where(now > 0, float(target[lvl]) / np.maximum(now, 1e-9) - 1, 0.0)
                    rows[f"dstyle_{m}"] = np.where(rows["mover"], np.clip(ch, -0.6, 1.5), 0.0)
                else:
                    rows[f"dstyle_{m}"] = 0.0
            else:
                rows[f"dusage_{m}"] = 0.0
                rows[f"dstyle_{m}"] = 0.0
            rows[f"y_{m}"] = np.nan
        rows["minutes_next"] = np.nan
        return rows.reset_index(drop=True)

    # ── player facts beyond the last Understat season ─────────
    def _profile_col(self, us_ids: pd.Series, col: str) -> pd.Series:
        prof = self.profiles
        if prof is None or prof.empty or col not in prof:
            return pd.Series([None] * len(us_ids), index=us_ids.index, dtype=object)
        return us_ids.map(prof[col])

    def position_share(self, us_ids: pd.Series, allowed: Iterable[str]) -> pd.Series:
        """Share of each player's starts over the last two seasons at the allowed Transfermarkt
        positions; NaN when there is no lineup data for him."""
        allowed = set(allowed)
        return self._profile_col(us_ids, "position_shares").map(
            lambda d: sum(v for k, v in d.items() if k in allowed) if isinstance(d, dict) else np.nan).astype(float)

    def recent_position(self, rows: pd.DataFrame) -> pd.Series:
        """Where he started most often over the last two seasons, else his Transfermarkt main position."""
        shares = self._profile_col(rows["us_id"], "position_shares")
        main = rows["tm_sub_position"] if "tm_sub_position" in rows else pd.Series(None, index=rows.index)
        return pd.Series([max(d, key=d.get) if isinstance(d, dict) and d else m for d, m in zip(shares, main)],
                         index=rows.index)

    def league_minutes(self, us_ids: pd.Series, leagues: Iterable[str]) -> pd.Series:
        """Most minutes he has played in any one of `leagues`, over every season in the data."""
        leagues = list(leagues)
        tot = self.ps[self.ps["league"].isin(leagues)].groupby(["us_id", "league"])["minutes"].sum()
        best = tot.groupby(level=0).max() if len(tot) else pd.Series(dtype=float)
        return us_ids.map(best).fillna(0.0)

    def recent_penalties(self, us_ids: pd.Series, seasons: int = 2) -> pd.Series:
        """Penalties taken over his last `seasons` seasons in the data (NaN if the build has no penalty data)."""
        if "pens" not in self.ps or self.ps["pens"].isna().all():
            return pd.Series(np.nan, index=us_ids.index)
        recent = self.ps[self.ps["season"] > self.latest_season - seasons]
        return us_ids.map(recent.groupby("us_id")["pens"].sum()).fillna(0.0)

    def annotate(self, out: pd.DataFrame) -> pd.DataFrame:
        """Add current club and league (Transfermarkt, newer than Understat), the recent position, and
        penalties taken. Players without a profile keep their last Understat club."""
        out = out.copy()
        club = self._profile_col(out["us_id"], "current_club")
        out["current_club"] = club.where(club.notna(), out["team"])
        lg = self._profile_col(out["us_id"], "current_league")
        out["current_league"] = lg.where(lg.notna(), out["league"])
        out["moved"] = self._profile_col(out["us_id"], "moved").fillna(False).astype(bool)
        out["recent_position"] = self.recent_position(out)
        out["pens"] = self.recent_penalties(out["us_id"]).to_numpy()
        return out

    # ── defensive profile ─────────────────────────────────────
    def defensive_profile(self) -> pd.DataFrame:
        """Recent observed defensive rates for every latest-season player (`def_*` columns,
        `def_minutes_used` = 0 when uncovered). Empty frame for a model built before the
        defensive data existed."""
        cached = self.__dict__.get("_def_profile")
        if cached is not None:
            return cached
        from forecast.config import BALL_RATES
        from forecast.data.defensive import recent_rates
        latest = self.in_latest_season(self.current())
        prof = recent_rates(self.ps, latest, seasons_back=3) if "def_minutes" in self.ps else pd.DataFrame(
            {"us_id": latest["us_id"]})
        if "ball_minutes" in self.ps:   # passing and take-ons: last two seasons with FBref data, own minutes
            ball = recent_rates(self.ps, latest, seasons_back=2, minutes_col="ball_minutes",
                                cols=list(BALL_RATES), used_col="ball_minutes_used")
            prof = prof.drop(columns=[f"def_{c}" for c in BALL_RATES if f"def_{c}" in prof]).merge(
                ball, on="us_id", how="left")
        self.__dict__["_def_profile"] = prof
        return prof

    def with_defensive(self, df: pd.DataFrame) -> pd.DataFrame:
        prof = self.defensive_profile()
        extra = [c for c in prof.columns if c != "us_id" and c not in df]
        return df.merge(prof[["us_id", *extra]], on="us_id", how="left") if extra else df

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_def_profile", None)
        return state

    # ── retrieval ─────────────────────────────────────────────
    def _style_vectors(self, rows: pd.DataFrame) -> np.ndarray:
        ref = self.in_latest_season(self.ps)
        scales, centres = metric_scales(ref), group_centres(ref)
        return np.vstack([style_vector({m: r[f"{m}_p90"] for m in METRICS}, scales, centres.get(r["group"]))
                          for _, r in rows.iterrows()])

    def _retrieve(self, cur: pd.DataFrame, role: RoleProfile, k: int, target: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Nearest styles to the role (or to `target`, e.g. a named player's style vector): ChromaDB
        when enabled and available, numpy otherwise. Filters that ChromaDB cannot see (age, value,
        exclusions) are already applied to `cur`."""
        tv = target_vector(role) if target is None else target
        if settings.ENABLE_VECTOR_INDEX:
            try:
                from forecast.index import StyleIndex
                everyone = self.current()
                everyone = self.in_latest_season(everyone)
                index = StyleIndex(settings.cache_path / "chroma", self.latest_season)
                index.sync(everyone, self._style_vectors(everyone))
                hits = index.query(tv, n=len(everyone), groups=role.groups)
                cos = {h["us_id"]: h["similarity"] / 100 for h in hits}      # cosine distance -> cosine
                pool = cur[cur["us_id"].isin(cos)].assign(style_cos=lambda d: d["us_id"].map(cos))
                return pool.nlargest(k, "style_cos").assign(style_similarity=lambda d: style_score(d["style_cos"]))
            except Exception as e:  # noqa: BLE001 - retrieval must not fail the run
                logger.warning("ChromaDB unavailable, using in-memory search: %s", e)
        cos = self._style_vectors(cur) @ tv
        return cur.assign(style_cos=cos).nlargest(k, "style_cos").assign(
            style_similarity=lambda d: style_score(d["style_cos"]))

    # ── the two questions ─────────────────────────────────────
    def project(self, players: pd.DataFrame, to_team: Optional[str] = None, age: Optional[float] = None,
                minutes: Optional[float] = None) -> pd.DataFrame:
        """`age` and `minutes` override the player's age and his expected minutes next season
        (what-if inputs for the command line; the agent leaves them alone)."""
        rows = self._pair_rows(players, to_team)
        if age is not None:
            rows["age"] = float(age)
        hist = history_features(self.ps, rows)
        for m in METRICS:
            rows[f"trend_{m}"] = hist[f"trend_{m}"].to_numpy()
        mins = self.minutes.predict(rows, self.ps) if minutes is None else np.full(len(rows), float(minutes))
        pred = self.model.predict(rows, hist, minutes=np.maximum(mins, 900))
        cols = ["us_id", "name", "team", "league", "group", "age", "minutes", "team_next", "league_next"]
        out = pd.concat([rows[cols + [c for c in ("tm_sub_position",) if c in rows]], pred], axis=1)
        out["expected_minutes"] = mins.round()
        out["market_value_eur"] = out["us_id"].map(self.values)
        return self.annotate(out)

    def candidates(self, role_key: Union[str, RoleProfile], n: int = 25, to_team: Optional[str] = None,
                   max_age: Optional[float] = None, max_value_eur: Optional[float] = None, min_minutes: int = 600,
                   leagues: Optional[Iterable[str]] = None, exclude: Iterable[str] = (),
                   min_age: Optional[float] = None, exclude_ids: Iterable[str] = (),
                   target: Optional[np.ndarray] = None, sub_positions: Optional[Iterable[str]] = None,
                   exclude_teams: Iterable[str] = (), min_position_share: float = 0.5,
                   also_positions: Optional[Iterable[str]] = None, min_also_share: float = 0.1,
                   experience: Optional[Tuple[Iterable[str], int]] = None,
                   penalty_taker: bool = False, style_match: bool = False,
                   min_pass_pct: Optional[float] = None) -> pd.DataFrame:
        """Closest players to the role's style (or to `target`), projected to `to_team`, ranked by
        role score. `role_key` is a key of ROLES or a custom RoleProfile (see forecast.brief).
        Adds `value_gap`: how cheap the player looks for his projected output.

        sub_positions       Transfermarkt positions wanted. A player passes if at least
                            `min_position_share` of his starts over the last two seasons were there
                            (lineup data), else if his Transfermarkt main position is one of them.
        also_positions      positions he must ALSO play: at least `min_also_share` of his recent starts
                            there, a quarter at `sub_positions`, and half across the two together.
                            Not checkable without lineup data (then only `sub_positions` applies).
        experience          (leagues, minutes): at least that many career minutes in one of the leagues.
        penalty_taker       took 2+ penalties over the last two seasons.
        style_match         `target` is a player's style ('like X', or the player being replaced): only
                            players whose style clearly leans his way (style score above 52) are kept, and
                            the ranking adds STYLE_WEIGHT x style (cosine) to the projected output score.
        min_pass_pct        pass completion (0-1) over his last two seasons with FBref passing data;
                            players without that data are left out.
        Leagues and clubs to exclude are matched on the current club (Transfermarkt) as well as the
        last Understat one; players who have retired or left the covered leagues are dropped."""
        role: RoleProfile = ROLES[role_key] if isinstance(role_key, str) else role_key
        cur = self.current()
        cur = self.in_latest_season(cur)
        cur = cur[cur["group"].isin(role.groups) & (cur["minutes"] >= min_minutes)]
        active = self._profile_col(cur["us_id"], "active")
        cur = cur[active.isna() | active.astype(bool)]
        now_club, now_league = self._profile_col(cur["us_id"], "current_club"), self._profile_col(cur["us_id"], "current_league")
        if leagues:
            cur = cur[now_league.where(now_league.notna(), cur["league"]).isin(list(leagues))]
            now_club = now_club.loc[cur.index]
        # hard limits: an unknown age or value (no Transfermarkt match) does not pass them
        if max_age is not None:
            cur = cur[cur["age"].notna() & (cur["age"] <= max_age)]
        if min_age is not None:
            cur = cur[cur["age"].notna() & (cur["age"] >= min_age)]
        teams = set(exclude_teams)
        cur = cur[~cur["us_id"].isin(set(exclude_ids)) & ~cur["team"].isin(teams)
                  & ~now_club.reindex(cur.index).isin(teams)]
        if sub_positions:
            # where he lined up over the last two seasons; Transfermarkt's main position if no lineups;
            # players with neither stay if their group fits (filtered above)
            main, also = set(sub_positions), set(also_positions or ())
            share = self.position_share(cur["us_id"], main | also)
            ok = share >= min_position_share
            if also:
                ok &= ((self.position_share(cur["us_id"], main) >= 0.25)
                       & (self.position_share(cur["us_id"], also) >= min_also_share))
            sub = cur["tm_sub_position"] if "tm_sub_position" in cur else pd.Series(None, index=cur.index)
            by_main = sub.isna() | sub.isin(main)
            cur = cur[np.where(share.notna(), ok, by_main)]
        if experience:
            exp_leagues, exp_minutes = experience
            cur = cur[self.league_minutes(cur["us_id"], exp_leagues) >= exp_minutes]
        if penalty_taker:
            pens = self.recent_penalties(cur["us_id"])
            if pens.isna().all():
                raise ValueError("This saved model has no penalty data. Rebuild it (python -m forecast.build) "
                                 "to search for penalty takers.")
            cur = cur[pens >= 2]
        excl = {e.lower() for e in exclude}
        cur = cur[~cur["name"].str.lower().isin(excl)]
        if max_value_eur is not None:
            vals = cur["us_id"].map(self.values)
            cur = cur[vals.notna() & (vals <= max_value_eur)]
        if min_pass_pct is not None:
            prof = self.defensive_profile()
            if "def_pass_pct" not in prof:
                raise ValueError("This saved model has no passing data. Rebuild it (python -m forecast.build) "
                                 "to filter on pass completion.")
            known = prof[prof.get("ball_minutes_used", pd.Series(1.0, index=prof.index)) > 0]
            ok = set(known.loc[known["def_pass_pct"] >= min_pass_pct, "us_id"])
            cur = cur[cur["us_id"].isin(ok)]
        if cur.empty:
            return pd.DataFrame()

        # Rank the whole filtered pool (up to 800 closest in style): cutting to the few closest in
        # style first dropped players who fit better overall (a top-10 list must be the top 10 by fit,
        # the same as the 10 first of a longer list).
        pool = self._retrieve(cur, role, min(len(cur), 800), target)
        if style_match and target is not None:
            pool = pool[pool["style_cos"] > 0.05]       # must clearly lean his way (style above 52)
        if pool.empty:
            return pd.DataFrame()

        proj = self.with_defensive(self.project(pool.drop(columns=["style_similarity", "style_cos"]), to_team))
        reference = self.with_defensive(self.in_latest_season(self.ps))
        proj["role_score"] = np.round(role_scores(proj, role, reference), 2)
        proj["style_similarity"] = pool["style_similarity"].to_numpy()
        cos = pool["style_cos"].to_numpy(float)
        proj["fit"] = proj["role_score"] + (STYLE_WEIGHT * cos if style_match and target is not None else 0.0)
        if target is None and not any(m in METRICS for m in role.weights):
            proj["style_similarity"] = np.nan      # a purely defensive profile has no attacking style to match
        proj["value_gap"] = np.round(value_gap(proj["role_score"].to_numpy(float),
                                               proj["market_value_eur"].to_numpy(float)), 2)
        if sub_positions:
            proj["position_share"] = self.position_share(proj["us_id"], set(sub_positions) | set(also_positions or ())).to_numpy()
        if also_positions:
            proj["also_share"] = self.position_share(proj["us_id"], set(also_positions)).to_numpy()
        if experience:
            proj["experience_minutes"] = self.league_minutes(proj["us_id"], experience[0]).to_numpy()
        return proj.sort_values("fit", ascending=False).head(n).reset_index(drop=True)
