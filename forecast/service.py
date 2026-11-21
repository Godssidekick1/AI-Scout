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
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from config import settings
from forecast.config import METRICS
from forecast.model import MinutesModel, ProjectionModel, history_features
from forecast.roles import ROLES, RoleProfile, metric_scales, role_scores, style_vector, target_vector, value_gap
from forecast.table import expected_share, team_totals

logger = logging.getLogger(__name__)


_FOLD = str.maketrans({"ø": "o", "æ": "ae", "ß": "ss", "ı": "i", "ł": "l", "đ": "d", "-": " ", "'": ""})


def _plain(text: str) -> str:
    """Lower case, accents removed: 'Ødegaard' -> 'odegaard'."""
    import unicodedata
    t = unicodedata.normalize("NFKD", str(text).lower().translate(_FOLD))
    return "".join(c for c in t if not unicodedata.combining(c)).strip()


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
            else:
                rows[f"dusage_{m}"] = 0.0
            rows[f"y_{m}"] = np.nan
        rows["minutes_next"] = np.nan
        return rows.reset_index(drop=True)

    # ── retrieval ─────────────────────────────────────────────
    def _style_vectors(self, rows: pd.DataFrame) -> np.ndarray:
        scales = metric_scales(self.in_latest_season(self.ps))
        return np.vstack([style_vector({m: r[f"{m}_p90"] for m in METRICS}, scales) for _, r in rows.iterrows()])

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
                sim = {h["us_id"]: h["similarity"] for h in hits}
                pool = cur[cur["us_id"].isin(sim)].assign(style_similarity=lambda d: d["us_id"].map(sim))
                return pool.nlargest(k, "style_similarity")
            except Exception as e:  # noqa: BLE001 - retrieval must not fail the run
                logger.warning("ChromaDB unavailable, using in-memory search: %s", e)
        sim = self._style_vectors(cur) @ tv
        return cur.assign(style_similarity=np.round(sim * 100, 1)).nlargest(k, "style_similarity")

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
        return out

    def candidates(self, role_key: Union[str, RoleProfile], n: int = 25, to_team: Optional[str] = None,
                   max_age: Optional[float] = None, max_value_eur: Optional[float] = None, min_minutes: int = 600,
                   leagues: Optional[Iterable[str]] = None, exclude: Iterable[str] = (),
                   min_age: Optional[float] = None, exclude_ids: Iterable[str] = (),
                   target: Optional[np.ndarray] = None, sub_positions: Optional[Iterable[str]] = None,
                   exclude_teams: Iterable[str] = ()) -> pd.DataFrame:
        """Closest players to the role's style (or to `target`), projected to `to_team`, ranked by
        role score. `role_key` is a key of ROLES or a custom RoleProfile (see forecast.brief).
        Adds `value_gap`: how cheap the player looks for his projected output."""
        role: RoleProfile = ROLES[role_key] if isinstance(role_key, str) else role_key
        cur = self.current()
        cur = self.in_latest_season(cur)
        cur = cur[cur["group"].isin(role.groups) & (cur["minutes"] >= min_minutes)]
        if leagues:
            cur = cur[cur["league"].isin(list(leagues))]
        if max_age is not None:
            cur = cur[cur["age"].isna() | (cur["age"] <= max_age)]
        if min_age is not None:
            cur = cur[cur["age"].isna() | (cur["age"] >= min_age)]
        cur = cur[~cur["us_id"].isin(set(exclude_ids)) & ~cur["team"].isin(set(exclude_teams))]
        if sub_positions:
            # Transfermarkt main position; players without one stay if their group fits (filtered above)
            sub = cur["tm_sub_position"] if "tm_sub_position" in cur else pd.Series(None, index=cur.index)
            cur = cur[sub.isna() | sub.isin(set(sub_positions))]
        excl = {e.lower() for e in exclude}
        cur = cur[~cur["name"].str.lower().isin(excl)]
        if max_value_eur is not None:
            vals = cur["us_id"].map(self.values)
            cur = cur[vals.isna() | (vals <= max_value_eur)]
        if cur.empty:
            return pd.DataFrame()

        pool = self._retrieve(cur, role, max(n * 2, 40), target)
        if pool.empty:
            return pd.DataFrame()

        proj = self.project(pool.drop(columns=["style_similarity"]), to_team)
        reference = self.in_latest_season(self.ps)
        proj["role_score"] = np.round(role_scores(proj, role, reference), 2)
        proj["style_similarity"] = pool["style_similarity"].to_numpy()
        proj["value_gap"] = np.round(value_gap(proj["role_score"].to_numpy(float),
                                               proj["market_value_eur"].to_numpy(float)), 2)
        return proj.sort_values("role_score", ascending=False).head(n).reset_index(drop=True)
