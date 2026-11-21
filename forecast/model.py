"""
Projection model
================
For player i, metric m (per 90), next season:

    y_hat = theta * (1 + lambda[L_next - L_now, m, g]    league tax, per metric and position group g
                     + beta_m  * dElo / 100             change in team strength
                     + gamma_m * dUsage                 expected change in his share of the team's output
                     + a_m * age_step(g, age)           position age curve (delta method)
                     + d_m * trend / theta)             his own recent trajectory (late bloomers)

theta is his ability: a recency-weighted average of his past per-90 rates,
shrunk toward his position group's mean (empirical Bayes, strength k fitted per
metric). The context terms are relative, because per-90 rates scale: a league
tax reads as "moving from Ligue 1 to the Premier League changes a striker's
npxG/90 by -X%". They are fitted together by weighted ridge regression on
(y_next - theta) / theta, weighted by theta^2 so it is the same fit as on the
per-90 error, and each coefficient is readable on its own. League effects are partially pooled: a shared effect
per league plus a smaller, more heavily penalised deviation per position group.

`terms` switches terms on and off, which is how the backtest builds its ladder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from forecast.config import LEAGUES, METRICS, POSITION_GROUPS, REFERENCE_LEAGUE

ALL_TERMS: Tuple[str, ...] = ("league", "team", "usage", "age", "trend")
DECAY = 0.6                        # weight of a season relative to the one after it
K_GRID = (150, 300, 600, 900, 1350, 2000, 3000, 4500)   # shrinkage strength, in minutes
Z80 = 1.2816                       # half-width of an 80% normal interval, in standard deviations
AGE_RANGE = (17, 37)
THETA_FLOOR = 0.25
TERM_OF = {"league": "league", "team_elo_per_100": "team", "usage_share_change": "usage",
           "age_curve": "age", "personal_trend": "trend", "intercept": "drift"}                 # relative terms use max(theta, 25% of the group mean) as the base


# ── history features (computed once per pair, reused for every k) ──────────

def history_features(ps: pd.DataFrame, pairs: pd.DataFrame, decay: float = DECAY) -> pd.DataFrame:
    """For each pair (player, season t): recency-weighted sums over seasons <= t, and
    a minutes-weighted trend over the last three seasons, per metric."""
    by_player = {pid: g.sort_values("season") for pid, g in ps.groupby("us_id")}
    out = {f"sw": [], **{f"swx_{m}": [] for m in METRICS}, **{f"trend_{m}": [] for m in METRICS}}
    for pid, t in zip(pairs["us_id"], pairs["season"]):
        hist = by_player[pid]
        hist = hist[hist["season"] <= t]
        w = hist["minutes"].to_numpy(float) * decay ** (t - hist["season"].to_numpy(float))
        out["sw"].append(w.sum())
        recent = hist[(hist["season"] >= t - 2) & (hist["minutes"] >= 450)]
        for m in METRICS:
            x = hist[f"{m}_p90"].to_numpy(float)
            out[f"swx_{m}"].append(float((w * x).sum()))
            out[f"trend_{m}"].append(_weighted_slope(recent["season"].to_numpy(float),
                                                     recent[f"{m}_p90"].to_numpy(float),
                                                     recent["minutes"].to_numpy(float)))
    return pd.DataFrame(out, index=pairs.index)


def _weighted_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    var = np.average((x - xm) ** 2, weights=w)
    return float(np.average((x - xm) * (y - ym), weights=w) / var) if var > 0 else 0.0


def pair_weights(pairs: pd.DataFrame) -> np.ndarray:
    """A pair is as reliable as its noisier side: harmonic mean of the two seasons' minutes."""
    a, b = pairs["minutes"].to_numpy(float), pairs["minutes_next"].to_numpy(float)
    return 2 * a * b / np.maximum(a + b, 1)


def weighted_rmse(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    return float(math.sqrt(np.sum(w * (y - yhat) ** 2) / max(np.sum(w), 1e-9)))


# ── fitted pieces ──────────────────────────────────────────────

@dataclass
class MetricFit:
    metric: str
    k: float                                   # shrinkage strength (minutes)
    prior: Dict[str, float]                    # group -> mean per 90
    age_step: Dict[str, Dict[int, float]]      # group -> {age: expected yearly change, centred}
    columns: List[str] = field(default_factory=list)
    coef: Optional[np.ndarray] = None
    resid_a: float = 0.0                       # residual variance = a + b * 900 / minutes
    resid_b: float = 0.0

    def coefficients(self) -> Dict[str, float]:
        return dict(zip(self.columns, (self.coef if self.coef is not None else [])))


class ProjectionModel:
    def __init__(self, terms: Iterable[str] = ALL_TERMS, ridge: float = 2.0, deviation_ridge: float = 20.0,
                 metrics: Sequence[str] = METRICS):
        self.terms = tuple(t for t in ALL_TERMS if t in set(terms))
        self.ridge = ridge
        self.deviation_ridge = deviation_ridge
        self.metrics = tuple(metrics)
        self.fits: Dict[str, MetricFit] = {}

    # ── public ────────────────────────────────────────────────
    def fit(self, pairs: pd.DataFrame, hist: pd.DataFrame) -> "ProjectionModel":
        w = pair_weights(pairs)
        for m in self.metrics:
            prior = self._group_prior(pairs, m, w)
            mu = pairs["group"].map(prior).to_numpy(float)
            k = self._fit_k(pairs, hist, m, mu, w)
            theta = self._theta(hist, m, mu, k)
            age_step = self._age_curve(pairs, m, w) if "age" in self.terms else {}
            fit = MetricFit(metric=m, k=k, prior=prior, age_step=age_step)
            X, cols, pen = self._design(pairs, m, fit)
            y = pairs[f"y_{m}"].to_numpy(float)
            resid = y - theta
            if X.shape[1]:
                base = np.maximum(theta, THETA_FLOOR * max(np.mean(list(prior.values())), 1e-6))
                fit.coef = _ridge(X, resid / base, w * base ** 2, pen)
                fit.columns = cols
                resid = y - theta * (1 + X @ fit.coef)
            fit.resid_a, fit.resid_b = _residual_variance(resid, pairs["minutes_next"].to_numpy(float), w)
            self.fits[m] = fit
        return self

    def predict(self, pairs: pd.DataFrame, hist: pd.DataFrame, minutes: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Per-90 projection for every pair and metric, with an 80% interval.
        `minutes` is the expected minutes next season (defaults to the actual, for backtests)."""
        mins = pairs["minutes_next"].to_numpy(float) if minutes is None else np.asarray(minutes, float)
        out = pd.DataFrame(index=pairs.index)
        for m, fit in self.fits.items():
            mu = pairs["group"].map(fit.prior).fillna(np.mean(list(fit.prior.values()))).to_numpy(float)
            yhat = self._theta(hist, m, mu, fit.k)
            if fit.coef is not None:
                X, _, _ = self._design(pairs, m, fit)
                yhat = yhat * (1 + X @ fit.coef)
            yhat = np.maximum(yhat, 0.0)                      # per-90 rates cannot be negative
            sd = np.sqrt(np.maximum(fit.resid_a + fit.resid_b * 900 / np.maximum(mins, 90), 1e-12))
            out[f"pred_{m}"] = yhat
            out[f"lo_{m}"] = np.maximum(yhat - Z80 * sd, 0.0)
            out[f"hi_{m}"] = yhat + Z80 * sd
        return out

    def contributions(self, pairs: pd.DataFrame, hist: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Per metric: ability (theta) and each term's relative effect for every row, so a
        projection can be read as theta * (1 + league + team + usage + age + trend + drift)."""
        out: Dict[str, pd.DataFrame] = {}
        for m, fit in self.fits.items():
            mu = pairs["group"].map(fit.prior).fillna(np.mean(list(fit.prior.values()))).to_numpy(float)
            df = pd.DataFrame({"ability": self._theta(hist, m, mu, fit.k)}, index=pairs.index)
            if fit.coef is not None:
                X, cols, _ = self._design(pairs, m, fit)
                for col, effect in zip(cols, (X * fit.coef).T):
                    term = TERM_OF.get(col.split("[")[0], col)
                    df[term] = df.get(term, 0.0) + effect
            out[m] = df
        return out

    def explain(self, metric: str) -> Dict[str, float]:
        """Readable coefficients: league effects (per 90, vs the reference league), team, usage, age, trend."""
        fit = self.fits[metric]
        return {"shrinkage_k_minutes": fit.k, **fit.coefficients()}

    # ── pieces ────────────────────────────────────────────────
    @staticmethod
    def _group_prior(pairs: pd.DataFrame, m: str, w: np.ndarray) -> Dict[str, float]:
        df = pd.DataFrame({"g": pairs["group"], "x": pairs[f"{m}_p90"], "w": pairs["minutes"]})
        prior = {g: float(np.average(d["x"], weights=d["w"])) for g, d in df.groupby("g") if d["w"].sum() > 0}
        overall = float(np.average(df["x"], weights=df["w"])) if df["w"].sum() > 0 else 0.0
        return {g: prior.get(g, overall) for g in POSITION_GROUPS}

    @staticmethod
    def _theta(hist: pd.DataFrame, m: str, mu: np.ndarray, k: float) -> np.ndarray:
        sw = hist.loc[:, "sw"].to_numpy(float)
        swx = hist.loc[:, f"swx_{m}"].to_numpy(float)
        return (swx + k * mu) / (sw + k)

    def _fit_k(self, pairs, hist, m, mu, w) -> float:
        y = pairs[f"y_{m}"].to_numpy(float)
        scores = {k: weighted_rmse(y, self._theta(hist, m, mu, k), w) for k in K_GRID}
        return float(min(scores, key=scores.get))

    @staticmethod
    def _age_curve(pairs: pd.DataFrame, m: str, w: np.ndarray) -> Dict[str, Dict[int, float]]:
        """Delta method: the same player's change from one season to the next, as a share of
        his group's typical level, averaged by age within each position group (weighted by
        minutes) and smoothed with a quadratic. One constant, shared by all groups, is then
        subtracted so the curves carry age shape and group differences (centre-backs peaking
        later than wingers) but not the overall regression to the mean that theta handles."""
        df = pd.DataFrame({"g": pairs["group"], "age": pairs["age"],
                           "d": pairs[f"y_{m}"] - pairs[f"{m}_p90"], "lvl": pairs[f"{m}_p90"], "w": w}).dropna()
        ages = np.arange(AGE_RANGE[0], AGE_RANGE[1] + 1)
        raw: Dict[str, np.ndarray] = {}
        fitted_at_obs, obs_w = [], []
        for g in POSITION_GROUPS:
            d = df[df["g"] == g]
            if len(d) < 30:
                continue
            level = float(np.average(d["lvl"], weights=d["w"])) or 1.0
            a = d["age"].clip(*AGE_RANGE).to_numpy(float)
            coef = np.polyfit(a, d["d"].to_numpy(float) / level, 2, w=np.sqrt(d["w"].to_numpy(float)))
            raw[g] = np.polyval(coef, ages)
            fitted_at_obs.append(np.polyval(coef, a)); obs_w.append(d["w"].to_numpy(float))
        centre = float(np.average(np.concatenate(fitted_at_obs), weights=np.concatenate(obs_w))) if raw else 0.0
        return {g: ({int(x): float(v - centre) for x, v in zip(ages, raw[g])} if g in raw
                    else {int(x): 0.0 for x in ages}) for g in POSITION_GROUPS}

    def _design(self, pairs: pd.DataFrame, m: str, fit: MetricFit):
        cols, blocks, pens = [], [], []
        n = len(pairs)
        if "league" in self.terms:
            others = [lg for lg in LEAGUES if lg != REFERENCE_LEAGUE]
            frm, to = pairs["league"].to_numpy(), pairs["league_next"].to_numpy()
            grp = pairs["group"].to_numpy()
            for lg in others:
                shared = (to == lg).astype(float) - (frm == lg).astype(float)
                cols.append(f"league[{lg}]"); blocks.append(shared); pens.append(self.ridge)
                for g in POSITION_GROUPS:
                    cols.append(f"league[{lg}]|{g}"); blocks.append(shared * (grp == g)); pens.append(self.deviation_ridge)
        if "team" in self.terms:
            cols.append("team_elo_per_100"); pens.append(self.ridge)
            blocks.append((pairs["elo_next"].to_numpy(float) - pairs["elo"].to_numpy(float)) / 100)
        if "usage" in self.terms:
            cols.append("usage_share_change"); pens.append(self.ridge)
            blocks.append(pairs[f"dusage_{m}"].to_numpy(float))
        if "age" in self.terms:
            steps = [fit.age_step.get(g, {}).get(int(min(max(round(a), AGE_RANGE[0]), AGE_RANGE[1])), 0.0)
                     if not (isinstance(a, float) and math.isnan(a)) else 0.0
                     for g, a in zip(pairs["group"], pairs["age"])]
            cols.append("age_curve"); blocks.append(np.array(steps, float)); pens.append(self.ridge)
        if "trend" in self.terms:
            # hist is aligned to pairs; the trend column is attached by the caller (see backtest/fit_all)
            level = max(np.mean(list(fit.prior.values())), 1e-6)
            cols.append("personal_trend"); pens.append(self.ridge * 5)
            blocks.append(pairs[f"trend_{m}"].to_numpy(float) / np.maximum(pairs[f"{m}_p90"].to_numpy(float), THETA_FLOOR * level))
        if not blocks:
            return np.zeros((n, 0)), [], np.zeros(0)
        # intercept: average drift (e.g. league-wide scoring changes), lightly penalised
        cols.append("intercept"); blocks.append(np.ones(n)); pens.append(1e-6)
        return np.column_stack(blocks), cols, np.array(pens, float)


def _ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, penalties: np.ndarray) -> np.ndarray:
    """Weighted ridge: minimise sum w (y - Xb)^2 / mean(w) + sum p_j b_j^2."""
    wn = w / max(w.mean(), 1e-9)
    A = X.T @ (X * wn[:, None]) + np.diag(penalties)
    return np.linalg.solve(A, X.T @ (wn * y))


def _residual_variance(resid: np.ndarray, minutes: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Fit resid^2 ~ a + b * 900/minutes: short seasons are noisier."""
    z = 900 / np.maximum(minutes, 90)
    X = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(X, resid ** 2, rcond=None)
    a, b = float(max(coef[0], 0.0)), float(max(coef[1], 0.0))
    if a + b <= 0:
        a = float(np.mean(resid ** 2))
    return a, b


def prepare(ps: pd.DataFrame, pairs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute history features once and attach the trend columns the design matrix uses."""
    hist = history_features(ps, pairs)
    pairs = pairs.copy()
    for m in METRICS:
        pairs[f"trend_{m}"] = hist[f"trend_{m}"].to_numpy()
    return pairs, hist


# ── minutes ────────────────────────────────────────────────────

class MinutesModel:
    """Expected league minutes next season, from past minutes, age, a move and team change.
    Only players who appear next season are in the data, so this predicts minutes given
    he plays in the big five, not the chance he disappears from it."""

    FEATURES = ("minutes", "minutes_prev", "age", "age_sq", "mover", "delo")

    def __init__(self, ridge: float = 1.0):
        self.ridge = ridge
        self.coef: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None

    @staticmethod
    def _X(pairs: pd.DataFrame, ps: pd.DataFrame) -> np.ndarray:
        prev = ps.set_index(["us_id", "season"])["minutes"]
        keys = list(zip(pairs["us_id"], pairs["season"] - 1))
        minutes_prev = np.array([prev.get(k, 0.0) for k in keys], float)
        age = pairs["age"].fillna(pairs["age"].median() if pairs["age"].notna().any() else 25).to_numpy(float)
        return np.column_stack([pairs["minutes"].to_numpy(float), minutes_prev, age, (age - 26) ** 2,
                                pairs["mover"].to_numpy(float),
                                (pairs["elo_next"] - pairs["elo"]).to_numpy(float) / 100])

    def fit(self, pairs: pd.DataFrame, ps: pd.DataFrame) -> "MinutesModel":
        X = self._X(pairs, ps)
        self.mean, self.scale = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([(X - self.mean) / self.scale, np.ones(len(X))])
        pen = np.r_[np.full(X.shape[1], self.ridge), 1e-6]
        self.coef = _ridge(Z, pairs["minutes_next"].to_numpy(float), np.ones(len(X)), pen)
        return self

    def predict(self, pairs: pd.DataFrame, ps: pd.DataFrame) -> np.ndarray:
        X = self._X(pairs, ps)
        Z = np.column_stack([(X - self.mean) / self.scale, np.ones(len(X))])
        return np.clip(Z @ self.coef, 0, 38 * 90)
