"""
Backtest
========
Rolling-origin evaluation on real past seasons. For each test season S, every
model is fitted only on pairs whose target season is before S, then asked to
predict the pairs whose target season is S. Nothing from S leaks into the fit.

The ladder adds one term at a time, so the report shows what each idea is
worth, against three baselines:
  last_season  next season = this season (per 90)
  handset      this season scaled by the old hand-set league coefficients
  ability      shrunk, recency-weighted history only (no context terms)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.league_normalizer import get_coefficient
from forecast.config import METRICS
from forecast.model import (ALL_TERMS, MinutesModel, ProjectionModel, Z80, pair_weights, prepare,
                            weighted_rmse)

LADDER: List[tuple] = [
    ("ability", ()),
    ("+league", ("league",)),
    ("+team", ("league", "team")),
    ("+usage", ("league", "team", "usage")),
    ("+age", ("league", "team", "usage", "age")),
    ("+trend", ALL_TERMS),
]


def baseline_last_season(pairs: pd.DataFrame, m: str) -> np.ndarray:
    return pairs[f"{m}_p90"].to_numpy(float)


def baseline_handset(pairs: pd.DataFrame, m: str) -> np.ndarray:
    """Old AI Scout: scale by coefficient(to) / coefficient(from)."""
    ratio = np.array([get_coefficient(to) / get_coefficient(frm) for frm, to in zip(pairs["league"], pairs["league_next"])])
    return pairs[f"{m}_p90"].to_numpy(float) * ratio


def run(ps: pd.DataFrame, pairs: pd.DataFrame, test_seasons: Sequence[int],
        metrics: Sequence[str] = METRICS, min_train_pairs: int = 200) -> Dict[str, pd.DataFrame]:
    """Returns {"scores": per-model x metric x subset RMSE, "coverage": 80% interval coverage,
    "minutes": minutes-model RMSE vs 'same minutes', "predictions": the full model's out-of-sample predictions}."""
    pairs, hist = prepare(ps, pairs)
    preds: Dict[str, Dict[str, List[np.ndarray]]] = {}
    truth, weights, subsets, cover = [], [], {"all": [], "movers": [], "league_change": []}, []
    minutes_rows, oos = [], []

    for S in test_seasons:
        train = (pairs["target_season"] < S).to_numpy()
        test = (pairs["target_season"] == S).to_numpy()
        if train.sum() < min_train_pairs or test.sum() == 0:
            continue
        tr, te = pairs[train], pairs[test]
        htr, hte = hist[train], hist[test]

        truth.append(te[[f"y_{m}" for m in metrics]].to_numpy(float))
        weights.append(pair_weights(te))
        subsets["all"].append(np.ones(len(te), bool))
        subsets["movers"].append(te["mover"].to_numpy(bool))
        subsets["league_change"].append(te["league_change"].to_numpy(bool))

        def add(name, matrix):
            preds.setdefault(name, []).append(matrix)

        add("last_season", np.column_stack([baseline_last_season(te, m) for m in metrics]))
        add("handset", np.column_stack([baseline_handset(te, m) for m in metrics]))
        for name, terms in LADDER:
            model = ProjectionModel(terms=terms, metrics=metrics).fit(tr, htr)
            p = model.predict(te, hte)
            add(name, p[[f"pred_{m}" for m in metrics]].to_numpy(float))
            if terms == ALL_TERMS:
                y = te[[f"y_{m}" for m in metrics]].to_numpy(float)
                lo = p[[f"lo_{m}" for m in metrics]].to_numpy(float)
                hi = p[[f"hi_{m}" for m in metrics]].to_numpy(float)
                cover.append((y >= lo) & (y <= hi))
                oos.append(pd.concat([te[["us_id", "name", "season", "team", "team_next", "league", "league_next"]], p], axis=1))

        mm = MinutesModel().fit(tr, ps)
        mins_hat = mm.predict(te, ps)
        y_min = te["minutes_next"].to_numpy(float)
        minutes_rows.append({"season": S, "n": len(te),
                             "same_minutes_rmse": float(np.sqrt(np.mean((y_min - te["minutes"].to_numpy(float)) ** 2))),
                             "model_rmse": float(np.sqrt(np.mean((y_min - mins_hat) ** 2)))})

    if not truth:
        raise ValueError("Not enough seasons to backtest: need training pairs before the first test season.")

    Y, W = np.vstack(truth), np.concatenate(weights)
    masks = {k: np.concatenate(v) for k, v in subsets.items()}
    rows = []
    for name, chunks in preds.items():
        P = np.vstack(chunks)
        for j, m in enumerate(metrics):
            for subset, mask in masks.items():
                if mask.sum() == 0:
                    continue
                rows.append({"model": name, "metric": m, "subset": subset, "n": int(mask.sum()),
                             "rmse": weighted_rmse(Y[mask, j], P[mask, j], W[mask])})
    scores = pd.DataFrame(rows)
    C = np.vstack(cover) if cover else np.zeros((0, len(metrics)), bool)
    coverage = pd.DataFrame({"metric": list(metrics), "coverage_80": C.mean(0) if len(C) else np.nan})
    return {"scores": scores, "coverage": coverage, "minutes": pd.DataFrame(minutes_rows),
            "predictions": pd.concat(oos) if oos else pd.DataFrame()}


def ladder_table(scores: pd.DataFrame, subset: str = "all") -> pd.DataFrame:
    """Model x metric RMSE, plus the mean improvement over 'last_season' in percent."""
    s = scores[scores["subset"] == subset].pivot(index="model", columns="metric", values="rmse")
    order = ["last_season", "handset"] + [n for n, _ in LADDER]
    s = s.reindex([o for o in order if o in s.index])
    base = s.loc["last_season"] if "last_season" in s.index else None
    if base is not None:
        s["improvement_vs_last_season_%"] = ((1 - s.div(base, axis=1)).mean(axis=1) * 100).round(1)
    return s


def to_markdown(table: pd.DataFrame, digits: int = 4) -> str:
    cols = ["model", *table.columns]
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, row in table.iterrows():
        vals = [f"{v:.{digits}f}" if isinstance(v, float) and not v.is_integer() else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx), *vals]) + " |")
    return "\n".join(lines)
