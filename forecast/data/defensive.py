"""
Defensive and duel data
=======================
Understat has no defending, duels, aerials or progression. This module adds
them from free sources, one row per player per team per league-season, in one
schema (`COLUMNS`), then links each row to the Understat player (`us_id`).

Sources, best first (see forecast/data/*_stats.py for each):

  premier_league  premierleague.com stats API (Opta)        2008-09 onward
  serie_a         legaseriea.it stats API (Opta)              recent seasons
  all big five    FBref big-five tables (Opta), via the       2017-18 to 2023-24
                  worldfootballR_data GitHub release          (+ ~9 games of 2024-25)

For each league-season the best source that has it wins; the others are not
mixed in, so a player's numbers always come from one provider's definitions.

    python -m forecast.data.defensive            # fetch everything into the cache, print coverage
    python -m forecast.data.defensive --check    # one request per source

Every loader is cached on disk (past seasons never expire), rate-limited and
retrying, and an outage in one source only loses that source.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from thefuzz import fuzz

from config import settings
from forecast.config import DEF_METRICS, DEF_RATIOS, LEAGUES

logger = logging.getLogger(__name__)

# Season totals each source provides (NaN where a source lacks one).
TOTALS: Tuple[str, ...] = ("tackles", "tackles_won", "interceptions", "blocks", "clearances", "aerials_won",
                           "aerials_lost", "recoveries", "duels_won", "duels_lost", "prog_passes", "prog_carries")
COLUMNS = ["source", "league", "season", "src_id", "name", "team", "birth_year", "minutes", *TOTALS]
PRIOR_MINUTES = 900.0      # shrinkage weight toward the position mean when a player has little data


# ── polite HTTP with a disk cache ──────────────────────────────

class SourceUnavailable(RuntimeError):
    pass


class PoliteClient:
    """GET with a per-host minimum interval, retries with backoff on 429/5xx and network
    errors, and a JSON/bytes disk cache. `ttl_days=None` caches forever (past seasons)."""

    def __init__(self, name: str, min_interval: float = 0.5, headers: Optional[Dict[str, str]] = None,
                 retries: int = 4, timeout: float = 30.0):
        self.name = name
        self.min_interval = min_interval
        self.retries = retries
        self.headers = {"User-Agent": "ai-scout/3.0 (personal scouting research; cached, rate-limited)",
                        **(headers or {})}
        self.timeout = timeout
        self._last = 0.0
        self.dir = settings.cache_path / "defensive" / name
        self.requests = 0

    def _path(self, key: str, suffix: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:180]
        return self.dir / f"{safe}{suffix}"

    def _fresh(self, path: Path, ttl_days: Optional[float]) -> bool:
        if not path.exists():
            return False
        return ttl_days is None or (time.time() - path.stat().st_mtime) < ttl_days * 86400

    def _get(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> httpx.Response:
        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                self.requests += 1
                r = httpx.get(url, params=params, headers={**self.headers, **(headers or {})},
                              timeout=self.timeout, follow_redirects=True)
            except httpx.HTTPError as e:
                last_err = e
            else:
                if r.status_code < 400:
                    return r
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise SourceUnavailable(f"{self.name}: HTTP {r.status_code} for {url}")
                last_err = SourceUnavailable(f"{self.name}: HTTP {r.status_code}")
                retry_after = r.headers.get("retry-after", "")
                if retry_after.isdigit():
                    time.sleep(min(int(retry_after), 60))
            time.sleep(min(2 ** attempt, 30))
        raise SourceUnavailable(f"{self.name}: {url} failed after {self.retries + 1} tries ({last_err})")

    def json(self, key: str, url: str, params: Optional[Dict] = None, ttl_days: Optional[float] = None,
             headers: Optional[Dict] = None):
        path = self._path(key, ".json")
        if self._fresh(path, ttl_days):
            return json.loads(path.read_text(encoding="utf-8"))
        data = self._get(url, params, headers).json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def file(self, key: str, url: str, ttl_days: Optional[float] = None) -> Path:
        path = self._path(key, "")
        if self._fresh(path, ttl_days):
            return path
        r = self._get(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.replace(path)
        return path


def frame(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("minutes", *TOTALS):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── combining sources ──────────────────────────────────────────

def _sources() -> List[Tuple[str, Callable[[Iterable[int]], pd.DataFrame]]]:
    from forecast.data import fbref_stats, pl_stats, seriea_stats
    return [("premierleague.com", pl_stats.load), ("legaseriea.it", seriea_stats.load),
            ("fbref", fbref_stats.load)]


def usable(grp: pd.DataFrame) -> bool:
    """A league-season counts as covered by a source only if it has real defensive counts:
    at least ten matchdays' worth of minutes and tackles + interceptions for most players."""
    if grp["minutes"].sum() < 10 * 20 * 11 * 90 * 0.9 / 2:
        return False
    played = grp[grp["minutes"] > 0]
    return bool(len(played)) and float((played["tackles"].notna() & played["interceptions"].notna()).mean()) > 0.5


def load_all(seasons: Iterable[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(rows, coverage). Rows: one source per league-season, best first. Coverage: which source
    supplied each league-season (and which failed), for the build report."""
    seasons = list(seasons)
    taken: Dict[Tuple[str, int], str] = {}
    parts, notes = [], []
    for name, loader in _sources():
        try:
            df = loader(seasons)
        except Exception as e:  # noqa: BLE001 - one source down must not stop the build
            logger.warning("Defensive data: %s unavailable (%s)", name, e)
            notes.append({"source": name, "league": "*", "season": -1, "status": f"unavailable: {str(e)[:120]}"})
            continue
        if df.empty:
            continue
        keep = []
        for (lg, s), grp in df.groupby(["league", "season"]):
            if (lg, s) in taken or not usable(grp):
                continue
            taken[(lg, s)] = name
            keep.append(grp)
        if keep:
            parts.append(pd.concat(keep, ignore_index=True))
    rows = pd.concat(parts, ignore_index=True) if parts else frame([])
    cov = pd.DataFrame([{"source": v, "league": k[0], "season": k[1], "status": "ok"} for k, v in taken.items()] + notes)
    return rows, cov


def coverage_table(cov: pd.DataFrame, seasons: Iterable[int]) -> pd.DataFrame:
    ok = cov[cov["status"] == "ok"] if not cov.empty else cov
    table = pd.DataFrame(index=list(LEAGUES), columns=[int(s) for s in seasons], dtype=object).fillna("-")
    for _, r in ok.iterrows():
        if r["league"] in table.index and int(r["season"]) in table.columns:
            table.at[r["league"], int(r["season"])] = {"premierleague.com": "PL", "legaseriea.it": "SA",
                                                       "fbref": "FB"}.get(r["source"], r["source"][:2])
    return table


# ── linking to Understat ───────────────────────────────────────

def link(rows: pd.DataFrame, us: pd.DataFrame) -> pd.DataFrame:
    """Add `us_id` to each source row by matching within the same league-season on name,
    team and minutes played (both count the same matches, so minutes agree closely). When both
    sides know the birth year (`birth_year` column on `us`), a gap of more than a year rules a pair out."""
    from forecast.data.transfermarkt import normalise
    rows = rows.copy()
    rows["us_id"] = None
    if rows.empty or us.empty:
        return rows
    us = us.assign(_n=us["name"].map(normalise), _t=us["team"].fillna("").str.lower())
    groups = {k: g for k, g in us.groupby(["league", "season"])}
    for (lg, s), grp in rows.groupby(["league", "season"]):
        cand = groups.get((lg, s))
        if cand is None:
            continue
        cn, ct, cm, cid = cand["_n"].tolist(), cand["_t"].tolist(), cand["minutes"].to_numpy(float), cand["us_id"].tolist()
        cb = cand["birth_year"].to_numpy(float) if "birth_year" in cand else np.full(len(cand), np.nan)
        taken: Dict[str, Tuple[int, float]] = {}
        for idx, name, team, mins, born in zip(grp.index, grp["name"], grp["team"], grp["minutes"],
                                               grp["birth_year"].astype(float)):
            n, t = normalise(str(name)), str(team or "").lower()
            if not n:
                continue
            best, best_score = None, 0.0
            for j in range(len(cn)):
                if np.isfinite(born) and np.isfinite(cb[j]) and abs(born - cb[j]) > 1:
                    continue                                   # two different Emersons
                same_team = bool(t) and fuzz.token_set_ratio(t, ct[j]) >= 70
                close = (np.isfinite(mins) and mins > 0 and cm[j] > 0
                         and abs(mins - cm[j]) / max(mins, cm[j]) < 0.03)
                ns = fuzz.token_sort_ratio(n, cn[j])
                if ns < 60:
                    ns = max(ns, fuzz.token_set_ratio(n, cn[j]) - 10)
                if ns < 60 and same_team and close:
                    ns = fuzz.partial_ratio(n, cn[j]) - 10      # known-as name: 'Rodri' vs 'Rodrigo Hernandez'
                if ns < 60:
                    continue
                score = ns + (10 if same_team else 0)
                if np.isfinite(mins) and mins > 0 and cm[j] > 0:
                    score += 15 * max(0.0, 1 - abs(mins - cm[j]) / max(mins, cm[j], 270))
                if score > best_score:
                    best, best_score = cid[j], score
            if best is not None and best_score >= 88:
                prev = taken.get(best)
                if prev is None or best_score > prev[1]:
                    if prev is not None:
                        rows.at[prev[0], "us_id"] = None
                    taken[best] = (idx, best_score)
                    rows.at[idx, "us_id"] = best
    logger.info("Defensive data: linked %d of %d rows to Understat", rows["us_id"].notna().sum(), len(rows))
    return rows


def player_season_table(linked: pd.DataFrame) -> pd.DataFrame:
    """One row per (us_id, season): totals summed over teams, with the source(s) used."""
    df = linked[linked["us_id"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["us_id", "season", "def_minutes", "def_source", *TOTALS])
    agg = {"minutes": "sum", "source": lambda s: "+".join(sorted(set(s))),
           **{c: (lambda x: x.sum(min_count=1)) for c in TOTALS}}
    out = df.groupby(["us_id", "season"], as_index=False).agg(agg)
    return out.rename(columns={"minutes": "def_minutes", "source": "def_source"})


def add_rates(ps: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Join defensive totals onto the player-season table as observed per-90 rates
    (`{metric}_p90` for DEF_METRICS) and success ratios (DEF_RATIOS), NaN where uncovered."""
    ps = ps.drop(columns=[c for c in ps.columns if c in ("def_minutes", "def_source", *TOTALS)
                          or c in [f"{m}_p90" for m in DEF_METRICS] or c in DEF_RATIOS], errors="ignore")
    ps = ps.merge(table, on=["us_id", "season"], how="left")
    mins = ps["def_minutes"].to_numpy(float)
    ok = np.isfinite(mins) & (mins > 0)
    derived = derived_totals(ps)
    for m in DEF_METRICS:
        tot = derived[m]
        ps[f"{m}_p90"] = np.where(ok, tot / np.where(ok, mins, 1) * 90, np.nan)
    for r, (num, den) in DEF_RATIOS.items():
        a, b = derived[num], derived[den]
        ps[r] = np.where(np.isfinite(b) & (b >= 5), a / np.where(b > 0, b, 1), np.nan)
    return ps


def derived_totals(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    get = lambda c: df[c].to_numpy(float) if c in df else np.full(len(df), np.nan)
    d = {c: get(c) for c in TOTALS}
    d["tkl_int"] = d["tackles"] + d["interceptions"]
    d["aerials"] = d["aerials_won"] + d["aerials_lost"]
    d["duels"] = d["duels_won"] + d["duels_lost"]
    return d


def recent_rates(ps: pd.DataFrame, latest: pd.DataFrame, seasons_back: int = 2) -> pd.DataFrame:
    """Each latest-season player's defensive profile: minutes-weighted over his last
    `seasons_back` covered seasons, shrunk toward his position group's mean so 300 good
    minutes don't top a list. Returns one row per us_id with `def_{metric}` columns and
    `def_minutes_used` (0 = no defensive data; his values are then the group mean)."""
    cols = [f"{m}_p90" for m in DEF_METRICS] + list(DEF_RATIOS)
    have = [c for c in cols if c in ps]
    if not have or "def_minutes" not in ps:
        return pd.DataFrame({"us_id": latest["us_id"]})
    last = int(latest["season"].max())
    hist = ps[(ps["season"] > last - seasons_back) & ps["def_minutes"].gt(0)]
    hist = hist[hist["us_id"].isin(set(latest["us_id"]))]
    prior = ps[ps["def_minutes"].gt(0)].groupby("group")[have].mean() if "group" in ps else None
    out = []
    groups = dict(zip(latest["us_id"], latest.get("group", pd.Series([None] * len(latest)))))
    for us_id, g in hist.groupby("us_id"):
        w = g["def_minutes"].to_numpy(float)
        row = {"us_id": us_id, "def_minutes_used": float(w.sum()),
               "def_seasons": ",".join(str(int(s)) for s in sorted(g["season"].unique()))}
        for c in have:
            v = g[c].to_numpy(float)
            ok = np.isfinite(v)
            mu = float(prior.at[groups.get(us_id), c]) if prior is not None and groups.get(us_id) in prior.index else np.nan
            if not ok.any():
                row[f"def_{c}"] = mu
                continue
            obs = float(np.average(v[ok], weights=w[ok]))
            k = w[ok].sum()
            row[f"def_{c}"] = (obs * k + mu * PRIOR_MINUTES) / (k + PRIOR_MINUTES) if np.isfinite(mu) else obs
        out.append(row)
    res = pd.DataFrame(out) if out else pd.DataFrame(columns=["us_id", "def_minutes_used"])
    res = pd.DataFrame({"us_id": latest["us_id"].unique()}).merge(res, on="us_id", how="left")
    res["def_minutes_used"] = res["def_minutes_used"].fillna(0.0)
    if prior is not None:   # uncovered players: the group mean, flagged by def_minutes_used == 0
        for c in have:
            fill = res["us_id"].map(groups).map(prior[c]) if c in prior else np.nan
            res[f"def_{c}"] = res[f"def_{c}"].fillna(fill) if f"def_{c}" in res else fill
    return res


# ── CLI ────────────────────────────────────────────────────────

def check() -> bool:
    ok_all = True
    from forecast.data import fbref_stats, pl_stats, seriea_stats
    for name, probe in (("premierleague.com", pl_stats.check), ("legaseriea.it", seriea_stats.check),
                        ("FBref (GitHub release)", fbref_stats.check)):
        try:
            status = probe()
        except Exception as e:  # noqa: BLE001
            status = f"DOWN ({type(e).__name__}: {str(e)[:80]})"
        ok_all &= status == "ok"
        print(f"{name:26s} {status}")
    return ok_all


def main(argv: Optional[List[str]] = None) -> None:
    from config import current_understat_season
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--first-season", type=int, default=2014)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if a.check:
        raise SystemExit(0 if check() else 1)
    seasons = list(range(a.first_season, current_understat_season() + 1))
    rows, cov = load_all(seasons)
    print(f"{len(rows)} player rows\n")
    print(coverage_table(cov, seasons).to_string())
    print("\nPL = premierleague.com, SA = legaseriea.it, FB = FBref copy, - = none")
    bad = cov[cov["status"] != "ok"] if not cov.empty else cov
    for _, r in bad.iterrows():
        print(f"{r['source']}: {r['status']}")


if __name__ == "__main__":
    main()


def synthetic_table(ps: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Plausible defensive totals for the synthetic demo (tests and offline runs), by position."""
    rng = np.random.default_rng(seed + 1)
    base = {"CB": (0.9, 1.1, 0.35, 4.5, 2.6, 5.0), "FB": (1.7, 1.1, 0.25, 1.9, 1.0, 5.2),
            "CM": (1.9, 1.0, 0.2, 1.1, 1.0, 6.0), "AM": (1.0, 0.4, 0.1, 0.4, 0.8, 4.0),
            "ST": (0.5, 0.2, 0.08, 0.5, 2.2, 2.8)}   # tackles, int, blocks, clr, aerials won, recov per 90
    rows = []
    for us_id, season, group, mins in zip(ps["us_id"], ps["season"], ps["group"], ps["minutes"]):
        t, i, b, c, a, r = base.get(group, base["CM"])
        skill = rng.lognormal(0, 0.3, 6)
        n90 = mins / 90
        won_pct = float(np.clip(rng.normal(0.55 if group == "CB" else 0.45, 0.08), 0.2, 0.85))
        aw = a * skill[4] * n90
        rows.append({"us_id": us_id, "season": season, "def_minutes": mins, "def_source": "synthetic",
                     "tackles": t * skill[0] * n90, "tackles_won": t * skill[0] * n90 * 0.6,
                     "interceptions": i * skill[1] * n90, "blocks": b * skill[2] * n90,
                     "clearances": c * skill[3] * n90, "aerials_won": aw, "aerials_lost": aw * (1 - won_pct) / won_pct,
                     "recoveries": r * skill[5] * n90, "duels_won": (t + a) * 2 * skill[0] * n90,
                     "duels_lost": (t + a) * 2 * n90, "prog_passes": (4 if group in ("CB", "CM") else 2.5) * n90,
                     "prog_carries": 1.4 * n90})
    return pd.DataFrame(rows)
