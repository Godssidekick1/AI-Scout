"""
Understat history from the Internet Archive
===========================================
Fallback for when understat.com is down. The Wayback Machine holds copies of
Understat's league pages, and each copy still carries the embedded
`playersData` (season totals, including xGChain and xGBuildup) and `datesData`
(the fixture list). For every big-five league and completed season we take a
copy saved after the season ended, check from `datesData` that it really is
that season and that every match had been played, and write the players into
the same cache key the live loader uses. The build then runs unchanged.

    python -m forecast.data.understat_archive            # fill the cache, 2014 to last season
    python -m forecast.data.understat_archive --dry-run  # just report coverage
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import time
import urllib.parse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import httpx

from config import current_understat_season, settings
from data_sources.data_cache import DataCache
from data_sources.understat_source import UNDERSTAT_LEAGUES
from forecast.data.understat_history import FIRST_SEASON

logger = logging.getLogger(__name__)

GZIP_MAGIC = bytes([0x1F, 0x8B])
CDX = "https://web.archive.org/cdx/search/cdx"
RAW = "https://web.archive.org/web/{ts}id_/{url}"   # id_ = the page as archived, no Wayback toolbar
HEADERS = {"User-Agent": "ai-scout/2.0 (portfolio project; Wayback fallback for Understat history)"}
_URL = re.compile(r"understat\.com/(league|getLeagueData)/([A-Za-z0-9_ ]+)/?(\d{4})?/?(?:[?#].*)?$")


def _embedded(html: str, var: str) -> Optional[list]:
    m = re.search(rf"var\s+{var}\s*=\s*JSON\.parse\('(.+?)'\)", html)
    if not m:
        return None
    try:
        return json.loads(m.group(1).encode("utf-8").decode("unicode_escape"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def season_of(dates: List[Dict]) -> Optional[int]:
    """Season (starting year) of a fixture list: the year of its earliest match, minus one if before July."""
    stamps = sorted(d["datetime"] for d in dates if d.get("datetime"))
    if not stamps:
        return None
    year, month = int(stamps[0][:4]), int(stamps[0][5:7])
    return year if month >= 7 else year - 1


# Seasons that ended early and will never have every match played.
ABANDONED = {("Ligue_1", 2019)}   # stopped in April 2020 (COVID)


def _parse(text: str) -> Tuple[Optional[list], Optional[list]]:
    """(players, dates) from an old-style league page or a newer getLeagueData JSON response."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            return data.get("players"), data.get("dates")
        except (json.JSONDecodeError, AttributeError):
            return None, None
    return _embedded(text, "playersData"), _embedded(text, "datesData")


def validate(html: str, season: int, slug: str = "") -> Tuple[Optional[list], str]:
    """(players, "ok") if the page is a complete copy of that season, else (None, reason)."""
    players, dates = _parse(html)
    if not players:
        return None, "no playersData"
    if not dates:
        return None, "no datesData"
    got = season_of(dates)
    if got != season:
        return None, f"page shows season {got}"
    unplayed = sum(not d.get("isResult") for d in dates)
    # A handful of abandoned/annulled fixtures never get a result; allow up to 2.
    if unplayed > 2 and (slug, season) not in ABANDONED:
        return None, f"{unplayed} matches not yet played"
    return players, "ok"


def candidates(snapshots: List[List[str]]) -> Dict[Tuple[str, int], List[Tuple[str, str]]]:
    """Group CDX rows into {(slug, season): [(timestamp, url), ...]} for copies saved after the season ended.
    Year-specific URLs count from 20 May of the following year; the yearless URL only between
    20 May and 15 August (after that it switches to the new season)."""
    out: Dict[Tuple[str, int], set] = defaultdict(set)
    slugs = set(UNDERSTAT_LEAGUES.values())
    for ts, url in snapshots:
        m = _URL.search(urllib.parse.unquote(url))
        if not m:
            continue
        slug, year = m.group(2).replace(" ", "_"), m.group(3)
        if slug not in slugs:
            continue
        if m.group(1) == "getLeagueData" and not year:
            continue
        if year:
            season = int(year)
            if ts >= f"{season + 1}0520":
                out[(slug, season)].add((ts, url))
        else:
            y = int(ts[:4])
            if f"{y}0520" <= ts <= f"{y}0815":
                out[(slug, y - 1)].add((ts, url))
    # JSON responses first (newer site), then earliest copies (closest to season end).
    return {k: sorted(v, key=lambda c: ("getLeagueData" not in c[1], c[0])) for k, v in out.items()}


def list_snapshots(client: httpx.Client,
                   prefixes: Tuple[str, ...] = ("understat.com/league/", "understat.com/getLeagueData/")) -> List[List[str]]:
    """CDX rows (timestamp, url) for archived league pages (embedded data) and getLeagueData JSON (newer site)."""
    rows: List[List[str]] = []
    for prefix in prefixes:
        params = {"url": prefix, "matchType": "prefix", "output": "json",
                  "filter": "statuscode:200", "fl": "timestamp,original", "limit": "200000"}
        for attempt in range(4):
            try:
                r = client.get(CDX, params=params, timeout=180)
                r.raise_for_status()
                rows += r.json()[1:] if r.text.strip() else []
                break
            except (httpx.HTTPError, ValueError) as e:
                logger.warning("Wayback index failed (%s); retrying", e)
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit("Could not reach the Wayback Machine index (web.archive.org).")
    return rows


def fetch(client: httpx.Client, ts: str, url: str) -> Optional[str]:
    for attempt in range(3):
        try:
            r = client.get(RAW.format(ts=ts, url=url), timeout=120, follow_redirects=True)
            if r.status_code == 200:
                body = r.content
                if body[:2] == GZIP_MAGIC:   # archived with its gzip encoding intact
                    body = gzip.decompress(body)
                if body.strip():          # Wayback sometimes answers 200 with an empty body under load
                    return body.decode("utf-8", errors="replace")
            if r.status_code not in (429, 500, 502, 503, 504):
                return None
        except httpx.HTTPError:
            pass
        time.sleep(3 * (attempt + 1))
    return None


NUMERIC = ("games", "time", "goals", "xG", "assists", "xA", "shots", "key_passes", "yellow_cards",
           "red_cards", "npg", "npxG", "xGChain", "xGBuildup")


def merge_team_players(per_team: Dict[str, list]) -> list:
    """Team pages list a player's stats for that team only. Combine them into one league-style row
    per player: totals summed, teams joined by commas with the team he played most for last."""
    rows: Dict[str, list] = defaultdict(list)
    for team, players in per_team.items():
        for p in players:
            rows[str(p["id"])].append({**p, "team_title": team})
    out = []
    for pid, parts in rows.items():
        parts.sort(key=lambda p: float(p.get("time") or 0))
        merged = dict(parts[-1])
        for k in NUMERIC:
            merged[k] = str(sum(float(p.get(k) or 0) for p in parts))
        merged["team_title"] = ",".join(p["team_title"] for p in parts)
        out.append(merged)
    return out


def team_candidates(snapshots: List[List[str]]) -> Dict[Tuple[str, int], List[Tuple[str, str]]]:
    """{(team title, season): [(timestamp, url)]} for team pages saved after 20 May of the following year."""
    out: Dict[Tuple[str, int], set] = defaultdict(set)
    for ts, url in snapshots:
        m = re.search(r"understat\.com/team/([^/?#]+)/(\d{4})/?$", urllib.parse.unquote(url))
        if m and ts >= f"{int(m.group(2)) + 1}0520":
            out[(m.group(1).replace("_", " "), int(m.group(2)))].add((ts, url))
    return {k: sorted(v) for k, v in out.items()}


def from_team_pages(client: httpx.Client, slug: str, season: int, seeds: List[str],
                    tcands: Dict[Tuple[str, int], List[Tuple[str, str]]], pause_s: float) -> Tuple[Optional[list], str]:
    """Rebuild a league-season from every team's archived page. The first team page found gives the
    league's team list (its fixtures); every team must then be found and complete."""
    def team_page(team: str) -> Optional[Tuple[list, list]]:
        for ts, url in tcands.get((team, season), [])[:4]:
            text = fetch(client, ts, url)
            time.sleep(pause_s)
            if text is None:
                continue
            players, dates = _parse(text)
            if players and dates and season_of(dates) == season and                     (sum(not d.get("isResult") for d in dates) <= 1 or (slug, season) in ABANDONED):
                return players, dates
        return None

    per_team: Dict[str, list] = {}
    teams: List[str] = []
    for seed in seeds:
        got = team_page(seed)
        if got:
            per_team[seed] = got[0]
            teams = sorted({d["h"]["title"] for d in got[1]} | {d["a"]["title"] for d in got[1]})
            break
    if not teams:
        return None, "no complete team page for any known team"
    missing = []
    for team in teams:
        if team in per_team:
            continue
        got = team_page(team)
        if got:
            per_team[team] = got[0]
        else:
            missing.append(team)
    if missing:
        return None, f"team pages missing for {', '.join(missing)}"
    return merge_team_players(per_team), "ok"


def fill_cache(first: int = FIRST_SEASON, last: Optional[int] = None, dry_run: bool = False,
               force: bool = False, pause_s: float = 1.0) -> Dict[Tuple[str, int], str]:
    """Write archived players for every (league, season) into the Understat history cache.
    Returns {(slug, season): status} where status is 'cached', 'ok (dry run)', 'already cached' or why it failed."""
    last = current_understat_season() - 1 if last is None else last
    cache = DataCache(cache_dir=str(settings.cache_path), ttl_hours=24 * 365 * 5)
    status: Dict[Tuple[str, int], str] = {}
    with httpx.Client(headers=HEADERS) as client:
        cands = candidates(list_snapshots(client))
        tcands = None
        for slug in UNDERSTAT_LEAGUES.values():
            for season in range(first, last + 1):
                key = f"understat:league:{slug}:{season}"
                if not force and not dry_run and cache.get(key) is not None:
                    status[(slug, season)] = "already cached"
                    continue
                reasons = []
                for ts, url in cands.get((slug, season), [])[:6]:
                    html = fetch(client, ts, url)
                    time.sleep(pause_s)
                    if html is None:
                        reasons.append(f"{ts}: fetch failed")
                        continue
                    players, why = validate(html, season, slug)
                    if players is None:
                        reasons.append(f"{ts}: {why}")
                        continue
                    if not dry_run:
                        cache.set(key, {"players": players, "source": f"wayback:{ts}:{url}"})
                    status[(slug, season)] = f"{'ok (dry run)' if dry_run else 'cached'}: {len(players)} players, copy of {ts[:8]}"
                    break
                else:
                    status[(slug, season)] = "missing: " + ("; ".join(reasons) if reasons else "no archived copy after season end")
                    # Second route: every team's own page for that season.
                    if tcands is None:
                        tcands = team_candidates(list_snapshots(client, ("understat.com/team/",)))
                    seeds = []
                    for near in (season - 1, season + 1, season - 2, season + 2):
                        near_data = cache.get(f"understat:league:{slug}:{near}") or {}
                        for p in near_data.get("players", []):
                            for t in (p.get("team_title") or "").split(","):
                                if t.strip() and t.strip() not in seeds:
                                    seeds.append(t.strip())
                    players, why = from_team_pages(client, slug, season, seeds[:8], tcands, pause_s)
                    if players:
                        if not dry_run:
                            cache.set(key, {"players": players, "source": "wayback:team-pages"})
                        status[(slug, season)] = f"{'ok (dry run)' if dry_run else 'cached'}: {len(players)} players, from team pages"
                    else:
                        status[(slug, season)] += f"; team pages: {why}"
                logger.info("%s %s -> %s", slug, season, status[(slug, season)])
    return status


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first-season", type=int, default=FIRST_SEASON)
    ap.add_argument("--last-season", type=int, default=None, help="default: last completed season")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="refetch seasons already in the cache")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    status = fill_cache(args.first_season, args.last_season, args.dry_run, args.force)
    ok = sum(not s.startswith("missing") for s in status.values())
    print(f"\n{ok} of {len(status)} league-seasons available")
    for (slug, season), s in sorted(status.items()):
        if s.startswith("missing"):
            print(f"  {slug} {season}: {s}")


if __name__ == "__main__":
    main()
