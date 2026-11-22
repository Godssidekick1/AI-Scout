"""
Bzzoiro Sports Data (sports.bzzoiro.com): can it stand in for Understat?
=======================================================================
    set BZZOIRO_TOKEN=...          (free token from https://sports.bzzoiro.com/register/)
    python -m forecast.data.bzzoiro_probe

What its public docs say (checked 22 Nov 2026): football is free with a token (7,500 requests a day),
the licence allows storing data for internal use and using it for models, and forbids redistributing
the raw data. The docs do NOT list player-level xG / xA / npxG / key passes / minutes, how far back
the big-five seasons go, or where its xG comes from. This probe answers those three questions
against the real API, and compares any season xG it finds with the Understat seasons already in the
local cache. It changes nothing in the model; every response is cached under cache/bzzoiro/.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config import settings

BASE = "https://sports.bzzoiro.com/api/v2"
BIG_FIVE = ("Premier League", "LaLiga", "La Liga", "Bundesliga", "Serie A", "Ligue 1")
XG_WORDS = ("xg", "expected", "xa", "key_pass", "keypass", "shot", "minute")


def _get(path: str, token: str, params: Optional[Dict[str, Any]] = None) -> Any:
    cache = settings.cache_path / "bzzoiro" / (path.strip("/").replace("/", "_") +
                                               ("_" + "_".join(f"{k}-{v}" for k, v in sorted((params or {}).items())) if params else "") + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    r = httpx.get(f"{BASE}/{path.strip('/')}/", params=params, timeout=30,
                  headers={"Authorization": f"Token {token}"})
    if r.status_code == 429:
        time.sleep(float(r.headers.get("Retry-After", "5")))
        return _get(path, token, params)
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(r.text)
    time.sleep(0.5)
    return r.json()


def _rows(payload: Any) -> List[Dict]:
    if isinstance(payload, dict):
        for k in ("results", "data", "items"):
            if isinstance(payload.get(k), list):
                return payload[k]
        return [payload]
    return payload if isinstance(payload, list) else []


def _keys(obj: Any, prefix: str = "") -> List[str]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += [f"{prefix}{k}"] + _keys(v, f"{prefix}{k}.")
    elif isinstance(obj, list) and obj:
        out += _keys(obj[0], prefix)
    return out


def main() -> None:
    token = os.environ.get("BZZOIRO_TOKEN", "")
    if not token:
        raise SystemExit("Set BZZOIRO_TOKEN first (free at https://sports.bzzoiro.com/register/).")
    leagues = [lg for lg in _rows(_get("leagues", token, {"limit": 200}))
               if any(b.lower() in str(lg.get("name", "")).lower() for b in BIG_FIVE)]
    print(f"Big-five leagues found: {[lg.get('name') for lg in leagues]}")
    for lg in leagues[:5]:
        seasons = _rows(_get(f"leagues/{lg['id']}/seasons", token))
        years = sorted(str(s.get("year") or s.get("name")) for s in seasons)
        print(f"  {lg.get('name')}: {len(seasons)} seasons, earliest {years[:1]}, latest {years[-1:]}")

    pl = next((lg for lg in leagues if "premier" in str(lg.get("name", "")).lower()), None)
    if pl is None:
        return
    top = _rows(_get(f"leagues/{pl['id']}/top/scorers", token))[:5]
    for row in top:
        pid = row.get("player_id")
        career = _get(f"players/{pid}/career", token)
        keys = _keys(career)
        hits = sorted({k for k in keys if any(w in k.lower() for w in XG_WORDS)})
        print(f"\n{row.get('player_name')}: career fields with xG/shots/minutes: {hits or 'none'}")
        print(json.dumps(_rows(career)[:2], indent=1)[:1500])
    print("\nIf season xG is there, compare it with Understat: the cached Understat rows are in "
          "forecast.data.understat_history.load_seasons (cache only, no network needed).")


if __name__ == "__main__":
    main()
