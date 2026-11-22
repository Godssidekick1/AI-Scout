"""
Style index (ChromaDB)
======================
Every current player's style vector (the shape of his per-90 output, level
removed; see forecast/roles.py) in a persistent ChromaDB collection, filtered
by position group, league and minutes inside the query.

A brief's role profile is a target vector in the same space, so retrieval
means "players whose output looks most like what this role asks for".
Rebuilt only when the underlying data changes (fingerprint check).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_BATCH = 500


def _fingerprint(ids: List[str], vecs: np.ndarray) -> str:
    h = hashlib.sha256()
    for i, v in zip(ids, vecs):
        h.update(i.encode())
        h.update(np.round(v, 4).tobytes())
    return h.hexdigest()[:16]


class StyleIndex:
    def __init__(self, path: Path, season: int):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        # We supply the embeddings, so no embedding model is downloaded.
        self.collection = self.client.get_or_create_collection(
            name=f"styles_{season}", configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)

    def sync(self, rows: pd.DataFrame, vectors: np.ndarray) -> bool:
        """Upsert every player if the data changed. `rows` needs us_id, name, team, league, group, minutes."""
        ids = rows["us_id"].astype(str).tolist()
        fp = _fingerprint(ids, vectors)
        meta = self.collection.metadata or {}
        if meta.get("fingerprint") == fp and self.collection.count() == len(ids):
            return False
        existing = self.collection.get(include=[])["ids"]
        stale = sorted(set(existing) - set(ids))
        if stale:
            self.collection.delete(ids=stale)
        for i in range(0, len(ids), _BATCH):
            chunk = rows.iloc[i:i + _BATCH]
            self.collection.upsert(
                ids=ids[i:i + _BATCH],
                embeddings=vectors[i:i + _BATCH].tolist(),
                metadatas=[{"name": r.name_, "team": r.team, "league": r.league, "group": r.group,
                            "minutes": int(r.minutes)}
                           for r in chunk.rename(columns={"name": "name_"}).itertuples()],
                documents=chunk["name"].tolist(),
            )
        self.collection.modify(metadata={"fingerprint": fp})
        logger.info("Style index rebuilt: %d players", len(ids))
        return True

    def query(self, target: np.ndarray, n: int, groups: Iterable[str] = (), leagues: Iterable[str] = (),
              min_minutes: int = 0) -> List[Dict]:
        """[{us_id, similarity (0-100)}], nearest first."""
        total = self.collection.count()
        if total == 0:
            return []
        res = self.collection.query(query_embeddings=[np.asarray(target, float).tolist()],
                                    n_results=min(total, n), where=build_where(groups, leagues, min_minutes))
        return [{"us_id": pid, "similarity": round((1 - d) * 100, 1)}
                for pid, d in zip(res["ids"][0], res["distances"][0])]


def build_where(groups: Iterable[str], leagues: Iterable[str], min_minutes: int) -> Optional[Dict]:
    clauses: List[Dict] = []
    if min_minutes:
        clauses.append({"minutes": {"$gte": int(min_minutes)}})
    groups, leagues = list(groups or []), list(leagues or [])
    if groups:
        clauses.append({"group": {"$in": groups}})
    if leagues:
        clauses.append({"league": {"$in": leagues}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
