"""
Player Comparison Engine
========================
Finds players with similar statistical profiles by cosine similarity over a
standardised per-90 feature vector. A pool of 10-20 players fits trivially in
memory, so plain numpy replaces the vector database the first version used.
"""

from typing import Dict, List, Optional

import numpy as np

SIMILARITY_FEATURES = [
    "goals_p90", "assists_p90", "xG_p90", "xA_p90", "npxG_p90",
    "shots_p90", "key_passes_p90", "xGChain_p90", "xGBuildup_p90",
    # legacy / optional fields, zero when absent
    "xT_p90", "tackles_p90",
]

# Rough population spreads so one large-valued stat (shots) doesn't dominate small ones (xA).
FEATURE_SCALE = {
    "goals_p90": 0.3, "assists_p90": 0.2, "xG_p90": 0.3, "xA_p90": 0.15, "npxG_p90": 0.25,
    "shots_p90": 1.5, "key_passes_p90": 1.0, "xGChain_p90": 0.3, "xGBuildup_p90": 0.2,
    "xT_p90": 0.2, "tackles_p90": 1.0,
}


def stats_to_vector(stats: Dict) -> np.ndarray:
    v = np.array([float(stats.get(f) or 0.0) / FEATURE_SCALE[f] for f in SIMILARITY_FEATURES])
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def cosine_similarity(a: Dict, b: Dict) -> float:
    """0-100 similarity between two stat profiles."""
    return round(float(np.dot(stats_to_vector(a), stats_to_vector(b))) * 100, 1)


class PlayerComparisonEngine:
    def __init__(self, collection_name: str = "player_vectors"):
        self.collection_name = collection_name
        self._ids: List[str] = []
        self._meta: List[Dict] = []
        self._vectors: List[np.ndarray] = []

    def add_player(self, player_id: str, name: str, stats: Dict, metadata: Optional[Dict] = None):
        meta = {"name": name, "position": stats.get("position", ""), "team": stats.get("team", ""),
                "league": stats.get("league", ""), **(metadata or {})}
        vec = stats_to_vector(stats)
        if player_id in self._ids:
            i = self._ids.index(player_id)
            self._meta[i], self._vectors[i] = meta, vec
        else:
            self._ids.append(player_id)
            self._meta.append(meta)
            self._vectors.append(vec)

    def find_similar(self, stats: Dict, n: int = 5, exclude_id: Optional[str] = None) -> List[Dict]:
        if not self._ids:
            return []
        sims = np.stack(self._vectors) @ stats_to_vector(stats)
        out = []
        for i in np.argsort(-sims):
            if self._ids[i] == exclude_id:
                continue
            out.append({"player_id": self._ids[i], **self._meta[i],
                        "similarity_score": round(float(sims[i]) * 100, 1)})
            if len(out) == n:
                break
        return out

    def reset(self):
        self._ids, self._meta, self._vectors = [], [], []
