"""
Player Comparison Engine
========================
Uses ChromaDB vector embeddings to find players with similar statistical profiles.
"""

from typing import Dict, List
import numpy as np

import chromadb
from chromadb.config import Settings


class PlayerComparisonEngine:
    """
    Finds players with similar playing styles using vector similarity.
    Converts player stats into a 32-dimensional vector and uses
    ChromaDB for fast cosine-similarity search.
    """

    def __init__(self, collection_name: str = "player_vectors"):
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _stats_to_vector(self, stats: Dict) -> List[float]:
        """
        Convert player stats to a normalized 32-dimensional vector.

        The vector captures 4 dimensions of play:
        - Attacking output (8 dims)
        - Defensive contribution (8 dims)
        - Ball progression (8 dims)
        - Style indicators (8 dims)
        """

        def s(key, default=0.0):
            return float(stats.get(key, default))

        vector = [
            # === Attacking (8 dims) ===
            s("goals_p90"),
            s("assists_p90"),
            s("xG_p90"),
            s("xA_p90"),
            s("xT_p90"),
            s("shot_creating_actions_p90"),
            s("goals_p90") + s("assists_p90"),                    # G+A
            s("goals_p90") / max(s("xG_p90"), 0.01),             # Finishing quality

            # === Defensive (8 dims) ===
            s("tackles_p90"),
            s("interceptions_p90"),
            s("pressing_success_pct") / 100,
            s("defensive_duel_win_rate") / 100,
            s("blocks_p90"),
            s("tackles_p90") + s("interceptions_p90"),            # Total defensive actions
            s("pressing_success_pct") / 100 * s("tackles_p90"),   # Pressing intensity
            s("defensive_duel_win_rate") / 100 * s("tackles_p90"),

            # === Progression (8 dims) ===
            s("progressive_carries_p90"),
            s("progressive_passes_p90"),
            s("passes_into_penalty_area_p90"),
            s("progressive_carries_distance") / 100,
            s("pv_plus"),
            (s("progressive_carries_p90") + s("progressive_passes_p90")) / 2,
            s("passes_into_penalty_area_p90") / max(s("progressive_carries_p90"), 0.01),
            s("high_intensity_sprints_p90") / 10,

            # === Style indicators (8 dims) ===
            1.0 if s("goals_p90") > s("assists_p90") else 0.0,   # Scorer vs creator
            s("goals_p90") / max(s("goals_p90") + s("assists_p90"), 0.01),
            s("tackles_p90") / max(s("tackles_p90") + s("interceptions_p90"), 0.01),
            1.0 if s("progressive_carries_p90") > 5.0 else 0.0,  # Ball-carrier flag
            1.0 if s("pressing_success_pct") > 30 else 0.0,      # Active presser
            (s("goals_p90") + s("tackles_p90")) / 2,              # Attack-defense balance
            s("xT_p90") / max(s("shot_creating_actions_p90"), 0.01),
            s("high_intensity_sprints_p90") / max(s("progressive_carries_p90"), 0.01),
        ]

        # Normalize
        magnitude = np.linalg.norm(vector)
        if magnitude > 0:
            vector = [v / magnitude for v in vector]

        return vector[:32]

    def add_player(self, player_id: str, name: str, stats: Dict, metadata: Dict = None):
        """Add a player to the similarity database."""
        vector = self._stats_to_vector(stats)
        meta = {
            "name": name,
            "position": stats.get("position", ""),
            "team": stats.get("team", ""),
            "league": stats.get("league", ""),
            "age": int(stats.get("age", 0)),
        }
        if metadata:
            meta.update(metadata)

        self.collection.upsert(
            embeddings=[vector],
            documents=[name],
            metadatas=[meta],
            ids=[player_id],
        )

    def find_similar(self, stats: Dict, n: int = 5, exclude_id: str = None) -> List[Dict]:
        """
        Find the most similar players to the given stat profile.

        Returns:
            List of dicts with name, position, team, similarity_score (0-100)
        """
        vector = self._stats_to_vector(stats)

        results = self.collection.query(
            query_embeddings=[vector],
            n_results=n + (1 if exclude_id else 0),  # Fetch extra in case we need to exclude
        )

        similar = []
        if results["ids"] and results["ids"][0]:
            for i, pid in enumerate(results["ids"][0]):
                if exclude_id and pid == exclude_id:
                    continue

                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = round((1 - distance) * 100, 1)

                similar.append({
                    "player_id": pid,
                    "name": meta.get("name", ""),
                    "position": meta.get("position", ""),
                    "team": meta.get("team", ""),
                    "league": meta.get("league", ""),
                    "similarity_score": similarity,
                })

        return similar[:n]

    def reset(self):
        """Clear all player data."""
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
