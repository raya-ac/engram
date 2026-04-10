"""Surprise-based importance scoring at write time.

Inspired by the Titans paper (2501.00663, Behrouz et al., Google):
memory updates proportional to gradient of loss = surprise.

Implementation: compute k-NN cosine distance against existing embeddings
before inserting. Novel memories (far from existing) get high surprise
scores → boosted importance. Redundant ones (close to existing) get
low surprise → flagged as potential duplicates.
"""

from __future__ import annotations

import numpy as np

from engram.store import Store


def compute_surprise(embedding: np.ndarray, store: Store,
                     k: int = 5, dedup_threshold: float = 0.92) -> dict:
    """Compute how surprising/novel a new memory is relative to existing ones.

    Returns:
        {
            "surprise": float 0-1 (1 = completely novel, 0 = exact duplicate),
            "nearest_distance": float (cosine distance to closest neighbor),
            "nearest_id": str | None (id of most similar existing memory),
            "is_duplicate": bool (True if nearest_distance < dedup_threshold),
            "k_distances": list[float] (distances to k nearest neighbors),
            "importance_modifier": float (-0.3 to +0.3, additive adjustment),
        }
    """
    ids, vecs = store.get_all_embeddings()

    # no existing memories → maximum surprise
    if len(ids) == 0:
        return {
            "surprise": 1.0,
            "nearest_distance": 1.0,
            "nearest_id": None,
            "is_duplicate": False,
            "k_distances": [],
            "importance_modifier": 0.15,
        }

    # cosine similarity (vectors are pre-normalized)
    similarities = vecs @ embedding
    k_actual = min(k, len(ids))
    top_indices = np.argsort(similarities)[::-1][:k_actual]

    k_similarities = [float(similarities[i]) for i in top_indices]
    k_distances = [1.0 - s for s in k_similarities]

    nearest_idx = top_indices[0]
    nearest_sim = k_similarities[0]
    nearest_dist = k_distances[0]

    # surprise = mean distance to k nearest neighbors
    # high mean distance = novel content = high surprise
    mean_distance = float(np.mean(k_distances))

    # map to 0-1 range with sigmoid-like curve
    # most memories cluster around 0.2-0.5 distance
    # we want surprise=0.5 at distance=0.35, steep around that
    surprise = _sigmoid(mean_distance, midpoint=0.35, steepness=10.0)

    # importance modifier: -0.3 (very redundant) to +0.3 (very novel)
    # linear map from surprise
    importance_modifier = (surprise - 0.5) * 0.6

    is_duplicate = nearest_sim >= dedup_threshold

    return {
        "surprise": round(surprise, 4),
        "nearest_distance": round(nearest_dist, 4),
        "nearest_id": ids[nearest_idx],
        "nearest_ids": [ids[i] for i in top_indices],
        "is_duplicate": is_duplicate,
        "k_distances": [round(d, 4) for d in k_distances],
        "importance_modifier": round(importance_modifier, 4),
    }


def adjust_importance(base_importance: float, surprise_result: dict) -> float:
    """Adjust a memory's importance score based on surprise.

    Clamps to [0.05, 1.0] — even redundant memories keep a floor.
    """
    adjusted = base_importance + surprise_result["importance_modifier"]
    return round(min(1.0, max(0.05, adjusted)), 4)


def _sigmoid(x: float, midpoint: float = 0.35, steepness: float = 10.0) -> float:
    """Sigmoid mapping: values near midpoint → 0.5, far above → 1.0, far below → 0.0."""
    z = steepness * (x - midpoint)
    # clamp to avoid overflow
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + np.exp(-z))
