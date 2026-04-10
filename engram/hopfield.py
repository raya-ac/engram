"""Hopfield associative memory channel (SuperLocalMemory V3.3).

Content-addressable pattern completion: given a partial cue, reconstruct
the most likely full memory pattern. Different from vector similarity —
it does pattern *completion* not just pattern *matching*.

Modern Hopfield network with exponential storage capacity:
  ξ_new = X^T · softmax(β · X · ξ)

Where X is the memory matrix and β controls pattern sharpness.
"""

from __future__ import annotations

import numpy as np

from engram.store import Store


def hopfield_retrieve(query_embedding: np.ndarray, store: Store,
                      beta: float = 8.0, top_k: int = 5) -> list[tuple[str, float]]:
    """Hopfield associative retrieval — pattern completion from partial cue.

    Args:
        query_embedding: the partial cue (query vector)
        store: memory store
        beta: inverse temperature (higher = sharper pattern selection)
        top_k: number of results

    Returns:
        List of (memory_id, association_score) tuples
    """
    ids, vecs = store.get_all_embeddings()
    if not ids or len(ids) == 0:
        return []

    # X = memory matrix (N x D), ξ = query (D,)
    X = vecs  # already normalized from store
    xi = query_embedding

    # Modern Hopfield update: ξ_new = X^T · softmax(β · X · ξ)
    # Compute attention scores
    scores = X @ xi  # (N,) — raw similarities
    # softmax with temperature
    scores_scaled = beta * scores
    # numerical stability
    scores_scaled -= scores_scaled.max()
    exp_scores = np.exp(scores_scaled)
    attention = exp_scores / (exp_scores.sum() + 1e-8)

    # the attention weights ARE the association scores
    # (they tell us which stored patterns the query most strongly activates)
    top_indices = np.argsort(attention)[::-1][:top_k]

    return [(ids[i], float(attention[i])) for i in top_indices]


def hopfield_complete(query_embedding: np.ndarray, store: Store,
                      beta: float = 8.0, iterations: int = 3) -> np.ndarray:
    """Run Hopfield dynamics to completion — reconstruct a full pattern from a cue.

    Iteratively updates the query toward the nearest stored attractor.
    Returns the completed pattern embedding.
    """
    ids, vecs = store.get_all_embeddings()
    if not ids:
        return query_embedding

    X = vecs
    xi = query_embedding.copy()

    for _ in range(iterations):
        scores = beta * (X @ xi)
        scores -= scores.max()
        attention = np.exp(scores) / (np.exp(scores).sum() + 1e-8)
        # update: weighted combination of stored patterns
        xi = X.T @ attention
        # normalize
        norm = np.linalg.norm(xi)
        if norm > 1e-8:
            xi = xi / norm

    return xi
