"""Semantic deduplication — find and merge near-duplicate memories.

Uses embedding cosine similarity to find pairs that say the same thing
in different words. Keeps the higher-importance version, merges metadata.
"""

from __future__ import annotations

import time
import numpy as np

from engram.store import Store, Memory


def find_duplicates(store: Store, threshold: float = 0.92,
                    limit: int = 500) -> list[tuple[Memory, Memory, float]]:
    """Find pairs of memories with cosine similarity >= threshold."""
    ids, vecs = store.get_all_embeddings()
    if len(ids) < 2:
        return []

    # compute similarity matrix (upper triangle only)
    # only check the first `limit` memories to avoid O(n^2) blowup
    n = min(len(ids), limit)
    sim_matrix = vecs[:n] @ vecs[:n].T

    duplicates = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                m1 = store.get_memory(ids[i])
                m2 = store.get_memory(ids[j])
                if m1 and m2:
                    duplicates.append((m1, m2, float(sim_matrix[i, j])))

    # sort by similarity descending
    duplicates.sort(key=lambda x: x[2], reverse=True)
    return duplicates


def merge_duplicate_pair(store: Store, keep: Memory, discard: Memory):
    """Keep one memory, forget the other. Merge access counts and metadata."""
    # merge access counts
    new_count = keep.access_count + discard.access_count
    store.conn.execute("UPDATE memories SET access_count = ? WHERE id = ?",
                       (new_count, keep.id))

    # merge importance (keep the higher one)
    new_importance = max(keep.importance, discard.importance)
    store.update_importance(keep.id, new_importance)

    # merge tags
    keep_tags = set(keep.metadata.get("tags", []))
    discard_tags = set(discard.metadata.get("tags", []))
    if discard_tags - keep_tags:
        import json
        merged_tags = sorted(keep_tags | discard_tags)
        keep.metadata["tags"] = merged_tags
        store.conn.execute("UPDATE memories SET metadata = ? WHERE id = ?",
                           (json.dumps(keep.metadata), keep.id))

    # transfer entity links
    store.conn.execute(
        "UPDATE OR IGNORE entity_mentions SET memory_id = ? WHERE memory_id = ?",
        (keep.id, discard.id),
    )

    # forget the duplicate
    store.forget_memory(discard.id)
    store.conn.commit()


def auto_dedup(store: Store, threshold: float = 0.92,
               max_merges: int = 50) -> dict:
    """Find and merge duplicates automatically. Returns stats."""
    dupes = find_duplicates(store, threshold)
    stats = {"found": len(dupes), "merged": 0, "skipped": 0}

    merged_ids = set()
    for m1, m2, similarity in dupes:
        if stats["merged"] >= max_merges:
            break
        if m1.id in merged_ids or m2.id in merged_ids:
            stats["skipped"] += 1
            continue

        # keep the one with higher importance, or if tied, the longer one
        if m1.importance > m2.importance:
            keep, discard = m1, m2
        elif m2.importance > m1.importance:
            keep, discard = m2, m1
        elif len(m1.content) >= len(m2.content):
            keep, discard = m1, m2
        else:
            keep, discard = m2, m1

        merge_duplicate_pair(store, keep, discard)
        merged_ids.add(discard.id)
        stats["merged"] += 1

    store.invalidate_embedding_cache()
    return stats
