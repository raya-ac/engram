"""Memory lifecycle: importance scoring, Ebbinghaus forgetting, promotion/demotion."""

from __future__ import annotations

import math
import time

from engram.config import Config
from engram.store import Store, Memory, MemoryLayer


def compute_importance(mem: Memory) -> float:
    """7-factor composite importance score."""
    base = mem.importance

    # access frequency (log scale, capped)
    access_factor = min(1.0, 0.1 * math.log(1 + mem.access_count))

    # recency (exponential decay, 30-day half-life)
    age_days = (time.time() - mem.last_accessed) / 86400
    recency = math.exp(-0.693 * age_days / 30)

    # emotional valence boost (strong emotions = more memorable)
    emotion = abs(mem.emotional_valence) * 0.3

    # stability (accessed frequently over time = stable)
    if mem.access_count > 0:
        span = max(1, (mem.last_accessed - mem.created_at) / 86400)
        stability = min(1.0, mem.access_count / (span + 1))
    else:
        stability = 0.0

    # layer boost (semantic memories are more valuable by definition)
    layer_boost = {
        MemoryLayer.WORKING: 0.0,
        MemoryLayer.EPISODIC: 0.1,
        MemoryLayer.SEMANTIC: 0.3,
        MemoryLayer.PROCEDURAL: 0.2,
    }.get(mem.layer, 0.0)

    score = (
        base * 0.30
        + access_factor * 0.15
        + recency * 0.15
        + emotion * 0.10
        + stability * 0.10
        + layer_boost * 0.20
    )
    return min(1.0, max(0.0, score))


def should_forget(mem: Memory, config: Config) -> bool:
    """Should this memory be archived (soft-forgotten)?"""
    if mem.layer == MemoryLayer.SEMANTIC:
        return False  # semantic memories are permanent
    if mem.layer == MemoryLayer.PROCEDURAL:
        return False  # procedural memories persist

    age_days = (time.time() - mem.created_at) / 86400
    lc = config.lifecycle

    return (
        age_days > lc.archive_after_days
        and mem.importance < lc.archive_min_importance
        and mem.access_count < lc.archive_min_accesses
    )


def should_promote(mem: Memory, config: Config) -> str | None:
    """Should this memory be promoted to a higher layer? Returns target layer or None."""
    lc = config.lifecycle

    if mem.layer == MemoryLayer.EPISODIC:
        if mem.importance >= lc.promote_importance and mem.access_count >= lc.promote_accesses:
            return MemoryLayer.SEMANTIC
    elif mem.layer == MemoryLayer.WORKING:
        age_minutes = (time.time() - mem.created_at) / 60
        # auto-promote working to episodic after 30 min or if accessed twice
        if age_minutes > 30 or mem.access_count >= 2:
            return MemoryLayer.EPISODIC

    return None


def apply_forgetting_curve(store: Store, config: Config) -> dict:
    """Apply Ebbinghaus decay to all episodic memories. Returns stats."""
    stats = {"forgotten": 0, "promoted": 0, "updated": 0}

    rows = store.conn.execute(
        "SELECT * FROM memories WHERE forgotten = 0 AND layer IN ('episodic', 'working')"
    ).fetchall()

    for row in rows:
        mem = store._row_to_memory(row)

        # check for promotion first
        target = should_promote(mem, config)
        if target:
            store.update_layer(mem.id, target)
            stats["promoted"] += 1
            continue

        # check for forgetting
        if should_forget(mem, config):
            store.forget_memory(mem.id)
            stats["forgotten"] += 1
            continue

        # update importance with decay
        new_importance = compute_importance(mem)
        if abs(new_importance - mem.importance) > 0.01:
            store.update_importance(mem.id, new_importance)
            stats["updated"] += 1

    return stats
