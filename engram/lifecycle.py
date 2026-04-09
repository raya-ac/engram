"""Memory lifecycle: retention regularization, importance scoring, promotion/demotion.

Retention model inspired by Miras (2504.13173, Behrouz et al., Google):
forgetting mechanisms are retention regularization. Three modes:

- L2 (default): classic Ebbinghaus exponential decay. Smooth, all memories
  fade gradually. Equivalent to weight decay.

- Huber: robust to outlier access patterns. Memories with burst-then-quiet
  usage get a gentler transition instead of falling off a cliff. Below delta,
  acts like L2; above delta, linear decay (less aggressive on old-but-once-hot
  memories).

- Elastic net (L1 + L2): sparse retention. Strongly-held memories stay near
  full strength, weakly-held ones decay faster than L2 alone. Produces a
  cleaner separation between "keeper" and "forgettable" memories.
"""

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


# --- Retention regularization functions ---

def retention_l2(age_days: float, half_life: float) -> float:
    """Classic Ebbinghaus exponential decay (L2 regularization).

    Smooth, gradual forgetting. All memories decay at the same rate
    regardless of their history.
    """
    return math.exp(-0.693 * age_days / half_life)


def retention_huber(age_days: float, half_life: float, delta: float = 0.5) -> float:
    """Huber retention: smooth near-recent, linear for old memories.

    Below delta (in half-lives), behaves like L2 — exponential decay.
    Above delta, transitions to linear decay — gentler on old-but-once-hot
    memories. Robust to outlier access patterns (burst-then-quiet).

    delta controls the transition point:
    - delta=0.3: quick transition to linear (aggressive on recent)
    - delta=0.5: balanced (default)
    - delta=1.0: mostly exponential, linear only for very old
    """
    # normalize age to half-life units
    t = age_days / half_life

    if t <= delta:
        # quadratic region (equivalent to L2): standard exponential
        return math.exp(-0.693 * age_days / half_life)
    else:
        # linear region: value at transition point, then linear decay
        transition_val = math.exp(-0.693 * delta)
        slope = 0.693 * transition_val / 1.0  # derivative at transition
        linear = transition_val - slope * (t - delta)
        return max(0.0, linear)


def retention_elastic(age_days: float, half_life: float,
                      l1_ratio: float = 0.3) -> float:
    """Elastic net retention: L1 + L2 mix for sparse retention.

    Strongly-held memories (recently/frequently accessed) stay near full
    strength. Weakly-held ones decay faster than pure L2. Produces a
    cleaner binary separation.

    l1_ratio: blend between L1 (sparse, hard threshold) and L2 (smooth).
    - 0.0: pure L2 (standard Ebbinghaus)
    - 0.5: balanced sparse + smooth
    - 1.0: pure L1 (hard cutoff)
    """
    l2_component = math.exp(-0.693 * age_days / half_life)

    # L1 component: linear decay to zero at 2x half-life
    l1_component = max(0.0, 1.0 - (age_days / (2.0 * half_life)))

    return l1_ratio * l1_component + (1.0 - l1_ratio) * l2_component


def compute_retention(mem: Memory, config: Config) -> float:
    """Compute retention score using the configured regularization mode.

    Returns a value in [0, 1] where 1 = full retention, 0 = fully decayed.
    """
    age_days = (time.time() - mem.last_accessed) / 86400
    half_life = config.lifecycle.forgetting_half_life_days
    mode = getattr(config.lifecycle, 'retention_mode', 'l2')

    if mode == 'huber':
        delta = getattr(config.lifecycle, 'huber_delta', 0.5)
        base_retention = retention_huber(age_days, half_life, delta)
    elif mode == 'elastic':
        l1_ratio = getattr(config.lifecycle, 'elastic_l1_ratio', 0.3)
        base_retention = retention_elastic(age_days, half_life, l1_ratio)
    else:  # 'l2' default
        base_retention = retention_l2(age_days, half_life)

    # access count reinforcement: each access strengthens retention
    # (spaced repetition effect — accessed memories resist forgetting)
    if mem.access_count > 0:
        reinforcement = min(0.3, 0.05 * math.log(1 + mem.access_count))
        base_retention = min(1.0, base_retention + reinforcement)

    return base_retention


def should_forget(mem: Memory, config: Config) -> bool:
    """Should this memory be archived (soft-forgotten)?"""
    if mem.metadata.get("pinned"):
        return False
    if mem.layer == MemoryLayer.SEMANTIC:
        return False
    if mem.layer == MemoryLayer.PROCEDURAL:
        return False

    age_days = (time.time() - mem.created_at) / 86400
    lc = config.lifecycle

    # use retention score for a more nuanced decision
    retention = compute_retention(mem, config)

    return (
        age_days > lc.archive_after_days
        and retention < 0.15  # retention-based threshold
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
        if age_minutes > 30 or mem.access_count >= 2:
            return MemoryLayer.EPISODIC

    return None


def apply_forgetting_curve(store: Store, config: Config) -> dict:
    """Apply retention regularization to all episodic memories. Returns stats."""
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
            store.record_importance(mem.id, new_importance)
            stats["updated"] += 1

    return stats
