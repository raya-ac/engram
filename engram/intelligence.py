"""Higher-level memory intelligence helpers for web and MCP surfaces."""

from __future__ import annotations

import collections
import time
from typing import Any

from engram.retrieval import search as hybrid_search
from engram.store import Store


def _trim(text: str, limit: int = 180) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _entity_counts_for_memory_ids(store: Store, memory_ids: list[str]) -> collections.Counter[str]:
    if not memory_ids:
        return collections.Counter()
    placeholders = ",".join("?" * len(memory_ids))
    rows = store.conn.execute(
        f"""SELECT e.canonical_name
            FROM entity_mentions em
            JOIN entities e ON e.id = em.entity_id
            WHERE em.memory_id IN ({placeholders})""",
        memory_ids,
    ).fetchall()
    return collections.Counter(r["canonical_name"] for r in rows if r["canonical_name"])


def _memory_payload(memory) -> dict[str, Any]:
    return {
        "id": memory.id,
        "content": _trim(memory.content),
        "layer": memory.layer,
        "memory_type": memory.memory_type,
        "status": memory.status,
        "importance": round(memory.importance or 0.0, 3),
        "created_at": memory.created_at,
        "fact_date": memory.fact_date,
    }


def build_query_brief(query: str, store: Store, config, top_k: int = 8) -> dict[str, Any]:
    results = hybrid_search(query, store, config, top_k=top_k, mode="full_context")
    memories = [r.memory for r in results]
    memory_ids = [m.id for m in memories]

    layer_counts = collections.Counter(m.layer for m in memories)
    type_counts = collections.Counter(m.memory_type for m in memories)
    status_counts = collections.Counter(m.status for m in memories)
    source_counts = collections.Counter(m.source_type for m in memories)
    entity_counts = _entity_counts_for_memory_ids(store, memory_ids)

    suggested_queries: list[str] = []
    for name, _count in entity_counts.most_common(4):
        suggested_queries.append(f"recent work involving {name}")
    for layer, _count in layer_counts.most_common(2):
        suggested_queries.append(f"{query} in {layer}")
    if memories:
        suggested_queries.append(f"timeline for {query}")

    follow_ups: list[str] = []
    seen = set()
    for item in suggested_queries:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        follow_ups.append(item)

    top_entities = [{"name": name, "count": count} for name, count in entity_counts.most_common(6)]
    top_layers = [{"name": name, "count": count} for name, count in layer_counts.most_common()]
    top_types = [{"name": name, "count": count} for name, count in type_counts.most_common()]
    top_statuses = [{"name": name, "count": count} for name, count in status_counts.most_common()]
    top_sources = [{"name": name, "count": count} for name, count in source_counts.most_common()]

    start_at = min((m.created_at for m in memories), default=None)
    end_at = max((m.created_at for m in memories), default=None)
    highest = max(memories, key=lambda m: m.importance, default=None)

    summary_parts = [f"Retrieved {len(memories)} memories for “{query}”."]
    if top_layers:
        summary_parts.append(
            "Most of the weight sits in "
            + ", ".join(f"{item['name']} ({item['count']})" for item in top_layers[:3])
            + "."
        )
    if top_entities:
        summary_parts.append(
            "Dominant entities: "
            + ", ".join(f"{item['name']} ({item['count']})" for item in top_entities[:4])
            + "."
        )
    if highest:
        summary_parts.append(f"Highest-importance memory: {_trim(highest.content, 110)}")

    return {
        "query": query,
        "summary": " ".join(summary_parts),
        "result_count": len(memories),
        "time_window": {"start": start_at, "end": end_at},
        "top_layers": top_layers,
        "top_types": top_types,
        "top_statuses": top_statuses,
        "top_sources": top_sources,
        "top_entities": top_entities,
        "key_memories": [_memory_payload(m) for m in memories[:6]],
        "suggested_queries": follow_ups[:6],
    }


def compare_queries(query_a: str, query_b: str, store: Store, config, top_k: int = 8) -> dict[str, Any]:
    left = hybrid_search(query_a, store, config, top_k=top_k, mode="full_context")
    right = hybrid_search(query_b, store, config, top_k=top_k, mode="full_context")

    left_memories = [r.memory for r in left]
    right_memories = [r.memory for r in right]
    left_ids = {m.id for m in left_memories}
    right_ids = {m.id for m in right_memories}

    overlap_ids = left_ids & right_ids
    only_left_ids = left_ids - right_ids
    only_right_ids = right_ids - left_ids

    left_entities = _entity_counts_for_memory_ids(store, list(left_ids))
    right_entities = _entity_counts_for_memory_ids(store, list(right_ids))

    shared_entities = left_entities & right_entities
    left_only_entities = left_entities - right_entities
    right_only_entities = right_entities - left_entities

    left_by_id = {m.id: m for m in left_memories}
    right_by_id = {m.id: m for m in right_memories}

    return {
        "query_a": query_a,
        "query_b": query_b,
        "overlap": {
            "memory_count": len(overlap_ids),
            "entity_count": len(shared_entities),
            "memories": [_memory_payload(left_by_id[mid]) for mid in list(overlap_ids)[:6]],
            "entities": [{"name": name, "count": count} for name, count in shared_entities.most_common(6)],
        },
        "only_a": {
            "memory_count": len(only_left_ids),
            "entity_count": len(left_only_entities),
            "memories": [_memory_payload(left_by_id[mid]) for mid in list(only_left_ids)[:6]],
            "entities": [{"name": name, "count": count} for name, count in left_only_entities.most_common(6)],
        },
        "only_b": {
            "memory_count": len(only_right_ids),
            "entity_count": len(right_only_entities),
            "memories": [_memory_payload(right_by_id[mid]) for mid in list(only_right_ids)[:6]],
            "entities": [{"name": name, "count": count} for name, count in right_only_entities.most_common(6)],
        },
    }


def activity_hotspots(store: Store, hours: float = 72.0, limit: int = 8) -> dict[str, Any]:
    since = time.time() - max(hours, 1.0) * 3600
    rows = store.conn.execute(
        """SELECT * FROM memories
           WHERE forgotten = 0 AND (created_at >= ? OR last_accessed >= ?)
           ORDER BY COALESCE(last_accessed, created_at) DESC
           LIMIT ?""",
        (since, since, max(limit * 8, 32)),
    ).fetchall()
    memories = [store._row_to_memory(row) for row in rows]
    memory_ids = [m.id for m in memories]

    entity_counts = _entity_counts_for_memory_ids(store, memory_ids)
    layer_counts = collections.Counter(m.layer for m in memories)
    source_counts = collections.Counter(m.source_type for m in memories)
    type_counts = collections.Counter(m.memory_type for m in memories)

    hot_memories = sorted(
        memories,
        key=lambda m: ((m.importance or 0.0) * 2.0) + (m.access_count or 0) * 0.15 + ((m.last_accessed or 0) - since) / 3600.0,
        reverse=True,
    )[:limit]

    bucket_counts = collections.Counter()
    for m in memories:
        ts = m.last_accessed or m.created_at
        bucket = time.strftime("%Y-%m-%d %H:00", time.localtime(ts))
        bucket_counts[bucket] += 1

    return {
        "hours": hours,
        "memory_count": len(memories),
        "hot_entities": [{"name": name, "count": count} for name, count in entity_counts.most_common(limit)],
        "hot_layers": [{"name": name, "count": count} for name, count in layer_counts.most_common(limit)],
        "hot_sources": [{"name": name, "count": count} for name, count in source_counts.most_common(limit)],
        "hot_types": [{"name": name, "count": count} for name, count in type_counts.most_common(limit)],
        "hot_memories": [_memory_payload(m) for m in hot_memories],
        "activity_buckets": [{"bucket": key, "count": value} for key, value in bucket_counts.most_common(limit)],
    }
