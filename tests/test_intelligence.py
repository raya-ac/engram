"""Tests for higher-level intelligence helpers."""

import time
import uuid

import numpy as np

from engram.intelligence import build_query_brief, compare_queries, activity_hotspots
from engram.store import Entity, Memory


def test_build_query_brief_surfaces_key_sections(store_with_memories, config):
    result = build_query_brief("Engram search", store_with_memories, config, top_k=5)

    assert result["query"] == "Engram search"
    assert result["result_count"] > 0
    assert isinstance(result["summary"], str)
    assert len(result["key_memories"]) > 0
    assert "top_layers" in result
    assert "suggested_queries" in result


def test_compare_queries_returns_overlap_and_unique_sections(store_with_memories, config):
    result = compare_queries(
        "Engram nearest neighbor",
        "Engram reranking precision",
        store_with_memories,
        config,
        top_k=5,
    )

    assert result["query_a"] == "Engram nearest neighbor"
    assert result["query_b"] == "Engram reranking precision"
    assert set(result.keys()) >= {"overlap", "only_a", "only_b"}
    assert "memory_count" in result["overlap"]
    assert "memories" in result["only_a"]
    assert "entities" in result["only_b"]


def test_activity_hotspots_surfaces_recent_entities(store, config):
    entity = Entity(
        id=str(uuid.uuid4()),
        canonical_name="Engram",
        entity_type="concept",
        first_seen=time.time(),
        last_seen=time.time(),
    )
    store.save_entity(entity)

    for idx in range(3):
        mem = Memory(
            id=str(uuid.uuid4()),
            content=f"Engram hotspot memory {idx}",
            layer="semantic" if idx % 2 == 0 else "episodic",
            importance=0.8,
            created_at=time.time(),
            last_accessed=time.time(),
        )
        mem.embedding = np.random.randn(384).astype(np.float32)
        store.save_memory(mem)
        store.link_entity_memory(entity.id, mem.id)

    result = activity_hotspots(store, hours=24, limit=5)

    assert result["memory_count"] >= 3
    assert len(result["hot_memories"]) > 0
    assert any(item["name"] == "Engram" for item in result["hot_entities"])
    assert any(item["name"] in {"semantic", "episodic"} for item in result["hot_layers"])
