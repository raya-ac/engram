"""Tests for store.py — CRUD, FTS, entities, events."""

import time
import uuid
import numpy as np
import pytest

from engram.store import Store, Memory, Entity, Relationship, MemoryLayer


class TestMemoryCRUD:
    def test_save_and_get(self, store):
        mem = Memory(id=str(uuid.uuid4()), content="test memory", layer="episodic")
        mem.embedding = np.random.randn(384).astype(np.float32)
        store.save_memory(mem)

        loaded = store.get_memory(mem.id)
        assert loaded is not None
        assert loaded.content == "test memory"
        assert loaded.layer == "episodic"
        assert loaded.embedding is not None

    def test_forget(self, store):
        mem = Memory(id=str(uuid.uuid4()), content="forgettable", layer="episodic")
        store.save_memory(mem)
        store.forget_memory(mem.id)

        loaded = store.get_memory(mem.id)
        assert loaded.forgotten is True

    def test_update_layer(self, store):
        mem = Memory(id=str(uuid.uuid4()), content="promotable", layer="episodic")
        store.save_memory(mem)
        store.update_layer(mem.id, "semantic")

        loaded = store.get_memory(mem.id)
        assert loaded.layer == "semantic"

    def test_count_memories(self, store_with_memories):
        counts = store_with_memories.count_memories()
        assert counts["total"] == 10
        assert counts.get("episodic", 0) == 10

    def test_recent_memories_sorted(self, store_with_memories):
        recent = store_with_memories.get_recent_memories(5)
        assert len(recent) == 5
        times = [m.created_at for m in recent]
        assert times == sorted(times, reverse=True)

    def test_access_tracking(self, store_with_memories):
        mem = store_with_memories.get_recent_memories(1)[0]
        old_count = mem.access_count
        store_with_memories.record_access(mem.id, "test query")
        updated = store_with_memories.get_memory(mem.id)
        assert updated.access_count == old_count + 1


class TestFTS:
    def test_search_fts(self, store_with_memories):
        results = store_with_memories.search_fts("engram HNSW", limit=10)
        assert len(results) > 0
        # should find the HNSW memory
        ids = [mid for mid, _ in results]
        contents = [store_with_memories.get_memory(mid).content for mid in ids]
        assert any("HNSW" in c for c in contents)

    def test_empty_query(self, store):
        results = store.search_fts("", limit=10)
        assert results == []


class TestEntities:
    def test_save_and_find_entity(self, store):
        entity = Entity(
            id=str(uuid.uuid4()),
            canonical_name="TestEntity",
            entity_type="concept",
            first_seen=1000.0,
            last_seen=2000.0,
        )
        store.save_entity(entity)
        found = store.find_entity_by_name("TestEntity")
        assert found is not None
        assert found.canonical_name == "TestEntity"

    def test_find_by_alias(self, store):
        entity = Entity(
            id=str(uuid.uuid4()),
            canonical_name="MainName",
            aliases=["Alias1", "Alias2"],
            entity_type="person",
            first_seen=1000.0,
            last_seen=2000.0,
        )
        store.save_entity(entity)
        found = store.find_entity_by_name("alias1")
        assert found is not None
        assert found.canonical_name == "MainName"

    def test_entity_memory_link(self, store):
        entity = Entity(id=str(uuid.uuid4()), canonical_name="LinkedEntity",
                        first_seen=0, last_seen=0)
        mem = Memory(id=str(uuid.uuid4()), content="linked content", layer="episodic")
        store.save_entity(entity)
        store.save_memory(mem)
        store.link_entity_memory(entity.id, mem.id)

        mems = store.get_entity_memories(entity.id)
        assert len(mems) == 1
        assert mems[0].id == mem.id


class TestEmbeddingCache:
    def test_cache_invalidation(self, store_with_memories):
        ids1, _ = store_with_memories.get_all_embeddings()
        assert len(ids1) == 10

        # add a new memory — cache should invalidate
        mem = Memory(id=str(uuid.uuid4()), content="new", layer="episodic")
        mem.embedding = np.random.randn(384).astype(np.float32)
        store_with_memories.save_memory(mem)

        ids2, _ = store_with_memories.get_all_embeddings()
        assert len(ids2) == 11

    def test_hot_cache(self, store_with_memories):
        import time
        # cold
        store_with_memories._embedding_cache = None
        t0 = time.time()
        store_with_memories.get_all_embeddings()
        cold = time.time() - t0

        # hot
        t0 = time.time()
        store_with_memories.get_all_embeddings()
        hot = time.time() - t0

        assert hot < cold


class TestDiary:
    def test_write_and_read(self, store):
        store.write_diary("test entry")
        entries = store.get_diary(limit=1)
        assert len(entries) == 1
        assert entries[0]["text"] == "test entry"

    def test_session_handoff_roundtrip(self, store):
        metadata = {
            "session_id": "sess-1",
            "summary": "Current state summary",
            "open_loops": ["Finish MCP resume flow"],
            "decisions": [{"content": "Use structured handoff packets"}],
        }
        store.save_session_handoff("sess-1", "Current state summary", metadata)

        handoff = store.get_session_handoff("sess-1")
        assert handoff is not None
        assert handoff["session_id"] == "sess-1"
        assert handoff["summary"] == "Current state summary"
        assert handoff["metadata"]["open_loops"] == ["Finish MCP resume flow"]

    def test_session_handoff_list_sorted(self, store):
        store.save_session_handoff("older", "older summary", {"summary": "older"})
        time.sleep(0.01)
        store.save_session_handoff("newer", "newer summary", {"summary": "newer"})

        handoffs = store.list_session_handoffs(limit=2)
        assert [item["session_id"] for item in handoffs] == ["newer", "older"]

    def test_search_cache_invalidates_after_write(self, store):
        key = ("query", "full_context", 5, False, "model-a", "model-b")
        store.set_search_cache(key, [{"memory_id": "cached", "score": 0.9, "sources": {}}])
        assert store.get_search_cache(key) is not None

        mem = Memory(id=str(uuid.uuid4()), content="new write", layer="episodic")
        store.save_memory(mem)

        assert store.get_search_cache(key) is None


class TestEvents:
    def test_events_logged(self, store):
        mem = Memory(id=str(uuid.uuid4()), content="event test", layer="episodic")
        store.save_memory(mem)
        events = store.get_recent_events(limit=5)
        assert any(e["event_type"] == "memory_write" for e in events)
