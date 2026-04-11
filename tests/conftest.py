"""Shared fixtures for engram tests."""

import os
import tempfile
import uuid

import pytest

# force offline mode for tests
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


@pytest.fixture
def config():
    """Config with a temp DB path."""
    from engram.config import Config
    cfg = Config()
    cfg.db_path = os.path.join(tempfile.mkdtemp(), "test_memory.db")
    return cfg


@pytest.fixture
def store(config):
    """Initialized store with temp DB."""
    from engram.store import Store
    s = Store(config)
    s.init_db()
    yield s
    s.close()


@pytest.fixture
def store_with_memories(store, config):
    """Store with 10 sample memories embedded."""
    from engram.store import Memory
    from engram.embeddings import embed_documents

    contents = [
        "Ari prefers casual tone and directness",
        "The auth system was migrated to Clerk in January 2026",
        "Engram uses HNSW for approximate nearest neighbor search",
        "Bug bounty hunting on HackerOne, focus on web vulns",
        "Melee.garden is a competitive Super Smash Bros platform",
        "The dream cycle clusters similar memories and archives old ones",
        "Python 3.12 is required, MLX for Apple Silicon GPU embedding",
        "Voyage-3.5 is the recommended embedding model from Anthropic",
        "Entity graphs use recursive SQL CTEs for multi-hop traversal",
        "Cross-encoder reranking adds ~300ms but improves precision",
    ]

    vecs = embed_documents(contents, config.embedding_model)

    for content, vec in zip(contents, vecs):
        mem = Memory(
            id=str(uuid.uuid4()),
            content=content,
            layer="episodic",
            importance=0.7,
        )
        mem.embedding = vec
        store.save_memory(mem)

    return store
