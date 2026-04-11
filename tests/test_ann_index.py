"""Tests for ann_index.py — HNSW index operations."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from engram.ann_index import ANNIndex


@pytest.fixture
def index():
    return ANNIndex(dim=384, m=16, ef_construction=100, ef_search=50)


@pytest.fixture
def populated_index(index):
    n = 100
    ids = [f"mem-{i}" for i in range(n)]
    vecs = np.random.randn(n, 384).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index.build(ids, vecs)
    return index, ids, vecs


class TestBuild:
    def test_build(self, populated_index):
        index, ids, _ = populated_index
        assert index.ready
        assert index.count == 100

    def test_build_empty(self, index):
        index.build([], np.array([]))
        assert index.ready
        assert index.count == 0


class TestSearch:
    def test_search_returns_results(self, populated_index):
        index, _, _ = populated_index
        query = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query)
        results = index.search(query, top_k=10)
        assert len(results) == 10
        assert all(isinstance(mid, str) for mid, _ in results)
        assert all(isinstance(s, float) for _, s in results)

    def test_search_self(self, populated_index):
        index, ids, vecs = populated_index
        results = index.search(vecs[0], top_k=1)
        assert results[0][0] == ids[0]
        assert results[0][1] > 0.99  # cosine similarity ~1.0 for self

    def test_search_top_k_capped(self, populated_index):
        index, _, _ = populated_index
        query = np.random.randn(384).astype(np.float32)
        results = index.search(query, top_k=200)
        assert len(results) == 100  # can't return more than we have

    def test_search_not_ready(self, index):
        query = np.random.randn(384).astype(np.float32)
        results = index.search(query, top_k=10)
        assert results == []


class TestAddRemove:
    def test_add(self, populated_index):
        index, _, _ = populated_index
        vec = np.random.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        index.add("new-mem", vec)
        assert index.count == 101

        results = index.search(vec, top_k=1)
        assert results[0][0] == "new-mem"

    def test_remove(self, populated_index):
        index, ids, _ = populated_index
        index.remove(ids[0])
        assert index.count == 99

        # removed memory should not appear in results
        query = np.ones(384, dtype=np.float32)
        results = index.search(query, top_k=100)
        result_ids = {mid for mid, _ in results}
        assert ids[0] not in result_ids

    def test_add_duplicate_replaces(self, populated_index):
        index, ids, _ = populated_index
        new_vec = np.random.randn(384).astype(np.float32)
        new_vec /= np.linalg.norm(new_vec)
        index.add(ids[0], new_vec)
        assert index.count == 100  # count unchanged

    def test_remove_nonexistent(self, populated_index):
        index, _, _ = populated_index
        index.remove("nonexistent-id")  # should not error
        assert index.count == 100


class TestPersistence:
    def test_save_and_load(self, populated_index):
        index, ids, vecs = populated_index
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.index"
            index.index_path = path
            index.save(path)

            assert path.exists()
            assert path.with_suffix(".meta.json").exists()

            new_index = ANNIndex(dim=384, index_path=str(path))
            assert new_index.load(path)
            assert new_index.ready
            assert new_index.count == 100

            # verify search still works
            results = new_index.search(vecs[0], top_k=1)
            assert results[0][0] == ids[0]


class TestRecall:
    def test_recall_at_10(self, populated_index):
        """HNSW should have perfect recall on 100 vectors."""
        index, ids, vecs = populated_index
        perfect = 0
        for i in range(len(ids)):
            results = index.search(vecs[i], top_k=10)
            if results[0][0] == ids[i]:
                perfect += 1
        recall = perfect / len(ids)
        assert recall >= 0.95
