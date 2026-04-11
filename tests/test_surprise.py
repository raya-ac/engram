"""Tests for surprise.py — novelty scoring."""

import numpy as np
import pytest

from engram.embeddings import embed_query
from engram.surprise import compute_surprise, adjust_importance


class TestSurprise:
    def test_novel_content(self, store_with_memories):
        vec = embed_query("completely unrelated topic about underwater basket weaving")
        result = compute_surprise(vec, store_with_memories)
        assert result["surprise"] > 0.5
        assert result["importance_modifier"] > 0

    def test_duplicate_content(self, store_with_memories):
        # embed something similar to an existing memory
        vec = embed_query("Engram uses HNSW for approximate nearest neighbor search")
        result = compute_surprise(vec, store_with_memories)
        assert result["surprise"] < 0.7  # should be lower than a novel query
        assert result["nearest_id"] is not None

    def test_empty_store(self, store):
        vec = embed_query("anything")
        result = compute_surprise(vec, store)
        assert result["surprise"] == 1.0
        assert result["nearest_id"] is None

    def test_fields_present(self, store_with_memories):
        vec = embed_query("test")
        result = compute_surprise(vec, store_with_memories)
        for field in ["surprise", "nearest_distance", "nearest_id", "is_duplicate",
                       "k_distances", "importance_modifier"]:
            assert field in result


class TestAdjustImportance:
    def test_novel_boosts(self):
        result = adjust_importance(0.5, {"importance_modifier": 0.3})
        assert result == 0.8

    def test_duplicate_reduces(self):
        result = adjust_importance(0.5, {"importance_modifier": -0.3})
        assert result == 0.2

    def test_clamped_high(self):
        result = adjust_importance(0.9, {"importance_modifier": 0.3})
        assert result == 1.0

    def test_clamped_low(self):
        result = adjust_importance(0.1, {"importance_modifier": -0.3})
        assert result == 0.05  # floor
