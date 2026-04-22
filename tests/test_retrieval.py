"""Tests for retrieval.py — hybrid search pipeline."""

import pytest

from engram.retrieval import search, classify_intent, RetrievalResult


class TestIntentClassification:
    @pytest.mark.parametrize("query,expected", [
        ("why did the deploy fail", "why"),
        ("when was the last release", "when"),
        ("who built the auth module", "who"),
        ("how to fix the login bug", "how"),
        ("what is engram", "what"),
        ("show me the code", "what"),
    ])
    def test_classify(self, query, expected):
        assert classify_intent(query) == expected


class TestSearch:
    def test_search_returns_results(self, store_with_memories, config):
        results = search("HNSW nearest neighbor", store_with_memories, config, top_k=5, rerank=False)
        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)
        assert results[0].memory is not None
        assert results[0].score > 0

    def test_search_relevance(self, store_with_memories, config):
        results = search("embedding model from Anthropic", store_with_memories, config, top_k=3, rerank=False)
        # Voyage memory should be in top results
        contents = [r.memory.content for r in results]
        assert any("Voyage" in c for c in contents)

    def test_search_with_debug(self, store_with_memories, config):
        results, dbg = search("test query", store_with_memories, config, top_k=5, rerank=False, debug=True)
        assert dbg is not None
        assert dbg.latency_ms > 0
        assert len(dbg.dense_candidates) > 0

    def test_search_debug_surfaces_query_features(self, store_with_memories, config):
        results, dbg = search('"auth" bug', store_with_memories, config, top_k=5, rerank=False, debug=True)
        assert results is not None
        assert dbg.intent == "what"
        assert "auth" in dbg.phrase_terms
        assert "authentication" in dbg.expanded_terms
        assert dbg.cache_hit is False

    def test_search_populates_cache_for_repeat_queries(self, store_with_memories, config):
        search("auth bug", store_with_memories, config, top_k=5, rerank=False)
        cache_key = (
            "auth bug",
            "full_context",
            5,
            False,
            config.embedding_model,
            config.cross_encoder_model,
        )
        cached = store_with_memories.get_search_cache(cache_key)
        assert cached is not None
        assert len(cached) > 0

    def test_search_records_access(self, store_with_memories, config):
        results = search("HNSW", store_with_memories, config, top_k=1, rerank=False)
        if results:
            mem = store_with_memories.get_memory(results[0].memory.id)
            assert mem.access_count > 0

    def test_search_sources(self, store_with_memories, config):
        results = search("test", store_with_memories, config, top_k=1, rerank=False)
        if results:
            sources = results[0].sources
            assert "dense" in sources
            assert "bm25" in sources
            assert "rrf" in sources

    def test_empty_store(self, store, config):
        results = search("anything", store, config, top_k=5, rerank=False)
        assert results == []
