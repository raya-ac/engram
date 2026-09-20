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


class TestTemporalResolution:
    def test_parse_date_formats(self):
        from datetime import date, datetime
        from engram.retrieval import _parse_date

        assert _parse_date("2026/04/10") == date(2026, 4, 10)
        assert _parse_date("2026-04-10") == date(2026, 4, 10)
        assert _parse_date(date(2026, 4, 10)) == date(2026, 4, 10)
        assert _parse_date(datetime(2026, 4, 10, 15, 30)) == date(2026, 4, 10)
        assert _parse_date(None) is None
        assert _parse_date("invalid-date-format") is None

    def test_resolve_relative_patterns(self):
        from datetime import date, timedelta
        from engram.retrieval import _resolve_temporal_window

        ref = "2026/04/10"  # Friday
        ref_d = date(2026, 4, 10)

        # "3 days ago"
        win = _resolve_temporal_window("what did I do 3 days ago?", ref)
        assert win == (ref_d - timedelta(days=3), 2)

        # "four weeks ago"
        win = _resolve_temporal_window("milestone mentioned four weeks ago", ref)
        assert win == (ref_d - timedelta(weeks=4), 4)

        # "a week ago"
        win = _resolve_temporal_window("call from a week ago", ref)
        assert win == (ref_d - timedelta(weeks=1), 4)

        # "yesterday"
        win = _resolve_temporal_window("notes from yesterday", ref)
        assert win == (ref_d - timedelta(days=1), 1)

        # "last Monday" (ref is Friday, so last Monday was 4 days ago)
        win = _resolve_temporal_window("commit from last Monday", ref)
        assert win == (ref_d - timedelta(days=4), 2)

        # non-temporal
        assert _resolve_temporal_window("what is HNSW?", ref) is None

    def test_temporal_search_boosts_matching_window(self, store, config):
        from engram.store import Memory
        from engram.embeddings import embed_documents

        # add memories with different dates
        c_target = "client contract signed for branding project"
        c_distractor = "client contract signed for website design"
        vecs = embed_documents([c_target, c_distractor], config.embedding_model)

        m_target = Memory(
            id="mem_target",
            content=c_target,
            fact_date="2026-04-07",  # 3 days before 2026-04-10
            layer="episodic",
            embedding=vecs[0],
        )
        m_distractor = Memory(
            id="mem_distractor",
            content=c_distractor,
            fact_date="2026-01-10",  # 3 months before
            layer="episodic",
            embedding=vecs[1],
        )
        store.save_memory(m_target)
        store.save_memory(m_distractor)

        results = search(
            "client contract signed 3 days ago",
            store,
            config,
            top_k=2,
            rerank=False,
            reference_date="2026-04-10",
        )
        assert len(results) >= 1
        assert results[0].memory.id == "mem_target"

    def test_search_cache_with_reference_date(self, store_with_memories, config):
        search("auth bug", store_with_memories, config, top_k=5, rerank=False, reference_date="2026-04-10")
        cache_key = (
            "auth bug",
            "full_context",
            5,
            False,
            config.embedding_model,
            config.cross_encoder_model,
            "2026-04-10",
        )
        cached = store_with_memories.get_search_cache(cache_key)
        assert cached is not None

