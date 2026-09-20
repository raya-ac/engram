"""Model-free scoring checks through the production SQLite retrieval path."""

import math

import pytest


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('normalized', [False, True])
def test_nonfinite_model_scores_are_rejected(value, normalized):
    from engram.rerank_scoring import rerank_score
    with pytest.raises(ValueError, match='finite'):
        rerank_score(value, prior_rank=0, normalized=normalized)

import engram.retrieval as retrieval
from engram.config import Config
from engram.rerank_scoring import rerank_score
from engram.store import Memory, Store


@pytest.fixture
def search_case(tmp_path, monkeypatch):
    stores = []
    monkeypatch.setattr(retrieval.random, "gauss", lambda *_: 0.0)

    def make(records, *, model="BAAI/bge-reranker-base"):
        config = Config()
        config.storage_backend = "sqlite"
        config.db_path = str(tmp_path / f"case-{len(stores)}.sqlite")
        config.cross_encoder_model = model
        config.retrieval.min_confidence = 0.0
        config.dormant_recall.mode = "off"
        store = Store(config)
        store.init_db()
        stores.append(store)
        scores = {}
        for record in records:
            memory_id, content, logit, *fact_date = record
            store.save_memory(Memory(
                id=memory_id,
                content=content,
                fact_date=fact_date[0] if fact_date else "2026-01-01",
                layer="semantic",
            ))
            scores[content] = logit
        candidates = [(row[0], 1.0 - index * 0.01) for index, row in enumerate(records)]
        monkeypatch.setattr(retrieval, "_dense_search", lambda *_: candidates)
        monkeypatch.setattr(retrieval, "_bm25_search", lambda *_: candidates)
        monkeypatch.setattr(retrieval, "_graph_search", lambda *_: [])
        monkeypatch.setattr(retrieval, "_hopfield_search", lambda *_: [])
        calls = []

        def rerank(query, docs, model_name):
            calls.append((query, model_name))
            return sorted(enumerate(scores[doc] for doc in docs), key=lambda pair: -pair[1])

        monkeypatch.setattr(retrieval, "cross_encoder_rerank", rerank)
        return config, store, calls

    yield make
    for store in stores:
        store.close()


def test_default_production_ranking_has_no_lexical_or_prior_bonus(search_case):
    config, store, _ = search_case([
        ("literal", "chocolate chip cookies", -8.0),
        ("semantic", "a remembered baking preference", -7.9),
    ])
    results = retrieval.search("chocolate chip cookies", store, config, top_k=2)
    assert config.retrieval.rerank_fusion_alpha == 0.0
    assert [result.memory.id for result in results] == ["semantic", "literal"]
    assert [result.score for result in results] == pytest.approx([
        1.0 / (1.0 + math.exp(7.9)), 1.0 / (1.0 + math.exp(8.0)),
    ])


@pytest.mark.parametrize("raw_score", [-1000.0, -8.0, 0.0, 8.0, 1000.0])
def test_local_scores_are_bounded_and_default_ignores_prior(raw_score):
    first = rerank_score(raw_score, prior_rank=0)
    last = rerank_score(raw_score, prior_rank=99)
    assert 0.0 <= first <= 1.0
    assert first == last


def test_nonzero_fusion_changes_production_ranking_without_exceeding_one(search_case):
    config, store, _ = search_case([
        ("prior", "first candidate", -3.0),
        ("semantic", "second candidate", 3.0),
    ])
    config.retrieval.rerank_fusion_alpha = 1.0
    results = retrieval.search("preferences", store, config, top_k=2)
    assert [result.memory.id for result in results] == ["prior", "semantic"]
    assert [result.score for result in results] == pytest.approx([1.0, 0.5])


@pytest.mark.parametrize("alpha", [-0.01, 1.01, float("nan")])
def test_invalid_fusion_weight_rejected(alpha):
    with pytest.raises(ValueError, match="between 0 and 1"):
        rerank_score(1.0, prior_rank=0, fusion_alpha=alpha)


def test_hosted_scores_are_not_sigmoided_again_in_results_or_evidence(search_case):
    config, store, _ = search_case([
        ("high", "strong hosted result", 0.8),
        ("low", "weak hosted result", 0.2),
    ], model="rerank-2.5")
    results = retrieval.search("preferences", store, config, top_k=2)
    assert [result.score for result in results] == pytest.approx([0.8, 0.2])
    assert [result.sources["cross_encoder_calibrated"] for result in results] == pytest.approx([0.8, 0.2])


@pytest.mark.parametrize(
    "model,raw_score,expected_in,expected_out",
    [
        ("BAAI/bge-reranker-base", -5.0, 0.5, 1.0 / (1.0 + math.exp(5.0))),
        ("rerank-2.5", 0.5, 1.0 / (1.0 + math.exp(-5.0)), 0.5),
    ],
)
def test_temporal_evidence_applies_once_in_logit_space(
    search_case, model, raw_score, expected_in, expected_out,
):
    config, store, _ = search_case([
        ("in-window", "a dated preference", raw_score, "2026-09-19"),
        ("outside", "an earlier preference", raw_score, "2026-08-01"),
    ], model=model)
    results = retrieval.search(
        "preferences from yesterday", store, config, top_k=2, reference_date="2026-09-20",
    )
    by_id = {result.memory.id: result for result in results}
    assert by_id["in-window"].score == pytest.approx(expected_in)
    assert by_id["outside"].score == pytest.approx(expected_out)
    assert by_id["in-window"].sources["temporal_boost"] == 5.0
    assert by_id["outside"].sources["temporal_boost"] == 0.0


def test_cache_cannot_reuse_scores_after_fusion_changes(search_case):
    config, store, calls = search_case([
        ("prior", "first candidate", -3.0),
        ("semantic", "second candidate", 3.0),
    ])
    first = retrieval.search("preferences", store, config, top_k=2)
    assert first[0].memory.id == "semantic"
    config.retrieval.rerank_fusion_alpha = 1.0
    second = retrieval.search("preferences", store, config, top_k=2)
    assert [result.memory.id for result in second] == ["prior", "semantic"]
    assert [result.score for result in second] == pytest.approx([1.0, 0.5])
    assert len(calls) == 2


@pytest.mark.parametrize("before,after,expected_count", [(0.0, 0.8, 1), (0.8, 0.0, 2)])
def test_cache_cannot_reuse_results_after_confidence_changes(search_case, before, after, expected_count):
    config, store, _ = search_case([
        ("high", "strong result", math.log(0.9 / 0.1)),
        ("low", "moderate result", math.log(0.7 / 0.3)),
    ])
    config.retrieval.min_confidence = before
    retrieval.search("preferences", store, config, top_k=2)
    config.retrieval.min_confidence = after
    results = retrieval.search("preferences", store, config, top_k=2)
    assert len(results) == expected_count
    assert all(result.score >= after for result in results)


def test_random_noise_cannot_reshuffle_cross_encoder_results(search_case, monkeypatch):
    config, store, _ = search_case([
        ("stronger", "more relevant result", 2.0),
        ("weaker", "slightly less relevant result", 1.98),
    ])
    noise = iter([-0.02, 0.02])
    monkeypatch.setattr(retrieval.random, "gauss", lambda *_: next(noise))
    results = retrieval.search("preferences", store, config, top_k=2)
    assert [result.memory.id for result in results] == ["stronger", "weaker"]
    assert [result.score for result in results] == pytest.approx([
        1.0 / (1.0 + math.exp(-2.0)), 1.0 / (1.0 + math.exp(-1.98)),
    ])


def test_long_memory_fallback_can_recover_evidence_and_has_separate_cache(search_case, monkeypatch):
    content = "Ordinary unrelated context. " * 80 + "The maintenance key lives in the amber drawer."
    config, store, _ = search_case([("key", content, -8.0)])
    config.retrieval.min_confidence = 0.6
    config.retrieval.rerank_passage_fallback = False
    calls = []
    query = "Where is the maintenance key?"

    def score(original_query, documents, model_name):
        assert original_query == query
        calls.append(documents)
        return [(index, -8.0 if document == content else 3.0)
                for index, document in enumerate(documents)]

    monkeypatch.setattr(retrieval, "cross_encoder_rerank", score)
    assert retrieval.search(query, store, config, top_k=2) == []
    config.retrieval.rerank_passage_fallback = True
    results = retrieval.search(query, store, config, top_k=2)
    assert [result.memory.id for result in results] == ["key"]
    assert len(calls) == 3
    assert calls[0] == calls[1] == [content]
    assert "maintenance key" in calls[2][0]
    assert len(calls[2][0].split()) <= 160
    assert results[0].score == pytest.approx(1.0 / (1.0 + math.exp(-3.0)))
    assert results[0].sources["base_raw_score"] == -8.0
    assert results[0].sources["excerpt_raw_score"] == 3.0


@pytest.mark.parametrize(
    "limit,expected_ids",
    [(1, ["winner"]), (2, ["winner", "prior"]), (3, ["winner", "second", "prior"])],
)
def test_prior_coverage_keeps_model_winner_with_unchanged_scores(search_case, limit, expected_ids):
    logits = {"prior": 0.6, "winner": 4.0, "second": 3.0, "third": 2.0}
    config, store, _ = search_case([
        ("prior", "selected anchor canonical", logits["prior"]),
        ("winner", "model winner", logits["winner"]),
        ("second", "model second", logits["second"]),
        ("third", "model third", logits["third"]),
    ])
    assert config.retrieval.preserve_prior_candidate is True
    results = retrieval.search('"selected anchor"', store, config, top_k=limit)
    assert [result.memory.id for result in results] == expected_ids
    for result in results:
        logit = logits[result.memory.id]
        assert result.score == pytest.approx(1.0 / (1.0 + math.exp(-logit)))
        assert result.sources["cross_encoder"] == logit
        assert result.sources["temporal_boost"] == 0.0


def test_confidence_filter_precedes_prior_coverage_without_losing_a_slot(search_case):
    config, store, _ = search_case([
        ("rejected-prior", "first prior candidate", -3.0),
        ("eligible-prior", "second prior candidate", 1.0),
        ("winner", "model winner", 5.0),
        ("second", "model second", 4.0),
    ])
    config.retrieval.min_confidence = 0.7
    results = retrieval.search("preferences", store, config, top_k=2)
    assert [result.memory.id for result in results] == ["winner", "eligible-prior"]
    assert len(results) == 2
    assert all(result.score >= config.retrieval.min_confidence for result in results)
    assert [result.score for result in results] == pytest.approx([
        1.0 / (1.0 + math.exp(-5.0)), 1.0 / (1.0 + math.exp(-1.0)),
    ])


def test_disabling_prior_coverage_preserves_pure_model_order(search_case):
    config, store, _ = search_case([
        ("prior", "selected anchor canonical", 0.6),
        ("winner", "model winner", 4.0),
        ("second", "model second", 3.0),
        ("third", "model third", 2.0),
    ])
    config.retrieval.preserve_prior_candidate = False
    results = retrieval.search('"selected anchor"', store, config, top_k=3)
    assert [result.memory.id for result in results] == ["winner", "second", "third"]
    assert [result.score for result in results] == pytest.approx([
        1.0 / (1.0 + math.exp(-logit)) for logit in (4.0, 3.0, 2.0)
    ])


@pytest.mark.parametrize("enabled_before", [False, True])
def test_changing_prior_coverage_uses_a_distinct_cache_entry(search_case, enabled_before):
    config, store, calls = search_case([
        ("prior", "selected anchor canonical", 0.6),
        ("winner", "model winner", 4.0),
        ("second", "model second", 3.0),
        ("third", "model third", 2.0),
    ])
    config.retrieval.preserve_prior_candidate = enabled_before
    first = retrieval.search('"selected anchor"', store, config, top_k=2)
    config.retrieval.preserve_prior_candidate = not enabled_before
    second = retrieval.search('"selected anchor"', store, config, top_k=2)
    assert [result.memory.id for result in first] == ["winner", "prior" if enabled_before else "second"]
    assert [result.memory.id for result in second] == ["winner", "second" if enabled_before else "prior"]
    assert len(calls) == 2
    assert first[0].score == second[0].score


def test_passage_activation_floor_is_below_the_production_gate(search_case):
    content = "Ordinary unrelated context. " * 80 + "The copper valve needs a replacement seal."
    config, store, calls = search_case([("valve", content, -5.0)])
    config.retrieval.min_confidence = 0.6
    assert config.retrieval.rerank_passage_floor == 0.001

    # sigmoid(-5) is below the final gate but above the excerpt activation floor.
    assert retrieval.search("copper valve", store, config, top_k=1) == []
    assert len(calls) == 1


@pytest.mark.parametrize("gate,expected_count", [(0.0, 1), (0.6, 0)])
def test_passage_retry_and_final_confidence_gate_are_independent(search_case, monkeypatch, gate, expected_count):
    content = "Ordinary unrelated context. " * 80 + "The copper valve needs a replacement seal."
    config, store, _ = search_case([("valve", content, -8.0)])
    config.retrieval.min_confidence = gate
    config.retrieval.rerank_passage_floor = 0.001
    calls = []

    def score(query, documents, model_name):
        calls.append(documents)
        return [(index, -8.0 if document == content else 0.0)
                for index, document in enumerate(documents)]

    monkeypatch.setattr(retrieval, "cross_encoder_rerank", score)
    results = retrieval.search("copper valve", store, config, top_k=1)
    assert len(calls) == 2
    assert calls[0] == [content]
    assert len(calls[1][0].split()) <= 160
    assert len(results) == expected_count
    if results:
        # Max aggregation keeps the excerpt logit 0, rather than averaging it with -8.
        assert results[0].score == pytest.approx(0.5)
        assert results[0].sources["base_raw_score"] == -8.0
        assert results[0].sources["excerpt_raw_score"] == 0.0
        assert results[0].sources["cross_encoder"] == 0.0


def test_changing_passage_floor_invalidates_cached_rerank_results(search_case, monkeypatch):
    content = "Ordinary unrelated context. " * 80 + "The copper valve needs a replacement seal."
    config, store, _ = search_case([("valve", content, -5.0)])
    config.retrieval.min_confidence = 0.0
    config.retrieval.rerank_passage_floor = 0.001
    calls = []

    def score(query, documents, model_name):
        calls.append(documents)
        return [(index, -5.0 if document == content else 3.0)
                for index, document in enumerate(documents)]

    monkeypatch.setattr(retrieval, "cross_encoder_rerank", score)
    first = retrieval.search("copper valve", store, config, top_k=1)
    assert len(calls) == 1
    assert first[0].score == pytest.approx(1.0 / (1.0 + math.exp(5.0)))

    config.retrieval.rerank_passage_floor = 0.1
    second = retrieval.search("copper valve", store, config, top_k=1)
    assert len(calls) == 3
    assert calls[0] == calls[1] == [content]
    assert second[0].score == pytest.approx(1.0 / (1.0 + math.exp(-3.0)))
