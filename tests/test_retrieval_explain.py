"""Explanation contracts through a temporary store, without model inference."""

import copy
import json
import math
from unittest.mock import Mock

import pytest

import engram.retrieval as retrieval
from engram.config import Config
from engram.store import Memory, Store


@pytest.fixture
def case(tmp_path, monkeypatch):
    config = Config()
    config.storage_backend = "sqlite"
    config.db_path = str(tmp_path / "explain.sqlite")
    config.dormant_recall.mode = "off"
    config.retrieval.rerank_passage_fallback = False
    store = Store(config)
    store.init_db()
    dense, bm25, graph, hopfield = [], [], [], []
    scores = {}
    calls = []
    monkeypatch.setattr(retrieval, "_dense_search", lambda *_: list(dense))
    monkeypatch.setattr(retrieval, "_bm25_search", lambda *_: list(bm25))
    monkeypatch.setattr(retrieval, "_graph_search", lambda *_: list(graph))
    monkeypatch.setattr(retrieval, "_hopfield_search", lambda *_: list(hopfield))
    monkeypatch.setattr(retrieval, "RETRIEVAL_NOISE_SCALE", 0.0)

    def rerank(query, docs, model):
        calls.append((query, list(docs), model))
        return sorted(enumerate(scores[doc] for doc in docs), key=lambda item: -item[1])

    monkeypatch.setattr(retrieval, "cross_encoder_rerank", rerank)

    def add(mid, raw=2.0, *, content=None, **kwargs):
        content = content or f"Remembered evidence for {mid}."
        memory = Memory(id=mid, content=content, layer="semantic", memory_type="fact",
                        fact_date="2026-01-01", **kwargs)
        store.save_memory(memory)
        dense.append((mid, 0.95 - len(dense) * 0.01))
        scores[content] = raw
        return memory

    yield config, store, add, (dense, bm25, graph, hopfield), calls
    store.close()


def explain(config, store, **kwargs):
    results, dbg = retrieval.search("Where is the maintenance key?", store, config,
                                    debug=True, **kwargs)
    report = dbg.to_dict()
    json.dumps(report, allow_nan=False)
    return results, dbg, report, {row["memory_id"]: row for row in report["candidates"]}


def test_fact_below_confidence_is_visible_with_observed_score_and_reason(case):
    config, store, add, _, _ = case
    add("key", raw=-1.0, content="The maintenance key is in the amber drawer.")
    results, dbg, report, rows = explain(config, store)
    assert results == []
    row = rows["key"]
    assert row["content"] == "The maintenance key is in the amber drawer."
    assert row["outcome"] == "below_confidence"
    assert row["confidence"] == {"applied": True, "threshold": 0.6, "passed": False}
    assert row["scores"]["raw_logit"] == -1.0
    assert row["scores"]["final"] == pytest.approx(1 / (1 + math.e))
    assert "below the configured minimum 0.6" in row["reason"]
    assert "bm25" not in row["scores"]  # an absent signal is not a measured zero
    assert report["final_ids"] == []
    assert dbg.dense_candidates and dbg.rrf_scores  # legacy debug compatibility
    assert "not probabilities" in report["score_semantics"]["description"]


def test_confidence_gate_precedes_prior_coverage_and_top_k_explained(case):
    config, store, add, _, _ = case
    add("rejected-prior", -3.0)
    add("eligible-prior", 1.0)
    add("winner", 5.0)
    add("runner-up", 4.0)
    results, _, report, rows = explain(config, store, top_k=2)
    assert [r.memory.id for r in results] == ["winner", "eligible-prior"]
    assert report["final_ids"] == ["winner", "eligible-prior"]
    assert rows["rejected-prior"]["outcome"] == "below_confidence"
    assert rows["runner-up"]["outcome"] == "outside_top_k"
    kept = rows["eligible-prior"]
    assert kept["prior_coverage"]["promoted"] is True
    assert kept["ranks"]["before_coverage"] == 3
    assert kept["ranks"]["after_coverage"] == 2
    assert kept["scores"]["final"] == pytest.approx(1 / (1 + math.exp(-1)))
    assert "score is unchanged" in kept["reason"]
    assert report["counts"]["reranked"] == 4
    assert report["counts"]["confidence_eligible"] == 3


def test_bounded_union_includes_hopfield_and_budget_exclusions(case):
    config, store, add, channels, _ = case
    add("selected")
    add("outside-budget")
    channels[3].append(("outside-budget", 0.125))
    config.retrieval.rerank_candidates = 1
    # Add an independent signal to keep the first candidate atop RRF.
    channels[1].append(("selected", 2.0))
    _, dbg, report, rows = explain(config, store)
    assert report["counts"]["hopfield"] == 1
    assert dbg.hopfield_candidates == [("outside-budget", 0.125)]
    assert rows["outside-budget"]["scores"]["hopfield"] == 0.125
    assert rows["outside-budget"]["outcome"] == "outside_rerank_candidates"
    assert "raw_logit" not in rows["outside-budget"]["scores"]


def test_filtered_and_unavailable_candidates_never_expose_content(case):
    config, store, add, channels, calls = case
    add("active")
    add("forgotten", forgotten=True, content="PRIVATE forgotten text")
    add("inactive", status="archived", content="PRIVATE inactive text")
    narrative = add("profile", content="PRIVATE narrative text")
    store.conn.execute("UPDATE memories SET memory_type = 'narrative' WHERE id = ?", (narrative.id,))
    store.conn.commit()
    channels[0].append(("removed", 0.7))
    results, _, report, rows = explain(config, store, mode="facts_only")
    assert [r.memory.id for r in results] == ["active"]
    for mid, outcome in [("forgotten", "forgotten"), ("inactive", "inactive"),
                         ("profile", "profile_filtered"), ("removed", "unavailable")]:
        assert rows[mid]["eligible"] is False
        assert rows[mid]["outcome"] == outcome
        assert "content" not in rows[mid]
        assert "memory_type" not in rows[mid]
    assert "PRIVATE" not in json.dumps(report)
    assert calls[0][1] == ["Remembered evidence for active."]


def test_diagnostic_bypasses_cache_access_and_dormant_writes(case, monkeypatch):
    config, store, add, _, calls = case
    add("key")
    config.dormant_recall.mode = "shadow"
    shadow = Mock()
    monkeypatch.setattr("engram.dormant.evaluate_shadow", shadow)
    # Existing entries must be neither consumed nor replaced by diagnostics.
    store._search_cache[("sentinel",)] = (store._search_cache_version, [{"sentinel": True}])
    cache_before = copy.deepcopy(store._search_cache)
    before = list(store.conn.iterdump())
    for method in ("get_search_cache", "set_search_cache", "record_search"):
        monkeypatch.setattr(store, method, Mock(side_effect=AssertionError(f"called {method}")))
    _, _, report, _ = explain(config, store)
    assert len(calls) == 1
    assert list(store.conn.iterdump()) == before
    assert store._search_cache == cache_before
    shadow.assert_not_called()
    assert report["cache"]["bypassed"] is True
    assert report["side_effects"] == {"record_search": False, "dormant_evaluation": False}


def test_ordinary_search_retains_cache_access_and_shadow_behavior(case, monkeypatch):
    config, store, add, _, calls = case
    add("key")
    config.dormant_recall.mode = "shadow"
    shadow = Mock()
    monkeypatch.setattr("engram.dormant.evaluate_shadow", shadow)
    results = retrieval.search("key", store, config)
    assert [r.memory.id for r in results] == ["key"]
    assert store.get_memory("key").access_count == 1
    assert store._search_cache
    retrieval.search("key", store, config)
    assert len(calls) == 1
    assert store.get_memory("key").access_count == 2
    assert shadow.call_count == 2


def test_explanation_does_not_change_scores_or_selected_order(case):
    config, store, add, _, _ = case
    add("prior", 1.0)
    add("winner", 4.0)
    add("other", 3.0)
    diagnostic, dbg, report, _ = explain(config, store, top_k=2)
    ordinary = retrieval.search("Where is the maintenance key?", store, config, top_k=2)
    assert [(r.memory.id, r.score) for r in diagnostic] == [(r.memory.id, r.score) for r in ordinary]
    # Clients cannot mutate the debug object by editing a serialized response.
    report["candidates"][0]["reason"] = "changed by caller"
    assert "changed by caller" not in json.dumps(dbg.to_dict())


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("field,value,outcome", [
    ("forgotten", 1, "forgotten"),
    ("status", "superseded", "inactive"),
    ("memory_type", "narrative", "profile_filtered"),
])
def test_lifecycle_change_during_rerank_cannot_expose_old_content(case, monkeypatch, debug, field, value, outcome):
    config, store, add, _, _ = case
    add("key", 3.0, content="PRIVATE content before concurrent lifecycle change")
    def transition(*args):
        store.conn.execute(f"UPDATE memories SET {field} = ? WHERE id = 'key'", (value,))
        store.conn.commit()
        return [(0, 3.0)]
    monkeypatch.setattr(retrieval, "cross_encoder_rerank", transition)
    result = retrieval.search("key", store, config, mode="facts_only", debug=debug)
    if debug:
        results, dbg = result
        report = dbg.to_dict()
        assert report["final_ids"] == []
        assert report["counts"]["returned"] == 0
        assert report["candidates"][0]["outcome"] == outcome
        assert "PRIVATE" not in json.dumps(report)
        assert dbg.final_results == []
    else:
        results = result
    assert results == []
    assert store.get_memory("key").access_count == 0


@pytest.mark.parametrize("field,value", [("forgotten", 1), ("status", "superseded"),
                                         ("memory_type", "narrative")])
def test_old_cached_candidate_is_rechecked_without_exposing_ineligible_content(case, field, value):
    config, store, add, _, calls = case
    add("key", 3.0)
    assert retrieval.search("key", store, config, mode="facts_only")
    assert len(calls) == 1 and store._search_cache
    # Simulate a separate writer whose lifecycle change cannot invalidate this
    # process's in-memory cache directly.
    store.conn.execute(f"UPDATE memories SET {field} = ? WHERE id = 'key'", (value,))
    store.conn.commit()
    assert retrieval.search("key", store, config, mode="facts_only") == []
    assert len(calls) == 1
    assert store.get_memory("key").access_count == 1


def test_deep_reranker_order_and_omissions_are_explained_without_invented_scores(case, monkeypatch):
    config, store, add, _, _ = case
    add("winner", 4.0)
    add("other", 3.0)
    monkeypatch.setattr(retrieval, "embed_query", lambda *_: [1.0, 0.0])
    deep = Mock(is_trained=True)
    deep.rerank.return_value = [{"id": "other", "deep_score": 0.88}]
    results, _, report, rows = explain(config, store, top_k=2, deep_reranker=deep)
    assert [r.memory.id for r in results] == ["other"]
    assert rows["winner"]["outcome"] == "deep_reranker_excluded"
    assert "deep_reranker" not in rows["winner"]["scores"]
    assert rows["other"]["scores"]["deep_reranker"] == 0.88
    assert rows["other"]["scores"]["final"] == pytest.approx(1 / (1 + math.exp(-3)))
    assert report["settings"]["deep_reranker_applied"] is True


def test_empty_search_report_is_clean_and_nonsecret(case):
    config, store, _, _, calls = case
    config.postgres_dsn = "DO_NOT_EXPOSE_DSN"
    config.llm.api_key = "DO_NOT_EXPOSE_KEY"
    results, _, report, rows = explain(config, store)
    assert results == [] and rows == {} and calls == []
    assert report["counts"]["considered"] == 0
    assert report["counts"]["returned"] == 0
    assert report["final_ids"] == []
    assert "DO_NOT_EXPOSE" not in json.dumps(report)
    assert report["settings"]["cross_encoder_model"] == config.cross_encoder_model


def test_hosted_score_is_identified_without_inventing_a_logit(case):
    config, store, add, _, _ = case
    config.cross_encoder_model = "rerank-2.5"
    add("key", 0.8)
    _, _, report, rows = explain(config, store)
    scores = rows["key"]["scores"]
    assert "raw_logit" not in scores
    assert scores["normalized_model_score"] == scores["cross_encoder_calibrated"] == scores["final"] == 0.8
    assert report["score_semantics"]["model_output"] == "normalized_score"


def test_noise_without_reranking_reports_actual_adjustment(case, monkeypatch):
    config, store, add, _, calls = case
    add("key")
    monkeypatch.setattr(retrieval, "RETRIEVAL_NOISE_SCALE", 0.02)
    monkeypatch.setattr(retrieval.random, "gauss", lambda *_: -1.0)
    results, _, _, rows = explain(config, store, rerank=False)
    assert calls == []
    row = rows["key"]
    assert row["noise"]["sample"] == -1.0
    assert row["noise"]["before"] == row["scores"]["boosted"]
    assert row["noise"]["after"] == row["scores"]["final"] == results[0].score == 0.0
    assert row["confidence"]["applied"] is False
    assert "cross_encoder" not in row["scores"]


def test_passage_trace_and_temporal_boost_are_preserved_exactly(case, monkeypatch):
    config, store, add, _, _ = case
    add("key")
    config.retrieval.rerank_passage_fallback = True
    trace = {0: {"base_raw_score": -8.0, "excerpt_raw_score": -1.0,
                 "source_start": 10, "source_end": 30, "passage_words": 4}}
    monkeypatch.setattr(retrieval, "rerank_with_passages", lambda *a, **kw: ([(0, -1.0)], trace))
    _, _, _, rows = explain(config, store, reference_date="2026-01-02")
    row = rows["key"]
    assert row["passage"] == trace[0]
    assert row["scores"]["raw_logit"] == -1.0
    assert row["scores"]["temporal_boost"] == 0.0
    assert row["scores"]["final"] == pytest.approx(1 / (1 + math.e))


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"top_k": -1}, {"top_k": True},
                                   {"top_k": 1.5}, {"mode": "unknown"},
                                   {"mode": []}, {"reference_date": True},
                                   {"reference_date": "not a date"}])
def test_invalid_search_inputs_are_rejected_before_retrieval(case, kwargs):
    config, store, _, _, calls = case
    with pytest.raises(ValueError):
        explain(config, store, **kwargs)
    assert calls == []


@pytest.mark.parametrize("query", [None, "", " \n ", 12])
def test_invalid_query_is_rejected(case, query):
    config, store, _, _, _ = case
    with pytest.raises(ValueError, match="query"):
        retrieval.search(query, store, config, debug=True)
