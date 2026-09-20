"""Synthetic score callbacks exercise bounded fallback without model inference."""

import math

import pytest

from engram.rerank_passages import rerank_with_passages, _query_passage


def test_repeated_unrelated_neighbor_does_not_dilute_complete_fact():
    filler = "The workshop has clean windows and freshly painted walls. "
    fact = "The spare compass is stored in the violet locker."
    document = filler * 30 + fact
    passage, start, end = _query_passage("Where is the spare compass stored?", document)
    assert passage == fact
    assert document[start:end] == fact


@pytest.mark.parametrize("context", [
    "This is an unverified proposal, not a confirmed decision.",
    "This draft was rejected during the earlier review.",
    "The previous information is incorrect and superseded.",
])
def test_repeated_qualifying_neighbor_is_retained(context):
    fact = "The spare compass is stored in the violet locker."
    document = (context + " ") * 30 + fact
    passage, _, _ = _query_passage("Where is the spare compass stored?", document)
    assert passage == context + " " + fact


def test_unique_neighbor_and_incomplete_query_coverage_keep_context():
    filler = "The workshop has clean windows and freshly painted walls. "
    context = "A volunteer added the following detail during the inspection. "
    fact = "The spare compass is stored in the violet locker."
    passage, _, _ = _query_passage("Where is the spare compass stored?", filler * 30 + context + fact)
    assert passage == context + fact
    passage, _, _ = _query_passage("Where is the nautical compass stored?", filler * 30 + fact)
    assert passage == filler + fact


def long_document(subject="valve"):
    return ("unrelated introduction. " * 90 +
            f"A technician began the inspection. The copper {subject} failed. She replaced the damaged seal." +
            " unrelated appendix." * 90)


@pytest.mark.parametrize("floor", [0.0, -0.1, 0.6])
def test_confident_or_disabled_retry_uses_only_full_documents(floor):
    calls = []
    documents = [long_document()]

    def rerank(query, docs, model):
        calls.append((query, docs, model))
        return [(0, 2.0 if floor > 0 else -4.0)]

    result, trace = rerank_with_passages("valve", documents, "local", rerank_fn=rerank, confidence_floor=floor)
    assert len(calls) == 1
    assert calls[0][1] is documents
    assert result == [(0, 2.0 if floor > 0 else -4.0)]
    assert "excerpt_raw_score" not in trace[0]


def test_exact_confidence_boundary_does_not_retry():
    calls = []

    def rerank(*args):
        calls.append(args)
        return [(0, 0.0)]

    assert rerank_with_passages("valve", [long_document()], "local", rerank_fn=rerank, confidence_floor=0.5)[0] == [(0, 0.0)]
    assert len(calls) == 1


def test_one_confident_candidate_disables_retry_for_the_whole_batch():
    calls = []

    def rerank(*args):
        calls.append(args)
        return [(0, -5.0), (1, 2.0)]

    result, trace = rerank_with_passages("valve", [long_document(), long_document()], "local", rerank_fn=rerank)
    assert result == [(1, 2.0), (0, -5.0)]
    assert len(calls) == 1
    assert all("excerpt_raw_score" not in evidence for evidence in trace.values())


def test_hosted_scores_never_take_local_passage_path():
    calls = []

    def rerank(*args):
        calls.append(args)
        return [(0, 0.001)]

    result, trace = rerank_with_passages("valve", [long_document()], "rerank-2.5", rerank_fn=rerank)
    assert result == [(0, 0.001)]
    assert trace == {0: {"base_raw_score": 0.001}}
    assert len(calls) == 1


def test_original_evidence_is_never_lowered_by_excerpt_score():
    calls = []

    def rerank(query, docs, model):
        calls.append(docs)
        return [(0, -4.0), (1, -3.0)] if len(calls) == 1 else [(0, -5.0), (1, 2.0)]

    result, trace = rerank_with_passages("valve", [long_document(), long_document()], "local", rerank_fn=rerank)
    assert len(calls) == 2
    assert result == [(1, 2.0), (0, -4.0)]
    assert trace[0]["base_raw_score"] == -4.0
    assert trace[0]["excerpt_raw_score"] == -5.0


def test_subset_indices_map_back_to_original_documents_and_query_is_preserved():
    documents = ["short valve note", long_document(), long_document("gasket").replace("failed", "passed"), long_document()]
    calls = []
    query = "Which valve failed last Tuesday?"

    def rerank(semantic_query, docs, model):
        calls.append((semantic_query, docs, model))
        return [(3, -4.0), (1, -5.0), (2, -6.0), (0, -7.0)] if len(calls) == 1 else [(1, 1.5), (0, -4.5)]

    result, trace = rerank_with_passages(query, documents, "local", rerank_fn=rerank, passage_query="Which valve failed?")
    assert result == [(3, 1.5), (1, -4.5), (2, -6.0), (0, -7.0)]
    assert [call[0] for call in calls] == [query, query]
    assert len(calls[1][1]) == 2
    assert set(index for index, item in trace.items() if "excerpt_raw_score" in item) == {1, 3}


def test_short_and_no_overlap_documents_are_not_rescored():
    documents = ["valve " * 160, long_document("gasket")]
    calls = []

    def rerank(*args):
        calls.append(args)
        return [(1, -4.0), (0, -4.0)]

    result, trace = rerank_with_passages("valve", documents, "local", rerank_fn=rerank)
    assert len(calls) == 1
    assert result == [(0, -4.0), (1, -4.0)]  # Ties use original input position.
    assert trace == {1: {"base_raw_score": -4.0}, 0: {"base_raw_score": -4.0}}


def test_excerpt_offsets_preserve_context_without_copying_content_into_trace():
    document = long_document()
    calls = []

    def rerank(query, docs, model):
        calls.append(docs)
        return [(0, -4.0)]

    _, trace = rerank_with_passages("valve", [document], "local", rerank_fn=rerank)
    evidence = trace[0]
    excerpt = document[evidence["source_start"]:evidence["source_end"]]
    assert excerpt == calls[1][0]
    assert excerpt == "A technician began the inspection. The copper valve failed. She replaced the damaged seal."
    assert evidence["passage_words"] <= 160
    assert all(not isinstance(value, str) for value in evidence.values())


def test_merged_ties_use_original_input_order():
    calls = []

    def rerank(*args):
        calls.append(args)
        return [(1, -4.0), (0, -5.0)] if len(calls) == 1 else [(1, -6.0), (0, -4.0)]

    result, _ = rerank_with_passages("valve", [long_document(), long_document()], "local", rerank_fn=rerank)
    assert result == [(0, -4.0), (1, -4.0)]


def test_oversized_sentence_keeps_query_hit_within_word_cap():
    document = " ".join(["filler"] * 250 + ["valve"] + ["filler"] * 250) + "."
    calls = []

    def rerank(query, docs, model):
        calls.append(docs)
        return [(0, -4.0)]

    _, trace = rerank_with_passages("valve", [document], "local", rerank_fn=rerank)
    assert trace[0]["passage_words"] == 160
    assert "valve" in calls[1][0]


@pytest.mark.parametrize("output", [
    [], [(0, -4.0), (0, -3.0)], [(-1, -4.0)], [(1, -4.0)],
    [(True, -4.0)], [(0.0, -4.0)], [(0, math.nan)], [(0, math.inf)],
    [(0, -math.inf)], [(0, "bad")], [(0, True)], [(0,)], None,
])
def test_invalid_full_reranker_output_is_rejected(output):
    with pytest.raises(ValueError):
        rerank_with_passages("valve", [long_document()], "local", rerank_fn=lambda *_: output)


def test_invalid_excerpt_output_is_also_rejected():
    calls = []

    def rerank(*args):
        calls.append(args)
        return [(0, -4.0)] if len(calls) == 1 else [(0, math.nan)]

    with pytest.raises(ValueError, match="finite"):
        rerank_with_passages("valve", [long_document()], "local", rerank_fn=rerank)


@pytest.mark.parametrize("floor", [math.nan, math.inf, 1.1, True, "0.6"])
def test_invalid_floor_is_rejected_before_inference(floor):
    def unexpected(*args):
        pytest.fail("Invalid policy must fail before inference")

    with pytest.raises(ValueError, match="confidence_floor"):
        rerank_with_passages("valve", [long_document()], "local", rerank_fn=unexpected, confidence_floor=floor)


def test_empty_documents_do_not_call_reranker():
    def unexpected(*args):
        pytest.fail("Empty input must not invoke a model")

    assert rerank_with_passages("valve", [], "local", rerank_fn=unexpected) == ([], {})


@pytest.mark.parametrize("singular,plural", [
    ("doctor", "doctors"), ("library", "libraries"), ("class", "classes"),
    ("movie", "movies"), ("box", "boxes"),
])
def test_regular_plural_surfaces_match_both_directions(singular, plural):
    from engram.rerank_passages import _surface_equivalent
    assert _surface_equivalent(singular, plural)
    assert _surface_equivalent(plural, singular)


@pytest.mark.parametrize("word,invalid", [
    ("class", "clas"), ("glass", "glas"), ("status", "statu"),
    ("analysis", "analysi"), ("bus", "bu"),
])
def test_protected_endings_do_not_create_false_singular_matches(word, invalid):
    from engram.rerank_passages import _surface_equivalent
    assert not _surface_equivalent(word, invalid)
    assert not _surface_equivalent(invalid, word)


def test_variant_repetitions_contribute_once_per_query_term():
    from engram.rerank_passages import _matched_query_terms
    assert _matched_query_terms({"doctors"}, {"doctor", "doctors", "staff"}) == {"doctors"}


def test_plural_selection_preserves_original_query_and_context():
    document = long_document("valve")
    query = "Which valves failed?"
    calls = []

    def rerank(semantic_query, documents, model):
        assert semantic_query == query
        calls.append(documents)
        return [(0, -4.0)]

    _, trace = rerank_with_passages(query, [document], "local", rerank_fn=rerank)
    assert len(calls) == 2
    assert "The copper valve failed." in calls[1][0]
    assert calls[1][0] == document[trace[0]["source_start"]:trace[0]["source_end"]]


def test_plural_only_overlap_can_select_a_long_document():
    from engram.rerank_passages import _query_passage
    document = "unrelated background. " * 90 + "A doctor arrived. The appointment began."
    passage = _query_passage("doctors", document)
    assert passage is not None
    text, start, end = passage
    assert "A doctor arrived." in text
    assert text == document[start:end]


def test_oversized_sentence_centers_on_plural_equivalent_hit():
    from engram.rerank_passages import _query_passage
    document = " ".join(["filler"] * 250 + ["doctor"] + ["filler"] * 250) + "."
    text, start, end = _query_passage("doctors", document)
    assert "doctor" in text
    assert len(text.split()) == 160
    assert text == document[start:end]
