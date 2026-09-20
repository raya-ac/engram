"""Model-free checks for benchmark metrics and safe checkpoint reuse."""

import json
import math

import pytest

from benchmarks.longmemeval.evaluation import (
    build_run_metadata,
    compute_metrics,
    load_resume_rows,
    write_run_metadata,
)


def test_second_position_has_discounted_gain():
    metrics = compute_metrics(["unrelated", "answer"], {"answer"}, ks=(1, 2, 5))
    assert metrics["recall_any@1"] == 0.0
    assert metrics["recall_any@2"] == 1.0
    assert metrics["ndcg_any@2"] == pytest.approx(1.0 / math.log2(3))
    assert metrics["ndcg_any@5"] < 1.0


def test_missing_answers_remain_in_ideal_ranking():
    metrics = compute_metrics(["first", "unrelated"], {"first", "missing"}, ks=(1, 5))
    assert metrics["recall_any@5"] == 1.0
    assert metrics["recall_all@5"] == 0.0
    assert metrics["ndcg_any@1"] == 1.0
    assert metrics["ndcg_any@5"] == pytest.approx(1.0 / (1.0 + 1.0 / math.log2(3)))


def test_duplicate_retrieval_ids_cannot_inflate_gain():
    with pytest.raises(ValueError, match="unique"):
        compute_metrics(["answer", "answer"], {"answer"})


@pytest.fixture
def checkpoint(tmp_path):
    dataset = tmp_path / "dataset.json"
    source = tmp_path / "retrieval.py"
    output = tmp_path / "results.jsonl"
    dataset.write_text("[]", encoding="utf-8")
    source.write_text("def retrieve(): pass\n", encoding="utf-8")

    def metadata(**overrides):
        kwargs = {
            "source_paths": {"retrieval.py": source},
            "embedding_model": "local-embedding",
            "cross_encoder_model": "local-reranker",
            "config": {"rerank": True, "top_k": 50},
        }
        kwargs.update(overrides)
        return build_run_metadata(dataset, **kwargs)

    row = {"question_id": "one", "retrieval_results": {}}
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")
    write_run_metadata(output, metadata())
    return output, source, metadata, row


def test_matching_resume_preserves_rows(checkpoint):
    output, _, metadata, row = checkpoint
    assert load_resume_rows(output, metadata()) == [row]


def test_changed_source_refuses_resume(checkpoint):
    output, source, metadata, _ = checkpoint
    source.write_text("def retrieve(): return []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance differs"):
        load_resume_rows(output, metadata())


@pytest.mark.parametrize(
    "overrides",
    [
        {"config": {"rerank": False, "top_k": 50}},
        {"cross_encoder_model": "another-reranker"},
    ],
)
def test_changed_settings_refuse_resume(checkpoint, overrides):
    output, _, metadata, _ = checkpoint
    with pytest.raises(ValueError, match="provenance differs"):
        load_resume_rows(output, metadata(**overrides))


def test_duplicate_questions_refuse_resume(checkpoint):
    output, _, metadata, row = checkpoint
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate question ID"):
        load_resume_rows(output, metadata())


def test_legacy_results_without_provenance_refuse_resume(tmp_path):
    output = tmp_path / "legacy.jsonl"
    output.write_text('{"question_id": "one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="metadata sidecar"):
        load_resume_rows(output, {})
