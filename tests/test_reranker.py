"""Cross-encoder score contracts without loading a model or calling an API."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from engram import embeddings


class FakeEncoder:
    def __init__(self, *, failure=None, default_sigmoid=True):
        self.failure = failure
        self.default_sigmoid = default_sigmoid
        self.calls = []
        self.devices = []
        self.model = SimpleNamespace(to=self.devices.append)

    def predict(self, pairs, *, batch_size, show_progress_bar, activation_fn=None):
        self.calls.append((pairs, batch_size, show_progress_bar, activation_fn))
        if self.failure is not None and len(self.calls) == 1:
            raise self.failure
        logits = np.array([-3.0, 2.0, 0.5], dtype=np.float32)
        if activation_fn is not None:
            return activation_fn(logits)
        return 1.0 / (1.0 + np.exp(-logits)) if self.default_sigmoid else logits


@pytest.mark.parametrize("model_name, default_sigmoid", [
    ("BAAI/bge-reranker-base", True),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", False),
])
def test_local_models_return_raw_logits(monkeypatch, model_name, default_sigmoid):
    encoder = FakeEncoder(default_sigmoid=default_sigmoid)
    monkeypatch.setattr(embeddings, "_get_cross_encoder", lambda name: encoder)

    result = embeddings.cross_encoder_rerank("query", ["a", "b", "c"], model_name)

    assert result == [(1, 2.0), (2, 0.5), (0, -3.0)]
    pairs, batch_size, progress, activation = encoder.calls[0]
    assert pairs == [("query", "a"), ("query", "b"), ("query", "c")]
    assert batch_size == 16
    assert progress is False
    assert isinstance(activation, torch.nn.Identity)


def test_legacy_predict_activation_keyword(monkeypatch):
    class LegacyEncoder:
        def predict(self, pairs, *, batch_size, show_progress_bar, activation_fct=None):
            assert isinstance(activation_fct, torch.nn.Identity)
            return activation_fct(np.array([-2.0, 4.0]))

    monkeypatch.setattr(embeddings, "_get_cross_encoder", lambda name: LegacyEncoder())
    assert embeddings.cross_encoder_rerank("query", ["a", "b"]) == [(1, 4.0), (0, -2.0)]


@pytest.mark.parametrize("message", ["MPS backend out of memory", "MPS execution failed"])
def test_cpu_retry_preserves_raw_logits(monkeypatch, message):
    encoder = FakeEncoder(failure=RuntimeError(message))
    monkeypatch.setattr(embeddings, "_get_cross_encoder", lambda name: encoder)
    cache_clears = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: cache_clears.append(True))

    result = embeddings.cross_encoder_rerank("query", ["a", "b", "c"], "BAAI/bge-reranker-base")

    assert result == [(1, 2.0), (2, 0.5), (0, -3.0)]
    assert [call[1] for call in encoder.calls] == [16, 8]
    assert isinstance(encoder.calls[0][3], torch.nn.Identity)
    assert encoder.calls[0][3] is encoder.calls[1][3]
    assert encoder.devices == ["cpu"]
    assert cache_clears == [True]


def test_unrelated_failure_is_not_retried(monkeypatch):
    encoder = FakeEncoder(failure=ValueError("invalid model output"))
    monkeypatch.setattr(embeddings, "_get_cross_encoder", lambda name: encoder)

    with pytest.raises(ValueError, match="invalid model output"):
        embeddings.cross_encoder_rerank("query", ["a", "b", "c"])

    assert len(encoder.calls) == 1
    assert encoder.devices == []


def test_voyage_scores_are_unchanged(monkeypatch):
    expected = [(1, 0.9), (0, 0.1)]
    calls = []

    def voyage(query, documents, model_name):
        calls.append((query, documents, model_name))
        return expected

    def unexpected_local_load(model_name):
        pytest.fail("cloud reranking must not load a local encoder")

    monkeypatch.setattr(embeddings, "_rerank_voyage", voyage)
    monkeypatch.setattr(embeddings, "_get_cross_encoder", unexpected_local_load)

    assert embeddings.cross_encoder_rerank("query", ["a", "b"], "rerank-2.5") == expected
    assert calls == [("query", ["a", "b"], "rerank-2.5")]


def test_empty_documents_do_not_load_encoder(monkeypatch):
    def unexpected_load(model_name):
        pytest.fail("empty reranking must not load a model")

    monkeypatch.setattr(embeddings, "_get_cross_encoder", unexpected_load)
    assert embeddings.cross_encoder_rerank("query", []) == []
