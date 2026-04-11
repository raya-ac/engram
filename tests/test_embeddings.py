"""Tests for embeddings.py — multi-backend embedding and cross-encoder."""

import numpy as np
import pytest

from engram.embeddings import (
    embed_query, embed_documents, embed_texts,
    cosine_similarity_search, get_backend, get_model_dim,
    set_default_model, _default_model, MODEL_DIMS, MODEL_BACKENDS,
)


class TestLocalEmbedding:
    def test_embed_query_shape(self):
        v = embed_query("test query")
        assert v.shape == (384,)
        assert v.dtype == np.float32

    def test_embed_query_normalized(self):
        v = embed_query("test query")
        assert abs(np.linalg.norm(v) - 1.0) < 0.01

    def test_embed_documents_batch(self):
        vecs = embed_documents(["hello", "world", "test"])
        assert vecs.shape == (3, 384)

    def test_embed_empty(self):
        result = embed_texts([])
        assert result.size == 0

    def test_different_texts_different_vectors(self):
        v1 = embed_query("cats are great")
        v2 = embed_query("quantum physics equations")
        sim = float(v1 @ v2)
        assert sim < 0.9  # should not be identical


class TestCosineSearch:
    def test_basic_search(self):
        docs = embed_documents(["cat", "dog", "physics", "quantum"])
        query = embed_query("animal")
        results = cosine_similarity_search(query, docs, top_k=2)
        assert len(results) == 2
        assert all(isinstance(idx, int) for idx, _ in results)
        assert all(isinstance(s, float) for _, s in results)

    def test_sorted_descending(self):
        docs = embed_documents(["cat", "dog", "physics"])
        query = embed_query("cat")
        results = cosine_similarity_search(query, docs, top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_docs(self):
        query = embed_query("test")
        results = cosine_similarity_search(query, np.array([]), top_k=5)
        assert results == []


class TestBackendDetection:
    def test_local_model_backend(self):
        backend = get_backend("BAAI/bge-small-en-v1.5")
        assert backend in ("mlx", "sentence_transformers")

    def test_voyage_model_detected(self):
        assert get_backend("voyage-3.5") == "voyage"
        assert get_backend("voyage-3.5-lite") == "voyage"
        assert get_backend("voyage-code-3") == "voyage"

    def test_openai_model_detected(self):
        assert get_backend("text-embedding-3-small") == "openai"
        assert get_backend("text-embedding-3-large") == "openai"

    def test_gemini_model_detected(self):
        assert get_backend("gemini-embedding-001") == "gemini"


class TestModelDims:
    def test_known_models(self):
        assert get_model_dim("BAAI/bge-small-en-v1.5") == 384
        assert get_model_dim("voyage-3.5") == 1024
        assert get_model_dim("text-embedding-3-small") == 1536
        assert get_model_dim("gemini-embedding-001") == 768

    def test_unknown_model(self):
        assert get_model_dim("some-unknown-model") is None


class TestDefaultModel:
    def test_set_default(self):
        original = _default_model
        set_default_model("voyage-3.5")
        from engram.embeddings import _default_model as current
        assert current == "voyage-3.5"
        set_default_model(original)  # restore


class TestAPIBackendErrors:
    def test_voyage_no_key(self):
        import os
        old = os.environ.pop("VOYAGE_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
                embed_query("test", "voyage-3.5")
        finally:
            if old:
                os.environ["VOYAGE_API_KEY"] = old

    def test_openai_no_key(self):
        import os
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                embed_query("test", "text-embedding-3-small")
        finally:
            if old:
                os.environ["OPENAI_API_KEY"] = old
