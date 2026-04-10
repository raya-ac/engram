"""Sentence-transformer embedding and cross-encoder reranking."""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np

# suppress model loading noise (BERT position_ids UNEXPECTED warnings, HF token nag)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

_bi_encoder = None
_cross_encoder = None
_models_warmed = False


def warmup(bi_model: str = "BAAI/bge-small-en-v1.5",
           ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Pre-load both models so first query isn't slow."""
    global _models_warmed
    _get_bi_encoder(bi_model)
    _get_cross_encoder(ce_model)
    _models_warmed = True


def _get_bi_encoder(model_name: str = "BAAI/bge-small-en-v1.5"):
    global _bi_encoder
    if _bi_encoder is None:
        from sentence_transformers import SentenceTransformer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _bi_encoder = SentenceTransformer(model_name)
    return _bi_encoder


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers.cross_encoder import CrossEncoder
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_texts(texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5",
                is_query: bool = False, normalize: bool = True) -> np.ndarray:
    if not texts:
        return np.array([])
    model = _get_bi_encoder(model_name)
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    embeddings = model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)


def embed_query(text: str, model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    return embed_texts([text], model_name=model_name, is_query=True)[0]


def embed_documents(texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    return embed_texts(texts, model_name=model_name, is_query=False)


def cosine_similarity_search(query_vec: np.ndarray, doc_vecs: np.ndarray,
                              top_k: int = 10) -> list[tuple[int, float]]:
    if doc_vecs.size == 0:
        return []
    # vectors are pre-normalized, so dot product = cosine similarity
    scores = doc_vecs @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices]


def cross_encoder_rerank(query: str, documents: list[str],
                          model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> list[tuple[int, float]]:
    if not documents:
        return []
    model = _get_cross_encoder(model_name)
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(int(i), float(s)) for i, s in ranked]
