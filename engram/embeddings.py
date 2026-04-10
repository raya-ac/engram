"""Multi-backend embedding and cross-encoder reranking.

Backends:
  - sentence_transformers (default): CPU, works everywhere
  - mlx: Apple Silicon GPU via MLX, 10-50x faster for batch operations

Configurable via config.yaml:
  embedding_backend: sentence_transformers | mlx
  embedding_model: BAAI/bge-small-en-v1.5  (or any sentence-transformers model)
  cross_encoder_model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np

# suppress model loading noise
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*LOAD REPORT.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

_bi_encoder = None
_cross_encoder = None
_mlx_model = None
_mlx_tokenizer = None
_backend = None  # auto-detected or set via config
_models_warmed = False


def _detect_backend() -> str:
    """Auto-detect best available backend."""
    global _backend
    if _backend:
        return _backend
    try:
        import mlx.core
        _backend = "mlx"
    except ImportError:
        _backend = "sentence_transformers"
    return _backend


def set_backend(backend: str):
    """Override the embedding backend. Call before any embedding operations."""
    global _backend, _bi_encoder, _mlx_model, _mlx_tokenizer
    if backend not in ("sentence_transformers", "mlx"):
        raise ValueError(f"Unknown backend: {backend}. Use 'sentence_transformers' or 'mlx'")
    _backend = backend
    _bi_encoder = None
    _mlx_model = None
    _mlx_tokenizer = None


def get_backend() -> str:
    """Return the current embedding backend."""
    return _detect_backend()


def warmup(bi_model: str = "BAAI/bge-small-en-v1.5",
           ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Pre-load models so first query isn't slow."""
    global _models_warmed
    backend = _detect_backend()
    if backend == "mlx":
        _get_mlx_model(bi_model)
    else:
        _get_bi_encoder(bi_model)
    _get_cross_encoder(ce_model)
    _models_warmed = True


# --- Sentence Transformers backend (CPU) ---

def _get_bi_encoder(model_name: str = "BAAI/bge-small-en-v1.5"):
    global _bi_encoder
    if _bi_encoder is None:
        from sentence_transformers import SentenceTransformer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _bi_encoder = SentenceTransformer(model_name)
    return _bi_encoder


def _embed_st(texts: list[str], model_name: str, is_query: bool, normalize: bool) -> np.ndarray:
    """Embed via sentence-transformers (CPU)."""
    model = _get_bi_encoder(model_name)
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    embeddings = model.encode(texts, normalize_embeddings=normalize,
                               show_progress_bar=False, batch_size=256)
    return np.array(embeddings, dtype=np.float32)


# --- MLX backend (Apple Silicon GPU via mlx-embeddings) ---

def _get_mlx_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """Load model via mlx-embeddings for native GPU inference."""
    global _mlx_model, _mlx_tokenizer
    if _mlx_model is not None:
        return _mlx_model, _mlx_tokenizer

    from mlx_embeddings.utils import load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _mlx_model, _mlx_tokenizer = load(model_name)
    return _mlx_model, _mlx_tokenizer


def _embed_mlx(texts: list[str], model_name: str, is_query: bool, normalize: bool) -> np.ndarray:
    """Embed via mlx-embeddings (Apple Silicon GPU). ~2000 texts/sec."""
    from mlx_embeddings.utils import generate

    model, tokenizer = _get_mlx_model(model_name)

    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]

    # mlx-embeddings handles batching internally
    batch_size = 1024
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        result = generate(model, tokenizer, batch)
        embs = np.array(result.text_embeds)
        if normalize:
            norms = np.linalg.norm(embs, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            embs = embs / norms
        all_embeddings.append(embs)

    return np.vstack(all_embeddings).astype(np.float32)


# --- Public API ---

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_texts(texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5",
                is_query: bool = False, normalize: bool = True) -> np.ndarray:
    if not texts:
        return np.array([])
    backend = _detect_backend()
    if backend == "mlx":
        return _embed_mlx(texts, model_name, is_query, normalize)
    return _embed_st(texts, model_name, is_query, normalize)


def embed_query(text: str, model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    return embed_texts([text], model_name=model_name, is_query=True)[0]


def embed_documents(texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    return embed_texts(texts, model_name=model_name, is_query=False)


def cosine_similarity_search(query_vec: np.ndarray, doc_vecs: np.ndarray,
                              top_k: int = 10) -> list[tuple[int, float]]:
    if doc_vecs.size == 0:
        return []
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


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers.cross_encoder import CrossEncoder
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder
