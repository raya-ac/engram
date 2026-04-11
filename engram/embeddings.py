"""Multi-backend embedding and cross-encoder reranking.

Backends:
  - auto: picks mlx > sentence_transformers for local models
  - mlx: Apple Silicon GPU via MLX (local models only)
  - sentence_transformers: CPU (local models only)
  - voyage: Voyage AI API (voyage-3.5, voyage-3.5-lite, voyage-code-3, etc.)
  - openai: OpenAI API (text-embedding-3-small, text-embedding-3-large)
  - gemini: Google Gemini API (gemini-embedding-001)

Configurable via config.yaml:
  embedding_backend: auto | mlx | sentence_transformers | voyage | openai | gemini
  embedding_model: BAAI/bge-small-en-v1.5  (local) | voyage-3.5 (API) | text-embedding-3-small (API)
  embedding_dim: 384  (auto-detected from model if not set)
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
_backend = None
_models_warmed = False
_default_model = "BAAI/bge-small-en-v1.5"

# known model → dim mappings (so we don't need a probe call)
MODEL_DIMS = {
    # local
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    # voyage
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-3-large": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
    # openai
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # gemini
    "gemini-embedding-001": 768,
    "text-embedding-004": 768,
}

# which backend a model needs
MODEL_BACKENDS = {
    "voyage-3.5": "voyage", "voyage-3.5-lite": "voyage",
    "voyage-3-large": "voyage", "voyage-3-lite": "voyage",
    "voyage-code-3": "voyage", "voyage-finance-2": "voyage",
    "voyage-law-2": "voyage",
    "text-embedding-3-small": "openai", "text-embedding-3-large": "openai",
    "text-embedding-ada-002": "openai",
    "gemini-embedding-001": "gemini", "text-embedding-004": "gemini",
}


def get_model_dim(model_name: str) -> int | None:
    """Get embedding dimension for a known model. Returns None if unknown."""
    return MODEL_DIMS.get(model_name)


def _detect_backend(model_name: str | None = None) -> str:
    """Detect backend. Priority: model name → explicit _backend → auto-detect local."""
    global _backend

    # API models always use their specific backend
    if model_name and model_name in MODEL_BACKENDS:
        return MODEL_BACKENDS[model_name]

    # explicit override via set_backend() or config
    if _backend and _backend != "auto":
        return _backend

    # auto-detect best local backend
    try:
        import mlx.core
        return "mlx"
    except ImportError:
        return "sentence_transformers"


def set_backend(backend: str):
    """Override the embedding backend. Called during config init."""
    global _backend, _bi_encoder, _mlx_model, _mlx_tokenizer
    valid = ("auto", "sentence_transformers", "mlx", "voyage", "openai", "gemini")
    if backend not in valid:
        raise ValueError(f"Unknown backend: {backend}. Use one of: {valid}")
    _backend = backend
    _bi_encoder = None
    _mlx_model = None
    _mlx_tokenizer = None


def set_default_model(model_name: str):
    """Set the default model used when callers don't specify one."""
    global _default_model
    _default_model = model_name


def get_backend(model_name: str | None = None) -> str:
    return _detect_backend(model_name)


def warmup(bi_model: str = "BAAI/bge-small-en-v1.5",
           ce_model: str | None = None):
    """Pre-load models so first query isn't slow."""
    global _models_warmed
    backend = _detect_backend(bi_model)
    if backend == "mlx":
        _get_mlx_model(bi_model)
    elif backend == "sentence_transformers":
        _get_bi_encoder(bi_model)
    # API backends don't need warmup
    if ce_model:
        _get_cross_encoder(ce_model)
    _models_warmed = True


# ── Sentence Transformers backend (CPU) ──────────────────────────

def _get_bi_encoder(model_name: str = "BAAI/bge-small-en-v1.5"):
    global _bi_encoder
    if _bi_encoder is None:
        from sentence_transformers import SentenceTransformer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _bi_encoder = SentenceTransformer(model_name)
    return _bi_encoder


def _embed_st(texts: list[str], model_name: str, is_query: bool, normalize: bool) -> np.ndarray:
    model = _get_bi_encoder(model_name)
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    embeddings = model.encode(texts, normalize_embeddings=normalize,
                               show_progress_bar=False, batch_size=256)
    return np.array(embeddings, dtype=np.float32)


# ── MLX backend (Apple Silicon GPU) ─────────────────────────────

def _get_mlx_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    global _mlx_model, _mlx_tokenizer
    if _mlx_model is not None:
        return _mlx_model, _mlx_tokenizer

    from mlx_embeddings.utils import load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _mlx_model, _mlx_tokenizer = load(model_name)
    return _mlx_model, _mlx_tokenizer


def _embed_mlx(texts: list[str], model_name: str, is_query: bool, normalize: bool) -> np.ndarray:
    from mlx_embeddings.utils import generate

    model, tokenizer = _get_mlx_model(model_name)

    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]

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


# ── Voyage AI backend ───────────────────────────────────────────

def _embed_voyage(texts: list[str], model_name: str, is_query: bool, normalize: bool) -> np.ndarray:
    """Embed via Voyage AI API. Requires VOYAGE_API_KEY env var."""
    try:
        import voyageai
    except ImportError:
        raise ImportError("voyageai not installed: pip install voyageai")

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY environment variable not set. Get one at https://dash.voyageai.com/")

    vo = voyageai.Client(api_key=api_key)
    input_type = "query" if is_query else "document"

    # voyage API limit: 128 texts per call, 320k tokens total
    batch_size = 128
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        result = vo.embed(batch, model=model_name, input_type=input_type)
        all_embeddings.extend(result.embeddings)

    vecs = np.array(all_embeddings, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        vecs = vecs / norms
    return vecs


# ── OpenAI backend ──────────────────────────────────────────────

def _embed_openai(texts: list[str], model_name: str, is_query: bool, normalize: bool) -> np.ndarray:
    """Embed via OpenAI API. Requires OPENAI_API_KEY env var."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai not installed: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)

    # openai limit: 2048 texts per call
    batch_size = 2048
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(input=batch, model=model_name)
        batch_embs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embs)

    vecs = np.array(all_embeddings, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        vecs = vecs / norms
    return vecs


# ── Gemini backend ──────────────────────────────────────────────

def _embed_gemini(texts: list[str], model_name: str, is_query: bool, normalize: bool) -> np.ndarray:
    """Embed via Google Gemini API. Requires GEMINI_API_KEY env var."""
    try:
        from google import genai
    except ImportError:
        raise ImportError("google-genai not installed: pip install google-genai")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"

    # gemini limit: 100 texts per call
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        result = client.models.embed_content(
            model=model_name,
            contents=batch,
            config={"task_type": task},
        )
        all_embeddings.extend([e.values for e in result.embeddings])

    vecs = np.array(all_embeddings, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        vecs = vecs / norms
    return vecs


# ── Public API ───────────────────────────────────────────────────

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_texts(texts: list[str], model_name: str | None = None,
                is_query: bool = False, normalize: bool = True) -> np.ndarray:
    if not texts:
        return np.array([])
    model = model_name or _default_model
    backend = _detect_backend(model)
    if backend == "voyage":
        return _embed_voyage(texts, model, is_query, normalize)
    elif backend == "openai":
        return _embed_openai(texts, model, is_query, normalize)
    elif backend == "gemini":
        return _embed_gemini(texts, model, is_query, normalize)
    elif backend == "mlx":
        return _embed_mlx(texts, model, is_query, normalize)
    return _embed_st(texts, model, is_query, normalize)


def embed_query(text: str, model_name: str | None = None) -> np.ndarray:
    return embed_texts([text], model_name=model_name, is_query=True)[0]


def embed_documents(texts: list[str], model_name: str | None = None) -> np.ndarray:
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
