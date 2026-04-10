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


# --- MLX backend (Apple Silicon GPU) ---

def _get_mlx_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """Load a sentence-transformers model into MLX for GPU inference."""
    global _mlx_model, _mlx_tokenizer
    if _mlx_model is not None:
        return _mlx_model, _mlx_tokenizer

    import mlx.core as mx
    import mlx.nn as nn
    from transformers import AutoTokenizer, AutoModel
    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_model = AutoModel.from_pretrained(model_name)

    # extract weights and convert to MLX
    state_dict = torch_model.state_dict()
    mlx_weights = {}
    for k, v in state_dict.items():
        arr = v.detach().cpu().numpy()
        mlx_weights[k] = mx.array(arr)

    _mlx_model = mlx_weights
    _mlx_tokenizer = tokenizer
    return _mlx_model, _mlx_tokenizer


def _mlx_mean_pool(hidden_states, attention_mask):
    """Mean pooling over token embeddings, respecting attention mask."""
    import mlx.core as mx
    mask_expanded = mx.expand_dims(attention_mask, -1)  # (B, T, 1)
    masked = hidden_states * mask_expanded
    summed = mx.sum(masked, axis=1)
    counts = mx.maximum(mx.sum(mask_expanded, axis=1), mx.array(1e-8))
    return summed / counts


def _mlx_forward(tokens, weights):
    """Minimal BERT forward pass in MLX — enough for embedding extraction."""
    import mlx.core as mx

    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    token_type_ids = tokens.get("token_type_ids", mx.zeros_like(input_ids))

    # embeddings
    word_emb = mx.take(weights["embeddings.word_embeddings.weight"], input_ids, axis=0)
    pos_ids = mx.arange(input_ids.shape[1])
    pos_emb = mx.take(weights["embeddings.position_embeddings.weight"], pos_ids, axis=0)
    tok_emb = mx.take(weights["embeddings.token_type_embeddings.weight"], token_type_ids, axis=0)

    hidden = word_emb + pos_emb + tok_emb

    # layer norm
    ln_w = weights["embeddings.LayerNorm.weight"]
    ln_b = weights["embeddings.LayerNorm.bias"]
    eps = 1e-12
    mean = mx.mean(hidden, axis=-1, keepdims=True)
    var = mx.var(hidden, axis=-1, keepdims=True)
    hidden = (hidden - mean) / mx.sqrt(var + eps) * ln_w + ln_b

    # transformer layers
    n_layers = 0
    while f"encoder.layer.{n_layers}.attention.self.query.weight" in weights:
        n_layers += 1

    for i in range(n_layers):
        prefix = f"encoder.layer.{i}"

        # self-attention
        q = hidden @ weights[f"{prefix}.attention.self.query.weight"].T + weights[f"{prefix}.attention.self.query.bias"]
        k = hidden @ weights[f"{prefix}.attention.self.key.weight"].T + weights[f"{prefix}.attention.self.key.bias"]
        v = hidden @ weights[f"{prefix}.attention.self.value.weight"].T + weights[f"{prefix}.attention.self.value.bias"]

        d = q.shape[-1]
        n_heads = 12 if d >= 384 else 6
        head_dim = d // n_heads
        B, T, _ = q.shape

        q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(float(head_dim)))
        # attention mask
        mask = mx.expand_dims(mx.expand_dims(attention_mask, 1), 1)  # (B,1,1,T)
        scores = scores + (1 - mask) * (-1e9)
        attn = mx.softmax(scores, axis=-1)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, d)

        # attention output projection
        ctx = ctx @ weights[f"{prefix}.attention.output.dense.weight"].T + weights[f"{prefix}.attention.output.dense.bias"]
        hidden = hidden + ctx
        # layer norm
        ln_w = weights[f"{prefix}.attention.output.LayerNorm.weight"]
        ln_b = weights[f"{prefix}.attention.output.LayerNorm.bias"]
        mean = mx.mean(hidden, axis=-1, keepdims=True)
        var = mx.var(hidden, axis=-1, keepdims=True)
        hidden = (hidden - mean) / mx.sqrt(var + eps) * ln_w + ln_b

        # FFN
        ff = hidden @ weights[f"{prefix}.intermediate.dense.weight"].T + weights[f"{prefix}.intermediate.dense.bias"]
        # GELU activation
        ff = ff * 0.5 * (1.0 + mx.tanh(0.7978845608 * (ff + 0.044715 * ff * ff * ff)))
        ff = ff @ weights[f"{prefix}.output.dense.weight"].T + weights[f"{prefix}.output.dense.bias"]
        hidden = hidden + ff
        # layer norm
        ln_w = weights[f"{prefix}.output.LayerNorm.weight"]
        ln_b = weights[f"{prefix}.output.LayerNorm.bias"]
        mean = mx.mean(hidden, axis=-1, keepdims=True)
        var = mx.var(hidden, axis=-1, keepdims=True)
        hidden = (hidden - mean) / mx.sqrt(var + eps) * ln_w + ln_b

    return hidden


def _embed_mlx(texts: list[str], model_name: str, is_query: bool, normalize: bool) -> np.ndarray:
    """Embed via MLX (Apple Silicon GPU)."""
    import mlx.core as mx

    weights, tokenizer = _get_mlx_model(model_name)

    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]

    batch_size = 512
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="np")

        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])
        token_type_ids = mx.array(encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"])))

        tokens = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        hidden = _mlx_forward(tokens, weights)
        pooled = _mlx_mean_pool(hidden, attention_mask)

        if normalize:
            norms = mx.sqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True))
            pooled = pooled / mx.maximum(norms, mx.array(1e-8))

        mx.eval(pooled)
        all_embeddings.append(np.array(pooled))

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
