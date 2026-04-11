# Embedding Backends

engram supports local and cloud embedding models. the backend is auto-detected from the model name.

## supported models

| model | provider | dim | price/1M tokens | notes |
|-------|----------|-----|-----------------|-------|
| `BAAI/bge-small-en-v1.5` | local | 384 | free | default, MLX GPU or CPU |
| `BAAI/bge-base-en-v1.5` | local | 768 | free | |
| `BAAI/bge-large-en-v1.5` | local | 1024 | free | |
| `voyage-3.5` | Voyage AI | 1024 | $0.18 | best retrieval quality |
| `voyage-3.5-lite` | Voyage AI | 1024 | $0.02 | 94% of 3.5 quality |
| `voyage-code-3` | Voyage AI | 1024 | $0.18 | optimized for code |
| `voyage-finance-2` | Voyage AI | 1024 | $0.18 | optimized for finance |
| `voyage-law-2` | Voyage AI | 1024 | $0.18 | optimized for legal |
| `text-embedding-3-small` | OpenAI | 1536 | $0.02 | cheapest API option |
| `text-embedding-3-large` | OpenAI | 3072 | $0.13 | highest dimensionality |
| `gemini-embedding-001` | Google | 768 | free tier | top MTEB retrieval score |

## switching models

1. change the model in `config.yaml`:

```yaml
embedding_model: voyage-3.5
```

2. set the API key:

```bash
export VOYAGE_API_KEY="your-key"
```

3. re-embed all existing memories:

```bash
engram reembed
```

this re-embeds every memory with the new model and rebuilds the ANN index. the embedding dimension is auto-detected from the model name.

## how auto-detection works

engram looks at the model name to determine the backend:

- `voyage-*` → Voyage AI API
- `text-embedding-*` → OpenAI API
- `gemini-*` → Google Gemini API
- anything else → local (MLX on Apple Silicon, sentence-transformers on CPU)

you can override with `embedding_backend: voyage` in config.yaml.

## reranker models

the cross-encoder reranker also supports cloud backends:

| model | provider | notes |
|-------|----------|-------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | local | default, ~300ms for 20 docs |
| `rerank-2.5` | Voyage AI | best quality, 32k context |
| `rerank-2.5-lite` | Voyage AI | faster/cheaper |

set in config.yaml:

```yaml
cross_encoder_model: rerank-2.5
```

## local backends

### MLX (Apple Silicon)

auto-detected on macOS with Apple Silicon. uses the GPU for ~10x faster embedding than CPU.

### sentence-transformers (CPU)

fallback on Linux/Windows or when MLX isn't available. works everywhere.

force a specific backend:

```yaml
embedding_backend: sentence_transformers  # force CPU
embedding_backend: mlx                    # force Apple GPU
```
