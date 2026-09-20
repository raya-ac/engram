# embedding and reranking backends

Engram uses an embedding model to represent notes and queries, and an optional
reranker to score query/document pairs. those are separate from the generative
LLM used for extraction and query enrichment.

this page describes the registries and routing in Engram 0.8.1. inclusion here
means Engram has a routing or dimension entry; it does not promise that a hosted
provider still offers that model, or that one model is better for your data.

## start with the local defaults

```yaml
embedding_model: BAAI/bge-small-en-v1.5
embedding_backend: auto
cross_encoder_model: BAAI/bge-reranker-base
```

local weights download on first use and can be reused from cache. no generative
LLM or Claude account is needed for embeddings and reranking. inspect the chosen
settings and run a real model check with:

```sh
engram --config /absolute/path/to/config.yaml config show
engram --config /absolute/path/to/config.yaml doctor --check-models
```

that check uses synthetic text. it can download local weights or call the
configured hosted provider. plain `doctor` inspects package/cache readiness
without model inference.

## registered embedding dimensions

these are Engram's built-in dimension entries in
[`engram/embeddings.py`](https://github.com/raya-ac/engram/blob/main/engram/embeddings.py).
configuration derives the dimension from this registry when `embedding_dim`
has not been explicitly supplied.

| model | route | registered dimensions |
| --- | --- | --- |
| `BAAI/bge-small-en-v1.5` | local | 384 |
| `BAAI/bge-base-en-v1.5` | local | 768 |
| `BAAI/bge-large-en-v1.5` | local | 1024 |
| `BAAI/bge-m3` | local | 1024 |
| `nomic-ai/nomic-embed-text-v1.5` | local | 768 |
| `voyage-3.5` | Voyage | 1024 |
| `voyage-3.5-lite` | Voyage | 1024 |
| `voyage-3-large` | Voyage | 1024 |
| `voyage-3-lite` | Voyage | 512 |
| `voyage-code-3` | Voyage | 1024 |
| `voyage-finance-2` | Voyage | 1024 |
| `voyage-law-2` | Voyage | 1024 |
| `text-embedding-3-small` | OpenAI | 1536 |
| `text-embedding-3-large` | OpenAI | 3072 |
| `text-embedding-ada-002` | OpenAI | 1536 |
| `gemini-embedding-001` | Gemini | 768 |
| `text-embedding-004` | Gemini | 768 |

other model names can be passed to a selected backend, but their compatibility
is not guaranteed by the registry. set the correct `embedding_dim` explicitly
for an unregistered model, then check real output dimensions with doctor.
`init` accepts its known local model choices; configure other models in the
existing YAML file.

## how embedding routing works

routing follows this order:

1. an **exact** name in `MODEL_BACKENDS` selects its registered hosted provider;
2. otherwise, a non-`auto` `embedding_backend` selects that backend;
3. otherwise, Engram tries to import MLX and falls back to
   `sentence_transformers` if MLX is unavailable.

these are exact registry lookups, not wildcard rules for every name beginning
with `voyage-`, `text-embedding-` or `gemini-`. an exact hosted-model match takes
priority even when `embedding_backend` names a different backend. use
`config show` and doctor together to inspect the configured value and the
resolved readiness route.

### local runtime selection

| setting | behavior |
| --- | --- |
| `auto` | try the local MLX runtime, otherwise sentence-transformers |
| `sentence_transformers` | load the configured model through `SentenceTransformer` |
| `mlx` | load through `mlx_embeddings`; requires its compatible MLX installation |

for example:

```yaml
embedding_backend: sentence_transformers
```

this selects the library, **not a forced CPU device**. Engram does not pass an
explicit device to `SentenceTransformer`. the `portable` init preset uses this
same setting. an explicitly selected MLX backend does not switch to
sentence-transformers when MLX is missing; use a suitable installed runtime.
MLX embedding dependencies are not included by the base Engram package.

local cross-encoder reranking has its own device choice: the current code tries
PyTorch MPS when available, otherwise CPU, and retries construction on CPU if
needed. choosing MLX for embeddings does not move the reranker to MLX. runtime,
model and hardware differences affect latency and memory use; measure them on
your workload.

## hosted embedding providers

install the needed extra in the same environment that runs Engram:

| backend | install | credential environment |
| --- | --- | --- |
| Voyage | `python -m pip install 'engram-memory-system[voyage]'` | `VOYAGE_API_KEY` |
| OpenAI | `python -m pip install 'engram-memory-system[openai]'` | `OPENAI_API_KEY` |
| Gemini | `python -m pip install 'engram-memory-system[gemini]'` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |

then choose the model, for example:

```yaml
embedding_model: text-embedding-3-small
```

provide the credential through your process environment. an agent launched by
a desktop app may not inherit your shell's environment; give its Engram process
the required variable through the client's supported configuration. the
[agent setup hub](client-configs.md) covers those launch settings.

embedding providers read their own API-key variables. `llm.api_key` belongs to
generative extraction and does not substitute for them. provider availability,
pricing and account limits are outside Engram's model registry.

## choose a reranker independently

```yaml
cross_encoder_model: BAAI/bge-reranker-base
```

| model | implementation |
| --- | --- |
| `BAAI/bge-reranker-base` | default local cross-encoder |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | local alternative selected by the `light` init preset |
| `rerank-2` | registered Voyage reranker |
| `rerank-2-lite` | registered Voyage reranker |
| `rerank-2.5` | registered Voyage reranker |
| `rerank-2.5-lite` | registered Voyage reranker |

Voyage reranking requires its package and `VOYAGE_API_KEY`. other reranker names
are treated as local cross-encoder models; compatibility must be checked with
the model/runtime rather than inferred from a name prefix.

the CLI enables reranking with `search --rerank`. local cross-encoders return raw
logits, which retrieval maps through one sigmoid before applicable temporal/prior
adjustments. hosted Voyage scores already have a normalized relevance scale.
these are ranking scores, not calibrated probabilities that a memory is true.

### focused excerpts and the final gate

with `retrieval.rerank_passage_fallback: true`, a local reranker can retry an
eligible long memory using one excerpt. in 0.8.1 there are two triggers:

- every full-document bounded score is below `rerank_passage_floor`
  (default `0.001`);
- an individual candidate would otherwise fail the final `min_confidence`
  gate, even if another candidate passes it.

an excerpt needs matching query terms and is capped at 160 words. the same
semantic query scores the full text and excerpt; the larger raw score survives.
each candidate gets at most one excerpt attempt across both triggers. hosted
rerankers skip this local fallback.

the final `min_confidence` cutoff, default `0.6`, still applies after the retry.
`rerank_passage_floor` is an activation setting, not the final return threshold.
set `rerank_passage_fallback: false` or `rerank_passage_floor: 0` to disable these
retries. use `search --rerank --explain` to inspect decisions and source offsets.
see [retrieval internals](../architecture/retrieval.md#focused-excerpt-retry) and
[the configuration reference](../reference/config.md#rerank-scores) for the
scoring and context-selection details.

## change models on an existing store

query and document vectors must use the intended model and dimension. changing
an embedding model requires re-embedding existing non-forgotten records;
changing only the reranker does not.

1. back up the store and keep its old config. stop other Engram processes that
   read/write that store during the conversion.
2. select the new `embedding_model` and backend. remove an obsolete explicit
   `embedding_dim` to use the registered dimension, or set it correctly for the
   new model. check for environment overrides too.
3. select a **new, nonexistent** `ann.index_path` for the new vectors. retain the
   old index with the backup rather than reusing it.
4. inspect and verify the new model, then re-embed:

```sh
engram --config /absolute/path/to/config.yaml config show
engram --config /absolute/path/to/config.yaml doctor --check-models
engram --config /absolute/path/to/config.yaml reembed --dry-run
engram --config /absolute/path/to/config.yaml reembed
```

re-embedding updates non-forgotten records in batches. the fresh enabled index
path lets initialization build an index from those vectors; an already-existing
loadable index can otherwise be reused by the current initialization path.
`--dry-run` reports the record count without computing replacement vectors.

restart connected processes with the updated config and check representative
queries. if re-embedding fails partway through, resolve that failure before
resuming ordinary retrieval: the store may contain a mixture of old and new
vectors. use [troubleshooting](../getting-started/troubleshooting.md) for model,
dimension or connection errors.
