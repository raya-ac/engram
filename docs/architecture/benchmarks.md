# Benchmarks

## LongMemEval (ICLR 2025)

[LongMemEval](https://arxiv.org/abs/2410.10813) — 500 questions testing 5 long-term memory abilities across ~40 conversation sessions per question (~115k tokens).

### results

| system | R@5 | method |
|--------|-----|--------|
| **engram v2** | **98.1%** | HNSW + BM25 + assistant BM25 + temporal boost + cross-encoder |
| MemPalace (raw) | 96.6% | ChromaDB cosine, verbatim storage |
| engram v1 | 94.7% | HNSW + BM25 + RRF |
| Emergence AI | 86.0% | RAG |
| MemPalace (AAAK) | 84.2% | compressed storage |
| EverMemOS | 83.0% | — |
| TiMem | 76.9% | temporal hierarchical |

### per question type

| type | n | R@5 | R@10 |
|------|---|-----|------|
| knowledge-update | 72 | 100.0% | 100.0% |
| single-session-user | 64 | 100.0% | 100.0% |
| multi-session | 121 | 99.2% | 99.2% |
| temporal-reasoning | 127 | 96.9% | 97.6% |
| single-session-assistant | 56 | 96.4% | 96.4% |
| single-session-preference | 30 | 93.3% | 96.7% |

### what makes it work

v2 adds three improvements over v1:

1. **assistant-turn BM25** (weight 0.5) — catches answers in assistant responses without polluting the dense index
2. **timestamp proximity boost** — favors sessions closer to the question date
3. **cross-encoder reranking** — jointly scores top-20 candidates against the query

run the benchmark: `python benchmarks/longmemeval/run_engram.py data/longmemeval_s_cleaned.json --rerank`

## system benchmark (72 tests)

43 subsystem tests across 20 modules:

| subsystem | tests | result |
|-----------|-------|--------|
| embedding | 3/3 | dim=384, norm=1.0, avg 5.1ms |
| ANN index (HNSW) | 7/7 | 0.09ms search, 100% recall@10, 5,304 inserts/sec |
| brute-force dense | 2/2 | 0.016ms avg |
| intent classification | 1/1 | 6/6 correct |
| full pipeline (no rerank) | 3/3 | 15.5ms avg |
| full pipeline (+ cross-encoder) | 1/1 | 252ms avg |
| cross-encoder | 2/2 | 2.9ms/doc |
| surprise gate | 4/4 | 0.10ms avg |
| Hopfield channel | 1/1 | <1ms |
| BM25 / FTS5 | 2/2 | 3.5ms avg |
| entity graph | 4/4 | 2-hop traversal |
| memory CRUD | 2/2 | write → ANN → forget |
| layers (L0-L3) | 1/1 | 248ms |

plus 72 pytest tests across store, embeddings, ann_index, retrieval, surprise, and config.

## latency

| operation | time |
|-----------|------|
| ANN dense search | 0.09ms |
| full pipeline (no rerank) | 15.5ms |
| full pipeline (+ cross-encoder) | 252ms |
| embedding | 5.1ms |
| surprise gate | 0.10ms |
| ANN insert | 0.19ms |
| BM25 / FTS5 | 3.5ms |

## ANN scaling

| vectors | brute-force | HNSW | speedup |
|---------|------------|------|---------|
| 1k | 0.1ms | 0.12ms | 1x |
| 10k | 0.9ms | 0.16ms | 5x |
| 100k | 8.7ms | 0.20ms | 45x |
| 500k | 43.7ms | 0.22ms | 198x |
| 1M | 87.3ms | 0.23ms | 377x |

## throughput

| operation | rate |
|-----------|------|
| embedding (MLX GPU) | 1,879 texts/sec |
| embedding (CPU) | 176 texts/sec |
| SQLite bulk insert | 51,000 rows/sec |
| ANN insert | 5,304 ops/sec |
