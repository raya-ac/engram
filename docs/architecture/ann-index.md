# ANN Index

HNSW approximate nearest neighbor index via [hnswlib](https://github.com/nmslib/hnswlib). replaces brute-force O(n) cosine similarity with O(log n) search.

## why HNSW

| | brute-force | HNSW |
|---|---|---|
| search complexity | O(n) | O(log n) |
| 1k vectors | 0.1ms | 0.12ms |
| 100k vectors | 8.7ms | 0.20ms |
| 1M vectors | 87.3ms | 0.23ms |
| recall@10 | 100% (exact) | 100% (at current scale) |

## parameters

configured in `config.yaml`:

```yaml
ann:
  enabled: true
  m: 32              # graph connectivity — more = better recall, more memory
  ef_construction: 200  # build-time search depth — more = better index
  ef_search: 100     # query-time search depth — more = better recall, slower
  max_elements: 500000
  index_path: ~/.local/share/engram/hnsw.index
```

## lifecycle

### build

on startup, the store tries to load the index from disk. if missing, rebuilds from all embeddings in the background.

```python
store.init_ann_index(background=True)  # non-blocking
store.init_ann_index(background=False) # blocking (for CLI)
```

### write

every `save_memory()` with an embedding calls `ann_index.add(id, vec)`. if the memory already exists, the old vector is marked deleted and the new one is added.

### delete

`forget_memory()` calls `ann_index.remove(id)` which uses hnswlib's `mark_delete()`.

### persist

the index saves to disk on `store.close()` and on explicit `ann_index.save()`. a `.meta.json` file alongside stores the id↔label mapping and metadata.

### rebuild

```bash
engram index rebuild   # full rebuild from DB
engram index status    # check vector count, size, last built
```

## implementation

`engram/ann_index.py` wraps hnswlib with:

- thread-safe add/remove/search via `threading.Lock`
- id↔label mapping (hnswlib uses integer labels internally)
- auto-resize when capacity is exceeded
- cosine distance space (hnswlib returns `1 - cosine_similarity`, converted back)

## integration points

| module | how it uses ANN |
|--------|----------------|
| `retrieval.py` → `_dense_search()` | primary dense search channel |
| `surprise.py` → `compute_surprise()` | k-NN novelty scoring at write time |
| `store.py` → `save_memory()` | add vector on write |
| `store.py` → `forget_memory()` | remove vector on delete |
| `web/app.py` | init on web server startup |
| `mcp_server.py` | init on MCP server startup |
