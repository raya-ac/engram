# ANN index

Engram can use [hnswlib](https://github.com/nmslib/hnswlib) for approximate cosine
nearest-neighbor search over stored embeddings. it is a retrieval cache; the
database remains the source of memory records.

HNSW avoids comparing every vector on each query, but its speed and recall depend
on the data, settings and hardware. approximate search can miss candidates.
measure it against your workload rather than assuming a fixed latency or perfect
recall. when no ready ANN index is attached to the store, dense retrieval uses
the database embedding pool instead.

## settings

```yaml
ann:
  enabled: true
  m: 32
  ef_construction: 200
  ef_search: 100
  max_elements: 500000
  index_path: ~/.local/share/engram/hnsw.index
```

`m` controls graph connectivity; `ef_construction` controls build effort;
`ef_search` controls search effort. larger values can trade memory or time for
recall, without guaranteeing a particular result.

the example shows the package default index path. `init` derives an index path
from the new database location, so inspect your effective `ann.index_path` with
`config show`. changing embedding models requires compatible dimensions and a
fresh index; see [model changes](../guides/embedding-backends.md#change-models-on-an-existing-store).

## initialization

`Store.init_ann_index()` first tries to load the configured index and its metadata.
if loading fails or the files are absent, it builds from the database's
non-forgotten records that have embeddings. this pool is not restricted to
`status: active`; normal retrieval applies eligibility rules afterward.

```python
store.init_ann_index(background=True)   # build in the background if loading fails
store.init_ann_index(background=False)  # wait for a required build
```

web and MCP startup call this initialization path. merely constructing a `Store`
does not attach an index. initialization returns without an index when ANN is
disabled or hnswlib is unavailable. on a store with no embedded records, there
may be no ready or persisted index yet.

## writes and forgetting

`save_memory()` updates the index only when that process has an attached, ready
index and the saved memory has an embedding. an update to an already-indexed ID
marks its previous hnswlib label deleted and adds a new label.

`forget_memory()` marks the database record forgotten and removes its vector
from an attached, ready index. these are updates to that process's in-memory
index, not a mechanism for synchronizing every running process. an index loaded
by another process can be stale.

## persistence

a ready attached index is saved by `store.close()` or an explicit
`ann_index.save()`. the binary index has a companion metadata file obtained by
replacing its final suffix with `.meta.json`:

| index path | companion metadata |
| --- | --- |
| `memory.hnsw.index` | `memory.hnsw.meta.json` |

the metadata holds memory-ID/label mappings, the next label, a saved element
count and a timestamp. keep both files together. a loadable cache does not by
itself establish that it matches the current database or embedding model.

## CLI behavior in 0.8.1

```sh
engram --config /absolute/path/to/config.yaml index status
engram --config /absolute/path/to/config.yaml index rebuild
```

`index status` reads the file's saved metadata, size and next label. its saved
element count is not a fresh database count and can include deleted hnswlib
labels.

despite its name, `index rebuild` currently calls the same **load-or-build**
initialization path. an existing loadable index can be reused. the CLI does not
force its replacement. `reembed` and `import` also finish through that path, so
their status messages alone do not prove an existing index was rebuilt.

for a fresh build without deleting the old cache:

1. stop other processes using the store and retain the old config/index files;
2. set `ann.enabled: true` and choose a new `ann.index_path` whose index and
   companion metadata files do not exist;
3. run `index rebuild` with that config;
4. check `index status`, restart clients with the same config, and verify
   representative queries.

for example, edit the intended config to use a new filename:

```yaml
ann:
  enabled: true
  index_path: /absolute/path/to/memory-rebuilt.hnsw.index
```

then run:

```sh
engram --config /absolute/path/to/config.yaml index rebuild
engram --config /absolute/path/to/config.yaml index status
```

the Python store API also exposes `rebuild_ann_index()` for a synchronous rebuild
of an already-attached index. it is not what the current CLI dispatch calls.

## implementation

[`engram/ann_index.py`](https://github.com/raya-ac/engram/blob/main/engram/ann_index.py)
wraps hnswlib with integer-label mappings, locking around index operations,
capacity growth, persistence and cosine-distance conversion. hnswlib distances
are converted to similarity with `1 - distance`.

| caller | use |
| --- | --- |
| retrieval dense search | ANN candidates when a ready index is attached |
| surprise scoring | nearest-neighbor comparison when available |
| `Store.save_memory()` | update an attached ready index |
| `Store.forget_memory()` | remove from an attached ready index |
| web and MCP startup | load or build the configured index |
