# Export & Import

portable backup and migration for engram memories.

## export

```bash
engram export backup.json                       # memories + entities + relationships
engram export backup.json --include-embeddings  # include embedding vectors
engram export backup.jsonl --layer procedural   # filter by layer
```

### JSON format

full export includes:

```json
{
  "version": "1.0",
  "exported_at": 1712345678.0,
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "embedding_dim": 384,
  "memories": [...],
  "entities": [...],
  "relationships": [...],
  "entity_mentions": [...]
}
```

### JSONL format

one memory per line (no entities/relationships):

```json
{"id": "...", "content": "...", "layer": "episodic", "importance": 0.7, ...}
```

### with embeddings

`--include-embeddings` adds base64-encoded embedding vectors. this makes the export larger but allows importing without re-embedding (useful when migrating between machines with the same model).

## import

```bash
engram import backup.json                   # restore everything
engram import backup.json --skip-duplicates # skip memories with matching content hash
```

import handles:

- memories (with optional embedded vectors)
- entities and their metadata
- relationships
- entity-memory links
- ANN index rebuild after import

### model mismatch warning

if the export used a different embedding model than your current config, engram warns you and re-embeds memories with the current model.

## use cases

- **backup before major changes**: `engram export pre-refactor.json --include-embeddings`
- **migrate between machines**: export with embeddings, import on new machine
- **share knowledge**: export a layer and share the JSON file
- **rollback**: import from a backup after a bad bulk operation
