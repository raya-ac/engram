# export and import

transfer selected memory records between Engram stores as JSON or JSONL. the
current CLI export is **not a complete database backup**, and import is not a
rollback operation. use a SQLite or Postgres database backup when you need exact
restoration of lifecycle state, history and every table.

## export records

```sh
engram --config /absolute/path/to/config.yaml export records.json
engram --config /absolute/path/to/config.yaml export records.json --include-embeddings
engram --config /absolute/path/to/config.yaml export procedures.jsonl --layer procedural
```

exports select `forgotten = 0`, optionally restricted by `--layer`, ordered by
creation time. they do **not** require `status = 'active'`: challenged,
invalidated, superseded or merged records can be included if not forgotten.

### JSON or JSONL

| format | contents |
| --- | --- |
| `.jsonl` | one selected memory per line |
| other output suffixes, normally `.json` | selected memories plus export metadata and the entity/relationship/mention tables |

JSON graph tables are currently exported in full, even with a memory-layer
filter. entity mentions can refer to memories excluded by that filter or by
forgotten status. **`--layer` is not a privacy boundary for the JSON graph data.**
review the actual file before sharing it or moving it into a different trust
boundary.

memory records include ID, content, source, layer, importance, timestamps,
access count, fact dates, emotional valence, chunk hash and metadata. they omit
the dedicated `memory_type`, `status`, `forgotten` and `previous_memory_id`
fields. access/event histories, hypothetical queries, session checkpoints and
other store tables are not part of this export format.

### include embeddings

`--include-embeddings` adds base64 vectors for records with embeddings, together
with dimension/model labels. JSON also includes the configured model and
dimension at the export's top level.

these labels describe the selected configuration; export does not verify that
every stored vector was produced by it. keep the correct config with a transfer,
especially if a previous re-embedding run was interrupted.

## import into the intended destination

for a separate destination, first create it with `init`, then import using its
config path:

```sh
engram --config /absolute/new/config.yaml init --yes --db-path /absolute/new/memory.db
engram --config /absolute/new/config.yaml import records.json --skip-duplicates
```

import writes into the selected store and may update records with matching IDs.
it does not clear records absent from the file. `--skip-duplicates` skips a memory
when an existing record has the same `chunk_hash`; it does not make all graph
operations idempotent or restore a previous database snapshot.

JSON imports also attempt to restore entities, relationships and entity-memory
links. JSONL has none of those graph sections. filtered JSON exports with links
to absent memories can fail foreign-key checks in a fresh destination. use
JSONL for a record-only layer transfer, or verify and prepare the graph references
before importing a filtered JSON file.

imports save records incrementally. a failure can leave a partially imported
store; use a new destination for a transfer you need to inspect before adoption.

## lifecycle state is not round-tripped

import currently constructs memories with default type/status:
`memory_type: narrative`, `status: active`, and `forgotten: false`. it does not
restore the omitted lifecycle fields, previous-memory links or histories.

in particular, a non-forgotten superseded/challenged record from an export can
become active after import. preserve the database itself when these distinctions
matter; do not use record import as a way to undo a bulk operation.

## model mismatch does not trigger automatic conversion

when JSON's top-level embedding model differs from the destination config,
import prints a warning and recommends `reembed`. it **does not automatically
replace supplied vectors**. JSONL has no top-level model warning, and supplied
per-record vector labels are not used to enforce compatibility.

| imported record | current behavior |
| --- | --- |
| has `embedding_b64` | decode and retain that vector without validating its model/dimension against the destination |
| has no embedded vector | embed its content using the destination's configured model |

for a transfer into a different embedding model, export without embeddings so
import generates destination-model vectors, or deliberately re-embed the
imported records before using the destination for retrieval. provider-backed
embedding can send imported text to that provider. the
[model-switch guide](embedding-backends.md#change-models-on-an-existing-store)
explains config, dimensions and fresh index paths.

## use a fresh index after a bulk transfer

import finishes by calling ANN initialization. that path can load an existing
index rather than force a rebuild. an `ANN index rebuilt` message alone does not
prove a pre-existing cache now covers the imported records.

use a new destination/index path, or retain the old cache and set a new,
nonexistent `ann.index_path` before rebuilding. then restart other processes
using the store and check representative queries. follow the
[ANN index procedure](../architecture/ann-index.md#cli-behavior-in-081).

## exact backup and recovery

for full rollback, retain a database-level backup using the procedure appropriate
to SQLite or Postgres, plus the configuration needed to interpret its vectors.
record exports remain useful for inspection and selected transfers, within the
field and graph limitations above. they do not replace that backup.
