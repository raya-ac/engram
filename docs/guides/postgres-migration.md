# Postgres Migration

Move an existing Engram install from sqlite to postgres without losing the old database.

## When to migrate

Stay on sqlite if Engram is mostly:

- one local user
- one CLI process
- light web usage

Move to postgres if you're running:

- the web dashboard all the time
- MCP clients alongside the web UI
- multiple concurrent readers and writers
- maintenance operations like drift fixes while other clients are active

## 1. Create an empty Postgres database

Use any normal Postgres instance. Example DSN:

```text
postgresql://user:pass@localhost:5432/engram
```

## 2. Run the migration

```bash
engram migrate-postgres \
  --dsn postgresql://user:pass@localhost:5432/engram \
  --switch-config
```

What this does:

- initializes the target postgres schema
- copies memories, entities, relationships, diary entries, handoffs, and history tables
- rebuilds full-text search data in postgres
- verifies key row counts after the copy
- backs up `config.yaml`
- switches `storage_backend` and `postgres_dsn`

Your original sqlite database is left in place.

## 3. Restart Engram

Restart any running:

- web dashboard
- MCP server
- local agents using Engram

They need to reconnect after the backend switch.

## Preview or verify only

If you want to test the target first:

```bash
engram migrate-postgres \
  --verify-only \
  --dsn postgresql://user:pass@localhost:5432/engram
```

That checks the target connection and prints source/target counts without copying data.

## Reset and rerun

If you want to overwrite the target during another migration attempt:

```bash
engram migrate-postgres \
  --dsn postgresql://user:pass@localhost:5432/engram \
  --force-reset
```

## Config after migration

```yaml
storage_backend: postgres
postgres_dsn: postgresql://user:pass@localhost:5432/engram
db_path: ~/.local/share/engram/memory.db
```

`db_path` can stay there as a rollback backup. Engram ignores it while postgres is the active backend.
