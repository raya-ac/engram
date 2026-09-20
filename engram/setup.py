"""Create a new Engram store and config, without installing or editing clients."""

from __future__ import annotations

import copy
import json
import os
import shlex
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import yaml

from engram.config import Config, ConfigError


class SetupError(ValueError):
    """A setup failure safe to display without underlying credentials."""


PRESETS = {
    "local": {
        "description": "automatic local embedding runtime, BGE embeddings and reranker",
        "embedding_backend": "auto",
        "cross_encoder_model": "BAAI/bge-reranker-base",
    },
    "portable": {
        "description": "sentence-transformers runtime, BGE embeddings and reranker",
        "embedding_backend": "sentence_transformers",
        "cross_encoder_model": "BAAI/bge-reranker-base",
    },
    "light": {
        "description": "sentence-transformers runtime, BGE embeddings and smaller MiniLM reranker",
        "embedding_backend": "sentence_transformers",
        "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    },
}
LOCAL_BACKENDS = ("auto", "mlx", "sentence_transformers")


def _path(value, field: str) -> Path:
    try:
        if not os.fspath(value).strip() or "\0" in os.fspath(value):
            raise ValueError
        return Path(os.path.abspath(os.path.expanduser(value)))
    except (OSError, TypeError, ValueError, RuntimeError):
        raise SetupError(f"{field} must be a valid nonempty file path") from None


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _new_target(path: Path, field: str) -> None:
    if _exists(path):
        raise SetupError(f"{field} already exists; choose a new path. Existing files are never overwritten")


def _new_sqlite_target(path: Path) -> None:
    _new_target(path, "db_path")
    for suffix in ("-wal", "-shm", "-journal"):
        if _exists(Path(str(path) + suffix)):
            raise SetupError("db_path has existing SQLite sidecar files; choose a new path")


def _prompt(input_fn, prompt, default):
    try:
        value = input_fn(f"{prompt} [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise SetupError("Setup cancelled; no files were created") from None
    return value or default


def _private_temp(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=prefix, dir=parent)
    os.close(fd)
    return Path(name)


def _publish(source: Path, target: Path) -> tuple[int, int]:
    """An atomic hard link fails if any file or symlink already owns the name."""
    try:
        os.link(source, target)
    except FileExistsError:
        raise SetupError("setup destination was created by another process; no existing file was overwritten") from None
    stat = source.stat()
    return stat.st_dev, stat.st_ino


def _remove_owned(path: Path, identity: tuple[int, int]) -> None:
    """Rollback only a file still owned by this setup attempt."""
    try:
        stat = path.lstat()
        if (stat.st_dev, stat.st_ino) == identity:
            path.unlink()
    except FileNotFoundError:
        pass


def _new_store(config):
    from engram.store import Store
    if config.normalized_storage_backend == "postgres":
        from psycopg.conninfo import conninfo_to_dict, make_conninfo
        options = conninfo_to_dict(config.postgres_dsn)
        options.setdefault("connect_timeout", os.environ.get("PGCONNECT_TIMEOUT", "10"))
        config = copy.deepcopy(config)
        config.postgres_dsn = make_conninfo(**options)
    return Store(config)


def _initialize_sqlite(config: Config, destination: Path, config_temp: Path, config_path: Path) -> None:
    _new_sqlite_target(destination)
    database_temp = _private_temp(destination.parent, ".engram-db-")
    published = None
    store = None
    try:
        temporary_config = copy.deepcopy(config)
        temporary_config.db_path = str(database_temp)
        store = _new_store(temporary_config)
        store.init_db()
        # A fresh schema must support both memory reads and native checkpoints.
        store.conn.execute("SELECT id, status, forgotten FROM memories LIMIT 0")
        store.conn.execute("SELECT session_id FROM session_handoffs LIMIT 0")
        store.close()
        store = None
        with database_temp.open("rb") as handle:
            os.fsync(handle.fileno())
        _new_sqlite_target(destination)
        published = _publish(database_temp, destination)
        _publish(config_temp, config_path)
    except BaseException:
        if published is not None:
            _remove_owned(destination, published)
        raise
    finally:
        if store is not None:
            store.close()
        for path in (database_temp, *(Path(str(database_temp) + suffix) for suffix in ("-wal", "-shm", "-journal"))):
            path.unlink(missing_ok=True)


def _initialize_postgres(config: Config, config_temp: Path, config_path: Path) -> None:
    # The schema must be empty. Store.init_db on an existing store could perform
    # migrations, which are outside this new-store setup operation.
    store = _new_store(config)
    published = None
    try:
        lock = store.conn.execute("SELECT pg_try_advisory_lock(hashtext(current_database()), hashtext(current_schema())) AS acquired").fetchone()
        if not lock["acquired"]:
            raise SetupError("postgres setup is already running for this schema; retry after it finishes")
        row = store.conn.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_class "
            "WHERE relnamespace = current_schema()::regnamespace "
            "AND relkind IN ('r', 'p', 'v', 'm', 'f', 'S')) AS occupied"
        ).fetchone()
        if row["occupied"]:
            raise SetupError("postgres storage requires an empty current schema; existing tables are never changed")
        # Publish first so a competing config creator cannot leave a newly
        # committed schema with no configuration. Failed DDL rolls back on close.
        published = _publish(config_temp, config_path)
        store.init_db()
    except BaseException:
        if published is not None:
            _remove_owned(config_path, published)
        raise
    finally:
        store.close()


def initialize(
    config_path: str | Path | None = None,
    *,
    db_path: str | Path | None = None,
    storage: str = "sqlite",
    postgres_dsn: str | None = None,
    preset: str = "local",
    embedding_backend: str | None = None,
    embedding_model: str | None = None,
    cross_encoder_model: str | None = None,
    yes: bool = False,
    input_fn=None,
    output_fn=print,
) -> dict:
    """Guide setup, or accept the chosen defaults noninteractively with yes=True.

    Explicit options become file settings; inherited environment overrides retain
    Config's usual precedence. No inherited secret is copied into the file or
    agent snippet. The caller can suppress output_fn for a structured CLI result.
    """
    if not yes and input_fn is None and not sys.stdin.isatty():
        raise SetupError("interactive setup requires a terminal; use --yes for noninteractive setup")
    input_fn = input_fn or input
    config_path = config_path if config_path is not None else Path.home() / ".config" / "engram" / "config.yaml"
    db_path = db_path if db_path is not None else Config().db_path
    if not yes:
        storage = _prompt(input_fn, "storage: sqlite or postgres", storage)
        for name, details in PRESETS.items():
            output_fn(f"{name}: {details['description']}")
        preset = _prompt(input_fn, "model preset: local, portable or light", preset)
        config_path = _prompt(input_fn, "new config path", str(config_path))
        if storage == "sqlite":
            db_path = _prompt(input_fn, "new SQLite database path", str(db_path))
    if storage not in {"sqlite", "postgres"}:
        raise SetupError("storage must be sqlite or postgres")
    if preset not in PRESETS:
        raise SetupError("preset must be local, portable, or light")
    if embedding_backend is not None and embedding_backend not in LOCAL_BACKENDS:
        raise SetupError("embedding_backend must be auto, mlx, or sentence_transformers for init")

    destination = _path(config_path, "config_file")
    _new_target(destination, "config_file")
    selected_db = _path(db_path, "db_path")
    values = asdict(Config())
    values.update({
        "storage_backend": storage,
        "db_path": str(selected_db),
        "embedding_backend": embedding_backend if embedding_backend is not None else PRESETS[preset]["embedding_backend"],
        "cross_encoder_model": cross_encoder_model if cross_encoder_model is not None else PRESETS[preset]["cross_encoder_model"],
    })
    # Keep dimension derivation working if an inherited model override is used.
    values.pop("embedding_dim")
    if embedding_model is not None:
        values["embedding_model"] = embedding_model
    if postgres_dsn is not None:
        values["postgres_dsn"] = postgres_dsn
    index_path = selected_db.with_suffix(".hnsw.index") if storage == "sqlite" else destination.with_suffix(".hnsw.index")
    values["ann"]["index_path"] = str(index_path)
    try:
        config = Config.from_mapping(values)
    except ConfigError as exc:
        raise SetupError(str(exc)) from None
    from engram.embeddings import MODEL_BACKENDS, MODEL_DIMS, RERANKER_BACKENDS
    if config.embedding_backend not in LOCAL_BACKENDS:
        raise SetupError("embedding_backend must select a local runtime for init; check environment overrides")
    if config.embedding_model not in MODEL_DIMS or config.embedding_model in MODEL_BACKENDS:
        raise SetupError("embedding_model must be a known local model for init; check the model and environment overrides")
    if config.cross_encoder_model in RERANKER_BACKENDS:
        raise SetupError("cross_encoder_model must select a local model for init")
    actual_storage = config.normalized_storage_backend
    actual_db = _path(config.db_path, "db_path") if actual_storage == "sqlite" else None
    actual_index = _path(config.ann.index_path, "ann.index_path")
    if destination.resolve() == actual_index.resolve() or (actual_db and actual_db.resolve() in {destination.resolve(), actual_index.resolve()}):
        raise SetupError("config_file, db_path and ann.index_path must use distinct file paths")
    if actual_db:
        sidecars = {Path(str(actual_db) + suffix).resolve() for suffix in ("-wal", "-shm", "-journal")}
        if destination.resolve() in sidecars or actual_index.resolve() in sidecars:
            raise SetupError("config_file and ann.index_path must not use SQLite sidecar paths")
        _new_sqlite_target(actual_db)
    report = config.describe()
    environment = [{"field": field, "variable": source.removeprefix("env:")}
                   for field, source in report["sources"].items() if source.startswith("env:")]
    warnings = list(report["warnings"])
    if environment:
        names = sorted({item["variable"] for item in environment})
        warnings.append("Environment overrides are active: " + ", ".join(names) + ". Preserve these variables in the agent process; their values are not copied into its config snippet.")
    if actual_storage == "postgres" and not values["postgres_dsn"]:
        warnings.append("The new config relies on ENGRAM_POSTGRES_DSN in each process that uses it.")
    for name in ("db_path", "ann.index_path"):
        value = config.db_path if name == "db_path" else config.ann.index_path
        if not Path(os.path.expanduser(value)).is_absolute():
            warnings.append(f"{name} is relative in the inherited environment; use an absolute path before launching an agent from another directory.")
    if config.ann.enabled and _exists(actual_index):
        raise SetupError("ann.index_path already exists; choose a separate index path with ENGRAM_ANN_INDEX_PATH")
    output_fn(f"config: {destination}")
    output_fn(f"storage: {actual_storage}" + (f" ({actual_db})" if actual_db else " (empty current schema required)"))
    output_fn(f"models: {config.embedding_model}; {config.cross_encoder_model}; backend {config.embedding_backend}")
    for warning in warnings:
        output_fn(f"warning: {warning}")
    if not yes and _prompt(input_fn, "create this config and initialize new storage? y/N", "n").lower() not in {"y", "yes"}:
        raise SetupError("Setup cancelled; no files were created")

    config_temp = None
    try:
        config_temp = _private_temp(destination.parent, ".engram-config-")
        with config_temp.open("w", encoding="utf-8") as handle:
            handle.write("# Created by engram init. Environment overrides still take precedence.\n")
            yaml.safe_dump(values, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        if actual_db is not None:
            _initialize_sqlite(config, actual_db, config_temp, destination)
        else:
            _initialize_postgres(config, config_temp, destination)
    except SetupError:
        raise
    except (OSError, RuntimeError, ValueError, ImportError):
        raise SetupError("setup could not initialize the selected storage or publish its config; check filesystem permissions and storage connectivity") from None
    except Exception:
        raise SetupError("setup could not initialize the selected storage; check connectivity and schema permissions") from None
    finally:
        if config_temp is not None:
            config_temp.unlink(missing_ok=True)

    args = ["-m", "engram", "--config", str(destination)]
    launch = [sys.executable, *args, "serve", "--mcp"]
    agent_setup = {
        "command": shlex.join(launch),
        "mcp_config": {"mcpServers": {"engram": {"command": sys.executable, "args": [*args, "serve", "--mcp"]}}},
        "environment_variables": sorted({item["variable"] for item in environment}),
    }
    doctor = shlex.join([sys.executable, *args, "doctor", "--full"])
    output_fn("new config and storage initialized; models have not been loaded")
    output_fn("agent MCP command: " + agent_setup["command"])
    output_fn(json.dumps(agent_setup["mcp_config"], indent=2))
    output_fn("next: " + doctor)
    return {
        "config_file": str(destination), "storage": actual_storage,
        "db_path": str(actual_db) if actual_db else None,
        "database_initialized": True, "preset": preset,
        "models": {name: getattr(config, name) for name in ("embedding_backend", "embedding_model", "embedding_dim", "cross_encoder_model")},
        "environment_overrides": environment, "warnings": warnings,
        "agent_setup": agent_setup, "next_steps": [doctor],
    }
