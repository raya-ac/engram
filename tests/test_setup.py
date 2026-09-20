"""Isolated, model-free setup tests. Never use a developer's config or store."""

import builtins
import json
import os
import shlex
import stat
import sys
from pathlib import Path

import pytest
import yaml

from engram.config import Config
from engram.setup import SetupError, initialize
from engram import setup


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    names = {item["env"] for item in Config.schema()["fields"].values()}
    names.update({"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"})
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def run_setup(tmp_path, **kwargs):
    options = {"config_path": tmp_path / "config.yaml", "db_path": tmp_path / "memory.db", "yes": True, "output_fn": lambda text: None}
    options.update(kwargs)
    return initialize(**options)


def test_fresh_sqlite_store_is_private_and_usable_without_llm(tmp_path):
    from engram.store import Store, Memory
    result = run_setup(tmp_path)
    config = Config.load(result["config_file"])
    assert config.db_path == str(tmp_path / "memory.db")
    assert result["database_initialized"] is True
    assert result["models"]["embedding_dim"] == 384
    assert stat.S_IMODE((tmp_path / "config.yaml").stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "memory.db").stat().st_mode) == 0o600
    store = Store(config)
    try:
        memory = Memory(id="setup-test", content="The fictional release token is lilac fern.")
        store.save_memory(memory)
        assert store.get_memory("setup-test").content == memory.content
        assert store.search_fts("lilac fern", limit=5)[0][0] == "setup-test"
    finally:
        store.close()
    assert not list(tmp_path.glob(".engram-*"))


@pytest.mark.parametrize("preset, backend, reranker", [
    ("local", "auto", "BAAI/bge-reranker-base"),
    ("portable", "sentence_transformers", "BAAI/bge-reranker-base"),
    ("light", "sentence_transformers", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
])
def test_practical_local_presets(tmp_path, preset, backend, reranker):
    result = run_setup(tmp_path, preset=preset)
    assert result["models"]["embedding_backend"] == backend
    assert result["models"]["cross_encoder_model"] == reranker


def test_model_override_derives_dimension_and_backend_override(tmp_path):
    result = run_setup(tmp_path, embedding_model="BAAI/bge-base-en-v1.5", embedding_backend="sentence_transformers")
    assert result["models"]["embedding_dim"] == 768
    assert result["models"]["embedding_backend"] == "sentence_transformers"
    assert "embedding_dim" not in yaml.safe_load(Path(result["config_file"]).read_text())


@pytest.mark.parametrize("options, match", [
    ({"storage": "private-secret"}, "storage"),
    ({"preset": "private-secret"}, "preset"),
    ({"embedding_backend": "voyage"}, "embedding_backend"),
    ({"embedding_model": "voyage-3.5"}, "embedding_model"),
    ({"embedding_model": "private-secret"}, "embedding_model"),
    ({"cross_encoder_model": "rerank-2.5"}, "cross_encoder_model"),
    ({"storage": "postgres"}, "postgres_dsn"),
])
def test_invalid_plan_creates_no_files_and_no_secret_errors(tmp_path, options, match):
    with pytest.raises(SetupError, match=match) as exc:
        run_setup(tmp_path, **options)
    assert "private-secret" not in str(exc.value)
    assert list(tmp_path.iterdir()) == []


def test_existing_config_is_never_changed(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_bytes(b"existing config contents")
    with pytest.raises(SetupError, match="config_file already exists"):
        run_setup(tmp_path)
    assert config.read_bytes() == b"existing config contents"
    assert not (tmp_path / "memory.db").exists()


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_existing_database_or_sidecar_is_never_changed(tmp_path, suffix):
    path = tmp_path / ("memory.db" + suffix)
    path.write_bytes(b"existing database contents")
    with pytest.raises(SetupError, match="db_path"):
        run_setup(tmp_path)
    assert path.read_bytes() == b"existing database contents"
    assert not (tmp_path / "config.yaml").exists()


def test_broken_config_symlink_is_not_followed(tmp_path):
    target = tmp_path / "missing.yaml"
    (tmp_path / "config.yaml").symlink_to(target)
    with pytest.raises(SetupError, match="config_file"):
        run_setup(tmp_path)
    assert (tmp_path / "config.yaml").is_symlink()
    assert not target.exists()


@pytest.mark.parametrize("path", ["config.yaml", "memory.hnsw.index"])
def test_database_cannot_share_config_or_index_path(tmp_path, path):
    options = {"db_path": tmp_path / path}
    if path.endswith("index"):
        options["config_path"] = tmp_path / "memory.hnsw.index"
    with pytest.raises(SetupError, match="distinct"):
        run_setup(tmp_path, **options)
    assert list(tmp_path.iterdir()) == []


def test_existing_ann_file_is_preserved(tmp_path):
    path = tmp_path / "memory.hnsw.index"
    path.write_bytes(b"existing index")
    with pytest.raises(SetupError, match="ann.index_path"):
        run_setup(tmp_path)
    assert path.read_bytes() == b"existing index"
    assert not (tmp_path / "config.yaml").exists()


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_config_must_not_be_written_as_a_database_sidecar(tmp_path, suffix):
    with pytest.raises(SetupError, match="sidecar"):
        run_setup(tmp_path, config_path=tmp_path / ("memory.db" + suffix))
    assert list(tmp_path.iterdir()) == []


def test_invalid_path_is_rejected_before_creating_files(tmp_path):
    with pytest.raises(SetupError, match="config_file"):
        run_setup(tmp_path, config_path=str(tmp_path / "config.yaml") + "\0")
    assert list(tmp_path.iterdir()) == []


def test_invalid_environment_is_rejected_before_any_file_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_RETRIEVAL_MIN_CONFIDENCE", "nan")
    with pytest.raises(SetupError, match="retrieval.min_confidence"):
        run_setup(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_inherited_environment_applies_and_secrets_are_not_copied(tmp_path, monkeypatch):
    actual_db = tmp_path / "env-store.db"
    monkeypatch.setenv("ENGRAM_DB_PATH", str(actual_db))
    monkeypatch.setenv("ENGRAM_RETRIEVAL_TOP_K", "4")
    monkeypatch.setenv("ENGRAM_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    monkeypatch.setenv("ENGRAM_HF_TOKEN", "private-secret-token")
    output = []
    result = run_setup(tmp_path, output_fn=output.append)
    assert actual_db.exists()
    assert not (tmp_path / "memory.db").exists()
    assert result["models"]["embedding_dim"] == 768
    assert result["db_path"] == str(actual_db)
    assert {row["variable"] for row in result["environment_overrides"]} >= {"ENGRAM_DB_PATH", "ENGRAM_RETRIEVAL_TOP_K", "ENGRAM_HF_TOKEN"}
    assert "private-secret-token" not in Path(result["config_file"]).read_text()
    assert "private-secret-token" not in json.dumps(result)
    assert "private-secret-token" not in "\n".join(output)
    assert os.environ["ENGRAM_DB_PATH"] == str(actual_db)
    assert Config.load(result["config_file"]).retrieval.top_k == 4


def test_existing_database_from_environment_is_not_touched(tmp_path, monkeypatch):
    actual_db = tmp_path / "existing.db"
    actual_db.write_bytes(b"private existing store")
    monkeypatch.setenv("ENGRAM_DB_PATH", str(actual_db))
    with pytest.raises(SetupError, match="db_path already exists"):
        run_setup(tmp_path)
    assert actual_db.read_bytes() == b"private existing store"
    assert not (tmp_path / "config.yaml").exists()


def test_agent_snippet_uses_absolute_config_without_secret_or_shell_interpolation(tmp_path):
    path = tmp_path / "config spaces $(do-not-run).yaml"
    result = run_setup(tmp_path, config_path=path)
    entry = result["agent_setup"]["mcp_config"]["mcpServers"]["engram"]
    expected = [sys.executable, "-m", "engram", "--config", str(path), "serve", "--mcp"]
    assert [entry["command"], *entry["args"]] == expected
    assert shlex.split(result["agent_setup"]["command"]) == expected
    assert shlex.split(result["next_steps"][0]) == [sys.executable, "-m", "engram", "--config", str(path), "doctor", "--full"]


def test_guided_setup_confirms_before_creating_files(tmp_path):
    answers = iter(["", "light", "", "", "yes"])
    result = run_setup(tmp_path, yes=False, input_fn=lambda prompt: next(answers))
    assert result["preset"] == "light"


def test_cancelled_setup_creates_no_directories(tmp_path):
    answers = iter(["", "", "", "", "no"])
    with pytest.raises(SetupError, match="cancelled"):
        run_setup(tmp_path, config_path=tmp_path / "config-dir" / "config.yaml", db_path=tmp_path / "data-dir" / "memory.db", yes=False, input_fn=lambda prompt: next(answers))
    assert list(tmp_path.iterdir()) == []


def test_noninteractive_setup_requires_yes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SetupError, match="--yes"):
        run_setup(tmp_path, yes=False)
    assert list(tmp_path.iterdir()) == []


def test_config_race_preserves_other_file_and_rolls_back_only_new_database(tmp_path, monkeypatch):
    original_publish = setup._publish
    def racing_publish(source, target):
        if target.name == "config.yaml":
            target.write_bytes(b"another process config")
        return original_publish(source, target)
    monkeypatch.setattr(setup, "_publish", racing_publish)
    with pytest.raises(SetupError, match="another process"):
        run_setup(tmp_path)
    assert (tmp_path / "config.yaml").read_bytes() == b"another process config"
    assert not (tmp_path / "memory.db").exists()
    assert not list(tmp_path.glob(".engram-*"))


def test_database_race_never_overwrites_other_file(tmp_path, monkeypatch):
    original_publish = setup._publish
    def racing_publish(source, target):
        if target.name == "memory.db":
            target.write_bytes(b"another process database")
        return original_publish(source, target)
    monkeypatch.setattr(setup, "_publish", racing_publish)
    with pytest.raises(SetupError, match="another process"):
        run_setup(tmp_path)
    assert (tmp_path / "memory.db").read_bytes() == b"another process database"
    assert not (tmp_path / "config.yaml").exists()


def test_failed_storage_initialization_has_safe_error_and_no_published_files(tmp_path, monkeypatch):
    class BrokenStore:
        def init_db(self):
            raise RuntimeError("postgresql://private:secret@database")
        def close(self):
            pass
    monkeypatch.setattr(setup, "_new_store", lambda config: BrokenStore())
    with pytest.raises(SetupError) as exc:
        run_setup(tmp_path)
    assert "private" not in str(exc.value)
    assert "secret" not in str(exc.value)
    assert list(tmp_path.iterdir()) == []


class FakePostgres:
    def __init__(self, occupied=False, acquired=True):
        self.occupied = occupied
        self.acquired = acquired
        self.queries = []
        self.initialized = False
        self.closed = False
        self.conn = self
    def execute(self, query):
        self.queries.append(query)
        return self
    def fetchone(self):
        return {"occupied": self.occupied, "acquired": self.acquired}
    def init_db(self):
        self.initialized = True
    def close(self):
        self.closed = True


def test_postgres_uses_environment_without_printing_or_saving_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_POSTGRES_DSN", "postgresql://private:secret@database")
    store = FakePostgres()
    monkeypatch.setattr(setup, "_new_store", lambda config: store)
    result = run_setup(tmp_path, storage="postgres")
    assert store.initialized and store.closed
    assert result["storage"] == "postgres"
    assert result["db_path"] is None
    assert "private" not in json.dumps(result)
    assert "secret" not in Path(result["config_file"]).read_text()
    assert any("pg_try_advisory_lock" in query for query in store.queries)


def test_postgres_refuses_nonempty_schema_without_initialization(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_POSTGRES_DSN", "postgresql://private:secret@database")
    store = FakePostgres(occupied=True)
    monkeypatch.setattr(setup, "_new_store", lambda config: store)
    with pytest.raises(SetupError, match="empty current schema"):
        run_setup(tmp_path, storage="postgres")
    assert not store.initialized
    assert store.closed
    assert list(tmp_path.iterdir()) == []


def test_postgres_refuses_concurrent_initialization_without_waiting(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_POSTGRES_DSN", "postgresql://private:secret@database")
    store = FakePostgres(acquired=False)
    monkeypatch.setattr(setup, "_new_store", lambda config: store)
    with pytest.raises(SetupError, match="already running"):
        run_setup(tmp_path, storage="postgres")
    assert not store.initialized
    assert store.closed
    assert list(tmp_path.iterdir()) == []


def test_postgres_connection_default_timeout_does_not_modify_original_config(monkeypatch):
    from psycopg.conninfo import conninfo_to_dict
    import engram.store
    monkeypatch.delenv("PGCONNECT_TIMEOUT", raising=False)
    config = Config(storage_backend="postgres", postgres_dsn="postgresql://example.invalid/memory")
    captured = []
    monkeypatch.setattr(engram.store, "Store", lambda candidate: captured.append(candidate))
    setup._new_store(config)
    assert conninfo_to_dict(captured[0].postgres_dsn)["connect_timeout"] == "10"
    assert "connect_timeout" not in config.postgres_dsn
    monkeypatch.setenv("PGCONNECT_TIMEOUT", "7")
    setup._new_store(config)
    assert conninfo_to_dict(captured[1].postgres_dsn)["connect_timeout"] == "7"
    config.postgres_dsn += "?connect_timeout=4"
    setup._new_store(config)
    assert conninfo_to_dict(captured[2].postgres_dsn)["connect_timeout"] == "4"


def test_setup_never_imports_model_runtimes_or_clients(tmp_path, monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "sentence_transformers", "mlx", "openai", "anthropic", "voyageai"}:
            raise AssertionError("setup attempted a model or provider import")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    run_setup(tmp_path)
