"""Doctor safety boundaries and actual isolated MCP/storage acceptance."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

from engram import __version__
from engram.config import Config
from engram import doctor as diagnostic


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    config = Config(db_path=str(tmp_path / "existing" / "memory.db"), embedding_backend="sentence_transformers")
    config.ann.index_path = str(tmp_path / "existing" / "hnsw.index")
    return config


@pytest.fixture
def ready(monkeypatch):
    monkeypatch.setattr(diagnostic, "_model_readiness", lambda config: [
        diagnostic._check("embedding_readiness", "pass", "fixture"),
        diagnostic._check("reranker_readiness", "pass", "fixture"),
    ])


def check(report, name):
    return next(item for item in report["checks"] if item["name"] == name)


def test_default_does_not_create_database_or_spawn_process(cfg, ready, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("default doctor spawned a process"))
    result = diagnostic.doctor(cfg)
    assert result["status"] == "fail"
    assert check(result, "storage")["status"] == "fail"
    assert check(result, "embedding_inference")["status"] == "skipped"
    assert check(result, "mcp_connection")["status"] == "skipped"
    assert not Path(cfg.db_path).parent.exists()
    assert capsys.readouterr().out == ""


def test_existing_empty_database_is_not_initialized(cfg, ready):
    path = Path(cfg.db_path)
    path.parent.mkdir()
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    result = diagnostic.doctor(cfg)
    assert check(result, "storage")["status"] == "fail"
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("unrelated",)]


def test_live_wal_counts_without_memory_or_access_mutations(cfg, ready):
    from engram.store import Store, Memory
    store = Store(cfg)
    try:
        store.init_db()
        store.save_memory(Memory(id="existing-record", content="keep this original fact", memory_type="fact"))
        before = tuple(store.conn.execute("SELECT content, access_count, last_accessed, status FROM memories").fetchone())
        result = diagnostic.doctor(cfg)
        observed = check(result, "storage")
        assert observed["status"] == "pass"
        assert observed["details"]["counts"]["memories"] == 1
        assert observed["details"]["counts"]["access_log"] == 0
        assert tuple(store.conn.execute("SELECT content, access_count, last_accessed, status FROM memories").fetchone()) == before
        assert result["ok"] is True and result["status"] == "incomplete"
    finally:
        store.close()


def test_sqlite_uri_special_characters(cfg):
    from engram.store import Store
    cfg.db_path = str(Path(cfg.db_path).parent / "memory?doctor#1.db")
    store = Store(cfg)
    store.init_db()
    store.close()
    assert diagnostic._storage_check(cfg, False)["status"] == "pass"


def test_postgres_is_not_contacted_by_default(cfg, ready, monkeypatch):
    cfg.storage_backend = "postgres"
    cfg.postgres_dsn = "postgresql://doctor:secret@localhost/store"
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda *a, **kw: pytest.fail("network attempted")))
    result = diagnostic.doctor(cfg)
    assert check(result, "storage")["status"] == "skipped"
    assert result["status"] == "incomplete"
    assert "secret" not in json.dumps(result)


def test_postgres_explicit_probe_enforces_readonly_and_timeout(cfg, monkeypatch):
    cfg.storage_backend = "postgres"
    cfg.postgres_dsn = "postgresql://doctor:secret@localhost/store"
    statements = []
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql):
            statements.append(sql)
            if "information_schema" in sql:
                return SimpleNamespace(fetchall=lambda: [(table, column) for table, columns in diagnostic._REQUIRED_COLUMNS.items() for column in columns])
            return SimpleNamespace(fetchone=lambda: (0, 0, 0))
    def connect(dsn, **kwargs):
        assert dsn == cfg.postgres_dsn
        assert kwargs["connect_timeout"] == 3
        assert "default_transaction_read_only=on" in kwargs["options"]
        assert "statement_timeout=3000" in kwargs["options"]
        return Connection()
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=connect, conninfo=SimpleNamespace(conninfo_to_dict=lambda dsn: {})))
    assert diagnostic._storage_check(cfg, True)["status"] == "pass"
    assert statements[0] == "SET TRANSACTION READ ONLY"
    assert not any("CREATE" in statement or "UPDATE" in statement for statement in statements)


def test_credentials_and_driver_error_text_are_never_reported(cfg, ready, monkeypatch):
    cfg.storage_backend = "postgres"
    cfg.postgres_dsn = "postgresql://doctor:dsn-secret@localhost/store"
    cfg.hf_token = "hf-private-token"
    cfg.llm.api_key = "provider-private-key"
    cfg.web.auth_token = "web-private-key"
    def failed_connection(*args, **kwargs):
        raise RuntimeError("authentication failed: " + cfg.postgres_dsn + " token=" + cfg.llm.api_key)
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=failed_connection, conninfo=SimpleNamespace(conninfo_to_dict=lambda dsn: {})))
    monkeypatch.setattr(diagnostic, "_connection_check", lambda config: diagnostic._check("mcp_connection", "pass", "fixture"))
    result = diagnostic.doctor(cfg, check_connection=True)
    serialized = json.dumps(result)
    for secret in (cfg.postgres_dsn, cfg.hf_token, cfg.llm.api_key, cfg.web.auth_token):
        assert secret not in serialized
    assert "<redacted>" in serialized
    assert check(result, "storage")["details"]["error_type"] == "RuntimeError"


def test_invalid_config_returns_safe_failure_before_storage(cfg, monkeypatch):
    cfg.retrieval.min_confidence = -1
    monkeypatch.setattr(diagnostic, "_storage_check", lambda *a: pytest.fail("invalid config reached storage"))
    result = diagnostic.doctor(cfg)
    assert result["status"] == "fail"
    assert result["configuration"] is None
    assert "min_confidence" in result["checks"][0]["message"]


def test_cache_check_requires_weights_and_tokenizer(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snapshot = tmp_path / "models--example--model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    assert not diagnostic._cached_model("example/model")["files_present"]
    (snapshot / "model.safetensors").write_bytes(b"fixture")
    assert not diagnostic._cached_model("example/model")["files_present"]
    (snapshot / "tokenizer.json").write_text("{}")
    result = diagnostic._cached_model("example/model")
    assert result["files_present"] and not result["runtime_load_verified"]


def test_basic_doctor_does_not_import_model_runtimes(cfg):
    code = (
        "import json, sys; from engram.config import Config; from engram.doctor import doctor; "
        "doctor(Config(db_path=sys.argv[1])); "
        "print(json.dumps([name for name in ('torch', 'mlx', 'sentence_transformers') if name in sys.modules]))"
    )
    result = subprocess.run([sys.executable, "-c", code, cfg.db_path], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=20, check=True)
    assert json.loads(result.stdout) == []


def test_real_mcp_handshake_is_isolated_and_does_not_warm_models(cfg, monkeypatch):
    monkeypatch.setenv("ENGRAM_STORAGE_BACKEND", "postgres")
    monkeypatch.setenv("ENGRAM_POSTGRES_DSN", "postgresql://do-not-contact/private")
    monkeypatch.setenv("ENGRAM_DB_PATH", cfg.db_path)
    monkeypatch.setenv("ENGRAM_ANN_ENABLED", "true")
    monkeypatch.setenv("ENGRAM_ANN_INDEX_PATH", cfg.ann.index_path)
    monkeypatch.setenv("ENGRAM_DORMANT_RECALL_MODE", "shadow")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    result = diagnostic._connection_check(cfg)
    assert result["status"] == "pass", result
    assert result["details"]["tool_count"] > 0
    assert result["details"]["external_agent_connection_verified"] is False
    assert not Path(cfg.db_path).parent.exists()


def test_mcp_errors_do_not_echo_captured_process_output(cfg, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout="private-api-key", stderr="postgresql://private"))
    result = diagnostic._connection_check(cfg)
    assert result["status"] == "fail"
    assert "private" not in json.dumps(result)


def test_bounded_mcp_timeout(cfg, monkeypatch):
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == diagnostic._CONNECTION_TIMEOUT
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="private-api-key")
    monkeypatch.setattr(subprocess, "run", timeout)
    result = diagnostic._connection_check(cfg)
    assert result["status"] == "fail" and "timed out" in result["message"]
    assert "private-api-key" not in json.dumps(result)


@pytest.fixture
def synthetic_models(monkeypatch):
    import numpy as np
    import engram.embeddings as embeddings
    import engram.retrieval as retrieval
    def rank(query, documents, model):
        return sorted([(index, 5.0 if "marker is copper" in text else -5.0) for index, text in enumerate(documents)], key=lambda item: -item[1])
    monkeypatch.setattr(embeddings, "set_backend", lambda value: None)
    monkeypatch.setattr(embeddings, "set_default_model", lambda value: None)
    monkeypatch.setattr(embeddings, "embed_documents", lambda texts, model: np.array([[1., 0.], [0., 1.], [0., 1.]], dtype=np.float32))
    monkeypatch.setattr(embeddings, "embed_query", lambda text, model=None: np.array([1., 0.], dtype=np.float32))
    monkeypatch.setattr(embeddings, "cross_encoder_rerank", rank)
    monkeypatch.setattr(retrieval, "embed_query", embeddings.embed_query)
    monkeypatch.setattr(retrieval, "cross_encoder_rerank", rank)


def test_smoke_uses_real_isolated_storage_and_keeps_user_config(cfg, synthetic_models):
    cfg.embedding_dim = 2
    before = asdict(cfg)
    results = diagnostic._inference_checks(cfg, smoke=True)
    result = next(item for item in results if item["name"] == "isolated_retrieval")
    assert result["status"] == "pass", results
    assert result["details"]["stored_count"] == 3
    assert result["details"]["expected_rank"] == 1
    assert asdict(cfg) == before
    assert not Path(cfg.db_path).parent.exists()
    assert not list(Path(tempfile.tempdir).glob("engram-doctor-*"))


def test_smoke_does_not_lower_configured_confidence_gate(cfg, synthetic_models):
    cfg.embedding_dim = 2
    cfg.retrieval.min_confidence = 1.0
    results = diagnostic._inference_checks(cfg, smoke=True)
    result = next(item for item in results if item["name"] == "isolated_retrieval")
    assert result["status"] == "fail", results
    assert result["details"]["returned_count"] == 0


def test_embedding_dimension_failure_skips_smoke(cfg, synthetic_models):
    cfg.embedding_dim = 384
    results = diagnostic._inference_checks(cfg, smoke=True)
    assert next(item for item in results if item["name"] == "embedding_inference")["status"] == "fail"
    assert next(item for item in results if item["name"] == "isolated_retrieval")["status"] == "skipped"


def test_model_worker_has_bounded_timeout_and_no_error_echo(cfg, monkeypatch):
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == diagnostic._INFERENCE_TIMEOUT
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], stderr="provider-secret")
    monkeypatch.setattr(subprocess, "run", timeout)
    results = diagnostic._run_inference(cfg, False)
    assert results[0]["status"] == "fail"
    assert "provider-secret" not in json.dumps(results)


def test_full_status_requires_every_requested_check_to_pass(cfg, ready, monkeypatch):
    monkeypatch.setattr(diagnostic, "_storage_check", lambda *a: diagnostic._check("storage", "pass", "fixture"))
    monkeypatch.setattr(diagnostic, "_connection_check", lambda *a: diagnostic._check("mcp_connection", "pass", "fixture"))
    monkeypatch.setattr(diagnostic, "_run_inference", lambda config, smoke: [
        diagnostic._check("embedding_inference", "pass", "fixture"),
        diagnostic._check("reranker_inference", "pass", "fixture"),
        diagnostic._check("isolated_retrieval", "pass", "fixture"),
    ])
    assert diagnostic.doctor(cfg)["status"] == "incomplete"
    assert diagnostic.doctor(cfg, check_models=True, check_connection=True, smoke=True)["status"] == "pass"


def test_successful_inference_resolves_initial_cache_warning(cfg, monkeypatch):
    monkeypatch.setattr(diagnostic, "_storage_check", lambda *a: diagnostic._check("storage", "pass", "fixture"))
    monkeypatch.setattr(diagnostic, "_connection_check", lambda *a: diagnostic._check("mcp_connection", "pass", "fixture"))
    monkeypatch.setattr(diagnostic, "_model_readiness", lambda config: [diagnostic._check("embedding_readiness", "warning", "files not present yet")])
    monkeypatch.setattr(diagnostic, "_run_inference", lambda config, smoke: [
        diagnostic._check("embedding_inference", "pass", "downloaded and loaded"),
        diagnostic._check("reranker_inference", "pass", "fixture"),
        diagnostic._check("isolated_retrieval", "pass", "fixture"),
    ])
    result = diagnostic.doctor(cfg, check_models=True, check_connection=True, smoke=True)
    assert result["status"] == "pass"
    assert check(result, "embedding_readiness")["details"]["runtime_load_verified"] is True
