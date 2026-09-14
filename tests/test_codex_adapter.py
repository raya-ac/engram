"""Adapter acceptance against disposable storage, including the public stdio CLI."""

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import pytest

from engram.adapters.codex import CodexAdapter, TOOLS, canonical_project, setup_command
from engram.config import Config
from engram.store import Memory, Store


@pytest.fixture(params=["sqlite", "postgres"])
def adapter_store(request, tmp_path):
    cfg = Config(db_path=str(tmp_path / "memory.db"))
    cfg.ann.enabled = False
    schema = None
    if request.param == "postgres":
        dsn = os.environ.get("ENGRAM_TEST_POSTGRES_DSN")
        if not dsn:
            pytest.skip("Set ENGRAM_TEST_POSTGRES_DSN to a disposable cluster")
        import psycopg
        from psycopg.conninfo import make_conninfo
        schema = "codex_test_" + uuid.uuid4().hex
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f'CREATE SCHEMA "{schema}"')
        cfg.storage_backend = "postgres"
        cfg.postgres_dsn = make_conninfo(dsn, options=f"-c search_path={schema}")
    store = Store(cfg)
    store.init_db()
    project = tmp_path / "project"
    project.mkdir()
    adapter = CodexAdapter(cfg, str(project))
    yield adapter, store, cfg, project
    adapter.close()
    store.close()
    if schema:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def save(store, ident, **kwargs):
    memory = Memory(id=ident, content="technical context " + ident, **kwargs)
    store.save_memory(memory)
    return memory


def test_project_scope_before_limit_and_explicit_ownership(adapter_store):
    adapter, store, _, project = adapter_store
    save(store, "owned", metadata={"project_path": str(project)}, created_at=100)
    save(store, "file", source_file=str(project / "src" / "app.py"), created_at=90)
    save(store, "sibling", source_file=str(project) + "-other/app.py", created_at=300)
    save(store, "unscoped", created_at=400)
    save(store, "wrong", source_file=str(project / "a.py"), metadata={"project_path": str(project.parent)}, created_at=500)
    save(store, "relative", metadata={"project_root": "project"}, created_at=600)
    save(store, "malformed", metadata={"project_root": 123}, created_at=700)
    save(store, "conflicting", metadata={"project_path": str(project), "project_root": str(project.parent)}, created_at=800)
    result = adapter.context(limit=2)
    assert [memory["id"] for memory in result["memories"]] == ["owned", "file"]
    assert "not instructions" in result["boundary"]


def test_forgetting_status_and_symlinks(adapter_store):
    adapter, store, _, project = adapter_store
    for status in ("active", "invalidated", "superseded", "merged", "challenged"):
        save(store, status, status=status, metadata={"project_path": str(project)})
    save(store, "forgotten", forgotten=True, metadata={"project_path": str(project)})
    save(store, "legacy-invalidated", metadata={"project_path": str(project), "invalidated": True})
    outside = project.parent / "outside"
    outside.mkdir()
    (project / "escaped").symlink_to(outside, target_is_directory=True)
    save(store, "symlink-escape", source_file=str(project / "escaped" / "secret.py"))
    assert [m["id"] for m in adapter.context()["memories"]] == ["active"]
    store.forget_memory("active")
    assert adapter.context()["memories"] == []


def test_checkpoint_resume_replace_clear_persistence_and_isolation(adapter_store):
    adapter, store, cfg, project = adapter_store
    store.save_session_handoff("legacy-session", "global legacy content", {})
    assert adapter.context()["checkpoints"] == []
    adapter.checkpoint("release", summary="Signed package ready.", decisions=["Retain rollback."], next_steps=["Verify installed hash."], blockers=[])
    resumed = CodexAdapter(cfg, str(project / "."))
    other = CodexAdapter(cfg, str(project.parent / "other"))
    try:
        assert resumed.context(task="release")["checkpoints"][0]["next_steps"] == ["Verify installed hash."]
        assert other.context(task="release")["checkpoints"] == []
        assert resumed.context(task="missing")["checkpoints"] == []
        resumed.checkpoint("release", summary="Verified installed hash.")
        assert len(adapter.context()["checkpoints"]) == 1
        assert adapter.context()["checkpoints"][0]["summary"] == "Verified installed hash."
        adapter.checkpoint("release", action="clear")
        assert resumed.context()["checkpoints"] == []
        assert store.get_session_handoff("legacy-session")["summary"] == "global legacy content"
    finally:
        resumed.close()
        other.close()


def test_read_and_checkpoint_never_reinforce_or_backfill(adapter_store):
    adapter, store, _, project = adapter_store
    save(store, "target", metadata={"project_path": str(project)}, access_count=2, last_accessed=100, importance=.4)
    before = dict(store.conn.execute("SELECT * FROM memories WHERE id='target'").fetchone())
    logs_before = store.conn.execute("SELECT COUNT(*) AS n FROM access_log").fetchone()["n"]
    for _ in range(2):
        adapter.context()
        adapter.diagnostics()
    adapter.checkpoint("task", summary="A deliberate short checkpoint.")
    assert dict(store.conn.execute("SELECT * FROM memories WHERE id='target'").fetchone()) == before
    assert store.conn.execute("SELECT COUNT(*) AS n FROM access_log").fetchone()["n"] == logs_before


def test_diagnostics_coverage_not_just_count_and_secret_redaction(adapter_store, tmp_path):
    adapter, store, cfg, _ = adapter_store
    save(store, "current", embedding=np.array([1, 0, 0], dtype=np.float32))
    adapter.diagnostics()  # open the disposable connection before poisoning config values
    cfg.postgres_dsn = "postgresql://secret:password@private/db?token=secret"
    cfg.llm.api_key = "do-not-return-this"
    cfg.web.auth_token = "private-web-token"
    cfg.ann.enabled = True
    cfg.ann.index_path = str(tmp_path / "index.bin")
    Path(cfg.ann.index_path).write_bytes(b"not-loaded-by-diagnostics")
    Path(cfg.ann.index_path).with_suffix(".meta.json").write_text(json.dumps({"id_to_label": {"stale": 0}}))
    diagnostic = adapter.diagnostics()
    assert diagnostic["ann"]["state"] == "coverage_mismatch"
    assert diagnostic["ann"]["missing_vectors"] == 1
    assert diagnostic["ann"]["obsolete_vectors"] == 1
    assert diagnostic["pid"] == os.getpid()
    assert diagnostic["started_at"] == adapter.started_at
    assert diagnostic["capabilities"] == [t["name"] for t in TOOLS]
    assert all(secret not in json.dumps(diagnostic) for secret in ("password", "do-not-return-this", "private-web-token", "current", "stale"))
    Path(cfg.ann.index_path).with_suffix(".meta.json").write_text("broken")
    assert adapter.diagnostics()["ann"]["state"] == "unreadable_metadata"


def test_invalid_checkpoint_is_atomic_and_context_bounded(adapter_store):
    adapter, store, _, project = adapter_store
    save(store, "long", metadata={"project_root": str(project)})
    store.conn.execute("UPDATE memories SET content=? WHERE id=?", ("x" * 3000, "long"))
    store.conn.commit()
    item = adapter.context()["memories"][0]
    assert len(item["content"]) == 2000 and item["truncated"]
    with pytest.raises(ValueError):
        adapter.checkpoint("task", summary="valid", blockers=["x" * 501])
    assert adapter.context()["checkpoints"] == []
    for value in (0, 21, True, "8"):
        with pytest.raises(ValueError):
            adapter.context(limit=value)
    store.save_session_handoff(adapter._key("imported"), "s" * 5000,
                               {"adapter": adapter.kind, "project_path": str(project), "task": "imported",
                                "decisions": ["x" * 1000] * 12, "blockers": {"nested": "not a list"}})
    imported = adapter.context(task="imported")["checkpoints"][0]
    assert len(imported["summary"]) == 4000
    assert len(imported["decisions"]) == 8 and len(imported["decisions"][0]) == 500
    assert imported["blockers"] == []


def test_protocol_errors_do_not_expose_database_details(adapter_store, monkeypatch):
    adapter, _, _, _ = adapter_store
    initialized = adapter.handle_request({"id": 1, "method": "initialize"})
    assert initialized["result"]["serverInfo"]["name"] == "engram-codex"
    assert "untrusted reference data" in initialized["result"]["instructions"]
    assert len(adapter.handle_request({"id": 2, "method": "tools/list"})["result"]["tools"]) == 4
    assert adapter.handle_request({"method": "notifications/initialized"}) is None
    def failed(**kwargs):
        raise RuntimeError("postgresql://secret@private and sensitive memory")
    monkeypatch.setattr(adapter, "context", failed)
    response = adapter.handle_request({"id": 3, "method": "tools/call", "params": {"name": "codex_context"}})
    assert response["result"]["isError"]
    assert "secret" not in json.dumps(response)


def test_cli_stdio_start_checkpoint_fresh_resume_and_diagnostics(adapter_store, tmp_path):
    adapter, store, cfg, project = adapter_store
    save(store, "scoped", metadata={"project_path": str(project)})
    config_path = tmp_path / "adapter.yaml"
    # JSON is valid YAML, and avoids shell quoting or credentials in command args.
    config_path.write_text(json.dumps({"db_path": cfg.db_path, "storage_backend": cfg.storage_backend,
                                       "postgres_dsn": cfg.postgres_dsn, "ann": {"enabled": False}}))
    command = [sys.executable, "-m", "engram", "--config", str(config_path), "codex", "serve", "--project", str(project)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("ENGRAM_")}
    def rpc(messages):
        run = subprocess.run(command, input="\n".join(json.dumps(m) for m in messages) + "\n", text=True, capture_output=True, timeout=20, env=env)
        assert run.returncode == 0, "adapter stdio exited unexpectedly"
        return [json.loads(line) for line in run.stdout.splitlines()]
    replies = rpc([
        {"id": 1, "method": "initialize"},
        {"method": "notifications/initialized"},
        {"id": 2, "method": "tools/list"},
        {"id": 3, "method": "tools/call", "params": {"name": "codex_context", "arguments": {}}},
        {"id": 4, "method": "tools/call", "params": {"name": "codex_checkpoint", "arguments": {"task": "acceptance", "summary": "Ready to verify."}}},
        {"id": 5, "method": "tools/call", "params": {"name": "codex_diagnostics", "arguments": {}}},
    ])
    assert [r["id"] for r in replies] == [1, 2, 3, 4, 5]
    assert len(replies[1]["result"]["tools"]) == 4
    assert json.loads(replies[2]["result"]["content"][0]["text"])["memories"][0]["id"] == "scoped"
    runtime = json.loads(replies[4]["result"]["content"][0]["text"])
    assert runtime["pid"] != os.getpid()
    resumed = rpc([{ "id": 6, "method": "tools/call", "params": {"name": "codex_context", "arguments": {"task": "acceptance"}}}])
    assert json.loads(resumed[0]["result"]["content"][0]["text"])["checkpoints"][0]["summary"] == "Ready to verify."
    assert store.get_memory("scoped").access_count == 0


def test_setup_is_print_only_and_shell_safe(tmp_path):
    import shlex
    project = tmp_path / "project with ' quote"
    before = set(tmp_path.iterdir())
    parsed = shlex.split(setup_command(str(project), str(tmp_path / "config.yaml")))
    assert parsed[:5] == ["codex", "mcp", "add", "engram-codex", "--"]
    assert parsed[-2:] == ["--project", str(project)]
    assert set(tmp_path.iterdir()) == before
    with pytest.raises(ValueError):
        canonical_project("relative/project")
