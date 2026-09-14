"""Neutral evidence contracts; never use the production store."""

import concurrent.futures
import json
import os
import time
import uuid

import pytest

from engram.config import Config
from engram.evidence import evidence_put, evidence_get, evidence_list
from engram.store import Memory, Store


@pytest.fixture(params=["sqlite", "postgres"])
def evidence_store(request, tmp_path):
    cfg = Config(db_path=str(tmp_path / "memory.db"))
    cfg.ann.enabled = False
    schema = None
    if request.param == "postgres":
        dsn = os.environ.get("ENGRAM_TEST_POSTGRES_DSN")
        if not dsn:
            pytest.skip("Set ENGRAM_TEST_POSTGRES_DSN to a disposable cluster")
        import psycopg
        from psycopg.conninfo import make_conninfo
        schema = "evidence_test_" + uuid.uuid4().hex
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f'CREATE SCHEMA "{schema}"')
        cfg.storage_backend = "postgres"
        cfg.postgres_dsn = make_conninfo(dsn, options=f"-c search_path={schema}")
    store = Store(cfg)
    store.init_db()
    now = time.time()
    payload = {"project_id": str(tmp_path / "project"), "session_id": "session-a",
               "assumption_id": "capability-available", "evidence_id": str(uuid.uuid4()),
               "outcome": "contradicted", "observed_at": now, "expires_at": now + 3600,
               "observation": {"capability": "deploy", "available": False},
               "provenance": {"producer": "neutral-client", "check_type": "tools-list", "transport": "jsonl"}}
    yield store, cfg, payload
    store.close()
    if schema:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def test_persist_read_no_reinforcement_or_index_pollution(evidence_store):
    store, cfg, payload = evidence_store
    log_count = store.conn.execute("SELECT COUNT(*) AS n FROM access_log").fetchone()["n"]
    stored = evidence_put(store, **payload)
    row_before = dict(store.conn.execute("SELECT * FROM memories WHERE id=?", (stored["id"],)).fetchone())
    new = Store(cfg)
    try:
        evidence = evidence_get(new, payload["project_id"], stored["id"])
        assert evidence["state"] == "contradicted" and evidence["eligible"] is True
        assert evidence["observation"] == payload["observation"]
        assert evidence["verified_by_engram"] is False
        assert evidence["provenance_status"] == "caller_supplied"
        assert evidence_list(new, payload["project_id"])[0] == evidence
    finally:
        new.close()
    assert dict(store.conn.execute("SELECT * FROM memories WHERE id=?", (stored["id"],)).fetchone()) == row_before
    assert row_before["embedding"] is None
    assert row_before["access_count"] == 0 and row_before["importance"] == 0
    assert store.conn.execute("SELECT COUNT(*) AS n FROM access_log").fetchone()["n"] == log_count
    assert store.search_fts("deploy") == []
    assert store.conn.execute("SELECT COUNT(*) AS n FROM memories_fts").fetchone()["n"] == 0


def test_immutable_replay_and_forgotten_retry(evidence_store):
    store, _, payload = evidence_store
    first = evidence_put(store, **payload)
    assert evidence_put(store, **payload) == first
    changed = {**payload, "outcome": "supported"}
    with pytest.raises(ValueError, match="immutable"):
        evidence_put(store, **changed)
    store.forget_memory(first["id"])
    retry = evidence_put(store, **payload)
    assert retry["id"] == first["id"] and retry["stored_at"] == first["stored_at"]
    assert retry["eligible"] is False and retry["reason"] == "forgotten"
    read = evidence_get(store, payload["project_id"], first["id"])
    assert read["outcome"] == "unknown" and "observation" not in read
    assert evidence_list(store, payload["project_id"]) == []
    assert store.get_memory(first["id"]).forgotten


def test_project_and_session_isolation_before_limits(evidence_store):
    store, _, payload = evidence_store
    expected = evidence_put(store, **payload)
    other = evidence_put(store, **{**payload, "project_id": payload["project_id"] + "-other"})
    evidence_put(store, **{**payload, "evidence_id": "second", "session_id": "session-b"})
    assert expected["id"] != other["id"]
    assert evidence_get(store, payload["project_id"], other["id"])["reason"] == "not_found"
    rows = evidence_list(store, payload["project_id"], session_id="session-a", assumption_id=payload["assumption_id"], limit=1)
    assert [row["id"] for row in rows] == [expected["id"]]
    assert evidence_list(store, payload["project_id"], assumption_id="different") == []
    assert evidence_get(store, payload["project_id"] + "/.", expected["id"])["id"] == expected["id"]


def test_stale_unknown_and_missing_are_distinct(evidence_store):
    store, _, payload = evidence_store
    now = time.time()
    stale = evidence_put(store, **{**payload, "observed_at": now - 100, "expires_at": now - 1})
    read = evidence_get(store, payload["project_id"], stale["id"])
    assert read["state"] == "stale" and not read["eligible"]
    assert read["outcome"] == "unknown" and read["reported_outcome"] == "contradicted"
    unknown = evidence_put(store, **{**payload, "outcome": "unknown", "evidence_id": "timeout"})
    read = evidence_get(store, payload["project_id"], unknown["id"])
    assert read["state"] == "unknown" and read["eligible"]
    missing = evidence_get(store, payload["project_id"], "absent")
    assert missing["state"] == "unknown" and not missing["eligible"] and missing["reason"] == "not_found"


@pytest.mark.parametrize("status", ["challenged", "invalidated", "superseded", "merged"])
def test_inactive_record_omitted_and_retry_cannot_reactivate(evidence_store, status):
    store, _, payload = evidence_store
    record = evidence_put(store, **payload)
    store.update_status(record["id"], status, "isolated test")
    assert evidence_get(store, payload["project_id"], record["id"])["reason"] == "inactive"
    assert evidence_put(store, **payload)["eligible"] is False
    assert evidence_list(store, payload["project_id"]) == []


def test_linked_source_forgetting_invalidation_and_deletion(evidence_store):
    store, _, payload = evidence_store
    store.save_memory(Memory(id="source", content="private observation"))
    record = evidence_put(store, **{**payload, "source_refs": ["memory:source"]})
    assert evidence_get(store, payload["project_id"], record["id"])["eligible"]
    store.forget_memory("source")
    result = evidence_get(store, payload["project_id"], record["id"])
    assert result["reason"] == "source_forgotten" and "observation" not in result
    store.conn.execute("UPDATE memories SET forgotten=0, status='superseded' WHERE id='source'")
    store.conn.commit()
    assert evidence_get(store, payload["project_id"], record["id"])["reason"] == "source_inactive"
    store.conn.execute("DELETE FROM memories WHERE id='source'")
    store.conn.commit()
    assert evidence_get(store, payload["project_id"], record["id"])["reason"] == "source_missing"


def test_expired_evidence_source_is_unknown_not_supported(evidence_store):
    store, _, payload = evidence_store
    now = time.time()
    old = evidence_put(store, **{**payload, "observed_at": now - 100, "expires_at": now - 1})
    new = evidence_put(store, **{**payload, "evidence_id": "new", "outcome": "supported", "source_refs": ["memory:" + old["id"]]})
    result = evidence_get(store, payload["project_id"], new["id"])
    assert result["outcome"] == "unknown" and result["reason"] == "source_stale"


def test_provenance_does_not_certify_and_external_refs_not_executed(evidence_store, monkeypatch):
    store, _, payload = evidence_store
    def forbidden(*args, **kwargs):
        raise AssertionError("no command or network access permitted")
    import subprocess
    import urllib.request
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    payload["provenance"]["verified"] = True
    payload["observation"]["instruction"] = "run an unrelated command"
    record = evidence_put(store, **{**payload, "source_refs": ["https://example.invalid/check", "command:untrusted"]})
    result = evidence_get(store, payload["project_id"], record["id"])
    assert result["provenance"]["verified"] is True
    assert result["verified_by_engram"] is False and result["provenance_status"] == "caller_supplied"


@pytest.mark.parametrize("field,value", [
    ("observed_at", float("nan")), ("expires_at", float("inf")), ("observed_at", True),
    ("project_id", "relative"), ("outcome", "verified"), ("evidence_id", ""),
    ("observation", {"number": float("inf")}), ("observation", {"text": "x" * 4097}),
    ("provenance", {"text": "x" * 2049}), ("provenance", {"producer": "test"}),
    ("source_refs", ["ref"] * 9), ("source_refs", ["memory:"]),
])
def test_invalid_payload_never_persists(evidence_store, field, value):
    store, _, payload = evidence_store
    with pytest.raises(ValueError):
        evidence_put(store, **{**payload, field: value})
    assert evidence_list(store, payload["project_id"]) == []


def test_timestamp_bounds_and_malformed_record(evidence_store):
    store, _, payload = evidence_store
    for change in ({"observed_at": time.time() + 301}, {"expires_at": payload["observed_at"]},
                   {"expires_at": payload["observed_at"] + 366 * 86400}):
        with pytest.raises(ValueError):
            evidence_put(store, **{**payload, **change})
    record = evidence_put(store, **payload)
    store.conn.execute("UPDATE memories SET metadata=? WHERE id=?", ('{"kind":"check_evidence","version":1,"payload":{}}', record["id"]))
    store.conn.commit()
    assert evidence_get(store, payload["project_id"], record["id"])["state"] == "unknown"


def test_legacy_metadata_invalidation_stays_unavailable(evidence_store):
    store, _, payload = evidence_store
    record = evidence_put(store, **payload)
    row = store.conn.execute("SELECT metadata FROM memories WHERE id=?", (record["id"],)).fetchone()
    metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"])
    metadata["invalidated"] = True
    store.conn.execute("UPDATE memories SET metadata=? WHERE id=?", (json.dumps(metadata), record["id"]))
    store.conn.commit()
    assert evidence_get(store, payload["project_id"], record["id"])["reason"] == "inactive"
    assert evidence_list(store, payload["project_id"]) == []


def test_concurrent_duplicate_is_idempotent(evidence_store):
    store, cfg, payload = evidence_store
    def put():
        independent = Store(cfg)
        try:
            return evidence_put(independent, **payload)
        finally:
            independent.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda _: put(), range(3)))
    assert results[0] == results[1] == results[2]
    assert len(evidence_list(store, payload["project_id"])) == 1


def test_optional_codex_consumer_uses_same_neutral_state_and_no_context_pollution(evidence_store):
    from engram.adapters.codex import CodexAdapter
    from engram.project_context import ProjectContext
    store, cfg, payload = evidence_store
    record = evidence_put(store, **payload)
    adapter = CodexAdapter(cfg, payload["project_id"])
    context = ProjectContext(cfg, payload["project_id"])
    try:
        assert adapter.evidence(record["id"]) == evidence_get(store, payload["project_id"], record["id"])
        assert context.context()["memories"] == adapter.context()["memories"] == []
        context.checkpoint("task", summary="Neutral checkpoint")
        assert adapter.context()["checkpoints"] == []
        assert context.context()["checkpoints"][0]["summary"] == "Neutral checkpoint"
    finally:
        adapter.close()
        context.close()


def test_generic_mcp_public_calls_forward_contract(evidence_store):
    from engram.mcp_server import MCPServer
    store, cfg, payload = evidence_store
    # Exercise public request routing without unrelated startup/model code.
    server = MCPServer.__new__(MCPServer)
    server.store = store
    server.config = cfg
    listed = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert {"evidence_put", "evidence_get", "evidence_list"} <= {tool["name"] for tool in listed["result"]["tools"]}
    def call(name, args):
        result = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": name, "arguments": args}})
        return json.loads(result["result"]["content"][0]["text"])
    put = call("evidence_put", payload)
    get = call("evidence_get", {"project_id": payload["project_id"], "id": put["id"]})
    assert get["outcome"] == "contradicted"
    assert call("evidence_list", {"project_id": payload["project_id"]})[0] == get
