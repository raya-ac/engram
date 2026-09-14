"""Isolated dormant recall contracts. No model downloads or production databases."""

import copy
import os
import time
import uuid
from dataclasses import replace

import numpy as np
import pytest

from engram.config import Config, DormantRecallConfig
from engram.dormant import DAY, evaluate_shadow, feedback, inspect_event, review
from engram.store import Memory, Store
from engram import retrieval

NOW = 2_000_000_000.0
TYPES = {"fact", "procedure", "narrative"}
QUERY = "postgres recovery rollback"


@pytest.fixture(params=["sqlite", "postgres"])
def isolated(request, tmp_path, monkeypatch):
    cfg = Config(db_path=str(tmp_path / "isolated.db"))
    cfg.ann.enabled = False
    cfg.dormant_recall.mode = "shadow"
    schema = None
    if request.param == "postgres":
        dsn = os.environ.get("ENGRAM_TEST_POSTGRES_DSN")
        if not dsn:
            pytest.skip("Set ENGRAM_TEST_POSTGRES_DSN to a disposable test cluster")
        import psycopg
        from psycopg.conninfo import make_conninfo
        schema = "dormant_test_" + uuid.uuid4().hex
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f'CREATE SCHEMA "{schema}"')
        cfg.storage_backend = "postgres"
        cfg.postgres_dsn = make_conninfo(dsn, options=f"-c search_path={schema}")
    store = Store(cfg)
    store.init_db()
    monkeypatch.setattr("engram.dormant.time.time", lambda: NOW)
    yield store, cfg
    store.close()
    if schema:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def memory(store, name="dormant", days=90, **kwargs):
    mem = Memory(id=name, content=f"postgres recovery rollback notes {name}",
                 created_at=NOW - days * DAY, last_accessed=NOW - days * DAY,
                 importance=0.8, **kwargs)
    store.save_memory(mem)
    return mem


def candidates(monkeypatch, hits):
    monkeypatch.setattr(retrieval, "_dense_search", lambda q, s, c, limit: hits[:limit])


def snapshot(store, mid):
    mem = store.get_memory(mid)
    return mem.access_count, mem.last_accessed, mem.importance, copy.deepcopy(mem.metadata)


def event(store, cfg, ordinary=(), types=TYPES):
    eid = evaluate_shadow(QUERY, store, cfg, set(ordinary), types)
    assert eid
    return next(row for row in review(cfg) if row["id"] == eid)


@pytest.mark.parametrize("changes", [
    {"forgotten": True}, {"status": "superseded"}, {"status": "merged"},
    {"status": "invalidated"}, {"status": "challenged"}, {"status": "inactive"},
    {"status": "deleted"}, {"status": "archived"},
])
def test_ineligible_never_revived(isolated, monkeypatch, changes):
    store, cfg = isolated
    memory(store, **changes)
    candidates(monkeypatch, [("missing", 1), ("dormant", 0.99)])
    before = snapshot(store, "dormant")
    assert event(store, cfg)["outcome"] == "none"
    assert snapshot(store, "dormant") == before


def test_scope_and_duplicates(isolated, monkeypatch):
    store, cfg = isolated
    memory(store, memory_type="narrative")
    candidates(monkeypatch, [("dormant", 0.9)])
    assert event(store, cfg, types={"fact"})["outcome"] == "none"
    assert event(store, cfg, ordinary={"dormant"})["outcome"] == "none"


def test_relevance_gap_over_age_and_importance(isolated, monkeypatch):
    store, cfg = isolated
    memory(store, "old", days=2000)
    memory(store, "relevant", days=31)
    candidates(monkeypatch, [("old", 0.80), ("relevant", 0.86)])
    row = event(store, cfg)
    assert row["memory_id"] == "relevant"
    assert row["relevance"] == 0.86
    assert 0 <= row["bonus"] <= cfg.dormant_recall.max_bonus


def test_dormancy_changes_selection_among_close_relevant_matches(isolated, monkeypatch):
    store, cfg = isolated
    memory(store, "recent", days=31)
    memory(store, "old", days=200)
    candidates(monkeypatch, [("recent", 0.84), ("old", 0.82)])
    assert event(store, cfg)["memory_id"] == "old"


@pytest.mark.parametrize("relevance", [0.74, float("nan"), float("inf")])
def test_old_irrelevant_or_invalid_never_passes(isolated, monkeypatch, relevance):
    store, cfg = isolated
    memory(store, days=3000)
    candidates(monkeypatch, [("dormant", relevance)])
    assert event(store, cfg)["outcome"] == "none"


def test_semantic_only_connection_without_invented_bridge(isolated, monkeypatch):
    store, cfg = isolated
    store.save_memory(Memory(id="blocker", content="The travel assistant was shelved because inference required a remote GPU and connectivity.", created_at=NOW-180*DAY))
    candidates(monkeypatch, [("blocker", 0.88)])
    eid = evaluate_shadow("Can we run a compact model offline now?", store, cfg, set(), TYPES)
    row = review(cfg)[0]
    assert row["memory_id"] == "blocker" and row["overlap_count"] == 0
    detail = inspect_event(cfg, eid)
    assert "embedding-only" in detail["connection"]
    assert "requiring review" in detail["connection"]
    assert "solved" not in detail["connection"]
    # The extra dense query must be the original, not ordinary query expansion.
    observed = []
    monkeypatch.setattr(retrieval, "_dense_search", lambda q, *a: observed.append(q) or [])
    evaluate_shadow("memory", store, cfg, set(), TYPES)
    assert observed == ["memory"]


def test_never_accessed_uses_creation_and_recent_access_blocks(isolated, monkeypatch):
    store, cfg = isolated
    memory(store, "new", days=1)
    memory(store, "old", days=90)
    store.record_access("old")
    candidates(monkeypatch, [("new", 0.95), ("old", 0.9)])
    assert event(store, cfg)["outcome"] == "none"


def test_shadow_inspect_feedback_are_separate_and_persist(isolated, monkeypatch):
    store, cfg = isolated
    memory(store)
    candidates(monkeypatch, [("dormant", 0.9)])
    before = snapshot(store, "dormant")
    row = event(store, cfg)
    assert row["shown_at"] is None and row["feedback"] is None
    with pytest.raises(ValueError, match="Inspect"):
        feedback(cfg, row["id"], "useful")
    shown = inspect_event(cfg, row["id"])
    assert shown["content"] == store.get_memory("dormant").content
    assert "3 distinct" in shown["connection"]
    feedback(cfg, row["id"], "useful")
    feedback(cfg, row["id"], "useful")  # idempotent retry
    with pytest.raises(ValueError, match="already"):
        feedback(cfg, row["id"], "dismissed")
    assert snapshot(store, "dormant") == before
    assert store.conn.execute("SELECT COUNT(*) AS n FROM access_log").fetchone()["n"] == 0
    assert event(store, cfg)["outcome"] == "none"
    # Every operation above uses a fresh connection; this proves durable state.
    state = store.conn.execute("SELECT * FROM dormant_recall_state").fetchone()
    assert state["used_at"] == NOW
    assert state["shown_at"] == NOW


@pytest.mark.parametrize("category", ["irrelevant", "dismissed"])
def test_negative_feedback_cooldown_without_use(isolated, monkeypatch, category):
    store, cfg = isolated
    memory(store)
    candidates(monkeypatch, [("dormant", 0.9)])
    row = event(store, cfg)
    inspect_event(cfg, row["id"])
    feedback(cfg, row["id"], category)
    state = store.conn.execute("SELECT * FROM dormant_recall_state").fetchone()
    assert state["used_at"] is None
    monkeypatch.setattr("engram.dormant.time.time", lambda: NOW + 8 * DAY)
    assert event(store, cfg)["outcome"] == "none"
    monkeypatch.setattr("engram.dormant.time.time", lambda: NOW + 31 * DAY)
    assert event(store, cfg)["outcome"] == "candidate"


def test_silence_and_rotation_do_not_reset_cooldown(isolated, monkeypatch):
    store, cfg = isolated
    memory(store)
    candidates(monkeypatch, [("dormant", 0.9)])
    cfg.dormant_recall.log_max_events = 1
    row = event(store, cfg)
    assert event(store, cfg)["outcome"] == "none"
    assert len(review(cfg)) == 1
    state = store.conn.execute("SELECT * FROM dormant_recall_state").fetchone()
    assert state["used_at"] is None and state["shown_at"] is None
    monkeypatch.setattr("engram.dormant.time.time", lambda: NOW + 8 * DAY)
    assert event(store, cfg)["outcome"] == "candidate"


def test_review_does_not_expose_forgotten_or_inactive_content(isolated, monkeypatch):
    store, cfg = isolated
    memory(store)
    candidates(monkeypatch, [("dormant", 0.9)])
    row = event(store, cfg)
    store.forget_memory("dormant")
    with pytest.raises(ValueError):
        inspect_event(cfg, row["id"])
    assert review(cfg) == []


def test_shadow_logs_bounded_no_text_and_expire(isolated, monkeypatch):
    store, cfg = isolated
    memory(store)
    candidates(monkeypatch, [("dormant", 0.9)])
    cfg.dormant_recall.log_max_events = 3
    for _ in range(5):
        evaluate_shadow(QUERY, store, cfg, set(), TYPES)
    rows = review(cfg)
    assert len(rows) == 3
    assert "postgres" not in str(rows) and "notes" not in str(rows)
    monkeypatch.setattr("engram.dormant.time.time", lambda: NOW + 31 * DAY)
    assert review(cfg) == []


def test_schema_addition_preserves_memory_rows(isolated, monkeypatch):
    store, cfg = isolated
    memory(store, metadata={"type": "factual", "source_trust": 0.2})
    before = dict(store.conn.execute("SELECT * FROM memories").fetchone())
    candidates(monkeypatch, [("dormant", 0.9)])
    event(store, cfg)
    review(cfg)
    assert dict(store.conn.execute("SELECT * FROM memories").fetchone()) == before


def test_ordinary_results_unchanged_independent_candidates_and_cache(isolated, monkeypatch):
    store, cfg = isolated
    for i in range(4):
        memory(store, f"normal{i}", days=1)
    memory(store)
    hits = [(f"normal{i}", 0.99-i/100) for i in range(4)] + [("dormant", 0.9)]
    calls = []
    def dense(q, s, c, limit):
        calls.append(limit)
        return hits[:limit]
    monkeypatch.setattr(retrieval, "_dense_search", dense)
    monkeypatch.setattr(retrieval, "_bm25_search", lambda *a: [])
    monkeypatch.setattr(retrieval, "_graph_search", lambda *a: [])
    monkeypatch.setattr(retrieval, "_hopfield_search", lambda *a: [])
    monkeypatch.setattr(retrieval, "RETRIEVAL_NOISE_SCALE", 0)
    before = snapshot(store, "dormant")
    cfg.dormant_recall.mode = "off"
    baseline = retrieval.search(QUERY, store, cfg, top_k=1, rerank=False)
    cfg.dormant_recall.mode = "shadow"
    results = retrieval.search(QUERY, store, cfg, top_k=1, rerank=False)
    assert [(r.memory.id, r.score, r.sources) for r in results] == [(r.memory.id, r.score, r.sources) for r in baseline]
    assert calls == [3, 50]  # ordinary cache hit still does independent search
    assert review(cfg)[0]["memory_id"] == "dormant"
    assert snapshot(store, "dormant") == before
    assert store.get_memory(results[0].memory.id).access_count == 2


def test_disabled_no_schema_or_search(isolated, monkeypatch):
    store, cfg = isolated
    cfg.dormant_recall.mode = "off"
    monkeypatch.setattr(retrieval, "_dense_search", lambda *a: pytest.fail("called when off"))
    assert evaluate_shadow(QUERY, store, cfg, set(), TYPES) is None
    if cfg.storage_backend == "sqlite":
        assert not store.conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'dormant_%'").fetchall()


def test_failure_does_not_poison_normal_store(isolated, monkeypatch, caplog):
    store, cfg = isolated
    memory(store)
    candidates(monkeypatch, [("dormant", 0.9)])
    def fail(db, *a):
        db.conn.execute("SELECT SECRET_QUERY_AND_DSN FROM nonexistent_shadow_table")
    monkeypatch.setattr("engram.dormant._prune", fail)
    assert evaluate_shadow(QUERY, store, cfg, set(), TYPES) is None
    assert "SECRET" not in caplog.text
    store.record_access("dormant")
    assert store.get_memory("dormant").access_count == 1


def test_concurrent_evaluations_only_one_candidate(isolated, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    store, cfg = isolated
    memory(store)
    candidates(monkeypatch, [("dormant", 0.9)])
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(lambda _: evaluate_shadow(QUERY, store, cfg, set(), TYPES), range(2)))
    rows = review(cfg)
    assert all(ids) and len(rows) == 2
    assert sum(row["outcome"] == "candidate" for row in rows) == 1


def test_mcp_review_inspect_and_feedback_handlers(isolated, monkeypatch):
    from engram.mcp_server import MCPServer, TOOLS
    store, cfg = isolated
    memory(store)
    candidates(monkeypatch, [("dormant", 0.9)])
    row = event(store, cfg)
    before = snapshot(store, "dormant")
    server = MCPServer.__new__(MCPServer)
    server.config = cfg
    server.store = store
    assert {"dormant_review", "dormant_inspect", "dormant_feedback"} <= {t["name"] for t in TOOLS}
    assert server._call_tool("dormant_review", {})[0]["id"] == row["id"]
    assert server._call_tool("dormant_inspect", {"event_id": row["id"]})["memory_id"] == "dormant"
    assert server._call_tool("dormant_feedback", {"event_id": row["id"], "category": "dismissed"})["feedback"] == "dismissed"
    assert snapshot(store, "dormant") == before


def test_real_dense_search_not_ordinary_candidate_pool(isolated, monkeypatch):
    store, cfg = isolated
    # Use real cosine implementation with synthetic vectors, no model/API calls.
    memory(store)
    mem = store.get_memory("dormant")
    mem.embedding = np.array([1, 0, 0], dtype=np.float32)
    store.save_memory(mem)
    monkeypatch.setattr(retrieval, "embed_query", lambda *a: np.array([1, 0, 0], dtype=np.float32))
    assert event(store, cfg)["memory_id"] == "dormant"


def test_config_switch_and_bounds(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text('dormant_recall:\n  mode: shadow\n  candidate_limit: 12\n')
    assert Config.load(path).dormant_recall.candidate_limit == 12
    monkeypatch.setenv("ENGRAM_DORMANT_RECALL_MODE", "off")
    assert Config.load(path).dormant_recall.mode == "off"
    assert DormantRecallConfig().mode == "off"
    for field, value in [("mode", "visible"), ("max_bonus", 1), ("min_relevance", 0),
                         ("dormancy_days", float("nan")), ("candidate_limit", 2.5),
                         ("log_max_events", 0), ("cooldown_days", True)]:
        with pytest.raises(ValueError):
            replace(DormantRecallConfig(), **{field: value}).validate()
