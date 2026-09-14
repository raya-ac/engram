"""Opt-in dormant recall, with independent, non-reinforcing shadow telemetry.

Tables are additive and lazily created on a separate connection. We deliberately
do not run Store.init_db(): its legacy backfills can modify existing memories.
No full query, memory text, literal matching terms or query hashes are persisted.
"""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace

from engram.config import Config
from engram.store import Memory, Store

logger = logging.getLogger(__name__)
DAY = 86400
STOP_WORDS = frozenset("a an and are as at be been but by can could did do does for from had has have how i if in is it its me my of on or our please show that the their them then there these they this to was we were what when where which who why will with would you your".split())

# DOUBLE PRECISION keeps Unix timestamp precision on both SQLite and PostgreSQL.
# State is separate from bounded evaluation history so log rotation cannot defeat
# cooldown or forget explicit use. At most one state row per extant memory.
SCHEMA = """
CREATE TABLE IF NOT EXISTS dormant_recall_state (
    memory_id TEXT PRIMARY KEY,
    retrieved_at DOUBLE PRECISION NOT NULL,
    shown_at DOUBLE PRECISION,
    used_at DOUBLE PRECISION,
    cooldown_until DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS dormant_recall_events (
    id TEXT PRIMARY KEY,
    sequence INTEGER NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    memory_id TEXT,
    outcome TEXT NOT NULL,
    candidate_count INTEGER NOT NULL,
    relevance DOUBLE PRECISION,
    bonus DOUBLE PRECISION,
    dormant_days DOUBLE PRECISION,
    overlap_count INTEGER,
    query_term_count INTEGER NOT NULL,
    shown_at DOUBLE PRECISION,
    feedback TEXT,
    feedback_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_dormant_events_created
ON dormant_recall_events(created_at DESC, sequence DESC);
"""


def _eligible(mem: Memory | None, allowed_types=None) -> bool:
    # NULL is a legacy active status, matching ordinary retrieval.
    return bool(mem and not mem.forgotten and mem.status in ("active", None)
                and (allowed_types is None or mem.memory_type in allowed_types))


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", text.lower())
            if len(t) > 2 and t not in STOP_WORDS}


@contextmanager
def _transaction(config: Config):
    """Serialize cooldown decisions, without touching the caller's transaction."""
    config.dormant_recall.validate()
    if config.normalized_storage_backend == "postgres":
        from psycopg.conninfo import make_conninfo
        config = replace(config, postgres_dsn=make_conninfo(config.postgres_dsn, connect_timeout=2))
    db = Store(config)
    try:
        if config.normalized_storage_backend == "postgres":
            # Bounded lock/statement waits; no changes to live server settings.
            db.conn.execute("SET LOCAL lock_timeout = '250ms'")
            db.conn.execute("SET LOCAL statement_timeout = '1000ms'")
            db.conn.execute("SELECT pg_advisory_xact_lock(731927416)")
        else:
            db.conn.execute("PRAGMA busy_timeout=250")
            db.conn.execute("BEGIN IMMEDIATE")
        for statement in SCHEMA.split(";"):
            if statement.strip():
                db.conn.execute(statement)
        yield db
        db.conn.commit()
    finally:
        # Closing an uncommitted connection rolls back this feature alone.
        db.close()


def _prune(db: Store, now: float):
    cfg = db.config.dormant_recall
    db.conn.execute("DELETE FROM dormant_recall_events WHERE created_at < ?",
                    (now - cfg.log_retention_days * DAY,))
    db.conn.execute("""DELETE FROM dormant_recall_events WHERE id NOT IN
        (SELECT id FROM dormant_recall_events ORDER BY sequence DESC LIMIT ?)""",
                    (cfg.log_max_events,))
    # Remove both telemetry and state when a memory is deleted/forgotten/inactive.
    for table in ("dormant_recall_events", "dormant_recall_state"):
        db.conn.execute(f"""DELETE FROM {table} WHERE memory_id IS NOT NULL AND
            NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = {table}.memory_id
            AND m.forgotten = 0 AND (m.status = 'active' OR m.status IS NULL))""")


def evaluate_shadow(query: str, store: Store, config: Config, ordinary_ids: set[str],
                    allowed_types: set[str]) -> str | None:
    """Record one independent shadow evaluation; never change ordinary results.

    Returns only an evaluation id for diagnostics, never a suggestion to inject.
    All failures are contained, including schema/permission/embedding failures.
    """
    if config.dormant_recall.mode != "shadow":
        return None
    try:
        from engram.retrieval import _dense_search

        cfg = config.dormant_recall
        cfg.validate()
        terms = _terms(query)
        # Original query only: expansions cannot manufacture the connection.
        # Independent of ordinary top_k, RRF cutoff, recency/frequency and cache.
        candidates = _dense_search(query, store, config, cfg.candidate_limit) if terms else []
        now = time.time()
        with _transaction(config) as db:
            _prune(db, now)
            choices = []
            for memory_id, relevance in candidates[:cfg.candidate_limit]:
                if memory_id in ordinary_ids or not math.isfinite(relevance) or not cfg.min_relevance <= relevance <= 1.00001:
                    continue
                relevance = min(1.0, relevance)  # allow floating-point cosine roundoff
                mem = db.get_memory(memory_id)
                if not _eligible(mem, allowed_types):
                    continue
                state = db.conn.execute("SELECT * FROM dormant_recall_state WHERE memory_id = ?",
                                        (memory_id,)).fetchone()
                if state and state["cooldown_until"] > now:
                    continue
                # Legacy accesses are a conservative proxy, not proof of use.
                last_use = max(mem.created_at, mem.last_accessed,
                               (state["used_at"] or 0) if state else 0)
                days = max(0, (now - last_use) / DAY)
                if not math.isfinite(days) or days < cfg.dormancy_days:
                    continue
                overlap = terms & _terms(mem.content)
                importance = mem.importance
                if not math.isfinite(importance):
                    continue
                bonus = cfg.max_bonus * min(1, max(0, importance)) * min(1, days / (4 * cfg.dormancy_days))
                choices.append((relevance, bonus, memory_id, days, len(overlap)))

            # Only relevance-gated candidates reach this point. A bounded bonus
            # may reorder close matches, never admit an irrelevant candidate or
            # increase the stored raw relevance / claimed truth confidence.
            choices.sort(key=lambda c: (-(c[0] + c[1]), -c[0], c[2]))
            chosen = choices[0] if choices else None
            event_id = str(uuid.uuid4())
            sequence = db.conn.execute("SELECT COALESCE(MAX(sequence), 0) + 1 AS next FROM dormant_recall_events").fetchone()["next"]
            if chosen:
                relevance, bonus, memory_id, days, overlap_count = chosen
                db.conn.execute("""INSERT INTO dormant_recall_state
                    (memory_id, retrieved_at, cooldown_until) VALUES (?, ?, ?)
                    ON CONFLICT (memory_id) DO UPDATE SET
                    retrieved_at = excluded.retrieved_at, cooldown_until = excluded.cooldown_until""",
                    (memory_id, now, now + cfg.cooldown_days * DAY))
            else:
                relevance = bonus = memory_id = days = overlap_count = None
            db.conn.execute("""INSERT INTO dormant_recall_events
                (id, sequence, created_at, memory_id, outcome, candidate_count, relevance, bonus,
                 dormant_days, overlap_count, query_term_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, sequence, now, memory_id, "candidate" if chosen else "none",
                 min(len(candidates), cfg.candidate_limit), relevance, bonus, days,
                 overlap_count, len(terms)))
            _prune(db, now)
        return event_id
    except Exception as exc:
        # Exception text can contain DSNs, queries or memory content.
        logger.warning("Dormant recall skipped (%s)", type(exc).__name__)
        return None


def review(config: Config, limit: int = 20) -> list[dict]:
    """Metadata only. Listing a row is not exposure of its memory content."""
    with _transaction(config) as db:
        _prune(db, time.time())
        return [dict(row) for row in db.conn.execute(
            "SELECT * FROM dormant_recall_events ORDER BY sequence DESC LIMIT ?",
            (max(1, min(int(limit), 100)),)).fetchall()]


def inspect_event(config: Config, event_id: str) -> dict:
    """Explicit review exposure; fresh eligibility check, no access reinforcement."""
    with _transaction(config) as db:
        now = time.time()
        _prune(db, now)
        row = db.conn.execute("SELECT * FROM dormant_recall_events WHERE id = ?", (event_id,)).fetchone()
        if not row or not row["memory_id"]:
            raise ValueError("No retained candidate for that evaluation")
        mem = db.get_memory(row["memory_id"])
        if not _eligible(mem):
            raise ValueError("Candidate is no longer eligible")
        db.conn.execute("UPDATE dormant_recall_events SET shown_at = COALESCE(shown_at, ?) WHERE id = ?", (now, event_id))
        db.conn.execute("""UPDATE dormant_recall_state SET shown_at = ?,
            cooldown_until = CASE WHEN cooldown_until > ? THEN cooldown_until ELSE ? END
            WHERE memory_id = ?""", (now, now + config.dormant_recall.cooldown_days * DAY,
                                     now + config.dormant_recall.cooldown_days * DAY, mem.id))
        connection = f"Embedding similarity to the original query: raw cosine {row['relevance']:.3f}. "
        if row["overlap_count"]:
            connection += f"The query also shared {row['overlap_count']} distinct non-stopword terms with this memory."
        else:
            connection += "No literal term overlap; this is an embedding-only connection requiring review against the original task."
        return {"event_id": event_id, "memory_id": mem.id, "content": mem.content,
                "status": mem.status, "source_type": mem.source_type,
                "source_trust": mem.metadata.get("source_trust"),
                "connection": connection,
                "note": "Query and matching terms were not saved. Check against the original task. Similarity and dormancy are not evidence of truth."}


def feedback(config: Config, event_id: str, category: str) -> dict:
    """Explicit useful means meaningful use, not approval inferred from silence.

    One immutable category per inspected event (same-category retries are safe).
    Useful affects only separate use/cooldown state, never ordinary ranking.
    """
    if category not in {"useful", "irrelevant", "dismissed"}:
        raise ValueError("Feedback must be useful, irrelevant or dismissed")
    with _transaction(config) as db:
        now = time.time()
        _prune(db, now)
        row = db.conn.execute("SELECT * FROM dormant_recall_events WHERE id = ?", (event_id,)).fetchone()
        if not row or row["shown_at"] is None:
            raise ValueError("Inspect a retained candidate before giving feedback")
        if row["feedback"]:
            if row["feedback"] != category:
                raise ValueError("Feedback already recorded for this evaluation")
            return {"event_id": event_id, "feedback": category}
        mem = db.get_memory(row["memory_id"])
        if not _eligible(mem):
            raise ValueError("Candidate is no longer eligible")
        cfg = config.dormant_recall
        delay = max(cfg.cooldown_days, cfg.feedback_cooldown_days,
                    cfg.dormancy_days if category == "useful" else 0)
        until = now + delay * DAY
        db.conn.execute("UPDATE dormant_recall_events SET feedback = ?, feedback_at = ? WHERE id = ?",
                        (category, now, event_id))
        db.conn.execute("""UPDATE dormant_recall_state SET
            used_at = CASE WHEN ? = 'useful' THEN ? ELSE used_at END,
            cooldown_until = CASE WHEN cooldown_until > ? THEN cooldown_until ELSE ? END
            WHERE memory_id = ?""", (category, now, until, until, mem.id))
        return {"event_id": event_id, "feedback": category}
