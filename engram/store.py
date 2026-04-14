"""SQLite storage with FTS5, entity graph, and memory lifecycle."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from engram.config import Config


class MemoryLayer(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    CODEBASE = "codebase"


class MemoryType(str, Enum):
    """Semantic type of the memory content — orthogonal to layer."""
    FACT = "fact"              # structured knowledge, statuses, states
    PROCEDURE = "procedure"    # how-to, playbooks, rules
    NARRATIVE = "narrative"    # session logs, stories, raw context


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory."""
    ACTIVE = "active"
    CHALLENGED = "challenged"
    INVALIDATED = "invalidated"
    MERGED = "merged"
    SUPERSEDED = "superseded"


class SourceType(str, Enum):
    INGEST = "ingest"
    HUMAN = "remember:human"
    AI = "remember:ai"
    DREAM = "dream"
    INTERACTION = "interaction"
    CODE_SCAN = "code_scan"


@dataclass
class Memory:
    id: str
    content: str
    source_file: str | None = None
    source_type: str = SourceType.INGEST
    layer: str = MemoryLayer.EPISODIC
    memory_type: str = MemoryType.NARRATIVE
    status: str = MemoryStatus.ACTIVE
    embedding: np.ndarray | None = None
    importance: float = 0.5
    access_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    fact_date: str | None = None
    fact_date_end: str | None = None
    emotional_valence: float = 0.0
    chunk_hash: str = ""
    metadata: dict = field(default_factory=dict)
    forgotten: bool = False

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_accessed:
            self.last_accessed = self.created_at
        if not self.chunk_hash:
            src = self.source_file or ""
            self.chunk_hash = hashlib.sha256(f"{src}:{self.content}".encode()).hexdigest()


@dataclass
class Entity:
    id: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    entity_type: str = "concept"
    first_seen: float = 0.0
    last_seen: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class Relationship:
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    strength: float = 1.0
    evidence_count: int = 1
    created_at: float = 0.0
    last_seen: float = 0.0


def _pack_embedding(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _unpack_embedding(blob: bytes, dim: int = 384) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_file TEXT,
    source_type TEXT NOT NULL DEFAULT 'ingest',
    layer TEXT NOT NULL DEFAULT 'episodic',
    memory_type TEXT NOT NULL DEFAULT 'narrative',
    status TEXT NOT NULL DEFAULT 'active',
    embedding BLOB,
    importance REAL NOT NULL DEFAULT 0.5,
    access_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL,
    fact_date TEXT,
    fact_date_end TEXT,
    emotional_valence REAL DEFAULT 0.0,
    chunk_hash TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    forgotten INTEGER DEFAULT 0,
    previous_memory_id TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_layer ON memories(layer);
CREATE INDEX IF NOT EXISTS idx_memories_chunk_hash ON memories(chunk_hash);
CREATE INDEX IF NOT EXISTS idx_memories_fact_date ON memories(fact_date);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_forgotten ON memories(forgotten);
CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    hypothetical_queries,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS hypothetical_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hq_memory ON hypothetical_queries(memory_id);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    aliases TEXT DEFAULT '[]',
    entity_type TEXT DEFAULT 'concept',
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(canonical_name);

CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, memory_id)
);

CREATE TABLE IF NOT EXISTS relationships (
    source_entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    strength REAL DEFAULT 1.0,
    evidence_count INTEGER DEFAULT 1,
    created_at REAL NOT NULL,
    last_seen REAL NOT NULL,
    valid_from REAL DEFAULT NULL,
    valid_to REAL DEFAULT NULL,
    embedding BLOB DEFAULT NULL,
    PRIMARY KEY (source_entity_id, target_entity_id, relation_type)
);

CREATE TABLE IF NOT EXISTS access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    accessed_at REAL NOT NULL,
    query_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_access_log_memory ON access_log(memory_id);

CREATE TABLE IF NOT EXISTS ingest_log (
    file_path TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    last_ingested REAL NOT NULL,
    memory_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    memory_id TEXT,
    entity_id TEXT,
    detail TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at DESC);

CREATE TABLE IF NOT EXISTS diary_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    session_id TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_diary_created ON diary_entries(created_at DESC);

CREATE TABLE IF NOT EXISTS importance_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    score REAL NOT NULL,
    recorded_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_imp_hist_memory ON importance_history(memory_id);

CREATE TABLE IF NOT EXISTS status_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    old_status TEXT NOT NULL,
    new_status TEXT NOT NULL,
    reason TEXT,
    changed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_status_hist_memory ON status_history(memory_id);
"""


class Store:
    def __init__(self, config: Config):
        self.config = config
        self.db_path = config.resolved_db_path
        self._conn: sqlite3.Connection | None = None
        self._embedding_cache: tuple[list[str], np.ndarray] | None = None
        self.ann_index = None  # initialized in init_ann_index()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def init_db(self):
        for stmt in SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    self.conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # column/index may not exist yet — migration will fix
        self.conn.commit()
        # migrations — add new columns if missing (safe for existing DBs)
        self._migrate()
        # retry any indexes that failed pre-migration
        for stmt in SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt and "CREATE INDEX" in stmt.upper():
                try:
                    self.conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass
        self.conn.commit()

    def _migrate(self):
        """Add new columns to existing databases without breaking them."""
        migrations = [
            ("memories", "previous_memory_id", "TEXT DEFAULT NULL"),
            ("relationships", "embedding", "BLOB DEFAULT NULL"),
            ("relationships", "valid_from", "REAL DEFAULT NULL"),
            ("relationships", "valid_to", "REAL DEFAULT NULL"),
            ("memories", "memory_type", "TEXT DEFAULT 'narrative'"),
            ("memories", "status", "TEXT DEFAULT 'active'"),
        ]
        for table, column, coltype in migrations:
            try:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
                self.conn.commit()
            except Exception:
                pass  # column already exists

        # create status_history table if missing (for pre-0.3.0 DBs)
        try:
            self.conn.execute("""CREATE TABLE IF NOT EXISTS status_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                old_status TEXT NOT NULL, new_status TEXT NOT NULL,
                reason TEXT, changed_at REAL NOT NULL)""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_status_hist_memory ON status_history(memory_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)")
            self.conn.commit()
        except Exception:
            pass

        # backfill memory_type from metadata for existing memories
        self._backfill_memory_type()

    def _backfill_memory_type(self):
        """Backfill memory_type from metadata['type'] for pre-0.3.0 memories."""
        try:
            rows = self.conn.execute(
                "SELECT id, metadata, layer FROM memories WHERE memory_type = 'narrative' OR memory_type IS NULL"
            ).fetchall()
            type_map = {"factual": "fact", "procedural": "procedure", "experiential": "narrative"}
            updated = 0
            for row in rows:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
                raw_type = meta.get("type", "")
                # infer from metadata or layer
                if raw_type in type_map:
                    mtype = type_map[raw_type]
                elif row["layer"] == "procedural":
                    mtype = "procedure"
                elif row["layer"] == "semantic":
                    mtype = "fact"
                else:
                    continue  # leave as narrative
                self.conn.execute("UPDATE memories SET memory_type = ? WHERE id = ?", (mtype, row["id"]))
                updated += 1
            if updated:
                self.conn.commit()
        except Exception:
            pass

    def update_status(self, memory_id: str, new_status: str, reason: str | None = None):
        """Transition a memory's status with audit trail."""
        row = self.conn.execute("SELECT status FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return
        old_status = row["status"] or "active"
        self.conn.execute("UPDATE memories SET status = ? WHERE id = ?", (new_status, memory_id))
        self.conn.execute(
            "INSERT INTO status_history (memory_id, old_status, new_status, reason, changed_at) VALUES (?, ?, ?, ?, ?)",
            (memory_id, old_status, new_status, reason, time.time()),
        )
        self._emit_event("status_change", memory_id=memory_id, detail=f"{old_status} → {new_status}: {reason}")
        self.conn.commit()

    def get_status_history(self, memory_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM status_history WHERE memory_id = ? ORDER BY changed_at ASC",
            (memory_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_memories_by_type(self, memory_type: str, limit: int = 50) -> list['Memory']:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE memory_type = ? AND forgotten = 0 AND status = 'active' ORDER BY importance DESC LIMIT ?",
            (memory_type, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def init_ann_index(self, background: bool = True):
        """Initialize the ANN index — load from disk or rebuild."""
        if not self.config.ann.enabled:
            return
        try:
            from engram.ann_index import ANNIndex, HAS_HNSWLIB
            if not HAS_HNSWLIB:
                return
        except ImportError:
            return

        ann = ANNIndex(
            dim=self.config.embedding_dim,
            m=self.config.ann.m,
            ef_construction=self.config.ann.ef_construction,
            ef_search=self.config.ann.ef_search,
            max_elements=self.config.ann.max_elements,
            index_path=str(self.config.ann.resolved_index_path),
        )

        if ann.load():
            self.ann_index = ann
        elif background:
            import threading
            self.ann_index = ann  # set now so it's available (not ready yet)
            t = threading.Thread(target=self._rebuild_ann_blocking, daemon=True)
            t.start()
        else:
            self.ann_index = ann
            self._rebuild_ann_blocking()

    def _rebuild_ann_blocking(self):
        """Rebuild ANN index from all embeddings in DB."""
        if self.ann_index is None:
            return
        ids, vecs = self.get_all_embeddings()
        if len(ids) == 0:
            self.ann_index._ready = True
            return
        self.ann_index.build(ids, vecs)
        self.ann_index.save()

    def rebuild_ann_index(self):
        """Public API: full synchronous rebuild + save."""
        self._rebuild_ann_blocking()

    def close(self):
        if self.ann_index and self.ann_index.ready:
            self.ann_index.save()
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- Memory CRUD ---

    def save_memory(self, mem: Memory, hypothetical_queries: list[str] | None = None):
        emb_blob = _pack_embedding(mem.embedding) if mem.embedding is not None else None

        # temporal backbone: link to the most recent memory
        prev_id = None
        try:
            prev_row = self.conn.execute(
                "SELECT id FROM memories WHERE forgotten=0 AND id != ? ORDER BY created_at DESC LIMIT 1",
                (mem.id,),
            ).fetchone()
            if prev_row:
                prev_id = prev_row["id"]
        except Exception:
            pass

        self.conn.execute(
            """INSERT OR REPLACE INTO memories
            (id, content, source_file, source_type, layer, memory_type, status,
             embedding, importance, access_count, created_at, last_accessed,
             fact_date, fact_date_end, emotional_valence, chunk_hash, metadata,
             forgotten, previous_memory_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (mem.id, mem.content, mem.source_file, mem.source_type, mem.layer,
             mem.memory_type, mem.status,
             emb_blob, mem.importance, mem.access_count, mem.created_at,
             mem.last_accessed, mem.fact_date, mem.fact_date_end,
             mem.emotional_valence, mem.chunk_hash, json.dumps(mem.metadata),
             int(mem.forgotten), prev_id),
        )

        # get the rowid for FTS
        row = self.conn.execute("SELECT rowid FROM memories WHERE id = ?", (mem.id,)).fetchone()
        rowid = row[0]

        # clear old FTS + hypothetical queries
        self.conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (rowid,))
        self.conn.execute("DELETE FROM hypothetical_queries WHERE memory_id = ?", (mem.id,))

        hq_text = ""
        if hypothetical_queries:
            hq_text = " ".join(hypothetical_queries)
            for q in hypothetical_queries:
                self.conn.execute(
                    "INSERT INTO hypothetical_queries (memory_id, query_text) VALUES (?, ?)",
                    (mem.id, q),
                )

        self.conn.execute(
            "INSERT INTO memories_fts (rowid, content, hypothetical_queries) VALUES (?, ?, ?)",
            (rowid, mem.content, hq_text),
        )

        self._embedding_cache = None  # invalidate cache
        if self.ann_index and self.ann_index.ready and mem.embedding is not None:
            self.ann_index.add(mem.id, mem.embedding)
        self._emit_event("memory_write", memory_id=mem.id,
                         detail=f"layer={mem.layer} source={mem.source_type}")
        self.conn.commit()

    def get_memory(self, memory_id: str) -> Memory | None:
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return None
        return self._row_to_memory(row)

    def invalidate_embedding_cache(self):
        self._embedding_cache = None

    def get_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        if self._embedding_cache is not None:
            return self._embedding_cache
        rows = self.conn.execute(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL AND forgotten = 0"
        ).fetchall()
        if not rows:
            return [], np.array([])
        ids = [r["id"] for r in rows]
        vecs = np.stack([_unpack_embedding(r["embedding"]) for r in rows])
        self._embedding_cache = (ids, vecs)
        return ids, vecs

    def search_fts(self, query: str, limit: int = 30) -> list[tuple[str, float]]:
        words = []
        for w in query.split():
            cleaned = "".join(c for c in w if c.isalnum())
            if len(cleaned) >= 2:
                words.append(f'"{cleaned}"')
        if not words:
            return []
        fts_query = " OR ".join(words)
        rows = self.conn.execute(
            """SELECT m.id, bm25(memories_fts) as score
            FROM memories_fts
            JOIN memories m ON m.rowid = memories_fts.rowid
            WHERE memories_fts MATCH ? AND m.forgotten = 0
            ORDER BY score
            LIMIT ?""",
            (fts_query, limit),
        ).fetchall()
        return [(r["id"], r["score"]) for r in rows]

    def record_access(self, memory_id: str, query: str | None = None):
        now = time.time()
        self.conn.execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, memory_id),
        )
        self.conn.execute(
            "INSERT INTO access_log (memory_id, accessed_at, query_text) VALUES (?, ?, ?)",
            (memory_id, now, query),
        )
        self._emit_event("memory_read", memory_id=memory_id, detail=query)
        self.conn.commit()

    def record_search(self, memory_ids: list[str], query: str | None = None):
        """Record a search — one event, updates all returned memories."""
        now = time.time()
        for mid in memory_ids:
            self.conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (now, mid),
            )
            self.conn.execute(
                "INSERT INTO access_log (memory_id, accessed_at, query_text) VALUES (?, ?, ?)",
                (mid, now, query),
            )
        self._emit_event("recall", detail=f"{query} ({len(memory_ids)} results)")
        self.conn.commit()

    def forget_memory(self, memory_id: str):
        self.conn.execute("UPDATE memories SET forgotten = 1 WHERE id = ?", (memory_id,))
        if self.ann_index and self.ann_index.ready:
            self.ann_index.remove(memory_id)
        self._emit_event("memory_forget", memory_id=memory_id)
        self.conn.commit()

    def update_layer(self, memory_id: str, new_layer: str):
        self.conn.execute("UPDATE memories SET layer = ? WHERE id = ?", (new_layer, memory_id))
        self._emit_event("memory_promote", memory_id=memory_id, detail=f"to={new_layer}")
        self.conn.commit()

    def update_importance(self, memory_id: str, importance: float):
        self.conn.execute("UPDATE memories SET importance = ? WHERE id = ?", (importance, memory_id))
        self.conn.commit()

    def get_memories_by_layer(self, layer: str, limit: int = 100) -> list[Memory]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE layer = ? AND forgotten = 0 ORDER BY importance DESC LIMIT ?",
            (layer, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memories_by_date_range(self, start: str, end: str | None = None, limit: int = 50) -> list[Memory]:
        if end:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE fact_date >= ? AND fact_date <= ? AND forgotten = 0 ORDER BY fact_date LIMIT ?",
                (start, end, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE fact_date LIKE ? AND forgotten = 0 ORDER BY fact_date LIMIT ?",
                (f"{start}%", limit),
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_recent_memories(self, limit: int = 20) -> list[Memory]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE forgotten = 0 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def count_memories(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT layer, COUNT(*) as cnt FROM memories WHERE forgotten = 0 GROUP BY layer"
        ).fetchall()
        counts = {r["layer"]: r["cnt"] for r in rows}
        total = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE forgotten = 0"
        ).fetchone()
        counts["total"] = total["cnt"]
        return counts

    # --- Entity CRUD ---

    def save_entity(self, entity: Entity):
        self.conn.execute(
            """INSERT OR REPLACE INTO entities
            (id, canonical_name, aliases, entity_type, first_seen, last_seen, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (entity.id, entity.canonical_name, json.dumps(entity.aliases),
             entity.entity_type, entity.first_seen, entity.last_seen,
             json.dumps(entity.metadata)),
        )
        self.conn.commit()

    def get_entity(self, entity_id: str) -> Entity | None:
        row = self.conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        if not row:
            return None
        return self._row_to_entity(row)

    def find_entity_by_name(self, name: str) -> Entity | None:
        name_lower = name.lower()
        row = self.conn.execute(
            "SELECT * FROM entities WHERE LOWER(canonical_name) = ?", (name_lower,)
        ).fetchone()
        if row:
            return self._row_to_entity(row)
        # search aliases
        rows = self.conn.execute("SELECT * FROM entities").fetchall()
        for r in rows:
            aliases = json.loads(r["aliases"])
            if any(a.lower() == name_lower for a in aliases):
                return self._row_to_entity(r)
        return None

    def link_entity_memory(self, entity_id: str, memory_id: str):
        self.conn.execute(
            "INSERT OR IGNORE INTO entity_mentions (entity_id, memory_id) VALUES (?, ?)",
            (entity_id, memory_id),
        )
        self.conn.commit()

    def get_entity_memories(self, entity_id: str, limit: int = 50) -> list[Memory]:
        rows = self.conn.execute(
            """SELECT m.* FROM memories m
            JOIN entity_mentions em ON em.memory_id = m.id
            WHERE em.entity_id = ? AND m.forgotten = 0
            ORDER BY m.created_at DESC LIMIT ?""",
            (entity_id, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def save_relationship(self, rel: Relationship):
        self.conn.execute(
            """INSERT INTO relationships
            (source_entity_id, target_entity_id, relation_type, strength,
             evidence_count, created_at, last_seen, valid_from)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_entity_id, target_entity_id, relation_type)
            DO UPDATE SET
                strength = MAX(strength, excluded.strength),
                evidence_count = evidence_count + 1,
                last_seen = excluded.last_seen""",
            (rel.source_entity_id, rel.target_entity_id, rel.relation_type,
             rel.strength, rel.evidence_count, rel.created_at, rel.last_seen,
             getattr(rel, 'valid_from', rel.created_at)),
        )
        self.conn.commit()

    def invalidate_relationship(self, source_id: str, target_id: str,
                                 relation_type: str, valid_to: float | None = None):
        """Mark a relationship as no longer valid (Graphiti temporal invalidation).

        Sets valid_to timestamp instead of deleting — preserves historical record.
        """
        import time as _time
        vt = valid_to or _time.time()
        self.conn.execute(
            """UPDATE relationships SET valid_to = ?
               WHERE source_entity_id = ? AND target_entity_id = ? AND relation_type = ?
               AND valid_to IS NULL""",
            (vt, source_id, target_id, relation_type),
        )
        self.conn.commit()

    def get_related_entities(self, entity_id: str, max_hops: int = 2) -> list[dict]:
        rows = self.conn.execute(
            """WITH RECURSIVE traversal(eid, depth, path) AS (
                SELECT ?, 0, ?
                UNION ALL
                SELECT
                    CASE WHEN r.source_entity_id = t.eid
                         THEN r.target_entity_id
                         ELSE r.source_entity_id END,
                    t.depth + 1,
                    t.path || ',' || CASE WHEN r.source_entity_id = t.eid
                         THEN r.target_entity_id
                         ELSE r.source_entity_id END
                FROM traversal t
                JOIN relationships r ON (r.source_entity_id = t.eid OR r.target_entity_id = t.eid)
                WHERE t.depth < ?
                  AND INSTR(t.path, CASE WHEN r.source_entity_id = t.eid
                         THEN r.target_entity_id
                         ELSE r.source_entity_id END) = 0
            )
            SELECT DISTINCT t.eid, t.depth, e.canonical_name, e.entity_type
            FROM traversal t
            JOIN entities e ON e.id = t.eid
            WHERE t.eid != ?
            ORDER BY t.depth""",
            (entity_id, entity_id, max_hops, entity_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_entity_relationships(self, entity_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT r.*, e1.canonical_name as source_name, e2.canonical_name as target_name
            FROM relationships r
            JOIN entities e1 ON e1.id = r.source_entity_id
            JOIN entities e2 ON e2.id = r.target_entity_id
            WHERE r.source_entity_id = ? OR r.target_entity_id = ?
            ORDER BY r.strength DESC""",
            (entity_id, entity_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_entities(self, limit: int = 100) -> list[Entity]:
        rows = self.conn.execute(
            "SELECT * FROM entities ORDER BY last_seen DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    # --- Ingest log ---

    def get_file_hash(self, file_path: str) -> str | None:
        row = self.conn.execute(
            "SELECT file_hash FROM ingest_log WHERE file_path = ?", (file_path,)
        ).fetchone()
        return row["file_hash"] if row else None

    def set_file_hash(self, file_path: str, file_hash: str, memory_count: int = 0):
        self.conn.execute(
            """INSERT OR REPLACE INTO ingest_log (file_path, file_hash, last_ingested, memory_count)
            VALUES (?, ?, ?, ?)""",
            (file_path, file_hash, time.time(), memory_count),
        )
        self.conn.commit()

    # --- Events ---

    def _emit_event(self, event_type: str, memory_id: str | None = None,
                    entity_id: str | None = None, detail: str | None = None):
        self.conn.execute(
            "INSERT INTO events (event_type, memory_id, entity_id, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (event_type, memory_id, entity_id, detail, time.time()),
        )

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Diary ---

    def write_diary(self, text: str, session_id: str | None = None):
        self.conn.execute(
            "INSERT INTO diary_entries (text, session_id, created_at) VALUES (?, ?, ?)",
            (text, session_id, time.time()),
        )
        self.conn.commit()

    def get_diary(self, limit: int = 50, session_id: str | None = None) -> list[dict]:
        if session_id:
            rows = self.conn.execute(
                "SELECT * FROM diary_entries WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM diary_entries ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --- Importance history ---

    def record_importance(self, memory_id: str, score: float):
        self.conn.execute(
            "INSERT INTO importance_history (memory_id, score, recorded_at) VALUES (?, ?, ?)",
            (memory_id, round(score, 4), time.time()),
        )

    def get_importance_history(self, memory_id: str, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT score, recorded_at FROM importance_history WHERE memory_id = ? ORDER BY recorded_at ASC LIMIT ?",
            (memory_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        counts = self.count_memories()
        entity_count = self.conn.execute("SELECT COUNT(*) as cnt FROM entities").fetchone()["cnt"]
        rel_count = self.conn.execute("SELECT COUNT(*) as cnt FROM relationships").fetchone()["cnt"]
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "memories": counts,
            "entities": entity_count,
            "relationships": rel_count,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
        }

    # --- Helpers ---

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        emb = _unpack_embedding(row["embedding"]) if row["embedding"] else None
        # safe access for new columns (may not exist in old DBs mid-migration)
        keys = row.keys() if hasattr(row, "keys") else []
        return Memory(
            id=row["id"],
            content=row["content"],
            source_file=row["source_file"],
            source_type=row["source_type"],
            layer=row["layer"],
            memory_type=row["memory_type"] if "memory_type" in keys else MemoryType.NARRATIVE,
            status=row["status"] if "status" in keys else MemoryStatus.ACTIVE,
            embedding=emb,
            importance=row["importance"],
            access_count=row["access_count"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            fact_date=row["fact_date"],
            fact_date_end=row["fact_date_end"],
            emotional_valence=row["emotional_valence"],
            chunk_hash=row["chunk_hash"],
            metadata=json.loads(row["metadata"]),
            forgotten=bool(row["forgotten"]),
        )

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            canonical_name=row["canonical_name"],
            aliases=json.loads(row["aliases"]),
            entity_type=row["entity_type"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            metadata=json.loads(row["metadata"]),
        )
