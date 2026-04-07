"""Conversation session ingestion — extract from Claude Code JSONL logs.

Finds Claude Code conversation files, parses them into exchange pairs,
extracts key moments (decisions, corrections, task completions), and
stores them as structured memories.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from collections import defaultdict

from engram.store import Store, Memory, MemoryLayer, SourceType
from engram.embeddings import embed_documents
from engram.entities import process_entities_for_memory


def find_claude_sessions(base: str | None = None) -> list[Path]:
    """Find all Claude Code conversation JSONL files."""
    if base is None:
        base = Path.home() / ".claude" / "projects"
    else:
        base = Path(base)

    sessions = []
    for jsonl in base.rglob("*.jsonl"):
        if jsonl.stat().st_size > 100:  # skip empty
            sessions.append(jsonl)

    return sorted(sessions, key=lambda p: p.stat().st_mtime, reverse=True)


def ingest_session(path: Path, store: Store, summarize: bool = True) -> dict:
    """Ingest a single Claude Code conversation session."""
    stats = {"exchanges": 0, "memories_created": 0, "decisions": 0, "corrections": 0}

    exchanges = _parse_jsonl(path)
    if not exchanges:
        return stats

    # group into Q+A pairs
    pairs = _pair_exchanges(exchanges)
    stats["exchanges"] = len(pairs)

    for pair in pairs:
        content = pair["content"]
        if len(content.strip()) < 30:
            continue

        # detect memory type
        mem_type = _classify_exchange(content)

        layer = MemoryLayer.EPISODIC
        importance = 0.4
        source_type = SourceType.INTERACTION

        if mem_type == "decision":
            layer = MemoryLayer.PROCEDURAL
            importance = 0.7
            stats["decisions"] += 1
        elif mem_type == "correction":
            layer = MemoryLayer.PROCEDURAL
            importance = 0.8
            stats["corrections"] += 1
        elif mem_type == "task_complete":
            importance = 0.6
        elif mem_type == "error":
            layer = MemoryLayer.PROCEDURAL
            importance = 0.7

        mem = Memory(
            id=str(uuid.uuid4()),
            content=content[:3000],
            source_file=str(path),
            source_type=source_type,
            layer=layer,
            importance=importance,
            fact_date=pair.get("date"),
            metadata={"type": mem_type, "session": path.stem},
        )

        emb = embed_documents([mem.content])
        if emb.size > 0:
            mem.embedding = emb[0]
        store.save_memory(mem)
        process_entities_for_memory(store, mem.id, mem.content)
        stats["memories_created"] += 1

    return stats


def _parse_jsonl(path: Path) -> list[dict]:
    """Parse Claude Code JSONL format."""
    exchanges = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                role = obj.get("role", "unknown")
                content = obj.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            text_parts.append(part.get("text", ""))
                    content = " ".join(text_parts)
                if content.strip():
                    exchanges.append({
                        "role": role,
                        "content": content.strip(),
                        "timestamp": obj.get("timestamp"),
                    })
            except json.JSONDecodeError:
                continue
    return exchanges


def _pair_exchanges(exchanges: list[dict]) -> list[dict]:
    """Group consecutive human/assistant exchanges."""
    pairs = []
    i = 0
    while i < len(exchanges):
        ex = exchanges[i]
        if ex["role"] in ("human", "user"):
            q = ex["content"]
            a = ""
            ts = ex.get("timestamp")
            if i + 1 < len(exchanges) and exchanges[i + 1]["role"] in ("assistant",):
                a = exchanges[i + 1]["content"]
                i += 2
            else:
                i += 1
            # truncate long assistant responses
            if len(a) > 2000:
                a = a[:2000] + "..."
            content = f"Q: {q}\nA: {a}" if a else q
            date = None
            if ts:
                try:
                    date = time.strftime("%Y-%m-%d", time.localtime(float(ts)))
                except (ValueError, TypeError):
                    pass
            pairs.append({"content": content, "date": date})
        else:
            i += 1
    return pairs


def _classify_exchange(content: str) -> str:
    """Classify an exchange into a memory type."""
    content_lower = content.lower()

    # decisions
    if any(w in content_lower for w in ["decided to", "let's go with", "i'll use",
                                         "the approach is", "decision:", "chose to"]):
        return "decision"

    # corrections
    if any(w in content_lower for w in ["no not that", "wrong", "don't do",
                                         "that's incorrect", "fix it", "you're going in a loop",
                                         "stop doing", "that's not what i"]):
        return "correction"

    # errors
    if any(w in content_lower for w in ["error:", "traceback", "exception",
                                         "failed:", "bug:", "crash"]):
        return "error"

    # task completion
    if any(w in content_lower for w in ["done", "finished", "completed",
                                         "shipped", "deployed", "merged"]):
        return "task_complete"

    return "conversation"


def ingest_all_sessions(store: Store, limit: int = 20) -> dict:
    """Ingest recent Claude Code sessions."""
    sessions = find_claude_sessions()
    total_stats = {"sessions": 0, "memories_created": 0, "exchanges": 0}

    for path in sessions[:limit]:
        # skip if already ingested
        existing_hash = store.get_file_hash(str(path))
        import hashlib
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if existing_hash == current_hash:
            continue

        stats = ingest_session(path, store)
        store.set_file_hash(str(path), current_hash, stats["memories_created"])
        total_stats["sessions"] += 1
        total_stats["memories_created"] += stats["memories_created"]
        total_stats["exchanges"] += stats["exchanges"]

    return total_stats
