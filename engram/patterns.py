"""Session pattern extraction — distill reusable procedural knowledge from work.

After a session, analyze what was done and extract patterns that would help
future sessions. Inspired by mex's GROW step: "after every task, if no pattern
exists for this task type, create one."

This module:
1. Reads recent session activity (diary, events, new memories)
2. Classifies what kinds of work happened
3. Checks if procedural memories already cover those patterns
4. Generates new procedural memories for uncovered patterns
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from engram.store import Store, Memory, MemoryLayer, SourceType
from engram.embeddings import embed_documents, cosine_similarity_search
from engram.config import Config


@dataclass
class ExtractedPattern:
    """A pattern extracted from session activity."""
    title: str
    content: str
    category: str       # workflow, gotcha, decision, integration, debug
    source_events: int  # how many events contributed
    novelty: float      # 0-1, how novel vs existing procedural memories
    should_store: bool   # whether novelty is high enough to store


# --- Pattern categories ---

CATEGORY_SIGNALS = {
    "workflow": [
        "created", "added", "implemented", "built", "set up", "configured",
        "deployed", "migrated", "refactored", "updated",
    ],
    "gotcha": [
        "error", "bug", "fix", "broke", "failed", "wrong", "issue",
        "workaround", "gotcha", "careful", "watch out", "trap",
    ],
    "decision": [
        "decided", "chose", "picked", "went with", "instead of",
        "tradeoff", "trade-off", "because", "rationale",
    ],
    "integration": [
        "api", "endpoint", "webhook", "auth", "token", "oauth",
        "connection", "config", "environment", "secret",
    ],
    "debug": [
        "debug", "diagnose", "investigate", "traced", "found that",
        "root cause", "stack trace", "log", "breakpoint",
    ],
}


def classify_content(text: str) -> str:
    """Classify text into a pattern category based on signal words."""
    text_lower = text.lower()
    scores = {}
    for category, signals in CATEGORY_SIGNALS.items():
        score = sum(1 for s in signals if s in text_lower)
        scores[category] = score

    if max(scores.values()) == 0:
        return "workflow"  # default
    return max(scores, key=scores.get)


def check_novelty(content: str, store: Store, config: Config, threshold: float = 0.78) -> float:
    """Check how novel a pattern is vs existing procedural memories.

    Returns novelty score 0-1 where 1 = completely novel.
    """
    emb = embed_documents([content], config.embedding_model)
    if emb.size == 0:
        return 1.0

    # get procedural memory embeddings
    rows = store.conn.execute(
        "SELECT id, embedding FROM memories WHERE layer = 'procedural' AND forgotten = 0 AND embedding IS NOT NULL"
    ).fetchall()

    if not rows:
        return 1.0  # no procedural memories exist

    import numpy as np
    ids = []
    vecs = []
    for row in rows:
        if row["embedding"]:
            vec = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            vecs.append(vec)
            ids.append(row["id"])

    if not vecs:
        return 1.0

    vec_matrix = np.stack(vecs)
    hits = cosine_similarity_search(emb[0], vec_matrix, top_k=3)

    if not hits:
        return 1.0

    # novelty = 1 - max_similarity
    max_sim = max(score for _, score in hits)
    return round(1.0 - max_sim, 3)


def extract_patterns_from_session(
    store: Store,
    config: Config,
    hours: float = 4.0,
    novelty_threshold: float = 0.25,
) -> list[ExtractedPattern]:
    """Extract reusable patterns from recent session activity.

    Looks at:
    1. Recent diary entries
    2. Recent memories written
    3. Recent events (recalls, writes, edits)

    Groups related activity, classifies it, checks novelty against
    existing procedural memories, and returns patterns worth storing.
    """
    cutoff = time.time() - (hours * 3600)
    patterns = []

    # --- Gather session material ---

    # Recent diary entries
    diary_rows = store.conn.execute(
        "SELECT text FROM diary_entries WHERE created_at > ? ORDER BY created_at",
        (cutoff,),
    ).fetchall()
    diary_texts = [r["text"] for r in diary_rows]

    # Recent memories written (non-codebase, non-working)
    mem_rows = store.conn.execute(
        "SELECT * FROM memories WHERE created_at > ? AND forgotten = 0 "
        "AND layer NOT IN ('working', 'codebase') ORDER BY created_at",
        (cutoff,),
    ).fetchall()
    recent_memories = [store._row_to_memory(row) for row in mem_rows]

    # Recent decision/error memories (already procedural — skip these as sources)
    procedural_ids = {m.id for m in recent_memories if m.layer == MemoryLayer.PROCEDURAL}

    # Recent events for context
    events = store.get_recent_events(limit=100)
    recent_events = [e for e in events if e.get("created_at", 0) > cutoff]

    # --- Build activity clusters ---

    # Group 1: Diary-based patterns (if diary has enough substance)
    if len(diary_texts) >= 2:
        combined_diary = "\n".join(diary_texts)
        if len(combined_diary) > 100:
            category = classify_content(combined_diary)
            title = _extract_title(combined_diary, category)
            pattern_content = _format_pattern(title, combined_diary, category, "diary")
            novelty = check_novelty(pattern_content, store, config)

            patterns.append(ExtractedPattern(
                title=title,
                content=pattern_content,
                category=category,
                source_events=len(diary_texts),
                novelty=novelty,
                should_store=novelty >= novelty_threshold,
            ))

    # Group 2: Error/fix patterns from recent memories
    error_memories = [m for m in recent_memories
                      if m.id not in procedural_ids
                      and any(w in m.content.lower() for w in ['error', 'fix', 'bug', 'broke', 'failed'])]

    for mem in error_memories:
        category = "gotcha"
        title = _extract_title(mem.content, category)
        pattern_content = _format_pattern(title, mem.content, category, "memory")
        novelty = check_novelty(pattern_content, store, config)

        patterns.append(ExtractedPattern(
            title=title,
            content=pattern_content,
            category=category,
            source_events=1,
            novelty=novelty,
            should_store=novelty >= novelty_threshold,
        ))

    # Group 3: Decision patterns from recent non-procedural memories
    decision_memories = [m for m in recent_memories
                         if m.id not in procedural_ids
                         and any(w in m.content.lower() for w in ['decided', 'chose', 'because', 'instead of', 'tradeoff'])]

    for mem in decision_memories:
        category = "decision"
        title = _extract_title(mem.content, category)
        pattern_content = _format_pattern(title, mem.content, category, "memory")
        novelty = check_novelty(pattern_content, store, config)

        patterns.append(ExtractedPattern(
            title=title,
            content=pattern_content,
            category=category,
            source_events=1,
            novelty=novelty,
            should_store=novelty >= novelty_threshold,
        ))

    # Group 4: Workflow patterns from clusters of related writes
    workflow_memories = [m for m in recent_memories
                         if m.id not in procedural_ids
                         and m.layer == MemoryLayer.EPISODIC
                         and len(m.content) > 80]

    if len(workflow_memories) >= 3:
        # combine into a workflow description
        combined = "\n---\n".join(m.content for m in workflow_memories[:5])
        category = classify_content(combined)
        title = _extract_title(combined, category)
        pattern_content = _format_pattern(title, combined, category, "session_workflow")
        novelty = check_novelty(pattern_content, store, config)

        patterns.append(ExtractedPattern(
            title=title,
            content=pattern_content,
            category=category,
            source_events=len(workflow_memories),
            novelty=novelty,
            should_store=novelty >= novelty_threshold,
        ))

    return patterns


def store_patterns(
    patterns: list[ExtractedPattern],
    store: Store,
    config: Config,
) -> dict:
    """Store novel patterns as procedural memories.

    Only stores patterns where should_store=True (novelty above threshold).
    Returns summary of what was stored.
    """
    stored = []
    skipped = []

    for pattern in patterns:
        if not pattern.should_store:
            skipped.append({
                "title": pattern.title,
                "novelty": pattern.novelty,
                "reason": "below novelty threshold",
            })
            continue

        # create procedural memory
        import uuid
        mem = Memory(
            id=str(uuid.uuid4()),
            content=pattern.content,
            source_type=SourceType.AI,
            layer=MemoryLayer.PROCEDURAL,
            importance=0.6 + (pattern.novelty * 0.3),  # higher novelty = higher importance
        )

        emb = embed_documents([mem.content], config.embedding_model)
        if emb.size > 0:
            mem.embedding = emb[0]

        from engram.extractor import generate_hypothetical_queries
        hqs = []
        try:
            hqs = generate_hypothetical_queries(mem.content, config)
        except Exception:
            pass

        store.save_memory(mem, hypothetical_queries=hqs)

        from engram.entities import process_entities_for_memory
        process_entities_for_memory(store, mem.id, mem.content)

        stored.append({
            "id": mem.id,
            "title": pattern.title,
            "category": pattern.category,
            "novelty": pattern.novelty,
            "importance": mem.importance,
        })

    return {
        "stored": stored,
        "skipped": skipped,
        "total_patterns": len(patterns),
        "total_stored": len(stored),
        "total_skipped": len(skipped),
    }


# --- Helpers ---

def _extract_title(text: str, category: str) -> str:
    """Extract a short title from content."""
    # take first meaningful line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return f"Untitled {category} pattern"

    first = lines[0]
    # strip common prefixes
    for prefix in ['Q:', 'A:', 'Error:', 'Decision:', 'Fix:', 'Note:', '-', '*', '#']:
        if first.startswith(prefix):
            first = first[len(prefix):].strip()

    # truncate to reasonable title length
    if len(first) > 80:
        first = first[:77] + "..."

    return first or f"Untitled {category} pattern"


def _format_pattern(title: str, content: str, category: str, source: str) -> str:
    """Format a pattern for storage as procedural memory."""
    return (
        f"Pattern [{category}]: {title}\n"
        f"Source: {source}\n"
        f"---\n"
        f"{content}"
    )
