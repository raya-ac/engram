"""Memory evolution — enrich at write time, evolve neighbors on new memories.

Implements two ideas from the papers:
1. Enriched embeddings (A-Mem): generate keywords + tags + summary, embed the concatenation
2. Memory evolution (A-Mem): when near-neighbors detected, ask LLM if they should be updated
3. Trust-weighted decay (SuperLocalMemory): source-based trust scoring
4. Confirmation count (SuperLocalMemory): track independent corroboration
"""

from __future__ import annotations

import json
import time
from typing import Any

from engram.config import Config
from engram.llm import query_llm
from engram.store import Store, Memory


# --- Source trust mapping ---

SOURCE_TRUST = {
    "remember:human": 1.0,      # user explicitly stored this
    "remember:ai": 0.7,         # AI-generated but reviewed
    "interaction": 0.6,         # conversation extract
    "ingest": 0.5,              # bulk file ingestion
    "dream": 0.4,               # consolidation-generated
    "code_scan": 0.5,           # codebase scan
}

def get_source_trust(source_type: str) -> float:
    """Get trust score for a memory source type."""
    return SOURCE_TRUST.get(source_type, 0.5)


# --- Enriched embeddings ---

ENRICH_SYSTEM = """Extract structured metadata from this memory content.

Return JSON with:
- "keywords": list of 3-7 specific keywords (not generic words)
- "tags": list of 1-3 categorical tags (e.g. "security", "architecture", "debugging")
- "summary": one sentence contextual description of what this memory captures

Be specific. Keywords should be names, tools, concepts — not "the", "about", "related".
Respond with ONLY the JSON object."""


def enrich_memory(content: str, config: Config) -> dict:
    """Generate keywords, tags, and summary for a memory.

    Returns dict with keys: keywords, tags, summary, enriched_text
    The enriched_text is the concatenation used for embedding.
    """
    try:
        response = query_llm(content[:2000], system=ENRICH_SYSTEM, config=config)
        # parse JSON from response
        # handle potential markdown wrapping
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(text)

        keywords = result.get("keywords", [])
        tags = result.get("tags", [])
        summary = result.get("summary", "")

        # build enriched text for embedding
        parts = [content]
        if keywords:
            parts.append("Keywords: " + ", ".join(keywords))
        if tags:
            parts.append("Tags: " + ", ".join(tags))
        if summary:
            parts.append("Context: " + summary)

        return {
            "keywords": keywords,
            "tags": tags,
            "summary": summary,
            "enriched_text": "\n".join(parts),
        }
    except Exception:
        return {
            "keywords": [],
            "tags": [],
            "summary": "",
            "enriched_text": content,
        }


# --- Memory evolution ---

EVOLVE_SYSTEM = """A new memory has arrived that is related to an existing memory.
Determine if the existing memory should be updated given the new information.

If the new memory adds context, corrects, supersedes, or extends the existing memory,
produce an updated version of the existing memory that incorporates the new information.

If the existing memory is still accurate and the new memory doesn't meaningfully change it,
respond with: {"evolved": false}

If it should be updated, respond with:
{"evolved": true, "new_content": "the updated memory content"}

Respond with ONLY the JSON object."""


def evolve_neighbors(new_memory: Memory, neighbors: list[Memory],
                     store: Store, config: Config, max_evolve: int = 3) -> list[str]:
    """Check if existing memories should evolve given a new memory.

    Returns list of memory IDs that were evolved.
    """
    evolved_ids = []

    for neighbor in neighbors[:max_evolve]:
        # skip if neighbor is much more important (don't mess with core knowledge)
        if neighbor.importance > 0.85:
            continue
        # skip if neighbor is in semantic/procedural (consolidated knowledge)
        if neighbor.layer in ("semantic", "procedural"):
            continue

        prompt = (
            f"NEW MEMORY:\n{new_memory.content[:1000]}\n\n"
            f"EXISTING MEMORY:\n{neighbor.content[:1000]}"
        )

        try:
            response = query_llm(prompt, system=EVOLVE_SYSTEM, config=config)
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(text)

            if result.get("evolved") and result.get("new_content"):
                new_content = result["new_content"]
                # update the memory content
                from engram.embeddings import embed_documents
                emb = embed_documents([new_content], config.embedding_model)
                emb_blob = emb[0].astype('float32').tobytes() if emb.size > 0 else None

                store.conn.execute(
                    "UPDATE memories SET content = ?, embedding = ? WHERE id = ?",
                    (new_content, emb_blob, neighbor.id),
                )

                # update metadata
                meta = neighbor.metadata
                meta["evolved_at"] = time.time()
                meta["evolved_by"] = new_memory.id
                evolution_count = meta.get("evolution_count", 0) + 1
                meta["evolution_count"] = evolution_count
                store.conn.execute(
                    "UPDATE memories SET metadata = ? WHERE id = ?",
                    (json.dumps(meta), neighbor.id),
                )

                # rebuild FTS
                row = store.conn.execute(
                    "SELECT rowid FROM memories WHERE id = ?", (neighbor.id,)
                ).fetchone()
                if row:
                    store.conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (row[0],))
                    hqs = store.conn.execute(
                        "SELECT query_text FROM hypothetical_queries WHERE memory_id = ?",
                        (neighbor.id,),
                    ).fetchall()
                    hq_text = " ".join(r["query_text"] for r in hqs)
                    store.conn.execute(
                        "INSERT INTO memories_fts (rowid, content, hypothetical_queries) VALUES (?, ?, ?)",
                        (row[0], new_content, hq_text),
                    )

                store.invalidate_embedding_cache()
                evolved_ids.append(neighbor.id)
        except Exception:
            continue

    if evolved_ids:
        store.conn.commit()

    return evolved_ids


# --- Confirmation tracking ---

def check_confirmation(new_content: str, neighbor: Memory, store: Store) -> bool:
    """Check if a new memory independently confirms an existing one.

    If so, increment the confirmation count on the existing memory.
    Returns True if confirmed.
    """
    # simple heuristic: if content is similar enough but from a different source,
    # it's an independent confirmation
    if neighbor.source_type == "dream":
        return False  # dream-generated can't confirm

    meta = neighbor.metadata
    confirmations = meta.get("confirmations", 0)
    confirmed_by = meta.get("confirmed_by", [])

    # don't double-count
    # (caller should check similarity > 0.8 before calling this)
    confirmations += 1
    meta["confirmations"] = confirmations
    meta["confirmed_by"] = confirmed_by
    meta["last_confirmed"] = time.time()

    store.conn.execute(
        "UPDATE memories SET metadata = ? WHERE id = ?",
        (json.dumps(meta), neighbor.id),
    )
    return True
