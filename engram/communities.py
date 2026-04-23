"""Community detection over the entity graph (Graphiti/Zep-inspired).

Uses label propagation to discover clusters of related entities.
Generates community summaries for higher-level retrieval.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter

from engram.store import Store, Memory, MemoryLayer, SourceType


def detect_communities(store: Store, min_community_size: int = 3) -> dict:
    """Run label propagation over the entity graph.

    Returns dict with stats and community assignments.
    """
    # load all entities and relationships
    entities = store.conn.execute(
        "SELECT id, canonical_name, entity_type FROM entities"
    ).fetchall()
    if not entities:
        return {"communities": 0, "entities_assigned": 0}

    entity_ids = [e["id"] for e in entities]
    entity_map = {e["id"]: dict(e) for e in entities}

    # build adjacency list
    adjacency: dict[str, list[tuple[str, float]]] = {eid: [] for eid in entity_ids}
    rels = store.conn.execute(
        "SELECT source_entity_id, target_entity_id, strength FROM relationships WHERE valid_to IS NULL"
    ).fetchall()
    for r in rels:
        src, tgt = r["source_entity_id"], r["target_entity_id"]
        strength = r["strength"] or 1.0
        if src in adjacency:
            adjacency[src].append((tgt, strength))
        if tgt in adjacency:
            adjacency[tgt].append((src, strength))

    # label propagation — each entity starts as its own community
    labels = {eid: eid for eid in entity_ids}

    max_iterations = 20
    for _ in range(max_iterations):
        changed = False
        # shuffle order to avoid bias
        import random
        order = entity_ids.copy()
        random.shuffle(order)

        for eid in order:
            neighbors = adjacency.get(eid, [])
            if not neighbors:
                continue

            # weighted vote from neighbors
            votes: dict[str, float] = Counter()
            for neighbor_id, strength in neighbors:
                votes[labels[neighbor_id]] += strength

            if votes:
                best_label = max(votes, key=votes.get)
                if best_label != labels[eid]:
                    labels[eid] = best_label
                    changed = True

        if not changed:
            break

    # group entities by community label
    communities: dict[str, list[str]] = {}
    for eid, label in labels.items():
        communities.setdefault(label, []).append(eid)

    # filter to minimum size
    communities = {k: v for k, v in communities.items() if len(v) >= min_community_size}

    # store community assignments in entity metadata
    for community_id, member_ids in communities.items():
        for eid in member_ids:
            entity = store.conn.execute(
                "SELECT metadata FROM entities WHERE id = ?", (eid,)
            ).fetchone()
            if entity:
                meta = json.loads(entity["metadata"] or "{}")
                meta["community_id"] = community_id
                meta["community_size"] = len(member_ids)
                store.conn.execute(
                    "UPDATE entities SET metadata = ? WHERE id = ?",
                    (json.dumps(meta), eid),
                )

    store.conn.commit()

    return {
        "communities": len(communities),
        "entities_assigned": sum(len(v) for v in communities.values()),
        "largest_community": max(len(v) for v in communities.values()) if communities else 0,
        "community_details": [
            {
                "id": cid,
                "size": len(members),
                "members": [entity_map[m]["canonical_name"] for m in members if m in entity_map][:10],
            }
            for cid, members in sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)[:20]
        ],
    }


def generate_community_summaries(store: Store, config=None) -> int:
    """Generate LLM summaries for detected communities.

    Creates a semantic memory for each community with a summary of
    its member entities and their relationships.
    """
    from engram.llm import query_llm
    from engram.embeddings import embed_documents
    if config is None:
        from engram.config import Config
        config = Config.load()

    rows = store.conn.execute("SELECT * FROM entities ORDER BY canonical_name").fetchall()
    entities = [store._row_to_entity(r) for r in rows]
    members_with_community = []
    for entity in entities:
        cid = entity.metadata.get("community_id")
        if cid is None:
            continue
        members_with_community.append({
            "cid": cid,
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type,
            "id": entity.id,
        })

    if not members_with_community:
        return 0

    # group by community
    communities: dict[str, list[dict]] = {}
    for r in members_with_community:
        cid = r["cid"]
        communities.setdefault(cid, []).append(dict(r))

    summaries_created = 0
    for cid, members in communities.items():
        if len(members) < 3:
            continue

        # check if we already have a summary for this community
        existing = False
        for row in store.conn.execute("SELECT * FROM memories WHERE forgotten=0").fetchall():
            mem = store._row_to_memory(row)
            if mem.metadata.get("community_id") == cid:
                existing = True
                break
        if existing:
            continue

        # gather entity names and sample memories
        names = [m["canonical_name"] for m in members[:15]]
        sample_memories = []
        for m in members[:5]:
            mems = store.get_entity_memories(m["id"], limit=3)
            for mem in mems:
                sample_memories.append(f"[{m['canonical_name']}] {mem.content[:150]}")

        prompt = (
            f"Community members: {', '.join(names)}\n\n"
            f"Sample facts:\n" + "\n".join(sample_memories[:15]) + "\n\n"
            f"Write a 2-3 sentence summary of what connects these entities."
        )

        try:
            summary = query_llm(prompt, system="Summarize what connects a group of related entities. Be concise.", config=config)
        except Exception:
            continue

        # store as semantic memory
        mem = Memory(
            id=str(uuid.uuid4()),
            content=f"[Community: {', '.join(names[:5])}{'...' if len(names) > 5 else ''}]\n\n{summary}",
            source_type=SourceType.DREAM,
            layer=MemoryLayer.SEMANTIC,
            importance=0.6,
            metadata={"type": "community_summary", "community_id": cid, "member_count": len(members)},
        )

        emb = embed_documents([mem.content])
        if emb.size > 0:
            mem.embedding = emb[0]

        store.save_memory(mem)
        summaries_created += 1

    return summaries_created
