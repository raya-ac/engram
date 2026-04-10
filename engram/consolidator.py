"""Dream cycle: cluster similar memories, summarize, generate entity peer cards."""

from __future__ import annotations

import time
import uuid

import numpy as np

from engram.config import Config
from engram.embeddings import embed_documents
from engram.llm import query_llm, extract_json_from_response
from engram.lifecycle import apply_forgetting_curve
from engram.drift import run_drift_check, auto_fix_drift
from engram.store import Store, Memory, MemoryLayer, SourceType

SUMMARIZE_SYSTEM = """You are a memory consolidation system. Given a cluster of related memories, produce a single consolidated summary that preserves all important information while removing redundancy.

Rules:
1. Keep all specific dates, names, and facts
2. Merge duplicate information
3. Preserve the most important details
4. Be concise but complete
5. Name all subjects explicitly

Respond with just the consolidated text."""

PEER_CARD_SYSTEM = """You are generating a biographical peer card about a specific entity based on collected memories. Produce a structured summary of everything known about this entity.

Include:
- Who/what they are
- Key facts and attributes
- Relationships to other entities
- Timeline of significant events
- Notable patterns or behaviors

Respond with a structured text summary, 200-400 words."""


SYNTHESIS_SYSTEM = """You are a cross-domain synthesis engine. Given two entities from different domains/contexts along with their associated memories, identify meaningful connections between them.

Rules:
1. Only report genuine, non-trivial connections
2. Don't force connections that aren't there — return "no meaningful connection" if appropriate
3. Be specific about what connects them
4. Note potential implications or insights from the connection
5. Keep it to 2-3 sentences

Respond with JSON: {"connected": true/false, "synthesis": "description of connection", "insight": "what this means"}"""


def consolidate(store: Store, config: Config | None = None) -> dict:
    if config is None:
        config = Config.load()

    stats = {
        "clusters_found": 0,
        "memories_merged": 0,
        "peer_cards_generated": 0,
        "forgotten": 0,
        "promoted": 0,
        "cross_domain_bridges": 0,
    }

    # Step 1: Apply forgetting curve
    lifecycle_stats = apply_forgetting_curve(store, config)
    stats["forgotten"] = lifecycle_stats["forgotten"]
    stats["promoted"] = lifecycle_stats["promoted"]

    # Step 2: Cluster similar memories
    clusters = _find_clusters(store, config)
    stats["clusters_found"] = len(clusters)

    for cluster in clusters:
        if len(cluster) >= config.lifecycle.cluster_min_size:
            _merge_cluster(cluster, store, config)
            stats["memories_merged"] += len(cluster)

    # Step 3: Generate entity peer cards
    entities = store.list_entities(limit=50)
    for entity in entities:
        memories = store.get_entity_memories(entity.id, limit=100)
        non_dream = [m for m in memories if m.source_type != SourceType.DREAM]
        if len(non_dream) >= 5:
            _generate_peer_card(entity, non_dream, store, config)
            stats["peer_cards_generated"] += 1

    # Step 4: Cross-domain synthesis
    bridges = _cross_domain_synthesis(store, config)
    stats["cross_domain_bridges"] = bridges

    # Step 5: Drift detection — validate memories against filesystem
    try:
        drift_report = run_drift_check(store, check_functions=False)
        stats["drift_score"] = drift_report.score
        stats["drift_issues"] = len(drift_report.issues)
        # auto-fix errors (invalidate dead paths, forget invalidated-but-active)
        if drift_report.issues:
            fix_result = auto_fix_drift(store, drift_report, dry_run=False)
            stats["drift_invalidated"] = fix_result["invalidated"]
            stats["drift_forgotten"] = fix_result["forgotten"]
    except Exception:
        stats["drift_score"] = -1  # signal that drift check failed

    # Step 6: Prune old access_log and events entries (>90 days)
    cutoff = time.time() - 90 * 86400
    pruned_access = store.conn.execute(
        "DELETE FROM access_log WHERE accessed_at < ?", (cutoff,)
    ).rowcount
    pruned_events = store.conn.execute(
        "DELETE FROM events WHERE created_at < ?", (cutoff,)
    ).rowcount
    stats["pruned_access_log"] = pruned_access
    stats["pruned_events"] = pruned_events

    store._emit_event("consolidation", detail=str(stats))
    store.conn.commit()
    return stats


def _find_clusters(store: Store, config: Config) -> list[list[Memory]]:
    ids, vecs = store.get_all_embeddings()
    if len(ids) < 2:
        return []

    threshold = config.lifecycle.cluster_threshold
    used = set()
    clusters = []

    # greedy clustering: for each memory, find all similar ones
    sim_matrix = vecs @ vecs.T

    for i in range(len(ids)):
        if i in used:
            continue
        cluster_indices = [i]
        used.add(i)
        for j in range(i + 1, len(ids)):
            if j in used:
                continue
            if sim_matrix[i, j] >= threshold:
                cluster_indices.append(j)
                used.add(j)

        if len(cluster_indices) >= 2:
            cluster_mems = []
            for idx in cluster_indices:
                mem = store.get_memory(ids[idx])
                if mem:
                    cluster_mems.append(mem)
            if len(cluster_mems) >= 2:
                clusters.append(cluster_mems)

    return clusters


def _merge_cluster(cluster: list[Memory], store: Store, config: Config):
    contents = [m.content for m in cluster]
    combined = "\n---\n".join(contents)

    prompt = f"Consolidate these {len(cluster)} related memories into one:\n\n{combined}"
    try:
        summary = query_llm(prompt, system=SUMMARIZE_SYSTEM, config=config)
    except Exception:
        # fallback: keep the most important one
        summary = max(cluster, key=lambda m: m.importance).content

    # create consolidated memory
    merged = Memory(
        id=str(uuid.uuid4()),
        content=summary,
        source_type=SourceType.DREAM,
        layer=MemoryLayer.SEMANTIC,
        importance=max(m.importance for m in cluster),
        fact_date=min((m.fact_date for m in cluster if m.fact_date), default=None),
        metadata={"merged_from": [m.id for m in cluster]},
    )

    # embed and save
    from engram.embeddings import embed_documents
    emb = embed_documents([summary])
    if emb.size > 0:
        merged.embedding = emb[0]

    store.save_memory(merged)

    # soft-forget the originals
    for m in cluster:
        store.forget_memory(m.id)


def _generate_peer_card(entity, memories: list[Memory], store: Store, config: Config):
    facts = "\n".join(f"- {m.content}" for m in memories[:50])
    prompt = f"Entity: {entity.canonical_name} ({entity.entity_type})\n\nKnown facts:\n{facts}"

    try:
        card = query_llm(prompt, system=PEER_CARD_SYSTEM, config=config)
    except Exception:
        return

    card_memory = Memory(
        id=str(uuid.uuid4()),
        content=f"[Peer Card: {entity.canonical_name}]\n\n{card}",
        source_type=SourceType.DREAM,
        layer=MemoryLayer.SEMANTIC,
        importance=0.8,
        metadata={"entity_id": entity.id, "type": "peer_card"},
    )

    from engram.embeddings import embed_documents
    emb = embed_documents([card_memory.content])
    if emb.size > 0:
        card_memory.embedding = emb[0]

    store.save_memory(card_memory)
    store.link_entity_memory(entity.id, card_memory.id)


def _cross_domain_synthesis(store: Store, config: Config, max_bridges: int = 5) -> int:
    """Find entity pairs in different domains that share no direct relationship.

    For each pair, check if their memory embeddings have unexpected similarity
    (semantically related despite being in different contexts). If so, generate
    a synthesis memory bridging them.
    """
    # get entities with enough memories (≥3) and their domain signals
    entities = store.list_entities(limit=100)
    entity_profiles = []

    for entity in entities:
        memories = store.get_entity_memories(entity.id, limit=50)
        non_dream = [m for m in memories if m.source_type != SourceType.DREAM]
        if len(non_dream) < 3:
            continue

        # determine domain from memory layers and content patterns
        layers = set(m.layer for m in non_dream)
        content_concat = " ".join(m.content[:200] for m in non_dream[:10])

        entity_profiles.append({
            "entity": entity,
            "memories": non_dream,
            "layers": layers,
            "content_sample": content_concat,
        })

    if len(entity_profiles) < 2:
        return 0

    # find pairs that are NOT directly related but share embedding similarity
    bridges_created = 0

    # compute mean embeddings per entity
    entity_embeddings = {}
    for profile in entity_profiles:
        embs = [m.embedding for m in profile["memories"] if m.embedding is not None]
        if embs:
            entity_embeddings[profile["entity"].id] = np.mean(embs, axis=0)

    # get existing relationships to exclude
    existing_pairs = set()
    for profile in entity_profiles:
        rels = store.get_entity_relationships(profile["entity"].id)
        for r in rels:
            pair = tuple(sorted([r["source_entity_id"], r["target_entity_id"]]))
            existing_pairs.add(pair)

    # find candidate bridges: entities with similar embedding centroids but no relationship
    candidates = []
    profiles_with_emb = [p for p in entity_profiles if p["entity"].id in entity_embeddings]

    for i, p1 in enumerate(profiles_with_emb):
        for p2 in profiles_with_emb[i+1:]:
            eid1, eid2 = p1["entity"].id, p2["entity"].id
            pair = tuple(sorted([eid1, eid2]))

            if pair in existing_pairs:
                continue

            emb1 = entity_embeddings[eid1]
            emb2 = entity_embeddings[eid2]
            # normalize for cosine similarity
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                continue
            sim = float(np.dot(emb1, emb2) / (norm1 * norm2))

            # we want moderate similarity — too high means they're already
            # the same domain, too low means no real connection.
            # empirically, mean entity embeddings cluster 0.7-0.95 in a
            # single-user system, so the interesting range is 0.75-0.90
            if 0.75 <= sim <= 0.90:
                candidates.append((p1, p2, sim))

    # sort by similarity (higher = more likely real connection)
    candidates.sort(key=lambda x: x[2], reverse=True)

    for p1, p2, sim in candidates[:max_bridges]:
        e1, e2 = p1["entity"], p2["entity"]

        # use LLM to check if there's a genuine connection
        facts1 = "\n".join(f"- {m.content[:150]}" for m in p1["memories"][:5])
        facts2 = "\n".join(f"- {m.content[:150]}" for m in p2["memories"][:5])

        prompt = (
            f"Entity A: {e1.canonical_name} ({e1.entity_type})\n"
            f"Facts about A:\n{facts1}\n\n"
            f"Entity B: {e2.canonical_name} ({e2.entity_type})\n"
            f"Facts about B:\n{facts2}\n\n"
            f"Embedding similarity: {sim:.2f}"
        )

        try:
            response = query_llm(prompt, system=SYNTHESIS_SYSTEM, config=config)
            result = extract_json_from_response(response)
        except Exception:
            continue

        if not result or not result.get("connected"):
            continue

        # create a bridge memory
        synthesis = result.get("synthesis", "")
        insight = result.get("insight", "")
        bridge_content = (
            f"[Cross-domain bridge: {e1.canonical_name} ↔ {e2.canonical_name}]\n\n"
            f"Connection: {synthesis}\n"
            f"Insight: {insight}\n"
            f"Embedding similarity: {sim:.2f}"
        )

        bridge = Memory(
            id=str(uuid.uuid4()),
            content=bridge_content,
            source_type=SourceType.DREAM,
            layer=MemoryLayer.SEMANTIC,
            importance=0.6,
            metadata={
                "type": "cross_domain_bridge",
                "entity_a": e1.canonical_name,
                "entity_b": e2.canonical_name,
                "similarity": round(sim, 4),
            },
        )

        emb = embed_documents([bridge_content])
        if emb.size > 0:
            bridge.embedding = emb[0]

        store.save_memory(bridge)

        # link to both entities
        store.link_entity_memory(e1.id, bridge.id)
        store.link_entity_memory(e2.id, bridge.id)

        # create a SYNTHESIZED_WITH relationship
        from engram.store import Relationship
        rel = Relationship(
            source_entity_id=e1.id,
            target_entity_id=e2.id,
            relation_type="SYNTHESIZED_WITH",
            created_at=time.time(),
            last_seen=time.time(),
        )
        store.save_relationship(rel)

        bridges_created += 1

    return bridges_created
