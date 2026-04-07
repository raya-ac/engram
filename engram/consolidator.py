"""Dream cycle: cluster similar memories, summarize, generate entity peer cards."""

from __future__ import annotations

import time
import uuid

import numpy as np

from engram.config import Config
from engram.embeddings import embed_documents
from engram.llm import query_llm, extract_json_from_response
from engram.lifecycle import apply_forgetting_curve
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


def consolidate(store: Store, config: Config | None = None) -> dict:
    if config is None:
        config = Config.load()

    stats = {
        "clusters_found": 0,
        "memories_merged": 0,
        "peer_cards_generated": 0,
        "forgotten": 0,
        "promoted": 0,
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
        # only generate if enough non-dream memories
        non_dream = [m for m in memories if m.source_type != SourceType.DREAM]
        if len(non_dream) >= 5:
            _generate_peer_card(entity, non_dream, store, config)
            stats["peer_cards_generated"] += 1

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
