"""Entity graph exploration.

Shows how to build, query, and traverse the entity relationship
graph. Entities are extracted automatically from memories — this
example demonstrates querying the graph directly.
"""

import uuid

from engram.config import Config
from engram.store import Store, Memory, Entity, Relationship
from engram.embeddings import embed_documents
from engram.entities import process_entities_for_memory


def main():
    config = Config.load()
    store = Store(config)
    store.init_db()

    # --- store memories that mention entities ---
    memories_text = [
        "Ari built the engram memory system using Python and SQLite",
        "engram uses HNSW via hnswlib for approximate nearest neighbor search",
        "the web dashboard is built with FastAPI and serves at port 8420",
        "Ari uses Claude Code for daily development work",
        "Claude Code integrates with engram via the MCP protocol",
        "the dream cycle consolidates memories like biological sleep",
    ]

    print("storing memories and extracting entities...\n")
    for content in memories_text:
        mem = Memory(id=str(uuid.uuid4()), content=content, layer="semantic", importance=0.7)
        emb = embed_documents([content], config.embedding_model)
        if emb.size > 0:
            mem.embedding = emb[0]
        store.save_memory(mem)
        process_entities_for_memory(store, mem.id, content)

    # --- list entities ---
    entities = store.list_entities(limit=20)
    print(f"entities found: {len(entities)}")
    for e in entities:
        print(f"  {e.canonical_name} ({e.entity_type})")

    # --- find a specific entity ---
    ari = store.find_entity_by_name("Ari")
    if ari:
        print(f"\n--- entity: {ari.canonical_name} ---")

        # get relationships
        rels = store.get_entity_relationships(ari.id)
        print(f"relationships: {len(rels)}")
        for r in rels:
            direction = "→" if r["source_entity_id"] == ari.id else "←"
            other = r["target_name"] if r["source_entity_id"] == ari.id else r["source_name"]
            print(f"  {direction} {r['relation_type']} {other} (strength={r['strength']:.1f})")

        # get memories mentioning this entity
        mems = store.get_entity_memories(ari.id)
        print(f"\nmemories: {len(mems)}")
        for m in mems:
            print(f"  [{m.layer}] {m.content[:70]}...")

        # multi-hop traversal
        related = store.get_related_entities(ari.id, max_hops=2)
        print(f"\nrelated entities (2 hops): {len(related)}")
        for r in related:
            indent = "  " * (r["depth"] + 1)
            print(f"{indent}{r['canonical_name']} ({r['entity_type']}) — depth {r['depth']}")

    store.close()


if __name__ == "__main__":
    main()
