"""Standalone Python client for engram.

Use this when building your own agent or application that needs
persistent memory without going through MCP. Talks directly to
the engram library.
"""

from engram.config import Config
from engram.store import Store, Memory, MemoryLayer, SourceType
from engram.embeddings import embed_documents, embed_query
from engram.retrieval import search as hybrid_search
from engram.surprise import compute_surprise, adjust_importance
from engram.entities import process_entities_for_memory
from engram.deep_retrieval import DeepReranker
from engram.lifecycle import apply_forgetting_curve, compute_retention

import uuid


def main():
    # --- setup ---
    config = Config.load()
    store = Store(config)
    store.init_db()

    # load the deep reranker (if trained)
    reranker_path = config.resolved_db_path.parent / "reranker.npz"
    reranker = DeepReranker(model_path=reranker_path)

    # --- store a memory with surprise scoring ---
    content = "The deploy pipeline uses blue-green deployment with a 5-minute canary window"
    mem = Memory(
        id=str(uuid.uuid4()),
        content=content,
        source_type=SourceType.HUMAN,
        layer=MemoryLayer.PROCEDURAL,
        importance=0.8,
    )

    # embed
    emb = embed_documents([content], config.embedding_model)
    if emb.size > 0:
        mem.embedding = emb[0]

        # compute surprise before storing
        surprise = compute_surprise(mem.embedding, store)
        mem.importance = adjust_importance(mem.importance, surprise)
        mem.metadata["surprise"] = surprise["surprise"]

        print(f"Surprise: {surprise['surprise']:.3f}")
        print(f"Adjusted importance: {mem.importance:.3f}")

        if surprise["is_duplicate"]:
            print(f"Warning: near-duplicate of {surprise['nearest_id']}")

    # save
    store.save_memory(mem)
    process_entities_for_memory(store, mem.id, content)
    print(f"Stored memory {mem.id}")

    # --- search with deep reranker ---
    results = hybrid_search(
        "deployment process",
        store, config,
        top_k=5,
        deep_reranker=reranker,
    )

    print(f"\nSearch results for 'deployment process':")
    for r in results:
        print(f"  [{r.memory.layer}] {r.memory.content[:80]}...")
        print(f"    score={r.score:.4f} sources={r.sources}")

    # --- check retention for a memory ---
    for r in results[:1]:
        retention = compute_retention(r.memory, config)
        print(f"\n  Retention score: {retention:.4f}")
        print(f"  Access count: {r.memory.access_count}")
        print(f"  Layer: {r.memory.layer}")

    # --- train the reranker (if enough data) ---
    if not reranker.is_trained:
        train_result = reranker.train(store, epochs=50)
        print(f"\nReranker training: {train_result}")

    # --- run lifecycle sweep ---
    stats = apply_forgetting_curve(store, config)
    print(f"\nLifecycle sweep: {stats}")

    # --- stats ---
    print(f"\nSystem stats: {store.get_stats()}")

    store.close()


if __name__ == "__main__":
    main()
