"""Negative knowledge — storing what does NOT exist.

Prevents hallucinated recommendations by explicitly recording
things that were tried and rejected, features that don't exist,
or approaches that should not be used.
"""

import uuid

from engram.config import Config
from engram.store import Store, Memory
from engram.embeddings import embed_documents
from engram.retrieval import search as hybrid_search


def main():
    config = Config.load()
    store = Store(config)
    store.init_db()

    # --- store negative knowledge ---
    negatives = [
        "NEGATIVE KNOWLEDGE: There is no caching layer in this project. "
        "We evaluated Redis and Memcached but decided the SQLite WAL mode "
        "was sufficient for our read patterns.",

        "NEGATIVE KNOWLEDGE: We do NOT use Redux for state management. "
        "React context + useReducer covers all our needs. Redux was "
        "considered and explicitly rejected for complexity reasons.",

        "NEGATIVE KNOWLEDGE: The /admin API endpoint was removed in v2.0. "
        "Admin operations now go through the CLI. Do not recommend building "
        "admin features that depend on this endpoint.",
    ]

    for content in negatives:
        mem = Memory(
            id=str(uuid.uuid4()),
            content=content,
            layer="semantic",
            importance=0.75,
            source_type="remember:human",
        )
        emb = embed_documents([content], config.embedding_model)
        if emb.size > 0:
            mem.embedding = emb[0]
        store.save_memory(mem)
        print(f"stored: {content[:60]}...")

    # --- now search for the thing that doesn't exist ---
    print("\n--- searching for 'caching layer' ---")
    results = hybrid_search("caching layer Redis", store, config, top_k=3, rerank=False)
    for r in results:
        print(f"  [{r.memory.layer}] {r.memory.content[:80]}...")
        print(f"    score={r.score:.4f}")

    print("\n--- searching for 'Redux state management' ---")
    results = hybrid_search("Redux state management", store, config, top_k=3, rerank=False)
    for r in results:
        print(f"  [{r.memory.layer}] {r.memory.content[:80]}...")

    print("\n--- searching for 'admin endpoint' ---")
    results = hybrid_search("admin API endpoint", store, config, top_k=3, rerank=False)
    for r in results:
        print(f"  [{r.memory.layer}] {r.memory.content[:80]}...")

    store.close()
    print("\nthe negative knowledge surfaces when someone searches for")
    print("the thing that doesn't exist — preventing bad recommendations.")


if __name__ == "__main__":
    main()
