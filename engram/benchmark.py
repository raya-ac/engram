"""Engram self-benchmark — measure retrieval quality, enrichment impact, and system health.

Not a standardized benchmark (those require multi-turn LLM conversations).
This tests the actual memory system: can it find what it stored? Does enrichment
help? Is the intent router working? How fast is each retrieval channel?

Run: engram benchmark
"""

from __future__ import annotations

import json
import time
import random
import math

import numpy as np

from engram.config import Config
from engram.store import Store
from engram.retrieval import search, classify_intent
from engram.embeddings import embed_query, cosine_similarity_search
from engram.hopfield import hopfield_retrieve
from engram.lifecycle import compute_importance, compute_retention
from engram.evolution import get_source_trust


def run_benchmark(config: Config | None = None) -> dict:
    """Run full benchmark suite against the live memory system."""
    if config is None:
        config = Config.load()

    store = Store(config)
    store.init_db()

    results = {}

    # 1. Retrieval quality — can we find memories we know exist?
    results["retrieval"] = _bench_retrieval(store, config)

    # 2. Channel comparison — which channels contribute most?
    results["channels"] = _bench_channels(store, config)

    # 3. Intent classification accuracy
    results["intent"] = _bench_intent()

    # 4. Importance scoring distribution
    results["importance"] = _bench_importance(store, config)

    # 5. Retention distribution
    results["retention"] = _bench_retention(store, config)

    # 6. Trust distribution
    results["trust"] = _bench_trust(store)

    # 7. Latency benchmarks
    results["latency"] = _bench_latency(store, config)

    # 8. Coverage — what fraction of memories are retrievable?
    results["coverage"] = _bench_coverage(store, config)

    # 9. Enrichment stats
    results["enrichment"] = _bench_enrichment(store)

    # 10. Graph connectivity
    results["graph"] = _bench_graph(store)

    store.close()
    return results


def _bench_retrieval(store: Store, config: Config) -> dict:
    """Test retrieval by searching for content we know is stored."""
    # sample 20 random memories
    rows = store.conn.execute(
        "SELECT id, content FROM memories WHERE forgotten=0 AND layer != 'codebase' ORDER BY RANDOM() LIMIT 20"
    ).fetchall()

    if len(rows) < 5:
        return {"error": "not enough memories to benchmark"}

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total = 0

    for row in rows:
        # extract a search query from the first line of the memory
        content = row["content"]
        lines = [l.strip() for l in content.split('\n') if l.strip() and len(l.strip()) > 10]
        if not lines:
            continue

        # use first meaningful line as query
        query = lines[0][:100]
        results = search(query, store, config, top_k=10, rerank=False)
        result_ids = [r.memory.id for r in results]

        total += 1
        if row["id"] in result_ids[:1]:
            hits_at_1 += 1
        if row["id"] in result_ids[:5]:
            hits_at_5 += 1
        if row["id"] in result_ids[:10]:
            hits_at_10 += 1

    return {
        "total_queries": total,
        "recall_at_1": round(hits_at_1 / max(1, total), 3),
        "recall_at_5": round(hits_at_5 / max(1, total), 3),
        "recall_at_10": round(hits_at_10 / max(1, total), 3),
    }


def _bench_channels(store: Store, config: Config) -> dict:
    """Compare retrieval channels by measuring overlap and unique contributions."""
    rows = store.conn.execute(
        "SELECT content FROM memories WHERE forgotten=0 AND layer != 'codebase' ORDER BY RANDOM() LIMIT 10"
    ).fetchall()

    if not rows:
        return {"error": "no memories"}

    channel_hits = {"dense": 0, "bm25": 0, "graph": 0, "hopfield": 0}
    total = 0

    for row in rows:
        query = row["content"].split('\n')[0][:80]
        results = search(query, store, config, top_k=10, rerank=False)

        for r in results[:5]:
            for channel in ["dense", "bm25", "graph"]:
                if r.sources.get(channel, 0) > 0:
                    channel_hits[channel] += 1
            # hopfield doesn't have separate source tracking in current RRF
            # but we can check if it was fused
        total += min(5, len(results))

    return {
        "queries": len(rows),
        "channel_contribution": {
            k: round(v / max(1, total), 3) for k, v in channel_hits.items()
        },
    }


def _bench_intent() -> dict:
    """Test intent classification on known queries."""
    test_cases = [
        ("why did the build fail", "why"),
        ("what caused the error", "why"),
        ("when was the last deploy", "when"),
        ("what date did we ship v2", "when"),
        ("who built the auth system", "who"),
        ("who created engram", "who"),
        ("how to run the tests", "how"),
        ("how do I debug this", "how"),
        ("what is the architecture", "what"),
        ("list all endpoints", "what"),
    ]

    correct = 0
    results = []
    for query, expected in test_cases:
        actual = classify_intent(query)
        is_correct = actual == expected
        if is_correct:
            correct += 1
        results.append({"query": query, "expected": expected, "actual": actual, "correct": is_correct})

    return {
        "accuracy": round(correct / len(test_cases), 3),
        "correct": correct,
        "total": len(test_cases),
        "details": results,
    }


def _bench_importance(store: Store, config: Config) -> dict:
    """Analyze importance score distribution."""
    rows = store.conn.execute(
        "SELECT * FROM memories WHERE forgotten=0"
    ).fetchall()

    scores = []
    for row in rows:
        mem = store._row_to_memory(row)
        score = compute_importance(mem)
        scores.append(score)

    if not scores:
        return {"error": "no memories"}

    arr = np.array(scores)
    return {
        "count": len(scores),
        "mean": round(float(arr.mean()), 3),
        "median": round(float(np.median(arr)), 3),
        "std": round(float(arr.std()), 3),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
        "above_0.5": int((arr > 0.5).sum()),
        "above_0.7": int((arr > 0.7).sum()),
        "below_0.2": int((arr < 0.2).sum()),
    }


def _bench_retention(store: Store, config: Config) -> dict:
    """Analyze retention score distribution."""
    rows = store.conn.execute(
        "SELECT * FROM memories WHERE forgotten=0 AND layer IN ('episodic', 'working')"
    ).fetchall()

    scores = []
    for row in rows:
        mem = store._row_to_memory(row)
        score = compute_retention(mem, config)
        scores.append(score)

    if not scores:
        return {"count": 0}

    arr = np.array(scores)
    return {
        "count": len(scores),
        "mean": round(float(arr.mean()), 3),
        "median": round(float(np.median(arr)), 3),
        "at_risk": int((arr < 0.2).sum()),
        "healthy": int((arr > 0.5).sum()),
    }


def _bench_trust(store: Store) -> dict:
    """Analyze trust distribution by source type."""
    rows = store.conn.execute(
        "SELECT source_type, COUNT(*) as cnt FROM memories WHERE forgotten=0 GROUP BY source_type"
    ).fetchall()

    breakdown = {}
    for row in rows:
        st = row["source_type"]
        trust = get_source_trust(st)
        breakdown[st] = {"count": row["cnt"], "trust": trust}

    return breakdown


def _bench_latency(store: Store, config: Config) -> dict:
    """Measure latency of key operations."""
    query = "test query for latency benchmark"

    # warm up
    search(query, store, config, top_k=5, rerank=False)

    # measure search without rerank
    t0 = time.time()
    for _ in range(3):
        search(query, store, config, top_k=10, rerank=False)
    search_no_rerank = (time.time() - t0) / 3 * 1000

    # measure search with rerank (skip if DB is large — too slow)
    total_mems = store.conn.execute("SELECT COUNT(*) as cnt FROM memories WHERE forgotten=0").fetchone()["cnt"]
    if total_mems < 5000:
        t0 = time.time()
        search(query, store, config, top_k=10, rerank=True)
        search_with_rerank = (time.time() - t0) * 1000
    else:
        search_with_rerank = None  # skipped

    # measure hopfield alone
    query_vec = embed_query(query, config.embedding_model)
    t0 = time.time()
    for _ in range(10):
        hopfield_retrieve(query_vec, store, beta=8.0, top_k=10)
    hopfield_ms = (time.time() - t0) / 10 * 1000

    # measure embedding
    t0 = time.time()
    for _ in range(5):
        embed_query("another test query", config.embedding_model)
    embed_ms = (time.time() - t0) / 5 * 1000

    return {
        "search_no_rerank_ms": round(search_no_rerank, 1),
        "search_with_rerank_ms": round(search_with_rerank, 1) if search_with_rerank is not None else "skipped",
        "hopfield_ms": round(hopfield_ms, 2),
        "embed_ms": round(embed_ms, 1),
    }


def _bench_coverage(store: Store, config: Config) -> dict:
    """What fraction of memories are findable via search?"""
    # sample 30 memories, search for each, check if found in top 20
    rows = store.conn.execute(
        "SELECT id, content FROM memories WHERE forgotten=0 AND embedding IS NOT NULL ORDER BY RANDOM() LIMIT 30"
    ).fetchall()

    found = 0
    total = 0
    for row in rows:
        query = row["content"][:80]
        if len(query) < 10:
            continue
        results = search(query, store, config, top_k=20, rerank=False)
        result_ids = {r.memory.id for r in results}
        total += 1
        if row["id"] in result_ids:
            found += 1

    return {
        "sampled": total,
        "found_in_top_20": found,
        "coverage": round(found / max(1, total), 3),
    }


def _bench_enrichment(store: Store) -> dict:
    """How many memories have enriched metadata?"""
    total = store.conn.execute("SELECT COUNT(*) as cnt FROM memories WHERE forgotten=0").fetchone()["cnt"]
    rows = store.conn.execute("SELECT * FROM memories WHERE forgotten=0").fetchall()
    memories = [store._row_to_memory(r) for r in rows]
    enriched = sum(1 for mem in memories if mem.metadata.get("keywords") is not None)
    evolved = sum(1 for mem in memories if (mem.metadata.get("evolution_count") or 0) > 0)
    confirmed = sum(1 for mem in memories if (mem.metadata.get("confirmations") or 0) > 0)
    with_surprise = sum(1 for mem in memories if mem.metadata.get("surprise") is not None)

    return {
        "total": total,
        "enriched": enriched,
        "enrichment_ratio": round(enriched / max(1, total), 3),
        "evolved": evolved,
        "confirmed": confirmed,
        "with_surprise": with_surprise,
    }


def _bench_graph(store: Store) -> dict:
    """Analyze entity graph connectivity."""
    entity_count = store.conn.execute("SELECT COUNT(*) as cnt FROM entities").fetchone()["cnt"]
    try:
        rel_count = store.conn.execute(
            "SELECT COUNT(*) as cnt FROM relationships WHERE valid_to IS NULL"
        ).fetchone()["cnt"]
    except Exception:
        rel_count = store.conn.execute(
            "SELECT COUNT(*) as cnt FROM relationships"
        ).fetchone()["cnt"]
    orphaned = store.conn.execute(
        "SELECT COUNT(*) as cnt FROM entities WHERE NOT EXISTS (SELECT 1 FROM entity_mentions em WHERE em.entity_id = entities.id)"
    ).fetchone()["cnt"]

    # average degree
    if entity_count > 0:
        try:
            avg_degree = store.conn.execute(
                """SELECT AVG(deg) FROM (
                    SELECT COUNT(*) as deg FROM relationships
                    GROUP BY source_entity_id
                )"""
            ).fetchone()[0] or 0
        except Exception:
            avg_degree = 0
    else:
        avg_degree = 0

    # connected components estimate (via largest community)
    entities = [store._row_to_entity(r) for r in store.conn.execute("SELECT * FROM entities").fetchall()]
    community_counts = {}
    for entity in entities:
        cid = entity.metadata.get("community_id")
        if cid is None:
            continue
        community_counts[cid] = community_counts.get(cid, 0) + 1
    communities = [
        {"cid": cid, "cnt": cnt}
        for cid, cnt in sorted(community_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    ]

    return {
        "entities": entity_count,
        "relationships": rel_count,
        "orphaned": orphaned,
        "avg_degree": round(avg_degree, 2),
        "top_communities": [{"id": r["cid"][:8] if r["cid"] else "?", "size": r["cnt"]} for r in communities],
    }


def run_stress_test(n_memories: int = 500, config: Config | None = None) -> dict:
    """Run benchmark on a fresh temp DB with synthetic data.

    Creates n_memories across all layers with realistic content,
    then runs the full benchmark suite against it.
    """
    import tempfile
    import uuid
    import os

    if config is None:
        config = Config.load()

    # create temp DB
    tmp_dir = tempfile.mkdtemp(prefix="engram_bench_")
    tmp_db = os.path.join(tmp_dir, "bench.db")
    config.db_path = tmp_db

    store = Store(config)
    store.init_db()

    print(f"Creating {n_memories} synthetic memories in {tmp_db}...")

    # synthetic content pools
    TOPICS = [
        "authentication", "database", "deployment", "testing", "API design",
        "caching", "logging", "monitoring", "security", "performance",
        "React components", "TypeScript", "Python", "Docker", "Kubernetes",
        "CI/CD pipeline", "error handling", "user onboarding", "payments",
        "search indexing", "file uploads", "webhooks", "rate limiting",
        "session management", "email notifications", "background jobs",
    ]
    PEOPLE = ["Ari", "Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey"]
    TOOLS = ["Flask", "Next.js", "PostgreSQL", "Redis", "Stripe", "AWS", "Vercel", "Docker"]
    ACTIONS = [
        "implemented", "fixed a bug in", "refactored", "debugged", "deployed",
        "reviewed", "documented", "optimized", "tested", "migrated",
    ]
    LAYERS = ["episodic", "episodic", "episodic", "semantic", "procedural"]
    SOURCES = ["remember:human", "remember:human", "remember:ai", "ingest", "interaction"]

    from engram.store import Memory, MemoryLayer, SourceType
    from engram.embeddings import embed_documents
    from engram.entities import process_entities_for_memory

    # batch embed for speed
    contents = []
    memories = []
    for i in range(n_memories):
        topic = random.choice(TOPICS)
        person = random.choice(PEOPLE)
        tool = random.choice(TOOLS)
        action = random.choice(ACTIONS)
        layer = random.choice(LAYERS)
        source = random.choice(SOURCES)

        # generate varied content
        templates = [
            f"{person} {action} the {topic} system using {tool}. This involved changes to the core module and required updating the configuration.",
            f"Decision: use {tool} for {topic} instead of alternatives. Rationale: better ecosystem support, {person} has experience with it.",
            f"Error in {topic}: {tool} connection timeout after deploy. Fix: increase pool size and add retry logic. Prevention: always test with production-like load.",
            f"The {topic} architecture uses {tool} as the primary backend. Key constraint: all writes must go through the service layer, never direct DB access.",
            f"Meeting with {person} about {topic} roadmap. Agreed to prioritize {tool} integration before the next release. Timeline: 2 weeks.",
            f"{person} discovered that {topic} performance degrades when {tool} cache is cold. Added warmup step to deployment pipeline.",
        ]
        content = random.choice(templates)
        # add some unique identifier so we can find it
        content += f" [ref:{i}]"

        mem = Memory(
            id=str(uuid.uuid4()),
            content=content,
            source_type=source,
            layer=layer,
            importance=0.3 + random.random() * 0.5,
            created_at=time.time() - random.randint(0, 90) * 86400,
            emotional_valence=random.uniform(-0.3, 0.3),
        )
        mem.last_accessed = mem.created_at + random.randint(0, 30) * 86400
        mem.access_count = random.randint(0, 20)
        contents.append(content)
        memories.append(mem)

    # batch embed
    import sys
    print("Embedding...")
    batch_size = 512
    all_embeddings = []
    t_embed = time.time()
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i+batch_size]
        embs = embed_documents(batch, config.embedding_model)
        all_embeddings.append(embs)
        if i > 0 and i % 5000 == 0:
            rate = i / (time.time() - t_embed)
            eta = (len(contents) - i) / max(1, rate)
            print(f"  {i}/{len(contents)} ({rate:.0f}/sec, ETA {eta:.0f}s)")
            sys.stdout.flush()
    all_embeddings = np.vstack(all_embeddings)
    print(f"  Embedded {len(contents)} in {time.time()-t_embed:.1f}s ({len(contents)/(time.time()-t_embed):.0f}/sec)")

    # bulk store — bypass save_memory for speed (no FTS, no entity extraction)
    import json as _json
    print("Storing...")
    t_store = time.time()
    if getattr(store.config, "normalized_storage_backend", "sqlite") == "sqlite":
        store.conn.execute("PRAGMA synchronous = OFF")
        store.conn.execute("PRAGMA journal_mode = MEMORY")
    for i, mem in enumerate(memories):
        emb_blob = all_embeddings[i].astype(np.float32).tobytes() if i < len(all_embeddings) else None
        store.conn.execute(
            """INSERT INTO memories
            (id, content, source_type, layer, embedding, importance,
             access_count, created_at, last_accessed, emotional_valence,
             chunk_hash, metadata, forgotten)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (mem.id, mem.content, mem.source_type, mem.layer, emb_blob,
             mem.importance, mem.access_count, mem.created_at, mem.last_accessed,
             mem.emotional_valence, mem.chunk_hash, _json.dumps(mem.metadata)),
        )
        # FTS for a subset only (bulk FTS is slow)
        if i < 10000:
            store.refresh_fts_entry(mem.id, mem.content, "")
        if i > 0 and i % 10000 == 0:
            store.conn.commit()
            print(f"  {i}/{len(memories)} stored ({i/(time.time()-t_store):.0f}/sec)")
    store.conn.commit()
    store.invalidate_embedding_cache()
    print(f"  Stored {len(memories)} in {time.time()-t_store:.1f}s")

    print(f"Created {n_memories} memories, running benchmark...\n")

    # override the store for benchmark
    results = {}
    results["retrieval"] = _bench_retrieval(store, config)
    results["channels"] = _bench_channels(store, config)
    results["intent"] = _bench_intent()
    results["importance"] = _bench_importance(store, config)
    results["retention"] = _bench_retention(store, config)
    results["trust"] = _bench_trust(store)
    results["latency"] = _bench_latency(store, config)
    results["coverage"] = _bench_coverage(store, config)
    results["enrichment"] = _bench_enrichment(store)
    results["graph"] = _bench_graph(store)
    results["db_path"] = tmp_db
    results["n_memories"] = n_memories

    store.close()

    # cleanup
    try:
        os.remove(tmp_db)
        os.rmdir(tmp_dir)
    except Exception:
        pass

    return results


def print_benchmark(results: dict):
    """Pretty-print benchmark results."""
    print("\n\033[1m=== ENGRAM BENCHMARK ===\033[0m\n")

    # retrieval
    r = results.get("retrieval", {})
    print(f"\033[36mRetrieval Quality\033[0m")
    print(f"  Recall@1:  {r.get('recall_at_1', '?')}")
    print(f"  Recall@5:  {r.get('recall_at_5', '?')}")
    print(f"  Recall@10: {r.get('recall_at_10', '?')}")
    print(f"  Queries:   {r.get('total_queries', '?')}")

    # intent
    i = results.get("intent", {})
    print(f"\n\033[36mIntent Classification\033[0m")
    print(f"  Accuracy: {i.get('accuracy', '?')} ({i.get('correct', '?')}/{i.get('total', '?')})")

    # latency
    l = results.get("latency", {})
    print(f"\n\033[36mLatency\033[0m")
    print(f"  Search (no rerank): {l.get('search_no_rerank_ms', '?')}ms")
    print(f"  Search (rerank):    {l.get('search_with_rerank_ms', '?')}ms")
    print(f"  Hopfield:           {l.get('hopfield_ms', '?')}ms")
    print(f"  Embedding:          {l.get('embed_ms', '?')}ms")

    # coverage
    c = results.get("coverage", {})
    print(f"\n\033[36mCoverage\033[0m")
    print(f"  Found in top 20: {c.get('found_in_top_20', '?')}/{c.get('sampled', '?')} ({c.get('coverage', '?')})")

    # importance
    imp = results.get("importance", {})
    print(f"\n\033[36mImportance Distribution\033[0m")
    print(f"  Mean: {imp.get('mean', '?')} | Median: {imp.get('median', '?')} | Std: {imp.get('std', '?')}")
    print(f"  >0.7: {imp.get('above_0.7', '?')} | >0.5: {imp.get('above_0.5', '?')} | <0.2: {imp.get('below_0.2', '?')}")

    # retention
    ret = results.get("retention", {})
    print(f"\n\033[36mRetention (episodic/working)\033[0m")
    print(f"  Mean: {ret.get('mean', '?')} | Healthy (>0.5): {ret.get('healthy', '?')} | At risk (<0.2): {ret.get('at_risk', '?')}")

    # enrichment
    e = results.get("enrichment", {})
    print(f"\n\033[36mEnrichment\033[0m")
    print(f"  Enriched: {e.get('enriched', '?')}/{e.get('total', '?')} ({e.get('enrichment_ratio', '?')})")
    print(f"  Evolved: {e.get('evolved', '?')} | Confirmed: {e.get('confirmed', '?')} | With surprise: {e.get('with_surprise', '?')}")

    # graph
    g = results.get("graph", {})
    print(f"\n\033[36mEntity Graph\033[0m")
    print(f"  Entities: {g.get('entities', '?')} | Relationships: {g.get('relationships', '?')} | Orphaned: {g.get('orphaned', '?')}")
    print(f"  Avg degree: {g.get('avg_degree', '?')}")

    print()
