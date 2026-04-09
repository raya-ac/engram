"""Interactive demo — walk through every engram feature with synthetic data.

Usage:
    engram demo              # CLI walkthrough
    engram demo --web        # also start the web dashboard so you can watch
    engram demo --keep       # keep the demo database after (default: cleanup)
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
import uuid

# terminal colors
C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "green": "\033[32m", "cyan": "\033[36m", "yellow": "\033[33m",
    "red": "\033[31m", "magenta": "\033[35m", "blue": "\033[34m",
}


def c(text, color):
    return f"{C[color]}{text}{C['reset']}"


def header(text):
    w = 60
    print(f"\n{c('━' * w, 'dim')}")
    print(f"  {c(text, 'bold')}")
    print(f"{c('━' * w, 'dim')}\n")


def step(text):
    print(f"  {c('→', 'cyan')} {text}")


def result(text):
    print(f"    {c(text, 'green')}")


def wait(msg="Press Enter to continue..."):
    input(f"\n  {c(msg, 'dim')}")
    print()


def run_demo(keep_db=False, start_web=False, web_port=8421):
    """Run the full interactive demo."""

    # create temporary database
    tmp_dir = tempfile.mkdtemp(prefix="engram_demo_")
    db_path = os.path.join(tmp_dir, "demo.db")

    print(f"""
{c('╔══════════════════════════════════════════════════════════╗', 'magenta')}
{c('║', 'magenta')}  {c('engram', 'bold')} — interactive demo                              {c('║', 'magenta')}
{c('║', 'magenta')}                                                          {c('║', 'magenta')}
{c('║', 'magenta')}  a cognitive memory system that actually remembers things {c('║', 'magenta')}
{c('╚══════════════════════════════════════════════════════════╝', 'magenta')}

  Using temporary database: {c(db_path, 'dim')}
""")

    # suppress model loading noise
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    step("Loading embedding models...")

    # setup
    from engram.config import Config
    config = Config()
    config.db_path = db_path

    from engram.store import Store, Memory, MemoryLayer, SourceType
    from engram.embeddings import embed_documents
    from engram.surprise import compute_surprise, adjust_importance
    from engram.lifecycle import retention_l2, retention_huber, retention_elastic, compute_retention
    from engram.entities import process_entities_for_memory
    from engram.deep_retrieval import DeepReranker
    from engram.retrieval import search as hybrid_search

    store = Store(config)
    store.init_db()

    # optionally start web server in background
    web_proc = None
    if start_web:
        import subprocess
        env = os.environ.copy()
        env["ENGRAM_DB_PATH"] = db_path
        web_proc = subprocess.Popen(
            [sys.executable, "-m", "engram", "serve", "--web", "--port", str(web_port)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        print(f"  {c('Web dashboard running at', 'dim')} {c(f'http://127.0.0.1:{web_port}', 'cyan')}")
        print(f"  {c('Open it in your browser to watch the neural map light up!', 'dim')}\n")

    # ================================================================
    # PHASE 1: Basic memory storage
    # ================================================================
    header("Phase 1: Storing memories")

    memories_data = [
        ("Ari prefers short, direct responses. No filler, no preamble.", "semantic", 0.8),
        ("The deploy pipeline uses blue-green deployment with a 5-minute canary window.", "procedural", 0.8),
        ("Decision: Use SQLite instead of PostgreSQL for the config service. Rationale: single-node, no concurrent writes, eliminates ops complexity.", "procedural", 0.9),
        ("melee.garden is a competitive Melee platform with 45+ pages, AI coaching, Slippi AI Arena, and stream overlays.", "semantic", 0.7),
        ("Error: Mock database tests passed but production migration failed. Prevention: always use real database connections for integration tests.", "procedural", 0.85),
        ("2026-03-28: Built and deployed melee.garden in ~20 hours. Full platform: frame data for all 26 characters, Supabase auth, cloud sync.", "episodic", 0.6),
        ("2026-04-07: Built engram memory system. 4-stage hybrid retrieval, entity graph, dream cycle, web dashboard.", "episodic", 0.7),
        ("The AI coach in melee.garden uses Qwen 3.5 2B via MLX on Mac, with a direct personality and memory system using localStorage.", "semantic", 0.6),
        ("Kubernetes pods should have resource limits set. Without them, a single pod can starve the node.", "procedural", 0.7),
        ("React server components reduce client bundle size by keeping data-fetching logic on the server.", "semantic", 0.5),
        ("The nginx rate limiter uses leaky bucket algorithm. Config: limit_req_zone with burst and nodelay.", "procedural", 0.65),
        ("2026-04-09: Read Titans paper — surprise-based memorization where memory updates are proportional to loss gradient.", "episodic", 0.8),
    ]

    step("Storing 12 memories across episodic, semantic, and procedural layers...")
    print()

    stored_ids = []
    for content, layer, importance in memories_data:
        mem = Memory(
            id=str(uuid.uuid4()), content=content,
            source_type=SourceType.HUMAN, layer=layer, importance=importance,
        )
        emb = embed_documents([content], config.embedding_model)
        if emb.size > 0:
            mem.embedding = emb[0]

        # surprise scoring
        surprise_info = compute_surprise(mem.embedding, store)
        mem.importance = adjust_importance(mem.importance, surprise_info)
        mem.metadata["surprise"] = surprise_info["surprise"]

        store.save_memory(mem)
        process_entities_for_memory(store, mem.id, content)
        stored_ids.append(mem.id)

        surprise_bar = "█" * int(surprise_info["surprise"] * 20)
        surprise_empty = "░" * (20 - int(surprise_info["surprise"] * 20))
        surprise_color = "green" if surprise_info["surprise"] > 0.5 else "yellow" if surprise_info["surprise"] > 0.3 else "red"
        print(f"    {c(f'[{layer:10s}]', 'dim')} surprise={c(f'{surprise_bar}{surprise_empty}', surprise_color)} {surprise_info['surprise']:.2f}  imp={mem.importance:.2f}")
        print(f"             {c(content[:70] + ('...' if len(content) > 70 else ''), 'dim')}")

    stats = store.get_stats()
    print()
    result(f"Stored {stats['memories']['total']} memories, {stats['entities']} entities, {stats['relationships']} relationships")

    wait()

    # ================================================================
    # PHASE 2: Surprise-based dedup detection
    # ================================================================
    header("Phase 2: Surprise scoring — detecting redundancy")

    step("Storing a near-duplicate of the deploy pipeline memory...")
    print()

    dup_content = "Our deployment uses blue-green with canary releases lasting about 5 minutes before full cutover."
    dup_emb = embed_documents([dup_content], config.embedding_model)
    dup_surprise = compute_surprise(dup_emb[0], store)

    print(f"    Original: {c('The deploy pipeline uses blue-green deployment with a 5-minute canary window.', 'dim')}")
    print(f"    Duplicate: {c(dup_content, 'dim')}")
    print()
    print(f"    Surprise score:  {c(f'{dup_surprise['surprise']:.4f}', 'red')} {'(very low — system knows this already)' if dup_surprise['surprise'] < 0.3 else ''}")
    print(f"    Nearest distance: {c(f'{dup_surprise['nearest_distance']:.4f}', 'red')}")
    print(f"    Is duplicate:    {c(str(dup_surprise['is_duplicate']), 'red' if dup_surprise['is_duplicate'] else 'green')}")
    print(f"    Importance adj:  {c(f'{dup_surprise['importance_modifier']:+.4f}', 'red')}")

    if dup_surprise["nearest_id"]:
        nearest = store.get_memory(dup_surprise["nearest_id"])
        if nearest:
            print(f"    Nearest memory:  {c(nearest.content[:80], 'cyan')}")

    print()
    step("Now storing something genuinely novel...")
    print()

    novel_content = "Black holes emit Hawking radiation due to quantum effects near the event horizon, causing them to slowly evaporate."
    novel_emb = embed_documents([novel_content], config.embedding_model)
    novel_surprise = compute_surprise(novel_emb[0], store)

    print(f"    Content: {c(novel_content[:80], 'dim')}")
    print(f"    Surprise score:  {c(f'{novel_surprise['surprise']:.4f}', 'green')} {'(novel — nothing like this in memory)' if novel_surprise['surprise'] > 0.5 else ''}")
    print(f"    Importance adj:  {c(f'{novel_surprise['importance_modifier']:+.4f}', 'green')}")

    wait()

    # ================================================================
    # PHASE 3: Hybrid retrieval
    # ================================================================
    header("Phase 3: Hybrid retrieval — dense + BM25 + graph")

    queries = [
        "deployment strategy",
        "what happened on march 28",
        "database testing mistakes",
    ]

    for query in queries:
        step(f"Searching: {c(query, 'cyan')}")
        results = hybrid_search(query, store, config, top_k=3)
        for i, r in enumerate(results):
            print(f"    {c(f'[{i+1}]', 'dim')} score={r.score:.3f} {c(f'[{r.memory.layer}]', 'dim')} {r.memory.content[:80]}")
        print()

    wait()

    # ================================================================
    # PHASE 4: Retention curves
    # ================================================================
    header("Phase 4: Retention regularization — three decay models")

    step("Comparing L2 (exponential), Huber (robust), and Elastic (sparse) retention...")
    print()

    half_life = 30
    print(f"    {'Age (days)':>12}  {'L2':>8}  {'Huber':>8}  {'Elastic':>8}")
    print(f"    {'─' * 45}")

    for days in [0, 7, 15, 30, 45, 60, 90]:
        l2 = retention_l2(days, half_life)
        hub = retention_huber(days, half_life, 0.5)
        ela = retention_elastic(days, half_life, 0.3)

        def bar(v):
            filled = int(v * 8)
            return c("█" * filled + "░" * (8 - filled), "green" if v > 0.5 else "yellow" if v > 0.2 else "red")

        print(f"    {days:>12}  {bar(l2)} {l2:.3f}  {bar(hub)} {hub:.3f}  {bar(ela)} {ela:.3f}")

    print()
    print(f"    {c('L2:', 'cyan')} classic exponential — everything fades smoothly")
    print(f"    {c('Huber:', 'cyan')} robust to burst-then-quiet — gentler on old memories")
    print(f"    {c('Elastic:', 'cyan')} sparse — strong memories stay, weak ones drop faster")

    wait()

    # ================================================================
    # PHASE 5: Deep reranker
    # ================================================================
    header("Phase 5: Deep retrieval — learning from access patterns")

    step("Simulating access patterns (some memories get recalled more than others)...")
    print()

    # simulate access patterns
    import random
    random.seed(42)
    for _ in range(50):
        idx = random.choice(range(len(stored_ids)))
        store.record_access(stored_ids[idx], f"simulated query {random.randint(1,100)}")

    # train reranker
    reranker_path = config.resolved_db_path.parent / "reranker.npz"
    reranker = DeepReranker(model_path=reranker_path)

    step("Training the MLP reranker on access log data...")
    train_result = reranker.train(store, epochs=30)

    if train_result.get("status") == "trained":
        result(f"Trained on {train_result['samples']} samples, final loss: {train_result['final_loss']:.4f}")
    else:
        print(f"    {c(f'Not enough data: {train_result}', 'yellow')}")

    print()
    step("Searching with the deep reranker active...")
    results_with = hybrid_search("deployment", store, config, top_k=3, deep_reranker=reranker)
    for i, r in enumerate(results_with):
        deep_score = r.sources.get("deep_reranker", "n/a")
        print(f"    {c(f'[{i+1}]', 'dim')} deep={deep_score:.3f if isinstance(deep_score, float) else deep_score} {c(f'[{r.memory.layer}]', 'dim')} {r.memory.content[:70]}")

    wait()

    # ================================================================
    # PHASE 6: Entity graph
    # ================================================================
    header("Phase 6: Entity graph — automatic knowledge linking")

    step("Entities extracted automatically from memory content:")
    print()

    entities = store.list_entities(limit=20)
    type_colors = {"person": "cyan", "tool": "green", "concept": "yellow", "date": "magenta", "url": "blue", "path": "dim"}
    for e in entities[:15]:
        mem_count = store.conn.execute(
            "SELECT COUNT(*) as cnt FROM entity_mentions WHERE entity_id = ?", (e.id,)
        ).fetchone()["cnt"]
        color = type_colors.get(e.entity_type, "dim")
        print(f"    {c(e.canonical_name, color)} {c(f'({e.entity_type})', 'dim')} — {mem_count} memories")

    print()

    # show a relationship
    for e in entities[:5]:
        rels = store.get_entity_relationships(e.id)
        if rels:
            step(f"Relationships for {c(e.canonical_name, 'cyan')}:")
            for r in rels[:3]:
                print(f"      → {r['relation_type']} → {r['target_name']}")
            break

    wait()

    # ================================================================
    # PHASE 7: Cognitive scaffolding
    # ================================================================
    header("Phase 7: Cognitive scaffolding — hints vs full recall")

    step("Full recall dumps entire memory content:")
    full = hybrid_search("testing approach", store, config, top_k=2)
    for r in full:
        print(f"    {c('[FULL]', 'cyan')} {r.memory.content}")
    print()

    step("Hint mode returns just enough to trigger recognition:")
    for r in full:
        hint = r.memory.content[:60] + "..."
        entities_for = store.conn.execute(
            "SELECT e.canonical_name FROM entity_mentions em JOIN entities e ON e.id = em.entity_id WHERE em.memory_id = ? LIMIT 3",
            (r.memory.id,),
        ).fetchall()
        ent_names = [row["canonical_name"] for row in entities_for]
        print(f"    {c('[HINT]', 'yellow')} {c(hint, 'dim')}  entities: {c(', '.join(ent_names), 'cyan')}")

    print()
    print(f"    {c('The idea:', 'dim')} hints trigger recognition without replacing cognition.")
    print(f"    {c('Pull full context only when you actually need it.', 'dim')}")

    wait()

    # ================================================================
    # PHASE 8: System stats
    # ================================================================
    header("Phase 8: System overview")

    final_stats = store.get_stats()
    mems = final_stats["memories"]

    print(f"    {c('Memories:', 'bold')}       {mems['total']}")
    for layer in ["episodic", "semantic", "procedural"]:
        count = mems.get(layer, 0)
        bar = "█" * count + "░" * (12 - count)
        color = {"episodic": "cyan", "semantic": "green", "procedural": "magenta"}.get(layer, "dim")
        print(f"      {layer:12s}  {c(bar, color)} {count}")

    print(f"    {c('Entities:', 'bold')}       {final_stats['entities']}")
    print(f"    {c('Relationships:', 'bold')}  {final_stats['relationships']}")
    print(f"    {c('Database:', 'bold')}       {final_stats['db_size_mb']} MB")
    print(f"    {c('Reranker:', 'bold')}       {'trained' if reranker.is_trained else 'untrained'}")

    # ================================================================
    # Cleanup
    # ================================================================
    print()
    if keep_db:
        result(f"Demo database kept at: {db_path}")
        print(f"    Run: engram --config /dev/null status  # with ENGRAM_DB_PATH={db_path}")
    else:
        store.close()
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        result("Demo database cleaned up.")

    if web_proc:
        if keep_db:
            result(f"Web dashboard still running at http://127.0.0.1:{web_port}")
        else:
            web_proc.terminate()
            result("Web dashboard stopped.")

    print(f"""
{c('━' * 60, 'dim')}

  {c('What you just saw:', 'bold')}

  1. {c('Surprise scoring', 'cyan')} — novel memories get boosted, duplicates get flagged
  2. {c('Hybrid retrieval', 'cyan')} — dense + BM25 + graph fused with RRF
  3. {c('Retention curves', 'cyan')} — three forgetting models (L2, Huber, elastic)
  4. {c('Deep reranker', 'cyan')} — MLP trained on actual access patterns
  5. {c('Entity graph', 'cyan')} — automatic knowledge linking from text
  6. {c('Cognitive scaffolding', 'cyan')} — hints that trigger recognition without replacing it

  {c('Get started:', 'bold')}
    pip install -e .
    engram remember "your first memory"
    engram search "what do I know"
    engram serve --web

  {c('See examples/ for agent integration guides.', 'dim')}

{c('━' * 60, 'dim')}
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="Keep demo database after")
    parser.add_argument("--web", action="store_true", help="Also start web dashboard")
    parser.add_argument("--port", type=int, default=8421, help="Web port")
    args = parser.parse_args()
    run_demo(keep_db=args.keep, start_web=args.web, web_port=args.port)
