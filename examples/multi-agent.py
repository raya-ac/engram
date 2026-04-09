"""Multi-agent shared memory experiment.

Spins up 3 specialized agents sharing the same engram database.
Each agent has a different domain focus. They work in rounds,
storing domain-specific knowledge and recalling across domains.

Watch the neural map light up as agents build a shared knowledge
graph — the cross-domain synthesis feature will fire naturally
when agents' memories bridge unexpected connections.

Usage:
    python examples/multi-agent.py                    # run experiment
    python examples/multi-agent.py --web              # also start dashboard
    python examples/multi-agent.py --rounds 10        # more rounds
    python examples/multi-agent.py --db /path/to.db   # use existing db
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import uuid
import random

# add parent to path for engram imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engram.config import Config
from engram.store import Store, Memory, MemoryLayer, SourceType
from engram.embeddings import embed_documents
from engram.surprise import compute_surprise, adjust_importance
from engram.entities import process_entities_for_memory
from engram.retrieval import search as hybrid_search
from engram.deep_retrieval import DeepReranker
from engram.consolidator import consolidate

# suppress model loading noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# terminal colors
C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "green": "\033[32m", "cyan": "\033[36m", "yellow": "\033[33m",
    "red": "\033[31m", "magenta": "\033[35m", "blue": "\033[34m",
    "white": "\033[37m",
}


def c(text, color):
    return f"{C[color]}{text}{C['reset']}"


# --- Agent Definitions ---

AGENTS = {
    "code": {
        "name": "CodeBot",
        "color": "cyan",
        "emoji": "⚙",
        "description": "Systems engineer focused on infrastructure, deployment, and code architecture",
        "knowledge": [
            "PostgreSQL connection pooling with PgBouncer reduces connection overhead by 10x. Set pool_mode=transaction for short queries.",
            "Blue-green deployments eliminate downtime during releases. The inactive environment serves as instant rollback.",
            "Git bisect automates binary search across commits to find the exact commit that introduced a bug.",
            "Docker multi-stage builds reduce image size by separating build dependencies from runtime. A Go service went from 1.2GB to 12MB.",
            "Rate limiting with token bucket algorithm allows burst traffic while maintaining average throughput limits.",
            "Database migrations should always be backward compatible. Deploy new code first, then migrate, then remove old columns.",
            "Connection timeouts should be set at every layer: application (30s), load balancer (60s), database (15s). Mismatched timeouts cause cascading failures.",
            "Prometheus + Grafana for metrics: USE method (Utilization, Saturation, Errors) for resources, RED method (Rate, Errors, Duration) for services.",
            "Kubernetes horizontal pod autoscaler uses CPU/memory metrics by default, but custom metrics (queue depth, request latency p99) are more useful for autoscaling.",
            "SQLite WAL mode allows concurrent readers with a single writer. Good enough for most single-node applications.",
        ],
        "queries": [
            "deployment strategy",
            "database performance",
            "monitoring and observability",
            "error handling patterns",
            "security best practices",
        ],
    },
    "research": {
        "name": "ResearchBot",
        "color": "magenta",
        "emoji": "🔬",
        "description": "AI/ML researcher focused on papers, experiments, and cognitive systems",
        "knowledge": [
            "Titans paper (Behrouz et al., 2025): memory updates proportional to surprise — the gradient of the loss function. Three branches: Core (attention), Contextual (neural LTM), Persistent (task knowledge).",
            "Retrieval-augmented generation (RAG) works best with hybrid search: BM25 for exact keyword matches, dense embeddings for semantic similarity, fused with reciprocal rank fusion.",
            "Chain-of-thought prompting improves reasoning by 20-40% on math and logic tasks. Works even with small models when combined with self-consistency sampling.",
            "Mixture of Experts (MoE) models route each token to the top-k experts. Sparse activation means only 20% of parameters fire per forward pass, but total parameter count is 8x larger.",
            "Neural scaling laws (Chinchilla): optimal training uses roughly equal compute for data and model parameters. Most models are overtrained on too little data.",
            "RLHF aligns model outputs with human preferences but can cause reward hacking — the model learns to game the reward model rather than genuinely satisfying the intent.",
            "Retrieval systems should compute cosine similarity on L2-normalized embeddings — dot product equals cosine when vectors are unit length, saving a division per comparison.",
            "Cross-encoder rerankers (like ms-marco-MiniLM) jointly encode query and document, scoring relevance with O(n*q) complexity. More accurate than bi-encoders but too slow for first-stage retrieval.",
            "Memory consolidation in biological brains happens during sleep — hippocampus replays experiences, strengthening important connections and pruning weak ones. Engram's dream cycle mimics this.",
            "Surprise-based learning: the brain allocates more attention and memory resources to unexpected events. Shannon information theory: surprise = -log(P(event)).",
        ],
        "queries": [
            "how does memory consolidation work",
            "retrieval augmented generation",
            "surprise and learning",
            "neural architecture scaling",
            "model training efficiency",
        ],
    },
    "ops": {
        "name": "OpsBot",
        "color": "yellow",
        "emoji": "🛡",
        "description": "Security and operations specialist focused on incidents, hardening, and monitoring",
        "knowledge": [
            "nginx rate limiting: limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s; with burst=20 nodelay allows short bursts while blocking sustained abuse.",
            "OWASP Top 10 (2025): broken access control is #1, not injection. Most vulnerabilities come from missing authorization checks, not SQL injection.",
            "TLS 1.3 reduces handshake from 2 round-trips to 1 (or 0 with PSK). Removes RSA key exchange, HMAC-based PRF, and all CBC cipher suites.",
            "Incident response: detect → triage → contain → eradicate → recover → lessons. The most common failure is skipping containment to rush to fix.",
            "Log aggregation with structured JSON logging: every log line should have timestamp, request_id, user_id, action, result, and duration_ms. grep is not a log analysis strategy.",
            "Secret rotation: API keys should rotate every 90 days. Use HashiCorp Vault or AWS Secrets Manager for automatic rotation. Never commit secrets to git.",
            "Container security: run as non-root, use read-only filesystem, drop all capabilities then add back only what's needed. seccomp profiles block dangerous syscalls.",
            "DNS rebinding attacks bypass same-origin policy by manipulating DNS TTLs. Mitigation: validate Host header, use HTTPS-only cookies, pin DNS responses.",
            "Zero-trust networking: never trust the network. Every request must be authenticated, authorized, and encrypted regardless of source. BeyondCorp is Google's implementation.",
            "Chaos engineering: regularly inject failures (latency, packet loss, pod kills) in staging to find weaknesses before they hit production. Netflix's Chaos Monkey started this.",
        ],
        "queries": [
            "rate limiting configuration",
            "security vulnerabilities",
            "incident response process",
            "container hardening",
            "monitoring alerts",
        ],
    },
}


def run_experiment(db_path: str | None = None, rounds: int = 5,
                   start_web: bool = False, web_port: int = 8422):
    """Run the multi-agent shared memory experiment."""

    # setup
    if db_path:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="engram_multi_")
        db_path = os.path.join(tmp_dir, "shared.db")

    config = Config()
    config.db_path = db_path
    store = Store(config)
    store.init_db()

    reranker_path = config.resolved_db_path.parent / "reranker.npz"
    reranker = DeepReranker(model_path=reranker_path)

    # optionally start web dashboard
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
        print(f"  {c(f'Dashboard: http://127.0.0.1:{web_port}', 'cyan')}")
        print(f"  {c('Open it to watch the neural map light up as agents work.', 'dim')}\n")

    print(f"""
{c('╔══════════════════════════════════════════════════════════╗', 'blue')}
{c('║', 'blue')}  {c('Multi-Agent Shared Memory Experiment', 'bold')}                  {c('║', 'blue')}
{c('║', 'blue')}                                                          {c('║', 'blue')}
{c('║', 'blue')}  3 agents, 1 shared engram database, {rounds} rounds              {c('║', 'blue')}
{c('╚══════════════════════════════════════════════════════════╝', 'blue')}
""")

    for name, agent in AGENTS.items():
        print(f"  {agent['emoji']}  {c(agent['name'], agent['color'])} — {agent['description']}")
    print()

    print(f"  {c('Loading embedding models...', 'dim')}")
    # warm up embeddings
    embed_documents(["warmup"], config.embedding_model)
    print(f"  {c('Ready.', 'green')}\n")

    # --- ROUNDS ---
    all_ids = {name: [] for name in AGENTS}

    for round_num in range(1, rounds + 1):
        print(f"\n{c(f'━━━ Round {round_num}/{rounds} ━━━', 'bold')}\n")

        for agent_name, agent in AGENTS.items():
            color = agent["color"]
            emoji = agent["emoji"]

            # Phase 1: STORE — each agent stores 2 random memories per round
            memories_to_store = random.sample(agent["knowledge"], min(2, len(agent["knowledge"])))
            for content in memories_to_store:
                # remove from pool so we don't repeat
                agent["knowledge"].remove(content)

                mem = Memory(
                    id=str(uuid.uuid4()), content=content,
                    source_type=f"agent:{agent_name}",
                    layer=MemoryLayer.SEMANTIC if random.random() > 0.4 else MemoryLayer.PROCEDURAL,
                    importance=0.6 + random.random() * 0.3,
                )
                emb = embed_documents([content], config.embedding_model)
                if emb.size > 0:
                    mem.embedding = emb[0]
                    surprise_info = compute_surprise(mem.embedding, store)
                    mem.importance = adjust_importance(mem.importance, surprise_info)
                    mem.metadata["surprise"] = surprise_info["surprise"]
                    mem.metadata["agent"] = agent_name

                store.save_memory(mem)
                process_entities_for_memory(store, mem.id, content)
                all_ids[agent_name].append(mem.id)

                surprise = mem.metadata.get("surprise", 0)
                s_color = "green" if surprise > 0.5 else "yellow" if surprise > 0.3 else "red"
                print(f"  {emoji} {c(agent['name'], color)} {c('stores:', 'dim')} {content[:60]}...")
                print(f"     {c(f'surprise={surprise:.2f}', s_color)} imp={mem.importance:.2f}")

            # Phase 2: RECALL — each agent searches for something (possibly cross-domain)
            query = random.choice(agent["queries"])
            results = hybrid_search(query, store, config, top_k=3,
                                    deep_reranker=reranker if reranker.is_trained else None)

            if results:
                print(f"  {emoji} {c(agent['name'], color)} {c('recalls:', 'dim')} \"{query}\"")
                for r in results[:2]:
                    source_agent = r.memory.metadata.get("agent", "?")
                    cross = source_agent != agent_name
                    cross_badge = c(" [cross-domain]", "magenta") if cross else ""
                    print(f"     → [{r.memory.layer}] from {source_agent}{cross_badge}: {r.memory.content[:50]}...")

        print()

    # --- POST-EXPERIMENT ANALYSIS ---
    print(f"\n{c('━━━ Experiment Complete ━━━', 'bold')}\n")

    stats = store.get_stats()
    print(f"  {c('Final state:', 'bold')}")
    print(f"    Memories:      {stats['memories']['total']}")
    print(f"    Entities:      {stats['entities']}")
    print(f"    Relationships: {stats['relationships']}")

    # per-agent breakdown
    print(f"\n  {c('Per-agent contribution:', 'bold')}")
    for name, agent in AGENTS.items():
        count = len(all_ids[name])
        print(f"    {agent['emoji']} {c(agent['name'], agent['color'])}: {count} memories stored")

    # cross-domain recall analysis
    print(f"\n  {c('Cross-domain recall analysis:', 'bold')}")
    cross_count = 0
    total_recalls = 0
    for agent_name, agent in AGENTS.items():
        for query in agent["queries"][:2]:
            results = hybrid_search(query, store, config, top_k=3)
            for r in results:
                total_recalls += 1
                source = r.memory.metadata.get("agent", "?")
                if source != agent_name:
                    cross_count += 1

    if total_recalls > 0:
        cross_pct = cross_count / total_recalls * 100
        print(f"    {cross_count}/{total_recalls} recalls hit another agent's memories ({cross_pct:.0f}%)")
        if cross_pct > 30:
            print(f"    {c('High cross-pollination — agents are learning from each other.', 'green')}")
        elif cross_pct > 10:
            print(f"    {c('Moderate cross-domain overlap — some shared knowledge.', 'yellow')}")
        else:
            print(f"    {c('Low overlap — agents are mostly in their own domains.', 'dim')}")

    # train reranker on the accumulated data
    print(f"\n  {c('Training deep reranker on multi-agent access patterns...', 'dim')}")
    train_result = reranker.train(store, epochs=30)
    if train_result.get("status") == "trained":
        print(f"    {c(f'Trained on {train_result[\"samples\"]} samples, loss={train_result[\"final_loss\"]:.4f}', 'green')}")
    else:
        print(f"    {c(f'Insufficient data: {train_result.get(\"samples\", 0)} samples', 'yellow')}")

    # run dream cycle to find cross-domain bridges
    print(f"\n  {c('Running dream cycle for cross-domain synthesis...', 'dim')}")
    dream_stats = consolidate(store, config)
    bridges = dream_stats.get("cross_domain_bridges", 0)
    if bridges > 0:
        print(f"    {c(f'{bridges} cross-domain bridges discovered!', 'magenta')}")
        # show them
        bridge_rows = store.conn.execute(
            "SELECT content FROM memories WHERE forgotten=0 AND json_extract(metadata, '$.type') = 'cross_domain_bridge' LIMIT 5"
        ).fetchall()
        for row in bridge_rows:
            print(f"    → {row['content'][:120]}")
    else:
        print(f"    {c('No bridges yet — needs more diverse data or lower similarity threshold.', 'dim')}")

    # entity overlap
    print(f"\n  {c('Shared entities (mentioned by 2+ agents):', 'bold')}")
    shared = store.conn.execute("""
        SELECT e.canonical_name, e.entity_type,
               COUNT(DISTINCT json_extract(m.metadata, '$.agent')) as agent_count,
               COUNT(em.memory_id) as mem_count
        FROM entity_mentions em
        JOIN entities e ON e.id = em.entity_id
        JOIN memories m ON m.id = em.memory_id
        WHERE m.forgotten = 0 AND json_extract(m.metadata, '$.agent') IS NOT NULL
        GROUP BY e.id
        HAVING agent_count >= 2
        ORDER BY agent_count DESC, mem_count DESC
        LIMIT 10
    """).fetchall()

    if shared:
        for row in shared:
            print(f"    {c(row['canonical_name'], 'cyan')} ({row['entity_type']}) — {row['agent_count']} agents, {row['mem_count']} memories")
    else:
        print(f"    {c('No shared entities yet.', 'dim')}")

    print(f"""
{c('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'dim')}

  {c('What just happened:', 'bold')}

  Three specialized agents stored domain-specific knowledge into a
  shared engram database. When one agent searched, it could find
  relevant memories from ANY agent — the hybrid retrieval doesn't
  care who stored the memory, just how relevant it is.

  The surprise scoring prevented redundant memories from inflating
  importance. The deep reranker learned which memories were actually
  useful across agents. And the dream cycle tried to bridge
  connections between unrelated domains.

  This is memory as infrastructure — not a feature bolted onto
  one agent, but a shared cognitive layer for all of them.

  Database: {c(db_path, 'dim')}
{c('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'dim')}
""")

    if web_proc:
        print(f"  Dashboard still running at http://127.0.0.1:{web_port}")
        print(f"  Press Ctrl+C to stop.\n")
        try:
            web_proc.wait()
        except KeyboardInterrupt:
            web_proc.terminate()
    else:
        store.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-agent shared memory experiment")
    parser.add_argument("--web", action="store_true", help="Start dashboard to watch")
    parser.add_argument("--port", type=int, default=8422, help="Dashboard port")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--db", type=str, help="Path to database (default: temp)")
    args = parser.parse_args()
    run_experiment(db_path=args.db, rounds=args.rounds,
                   start_web=args.web, web_port=args.port)
