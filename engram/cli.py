"""CLI: ingest, search, consolidate, status, serve."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

from engram.config import Config
from engram.store import _json_loads_maybe


def main():
    parser = argparse.ArgumentParser(prog="engram", description="Cognitive memory system")
    parser.add_argument("--config", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest files into memory")
    p_ingest.add_argument("paths", nargs="+", help="Files or directories to ingest")
    p_ingest.add_argument("-j", "--jobs", type=int, default=1, help="Parallel extraction jobs")
    p_ingest.add_argument("--no-queries", action="store_true", help="Skip hypothetical query generation")

    # search
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("-k", "--top-k", type=int, default=10)
    p_search.add_argument("--debug", action="store_true", help="Show retrieval debug info")
    p_search.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking (slower, better)")
    p_search.add_argument("--json", action="store_true", dest="json_output")

    # remember
    p_remember = sub.add_parser("remember", help="Store a memory directly")
    p_remember.add_argument("content", nargs="+")
    p_remember.add_argument("--source", default="remember:human")
    p_remember.add_argument("--layer", default="episodic")
    p_remember.add_argument("--importance", type=float, default=0.7)

    # entity
    p_entity = sub.add_parser("entity", help="Query entity information")
    p_entity.add_argument("name", help="Entity name")
    p_entity.add_argument("--graph", action="store_true", help="Show relationship graph")

    # consolidate
    p_consolidate = sub.add_parser("consolidate", help="Run dream cycle")

    # status
    p_status = sub.add_parser("status", help="Show system stats")

    # demo
    p_demo = sub.add_parser("demo", help="Interactive demo walkthrough")
    p_demo.add_argument("--keep", action="store_true", help="Keep demo database after")
    p_demo.add_argument("--web", action="store_true", help="Also start web dashboard")
    p_demo.add_argument("--port", type=int, default=8421, help="Web dashboard port")

    # drift
    p_drift = sub.add_parser("drift", help="Check memory drift against filesystem reality")
    p_drift.add_argument("--search-roots", nargs="*", help="Directories to search for function verification")
    p_drift.add_argument("--project-root", help="Project root for command verification")
    p_drift.add_argument("--fix", action="store_true", help="Auto-fix drift issues (invalidate dead refs, flag stale)")
    p_drift.add_argument("--dry-run", action="store_true", help="Preview fixes without applying")
    p_drift.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    p_drift.add_argument("--no-functions", action="store_true", help="Skip function/class name grep (faster)")

    # patterns
    p_patterns = sub.add_parser("patterns", help="Extract reusable patterns from recent session activity")
    p_patterns.add_argument("--hours", type=float, default=4.0, help="How far back to look (hours)")
    p_patterns.add_argument("--threshold", type=float, default=0.25, help="Minimum novelty to store")
    p_patterns.add_argument("--dry-run", action="store_true", help="Preview patterns without storing")

    # index
    p_index = sub.add_parser("index", help="Manage ANN index")
    p_index.add_argument("action", choices=["rebuild", "status"], help="Action to perform")

    # reembed
    p_reembed = sub.add_parser("reembed", help="Re-embed all memories (use after switching embedding model)")
    p_reembed.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding")
    p_reembed.add_argument("--dry-run", action="store_true", help="Count memories without re-embedding")

    # watch
    p_watch = sub.add_parser("watch", help="Watch a directory for new files and auto-ingest")
    p_watch.add_argument("path", help="Directory to watch")
    p_watch.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")

    # export / import
    p_export = sub.add_parser("export", help="Export memories to portable format")
    p_export.add_argument("output", help="Output file path (.json or .jsonl)")
    p_export.add_argument("--layer", help="Filter by layer")
    p_export.add_argument("--include-embeddings", action="store_true", help="Include embedding vectors")

    p_import = sub.add_parser("import", help="Import memories from exported file")
    p_import.add_argument("input", help="Input file path (.json or .jsonl)")
    p_import.add_argument("--skip-duplicates", action="store_true", help="Skip memories with matching chunk_hash")

    # postgres migration
    p_pg = sub.add_parser("migrate-postgres", help="Copy an existing SQLite Engram store into PostgreSQL")
    p_pg.add_argument("--dsn", help="Postgres DSN; defaults to postgres_dsn from config")
    p_pg.add_argument("--from-sqlite", help="SQLite database path; defaults to current config db_path")
    p_pg.add_argument("--switch-config", action="store_true", help="Update config.yaml to use Postgres after a successful migration")
    p_pg.add_argument("--force-reset", action="store_true", help="Wipe target Postgres tables before importing")
    p_pg.add_argument("--verify-only", action="store_true", help="Check connectivity + compare counts without copying data")

    # serve
    p_serve = sub.add_parser("serve", help="Start web UI and/or MCP server")
    p_serve.add_argument("--web", action="store_true", help="Start web UI")
    p_serve.add_argument("--mcp", action="store_true", help="Start MCP server (stdio)")
    p_serve.add_argument("--mcp-sse", action="store_true", help="Start MCP server (HTTP/SSE transport)")
    p_serve.add_argument("--port", type=int, help="Port override")

    args = parser.parse_args()
    config = Config.load(args.config)

    # set embedding backend + default model from config
    from engram.embeddings import set_backend, set_default_model
    if config.embedding_backend and config.embedding_backend != "auto":
        set_backend(config.embedding_backend)
    set_default_model(config.embedding_model)

    if args.command == "ingest":
        cmd_ingest(args, config)
    elif args.command == "search":
        cmd_search(args, config)
    elif args.command == "remember":
        cmd_remember(args, config)
    elif args.command == "entity":
        cmd_entity(args, config)
    elif args.command == "consolidate":
        cmd_consolidate(args, config)
    elif args.command == "status":
        cmd_status(args, config)
    elif args.command == "drift":
        cmd_drift(args, config)
    elif args.command == "patterns":
        cmd_patterns(args, config)
    elif args.command == "index":
        cmd_index(args, config)
    elif args.command == "reembed":
        cmd_reembed(args, config)
    elif args.command == "watch":
        cmd_watch(args, config)
    elif args.command == "export":
        cmd_export(args, config)
    elif args.command == "import":
        cmd_import(args, config)
    elif args.command == "migrate-postgres":
        cmd_migrate_postgres(args, config)
    elif args.command == "demo":
        from engram.demo import run_demo
        run_demo(keep_db=args.keep, start_web=args.web, web_port=args.port)
    elif args.command == "serve":
        cmd_serve(args, config)
    else:
        parser.print_help()


def cmd_ingest(args, config: Config):
    from engram.store import Store, Memory, SourceType
    from engram.extractor import extract_facts, facts_to_memories, generate_hypothetical_queries
    from engram.embeddings import embed_documents
    from engram.entities import process_entities_for_memory
    from engram.formats import parse_file, group_exchanges, detect_format

    store = Store(config)
    store.init_db()

    files = []
    for p in args.paths:
        path = Path(p).expanduser()
        if path.is_dir():
            for ext in ("*.md", "*.txt", "*.json", "*.jsonl", "*.pdf"):
                files.extend(path.rglob(ext))
        elif path.exists():
            files.append(path)

    # sort smallest first for early checkpoints
    files.sort(key=lambda f: f.stat().st_size)

    total = len(files)
    ingested = 0
    memories_created = 0

    for i, fpath in enumerate(files):
        file_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
        existing_hash = store.get_file_hash(str(fpath))
        if existing_hash == file_hash:
            print(f"  [{i+1}/{total}] skip (unchanged): {fpath.name}")
            continue

        print(f"  [{i+1}/{total}] ingesting: {fpath.name}")

        # parse file into exchanges
        exchanges = parse_file(fpath)
        if not exchanges:
            continue

        # group into chunks
        fmt = detect_format(fpath)
        if fmt in ("claude_code", "claude_ai", "chatgpt", "slack"):
            chunks = group_exchanges(exchanges)
        else:
            chunks = [ex["content"] for ex in exchanges]

        file_memories = 0
        for chunk in chunks:
            # extract facts via LLM
            facts = extract_facts(chunk, source_file=str(fpath), config=config)
            mem_pairs = facts_to_memories(facts, source_file=str(fpath))

            if not mem_pairs:
                # fallback: store chunk directly
                mem = Memory(
                    id=str(uuid.uuid4()),
                    content=chunk[:2000],
                    source_file=str(fpath),
                    source_type=SourceType.INGEST,
                )
                mem_pairs = [(mem, [])]

            # batch embed
            contents = [m.content for m, _ in mem_pairs]
            embeddings = embed_documents(contents, config.embedding_model)

            for j, (mem, _) in enumerate(mem_pairs):
                if j < len(embeddings):
                    mem.embedding = embeddings[j]

                # generate hypothetical queries
                hqs = []
                if not args.no_queries:
                    try:
                        hqs = generate_hypothetical_queries(mem.content, config)
                    except Exception:
                        pass

                store.save_memory(mem, hypothetical_queries=hqs)

                # process entities
                try:
                    process_entities_for_memory(store, mem.id, mem.content)
                except Exception:
                    pass

                file_memories += 1

        store.set_file_hash(str(fpath), file_hash, file_memories)
        ingested += 1
        memories_created += file_memories
        print(f"    → {file_memories} memories extracted")

    print(f"\nDone: {ingested} files ingested, {memories_created} memories created")
    store.close()


def cmd_search(args, config: Config):
    from engram.store import Store
    from engram.retrieval import search

    store = Store(config)
    store.init_db()

    query = " ".join(args.query)
    results = search(query, store, config, top_k=args.top_k, debug=args.debug, rerank=args.rerank)

    if args.debug:
        results, debug = results
        print(f"\n[Debug] Latency: {debug.latency_ms:.0f}ms")
        print(f"[Debug] Dense candidates: {len(debug.dense_candidates)}")
        print(f"[Debug] BM25 candidates: {len(debug.bm25_candidates)}")
        print(f"[Debug] Graph candidates: {len(debug.graph_candidates)}")
        print(f"[Debug] RRF merged: {len(debug.rrf_scores)}")
        print()

    if not results:
        print("No memories found.")
        store.close()
        return

    if args.json_output:
        out = []
        for r in results:
            out.append({
                "id": r.memory.id,
                "content": r.memory.content,
                "score": round(r.score, 4),
                "layer": r.memory.layer,
                "importance": r.memory.importance,
                "fact_date": r.memory.fact_date,
                "sources": {k: round(v, 4) for k, v in r.sources.items()},
            })
        print(json.dumps(out, indent=2))
    else:
        for i, r in enumerate(results):
            print(f"\n{'='*60}")
            print(f"[{i+1}] score={r.score:.3f} layer={r.memory.layer} importance={r.memory.importance:.2f}")
            if r.memory.fact_date:
                print(f"    date={r.memory.fact_date}")
            print(f"    {r.memory.content[:200]}")

    store.close()


def cmd_remember(args, config: Config):
    from engram.store import Store, Memory
    from engram.embeddings import embed_documents
    from engram.extractor import generate_hypothetical_queries
    from engram.entities import process_entities_for_memory

    store = Store(config)
    store.init_db()

    content = " ".join(args.content)
    mem = Memory(
        id=str(uuid.uuid4()),
        content=content,
        source_type=args.source,
        layer=args.layer,
        importance=args.importance,
    )

    emb = embed_documents([content], config.embedding_model)
    if emb.size > 0:
        mem.embedding = emb[0]

    hqs = []
    try:
        hqs = generate_hypothetical_queries(content, config)
    except Exception:
        pass

    store.save_memory(mem, hypothetical_queries=hqs)
    process_entities_for_memory(store, mem.id, content)

    print(f"Remembered: {mem.id}")
    print(f"  layer={mem.layer} importance={mem.importance}")
    if hqs:
        print(f"  hypothetical queries: {hqs}")
    store.close()


def cmd_entity(args, config: Config):
    from engram.store import Store

    store = Store(config)
    store.init_db()

    entity = store.find_entity_by_name(args.name)
    if not entity:
        print(f"Entity '{args.name}' not found.")
        store.close()
        return

    print(f"\n{entity.canonical_name} ({entity.entity_type})")
    print(f"  aliases: {entity.aliases}")
    print(f"  first seen: {time.strftime('%Y-%m-%d', time.localtime(entity.first_seen))}")
    print(f"  last seen: {time.strftime('%Y-%m-%d', time.localtime(entity.last_seen))}")

    memories = store.get_entity_memories(entity.id, limit=20)
    print(f"\n  Memories ({len(memories)}):")
    for m in memories:
        print(f"    [{m.fact_date or '?'}] {m.content[:100]}")

    if args.graph:
        rels = store.get_entity_relationships(entity.id)
        print(f"\n  Relationships ({len(rels)}):")
        for r in rels:
            direction = "→" if r["source_entity_id"] == entity.id else "←"
            other = r["target_name"] if r["source_entity_id"] == entity.id else r["source_name"]
            print(f"    {direction} {r['relation_type']} {other} (strength={r['strength']:.2f})")

        related = store.get_related_entities(entity.id, max_hops=2)
        print(f"\n  Related entities (2 hops):")
        for r in related:
            print(f"    {'  ' * r['depth']}{r['canonical_name']} ({r['entity_type']})")

    store.close()


def cmd_consolidate(args, config: Config):
    from engram.store import Store
    from engram.consolidator import consolidate

    store = Store(config)
    store.init_db()

    print("Running dream cycle...")
    stats = consolidate(store, config)
    print(f"  Clusters found: {stats['clusters_found']}")
    print(f"  Memories merged: {stats['memories_merged']}")
    print(f"  Peer cards generated: {stats['peer_cards_generated']}")
    print(f"  Forgotten: {stats['forgotten']}")
    print(f"  Promoted: {stats['promoted']}")
    store.close()


def cmd_status(args, config: Config):
    from engram.store import Store

    store = Store(config)
    store.init_db()

    stats = store.get_stats()
    print(f"\nEngram Memory System")
    print(f"{'='*40}")
    if config.normalized_storage_backend == "postgres":
        print(f"Database: postgres")
        print(f"DSN: {config.postgres_dsn}")
    else:
        print(f"Database: {config.resolved_db_path}")
    print(f"Size: {stats['db_size_mb']} MB")
    print(f"\nMemories:")
    for layer, count in stats["memories"].items():
        print(f"  {layer}: {count}")
    print(f"\nEntities: {stats['entities']}")
    print(f"Relationships: {stats['relationships']}")

    # ANN index status
    index_path = config.ann.resolved_index_path
    if index_path.exists():
        size_mb = index_path.stat().st_size / (1024 * 1024)
        print(f"\nANN Index: {index_path}")
        print(f"  Size: {size_mb:.1f} MB")
        meta_path = index_path.with_suffix(".meta.json")
        if meta_path.exists():
            import json as _json
            meta = _json.loads(meta_path.read_text())
            print(f"  Vectors: {meta.get('count', '?')}")
            saved_at = meta.get("saved_at")
            if saved_at:
                import datetime
                dt = datetime.datetime.fromtimestamp(saved_at)
                print(f"  Last built: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"\nANN Index: not built (run 'engram index rebuild')")

    store.close()


def cmd_index(args, config: Config):
    from engram.store import Store

    store = Store(config)
    store.init_db()

    if args.action == "rebuild":
        print("Rebuilding ANN index...")
        store.init_ann_index(background=False)
        if store.ann_index and store.ann_index.ready:
            print(f"  Indexed {store.ann_index.count} vectors")
            print(f"  Saved to {config.ann.resolved_index_path}")
        else:
            print("  Failed — is hnswlib installed?")
    elif args.action == "status":
        index_path = config.ann.resolved_index_path
        if index_path.exists():
            meta_path = index_path.with_suffix(".meta.json")
            if meta_path.exists():
                import json as _json
                meta = _json.loads(meta_path.read_text())
                size_mb = index_path.stat().st_size / (1024 * 1024)
                print(f"ANN Index: {index_path}")
                print(f"  Vectors: {meta.get('count', '?')}")
                print(f"  Size: {size_mb:.1f} MB")
                print(f"  Next label: {meta.get('next_label', '?')}")
            else:
                print(f"ANN Index file exists but metadata missing")
        else:
            print("ANN Index: not built")
            print(f"  Run: engram index rebuild")

    store.close()


def cmd_serve(args, config: Config):
    if args.mcp:
        from engram.mcp_server import run_mcp
        run_mcp(config)
    elif getattr(args, 'mcp_sse', False):
        from engram.mcp_server import run_mcp_sse
        port = args.port or 8421
        run_mcp_sse(config, port=port)
    elif args.web:
        from engram.web.app import create_app
        import uvicorn
        app = create_app(config)
        port = args.port or config.web.port
        print(f"Starting Engram web UI on http://{config.web.host}:{port}")
        uvicorn.run(app, host=config.web.host, port=port)
    else:
        # default: both web
        from engram.web.app import create_app
        import uvicorn
        app = create_app(config)
        port = args.port or config.web.port
        print(f"Starting Engram web UI on http://{config.web.host}:{port}")
        uvicorn.run(app, host=config.web.host, port=port)


def cmd_drift(args, config: Config):
    from engram.store import Store
    from engram.drift import run_drift_check, auto_fix_drift

    store = Store(config)
    store.init_db()

    report = run_drift_check(
        store,
        search_roots=args.search_roots,
        project_root=args.project_root,
        check_functions=not args.no_functions,
    )

    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        # colored summary
        score = report.score
        color = "\033[32m" if score >= 80 else "\033[33m" if score >= 50 else "\033[31m"
        reset = "\033[0m"

        print(f"\n{color}Drift score: {score}/100{reset}")
        print(f"  Memories checked: {report.memories_checked}")
        print(f"  Claims extracted: {report.claims_extracted}")
        print(f"  Claims verified:  {report.claims_verified} ({report.claims_valid} valid)")
        print(f"  Stale memories:   {report.stale_memories}")

        if report.issues:
            print(f"\n  Issues ({len(report.issues)}):")
            for issue in report.issues:
                sev_color = {"error": "\033[31m", "warning": "\033[33m", "info": "\033[36m"}.get(issue.severity, "")
                print(f"    {sev_color}[{issue.severity.upper()}]{reset} {issue.code}: {issue.message}")
                print(f"           Memory: {issue.memory_preview[:80]}...")
        else:
            print("\n  No issues found.")

    if args.fix:
        dry_run = args.dry_run
        result = auto_fix_drift(store, report, dry_run=dry_run)
        if dry_run:
            print(f"\n  [DRY RUN] Would fix {result['total_actions']} issues:")
        else:
            print(f"\n  Fixed {result['total_actions']} issues:")
        print(f"    Invalidated: {result['invalidated']}")
        print(f"    Flagged stale: {result['flagged_stale']}")
        print(f"    Forgotten: {result['forgotten']}")

    store.close()


def cmd_patterns(args, config: Config):
    from engram.store import Store
    from engram.patterns import extract_patterns_from_session, store_patterns

    store = Store(config)
    store.init_db()

    patterns = extract_patterns_from_session(
        store, config,
        hours=args.hours,
        novelty_threshold=args.threshold,
    )

    if not patterns:
        print("No patterns found in recent session activity.")
        store.close()
        return

    print(f"\nFound {len(patterns)} potential patterns:\n")
    for i, p in enumerate(patterns, 1):
        novel_color = "\033[32m" if p.should_store else "\033[33m"
        reset = "\033[0m"
        status = "STORE" if p.should_store else "SKIP"
        print(f"  {i}. [{p.category}] {p.title}")
        print(f"     {novel_color}Novelty: {p.novelty:.2f} → {status}{reset} | Sources: {p.source_events}")

    if args.dry_run:
        would_store = sum(1 for p in patterns if p.should_store)
        print(f"\n  [DRY RUN] Would store {would_store}/{len(patterns)} patterns.")
    else:
        result = store_patterns(patterns, store, config)
        print(f"\n  Stored {result['total_stored']} patterns, skipped {result['total_skipped']}.")
        for s in result["stored"]:
            print(f"    + {s['title']} (novelty={s['novelty']:.2f}, importance={s['importance']:.2f})")

    store.close()


def cmd_reembed(args, config: Config):
    from engram.store import Store
    from engram.embeddings import embed_documents

    store = Store(config)
    store.init_db()

    # count memories with embeddings
    total = store.conn.execute(
        "SELECT COUNT(*) as cnt FROM memories WHERE forgotten = 0"
    ).fetchone()["cnt"]
    with_emb = store.conn.execute(
        "SELECT COUNT(*) as cnt FROM memories WHERE forgotten = 0 AND embedding IS NOT NULL"
    ).fetchone()["cnt"]

    print(f"Re-embedding with model: {config.embedding_model} (dim={config.embedding_dim})")
    print(f"Memories: {total} total, {with_emb} with embeddings")

    if args.dry_run:
        print(f"[DRY RUN] Would re-embed {total} memories in batches of {args.batch_size}")
        store.close()
        return

    # fetch all memories
    rows = store.conn.execute(
        "SELECT id, content FROM memories WHERE forgotten = 0 ORDER BY created_at"
    ).fetchall()

    batch_size = args.batch_size
    updated = 0

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        ids = [r["id"] for r in batch]
        texts = [r["content"] for r in batch]

        vecs = embed_documents(texts, config.embedding_model)

        for mid, vec in zip(ids, vecs):
            blob = vec.astype(__import__("numpy").float32).tobytes()
            store.conn.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?", (blob, mid)
            )
        store.conn.commit()
        updated += len(batch)
        print(f"  [{updated}/{len(rows)}] embedded")

    store._embedding_cache = None

    # rebuild ANN index
    print("Rebuilding ANN index...")
    store.init_ann_index(background=False)
    if store.ann_index and store.ann_index.ready:
        print(f"  ANN index: {store.ann_index.count} vectors")

    print(f"Done. Re-embedded {updated} memories.")
    store.close()


def cmd_watch(args, config: Config):
    import signal
    from engram.store import Store

    path = Path(args.path).expanduser().resolve()
    if not path.is_dir():
        print(f"Error: {path} is not a directory")
        sys.exit(1)

    store = Store(config)
    store.init_db()

    interval = args.interval
    print(f"Watching {path} every {interval}s for new files...")
    print("Press Ctrl+C to stop.\n")

    def _shutdown(sig, frame):
        print("\nStopping watcher.")
        store.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    import hashlib
    from engram.extractor import extract_facts, facts_to_memories, generate_hypothetical_queries
    from engram.embeddings import embed_documents
    from engram.entities import process_entities_for_memory
    from engram.formats import parse_file, group_exchanges, detect_format

    while True:
        files = []
        for ext in ("*.md", "*.txt", "*.json", "*.jsonl", "*.pdf"):
            files.extend(path.rglob(ext))

        new_files = 0
        for fpath in sorted(files, key=lambda f: f.stat().st_mtime):
            file_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
            existing = store.get_file_hash(str(fpath))
            if existing == file_hash:
                continue

            new_files += 1
            print(f"  ingesting: {fpath.name}")

            exchanges = parse_file(fpath)
            if not exchanges:
                continue

            fmt = detect_format(fpath)
            groups = group_exchanges(exchanges, fmt)

            from engram.store import Memory, SourceType
            for group in groups:
                content = "\n".join(f"{e.role}: {e.content}" for e in group)
                mem = Memory(
                    id=str(uuid.uuid4()),
                    content=content,
                    source_file=str(fpath),
                    source_type=SourceType.INGEST,
                )
                emb = embed_documents([content], config.embedding_model)
                mem.embedding = emb[0]

                store.save_memory(mem)
                process_entities_for_memory(mem, store)

            store.set_file_hash(str(fpath), file_hash)

        if new_files:
            print(f"  → ingested {new_files} new files")

        time.sleep(interval)


def cmd_export(args, config: Config):
    import base64
    from engram.store import Store

    store = Store(config)
    store.init_db()

    query = "SELECT * FROM memories WHERE forgotten = 0"
    params = []
    if args.layer:
        query += " AND layer = ?"
        params.append(args.layer)
    query += " ORDER BY created_at"

    rows = store.conn.execute(query, params).fetchall()

    output_path = Path(args.output)
    memories = []

    for row in rows:
        mem = {
            "id": row["id"],
            "content": row["content"],
            "source_file": row["source_file"],
            "source_type": row["source_type"],
            "layer": row["layer"],
            "importance": row["importance"],
            "access_count": row["access_count"],
            "created_at": row["created_at"],
            "last_accessed": row["last_accessed"],
            "fact_date": row["fact_date"],
            "fact_date_end": row["fact_date_end"],
            "emotional_valence": row["emotional_valence"],
            "chunk_hash": row["chunk_hash"],
            "metadata": _json_loads_maybe(row["metadata"], {}),
        }
        if args.include_embeddings and row["embedding"]:
            mem["embedding_b64"] = base64.b64encode(row["embedding"]).decode()
            mem["embedding_dim"] = config.embedding_dim
            mem["embedding_model"] = config.embedding_model
        memories.append(mem)

    # also export entities and relationships
    entities = [dict(r) for r in store.conn.execute("SELECT * FROM entities").fetchall()]
    for e in entities:
        e["aliases"] = _json_loads_maybe(e["aliases"], [])
        e["metadata"] = _json_loads_maybe(e["metadata"], {})

    relationships = [dict(r) for r in store.conn.execute("SELECT * FROM relationships").fetchall()]

    entity_mentions = [dict(r) for r in store.conn.execute("SELECT * FROM entity_mentions").fetchall()]

    export_data = {
        "version": "1.0",
        "exported_at": time.time(),
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
        "memories": memories,
        "entities": entities,
        "relationships": relationships,
        "entity_mentions": entity_mentions,
    }

    if str(output_path).endswith(".jsonl"):
        with open(output_path, "w") as f:
            for mem in memories:
                f.write(json.dumps(mem) + "\n")
    else:
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported {len(memories)} memories, {len(entities)} entities, {len(relationships)} relationships")
    print(f"  → {output_path} ({size_mb:.1f} MB)")
    if not args.include_embeddings:
        print(f"  (embeddings excluded — use --include-embeddings to include)")
    store.close()


def cmd_import(args, config: Config):
    import base64
    import numpy as np
    from engram.store import Store, Memory

    store = Store(config)
    store.init_db()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    if str(input_path).endswith(".jsonl"):
        with open(input_path) as f:
            memories_data = [json.loads(line) for line in f if line.strip()]
        entities_data = []
        relationships_data = []
        mentions_data = []
    else:
        with open(input_path) as f:
            export_data = json.load(f)
        memories_data = export_data.get("memories", [])
        entities_data = export_data.get("entities", [])
        relationships_data = export_data.get("relationships", [])
        mentions_data = export_data.get("entity_mentions", [])

        if "embedding_model" in export_data:
            exp_model = export_data["embedding_model"]
            if exp_model != config.embedding_model:
                print(f"Warning: exported with model '{exp_model}', current model is '{config.embedding_model}'")
                print(f"  Embeddings may be incompatible. Run 'engram reembed' after import.")

    imported = 0
    skipped = 0

    for mem_data in memories_data:
        if args.skip_duplicates:
            existing = store.conn.execute(
                "SELECT id FROM memories WHERE chunk_hash = ?",
                (mem_data.get("chunk_hash", ""),)
            ).fetchone()
            if existing:
                skipped += 1
                continue

        mem = Memory(
            id=mem_data["id"],
            content=mem_data["content"],
            source_file=mem_data.get("source_file"),
            source_type=mem_data.get("source_type", "ingest"),
            layer=mem_data.get("layer", "episodic"),
            importance=mem_data.get("importance", 0.5),
            access_count=mem_data.get("access_count", 0),
            created_at=mem_data.get("created_at", time.time()),
            last_accessed=mem_data.get("last_accessed", time.time()),
            fact_date=mem_data.get("fact_date"),
            fact_date_end=mem_data.get("fact_date_end"),
            emotional_valence=mem_data.get("emotional_valence", 0.0),
            chunk_hash=mem_data.get("chunk_hash", ""),
            metadata=mem_data.get("metadata", {}),
        )

        # restore embedding if present and compatible
        if "embedding_b64" in mem_data:
            emb_bytes = base64.b64decode(mem_data["embedding_b64"])
            mem.embedding = np.frombuffer(emb_bytes, dtype=np.float32).copy()
        else:
            # embed with current model
            from engram.embeddings import embed_documents
            emb = embed_documents([mem.content], config.embedding_model)
            mem.embedding = emb[0]

        store.save_memory(mem)
        imported += 1

        if imported % 100 == 0:
            print(f"  [{imported}] imported...")

    # import entities
    from engram.store import Entity, Relationship
    for e_data in entities_data:
        entity = Entity(
            id=e_data["id"],
            canonical_name=e_data["canonical_name"],
            aliases=e_data.get("aliases", []),
            entity_type=e_data.get("entity_type", "concept"),
            first_seen=e_data.get("first_seen", 0),
            last_seen=e_data.get("last_seen", 0),
            metadata=e_data.get("metadata", {}),
        )
        store.save_entity(entity)

    # import relationships
    for r_data in relationships_data:
        rel = Relationship(
            source_entity_id=r_data["source_entity_id"],
            target_entity_id=r_data["target_entity_id"],
            relation_type=r_data["relation_type"],
            strength=r_data.get("strength", 1.0),
            evidence_count=r_data.get("evidence_count", 1),
            created_at=r_data.get("created_at", 0),
            last_seen=r_data.get("last_seen", 0),
        )
        store.save_relationship(rel)

    # import entity mentions
    for m_data in mentions_data:
        store.link_entity_memory(m_data["entity_id"], m_data["memory_id"])

    print(f"\nImported {imported} memories, {len(entities_data)} entities, {len(relationships_data)} relationships")
    if skipped:
        print(f"  Skipped {skipped} duplicates")

    # rebuild ANN index
    store.init_ann_index(background=False)
    if store.ann_index and store.ann_index.ready:
        print(f"  ANN index rebuilt: {store.ann_index.count} vectors")

    store.close()


def cmd_migrate_postgres(args, config: Config):
    import yaml
    from engram.store import Store

    sqlite_path = str(Path(args.from_sqlite or config.db_path).expanduser())
    postgres_dsn = args.dsn or config.postgres_dsn
    if not postgres_dsn:
        print("Error: provide --dsn or set postgres_dsn in config.")
        sys.exit(1)

    sqlite_cfg = copy.deepcopy(config)
    sqlite_cfg.storage_backend = "sqlite"
    sqlite_cfg.db_path = sqlite_path

    pg_cfg = copy.deepcopy(config)
    pg_cfg.storage_backend = "postgres"
    pg_cfg.postgres_dsn = postgres_dsn

    src = Store(sqlite_cfg)
    dst = Store(pg_cfg)
    src.init_db()
    dst.init_db()

    source_counts = {
        "memories": src.conn.execute("SELECT COUNT(*) AS cnt FROM memories").fetchone()["cnt"],
        "entities": src.conn.execute("SELECT COUNT(*) AS cnt FROM entities").fetchone()["cnt"],
        "relationships": src.conn.execute("SELECT COUNT(*) AS cnt FROM relationships").fetchone()["cnt"],
        "entity_mentions": src.conn.execute("SELECT COUNT(*) AS cnt FROM entity_mentions").fetchone()["cnt"],
        "diary_entries": src.conn.execute("SELECT COUNT(*) AS cnt FROM diary_entries").fetchone()["cnt"],
        "session_handoffs": src.conn.execute("SELECT COUNT(*) AS cnt FROM session_handoffs").fetchone()["cnt"],
    }

    print("Migration plan:")
    print(f"  source sqlite: {sqlite_path}")
    print(f"  target postgres: {postgres_dsn}")
    print(f"  source counts: {source_counts}")

    if args.verify_only:
        target_counts = {
            "memories": dst.conn.execute("SELECT COUNT(*) AS cnt FROM memories").fetchone()["cnt"],
            "entities": dst.conn.execute("SELECT COUNT(*) AS cnt FROM entities").fetchone()["cnt"],
            "relationships": dst.conn.execute("SELECT COUNT(*) AS cnt FROM relationships").fetchone()["cnt"],
            "entity_mentions": dst.conn.execute("SELECT COUNT(*) AS cnt FROM entity_mentions").fetchone()["cnt"],
            "diary_entries": dst.conn.execute("SELECT COUNT(*) AS cnt FROM diary_entries").fetchone()["cnt"],
            "session_handoffs": dst.conn.execute("SELECT COUNT(*) AS cnt FROM session_handoffs").fetchone()["cnt"],
        }
        print(f"  target counts: {target_counts}")
        src.close()
        dst.close()
        return

    if args.force_reset:
        for table in [
            "memories_fts", "hypothetical_queries", "entity_mentions", "relationships",
            "access_log", "events", "diary_entries", "session_handoffs",
            "importance_history", "status_history", "ingest_log", "entities", "memories",
        ]:
            dst.conn.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
        dst.conn.commit()
        print("  target reset: done")

    print("Copying rows...")

    memories = src.conn.execute("SELECT * FROM memories ORDER BY created_at").fetchall()
    entities = src.conn.execute("SELECT * FROM entities").fetchall()
    relationships = src.conn.execute("SELECT * FROM relationships").fetchall()
    entity_mentions = src.conn.execute("SELECT * FROM entity_mentions").fetchall()
    access_log = src.conn.execute("SELECT * FROM access_log").fetchall()
    ingest_log = src.conn.execute("SELECT * FROM ingest_log").fetchall()
    events = src.conn.execute("SELECT * FROM events").fetchall()
    diary_entries = src.conn.execute("SELECT * FROM diary_entries").fetchall()
    session_handoffs = src.conn.execute("SELECT * FROM session_handoffs").fetchall()
    importance_history = src.conn.execute("SELECT * FROM importance_history").fetchall()
    status_history = src.conn.execute("SELECT * FROM status_history").fetchall()
    hqs = src.conn.execute("SELECT * FROM hypothetical_queries").fetchall()

    for row in memories:
        dst.conn.execute(
            """INSERT INTO memories
               (id, content, source_file, source_type, layer, memory_type, status, embedding,
                importance, access_count, created_at, last_accessed, fact_date, fact_date_end,
                emotional_valence, chunk_hash, metadata, forgotten, previous_memory_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (id) DO NOTHING""",
            (
                row["id"], row["content"], row["source_file"], row["source_type"], row["layer"],
                row["memory_type"] if "memory_type" in row.keys() else "narrative",
                row["status"] if "status" in row.keys() else "active",
                row["embedding"], row["importance"], row["access_count"], row["created_at"],
                row["last_accessed"], row["fact_date"], row["fact_date_end"],
                row["emotional_valence"], row["chunk_hash"], row["metadata"],
                row["forgotten"], row["previous_memory_id"] if "previous_memory_id" in row.keys() else None,
            ),
        )
    dst.conn.commit()

    hq_map: dict[str, list[str]] = {}
    for row in hqs:
        dst.conn.execute(
            "INSERT INTO hypothetical_queries (memory_id, query_text) VALUES (?, ?)",
            (row["memory_id"], row["query_text"]),
        )
        hq_map.setdefault(row["memory_id"], []).append(row["query_text"])
    dst.conn.commit()

    for row in memories:
        dst.refresh_fts_entry(row["id"], row["content"], " ".join(hq_map.get(row["id"], [])))
    dst.conn.commit()

    valid_memory_ids = {row["id"] for row in memories}
    valid_entity_ids = {row["id"] for row in entities}
    skipped_counts = {
        "entity_mentions": 0,
        "relationships": 0,
        "importance_history": 0,
        "status_history": 0,
    }

    for row in entities:
        dst.conn.execute(
            """INSERT INTO entities (id, canonical_name, aliases, entity_type, first_seen, last_seen, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (id) DO NOTHING""",
            (row["id"], row["canonical_name"], row["aliases"], row["entity_type"], row["first_seen"], row["last_seen"], row["metadata"]),
        )
    for row in relationships:
        if row["source_entity_id"] not in valid_entity_ids or row["target_entity_id"] not in valid_entity_ids:
            skipped_counts["relationships"] += 1
            continue
        dst.conn.execute(
            """INSERT INTO relationships
               (source_entity_id, target_entity_id, relation_type, strength, evidence_count,
                created_at, last_seen, valid_from, valid_to, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (source_entity_id, target_entity_id, relation_type) DO NOTHING""",
            (
                row["source_entity_id"], row["target_entity_id"], row["relation_type"], row["strength"],
                row["evidence_count"], row["created_at"], row["last_seen"],
                row["valid_from"] if "valid_from" in row.keys() else None,
                row["valid_to"] if "valid_to" in row.keys() else None,
                row["embedding"] if "embedding" in row.keys() else None,
            ),
        )
    for row in entity_mentions:
        if row["entity_id"] not in valid_entity_ids or row["memory_id"] not in valid_memory_ids:
            skipped_counts["entity_mentions"] += 1
            continue
        dst.conn.execute(
            "INSERT INTO entity_mentions (entity_id, memory_id) VALUES (?, ?) ON CONFLICT DO NOTHING",
            (row["entity_id"], row["memory_id"]),
        )
    for row in access_log:
        dst.conn.execute(
            "INSERT INTO access_log (memory_id, accessed_at, query_text) VALUES (?, ?, ?)",
            (row["memory_id"], row["accessed_at"], row["query_text"]),
        )
    for row in ingest_log:
        dst.conn.execute(
            """INSERT INTO ingest_log (file_path, file_hash, last_ingested, memory_count)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (file_path) DO NOTHING""",
            (row["file_path"], row["file_hash"], row["last_ingested"], row["memory_count"]),
        )
    for row in events:
        dst.conn.execute(
            "INSERT INTO events (event_type, memory_id, entity_id, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (row["event_type"], row["memory_id"], row["entity_id"], row["detail"], row["created_at"]),
        )
    for row in diary_entries:
        dst.conn.execute(
            "INSERT INTO diary_entries (text, session_id, created_at) VALUES (?, ?, ?)",
            (row["text"], row["session_id"], row["created_at"]),
        )
    for row in session_handoffs:
        dst.conn.execute(
            """INSERT INTO session_handoffs (session_id, summary, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (session_id) DO NOTHING""",
            (row["session_id"], row["summary"], row["metadata"], row["created_at"], row["updated_at"]),
        )
    for row in importance_history:
        if row["memory_id"] not in valid_memory_ids:
            skipped_counts["importance_history"] += 1
            continue
        dst.conn.execute(
            "INSERT INTO importance_history (memory_id, score, recorded_at) VALUES (?, ?, ?)",
            (row["memory_id"], row["score"], row["recorded_at"]),
        )
    for row in status_history:
        if row["memory_id"] not in valid_memory_ids:
            skipped_counts["status_history"] += 1
            continue
        dst.conn.execute(
            "INSERT INTO status_history (memory_id, old_status, new_status, reason, changed_at) VALUES (?, ?, ?, ?, ?)",
            (row["memory_id"], row["old_status"], row["new_status"], row["reason"], row["changed_at"]),
        )
    dst.conn.commit()

    migrated_counts = {
        "memories": dst.conn.execute("SELECT COUNT(*) AS cnt FROM memories").fetchone()["cnt"],
        "entities": dst.conn.execute("SELECT COUNT(*) AS cnt FROM entities").fetchone()["cnt"],
        "relationships": dst.conn.execute("SELECT COUNT(*) AS cnt FROM relationships").fetchone()["cnt"],
        "entity_mentions": dst.conn.execute("SELECT COUNT(*) AS cnt FROM entity_mentions").fetchone()["cnt"],
        "diary_entries": dst.conn.execute("SELECT COUNT(*) AS cnt FROM diary_entries").fetchone()["cnt"],
        "session_handoffs": dst.conn.execute("SELECT COUNT(*) AS cnt FROM session_handoffs").fetchone()["cnt"],
    }
    print(f"  target counts after copy: {migrated_counts}")

    expected_counts = dict(source_counts)
    for key, skipped in skipped_counts.items():
        if key in expected_counts:
            expected_counts[key] = max(0, expected_counts[key] - skipped)

    mismatches = {
        key: (expected_counts[key], migrated_counts[key])
        for key in expected_counts
        if expected_counts[key] != migrated_counts[key]
    }
    if mismatches:
        print(f"Error: migration verification failed: {mismatches}")
        src.close()
        dst.close()
        sys.exit(1)

    print("  verification: passed")
    skipped_counts = {key: value for key, value in skipped_counts.items() if value}
    if skipped_counts:
        print(f"  skipped stale references: {skipped_counts}")

    if args.switch_config:
        config_path = Path(args.config).expanduser() if args.config else (Path.home() / ".config" / "engram" / "config.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if config_path.exists():
            with open(config_path) as f:
                existing = yaml.safe_load(f) or {}
            backup = config_path.with_suffix(config_path.suffix + ".bak")
            backup.write_text(config_path.read_text())
            print(f"  backed up config to {backup}")
        existing["storage_backend"] = "postgres"
        existing["postgres_dsn"] = postgres_dsn
        with open(config_path, "w") as f:
            yaml.safe_dump(existing, f, sort_keys=False)
        print(f"  updated config at {config_path}")
        print("  next step: restart any running engram web or MCP processes so they reconnect to Postgres")
    else:
        print("  next step: rerun with --switch-config when you're ready to flip the default backend")

    src.close()
    dst.close()


if __name__ == "__main__":
    main()
