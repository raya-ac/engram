"""CLI: ingest, search, consolidate, status, serve."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

from engram.config import Config


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

    # serve
    p_serve = sub.add_parser("serve", help="Start web UI and/or MCP server")
    p_serve.add_argument("--web", action="store_true", help="Start web UI")
    p_serve.add_argument("--mcp", action="store_true", help="Start MCP server (stdio)")
    p_serve.add_argument("--port", type=int, help="Web UI port override")

    args = parser.parse_args()
    config = Config.load(args.config)

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


if __name__ == "__main__":
    main()
