"""API routes and HTML endpoints."""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from engram.store import Store, Memory, MemoryLayer, SourceType
from engram.retrieval import search as hybrid_search, RetrievalResult
from engram.embeddings import embed_documents
from engram.extractor import generate_hypothetical_queries
from engram.entities import process_entities_for_memory
from engram.consolidator import consolidate
from engram.layers import get_context_layers
from engram.compress import compress_memories
from engram.web.events import event_generator, get_recent_events, push_event
from engram.surprise import compute_surprise, adjust_importance
from engram.lifecycle import retention_l2, retention_huber, retention_elastic, compute_retention
from engram.deep_retrieval import DeepReranker
from engram.skill_select import select_skills, format_skills
from engram.drift import run_drift_check, auto_fix_drift
from engram.patterns import extract_patterns_from_session, store_patterns

router = APIRouter()


def _store(request: Request) -> Store:
    return request.app.state.store


def _config(request: Request):
    return request.app.state.config


# --- HTML ---

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return request.app.state.templates.TemplateResponse(request, "index.html")


# --- SSE ---

@router.get("/api/stream")
async def stream(request: Request):
    return EventSourceResponse(event_generator())


# --- API: Read ---

@router.get("/api/memories")
async def list_memories(request: Request, layer: str | None = None,
                        limit: int = 50, offset: int = 0):
    store = _store(request)
    if layer:
        mems = store.get_memories_by_layer(layer, limit=limit)
    else:
        mems = store.get_recent_memories(limit=limit)
    return [_mem_dict(m) for m in mems]


@router.get("/api/memories/{memory_id}")
async def get_memory(request: Request, memory_id: str):
    store = _store(request)
    mem = store.get_memory(memory_id)
    if not mem:
        return JSONResponse({"error": "not found"}, status_code=404)
    hqs = store.conn.execute(
        "SELECT query_text FROM hypothetical_queries WHERE memory_id = ?", (memory_id,)
    ).fetchall()
    d = _mem_dict(mem)
    d["hypothetical_queries"] = [r["query_text"] for r in hqs]
    # access log
    accesses = store.conn.execute(
        "SELECT accessed_at, query_text FROM access_log WHERE memory_id = ? ORDER BY accessed_at DESC LIMIT 20",
        (memory_id,),
    ).fetchall()
    d["access_history"] = [dict(a) for a in accesses]
    # linked entities
    entities = store.conn.execute(
        """SELECT e.canonical_name, e.entity_type FROM entity_mentions em
        JOIN entities e ON e.id = em.entity_id WHERE em.memory_id = ?""",
        (memory_id,),
    ).fetchall()
    d["entities"] = [dict(e) for e in entities]
    return d


@router.get("/api/search")
async def search_memories(request: Request, q: str, top_k: int = 10, debug: bool = False):
    store = _store(request)
    config = _config(request)
    result = hybrid_search(q, store, config, top_k=top_k, debug=debug)

    push_event("search", {"query": q, "results": len(result) if not debug else len(result[0])})

    if debug:
        results, dbg = result
        entity_ids = _collect_entity_ids(store, [r.memory.id for r in results])
        return {
            "results": [_result_dict(r) for r in results],
            "entity_ids": entity_ids,
            "debug": {
                "latency_ms": round(dbg.latency_ms, 1),
                "dense_count": len(dbg.dense_candidates),
                "bm25_count": len(dbg.bm25_candidates),
                "graph_count": len(dbg.graph_candidates),
                "rrf_count": len(dbg.rrf_scores),
            },
        }
    results = result
    entity_ids = _collect_entity_ids(store, [r.memory.id for r in results])
    return {"results": [_result_dict(r) for r in results], "entity_ids": entity_ids}


@router.get("/api/entities")
async def list_entities(request: Request, limit: int = 100):
    store = _store(request)
    entities = store.list_entities(limit=limit)
    result = []
    for e in entities:
        mem_count = store.conn.execute(
            "SELECT COUNT(*) as cnt FROM entity_mentions WHERE entity_id = ?", (e.id,)
        ).fetchone()["cnt"]
        result.append({
            "id": e.id, "name": e.canonical_name, "type": e.entity_type,
            "aliases": e.aliases, "memory_count": mem_count,
            "first_seen": e.first_seen, "last_seen": e.last_seen,
        })
    return result


@router.get("/api/entities/{entity_id}/graph")
async def entity_graph(request: Request, entity_id: str):
    store = _store(request)
    entity = store.get_entity(entity_id)
    if not entity:
        return JSONResponse({"error": "not found"}, status_code=404)
    rels = store.get_entity_relationships(entity_id)
    related = store.get_related_entities(entity_id, max_hops=2)
    nodes = [{"id": entity.id, "name": entity.canonical_name, "type": entity.entity_type, "depth": 0}]
    for r in related:
        nodes.append({"id": r["eid"], "name": r["canonical_name"], "type": r["entity_type"], "depth": r["depth"]})
    return {"nodes": nodes, "edges": [dict(r) for r in rels]}


@router.get("/api/neural")
async def neural_graph(request: Request, limit: int = 80):
    """Full entity-relationship graph for neural visualization."""
    store = _store(request)
    # get top entities by memory count
    # layer priority: semantic > procedural > episodic > working
    # assign entity to its highest-priority layer
    rows = store.conn.execute(
        """SELECT e.id, e.canonical_name, e.entity_type,
                  COUNT(em.memory_id) as mem_count,
                  MAX(m.last_accessed) as last_active,
                  CASE
                    WHEN EXISTS(SELECT 1 FROM entity_mentions em2 JOIN memories m2 ON m2.id=em2.memory_id
                                WHERE em2.entity_id=e.id AND m2.layer='semantic' AND m2.forgotten=0) THEN 'semantic'
                    WHEN EXISTS(SELECT 1 FROM entity_mentions em2 JOIN memories m2 ON m2.id=em2.memory_id
                                WHERE em2.entity_id=e.id AND m2.layer='procedural' AND m2.forgotten=0) THEN 'procedural'
                    WHEN EXISTS(SELECT 1 FROM entity_mentions em2 JOIN memories m2 ON m2.id=em2.memory_id
                                WHERE em2.entity_id=e.id AND m2.layer='working' AND m2.forgotten=0) THEN 'working'
                    ELSE 'episodic'
                  END as dominant_layer
           FROM entities e
           JOIN entity_mentions em ON em.entity_id = e.id
           JOIN memories m ON m.id = em.memory_id AND m.forgotten = 0
           GROUP BY e.id
           ORDER BY mem_count DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()

    nodes = []
    node_ids = set()
    for r in rows:
        nodes.append({
            "id": r["id"],
            "name": r["canonical_name"],
            "type": r["entity_type"],
            "size": r["mem_count"],
            "lastActive": r["last_active"],
            "layer": r["dominant_layer"] or "episodic",
        })
        node_ids.add(r["id"])

    # get relationships between these nodes
    edges = []
    if node_ids:
        placeholders = ",".join("?" * len(node_ids))
        rels = store.conn.execute(
            f"""SELECT r.source_entity_id, r.target_entity_id,
                       r.relation_type, r.strength, r.evidence_count,
                       e1.canonical_name as source_name,
                       e2.canonical_name as target_name
                FROM relationships r
                JOIN entities e1 ON e1.id = r.source_entity_id
                JOIN entities e2 ON e2.id = r.target_entity_id
                WHERE r.source_entity_id IN ({placeholders})
                  AND r.target_entity_id IN ({placeholders})""",
            list(node_ids) + list(node_ids),
        ).fetchall()
        for r in rels:
            edges.append(dict(r))

    # get recent access events (reads + writes for firing animation)
    cutoff = time.time() - 300  # last 5 minutes
    read_fires = store.conn.execute(
        """SELECT al.memory_id, al.accessed_at as ts, al.query_text,
                  GROUP_CONCAT(e.id) as entity_ids, 'read' as fire_type
           FROM access_log al
           JOIN entity_mentions em ON em.memory_id = al.memory_id
           JOIN entities e ON e.id = em.entity_id
           WHERE al.accessed_at > ?
           GROUP BY al.id
           ORDER BY al.accessed_at DESC
           LIMIT 30""",
        (cutoff,),
    ).fetchall()
    write_fires = store.conn.execute(
        """SELECT ev.memory_id, ev.created_at as ts, ev.detail as query_text,
                  GROUP_CONCAT(e.id) as entity_ids, 'write' as fire_type
           FROM events ev
           JOIN entity_mentions em ON em.memory_id = ev.memory_id
           JOIN entities e ON e.id = em.entity_id
           WHERE ev.created_at > ?
             AND ev.event_type IN ('memory_write', 'memory_promote', 'memory_forget')
           GROUP BY ev.id
           ORDER BY ev.created_at DESC
           LIMIT 30""",
        (cutoff,),
    ).fetchall()
    fires = [dict(r) for r in read_fires] + [dict(r) for r in write_fires]
    fires.sort(key=lambda f: f.get("ts", 0), reverse=True)
    fires = fires[:30]

    return {"nodes": nodes, "edges": edges, "fires": fires}


@router.get("/api/neural/fires")
async def neural_fires(request: Request, since: float = 0):
    """Lightweight poll endpoint — returns recent read AND write events since timestamp."""
    store = _store(request)
    cutoff = since if since > 0 else time.time() - 10

    # reads from access_log
    reads = store.conn.execute(
        """SELECT al.memory_id, al.accessed_at as ts, al.query_text,
                  GROUP_CONCAT(DISTINCT e.canonical_name) as entity_names,
                  GROUP_CONCAT(DISTINCT e.id) as entity_ids,
                  'read' as fire_type
           FROM access_log al
           JOIN entity_mentions em ON em.memory_id = al.memory_id
           JOIN entities e ON e.id = em.entity_id
           WHERE al.accessed_at > ?
           GROUP BY al.id
           ORDER BY al.accessed_at DESC
           LIMIT 50""",
        (cutoff,),
    ).fetchall()

    # writes from events table
    writes = store.conn.execute(
        """SELECT ev.memory_id, ev.created_at as ts, ev.detail as query_text,
                  GROUP_CONCAT(DISTINCT e.canonical_name) as entity_names,
                  GROUP_CONCAT(DISTINCT e.id) as entity_ids,
                  'write' as fire_type
           FROM events ev
           JOIN entity_mentions em ON em.memory_id = ev.memory_id
           JOIN entities e ON e.id = em.entity_id
           WHERE ev.created_at > ?
             AND ev.event_type IN ('memory_write', 'memory_promote', 'memory_forget')
           GROUP BY ev.id
           ORDER BY ev.created_at DESC
           LIMIT 50""",
        (cutoff,),
    ).fetchall()

    # merge and sort by timestamp descending
    fires = [dict(r) for r in reads] + [dict(r) for r in writes]
    fires.sort(key=lambda f: f.get("ts", 0), reverse=True)
    return {"fires": fires[:50], "timestamp": time.time()}


@router.get("/api/timeline")
async def timeline(request: Request, start: str, end: str | None = None, limit: int = 50):
    store = _store(request)
    mems = store.get_memories_by_date_range(start, end, limit=limit)
    return [_mem_dict(m) for m in mems]


@router.get("/api/stats")
async def stats(request: Request):
    store = _store(request)
    return store.get_stats()


@router.get("/api/events")
async def events(request: Request, limit: int = 50):
    store = _store(request)
    return store.get_recent_events(limit=limit)


@router.get("/api/pulse")
async def session_pulse(request: Request):
    """Real-time session activity counters + hourly sparkline."""
    store = _store(request)
    now = time.time()
    hour_ago = now - 3600

    # count events by type in last hour
    rows = store.conn.execute(
        """SELECT event_type, COUNT(*) as cnt
           FROM events WHERE created_at > ?
           GROUP BY event_type""",
        (hour_ago,),
    ).fetchall()
    counts = {r["event_type"]: r["cnt"] for r in rows}

    # sparkline: activity per 5-min bucket over last hour (12 buckets)
    sparkline = []
    for i in range(12):
        bucket_start = hour_ago + i * 300
        bucket_end = bucket_start + 300
        cnt = store.conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE created_at >= ? AND created_at < ?",
            (bucket_start, bucket_end),
        ).fetchone()["cnt"]
        sparkline.append(cnt)

    return {
        "created": counts.get("memory_write", 0),
        "recalled": counts.get("recall", 0) + counts.get("memory_read", 0),
        "promoted": counts.get("memory_promote", 0),
        "forgotten": counts.get("memory_forget", 0),
        "total_events": sum(counts.values()),
        "sparkline": sparkline,
        "timestamp": now,
    }


@router.get("/api/heatmap")
async def activity_heatmap(request: Request, days: int = 30):
    """GitHub-style activity heatmap: day x hour bucketed event counts."""
    store = _store(request)
    cutoff = time.time() - days * 86400
    rows = store.conn.execute(
        """SELECT
             CAST((created_at - ?) / 86400 AS INT) as day_offset,
             CAST(((created_at - ?) % 86400) / 3600 AS INT) as hour,
             event_type,
             COUNT(*) as cnt
           FROM events WHERE created_at > ?
           GROUP BY day_offset, hour, event_type
           ORDER BY day_offset, hour""",
        (cutoff, cutoff, cutoff),
    ).fetchall()
    cells = [dict(r) for r in rows]
    return {"cells": cells, "days": days}


# --- API: Write ---

@router.post("/api/remember")
async def remember(request: Request):
    body = await request.json()
    store = _store(request)
    config = _config(request)

    mem = Memory(
        id=str(uuid.uuid4()),
        content=body["content"],
        source_type=body.get("source_type", SourceType.HUMAN),
        layer=body.get("layer", MemoryLayer.EPISODIC),
        importance=body.get("importance", 0.7),
    )
    emb = embed_documents([mem.content], config.embedding_model)
    surprise_info = {}
    if emb.size > 0:
        mem.embedding = emb[0]
        surprise_info = compute_surprise(mem.embedding, store)
        mem.importance = adjust_importance(mem.importance, surprise_info)
        mem.metadata["surprise"] = surprise_info["surprise"]
        if surprise_info["is_duplicate"] and surprise_info["nearest_id"]:
            mem.metadata["duplicate_of"] = surprise_info["nearest_id"]
    hqs = []
    try:
        hqs = generate_hypothetical_queries(mem.content, config)
    except Exception:
        pass
    store.save_memory(mem, hypothetical_queries=hqs)
    process_entities_for_memory(store, mem.id, mem.content)
    push_event("remember", {"id": mem.id, "content": mem.content[:100]})
    result = {"id": mem.id, "status": "stored"}
    if surprise_info:
        result["surprise"] = surprise_info["surprise"]
        result["importance"] = mem.importance
        if surprise_info["is_duplicate"]:
            result["warning"] = "near-duplicate detected"
    return result


@router.post("/api/consolidate")
async def run_consolidation(request: Request):
    store = _store(request)
    config = _config(request)
    result = consolidate(store, config)
    push_event("consolidation", result)
    return result


# --- Memory actions ---

@router.post("/api/memories/{memory_id}/promote")
async def promote_memory(request: Request, memory_id: str):
    body = await request.json()
    store = _store(request)
    target = body.get("layer", "semantic")
    store.update_layer(memory_id, target)
    return {"status": "promoted", "layer": target}


@router.post("/api/memories/{memory_id}/demote")
async def demote_memory(request: Request, memory_id: str):
    body = await request.json()
    store = _store(request)
    target = body.get("layer", "episodic")
    store.update_layer(memory_id, target)
    return {"status": "demoted", "layer": target}


@router.post("/api/memories/{memory_id}/forget")
async def forget_memory(request: Request, memory_id: str):
    store = _store(request)
    store.forget_memory(memory_id)
    return {"status": "forgotten"}


@router.post("/api/memories/{memory_id}/invalidate")
async def invalidate_memory(request: Request, memory_id: str):
    body = await request.json()
    store = _store(request)
    reason = body.get("reason", "invalidated")
    mem = store.get_memory(memory_id)
    if not mem:
        return JSONResponse({"error": "not found"}, status_code=404)
    meta = mem.metadata
    meta["invalidated"] = True
    meta["invalidation_reason"] = reason
    meta["invalidated_at"] = time.time()
    store.conn.execute("UPDATE memories SET metadata = ? WHERE id = ?",
                       (json.dumps(meta), memory_id))
    store.conn.commit()
    return {"status": "invalidated"}


# --- Importance history ---

@router.get("/api/memories/{memory_id}/history")
async def importance_history(request: Request, memory_id: str):
    store = _store(request)
    history = store.get_importance_history(memory_id)
    return {"memory_id": memory_id, "history": history}


# --- Entity management ---

@router.post("/api/entities/{entity_id}/alias")
async def add_entity_alias(request: Request, entity_id: str):
    body = await request.json()
    store = _store(request)
    entity = store.get_entity(entity_id)
    if not entity:
        return JSONResponse({"error": "not found"}, status_code=404)
    alias = body.get("alias", "")
    if alias and alias not in entity.aliases:
        aliases = entity.aliases + [alias]
        store.conn.execute("UPDATE entities SET aliases = ? WHERE id = ?",
                           (json.dumps(aliases), entity_id))
        store.conn.commit()
    return {"status": "alias_added", "aliases": entity.aliases + [alias]}


@router.post("/api/entities/{entity_id}/type")
async def change_entity_type(request: Request, entity_id: str):
    body = await request.json()
    store = _store(request)
    new_type = body.get("type", "concept")
    store.conn.execute("UPDATE entities SET entity_type = ? WHERE id = ?", (new_type, entity_id))
    store.conn.commit()
    return {"status": "type_changed", "type": new_type}


# --- Analytics ---

@router.get("/api/analytics")
async def analytics(request: Request):
    store = _store(request)

    # most accessed memories
    top_accessed = store.conn.execute(
        """SELECT id, content, access_count, layer, importance
           FROM memories WHERE forgotten = 0 AND access_count > 0
           ORDER BY access_count DESC LIMIT 15"""
    ).fetchall()

    # layer distribution
    layer_dist = store.conn.execute(
        "SELECT layer, COUNT(*) as cnt FROM memories WHERE forgotten = 0 GROUP BY layer"
    ).fetchall()

    # activity by hour (last 24h)
    hourly = store.conn.execute(
        """SELECT CAST((created_at - ?) / 3600 AS INT) as hour_ago, COUNT(*) as cnt
           FROM events WHERE created_at > ? GROUP BY hour_ago ORDER BY hour_ago""",
        (time.time() - 86400, time.time() - 86400),
    ).fetchall()

    # top entities by memory count
    top_entities = store.conn.execute(
        """SELECT e.canonical_name, e.entity_type, COUNT(em.memory_id) as cnt
           FROM entities e JOIN entity_mentions em ON em.entity_id = e.id
           GROUP BY e.id ORDER BY cnt DESC LIMIT 20"""
    ).fetchall()

    # source type distribution
    source_dist = store.conn.execute(
        "SELECT source_type, COUNT(*) as cnt FROM memories WHERE forgotten = 0 GROUP BY source_type"
    ).fetchall()

    return {
        "top_accessed": [dict(r) for r in top_accessed],
        "layer_distribution": [dict(r) for r in layer_dist],
        "hourly_activity": [dict(r) for r in hourly],
        "top_entities": [dict(r) for r in top_entities],
        "source_distribution": [dict(r) for r in source_dist],
    }


# --- Diary (persistent) ---

@router.get("/api/diary")
async def get_diary(request: Request):
    store = _store(request)
    entries = store.get_diary(limit=100)
    return {"entries": [{"text": e["text"], "timestamp": e["created_at"]} for e in entries]}


@router.post("/api/diary")
async def write_diary(request: Request):
    body = await request.json()
    store = _store(request)
    text = body.get("entry", "")
    store.write_diary(text)
    return {"status": "written"}


# --- Context layers ---

@router.get("/api/context")
async def get_context(request: Request, query: str | None = None, max_tokens: int = 4000):
    store = _store(request)
    config = _config(request)
    layers_data = get_context_layers(store, query, config, max_tokens)
    result = {}
    for k, v in layers_data.items():
        tokens = int(len(v.split()) * 1.3) if v else 0
        result[k] = {"text": v, "tokens": tokens}
    return result


# --- Ingest log ---

@router.get("/api/ingest/log")
async def ingest_log(request: Request):
    store = _store(request)
    rows = store.conn.execute(
        "SELECT file_path, file_hash, last_ingested, memory_count FROM ingest_log ORDER BY last_ingested DESC LIMIT 50"
    ).fetchall()
    return [dict(r) for r in rows]


# --- Filtered search ---

@router.get("/api/search/filtered")
async def filtered_search(request: Request, q: str, top_k: int = 10,
                           layer: str | None = None, min_importance: float = 0,
                           source_type: str | None = None,
                           date_from: str | None = None, date_to: str | None = None):
    store = _store(request)
    config = _config(request)
    results = hybrid_search(q, store, config, top_k=top_k * 3, rerank=False)

    # post-filter
    filtered = []
    for r in results:
        if layer and r.memory.layer != layer:
            continue
        if r.memory.importance < min_importance:
            continue
        if source_type and r.memory.source_type != source_type:
            continue
        if date_from and r.memory.fact_date and r.memory.fact_date < date_from:
            continue
        if date_to and r.memory.fact_date and r.memory.fact_date > date_to:
            continue
        filtered.append(r)
        if len(filtered) >= top_k:
            break

    return {"results": [_result_dict(r) for r in filtered]}


# --- Health & map ---

@router.get("/api/health")
async def health_check(request: Request):
    store = _store(request)
    stats = store.get_stats()
    cache_loaded = store._embedding_cache is not None
    cache_size = len(store._embedding_cache[0]) if cache_loaded else 0
    orphaned = store.conn.execute(
        "SELECT COUNT(*) as cnt FROM entities e WHERE NOT EXISTS (SELECT 1 FROM entity_mentions em WHERE em.entity_id = e.id)"
    ).fetchone()["cnt"]
    stale = store.conn.execute(
        "SELECT COUNT(*) as cnt FROM memories WHERE layer='working' AND forgotten=0 AND created_at < ?",
        (time.time() - 1800,),
    ).fetchone()["cnt"]
    fts = store.conn.execute("SELECT COUNT(*) as cnt FROM memories_fts").fetchone()["cnt"]
    no_emb = store.conn.execute("SELECT COUNT(*) as cnt FROM memories WHERE embedding IS NULL AND forgotten=0").fetchone()["cnt"]
    # ANN index status
    ann_ready = store.ann_index.ready if store.ann_index else False
    ann_count = store.ann_index.count if store.ann_index and store.ann_index.ready else 0

    # embedding backend info
    from engram.embeddings import get_backend, _default_model
    config = request.app.state.config
    emb_backend = get_backend(config.embedding_model)

    return {
        **stats,
        "embedding_cache_loaded": cache_loaded,
        "embedding_cache_size": cache_size,
        "orphaned_entities": orphaned,
        "stale_working": stale,
        "fts_indexed": fts,
        "no_embedding": no_emb,
        "ann_index_ready": ann_ready,
        "ann_index_count": ann_count,
        "embedding_backend": emb_backend,
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
        "reranker_model": config.cross_encoder_model,
    }


@router.get("/api/drift")
async def drift_check(request: Request,
                       check_functions: bool = False,
                       search_roots: str = Query(None)):
    store = _store(request)
    roots = search_roots.split(",") if search_roots else None
    report = run_drift_check(store, search_roots=roots, check_functions=check_functions)
    result = report.to_dict()
    error_count = sum(1 for i in report.issues if i.severity == "error")
    warn_count = sum(1 for i in report.issues if i.severity == "warning")
    result["summary"] = (
        f"Drift score: {report.score}/100 | "
        f"{error_count} errors, {warn_count} warnings | "
        f"{report.claims_valid}/{report.claims_verified} claims valid"
    )
    return result


@router.post("/api/drift/fix")
async def drift_fix(request: Request):
    store = _store(request)
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    dry_run = body.get("dry_run", False)
    report = run_drift_check(store, check_functions=False)
    result = auto_fix_drift(store, report, dry_run=dry_run)
    push_event("drift_fix", {"total": result["total_actions"], "dry_run": dry_run})
    return result


@router.get("/api/patterns")
async def get_patterns(request: Request, hours: float = 4.0):
    store = _store(request)
    config = _config(request)
    patterns = extract_patterns_from_session(store, config, hours=hours)
    return {
        "patterns": [
            {"title": p.title, "category": p.category, "novelty": p.novelty,
             "should_store": p.should_store, "source_events": p.source_events,
             "content_preview": p.content[:300]}
            for p in patterns
        ],
        "total": len(patterns),
        "would_store": sum(1 for p in patterns if p.should_store),
    }


@router.post("/api/patterns/extract")
async def extract_and_store_patterns(request: Request):
    store = _store(request)
    config = _config(request)
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    hours = body.get("hours", 4.0)
    threshold = body.get("novelty_threshold", 0.25)
    patterns = extract_patterns_from_session(store, config, hours=hours, novelty_threshold=threshold)
    result = store_patterns(patterns, store, config)
    push_event("patterns_extracted", {"stored": result["total_stored"]})
    return result


@router.get("/api/memory-map")
async def memory_map(request: Request):
    store = _store(request)
    stats = store.get_stats()
    layers_detail = {}
    for layer in ["working", "episodic", "semantic", "procedural", "codebase"]:
        top = store.conn.execute(
            """SELECT e.canonical_name, COUNT(em.memory_id) as cnt
               FROM entity_mentions em JOIN memories m ON m.id = em.memory_id
               JOIN entities e ON e.id = em.entity_id
               WHERE m.layer = ? AND m.forgotten = 0
               GROUP BY e.id ORDER BY cnt DESC LIMIT 5""", (layer,),
        ).fetchall()
        layers_detail[layer] = {"count": stats["memories"].get(layer, 0),
                                "top_entities": [dict(r) for r in top]}
    oldest = store.conn.execute("SELECT fact_date FROM memories WHERE forgotten=0 AND fact_date IS NOT NULL ORDER BY fact_date ASC LIMIT 1").fetchone()
    newest = store.conn.execute("SELECT fact_date FROM memories WHERE forgotten=0 AND fact_date IS NOT NULL ORDER BY fact_date DESC LIMIT 1").fetchone()
    return {**stats, "layers": layers_detail,
            "oldest": oldest["fact_date"] if oldest else None,
            "newest": newest["fact_date"] if newest else None}


# --- Similar & dedup ---

@router.get("/api/memories/{memory_id}/similar")
async def similar_memories(request: Request, memory_id: str, top_k: int = 5):
    store = _store(request)
    mem = store.get_memory(memory_id)
    if not mem or mem.embedding is None:
        return JSONResponse({"error": "not found or no embedding"}, status_code=404)
    from engram.embeddings import cosine_similarity_search
    ids, vecs = store.get_all_embeddings()
    hits = cosine_similarity_search(mem.embedding, vecs, top_k=top_k + 1)
    results = []
    for idx, score in hits:
        if ids[idx] == memory_id:
            continue
        other = store.get_memory(ids[idx])
        if other:
            results.append({**_mem_dict(other), "similarity": round(score, 4)})
    return results[:top_k]


@router.get("/api/duplicates")
async def find_dups(request: Request, threshold: float = 0.92, limit: int = 20):
    from engram.dedup import find_duplicates
    store = _store(request)
    dupes = find_duplicates(store, threshold=threshold, limit=500)
    return [{"memory_1": _mem_dict(m1), "memory_2": _mem_dict(m2), "similarity": round(sim, 4)}
            for m1, m2, sim in dupes[:limit]]


@router.post("/api/dedup")
async def run_dedup(request: Request):
    from engram.dedup import auto_dedup
    store = _store(request)
    return auto_dedup(store)


# --- Importance explain ---

@router.get("/api/memories/{memory_id}/importance")
async def explain_importance(request: Request, memory_id: str):
    import math
    store = _store(request)
    mem = store.get_memory(memory_id)
    if not mem:
        return JSONResponse({"error": "not found"}, status_code=404)
    age_days = (time.time() - mem.last_accessed) / 86400
    recency = math.exp(-0.693 * age_days / 30)
    access_factor = min(1.0, 0.1 * math.log(1 + mem.access_count))
    emotion = abs(mem.emotional_valence) * 0.3
    stability = min(1.0, mem.access_count / (max(1, (mem.last_accessed - mem.created_at) / 86400) + 1)) if mem.access_count > 0 else 0.0
    layer_boost = {"working": 0.0, "episodic": 0.1, "semantic": 0.3, "procedural": 0.2, "codebase": 0.15}.get(mem.layer, 0.0)
    factors = [
        {"name": "Base", "value": mem.importance, "weight": 0.30, "color": "#818cf8"},
        {"name": "Access", "value": access_factor, "weight": 0.15, "color": "#06b6d4", "detail": f"{mem.access_count} accesses"},
        {"name": "Recency", "value": recency, "weight": 0.15, "color": "#22c55e", "detail": f"{age_days:.1f} days ago"},
        {"name": "Emotion", "value": emotion, "weight": 0.10, "color": "#ef4444"},
        {"name": "Stability", "value": stability, "weight": 0.10, "color": "#eab308"},
        {"name": "Layer", "value": layer_boost, "weight": 0.20, "color": "#6366f1", "detail": mem.layer},
    ]
    composite = sum(f["value"] * f["weight"] for f in factors)
    return {"composite": round(min(1.0, max(0.0, composite)), 4), "factors": factors}


# --- Entity timeline ---

@router.get("/api/entities/{entity_id}/timeline")
async def entity_timeline_view(request: Request, entity_id: str):
    store = _store(request)
    entity = store.get_entity(entity_id)
    if not entity:
        return JSONResponse({"error": "not found"}, status_code=404)
    memories = store.get_entity_memories(entity_id, limit=100)
    def sort_key(m):
        return m.fact_date or time.strftime("%Y-%m-%d", time.localtime(m.created_at))
    memories.sort(key=sort_key)
    return {"entity": entity.canonical_name, "memories": [
        {**_mem_dict(m), "sort_date": sort_key(m)} for m in memories
    ]}


# --- Pin/unpin ---

@router.post("/api/memories/{memory_id}/pin")
async def pin_memory(request: Request, memory_id: str):
    store = _store(request)
    mem = store.get_memory(memory_id)
    if not mem:
        return JSONResponse({"error": "not found"}, status_code=404)
    mem.metadata["pinned"] = True
    store.conn.execute("UPDATE memories SET metadata = ? WHERE id = ?", (json.dumps(mem.metadata), memory_id))
    store.conn.commit()
    return {"status": "pinned"}


@router.post("/api/memories/{memory_id}/unpin")
async def unpin_memory(request: Request, memory_id: str):
    store = _store(request)
    mem = store.get_memory(memory_id)
    if not mem:
        return JSONResponse({"error": "not found"}, status_code=404)
    mem.metadata.pop("pinned", None)
    store.conn.execute("UPDATE memories SET metadata = ? WHERE id = ?", (json.dumps(mem.metadata), memory_id))
    store.conn.commit()
    return {"status": "unpinned"}


# --- Surprise ---

@router.post("/api/surprise/preview")
async def surprise_preview(request: Request):
    """Compute surprise score for text before storing."""
    body = await request.json()
    store = _store(request)
    config = _config(request)
    text = body.get("content", "")
    if not text:
        return JSONResponse({"error": "content required"}, status_code=400)
    emb = embed_documents([text], config.embedding_model)
    if emb.size == 0:
        return JSONResponse({"error": "embedding failed"}, status_code=500)
    result = compute_surprise(emb[0], store)
    # add nearest memory content snippet
    if result.get("nearest_id"):
        nearest = store.get_memory(result["nearest_id"])
        if nearest:
            result["nearest_content"] = nearest.content[:150]
    return result


@router.get("/api/surprise/{memory_id}")
async def memory_surprise(request: Request, memory_id: str):
    """Get the stored surprise score for a memory."""
    store = _store(request)
    mem = store.get_memory(memory_id)
    if not mem:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "memory_id": memory_id,
        "surprise": mem.metadata.get("surprise"),
        "duplicate_of": mem.metadata.get("duplicate_of"),
    }


# --- Retention curves ---

@router.get("/api/retention/curves")
async def retention_curves(request: Request,
                           half_life: int = 30, huber_delta: float = 0.5,
                           l1_ratio: float = 0.3, points: int = 100):
    """Generate retention curve data for visualization."""
    max_days = half_life * 3
    step = max_days / points
    curves = {"l2": [], "huber": [], "elastic": []}
    for i in range(points + 1):
        d = i * step
        curves["l2"].append({"day": round(d, 1), "retention": round(retention_l2(d, half_life), 4)})
        curves["huber"].append({"day": round(d, 1), "retention": round(retention_huber(d, half_life, huber_delta), 4)})
        curves["elastic"].append({"day": round(d, 1), "retention": round(retention_elastic(d, half_life, l1_ratio), 4)})
    config = _config(request)
    return {
        "curves": curves,
        "config": {
            "retention_mode": getattr(config.lifecycle, 'retention_mode', 'l2'),
            "half_life": config.lifecycle.forgetting_half_life_days,
            "huber_delta": getattr(config.lifecycle, 'huber_delta', 0.5),
            "l1_ratio": getattr(config.lifecycle, 'elastic_l1_ratio', 0.3),
        },
    }


@router.get("/api/retention/scatter")
async def retention_scatter(request: Request, limit: int = 200):
    """Get real memory age vs retention for scatter plot."""
    import math
    store = _store(request)
    config = _config(request)
    rows = store.conn.execute(
        "SELECT * FROM memories WHERE forgotten = 0 AND layer IN ('episodic', 'working') LIMIT ?",
        (limit,),
    ).fetchall()
    points = []
    for row in rows:
        mem = store._row_to_memory(row)
        age = (time.time() - mem.last_accessed) / 86400
        ret = compute_retention(mem, config)
        points.append({
            "id": mem.id,
            "age_days": round(age, 1),
            "retention": round(ret, 4),
            "layer": mem.layer,
            "importance": mem.importance,
            "access_count": mem.access_count,
        })
    return {"points": points}


# --- Reranker ---

@router.get("/api/reranker/status")
async def reranker_status(request: Request):
    config = _config(request)
    model_path = config.resolved_db_path.parent / "reranker.npz"
    reranker = DeepReranker(model_path=model_path)
    result = {"trained": reranker.is_trained, "model_path": str(model_path),
              "model_exists": model_path.exists()}
    if model_path.exists():
        result["model_size_kb"] = round(model_path.stat().st_size / 1024, 1)
    return result


@router.post("/api/reranker/train")
async def train_reranker(request: Request):
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    store = _store(request)
    config = _config(request)
    model_path = config.resolved_db_path.parent / "reranker.npz"
    reranker = DeepReranker(model_path=model_path)
    result = reranker.train(store, lr=body.get("learning_rate", 0.01),
                            epochs=body.get("epochs", 50))
    return result


# --- Bridges ---

@router.get("/api/bridges")
async def list_bridges(request: Request, limit: int = 20):
    store = _store(request)
    rows = store.conn.execute(
        """SELECT * FROM memories
           WHERE forgotten = 0 AND json_extract(metadata, '$.type') = 'cross_domain_bridge'
           ORDER BY created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    bridges = []
    for row in rows:
        mem = store._row_to_memory(row)
        bridges.append({
            **_mem_dict(mem),
            "entity_a": mem.metadata.get("entity_a"),
            "entity_b": mem.metadata.get("entity_b"),
            "similarity": mem.metadata.get("similarity"),
        })
    return {"bridges": bridges}


# --- Skill selection ---

@router.get("/api/skills")
async def get_skills_api(request: Request, query: str, max_skills: int = 3):
    store = _store(request)
    config = _config(request)
    selection = select_skills(query, store, config, max_skills=max_skills)
    return {
        "should_inject": selection.should_inject,
        "confidence": round(selection.confidence, 3),
        "task_novelty": round(selection.task_novelty, 3),
        "domain_coverage": round(selection.domain_coverage, 3),
        "reason": selection.reason,
        "skill_count": len(selection.skills),
        "context": format_skills(selection) if selection.should_inject else None,
        "skills": [
            {"id": m.id, "content": m.content[:200], "layer": m.layer,
             "importance": m.importance}
            for m in selection.skills
        ],
    }


# --- Edit & Annotate (web) ---

@router.post("/api/memories/{memory_id}/edit")
async def edit_memory(request: Request, memory_id: str):
    body = await request.json()
    store = _store(request)
    config = _config(request)
    mem = store.get_memory(memory_id)
    if not mem:
        return JSONResponse({"error": "not found"}, status_code=404)
    new_content = body.get("content", "")
    if not new_content:
        return JSONResponse({"error": "content required"}, status_code=400)
    emb = embed_documents([new_content], config.embedding_model)
    emb_blob = emb[0].astype('float32').tobytes() if emb.size > 0 else None
    store.conn.execute("UPDATE memories SET content = ?, embedding = ? WHERE id = ?",
                       (new_content, emb_blob, memory_id))
    row = store.conn.execute("SELECT rowid FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if row:
        store.conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (row[0],))
        store.conn.execute("INSERT INTO memories_fts (rowid, content, hypothetical_queries) VALUES (?, ?, '')",
                           (row[0], new_content))
    store.invalidate_embedding_cache()
    store._emit_event("memory_edit", memory_id=memory_id)
    store.conn.commit()
    return {"status": "edited"}


@router.post("/api/memories/{memory_id}/annotate")
async def annotate_memory(request: Request, memory_id: str):
    body = await request.json()
    store = _store(request)
    mem = store.get_memory(memory_id)
    if not mem:
        return JSONResponse({"error": "not found"}, status_code=404)
    annotations = mem.metadata.get("annotations", [])
    annotations.append({"note": body.get("note", ""), "timestamp": time.time()})
    mem.metadata["annotations"] = annotations
    store.conn.execute("UPDATE memories SET metadata = ? WHERE id = ?",
                       (json.dumps(mem.metadata), memory_id))
    store.conn.commit()
    return {"status": "annotated", "count": len(annotations)}


# --- Bulk actions ---

@router.post("/api/memories/bulk")
async def bulk_action(request: Request):
    body = await request.json()
    store = _store(request)
    action = body.get("action")
    ids = body.get("memory_ids", [])
    if not ids or not action:
        return JSONResponse({"error": "action and memory_ids required"}, status_code=400)
    affected = 0
    for mid in ids:
        if action == "forget":
            store.forget_memory(mid)
            affected += 1
        elif action == "promote":
            target = body.get("layer", "semantic")
            store.update_layer(mid, target)
            affected += 1
        elif action == "demote":
            target = body.get("layer", "episodic")
            store.update_layer(mid, target)
            affected += 1
        elif action == "tag":
            mem = store.get_memory(mid)
            if mem:
                tags = set(mem.metadata.get("tags", []))
                tags.update(body.get("tags", []))
                mem.metadata["tags"] = sorted(tags)
                store.conn.execute("UPDATE memories SET metadata = ? WHERE id = ?",
                                   (json.dumps(mem.metadata), mid))
                affected += 1
    store.conn.commit()
    store.invalidate_embedding_cache()
    return {"affected": affected, "action": action}


# --- Export ---

@router.get("/api/export")
async def export_memories(request: Request, format: str = "json",
                          layer: str | None = None, limit: int = 200):
    store = _store(request)
    if layer:
        mems = store.get_memories_by_layer(layer, limit=limit)
    else:
        mems = store.get_recent_memories(limit=limit)
    if format == "markdown":
        lines = []
        for m in mems:
            date = m.fact_date or time.strftime("%Y-%m-%d", time.localtime(m.created_at))
            lines.append(f"## [{date}] {m.layer} (imp={m.importance:.2f})\n\n{m.content}\n")
        return {"format": "markdown", "content": "\n---\n\n".join(lines), "count": len(mems)}
    else:
        return {"format": "json", "memories": [_mem_dict(m) for m in mems], "count": len(mems)}


# --- Ingest (web) ---

@router.post("/api/ingest/path")
async def ingest_path(request: Request):
    body = await request.json()
    path = body.get("path", "")
    if not path:
        return JSONResponse({"error": "path required"}, status_code=400)
    from engram.cli import cmd_ingest
    import argparse
    config = _config(request)
    try:
        fake_args = argparse.Namespace(paths=[path], jobs=1, no_queries=False)
        cmd_ingest(fake_args, config)
        return {"status": "ingested", "path": path}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/ingest/sessions")
async def ingest_sessions(request: Request):
    store = _store(request)
    from engram.conversations import ingest_all_sessions
    result = ingest_all_sessions(store, limit=20)
    return result


# --- Search hints ---

@router.get("/api/search/hints")
async def search_hints(request: Request, q: str, top_k: int = 10, hint_length: int = 60):
    store = _store(request)
    config = _config(request)
    results = hybrid_search(q, store, config, top_k=top_k)
    hints = []
    for r in results:
        mem = r.memory
        lines = [l.strip() for l in mem.content.split("\n") if l.strip()]
        title = (lines[0] if lines else mem.content)[:hint_length]
        if len(title) < len(lines[0] if lines else mem.content):
            title += "..."
        entities = store.conn.execute(
            "SELECT e.canonical_name, e.entity_type FROM entity_mentions em JOIN entities e ON e.id = em.entity_id WHERE em.memory_id = ? LIMIT 5",
            (mem.id,),
        ).fetchall()
        hints.append({
            "id": mem.id, "hint": title, "layer": mem.layer,
            "importance": round(mem.importance, 2), "date": mem.fact_date,
            "entities": [dict(e) for e in entities], "score": round(r.score, 4),
        })
    return {"hints": hints, "query": q}


# --- Helpers ---

def _collect_entity_ids(store: Store, memory_ids: list[str]) -> list[str]:
    """Get unique entity IDs mentioned in the given memories."""
    if not memory_ids:
        return []
    placeholders = ",".join("?" * len(memory_ids))
    rows = store.conn.execute(
        f"SELECT DISTINCT entity_id FROM entity_mentions WHERE memory_id IN ({placeholders})",
        memory_ids,
    ).fetchall()
    return [r["entity_id"] for r in rows]


def _mem_dict(m: Memory) -> dict:
    return {
        "id": m.id, "content": m.content, "layer": m.layer,
        "source_type": m.source_type, "source_file": m.source_file,
        "importance": m.importance, "access_count": m.access_count,
        "created_at": m.created_at, "last_accessed": m.last_accessed,
        "fact_date": m.fact_date, "emotional_valence": m.emotional_valence,
        "metadata": m.metadata,
    }


def _result_dict(r: RetrievalResult) -> dict:
    d = _mem_dict(r.memory)
    d["score"] = round(r.score, 4)
    d["sources"] = {k: round(v, 4) for k, v in r.sources.items()}
    return d
