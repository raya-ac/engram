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
        return {
            "results": [_result_dict(r) for r in results],
            "debug": {
                "latency_ms": round(dbg.latency_ms, 1),
                "dense_count": len(dbg.dense_candidates),
                "bm25_count": len(dbg.bm25_candidates),
                "graph_count": len(dbg.graph_candidates),
                "rrf_count": len(dbg.rrf_scores),
            },
        }
    return {"results": [_result_dict(r) for r in result]}


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

    # get recent access events (for firing animation)
    recent = store.conn.execute(
        """SELECT al.memory_id, al.accessed_at, al.query_text,
                  GROUP_CONCAT(e.id) as entity_ids
           FROM access_log al
           JOIN entity_mentions em ON em.memory_id = al.memory_id
           JOIN entities e ON e.id = em.entity_id
           WHERE al.accessed_at > ?
           GROUP BY al.id
           ORDER BY al.accessed_at DESC
           LIMIT 30""",
        (time.time() - 300,),  # last 5 minutes
    ).fetchall()
    fires = [dict(r) for r in recent]

    return {"nodes": nodes, "edges": edges, "fires": fires}


@router.get("/api/neural/fires")
async def neural_fires(request: Request, since: float = 0):
    """Lightweight poll endpoint — returns recent access events since timestamp."""
    store = _store(request)
    cutoff = since if since > 0 else time.time() - 10
    recent = store.conn.execute(
        """SELECT al.memory_id, al.accessed_at, al.query_text,
                  GROUP_CONCAT(DISTINCT e.canonical_name) as entity_names,
                  GROUP_CONCAT(DISTINCT e.id) as entity_ids
           FROM access_log al
           JOIN entity_mentions em ON em.memory_id = al.memory_id
           JOIN entities e ON e.id = em.entity_id
           WHERE al.accessed_at > ?
           GROUP BY al.id
           ORDER BY al.accessed_at DESC
           LIMIT 50""",
        (cutoff,),
    ).fetchall()
    return {"fires": [dict(r) for r in recent], "timestamp": time.time()}


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
    if emb.size > 0:
        mem.embedding = emb[0]
    hqs = []
    try:
        hqs = generate_hypothetical_queries(mem.content, config)
    except Exception:
        pass
    store.save_memory(mem, hypothetical_queries=hqs)
    process_entities_for_memory(store, mem.id, mem.content)
    push_event("remember", {"id": mem.id, "content": mem.content[:100]})
    return {"id": mem.id, "status": "stored"}


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


# --- Diary ---

_web_diary: list[dict] = []

@router.get("/api/diary")
async def get_diary(request: Request):
    return {"entries": _web_diary}


@router.post("/api/diary")
async def write_diary(request: Request):
    body = await request.json()
    entry = {"text": body.get("entry", ""), "timestamp": time.time()}
    _web_diary.append(entry)
    return {"status": "written", "count": len(_web_diary)}


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
    return {**stats, "embedding_cache_loaded": cache_loaded, "embedding_cache_size": cache_size,
            "orphaned_entities": orphaned, "stale_working": stale, "fts_indexed": fts, "no_embedding": no_emb}


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


# --- Helpers ---

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
