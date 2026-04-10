"""5-stage hybrid retrieval pipeline: intent → dense + BM25 + graph → RRF → temporal boost → cross-encoder.

Intent-aware routing (MAGMA): classify query intent and dynamically weight signals.
Retrieval threshold (ACT-R): gate results by minimum score, don't always return top-k.
"""

from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass, field

import numpy as np

from engram.config import Config
from engram.embeddings import embed_query, cosine_similarity_search, cross_encoder_rerank
from engram.hopfield import hopfield_retrieve
from engram.store import Store, Memory


# --- Intent classification (MAGMA-inspired) ---

INTENT_PATTERNS = {
    "why": re.compile(r'\b(why|because|reason|cause|led to|resulted in)\b', re.I),
    "when": re.compile(r'\b(when|date|time|before|after|during|timeline|history)\b', re.I),
    "who": re.compile(r'\b(who|person|people|team|built|created|wrote)\b', re.I),
    "how": re.compile(r'\b(how to|steps|procedure|process|workflow|debug|fix)\b', re.I),
}

# intent → signal weight adjustments
INTENT_WEIGHTS = {
    "why":   {"dense": 1.0, "bm25": 0.8, "graph": 1.5},  # boost graph for causal
    "when":  {"dense": 0.8, "bm25": 1.2, "graph": 0.8},  # boost BM25 for date matching
    "who":   {"dense": 0.8, "bm25": 0.8, "graph": 1.8},  # boost graph for entity lookup
    "how":   {"dense": 1.2, "bm25": 1.0, "graph": 0.8},  # boost dense for procedural
    "what":  {"dense": 1.0, "bm25": 1.0, "graph": 1.0},  # default balanced
}

def classify_intent(query: str) -> str:
    """Classify query intent for adaptive retrieval weighting."""
    scores = {}
    for intent, pattern in INTENT_PATTERNS.items():
        scores[intent] = len(pattern.findall(query))
    if max(scores.values()) == 0:
        return "what"
    return max(scores, key=scores.get)


# retrieval noise scale (ACT-R inspired) — small logistic noise for beneficial variation
RETRIEVAL_NOISE_SCALE = 0.02


@dataclass
class RetrievalResult:
    memory: Memory
    score: float
    sources: dict[str, float] = field(default_factory=dict)  # signal -> score


@dataclass
class RetrievalDebug:
    query: str
    dense_candidates: list[tuple[str, float]]
    bm25_candidates: list[tuple[str, float]]
    graph_candidates: list[tuple[str, float]]
    rrf_scores: list[tuple[str, float]]
    boosted_scores: list[tuple[str, float]]
    reranked: list[tuple[str, float]]
    final_results: list[RetrievalResult]
    latency_ms: float


def search(query: str, store: Store, config: Config | None = None,
           top_k: int | None = None, debug: bool = False,
           rerank: bool = True,
           deep_reranker=None) -> list[RetrievalResult] | tuple[list[RetrievalResult], RetrievalDebug]:
    if config is None:
        config = Config.load()

    rc = config.retrieval
    k = top_k or rc.top_k
    t0 = time.time()

    # --- Stage 0: Intent classification (MAGMA-inspired) ---
    intent = classify_intent(query)
    weights = INTENT_WEIGHTS.get(intent, INTENT_WEIGHTS["what"])

    # --- Stage 1: Parallel candidate generation (4 channels) ---
    dense_candidates = _dense_search(query, store, config, k * rc.dense_multiplier)
    bm25_candidates = _bm25_search(query, store, k * rc.bm25_multiplier)
    graph_candidates = _graph_search(query, store, k)
    hopfield_candidates = _hopfield_search(query, store, config, k)

    # --- Stage 2: RRF fusion with intent-weighted signals ---
    rrf_scores = _rrf_fuse(
        [dense_candidates, bm25_candidates, graph_candidates, hopfield_candidates],
        k=rc.rrf_k,
        signal_weights=[weights["dense"], weights["bm25"], weights["graph"],
                        weights.get("hopfield", 0.6)],
    )
    rrf_top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:rc.rerank_candidates]

    # --- Stage 3: Temporal + importance boosting ---
    boosted = _apply_boosts(query, rrf_top, store, config)

    # --- Stage 4: Cross-encoder reranking (optional) ---
    candidate_ids = [mid for mid, _ in boosted[:rc.rerank_candidates]]
    candidate_memories = {}
    for mid in candidate_ids:
        mem = store.get_memory(mid)
        if mem:
            candidate_memories[mid] = mem

    if not candidate_memories:
        results = []
    elif rerank:
        docs = [candidate_memories[mid].content for mid in candidate_ids if mid in candidate_memories]
        valid_ids = [mid for mid in candidate_ids if mid in candidate_memories]
        reranked = cross_encoder_rerank(query, docs, config.cross_encoder_model)

        results = []
        for idx, ce_score in reranked[:k]:
            mid = valid_ids[idx]
            mem = candidate_memories[mid]
            sources = {
                "dense": dict(dense_candidates).get(mid, 0),
                "bm25": dict(bm25_candidates).get(mid, 0),
                "graph": dict(graph_candidates).get(mid, 0),
                "rrf": rrf_scores.get(mid, 0),
                "cross_encoder": ce_score,
            }
            results.append(RetrievalResult(memory=mem, score=ce_score, sources=sources))
    else:
        # skip cross-encoder, use boosted RRF scores directly
        results = []
        for mid, score in boosted[:k]:
            if mid in candidate_memories:
                mem = candidate_memories[mid]
                sources = {
                    "dense": dict(dense_candidates).get(mid, 0),
                    "bm25": dict(bm25_candidates).get(mid, 0),
                    "graph": dict(graph_candidates).get(mid, 0),
                    "rrf": rrf_scores.get(mid, 0),
                }
                results.append(RetrievalResult(memory=mem, score=score, sources=sources))

    # --- Optional: deep reranker pass ---
    if deep_reranker and deep_reranker.is_trained and results:
        query_vec = embed_query(query, config.embedding_model)
        candidates = []
        emb_map = {}
        for r in results:
            c = {
                "id": r.memory.id,
                "score": r.score,
                "importance": r.memory.importance,
                "access_count": r.memory.access_count,
                "created_at": r.memory.created_at,
                "layer": r.memory.layer,
            }
            candidates.append(c)
            if r.memory.embedding is not None:
                emb_map[r.memory.id] = r.memory.embedding

        reranked_candidates = deep_reranker.rerank(candidates, query_vec, emb_map)

        # rebuild results in new order
        mem_map = {r.memory.id: r for r in results}
        new_results = []
        for c in reranked_candidates[:k]:
            r = mem_map[c["id"]]
            r.sources["deep_reranker"] = c.get("deep_score", 0)
            new_results.append(r)
        results = new_results

    # --- Retrieval noise (ACT-R inspired) — beneficial variation ---
    if results and RETRIEVAL_NOISE_SCALE > 0:
        for r in results:
            noise = random.gauss(0, RETRIEVAL_NOISE_SCALE)
            r.score = max(0, r.score + noise)
        results.sort(key=lambda r: r.score, reverse=True)

    # --- Retrieval threshold gate (ACT-R) — don't return garbage ---
    min_threshold = getattr(rc, 'min_confidence', 0.0)
    if min_threshold > 0 and results:
        results = [r for r in results if r.score >= min_threshold]

    # record access — one event per search, not per result
    if results:
        result_ids = [r.memory.id for r in results]
        store.record_search(result_ids, query)

    latency = (time.time() - t0) * 1000

    if debug:
        dbg = RetrievalDebug(
            query=query,
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
            graph_candidates=graph_candidates,
            rrf_scores=list(rrf_scores.items()),
            boosted_scores=boosted,
            reranked=[(valid_ids[i], s) for i, s in reranked[:k]] if candidate_memories else [],
            final_results=results,
            latency_ms=latency,
        )
        return results, dbg

    return results


def _dense_search(query: str, store: Store, config: Config, limit: int) -> list[tuple[str, float]]:
    query_vec = embed_query(query, config.embedding_model)
    ids, vecs = store.get_all_embeddings()
    if not ids:
        return []
    hits = cosine_similarity_search(query_vec, vecs, top_k=limit)
    return [(ids[idx], score) for idx, score in hits]


def _bm25_search(query: str, store: Store, limit: int) -> list[tuple[str, float]]:
    results = store.search_fts(query, limit=limit)
    # FTS5 bm25() returns negative scores (lower = better), normalize
    if not results:
        return []
    # convert to positive scores where higher = better
    return [(mid, -score) for mid, score in results]


def _graph_search(query: str, store: Store, limit: int) -> list[tuple[str, float]]:
    """Entity graph search with multi-hop BFS (Graphiti/MAGMA-inspired).

    1. Extract entity names from query
    2. Get direct memories (hop 0, score 1.0)
    3. Traverse 1-hop related entities (score 0.5)
    4. Deduplicate, return weighted candidates
    """
    words = query.split()
    candidates = []
    matched_entity_ids = set()

    # find matching entities from query words
    for word in words:
        if len(word) >= 2:
            entity = store.find_entity_by_name(word)
            if entity:
                matched_entity_ids.add(entity.id)
                # hop 0: direct entity memories
                memories = store.get_entity_memories(entity.id, limit=limit)
                for mem in memories:
                    candidates.append((mem.id, 1.0))

    # hop 1: memories from related entities (graph BFS)
    for eid in list(matched_entity_ids):
        rels = store.get_entity_relationships(eid)
        for rel in rels:
            related_id = rel["target_entity_id"] if rel["source_entity_id"] == eid else rel["source_entity_id"]
            if related_id not in matched_entity_ids:
                related_mems = store.get_entity_memories(related_id, limit=max(3, limit // 4))
                strength = rel.get("strength", 1.0)
                for mem in related_mems:
                    # score decays with hop distance, weighted by relationship strength
                    candidates.append((mem.id, 0.5 * min(1.0, strength)))

    # deduplicate, keep highest score
    seen = {}
    for mid, score in candidates:
        if mid not in seen or score > seen[mid]:
            seen[mid] = score
    return list(seen.items())[:limit]


def _hopfield_search(query: str, store: Store, config: Config, limit: int) -> list[tuple[str, float]]:
    """Hopfield associative channel — pattern completion from partial cue."""
    try:
        query_vec = embed_query(query, config.embedding_model)
        return hopfield_retrieve(query_vec, store, beta=8.0, top_k=limit)
    except Exception:
        return []


def _rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60,
              signal_weights: list[float] | None = None) -> dict[str, float]:
    scores: dict[str, float] = {}
    if signal_weights is None:
        signal_weights = [1.0] * len(rankings)
    for ranking, weight in zip(rankings, signal_weights):
        for rank, (doc_id, _) in enumerate(ranking):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += weight * (1.0 / (k + rank + 1))
    return scores


def _apply_boosts(query: str, candidates: list[tuple[str, float]],
                  store: Store, config: Config) -> list[tuple[str, float]]:
    temporal_signal = _detect_temporal(query)
    boosted = []

    for mid, rrf_score in candidates:
        mem = store.get_memory(mid)
        if not mem:
            continue

        score = rrf_score

        # temporal boost
        if temporal_signal and mem.fact_date:
            if temporal_signal in (mem.fact_date or ""):
                score *= 2.0

        # importance boost (mild)
        score *= (0.8 + 0.4 * mem.importance)

        # recency decay (Ebbinghaus) — only for episodic
        if mem.layer == "episodic":
            age_days = (time.time() - mem.created_at) / 86400
            half_life = config.lifecycle.forgetting_half_life_days
            decay = math.exp(-0.693 * age_days / half_life)
            score *= (0.5 + 0.5 * decay)  # floor at 50%

        # access frequency boost (log scale)
        if mem.access_count > 0:
            score *= (1.0 + 0.1 * math.log(1 + mem.access_count))

        boosted.append((mid, score))

    return sorted(boosted, key=lambda x: x[1], reverse=True)


def _detect_temporal(query: str) -> str | None:
    patterns = [
        (r"\b(\d{4}-\d{2}-\d{2})\b", None),
        (r"\b(\d{4}-\d{2})\b", None),
        (r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b",
         lambda m: f"{m.group(2)}-{_month_num(m.group(1))}"),
        (r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})\b",
         lambda m: f"2026-{_month_num(m.group(1))}-{int(m.group(2)):02d}"),
        (r"\b(march|april)\s+(\d{1,2})\b",
         lambda m: f"2026-{_month_num(m.group(1))}-{int(m.group(2)):02d}"),
    ]
    query_lower = query.lower()
    for pat, transform in patterns:
        m = re.search(pat, query_lower)
        if m:
            if transform:
                return transform(m)
            return m.group(1)
    return None


def _month_num(name: str) -> str:
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    return months.get(name.lower(), "01")
