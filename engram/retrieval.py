"""5-stage hybrid retrieval pipeline: intent → dense + BM25 + graph → RRF → temporal boost → cross-encoder.

Intent-aware routing (MAGMA): classify query intent and dynamically weight signals.
Retrieval threshold (ACT-R): gate results by minimum score, don't always return top-k.
"""

from __future__ import annotations

import math
import random
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from engram.config import Config
from engram.embeddings import RERANKER_BACKENDS, cosine_similarity_search, cross_encoder_rerank, embed_query
from engram.rerank_scoring import rerank_score
from engram.rerank_selection import preserve_prior_leader
from engram.rerank_passages import rerank_with_passages, retry_passages
from engram.retrieval_explain import ExplanationTrace, build_explanation, eligibility_reason
from engram.hopfield import hopfield_retrieve
from engram.store import Memory, MemoryType, Store
from engram.temporal import _parse_date, _resolve_temporal_window, _temporal_content_query


INTENT_PATTERNS = {
    "why": re.compile(r"\b(why|because|reason|cause|led to|resulted in)\b", re.I),
    "when": re.compile(r"\b(when|date|time|before|after|during|timeline|history)\b", re.I),
    "who": re.compile(r"\b(who|person|people|team|built|created|wrote)\b", re.I),
    "how": re.compile(r"\b(how to|steps|procedure|process|workflow|debug|fix)\b", re.I),
}

INTENT_WEIGHTS = {
    "why": {"dense": 1.0, "bm25": 0.8, "graph": 1.5},
    "when": {"dense": 0.8, "bm25": 1.2, "graph": 0.8},
    "who": {"dense": 0.8, "bm25": 0.8, "graph": 1.8},
    "how": {"dense": 1.2, "bm25": 1.0, "graph": 0.8},
    "what": {"dense": 1.0, "bm25": 1.0, "graph": 1.0},
}

QUERY_EXPANSIONS = {
    "auth": ["authentication", "login", "oauth", "token"],
    "bug": ["issue", "error", "failure"],
    "deploy": ["deployment", "release", "ship"],
    "memory": ["recall", "context", "history"],
    "graph": ["entity", "relationship"],
    "code": ["function", "class", "file"],
}

RETRIEVAL_PROFILES = {
    "facts_only": {MemoryType.FACT},
    "facts_plus_rules": {MemoryType.FACT, MemoryType.PROCEDURE},
    "full_context": {MemoryType.FACT, MemoryType.PROCEDURE, MemoryType.NARRATIVE},
}

RETRIEVAL_NOISE_SCALE = 0.02


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid mapping logits to [0.0, 1.0]."""
    if x >= 40.0:
        return 1.0
    if x <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "yourselves", "you", "your", "yours",
    "he", "him", "his", "himself", "she", "her", "hers", "it", "its", "they", "them",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "ve", "d", "ll", "m", "o", "re", "y", "ain", "aren", "couldn", "didn", "doesn",
    "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn", "would", "think", "good", "idea", "lately", "feeling",
    "feel", "tell", "want", "like", "get", "give", "suggest", "know", "help",
}


def _extract_content_phrases(query: str) -> list[str]:
    words = [w for w in re.findall(r"[a-zA-Z0-9]+", query.lower()) if w not in _STOP_WORDS]
    return [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]


def classify_intent(query: str) -> str:
    scores = {}
    for intent, pattern in INTENT_PATTERNS.items():
        scores[intent] = len(pattern.findall(query))
    if max(scores.values()) == 0:
        return "what"
    return max(scores, key=scores.get)


@dataclass
class QueryFeatures:
    original: str
    intent: str
    tokens: list[str]
    phrase_terms: list[str]
    expanded_terms: list[str]
    dense_query: str
    bm25_query: str


@dataclass
class RetrievalResult:
    memory: Memory
    score: float
    sources: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalDebug:
    query: str
    intent: str
    expanded_terms: list[str]
    phrase_terms: list[str]
    cache_hit: bool
    dense_candidates: list[tuple[str, float]]
    bm25_candidates: list[tuple[str, float]]
    graph_candidates: list[tuple[str, float]]
    rrf_scores: list[tuple[str, float]]
    boosted_scores: list[tuple[str, float]]
    reranked: list[tuple[str, float]]
    final_results: list[RetrievalResult]
    latency_ms: float
    hopfield_candidates: list[tuple[str, float]] = field(default_factory=list)
    explanation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-friendly report without serializing Memory internals."""
        return deepcopy(self.explanation)


def search(
    query: str,
    store: Store,
    config: Config | None = None,
    top_k: int | None = None,
    debug: bool = False,
    rerank: bool = True,
    deep_reranker=None,
    mode: str = "full_context",
    reference_date: str | datetime | date | float | int | None = None,
) -> list[RetrievalResult] | tuple[list[RetrievalResult], RetrievalDebug]:
    if config is None:
        config = Config.load()

    rc = config.retrieval
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    k = rc.top_k if top_k is None else top_k
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(mode, str) or mode not in RETRIEVAL_PROFILES:
        raise ValueError("mode must be facts_only, facts_plus_rules, or full_context")
    t0 = time.time()
    allowed_types = RETRIEVAL_PROFILES[mode]
    features = _build_query_features(query, config)
    weights = INTENT_WEIGHTS.get(features.intent, INTENT_WEIGHTS["what"])
    if isinstance(reference_date, bool):
        raise ValueError("reference_date must be a valid date or timestamp")
    ref_d = _parse_date(reference_date) if reference_date is not None else None
    if reference_date is not None and ref_d is None:
        raise ValueError("reference_date must be a valid date or timestamp")

    if ref_d:
        cache_key = (
            features.original.lower(),
            mode,
            int(k),
            bool(rerank),
            config.embedding_model,
            config.cross_encoder_model,
            str(ref_d),
        )
    else:
        cache_key = (
            features.original.lower(),
            mode,
            int(k),
            bool(rerank),
            config.embedding_model,
            config.cross_encoder_model,
        )
    if rerank:
        cache_key += ("rerank-v5", rc.rerank_fusion_alpha, rc.min_confidence,
                      rc.preserve_prior_candidate, rc.rerank_passage_fallback,
                      rc.rerank_passage_floor)
    cache_hit = False
    dense_candidates: list[tuple[str, float]] = []
    bm25_candidates: list[tuple[str, float]] = []
    graph_candidates: list[tuple[str, float]] = []
    hopfield_candidates: list[tuple[str, float]] = []
    rrf_scores: dict[str, float] = {}
    boosted: list[tuple[str, float]] = []
    reranked: list[tuple[int, float]] = []
    boosted_reranked: list[tuple[str, float, float, float]] = []
    valid_ids: list[str] = []
    trace = ExplanationTrace() if debug else None

    cached_payload = None if debug else store.get_search_cache(cache_key)
    if cached_payload:
        results = _deserialize_results(cached_payload, store)
        cache_hit = True
    else:
        dense_candidates = _dense_search(features.dense_query, store, config, k * rc.dense_multiplier)
        bm25_candidates = _bm25_search(features.bm25_query, store, k * rc.bm25_multiplier)
        graph_candidates = _graph_search(features, store, k)
        hopfield_candidates = _hopfield_search(features.dense_query, store, config, k)

        rrf_scores = _rrf_fuse(
            [dense_candidates, bm25_candidates, graph_candidates, hopfield_candidates],
            k=rc.rrf_k,
            signal_weights=[weights["dense"], weights["bm25"], weights["graph"], weights.get("hopfield", 0.6)],
        )
        rrf_top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: rc.rerank_candidates]
        if trace is not None:
            trace.rrf_selected_ids = [mid for mid, _ in rrf_top]
        boosted = _apply_boosts(features, rrf_top, store, config, reference_date=ref_d)

        candidate_ids = [mid for mid, _ in boosted[: rc.rerank_candidates]]
        candidate_memories: dict[str, Memory] = {}
        for mid in candidate_ids:
            mem = store.get_memory(mid)
            if eligibility_reason(mem, allowed_types) is None:
                candidate_memories[mid] = mem

        if not candidate_memories:
            results = []
        elif rerank:
            docs = [candidate_memories[mid].content for mid in candidate_ids if mid in candidate_memories]
            valid_ids = [mid for mid in candidate_ids if mid in candidate_memories]
            if rc.rerank_passage_fallback:
                reranked, passage_details = rerank_with_passages(
                    features.dense_query, docs, config.cross_encoder_model,
                    rerank_fn=cross_encoder_rerank, confidence_floor=rc.rerank_passage_floor,
                    passage_query=_temporal_content_query(features.dense_query, ref_d),
                )
            else:
                reranked = cross_encoder_rerank(features.dense_query, docs, config.cross_encoder_model)
                passage_details = {}
            # post-rerank temporal boost — cross-encoder scores text semantics only,
            # so we condition final scores on temporal window proximity when a date is resolved
            window = _resolve_temporal_window(features.original, ref_d) if ref_d else None
            pre_rank_map = {mid: i for i, mid in enumerate(valid_ids)}

            def score_candidates(ranking):
                scored = []
                for idx, ce_score in ranking:
                    mid = valid_ids[idx]
                    d_date = _get_memory_date(candidate_memories[mid])
                    t_boost = 0.0
                    if window and d_date:
                        center, margin = window
                        if abs((d_date - center).days) <= margin:
                            t_boost = 5.0
                    final_score = rerank_score(
                        ce_score, prior_rank=pre_rank_map[mid],
                        fusion_alpha=rc.rerank_fusion_alpha, temporal_boost=t_boost,
                        normalized=config.cross_encoder_model in RERANKER_BACKENDS)
                    scored.append((mid, final_score, ce_score, t_boost))
                return sorted(scored, key=lambda row: row[1], reverse=True)

            boosted_reranked = score_candidates(reranked)
            rejected_indices = {pre_rank_map[mid] for mid, score, _, _ in boosted_reranked
                                if score < rc.min_confidence}
            confidence_gate_retry = (
                rc.rerank_passage_fallback and rc.rerank_passage_floor > 0
                and rc.min_confidence > 0 and rejected_indices
                and config.cross_encoder_model not in RERANKER_BACKENDS
            )
            if confidence_gate_retry:
                # Another candidate must not prevent inspection of a long
                # memory whose relevant passage may be truncated. Retry only
                # rejected candidates, reusing base scores and respecting the
                # existing per-document excerpt budget and confidence gate.
                prior_attempts = {index for index, detail in passage_details.items()
                                  if "excerpt_raw_score" in detail}
                reranked, passage_details = retry_passages(
                    features.dense_query, docs, config.cross_encoder_model,
                    reranked, passage_details, rerank_fn=cross_encoder_rerank,
                    passage_query=_temporal_content_query(features.dense_query, ref_d),
                    candidate_indices=rejected_indices)
                for index, detail in passage_details.items():
                    if "excerpt_raw_score" in detail and index not in prior_attempts:
                        detail["confidence_gate_retry"] = 1.0
                boosted_reranked = score_candidates(reranked)

            passage_by_id = {valid_ids[index]: detail for index, detail in passage_details.items()}
            if trace is not None:
                trace.prior_ids = list(valid_ids)
                trace.passages = passage_by_id
                trace.scored = list(boosted_reranked)
            # Confidence is a separate eligibility gate. Reserve only a
            # candidate that would otherwise be allowed in the result set.
            if rc.min_confidence > 0:
                boosted_reranked = [row for row in boosted_reranked if row[1] >= rc.min_confidence]
            if trace is not None:
                trace.before_coverage = [row[0] for row in boosted_reranked]
            if rc.preserve_prior_candidate:
                by_id = {row[0]: row for row in boosted_reranked}
                order = preserve_prior_leader(list(by_id), valid_ids, result_limit=k)
                boosted_reranked = [by_id[mid] for mid in order]
            if trace is not None:
                trace.after_coverage = [row[0] for row in boosted_reranked]

            results = []
            for mid, final_score, ce_score, t_boost in boosted_reranked[:k]:
                mem = candidate_memories[mid]
                results.append(
                    RetrievalResult(
                        memory=mem,
                        score=final_score,
                        sources={
                            **passage_by_id.get(mid, {}),
                            "dense": dict(dense_candidates).get(mid, 0),
                            "bm25": dict(bm25_candidates).get(mid, 0),
                            "graph": dict(graph_candidates).get(mid, 0),
                            "rrf": rrf_scores.get(mid, 0),
                            "boosted": dict(boosted).get(mid, 0),
                            "exact_match": _exact_match_signal(features, mem),
                            "cross_encoder": ce_score,
                            "cross_encoder_calibrated": rerank_score(
                                ce_score, prior_rank=pre_rank_map[mid],
                                normalized=config.cross_encoder_model in RERANKER_BACKENDS),
                            "temporal_boost": t_boost,
                        },
                    )
                )
        else:
            results = []
            for mid, score in boosted[:k]:
                if mid in candidate_memories:
                    mem = candidate_memories[mid]
                    results.append(
                        RetrievalResult(
                            memory=mem,
                            score=score,
                            sources={
                                "dense": dict(dense_candidates).get(mid, 0),
                                "bm25": dict(bm25_candidates).get(mid, 0),
                                "graph": dict(graph_candidates).get(mid, 0),
                                "rrf": rrf_scores.get(mid, 0),
                                "boosted": score,
                                "exact_match": _exact_match_signal(features, mem),
                            },
                        )
                    )

        if deep_reranker and deep_reranker.is_trained and results:
            if trace is not None:
                trace.deep_applied = True
                trace.deep_input_ids = [result.memory.id for result in results]
            query_vec = embed_query(features.dense_query, config.embedding_model)
            candidates = []
            emb_map = {}
            for r in results:
                candidates.append(
                    {
                        "id": r.memory.id,
                        "score": r.score,
                        "importance": r.memory.importance,
                        "access_count": r.memory.access_count,
                        "created_at": r.memory.created_at,
                        "layer": r.memory.layer,
                    }
                )
                if r.memory.embedding is not None:
                    emb_map[r.memory.id] = r.memory.embedding

            reranked_candidates = deep_reranker.rerank(candidates, query_vec, emb_map)
            if trace is not None:
                trace.deep_scores = {row["id"]: row["deep_score"] for row in reranked_candidates
                                     if "deep_score" in row}
            mem_map = {r.memory.id: r for r in results}
            new_results = []
            for c in reranked_candidates[:k]:
                r = mem_map[c["id"]]
                r.sources["deep_reranker"] = c.get("deep_score", 0)
                new_results.append(r)
            results = new_results

        if results and not rerank and RETRIEVAL_NOISE_SCALE > 0:
            for r in results:
                noise = random.gauss(0, RETRIEVAL_NOISE_SCALE)
                before_noise = r.score
                r.score = max(0.0, r.score + noise)
                if trace is not None:
                    trace.noise[r.memory.id] = {"sample": float(noise), "before": float(before_noise),
                                                "after": float(r.score), "scale": RETRIEVAL_NOISE_SCALE}
            results.sort(key=lambda r: r.score, reverse=True)

        if rerank and results:
            min_threshold = getattr(rc, "min_confidence", 0.0)
            if min_threshold > 0:
                results = [r for r in results if r.score >= min_threshold]

    # A model call can take long enough for another request to edit or forget
    # a candidate. Recheck immediately before caching or returning content.
    # Diagnostics use one snapshot for both results and their explanation.
    if trace is not None:
        trace.memories = {mid: store.get_memory(mid) for mid in rrf_scores}
    results = _refresh_results(results, store, allowed_types,
                               memories=trace.memories if trace is not None else None)

    if not debug and not cache_hit:
        store.set_search_cache(cache_key, _serialize_results(results))

    if results and not debug:
        store.record_search([r.memory.id for r in results], query)

    if not debug and config.dormant_recall.mode == "shadow":
        from engram.dormant import evaluate_shadow
        evaluate_shadow(query, store, config, {r.memory.id for r in results}, allowed_types)

    latency = (time.time() - t0) * 1000
    if debug:
        dbg = RetrievalDebug(
            query=query,
            intent=features.intent,
            expanded_terms=features.expanded_terms,
            phrase_terms=features.phrase_terms,
            cache_hit=cache_hit,
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
            graph_candidates=graph_candidates,
            hopfield_candidates=hopfield_candidates,
            rrf_scores=list(rrf_scores.items()),
            boosted_scores=boosted,
            reranked=[(mid, s) for mid, s, _, _ in boosted_reranked[:k]] if boosted_reranked else ([(valid_ids[i], s) for i, s in reranked[:k]] if valid_ids and reranked else []),
            final_results=results,
            latency_ms=latency,
        )
        dbg.explanation = build_explanation(
            dbg, trace=trace, config=config, features=features,
            mode=mode, allowed_types=allowed_types, top_k=k, rerank=rerank,
            reference_date=ref_d, weights=weights,
        )
        return results, dbg
    return results


def _build_query_features(query: str, config: Config) -> QueryFeatures:
    normalized = " ".join(query.strip().split())
    tokens = re.findall(r"[A-Za-z0-9_./:-]+", normalized.lower())
    phrase_terms = [p.strip().lower() for p in re.findall(r'"([^"]+)"', query) if p.strip()]
    expanded_terms: list[str] = []
    if config.retrieval.enable_query_expansion:
        for token in tokens:
            expanded_terms.extend(QUERY_EXPANSIONS.get(token, []))
    expanded_terms = list(dict.fromkeys(expanded_terms))

    dense_query = normalized
    if expanded_terms:
        dense_query = normalized + " " + " ".join(expanded_terms[:8])

    bm25_parts = [normalized]
    bm25_parts.extend(f'"{phrase}"' for phrase in phrase_terms)
    bm25_parts.extend(expanded_terms[:8])
    return QueryFeatures(
        original=normalized,
        intent=classify_intent(normalized),
        tokens=tokens,
        phrase_terms=phrase_terms,
        expanded_terms=expanded_terms,
        dense_query=dense_query,
        bm25_query=" ".join(part for part in bm25_parts if part).strip(),
    )


def _serialize_results(results: list[RetrievalResult]) -> list[dict]:
    return [{"memory_id": r.memory.id, "score": r.score, "sources": dict(r.sources)} for r in results]


def _refresh_results(results: list[RetrievalResult], store: Store, allowed_types,
                     *, memories: dict | None = None) -> list[RetrievalResult]:
    """Preserve scores/order while checking the latest lifecycle and profile."""
    refreshed = []
    for result in results:
        memory = (memories.get(result.memory.id) if memories is not None
                  else store.get_memory(result.memory.id))
        if eligibility_reason(memory, allowed_types) is None:
            refreshed.append(RetrievalResult(memory=memory, score=result.score, sources=result.sources))
    return refreshed


def _deserialize_results(payload: list[dict], store: Store) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    for item in payload:
        mem = store.get_memory(item["memory_id"])
        if not mem or mem.forgotten or mem.status not in ("active", None):
            continue
        results.append(RetrievalResult(memory=mem, score=item["score"], sources=dict(item.get("sources") or {})))
    return results


def _dense_search(query: str, store: Store, config: Config, limit: int) -> list[tuple[str, float]]:
    query_vec = embed_query(query, config.embedding_model)
    if store.ann_index and store.ann_index.ready:
        return store.ann_index.search(query_vec, top_k=limit)
    ids, vecs = store.get_all_embeddings()
    if not ids:
        return []
    hits = cosine_similarity_search(query_vec, vecs, top_k=limit)
    return [(ids[idx], score) for idx, score in hits]


def _bm25_search(query: str, store: Store, limit: int) -> list[tuple[str, float]]:
    results = store.search_fts(query, limit=limit)
    if not results:
        return []
    return [(mid, -score) for mid, score in results]


def _graph_search(features: QueryFeatures, store: Store, limit: int) -> list[tuple[str, float]]:
    words = list(dict.fromkeys(features.tokens + features.expanded_terms))
    candidates = []
    matched_entity_ids = set()

    for word in words:
        if len(word) < 2:
            continue
        entity = store.find_entity_by_name(word)
        if not entity:
            continue
        matched_entity_ids.add(entity.id)
        for mem in store.get_entity_memories(entity.id, limit=limit):
            candidates.append((mem.id, 1.0))

    for eid in list(matched_entity_ids):
        rels = store.get_entity_relationships(eid)
        for rel in rels:
            related_id = rel["target_entity_id"] if rel["source_entity_id"] == eid else rel["source_entity_id"]
            if related_id in matched_entity_ids:
                continue
            related_mems = store.get_entity_memories(related_id, limit=max(3, limit // 4))
            strength = rel.get("strength", 1.0)
            for mem in related_mems:
                candidates.append((mem.id, 0.5 * min(1.0, strength)))

    seen: dict[str, float] = {}
    for mid, score in candidates:
        if mid not in seen or score > seen[mid]:
            seen[mid] = score
    return list(seen.items())[:limit]


def _hopfield_search(query: str, store: Store, config: Config, limit: int) -> list[tuple[str, float]]:
    try:
        query_vec = embed_query(query, config.embedding_model)
        return hopfield_retrieve(query_vec, store, beta=8.0, top_k=limit)
    except Exception:
        return []


def _rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60, signal_weights: list[float] | None = None) -> dict[str, float]:
    scores: dict[str, float] = {}
    signal_weights = signal_weights or [1.0] * len(rankings)
    for ranking, weight in zip(rankings, signal_weights):
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / (k + rank + 1))
    return scores


def _apply_boosts(
    features: QueryFeatures,
    candidates: list[tuple[str, float]],
    store: Store,
    config: Config,
    reference_date: str | datetime | date | float | int | None = None,
) -> list[tuple[str, float]]:
    temporal_signal = _detect_temporal(features.original)
    ref_d = _parse_date(reference_date) if reference_date is not None else datetime.now().date()
    window = _resolve_temporal_window(features.original, ref_d) if ref_d else None

    boosted = []
    seen_ids = set()
    for mid, rrf_score in candidates:
        if mid in seen_ids:
            continue
        seen_ids.add(mid)
        mem = store.get_memory(mid)
        if not mem:
            continue
        score = rrf_score

        mem_d = _get_memory_date(mem)
        if ref_d and mem_d:
            diff_days = (ref_d - mem_d).days
            if window:
                center, margin = window
                dist = abs((mem_d - center).days)
                if dist <= margin:
                    score *= 2.5
                elif diff_days < 0:
                    score *= 0.90
            else:
                if diff_days < 0:
                    score *= 0.95
                elif diff_days <= 7:
                    score *= 1.15
                elif diff_days <= 30:
                    score *= 1.05

        if temporal_signal and mem.fact_date and temporal_signal in (mem.fact_date or ""):
            score *= 2.0
        score *= (0.8 + 0.4 * mem.importance)
        exact_signal = _exact_match_signal(features, mem)
        if exact_signal > 0:
            score *= (1.0 + exact_signal * (config.retrieval.exact_match_boost - 1.0))
        if mem.layer == "episodic":
            if ref_d and mem_d:
                age_days = max(0.0, float((ref_d - mem_d).days))
            else:
                age_days = max(0.0, (time.time() - mem.created_at) / 86400)
            half_life = config.lifecycle.forgetting_half_life_days
            decay = math.exp(-0.693 * age_days / half_life)
            score *= (0.5 + 0.5 * decay)
        if mem.access_count > 0:
            score *= (1.0 + 0.1 * math.log(1 + mem.access_count))
        boosted.append((mid, score))
    return sorted(boosted, key=lambda x: x[1], reverse=True)


def _exact_match_signal(features: QueryFeatures, mem: Memory) -> float:
    content = mem.content.lower()
    score = 0.0
    for phrase in features.phrase_terms:
        if phrase in content:
            score += 0.6
    token_hits = sum(1 for tok in features.tokens[:8] if len(tok) > 2 and tok in content)
    score += min(0.4, token_hits * 0.08)
    for hq in mem.metadata.get("hypothetical_queries", [])[:10]:
        hq_lower = hq.lower()
        if any(phrase in hq_lower for phrase in features.phrase_terms):
            score += 0.15
            break
    return min(1.0, score)


def _detect_temporal(query: str) -> str | None:
    patterns = [
        (r"\b(\d{4}-\d{2}-\d{2})\b", None),
        (r"\b(\d{4}-\d{2})\b", None),
        (
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b",
            lambda m: f"{m.group(2)}-{_month_num(m.group(1))}",
        ),
        (
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})\b",
            lambda m: f"2026-{_month_num(m.group(1))}-{int(m.group(2)):02d}",
        ),
        (r"\b(march|april)\s+(\d{1,2})\b", lambda m: f"2026-{_month_num(m.group(1))}-{int(m.group(2)):02d}"),
    ]
    query_lower = query.lower()
    for pat, transform in patterns:
        m = re.search(pat, query_lower)
        if not m:
            continue
        return transform(m) if transform else m.group(1)
    return None


def _month_num(name: str) -> str:
    months = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
    }
    return months.get(name.lower(), "01")


def _get_memory_date(mem: Memory) -> date | None:
    if mem.fact_date:
        d = _parse_date(mem.fact_date)
        if d:
            return d
    for key in ("timestamp", "date", "created_at_str"):
        if key in mem.metadata:
            d = _parse_date(mem.metadata[key])
            if d:
                return d
    if mem.created_at:
        return _parse_date(mem.created_at)
    return None
