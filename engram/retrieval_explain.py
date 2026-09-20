"""Read-only, bounded evidence for the retrieval decisions actually made."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from engram.embeddings import RERANKER_BACKENDS
from engram.rerank_scoring import rerank_score


@dataclass
class ExplanationTrace:
    """Transient observations; never persisted in the search result cache."""

    rrf_selected_ids: list[str] = field(default_factory=list)
    prior_ids: list[str] = field(default_factory=list)
    scored: list[tuple[str, float, float, float]] = field(default_factory=list)
    before_coverage: list[str] = field(default_factory=list)
    after_coverage: list[str] = field(default_factory=list)
    passages: dict[str, dict] = field(default_factory=dict)
    noise: dict[str, dict] = field(default_factory=dict)
    deep_input_ids: list[str] = field(default_factory=list)
    deep_scores: dict[str, float] = field(default_factory=dict)
    deep_applied: bool = False
    memories: dict = field(default_factory=dict)


def eligibility_reason(memory, allowed_types) -> str | None:
    """Check availability before any diagnostic content can be exposed."""
    if memory is None:
        return "unavailable"
    if memory.forgotten:
        return "forgotten"
    if memory.status not in ("active", None):
        return "inactive"
    if memory.memory_type not in allowed_types:
        return "profile_filtered"
    return None


def build_explanation(dbg, *, trace, config, features, mode, allowed_types,
                      top_k, rerank, reference_date, weights) -> dict:
    """Serialize only observed scores and explicitly eligible memory content.

    Candidates are the bounded union produced by the four retrieval channels,
    not an inventory of the store. Missing scores mean a stage did not run for
    that candidate. Ranks are one-based and final order is authoritative.
    """
    rc = config.retrieval
    channels = {
        "dense": dbg.dense_candidates, "bm25": dbg.bm25_candidates,
        "graph": dbg.graph_candidates, "hopfield": dbg.hopfield_candidates,
    }
    observations = {}
    for stage, pairs in {**channels, "rrf": dbg.rrf_scores,
                         "boosted": dbg.boosted_scores}.items():
        # RRF insertion order is not ranking order.
        if stage == "rrf":
            pairs = sorted(pairs, key=lambda item: item[1], reverse=True)
        for rank, (mid, score) in enumerate(pairs, 1):
            row = observations.setdefault(mid, {"scores": {}, "ranks": {}})
            row["scores"][stage] = float(score)
            row["ranks"][stage] = rank

    normalized = config.cross_encoder_model in RERANKER_BACKENDS
    prior = {mid: rank for rank, mid in enumerate(trace.prior_ids, 1)}
    before = {mid: rank for rank, mid in enumerate(trace.before_coverage, 1)}
    after = {mid: rank for rank, mid in enumerate(trace.after_coverage, 1)}
    final = {result.memory.id: (rank, result)
             for rank, result in enumerate(dbg.final_results, 1)}
    scored_ids = set()
    for rank, (mid, score, raw, temporal_boost) in enumerate(trace.scored, 1):
        scored_ids.add(mid)
        row = observations[mid]
        row["scores"].update({
            "cross_encoder": float(raw),
            "cross_encoder_calibrated": rerank_score(
                raw, prior_rank=prior[mid] - 1, normalized=normalized),
            "temporal_boost": float(temporal_boost), "final": float(score),
            "normalized_model_score" if normalized else "raw_logit": float(raw),
        })
        row["ranks"].update({"prior": prior[mid], "scored": rank})
        if mid in trace.passages:
            row["passage"] = dict(trace.passages[mid])

    reasons = {
        "unavailable": "The candidate is no longer available in the store.",
        "forgotten": "The memory is forgotten and cannot be returned.",
        "inactive": "The memory is not active and cannot be returned.",
        "profile_filtered": "The memory type is excluded by the requested retrieval profile.",
        "outside_rerank_candidates": "The candidate fell outside the configured candidate budget before reranking.",
        "not_scored": "The reranker did not return a score for this candidate.",
        "deep_reranker_excluded": "The optional deep reranker did not retain this candidate.",
        "outside_top_k": "The candidate was eligible but fell outside the requested result count.",
    }
    candidates = []
    for mid, observed in observations.items():
        # The same final snapshot gates returned results. Re-reading here
        # could contradict final_ids if lifecycle state changed between reads.
        memory = trace.memories.get(mid)
        excluded = eligibility_reason(memory, allowed_types)
        row = {"memory_id": mid, "eligible": excluded is None, **observed}
        # No content, metadata, source path, or embedding for filtered memories.
        if excluded is None:
            row["content"] = memory.content
            row["memory_type"] = memory.memory_type
        applied = rerank and mid in scored_ids and rc.min_confidence > 0
        passed = observed["scores"]["final"] >= rc.min_confidence if applied else None
        row["confidence"] = {
            "applied": applied, "threshold": rc.min_confidence if rerank else None,
            "passed": passed,
        }
        if mid in before:
            row["ranks"].update({"before_coverage": before[mid], "after_coverage": after[mid]})
            row["prior_coverage"] = {"promoted": after[mid] < before[mid]}
        if mid in trace.deep_scores:
            row["scores"]["deep_reranker"] = float(trace.deep_scores[mid])
        if mid in trace.noise:
            row["noise"] = dict(trace.noise[mid])
            row["scores"]["final"] = trace.noise[mid]["after"]

        if excluded:
            outcome = excluded
        elif mid in final:
            outcome = "returned"
            rank, result = final[mid]
            row["ranks"]["final"] = rank
            row["scores"]["final"] = float(result.score)
        elif mid not in trace.rrf_selected_ids:
            outcome = "outside_rerank_candidates"
        elif applied and not passed:
            outcome = "below_confidence"
        elif rerank and mid not in scored_ids:
            outcome = "not_scored"
        elif trace.deep_applied and mid in trace.deep_input_ids:
            outcome = "deep_reranker_excluded"
        else:
            outcome = "outside_top_k"
        row["outcome"] = outcome
        if outcome == "returned":
            row["reason"] = f"Returned at rank {row['ranks']['final']} within the requested top {top_k}."
            if row.get("prior_coverage", {}).get("promoted"):
                row["reason"] += " Kept the best confidence-eligible hybrid candidate in the result window; its score is unchanged."
        elif outcome == "below_confidence":
            row["reason"] = (
                f"Relevance score {row['scores']['final']:.6g} is below the configured "
                f"minimum {rc.min_confidence:.6g}; prior coverage cannot bypass this gate."
            )
        else:
            row["reason"] = reasons[outcome]
        candidates.append(row)

    outcomes = Counter(row["outcome"] for row in candidates)
    return {
        "schema_version": 1,
        "query": {"original": dbg.query, "dense_query": features.dense_query,
                  "bm25_query": features.bm25_query, "intent": dbg.intent,
                  "expanded_terms": list(dbg.expanded_terms), "phrase_terms": list(dbg.phrase_terms)},
        "settings": {
            "top_k": top_k, "mode": mode, "rerank": rerank,
            "embedding_model": config.embedding_model, "cross_encoder_model": config.cross_encoder_model,
            **{name: getattr(rc, name) for name in (
                "rrf_k", "min_confidence", "rerank_candidates", "dense_multiplier", "bm25_multiplier",
                "enable_query_expansion", "exact_match_boost", "rerank_fusion_alpha",
                "preserve_prior_candidate", "rerank_passage_fallback", "rerank_passage_floor")},
            "reference_date": reference_date.isoformat() if reference_date else None,
            "intent_weights": {**weights, "hopfield": weights.get("hopfield", 0.6)},
            "forgetting_half_life_days": config.lifecycle.forgetting_half_life_days,
            "deep_reranker_applied": trace.deep_applied,
        },
        "cache": {"bypassed": True, "read": False, "written": False,
                  "reason": "Diagnostics recompute candidate evidence without reading or writing the result cache."},
        "side_effects": {"record_search": False, "dormant_evaluation": False},
        "score_semantics": {
            "description": "Relevance scores are ranking signals, not probabilities of truth or answer correctness.",
            "eligibility": "eligible describes lifecycle/profile availability; confidence is a separate gate.",
            "model_output": "normalized_score" if normalized else "raw_logit",
            "final_order": "Use final_ids: prior coverage and optional deep reranking can change order without changing scores.",
            "noise": "Non-reranked results retain the ordinary Gaussian score noise; each realized adjustment is reported.",
            "missing_score": "A missing score means the stage did not score this candidate.",
            "rank_base": 1,
        },
        "counts": {**{name: len(values) for name, values in channels.items()},
                   "considered": len(candidates), "rrf": len(dbg.rrf_scores),
                   "candidate_budget": len(trace.rrf_selected_ids), "boosted": len(dbg.boosted_scores),
                   "reranked": len(scored_ids),
                   "confidence_eligible": len(trace.before_coverage) if rerank else None,
                   "returned": len(final), "outcomes": dict(outcomes)},
        "candidates": candidates, "final_ids": list(final), "latency_ms": float(dbg.latency_ms),
        "scope": "Only candidates produced by this bounded retrieval run are explained; absence is not proof that a memory is missing from the store.",
    }
