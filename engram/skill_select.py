"""Task-aware skill selection — inject procedural knowledge only when it helps.

Inspired by SkillsBench (2602.12670, Li et al.): curated skills improve agent
performance by +16.2pp, but self-generated skills are useless (-1.3pp), 2-3
focused skills beat 4+ comprehensive ones, and 16/84 tasks are hurt by skills.

Also informed by the AGENTS.md evaluation (2602.11988, Gloaguen et al.): static
context files reduce performance by adding redundant information.

The key insight: the value isn't in WHAT you inject, it's in WHEN and HOW MUCH.
This module implements a three-stage gate:

1. NEED assessment — does this task need procedural help?
   - Query surprise against existing memories (high = unfamiliar territory)
   - Domain coverage check (how much procedural knowledge exists here?)
   - If the agent likely already knows how → skip injection

2. SELECTION — which 2-3 procedures are most relevant?
   - Hybrid search scoped to procedural + semantic layers
   - Novelty filter: skip memories the agent has seen recently
   - Budget cap: return at most max_skills items (default 3)

3. CALIBRATION — is this actually worth injecting?
   - Relevance threshold: only include if retrieval score > min_relevance
   - Redundancy check: if all candidates overlap with recent context → skip
   - Return confidence score so the caller can decide
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from engram.config import Config
from engram.store import Store, Memory
from engram.embeddings import embed_query, cosine_similarity_search
from engram.surprise import compute_surprise


@dataclass
class SkillSelection:
    """Result of task-aware skill selection."""
    should_inject: bool
    skills: list[Memory]
    confidence: float          # 0-1, how confident we are injection helps
    task_novelty: float        # 0-1, how unfamiliar this task domain is
    domain_coverage: float     # 0-1, how much procedural knowledge we have
    reason: str                # human-readable explanation


def select_skills(query: str, store: Store, config: Config,
                  max_skills: int = 3,
                  min_relevance: float = 0.3,
                  novelty_threshold: float = 0.4,
                  recency_window: float = 60) -> SkillSelection:
    """Select relevant procedural skills for a task, or decide to skip injection.

    Args:
        query: the task description or search query
        store: engram store
        config: engram config
        max_skills: maximum number of skills to return (2-3 is optimal per SkillsBench)
        min_relevance: minimum retrieval score to include a skill
        novelty_threshold: below this task novelty score, skip injection
        recency_window: seconds — skip memories accessed within this window
    """

    # --- Stage 1: NEED assessment ---

    # compute task novelty: how unfamiliar is this query domain?
    query_emb = embed_query(query, config.embedding_model)
    surprise_info = compute_surprise(query_emb, store)
    task_novelty = surprise_info["surprise"]

    # domain coverage: how much procedural knowledge do we have in this area?
    # search procedural + semantic layers for related content
    proc_ids, proc_vecs = _get_layer_embeddings(store, ["procedural", "semantic"])

    if len(proc_ids) == 0:
        return SkillSelection(
            should_inject=False, skills=[], confidence=0.0,
            task_novelty=task_novelty, domain_coverage=0.0,
            reason="no procedural knowledge stored",
        )

    # domain coverage: use relative similarity spread, not absolute threshold.
    # small embedding models (bge-small) give 0.5+ cosine for any English text pair,
    # so absolute thresholds don't discriminate. instead, compare the top score against
    # the median — a large gap means genuinely relevant procedures exist.
    proc_hits = cosine_similarity_search(query_emb, proc_vecs, top_k=min(10, len(proc_ids)))
    if len(proc_hits) >= 3:
        top_score = proc_hits[0][1]
        median_score = proc_hits[len(proc_hits) // 2][1]
        spread = top_score - median_score
        # spread > 0.05 means the top result is meaningfully more relevant than average
        # scale coverage by how many are above median + half the spread
        threshold = median_score + spread * 0.5
        relevant_count = sum(1 for _, score in proc_hits if score > threshold)
        domain_coverage = min(1.0, relevant_count / 3.0)
    else:
        relevant_count = len(proc_hits)
        domain_coverage = min(1.0, relevant_count / 3.0)

    # decision: should we inject?
    # - high task novelty + low domain coverage = inject (agent needs help)
    # - low task novelty + high domain coverage = inject (we have specific knowledge)
    # - low task novelty + low domain coverage = skip (agent knows, we don't add value)
    # - high task novelty + high domain coverage = definitely inject

    inject_score = _compute_inject_score(task_novelty, domain_coverage)

    if inject_score < 0.3 and domain_coverage < 0.2:
        return SkillSelection(
            should_inject=False, skills=[], confidence=1.0 - inject_score,
            task_novelty=task_novelty, domain_coverage=domain_coverage,
            reason="task domain is well-covered by model pretraining, low procedural coverage — injection would add overhead without benefit",
        )

    # --- Stage 2: SELECTION ---

    # get the top candidates from procedural layer specifically
    proc_only_ids, proc_only_vecs = _get_layer_embeddings(store, ["procedural"])
    if len(proc_only_ids) == 0:
        # fall back to semantic if no procedural
        proc_only_ids, proc_only_vecs = proc_ids, proc_vecs

    candidates = cosine_similarity_search(query_emb, proc_only_vecs,
                                           top_k=min(max_skills * 3, len(proc_only_ids)))

    # filter and rank
    now = time.time()
    selected = []

    # compute adaptive relevance threshold from candidate distribution
    # use percentile-based: keep candidates in the top 30% of the score range
    if len(candidates) >= 3:
        cand_scores = [s for _, s in candidates]
        score_range = cand_scores[0] - cand_scores[-1]
        adaptive_threshold = max(min_relevance, cand_scores[-1] + score_range * 0.7)
    else:
        adaptive_threshold = min_relevance

    for idx, score in candidates:
        if score < adaptive_threshold:
            continue
        if len(selected) >= max_skills:
            break

        mem = store.get_memory(proc_only_ids[idx])
        if not mem:
            continue

        # skip recently accessed (agent probably already has this in context)
        if (now - mem.last_accessed) < recency_window and mem.access_count > 0:
            continue

        # skip invalidated memories
        if mem.metadata.get("invalidated"):
            continue

        selected.append(mem)

    # --- Stage 3: CALIBRATION ---

    if not selected:
        return SkillSelection(
            should_inject=False, skills=[], confidence=0.5,
            task_novelty=task_novelty, domain_coverage=domain_coverage,
            reason="no relevant procedural skills above relevance threshold",
        )

    # check redundancy: are all selected skills too similar to each other?
    if len(selected) >= 2:
        selected = _deduplicate_skills(selected, threshold=0.85)

    # compute final confidence — factor in absolute relevance
    avg_relevance = sum(
        float(np.dot(query_emb, mem.embedding)) if mem.embedding is not None else 0
        for mem in selected
    ) / len(selected)

    # penalize if absolute relevance is mediocre (below 0.65 for bge-small
    # means the skills aren't really about this topic, just nearest neighbors)
    relevance_quality = max(0, (avg_relevance - 0.55) / 0.15)  # 0 at 0.55, 1 at 0.70

    confidence = min(1.0, inject_score * 0.3 + avg_relevance * 0.3 + relevance_quality * 0.4)

    # final gate: if confidence is too low, don't bother
    # 0.5 threshold filters out borderline matches where the skills aren't
    # genuinely about the task (SkillsBench: irrelevant skills hurt performance)
    if confidence < 0.5:
        return SkillSelection(
            should_inject=False, skills=selected, confidence=confidence,
            task_novelty=task_novelty, domain_coverage=domain_coverage,
            reason=f"confidence too low ({confidence:.2f}) — skills available but marginal relevance",
        )

    # record access for selected skills
    for mem in selected:
        store.record_access(mem.id, f"skill_select:{query[:50]}")

    reason_parts = []
    if task_novelty > 0.5:
        reason_parts.append("unfamiliar task domain")
    if domain_coverage > 0.3:
        reason_parts.append(f"{len(selected)} relevant procedures found")
    reason_parts.append(f"confidence={confidence:.2f}")

    return SkillSelection(
        should_inject=True, skills=selected, confidence=confidence,
        task_novelty=task_novelty, domain_coverage=domain_coverage,
        reason=" — ".join(reason_parts) if reason_parts else "procedural guidance available",
    )


def format_skills(selection: SkillSelection, max_tokens: int = 1500) -> str:
    """Format selected skills into a compact context block.

    Follows SkillsBench finding: focused, step-oriented, with concrete examples.
    Comprehensive dumps hurt. Brief procedures help.
    """
    if not selection.should_inject or not selection.skills:
        return ""

    parts = [f"[Procedural guidance — {len(selection.skills)} skills, confidence={selection.confidence:.2f}]"]

    token_budget = max_tokens
    for i, mem in enumerate(selection.skills):
        content = mem.content.strip()
        # estimate tokens
        est_tokens = int(len(content.split()) * 1.3)
        if est_tokens > token_budget:
            # truncate to fit budget
            words = content.split()
            max_words = int(token_budget / 1.3)
            content = " ".join(words[:max_words]) + "..."
            est_tokens = token_budget

        parts.append(f"\n[{i+1}] {content}")
        token_budget -= est_tokens

        if token_budget <= 0:
            break

    return "\n".join(parts)


def _compute_inject_score(task_novelty: float, domain_coverage: float) -> float:
    """Compute whether injection is likely beneficial.

    High score = inject. Low score = skip.

    The quadrant logic:
    - Novel task + good coverage → definitely inject (0.9)
    - Novel task + no coverage → maybe inject (0.5) — we might not have what's needed
    - Familiar task + good coverage → inject specific procedures (0.7)
    - Familiar task + no coverage → skip (0.2) — agent knows, we don't add value
    """
    # weighted combination — domain coverage matters more than novelty
    # because having relevant procedures is the prerequisite
    score = domain_coverage * 0.6 + task_novelty * 0.3

    # bonus for having both
    if domain_coverage > 0.3 and task_novelty > 0.3:
        score += 0.15

    return min(1.0, score)


def _get_layer_embeddings(store: Store, layers: list[str]) -> tuple[list[str], np.ndarray]:
    """Get embeddings for memories in specific layers."""
    placeholders = ",".join("?" * len(layers))
    rows = store.conn.execute(
        f"SELECT id, embedding FROM memories WHERE layer IN ({placeholders}) "
        "AND embedding IS NOT NULL AND forgotten = 0",
        layers,
    ).fetchall()
    if not rows:
        return [], np.array([])
    ids = [r["id"] for r in rows]
    vecs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32).copy() for r in rows])
    return ids, vecs


def _deduplicate_skills(skills: list[Memory], threshold: float = 0.85) -> list[Memory]:
    """Remove skills that are too similar to each other."""
    if len(skills) <= 1:
        return skills

    kept = [skills[0]]
    for mem in skills[1:]:
        if mem.embedding is None:
            kept.append(mem)
            continue
        # check similarity against all kept
        too_similar = False
        for k in kept:
            if k.embedding is None:
                continue
            sim = float(np.dot(mem.embedding, k.embedding))
            if sim >= threshold:
                too_similar = True
                break
        if not too_similar:
            kept.append(mem)
    return kept
