"""LongMemEval retrieval benchmark for engram.

HNSW + BM25 + RRF per question, ~1.6s/question, ~12min total.

Usage:
    python run_engram.py data/longmemeval_s_cleaned.json [--limit N] [--rerank]
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engram.config import Config
from engram.embeddings import embed_query, embed_documents
from engram.ann_index import ANNIndex


# ── BM25 ─────────────────────────────────────────────────────────

def _simple_bm25(query: str, corpus: list[dict], top_k: int = 50) -> list[tuple[str, float]]:
    query_tokens = re.findall(r"\w+", query.lower())
    if not query_tokens:
        return []

    N = len(corpus)
    doc_tokens = [Counter(re.findall(r"\w+", doc["text"].lower())) for doc in corpus]
    avgdl = sum(sum(m.values()) for m in doc_tokens) / max(N, 1)
    k1, b = 1.5, 0.75

    df = Counter()
    for m in doc_tokens:
        for t in m:
            df[t] += 1

    scores = []
    for i, doc in enumerate(corpus):
        tf_map = doc_tokens[i]
        dl = sum(tf_map.values())
        score = 0
        for qt in query_tokens:
            if qt not in df:
                continue
            idf = math.log((N - df[qt] + 0.5) / (df[qt] + 0.5) + 1)
            tf = tf_map.get(qt, 0)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
            score += idf * tf_norm
        scores.append((doc["id"], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ── temporal boost ───────────────────────────────────────────────

_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}

_DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_RELATIVE_PATTERNS = [
    # "N days ago" / "ten days ago"
    (re.compile(r"(\d+)\s+days?\s+ago", re.I), "days"),
    (re.compile(r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+days?\s+ago", re.I), "days_word"),
    # "N weeks ago" / "four weeks ago"
    (re.compile(r"(\d+)\s+weeks?\s+ago", re.I), "weeks"),
    (re.compile(r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+weeks?\s+ago", re.I), "weeks_word"),
    # "a week ago"
    (re.compile(r"\ba\s+week\s+ago\b", re.I), "a_week"),
    # "N months ago"
    (re.compile(r"(\d+)\s+months?\s+ago", re.I), "months"),
    (re.compile(r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\s+ago", re.I), "months_word"),
    # "a month ago"
    (re.compile(r"\ba\s+month\s+ago\b", re.I), "a_month"),
    # "last Saturday" / "last Monday"
    (re.compile(r"last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", re.I), "last_day"),
    # "yesterday"
    (re.compile(r"\byesterday\b", re.I), "yesterday"),
    # "past N days" / "in the past two weeks" / "last N weeks"
    (re.compile(r"\b(?:in\s+the\s+)?(?:past|last)\s+(\d+)\s+days?\b", re.I), "past_days"),
    (re.compile(r"\b(?:in\s+the\s+)?(?:past|last)\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+days?\b", re.I), "past_days_word"),
    (re.compile(r"\b(?:in\s+the\s+)?(?:past|last)\s+(\d+)\s+weeks?\b", re.I), "past_weeks"),
    (re.compile(r"\b(?:in\s+the\s+)?(?:past|last)\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+weeks?\b", re.I), "past_weeks_word"),
    (re.compile(r"\b(?:in\s+the\s+)?(?:past|last)\s+(\d+)\s+months?\b", re.I), "past_months"),
    (re.compile(r"\b(?:in\s+the\s+)?(?:past|last)\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b", re.I), "past_months_word"),
]


def _parse_date(date_str: str):
    """Parse 'YYYY/MM/DD ...' into a datetime.date."""
    from datetime import date
    m = _DATE_RE.match(date_str)
    if not m:
        return None
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _resolve_temporal_window(query: str, question_date: str):
    """Parse relative time expressions and return (center_date, margin_days) or None."""
    from datetime import date, timedelta

    q_date = _parse_date(question_date)
    if not q_date:
        return None

    for pattern, kind in _RELATIVE_PATTERNS:
        m = pattern.search(query)
        if not m:
            continue

        if kind == "days":
            n = int(m.group(1))
            return (q_date - timedelta(days=n), 2)
        elif kind == "days_word":
            n = _WORD_TO_NUM.get(m.group(1).lower(), 0)
            if n:
                return (q_date - timedelta(days=n), 2)
        elif kind == "weeks":
            n = int(m.group(1))
            return (q_date - timedelta(weeks=n), 4)
        elif kind == "weeks_word":
            n = _WORD_TO_NUM.get(m.group(1).lower(), 0)
            if n:
                return (q_date - timedelta(weeks=n), 4)
        elif kind == "a_week":
            return (q_date - timedelta(weeks=1), 4)
        elif kind == "months":
            n = int(m.group(1))
            return (q_date - timedelta(days=n * 30), 7)
        elif kind == "months_word":
            n = _WORD_TO_NUM.get(m.group(1).lower(), 0)
            if n:
                return (q_date - timedelta(days=n * 30), 7)
        elif kind == "a_month":
            return (q_date - timedelta(days=30), 7)
        elif kind == "last_day":
            day_name = m.group(1).lower()
            target_dow = _DAY_NAMES[day_name]
            diff = (q_date.weekday() - target_dow) % 7
            if diff == 0:
                diff = 7  # "last Saturday" when today is Saturday means 7 days ago
            return (q_date - timedelta(days=diff), 2)
        elif kind == "yesterday":
            return (q_date - timedelta(days=1), 1)
        elif kind == "past_days":
            n = int(m.group(1))
            return (q_date - timedelta(days=n / 2), max(1, int(n / 2) + 1))
        elif kind == "past_days_word":
            n = _WORD_TO_NUM.get(m.group(1).lower(), 0)
            if n:
                return (q_date - timedelta(days=n / 2), max(1, int(n / 2) + 1))
        elif kind == "past_weeks":
            n = int(m.group(1))
            return (q_date - timedelta(days=n * 7 / 2), max(2, int(n * 7 / 2) + 2))
        elif kind == "past_weeks_word":
            n = _WORD_TO_NUM.get(m.group(1).lower(), 0)
            if n:
                return (q_date - timedelta(days=n * 7 / 2), max(2, int(n * 7 / 2) + 2))
        elif kind == "past_months":
            n = int(m.group(1))
            return (q_date - timedelta(days=n * 30 / 2), max(4, int(n * 30 / 2) + 4))
        elif kind == "past_months_word":
            n = _WORD_TO_NUM.get(m.group(1).lower(), 0)
            if n:
                return (q_date - timedelta(days=n * 30 / 2), max(4, int(n * 30 / 2) + 4))

    return None


def _apply_temporal_boost(scores: dict, corpus: list[dict], question_date: str,
                          query: str = ""):
    from datetime import timedelta

    q_date = _parse_date(question_date)
    if not q_date:
        return

    # try to resolve a specific temporal window from the query
    window = _resolve_temporal_window(query, question_date) if query else None

    for doc in corpus:
        d_date = _parse_date(doc.get("timestamp", ""))
        if not d_date or doc["id"] not in scores:
            continue

        diff_days = (q_date - d_date).days

        if window:
            center, margin = window
            dist = abs((d_date - center).days)
            kernel = math.exp(-0.5 * (dist / max(1.0, float(margin))) ** 2)
            scores[doc["id"]] *= (1.0 + 1.5 * kernel)
            if diff_days < 0:
                scores[doc["id"]] *= 0.90
        else:
            # generic proximity boost (no temporal expression detected)
            if diff_days < 0:
                scores[doc["id"]] *= 0.95
            elif diff_days < 7:
                scores[doc["id"]] *= 1.15
            elif diff_days < 30:
                scores[doc["id"]] *= 1.05


# ── retrieval ────────────────────────────────────────────────────

def engram_retrieve(query: str, entry: dict, config: Config,
                    top_k: int = 50, use_rerank: bool = False) -> list[dict]:
    sessions = entry["haystack_sessions"]
    sids = entry["haystack_session_ids"]
    dates = entry["haystack_dates"]

    # build corpora
    user_corpus = []
    asst_corpus = []
    for session, sid, date in zip(sessions, sids, dates):
        user_text = " ".join(t["content"] for t in session if t["role"] == "user")
        asst_text = " ".join(t["content"] for t in session if t["role"] == "assistant")
        if user_text.strip():
            user_corpus.append({"id": sid, "text": user_text, "timestamp": date})
        if asst_text.strip():
            asst_corpus.append({"id": sid, "text": asst_text, "timestamp": date})

    if not user_corpus:
        return []

    query_vec = embed_query(query, config.embedding_model)

    # embed + HNSW on user corpus
    user_texts = [doc["text"] for doc in user_corpus]
    doc_vecs = embed_documents(user_texts, config.embedding_model)

    ann = ANNIndex(dim=config.embedding_dim, m=16, ef_construction=100, ef_search=50)
    user_ids = [doc["id"] for doc in user_corpus]
    ann.build(user_ids, doc_vecs)
    dense_hits = ann.search(query_vec, top_k=min(top_k, len(user_corpus)))

    # BM25 on user corpus
    bm25_user = _simple_bm25(query, user_corpus, top_k=top_k)

    # BM25 on assistant corpus (no embed, weight 0.5)
    bm25_asst = _simple_bm25(query, asst_corpus, top_k=top_k) if asst_corpus else []

    # RRF fusion
    rrf_k = 60
    scores = {}
    for rank, (doc_id, _) in enumerate(dense_hits):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_user):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_asst):
        scores[doc_id] = scores.get(doc_id, 0) + 0.5 / (rrf_k + rank + 1)

    # temporal boost
    question_date = entry.get("question_date")
    if question_date:
        _apply_temporal_boost(scores, user_corpus + asst_corpus, question_date, query)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # cross-encoder rerank
    if use_rerank and ranked:
        from engram.embeddings import cross_encoder_rerank
        rerank_ids = [did for did, _ in ranked[:35]]
        id_to_user = {doc["id"]: doc["text"] for doc in user_corpus}
        id_to_asst = {doc["id"]: doc["text"] for doc in asst_corpus}
        rerank_texts = [(id_to_user.get(did, "") + " " + id_to_asst.get(did, "")).strip()
                        for did in rerank_ids]
        reranked = cross_encoder_rerank(query, rerank_texts, config.cross_encoder_model)
        new_ranked = [(rerank_ids[idx], score) for idx, score in reranked]

        # post-rerank temporal boost — cross-encoder wipes pre-rerank scores,
        # so we re-apply temporal signal to the final CE scores
        question_date = entry.get("question_date")
        window = _resolve_temporal_window(query, question_date) if question_date else None
        center, margin = window if window else (None, None)

        id_to_date = {}
        if window:
            for doc in user_corpus + asst_corpus:
                id_to_date[doc["id"]] = _parse_date(doc.get("timestamp", ""))

        fusion_alpha = getattr(config.retrieval, "rerank_fusion_alpha", 0.0)
        pre_rank_map = {did: i for i, (did, _) in enumerate(ranked[:len(rerank_ids)])}

        def _sig(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))

        boosted = []
        for did, score in new_ranked:
            d_date = id_to_date.get(did)
            t_boost = 0.0
            if window and d_date:
                dist = abs((d_date - center).days)
                kernel = math.exp(-0.5 * (dist / max(1.0, float(margin))) ** 2)
                t_boost = 5.0 * kernel

            calibrated_ce = _sig(score + t_boost)
            if fusion_alpha > 0.0 and did in pre_rank_map:
                pre_norm = 1.0 - (pre_rank_map[did] / len(rerank_ids))
                final_score = (1.0 - fusion_alpha) * calibrated_ce + fusion_alpha * pre_norm
            else:
                final_score = calibrated_ce
            boosted.append((did, final_score))

        new_ranked = sorted(boosted, key=lambda x: x[1], reverse=True)

        reranked_set = set(rerank_ids)
        for did, score in ranked:
            if did not in reranked_set:
                new_ranked.append((did, score))
        ranked = new_ranked

    # build results
    id_to_doc = {doc["id"]: doc for doc in user_corpus}
    for doc in asst_corpus:
        if doc["id"] not in id_to_doc:
            id_to_doc[doc["id"]] = doc

    return [{"corpus_id": did, "text": id_to_doc[did]["text"], "timestamp": id_to_doc[did]["timestamp"]}
            for did, _ in ranked if did in id_to_doc]


# ── metrics ──────────────────────────────────────────────────────

def compute_metrics(ranked_ids: list[str], correct_ids: set[str],
                    ks: list[int] = [1, 3, 5, 10, 30, 50]) -> dict:
    metrics = {}
    for k in ks:
        top_k_ids = set(ranked_ids[:k])
        recall_any = float(any(cid in top_k_ids for cid in correct_ids))
        recall_all = float(all(cid in top_k_ids for cid in correct_ids))

        relevances = [1.0 if rid in correct_ids else 0.0 for rid in ranked_ids[:k]]
        ideal = sorted([1.0 if rid in correct_ids else 0.0 for rid in ranked_ids], reverse=True)[:k]

        def dcg(rels):
            if not rels:
                return 0.0
            val = rels[0]
            for i, r in enumerate(rels[1:], 2):
                val += r / np.log2(i)
            return val

        idcg = dcg(ideal)
        ndcg_val = dcg(relevances) / idcg if idcg > 0 else 0.0
        metrics[f"recall_any@{k}"] = recall_any
        metrics[f"recall_all@{k}"] = recall_all
        metrics[f"ndcg_any@{k}"] = ndcg_val
    return metrics


# ── main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to longmemeval JSON")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rerank", action="store_true", help="Cross-encoder rerank top-35")
    parser.add_argument("--fusion-alpha", type=float, default=0.0, help="Late fusion weight for pre-rerank RRF (0.0=pure cross-encoder)")
    parser.add_argument("--output", help="Output JSONL path")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSONL")
    args = parser.parse_args()

    config = Config.load()
    if args.fusion_alpha > 0.0:
        config.retrieval.rerank_fusion_alpha = args.fusion_alpha

    print("warming up...")
    embed_query("warmup", config.embedding_model)
    if args.rerank:
        from engram.embeddings import cross_encoder_rerank
        cross_encoder_rerank("warmup", ["warmup"], config.cross_encoder_model)

    print(f"loading dataset: {args.dataset}")
    data = json.load(open(args.dataset))

    if args.limit > 0:
        data = data[:args.limit]

    eval_data = [d for d in data if not d["question_id"].endswith("_abs")]
    abs_data = [d for d in data if d["question_id"].endswith("_abs")]
    print(f"questions: {len(data)} total, {len(eval_data)} retrieval, {len(abs_data)} abstention (skipped)")

    mode = "v2_"
    if args.rerank:
        mode += "rerank_"
    output_path = args.output or f"engram_retrieval_{mode}results.jsonl"

    all_metrics = {f"{m}@{k}": [] for m in ["recall_any", "recall_all", "ndcg_any"] for k in [1, 3, 5, 10, 30, 50]}
    per_type_metrics = {}
    times = []

    processed_ids = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    qid = row["question_id"]
                    processed_ids.add(qid)
                    rm = row.get("retrieval_results", {}).get("metrics", {}).get("session", {})
                    for k, v in rm.items():
                        if k in all_metrics:
                            all_metrics[k].append(v)
                    qtype = row.get("question_type")
                    if qtype:
                        if qtype not in per_type_metrics:
                            per_type_metrics[qtype] = {k: [] for k in all_metrics}
                        for k, v in rm.items():
                            if k in per_type_metrics[qtype]:
                                per_type_metrics[qtype][k].append(v)
                except Exception:
                    pass
        print(f"resuming from {len(processed_ids)} already processed questions")

    out_mode = "a" if (args.resume and os.path.exists(output_path)) else "w"
    with open(output_path, out_mode) as out:
        for i, entry in enumerate(eval_data):
            if entry["question_id"] in processed_ids:
                continue
            t0 = time.time()

            correct_ids = set(entry.get("answer_session_ids", []))
            if not correct_ids:
                continue

            results = engram_retrieve(entry["question"], entry, config,
                                      top_k=50, use_rerank=args.rerank)

            elapsed = time.time() - t0
            times.append(elapsed)

            ranked_ids = [r["corpus_id"] for r in results]
            metrics = compute_metrics(ranked_ids, correct_ids)

            for k, v in metrics.items():
                all_metrics[k].append(v)

            qtype = entry["question_type"]
            if qtype not in per_type_metrics:
                per_type_metrics[qtype] = {k: [] for k in all_metrics}
            for k, v in metrics.items():
                per_type_metrics[qtype][k].append(v)

            log_entry = {
                "question_id": entry["question_id"],
                "question_type": entry["question_type"],
                "question": entry["question"],
                "answer": entry["answer"],
                "retrieval_results": {
                    "query": entry["question"],
                    "ranked_items": results,
                    "metrics": {"session": metrics},
                },
            }
            out.write(json.dumps(log_entry) + "\n")

            r5 = metrics["recall_any@5"]
            running_r5 = np.mean(all_metrics["recall_any@5"])
            if (i + 1) % 10 == 0 or i == 0 or r5 < 1.0:
                print(f"  [{i+1}/{len(eval_data)}] {elapsed:.1f}s | R@5={r5:.0f} running={running_r5:.3f} | {entry['question'][:55]}", flush=True)

    print()
    print("=" * 70)
    print(f"ENGRAM LongMemEval Results ({len(eval_data)} questions)")
    print("=" * 70)
    print()
    print(f"{'metric':<20} {'score':>8}")
    print("-" * 30)
    for k in ["recall_any@1", "recall_any@3", "recall_any@5", "recall_any@10",
              "recall_all@1", "recall_all@3", "recall_all@5", "recall_all@10",
              "ndcg_any@1", "ndcg_any@3", "ndcg_any@5", "ndcg_any@10"]:
        print(f"  {k:<18} {np.mean(all_metrics[k])*100:>7.1f}%")

    print()
    print("per question type:")
    print("-" * 70)
    for qtype, mets in sorted(per_type_metrics.items()):
        n = len(mets["recall_any@5"])
        r5 = np.mean(mets["recall_any@5"]) * 100
        r10 = np.mean(mets["recall_any@10"]) * 100
        print(f"  {qtype:<30} n={n:<4} R@5={r5:>5.1f}%  R@10={r10:>5.1f}%")

    print()
    avg_s = np.mean(times)
    total_s = sum(times)
    print(f"timing: avg={avg_s:.2f}s/question, total={total_s:.0f}s ({total_s/60:.1f}min)")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
