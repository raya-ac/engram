"""LongMemEval retrieval benchmark for engram.

HNSW + BM25 + RRF per question, ~1.6s/question, ~12min total.

Usage:
    python run_engram.py data/longmemeval_s_cleaned.json [--limit N] [--rerank]
"""

import argparse
import json
import math
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
    query_tokens = query.lower().split()
    if not query_tokens:
        return []

    N = len(corpus)
    avgdl = sum(len(doc["text"].split()) for doc in corpus) / max(N, 1)
    k1, b = 1.5, 0.75

    df = Counter()
    doc_tokens = []
    for doc in corpus:
        tokens = doc["text"].lower().split()
        doc_tokens.append(Counter(tokens))
        for t in set(tokens):
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

def _apply_temporal_boost(scores: dict, corpus: list[dict], question_date: str):
    m = _DATE_RE.match(question_date)
    if not m:
        return
    q_days = int(m.group(1)) * 365 + int(m.group(2)) * 30 + int(m.group(3))

    for doc in corpus:
        dm = _DATE_RE.match(doc.get("timestamp", ""))
        if not dm or doc["id"] not in scores:
            continue
        d_days = int(dm.group(1)) * 365 + int(dm.group(2)) * 30 + int(dm.group(3))
        diff = q_days - d_days
        if diff < 0:
            scores[doc["id"]] *= 0.95
        elif diff < 7:
            scores[doc["id"]] *= 1.15
        elif diff < 30:
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
        _apply_temporal_boost(scores, user_corpus + asst_corpus, question_date)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # cross-encoder rerank
    if use_rerank and ranked:
        from engram.embeddings import cross_encoder_rerank
        rerank_ids = [did for did, _ in ranked[:20]]
        id_to_user = {doc["id"]: doc["text"] for doc in user_corpus}
        id_to_asst = {doc["id"]: doc["text"] for doc in asst_corpus}
        rerank_texts = [(id_to_user.get(did, "") + " " + id_to_asst.get(did, "")).strip()
                        for did in rerank_ids]
        reranked = cross_encoder_rerank(query, rerank_texts, config.cross_encoder_model)
        new_ranked = [(rerank_ids[idx], score) for idx, score in reranked]
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
    parser.add_argument("--rerank", action="store_true", help="Cross-encoder rerank top-20")
    parser.add_argument("--output", help="Output JSONL path")
    args = parser.parse_args()

    config = Config.load()

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

    with open(output_path, "w") as out:
        for i, entry in enumerate(eval_data):
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
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(eval_data)}] {elapsed:.1f}s | R@5={r5:.0f} running={running_r5:.3f} | {entry['question'][:55]}")

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
