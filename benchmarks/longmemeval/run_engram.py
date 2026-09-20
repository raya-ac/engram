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
from engram.embeddings import RERANKER_BACKENDS, embed_query, embed_documents
from engram.rerank_scoring import rerank_score
from engram.rerank_selection import preserve_prior_leader
from engram.rerank_passages import rerank_with_passages
from engram.ann_index import ANNIndex
from benchmarks.longmemeval.evaluation import (
    compute_metrics, build_run_metadata, load_resume_rows, write_run_metadata,
)
from engram.temporal import _parse_date, _resolve_temporal_window, _temporal_content_query


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


def _extract_phrases(q: str) -> list[str]:
    words = [w for w in re.findall(r"[a-zA-Z0-9]+", q.lower()) if w not in _STOP_WORDS]
    return [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]


# ── temporal boost ───────────────────────────────────────────────

def _apply_temporal_boost(scores: dict, corpus: list[dict], question_date: str,
                          query: str = ""):
    q_date = _parse_date(question_date)
    if not q_date:
        return

    # try to resolve a specific temporal window from the query
    window = _resolve_temporal_window(query, question_date) if query else None

    seen = set()
    for doc in corpus:
        if doc["id"] in seen:
            continue
        seen.add(doc["id"])
        d_date = _parse_date(doc.get("timestamp", ""))
        if not d_date or doc["id"] not in scores:
            continue

        diff_days = (q_date - d_date).days

        if window:
            center, margin = window
            dist = abs((d_date - center).days)
            if dist <= margin:
                scores[doc["id"]] *= 2.5
            elif diff_days < 0:
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
                    top_k: int = 50, use_rerank: bool = False,
                    result_k: int = 5) -> list[dict]:
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

    score_details = {}
    # cross-encoder rerank
    if use_rerank and ranked:
        from engram.embeddings import cross_encoder_rerank
        rerank_ids = [did for did, _ in ranked[:35]]
        id_to_user = {doc["id"]: doc["text"] for doc in user_corpus}
        id_to_asst = {doc["id"]: doc["text"] for doc in asst_corpus}
        rerank_texts = [(id_to_user.get(did, "") + " " + id_to_asst.get(did, "")).strip()
                        for did in rerank_ids]
        if config.retrieval.rerank_passage_fallback:
            reranked, passage_details = rerank_with_passages(
                query, rerank_texts, config.cross_encoder_model,
                rerank_fn=cross_encoder_rerank,
                confidence_floor=config.retrieval.rerank_passage_floor,
                passage_query=_temporal_content_query(query, question_date),
            )
        else:
            reranked = cross_encoder_rerank(query, rerank_texts, config.cross_encoder_model)
            passage_details = {}
        passage_by_id = {rerank_ids[index]: trace for index, trace in passage_details.items()}
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

        boosted = []
        for did, score in new_ranked:
            d_date = id_to_date.get(did)
            t_boost = 0.0
            if window and d_date:
                dist = abs((d_date - center).days)
                if dist <= margin:
                    t_boost = 5.0

            final_score = rerank_score(score, prior_rank=pre_rank_map[did],
                                       fusion_alpha=fusion_alpha, temporal_boost=t_boost,
                                       normalized=config.cross_encoder_model in RERANKER_BACKENDS)
            boosted.append((did, final_score))
            score_details[did] = {**passage_by_id.get(did, {}), "cross_encoder": score,
                                  "prior_rank": pre_rank_map[did] + 1,
                                  "temporal_boost": t_boost}

        new_ranked = sorted(boosted, key=lambda x: x[1], reverse=True)
        if config.retrieval.preserve_prior_candidate:
            by_id = dict(new_ranked)
            order = preserve_prior_leader(list(by_id), rerank_ids, result_limit=result_k)
            new_ranked = [(did, by_id[did]) for did in order]

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

    return [{"corpus_id": did, "text": id_to_doc[did]["text"],
             "timestamp": id_to_doc[did]["timestamp"], "score": score,
             "sources": score_details.get(did, {})}
            for did, score in ranked if did in id_to_doc]


# ── main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to longmemeval JSON")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rerank", action="store_true", help="Cross-encoder rerank top-35")
    parser.add_argument("--fusion-alpha", type=float, default=None, help="Optional prior rank blend (default: configured value)")
    parser.add_argument("--output", help="Output JSONL path")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSONL")
    parser.add_argument("--config", help="Explicit config file; otherwise use package defaults")
    parser.add_argument("--reranker-model", help="Explicit reranker model")
    parser.add_argument("--embedding-backend", default="sentence_transformers", choices=["sentence_transformers", "mlx"])
    parser.add_argument("--question-id", action="append", default=[], help="Run only selected question IDs")
    parser.add_argument("--result-k", type=int, default=5, help="Requested result window for hybrid coverage (default: 5)")
    parser.add_argument("--no-preserve-prior", action="store_true", help="Disable hybrid candidate coverage for comparison")
    parser.add_argument("--no-passage-fallback", action="store_true", help="Disable focused excerpts on low-scoring long documents")
    parser.add_argument("--passage-floor", type=float, default=None, help="Override the local excerpt activation floor (0 to 1)")
    args = parser.parse_args()
    if args.fusion_alpha is not None and not 0.0 <= args.fusion_alpha <= 1.0:
        parser.error("--fusion-alpha must be between 0 and 1")
    if args.passage_floor is not None and not 0.0 <= args.passage_floor <= 1.0:
        parser.error("--passage-floor must be between 0 and 1")
    if args.result_k < 1:
        parser.error("--result-k must be positive")

    config = Config.load(args.config) if args.config else Config()
    if args.reranker_model:
        config.cross_encoder_model = args.reranker_model
    if args.no_preserve_prior:
        config.retrieval.preserve_prior_candidate = False
    if args.no_passage_fallback:
        config.retrieval.rerank_passage_fallback = False
    from engram.embeddings import set_backend
    set_backend(args.embedding_backend)
    if args.fusion_alpha is not None:
        config.retrieval.rerank_fusion_alpha = args.fusion_alpha
    if args.passage_floor is not None:
        config.retrieval.rerank_passage_floor = args.passage_floor

    print(f"loading dataset: {args.dataset}")
    data = json.load(open(args.dataset))

    if args.question_id:
        selected = set(args.question_id)
        missing = selected - {row["question_id"] for row in data}
        if missing:
            parser.error(f"Unknown question IDs: {sorted(missing)}")
        data = [row for row in data if row["question_id"] in selected]
    if args.limit > 0:
        data = data[:args.limit]

    eval_data = [d for d in data if not d["question_id"].endswith("_abs")]
    abs_data = [d for d in data if d["question_id"].endswith("_abs")]
    print(f"questions: {len(data)} total, {len(eval_data)} retrieval, {len(abs_data)} abstention (skipped)")

    mode = "v2_"
    if args.rerank:
        mode += "rerank_"
    output_path = args.output or f"engram_retrieval_{mode}results.jsonl"
    if not args.resume and (Path(output_path).exists() or Path(str(output_path) + ".metadata.json").exists()):
        parser.error("Output or metadata already exists; use --resume or a new output filename")

    all_metrics = {f"{m}@{k}": [] for m in ["recall_any", "recall_all", "ndcg_any"] for k in [1, 3, 5, 10, 30, 50]}
    per_type_metrics = {}
    times = []

    root = Path(__file__).resolve().parents[2]
    sources = ["benchmarks/longmemeval/run_engram.py", "benchmarks/longmemeval/evaluation.py",
               "engram/embeddings.py", "engram/ann_index.py", "engram/config.py",
               "engram/temporal.py", "engram/rerank_scoring.py", "engram/rerank_selection.py",
               "engram/rerank_passages.py"]
    metadata = build_run_metadata(args.dataset, source_paths={p: root / p for p in sources},
        embedding_model=config.embedding_model, cross_encoder_model=config.cross_encoder_model,
        config={"embedding_backend": args.embedding_backend, "embedding_dim": config.embedding_dim,
                "rerank": args.rerank, "fusion_alpha": config.retrieval.rerank_fusion_alpha,
                "top_k": 50, "rerank_candidates": 35, "limit": args.limit,
                "result_k": args.result_k, "preserve_prior_candidate": config.retrieval.preserve_prior_candidate,
                "rerank_passage_fallback": config.retrieval.rerank_passage_fallback,
                "passage_confidence_floor": config.retrieval.rerank_passage_floor,
                "confidence_filter": False,
                "question_ids": sorted(args.question_id)})
    resuming_existing = args.resume and Path(output_path).is_file()
    resumed = load_resume_rows(output_path, metadata) if args.resume else []
    processed_ids = set()
    for row in resumed:
        processed_ids.add(row["question_id"])
        rm = row["retrieval_results"]["metrics"]["session"]
        qtype = row["question_type"]
        per_type_metrics.setdefault(qtype, {key: [] for key in all_metrics})
        for key in all_metrics:
            all_metrics[key].append(rm[key])
            per_type_metrics[qtype][key].append(rm[key])
    if resumed:
        print(f"resuming from {len(processed_ids)} verified questions")
    print(f"models: {config.embedding_model} / {config.cross_encoder_model}")
    print(f"run fingerprint: {metadata['fingerprint']}")
    print("warming up...", flush=True)
    embed_query("warmup", config.embedding_model)
    if args.rerank:
        from engram.embeddings import cross_encoder_rerank
        cross_encoder_rerank("warmup", ["warmup"], config.cross_encoder_model)
    out_mode = "a" if resuming_existing else "x"
    with open(output_path, out_mode) as out:
        if out_mode == "x":
            write_run_metadata(output_path, metadata)
        for i, entry in enumerate(eval_data):
            if entry["question_id"] in processed_ids:
                continue
            t0 = time.time()

            correct_ids = set(entry.get("answer_session_ids", []))
            if not correct_ids:
                continue

            results = engram_retrieve(entry["question"], entry, config,
                                      top_k=50, use_rerank=args.rerank, result_k=args.result_k)

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
            out.flush()

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
    avg_s = np.mean(times) if times else 0.0
    total_s = sum(times)
    print(f"timing: avg={avg_s:.2f}s/question, total={total_s:.0f}s ({total_s/60:.1f}min)")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
