"""Small synthetic acceptance check through ordinary, confidence-gated search.

This is an engineering smoke evaluation, not a broad accuracy benchmark.
Fixtures are independent corpora; gold IDs are consumed only after retrieval.
The held-out split requires a previously frozen source manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import wraps
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time


CORE_FILES = (
    "engram/config.py", "engram/embeddings.py", "engram/retrieval.py",
    "engram/retrieval_explain.py", "engram/rerank_scoring.py",
    "engram/rerank_selection.py", "engram/rerank_passages.py",
    "engram/temporal.py", "engram/store.py", "engram/ann_index.py", "engram/hopfield.py",
)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_manifest(root):
    hashes = {name: sha256_file(root / name) for name in CORE_FILES}
    return {"files": hashes, "sha256": fingerprint(hashes)}


def read_fixture(path):
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("split") not in {"dev", "holdout"}:
        raise ValueError("unsupported fixture schema or split")
    seen = set()
    for case in data["cases"]:
        if not isinstance(case["id"], str) or case["id"] in seen or not case["query"].strip():
            raise ValueError("case IDs must be unique and questions nonempty")
        seen.add(case["id"])
        ids = [memory["id"] for memory in case["memories"]]
        if not ids or len(set(ids)) != len(ids) or any(not memory["content"].strip() for memory in case["memories"]):
            raise ValueError("each corpus needs unique IDs and nonempty memories")
        if len(set(case["gold_ids"])) != len(case["gold_ids"]) or not set(case["gold_ids"]) <= set(ids):
            raise ValueError("gold IDs must uniquely identify memories in that case")
    return data


def summary(rows, top_k):
    answerable = [row for row in rows if row["answerable"]]
    absent = [row for row in rows if not row["answerable"]]
    accepted = sum(len(row["returned_ids"]) for row in rows)
    correct = sum(len(row["matched_gold_ids"]) for row in rows)
    answerable_accepted = sum(len(row["returned_ids"]) for row in answerable)
    latencies = [row["search_seconds"] for row in rows]
    divide = lambda numerator, denominator: numerator / denominator if denominator else None
    return {
        "completed_cases": len(rows), "top_k": top_k,
        "answerable_cases": len(answerable), "unanswerable_cases": len(absent),
        "answerable_hits": sum(bool(row["matched_gold_ids"]) for row in answerable),
        "answerable_recall_any": divide(sum(bool(row["matched_gold_ids"]) for row in answerable), len(answerable)),
        "gold_memory_recall": divide(correct, sum(len(row["gold_ids"]) for row in answerable)),
        "accepted_memories": accepted, "gold_memories_accepted": correct,
        "accepted_memory_precision": divide(correct, accepted),
        "answerable_accepted_memory_precision": divide(correct, answerable_accepted),
        "non_gold_memories_accepted": accepted - correct,
        "unanswerable_wrong_accept_cases": sum(bool(row["returned_ids"]) for row in absent),
        "unanswerable_wrong_accept_memories": sum(len(row["returned_ids"]) for row in absent),
        "unanswerable_abstentions": sum(not row["returned_ids"] for row in absent),
        "unanswerable_abstention_rate": divide(sum(not row["returned_ids"] for row in absent), len(absent)),
        "search_seconds_total": sum(latencies),
        "search_seconds_median": statistics.median(latencies) if latencies else None,
        "search_seconds_max": max(latencies) if latencies else None,
    }


def cached_model_manifest(model_id):
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(model_id, local_files_only=True))
    relevant = (".json", ".txt", ".model", ".safetensors", ".bin")
    hashes = {str(file.relative_to(path)): sha256_file(file)
              for file in sorted(path.rglob("*")) if file.is_file() and file.suffix in relevant}
    return path, {"model_id": model_id, "snapshot_revision": path.name,
                  "files": hashes, "sha256": fingerprint(hashes)}


def write_progress(stream, report):
    stream.seek(0)
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write("\n")
    stream.truncate()
    stream.flush()


def run(args, fixture, sources):
    # All runtime setup is local and CPU-only. Do not call Config.load: this
    # fixture must never inherit the user's credentials, database or overrides.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(args.source_root))
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from engram import embeddings, retrieval
    from engram.config import Config
    from engram.store import Memory, Store
    if Path(retrieval.__file__).resolve() != args.source_root / "engram/retrieval.py":
        raise RuntimeError("loaded retrieval module does not match --source-root")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    config = Config()
    config.storage_backend = "sqlite"
    config.embedding_model, config.cross_encoder_model = EMBEDDING_MODEL, RERANK_MODEL
    config.embedding_backend = "sentence_transformers"
    config.retrieval.top_k = args.top_k
    config.retrieval.min_confidence = 0.6
    config.dormant_recall.mode = "off"
    config.validate()
    settings = asdict(config)
    settings.update(db_path="<isolated temporary SQLite database>")
    settings["ann"]["index_path"] = "<isolated temporary HNSW index>"
    settings["llm"]["api_key"] = ""
    embedding_path, embedding_provenance = cached_model_manifest(EMBEDDING_MODEL)
    rerank_path, rerank_provenance = cached_model_manifest(RERANK_MODEL)
    load_start = time.perf_counter()
    embeddings.set_backend("sentence_transformers")
    embeddings.set_default_model(EMBEDDING_MODEL)
    embeddings._bi_encoder = SentenceTransformer(str(embedding_path), device="cpu")
    embeddings._cross_encoders[RERANK_MODEL] = CrossEncoder(str(rerank_path), device="cpu")
    model_load_seconds = time.perf_counter() - load_start
    model_devices = {"embedding": str(embeddings._bi_encoder.device),
                     "reranker": str(embeddings._cross_encoders[RERANK_MODEL].model.device)}
    if set(model_devices.values()) != {"cpu"}:
        raise RuntimeError("this runner requires both models on CPU")
    meta = {
        "fixture": str(args.fixture), "fixture_sha256": sha256_file(args.fixture),
        "split": fixture["split"], "suite": fixture["suite"],
        "source_root": str(args.source_root), "sources": sources,
        "runner_sha256": sha256_file(__file__), "config": settings,
        "models": {"embedding": embedding_provenance, "reranker": rerank_provenance},
        "runtime": {"python": sys.version, "platform": platform.platform(), "device": model_devices,
                    "torch_threads": torch.get_num_threads(), "packages": {
                        name: importlib.metadata.version(name) for name in
                        ("torch", "sentence-transformers", "transformers", "numpy", "hnswlib")}},
        "model_load_seconds": model_load_seconds,
        "reference_date": fixture["reference_date"], "confidence_filter": True,
        "debug": False, "rerank": True, "result_cache_reused": False,
        "instrumentation": "Read-only wrappers observe channel counts and model outputs; arguments, scores and ordering are unchanged.",
        "scope": "Small authored synthetic production-path smoke test. Precision counts returned non-gold memories as errors; it does not measure generated-answer accuracy. Not evidence of broad generalization.",
    }
    meta["fingerprint"] = fingerprint({key: meta[key] for key in
                                       ("fixture_sha256", "sources", "runner_sha256", "config", "models")})
    report = {"schema_version": 1, "status": "running", "metadata": meta, "cases": [], "summary": {}}
    state = {}
    for channel in ("dense", "bm25", "graph", "hopfield"):
        original = getattr(retrieval, f"_{channel}_search")
        def observe_channel(*positional, _original=original, _channel=channel, **keywords):
            pairs = _original(*positional, **keywords)
            state["channels"][_channel] = [{"id": mid, "score": float(score)} for mid, score in pairs]
            return pairs
        setattr(retrieval, f"_{channel}_search", wraps(original)(observe_channel))
    original_rerank = retrieval.cross_encoder_rerank
    @wraps(original_rerank)
    def observe_rerank(query, documents, model):
        scored = original_rerank(query, documents, model)
        by_index = dict(scored)
        inputs = []
        for index, document in enumerate(documents):
            origins = [memory["id"] for memory in state["memories"] if document in memory["content"]]
            inputs.append({"document_sha256": hashlib.sha256(document.encode()).hexdigest(),
                           "source_ids": origins, "words": len(document.split()),
                           "raw_score": float(by_index[index]),
                           "full_document": any(document == memory["content"] for memory in state["memories"])})
        state["model_calls"].append({"query_sha256": hashlib.sha256(query.encode()).hexdigest(), "inputs": inputs})
        return scored
    retrieval.cross_encoder_rerank = observe_rerank
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        write_progress(output, report)
        try:
            with tempfile.TemporaryDirectory(prefix="production-retrieval-", dir=args.work_dir) as temporary:
                for case in fixture["cases"]:
                    case_root = Path(temporary) / case["id"]
                    case_root.mkdir()
                    config.db_path = str(case_root / "memory.sqlite")
                    config.ann.index_path = str(case_root / "hnsw.index")
                    state.update(channels={}, model_calls=[], memories=case["memories"])
                    store = Store(config)
                    try:
                        ingest_start = time.perf_counter()
                        store.init_db()
                        documents = [memory["content"] for memory in case["memories"]]
                        vectors = embeddings.embed_documents(documents, EMBEDDING_MODEL)
                        for memory, vector in zip(case["memories"], vectors):
                            store.save_memory(Memory(id=memory["id"], content=memory["content"],
                                memory_type="fact", layer="semantic", importance=0.5, embedding=vector,
                                fact_date="2026-06-20", created_at=1781913600.0, last_accessed=1781913600.0))
                        store.init_ann_index(background=False)
                        indexing_seconds = time.perf_counter() - ingest_start
                        search_start = time.perf_counter()
                        # Production gate and ordinary access/cache behavior remain active.
                        results = retrieval.search(case["query"], store, config, top_k=args.top_k,
                                                   rerank=True, debug=False,
                                                   reference_date=fixture["reference_date"])
                        search_seconds = time.perf_counter() - search_start
                        returned_ids = [result.memory.id for result in results]
                        # Gold annotations are used only below this point, for metrics.
                        gold_ids = case["gold_ids"]
                        matched = [mid for mid in returned_ids if mid in gold_ids]
                        row = {
                            "id": case["id"], "category": case["category"], "answerable": bool(gold_ids),
                            "case_sha256": fingerprint(case), "gold_ids": gold_ids,
                            "returned_ids": returned_ids, "matched_gold_ids": matched,
                            "non_gold_returned_ids": [mid for mid in returned_ids if mid not in gold_ids],
                            "results": [{"id": result.memory.id, "score": result.score, "sources": result.sources}
                                        for result in results],
                            "corpus_memories": len(case["memories"]),
                            "retrieval_channel_counts": {name: len(values) for name, values in state["channels"].items()},
                            "considered_unique_ids": list(dict.fromkeys(item["id"] for values in state["channels"].values() for item in values)),
                            "model_calls": state["model_calls"], "ann_ready": bool(store.ann_index and store.ann_index.ready),
                            "indexing_seconds": indexing_seconds, "search_seconds": search_seconds,
                            "ordinary_accesses_recorded": sum(store.get_memory(mid).access_count for mid in returned_ids),
                        }
                    finally:
                        store.close()
                    report["cases"].append(row)
                    report["summary"] = summary(report["cases"], args.top_k)
                    write_progress(output, report)
                    print(f"{len(report['cases'])}/{len(fixture['cases'])} {case['id']}: accepted={len(returned_ids)} gold_hits={len(matched)} seconds={search_seconds:.2f}", flush=True)
            if source_manifest(args.source_root) != sources or sha256_file(args.fixture) != meta["fixture_sha256"]:
                raise RuntimeError("source or fixture changed during evaluation")
            report["status"] = "complete"
        except BaseException:
            report["status"] = "failed"
            write_progress(output, report)
            raise
        write_progress(output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--freeze-source", type=Path)
    parser.add_argument("--frozen-source", type=Path)
    args = parser.parse_args()
    args.source_root, args.fixture = args.source_root.resolve(), args.fixture.resolve()
    fixture = read_fixture(args.fixture)
    sources = source_manifest(args.source_root)
    if args.freeze_source:
        args.freeze_source.parent.mkdir(parents=True, exist_ok=True)
        with args.freeze_source.open("x") as output:
            json.dump({"created_at": datetime.now(timezone.utc).isoformat(), "source_root": str(args.source_root),
                       "sources": sources}, output, indent=2)
    if args.validate_only:
        print(json.dumps({"split": fixture["split"], "cases": len(fixture["cases"]),
                          "fixture_sha256": sha256_file(args.fixture), "sources": sources["sha256"]}))
        return
    if not args.output or not args.work_dir or args.top_k < 1:
        parser.error("evaluation requires --output, --work-dir and positive --top-k")
    if args.output.exists():
        parser.error("output already exists; each evaluation must use a fresh path")
    if fixture["split"] == "holdout":
        if args.freeze_source or not args.frozen_source:
            parser.error("held-out evaluation requires an earlier --frozen-source manifest")
        frozen = json.loads(args.frozen_source.read_text())
        if frozen["sources"] != sources:
            parser.error("held-out source differs from the frozen candidate")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    run(args, fixture, sources)


if __name__ == "__main__":
    main()
