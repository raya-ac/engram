"""Manual public-interface smoke test with synthetic data and cached local models.

Run: python tests/dormant_smoke.py --work-dir /path/to/disposable/directory
No production config, database, service, embedding API or credentials are used.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

import yaml

from engram.config import Config
from engram.dormant import _terms
from engram.embeddings import embed_documents, embed_query, set_backend
from engram.store import Memory, Store

QUERY = "Is offline voice transcription possible?"
CONTENTS = {
    "current": "Offline voice transcription is possible with a compact local speech model on a laptop.",
    "blocker": "Speech recognition cannot run without internet access.",
    "irrelevant": "Bread dough needs water and a long fermentation before baking.",
    "forgotten": "Offline voice transcription previously needed a cloud service.",
    "superseded": "Offline voice transcription previously required an internet connection.",
}


def snapshot(cfg):
    store = Store(cfg)
    try:
        return {mid: (m.access_count, m.last_accessed, m.importance, m.forgotten, m.status)
                for mid in CONTENTS if (m := store.get_memory(mid))}
    finally:
        store.close()


def prepare(root, name, vectors, mode, threshold):
    folder = root / name
    folder.mkdir()
    cfg = Config(db_path=str(folder / "memory.db"), embedding_backend="sentence_transformers")
    cfg.ann.enabled = False
    cfg.dormant_recall.mode = mode
    cfg.dormant_recall.min_relevance = threshold
    store = Store(cfg)
    store.init_db()
    now = time.time()
    for i, (mid, content) in enumerate(CONTENTS.items()):
        created = now - (1 if mid == "current" else 180) * 86400
        store.save_memory(Memory(id=mid, content=content, embedding=vectors[i],
                                 created_at=created, last_accessed=created,
                                 importance=0.9, forgotten=mid == "forgotten",
                                 status="superseded" if mid == "superseded" else "active"))
    store.close()
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump({
        "db_path": cfg.db_path, "storage_backend": "sqlite",
        "embedding_backend": "sentence_transformers", "embedding_dim": 384,
        "embedding_model": cfg.embedding_model, "cross_encoder_model": cfg.cross_encoder_model,
        "ann": {"enabled": False},
        "dormant_recall": {"mode": mode, "min_relevance": threshold},
    }))
    return cfg, path


def decode_cli(text):
    # Libraries may emit startup text; accept only a complete trailing JSON value.
    for i, char in enumerate(text):
        if char in "[{":
            try:
                return json.loads(text[i:])
            except json.JSONDecodeError:
                pass
    raise AssertionError("CLI did not return JSON")


class MCP:
    def __init__(self, config, env, log):
        self.err = log.open("w")
        self.proc = subprocess.Popen([sys.executable, "-m", "engram", "--config", str(config), "serve", "--mcp"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                                     text=True, bufsize=1, env=env)
        self.seq = 0
        self.noise_lines = 0
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)

    def request(self, method, params=None):
        self.seq += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.seq, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if not self.selector.select(timeout=1):
                continue
            line = self.proc.stdout.readline()
            if not line:
                raise AssertionError("MCP process exited unexpectedly")
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                self.noise_lines += 1
                continue
            assert "error" not in response, response
            assert response["id"] == self.seq, response
            return response["result"]
        raise AssertionError("MCP response timed out")

    def call(self, name, arguments=None):
        result = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        return json.loads(result["content"][0]["text"])

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        self.selector.close()
        self.err.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ENGRAM_")}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    set_backend("sentence_transformers")
    vectors = embed_documents(list(CONTENTS.values()))
    similarities = {mid: float(score) for mid, score in zip(CONTENTS, vectors @ embed_query(QUERY))}
    assert not _terms(QUERY) & _terms(CONTENTS["blocker"])
    assert 0.70 <= similarities["blocker"] < 0.75
    assert similarities["irrelevant"] < 0.70
    report = {"synthetic_data_only": True, "embedding_model": "BAAI/bge-small-en-v1.5",
              "semantic_only_cosine": similarities["blocker"], "literal_overlap": 0,
              "default_threshold": 0.75, "isolated_pilot_threshold": 0.70}

    def cli(path, *arguments):
        proc = subprocess.run([sys.executable, "-m", "engram", "--config", str(path), *arguments],
                              env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr[-2000:]
        return decode_cli(proc.stdout)

    baseline_cfg, baseline_path = prepare(root, "cli-baseline", vectors, "off", 0.75)
    baseline = cli(baseline_path, "search", QUERY, "-k", "1", "--json")
    cfg, path = prepare(root, "cli-shadow", vectors, "shadow", 0.70)
    before = snapshot(cfg)
    current = cli(path, "search", QUERY, "-k", "1", "--json")
    assert [r["id"] for r in current] == [r["id"] for r in baseline] == ["current"]
    rows = cli(path, "dormant", "review")
    assert len(rows) == 1 and rows[0]["memory_id"] == "blocker"
    eid = rows[0]["id"]
    inspected = cli(path, "dormant", "inspect", eid)
    assert "embedding-only" in inspected["connection"]
    assert cli(path, "dormant", "feedback", eid, "useful")["feedback"] == "useful"
    after = snapshot(cfg)
    assert all(after[mid] == before[mid] for mid in CONTENTS if mid != "current")
    assert after["current"][0] == 1
    report["cli"] = {"ordinary_ids_unchanged": True, "candidate": "blocker", "no_candidate_reinforcement": True,
                     "review_inspect_feedback": "passed"}

    cfg, path = prepare(root, "default-threshold", vectors, "shadow", 0.75)
    cli(path, "search", QUERY, "-k", "1", "--json")
    assert cli(path, "dormant", "review")[0]["outcome"] == "none"
    report["default_threshold_result"] = "abstained on this semantic-only fixture"

    ordinary = None
    for mode in ("off", "shadow"):
        cfg, path = prepare(root, f"mcp-{mode}", vectors, mode, 0.70)
        before = snapshot(cfg)
        mcp = MCP(path, env, root / f"mcp-{mode}.stderr.log")
        try:
            mcp.request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "dormant-smoke", "version": "1"}})
            names = {tool["name"] for tool in mcp.request("tools/list")["tools"]}
            assert {"dormant_review", "dormant_inspect", "dormant_feedback"} <= names
            result = mcp.call("recall", {"query": QUERY, "top_k": 1})
            ids = [row["id"] for row in result]
            assert ids == ["current"]
            if mode == "off":
                ordinary = ids
            else:
                assert ids == ordinary
                rows = mcp.call("dormant_review")
                assert len(rows) == 1 and rows[0]["memory_id"] == "blocker"
                eid = rows[0]["id"]
                assert "embedding-only" in mcp.call("dormant_inspect", {"event_id": eid})["connection"]
                assert mcp.call("dormant_feedback", {"event_id": eid, "category": "dismissed"})["feedback"] == "dismissed"
                assert [r["id"] for r in mcp.call("recall", {"query": QUERY, "top_k": 1})] == ids
                assert mcp.call("dormant_review")[0]["outcome"] == "none"
                after = snapshot(cfg)
                assert all(after[mid] == before[mid] for mid in CONTENTS if mid != "current")
                assert after["current"][0] == 2
                report["mcp"] = {"initialize_tools_list_recall_review_inspect_feedback": "passed",
                                 "cooldown_on_cached_recall": True, "ordinary_ids_unchanged": True,
                                 "no_candidate_reinforcement": True}
            report[f"mcp_{mode}_non_json_stdout_lines"] = mcp.noise_lines
        finally:
            mcp.close()
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
