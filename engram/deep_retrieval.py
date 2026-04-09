"""Learned MLP reranker trained on access patterns.

Trains on (query_embedding, memory_embedding, features) → relevance signal
derived from actual access patterns. Which memories get accessed after being
returned in search results? That signal teaches the reranker what's useful
vs what's just semantically similar.

The MLP takes a feature vector:
- cosine similarity between query and memory
- memory importance score
- memory access count (log-scaled)
- memory age (days, log-scaled)
- memory layer (one-hot encoded)
- memory retention score

And learns to predict whether a memory will be accessed (clicked/used)
after being returned in search results.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from engram.store import Store


class DeepReranker:
    """2-layer MLP reranker trained on access patterns."""

    FEATURE_DIM = 10  # cosine_sim, importance, log_access, log_age, 5 layers, retention

    def __init__(self, hidden_dim: int = 16, model_path: Path | None = None):
        self.hidden_dim = hidden_dim
        self.model_path = model_path

        # weights initialized with Xavier/Glorot
        scale1 = math.sqrt(2.0 / (self.FEATURE_DIM + hidden_dim))
        scale2 = math.sqrt(2.0 / (hidden_dim + 1))

        self.w1 = np.random.randn(self.FEATURE_DIM, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = np.random.randn(hidden_dim, 1).astype(np.float32) * scale2
        self.b2 = np.zeros(1, dtype=np.float32)

        self._trained = False

        if model_path and model_path.exists():
            self.load(model_path)

    def extract_features(self, cosine_sim: float, memory: dict) -> np.ndarray:
        """Extract feature vector for a (query, memory) pair.

        memory is a dict with: importance, access_count, created_at, layer
        """
        importance = memory.get("importance", 0.5)
        access_count = memory.get("access_count", 0)
        created_at = memory.get("created_at", time.time())
        layer = memory.get("layer", "episodic")

        age_days = max(0.01, (time.time() - created_at) / 86400)

        # one-hot layer encoding
        layers = ["working", "episodic", "semantic", "procedural", "codebase"]
        layer_vec = [1.0 if layer == l else 0.0 for l in layers]

        # retention estimate (simplified — no config dependency)
        retention = math.exp(-0.693 * age_days / 30)

        features = [
            cosine_sim,
            importance,
            math.log(1 + access_count) / 5.0,  # normalized log access
            math.log(1 + age_days) / 5.0,       # normalized log age
            *layer_vec,
            retention,
        ]
        return np.array(features, dtype=np.float32)

    def predict(self, features: np.ndarray) -> float:
        """Forward pass: features → relevance score."""
        # layer 1: ReLU
        h = features @ self.w1 + self.b1
        h = np.maximum(0, h)
        # layer 2: sigmoid
        out = h @ self.w2 + self.b2
        return float(1.0 / (1.0 + np.exp(-np.clip(out[0], -20, 20))))

    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Batch forward pass."""
        h = features_batch @ self.w1 + self.b1
        h = np.maximum(0, h)
        out = h @ self.w2 + self.b2
        return 1.0 / (1.0 + np.exp(-np.clip(out.flatten(), -20, 20)))

    def rerank(self, candidates: list[dict], query_embedding: np.ndarray,
               memory_embeddings: dict[str, np.ndarray]) -> list[dict]:
        """Rerank candidates using the learned model.

        candidates: list of dicts with 'id', 'score', and memory fields
        memory_embeddings: {memory_id: embedding_vector}

        Returns candidates sorted by learned relevance score.
        """
        if not self._trained:
            return candidates  # pass-through if no training data

        features_batch = []
        for c in candidates:
            mid = c["id"]
            emb = memory_embeddings.get(mid)
            if emb is not None:
                cos_sim = float(np.dot(query_embedding, emb))
            else:
                cos_sim = c.get("score", 0.5)
            features_batch.append(self.extract_features(cos_sim, c))

        features_arr = np.array(features_batch, dtype=np.float32)
        scores = self.predict_batch(features_arr)

        for c, s in zip(candidates, scores):
            c["deep_score"] = float(s)

        return sorted(candidates, key=lambda x: x["deep_score"], reverse=True)

    def train(self, store: Store, lr: float = 0.01, epochs: int = 50,
              min_samples: int = 20) -> dict:
        """Train on access patterns from the access_log table.

        Positive examples: memories that were accessed after being returned.
        Negative examples: memories that were returned but not accessed after.

        We approximate this by looking at access_log entries and comparing
        access counts to retrieval frequency.
        """
        # collect training data from access_log
        rows = store.conn.execute("""
            SELECT m.id, m.importance, m.access_count, m.created_at, m.layer,
                   m.embedding, COUNT(al.id) as log_count
            FROM memories m
            JOIN access_log al ON al.memory_id = m.id
            WHERE m.forgotten = 0 AND m.embedding IS NOT NULL
            GROUP BY m.id
            HAVING log_count >= 2
            ORDER BY log_count DESC
            LIMIT 500
        """).fetchall()

        if len(rows) < min_samples:
            return {"status": "insufficient_data", "samples": len(rows),
                    "min_required": min_samples}

        # build feature matrix and labels
        # label = normalized access count (how useful this memory proved to be)
        max_access = max(r["log_count"] for r in rows)
        features_list = []
        labels = []

        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            cos_sim = float(np.linalg.norm(emb))  # self-similarity (1.0 for normalized)
            mem_dict = {
                "importance": row["importance"],
                "access_count": row["access_count"],
                "created_at": row["created_at"],
                "layer": row["layer"],
            }
            features_list.append(self.extract_features(cos_sim, mem_dict))
            # label: sigmoid of log access count, normalized
            labels.append(row["log_count"] / max_access)

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels, dtype=np.float32).reshape(-1, 1)

        # mini-batch SGD with binary cross-entropy
        n = len(X)
        batch_size = min(32, n)
        stats = {"epochs": epochs, "samples": n, "final_loss": 0.0}

        for epoch in range(epochs):
            # shuffle
            perm = np.random.permutation(n)
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            epoch_loss = 0.0

            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                bs = len(X_batch)

                # forward
                h = X_batch @ self.w1 + self.b1
                h_relu = np.maximum(0, h)
                out = h_relu @ self.w2 + self.b2
                pred = 1.0 / (1.0 + np.exp(-np.clip(out, -20, 20)))

                # binary cross-entropy loss
                eps = 1e-7
                loss = -np.mean(y_batch * np.log(pred + eps) +
                                (1 - y_batch) * np.log(1 - pred + eps))
                epoch_loss += loss * bs

                # backward
                d_out = (pred - y_batch) / bs  # (bs, 1)
                d_w2 = h_relu.T @ d_out
                d_b2 = np.sum(d_out, axis=0)

                d_h = d_out @ self.w2.T
                d_h[h <= 0] = 0  # ReLU gradient

                d_w1 = X_batch.T @ d_h
                d_b1 = np.sum(d_h, axis=0)

                # update
                self.w1 -= lr * d_w1
                self.b1 -= lr * d_b1
                self.w2 -= lr * d_w2
                self.b2 -= lr * d_b2

            epoch_loss /= n
            stats["final_loss"] = float(epoch_loss)

        self._trained = True
        stats["status"] = "trained"

        # auto-save if path configured
        if self.model_path:
            self.save(self.model_path)

        return stats

    def save(self, path: Path):
        """Save model weights to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path),
                 w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2,
                 trained=np.array([self._trained]))

    def load(self, path: Path):
        """Load model weights from disk."""
        data = np.load(str(path))
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self._trained = bool(data["trained"][0])

    @property
    def is_trained(self) -> bool:
        return self._trained
