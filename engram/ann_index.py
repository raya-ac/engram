"""HNSW approximate nearest neighbor index for fast dense retrieval.

Wraps hnswlib to replace brute-force cosine similarity search.
At 200k vectors, drops dense search from ~5s to <10ms.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import hnswlib
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False


class ANNIndex:
    """HNSW index for approximate cosine nearest-neighbor search."""

    def __init__(self, dim: int = 384, m: int = 32, ef_construction: int = 200,
                 ef_search: int = 100, max_elements: int = 500_000,
                 index_path: str | None = None):
        if not HAS_HNSWLIB:
            raise ImportError("hnswlib not installed: pip install hnswlib")

        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_elements = max_elements
        self.index_path = Path(index_path) if index_path else None

        self._index: hnswlib.Index | None = None
        self._id_to_label: dict[str, int] = {}  # memory_id → hnswlib label
        self._label_to_id: dict[int, str] = {}  # hnswlib label → memory_id
        self._next_label: int = 0
        self._lock = threading.Lock()
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready and self._index is not None

    @property
    def count(self) -> int:
        """Active (non-deleted) vectors in the index."""
        return len(self._id_to_label)

    def build(self, ids: list[str], vecs: np.ndarray):
        """Build index from scratch. ids and vecs must be aligned."""
        if len(ids) == 0:
            # init empty index so add() works later
            idx = hnswlib.Index(space="cosine", dim=self.dim)
            idx.init_index(max_elements=self.max_elements, M=self.m, ef_construction=self.ef_construction)
            idx.set_ef(self.ef_search)
            self._index = idx
            self._ready = True
            return

        t0 = time.time()
        n = len(ids)

        idx = hnswlib.Index(space="cosine", dim=self.dim)
        max_el = max(self.max_elements, n + 10_000)
        idx.init_index(max_elements=max_el, M=self.m, ef_construction=self.ef_construction)
        idx.set_ef(self.ef_search)

        # add all vectors with integer labels
        labels = np.arange(n, dtype=np.int64)
        idx.add_items(vecs.astype(np.float32), labels, num_threads=4)

        with self._lock:
            self._index = idx
            self._id_to_label = {mid: int(i) for i, mid in enumerate(ids)}
            self._label_to_id = {int(i): mid for i, mid in enumerate(ids)}
            self._next_label = n
            self._ready = True

        elapsed = time.time() - t0
        logger.info(f"ANN index built: {n} vectors in {elapsed:.1f}s")

    def add(self, memory_id: str, vec: np.ndarray):
        """Add a single vector to the index. Thread-safe."""
        if self._index is None:
            return

        with self._lock:
            # if already exists, remove first
            if memory_id in self._id_to_label:
                old_label = self._id_to_label[memory_id]
                self._index.mark_deleted(old_label)
                del self._label_to_id[old_label]

            label = self._next_label
            self._next_label += 1

            # resize if needed
            if label >= self._index.get_max_elements():
                new_max = self._index.get_max_elements() + 100_000
                self._index.resize_index(new_max)

            self._index.add_items(
                vec.astype(np.float32).reshape(1, -1),
                np.array([label], dtype=np.int64),
            )
            self._id_to_label[memory_id] = label
            self._label_to_id[label] = memory_id

    def remove(self, memory_id: str):
        """Mark a vector as deleted. Thread-safe."""
        if self._index is None:
            return

        with self._lock:
            if memory_id in self._id_to_label:
                label = self._id_to_label[memory_id]
                self._index.mark_deleted(label)
                del self._id_to_label[memory_id]
                del self._label_to_id[label]

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Search for nearest neighbors. Returns [(memory_id, similarity_score), ...].

        Scores are cosine similarity (higher = more similar), matching
        the convention of the brute-force fallback.
        """
        if not self._ready or self._index is None or self.count == 0:
            return []

        k = min(top_k, self.count)
        q = query_vec.astype(np.float32).reshape(1, -1)

        with self._lock:
            labels, distances = self._index.knn_query(q, k=k)

        results = []
        for label, dist in zip(labels[0], distances[0]):
            label = int(label)
            if label in self._label_to_id:
                # hnswlib cosine distance = 1 - cosine_similarity
                similarity = 1.0 - float(dist)
                results.append((self._label_to_id[label], similarity))

        return results

    def save(self, path: Path | str | None = None):
        """Persist index to disk."""
        if self._index is None:
            return

        save_path = Path(path) if path else self.index_path
        if save_path is None:
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            self._index.save_index(str(save_path))

            # save id mappings alongside
            import json
            meta_path = save_path.with_suffix(".meta.json")
            meta = {
                "id_to_label": self._id_to_label,
                "next_label": self._next_label,
                "count": self._index.get_current_count(),
                "saved_at": time.time(),
            }
            meta_path.write_text(json.dumps(meta))

        logger.info(f"ANN index saved: {self.count} vectors → {save_path}")

    def load(self, path: Path | str | None = None) -> bool:
        """Load index from disk. Returns True if successful."""
        load_path = Path(path) if path else self.index_path
        if load_path is None or not load_path.exists():
            return False

        meta_path = load_path.with_suffix(".meta.json")
        if not meta_path.exists():
            return False

        try:
            import json
            meta = json.loads(meta_path.read_text())

            idx = hnswlib.Index(space="cosine", dim=self.dim)
            idx.load_index(str(load_path), max_elements=self.max_elements)
            idx.set_ef(self.ef_search)

            with self._lock:
                self._index = idx
                self._id_to_label = meta["id_to_label"]
                self._label_to_id = {int(v): k for k, v in self._id_to_label.items()}
                self._next_label = meta["next_label"]
                self._ready = True

            logger.info(f"ANN index loaded: {self.count} vectors from {load_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load ANN index: {e}")
            return False
