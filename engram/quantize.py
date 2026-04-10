"""Lifecycle-aware embedding compression (SuperLocalMemory V3.3).

As memories age and retention drops, quantize embeddings to lower precision
to save storage while keeping degraded retrieval possible.

Retention thresholds:
  Active  (R > 0.8):  32-bit float (no compression)
  Warm    (0.5 < R <= 0.8): 8-bit int
  Cold    (0.2 < R <= 0.5): 4-bit int
  Archive (0.05 < R <= 0.2): 2-bit int
  Forgotten (R <= 0.05): deleted

Fisher-Rao Quantization-Aware Distance (FRQAD):
When comparing embeddings at different precisions, inflate variance
proportional to quantization loss to prevent false similarity.
  σ²_eff = σ²_obs · (32/bits)^κ, κ=1.5
"""

from __future__ import annotations

import numpy as np

from engram.store import Store, Memory
from engram.lifecycle import compute_retention
from engram.config import Config


# precision tiers
TIERS = [
    ("active",  0.8, 32),
    ("warm",    0.5, 8),
    ("cold",    0.2, 4),
    ("archive", 0.05, 2),
]


def get_tier(retention: float) -> tuple[str, int]:
    """Get compression tier name and bit width for a retention score."""
    for name, threshold, bits in TIERS:
        if retention > threshold:
            return name, bits
    return "forgotten", 0


def quantize_embedding(embedding: np.ndarray, bits: int) -> bytes:
    """Quantize a float32 embedding to the specified bit width."""
    if bits >= 32:
        return embedding.astype(np.float32).tobytes()

    # normalize to [0, 1] range
    vmin, vmax = embedding.min(), embedding.max()
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    normalized = (embedding - vmin) / (vmax - vmin)

    if bits == 8:
        quantized = (normalized * 255).astype(np.uint8)
        # pack: header (8 bytes for min/max) + quantized
        header = np.array([vmin, vmax], dtype=np.float32).tobytes()
        return header + quantized.tobytes()
    elif bits == 4:
        quantized = (normalized * 15).astype(np.uint8)
        # pack pairs of 4-bit values into bytes
        packed = np.zeros(len(quantized) // 2 + 1, dtype=np.uint8)
        for i in range(0, len(quantized) - 1, 2):
            packed[i // 2] = (quantized[i] << 4) | quantized[i + 1]
        if len(quantized) % 2:
            packed[-1] = quantized[-1] << 4
        header = np.array([vmin, vmax], dtype=np.float32).tobytes()
        return header + packed.tobytes()
    elif bits == 2:
        quantized = (normalized * 3).astype(np.uint8)
        # pack 4 values per byte
        packed = np.zeros(len(quantized) // 4 + 1, dtype=np.uint8)
        for i in range(0, len(quantized) - 3, 4):
            packed[i // 4] = ((quantized[i] << 6) | (quantized[i+1] << 4) |
                              (quantized[i+2] << 2) | quantized[i+3])
        header = np.array([vmin, vmax], dtype=np.float32).tobytes()
        return header + packed.tobytes()

    return embedding.astype(np.float32).tobytes()


def dequantize_embedding(data: bytes, bits: int, dim: int = 384) -> np.ndarray:
    """Dequantize a compressed embedding back to float32."""
    if bits >= 32:
        return np.frombuffer(data, dtype=np.float32).copy()

    # extract header
    header = np.frombuffer(data[:8], dtype=np.float32)
    vmin, vmax = float(header[0]), float(header[1])
    payload = data[8:]

    if bits == 8:
        quantized = np.frombuffer(payload, dtype=np.uint8)[:dim]
        return (quantized.astype(np.float32) / 255.0) * (vmax - vmin) + vmin
    elif bits == 4:
        raw = np.frombuffer(payload, dtype=np.uint8)
        values = []
        for byte in raw:
            values.append((byte >> 4) & 0x0F)
            values.append(byte & 0x0F)
        quantized = np.array(values[:dim], dtype=np.float32)
        return (quantized / 15.0) * (vmax - vmin) + vmin
    elif bits == 2:
        raw = np.frombuffer(payload, dtype=np.uint8)
        values = []
        for byte in raw:
            values.append((byte >> 6) & 0x03)
            values.append((byte >> 4) & 0x03)
            values.append((byte >> 2) & 0x03)
            values.append(byte & 0x03)
        quantized = np.array(values[:dim], dtype=np.float32)
        return (quantized / 3.0) * (vmax - vmin) + vmin

    return np.frombuffer(data, dtype=np.float32).copy()


def frqad_distance(emb_a: np.ndarray, emb_b: np.ndarray,
                   bits_a: int, bits_b: int, kappa: float = 1.5) -> float:
    """Fisher-Rao Quantization-Aware Distance.

    Inflates effective variance for lower-precision embeddings to prevent
    false similarity from quantization noise.
    """
    # base cosine similarity
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0

    cosine_sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

    # quantization penalty: reduce similarity for lower precision
    penalty_a = (32 / max(1, bits_a)) ** kappa if bits_a < 32 else 1.0
    penalty_b = (32 / max(1, bits_b)) ** kappa if bits_b < 32 else 1.0
    penalty = max(penalty_a, penalty_b)

    # adjusted similarity — penalized by quantization noise
    adjusted = cosine_sim / (1.0 + 0.01 * (penalty - 1.0))
    return 1.0 - max(-1.0, min(1.0, adjusted))


def compress_old_embeddings(store: Store, config: Config, dry_run: bool = True) -> dict:
    """Compress embeddings for memories with low retention.

    Scans all memories, computes retention, and compresses embeddings
    that are above the current bit width for their tier.
    """
    import json

    rows = store.conn.execute(
        "SELECT * FROM memories WHERE forgotten = 0 AND embedding IS NOT NULL"
    ).fetchall()

    stats = {"scanned": len(rows), "compressed": 0, "by_tier": {}}

    for row in rows:
        mem = store._row_to_memory(row)
        retention = compute_retention(mem, config)
        tier_name, target_bits = get_tier(retention)

        current_bits = mem.metadata.get("embedding_bits", 32)

        if target_bits < current_bits and target_bits > 0:
            if not dry_run:
                # dequantize from current precision, requantize to target
                if mem.embedding is not None:
                    compressed = quantize_embedding(mem.embedding, target_bits)
                    store.conn.execute(
                        "UPDATE memories SET embedding = ? WHERE id = ?",
                        (compressed, mem.id),
                    )
                    mem.metadata["embedding_bits"] = target_bits
                    mem.metadata["compressed_at"] = __import__("time").time()
                    store.conn.execute(
                        "UPDATE memories SET metadata = ? WHERE id = ?",
                        (json.dumps(mem.metadata), mem.id),
                    )

            stats["compressed"] += 1
            stats["by_tier"][tier_name] = stats["by_tier"].get(tier_name, 0) + 1

    if not dry_run and stats["compressed"]:
        store.conn.commit()
        store.invalidate_embedding_cache()

    return stats
