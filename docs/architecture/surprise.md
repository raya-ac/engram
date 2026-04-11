# Surprise Scoring

every memory gets a novelty score at write time. novel memories get importance boosted, redundant ones get flagged.

## how it works

before inserting a new memory, compute k-NN (k=5) cosine distance against all existing embeddings:

1. find the 5 nearest neighbors (via ANN index or brute-force fallback)
2. compute mean distance to those neighbors
3. map through sigmoid: `surprise = 1 / (1 + exp(-10 * (mean_distance - 0.35)))`
4. derive importance modifier: `(surprise - 0.5) * 0.6` → range [-0.3, +0.3]

## output

```python
{
    "surprise": 0.85,           # 0-1, higher = more novel
    "nearest_distance": 0.49,   # cosine distance to closest neighbor
    "nearest_id": "abc-123",    # most similar existing memory
    "is_duplicate": False,      # True if nearest_distance < 0.08
    "k_distances": [0.49, 0.52, 0.55, 0.58, 0.61],
    "importance_modifier": +0.21  # additive adjustment
}
```

## importance adjustment

```python
adjusted = base_importance + importance_modifier
# clamped to [0.05, 1.0]
```

- surprise=0.85, base=0.7 → adjusted=0.91 (novel, boosted)
- surprise=0.15, base=0.7 → adjusted=0.49 (redundant, reduced)

## duplicate detection

if nearest distance < 0.08 (cosine similarity > 0.92), the memory is flagged as a potential duplicate. it still gets stored, but with reduced importance and a warning in the response.

## the sigmoid curve

the midpoint is 0.35 and steepness is 10.0. most memories cluster around 0.2-0.5 mean distance. the curve maps:

- distance < 0.2 → surprise ≈ 0.0 (very redundant)
- distance ≈ 0.35 → surprise = 0.5 (neutral)
- distance > 0.5 → surprise ≈ 1.0 (very novel)

## inspiration

from the [Titans paper](https://arxiv.org/abs/2501.00663) (Behrouz et al., Google) where memory updates are proportional to the gradient of the loss function — surprise.

## key file

`engram/surprise.py` — `compute_surprise()` and `adjust_importance()`
