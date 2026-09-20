"""Bounded rerank scores shared by retrieval and its benchmark."""
import math


def sigmoid(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def rerank_score(raw_score: float, *, prior_rank: int, fusion_alpha: float = 0.0,
                 temporal_boost: float = 0.0, normalized: bool = False) -> float:
    """Calibrate once, then optionally blend a reciprocal prior rank.

    Local cross-encoders return logits. Hosted rerankers return normalized
    scores; convert those to log-odds only when applying temporal evidence.
    The default preserves the model ordering without lexical/prior bonuses.
    """
    if not 0.0 <= fusion_alpha <= 1.0:
        raise ValueError('rerank_fusion_alpha must be between 0 and 1')
    if not math.isfinite(raw_score):
        raise ValueError('reranker scores must be finite')
    if normalized:
        score = max(0.0, min(1.0, raw_score))
        if temporal_boost:
            p = max(1e-12, min(1.0-1e-12, score))
            score = sigmoid(math.log(p / (1.0-p)) + temporal_boost)
    else:
        score = sigmoid(raw_score + temporal_boost)
    return (1.0-fusion_alpha) * score + fusion_alpha / (prior_rank+1)
