"""Preserve one prior leader within a confidence-filtered rerank result set."""

from __future__ import annotations

from collections.abc import Sequence


def preserve_prior_leader(
    reranked_ids: Sequence[str],
    prior_ids: Sequence[str],
    *,
    result_limit: int,
) -> list[str]:
    """Keep the model winner and include the best eligible prior when possible.

    Callers must confidence-filter candidates before calling this helper.
    The returned list contains every eligible scored ID once, preserving its
    rerank order except for at most one move: when result_limit >= 2 and
    the first eligible prior ID is absent from that prefix, move it to index
    result_limit - 1. A limit of 1 leaves the model winner first.

    All other relative order and tail candidates survive. Apply final
    truncation afterward. Relevance values stay unchanged in the caller;
    their numerical order does not encode this coverage choice.
    """
    if isinstance(result_limit, bool) or not isinstance(result_limit, int) or result_limit < 1:
        raise ValueError("result_limit must be a positive integer")

    result = list(dict.fromkeys(reranked_ids))
    if result_limit == 1 or not result:
        return result

    eligible = set(result)
    leader = next((candidate_id for candidate_id in prior_ids if candidate_id in eligible), None)
    if leader is not None and leader not in result[:result_limit]:
        result.remove(leader)
        result.insert(result_limit - 1, leader)
    return result
