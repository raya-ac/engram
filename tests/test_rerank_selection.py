"""Prior coverage invariants without model scores, content or benchmark labels."""

import pytest

from engram.rerank_selection import preserve_prior_leader


def test_coverage_keeps_model_winner_and_other_relative_order():
    reranked = ["a", "b", "c", "d", "e", "f"]
    prior = ["e", "d", "c", "b", "a", "f"]
    ordered = preserve_prior_leader(reranked, prior, result_limit=3)
    assert ordered == ["a", "b", "e", "c", "d", "f"]
    assert [candidate for candidate in ordered if candidate != "e"] == ["a", "b", "c", "d", "f"]
    assert reranked == ["a", "b", "c", "d", "e", "f"]
    assert prior == ["e", "d", "c", "b", "a", "f"]


def test_leader_already_in_prefix_does_not_reorder_results():
    reranked = ["a", "b", "c", "d"]
    assert preserve_prior_leader(reranked, ["b", "d", "c", "a"], result_limit=3) == reranked


@pytest.mark.parametrize(
    "limit,expected",
    [(1, ["a", "b", "c"]), (2, ["a", "c", "b"]),
     (3, ["a", "b", "c"]), (20, ["a", "b", "c"])],
)
def test_result_limit_boundaries(limit, expected):
    assert preserve_prior_leader(["a", "b", "c"], ["c", "b", "a"], result_limit=limit) == expected


def test_coverage_ignores_candidates_removed_by_confidence_gate():
    assert preserve_prior_leader(
        ["a", "b", "c"], ["rejected", "unscored", "c", "b", "a"], result_limit=2,
    ) == ["a", "c", "b"]


def test_stable_deduplication_retains_every_eligible_tail_candidate():
    ordered = preserve_prior_leader(
        ["a", "a", "b", "c", "b", "d", "e", "f"],
        ["unscored", "e", "e", "d"], result_limit=3,
    )
    assert ordered == ["a", "b", "e", "c", "d", "f"]
    assert len(ordered) == len(set(ordered)) == 6
    assert set(ordered) == {"a", "b", "c", "d", "e", "f"}


@pytest.mark.parametrize("prior", [[], ["unscored"]])
def test_no_eligible_prior_keeps_order(prior):
    assert preserve_prior_leader(["a", "b"], prior, result_limit=2) == ["a", "b"]


def test_empty_candidates_cannot_admit_prior_entries():
    assert preserve_prior_leader([], ["unscored"], result_limit=5) == []


def test_single_result_still_deduplicates_without_replacing_winner():
    assert preserve_prior_leader(["a", "a", "b"], ["b", "a"], result_limit=1) == ["a", "b"]


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, "2", None])
def test_invalid_result_limit_is_rejected(invalid):
    with pytest.raises(ValueError, match="result_limit"):
        preserve_prior_leader(["a"], [], result_limit=invalid)
