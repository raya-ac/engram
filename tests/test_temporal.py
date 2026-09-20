"""Model-free checks for date parsing and retrieval time windows."""

from datetime import date, datetime

import pytest

from engram.temporal import _parse_date, _resolve_temporal_window, _temporal_content_query


@pytest.mark.parametrize("query,reference,expected", [
    ("Who joined lunch last Tuesday?", "2026-09-20", "Who joined lunch?"),
    ("tasks in the past two weeks", "2026-09-20", "tasks"),
    ("places since I moved two months ago", "2026-09-20", "places since I moved"),
    ("last Tuesday?", "2026-09-20", "last Tuesday?"),
    ("  original Spelling buisiness?  ", "2026-09-20", "  original Spelling buisiness?  "),
    ("notes yesterday", "bad-date", "notes yesterday"),
    ("notes yesterday", None, "notes yesterday"),
])
def test_passage_terms_exclude_only_a_resolved_temporal_span(query, reference, expected):
    assert _temporal_content_query(query, reference) == expected


@pytest.mark.parametrize("value", [
    "2026/04/10",
    "2026-04-10",
    "2026/4/10",
    "2026-4-10",
    "  2026-04-10  ",
    "2026/04/10 (Fri) 15:30",
    "2026-04-10 (Fri) 15:30",
    "2026-04-10T15:30:00",
    "2026-04-10T23:30:00-10:00",
    date(2026, 4, 10),
    datetime(2026, 4, 10, 15, 30),
    datetime(2026, 4, 10, 15, 30).timestamp(),
])
def test_parse_supported_date_formats(value):
    assert _parse_date(value) == date(2026, 4, 10)


@pytest.mark.parametrize("value", [
    None,
    "",
    "  ",
    "invalid-date-format",
    "2026/00/10",
    "2026-13-10",
    "2026/04/00",
    "2026-04-31",
    "2026/02/29",
    "2026-02-29T12:00:00",
    "0000/01/01",
    float("nan"),
    float("inf"),
])
def test_invalid_dates_return_none(value):
    assert _parse_date(value) is None


def test_leap_day_is_valid():
    assert _parse_date("2024-02-29") == date(2024, 2, 29)


@pytest.mark.parametrize("reference", [
    "2023/03/28 (Tue) 20:35",
    "2023-03-28 (Tue) 20:35",
    "2023-03-28T20:35:00",
    date(2023, 3, 28),
])
def test_four_weeks_ago_resolves_across_formats(reference):
    assert _resolve_temporal_window(
        "What was the milestone I mentioned four weeks ago?", reference
    ) == (date(2023, 2, 28), 4)


@pytest.mark.parametrize("query, reference, expected", [
    ("notes from yesterday", "2026-01-01", (date(2025, 12, 31), 1)),
    ("notes from yesterday", "2024-03-01", (date(2024, 2, 29), 1)),
    ("commit from last Monday", "2026-04-13", (date(2026, 4, 6), 2)),
    ("commit from last Monday", "2026-04-10", (date(2026, 4, 6), 2)),
    ("progress since two weeks ago", "2026-04-10", (date(2026, 4, 3), 8)),
    ("progress in the past two weeks", "2026-04-10", (date(2026, 4, 3), 9)),
    ("what is HNSW?", "2026-04-10", None),
    ("notes from yesterday", "2026-02-30", None),
    ("notes from yesterday", "invalid-date", None),
])
def test_relative_window_boundaries(query, reference, expected):
    assert _resolve_temporal_window(query, reference) == expected


@pytest.mark.parametrize("separator", ["/", "-"])
def test_benchmark_boost_uses_window_boundaries_and_skips_invalid_dates(separator):
    from benchmarks.longmemeval.run_engram import _apply_temporal_boost

    timestamps = {
        "at_start": "2023-02-24",
        "at_end": "2023-03-04",
        "before_start": "2023-02-23",
        "after_end": "2023-03-05",
        "invalid": "2023-02-30",
        "missing": "",
    }
    corpus = [
        {"id": doc_id, "timestamp": timestamp.replace("-", separator)}
        for doc_id, timestamp in timestamps.items()
    ]
    scores = dict.fromkeys(timestamps, 1.0)

    _apply_temporal_boost(
        scores,
        corpus,
        f"2023{separator}03{separator}28 (Tue) 20:35",
        "milestone mentioned four weeks ago",
    )

    assert scores == {
        "at_start": 2.5,
        "at_end": 2.5,
        "before_start": 1.0,
        "after_end": 1.0,
        "invalid": 1.0,
        "missing": 1.0,
    }


def test_benchmark_boost_skips_invalid_reference_date():
    from benchmarks.longmemeval.run_engram import _apply_temporal_boost

    scores = {"memory": 1.0}
    _apply_temporal_boost(
        scores,
        [{"id": "memory", "timestamp": "2023-03-01"}],
        "2023-02-30",
        "milestone mentioned four weeks ago",
    )
    assert scores == {"memory": 1.0}


def test_benchmark_boost_counts_a_session_once_across_roles():
    from benchmarks.longmemeval.run_engram import _apply_temporal_boost

    doc = {"id": "session", "timestamp": "2023-03-01"}
    scores = {"session": 1.0}
    _apply_temporal_boost(scores, [doc, dict(doc)], "2023-03-28", "four weeks ago")
    assert scores == {"session": 2.5}
