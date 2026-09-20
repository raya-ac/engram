"""Date parsing and relative windows shared by retrieval and benchmarks."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta


_DATE_RE = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})")

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}

_DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_RELATIVE_PATTERNS = [
    # "N days ago" / "ten days ago"
    (re.compile(r"(\d+)\s+days?\s+ago", re.I), "days"),
    (re.compile(r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+days?\s+ago", re.I), "days_word"),
    # "N weeks ago" / "four weeks ago"
    (re.compile(r"(\d+)\s+weeks?\s+ago", re.I), "weeks"),
    (re.compile(r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+weeks?\s+ago", re.I), "weeks_word"),
    # "a week ago"
    (re.compile(r"\ba\s+week\s+ago\b", re.I), "a_week"),
    # "N months ago"
    (re.compile(r"(\d+)\s+months?\s+ago", re.I), "months"),
    (re.compile(r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\s+ago", re.I), "months_word"),
    # "a month ago"
    (re.compile(r"\ba\s+month\s+ago\b", re.I), "a_month"),
    # "last Saturday" / "last Monday"
    (re.compile(r"last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", re.I), "last_day"),
    # "yesterday"
    (re.compile(r"\byesterday\b", re.I), "yesterday"),
    # "past N days" / "in the past two weeks" / "last N weeks"
    (re.compile(r"\b(?:in\s+the\s+)?(?:past|last)\s+(\d+)\s+days?\b", re.I), "past_days"),
    (re.compile(r"\b(?:in\s+the\s+)?(?:past|last)\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+days?\b", re.I), "past_days_word"),
    (re.compile(r"\b(?:in\s+the\s+)?(?:past|last)\s+(\d+)\s+weeks?\b", re.I), "past_weeks"),
    (re.compile(r"\b(?:in\s+the\s+)?(?:past|last)\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+weeks?\b", re.I), "past_weeks_word"),
    (re.compile(r"\b(?:in\s+the\s+)?(?:past|last)\s+(\d+)\s+months?\b", re.I), "past_months"),
    (re.compile(r"\b(?:in\s+the\s+)?(?:past|last)\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b", re.I), "past_months_word"),
]


def _parse_date(date_val: str | datetime | date | float | int | None) -> date | None:
    if date_val is None:
        return None
    if isinstance(date_val, datetime):
        return date_val.date()
    if isinstance(date_val, date):
        return date_val
    if isinstance(date_val, (int, float)):
        try:
            return datetime.fromtimestamp(date_val).date()
        except Exception:
            return None
    if isinstance(date_val, str):
        s = date_val.strip()
        m = _DATE_RE.search(s)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(s).date()
        except Exception:
            pass
    return None


def _temporal_content_query(query: str, reference_date=None) -> str:
    """Remove a resolved relative-time span only for lexical passage selection.

    The original query still drives semantic inference and temporal scoring.
    Keep non-temporal queries byte-identical and retain event content in phrases
    such as 'since I moved two months ago'.
    """
    if reference_date is None or _resolve_temporal_window(query, reference_date) is None:
        return query
    for pattern, _ in _RELATIVE_PATTERNS:
        match = pattern.search(query)
        if match:
            content = " ".join((query[:match.start()] + query[match.end():]).split())
            content = re.sub(r"\s+([?.!,;:])", r"\1", content)
            return content if re.search(r"\w", content) else query
    return query


def _resolve_temporal_window(query: str, reference_date: str | datetime | date | float | int | None = None) -> tuple[date, int] | None:
    """Parse relative time expressions and return (center_date, margin_days) or None."""
    ref_d = _parse_date(reference_date) if reference_date is not None else datetime.now().date()
    if not ref_d:
        return None

    m_since = re.search(
        r"\bsince\s+.*?(?:(\d+)|(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve))\s+(days?|weeks?|months?)\s+ago\b",
        query,
        re.I,
    )
    if m_since:
        num_str = m_since.group(1) or m_since.group(2)
        unit = m_since.group(3).lower()
        n = int(num_str) if num_str.isdigit() else _WORD_TO_NUM.get(num_str.lower(), 1)
        total_days = n if "day" in unit else (n * 7 if "week" in unit else n * 30)
        return (ref_d - timedelta(days=total_days / 2), total_days / 2 + 1)

    for pattern, kind in _RELATIVE_PATTERNS:
        m = pattern.search(query)
        if not m:
            continue

        if kind == "days":
            n = int(m.group(1))
            return (ref_d - timedelta(days=n), 2)
        elif kind == "days_word":
            n = _WORD_TO_NUM.get(m.group(1).lower(), 0)
            if n:
                return (ref_d - timedelta(days=n), 2)
        elif kind == "weeks":
            n = int(m.group(1))
            return (ref_d - timedelta(weeks=n), 4)
        elif kind == "weeks_word":
            n = _WORD_TO_NUM.get(m.group(1).lower(), 0)
            if n:
                return (ref_d - timedelta(weeks=n), 4)
        elif kind == "a_week":
            return (ref_d - timedelta(weeks=1), 4)
        elif kind == "months":
            n = int(m.group(1))
            return (ref_d - timedelta(days=n * 30), 7)
        elif kind == "months_word":
            n = _WORD_TO_NUM.get(m.group(1).lower(), 0)
            if n:
                return (ref_d - timedelta(days=n * 30), 7)
        elif kind == "a_month":
            return (ref_d - timedelta(days=30), 7)
        elif kind == "last_day":
            day_name = m.group(1).lower()
            target_dow = _DAY_NAMES[day_name]
            diff = (ref_d.weekday() - target_dow) % 7
            if diff == 0:
                diff = 7
            return (ref_d - timedelta(days=diff), 2)
        elif kind == "yesterday":
            return (ref_d - timedelta(days=1), 1)
        elif kind == "past_days":
            n = int(m.group(1))
            return (ref_d - timedelta(days=n / 2), max(1, int(n / 2) + 1))
        elif kind == "past_days_word":
            n = _WORD_TO_NUM.get(m.group(1).lower(), 0)
            if n:
                return (ref_d - timedelta(days=n / 2), max(1, int(n / 2) + 1))
        elif kind == "past_weeks":
            n = int(m.group(1))
            return (ref_d - timedelta(days=n * 7 / 2), max(2, int(n * 7 / 2) + 2))
        elif kind == "past_weeks_word":
            n = _WORD_TO_NUM.get(m.group(1).lower(), 0)
            if n:
                return (ref_d - timedelta(days=n * 7 / 2), max(2, int(n * 7 / 2) + 2))
        elif kind == "past_months":
            n = int(m.group(1))
            return (ref_d - timedelta(days=n * 30 / 2), max(4, int(n * 30 / 2) + 4))
        elif kind == "past_months_word":
            n = _WORD_TO_NUM.get(m.group(1).lower(), 0)
            if n:
                return (ref_d - timedelta(days=n * 30 / 2), max(4, int(n * 30 / 2) + 4))

    return None
