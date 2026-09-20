"""One bounded extractive retry when a local reranker finds no confident result."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Callable, Iterable
import math
from numbers import Integral, Real
import re

from engram.embeddings import RERANKER_BACKENDS
from engram.rerank_scoring import rerank_score


PASSAGE_WORD_BUDGET = 160  # A word cap, not a tokenizer-token limit.
_STOP_WORDS = frozenset("""
a an the and or but if then than as at by for from in into of on onto out over
to under up upon with within without about after before during through
i me my mine we us our ours you your yours he him his she her hers it its they
them their theirs who whom whose what which when where why how
is am are was were be been being do does did have has had can could will would
shall should may might must not no nor any all both each few more most other
some such only own same so too very this that these those here there
s t ve re ll d m
""".split())
_CONTEXT_QUALIFIERS = re.compile(
    r"\b(?:not|no|never|nor|unless|except|however|but|although|instead|previous(?:ly)?|"
    r"former(?:ly)?|old|obsolete|outdated|draft|propos(?:ed|al)|rejected|unverified|"
    r"uncertain|examples?|hypothetical|disputed|superseded|cancelled|canceled|"
    r"incorrect|correction|retracted|warning|caution)\b", re.IGNORECASE)
_REFERENCE_START = re.compile(r"^[\s\"'(]*(?:it|they|he|she|this|that|these|those|such)\b", re.IGNORECASE)


def _forward_plural(token: str) -> str:
    """Return one conservative regular-English plural surface, without stripping."""
    if len(token) < 3 or not token.isascii() or not token.isalpha():
        return token
    if token.endswith(("us", "is")):
        return token
    if token.endswith(("ss", "sh", "ch", "x", "z")):
        return token + "es"
    if token.endswith("s"):
        return token
    if token.endswith("y") and token[-2] not in "aeiou":
        return token[:-1] + "ies"
    return token + "s"


def _surface_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True
    if left.endswith(("us", "is")) or right.endswith(("us", "is")):
        return False
    return _forward_plural(left) == right or left == _forward_plural(right)


def _matched_query_terms(query_terms: set[str], sentence_terms: set[str]) -> set[str]:
    """Count an original query term once across repetitions and surface variants."""
    return {term for term in query_terms
            if any(_surface_equivalent(term, word) for word in sentence_terms)}


def _query_passage(query: str, document: str) -> tuple[str, int, int] | None:
    """Choose a source-contiguous sentence and useful neighboring context.

    Unique query terms, including regular singular/plural surface matches, are
    weighted by inverse sentence frequency. Each original term contributes once. Ties favor the first source span. Short documents and documents
    without lexical evidence do not get a replacement passage. Only repeated,
    nonmatching neighboring boilerplate may be omitted around a complete anchor.
    """
    words = list(re.finditer(r"\S+", document))
    if len(words) <= PASSAGE_WORD_BUDGET:
        return None
    terms = set(re.findall(r"[a-z0-9]+", query.lower())) - _STOP_WORDS
    if not terms:
        return None
    sentences = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", document):
        source = match.group()
        text = source.strip()
        if text:
            start = match.start() + len(source) - len(source.lstrip())
            end = match.end() - len(source) + len(source.rstrip())
            sentence_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
            sentences.append((start, end, _matched_query_terms(terms, sentence_terms)))
    if not sentences:
        return None
    frequencies = Counter(term for _, _, sentence_terms in sentences for term in sentence_terms)
    weights = [
        sum(math.log((len(sentences) + 1) / (frequencies[term] + 1)) + 1
            for term in terms & sentence_terms)
        for _, _, sentence_terms in sentences
    ]
    best = max(range(len(sentences)), key=lambda index: weights[index])
    if weights[best] == 0:
        return None
    sentence_start, sentence_end, _ = sentences[best]
    normalized_sentences = [re.sub(r"\s+", " ", document[a:b]).strip().casefold()
                            for a, b, _ in sentences]
    repetitions = Counter(normalized_sentences)
    anchor = document[sentence_start:sentence_end]
    anchor_covers_query = len(terms) >= 2 and sentences[best][2] == terms

    def keep_neighbor(index):
        a, b, matched = sentences[index]
        # Repeated, unrelated boilerplate can overwhelm a short complete fact.
        # Keep unique context, references and qualifications; this is a narrow
        # duplication rule, not a general claim that context is unnecessary.
        return (not anchor_covers_query or _REFERENCE_START.search(anchor)
                or matched or repetitions[normalized_sentences[index]] == 1
                or _CONTEXT_QUALIFIERS.search(document[a:b]))

    start = sentences[best - 1][0] if best > 0 and keep_neighbor(best - 1) else sentence_start
    end = sentences[best + 1][1] if best + 1 < len(sentences) and keep_neighbor(best + 1) else sentence_end
    word_starts = [word.start() for word in words]
    word_ends = [word.end() for word in words]
    first = bisect_right(word_ends, start)
    last = bisect_left(word_starts, end)
    if last - first > PASSAGE_WORD_BUDGET:
        anchor_first = bisect_right(word_ends, sentence_start)
        anchor_last = bisect_left(word_starts, sentence_end)
        if anchor_last - anchor_first > PASSAGE_WORD_BUDGET:
            # Keep the first query hit visible even in an oversized sentence.
            hit = next((match for match in re.finditer(r"[a-z0-9]+", document[sentence_start:sentence_end].lower())
                        if any(_surface_equivalent(term, match.group()) for term in terms)), None)
            hit_offset = sentence_start + hit.start() if hit else sentence_start
            center = bisect_right(word_ends, hit_offset)
            first = max(first, center - PASSAGE_WORD_BUDGET // 2)
        else:
            remaining = PASSAGE_WORD_BUDGET - (anchor_last - anchor_first)
            first = max(first, anchor_first - remaining // 2)
        last = min(last, first + PASSAGE_WORD_BUDGET)
        start, end = words[first].start(), words[last - 1].end()
    return document[start:end], start, end


def _validated_scores(results: Iterable[tuple[int, float]], count: int) -> dict[int, float]:
    scores = {}
    try:
        iterator = iter(results)
    except TypeError as exc:
        raise ValueError("Reranker output must contain indexed scores") from exc
    for pair in iterator:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Reranker output must contain index/score pairs")
        index, score = pair
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < count:
            raise ValueError("Reranker returned an invalid document index")
        index = int(index)
        if index in scores:
            raise ValueError("Reranker returned a duplicate document index")
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
            raise ValueError("Reranker scores must be finite numbers")
        scores[index] = float(score)
    if len(scores) != count:
        raise ValueError("Reranker must score every supplied document exactly once")
    return scores


def rerank_with_passages(
    query: str,
    documents: list[str],
    model_name: str,
    *,
    rerank_fn: Callable[[str, list[str], str], Iterable[tuple[int, float]]],
    confidence_floor: float = 0.6,
    passage_query: str | None = None,
) -> tuple[list[tuple[int, float]], dict[int, dict]]:
    """Rerank full documents, then retry at most one passage per eligible doc.

    A local-only retry occurs when every base relevance score is below the
    floor. A nonpositive floor disables it. The lexical selection query may be
    supplied separately, but both model calls use the original semantic query.
    The final raw score is max(full, excerpt), with stable input-index ties.
    Trace data records raw scores and source offsets without copying documents.
    Ordinary confidence filtering remains the caller's responsibility.
    """
    if isinstance(confidence_floor, bool) or not isinstance(confidence_floor, Real):
        raise ValueError("confidence_floor must be a finite number no greater than 1")
    if not math.isfinite(confidence_floor) or confidence_floor > 1:
        raise ValueError("confidence_floor must be a finite number no greater than 1")
    if not documents:
        return [], {}
    scores = _validated_scores(rerank_fn(query, documents, model_name), len(documents))
    trace = {index: {"base_raw_score": score} for index, score in scores.items()}

    def ranked() -> list[tuple[int, float]]:
        return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))

    if model_name in RERANKER_BACKENDS or confidence_floor <= 0:
        return ranked(), trace
    if max(rerank_score(score, prior_rank=index) for index, score in scores.items()) >= confidence_floor:
        return ranked(), trace

    return retry_passages(query, documents, model_name, ranked(), trace,
                          rerank_fn=rerank_fn, passage_query=passage_query)


def retry_passages(
    query: str,
    documents: list[str],
    model_name: str,
    ranked_scores: Iterable[tuple[int, float]],
    previous_trace: dict[int, dict],
    *,
    rerank_fn: Callable[[str, list[str], str], Iterable[tuple[int, float]]],
    passage_query: str | None = None,
    candidate_indices: set[int] | None = None,
) -> tuple[list[tuple[int, float]], dict[int, dict]]:
    """Reuse full-document scores and retry each eligible excerpt at most once.

    The caller chooses when a retry is warranted. Hosted models never enter
    this local path, and an earlier excerpt attempt cannot be repeated.
    """
    scores = _validated_scores(ranked_scores, len(documents))
    trace = {index: dict(previous_trace.get(index, {"base_raw_score": score}))
             for index, score in scores.items()}

    def ranked() -> list[tuple[int, float]]:
        return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))

    if model_name in RERANKER_BACKENDS:
        return ranked(), trace

    selection_query = query if passage_query is None else passage_query
    selected = []
    for index, document in enumerate(documents):
        if (candidate_indices is not None and index not in candidate_indices) or "excerpt_raw_score" in trace[index]:
            continue
        passage = _query_passage(selection_query, document)
        if passage is not None:
            text, start, end = passage
            selected.append((index, text, start, end))
    if not selected:
        return ranked(), trace

    excerpts = [text for _, text, _, _ in selected]
    excerpt_scores = _validated_scores(rerank_fn(query, excerpts, model_name), len(excerpts))
    for subset_index, (index, text, start, end) in enumerate(selected):
        excerpt_score = excerpt_scores[subset_index]
        trace[index].update({
            "excerpt_raw_score": excerpt_score, "source_start": start,
            "source_end": end, "passage_words": len(text.split()),
        })
        scores[index] = max(scores[index], excerpt_score)
    return ranked(), trace
