"""Retrieval metrics and reproducible LongMemEval result provenance."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def compute_metrics(
    ranked_ids: Sequence[str],
    correct_ids: set[str],
    ks: Sequence[int] = (1, 3, 5, 10, 30, 50),
) -> dict[str, float]:
    """Compute binary session recall and NDCG, including unretrieved answers."""
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Ranked session IDs must be unique")
    metrics = {}
    for k in ks:
        if k < 1:
            raise ValueError("Metric cutoffs must be positive")
        selected = ranked_ids[:k]
        hits = set(selected) & correct_ids
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, session_id in enumerate(selected, 1)
            if session_id in correct_ids
        )
        ideal = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(k, len(correct_ids)) + 1)
        )
        metrics[f"recall_any@{k}"] = float(bool(hits))
        metrics[f"recall_all@{k}"] = float(bool(correct_ids) and hits == correct_ids)
        metrics[f"ndcg_any@{k}"] = dcg / ideal if ideal else 0.0
    return metrics


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_run_metadata(
    dataset_path: str | Path,
    *,
    source_paths: Mapping[str, str | Path],
    embedding_model: str,
    cross_encoder_model: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fingerprint inputs without recording dataset contents or host paths.

    ``source_paths`` maps stable labels to every relevant implementation file.
    ``config`` must contain only benchmark settings, never credentials or the
    complete Engram configuration. Include cutoffs, rerank enablement and limits.
    """
    metadata = {
        "schema_version": 1,
        "dataset_sha256": _sha256(dataset_path),
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "embedding_model": embedding_model,
        "cross_encoder_model": cross_encoder_model,
        "config": dict(config),
    }
    # Round-trip to normalize JSON values and detach mutable caller settings.
    payload = _canonical_json(metadata)
    metadata = json.loads(payload)
    metadata["fingerprint"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return metadata


def _metadata_path(output_path: str | Path) -> Path:
    return Path(str(output_path) + ".metadata.json")


def write_run_metadata(output_path: str | Path, metadata: Mapping[str, Any]) -> None:
    """Write provenance after the caller exclusively claims a fresh result file."""
    _metadata_path(output_path).write_text(
        json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_resume_rows(
    output_path: str | Path, expected_metadata: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Validate provenance and unique question IDs before resuming any work.

    A fresh path returns no rows. Existing results without a matching sidecar,
    including legacy result files, must be rerun under a new output filename.
    """
    output_path = Path(output_path)
    metadata_path = _metadata_path(output_path)
    if not output_path.exists() and not metadata_path.exists():
        return []
    if not output_path.is_file() or not metadata_path.is_file():
        raise ValueError("Cannot resume: result file and metadata sidecar are both required")
    try:
        actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError("Cannot resume: invalid metadata sidecar") from exc
    if actual_metadata != expected_metadata:
        raise ValueError("Cannot resume: dataset, source, model or config provenance differs")

    rows = []
    seen = set()
    with output_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Cannot resume: invalid JSON at line {line_number}") from exc
            question_id = row.get("question_id") if isinstance(row, dict) else None
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"Cannot resume: invalid question ID at line {line_number}")
            if question_id in seen:
                raise ValueError(f"Cannot resume: duplicate question ID at line {line_number}")
            seen.add(question_id)
            rows.append(row)
    return rows
