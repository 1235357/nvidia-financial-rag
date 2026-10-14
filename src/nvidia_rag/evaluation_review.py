"""Import supplied relevance judgments bound to an immutable retrieval run.

These records validate bookkeeping, not reviewer identity, financial truth,
dataset independence, or generated-answer correctness. No retrieval is rerun.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
import re


_MAX_BODY_BYTES = 64 * 1024 * 1024
_RUN_FILES = ("freeze.json", "results.json", "review-packet.json", "receipt.json")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _keys(value, fields, label: str) -> None:
    _require(type(value) is dict and set(value) == set(fields),
             f"{label} requires exact fields")


def _text(value, label: str) -> None:
    _require(type(value) is str and bool(value.strip()), f"{label} must be a nonblank string")


def _digest(value, label: str) -> None:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{label} must be a lowercase SHA-256 digest")


def _validate_candidate(candidate, rank: int, seen: set[str]) -> None:
    _keys(candidate, {"rank", "chunk", "score", "text_sha256"}, "candidate")
    _require(type(candidate["rank"]) is int and candidate["rank"] == rank,
             "candidate ranks must be complete and ordered")
    chunk = candidate["chunk"]
    _keys(chunk, {"chunk_id", "document_id", "page_number", "text",
                  "amount_bbox", "metric_bbox", "period_bbox"}, "candidate chunk")
    for field in ("chunk_id", "document_id", "text"):
        _text(chunk[field], f"chunk {field}")
    _require(chunk["chunk_id"] not in seen, "candidate chunk IDs must be unique per question")
    seen.add(chunk["chunk_id"])
    _require(type(chunk["page_number"]) is int and chunk["page_number"] > 0,
             "candidate page_number must be a positive integer")
    for field in ("amount_bbox", "metric_bbox", "period_bbox"):
        bbox = chunk[field]
        if bbox is None:
            continue
        _require(type(bbox) in (list, tuple) and len(bbox) == 4 and all(
            type(value) in (int, float) and math.isfinite(value) and value >= 0
            for value in bbox
        ) and bbox[0] < bbox[2] and bbox[1] < bbox[3],
                 f"candidate {field} must be a finite positive-area rectangle")
    score = candidate["score"]
    _require(type(score) in (int, float) and math.isfinite(score)
             and -1 - 1e-12 <= score <= 1 + 1e-12,
             "candidate score must be a finite cosine score")
    _digest(candidate["text_sha256"], "candidate text_sha256")
    _require(candidate["text_sha256"] == sha256(chunk["text"].encode("utf-8")).hexdigest(),
             "candidate text hash mismatch")


def build_review_packet(results, *, freeze_sha256, results_sha256) -> dict:
    """Retain complete ordered retrieval snapshots and one null slot per chunk."""
    _digest(freeze_sha256, "freeze_sha256")
    _digest(results_sha256, "results_sha256")
    _keys(results, {"schema_version", "freeze_sha256", "report", "calls"}, "results")
    _require(type(results["schema_version"]) is int and results["schema_version"] == 1,
             "results require schema version 1")
    _require(results["freeze_sha256"] == freeze_sha256, "results freeze hash mismatch")
    _require(type(results["report"]) is dict, "results report must be an object")
    _require(type(results["calls"]) is list, "results calls must be a list")
    rows = []
    question_ids: set[str] = set()
    for call in results["calls"]:
        _keys(call, {"question_id", "query", "query_sha256", "candidates"}, "retrieval call")
        _text(call["question_id"], "question_id")
        _require(call["question_id"] not in question_ids, "question IDs must be unique")
        question_ids.add(call["question_id"])
        _text(call["query"], "query")
        _digest(call["query_sha256"], "query_sha256")
        _require(call["query_sha256"] == sha256(call["query"].encode("utf-8")).hexdigest(),
                 "retrieval query hash mismatch")
        _require(type(call["candidates"]) is list, "candidates must be a list")
        seen: set[str] = set()
        for rank, candidate in enumerate(call["candidates"], start=1):
            _validate_candidate(candidate, rank, seen)
        rows.append({
            "question_id": call["question_id"], "query": call["query"],
            "query_sha256": call["query_sha256"], "candidates": deepcopy(call["candidates"]),
            "judgments": [{"chunk_id": candidate["chunk"]["chunk_id"], "grade": None,
                           "reviewer": None, "source_check": None}
                          for candidate in call["candidates"]],
        })
    return {"schema_version": 1, "kind": "retrieval-relevance-review",
            "freeze_sha256": freeze_sha256, "results_sha256": results_sha256, "rows": rows}


def _checked_path(value, label: str, *, exists: bool) -> Path:
    path = Path(value)
    _require(path.is_absolute(), f"{label} must be an absolute local path")
    for part in (path, *path.parents):
        _require(not part.is_symlink() and not getattr(part, "is_junction", lambda: False)(),
                 f"{label} must not use symlinks or junctions")
    return path.resolve(strict=exists)


def _paths(run_dir, review_path, output_path) -> tuple[Path, Path, Path]:
    run = _checked_path(run_dir, "run_dir", exists=True)
    review = _checked_path(review_path, "review_path", exists=True)
    output = _checked_path(output_path, "output_path", exists=False)
    _require(run.is_dir(), "run_dir must be a directory")
    _require(review.is_file(), "review_path must be a file")
    _require(output != run and not output.is_relative_to(run),
             "output_path must be outside the immutable run")
    _require(output != review, "output_path must not overlap review_path")
    _require(not output.exists(), "output_path must be a fresh exclusive file")
    _require(output.parent.is_dir(), "output_path parent must be an existing directory")
    return run, review, output


def _read_body(path: Path) -> bytes:
    _checked_path(path, "input artifact", exists=True)
    _require(path.is_file(), "input artifact must be a file")
    _require(path.stat().st_size <= _MAX_BODY_BYTES, "input artifact exceeds 64 MiB limit")
    with path.open("rb") as handle:
        body = handle.read(_MAX_BODY_BYTES + 1)
    _require(len(body) <= _MAX_BODY_BYTES, "input artifact exceeds 64 MiB limit")
    return body


def _run_bodies(run: Path, bundle: dict, canonical_json) -> dict[str, bytes]:
    _keys(bundle, {"freeze", "results", "packet", "receipt"}, "completed run")
    bodies = {}
    for name, key in zip(_RUN_FILES, ("freeze", "results", "packet", "receipt")):
        body = _read_body(run / name)
        _require(body == canonical_json(bundle[key]), "completed run artifact identity mismatch")
        bodies[name] = body
    return bodies


def _normalize_judgments(review: dict) -> dict:
    _keys(review, {"schema_version", "kind", "freeze_sha256", "results_sha256", "rows"},
          "review packet")
    _require(type(review["rows"]) is list, "review rows must be a list")
    normalized = deepcopy(review)
    for row in normalized["rows"]:
        _keys(row, {"question_id", "query", "query_sha256", "candidates", "judgments"},
              "review row")
        _require(type(row["judgments"]) is list, "review judgments must be a list")
        for judgment in row["judgments"]:
            _keys(judgment, {"chunk_id", "grade", "reviewer", "source_check"}, "judgment")
            grade = judgment["grade"]
            _require(grade is None or (type(grade) is int and grade >= 0),
                     "judgment grade must be null or a nonnegative integer")
            reviewer, source_check = judgment["reviewer"], judgment["source_check"]
            provenance = (type(reviewer) is str and bool(reviewer.strip())
                          and type(source_check) is str and bool(source_check.strip()))
            _require(provenance or (grade is None and reviewer is None and source_check is None),
                     "judgment requires both nonblank reviewer and source_check, or both null for unknown")
            # Only these three explicitly editable values may differ from the
            # null template. IDs, order and the full candidate body stay intact.
            judgment.update(grade=None, reviewer=None, source_check=None)
    return normalized


def import_review(*, run_dir, review_path, output_path) -> dict:
    """Write a fresh provenance-bound record of supplied chunk relevance.

    The completed source graph is verified before reading the review and again
    immediately before the exclusive write. Its directory is never modified.
    """
    # Importing this module or displaying CLI help does not load a semantic
    # backend. The completed-run loader also has no reason to construct one.
    from .evaluation_inputs import canonical_json, parse_json
    from .evaluation_run import load_completed_run
    from .evaluation import summarize_judgment_coverage_rows

    run, review_file, output = _paths(run_dir, review_path, output_path)
    bundle = load_completed_run(run)
    source_bodies = _run_bodies(run, bundle, canonical_json)
    source_hashes = {name: sha256(body).hexdigest() for name, body in source_bodies.items()}
    expected = build_review_packet(
        bundle["results"], freeze_sha256=source_hashes["freeze.json"],
        results_sha256=source_hashes["results.json"])
    _require(canonical_json(bundle["packet"]) == canonical_json(expected),
             "completed run review packet mismatch")
    review_body = _read_body(review_file)
    review = parse_json(review_body, max_bytes=_MAX_BODY_BYTES)
    normalized = _normalize_judgments(review)
    # Comparing canonical encodings preserves bool/int/float identity that
    # ordinary Python equality would conflate (True == 1 == 1.0).
    _require(canonical_json(normalized) == canonical_json(expected),
             "review changed immutable run data or judgment slots")
    coverage_rows = [{
        "question_id": row["question_id"], "assessment_unit": "chunk",
        "retrieved_ids": [candidate["chunk"]["chunk_id"] for candidate in row["candidates"]],
        "judgments": {judgment["chunk_id"]: judgment["grade"]
                      for judgment in row["judgments"] if judgment["grade"] is not None},
    } for row in review["rows"]]
    record = {
        "schema_version": 1, "kind": "imported-retrieval-relevance-review",
        "source_run": {
            "freeze_sha256": source_hashes["freeze.json"],
            "results_sha256": source_hashes["results.json"],
            "review_packet_sha256": source_hashes["review-packet.json"],
            "receipt_sha256": source_hashes["receipt.json"],
        },
        "review_sha256": sha256(review_body).hexdigest(), "review": review,
        "summary": summarize_judgment_coverage_rows(coverage_rows),
        "generation_runs": 0, "financial_correctness": "unestablished",
    }
    output_body = canonical_json(record)
    _require(len(output_body) <= _MAX_BODY_BYTES, "imported record exceeds 64 MiB limit")
    # A graph or path mutation during source inspection/review parsing must not
    # create an apparently completed imported record.
    _paths(run, review_file, output)
    final_bundle = load_completed_run(run)
    _require(_run_bodies(run, final_bundle, canonical_json) == source_bodies,
             "completed run changed during review import")
    _require(_read_body(review_file) == review_body, "review changed during import")
    with output.open("xb") as handle:
        handle.write(output_body)
    return record
