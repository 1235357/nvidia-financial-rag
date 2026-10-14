"""Result-bound relevance reviews preserve every immutable run candidate."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def digest(body):
    return sha256(body).hexdigest()


def review_module():
    assert importlib.util.find_spec("nvidia_rag.evaluation_review") is not None, (
        "result-bound review module has not been implemented"
    )
    return importlib.import_module("nvidia_rag.evaluation_review")


def candidate(chunk_id, rank, score=0.75):
    text = f"Original PDF text for {chunk_id}; revenue $42 million."
    return {"rank": rank, "chunk": {
        "chunk_id": chunk_id, "document_id": "filing-2025", "page_number": 12,
        "text": text, "amount_bbox": [1.0, 2.0, 3.0, 4.0],
        "metric_bbox": None, "period_bbox": None,
    }, "score": score, "text_sha256": digest(text.encode("utf-8"))}


def result_fixture():
    query = "  Which revenue? 财报\n"
    return {"schema_version": 1, "freeze_sha256": "a" * 64,
            "report": {"summary": {}, "records": []}, "calls": [
                {"question_id": "q1", "query": query,
                 "query_sha256": digest(query.encode("utf-8")),
                 "candidates": [candidate("c1", 1), candidate("c2", 2, 1.0)]},
                {"question_id": "q2", "query": "Empty retrieval?",
                 "query_sha256": digest(b"Empty retrieval?"), "candidates": []},
                {"question_id": "q3", "query": "Other metric?",
                 "query_sha256": digest(b"Other metric?"),
                 "candidates": [candidate("c3", 1)]},
            ]}


def test_packet_keeps_exact_queries_complete_candidates_empty_rows_and_null_slots():
    results = result_fixture()
    original = deepcopy(results)
    packet = review_module().build_review_packet(
        results, freeze_sha256="a" * 64, results_sha256="b" * 64)
    assert packet == {
        "schema_version": 1, "kind": "retrieval-relevance-review",
        "freeze_sha256": "a" * 64, "results_sha256": "b" * 64,
        "rows": [{"question_id": call["question_id"], "query": call["query"],
                  "query_sha256": call["query_sha256"],
                  "candidates": call["candidates"],
                  "judgments": [{"chunk_id": row["chunk"]["chunk_id"], "grade": None,
                                 "reviewer": None, "source_check": None}
                                for row in call["candidates"]]}
                 for call in results["calls"]],
    }
    packet["rows"][0]["candidates"][0]["chunk"]["text"] = "edit"
    assert results == original


@pytest.fixture
def completed(tmp_path, monkeypatch):
    module = review_module()
    run_dir = tmp_path / "immutable-run"
    run_dir.mkdir()
    freeze = {"schema_version": 1, "config": {}, "inputs": {}, "questions": [],
              "reference": []}
    results = result_fixture()
    freeze_body = canonical(freeze)
    results["freeze_sha256"] = digest(freeze_body)
    results_body = canonical(results)
    packet = module.build_review_packet(
        results, freeze_sha256=digest(freeze_body), results_sha256=digest(results_body))
    packet_body = canonical(packet)
    bodies = {"freeze.json": freeze_body, "results.json": results_body,
              "review-packet.json": packet_body}
    receipt = {"schema_version": 1, "status": "complete",
               "files": {name: {"bytes": len(body), "sha256": digest(body)}
                         for name, body in bodies.items()},
               "generation_runs": 0, "financial_correctness": "unestablished"}
    bodies["receipt.json"] = canonical(receipt)
    for name, body in bodies.items():
        (run_dir / name).write_bytes(body)
    bundle = {"freeze": freeze, "results": results, "packet": packet, "receipt": receipt}
    loader_module = ModuleType("nvidia_rag.evaluation_run")

    def load_completed_run(path):
        assert Path(path) == run_dir
        for name, expected in bodies.items():
            if (run_dir / name).read_bytes() != expected:
                raise ValueError("source run mutated")
        return deepcopy(bundle)

    loader_module.load_completed_run = load_completed_run
    monkeypatch.setitem(sys.modules, "nvidia_rag.evaluation_run", loader_module)
    review_path = tmp_path / "review.json"
    review_path.write_bytes(packet_body)
    return module, run_dir, review_path, tmp_path / "imported.json", packet, bodies, loader_module


def run_import(completed, packet=None, *, body=None, output_path=None):
    module, run_dir, review_path, output, expected, _, _ = completed
    if packet is not None:
        review_path.write_bytes(canonical(packet))
    if body is not None:
        review_path.write_bytes(body)
    return module.import_review(run_dir=run_dir, review_path=review_path,
                                output_path=output_path or output)


def test_import_preserves_all_unknown_judgments_and_the_full_denominator(completed):
    record = run_import(completed)
    module, run_dir, review_path, output, packet, bodies, _ = completed
    assert set(record) == {"schema_version", "kind", "source_run", "review_sha256",
                           "review", "summary", "generation_runs", "financial_correctness"}
    assert record["schema_version"] == 1
    assert record["kind"] == "imported-retrieval-relevance-review"
    assert record["review"] == packet
    assert record["review_sha256"] == digest(review_path.read_bytes())
    assert record["source_run"] == {
        "freeze_sha256": digest(bodies["freeze.json"]),
        "results_sha256": digest(bodies["results.json"]),
        "review_packet_sha256": digest(bodies["review-packet.json"]),
        "receipt_sha256": digest(bodies["receipt.json"]),
    }
    summary = record["summary"]
    assert summary["assessment_unit"] == "chunk"
    assert summary["question_count"] == 3
    assert summary["empty_retrieval_count"] == 1
    assert summary["retrieved_unique_count"] == 3
    assert summary["unjudged_count"] == 3
    assert summary["unjudged_ratio"] == 1
    assert summary["judgment_coverage"] == 0
    assert record["generation_runs"] == 0
    assert record["financial_correctness"] == "unestablished"
    assert output.read_bytes() == canonical(record)
    assert {path.name: path.read_bytes() for path in run_dir.iterdir()} == bodies


def test_supplied_grades_count_relevance_without_turning_null_into_irrelevant(completed):
    packet = deepcopy(completed[4])
    packet["rows"][0]["judgments"][0].update(
        grade=2, reviewer="Reviewer A", source_check="PDF page 12 metric and period checked")
    packet["rows"][0]["judgments"][1].update(
        grade=0, reviewer="Reviewer B", source_check="Wrong metric on PDF page 12")
    packet["rows"][2]["judgments"][0].update(
        reviewer="Reviewer A", source_check="Source inspected; relevance remains unknown")
    record = run_import(completed, packet)
    assert record["review"] == packet
    assert record["summary"]["judged_relevant_count"] == 1
    assert record["summary"]["judged_irrelevant_count"] == 1
    assert record["summary"]["unjudged_count"] == 1
    assert record["summary"]["judgment_coverage"] == 2 / 3


def _mutate(packet, case):
    first = packet["rows"][0]
    judgment = first["judgments"][0]
    if case == "unknown-root": packet["unexpected"] = None
    elif case == "unknown-row": first["unexpected"] = None
    elif case == "unknown-judgment": judgment["unexpected"] = None
    elif case == "unknown-candidate": first["candidates"][0]["unexpected"] = None
    elif case == "unknown-chunk": first["candidates"][0]["chunk"]["unexpected"] = None
    elif case == "freeze-transplant": packet["freeze_sha256"] = "0" * 64
    elif case == "result-transplant": packet["results_sha256"] = "0" * 64
    elif case == "schema-bool": packet["schema_version"] = True
    elif case == "schema-float": packet["schema_version"] = 1.0
    elif case == "missing-row": packet["rows"].pop()
    elif case == "empty-row-removed": packet["rows"].pop(1)
    elif case == "row-order": packet["rows"].reverse()
    elif case == "duplicate-row": packet["rows"][2] = deepcopy(first)
    elif case == "query": first["query"] = first["query"].strip()
    elif case == "query-hash": first["query_sha256"] = "0" * 64
    elif case == "question-id": first["question_id"] = "foreign"
    elif case == "candidate-order": first["candidates"].reverse()
    elif case == "candidate-missing": first["candidates"].pop()
    elif case == "candidate-text": first["candidates"][0]["chunk"]["text"] += " changed"
    elif case == "candidate-rank": first["candidates"][0]["rank"] = 2
    elif case == "rank-bool": first["candidates"][0]["rank"] = True
    elif case == "page-float": first["candidates"][0]["chunk"]["page_number"] = 12.0
    elif case == "bbox-int": first["candidates"][0]["chunk"]["amount_bbox"][0] = 1
    elif case == "score-int": first["candidates"][1]["score"] = 1
    elif case == "score-bool": first["candidates"][1]["score"] = True
    elif case == "text-hash": first["candidates"][0]["text_sha256"] = "0" * 64
    elif case == "judgment-order": first["judgments"].reverse()
    elif case == "judgment-missing": first["judgments"].pop()
    elif case == "judgment-foreign": judgment["chunk_id"] = "foreign"
    elif case == "judgment-duplicate": first["judgments"][1] = deepcopy(judgment)
    elif case == "judgment-new": first["judgments"].append(deepcopy(judgment))
    elif case == "grade-bool": judgment["grade"] = True
    elif case == "grade-float": judgment["grade"] = 1.0
    elif case == "grade-negative": judgment["grade"] = -1
    elif case == "grade-string": judgment["grade"] = "1"
    elif case == "graded-no-provenance": judgment["grade"] = 0
    elif case == "graded-blank-provenance": judgment.update(grade=1, reviewer="  ", source_check="checked")
    elif case == "unknown-half-provenance": judgment["reviewer"] = "Reviewer"
    elif case == "unknown-nonstring-provenance": judgment.update(reviewer=7, source_check="checked")
    else: raise AssertionError(case)


@pytest.mark.parametrize("case", [
    "unknown-root", "unknown-row", "unknown-judgment", "unknown-candidate", "unknown-chunk",
    "freeze-transplant", "result-transplant", "schema-bool", "schema-float", "missing-row",
    "empty-row-removed", "row-order", "duplicate-row", "query", "query-hash", "question-id",
    "candidate-order", "candidate-missing", "candidate-text", "candidate-rank", "rank-bool",
    "page-float", "bbox-int", "score-int", "score-bool", "text-hash", "judgment-order",
    "judgment-missing", "judgment-foreign", "judgment-duplicate", "judgment-new", "grade-bool",
    "grade-float", "grade-negative", "grade-string", "graded-no-provenance",
    "graded-blank-provenance", "unknown-half-provenance", "unknown-nonstring-provenance",
])
def test_import_rejects_immutable_edits_and_invalid_judgments_before_output(completed, case):
    packet = deepcopy(completed[4])
    _mutate(packet, case)
    with pytest.raises(ValueError):
        run_import(completed, packet)
    assert not completed[3].exists()


@pytest.mark.parametrize("body", [
    b'{"schema_version":1,"schema_version":1}',
    b'{"schema_version":1,"schema_versi\\u006fn":1}',
    b'{"value":NaN}', b'{"value":Infinity}', b'{"value":1e999}', b'\xff',
])
def test_import_rejects_ambiguous_nonfinite_and_non_utf8_json(completed, body):
    with pytest.raises(ValueError):
        run_import(completed, body=body)
    assert not completed[3].exists()


def test_raw_review_hash_preserves_whitespace_identity(completed):
    body = json.dumps(completed[4], ensure_ascii=False, indent=2).encode("utf-8")
    record = run_import(completed, body=body)
    assert record["review_sha256"] == digest(body)
    assert record["review_sha256"] != digest(canonical(completed[4]))


def test_import_refuses_existing_output_without_overwrite(completed):
    completed[3].write_bytes(b"retain me")
    with pytest.raises((ValueError, FileExistsError)):
        run_import(completed)
    assert completed[3].read_bytes() == b"retain me"


@pytest.mark.parametrize("case", ["inside-run", "run-file", "review-file"])
def test_import_refuses_output_overlap(completed, case):
    output = {"inside-run": completed[1] / "new-review.json",
              "run-file": completed[1] / "results.json", "review-file": completed[2]}[case]
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError):
        run_import(completed, output_path=output)
    assert (output.read_bytes() if output.exists() else None) == before


def test_run_mutation_during_review_read_cannot_create_output(completed, monkeypatch):
    original_read = completed[0]._read_body
    review_path = completed[2]

    def read_then_mutate(path):
        body = original_read(path)
        if path == review_path:
            (completed[1] / "results.json").write_bytes(b"changed during import")
        return body

    monkeypatch.setattr(completed[0], "_read_body", read_then_mutate)
    with pytest.raises(ValueError, match="mutat|changed|mismatch|identity"):
        run_import(completed)
    assert not completed[3].exists()


def test_incomplete_loader_failure_does_not_create_output(completed):
    def reject(path):
        raise ValueError("pending run cannot be reviewed")
    completed[6].load_completed_run = reject
    with pytest.raises(ValueError, match="pending"):
        run_import(completed)
    assert not completed[3].exists()


def test_import_refuses_symlink_input_and_parent_when_supported(completed, tmp_path):
    link = tmp_path / "linked-review.json"
    try:
        link.symlink_to(completed[2])
    except OSError:
        pytest.skip("host does not permit creating symlinks")
    with pytest.raises(ValueError, match="symlink|symbolic"):
        completed[0].import_review(run_dir=completed[1], review_path=link,
                                   output_path=completed[3])
    output_parent = tmp_path / "linked-output"
    output_parent.symlink_to(completed[1].parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|symbolic"):
        run_import(completed, output_path=output_parent / "new.json")


def test_import_enforces_raw_body_limit_before_parsing(completed):
    with pytest.raises(ValueError, match="large|limit|MiB|bytes"):
        run_import(completed, body=b" " * (64 * 1024 * 1024 + 1))
    assert not completed[3].exists()
