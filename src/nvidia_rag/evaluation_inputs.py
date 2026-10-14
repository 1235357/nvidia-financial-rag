"""Strict frozen-input boundaries for installed retrieval evaluation.

These checks validate supplied labels and JSON structure. They do not verify
source truth, independence, or the registration policy of a selected run mode.
"""

from __future__ import annotations

from dataclasses import fields
import json
import math

from .evaluation import EvaluationQuestion


MAX_INPUT_BYTES = 1024 * 1024
MAX_QUESTIONS = 1000
MAX_QUERY_CHARACTERS = 500

_REQUIRED_FIELDS = frozenset({"question_id", "query", "answerable", "evidence_pages"})
_QUESTION_FIELDS = frozenset(field.name for field in fields(EvaluationQuestion))
_USAGE_FIELDS = frozenset({"purpose", "frozen_on", "development_exposed", "registration_note"})


def _decode_body(body: bytes, *, max_bytes: int) -> str:
    if not isinstance(body, bytes):
        raise TypeError("JSON input must be bytes")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if len(body) > max_bytes:
        raise ValueError(f"JSON input exceeds {max_bytes} bytes")
    try:
        return body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("JSON input must be valid UTF-8") from exc


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate decoded JSON object member")
        result[key] = value
    return result


def _reject_constant(_: str) -> object:
    raise ValueError("JSON numbers must be finite")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    return value


def _require_utf8_strings(value: object) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            try:
                item.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise ValueError("JSON strings must contain valid Unicode scalars") from exc
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def _parse_text(text: str) -> object:
    try:
        value = json.loads(text, object_pairs_hook=_object_without_duplicates,
                           parse_constant=_reject_constant, parse_float=_finite_float)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON body") from exc
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    _require_utf8_strings(value)
    return value


def parse_json(body: bytes, *, max_bytes: int = MAX_INPUT_BYTES) -> object:
    """Decode bounded UTF-8 JSON without duplicates, nonfinite values or surrogates.

    The byte limit is inclusive; review readers can explicitly supply a larger
    bound. No schema, canonical-byte requirement, or type coercion is applied.
    Strings and keys must decode to Unicode scalars, including escaped valid
    surrogate pairs. Invalid input raises ValueError; non-bytes raises TypeError.
    """
    return _parse_text(_decode_body(body, max_bytes=max_bytes))


def canonical_json(value: object) -> bytes:
    """Encode sorted, compact UTF-8 JSON with one trailing newline.

    Nonfinite numbers are rejected. This serializer does not replace the
    schema validation required by each artifact consumer.
    """
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _nonblank_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")


def _ground_truth_types(value: object, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object or null")
    for item in value.values():
        _nonblank_string(item, f"{name} values")
    # The dataclass validates exact fields and the permitted vocabularies.


def _usage_types(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != _USAGE_FIELDS:
        raise ValueError("evaluation_usage requires exact registration fields")
    _nonblank_string(value["purpose"], "evaluation_usage purpose")
    _nonblank_string(value["registration_note"], "evaluation_usage registration_note")
    if type(value["development_exposed"]) is not bool:
        raise ValueError("evaluation_usage development_exposed must be boolean")
    if value["frozen_on"] is not None and not isinstance(value["frozen_on"], str):
        raise ValueError("evaluation_usage frozen_on must be an ISO date or null")
    # Calendar and purpose invariants are shared with EvaluationQuestion.


def _question(row: object) -> EvaluationQuestion:
    if not isinstance(row, dict):
        raise ValueError("question row must be an object")
    if not _REQUIRED_FIELDS <= set(row):
        raise ValueError("question row is missing required fields")
    if set(row) - _QUESTION_FIELDS:
        raise ValueError("question row contains unknown fields")

    _nonblank_string(row["question_id"], "question_id")
    _nonblank_string(row["query"], "query")
    if len(row["query"]) > MAX_QUERY_CHARACTERS:
        raise ValueError(f"query exceeds {MAX_QUERY_CHARACTERS} characters")
    if type(row["answerable"]) is not bool:
        raise ValueError("answerable must be a boolean")

    pages = row["evidence_pages"]
    if not isinstance(pages, list) or any(type(page) is not int or page < 1 for page in pages):
        raise ValueError("evidence_pages must be a list of positive one-based integers")
    documents = row.get("evidence_document_ids", [])
    if not isinstance(documents, list):
        raise ValueError("evidence_document_ids must be a list")
    for document in documents:
        _nonblank_string(document, "evidence_document_ids values")
    if len(documents) != len(pages):
        raise ValueError("evidence_document_ids must align one-to-one with evidence_pages")
    if len(set(zip(documents, pages))) != len(pages):
        raise ValueError("duplicate evidence document/page pair")

    expected = row.get("expected_answer")
    if expected is not None and not isinstance(expected, str):
        raise ValueError("expected_answer must be a string or null")
    _ground_truth_types(row.get("attribution_ground_truth"), "attribution_ground_truth")
    _ground_truth_types(row.get("financial_ground_truth"), "financial_ground_truth")
    _usage_types(row.get("evaluation_usage"))

    values = dict(row)
    values["evidence_pages"] = tuple(pages)
    values["evidence_document_ids"] = tuple(documents)
    return EvaluationQuestion(**values)


def parse_questions(body: bytes) -> tuple[EvaluationQuestion, ...]:
    """Parse bounded legacy-format JSONL without rewriting question text.

    Blank lines are ignored. Each answerable question requires aligned,
    document-qualified evidence pages; duplicate pairs and IDs are rejected.
    Optional annotations retain the existing EvaluationQuestion schema. Run
    mode registration and verified-corpus membership belong to the runner.
    """
    text = _decode_body(body, max_bytes=MAX_INPUT_BYTES)
    questions: list[EvaluationQuestion] = []
    seen: set[str] = set()
    # Unicode line/paragraph separators can occur inside a valid JSON string.
    # Only physical JSONL newlines delimit records; CRLF is JSON whitespace.
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        if len(questions) >= MAX_QUESTIONS:
            raise ValueError(f"question input exceeds {MAX_QUESTIONS} questions")
        try:
            question = _question(_parse_text(line))
        except ValueError as exc:
            raise ValueError(f"invalid question on line {line_number}: {exc}") from exc
        if question.question_id in seen:
            raise ValueError(f"duplicate question_id on line {line_number}")
        seen.add(question.question_id)
        questions.append(question)
    return tuple(questions)
