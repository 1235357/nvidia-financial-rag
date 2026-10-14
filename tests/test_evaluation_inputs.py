"""Strict frozen-input parsing, separate from registration or source truth."""

import importlib
import importlib.util
from dataclasses import asdict
import json
import math

import pytest

from nvidia_rag.evaluation import EvaluationQuestion


def inputs():
    assert importlib.util.find_spec("nvidia_rag.evaluation_inputs") is not None, \
        "strict installed evaluation input boundary is missing"
    return importlib.import_module("nvidia_rag.evaluation_inputs")


def row(**changes):
    return {
        "question_id": "q1",
        "query": "What was revenue?",
        "answerable": True,
        "evidence_pages": [3],
        "evidence_document_ids": ["report-a"],
        **changes,
    }


def body(*rows):
    return b"\n".join(json.dumps(value, ensure_ascii=False).encode("utf-8") for value in rows)


def attribution(**changes):
    return {
        "source_type": "official_filing",
        "actor": "NVIDIA",
        "claim_type": "reported_result",
        "source_locator": "PDF page 3",
        **changes,
    }


def financial(**changes):
    return {
        "document_id": "report-a", "statement": "income", "metric": "revenue",
        "period": "2026 Q2", "unit": "USD millions", "value": "123",
        "source_locator": "PDF page 3", "reviewer": "annotator",
        "source_check": "checked original PDF", **changes,
    }


def usage(**changes):
    return {
        "purpose": "development", "frozen_on": None,
        "development_exposed": True, "registration_note": "seen development set",
        **changes,
    }


def test_preserves_query_exactly_and_returns_frozen_question_tuple():
    query = " \tQuelle est la recette?\n财务\u00a0 "
    actual = inputs().parse_questions(b"\n\r\n" + body(row(query=query)) + b"\r\n\n")
    assert type(actual) is tuple and len(actual) == 1
    assert actual[0] == EvaluationQuestion("q1", query, True, (3,), ("report-a",))
    assert actual[0].query.encode("utf-8") == query.encode("utf-8")


def test_jsonl_separates_only_physical_newlines_and_preserves_unicode_separators():
    query = "What\u0085was\u2028the\u2029revenue?"
    actual, = inputs().parse_questions(body(row(query=query)))
    assert actual.query == query


def test_accepts_every_existing_optional_field_without_mode_inference():
    value = row(expected_answer="123", attribution_ground_truth=attribution(),
                financial_ground_truth=financial(), evaluation_usage=usage(development_exposed=False))
    actual, = inputs().parse_questions(body(value))
    assert actual.expected_answer == "123"
    assert actual.attribution_ground_truth == value["attribution_ground_truth"]
    assert actual.financial_ground_truth == value["financial_ground_truth"]
    assert actual.evaluation_usage == value["evaluation_usage"]


def test_accepts_canonical_dataclass_rows_with_all_optional_defaults_present():
    original = EvaluationQuestion("q1", "Exact query", True, (3,), ("report-a",))
    assert inputs().parse_questions(inputs().canonical_json(asdict(original))) == (original,)


def test_accepts_omitted_optional_fields_and_unregistered_questions():
    negative = row(question_id="negative", answerable=False, evidence_pages=[])
    del negative["evidence_document_ids"]
    actual, = inputs().parse_questions(body(negative))
    assert actual == EvaluationQuestion("negative", "What was revenue?", False, ())


def test_unanswerable_can_retain_valid_attribution_metadata():
    value = row(answerable=False, evidence_pages=[], evidence_document_ids=[],
                attribution_ground_truth=attribution(claim_type="question"))
    actual, = inputs().parse_questions(body(value))
    assert actual.attribution_ground_truth == value["attribution_ground_truth"]
    assert actual.evidence_pages == actual.evidence_document_ids == ()


def test_allows_distinct_document_page_pairs_and_repeated_document_ids():
    actual, = inputs().parse_questions(body(row(
        evidence_pages=[3, 3, 4], evidence_document_ids=["report-a", "report-b", "report-a"])))
    assert actual.evidence_pages == (3, 3, 4)


def test_empty_jsonl_has_no_questions():
    assert inputs().parse_questions(b"\n \t\r\n") == ()


@pytest.mark.parametrize("missing", ["question_id", "query", "answerable", "evidence_pages"])
def test_rejects_missing_required_field(missing):
    value = row()
    del value[missing]
    with pytest.raises(ValueError):
        inputs().parse_questions(body(value))


@pytest.mark.parametrize("value", [None, [], True, "q1", 42])
def test_rejects_nonobject_rows(value):
    with pytest.raises(ValueError):
        inputs().parse_questions(body(value))


def test_rejects_unknown_field_even_if_other_fields_are_valid():
    with pytest.raises(ValueError):
        inputs().parse_questions(body(row(extra="silently dropped by legacy loader")))


@pytest.mark.parametrize("field,value", [
    ("question_id", 1), ("question_id", None), ("question_id", True),
    ("question_id", " \t"), ("query", 1), ("query", []), ("query", None),
    ("query", " \t\n"), ("answerable", 1), ("answerable", 0),
    ("answerable", "true"), ("answerable", None),
    ("evidence_pages", "3"), ("evidence_pages", {}), ("evidence_pages", None),
    ("evidence_pages", [True]), ("evidence_pages", [3.0]), ("evidence_pages", ["3"]),
    ("evidence_pages", [None]), ("evidence_pages", [0]), ("evidence_pages", [-1]),
    ("evidence_pages", []), ("evidence_document_ids", "report-a"),
    ("evidence_document_ids", {}), ("evidence_document_ids", None),
    ("evidence_document_ids", [3]), ("evidence_document_ids", [True]),
    ("evidence_document_ids", [""]), ("evidence_document_ids", [" \t"]),
    ("evidence_document_ids", []), ("expected_answer", 123),
    ("expected_answer", True), ("expected_answer", {}),
])
def test_rejects_type_coercion_or_invalid_required_values(field, value):
    with pytest.raises(ValueError):
        inputs().parse_questions(body(row(**{field: value})))


def test_answerable_requires_document_qualified_labels():
    value = row()
    del value["evidence_document_ids"]
    with pytest.raises(ValueError):
        inputs().parse_questions(body(value))


@pytest.mark.parametrize("pages,documents", [([3, 4], ["report-a"]), ([3], ["report-a", "report-b"])])
def test_rejects_misaligned_evidence_labels(pages, documents):
    with pytest.raises(ValueError):
        inputs().parse_questions(body(row(evidence_pages=pages, evidence_document_ids=documents)))


def test_rejects_duplicate_document_page_pair():
    with pytest.raises(ValueError):
        inputs().parse_questions(body(row(evidence_pages=[3, 3], evidence_document_ids=["report-a"] * 2)))


@pytest.mark.parametrize("changes", [
    {"evidence_pages": [3]}, {"evidence_document_ids": ["report-a"]},
    {"expected_answer": "123"}, {"financial_ground_truth": financial()},
])
def test_unanswerable_rejects_evidence_or_answer_labels(changes):
    value = row(**{"answerable": False, "evidence_pages": [], "evidence_document_ids": [], **changes})
    with pytest.raises(ValueError):
        inputs().parse_questions(body(value))


@pytest.mark.parametrize("field", ["attribution_ground_truth", "financial_ground_truth", "evaluation_usage"])
@pytest.mark.parametrize("value", [[], "label", True, 1, {}])
def test_rejects_invalid_nested_object_types_or_fields(field, value):
    with pytest.raises(ValueError):
        inputs().parse_questions(body(row(**{field: value})))


@pytest.mark.parametrize("field,factory", [
    ("attribution_ground_truth", attribution), ("financial_ground_truth", financial),
])
@pytest.mark.parametrize("value", [None, [], {}, 1, True, " \t"])
def test_rejects_nonstrings_or_blank_strings_in_ground_truth(field, factory, value):
    labels = factory()
    labels[next(iter(labels))] = value
    with pytest.raises(ValueError):
        inputs().parse_questions(body(row(**{field: labels})))


@pytest.mark.parametrize("labels", [
    attribution(source_type="blog"), attribution(claim_type="fact"),
    {**attribution(), "extra": "x"},
])
def test_rejects_invalid_attribution_vocabulary_or_unknown_fields(labels):
    with pytest.raises(ValueError):
        inputs().parse_questions(body(row(attribution_ground_truth=labels)))


@pytest.mark.parametrize("labels", [
    usage(purpose=[]), usage(purpose="test"), usage(frozen_on=20261014),
    usage(frozen_on="2026-02-30"), usage(frozen_on="2026-1-1"),
    usage(development_exposed=1), usage(registration_note=None),
    usage(registration_note=" "), {**usage(), "extra": "x"},
    usage(purpose="acceptance_candidate", frozen_on=None, development_exposed=False),
    usage(purpose="acceptance_candidate", frozen_on="2026-10-14", development_exposed=True),
])
def test_rejects_invalid_registration_structure_but_does_not_choose_run_mode(labels):
    with pytest.raises(ValueError):
        inputs().parse_questions(body(row(evaluation_usage=labels)))


def test_accepts_existing_acceptance_candidate_registration_shape():
    value = usage(purpose="acceptance_candidate", frozen_on="2026-10-14", development_exposed=False)
    actual, = inputs().parse_questions(body(row(evaluation_usage=value)))
    assert actual.evaluation_usage == value


def test_rejects_duplicate_question_ids_across_jsonl_rows():
    with pytest.raises(ValueError, match="duplicate.*question_id|question_id.*duplicate"):
        inputs().parse_questions(body(row(), row(query="Different query")))


def test_duplicate_id_comparison_uses_decoded_string():
    raw = body(row()) + b'\n{"question_id":"\\u00711","query":"other","answerable":false,"evidence_pages":[]}'
    with pytest.raises(ValueError):
        inputs().parse_questions(raw)


def test_query_limit_is_decoded_characters_and_preserves_boundary():
    actual, = inputs().parse_questions(body(row(query="财" * 500)))
    assert actual.query == "财" * 500
    with pytest.raises(ValueError):
        inputs().parse_questions(body(row(query="x" * 501)))


def test_question_count_limit_counts_nonblank_rows():
    rows = [row(question_id=f"q{index}") for index in range(1001)]
    assert len(inputs().parse_questions(body(*rows[:1000]) + b"\n\n")) == 1000
    with pytest.raises(ValueError):
        inputs().parse_questions(body(*rows))


def test_question_raw_byte_limit_is_inclusive():
    raw = body(row())
    assert len(inputs().parse_questions(raw + b" " * (1024 * 1024 - len(raw)))) == 1
    with pytest.raises(ValueError):
        inputs().parse_questions(raw + b" " * (1024 * 1024 + 1 - len(raw)))


@pytest.mark.parametrize("raw", [b"\xff", b'{"key":"\xc0\xaf"}', b"{} {}", b"[1,", b"", b"\xef\xbb\xbf{}"])
def test_parse_json_rejects_invalid_utf8_or_non_single_json_body(raw):
    with pytest.raises(ValueError):
        inputs().parse_json(raw)


@pytest.mark.parametrize("raw", [
    b'{"query":"a","query":"b"}', b'{"a":1,"\\u0061":2}',
    b'{"outer":[{"grade":0,"\\u0067rade":1}]}',
    b'{"\\ud834\\udd1e":1,"\xf0\x9d\x84\x9e":2}',
])
def test_parse_json_rejects_duplicate_decoded_members_at_every_depth(raw):
    with pytest.raises(ValueError, match="duplicate"):
        inputs().parse_json(raw)


@pytest.mark.parametrize("raw", [
    b'"\\ud800"', b'{"\\ud800":"value"}', b'{"outer":[{"label":"\\udfff"}]}',
])
def test_parse_json_rejects_unpaired_surrogates_in_strings_and_keys(raw):
    with pytest.raises(ValueError, match="Unicode|surrogate"):
        inputs().parse_json(raw)


def test_parse_json_accepts_valid_escaped_surrogate_pairs_as_unicode_characters():
    actual = inputs().parse_json(b'{"\\ud834\\udd1e":"\\ud83d\\ude00"}')
    assert actual == {"𝄞": "😀"}
    assert inputs().canonical_json(actual) == '{"𝄞":"😀"}\n'.encode("utf-8")


def test_question_parser_rejects_escaped_unpaired_surrogate_query():
    raw = b'{"question_id":"q1","query":"\\ud800","answerable":false,"evidence_pages":[]}'
    with pytest.raises(ValueError, match="Unicode|surrogate"):
        inputs().parse_questions(raw)


@pytest.mark.parametrize("number", [b"NaN", b"Infinity", b"-Infinity", b"1e9999", b"-1e9999"])
def test_parse_json_rejects_nonfinite_numbers_at_every_depth(number):
    with pytest.raises(ValueError):
        inputs().parse_json(b'{"outer":[' + number + b"]}")


def test_parse_json_has_configurable_raw_byte_bound_for_review_artifacts():
    assert inputs().parse_json(b'{"a":1}', max_bytes=7) == {"a": 1}
    with pytest.raises(ValueError):
        inputs().parse_json(b'{"a":1}', max_bytes=6)
    large = b'"' + b"a" * (1024 * 1024) + b'"'
    with pytest.raises(ValueError):
        inputs().parse_json(large)
    assert inputs().parse_json(large, max_bytes=64 * 1024 * 1024) == "a" * (1024 * 1024)


def test_parse_json_retains_distinct_integer_float_and_boolean_types():
    actual = inputs().parse_json(b'{"values":[9007199254740993,1.0,true]}')
    integer, floating, boolean = actual["values"]
    assert type(integer) is int and integer == 9007199254740993
    assert type(floating) is float and floating == 1.0
    assert type(boolean) is bool and boolean is True
    assert inputs().canonical_json(actual) == b'{"values":[9007199254740993,1.0,true]}\n'


@pytest.mark.parametrize("raw", ["{}", bytearray(b"{}"), None])
def test_parsers_require_actual_bytes(raw):
    with pytest.raises(TypeError):
        inputs().parse_json(raw)
    with pytest.raises(TypeError):
        inputs().parse_questions(raw)


@pytest.mark.parametrize("raw", [
    b'{"question_id":"q1","query":"a","query":"b","answerable":false,"evidence_pages":[]}',
    b'{"question_id":"q1","query":"a","answerable":false,"evidence_pages":[],"unknown":NaN}',
    b'{"question_id":"q1","query":"\xff","answerable":false,"evidence_pages":[]}',
])
def test_question_parser_uses_strict_json_boundary(raw):
    with pytest.raises(ValueError):
        inputs().parse_questions(raw)


def test_canonical_json_uses_sorted_compact_utf8_and_exactly_one_newline():
    actual = inputs().canonical_json({"z": [True, None, 1.5], "a": "财务"})
    assert actual == '{"a":"财务","z":[true,null,1.5]}\n'.encode("utf-8")
    assert inputs().parse_json(actual) == {"z": [True, None, 1.5], "a": "财务"}


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_nonfinite_output_numbers(value):
    with pytest.raises(ValueError):
        inputs().canonical_json({"nested": [value]})
