"""Explicit-field selection and raw lexical coverage contracts, not NLU scores."""
from copy import deepcopy
from pathlib import Path
import json
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "acceptance" / "table_field_coverage_experiment.py"
UNIT = "In millions, except per share data"


def api():
    assert SCRIPT.exists(), "standalone field coverage experiment must be implemented"
    return runpy.run_path(str(SCRIPT))


def profile(metric="Net income", duration=3, end="2024-07-28", unit=UNIT):
    return {"metric": metric, "duration_months": duration,
            "period_end": end, "unit_literal": unit}


def cell(identifier="a", duration=3, end="2024-07-28", value="901"):
    return {"cell_id": identifier, **profile(duration=duration, end=end),
            "value_text": value, "value": int(value),
            "bindings": {"row": {"text": "Net income"}, "unit": {"text": UNIT},
                         "duration": {"text": "Three Months Ended" if duration == 3 else "Six Months Ended"},
                         "period_end": {"text": "Jul 28, 2024" if end == "2024-07-28" else "Jul 30, 2023"},
                         "value": {"text": value}}}


def test_exact_fields_select_source_value_not_expected_answer():
    ns = api()
    rows = [cell("prior", end="2023-07-30", value="303"), cell("six", duration=6, value="404"), cell()]
    selected = ns["select_cell"](rows, profile())
    assert selected == {"status": "selected", "cell_id": "a", "value_text": "901", "value": 901}
    assert ns["select_cell"](rows, profile(duration=6))["value_text"] == "404"
    assert ns["select_cell"](rows, profile(end="2023-07-30"))["value_text"] == "303"
    rows[-1] = cell(value="902")
    assert ns["select_cell"](rows, profile())["value_text"] == "902"


@pytest.mark.parametrize("change", [
    {"metric": "Revenue"}, {"duration_months": 12}, {"period_end": "2025-07-28"},
    {"unit_literal": "thousands"},
])
def test_no_guess_for_uncovered_exact_tuple(change):
    ns = api()
    request = {**profile(), **change}
    assert ns["select_cell"]([cell()], request) == {"status": "refused", "reason": "NO_EXACT_CELL"}


@pytest.mark.parametrize("change", [
    {"duration_months": None}, {"duration_months": True}, {"period_end": "2024-02-30"},
    {"period_end": "July 28 2024"}, {"metric": ""}, {"unit_literal": None}, {"expected_answer": "999"},
])
def test_invalid_or_incomplete_query_profile_refuses(change):
    ns = api()
    assert ns["select_cell"]([cell()], {**profile(), **change}) == {
        "status": "refused", "reason": "INVALID_QUERY_PROFILE"}
    missing = profile()
    del missing["duration_months"]
    assert ns["select_cell"]([cell()], missing)["reason"] == "INVALID_QUERY_PROFILE"


@pytest.mark.parametrize("second_value", ["901", "999"])
def test_duplicate_tuple_refuses_even_when_values_agree(second_value):
    ns = api()
    assert ns["select_cell"]([cell(), cell("other", value=second_value)], profile()) == {
        "status": "refused", "reason": "AMBIGUOUS_EXACT_CELL"}


def test_duplicate_identity_and_invalid_cell_fields_refuse():
    ns = api()
    assert ns["select_cell"]([cell(), cell(duration=6)], profile())["reason"] == "DUPLICATE_CELL_ID"
    invalid = cell()
    invalid["bindings"]["unit"]["text"] = "thousands"
    assert ns["select_cell"]([invalid], profile())["reason"] == "INVALID_CELL_FIELDS"
    invalid = cell()
    invalid["value"] = 12345
    assert ns["select_cell"]([invalid], profile())["reason"] == "INVALID_CELL_FIELDS"


def raw_fixture():
    text = "Statement\n(" + UNIT + ")\nThree Months Ended\nSix Months Ended\nJul 28, 2024\nJul 28, 2024\nNet income\n901\n"
    start = text.index("901")
    chunk = {"chunk_id": "chunk", "document_id": "doc", "page_number": 3, "text": text}
    candidate = {"source": {"row": "Net income", "duration_months": 3, "period_end": "2024-07-28",
                            "value_text": "901", "document_id": "doc", "physical_page": 3},
                 "original_chunk_binding": {"chunk_id": "chunk", "corpus_sha256": "digest",
                    "offset_unit": "unicode_code_points", "interval": "half_open",
                    "value_span": {"start": start, "end": start + 3, "text": "901"}}}
    return chunk, candidate


def test_raw_value_binding_and_lexical_presence_do_not_claim_entailment():
    ns = api()
    chunk, candidate = raw_fixture()
    report = ns["raw_chunk_coverage"](chunk, candidate, profile(), "digest")
    assert report["value_span_verified"] is True
    assert all(report["lexical_presence"].values())
    assert report["lexical_occurrences"]["period_end"] == 2
    assert report["semantic_field_entailment_established_here"] is False
    assert report["deterministic_cell_selection"] == "not-evaluated"


@pytest.mark.parametrize("mutate, reason", [
    (lambda c, q: q["original_chunk_binding"]["value_span"].update(start=0), "RAW_VALUE_SPAN_MISMATCH"),
    (lambda c, q: c.update(document_id="other"), "RAW_SOURCE_IDENTITY_MISMATCH"),
    (lambda c, q: q["original_chunk_binding"].update(offset_unit="utf16"), "RAW_BINDING_PROFILE_MISMATCH"),
    (lambda c, q: q["original_chunk_binding"].update(corpus_sha256="other"), "RAW_SOURCE_IDENTITY_MISMATCH"),
    (lambda c, q: q["source"].update(duration_months=6), "QUERY_SOURCE_PROFILE_MISMATCH"),
])
def test_raw_coverage_refuses_conflicting_binding(mutate, reason):
    ns = api()
    chunk, candidate = raw_fixture()
    mutate(chunk, candidate)
    with pytest.raises(ValueError, match=reason):
        ns["raw_chunk_coverage"](chunk, candidate, profile(), "digest")


def test_missing_raw_unit_is_lexical_gap_not_invented_support():
    ns = api()
    chunk, candidate = raw_fixture()
    # Replacement has the same length, preserving the controlled value offset.
    chunk["text"] = chunk["text"].replace(UNIT, "x" * len(UNIT))
    report = ns["raw_chunk_coverage"](chunk, candidate, profile(), "digest")
    assert report["value_span_verified"] is True
    assert report["lexical_presence"]["unit"] is False


def test_checked_digest_refuses_wrong_bytes_before_followup(tmp_path):
    ns = api()
    path = tmp_path / "source"
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="INPUT_DIGEST_MISMATCH"):
        ns["checked_digest"](path, "0" * 64)


def test_artifact_creation_exclusive_digest_and_code_validation(tmp_path):
    ns = api()
    directory = tmp_path / "exclusive"
    report = {"scope": "controlled", "model_runs": 0}
    receipt = ns["write_artifacts"](directory, report)
    original_body = (directory / "comparison.json").read_bytes()
    assert ns["verify_report_digest"](directory) == report
    with pytest.raises(FileExistsError):
        ns["write_artifacts"](directory, report)
    (directory / "comparison.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="REPORT_DIGEST_MISMATCH"):
        ns["verify_report_digest"](directory)
    (directory / "comparison.json").write_bytes(original_body)
    receipt["script_sha256"] = "0" * 64
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="BENCHMARK_CODE_MISMATCH"):
        ns["verify_report_digest"](directory)
