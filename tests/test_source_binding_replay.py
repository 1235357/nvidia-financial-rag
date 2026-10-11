"""Portable regression checks for candidate provenance, not financial accuracy."""

from copy import deepcopy
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "acceptance" / "source_binding_replay.py"
CANDIDATES = ROOT / "evaluation" / "source-candidates" / "fy2025-q2-income-duration.jsonl"


def functions():
    assert SCRIPT.exists(), "source-binding replay script must be implemented"
    return runpy.run_path(str(SCRIPT))


def test_four_candidate_records_preserve_original_questions_and_development_scope():
    records = functions()["load_candidates"](CANDIDATES)
    assert len(records) == 4
    assert len({row["question_id"] for row in records}) == 4
    for row in records:
        assert row["label_status"] == "assistant-source-reviewed-candidate"
        assert row["reviewer"] == "CodexAI"
        assert row["human_acceptance"] is False
        assert row["freeze_purpose"] == "development-replay"
        assert row["query"] == row["original_question"]["query"]


@pytest.mark.parametrize("change", [
    "human_acceptance", "label_status", "freeze_purpose", "query", "value", "duration",
])
def test_candidate_metadata_cannot_promote_review_or_change_original_qualifiers(change):
    namespace = functions()
    records = deepcopy(namespace["load_candidates"](CANDIDATES))
    if change == "human_acceptance":
        records[0][change] = True
    elif change == "label_status":
        records[0][change] = "independent-human-gold"
    elif change == "freeze_purpose":
        records[0][change] = "acceptance"
    elif change == "query":
        records[0][change] = "What was revenue in another fiscal year?"
    elif change == "value":
        records[0]["source"]["value"] = 56084
    else:
        records[0]["source"]["duration_months"] = 6
    with pytest.raises(ValueError):
        namespace["validate_candidates"](records)


def test_modified_original_question_bytes_fail_provenance_check():
    namespace = functions()
    records = deepcopy(namespace["load_candidates"](CANDIDATES))
    records[0]["original_question_record_utf8"] += " "
    with pytest.raises(ValueError, match="original question"):
        namespace["validate_candidates"](records)


def test_wrong_pdf_hash_fails_before_pdf_parser(tmp_path, monkeypatch):
    namespace = functions()
    wrong_pdf = tmp_path / "wrong-source.pdf"
    wrong_pdf.write_bytes(b"synthetic wrong PDF bytes")

    def forbidden_open(*args, **kwargs):
        pytest.fail("PDF parser must not run after source hash mismatch")

    monkeypatch.setattr(namespace["pymupdf"], "open", forbidden_open)
    with pytest.raises(ValueError, match="PDF SHA-256"):
        namespace["replay"](wrong_pdf, CANDIDATES)


def test_identical_date_text_in_other_duration_column_cannot_rebind_candidate():
    namespace = functions()
    records = namespace["load_candidates"](CANDIDATES)
    # Synthetic text tests the same-text/wrong-column boundary only. It is not
    # a new financial question, source label, or PDF extraction acceptance.
    characters = [" "] * 3500
    for record in records:
        for span in record["source_bindings"]["spans"].values():
            characters[span["start"]:span["end"]] = span["text"]
    page_text = "".join(characters)
    candidate = deepcopy(records[0])
    candidate["source_bindings"]["spans"]["period_end_heading"] = deepcopy(
        records[1]["source_bindings"]["spans"]["period_end_heading"])
    with pytest.raises(ValueError, match="column binding"):
        namespace["verify_spans"](page_text, candidate)


def test_cli_requires_absolute_pdf_path():
    assert SCRIPT.exists(), "source-binding replay script must be implemented"
    result = subprocess.run([sys.executable, str(SCRIPT), "--pdf", "relative.pdf"],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert "absolute" in result.stderr


def test_receipt_never_overwrites_existing_file(tmp_path):
    namespace = functions()
    output = tmp_path / "receipt.json"
    output.write_text("preserved", encoding="utf-8")
    with pytest.raises(FileExistsError):
        namespace["write_receipt"](output, {"source_binding_passed": True})
    assert output.read_text(encoding="utf-8") == "preserved"
