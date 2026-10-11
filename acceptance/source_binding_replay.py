"""Verify four frozen source candidates against an operator-supplied local PDF.

No model or retriever is loaded. The receipt verifies source bindings and original
question bytes; it is neither human adjudication nor a financial quality score.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re

import pymupdf

PDF_SHA256 = "35d94221f0ed30f974b4fe6536fa54ff40945265d8b354b0ae17066c8856a861"
QUESTIONS_SHA256 = "4027ca9d287894f940d9836467547adc0950082be3536b0c1cbb795247c10ee6"
PAGE_TEXT_SHA256 = "224e3c92eb9d84d69dff20d606f4b3e3286aff01719fbc114f29855179954d3d"
REPORT_URL = "https://s201.q4cdn.com/141608511/files/doc_financials/2025/q2/78501ce3-7816-4c4d-8688-53dd140df456.pdf"
PROFILE = {
    "library": "PyMuPDF", "version": "1.28.2", "mode": "text", "sort": True,
    "normalization": "none", "text_encoding": "utf-8",
    "offset_unit": "unicode_code_points", "interval": "half_open",
}
QUESTION_QUALIFIERS = {
    "q2_regression_revenue_three_months": ("Revenue", 3),
    "q2_regression_revenue_six_months": ("Revenue", 6),
    "q2_regression_net_income_three_months": ("Net income", 3),
    "q2_regression_net_income_six_months": ("Net income", 6),
}
DEFAULT_CANDIDATES = (Path(__file__).resolve().parents[1] / "evaluation" /
                      "source-candidates" / "fy2025-q2-income-duration.jsonl")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_candidates(records: list[dict]) -> None:
    """Check the bounded candidate scope and frozen original-question bytes."""
    require(len(records) == 4, "exactly four development candidate records are required")
    require([row.get("question_id") for row in records] == list(QUESTION_QUALIFIERS),
            "original question IDs and order must remain unchanged")
    original_bytes = bytearray()
    for index, row in enumerate(records, 1):
        require(row.get("schema_version") == 1, "unsupported candidate schema")
        require(row.get("label_status") == "assistant-source-reviewed-candidate"
                and row.get("reviewer") == "CodexAI", "candidate review status must remain provisional")
        require(row.get("human_acceptance") is False and row.get("human_verdict") is None,
                "human acceptance is not established by this replay")
        require(row.get("freeze_purpose") == "development-replay"
                and row.get("development_exposed") is True,
                "candidate scope must remain development replay")
        require(row.get("original_question_file_sha256") == QUESTIONS_SHA256,
                "original question file hash differs from the frozen source")
        raw = row.get("original_question_record_utf8")
        require(isinstance(raw, str), "original question record bytes are missing")
        encoded = raw.encode("utf-8")
        require(sha256(encoded).hexdigest() == row.get("original_question_record_sha256"),
                "original question record hash mismatch")
        require(row.get("original_question_record_index") == index,
                "original question record index mismatch")
        original = json.loads(raw)
        require(original == row.get("original_question"), "original question object differs from its bytes")
        for field in ("question_id", "query", "answerable", "evidence_pages",
                      "evidence_document_ids", "expected_answer", "source_note"):
            require(row.get(field) == original.get(field), f"original question {field} changed")
        require(row.get("answerable") is True and row.get("evidence_pages") == [3]
                and row.get("evidence_document_ids") == ["nvidia-fy2025-q2"],
                "the four candidate source locations must remain unchanged")
        original_bytes.extend(encoded)

        source = row.get("source", {})
        metric, duration = QUESTION_QUALIFIERS[row["question_id"]]
        require(source.get("document_id") == "nvidia-fy2025-q2"
                and source.get("pdf_sha256") == PDF_SHA256
                and source.get("report_url") == REPORT_URL,
                "registered report identity changed")
        require(source.get("physical_page") == 3 and source.get("printed_page") == "3"
                and source.get("page_count") == 80, "report page identity changed")
        require(source.get("statement") == "Condensed Consolidated Statements of Income"
                and source.get("row") == metric and source.get("duration_months") == duration
                and source.get("period_end") == "2024-07-28"
                and source.get("fiscal_year") == 2025 and source.get("fiscal_quarter") == 2,
                "original financial field or period qualifiers changed")
        require(source.get("unit") == "million dollars" and source.get("currency_literal") == "$"
                and source.get("normalized_currency") == "USD"
                and source.get("currency_normalization_status") == "assistant-source-reviewed-candidate",
                "source currency/unit qualification changed")
        amount = re.fullmatch(r"\$([\d,]+) million", original["expected_answer"])
        require(amount is not None and type(source.get("value")) is int
                and source["value"] == int(amount[1].replace(",", ""))
                and source.get("value_text") == amount[1], "original candidate value changed")
        bindings = row.get("source_bindings", {})
        require(bindings.get("page_text_sha256") == PAGE_TEXT_SHA256
                and bindings.get("extraction_profile") == PROFILE,
                "frozen page extraction profile changed")
    require(sha256(original_bytes).hexdigest() == QUESTIONS_SHA256,
            "reconstructed original question file hash mismatch")


def load_candidates(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_candidates(records)
    return records


def verify_spans(page_text: str, record: dict) -> None:
    source = record["source"]
    expected_text = {
        "statement": source["statement"],
        "unit_heading": "In millions, except per share data",
        "duration_heading": "Three Months Ended" if source["duration_months"] == 3 else "Six Months Ended",
        "period_end_heading": "Jul 28, 2024",
        "row": source["row"], "value": source["value_text"], "printed_page": "3",
    }
    spans = record["source_bindings"].get("spans", {})
    require(set(spans) == set(expected_text), "required source header/value spans are missing")
    for name, expected in expected_text.items():
        span = spans[name]
        start, end = span.get("start"), span.get("end")
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(page_text),
                f"invalid {name} source span")
        require(span.get("text") == expected and page_text[start:end] == expected,
                f"{name} source span does not match the frozen page")
    # The source hash freezes geometry; preserve the actual row/group/column
    # binding instead of accepting another occurrence of an identical header.
    expected_positions = {
        ("Revenue", 3): (696, 703, 492, 510, 619, 631, 766, 772),
        ("Revenue", 6): (696, 703, 530, 546, 661, 673, 804, 810),
        ("Net income", 3): (2523, 2533, 492, 510, 619, 631, 2594, 2600),
        ("Net income", 6): (2523, 2533, 530, 546, 661, 673, 2632, 2638),
    }
    observed = tuple(spans[name][key] for name in ("row", "duration_heading", "period_end_heading", "value")
                     for key in ("start", "end"))
    require(observed == expected_positions[(source["row"], source["duration_months"])],
            "financial row/duration/column binding differs from the reviewed candidate")


def replay(pdf_path: Path, candidates_path: Path = DEFAULT_CANDIDATES) -> dict:
    require(pdf_path.is_absolute(), "--pdf must be an absolute local path")
    records = load_candidates(candidates_path)
    pdf_bytes = pdf_path.read_bytes()
    require(sha256(pdf_bytes).hexdigest() == PDF_SHA256, "PDF SHA-256 differs from the registered source")
    require(pymupdf.version[0] == PROFILE["version"], "PyMuPDF version differs from the frozen extraction profile")
    with pymupdf.open(pdf_path) as document:
        require(len(document) == 80, "registered physical page count mismatch")
        page = document[2]
        page_text = page.get_text("text", sort=True)
        require(sha256(page_text.encode("utf-8")).hexdigest() == PAGE_TEXT_SHA256,
                "extracted page text SHA-256 mismatch")
        checks = []
        for record in records:
            verify_spans(page_text, record)
            boxes = page.search_for(record["source"]["value_text"])
            expected_box = record["source"].get("value_bbox_points")
            require(isinstance(expected_box, list) and len(expected_box) == 4
                    and any(all(abs(a - b) <= 0.01 for a, b in zip(rect, expected_box)) for rect in boxes),
                    "value bounding box differs from the reviewed source cell")
            checks.append({
                "question_id": record["question_id"], "source_binding_passed": True,
                "label_status": record["label_status"], "reviewer": "CodexAI",
                "human_acceptance": False, "human_verdict": None,
                "source": record["source"], "verified_spans": record["source_bindings"]["spans"],
                "original_question_record_sha256": record["original_question_record_sha256"],
            })
    return {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "frozen source-binding replay; assistant candidates; no human acceptance or model quality score",
        "freeze_purpose": "development-replay", "source_binding_passed": True,
        "human_acceptance": False, "model_runs": 0,
        "candidate_file": str(candidates_path.resolve()),
        "candidate_file_sha256": sha256(candidates_path.read_bytes()).hexdigest(),
        "candidate_count": len(records), "negative_count": 0,
        "source_pdf": str(pdf_path), "source_pdf_sha256": PDF_SHA256,
        "source_pdf_bytes": len(pdf_bytes), "original_question_file_sha256": QUESTIONS_SHA256,
        "reconstructed_original_question_bytes_match": True,
        "page_text_sha256": PAGE_TEXT_SHA256, "extraction_profile": PROFILE,
        "checks": checks,
        "not_evaluated": ["retrieval ranking", "refusal reliability", "generation",
                          "financial accuracy", "independent human adjudication", "latency", "throughput"],
    }


def write_receipt(path: Path, receipt: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True, help="Absolute path to the registered local PDF; no download or copy")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, help="Create a new JSON receipt without overwriting")
    args = parser.parse_args()
    if not args.pdf.is_absolute():
        parser.error("--pdf must be an absolute local path")
    try:
        receipt = replay(args.pdf, args.candidates)
        if args.output is not None:
            write_receipt(args.output, receipt)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
