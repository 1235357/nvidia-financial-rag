"""Frozen FY2025 Q2 page-3 Net income cells, derived without labels or models.

This local development experiment is outside the package/API. It is not a
general table parser, financial verifier, human gold set or retrieval adapter.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re

import pymupdf

PDF_SHA256 = "35d94221f0ed30f974b4fe6536fa54ff40945265d8b354b0ae17066c8856a861"
PAGE_TEXT_SHA256 = "224e3c92eb9d84d69dff20d606f4b3e3286aff01719fbc114f29855179954d3d"
DOCUMENT_ID = "nvidia-fy2025-q2"
PROFILE = {"library": "PyMuPDF", "version": "1.28.2", "mode": "text-and-words",
           "sort": True, "normalization": "none", "offset_unit": "unicode_code_points",
           "interval": "half_open", "geometry_unit": "PDF points",
           "policy": "q2-page3-net-income-cells-v1"}
STATEMENT = "Condensed Consolidated Statements of Income"
UNIT = "In millions, except per share data"
AMOUNT = re.compile(r"\(?\d{1,3}(?:,\d{3})*\)?\Z")
DATE = re.compile(r"Jul (\d{1,2}), (20\d{2})\Z")
ROOT = Path(__file__).resolve().parents[1]


def require(condition, code):
    if not condition:
        raise ValueError(code)


def digest_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounds(words):
    return [min(w[0] for w in words), min(w[1] for w in words),
            max(w[2] for w in words), max(w[3] for w in words)]


def phrase_groups(words):
    groups = defaultdict(list)
    for word in words:
        groups[(word[5], word[6])].append(word)
    result = []
    for group in groups.values():
        ordered = sorted(group, key=lambda word: word[0])
        result.append({"text": " ".join(w[4] for w in ordered), "words": ordered,
                       "bbox": bounds(ordered)})
    return result


def matches(text, phrase):
    pattern = r"\s+".join(re.escape(part) for part in phrase.split(" "))
    return list(re.finditer(pattern, text))


def binding(text, phrase, box, occurrence=0, total=1):
    found = matches(text, phrase)
    require(len(found) == total, "TEXT_GEOMETRY_MISMATCH")
    item = found[occurrence]
    return {"start": item.start(), "end": item.end(), "text": item.group(),
            "bbox_points": list(box)}


def derive_cells(page_text, raw_words, page_rect):
    """Pure geometry/text derivation. PDF authority is checked by extract_from_pdf."""
    require(isinstance(page_text, str) and len(page_text) <= 100000
            and 0 < len(raw_words) <= 10000, "INVALID_GEOMETRY")
    require(len(page_rect) == 4 and all(type(v) in (int, float) and math.isfinite(v)
                                      for v in page_rect), "INVALID_GEOMETRY")
    words = []
    for word in raw_words:
        require(len(word) == 8 and isinstance(word[4], str) and 0 < len(word[4]) < 2048
                and all(type(v) in (int, float) and math.isfinite(v) for v in word[:4])
                and page_rect[0] <= word[0] < word[2] <= page_rect[2]
                and page_rect[1] <= word[1] < word[3] <= page_rect[3]
                and all(type(v) is int and v >= 0 for v in word[5:]), "INVALID_GEOMETRY")
        words.append(tuple(word))
    groups = phrase_groups(words)

    def unique(phrase, code, strip_parens=False):
        found = [g for g in groups if (g["text"].strip("()") if strip_parens
                                      else g["text"]) == phrase]
        require(len(found) == 1, code)
        return found[0]

    statement = unique(STATEMENT, "AMBIGUOUS_STATEMENT")
    unit = unique(UNIT, "AMBIGUOUS_UNIT", True)
    row = unique("Net income", "AMBIGUOUS_ROW")
    three = unique("Three Months Ended", "AMBIGUOUS_DURATION")
    six = unique("Six Months Ended", "AMBIGUOUS_DURATION")
    center = lambda g: (g["bbox"][0] + g["bbox"][2]) / 2
    require(center(three) < center(six) and abs(three["bbox"][1] - six["bbox"][1]) <= 1.5,
            "AMBIGUOUS_DURATION")
    require(statement["bbox"][3] <= unit["bbox"][1]
            and unit["bbox"][3] < three["bbox"][1]
            and row["bbox"][1] > three["bbox"][3], "AMBIGUOUS_DURATION")
    dates = sorted([g for g in groups if DATE.fullmatch(g["text"])
                    and 0 < g["bbox"][1] - three["bbox"][1] <= 25], key=center)
    require(len(dates) == 4 and max(g["bbox"][1] for g in dates)
            - min(g["bbox"][1] for g in dates) <= 1.5, "AMBIGUOUS_DATE_COLUMNS")
    require(all(a["bbox"][2] < b["bbox"][0] for a, b in zip(dates, dates[1:])),
            "AMBIGUOUS_DATE_COLUMNS")
    boundary = (center(three) + center(six)) / 2
    duration_groups = [(3, three, [g for g in dates if center(g) < boundary]),
                       (6, six, [g for g in dates if center(g) > boundary])]
    for _, heading, members in duration_groups:
        require(len(members) == 2 and abs(sum(center(g) for g in members) / 2
                                        - center(heading)) <= 10,
                "AMBIGUOUS_DATE_COLUMNS")
    row_y = row["bbox"][1]
    amounts = sorted([w for w in words if AMOUNT.fullmatch(w[4])
                      and abs(w[1] - row_y) <= 1.5 and w[0] > row["bbox"][2]], key=lambda w: w[0])
    require(len(amounts) == 4, "AMBIGUOUS_ROW_VALUES")
    currencies = sorted([w for w in words if w[4] == "$" and abs(w[1] - row_y) <= 1.5
                         and w[0] > row["bbox"][2]], key=lambda w: w[0])
    require(len(currencies) == 4, "AMBIGUOUS_CURRENCY")
    for date, amount, currency in zip(dates, amounts, currencies):
        require(0 < amount[2] - date["bbox"][2] < 20,
                "AMBIGUOUS_VALUE_COLUMN")
        require(0 < amount[0] - currency[2] < 70
                and currency[0] < amount[0], "AMBIGUOUS_CURRENCY")
    row_matches = list(re.finditer(r"(?m)^(Net[ \t]+income)[ \t]+(?=[$(\d])([^\n]+)$", page_text))
    require(len(row_matches) == 1, "TEXT_GEOMETRY_MISMATCH")
    row_match = row_matches[0]
    values_in_text = list(re.finditer(r"(?<![\w,])\(?\d{1,3}(?:,\d{3})*\)?(?![\w,])", row_match[2]))
    require([m.group() for m in values_in_text] == [a[4] for a in amounts], "TEXT_GEOMETRY_MISMATCH")
    page_center = (page_rect[0] + page_rect[2]) / 2
    printed = [g["words"][0] for g in groups if len(g["words"]) == 1
               and re.fullmatch(r"\d+", g["text"])
               and abs(center(g) - page_center) <= 10
               and g["bbox"][1] > row["bbox"][3] + 20]
    require(len(printed) == 1, "AMBIGUOUS_PRINTED_PAGE")
    require(printed[0][1] >= max(w[3] for w in words if w != printed[0]) + 10,
            "AMBIGUOUS_PRINTED_PAGE")
    printed_text = list(re.finditer(r"(?m)^[ \t]*(\d+)[ \t]*$", page_text))
    require(len(printed_text) == 1 and printed_text[0][1] == printed[0][4],
            "TEXT_GEOMETRY_MISMATCH")
    printed_binding = {"start": printed_text[0].start(1), "end": printed_text[0].end(1),
                       "text": printed_text[0][1], "bbox_points": list(printed[0][:4])}
    row_binding = {"start": row_match.start(1), "end": row_match.end(1),
                   "text": row_match[1], "bbox_points": row["bbox"]}
    shared = {"statement": binding(page_text, STATEMENT, statement["bbox"]),
              "unit": binding(page_text, UNIT, unit["bbox"]), "row": row_binding,
              "printed_page": printed_binding}
    cells = []
    same_date_seen = defaultdict(int)
    page_sha = sha256(page_text.encode("utf-8")).hexdigest()
    for date_group, amount, value_match, currency in zip(dates, amounts, values_in_text, currencies):
        duration, heading, _ = next(item for item in duration_groups if date_group in item[2])
        date_match = DATE.fullmatch(date_group["text"])
        try:
            period_end = datetime(int(date_match[2]), 7, int(date_match[1])).date().isoformat()
        except ValueError:
            raise ValueError("AMBIGUOUS_DATE_COLUMNS") from None
        same_date_total = sum(g["text"] == date_group["text"] for g in dates)
        offset = row_match.start(2) + value_match.start()
        bindings = {**shared, "duration": binding(page_text, heading["text"], heading["bbox"]),
                    "period_end": binding(page_text, date_group["text"], date_group["bbox"],
                                          same_date_seen[date_group["text"]], same_date_total),
                    "value": {"start": offset, "end": offset + len(amount[4]),
                              "text": amount[4], "bbox_points": list(amount[:4])}}
        same_date_seen[date_group["text"]] += 1
        require(all(page_text[b["start"]:b["end"]] == b["text"] for b in bindings.values()),
                "TEXT_GEOMETRY_MISMATCH")
        number = int(amount[4].strip("()").replace(",", "")) * (-1 if amount[4].startswith("(") else 1)
        identity = json.dumps({"page_text_sha256": page_sha, "policy": PROFILE["policy"],
                               "bindings": bindings}, sort_keys=True, separators=(",", ":"))
        cells.append({"schema_version": 1, "cell_id": "net-income-cell:" + sha256(identity.encode()).hexdigest(),
                      "metric": row_binding["text"], "duration_months": duration,
                      "period_end": period_end, "value_text": amount[4], "value": number,
                      "unit_literal": bindings["unit"]["text"], "currency_literal": currency[4],
                      "currency_bbox_points": list(currency[:4]), "normalized_currency": None,
                      "currency_normalization": "not-established-by-this-extractor",
                      "printed_page": printed[0][4], "bindings": bindings,
                      "page_text_sha256": page_sha, "extraction_profile": PROFILE,
                      "review_status": "AI-source-reviewed-experimental", "reviewer": "CodexAI",
                      "human_acceptance": False, "human_verdict": None,
                      "development_exposed": True, "retrieval_integration": "not-run"})
    require(len({cell["cell_id"] for cell in cells}) == 4, "DUPLICATE_CELL")
    return cells


def extract_from_pdf(pdf_path):
    path = Path(pdf_path)
    require(path.is_absolute() and path.suffix.lower() == ".pdf", "INVALID_SOURCE_PATH")
    require(digest_file(path) == PDF_SHA256, "SOURCE_HASH_MISMATCH")
    require(pymupdf.VersionBind == PROFILE["version"], "EXTRACTION_VERSION_MISMATCH")
    with pymupdf.open(path) as document:
        require(len(document) == 80, "SOURCE_PAGE_COUNT_MISMATCH")
        page = document[2]
        text = page.get_text("text", sort=True)
        require(sha256(text.encode()).hexdigest() == PAGE_TEXT_SHA256, "SOURCE_TEXT_HASH_MISMATCH")
        cells = derive_cells(text, page.get_text("words", sort=True), tuple(page.rect))
    for cell in cells:
        cell["source"] = {"document_id": DOCUMENT_ID, "pdf_sha256": PDF_SHA256,
                          "physical_page": 3, "printed_page": cell["printed_page"]}
        cell["cell_id"] = DOCUMENT_ID + ":p3:net-income-cell:" + sha256(
            (PDF_SHA256 + "|" + cell["cell_id"]).encode()).hexdigest()
        cell["retrieval_text_candidate"] = (
            f"{STATEMENT} | {cell['metric']} | {cell['bindings']['duration']['text']} "
            f"{cell['bindings']['period_end']['text']} | raw value {cell['value_text']} | "
            f"{cell['unit_literal']} | source currency literal {cell['currency_literal']} | "
            f"{DOCUMENT_ID} physical page 3")
    return cells, {"source_sha256": PDF_SHA256, "page_text_sha256": PAGE_TEXT_SHA256,
                   "document_id": DOCUMENT_ID, "physical_page": 3, "physical_page_count": 80,
                   "profile": PROFILE, "source_hash_and_profile_checked": True}


def write_artifacts(output_dir, cells, source):
    require(len(cells) == 4 and len({cell["cell_id"] for cell in cells}) == 4, "DUPLICATE_CELL")
    body = b"".join((json.dumps(cell, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
                    for cell in cells)
    receipt = {"schema_version": 1, "status": "AI-source-reviewed-experimental",
               "human_acceptance": False, "human_verdict": None, "cell_count": 4,
               "cells_sha256": sha256(body).hexdigest(), "cells_bytes": len(body), "source": source,
               "script_sha256": digest_file(__file__), "embedding_runs": 0, "model_runs": 0,
               "retrieval_integration": "not-run", "raw_corpus_modified": False,
               "created_at_utc": datetime.now(timezone.utc).isoformat()}
    directory = Path(output_dir)
    directory.mkdir(parents=False, exist_ok=False)
    with (directory / "cells.jsonl").open("xb") as target:
        target.write(body)
    with (directory / "receipt.json").open("x", encoding="utf-8", newline="\n") as target:
        json.dump(receipt, target, indent=2, sort_keys=True)
        target.write("\n")
    return receipt


def verify_artifacts(directory, pdf_path=None):
    directory = Path(directory)
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    body = (directory / "cells.jsonl").read_bytes()
    require(sha256(body).hexdigest() == receipt["cells_sha256"]
            and len(body) == receipt["cells_bytes"], "ARTIFACT_DIGEST_MISMATCH")
    cells = [json.loads(line) for line in body.decode("utf-8").splitlines()]
    require(len(cells) == receipt["cell_count"] == 4, "ARTIFACT_DIGEST_MISMATCH")
    if pdf_path is not None:
        fresh, source = extract_from_pdf(pdf_path)
        require(fresh == cells and source == receipt["source"], "SOURCE_REPLAY_MISMATCH")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    require(args.output_dir.is_absolute() and args.output_dir.resolve().is_relative_to(
        (ROOT / "artifacts").resolve()), "OUTPUT_OUTSIDE_STAGED_ARTIFACTS")
    if args.verify_existing:
        receipt = verify_artifacts(args.output_dir, args.pdf)
    else:
        cells, source = extract_from_pdf(args.pdf)
        receipt = write_artifacts(args.output_dir, cells, source)
    print(json.dumps({"status": receipt["status"], "cell_count": receipt["cell_count"],
                      "cells_sha256": receipt["cells_sha256"], "output_dir": str(args.output_dir),
                      "source_replay": bool(args.verify_existing), "embedding_runs": 0,
                      "model_runs": 0, "human_acceptance": False}, indent=2))


if __name__ == "__main__":
    main()
