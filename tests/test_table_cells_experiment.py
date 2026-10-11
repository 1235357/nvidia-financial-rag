"""Controlled cell-geometry/provenance tests; not financial quality labels."""
from copy import deepcopy
from pathlib import Path
import json
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "acceptance" / "table_cells_experiment.py"


def api():
    assert SCRIPT.exists(), "bounded table-cell experiment must be implemented"
    return runpy.run_path(str(SCRIPT))


def fixture(values=("101", "202", "303", "404")):
    words = []

    def phrase(text, center, y, block, line):
        tokens = text.split(" ")
        width = sum(len(token) * 4 for token in tokens) + 2 * (len(tokens) - 1)
        x = center - width / 2
        for index, token in enumerate(tokens):
            words.append((x, y, x + len(token) * 4, y + 9, token, block, line, index))
            x += len(token) * 4 + 2
        return center + width / 2

    phrase("Condensed Consolidated Statements of Income", 380, 80, 0, 0)
    phrase("(In millions, except per share data)", 380, 100, 1, 0)
    phrase("Three Months Ended", 375, 140, 2, 0)
    phrase("Six Months Ended", 575, 140, 2, 1)
    dates = ("Jul 28, 2024", "Jul 30, 2023", "Jul 28, 2024", "Jul 30, 2023")
    for i, (center, date) in enumerate(zip((325, 425, 525, 625), dates)):
        right = phrase(date, center, 160, 3, i)
        words.append((right - 65, 350, right - 60, 359, "$", 5, 2 * i + 1, 0))
        words.append((right - 20, 350, right + 5, 359, values[i], 5, 2 * i + 2, 0))
    phrase("Net income", 45, 350, 5, 0)
    phrase("3", 400, 860, 6, 0)
    text = ("Condensed Consolidated Statements of Income\n"
            "(In millions, except per share data)\n"
            "Three Months Ended    Six Months Ended\n" + "    ".join(dates) + "\n"
            "Net income    " + "    ".join("$ " + value for value in values) + "\n"
            "Other notes contain 101 but are not the row\n3\n")
    return text, words, (0, 0, 800, 900)


def test_derives_four_cells_from_geometry_and_exact_row_spans():
    ns = api()
    text, words, rect = fixture()
    cells = ns["derive_cells"](text, words, rect)
    assert [(cell["duration_months"], cell["period_end"], cell["value_text"]) for cell in cells] == [
        (3, "2024-07-28", "101"), (3, "2023-07-30", "202"),
        (6, "2024-07-28", "303"), (6, "2023-07-30", "404"),
    ]
    for cell in cells:
        for binding in cell["bindings"].values():
            assert text[binding["start"]:binding["end"]] == binding["text"]
            assert len(binding["bbox_points"]) == 4
        assert cell["value"] == int(cell["value_text"])
        assert cell["review_status"] == "AI-source-reviewed-experimental"
        assert cell["human_acceptance"] is False
        assert cell["retrieval_integration"] == "not-run"
    # The duplicate 101 in a note cannot replace the row-bound occurrence.
    assert cells[0]["bindings"]["value"]["end"] < text.index("Other notes")


def test_cell_discovery_ignores_input_word_order_and_uses_changed_source_values():
    ns = api()
    text, words, rect = fixture(("901", "802", "703", "604"))
    cells = ns["derive_cells"](text, list(reversed(words)), rect)
    assert [cell["value_text"] for cell in cells] == ["901", "802", "703", "604"]


@pytest.mark.parametrize("part,code", [
    ("statement", "AMBIGUOUS_STATEMENT"), ("unit", "AMBIGUOUS_UNIT"),
    ("duration", "AMBIGUOUS_DURATION"), ("date", "AMBIGUOUS_DATE_COLUMNS"),
    ("row", "AMBIGUOUS_ROW"),
])
def test_duplicate_source_groups_refuse_without_picking_nearest(part, code):
    ns = api()
    text, words, rect = fixture()
    blocks = {"statement": (0, 0), "unit": (1, 0), "duration": (2, 0),
              "date": (3, 0), "row": (5, 0)}
    block, line = blocks[part]
    duplicates = [(*word[:5], 90, line, word[7]) for word in words if word[5:7] == (block, line)]
    with pytest.raises(ValueError, match=code):
        ns["derive_cells"](text, words + duplicates, rect)


@pytest.mark.parametrize("mutation,code", [
    ("missing_unit", "AMBIGUOUS_UNIT"), ("wrong_row", "AMBIGUOUS_ROW"),
    ("missing_column", "AMBIGUOUS_DATE_COLUMNS"), ("crossed_durations", "AMBIGUOUS_DURATION"),
    ("value_outside_row", "AMBIGUOUS_ROW_VALUES"), ("value_wrong_column", "AMBIGUOUS_VALUE_COLUMN"),
    ("missing_currency", "AMBIGUOUS_CURRENCY"), ("nonfinite_bbox", "INVALID_GEOMETRY"),
])
def test_wrong_or_missing_geometry_refuses_fixed_reason(mutation, code):
    ns = api()
    text, words, rect = fixture()
    changed = []
    for word in words:
        item = list(word)
        if mutation == "missing_unit" and item[5] == 1:
            continue
        if mutation == "wrong_row" and item[5:7] == [5, 0] and item[4] == "Net":
            item[4] = "Operating"
        if mutation == "missing_column" and item[5:7] == [3, 3]:
            continue
        if mutation == "crossed_durations" and item[5] == 2:
            if item[4] == "Three": item[4] = "Six"
            elif item[4] == "Six": item[4] = "Three"
        if mutation == "value_outside_row" and item[4] == "101":
            item[1] += 15; item[3] += 15
        if mutation == "value_wrong_column" and item[4] == "101":
            item[0] += 50; item[2] += 50
        if mutation == "missing_currency" and item[4] == "$":
            continue
        if mutation == "nonfinite_bbox" and item[4] == "101":
            item[0] = float("nan")
        changed.append(tuple(item))
    with pytest.raises(ValueError, match=code):
        ns["derive_cells"](text, changed, rect)


def test_geometry_text_disagreement_cannot_bind_a_value_from_other_column():
    ns = api()
    text, words, rect = fixture()
    with pytest.raises(ValueError, match="TEXT_GEOMETRY_MISMATCH"):
        ns["derive_cells"](text.replace("$ 101", "$ 202", 1), words, rect)


def test_centered_last_footer_can_be_above_large_blank_page_margin():
    ns = api()
    text, words, rect = fixture()
    changed = [(w[0], 520, w[2], 529, *w[4:]) if w[5] == 6 else w for w in words]
    assert all(cell["printed_page"] == "3" for cell in ns["derive_cells"](text, changed, rect))


def test_duplicate_centered_footer_is_ambiguous():
    ns = api()
    text, words, rect = fixture()
    footer = next(w for w in words if w[5] == 6)
    with pytest.raises(ValueError, match="AMBIGUOUS_PRINTED_PAGE"):
        ns["derive_cells"](text, words + [(*footer[:5], 91, 0, 0)], rect)


def test_sorted_text_footer_indentation_preserves_exact_digit_span():
    ns = api()
    text, words, rect = fixture()
    text = text.removesuffix("3\n") + "          3\n"
    cells = ns["derive_cells"](text, words, rect)
    assert all(cell["bindings"]["printed_page"]["text"] == "3" for cell in cells)
    assert all(text[cell["bindings"]["printed_page"]["start"]:cell["bindings"]["printed_page"]["end"]] == "3" for cell in cells)


def test_wrong_pdf_hash_refuses_before_parser(tmp_path, monkeypatch):
    ns = api()
    wrong = tmp_path / "wrong.pdf"
    wrong.write_bytes(b"synthetic wrong frozen source")
    monkeypatch.setattr(ns["pymupdf"], "open", lambda *args: pytest.fail("wrong source must not be parsed"))
    with pytest.raises(ValueError, match="SOURCE_HASH_MISMATCH"):
        ns["extract_from_pdf"](wrong)


def test_exclusive_output_and_digest_replay_reject_tampering(tmp_path):
    ns = api()
    text, words, rect = fixture()
    cells = ns["derive_cells"](text, words, rect)
    directory = tmp_path / "versioned-cells"
    receipt = ns["write_artifacts"](directory, cells, {"source_sha256": "a" * 64})
    assert receipt["cell_count"] == 4
    assert ns["verify_artifacts"](directory)["cells_sha256"] == receipt["cells_sha256"]
    with pytest.raises(FileExistsError):
        ns["write_artifacts"](directory, cells, {"source_sha256": "a" * 64})
    target = directory / "cells.jsonl"
    original = target.read_bytes()
    target.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="ARTIFACT_DIGEST_MISMATCH"):
        ns["verify_artifacts"](directory)
