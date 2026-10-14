"""Conservative, opt-in extraction of NVIDIA financial-statement table facts.

This is a research candidate generator, not a general PDF table parser or a
factual-answer verifier. Ambiguous geometry is deliberately omitted.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import pymupdf as fitz

from .core import Chunk

_MONTHS = {"jan", "january", "feb", "february", "mar", "march", "jul", "july"}
_AMOUNT = re.compile(r"^\(?\d{1,3}(?:,\d{3})*\)?$")
_YEAR = re.compile(r"^20\d{2}$")
_STATEMENT = re.compile(r"Consolidated Statements of (Income|Cash Flows)", re.IGNORECASE)
_BALANCE_SHEET = re.compile(r"Consolidated Balance Sheets", re.IGNORECASE)
_UNIT = re.compile(r"\(In millions(?:,|\))", re.IGNORECASE)


@dataclass(frozen=True)
class _Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass(frozen=True)
class _Column:
    fiscal_year: int
    right_edge: float
    header_left: float
    date: str
    period_bbox: tuple[float, float, float, float] | None


def _words(page: fitz.Page, bbox: tuple[float, float, float, float]) -> list[_Word]:
    return [_Word(*item[:5]) for item in page.get_text("words", clip=fitz.Rect(bbox))]


def _date_for_year(year_word: _Word, words: list[_Word]) -> str | None:
    months = [
        word for word in words
        if word.text.lower().rstrip(".") in _MONTHS
        and 0 <= year_word.y0 - word.y0 <= 16
        and 0 <= year_word.x0 - word.x0 <= 65
    ]
    if len(months) != 1:
        return None
    month = months[0]
    days = [
        word for word in words
        if re.fullmatch(r"\d{1,2},?", word.text)
        and abs(word.y0 - month.y0) < 1.5
        and month.x1 <= word.x0 <= year_word.x1 + 3
    ]
    if len(days) != 1:
        return None
    month_numbers = {'jan': 1, 'feb': 2, 'mar': 3, 'jul': 7}
    try:
        date(int(year_word.text), month_numbers[month.text.lower().rstrip('.')[:3]],
             int(days[0].text.rstrip(',')))
    except (ValueError, KeyError):
        return None
    return f"{month.text} {days[0].text.rstrip(',')}, {year_word.text}"


def _header_columns(words: list[_Word]) -> tuple[list[_Column], float] | None:
    ended_lines = [
        word for word in words if word.text.lower() == "ended"
        and any(
            other.text.lower() == "year"
            and abs(other.y0 - word.y0) < 1.5
            and other.x1 <= word.x0
            for other in words
        )
    ]
    if len(ended_lines) != 1:
        return None
    ended_y = ended_lines[0].y0
    year_words = sorted(
        (
            word for word in words
            if _YEAR.fullmatch(word.text)
            and ended_y < word.y0 < ended_y + 45
        ),
        key=lambda word: word.x0,
    )
    if not 2 <= len(year_words) <= 4:
        return None
    years = [int(word.text) for word in year_words]
    if len(set(years)) != len(years) or years != sorted(years, reverse=True):
        return None
    columns: list[_Column] = []
    for word in year_words:
        date = _date_for_year(word, words)
        if date is None:
            return None
        header_words = [item for item in words
                        if 0 <= word.y0 - item.y0 <= 16
                        and word.x0 - 65 <= item.x0 <= word.x1
                        and item.text.rstrip(',') in date.replace(',', '').split()]
        bbox = (min(item.x0 for item in header_words), min(item.y0 for item in header_words),
                max(item.x1 for item in header_words), max(item.y1 for item in header_words))
        columns.append(_Column(int(word.text), word.x1, word.x0, date, bbox))
    return columns, max(word.y1 for word in year_words)


def _balance_sheet_columns(words: list[_Word]) -> tuple[list[_Column], float] | None:
    """Map the two dated snapshot columns; never interpret them as cash flows."""
    dated_years = sorted(
        (
            (word, date)
            for word in words
            if _YEAR.fullmatch(word.text)
            if (date := _date_for_year(word, words)) is not None
        ),
        key=lambda item: item[0].y0,
    )
    if not dated_years:
        return None
    first_y = dated_years[0][0].y0
    header_years = sorted(
        (
            (word, date)
            for word, date in dated_years
            if abs(word.y0 - first_y) < 1.5
        ),
        key=lambda item: item[0].x0,
    )
    if len(header_years) != 2:
        return None
    years = [int(word.text) for word, _ in header_years]
    if years != sorted(set(years), reverse=True) or years[0] - years[1] != 1:
        return None
    columns = []
    for word, date in header_years:
        header_words = [item for item in words
                        if abs(item.y0 - word.y0) < 1.5
                        and word.x0 - 65 <= item.x0 <= word.x1
                        and item.text.rstrip(',') in date.replace(',', '').split()]
        bbox = (min(item.x0 for item in header_words), min(item.y0 for item in header_words),
                max(item.x1 for item in header_words), max(item.y1 for item in header_words))
        columns.append(_Column(int(word.text), word.x1, word.x0, date, bbox))
    return columns, max(word.y1 for word, _ in header_years)


def _normalized_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def _candidate_rows(
    words: list[_Word], columns: list[_Column], header_bottom: float
) -> list[tuple[str, dict[int, _Word], tuple[float, float, float, float]]]:
    lines: list[tuple[float, list[_Word]]] = []
    for word in sorted(words, key=lambda item: item.y0):
        if word.y0 <= header_bottom + 1:
            continue
        # Bold labels and regular-weight amounts can have different font
        # ascenders on the same visual row. Keep nearby baselines together,
        # while rejecting larger offsets that might belong to another row.
        if lines and word.y0 - lines[-1][0] <= 2.0:
            lines[-1][1].append(word)
        else:
            lines.append((word.y0, [word]))

    label_limit = min(column.header_left for column in columns) - 20
    rows: list[tuple[str, dict[int, _Word], tuple[float, float, float, float]]] = []
    per_share_section = False
    for _, line in lines:
        line.sort(key=lambda word: word.x0)
        # A currency glyph can sit to the left of the date-header boundary.
        # It belongs to the amount cell, never to the financial metric label.
        label_words = [
            word for word in line
            if word.x1 < label_limit and word.text not in {"$", "€", "£"}
        ]
        label = " ".join(word.text for word in label_words).strip()
        if "net income per share" in label.lower():
            per_share_section = True
        if per_share_section or not label or label.lower() in {"other", "year ended"}:
            continue
        amounts: dict[int, _Word] = {}
        ambiguous = False
        for word in line:
            if word.x0 < label_limit or not _AMOUNT.fullmatch(word.text):
                continue
            ranked = sorted(columns, key=lambda column: abs(word.x1 - column.right_edge))
            if len(ranked) > 1 and abs(word.x1 - ranked[0].right_edge) + 8 >= abs(word.x1 - ranked[1].right_edge):
                ambiguous = True
                break
            closest = ranked[0]
            if abs(word.x1 - closest.right_edge) > 26 or closest.fiscal_year in amounts:
                ambiguous = True
                break
            amounts[closest.fiscal_year] = word
        if not ambiguous and len(amounts) >= 2:
            metric_bbox = (min(w.x0 for w in label_words), min(w.y0 for w in label_words),
                           max(w.x1 for w in label_words), max(w.y1 for w in label_words))
            rows.append((label, amounts, metric_bbox))

    counts = Counter(_normalized_label(label) for label, _, _ in rows)
    return [
        (label, amounts, bbox) for label, amounts, bbox in rows
        if counts[_normalized_label(label)] == 1
    ]


def extract_table_candidate_chunks(
    pdf_path: Path, *, document_id: str, page_numbers: set[int] | None = None
) -> list[Chunk]:
    """Return only uniquely mapped fiscal-year/metric/USD-million candidates.

    The page's original text stays in the separate baseline. An omitted row is
    not an assertion that the PDF lacks the fact; it needs manual review.
    """
    if not document_id.strip():
        raise ValueError("document_id must be non-empty")
    candidates: list[Chunk] = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            if page_numbers is not None and page_number not in page_numbers:
                continue
            page_text = page.get_text("text")
            # SEC PDFs often use nonbreaking spaces in headings and table units.
            # Normalize only the recognition text; coordinates stay untouched.
            recognition_text = re.sub(r"[\u00a0\u2007\u202f]", " ", page_text)
            heading = _STATEMENT.search(recognition_text[:350])
            balance_sheet = _BALANCE_SHEET.search(recognition_text[:350])
            if not (heading or balance_sheet) or not _UNIT.search(recognition_text[:350]):
                continue
            if heading and "Year Ended" not in recognition_text[:500]:
                continue
            if "$" not in page_text:
                continue
            statement = (
                f"Consolidated Statements of {heading.group(1)}"
                if heading else "Consolidated Balance Sheets"
            )
            tables = page.find_tables(strategy="text").tables
            page_candidates: list[tuple[tuple[int, str], Chunk]] = []
            for table_number, table in enumerate(tables, start=1):
                words = _words(page, table.bbox)
                header = (
                    _header_columns(words) if heading
                    else _balance_sheet_columns(words)
                )
                if header is None:
                    continue
                columns, header_bottom = header
                for label, amounts, metric_bbox in _candidate_rows(words, columns, header_bottom):
                    for column in columns:
                        value = amounts.get(column.fiscal_year)
                        if value is None:
                            continue
                        amount_word = value
                        value = amount_word.text
                        money = f"(${value[1:-1]})" if value.startswith("(") else f"${value}"
                        time_label = (
                            f"For fiscal {column.fiscal_year} (year ended {column.date})"
                            if heading else f"As of {column.date}"
                        )
                        text = (
                            f"NVIDIA {statement} in {document_id}. "
                            f"{time_label}, {label} was {money} million USD."
                        )
                        digest = sha256(
                            f"{document_id}|{page_number}|{table_number}|{text}".encode()
                        ).hexdigest()[:16]
                        chunk_id = f"{document_id}:p{page_number}:table:{digest}"
                        key = (column.fiscal_year, _normalized_label(label))
                        page_candidates.append(
                            (key, Chunk(chunk_id, document_id, page_number, text,
                                        (amount_word.x0, amount_word.y0, amount_word.x1, amount_word.y1),
                                        metric_bbox, column.period_bbox))
                        )
            key_counts = Counter(key for key, _ in page_candidates)
            candidates.extend(
                chunk for key, chunk in page_candidates if key_counts[key] == 1
            )
    return candidates


def extract_quarter_income_revenue_chunks(
    pdf_path: Path, *, document_id: str, metrics: tuple[str, ...] = ("Revenue",)
) -> list[Chunk]:
    """Conservatively extract named income rows from a quarterly income table.

    This isolated experiment requires explicit three- and six-month headings,
    four dated columns, and one geometrically unambiguous row per metric.
    It is not a general quarterly-table parser or part of the default index.
    """
    if not document_id.strip():
        raise ValueError("document_id must be non-empty")
    if not metrics or any(metric not in {"Revenue", "Operating income"} for metric in metrics):
        raise ValueError("unsupported quarterly income metric")
    candidates: list[Chunk] = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")
            if (
                "Condensed Consolidated Statements of Income" not in text[:350]
                or "(In millions" not in text[:350]
                or "Three Months Ended" not in text[:600]
                or "Six Months Ended" not in text[:600]
            ):
                continue
            words = [_Word(*item[:5]) for item in page.get_text("words")]
            year_words = sorted(
                (word for word in words if _YEAR.fullmatch(word.text) and word.y0 < 220),
                key=lambda word: word.x0,
            )
            if len(year_words) != 4 or max(w.y0 for w in year_words) - min(w.y0 for w in year_words) > 1.5:
                continue
            dates = [_date_for_year(word, words) for word in year_words]
            if any(date is None for date in dates):
                continue
            for metric in dict.fromkeys(metrics):
                row_labels = []
                for word in words:
                    if word.x0 >= 120 or word.y0 <= year_words[0].y1:
                        continue
                    line = sorted(
                        (item for item in words if abs(item.y0 - word.y0) < 1.5 and item.x0 < 200),
                        key=lambda item: item.x0,
                    )
                    if line and line[0] == word and " ".join(item.text for item in line) == metric:
                        row_labels.append(word)
                if len(row_labels) != 1:
                    continue
                row_y = row_labels[0].y0
                amounts = sorted(
                    (
                        word for word in words
                        if _AMOUNT.fullmatch(word.text)
                        and abs(word.y0 - row_y) < 1.5 and word.x0 > 200
                    ),
                    key=lambda word: word.x0,
                )
                if len(amounts) != 4 or any(
                    not 0 < amount.x1 - year.x1 < 20
                    for amount, year in zip(amounts, year_words)
                ):
                    continue
                for index, (date, amount) in enumerate(zip(dates, amounts)):
                    period = "Three Months Ended" if index < 2 else "Six Months Ended"
                    candidate_text = (
                        f"NVIDIA Condensed Consolidated Statements of Income in {document_id}. "
                        f"{period} {date}, {metric} was ${amount.text} million USD."
                    )
                    digest = sha256(f"{document_id}|{page_number}|{candidate_text}".encode()).hexdigest()[:16]
                    candidates.append(Chunk(
                        f"{document_id}:p{page_number}:quarter-table:{digest}",
                        document_id, page_number, candidate_text,
                    ))
    return candidates


def select_fy2025_q2_revenue_candidate(query: str, candidates: Sequence[Chunk]) -> Chunk | None:
    """Select one verified Revenue cell for this one filing, or refuse.

    This opt-in experiment does not interpret arbitrary quarters, compute values,
    or modify the default RAG retriever. It requires the complete four-cell
    candidate set from the FY2025 Q2 income statement's PDF page 3.
    """
    normalized = re.sub(r"\s+", " ", query.lower()).strip()
    if "revenue" not in normalized or re.search(r"\b(net income|profit|margin)\b", normalized):
        return None
    if re.search(r"\bfy20\d{2}\b", normalized) and "fy2025" not in normalized:
        return None

    explicit_period = re.search(r"\b(three|six) months? ended\s+jul(?:y)?\s+(28|30),?\s+(2023|2024)\b", normalized)
    fiscal_alias = bool(re.search(r"\bfy2025\s+q2\b", normalized))
    if explicit_period:
        period = "Three" if explicit_period.group(1) == "three" else "Six"
        date = f"Jul {explicit_period.group(2)}, {explicit_period.group(3)}"
        if fiscal_alias and (period != "Three" or date != "Jul 28, 2024"):
            return None
        target = (period, date)
    elif fiscal_alias and not re.search(r"\b(three|six) months?\b|\bjul(?:y)?\b", normalized):
        target = ("Three", "Jul 28, 2024")
    else:
        return None

    expected = {
        ("Three", "Jul 28, 2024"),
        ("Three", "Jul 30, 2023"),
        ("Six", "Jul 28, 2024"),
        ("Six", "Jul 30, 2023"),
    }
    if len(candidates) != 4:
        return None
    by_period: dict[tuple[str, str], Chunk] = {}
    pattern = re.compile(
        r"\. (Three|Six) Months Ended (Jul (?:28, 2024|30, 2023)), "
        r"Revenue was \$\d{1,3}(?:,\d{3})* million USD\.$"
    )
    for candidate in candidates:
        if candidate.document_id != "nvidia-fy2025-q2" or candidate.page_number != 3:
            return None
        match = pattern.search(candidate.text)
        if match is None:
            return None
        key = (match.group(1), match.group(2))
        if key in by_period:
            return None
        by_period[key] = candidate
    if set(by_period) != expected:
        return None
    return by_period.get(target)
