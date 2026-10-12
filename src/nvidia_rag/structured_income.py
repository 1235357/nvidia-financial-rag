"""Opt-in source-bound income tables, separate from dense retrieval.

Build from pinned PDFs, never evaluation cases. Selection returns source
candidates, not generated answers or independently adjudicated financial facts.
Dollar-symbol normalization is deliberately unverified. Unsupported geometry,
units and requests never acquire a guessed value or a dense fallback.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from decimal import Decimal
from hashlib import sha256
import math
import json
from pathlib import Path
import re

import pymupdf

TITLE = 'Condensed Consolidated Statements of Income'
UNIT = '(In millions, except per share data)'
EPS = 'Net income per share:'
SHARES = 'Weighted average shares used in per share computation:'
FOOTER = 'See accompanying Notes to Condensed Consolidated Financial Statements.'
PROFILE = 'quarter-income-source-v1;PyMuPDF;text-sort-true;no-normalization;code-points'
ANNUAL_PROFILE = 'annual-income-source-v1;PyMuPDF;text-sort-true;no-normalization;code-points'
ANNUAL_TITLE = 'Consolidated Statements of Income'
ANNUAL_FOOTER = 'See accompanying Notes to the Consolidated Financial Statements.'
_BUILT_INDEX = object()
MONTHS = {name: number for number, name in enumerate(
    ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'), 1)}
QUERY_MONTHS = {**MONTHS, **{name: number for number, name in enumerate(
    ('january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
     'september', 'october', 'november', 'december'), 1)}}
DATE = re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{1,2}), (\d{4})\Z')
NUMBER = re.compile(r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\Z')


def require(condition, code):
    if not condition:
        raise ValueError(code)


@dataclass(frozen=True, slots=True)
class SourceSpan:
    start: int
    end: int
    literal: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ReportInput:
    document_id: str
    pdf_path: Path
    pdf_sha256: str
    fiscal_year: int
    fiscal_quarter: int | None


@dataclass(frozen=True, slots=True)
class IncomeCell:
    cell_id: str
    document_id: str
    pdf_sha256: str
    physical_page: int
    pdf_page_label: str | None
    fiscal_year: int
    fiscal_quarter: int | None
    parser_profile: str
    pymupdf_version: str
    page_text_sha256: str
    row_path: str
    row_literal: str
    column_number: int
    column_role: str
    period_end: str
    duration_months: int | None
    amount: str
    value_literal: str
    unit_kind: str
    scale: str
    currency_literal: str | None
    currency_normalization: str
    bindings: tuple[tuple[str, SourceSpan], ...]

    @property
    def period_kind(self):
        # Keep quarterly serialized fields and identities unchanged. A fiscal
        # year is source-qualified by its profile and Year Ended binding.
        return 'fiscal_year' if self.parser_profile == ANNUAL_PROFILE else 'months'


@dataclass(frozen=True, slots=True)
class Omission:
    document_id: str
    physical_page: int | None
    row_literal: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class IncomeIndex:
    reports: tuple[tuple[str, str, int, int | None], ...]
    cells: tuple[IncomeCell, ...]
    omissions: tuple[Omission, ...]
    # Prevent accidental selection from a hand-constructed/unreplayed index.
    # This is a factory contract, not a sandbox against arbitrary Python code.
    _construction_token: object = field(default=None, init=False, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Mention:
    field: str
    value: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ParsedRequest:
    query: str
    query_sha256: str
    operator_document_ids: tuple[str, ...]
    mentions: tuple[Mention, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Selection:
    disposition: str
    reason: str
    cells: tuple[IncomeCell, ...] = ()
    financial_correctness: str = 'unestablished'
    score_semantics: str = 'exact-source-field-selection;no-cosine-score'


def _bbox(words):
    return (min(w[0] for w in words), min(w[1] for w in words),
            max(w[2] for w in words), max(w[3] for w in words))


def _center(box):
    return (box[0] + box[2]) / 2


def _groups(raw_words):
    require(0 < len(raw_words) <= 20_000, 'PAGE_WORD_BUDGET')
    grouped = defaultdict(list)
    for w in raw_words:
        require(len(w) == 8 and type(w[4]) is str and 0 < len(w[4]) <= 4096
                and all(type(n) in (int, float) and math.isfinite(n) for n in w[:4])
                and w[0] < w[2] and w[1] < w[3], 'INVALID_WORD_GEOMETRY')
        grouped[w[5], w[6]].append(tuple(w))
    return sorted(((' '.join(w[4] for w in sorted(group, key=lambda w: w[0])),
                    _bbox(group), tuple(sorted(group, key=lambda w: w[0])))
                   for group in grouped.values()), key=lambda g: (g[1][1], g[1][0]))


def _unique(groups, literal):
    found = [group for group in groups if group[0] == literal]
    require(len(found) == 1, 'AMBIGUOUS_HEADER')
    return found[0]


def _span(text, groups, selected_words):
    tokens = tuple(w[4] for w in selected_words)
    literal = ' '.join(tokens)
    boxes = []
    for _, _, words in groups:
        for i in range(len(words) - len(tokens) + 1):
            subset = words[i:i + len(tokens)]
            if tuple(w[4] for w in subset) == tokens:
                boxes.append(_bbox(subset))
    boxes.sort(key=lambda box: (round(box[1], 3), box[0]))
    pattern = r'(?<![\w.,])' + r'\s+'.join(re.escape(t) for t in tokens) + r'(?![\w.,])'
    hits = list(re.finditer(pattern, text))
    box = _bbox(selected_words)
    candidates = [i for i, item in enumerate(boxes)
                  if all(abs(a - b) <= .001 for a, b in zip(item, box))]
    require(len(hits) == len(boxes) and len(candidates) == 1, 'AMBIGUOUS_TEXT_GEOMETRY')
    hit = hits[candidates[0]]
    return SourceSpan(hit.start(), hit.end(), hit.group(), box)


def _amount(literal, unit_kind):
    require(type(literal) is str and len(literal) <= 48, 'UNSUPPORTED_VALUE')
    negative = literal.startswith('(') and literal.endswith(')')
    body = literal[1:-1] if negative else literal
    require(NUMBER.fullmatch(body) is not None, 'UNSUPPORTED_VALUE')
    require(unit_kind == 'per_share' or '.' not in body, 'UNSUPPORTED_UNIT_VALUE')
    number = Decimal(body.replace(',', ''))
    if negative:
        number = number.copy_negate()  # Exact; multiplication can round at context precision.
    return format(number, 'f')


def _parse_table(page, spec, physical_page):
    text = page.get_text('text', sort=True)
    require(0 < len(text) <= 200_000, 'PAGE_TEXT_BUDGET')
    groups = _groups(page.get_text('words', sort=True))
    # Exact standalone groups exclude prose/notes that merely mention the title.
    title = _unique(groups, TITLE)
    unit = _unique(groups, UNIT)
    footer = _unique(groups, FOOTER)
    revenue = _unique(groups, 'Revenue')
    require(title[1][3] <= unit[1][1] + .1 < revenue[1][1]
            and revenue[1][3] < footer[1][1], 'INVALID_STATEMENT_LAYOUT')
    periods = [g for g in groups if g[0] in ('Three Months Ended', 'Six Months Ended')
               and unit[1][3] < g[1][1] < revenue[1][1]]
    periods.sort(key=lambda g: _center(g[1]))
    require([g[0] for g in periods] in (['Three Months Ended'],
            ['Three Months Ended', 'Six Months Ended']), 'UNSUPPORTED_DURATION_LAYOUT')
    dates = sorted((g for g in groups if DATE.fullmatch(g[0])
                    and max(p[1][3] for p in periods) < g[1][1] < revenue[1][1]),
                   key=lambda g: _center(g[1]))
    count = 2 * len(periods)
    require(len(dates) == count and max(g[1][1] for g in dates)
            - min(g[1][1] for g in dates) <= 1.5, 'AMBIGUOUS_DATE_COLUMNS')
    currencies = sorted((w for g in groups for w in g[2] if w[4] == '$'
                         and abs(w[1] - revenue[1][1]) <= 1.5), key=lambda w: w[0])
    require(len(currencies) == count and all(w[0] > revenue[1][2] for w in currencies),
            'AMBIGUOUS_CURRENCY_COLUMNS')
    bands = [(w[0] - .5, currencies[i + 1][0] - .5 if i + 1 < count else page.rect.x1)
             for i, w in enumerate(currencies)]
    header_values = []
    for i, band in enumerate(bands):
        vals = [w for g in groups for w in g[2] if band[0] <= w[0] and w[2] <= band[1]
                and abs(w[1] - revenue[1][1]) <= 1.5 and NUMBER.fullmatch(w[4])]
        require(len(vals) == 1, 'AMBIGUOUS_REVENUE_COLUMNS')
        header_values.append(vals[0])
        require(band[0] <= dates[i][1][0] < dates[i][1][2] <= band[1], 'DATE_COLUMN_MISMATCH')
    columns = []
    for group_number, period in enumerate(periods):
        pair = dates[group_number * 2:group_number * 2 + 2]
        require(abs(sum(_center(g[1]) for g in pair) / 2 - _center(period[1])) <= 10,
                'DURATION_COLUMN_MISMATCH')
        parsed_dates = []
        for g in pair:
            match = DATE.fullmatch(g[0])
            parsed_dates.append(date(int(match[3]), MONTHS[match[1].lower()], int(match[2])).isoformat())
        require(parsed_dates[0] > parsed_dates[1], 'COMPARISON_DATE_ORDER')
        for role, (g, parsed) in enumerate(zip(pair, parsed_dates)):
            columns.append((parsed, 3 if period[0].startswith('Three') else 6,
                            'current' if role == 0 else 'prior_comparison',
                            _span(text, groups, g[2]), _span(text, groups, period[2])))
    require(len({value[0] for value in columns[::2]}) == 1, 'CURRENT_DATE_CONFLICT')
    return _parse_statement_rows(page, spec, physical_page, text, groups, title,
                                 unit, footer, revenue, bands, header_values, columns, currencies)


def _parse_statement_rows(page, spec, physical_page, text, groups, title, unit,
                          footer, revenue, bands, header_values, columns, currencies,
                          *, profile=PROFILE, extra_date_bindings=()):
    count = len(columns)
    eps = _unique(groups, EPS)
    shares = _unique(groups, SHARES)
    require(revenue[1][1] < eps[1][1] < shares[1][1] < footer[1][1], 'UNIT_SECTION_ORDER')
    shared = (('statement', _span(text, groups, title[2])), ('unit', _span(text, groups, unit[2])))
    cells, omissions = [], []
    labels = [g for g in groups if revenue[1][1] <= g[1][1] < footer[1][1]
              and g[1][2] < bands[0][0] and g[0] not in (EPS, SHARES)]
    paths = Counter(('monetary' if g[1][1] < eps[1][1] else (
        'per_share' if g[1][1] < shares[1][1] else 'count')) + '/' +
        re.sub(r'[^a-z0-9]+', '_', g[0].lower()).strip('_') for g in labels)
    for label in labels:
        section = 'monetary' if label[1][1] < eps[1][1] else (
            'per_share' if label[1][1] < shares[1][1] else 'count')
        row_path = section + '/' + re.sub(r'[^a-z0-9]+', '_', label[0].lower()).strip('_')
        row_words = [w for g in groups for w in g[2]
                     if abs(w[1] - label[1][1]) <= 1.5 and w[0] >= bands[0][0]]
        # Section headings have no values. They are not missing financial rows.
        if not row_words:
            continue
        try:
            require(paths[row_path] == 1, 'DUPLICATE_ROW_PATH')
            row_binding = _span(text, groups, label[2])
            amounts, signs = [], []
            for i, band in enumerate(bands):
                column_words = [w for w in row_words if band[0] <= w[0] and w[2] <= band[1]]
                values = [w for w in column_words if w[4] != '$']
                require(len(values) == 1 and abs(values[0][2] - header_values[i][2]) <= 3,
                        'AMBIGUOUS_ROW_COLUMN')
                try:
                    value = _amount(values[0][4], section)
                except ValueError as error:
                    if profile != ANNUAL_PROFILE:
                        raise
                    # A source placeholder with unambiguous column geometry
                    # does not invalidate a neighboring numeric annual cell.
                    omissions.append(Omission(spec.document_id, physical_page, label[0],
                                              str(error) + '_COLUMN_' + str(i + 1)))
                    amounts.append(None)
                    signs.append(None)
                    continue
                amounts.append((values[0], value))
                glyphs = [w for w in column_words if w[4] == '$']
                require(len(glyphs) <= 1 and (section != 'count' or not glyphs),
                        'AMBIGUOUS_ROW_CURRENCY')
                signs.append(glyphs[0] if glyphs else currencies[i])
            if section != 'monetary':
                section_binding = _span(text, groups, (eps if section == 'per_share' else shares)[2])
            row_cells = []
            for i, (amount, column) in enumerate(zip(amounts, columns)):
                if amount is None:
                    continue
                value_word, value = amount
                period_end, duration, role, date_span, duration_span = column
                bindings = (*shared, ('row', row_binding), ('date', date_span),
                            ('duration', duration_span), ('value', _span(text, groups, (value_word,))))
                if extra_date_bindings:
                    bindings += extra_date_bindings[i]
                if section != 'monetary':
                    bindings += (('section', section_binding),)
                if section != 'count':
                    bindings += (('currency', _span(text, groups, (signs[i],))),)
                cell = IncomeCell(
                    '', spec.document_id, spec.pdf_sha256,
                    physical_page, page.get_label() or None, spec.fiscal_year, spec.fiscal_quarter,
                    profile, pymupdf.version[0], sha256(text.encode()).hexdigest(), row_path, label[0],
                    i + 1, role, period_end, duration, value, value_word[4], section,
                    '1' if section == 'per_share' else '1000000',
                    None if section == 'count' else '$',
                    'not_applicable' if section == 'count' else 'dollar-symbol;ISO-currency-unverified', bindings)
                identity = asdict(cell)
                identity.pop('cell_id')
                encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()
                row_cells.append(replace(cell, cell_id=sha256(encoded).hexdigest()))
            cells.extend(row_cells)
        except ValueError as error:
            omissions.append(Omission(spec.document_id, physical_page, label[0], str(error)))
    require(bool(cells), 'NO_RELIABLE_CELLS')
    return cells, omissions


def _parse_annual_table(page, spec, physical_page):
    """Three Year Ended columns; no assumed calendar-month duration."""
    text = page.get_text('text', sort=True)
    require(0 < len(text) <= 200_000, 'PAGE_TEXT_BUDGET')
    groups = _groups(page.get_text('words', sort=True))
    def unique_folded(literal):
        found = [g for g in groups if g[0].casefold() == literal.casefold()]
        require(len(found) == 1, 'AMBIGUOUS_HEADER')
        return found[0]
    title = unique_folded(ANNUAL_TITLE)
    unit = unique_folded(UNIT)
    footer = unique_folded(ANNUAL_FOOTER)
    revenue = _unique(groups, 'Revenue')
    require(title[1][3] <= unit[1][1] + 1 < revenue[1][1]
            and revenue[1][3] < footer[1][1], 'INVALID_STATEMENT_LAYOUT')
    period = _unique(groups, 'Year Ended')
    require(unit[1][3] < period[1][1] and period[1][3] < revenue[1][1],
            'UNSUPPORTED_DURATION_LAYOUT')
    currencies = sorted((w for g in groups for w in g[2] if w[4] == '$'
                         and abs(w[1] - revenue[1][1]) <= 1.5), key=lambda w: w[0])
    require(len(currencies) == 3 and all(w[0] > revenue[1][2] for w in currencies),
            'AMBIGUOUS_CURRENCY_COLUMNS')
    bands = [(w[0] - .5, currencies[i+1][0] - .5 if i < 2 else page.rect.x1)
             for i,w in enumerate(currencies)]
    header_values = []
    for band in bands:
        values = [w for g in groups for w in g[2] if band[0] <= w[0] and w[2] <= band[1]
                  and abs(w[1] - revenue[1][1]) <= 1.5 and NUMBER.fullmatch(w[4])]
        require(len(values) == 1, 'AMBIGUOUS_REVENUE_COLUMNS')
        header_values.append(values[0])
    months = '|'.join(sorted(QUERY_MONTHS, key=len, reverse=True))
    full_pattern = re.compile('(' + months + r') (\d{1,2}), (\d{4})\Z', re.I)
    partial_pattern = re.compile('(' + months + r') (\d{1,2}),\Z', re.I)
    headers = [g for g in groups if period[1][3] < g[1][1] < revenue[1][1]]
    full = sorted((g for g in headers if full_pattern.fullmatch(g[0])), key=lambda g: _center(g[1]))
    parts = sorted((g for g in headers if partial_pattern.fullmatch(g[0])), key=lambda g: _center(g[1]))
    years = sorted((g for g in headers if re.fullmatch(r'\d{4}', g[0])), key=lambda g: _center(g[1]))
    require((len(full) == 3 and not parts and not years)
            or (not full and len(parts) == len(years) == 3), 'AMBIGUOUS_DATE_COLUMNS')
    first_line = full or parts
    require(max(g[1][1] for g in first_line)-min(g[1][1] for g in first_line) <= 1.5,
            'AMBIGUOUS_DATE_COLUMNS')
    if years:
        require(max(g[1][1] for g in years)-min(g[1][1] for g in years) <= 1.5,
                'AMBIGUOUS_DATE_COLUMNS')
    require(abs(sum(_center(g[1]) for g in first_line)/3 - _center(period[1])) <= 10,
            'DURATION_COLUMN_MISMATCH')
    columns, extra_bindings, parsed_dates = [], [], []
    period_span = _span(text, groups, period[2])
    for i,(band,g) in enumerate(zip(bands,first_line)):
        require(band[0] <= g[1][0] < g[1][2] <= band[1], 'DATE_COLUMN_MISMATCH')
        match = (full_pattern if full else partial_pattern).fullmatch(g[0])
        extras = ()
        if full:
            year = int(match[3])
        else:
            year_group = years[i]
            require(band[0] <= year_group[1][0] < year_group[1][2] <= band[1]
                    and g[1][1] < year_group[1][1] < revenue[1][1]
                    and abs(_center(g[1]) - _center(year_group[1])) <= 12,
                    'DATE_COLUMN_MISMATCH')
            year = int(year_group[0])
            extras = (('date_year', _span(text,groups,year_group[2])),)
        normalized = date(year, QUERY_MONTHS[match[1].lower()], int(match[2])).isoformat()
        parsed_dates.append(normalized)
        columns.append((normalized,None,'current' if i == 0 else 'prior_comparison',
                        _span(text,groups,g[2]),period_span))
        extra_bindings.append(extras)
    require(parsed_dates[0] > parsed_dates[1] > parsed_dates[2], 'COMPARISON_DATE_ORDER')
    return _parse_statement_rows(page,spec,physical_page,text,groups,title,unit,footer,
        revenue,bands,header_values,columns,currencies,profile=ANNUAL_PROFILE,
        extra_date_bindings=tuple(extra_bindings))


def build_income_index(reports: tuple[ReportInput, ...]) -> IncomeIndex:
    """Read every pinned report page; case labels and expected values are absent.

    Fiscal identities are explicit operator metadata, not inferred from a date.
    PDF bytes, literal table dates and extraction identity are verified here.
    The caller must separately review the report's fiscal metadata declaration.
    """
    require(type(reports) is tuple and 0 < len(reports) <= 128, 'REPORT_BUDGET')
    cells, omissions, records = [], [], []
    for report in reports:
        require(type(report) is ReportInput and type(report.document_id) is str
                and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', report.document_id)
                and type(report.pdf_sha256) is str and re.fullmatch(r'[0-9a-f]{64}', report.pdf_sha256)
                and type(report.fiscal_year) is int and 1900 <= report.fiscal_year <= 9999
                and (report.fiscal_quarter is None or
                     (type(report.fiscal_quarter) is int and report.fiscal_quarter in (1, 2, 3, 4))),
                'INVALID_REPORT_DECLARATION')
        require(report.document_id not in {r[0] for r in records}, 'DUPLICATE_REPORT')
        path = report.pdf_path
        require(isinstance(path, Path) and path.is_absolute() and path.is_file()
                and not any(p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction())
                            for p in (path, *path.parents)) and path.stat().st_size <= 256_000_000,
                'INVALID_SOURCE_PATH')
        with path.open('rb') as stream:
            body = stream.read(256_000_001)
        require(len(body) <= 256_000_000 and sha256(body).hexdigest() == report.pdf_sha256,
                'SOURCE_IDENTITY_MISMATCH')
        records.append((report.document_id, report.pdf_sha256, report.fiscal_year, report.fiscal_quarter))
        with pymupdf.open(stream=body, filetype='pdf') as pdf:
            require(not pdf.needs_pass and 0 < len(pdf) <= 2048, 'UNSUPPORTED_PDF')
            matched_statement = False
            for number, page in enumerate(pdf, 1):
                # Check exact title groups before attempting the table parser.
                words = page.get_text('words', sort=True)
                annual = report.fiscal_quarter is None
                wanted = ANNUAL_TITLE if annual else TITLE
                if not words or not any((g[0].casefold() == wanted.casefold() if annual
                                         else g[0] == wanted) for g in _groups(words)):
                    continue
                matched_statement = True
                try:
                    parser = _parse_annual_table if annual else _parse_table
                    new_cells, new_omissions = parser(page, report, number)
                    cells.extend(new_cells)
                    omissions.extend(new_omissions)
                except (ValueError, OverflowError) as error:
                    code = str(error)
                    omissions.append(Omission(report.document_id, number, None,
                        code if re.fullmatch('[A-Z_]{1,80}', code) else 'UNSUPPORTED_STATEMENT_LAYOUT'))
            if not matched_statement:
                omissions.append(Omission(report.document_id,None,None,
                    'ANNUAL_STATEMENT_NOT_FOUND' if report.fiscal_quarter is None
                    else 'QUARTER_STATEMENT_NOT_FOUND'))
        require(sha256(path.read_bytes()).hexdigest() == report.pdf_sha256, 'SOURCE_CHANGED_DURING_BUILD')
    keys = [(c.document_id, c.row_path, c.duration_months, c.period_end) for c in cells]
    require(len(keys) == len(set(keys)), 'AMBIGUOUS_STATEMENT_IDENTITY')
    # Recheck all sources at the end, including earlier reports while later
    # ones were being scanned. Source bytes were parsed from immutable memory.
    for report in reports:
        require(not report.pdf_path.is_symlink() and
                sha256(report.pdf_path.read_bytes()).hexdigest() == report.pdf_sha256,
                'SOURCE_CHANGED_DURING_BUILD')
    result = IncomeIndex(tuple(records), tuple(cells), tuple(omissions))
    object.__setattr__(result, '_construction_token', _BUILT_INDEX)
    return result


def parse_statement_request(query: str, *, operator_document_ids: tuple[str, ...],
                            index: IncomeIndex) -> ParsedRequest:
    """Bounded literal parser with every recognized mention retained.

    It accepts explicit fiscal quarter, dated duration, and table row phrases.
    It does not claim general NLP; unknown or conflicting wording is withheld.
    No evaluation request annotation enters this interface.
    """
    require(type(query) is str and 0 < len(query) <= 4000 and query.strip(), 'INVALID_QUERY')
    require(type(index) is IncomeIndex and index._construction_token is _BUILT_INDEX
            and type(operator_document_ids) is tuple
            and 0 < len(operator_document_ids) <= 128
            and len(operator_document_ids) == len(set(operator_document_ids))
            and all(type(v) is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', v)
                    for v in operator_document_ids), 'INVALID_OPERATOR_SCOPE')
    mentions, issues = [], []
    def collect(field, pattern, convert):
        for match in re.finditer(pattern, query, re.I):
            try:
                value = str(convert(match))
                mentions.append(Mention(field, value, match.start(), match.end()))
            except (ValueError, KeyError):
                issues.append('INVALID_' + field.upper())
    collect('fiscal_year', r'\b(?:fiscal(?:\s+year)?\s*|FY\s*)(\d{4})\b', lambda m: int(m[1]))
    quarters = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4}
    collect('fiscal_quarter', r'\b(first|second|third|fourth)[-\s]+quarter\b|\bQ([1-4])\b',
            lambda m: quarters[m[1].lower()] if m[1] else int(m[2]))
    durations = {'one': 1, 'two': 2, 'three': 3, 'six': 6, 'nine': 9, 'twelve': 12}
    collect('duration_months', r'(?<![\w.])(one|two|three|six|nine|twelve|\d{1,2})[-\s]+months?\b',
            lambda m: durations[m[1].lower()] if m[1].lower() in durations else int(m[1]))
    collect('period_kind', r'\bannual\b|\byear\s+ended\b', lambda m: 'fiscal_year')
    collect('period_end', r'\bended\s+([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b',
            lambda m: date(int(m[3]), QUERY_MONTHS[m[1].lower()], int(m[2])).isoformat())
    collect('currency', r'\b(?:US(?:\s+|\.)?dollars?|USD)\b', lambda m: 'USD')
    collect('currency', r'\b(?:euros?|EUR)\b', lambda m: 'EUR')
    collect('currency', r'(?<!\w)\$|\bdollars?\b', lambda m: '$')
    collect('scale', r'\b(?:in\s+)?millions?\b', lambda m: '1000000')
    collect('scale', r'\b(?:in\s+)?billions?\b', lambda m: '1000000000')
    collect('scale', r'\b(?:in\s+)?thousands?\b', lambda m: '1000')
    labels = sorted({(cell.row_literal, cell.row_path) for cell in index.cells},
                    key=lambda pair: -len(pair[0]))
    matched = []
    for literal, row_path in labels:
        if row_path.startswith(('per_share/', 'count/')):
            continue  # Basic/Diluted require explicit section-qualified UI input in a later adapter.
        for match in re.finditer(r'(?<!\w)' + re.escape(literal) + r'(?!\w)', query, re.I):
            if any(match.start() >= a and match.end() <= b for a, b in matched):
                continue
            mentions.append(Mention('row_path', row_path, match.start(), match.end()))
            matched.append((match.start(), match.end()))
    if sum(m.field == 'row_path' for m in mentions) > 1:
        issues.append('UNSUPPORTED_MULTIPLE_METRICS')
    if re.search(r'\bper[-\s]+share\b|\bweighted\s+average\s+shares\b|\bmargin\b|\bpercent\b', query, re.I):
        issues.append('UNSUPPORTED_QUERY_UNIT')
    if re.search(r'\b(?:excluding|except|not|minus|difference|growth|ratio|compare|sum|average)\b', query, re.I):
        issues.append('UNSUPPORTED_QUERY_OPERATION')
    # Reject remaining material wording rather than silently discard a new
    # currency, scale, operation or negation. This is a narrow literal grammar.
    remainder = list(query)
    for mention in mentions:
        remainder[mention.start:mention.end] = ' ' * (mention.end - mention.start)
    remainder = ''.join(remainder)
    remainder = re.sub(r"\bNVIDIA(?:['’]s)?\b|\bForm\s+10-Q\b", ' ', remainder, flags=re.I)
    boilerplate = {'according', 'to', 'report', 'what', 'was', 'for', 'the', 'period',
                   'in', 'of', 'give', 'using', 'column', 'amount', 'reported', 'show', 'me'}
    if any(word.lower() not in boilerplate for word in re.findall(r'[A-Za-z0-9]+', remainder)):
        issues.append('UNRECOGNIZED_QUERY_CONSTRAINT')
    if any(not (char.isascii() and (char.isalnum() or char.isspace()
                                   or char in ',.?:;!\'"()')) for char in remainder):
        issues.append('UNRECOGNIZED_QUERY_SYMBOL')
    return ParsedRequest(query, sha256(query.encode()).hexdigest(), operator_document_ids,
                         tuple(sorted(mentions, key=lambda m: (m.start, m.end, m.field))), tuple(sorted(set(issues))))


def select_statement_candidates(index: IncomeIndex, request: ParsedRequest) -> Selection:
    require(type(index) is IncomeIndex and index._construction_token is _BUILT_INDEX
            and type(request) is ParsedRequest, 'INVALID_SELECTION_INPUT')
    require(request == parse_statement_request(request.query,
            operator_document_ids=request.operator_document_ids, index=index),
            'REQUEST_PARSE_IDENTITY_MISMATCH')
    fields = defaultdict(set)
    for mention in request.mentions:
        fields[mention.field].add(mention.value)
    # '$' is a printed symbol, not a second ISO currency or a USD proof.
    if '$' in fields['currency'] and len(fields['currency']) > 1:
        fields['currency'].remove('$')
    if any(len(values) > 1 for values in fields.values()):
        return Selection('clarify', 'REQUEST_CONFLICT')
    if fields['period_kind'] and fields['duration_months']:
        return Selection('clarify', 'REQUEST_CONFLICT')
    if request.issues:
        return Selection('unsupported', request.issues[0])
    reports = {row[0]: row for row in index.reports}
    scope = request.operator_document_ids
    if any(doc not in reports for doc in scope):
        return Selection('refuse', 'OPERATOR_REPORT_UNAVAILABLE')
    year = next(iter(fields['fiscal_year']), None)
    quarter = next(iter(fields['fiscal_quarter']), None)
    matching_reports = [doc for doc, _, fy, fq in index.reports
                        if (year is None or str(fy) == year) and (quarter is None or str(fq) == quarter)]
    if not matching_reports:
        return Selection('refuse', 'REPORT_UNAVAILABLE')
    chosen_reports = set(matching_reports) & set(scope)
    if not chosen_reports:
        return Selection('clarify', 'REPORT_SCOPE_CONFLICT')
    for field, reason in (('row_path', 'METRIC_REQUIRED'), ('period_end', 'DATE_REQUIRED'),
                          ('duration_months', 'DURATION_REQUIRED')):
        if not fields[field] and not (field == 'duration_months' and fields['period_kind']):
            return Selection('clarify', reason)
    value = lambda field: next(iter(fields[field]))
    candidates = tuple(c for c in index.cells if c.document_id in chosen_reports
                       and c.row_path == value('row_path') and c.period_end == value('period_end')
                       and ((c.period_kind == 'fiscal_year' and c.duration_months is None)
                            if fields['period_kind'] else str(c.duration_months) == value('duration_months')))
    if not candidates:
        return Selection('refuse', 'NO_INDEXED_CELL_MATCH')
    if len(candidates) != 1:
        return Selection('clarify', 'MULTIPLE_SOURCE_MATCHES')
    if fields['currency'] and fields['currency'] != {'$'}:
        return Selection('unsupported', 'ISO_CURRENCY_UNVERIFIED_NO_FX')
    if fields['scale'] and fields['scale'] != {'1000000'}:
        return Selection('unsupported', 'REQUESTED_SCALE_CONVERSION_UNSUPPORTED')
    return Selection('source_candidate', 'EXACT_LITERAL_FIELDS_MATCH', candidates)
