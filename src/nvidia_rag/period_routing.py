"""Literal report/column scope over an immutable, source-bound report catalog.

Report declarations and retained PDF cells supply metadata. Query restrictions
select that metadata before dense ranking; neither a route nor a score verifies
a financial field. No filenames, evaluation annotations or model outputs enter
catalog construction. Multi-report field requests currently require clarification.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
import re

from .core import Chunk, ScoredChunk
from .ingest import load_chunks
from .retrieval import DenseInspection, DenseRetriever
from .statement_startup import StatementSnapshot, _model_binding
from .structured_income import ANNUAL_PROFILE, PROFILE, IncomeCell, validate_income_index

_CATALOG_TOKEN = object()
_ID = re.compile(r'[A-Za-z0-9_.-]{1,128}\Z')
_DIGEST = re.compile(r'[0-9a-f]{64}\Z')
_MONTHS = {name: number for number, name in enumerate(
    ('january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
     'september', 'october', 'november', 'december'), 1)}
_MONTHS.update({name[:3]: number for name, number in tuple(_MONTHS.items())})
_ORDINALS = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4}
_DURATIONS = {name: number for number, name in enumerate(
    ('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'), 1)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ColumnPeriod:
    period_end: str
    duration_months: int | None  # None retains native annual Year Ended semantics.


@dataclass(frozen=True, slots=True)
class ReportScope:
    document_id: str
    pdf_sha256: str
    fiscal_year: int
    fiscal_quarter: int | None
    columns: frozenset[ColumnPeriod]


@dataclass(frozen=True, slots=True)
class ReportCatalog:
    reports: tuple[ReportScope, ...]
    chunks_sha256: str
    _chunks: tuple[Chunk, ...] = field(repr=False)
    _construction_token: object = field(default=None, init=False, repr=False, compare=False)


def validate_report_catalog(catalog: ReportCatalog, *, chunks: Sequence[Chunk] | None = None) -> None:
    """Internal factory ownership and exact ranking-input binding, not a sandbox."""
    _require(type(catalog) is ReportCatalog and catalog._construction_token is _CATALOG_TOKEN,
             'report catalog must be source-built')
    if chunks is not None:
        _require(tuple(chunks) == catalog._chunks, 'report catalog differs from bound corpus chunks')


def build_report_catalog(snapshot: StatementSnapshot, *, corpus_manifest: Path,
                         chunks_path: Path) -> ReportCatalog:
    """Bind selected source-built reports to the snapshot's full corpus inputs.

    A selected snapshot may intentionally omit other registered reports. Those
    reports never enter its catalog or unrestricted ranking scope. Unsupported
    source layouts keep their declared report but contribute no invented dates.
    """
    _require(type(snapshot) is StatementSnapshot, 'statement snapshot must be source-built')
    validate_income_index(snapshot.index)
    snapshot.verify_unchanged()
    _require(type(snapshot.sources) is tuple and bool(snapshot.sources), 'source registration is missing')
    sources = {}
    for source in snapshot.sources:
        _require(type(source.document_id) is str and _ID.fullmatch(source.document_id) is not None
                 and source.document_id not in sources, 'invalid or duplicate source document ID')
        _require(type(source.sha256) is str and _DIGEST.fullmatch(source.sha256) is not None,
                 'registered PDF must have a canonical source digest')
        sources[source.document_id] = source
    # Reuse the established model-corpus contract, including actual chunk counts
    # and complete source identity. API provenance alone permits legacy {} data.
    manifest_identity, chunks_identity = _model_binding(corpus_manifest, chunks_path, snapshot.sources)
    identities = {identity.path: identity for identity in snapshot.input_identities}
    _require(len(identities) == len(snapshot.input_identities), 'duplicate snapshot input identity')
    _require(identities.get(corpus_manifest) == manifest_identity
             and identities.get(chunks_path) == chunks_identity, 'corpus inputs are not bound to statement snapshot')
    records, identities_seen = {}, set()
    _require(type(snapshot.index.reports) is tuple and 0 < len(snapshot.index.reports) <= 128,
             'invalid report catalog budget')
    for record in snapshot.index.reports:
        _require(type(record) is tuple and len(record) == 4, 'invalid report declaration')
        document_id, digest, year, quarter = record
        _require(type(document_id) is str and _ID.fullmatch(document_id) is not None
                 and document_id not in records, 'invalid or duplicate report document ID')
        _require(type(digest) is str and _DIGEST.fullmatch(digest) is not None
                 and document_id in sources and sources[document_id].sha256 == digest,
                 'report PDF is not bound to source registration and corpus')
        _require(type(year) is int and 1900 <= year <= 9999
                 and (quarter is None or type(quarter) is int and quarter in (1, 2, 3, 4)),
                 'invalid fiscal report identity')
        _require((year, quarter) not in identities_seen, 'duplicate report identity')
        identities_seen.add((year, quarter))
        declared_period = sources[document_id].reporting_period
        if declared_period is not None:
            _require(declared_period['fiscal_year'] == year and declared_period['quarter'] == quarter,
                     'report identity contradicts source declaration')
        records[document_id] = (digest, year, quarter)
    snapshot.verify_report_bindings()
    columns: dict[str, set[ColumnPeriod]] = defaultdict(set)
    for cell in snapshot.index.cells:
        _require(type(cell) is IncomeCell and cell.document_id in records,
                 'source cell has an unknown report owner')
        _require((cell.pdf_sha256, cell.fiscal_year, cell.fiscal_quarter) == records[cell.document_id],
                 'source cell contradicts report identity')
        _require(type(cell.period_end) is str, 'invalid source column date')
        try:
            parsed_date = date.fromisoformat(cell.period_end)
        except ValueError as error:
            raise ValueError('invalid source column date') from error
        _require(parsed_date.isoformat() == cell.period_end, 'noncanonical source column date')
        _require((cell.fiscal_quarter is None and cell.parser_profile == ANNUAL_PROFILE
                  and cell.duration_months is None)
                 or (cell.fiscal_quarter is not None and cell.parser_profile == PROFILE
                     and type(cell.duration_months) is int and cell.duration_months in (3, 6)),
                 'invalid source column profile and duration')
        columns[cell.document_id].add(ColumnPeriod(cell.period_end, cell.duration_months))
    chunks = tuple(load_chunks(chunks_path))
    snapshot.verify_unchanged()
    catalog = ReportCatalog(tuple(ReportScope(document_id, *record, frozenset(columns[document_id]))
                                  for document_id, record in records.items()), chunks_identity.sha256, chunks)
    object.__setattr__(catalog, '_construction_token', _CATALOG_TOKEN)
    return catalog


@dataclass(frozen=True, slots=True)
class LiteralPeriodScope:
    query: str
    fiscal_years: frozenset[int]
    quarters: frozenset[int]
    annual: bool
    dates: frozenset[str]
    durations: frozenset[int]
    issues: tuple[str, ...]
    has_literal_scope: bool


def parse_literal_period_scope(query: str) -> LiteralPeriodScope:
    """Recognize a bounded literal grammar; preserve the original encoder text."""
    _require(type(query) is str and 0 < len(query) <= 4000 and bool(query.strip()), 'invalid routing query')
    years, quarters, dates, durations, issues = set(), set(), set(), set(), set()
    scoped = False
    # Reject malformed numeric period tokens before the bounded grammar can
    # consume a valid-looking suffix/prefix (Q2.5 -> Q2, -3 months -> 3).
    # A dot ending a sentence is allowed; a decimal or slash operand is not.
    # Keep the compact FY2025Q2 spelling and ordinary Q-prefixed prose intact.
    mantissa = r'(?:\d+(?:\.\d*)?|\.\d+)'
    exponent = r'[eE][+\-\u2212]?\d+'
    unsigned_number = mantissa + '(?:' + exponent + ')?'
    signed_number = r'[+\-\u2212]\s*' + unsigned_number
    decimal_number = r'(?:\d+\.\d+|\.\d+)'
    exponent_number = mantissa + exponent
    slash_operand = unsigned_number + r'(?:st|nd|rd|th)?'
    slash_number = slash_operand + r'\s*/\s*(?:Q\s*)?[+\-\u2212]?\s*' + slash_operand
    invalid_number = '(?:' + '|'.join((signed_number, decimal_number, slash_number, exponent_number)) + ')'
    # Before an explicit unit, even a trailing decimal dot is malformed.
    # Do not exclude punctuation/letters before a sign: the existing grammar
    # can consume its unsigned suffix there (income.-3 months, 1e-3 months).
    # Unsigned starts inside a digit run are redundant and cause quadratic
    # rescanning. Signs remain unrestricted so punctuation cannot hide them.
    invalid_unit_number = '(?:' + signed_number + r'|(?<!\d)(?:' + '|'.join((
        r'(?:\d+\.\d*|\.\d+)', slash_number, exponent_number)) + '))'
    invalid_numeric_periods = (
        r'(?<![A-Za-z])Q\s*' + invalid_number,
        invalid_unit_number + r'(?:st|nd|rd|th)?[-\s]+quarter\b',
        invalid_unit_number + r'[-\s]+months?\b',
    )
    if any(re.search(pattern, query, re.I) for pattern in invalid_numeric_periods):
        scoped = True
        issues.add('INVALID_PERIOD_LITERAL')
    fiscal_spans = []
    for match in re.finditer(r'\b(?:fiscal(?:\s+year)?\s*|FY\s*)(\d+)(?=Q\d|\b)', query, re.I):
        scoped = True
        fiscal_spans.append((match.start(), match.end()))
        value = int(match[1])
        if len(match[1]) != 4 or not 1900 <= value <= 9999:
            issues.add('INVALID_PERIOD_LITERAL')
        else:
            years.add(value)
    quarter_spans = []
    quarter_words = '|'.join((*_ORDINALS, 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'))
    for match in re.finditer(r'(?<![A-Za-z])Q(\d+)\b|\b(' + quarter_words
                            + r'|\d+(?:st|nd|rd|th)?)[-\s]+quarter\b', query, re.I):
        scoped = True
        quarter_spans.append((match.start(), match.end()))
        literal = match[2].lower() if match[2] else ''
        ordinal_number = re.fullmatch(r'(\d+)(?:st|nd|rd|th)?', literal)
        value = int(match[1]) if match[1] else int(ordinal_number[1]) if ordinal_number else _ORDINALS.get(literal)
        if value not in (1, 2, 3, 4):
            issues.add('INVALID_PERIOD_LITERAL')
        else:
            quarters.add(value)
    annual = bool(re.search(r'\bannual\b|\byear[-\s]+ended\b', query, re.I))
    scoped = scoped or annual
    for match in re.finditer(r'(?<![\w.])(\d+(?:\.\d+)?|[A-Za-z]+)[-\s]+months?\b', query, re.I):
        scoped = True
        duration = _DURATIONS.get(match[1].lower())
        if duration is None:
            if not match[1].isdigit():
                issues.add('INVALID_PERIOD_LITERAL')
                continue
            duration = int(match[1])
        if not 1 <= duration <= 12:
            issues.add('INVALID_PERIOD_LITERAL')
        else:
            durations.add(duration)
    # Calendar/ISO dates are column restrictions, independently of report FY.
    date_spans = []
    for match in re.finditer(r'\b(\d{4}-\d{2}-\d{2})\b|\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b', query):
        if match[1] is None and match[2].lower() not in _MONTHS:
            # Unknown month wording is relevant only when explicitly ended.
            if not re.search(r'\bended\s+(?:on\s+)?$', query[:match.start()], re.I):
                continue
        scoped = True
        date_spans.append((match.start(), match.end()))
        try:
            value = date.fromisoformat(match[1]) if match[1] else date(
                int(match[4]), _MONTHS[match[2].lower()], int(match[3]))
            dates.add(value.isoformat())
        except (ValueError, KeyError):
            issues.add('INVALID_PERIOD_LITERAL')
    # An explicit ended restriction with an unsupported date must not vanish
    # into global retrieval. It is deliberately not a general language parser.
    for match in re.finditer(r'\bended\s+(?:on\s+)?', query, re.I):
        scoped = True
        if not any(start == match.end() for start, _ in date_spans):
            issues.add('INVALID_PERIOD_LITERAL')
    generic_quarters = [match for match in re.finditer(r'\bquarter\b', query, re.I)
                        if not any(start <= match.start() and match.end() <= end
                                   for start, end in quarter_spans)]
    if generic_quarters:
        scoped = True
        issues.add('UNSUPPORTED_PERIOD_SCOPE')
    # Period-looking unsupported tokens cannot silently erase a restriction.
    for pattern, spans in ((r'\b(?:FY[\w.-]*|fiscal(?:\s+year)?\s*[\w.-]+)', fiscal_spans),
                           (r'(?<![A-Za-z])Q\d+\w*\b|\b\w+[-\s]+quarter\b', quarter_spans)):
        for match in re.finditer(pattern, query, re.I):
            if match[0].casefold() == 'quarter':
                continue  # Generic quarter wording has an explicit unsupported guard above.
            if not any(start <= match.start() and match.end() <= end for start, end in spans):
                # Compact FY2025Q2 spans two accepted literal fields.
                if re.fullmatch(r'FY\d{4}Q[1-4]', match[0], re.I):
                    continue
                scoped = True
                issues.add('INVALID_PERIOD_LITERAL')
    # Alphabetic Qfoo is a malformed report restriction only immediately after
    # a literal FY declaration. Ordinary Question/Quality prose is not a Q#.
    for _, end in fiscal_spans:
        match = re.match(r'[-\s]*(Q[A-Za-z]+)\b', query[end:], re.I)
        if match and match[1].casefold() != 'quarter':
            scoped = True
            issues.add('INVALID_PERIOD_LITERAL')
    if re.search(r'\bcalendar\s+(?:year|quarter|Q\d)\b', query, re.I):
        scoped = True
        issues.add('UNSUPPORTED_PERIOD_SCOPE')
    # In a stated fiscal comparison, retain a second year adjacent to Q wording
    # instead of silently discarding it. No bare calendar year becomes a FY.
    if years and re.search(r'\bcompare\b|\bversus\b|\bvs\.?\b|\band\b', query, re.I):
        for match in re.finditer(r'\b(\d{4})\s+Q[1-4]\b|\bQ[1-4]\s+(\d{4})\b', query, re.I):
            years.add(int(match[1] or match[2]))
    return LiteralPeriodScope(query, frozenset(years), frozenset(quarters), annual,
        frozenset(dates), frozenset(durations), tuple(sorted(issues)), scoped)


@dataclass(frozen=True, slots=True)
class ScopeDecision:
    disposition: str
    reason: str
    document_ids: frozenset[str] = frozenset()


def decide_period_scope(catalog: ReportCatalog, query: str) -> ScopeDecision:
    """Select one source report, or clarify/refuse without unrelated fallback."""
    validate_report_catalog(catalog)
    parsed = parse_literal_period_scope(query)
    if parsed.issues:
        return ScopeDecision('refuse', 'INVALID_PERIOD_LITERAL' if 'INVALID_PERIOD_LITERAL' in parsed.issues
                             else 'UNSUPPORTED_PERIOD_SCOPE')
    if len(parsed.fiscal_years) > 1 or len(parsed.quarters) > 1:
        return ScopeDecision('clarify', 'MULTIPLE_REPORT_SCOPE')
    if (len(parsed.dates) > 1 or len(parsed.durations) > 1
            or parsed.annual and (parsed.quarters or parsed.durations)):
        return ScopeDecision('clarify', 'CONFLICTING_PERIOD_SCOPE')
    if not parsed.has_literal_scope:
        return ScopeDecision('global', 'NO_LITERAL_PERIOD_SCOPE',
                             frozenset(report.document_id for report in catalog.reports))
    reports = [report for report in catalog.reports
               if (not parsed.fiscal_years or report.fiscal_year in parsed.fiscal_years)
               and (not parsed.quarters or report.fiscal_quarter in parsed.quarters)
               and (not parsed.annual or report.fiscal_quarter is None)]
    if not reports:
        return ScopeDecision('refuse', 'UNKNOWN_REPORT_SCOPE')
    if parsed.dates or parsed.durations:
        reports = [report for report in reports if any(
            (not parsed.dates or column.period_end in parsed.dates)
            and (not parsed.durations or column.duration_months in parsed.durations)
            for column in report.columns)]
        if not reports:
            return ScopeDecision('refuse', 'UNAVAILABLE_SOURCE_COLUMN')
    if len(reports) != 1:
        return ScopeDecision('clarify', 'AMBIGUOUS_REPORT_SCOPE')
    return ScopeDecision('selected', 'EXPLICIT_SOURCE_REPORT', frozenset({reports[0].document_id}))


class SourceBoundPeriodRetriever:
    """Apply report scope first; retain the dense retriever's exact arithmetic."""
    def __init__(self, retriever: DenseRetriever, *, report_catalog: ReportCatalog) -> None:
        validate_report_catalog(report_catalog, chunks=retriever.chunks)
        self._retriever = retriever
        self._catalog = report_catalog

    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        return list(self.inspect(query, top_k=top_k).candidates)

    def inspect(self, query: str, *, top_k: int) -> DenseInspection:
        decision = decide_period_scope(self._catalog, query)
        inspection = self._retriever.inspect(query, top_k=top_k, document_ids=set(decision.document_ids))
        document_id = next(iter(decision.document_ids)) if decision.disposition == 'selected' else None
        return replace(inspection, routed_document_id=document_id,
                       routed_document_ids=tuple(sorted(decision.document_ids)),
                       route_disposition=decision.disposition, route_reason=decision.reason)
