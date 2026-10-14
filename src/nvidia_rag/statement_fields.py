"""Explicit native income-field selection, without natural-language inference.

The caller supplies source-qualified field constraints. Results retain the
original cells and their unestablished financial correctness; no model, dense
score, arithmetic, unit conversion or source IO enters selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re

from .structured_income import IncomeIndex, Selection, validate_income_index


@dataclass(frozen=True, slots=True)
class StatementFieldRequest:
    row_path: str
    period_kind: str
    period_end: str
    duration_months: int | None
    unit_kind: str
    scale: str
    currency_literal: str | None
    document_ids: tuple[str, ...] | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    column_role: str | None = None


def validate_field_request(request: StatementFieldRequest) -> None:
    """Validate syntax and strict types; unsupported native units are outcomes."""
    def require(condition):
        if not condition:
            raise ValueError('INVALID_FIELD_REQUEST')

    require(type(request) is StatementFieldRequest)
    require(type(request.row_path) is str and 1 <= len(request.row_path) <= 256
            and re.fullmatch(r'[a-z][a-z0-9_]{0,63}/[a-z0-9]+(?:_[a-z0-9]+)*', request.row_path))
    require(type(request.period_kind) is str and request.period_kind in ('fiscal_year', 'months'))
    require(type(request.period_end) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}', request.period_end))
    try:
        require(date.fromisoformat(request.period_end).isoformat() == request.period_end)
    except (ValueError, OverflowError):
        raise ValueError('INVALID_FIELD_REQUEST') from None
    require((request.duration_months is None if request.period_kind == 'fiscal_year' else
             type(request.duration_months) is int and 1 <= request.duration_months <= 12))
    require(type(request.unit_kind) is str and request.unit_kind in ('monetary', 'per_share', 'count'))
    require(type(request.scale) is str and re.fullmatch(r'[1-9][0-9]{0,31}', request.scale))
    require(request.currency_literal is None or (type(request.currency_literal) is str
            and 0 < len(request.currency_literal) <= 32
            and all(char.isascii() and char.isprintable() and not char.isspace()
                    for char in request.currency_literal)))
    require(request.document_ids is None or (type(request.document_ids) is tuple
            and 0 < len(request.document_ids) <= 128
            and all(type(value) is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', value)
                    for value in request.document_ids)
            and len(set(request.document_ids)) == len(request.document_ids)))
    require(request.fiscal_year is None or (type(request.fiscal_year) is int
            and 1900 <= request.fiscal_year <= 9999))
    require(request.fiscal_quarter is None or (type(request.fiscal_quarter) is int
            and request.fiscal_quarter in (1, 2, 3, 4)))
    require(request.column_role is None or (type(request.column_role) is str
            and request.column_role in ('current', 'prior_comparison')))


def select_statement_fields(index: IncomeIndex, request: StatementFieldRequest) -> Selection:
    """Match every supplied native field, withholding absent or ambiguous cells.

    Fiscal identity constrains the source report, never the column's calendar
    year. Optional role means either observed role, rather than a current-column
    default. No source candidate is selected by ordering or an amount supplied
    by a caller.
    """
    validate_income_index(index)
    validate_field_request(request)
    reports = {row[0]: row for row in index.reports}
    scope = tuple(reports) if request.document_ids is None else request.document_ids
    if any(document not in reports for document in scope):
        return Selection('refuse', 'OPERATOR_REPORT_UNAVAILABLE')
    if request.period_kind == 'fiscal_year' and request.fiscal_quarter is not None:
        return Selection('clarify', 'REQUEST_CONFLICT')
    matching_reports = {document for document, _, year, quarter in index.reports
        if (request.fiscal_year is None or year == request.fiscal_year)
        and (request.fiscal_quarter is None or quarter == request.fiscal_quarter)}
    if not matching_reports:
        return Selection('refuse', 'REPORT_UNAVAILABLE')
    chosen_reports = matching_reports.intersection(scope)
    if not chosen_reports:
        return Selection('clarify', 'REPORT_SCOPE_CONFLICT')

    section = request.row_path.split('/', 1)[0]
    if section != request.unit_kind:
        return Selection('unsupported', 'REQUESTED_UNIT_KIND_MISMATCH')
    native_scale = '1' if request.unit_kind == 'per_share' else '1000000'
    if request.scale != native_scale:
        return Selection('unsupported', 'REQUESTED_SCALE_CONVERSION_UNSUPPORTED')
    if request.unit_kind == 'count':
        if request.currency_literal is not None:
            return Selection('unsupported', 'CURRENCY_NOT_APPLICABLE')
    elif request.currency_literal != '$':
        return Selection('unsupported', 'REQUESTED_CURRENCY_MISMATCH' if request.currency_literal is None
                         else 'ISO_CURRENCY_UNVERIFIED_NO_FX')

    candidates = tuple(cell for cell in index.cells
        if cell.document_id in chosen_reports
        and cell.row_path == request.row_path
        and cell.period_kind == request.period_kind
        and cell.period_end == request.period_end
        and cell.duration_months == request.duration_months
        and cell.unit_kind == request.unit_kind
        and cell.scale == request.scale
        and cell.currency_literal == request.currency_literal
        and (request.column_role is None or cell.column_role == request.column_role))
    if not candidates:
        return Selection('refuse', 'NO_INDEXED_CELL_MATCH')
    if len(candidates) != 1:
        return Selection('clarify', 'MULTIPLE_SOURCE_MATCHES')
    return Selection('source_candidate', 'EXACT_TYPED_FIELDS_MATCH', candidates)
