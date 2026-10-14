"""Typed selection exercises native PDF units and source identity, not gold accuracy."""
from dataclasses import FrozenInstanceError, replace
from datetime import date
import importlib
import importlib.util

import pytest

from nvidia_rag.structured_income import IncomeIndex, build_income_index
from test_annual_income import annual_pdf
from test_structured_income import source_pdf


def fields():
    assert importlib.util.find_spec('nvidia_rag.statement_fields') is not None, \
        'typed source-field selection is missing'
    return importlib.import_module('nvidia_rag.statement_fields')


@pytest.fixture
def indexed(tmp_path):
    annual = annual_pdf(tmp_path, split_date=True)
    quarter = source_pdf(tmp_path)
    return build_income_index((annual, quarter)), annual, quarter


def request(**changes):
    values = dict(row_path='per_share/basic', period_kind='months', period_end='2024-07-28',
                  duration_months=3, unit_kind='per_share', scale='1', currency_literal='$')
    values.update(changes)
    return fields().StatementFieldRequest(**values)


@pytest.mark.parametrize('row,kind,scale,currency,amount', [
    ('per_share/basic', 'per_share', '1', '$', '1.07'),
    ('per_share/diluted', 'per_share', '1', '$', '1.03'),
    ('count/basic', 'count', '1000000', None, '2888'),
    ('count/diluted', 'count', '1000000', None, '3111'),
    ('monetary/interest_expense', 'monetary', '1000000', '$', '-17'),
    ('monetary/revenue', 'monetary', '1000000', '$', '4111'),
])
def test_section_qualified_units_select_complete_unchanged_native_cells(indexed, row, kind, scale, currency, amount):
    index, _, quarter = indexed
    original = next(c for c in index.cells if c.document_id == quarter.document_id
        and c.row_path == row and c.period_end == '2024-07-28' and c.duration_months == 3)
    result = fields().select_statement_fields(index, request(row_path=row, unit_kind=kind,
        scale=scale, currency_literal=currency, document_ids=(quarter.document_id,)))
    assert result.disposition == 'source_candidate' and result.reason == 'EXACT_TYPED_FIELDS_MATCH'
    assert result.cells == (original,) and result.cells[0] is original
    assert original.amount == amount
    assert result.financial_correctness == 'unestablished'
    assert result.score_semantics == 'exact-source-field-selection;no-cosine-score'
    if kind != 'monetary':
        assert 'section' in dict(original.bindings)
    if kind == 'count':
        assert original.currency_normalization == 'not_applicable'


def test_duration_and_comparison_role_are_exact_constraints_not_default_current(indexed):
    index, _, quarter = indexed
    result = fields().select_statement_fields(index, request(period_end='2023-07-30', duration_months=6,
        document_ids=(quarter.document_id,)))
    assert result.cells[0].amount == '1.09' and result.cells[0].column_role == 'prior_comparison'
    assert fields().select_statement_fields(index, request(period_end='2023-07-30', duration_months=6,
        column_role='current')).reason == 'NO_INDEXED_CELL_MATCH'


def test_annual_null_duration_keeps_report_year_separate_from_comparison_date(indexed):
    index, annual, _ = indexed
    original = next(c for c in index.cells if c.document_id == annual.document_id
        and c.row_path == 'per_share/basic' and c.column_number == 2)
    result = fields().select_statement_fields(index, request(period_kind='fiscal_year', duration_months=None,
        period_end=original.period_end, fiscal_year=2025, document_ids=(annual.document_id,)))
    assert result.cells == (original,) and original.fiscal_year == 2025
    assert original.duration_months is None and original.period_kind == 'fiscal_year'


@pytest.mark.parametrize('changes,reason', [
    ({'unit_kind': 'count'}, 'REQUESTED_UNIT_KIND_MISMATCH'),
    ({'scale': '1000000'}, 'REQUESTED_SCALE_CONVERSION_UNSUPPORTED'),
    ({'currency_literal': 'USD'}, 'ISO_CURRENCY_UNVERIFIED_NO_FX'),
    ({'currency_literal': None}, 'REQUESTED_CURRENCY_MISMATCH'),
    ({'row_path': 'count/basic', 'unit_kind': 'count', 'scale': '1000000'}, 'CURRENCY_NOT_APPLICABLE'),
    ({'row_path': 'monetary/revenue', 'unit_kind': 'monetary', 'scale': '1'}, 'REQUESTED_SCALE_CONVERSION_UNSUPPORTED'),
])
def test_unit_constraints_are_never_dropped_or_converted(indexed, changes, reason):
    result = fields().select_statement_fields(indexed[0], request(**changes))
    assert (result.disposition, result.reason, result.cells) == ('unsupported', reason, ())


@pytest.mark.parametrize('changes,disposition,reason', [
    ({'document_ids': ('not-registered',)}, 'refuse', 'OPERATOR_REPORT_UNAVAILABLE'),
    ({'fiscal_year': 2035}, 'refuse', 'REPORT_UNAVAILABLE'),
    ({'row_path': 'per_share/not_present'}, 'refuse', 'NO_INDEXED_CELL_MATCH'),
    ({'period_end': '2024-07-29'}, 'refuse', 'NO_INDEXED_CELL_MATCH'),
    ({'duration_months': 9}, 'refuse', 'NO_INDEXED_CELL_MATCH'),
    ({'period_kind': 'fiscal_year', 'duration_months': None, 'fiscal_quarter': 2}, 'clarify', 'REQUEST_CONFLICT'),
])
def test_scope_absence_and_request_conflict_withhold_all_cells(indexed, changes, disposition, reason):
    result = fields().select_statement_fields(indexed[0], request(**changes))
    assert (result.disposition, result.reason, result.cells) == (disposition, reason, ())


def test_unknown_scope_member_withholds_known_source_and_report_conflict_is_distinct(indexed):
    index, annual, quarter = indexed
    assert fields().select_statement_fields(index, request(document_ids=(quarter.document_id, 'missing'))).reason == 'OPERATOR_REPORT_UNAVAILABLE'
    result = fields().select_statement_fields(index, request(document_ids=(annual.document_id,), fiscal_quarter=2))
    assert (result.disposition, result.reason) == ('clarify', 'REPORT_SCOPE_CONFLICT')


def test_overlapping_reports_require_scope_instead_of_first_match(indexed):
    index, annual, quarter = indexed
    other = replace(quarter, document_id='another-quarter')
    index = build_income_index((annual, quarter, other))
    result = fields().select_statement_fields(index, request())
    assert (result.disposition, result.reason, result.cells) == ('clarify', 'MULTIPLE_SOURCE_MATCHES', ())
    assert fields().select_statement_fields(index, request(document_ids=(other.document_id,))).cells[0].document_id == other.document_id


@pytest.mark.parametrize('changes', [
    {'duration_months': True}, {'duration_months': 0}, {'duration_months': 13},
    {'duration_months': '3'}, {'duration_months': None},
    {'period_kind': 'fiscal_year', 'duration_months': 12},
    {'period_end': '2024-7-28'}, {'period_end': '2024-02-30'},
    {'period_end': date(2024, 7, 28)}, {'period_end': '20240728'},
    {'period_kind': 'calendar_year'}, {'unit_kind': 'percentage'},
    {'row_path': '../basic'}, {'row_path': 'basic'}, {'row_path': 'per_share/basic\n'},
    {'scale': 1}, {'scale': ''}, {'currency_literal': 1},
    {'document_ids': ()}, {'document_ids': ['one']}, {'document_ids': ('one', 'one')},
    {'fiscal_year': True}, {'fiscal_quarter': True}, {'fiscal_quarter': 5},
    {'column_role': 'latest'},
])
def test_pure_domain_input_validation_rejects_coercion_and_invalid_fields(indexed, changes):
    with pytest.raises(ValueError, match='INVALID_FIELD_REQUEST'):
        fields().select_statement_fields(indexed[0], request(**changes))


def test_forged_index_is_rejected_and_requests_are_immutable(indexed):
    index = indexed[0]
    with pytest.raises(ValueError, match='INVALID_SELECTION_INPUT'):
        fields().select_statement_fields(IncomeIndex(index.reports, index.cells, index.omissions), request())
    with pytest.raises(FrozenInstanceError):
        request().scale = '1000000'
