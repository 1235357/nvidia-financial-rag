"""Real synthetic PDF tables exercise source binding and request withholding."""
from dataclasses import replace
from decimal import localcontext
from hashlib import sha256
from pathlib import Path

import pymupdf
import pytest

from nvidia_rag.structured_income import (
    EPS, FOOTER, SHARES, TITLE, UNIT, IncomeIndex, ReportInput,
    _amount, build_income_index, parse_statement_request, select_statement_candidates,
)


def source_pdf(tmp_path, *, quarter=2, duplicate=False, ambiguous=False):
    """Distinct report and non-gold metrics, pages, dates, signs and decimals."""
    pdf = pymupdf.open()
    pdf.new_page()  # Income table deliberately not physical page 3.
    page = pdf.new_page()
    def text(x, y, value):
        page.insert_text((x, y), value, fontsize=8)
    text(190, 110, TITLE)
    text(205, 130, UNIT)
    glyphs = [240, 330, 420, 510] if quarter == 2 else [360, 475]
    rights = [315, 405, 495, 585] if quarter == 2 else [460, 585]
    dates = ['Jul 28, 2024', 'Jul 30, 2023'] * 2 if quarter == 2 else ['Apr 28, 2024', 'Apr 30, 2023']
    duration_x = [290, 470] if quarter == 2 else [430]
    text(duration_x[0], 165, 'Three Months Ended')
    if quarter == 2:
        text(duration_x[1], 165, 'Six Months Ended')
    for i, (x, value) in enumerate(zip(glyphs, dates)):
        text(x + 20, 185, value)
    def row(label, y, values, dollars=False):
        text(20, y, label)
        for x, right, value in zip(glyphs, rights, values):
            if dollars:
                text(x, y, '$')
            text(right - pymupdf.get_text_length(value, fontsize=8), y, value)
    n = len(glyphs)
    row('Revenue', 210, ['4,111', '2,333', '8,777', '5,666'][:n], True)
    row('Cost of revenue', 230, ['1,234', '999', '2,222', '1,777'][:n])
    row('Operating income', 250, ['2,877', '1,334', '6,555', '3,889'][:n])
    row('Interest expense', 270, ['(17)', '(9)', '(30)', '(15)'][:n])
    if duplicate:
        row('Operating income', 290, ['1,111']*n)
    row('Net income', 310, ['2,860', '1,325', '6,525', '3,874'][:n], True)
    text(20, 345, EPS)
    row('Basic', 370, ['1.07', '0.59', '2.87', '1.09'][:n], True)
    row('Diluted', 390, ['1.03', '0.55', '2.71', '1.01'][:n], True)
    text(20, 420, SHARES)
    row('Basic', 445, ['2,888', '2,777', '2,555', '2,444'][:n])
    row('Diluted', 465, ['3,111', '3,001', '2,777', '2,666'][:n])
    if ambiguous:
        text(glyphs[0]+25, 230, '123')
    text(100, 510, FOOTER)
    # Another page has only a prose mention; it must not become a table.
    notes = pdf.new_page()
    notes.insert_text((20, 70), TITLE + ' include stock-based compensation.', fontsize=8)
    path = tmp_path/'quarter.pdf'
    pdf.save(path)
    pdf.close()
    return ReportInput('nvidia-fy2025-q' + str(quarter), path.resolve(),
                       sha256(path.read_bytes()).hexdigest(), 2025, quarter)


@pytest.fixture
def index(tmp_path):
    return build_income_index((source_pdf(tmp_path),))


def query(metric='revenue', *, duration='three', end='July 28, 2024', unit='millions of dollars'):
    return ("According to NVIDIA's fiscal 2025 second-quarter report, what was " + metric +
            ' for the ' + duration + '-month period ended ' + end + ', in ' + unit + '?')


def select(index, text, scope=None):
    parsed = parse_statement_request(text, operator_document_ids=scope or tuple(r[0] for r in index.reports), index=index)
    return select_statement_candidates(index, parsed)


def test_synthetic_pdf_whole_statement_and_source_identity(index):
    assert len(index.cells) == 36 and not index.omissions
    assert {c.physical_page for c in index.cells} == {2}
    assert sum(c.unit_kind == 'monetary' for c in index.cells) == 20
    assert sum(c.unit_kind == 'per_share' for c in index.cells) == 8
    assert sum(c.unit_kind == 'count' for c in index.cells) == 8
    assert len({c.cell_id for c in index.cells}) == 36
    assert {c.row_path for c in index.cells if c.row_literal == 'Basic'} == {'per_share/basic', 'count/basic'}
    assert all(c.financial_correctness == 'unestablished' for c in (select(index, query()),))


def test_q1_two_columns_april_and_comparison(tmp_path):
    index = build_income_index((source_pdf(tmp_path, quarter=1),))
    assert len(index.cells) == 18 and {c.duration_months for c in index.cells} == {3}
    assert {c.period_end for c in index.cells} == {'2024-04-28', '2023-04-30'}


@pytest.mark.parametrize('metric,duration,end,amount', [
    ('revenue', 'three', 'July 28, 2024', '4111'),
    ('revenue', 'six', 'Jul 28, 2024', '8777'),
    ('revenue', 'three', 'July 30, 2023', '2333'),
    ('operating income', 'six', 'Jul 30, 2023', '3889'),
    ('interest expense', 'three', 'July 28, 2024', '-17'),
])
def test_request_selects_report_row_period_column_and_signed_amount(index, metric, duration, end, amount):
    result = select(index, query(metric, duration=duration, end=end))
    assert result.disposition == 'source_candidate'
    assert result.cells[0].amount == amount
    assert {role for role, span in result.cells[0].bindings} == {'statement','unit','row','date','duration','value','currency'}
    assert result.cells[0].currency_normalization == 'dollar-symbol;ISO-currency-unverified'
    assert result.score_semantics.endswith('no-cosine-score')


@pytest.mark.parametrize('change', [
    {'unit':'thousands of dollars'}, {'unit':'billions of dollars'},
    {'unit':'millions of Canadian dollars'}, {'unit':'millions of yen'},
    {'unit':'millions of US dollars'}, {'unit':'GBP'},
    {'duration':'1.3'}, {'end':'Julian 28, 2024'},
    {'metric':'increase in revenue'}, {'metric':'revenue per customer'},
    {'metric':'forecast revenue'}, {'metric':'revenue multiplied by 2'},
    {'metric':'not revenue'}, {'metric':'revenue without taxes'},
    {'metric':'net income per share'},
    {'metric':'revenue / revenue'}, {'metric':'revenue - revenue'},
    {'metric':'revenue %'}, {'metric':'revenue \u00d7 revenue'}, {'metric':'revenue \u00f7 revenue'},
])
def test_unknown_or_unverified_request_constraints_never_disappear(index, change):
    result = select(index, query(**change))
    assert result.disposition in {'unsupported','refuse','clarify'} and not result.cells


def test_conflict_mentions_preserved_and_label_fields_not_available(index):
    text = ("According to NVIDIA's fiscal 2025 second-quarter report, give the six-month "
            'net income using the three-month column ended July 28, 2024.')
    parsed = parse_statement_request(text, operator_document_ids=('nvidia-fy2025-q2',), index=index)
    durations = [m for m in parsed.mentions if m.field == 'duration_months']
    assert {m.value for m in durations} == {'3','6'}
    assert {text[m.start:m.end] for m in durations} == {'three-month','six-month'}
    assert select_statement_candidates(index, parsed).reason == 'REQUEST_CONFLICT'


def test_missing_duration_date_and_absent_report_have_distinct_outcomes(index):
    missing = query().replace('three-month ', '')
    assert select(index, missing).reason == 'DURATION_REQUIRED'
    assert select(index, query(end='July 29, 2024')).reason == 'NO_INDEXED_CELL_MATCH'
    assert select(index, query(duration='one')).reason == 'NO_INDEXED_CELL_MATCH'
    assert select(index, query().replace('2025 second', '2035 second')).reason == 'REPORT_UNAVAILABLE'
    assert select(index, query(), scope=('unregistered',)).reason == 'OPERATOR_REPORT_UNAVAILABLE'


def test_reconstructed_request_or_index_cannot_bypass_factory(index):
    parsed = parse_statement_request(query(), operator_document_ids=('nvidia-fy2025-q2',), index=index)
    with pytest.raises(ValueError, match='REQUEST_PARSE_IDENTITY_MISMATCH'):
        select_statement_candidates(index, replace(parsed, query_sha256='not-a-hash'))
    mention = parsed.mentions[0]
    with pytest.raises(ValueError, match='REQUEST_PARSE_IDENTITY_MISMATCH'):
        select_statement_candidates(index, replace(parsed, mentions=(replace(mention, start=-1),)))
    with pytest.raises(ValueError, match='INVALID_SELECTION_INPUT'):
        select_statement_candidates(IncomeIndex(index.reports, index.cells, ()), parsed)
    with pytest.raises(ValueError, match='INVALID_SELECTION_INPUT'):
        select_statement_candidates(replace(index, cells=(replace(index.cells[0], amount='999999'),)), parsed)


def test_all_duplicate_rows_omitted_instead_of_first_value_winning(tmp_path):
    index = build_income_index((source_pdf(tmp_path, duplicate=True),))
    assert not any(c.row_path == 'monetary/operating_income' for c in index.cells)
    assert [o.reason for o in index.omissions].count('DUPLICATE_ROW_PATH') == 2
    assert select(index, query('operating income')).disposition != 'source_candidate'


def test_ambiguous_row_keeps_omission_and_never_guesses_column(tmp_path):
    index = build_income_index((source_pdf(tmp_path, ambiguous=True),))
    assert not any(c.row_path == 'monetary/cost_of_revenue' for c in index.cells)
    assert any(o.reason == 'AMBIGUOUS_ROW_COLUMN' for o in index.omissions)


def test_pinned_source_identity_and_no_silent_duplicate_reports(tmp_path):
    report = source_pdf(tmp_path)
    with pytest.raises(ValueError, match='SOURCE_IDENTITY_MISMATCH'):
        build_income_index((replace(report, pdf_sha256='0'*64),))
    with pytest.raises(ValueError, match='DUPLICATE_REPORT'):
        build_income_index((report, report))


def test_exact_numeric_literals_do_not_round_at_decimal_context():
    number = '123456789012345678901234567890'
    with localcontext() as ctx:
        ctx.prec = 3
        assert _amount(number, 'monetary') == number
        assert _amount('('+number+')', 'monetary') == '-'+number
        assert _amount('0.123456789012345678901234567890', 'per_share') == '0.123456789012345678901234567890'
    for value in ('-', '1,23', '1,234.50', '(17', '17)', '1e5'):
        with pytest.raises(ValueError):
            _amount(value, 'monetary')


def test_registered_annual_source_availability_is_distinct_from_parser_support(tmp_path):
    report = source_pdf(tmp_path)
    annual = replace(report, document_id='nvidia-fy2025-annual', fiscal_quarter=None)
    index = build_income_index((report, annual))
    assert len(index.reports) == 2 and len(index.cells) == 36
    assert index.omissions[0].reason == 'ANNUAL_STATEMENT_NOT_FOUND'
    assert index.omissions[0].physical_page is None
    assert select(index, query()).disposition == 'source_candidate'
    assert select(index, query(), scope=(annual.document_id,)).reason == 'REPORT_SCOPE_CONFLICT'
