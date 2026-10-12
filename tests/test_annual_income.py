"""Annual source geometry and fiscal-year semantics use real synthetic PDFs."""
from dataclasses import replace
from hashlib import sha256

import pymupdf
import pytest

from nvidia_rag.structured_income import (
    ReportInput, build_income_index, parse_statement_request, select_statement_candidates,
)


def annual_pdf(tmp_path, *, split_date=False, misplaced_year=False, reversed_dates=False, placeholder=False):
    pdf=pymupdf.open()
    pdf.new_page()
    page=pdf.new_page()
    def text(x,y,value):
        page.insert_text((x,y),value,fontsize=8)
    text(180,100,'CONSOLIDATED STATEMENTS OF INCOME' if split_date else 'Consolidated Statements of Income')
    text(195,120,'(In millions, except per share data)')
    text(420,150,'Year Ended')
    glyphs=[300,400,500]
    rights=[390,490,590]
    dates=['Jan 26, 2025','Jan 28, 2024','Jan 29, 2023']
    if reversed_dates:
        dates[0],dates[1]=dates[1],dates[0]
    for i,(x,value) in enumerate(zip(glyphs,dates)):
        if split_date:
            month,day,year=value.split()
            text(x+15,170,'January '+day)
            text(x+30+(110 if misplaced_year and i==0 else 0),184,year)
        else:
            text(x+15,175,value)
    def row(label,y,values,dollars=False):
        text(20,y,label)
        for x,right,value in zip(glyphs,rights,values):
            if dollars:
                text(x,y,'$')
            text(right-pymupdf.get_text_length(value,fontsize=8),y,value)
    row('Revenue',210,['12,111','9,333','7,555'],True)
    row('Cost of revenue',230,['3,444','2,222','1,999'])
    row('Interest expense',250,['(17)','(9)','(5)'])
    if placeholder:
        row('Acquisition termination cost',265,['-','999','-'])
    row('Net income',280,['8,650','7,102','5,551'],True)
    text(20,315,'Net income per share:')
    row('Basic',340,['1.27','0.91','0.72'],True)
    row('Diluted',360,['1.21','0.85','0.69'],True)
    text(20,400,'Weighted average shares used in per share computation:')
    row('Basic',425,['6,811','6,701','6,555'])
    row('Diluted',445,['7,011','6,999','6,811'])
    text(100,490,'See accompanying notes to the consolidated financial statements.')
    path=(tmp_path/('annual-split.pdf' if split_date else 'annual-full.pdf')).resolve()
    pdf.save(path)
    pdf.close()
    return ReportInput('nvidia-fy2025-annual',path,sha256(path.read_bytes()).hexdigest(),2025,None)


def select(index,query,scope=None):
    request=parse_statement_request(query,operator_document_ids=scope or tuple(r[0] for r in index.reports),index=index)
    return select_statement_candidates(index,request)


@pytest.mark.parametrize('split_date',[False,True])
def test_annual_three_columns_retain_native_period_and_date_components(tmp_path,split_date):
    report=annual_pdf(tmp_path,split_date=split_date)
    index=build_income_index((report,))
    assert len(index.cells)==24 and not index.omissions
    assert {c.physical_page for c in index.cells}=={2}
    assert {c.period_end for c in index.cells}=={'2025-01-26','2024-01-28','2023-01-29'}
    assert all(c.duration_months is None and c.period_kind=='fiscal_year' for c in index.cells)
    assert all(c.fiscal_quarter is None for c in index.cells)
    with pymupdf.open(report.pdf_path) as pdf:
        native=pdf[1].get_text('text',sort=True)
    for cell in index.cells:
        roles=dict(cell.bindings)
        assert roles['duration'].literal=='Year Ended'
        assert ('date_year' in roles)==split_date
        for span in roles.values():
            assert native[span.start:span.end]==span.literal
    negative=next(c for c in index.cells if c.row_path=='monetary/interest_expense' and c.column_number==1)
    assert negative.amount=='-17'
    assert {c.row_path for c in index.cells if c.row_literal=='Basic'}=={'per_share/basic','count/basic'}


@pytest.mark.parametrize('wording',['annual period ended','year ended'])
def test_annual_literal_request_selects_one_fiscal_year_cell(tmp_path,wording):
    index=build_income_index((annual_pdf(tmp_path),))
    result=select(index,'According to NVIDIA fiscal 2025 report, what was net income for the '+wording+' January 26, 2025, in millions of dollars?')
    assert result.disposition=='source_candidate' and result.financial_correctness=='unestablished'
    assert result.cells[0].amount=='8650' and result.cells[0].duration_months is None


def test_annual_does_not_satisfy_literal_twelve_month_request(tmp_path):
    index=build_income_index((annual_pdf(tmp_path),))
    result=select(index,'What was net income for the twelve-month period ended January 26, 2025, in millions of dollars?')
    assert result.disposition!='source_candidate'
    conflict=select(index,'What was annual net income for the three-month period ended January 26, 2025?')
    assert conflict.disposition=='clarify' and conflict.reason=='REQUEST_CONFLICT'


def test_split_annual_year_cannot_cross_into_neighbor_column(tmp_path):
    index=build_income_index((annual_pdf(tmp_path,split_date=True,misplaced_year=True),))
    assert not index.cells
    assert index.omissions and index.omissions[0].physical_page==2


def test_annual_current_and_comparison_dates_must_be_descending(tmp_path):
    index=build_income_index((annual_pdf(tmp_path,reversed_dates=True),))
    assert not index.cells and any(o.reason=='COMPARISON_DATE_ORDER' for o in index.omissions)


def test_annual_fiscal_declaration_cannot_parse_a_condensed_quarter_table(tmp_path):
    from test_structured_income import source_pdf
    report=source_pdf(tmp_path)
    index=build_income_index((replace(report,document_id='nvidia-fy2025-annual',fiscal_quarter=None),))
    assert not index.cells and index.omissions[0].reason=='ANNUAL_STATEMENT_NOT_FOUND'


def test_annual_missing_amount_does_not_erase_neighboring_supported_cell(tmp_path):
    index=build_income_index((annual_pdf(tmp_path,placeholder=True),))
    row=[c for c in index.cells if c.row_literal=='Acquisition termination cost']
    assert len(index.cells)==25 and len(row)==1
    assert row[0].column_number==2 and row[0].amount=='999'
    assert row[0].period_end=='2024-01-28'
    omitted=[o for o in index.omissions if o.row_literal=='Acquisition termination cost']
    assert {o.reason for o in omitted}=={'UNSUPPORTED_VALUE_COLUMN_1','UNSUPPORTED_VALUE_COLUMN_3'}
