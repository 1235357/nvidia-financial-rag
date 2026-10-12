from dataclasses import replace
from types import SimpleNamespace

import pytest

from acceptance import structured_income_comparison as comparison
from nvidia_rag.structured_income import Selection
from test_structured_income import source_pdf
from nvidia_rag.structured_income import build_income_index


def test_posthoc_target_comparison_sees_only_already_completed_selection(tmp_path):
    index = build_income_index((source_pdf(tmp_path),))
    cell = next(c for c in index.cells if c.row_path == 'monetary/revenue' and c.column_number == 1)
    target = {name:getattr(cell,name) for name in ('document_id','pdf_sha256','physical_page',
        'column_role','period_end','duration_months','amount','unit_kind','scale','currency_literal')}
    target.update(cell_id='target-cell', row=cell.row_literal,
                  report_fiscal_year=cell.fiscal_year, report_fiscal_quarter=cell.fiscal_quarter)
    case = SimpleNamespace(expected={'outcome':'evidence','cell_ids':['target-cell']}, source_cells=(target,))
    result = comparison.compare_source_candidate(case, Selection('source_candidate','EXACT', (cell,)))
    assert result['source_fields_match'] and result['financial_correctness'] == 'unestablished'
    wrong = comparison.compare_source_candidate(case, Selection('source_candidate','EXACT', (replace(cell, amount='999999'),)))
    assert not wrong['source_fields_match']
    assert not comparison.compare_source_candidate(case, Selection('refuse','NO_CELL'))['source_fields_match']


def test_negative_assessment_withholding_is_not_human_correctness():
    case = SimpleNamespace(expected={'outcome':'clarify'})
    assert comparison.compare_source_candidate(case, Selection('unsupported','NO_FX')) == {
        'kind':'provisional-negative-withholding','withheld':True,'human_gold':False}
    assert not comparison.compare_source_candidate(case, Selection('source_candidate','EXACT'))['withheld']


@pytest.mark.parametrize('late_change', ['input', 'body', 'pending'])
def test_failed_late_completion_cannot_leave_an_accepted_output(tmp_path, monkeypatch, late_change):
    output = (tmp_path/'comparison.json').resolve()
    original = comparison.write_replay_exclusive
    dirty = False
    def writer(path, value, final_check):
        nonlocal dirty
        original(path, value, final_check)
        if late_change == 'body':
            path.write_bytes(b'{"changed":true}\n')
        elif late_change == 'pending':
            path.with_name(path.name+'.pending.json').write_bytes(b'{}')
        else:
            dirty = True
    def final_check():
        if dirty:
            raise ValueError('INPUT_CHANGED')
    monkeypatch.setattr(comparison, 'write_replay_exclusive', writer)
    with pytest.raises(ValueError):
        comparison.write_completed(output, {'status':'complete'}, final_check)
    assert output.with_name(output.name+'.failure.json').is_file()
    with pytest.raises(ValueError):
        comparison.read_completed_replay(output)


def test_successful_completion_refuses_reuse_of_output(tmp_path):
    output = (tmp_path/'comparison.json').resolve()
    comparison.write_completed(output, {'status':'complete'}, lambda:None)
    assert comparison.read_completed_replay(output) == {'status':'complete'}
    with pytest.raises(ValueError):
        comparison.write_completed(output, {'status':'different'}, lambda:None)
    assert comparison.read_completed_replay(output) == {'status':'complete'}
