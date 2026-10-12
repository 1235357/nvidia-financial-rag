"""Independent annual CLI closure controls with real synthetic source PDFs."""
from hashlib import sha256
import json

import pytest

from acceptance import structured_income_comparison as comparison
from test_annual_income import annual_pdf
from test_expanded_cases import case, experiment, write_inputs


def comparison_inputs(tmp_path):
    source_dir = (tmp_path / 'sources').resolve()
    source_dir.mkdir()
    report = annual_pdf(source_dir, split_date=True)
    issuer_url = 'https://s201.q4cdn.com/534360573/files/doc_financials/2025/annual.pdf'
    registry = (tmp_path / 'registry.json').resolve()
    registry.write_text(json.dumps({'schema_version': 1, 'sources': [{
        'document_id': report.document_id, 'label': 'Synthetic annual source',
        'filename': report.pdf_path.name, 'url': issuer_url, 'sha256': report.pdf_sha256,
    }]}), encoding='utf-8')
    cfg = experiment()
    cfg['documents'] = [{
        'document_id': report.document_id, 'report_fiscal_year': report.fiscal_year,
        'report_fiscal_quarter': None, 'pdf_sha256': report.pdf_sha256,
        'issuer_url': issuer_url, 'page_count': 2,
    }]
    cfg['input_sha256']['registry'] = sha256(registry.read_bytes()).hexdigest()
    cfg['scopes'] = [{'id': 'all-reports', 'document_ids': None}]
    cfg['rationale_bindings'][0]['kind'] = 'registry-absence'
    row = case()
    row['query'] = 'What was net income for fiscal 2035?'
    row['request'].update(report_document_id='not-in-corpus', report_fiscal_year=2035,
                          report_fiscal_quarter=None, metric='net_income',
                          period_kind=None, period_end=None, duration_months=None)
    row['scope_ids'] = ['all-reports']
    row['source_cells'] = []
    row['expected'] = {'outcome': 'refuse', 'cell_ids': [],
        'negative_reason': 'Synthetic unavailable fiscal year control',
        'rationale_binding_ids': ['inventory:1']}
    cases, freeze, config = write_inputs(tmp_path, [row], cfg)
    return dict(source_dir=source_dir, source_registry=registry,
                cases=cases.resolve(), case_freeze=freeze.resolve(),
                experiment=config.resolve(), output_dir=(tmp_path / 'output').resolve())


def test_completed_annual_comparison_retains_query_and_readable_index(tmp_path):
    paths = comparison_inputs(tmp_path)
    receipt = comparison.run_comparison(**paths)
    index = comparison.read_completed_replay(paths['output_dir'] / 'index.json')
    completed = comparison.read_completed_replay(paths['output_dir'] / 'comparison.json')
    assert len(index['cells']) == 24
    assert receipt['period_kind_counts'] == {'fiscal_year': 24}
    assert completed['ordered_requests'] == 1
    assert completed['rows'][0]['parsed_request']['query'] == 'What was net income for fiscal 2035?'
    assert completed['human_gold_count'] == 0 and completed['financial_readiness'] == 'unestablished'


@pytest.mark.parametrize('input_name,loader_name', [
    ('experiment', 'load_experiment'), ('source_registry', 'load_source_manifest'),
])
def test_validated_input_changed_before_initial_capture_is_rejected(tmp_path, monkeypatch,
                                                                   input_name, loader_name):
    paths = comparison_inputs(tmp_path)
    load = getattr(comparison, loader_name)
    def changed(path):
        validated = load(path)
        path.write_bytes(path.read_bytes() + b'\n')
        return validated
    monkeypatch.setattr(comparison, loader_name, changed)
    with pytest.raises(ValueError):
        comparison.run_comparison(**paths)
    assert not (paths['output_dir'] / 'comparison.json').exists()


@pytest.mark.parametrize('marker', ['pending', 'failure'])
def test_index_completion_marker_added_after_selection_prevents_comparison(tmp_path, monkeypatch, marker):
    paths = comparison_inputs(tmp_path)
    assess = comparison.compare_source_candidate
    def changed(case, selection):
        index = paths['output_dir'] / 'index.json'
        assert comparison.read_completed_replay(index)['cells']
        index.with_name(index.name + '.' + marker + '.json').write_bytes(b'{}\n')
        return assess(case, selection)
    monkeypatch.setattr(comparison, 'compare_source_candidate', changed)
    with pytest.raises(ValueError):
        comparison.run_comparison(**paths)
    output = paths['output_dir'] / 'comparison.json'
    if output.exists():
        with pytest.raises(ValueError):
            comparison.read_completed_replay(output)


def test_completed_index_bytes_changed_before_initial_capture_are_rejected(tmp_path, monkeypatch):
    paths = comparison_inputs(tmp_path)
    write = comparison.write_completed
    def changed(path, value, final_check):
        write(path, value, final_check)
        if path.name == 'index.json':
            original = path.read_bytes()
            path.write_bytes(original + b'\n')
            assert comparison.read_completed_replay(path) == comparison.plain(value)
    monkeypatch.setattr(comparison, 'write_completed', changed)
    with pytest.raises(ValueError):
        comparison.run_comparison(**paths)
    assert not (paths['output_dir'] / 'comparison.json').exists()
