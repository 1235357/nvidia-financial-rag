"""Source-coordinate and corpus-coverage controls; no model or financial gold."""
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path

import pytest


def api():
    path = Path(__file__).resolve().parents[1] / 'acceptance/expanded_source_binding_replay.py'
    assert path.exists(), 'expanded source replay must be implemented'
    return importlib.import_module('acceptance.expanded_source_binding_replay')


def span(text, page_text, box, occurrence=0):
    positions = []
    start = 0
    while (at := page_text.find(text, start)) >= 0:
        positions.append(at)
        start = at + 1
    at = positions[occurrence]
    return {'start': at, 'end': at + len(text), 'text': text, 'bbox': box}


def source_fixture():
    text = 'Income\nIn millions\nThree Months Ended\nJul 28, 2024 Jul 28, 2024\nRevenue\n$ 30,040 30,040\n'
    boxes = {'Income': [[10, 10, 90, 20]], 'In millions': [[10, 25, 90, 35]],
             'Three Months Ended': [[100, 40, 240, 50]],
             'Jul 28, 2024': [[110, 55, 150, 65], [190, 55, 230, 65]],
             'Revenue': [[10, 75, 70, 85]], '$': [[105, 75, 110, 85]],
             '30,040': [[120, 75, 150, 85], [200, 75, 230, 85]]}
    names = {'statement': 'Income', 'unit': 'In millions', 'duration': 'Three Months Ended',
             'date': 'Jul 28, 2024', 'row': 'Revenue', 'currency': '$', 'value': '30,040'}
    spans = {name: span(value, text, boxes[value][0]) for name, value in names.items()}
    profile = {'library': 'PyMuPDF', 'version': '1.28.2', 'mode': 'text', 'sort': True,
               'normalization': 'none', 'text_encoding': 'utf-8',
               'offset_unit': 'unicode_code_points', 'interval': 'half_open'}
    cell = {'cell_id': 'cell:' + 'a' * 64, 'document_id': 'doc', 'pdf_sha256': 'b' * 64,
            'physical_page': 3, 'period_kind': 'duration', 'period_end': '2024-07-28',
            'duration_months': 3, 'column_role': 'current', 'unit_kind': 'monetary',
            'currency_literal': '$', 'bindings': {'page_text_sha256': sha256(text.encode()).hexdigest(),
            'page_text_chars': len(text), 'extraction_profile': profile, 'spans': spans,
            'association_receipt_sha256': None}}
    columns = []
    for i in range(2):
        columns.append({'column_id': f'c{i}', 'column_role': 'current' if i == 0 else 'prior_comparison',
                        'period_end': '2024-07-28', 'duration_months': 3,
                        'date_span': span('Jul 28, 2024', text, boxes['Jul 28, 2024'][i], i),
                        'duration_span': deepcopy(spans['duration']),
                        'x_range': [100 + i * 80, 160 + i * 80]})
    packet = {'schema_version': 2, 'cell_id': cell['cell_id'], 'pdf_sha256': cell['pdf_sha256'],
              'physical_page': 3, 'page_text_sha256': cell['bindings']['page_text_sha256'],
              'extraction_profile': profile, 'selected_column_id': 'c0',
              'spans': deepcopy(spans), 'columns': columns, 'row_band': [74, 86],
              'review_status': 'assistant-source-reviewed-candidate', 'human_acceptance': False}
    return text, boxes, cell, packet


def test_repeated_date_and_amount_bind_to_the_selected_column():
    m = api()
    text, boxes, cell, packet = source_fixture()
    result = m.verify_cell_geometry(cell, packet, text, boxes)
    assert result['source_binding_passed'] is True
    assert result['human_gold'] is False


@pytest.mark.parametrize('mutation', ['date', 'amount', 'column', 'profile', 'page', 'unit', 'currency'])
def test_wrong_source_tuple_refuses(mutation):
    m = api()
    text, boxes, cell, packet = source_fixture()
    if mutation == 'date':
        cell['bindings']['spans']['date'] = span('Jul 28, 2024', text, boxes['Jul 28, 2024'][1], 1)
        packet['spans'] = deepcopy(cell['bindings']['spans'])
    elif mutation == 'amount':
        cell['bindings']['spans']['value'] = span('30,040', text, boxes['30,040'][1], 1)
        packet['spans'] = deepcopy(cell['bindings']['spans'])
    elif mutation == 'column':
        packet['selected_column_id'] = 'c1'
    elif mutation == 'profile':
        packet['extraction_profile'] = {**packet['extraction_profile'], 'sort': False}
    elif mutation == 'page':
        packet['physical_page'] = 4
    else:
        cell['bindings']['spans'][mutation] = None
        packet['spans'][mutation] = None
    with pytest.raises(ValueError):
        m.verify_cell_geometry(cell, packet, text, boxes)


def test_bbox_cannot_select_another_identical_occurrence():
    m = api()
    text, boxes, cell, packet = source_fixture()
    cell['bindings']['spans']['value']['bbox'] = boxes['30,040'][1]
    packet['spans'] = deepcopy(cell['bindings']['spans'])
    with pytest.raises(ValueError, match='occurrence'):
        m.verify_cell_geometry(cell, packet, text, boxes)


def test_split_header_coverage_uses_exact_same_page_chunk_union():
    m = api()
    text = 'Income\nDate\nRevenue\n30,040'
    required = {'header': [0, 11], 'row': [12, 19], 'value': [20, 26]}
    mappings = m.map_chunk_ranges(text, [
        {'chunk_id': 'head', 'text': text[:8]}, {'chunk_id': 'tail', 'text': text[8:]}])
    assert m.cover_ranges(required, mappings, ['head', 'tail']) == 'passed'
    assert m.cover_ranges(required, mappings, ['tail']) == 'failed'


def test_ambiguous_chunk_mapping_remains_not_evaluated():
    m = api()
    mappings = m.map_chunk_ranges('Revenue 10 Revenue 10', [{'chunk_id': 'same', 'text': 'Revenue 10'}])
    assert mappings[0]['status'] == 'ambiguous'
    assert m.cover_ranges({'row': [0, 7]}, mappings, ['same']) == 'not_evaluated'


def test_unselected_ambiguous_chunk_does_not_corrupt_valid_coverage():
    m = api()
    mappings = m.map_chunk_ranges('AA BB AA', [
        {'chunk_id': 'all', 'text': 'AA BB AA'}, {'chunk_id': 'ambiguous', 'text': 'AA'}])
    assert m.cover_ranges({'row': [0, 2]}, mappings, ['all']) == 'passed'


def test_unknown_accepted_chunk_is_refused():
    m = api()
    with pytest.raises(ValueError, match='unknown'):
        m.cover_ranges({'row': [0, 1]}, [], ['missing'])


def test_config_binding_avoids_only_the_replay_hash_cycle():
    m = api()
    cfg = {'input_sha256': {'source_replay': 'a', 'cases': 'b'}, 'top_k': 5}
    edited = deepcopy(cfg)
    edited['input_sha256']['source_replay'] = 'other'
    assert m.experiment_binding_sha256(cfg) == m.experiment_binding_sha256(edited)
    edited['top_k'] = 4
    assert m.experiment_binding_sha256(cfg) != m.experiment_binding_sha256(edited)


def test_sorted_review_offsets_are_remapped_to_unsorted_corpus():
    m = api()
    text, boxes, cell, packet = source_fixture()
    corpus = 'Revenue\n$ 30,040 30,040\nIncome\nIn millions\nThree Months Ended\nJul 28, 2024 Jul 28, 2024'
    ranges = m.map_cell_corpus_spans(cell, corpus, boxes)
    assert ranges['status'] == 'complete'
    assert ranges['ranges']['row'] == [0, 7]
    assert ranges['ranges']['value'] != [cell['bindings']['spans']['value']['start'],
                                        cell['bindings']['spans']['value']['end']]


def test_corpus_occurrence_geometry_mismatch_is_not_guessed():
    m = api()
    text, boxes, cell, packet = source_fixture()
    boxes['30,040'].append([300, 75, 330, 85])
    assert m.map_cell_corpus_spans(cell, text, boxes)['status'] == 'not_evaluated'


def test_coverage_does_not_use_accepted_chunks_from_another_page():
    m = api()
    receipt = {'ordered_case_ids': ['case'], 'chunk_ids': ['right', 'other'],
               'cells': [{'case_id': 'case', 'cell_id': 'cell', 'source_binding_passed': True,
                          'corpus_spans': {'status': 'complete', 'ranges': {'value': [0, 6]}},
                          'chunk_mappings': [{'chunk_id': 'right', 'status': 'unique', 'range': [0, 6]}]}]}
    assert m.evaluate_candidate_coverage(receipt, 'case', ['other'])['joint_source_candidate_coverage'] == 'failed'
    assert m.evaluate_candidate_coverage(receipt, 'case', ['right'])['joint_source_candidate_coverage'] == 'passed'


def test_incomplete_source_cannot_acquire_coverage_pass():
    m = api()
    receipt = {'ordered_case_ids': ['case'], 'chunk_ids': ['right'],
               'cells': [{'case_id': 'case', 'cell_id': 'cell', 'source_binding_passed': False,
                          'corpus_spans': {'status': 'not_evaluated', 'ranges': {}},
                          'chunk_mappings': []}]}
    assert m.evaluate_candidate_coverage(receipt, 'case', ['right'])['joint_source_candidate_coverage'] == 'not_evaluated'


def test_missing_case_coverage_is_refused():
    m = api()
    with pytest.raises(ValueError, match='unknown'):
        m.evaluate_candidate_coverage({'ordered_case_ids': [], 'chunk_ids': []}, 'missing', [])


def test_source_body_changed_before_parser_never_opens_it(tmp_path, monkeypatch):
    m = api()
    source = tmp_path / 'source.pdf'
    source.write_bytes(b'changed source')
    opened = []
    monkeypatch.setattr(m, 'read_source_page', lambda *args: opened.append(args))
    with pytest.raises(ValueError, match='SHA'):
        m.checked_source_body(source, 'a' * 64)
    assert opened == []


def test_pending_replay_cannot_be_loaded(tmp_path):
    m = api()
    path = tmp_path / 'replay.json'
    path.write_text('{}', encoding='utf-8')
    path.with_name('replay.json.pending.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='pending'):
        m.read_completed_replay(path)


def test_receipt_write_is_exclusive_and_does_not_replace_old_bytes(tmp_path):
    m = api()
    path = tmp_path / 'replay.json'
    path.write_bytes(b'original')
    with pytest.raises(ValueError, match='exists'):
        m.write_replay_exclusive(path, {'status': 'complete'}, lambda: None)
    assert path.read_bytes() == b'original'


def test_failed_final_input_check_leaves_pending_receipt(tmp_path):
    m = api()
    path = tmp_path / 'replay.json'
    def changed():
        raise ValueError('input identity changed')
    with pytest.raises(ValueError, match='identity'):
        m.write_replay_exclusive(path, {'status': 'complete'}, changed)
    assert path.with_name('replay.json.pending.json').exists()
    with pytest.raises(ValueError, match='pending'):
        m.read_completed_replay(path)


def test_successful_receipt_has_no_pending_marker(tmp_path):
    m = api()
    path = tmp_path / 'replay.json'
    m.write_replay_exclusive(path, {'status': 'complete'}, lambda: None)
    assert m.read_completed_replay(path) == {'status': 'complete'}


def replay_inputs(tmp_path, monkeypatch):
    import test_expanded_cases as contracts
    from types import SimpleNamespace
    m = api()
    text, boxes, small_cell, packet = source_fixture()
    row, cfg = contracts.case(), contracts.experiment()
    source_dir, corpus_dir, binding_dir = [tmp_path / name for name in ('sources', 'corpus', 'bindings')]
    for directory in (source_dir, corpus_dir, binding_dir):
        directory.mkdir()
    pdf_body = b'owned PDF parser boundary fixture'
    digest = sha256(pdf_body).hexdigest()
    (source_dir / 'report.pdf').write_bytes(pdf_body)
    source = row['source_cells'][0]
    source.update(document_id='doc', pdf_sha256=digest, period_end='2024-07-28',
                  column_role='current', amount='30040', value_text='30,040',
                  statement='Income', unit_literal='In millions',
                  bindings=deepcopy(small_cell['bindings']))
    source['bindings']['status'] = 'candidate-complete'
    row['request'].update(report_document_id='doc', period_end='2024-07-28')
    cfg['documents'][0].update(document_id='doc', pdf_sha256=digest)
    cfg['scopes'][1]['document_ids'] = ['doc']
    source['cell_id'] = contracts.api().source_cell_id(source)
    row['expected']['cell_ids'] = [source['cell_id']]
    packet.update(cell_id=source['cell_id'], pdf_sha256=digest)
    packet_body = m.canonical_bytes(packet)
    packet_hash = sha256(packet_body).hexdigest()
    source['bindings']['association_receipt_sha256'] = packet_hash
    (binding_dir / (packet_hash + '.json')).write_bytes(packet_body)
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'schema_version': 1, 'sources': [{
        'document_id': 'doc', 'filename': 'report.pdf', 'sha256': digest,
        'url': source['issuer_url'], 'label': 'Synthetic source'}]}), encoding='utf-8')
    chunks_body = (json.dumps({'chunk_id': 'full', 'document_id': 'doc',
                              'page_number': 3, 'text': text}) + '\n').encode()
    (corpus_dir / 'chunks.jsonl').write_bytes(chunks_body)
    manifest = {'document_count': 1, 'chunk_count': 1, 'documents': [{
        'document_id': 'doc', 'page_count': 45}]}
    (corpus_dir / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (corpus_dir / 'build-receipt.json').write_text(json.dumps({
        'runtime': {'PyMuPDF': '1.28.2'}, 'profile': {'name': 'plain-word-safe-400-0-v1',
        'extraction_mode': 'plain', 'strategy': 'word_safe', 'max_tokens': 400,
        'overlap_tokens': 0, 'encoding_name': 'cl100k_base'}}), encoding='utf-8')
    for key, path in [('registry', registry), ('corpus_manifest', corpus_dir / 'manifest.json'),
                      ('corpus_chunks', corpus_dir / 'chunks.jsonl')]:
        cfg['input_sha256'][key] = sha256(path.read_bytes()).hexdigest()
    cfg['chunk_count'] = 1
    paths = contracts.write_inputs(tmp_path, [row], cfg)
    monkeypatch.setattr(m, 'read_source_page', lambda *args: (text, text, boxes))
    monkeypatch.setattr(m, 'load_corpus', lambda **args: SimpleNamespace(
        chunks=(SimpleNamespace(chunk_id='full', document_id='doc', page_number=3, text=text),),
        manifest=manifest))
    kwargs = dict(source_dir=source_dir, source_registry=registry, cases=paths[0],
                  case_freeze=paths[1], artifact_dir=corpus_dir, experiment=paths[2], binding_dir=binding_dir)
    return kwargs


def test_complete_replay_closes_inputs_and_preserves_candidate_status(tmp_path, monkeypatch):
    m = api()
    kwargs = replay_inputs(tmp_path, monkeypatch)
    receipt = m.replay_sources(**kwargs)
    assert receipt['status'] == 'complete'
    assert receipt['human_gold_count'] == receipt['generation_runs'] == 0
    assert receipt['cells'][0]['source_binding_passed'] is True
    assert m.evaluate_candidate_coverage(receipt, 'positive', ['full'])['joint_source_candidate_coverage'] == 'passed'


def test_changed_pdf_refuses_before_page_parser(tmp_path, monkeypatch):
    m = api()
    kwargs = replay_inputs(tmp_path, monkeypatch)
    (kwargs['source_dir'] / 'report.pdf').write_bytes(b'changed')
    opened = []
    monkeypatch.setattr(m, 'read_source_page', lambda *args: opened.append(args))
    with pytest.raises(ValueError, match='SHA'):
        m.replay_sources(**kwargs)
    assert opened == []


def test_source_mutation_during_replay_prevents_completion(tmp_path, monkeypatch):
    m = api()
    kwargs = replay_inputs(tmp_path, monkeypatch)
    text, boxes, _, _ = source_fixture()
    def changed(*args):
        (kwargs['source_dir'] / 'report.pdf').write_bytes(b'changed during replay')
        return text, text, boxes
    monkeypatch.setattr(m, 'read_source_page', changed)
    with pytest.raises(ValueError, match='changed'):
        m.replay_sources(**kwargs)


def test_overlapping_columns_cannot_bind_the_other_column_amount():
    m = api()
    text, boxes, cell, packet = source_fixture()
    packet['columns'][0]['x_range'] = [100, 240]
    cell['bindings']['spans']['value'] = span('30,040', text, boxes['30,040'][1], 1)
    packet['spans'] = deepcopy(cell['bindings']['spans'])
    with pytest.raises(ValueError, match='overlap'):
        m.verify_cell_geometry(cell, packet, text, boxes)


def test_huge_row_band_cannot_join_amount_to_another_row():
    m = api()
    text, boxes, cell, packet = source_fixture()
    boxes['Revenue'][0] = [10, 200, 70, 210]
    cell['bindings']['spans']['row']['bbox'] = boxes['Revenue'][0]
    packet['spans'] = deepcopy(cell['bindings']['spans'])
    packet['row_band'] = [74, 211]
    with pytest.raises(ValueError, match='row'):
        m.verify_cell_geometry(cell, packet, text, boxes)


def completed_inputs(tmp_path, monkeypatch, edit=None):
    m = api()
    kwargs = replay_inputs(tmp_path, monkeypatch)
    receipt = m.replay_sources(**kwargs)
    if edit:
        edit(receipt)
    path = tmp_path / 'replay.json'
    path.write_bytes(m.canonical_bytes(receipt) + b'\n')
    cfg = json.loads(kwargs['experiment'].read_bytes())
    cfg['input_sha256']['source_replay'] = sha256(path.read_bytes()).hexdigest()
    final = tmp_path / 'final.json'
    final.write_text(json.dumps(cfg), encoding='utf-8')
    from acceptance.expanded_cases import load_case_bundle, load_experiment
    bundle = load_case_bundle(kwargs['cases'], kwargs['case_freeze'], load_experiment(final))
    return path, bundle, kwargs


def test_completed_replay_loader_closes_final_config_and_actual_chunks(tmp_path, monkeypatch):
    m = api()
    path, bundle, kwargs = completed_inputs(tmp_path, monkeypatch)
    loaded = m.load_source_replay(path, bundle, kwargs['artifact_dir'], kwargs['source_registry'])
    assert loaded['cells'][0]['source_binding_passed'] is True


@pytest.mark.parametrize('marker', ['pending', 'failure'])
def test_completed_replay_loader_rejects_guard_added_during_corpus_load(tmp_path, monkeypatch, marker):
    m = api()
    path, bundle, kwargs = completed_inputs(tmp_path, monkeypatch)
    original = m.load_corpus
    def guarded(**arguments):
        m.guard_path(path, marker).write_text('{}', encoding='utf-8')
        return original(**arguments)
    monkeypatch.setattr(m, 'load_corpus', guarded)
    with pytest.raises(ValueError, match='guard'):
        m.load_source_replay(path, bundle, kwargs['artifact_dir'], kwargs['source_registry'])


@pytest.mark.parametrize('change', ['missing-cell', 'wrong-chunk', 'wrong-role', 'wrong-range', 'fake-gold'])
def test_hashed_but_structurally_invalid_receipt_is_rejected(tmp_path, monkeypatch, change):
    m = api()
    def edit(receipt):
        if change == 'missing-cell':
            receipt['cells'] = []
        elif change == 'wrong-chunk':
            receipt['cells'][0]['chunk_mappings'][0]['chunk_id'] = 'different'
        elif change == 'wrong-role':
            receipt['cells'][0]['essential_roles'] = ['value']
        elif change == 'wrong-range':
            receipt['cells'][0]['corpus_spans']['ranges']['value'] = [True, 900000]
        else:
            receipt['human_gold_count'] = 1
    path, bundle, kwargs = completed_inputs(tmp_path, monkeypatch, edit)
    with pytest.raises(ValueError):
        m.load_source_replay(path, bundle, kwargs['artifact_dir'], kwargs['source_registry'])


def test_registry_absence_rationale_checks_actual_registered_ids():
    m = api()
    cfg = {'input_sha256': {'registry': 'a' * 64}, 'documents': [{'document_id': 'present'}]}
    body = {'registry_sha256': 'a' * 64, 'missing_document_id': 'missing'}
    assert m.verify_rationale('registry-absence', body, cfg, {}) == 'candidate-bound'
    body['missing_document_id'] = 'present'
    with pytest.raises(ValueError):
        m.verify_rationale('registry-absence', body, cfg, {})


def test_source_inventory_rationale_requires_replayed_column_identity():
    m = api()
    _, _, _, packet = source_fixture()
    body = {'document_id': 'doc', 'pdf_sha256': 'b' * 64, 'physical_page': 3,
            'columns': packet['columns'], 'human_acceptance': False}
    cfg = {'documents': [{'document_id': 'doc', 'pdf_sha256': 'b' * 64}]}
    verified = {('doc', 3): packet['columns']}
    assert m.verify_rationale('source-column-inventory', body, cfg, verified) == 'candidate-bound'
    body['columns'][0]['period_end'] = '2024-07-29'
    # Distinct independent inventory: a shared mutable object would hide the edit.
    _, _, _, unchanged = source_fixture()
    with pytest.raises(ValueError):
        m.verify_rationale('source-column-inventory', body, cfg, {('doc', 3): unchanged['columns']})


def test_currency_capability_rationale_cannot_claim_fx_or_new_currency():
    m = api()
    cfg = {'supported_currencies': ['USD']}
    assert m.verify_rationale('experiment-capability', {'supported_currencies': ['USD'], 'fx_conversion': False}, cfg, {}) == 'candidate-bound'
    with pytest.raises(ValueError):
        m.verify_rationale('experiment-capability', {'supported_currencies': ['EUR'], 'fx_conversion': False}, cfg, {})


def test_experiment_changed_between_validation_and_snapshot_is_refused(tmp_path, monkeypatch):
    m = api()
    kwargs = replay_inputs(tmp_path, monkeypatch)
    original = m.load_experiment
    def changed(path):
        validated = original(path)
        data = json.loads(path.read_bytes())
        data['model']['instruction'] = 'Different instruction after validation: '
        path.write_text(json.dumps(data), encoding='utf-8')
        return validated
    monkeypatch.setattr(m, 'load_experiment', changed)
    with pytest.raises(ValueError, match='experiment'):
        m.replay_sources(**kwargs)
