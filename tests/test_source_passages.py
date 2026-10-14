"""Native synthetic sources exercise a new field sidecar, not model quality."""
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import pytest

from nvidia_rag.structured_income import IncomeIndex
from nvidia_rag.statement_startup import load_statement_snapshot
from test_statement_startup import Inputs


def module():
    assert importlib.util.find_spec('nvidia_rag.source_passages') is not None, 'typed source passage factory is missing'
    return importlib.import_module('nvidia_rag.source_passages')


@pytest.fixture
def source(tmp_path):
    inputs = Inputs(tmp_path)
    return inputs, inputs.load()


def test_all_native_cells_become_immutable_source_passages(source):
    inputs, snapshot = source
    records = module().derive_source_passages(snapshot)
    assert type(records) is tuple and len(records) == len(snapshot.index.cells) == 60
    assert tuple(record.cell for record in records) == snapshot.index.cells
    assert len({record.chunk.chunk_id for record in records}) == 60
    for record in records:
        cell = record.cell
        spans = dict(cell.bindings)
        assert record.chunk.document_id == cell.document_id
        assert record.chunk.page_number == cell.physical_page
        assert record.chunk.amount_bbox == spans['value'].bbox
        assert record.chunk.metric_bbox == spans['row'].bbox
        assert record.chunk.period_bbox == spans['date'].bbox
        assert cell.value_literal in record.chunk.text and cell.row_literal in record.chunk.text
        assert cell.period_end in record.chunk.text
        assert 'USD' not in record.chunk.text
        with pytest.raises((FrozenInstanceError, AttributeError)):
            record.cell = snapshot.index.cells[0]


def test_all_unit_kinds_and_split_annual_dates_preserve_semantics(source):
    records = module().derive_source_passages(source[1])
    monetary = next(record for record in records if record.cell.unit_kind == 'monetary')
    eps = next(record for record in records if record.cell.unit_kind == 'per_share')
    count = next(record for record in records if record.cell.unit_kind == 'count')
    assert 'Source scale: 1000000' in monetary.chunk.text and 'ISO currency unverified' in monetary.chunk.text
    assert 'Source scale: 1;' in eps.chunk.text and 'Net income per share:' in eps.chunk.text
    assert 'Source scale: 1000000' in count.chunk.text and 'Currency: not applicable' in count.chunk.text
    assert 'Weighted average shares' in count.chunk.text and count.cell.currency_literal is None
    annual = next(record for record in records if record.cell.fiscal_quarter is None)
    assert annual.cell.duration_months is None
    assert 'Year Ended' in annual.chunk.text and '12 months' not in annual.chunk.text
    assert 'date_year' in dict(annual.cell.bindings)
    assert '16,599' not in count.chunk.text  # No exposed development target influences construction.


def test_cell_ids_and_new_passage_ids_are_distinct_source_derived_identities(source):
    records = module().derive_source_passages(source[1])
    for record in records:
        payload = asdict(record.cell)
        payload.pop('cell_id')
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
        assert sha256(encoded).hexdigest() == record.cell.cell_id
        assert record.chunk.chunk_id.startswith(record.cell.document_id + ':p' + str(record.cell.physical_page) + ':source-cell:')
        assert record.chunk.chunk_id != record.cell.cell_id


def test_negative_and_decimal_native_values_remain_exact_strings(source):
    records=module().derive_source_passages(source[1])
    negative=next(record for record in records if record.cell.amount == '-17')
    decimal=next(record for record in records if record.cell.amount == '1.27')
    assert negative.cell.value_literal == '(17)' and 'Printed value: (17)' in negative.chunk.text
    assert decimal.cell.value_literal == '1.27' and 'Printed value: 1.27' in decimal.chunk.text


@pytest.mark.parametrize('change', ['unowned-index','cell-id','value','unit','duplicate-cell','count-currency','duplicate-report','bool-column'])
def test_invalid_or_substituted_cells_refuse_before_output(source, change, tmp_path):
    inputs, snapshot = source
    if change == 'unowned-index':
        index = IncomeIndex(snapshot.index.reports, snapshot.index.cells, snapshot.index.omissions)
    else:
        # Reusing the original factory token is not enough to legitimize an
        # accidentally replaced source field; native replay must disagree.
        index = snapshot.index
        cells = list(index.cells)
        if change == 'duplicate-cell':
            cells.append(cells[0])
        elif change == 'duplicate-report':
            object.__setattr__(index, 'reports', index.reports + (index.reports[0],))
        else:
            at = next((i for i,c in enumerate(cells) if c.unit_kind == 'count'), 0) if change == 'count-currency' else 0
            fields = {'cell-id': {'cell_id':'0'*64}, 'value': {'amount':'999999'},
                'unit': {'scale':'1'}, 'count-currency': {'currency_literal':'$'},
                'bool-column': {'column_number':True}}[change]
            cells[at] = replace(cells[at], **fields)
        object.__setattr__(index, 'cells', tuple(cells))
    with pytest.raises(ValueError):
        module().build_source_passage_artifact(snapshot=replace(snapshot,index=index), output_dir=tmp_path/'new-sidecar')
    assert not (tmp_path/'new-sidecar').exists()


def test_artifact_round_trip_replays_native_sources_and_keeps_old_inputs(source, tmp_path):
    inputs, snapshot = source
    before = {identity.path: identity.path.read_bytes() for identity in snapshot.input_identities}
    directory = tmp_path/'field-sidecar'
    receipt = module().build_source_passage_artifact(snapshot=snapshot, output_dir=directory)
    assert receipt['status'] == 'complete' and receipt['model_runs'] == receipt['embedding_runs'] == 0
    assert set(path.name for path in directory.iterdir()) == {'passages.jsonl','manifest.json','build-receipt.json'}
    result = module().load_source_passage_artifact(snapshot=inputs.load(), artifact_dir=directory)
    assert result == module().derive_source_passages(snapshot)
    assert all(path.read_bytes() == body for path,body in before.items())
    manifest = json.loads((directory/'manifest.json').read_bytes())
    assert manifest['counts'] == {'reports':2, 'passages':60, 'omissions':0,
        'unit_kinds':{'count':14,'monetary':32,'per_share':14}}
    assert manifest['serializer_profile'] == module().SERIALIZER_PROFILE
    assert tuple(map(tuple,manifest['reports'])) == snapshot.index.reports


@pytest.mark.parametrize('change', ['pending','failure','unknown-file','duplicate-key','invalid-utf8','nan',
                                  'cell-value','chunk-text','chunk-bbox','cell-kind','receipt-count','manifest-identity'])
def test_reader_refuses_tampered_or_partial_artifact_even_when_rehashed(source,tmp_path,change):
    inputs,snapshot = source
    directory=tmp_path/'field-sidecar'
    module().build_source_passage_artifact(snapshot=snapshot,output_dir=directory)
    target=directory/'passages.jsonl'
    if change in ('pending','failure','unknown-file'):
        (directory/({'pending':'pending.json','failure':'failure.json','unknown-file':'unreviewed.json'}[change])).write_bytes(b'{}')
    elif change == 'duplicate-key':
        target.write_bytes(target.read_bytes().replace(b'"schema_version":1',b'"schema_version":1,"schema_version":1',1))
    elif change == 'invalid-utf8':
        target.write_bytes(b'\xff')
    elif change == 'nan':
        target.write_bytes(target.read_bytes().replace(b'"schema_version":1',b'"schema_version":NaN',1))
    elif change in ('cell-value','chunk-text','chunk-bbox','cell-kind'):
        rows=[json.loads(line) for line in target.read_bytes().splitlines()]
        if change == 'cell-value': rows[0]['cell']['amount']='999999'
        elif change == 'cell-kind': rows[0]['cell']['unit_kind']='count'
        elif change == 'chunk-text': rows[0]['chunk']['text']='fabricated 999999 USD'
        else: rows[0]['chunk']['amount_bbox']=[0,0,1,1]
        target.write_bytes(b''.join((json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode() for row in rows))
        # Rehashing a tampered data body is not source replay or authority.
        manifest=json.loads((directory/'manifest.json').read_bytes())
        manifest['passages_sha256']=sha256(target.read_bytes()).hexdigest()
        (directory/'manifest.json').write_bytes((json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n').encode())
    elif change == 'receipt-count':
        path=directory/'build-receipt.json'
        value=json.loads(path.read_bytes()); value['passage_count']=999
        path.write_bytes(json.dumps(value).encode())
    else:
        path=directory/'manifest.json'
        value=json.loads(path.read_bytes()); value['input_identities'][0]['sha256']='0'*64
        path.write_bytes(json.dumps(value).encode())
    with pytest.raises(ValueError):
        module().load_source_passage_artifact(snapshot=inputs.load(),artifact_dir=directory)


def test_reused_or_input_overlapping_output_is_refused_without_modification(source,tmp_path):
    inputs,snapshot=source
    directory=tmp_path/'field-sidecar'
    module().build_source_passage_artifact(snapshot=snapshot,output_dir=directory)
    before={path.name:path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(ValueError): module().build_source_passage_artifact(snapshot=snapshot,output_dir=directory)
    assert before == {path.name:path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(ValueError):
        module().build_source_passage_artifact(snapshot=snapshot,output_dir=inputs.source_dir/'sidecar')
    assert not (inputs.source_dir/'sidecar').exists()


def test_write_failure_retains_refusal_guard_and_reader_refuses(source,tmp_path,monkeypatch):
    inputs,snapshot=source
    library=module()
    original=library._write_exclusive
    def failing(path,body):
        if path.name == 'manifest.json': raise OSError('synthetic partial writer failure')
        return original(path,body)
    monkeypatch.setattr(library,'_write_exclusive',failing)
    directory=tmp_path/'field-sidecar'
    with pytest.raises(OSError): library.build_source_passage_artifact(snapshot=snapshot,output_dir=directory)
    assert (directory/'pending.json').is_file() and (directory/'failure.json').is_file()
    with pytest.raises(ValueError): library.load_source_passage_artifact(snapshot=inputs.load(),artifact_dir=directory)


def test_source_change_during_build_refuses_completion(source,tmp_path,monkeypatch):
    inputs,snapshot=source
    library=module()
    original=library._write_exclusive
    def change(path,body):
        result=original(path,body)
        if path.name == 'passages.jsonl': inputs.declarations.write_bytes(inputs.declarations.read_bytes()+b'\n')
        return result
    monkeypatch.setattr(library,'_write_exclusive',change)
    directory=tmp_path/'field-sidecar'
    with pytest.raises(ValueError,match='changed'): library.build_source_passage_artifact(snapshot=snapshot,output_dir=directory)
    assert (directory/'pending.json').is_file() and (directory/'failure.json').is_file()


def test_late_closure_failure_with_unwritable_failure_receipt_stays_unloadable(source,tmp_path,monkeypatch):
    inputs,snapshot=source
    library=module()
    original_read=library._artifact_bodies
    original_write=library._write_exclusive
    calls=0
    def late_read(directory,**kwargs):
        nonlocal calls
        calls+=1
        if calls == 2: raise ValueError('synthetic final artifact closure failure')
        return original_read(directory,**kwargs)
    def deny_failure(path,body):
        if path.name == 'failure.json': raise OSError('synthetic failure receipt storage denial')
        return original_write(path,body)
    monkeypatch.setattr(library,'_artifact_bodies',late_read)
    monkeypatch.setattr(library,'_write_exclusive',deny_failure)
    directory=tmp_path/'field-sidecar'
    with pytest.raises(ValueError,match='closure'): library.build_source_passage_artifact(snapshot=snapshot,output_dir=directory)
    assert (directory/'pending.json').is_file()
    monkeypatch.setattr(library,'_artifact_bodies',original_read)
    with pytest.raises(ValueError): library.load_source_passage_artifact(snapshot=inputs.load(),artifact_dir=directory)


def test_source_directory_overlap_uses_resolved_dotdot_input_paths(source):
    inputs,_=source
    aliased=inputs.source_dir/'..'/'sources'
    snapshot=load_statement_snapshot(source_registry=inputs.registry,source_dir=aliased,
                                    fiscal_declarations=inputs.declarations)
    output=inputs.source_dir/'nested-field-sidecar'
    with pytest.raises(ValueError,match='overlap'):
        module().build_source_passage_artifact(snapshot=snapshot,output_dir=output)
    assert not output.exists()


def test_changed_pending_marker_is_not_removed_at_final_completion(source,tmp_path,monkeypatch):
    inputs,snapshot=source
    library=module()
    original=library._artifact_bodies
    calls=0
    def replace_marker(directory,**kwargs):
        nonlocal calls
        calls+=1
        result=original(directory,**kwargs)
        if calls == 2: (directory/'pending.json').write_bytes(b'{"status":"different-owner"}')
        return result
    monkeypatch.setattr(library,'_artifact_bodies',replace_marker)
    directory=tmp_path/'field-sidecar'
    with pytest.raises(ValueError,match='pending marker changed'):
        library.build_source_passage_artifact(snapshot=snapshot,output_dir=directory)
    assert (directory/'pending.json').read_bytes() == b'{"status":"different-owner"}'


def test_fresh_loader_rechecks_artifact_after_native_replay(source,tmp_path,monkeypatch):
    inputs,snapshot=source
    library=module()
    directory=tmp_path/'field-sidecar'
    library.build_source_passage_artifact(snapshot=snapshot,output_dir=directory)
    original=library.build_income_index
    def mutate_after_native(reports):
        result=original(reports)
        target=directory/'passages.jsonl'
        target.write_bytes(target.read_bytes()+b'\n')
        return result
    monkeypatch.setattr(library,'build_income_index',mutate_after_native)
    with pytest.raises(ValueError):
        library.load_source_passage_artifact(snapshot=inputs.load(),artifact_dir=directory)


def test_library_round_trip_accepts_valid_paired_optional_corpus(source,tmp_path):
    inputs,_=source
    directory=tmp_path/'raw-corpus'
    directory.mkdir()
    manifest,chunks=inputs.corpus_files()
    moved_manifest=directory/manifest.name
    moved_chunks=directory/chunks.name
    moved_manifest.write_bytes(manifest.read_bytes())
    moved_chunks.write_bytes(chunks.read_bytes())
    snapshot=inputs.load(corpus_manifest=moved_manifest,chunks_path=moved_chunks)
    output=tmp_path/'new-fields'
    module().build_source_passage_artifact(snapshot=snapshot,output_dir=output)
    assert module().load_source_passage_artifact(snapshot=snapshot,artifact_dir=output) == module().derive_source_passages(snapshot)


def test_output_is_separate_from_bound_optional_corpus_directory(source,tmp_path):
    inputs,_=source
    directory=tmp_path/'raw-corpus'
    directory.mkdir()
    manifest,chunks=inputs.corpus_files()
    copied_manifest=directory/manifest.name
    copied_chunks=directory/chunks.name
    copied_manifest.write_bytes(manifest.read_bytes())
    copied_chunks.write_bytes(chunks.read_bytes())
    snapshot=inputs.load(corpus_manifest=copied_manifest,chunks_path=copied_chunks)
    with pytest.raises(ValueError,match='overlap'):
        module().build_source_passage_artifact(snapshot=snapshot,output_dir=directory/'field-sidecar')
    assert not (directory/'field-sidecar').exists()


def test_snapshot_extra_corpus_bytes_must_still_replay_registration_binding(source,tmp_path):
    from nvidia_rag.statement_startup import InputIdentity
    inputs,_=source
    directory=tmp_path/'raw-corpus'
    directory.mkdir()
    manifest,chunks=inputs.corpus_files()
    value=json.loads(manifest.read_bytes())
    value['documents'][0]['source_sha256']='0'*64
    manifest.write_bytes(json.dumps(value).encode())
    # An accidentally assembled in-process snapshot with real byte identities
    # must not turn a broken corpus registration into a successful sidecar.
    snapshot=inputs.load()
    identities=tuple(InputIdentity(path,sha256(path.read_bytes()).hexdigest(),path.stat().st_size)
                     for path in (manifest,chunks))
    snapshot=replace(snapshot,input_identities=snapshot.input_identities[:2]+identities+snapshot.input_identities[2:])
    with pytest.raises(ValueError,match='corpus|source identity'):
        module().derive_source_passages(snapshot)


def test_cli_uses_package_snapshot_and_has_no_optional_model_imports(source,tmp_path,monkeypatch):
    inputs,_=source
    script=Path(__file__).resolve().parents[1]/'examples'/'build_source_passages.py'
    assert script.is_file(), 'source-only installed-package CLI is missing'
    monkeypatch.setattr(sys,'argv',[str(script),'--source-registry',str(inputs.registry),
        '--source-dir',str(inputs.source_dir),'--fiscal-declarations',str(inputs.declarations),
        '--output-dir',str(tmp_path/'cli-sidecar')])
    import builtins
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in {'sentence_transformers','transformers','uvicorn'}:
            pytest.fail('source-only CLI imported an optional model/server runtime')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    runpy.run_path(str(script),run_name='__main__')
    assert (tmp_path/'cli-sidecar'/'build-receipt.json').is_file()
