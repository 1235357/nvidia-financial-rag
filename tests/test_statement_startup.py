"""Startup owns reviewed declarations and native source bytes, not financial gold."""
from dataclasses import replace
from hashlib import sha256
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from nvidia_rag.sources import load_source_manifest
from nvidia_rag.structured_income import build_income_index
from test_annual_income import annual_pdf
from test_structured_income import source_pdf


def startup():
    assert importlib.util.find_spec('nvidia_rag.statement_startup') is not None, 'installed statement startup factory is missing'
    module = importlib.import_module('nvidia_rag.statement_startup')
    assert callable(getattr(module, 'load_statement_snapshot', None)), 'source-bound snapshot interface is missing'
    return module


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n').encode())


class Inputs:
    def __init__(self, root):
        self.root = root
        self.source_dir = root / 'sources'
        self.source_dir.mkdir()
        annual = annual_pdf(self.source_dir, split_date=True)
        quarter = source_pdf(self.source_dir)
        self.reports = (annual, quarter)
        self.rows = [dict(document_id=r.document_id, filename=r.pdf_path.name, sha256=r.pdf_sha256,
                          label='Generated source ' + r.document_id,
                          url='https://s201.q4cdn.com/141608511/files/test/' + r.pdf_path.name) for r in self.reports]
        self.registry = root / 'registry.json'
        self.declarations = root / 'fiscal-declarations.json'
        self.payload = {'schema_version': 1, 'source_registry_sha256': '', 'reports': [
            dict(document_id=r.document_id, pdf_sha256=r.pdf_sha256, fiscal_year=r.fiscal_year,
                 fiscal_quarter=r.fiscal_quarter, reviewed_by='Synthetic fixture author',
                 source_locator='Native generated report table on physical page 2') for r in self.reports]}
        self.sync()

    def sync(self):
        write_json(self.registry, {'schema_version': 1, 'sources': self.rows})
        self.payload['source_registry_sha256'] = digest(self.registry)
        write_json(self.declarations, self.payload)

    def load(self, **kwargs):
        return startup().load_statement_snapshot(source_registry=self.registry, source_dir=self.source_dir,
            fiscal_declarations=self.declarations, **kwargs)

    def corpus_files(self):
        chunks = self.root / 'chunks.jsonl'
        records = [dict(chunk_id=r.document_id + ':p2:w:0123456789abcdef', document_id=r.document_id,
                        page_number=2, text='Generated native report evidence') for r in self.reports]
        chunks.write_bytes(''.join(json.dumps(r) + '\n' for r in records).encode())
        self.manifest = self.root / 'manifest.json'
        self.manifest_value = {'schema_version': 1, 'chunks_sha256': digest(chunks),
            'document_count': 2, 'chunk_count': 2, 'documents': [
                dict(document_id=r['document_id'], document_label=r['label'], source_url=r['url'],
                     source_filename=r['filename'], source_sha256=r['sha256'], chunk_count=1)
                for r in self.rows]}
        write_json(self.manifest, self.manifest_value)
        return self.manifest, chunks


def no_parse(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid startup inputs reached native PDF parsing')
    monkeypatch.setattr(startup(), 'build_income_index', forbidden)


def test_native_annual_and_quarter_index_matches_direct_source_build(tmp_path):
    inputs = Inputs(tmp_path)
    snapshot = inputs.load()
    assert snapshot.index == build_income_index(inputs.reports)
    assert len(snapshot.index.cells) == 60
    assert next(c.amount for c in snapshot.index.cells if c.row_literal == 'Net income'
        and c.document_id == inputs.reports[0].document_id and c.column_role == 'current') == '8650'
    assert next(c.amount for c in snapshot.index.cells if c.row_literal == 'Net income'
        and c.document_id == inputs.reports[1].document_id and c.column_role == 'current'
        and c.duration_months == 3) == '2860'
    assert snapshot.verify_unchanged() is None


def test_scope_builds_only_selected_declared_sources(tmp_path):
    inputs = Inputs(tmp_path)
    annual, quarter = inputs.reports
    annual.pdf_path.unlink()  # Unselected source must not be read.
    snapshot = inputs.load(document_ids=frozenset({quarter.document_id}))
    assert snapshot.index == build_income_index((quarter,))
    assert {r[0] for r in snapshot.index.reports} == {quarter.document_id}


@pytest.mark.parametrize('scope', [frozenset(), frozenset({'unknown'}), {'nvidia-fy2025-q2'}, frozenset({True})])
def test_invalid_scope_refused_before_extraction(tmp_path, monkeypatch, scope):
    inputs = Inputs(tmp_path)
    no_parse(monkeypatch)
    with pytest.raises(ValueError):
        inputs.load(document_ids=scope)


def test_empty_declared_scope_refused_instead_of_unconfigured_fallback(tmp_path, monkeypatch):
    inputs = Inputs(tmp_path)
    inputs.payload['reports'] = inputs.payload['reports'][:1]
    inputs.sync()
    no_parse(monkeypatch)
    with pytest.raises(ValueError):
        inputs.load(document_ids=frozenset({inputs.reports[1].document_id}))


@pytest.mark.parametrize('field,value', [
    ('schema_version', True), ('schema_version', 2), ('source_registry_sha256', '0'*64),
    ('source_registry_sha256', 4), ('reports', []), ('reports', {}), ('unexpected', 'must refuse'),
])
def test_declaration_envelope_strict(tmp_path, monkeypatch, field, value):
    inputs = Inputs(tmp_path)
    inputs.payload[field] = value
    write_json(inputs.declarations, inputs.payload)
    no_parse(monkeypatch)
    with pytest.raises(ValueError):
        inputs.load()


@pytest.mark.parametrize('field,value', [
    ('document_id', 'unregistered'), ('document_id', ' nvidia-fy2025-annual'),
    ('pdf_sha256', '0'*64), ('pdf_sha256', 'A'*64), ('pdf_sha256', None),
    ('fiscal_year', True), ('fiscal_year', 1899), ('fiscal_year', 10000), ('fiscal_year', '2025'),
    ('fiscal_quarter', True), ('fiscal_quarter', 0), ('fiscal_quarter', 5), ('fiscal_quarter', '2'),
    ('reviewed_by', ''), ('reviewed_by', ' unreviewed whitespace '), ('reviewed_by', 5),
    ('source_locator', ''), ('source_locator', []), ('source_locator', 'x'*2049),
    ('pdf_path', '/arbitrary/source.pdf'),
])
def test_report_declarations_refused_before_extraction(tmp_path, monkeypatch, field, value):
    inputs = Inputs(tmp_path)
    inputs.payload['reports'][0][field] = value
    inputs.sync()
    no_parse(monkeypatch)
    with pytest.raises(ValueError):
        inputs.load()


@pytest.mark.parametrize('change', ['duplicate-report', 'missing-quarter', 'duplicate-member', 'constant', 'oversized', 'invalid-utf8'])
def test_strict_json_and_report_coverage(tmp_path, monkeypatch, change):
    inputs = Inputs(tmp_path)
    if change == 'duplicate-report':
        inputs.payload['reports'].append(dict(inputs.payload['reports'][0]))
        inputs.sync()
    elif change == 'missing-quarter':
        del inputs.payload['reports'][0]['fiscal_quarter']
        inputs.sync()
    elif change == 'duplicate-member':
        body = json.dumps(inputs.payload).replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
        inputs.declarations.write_bytes(body.encode())
    elif change == 'constant':
        inputs.declarations.write_bytes(json.dumps(inputs.payload).replace('"fiscal_year": 2025', '"fiscal_year": NaN', 1).encode())
    elif change == 'oversized':
        inputs.declarations.write_bytes(b' ' * (1_048_576 + 1))
    else:
        inputs.declarations.write_bytes(b'\xff')
    no_parse(monkeypatch)
    with pytest.raises(ValueError):
        inputs.load()


@pytest.mark.parametrize('filename', ['../report.pdf', '..\\report.pdf', 'C:report.pdf', '/report.pdf'])
def test_registered_path_escape_refused_before_extraction(tmp_path, monkeypatch, filename):
    inputs = Inputs(tmp_path)
    inputs.rows[0]['filename'] = filename
    inputs.sync()
    no_parse(monkeypatch)
    with pytest.raises(ValueError):
        inputs.load()


@pytest.mark.parametrize('field,value', [('fiscal_year', 2026), ('fiscal_quarter', 1), ('fiscal_quarter', None)])
def test_reviewed_quarter_metadata_cannot_be_contradicted(tmp_path, monkeypatch, field, value):
    inputs = Inputs(tmp_path)
    inputs.rows[1]['reporting_period'] = dict(fiscal_year=2025, quarter=2, end_date='2024-07-28',
        reviewed_by='Synthetic fixture author', source_locator='Native generated table')
    inputs.payload['reports'][1][field] = value
    inputs.sync()
    no_parse(monkeypatch)
    with pytest.raises(ValueError):
        inputs.load()


def test_expected_verified_corpus_sources_cannot_be_substituted(tmp_path, monkeypatch):
    inputs = Inputs(tmp_path)
    sources = load_source_manifest(inputs.registry)
    inputs.rows[0]['label'] = 'Changed registry while corpus retains original snapshot'
    inputs.sync()
    no_parse(monkeypatch)
    with pytest.raises(ValueError):
        inputs.load(expected_sources=sources)


@pytest.mark.parametrize('entry', ['registry', 'declarations', 'pdf'])
def test_changed_configuration_after_build_is_rejected(tmp_path, entry):
    inputs = Inputs(tmp_path)
    snapshot = inputs.load()
    path = inputs.registry if entry == 'registry' else inputs.declarations if entry == 'declarations' else inputs.reports[0].pdf_path
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError):
        snapshot.verify_unchanged()


@pytest.mark.parametrize('entry', ['registry', 'declarations', 'pdf'])
def test_changes_at_the_end_of_native_build_prevent_snapshot_release(tmp_path, monkeypatch, entry):
    inputs = Inputs(tmp_path)
    module = startup()
    original = module.build_income_index
    def changed(reports):
        index = original(reports)
        path = inputs.registry if entry == 'registry' else inputs.declarations if entry == 'declarations' else inputs.reports[0].pdf_path
        path.write_bytes(path.read_bytes() + b'\n')
        return index
    monkeypatch.setattr(module, 'build_income_index', changed)
    with pytest.raises(ValueError):
        inputs.load()


def test_pdf_substitution_is_refused(tmp_path):
    inputs = Inputs(tmp_path)
    inputs.reports[0].pdf_path.write_bytes(inputs.reports[1].pdf_path.read_bytes())
    with pytest.raises(ValueError):
        inputs.load()


def test_absolute_inputs_required(tmp_path):
    inputs = Inputs(tmp_path)
    with pytest.raises(ValueError):
        startup().load_statement_snapshot(source_registry=Path('registry.json'), source_dir=inputs.source_dir,
                                         fiscal_declarations=inputs.declarations)


def test_complete_model_corpus_binding_is_retained_and_rechecked(tmp_path):
    inputs = Inputs(tmp_path)
    manifest, chunks = inputs.corpus_files()
    snapshot = inputs.load(corpus_manifest=manifest, chunks_path=chunks)
    assert snapshot.verify_unchanged() is None
    chunks.write_bytes(chunks.read_bytes() + b'\n')
    with pytest.raises(ValueError):
        snapshot.verify_unchanged()


@pytest.mark.parametrize('change', ['hash', 'count', 'report-hash', 'filename', 'missing-report', 'duplicate-report', 'chunk-document'])
def test_invalid_model_corpus_binding_precedes_pdf_extraction(tmp_path, monkeypatch, change):
    inputs = Inputs(tmp_path)
    manifest, chunks = inputs.corpus_files()
    if change == 'hash':
        inputs.manifest_value['chunks_sha256'] = '0'*64
    elif change == 'count':
        inputs.manifest_value['chunk_count'] = 9
    elif change == 'report-hash':
        inputs.manifest_value['documents'][0]['source_sha256'] = '0'*64
    elif change == 'filename':
        inputs.manifest_value['documents'][0]['source_filename'] = 'another.pdf'
    elif change == 'missing-report':
        inputs.manifest_value['documents'].pop()
    elif change == 'duplicate-report':
        inputs.manifest_value['documents'][1] = dict(inputs.manifest_value['documents'][0])
    else:
        rows = [json.loads(line) for line in chunks.read_text().splitlines()]
        rows[0]['document_id'] = 'unregistered'
        chunks.write_bytes(''.join(json.dumps(row)+'\n' for row in rows).encode())
        inputs.manifest_value['chunks_sha256'] = digest(chunks)
    write_json(manifest, inputs.manifest_value)
    no_parse(monkeypatch)
    with pytest.raises(ValueError):
        inputs.load(corpus_manifest=manifest, chunks_path=chunks)
