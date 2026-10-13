"""Model launcher preflight is independent of optional model construction."""
import builtins
from pathlib import Path
import runpy
import sys
import types

import pytest

from test_statement_startup import Inputs

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'run_api.py'


def options(inputs):
    manifest, chunks = inputs.corpus_files()
    return ['--chunks', str(chunks), '--corpus-manifest', str(manifest),
        '--statement-source-registry', str(inputs.registry), '--statement-source-dir', str(inputs.source_dir),
        '--statement-fiscal-declarations', str(inputs.declarations)]


def test_model_launch_arguments_expose_explicit_statement_trio(tmp_path, monkeypatch):
    inputs = Inputs(tmp_path)
    ns = runpy.run_path(str(LAUNCHER))
    monkeypatch.setattr(sys, 'argv', [str(LAUNCHER)] + options(inputs))
    try:
        args = ns['parse_args']()
    except SystemExit:
        pytest.fail('model launcher does not expose optional statement startup arguments')
    assert args.statement_source_registry == inputs.registry
    assert args.statement_source_dir == inputs.source_dir
    assert args.statement_fiscal_declarations == inputs.declarations


@pytest.mark.parametrize('missing', ['--statement-source-registry', '--statement-source-dir', '--statement-fiscal-declarations'])
def test_partial_statement_trio_is_refused(tmp_path, monkeypatch, missing):
    inputs = Inputs(tmp_path)
    argv = options(inputs)
    at = argv.index(missing)
    del argv[at:at+2]
    ns = runpy.run_path(str(LAUNCHER))
    monkeypatch.setattr(sys, 'argv', [str(LAUNCHER)] + argv)
    with pytest.raises(SystemExit) as error:
        ns['parse_args']()
    assert error.value.code == 2


def test_model_launcher_builds_the_optional_bound_snapshot(tmp_path, monkeypatch):
    inputs = Inputs(tmp_path)
    ns = runpy.run_path(str(LAUNCHER))
    assert callable(ns.get('load_configured_statement')), 'model startup snapshot composition is missing'
    monkeypatch.setattr(sys, 'argv', [str(LAUNCHER)] + options(inputs))
    snapshot = ns['load_configured_statement'](ns['parse_args']())
    assert len(snapshot.index.cells) == 60
    assert {p.path.name for p in snapshot.input_identities} >= {'chunks.jsonl', 'manifest.json'}


def test_invalid_statement_source_refuses_before_optional_model_imports(tmp_path, monkeypatch):
    inputs = Inputs(tmp_path)
    argv = options(inputs)
    inputs.declarations.write_bytes(b'invalid JSON')
    monkeypatch.setenv('NVIDIA_RAG_API_TOKEN', 'synthetic-model-startup-'+'x'*32)
    monkeypatch.setattr(sys, 'argv', [str(LAUNCHER)] + argv)
    imported = []
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'sentence_transformers', 'uvicorn'}:
            imported.append(name)
            pytest.fail('invalid statement configuration reached optional runtime import')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(ValueError):
        runpy.run_path(str(LAUNCHER), run_name='__main__')
    assert imported == []


def test_token_refusal_precedes_optional_statement_snapshot_io(tmp_path, monkeypatch):
    inputs = Inputs(tmp_path)
    monkeypatch.setattr(sys, 'argv', [str(LAUNCHER)] + options(inputs))
    monkeypatch.delenv('NVIDIA_RAG_API_TOKEN', raising=False)
    original = Path.read_bytes
    def guarded(path):
        if path in {inputs.registry, inputs.declarations}:
            pytest.fail('statement configuration read before API token')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    with pytest.raises(ValueError, match='TOKEN'):
        runpy.run_path(str(LAUNCHER), run_name='__main__')


def test_model_launcher_refuses_conflicting_catalog_identity_before_model_import(tmp_path, monkeypatch):
    inputs = Inputs(tmp_path)
    inputs.payload['reports'][1]['fiscal_quarter'] = None  # Both now claim the same annual FY2025 identity.
    inputs.sync()
    monkeypatch.setenv('NVIDIA_RAG_API_TOKEN', 'synthetic-model-startup-'+'x'*32)
    monkeypatch.setattr(sys, 'argv', [str(LAUNCHER)] + options(inputs))
    imported = []
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'sentence_transformers', 'uvicorn'}:
            imported.append(name)
            pytest.fail('conflicting report identities reached optional model import')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(ValueError, match='duplicate report identity'):
        runpy.run_path(str(LAUNCHER), run_name='__main__')
    assert imported == []


def test_statement_routing_and_explicit_source_task_are_rejected_before_runtime(tmp_path, monkeypatch):
    inputs = Inputs(tmp_path)
    monkeypatch.setattr(sys, 'argv', [str(LAUNCHER)] + options(inputs)
        + ['--source-task', 'reported_results', '--source-registry', str(inputs.registry)])
    ns = runpy.run_path(str(LAUNCHER))
    with pytest.raises(SystemExit) as error:
        ns['parse_args']()
    assert error.value.code == 2


@pytest.mark.parametrize('mutate', [False, True])
def test_model_launcher_attaches_snapshot_and_rechecks_after_model_work(tmp_path, monkeypatch, mutate):
    from fastapi.testclient import TestClient
    from nvidia_rag import embedding_store, runtime
    from nvidia_rag.core import build_grounded_context
    inputs = Inputs(tmp_path)
    monkeypatch.setenv('NVIDIA_RAG_API_TOKEN', 'synthetic-model-startup-'+'x'*32)
    monkeypatch.setattr(sys, 'argv', [str(LAUNCHER)] + options(inputs))
    model = types.ModuleType('sentence_transformers')
    model.SentenceTransformer = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, 'sentence_transformers', model)
    server = types.ModuleType('uvicorn')
    observed = []
    server.run = lambda app, **kwargs: observed.append(app)
    monkeypatch.setitem(sys.modules, 'uvicorn', server)
    monkeypatch.setattr(embedding_store, 'load_passage_embedding_artifact', lambda **kwargs: None)
    class Service:
        def query(self, question):
            return build_grounded_context(question, evidence=[])
    def build(**kwargs):
        assert kwargs['report_catalog'] is not None
        assert kwargs['route_fiscal_year'] is False
        assert {r.document_id for r in kwargs['report_catalog'].reports} == {r.document_id for r in inputs.reports}
        if mutate:
            inputs.declarations.write_bytes(inputs.declarations.read_bytes() + b'\n')
        return Service()
    monkeypatch.setattr(runtime, 'build_bge_service', build)
    if mutate:
        with pytest.raises(ValueError, match='changed'):
            runpy.run_path(str(LAUNCHER), run_name='__main__')
        assert observed == []
    else:
        runpy.run_path(str(LAUNCHER), run_name='__main__')
        token = 'synthetic-model-startup-'+'x'*32
        response = TestClient(observed[0]).post('/v1/statement-candidates',
            headers={'Authorization':'Bearer '+token}, json={'question':
                'What was net income for the year ended January 26, 2025, in millions of dollars?',
                'document_ids':[inputs.reports[0].document_id]})
        assert response.status_code == 200 and response.json()['cells'][0]['amount'] == '8650'
