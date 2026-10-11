"""Run launcher startup without installing or constructing an embedding model."""
from pathlib import Path
import builtins
import runpy
import sys
import types

import pytest


@pytest.mark.parametrize('token', [None, 'short'])
def test_invalid_token_precedes_corpus_and_model_loading(monkeypatch, token):
    launcher = Path(__file__).resolve().parents[1] / 'run_api.py'
    calls = []

    def forbidden_model(*args, **kwargs):
        calls.append('model')
        raise AssertionError('model must not be loaded')

    def forbidden_server(*args, **kwargs):
        calls.append('server')
        raise AssertionError('server must not start')

    model_module = types.ModuleType('sentence_transformers')
    model_module.SentenceTransformer = forbidden_model
    server_module = types.ModuleType('uvicorn')
    server_module.run = forbidden_server
    monkeypatch.setitem(sys.modules, 'sentence_transformers', model_module)
    monkeypatch.setitem(sys.modules, 'uvicorn', server_module)
    monkeypatch.setattr(sys, 'argv', [str(launcher), '--chunks', 'absent-corpus.jsonl'])
    if token is None:
        monkeypatch.delenv('NVIDIA_RAG_API_TOKEN', raising=False)
    else:
        monkeypatch.setenv('NVIDIA_RAG_API_TOKEN', token)
    with pytest.raises(ValueError, match='at least 32 characters'):
        runpy.run_path(str(launcher), run_name='__main__')
    assert calls == []


@pytest.mark.parametrize('token', [None, 'short'])
def test_invalid_token_precedes_optional_runtime_imports(monkeypatch, token):
    launcher = Path(__file__).resolve().parents[1] / 'run_api.py'
    imported = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in {'sentence_transformers', 'uvicorn'}:
            imported.append(name)
            raise ModuleNotFoundError('optional runtime intentionally unavailable')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(sys, 'argv', [str(launcher), '--chunks', 'absent-corpus.jsonl'])
    if token is None:
        monkeypatch.delenv('NVIDIA_RAG_API_TOKEN', raising=False)
    else:
        monkeypatch.setenv('NVIDIA_RAG_API_TOKEN', token)
    with pytest.raises(ValueError, match='at least 32 characters'):
        runpy.run_path(str(launcher), run_name='__main__')
    assert imported == []
