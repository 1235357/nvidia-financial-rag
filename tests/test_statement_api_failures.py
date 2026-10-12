"""Operational source-selection faults never release a partial candidate."""
from dataclasses import replace
import importlib

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest

from nvidia_rag.api import create_app
from nvidia_rag.structured_income import Selection, build_income_index
from test_statement_api import (
    ANNUAL_QUESTION, HEADERS, PATH, SECRET, TOKEN, DenseService, indexed_reports,
)


def assert_unavailable(client):
    response = client.post(PATH, headers=HEADERS, json={'question': ANNUAL_QUESTION})
    assert response.status_code == 503
    body = response.json()
    assert body['disposition'] == 'unavailable'
    assert body['error_code'] == 'statement_selection_unavailable'
    assert body['cells'] == [] and body['parsed_request'] is None
    assert body['request_id'] == response.headers['X-Request-ID'] == 'statement-http-test'
    assert SECRET not in response.text


def test_source_selection_exception_is_safe_and_correlated(indexed_reports, monkeypatch, caplog):
    module = importlib.import_module('nvidia_rag.statement_api')
    def broken(*args, **kwargs):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(module, 'select_statement_candidates', broken)
    service = DenseService()
    with TestClient(create_app(service=service, api_token=TOKEN,
                              statement_index=indexed_reports[0])) as client:
        assert_unavailable(client)
    assert service.calls == [] and SECRET not in caplog.text


def test_candidate_serialization_exception_is_safe_and_correlated(indexed_reports, monkeypatch):
    module = importlib.import_module('nvidia_rag.statement_api')
    def broken(*args, **kwargs):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(module.StatementResponse, 'model_dump', broken)
    with TestClient(create_app(service=DenseService(), api_token=TOKEN,
                              statement_index=indexed_reports[0])) as client:
        assert_unavailable(client)


def test_json_response_failure_is_safe_and_correlated(indexed_reports, monkeypatch):
    module = importlib.import_module('nvidia_rag.statement_api')
    original = module.JSONResponse
    calls = []
    def first_response_fails(*args, **kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise RuntimeError(SECRET)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'JSONResponse', first_response_fails)
    with TestClient(create_app(service=DenseService(), api_token=TOKEN,
                              statement_index=indexed_reports[0])) as client:
        assert_unavailable(client)
    assert len(calls) == 2


@pytest.mark.parametrize('fault', ['nonfinite_bbox', 'empty_candidate', 'unknown_disposition'])
def test_malformed_selector_result_is_unavailable(indexed_reports, monkeypatch, fault):
    module = importlib.import_module('nvidia_rag.statement_api')
    index = indexed_reports[0]
    cell = next(c for c in index.cells if c.row_literal == 'Net income'
                and c.period_end == '2025-01-26')
    if fault == 'nonfinite_bbox':
        role, span = cell.bindings[0]
        cell = replace(cell, bindings=((role, replace(span, bbox=(0.0, 0.0, float('nan'), 1.0))),
                                     *cell.bindings[1:]))
        result = Selection('source_candidate', 'EXACT_LITERAL_FIELDS_MATCH', (cell,))
    elif fault == 'empty_candidate':
        result = Selection('source_candidate', 'EXACT_LITERAL_FIELDS_MATCH')
    else:
        result = Selection('private-' + SECRET, 'EXACT_LITERAL_FIELDS_MATCH')
    monkeypatch.setattr(module, 'select_statement_candidates', lambda *args, **kwargs: result)
    with TestClient(create_app(service=DenseService(), api_token=TOKEN,
                              statement_index=index)) as client:
        assert_unavailable(client)


def test_existing_query_also_rejects_non_ascii_bearer():
    service = DenseService()
    with TestClient(create_app(service=service, api_token=TOKEN)) as client:
        response = client.post('/v1/query', headers=[(b'authorization', b'Bearer \xff')],
                               json={'question': 'ordinary question'})
    assert response.status_code == 401 and service.calls == []


def test_overlapping_annual_source_date_requires_explicit_report_scope(indexed_reports):
    report = indexed_reports[1]
    second = replace(report, document_id='nvidia-fy2024-annual', fiscal_year=2024)
    index = build_income_index((report, second))
    question = ('What was net income for the year ended January 26, 2025, '
                'in millions of dollars?')
    with TestClient(create_app(service=DenseService(), api_token=TOKEN,
                              statement_index=index)) as client:
        response = client.post(PATH, headers=HEADERS, json={'question': question})
        assert response.status_code == 200
        assert response.json()['disposition'] == 'clarify'
        assert response.json()['reason'] == 'MULTIPLE_SOURCE_MATCHES'
        assert response.json()['cells'] == []
        scoped = client.post(PATH, headers=HEADERS,
                             json={'question': question, 'document_ids': [report.document_id]})
        assert scoped.status_code == 200 and scoped.json()['disposition'] == 'source_candidate'
        assert scoped.json()['cells'][0]['document_id'] == report.document_id


def test_deep_json_within_raw_limit_is_invalid_input_not_service_failure(indexed_reports):
    body = b'{"question":' + b'[' * 3900 + b'"private-deep-sentinel"' + b']' * 3900 + b'}'
    assert len(body) < 8192
    with TestClient(create_app(service=DenseService(), api_token=TOKEN,
                              statement_index=indexed_reports[0])) as client:
        response = client.post(PATH, headers={**HEADERS, 'Content-Type': 'application/json'}, content=body)
    assert response.status_code == 422
    assert response.json() == {'detail': 'Invalid statement request.'}


def test_selector_http_exception_cannot_escape_the_operational_boundary(indexed_reports, monkeypatch):
    module = importlib.import_module('nvidia_rag.statement_api')
    def broken(*args, **kwargs):
        raise HTTPException(status_code=422, detail=SECRET)
    monkeypatch.setattr(module, 'select_statement_candidates', broken)
    with TestClient(create_app(service=DenseService(), api_token=TOKEN,
                              statement_index=indexed_reports[0])) as client:
        assert_unavailable(client)
