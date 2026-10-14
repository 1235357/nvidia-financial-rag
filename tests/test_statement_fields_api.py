"""Independent typed HTTP contract over real synthetic native PDF cells."""
from dataclasses import asdict
from hashlib import sha256
import importlib
import importlib.util
import json

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from nvidia_rag.api import create_app
from nvidia_rag.structured_income import build_income_index
from test_annual_income import annual_pdf
from test_statement_api import DenseService, HEADERS, TOKEN
from test_structured_income import source_pdf

PATH = '/v1/statement-fields'
SECRET = 'private-field-error-sentinel'


def api():
    assert importlib.util.find_spec('nvidia_rag.statement_fields_api') is not None, \
        'typed source-field HTTP adapter is missing'
    return importlib.import_module('nvidia_rag.statement_fields_api')


def payload(**changes):
    values = dict(row_path='per_share/basic', period_kind='months', period_end='2024-07-28',
        duration_months=3, unit_kind='per_share', scale='1', currency_literal='$')
    values.update(changes)
    return values


@pytest.fixture
def configured(tmp_path):
    index = build_income_index((annual_pdf(tmp_path, split_date=True), source_pdf(tmp_path)))
    service = DenseService()
    return TestClient(create_app(service=service, api_token=TOKEN, statement_index=index)), index, service


def post(client, value=None):
    return client.post(PATH, headers=HEADERS, json=payload() if value is None else value)


def test_typed_http_route_is_registered_and_disabled_index_is_authenticated():
    service = DenseService()
    client = TestClient(create_app(service=service, api_token=TOKEN))
    response = post(client)
    assert response.status_code == 503, 'typed source-field route is missing'
    body = response.json()
    assert body['reason'] == 'STATEMENT_INDEX_NOT_CONFIGURED'
    assert body['disposition'] == 'unavailable' and body['cells'] == []
    assert body['selection_request'] is None and body['selection_request_sha256'] is None
    assert body['index_sha256'] is None and body['error_code'] == 'statement_selection_unavailable'
    assert client.post(PATH, content=b'not-json').status_code == 401
    assert service.calls == []


@pytest.mark.parametrize('row,kind,scale,currency', [
    ('per_share/basic', 'per_share', '1', '$'),
    ('count/basic', 'count', '1000000', None),
    ('monetary/revenue', 'monetary', '1000000', '$'),
])
def test_complete_native_cell_and_typed_request_identity_over_http(configured, row, kind, scale, currency):
    client, index, service = configured
    value = payload(row_path=row, unit_kind=kind, scale=scale, currency_literal=currency)
    response = post(client, value)
    assert response.status_code == 200
    body = response.json()
    assert body['schema_version'] == 1 and body['disposition'] == 'source_candidate'
    assert body['financial_correctness'] == 'unestablished'
    assert body['score_semantics'] == 'exact-source-field-selection;no-cosine-score'
    assert body['request_id'] == response.headers['X-Request-ID'] == 'statement-http-test'
    expected = next(c for c in index.cells if c.row_path == row and c.period_end == '2024-07-28' and c.duration_months == 3)
    assert body['cells'] == [json.loads(json.dumps({**asdict(expected), 'period_kind': expected.period_kind}))]
    echoed = {**value, 'document_ids': None, 'fiscal_year': None, 'fiscal_quarter': None, 'column_role': None}
    assert body['selection_request'] == echoed
    canonical = json.dumps(echoed, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode() + b'\n'
    assert body['selection_request_sha256'] == sha256(canonical).hexdigest()
    original = asdict(index)
    original.pop('_construction_token')
    encoded = json.dumps(original, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode() + b'\n'
    assert body['index_sha256'] == sha256(encoded).hexdigest()
    assert not {'parsed_request', 'mentions', 'answer', 'can_answer', 'score'} & set(body)
    assert 'pdf_path' not in response.text and service.calls == []


def test_annual_and_optional_fiscal_scope_preserve_report_and_column_identity(configured):
    client, index, _ = configured
    cell = next(c for c in index.cells if c.period_kind == 'fiscal_year' and c.row_path == 'count/diluted' and c.column_number == 2)
    result = post(client, payload(row_path=cell.row_path, period_kind='fiscal_year', period_end=cell.period_end,
        duration_months=None, unit_kind='count', scale='1000000', currency_literal=None,
        document_ids=[cell.document_id], fiscal_year=cell.fiscal_year, fiscal_quarter=None, column_role='prior_comparison'))
    assert result.status_code == 200 and result.json()['cells'][0]['cell_id'] == cell.cell_id


@pytest.mark.parametrize('value,disposition,reason', [
    (payload(scale='1000000'), 'unsupported', 'REQUESTED_SCALE_CONVERSION_UNSUPPORTED'),
    (payload(currency_literal='USD'), 'unsupported', 'ISO_CURRENCY_UNVERIFIED_NO_FX'),
    (payload(document_ids=['unknown']), 'refuse', 'OPERATOR_REPORT_UNAVAILABLE'),
    (payload(period_end='2024-07-29'), 'refuse', 'NO_INDEXED_CELL_MATCH'),
    (payload(period_kind='fiscal_year', duration_months=None, fiscal_quarter=2), 'clarify', 'REQUEST_CONFLICT'),
])
def test_typed_withheld_outcomes_have_no_cells_or_operational_error(configured, value, disposition, reason):
    response = post(configured[0], value)
    assert response.status_code == 200
    body = response.json()
    assert (body['disposition'], body['reason'], body['cells']) == (disposition, reason, [])
    assert body['selection_request'] is not None and body['error_code'] is None


@pytest.mark.parametrize('field', ['row_path', 'period_kind', 'period_end', 'duration_months', 'unit_kind', 'scale', 'currency_literal'])
def test_each_explicit_field_is_required_even_when_nullable(configured, field):
    value = payload()
    value.pop(field)
    response = post(configured[0], value)
    assert response.status_code == 422 and response.json() == {'detail': 'Invalid statement field request.'}


@pytest.mark.parametrize('change', [
    {'duration_months': True}, {'duration_months': '3'}, {'duration_months': None},
    {'period_end': '2024-7-28'}, {'period_end': '2024-02-30'},
    {'period_kind': 'fiscal_year'}, {'scale': 1}, {'unit_kind': 'percent'},
    {'currency_literal': 0}, {'document_ids': []}, {'document_ids': ['one', 'one']},
    {'fiscal_year': True}, {'fiscal_quarter': 0}, {'column_role': 'latest'},
    {'row_path': '../basic'}, {'question': SECRET}, {'pdf_path': SECRET},
])
def test_schema_is_strict_bounded_and_nonreflective(configured, change):
    response = post(configured[0], payload(**change))
    assert response.status_code == 422 and response.json() == {'detail': 'Invalid statement field request.'}
    assert SECRET not in response.text


@pytest.mark.parametrize('body', [
    b'not-json', b'[]', b'null', b'{"row_path":"a","row_path":"b"}',
    b'{"row_path":"a","row_\\u0070ath":"b"}', b'{"scale":NaN}',
    b'{"scale":Infinity}', b'{"scale":"\xff"}', b'[' * 1500 + b'0' + b']' * 1500,
])
def test_json_rejects_duplicate_decoded_keys_nonfinite_utf8_and_depth(configured, body):
    response = configured[0].post(PATH, headers={**HEADERS, 'Content-Type': 'application/json'}, content=body)
    assert response.status_code == 422 and response.json() == {'detail': 'Invalid statement field request.'}


def test_authentication_precedes_stream_read_and_actual_body_limit(configured):
    client = configured[0]
    assert client.post(PATH, content=b'x' * 9000).status_code == 401
    assert client.post(PATH, headers=HEADERS, content=b'{}').status_code == 415
    response = client.post(PATH, headers={**HEADERS, 'Content-Type': 'application/json', 'Content-Length': '1'}, content=b'x' * 8193)
    assert response.status_code == 413


@pytest.mark.parametrize('fault', ['selector', 'http_exception', 'typed_result', 'json_result'])
def test_operational_faults_are_private_empty_503_and_keep_snapshot(configured, monkeypatch, fault):
    client = configured[0]
    baseline = post(client).json()
    module = api()
    def broken(*args, **kwargs):
        raise HTTPException(status_code=418, detail=SECRET) if fault == 'http_exception' else RuntimeError(SECRET)
    if fault in {'selector', 'http_exception'}:
        monkeypatch.setattr(module, 'select_statement_fields', broken)
    elif fault == 'typed_result':
        monkeypatch.setattr(module.SourceCandidate, 'model_validate', broken)
    else:
        original = module.StatementFieldResponse.model_dump
        def unserializable(self, *args, **kwargs):
            return {**original(self, *args, **kwargs), 'injected': float('nan')}
        monkeypatch.setattr(module.StatementFieldResponse, 'model_dump', unserializable)
    response = post(client)
    assert response.status_code == 503
    body = response.json()
    assert body['cells'] == [] and body['disposition'] == 'unavailable'
    assert body['selection_request'] is None and body['selection_request_sha256'] is None
    assert body['index_sha256'] == baseline['index_sha256'] and SECRET not in response.text


def test_openapi_has_separate_typed_required_fields_and_existing_routes_stay_independent(configured):
    _, index, service = configured
    client = TestClient(create_app(service=service, api_token=TOKEN, statement_index=index, expose_docs=True))
    schema = client.get('/openapi.json').json()['paths'][PATH]['post']['requestBody']['content']['application/json']['schema']
    assert set(schema['required']) == {'row_path', 'period_kind', 'period_end', 'duration_months', 'unit_kind', 'scale', 'currency_literal'}
    assert post(client).status_code == 200 and service.calls == []
    question = 'What was revenue for three months ended July 28, 2024?'
    legacy = client.post('/v1/statement-candidates', headers=HEADERS, json={'question': question})
    assert legacy.status_code == 200 and legacy.json()['disposition'] == 'source_candidate'
    ordinary = client.post('/v1/query', headers=HEADERS, json={'question': 'ordinary question'})
    assert ordinary.status_code == 200 and service.calls == ['ordinary question']
