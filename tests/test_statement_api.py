"""HTTP source-candidate contracts use native synthetic PDF extraction."""
from dataclasses import asdict, replace
from hashlib import sha256
import json

from fastapi.testclient import TestClient
import pytest

from nvidia_rag.api import create_app
from nvidia_rag.core import build_grounded_context
from nvidia_rag.structured_income import IncomeIndex, build_income_index
from test_annual_income import annual_pdf
from test_structured_income import source_pdf

TOKEN = 'synthetic-statement-api-' + 'x' * 32
HEADERS = {'Authorization': 'Bearer ' + TOKEN, 'X-Request-ID': 'statement-http-test'}
PATH = '/v1/statement-candidates'
ANNUAL_QUESTION = ('According to NVIDIA fiscal 2025 annual report, what was net income '
                   'for the year ended January 26, 2025, in millions of dollars?')
SECRET = 'private-exception-and-input-sentinel'


class DenseService:
    def __init__(self):
        self.calls = []
    def query(self, question):
        self.calls.append(question)
        return build_grounded_context(question, evidence=[])


@pytest.fixture
def indexed_reports(tmp_path):
    annual = annual_pdf(tmp_path, split_date=True)
    quarterly = source_pdf(tmp_path)
    index = build_income_index((annual, quarterly))
    return index, annual, quarterly


def client_for(index=None, *, enabled=True, docs=False):
    service = DenseService()
    kwargs = {'statement_index': index} if enabled else {}
    app = create_app(service=service, api_token=TOKEN, expose_docs=docs, **kwargs)
    return TestClient(app), service


def post(client, question=ANNUAL_QUESTION, **extras):
    return client.post(PATH, headers=HEADERS, json={'question': question, **extras})


def test_disabled_statement_route_is_explicit_and_authenticated():
    client, service = client_for(enabled=False)
    response = post(client)
    assert response.status_code == 503
    body = response.json()
    assert body['disposition'] == 'unavailable'
    assert body['reason'] == 'STATEMENT_INDEX_NOT_CONFIGURED'
    assert body['error_code'] == 'statement_selection_unavailable'
    assert body['cells'] == [] and body['index_sha256'] is None
    assert body['financial_correctness'] == 'unestablished'
    assert response.headers['X-Request-ID'] == body['request_id'] == 'statement-http-test'
    assert service.calls == []
    assert client.post(PATH, content=b'not-json').status_code == 401


def test_annual_http_candidate_preserves_native_spans_and_snapshot_identity(indexed_reports):
    index, annual, _ = indexed_reports
    client, service = client_for(index)
    response = post(client, document_ids=[annual.document_id])
    assert response.status_code == 200
    body = response.json()
    assert body['schema_version'] == 1 and body['disposition'] == 'source_candidate'
    assert body['reason'] == 'EXACT_LITERAL_FIELDS_MATCH' and body['error_code'] is None
    assert body['financial_correctness'] == 'unestablished'
    assert body['score_semantics'] == 'exact-source-field-selection;no-cosine-score'
    assert body['request_id'] == response.headers['X-Request-ID'] == 'statement-http-test'
    assert body['parsed_request']['query'] == ANNUAL_QUESTION
    assert body['parsed_request']['query_sha256'] == sha256(ANNUAL_QUESTION.encode()).hexdigest()
    assert body['parsed_request']['operator_document_ids'] == [annual.document_id]
    cell = body['cells'][0]
    assert len(body['cells']) == 1 and cell['amount'] == '8650'
    assert cell['duration_months'] is None and cell['period_kind'] == 'fiscal_year'
    assert cell['pdf_sha256'] == annual.pdf_sha256 and cell['physical_page'] == 2
    assert cell['currency_normalization'] == 'dollar-symbol;ISO-currency-unverified'
    roles = dict(cell['bindings'])
    assert roles['duration']['literal'] == 'Year Ended'
    assert roles['date']['literal'] == 'January 26,' and roles['date_year']['literal'] == '2025'
    assert all(len(span['bbox']) == 4 for span in roles.values())
    expected = asdict(index)
    expected.pop('_construction_token')
    canonical = json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'
    assert body['index_sha256'] == sha256(canonical).hexdigest()
    assert not {'answer', 'can_answer', 'evidence_passed', 'citations', 'score'} & set(body)
    assert 'pdf_path' not in response.text and str(annual.pdf_path) not in response.text
    assert service.calls == []


def test_quarter_candidate_keeps_source_period_and_original_cell_id(indexed_reports):
    index, _, quarterly = indexed_reports
    client, service = client_for(index)
    question = ('According to NVIDIA fiscal 2025 Q2 report, what was net income '
                'for the three months ended July 28, 2024, in millions of dollars?')
    response = post(client, question, document_ids=[quarterly.document_id])
    assert response.status_code == 200
    cell = response.json()['cells'][0]
    assert cell['period_kind'] == 'months' and cell['duration_months'] == 3
    assert cell['cell_id'] == next(c.cell_id for c in index.cells
        if c.document_id == quarterly.document_id and c.row_literal == 'Net income'
        and c.duration_months == 3 and c.column_role == 'current')
    assert service.calls == []


@pytest.mark.parametrize('question,disposition,reason', [
    (ANNUAL_QUESTION.replace('January 26', 'January 27'), 'refuse', 'NO_INDEXED_CELL_MATCH'),
    (ANNUAL_QUESTION.replace('year ended', 'three months ended'), 'clarify', 'REQUEST_CONFLICT'),
    (ANNUAL_QUESTION.replace('net income', 'net income and revenue'), 'clarify', 'REQUEST_CONFLICT'),
    (ANNUAL_QUESTION.replace('dollars', 'USD'), 'unsupported', 'ISO_CURRENCY_UNVERIFIED_NO_FX'),
])
def test_withholding_stays_distinct_from_operational_failure(indexed_reports, question, disposition, reason):
    index, annual, _ = indexed_reports
    client, service = client_for(index)
    response = post(client, question, document_ids=[annual.document_id])
    assert response.status_code == 200
    assert response.json()['disposition'] == disposition
    assert response.json()['reason'] == reason
    assert response.json()['cells'] == [] and response.json()['error_code'] is None
    assert service.calls == []


def test_unknown_report_in_scope_does_not_silently_intersect(indexed_reports):
    index, annual, _ = indexed_reports
    client, _ = client_for(index)
    response = post(client, document_ids=[annual.document_id, 'unavailable-report'])
    assert response.status_code == 200
    assert response.json()['reason'] == 'OPERATOR_REPORT_UNAVAILABLE'
    assert response.json()['disposition'] == 'refuse' and response.json()['cells'] == []


def test_absent_scope_uses_all_registered_reports_without_rebuilding_or_rereading(indexed_reports):
    index, annual, quarterly = indexed_reports
    client, service = client_for(index)
    annual.pdf_path.write_bytes(b'changed after the accepted snapshot')
    first, second = post(client), post(client)
    assert first.status_code == second.status_code == 200
    assert first.json()['cells'] == second.json()['cells']
    assert first.json()['index_sha256'] == second.json()['index_sha256']
    assert first.json()['parsed_request']['operator_document_ids'] == [annual.document_id, quarterly.document_id]
    assert service.calls == []


@pytest.mark.parametrize('payload', [
    {'question': True}, {'question': 7}, {'question': '  '}, {'question': 'x' * 501},
    {'question': SECRET, 'pdf_path': 'C:/private/report.pdf'},
    {'question': SECRET, 'parsed_request': {'amount': '999'}},
    {'question': SECRET, 'document_ids': []},
    {'question': SECRET, 'document_ids': ['same', 'same']},
    {'question': SECRET, 'document_ids': [True]},
    {'question': SECRET, 'document_ids': ['C:/private/report.pdf']},
    {'question': SECRET, 'document_ids': ['x' * 129]},
    {'question': SECRET, 'document_ids': [f'report-{i}' for i in range(129)]},
])
def test_invalid_http_input_is_bounded_and_not_reflected(indexed_reports, payload):
    client, service = client_for(indexed_reports[0])
    response = client.post(PATH, headers=HEADERS, json=payload)
    assert response.status_code == 422
    assert response.json() == {'detail': 'Invalid statement request.'}
    assert SECRET not in response.text and 'C:/private' not in response.text
    assert service.calls == []


@pytest.mark.parametrize('body', [
    b'not-json', b'[]', b'null', b'{"question":"first","question":"second"}',
    b'{"question":"first","ques\\u0074ion":"second"}',
    b'{"question":NaN}', b'{"question":"\xff"}',
])
def test_json_parse_errors_and_duplicate_decoded_keys_are_safe(indexed_reports, body):
    client, _ = client_for(indexed_reports[0])
    response = client.post(PATH, headers={**HEADERS, 'Content-Type': 'application/json'}, content=body)
    assert response.status_code == 422 and response.json() == {'detail': 'Invalid statement request.'}


def test_non_json_media_type_is_415_before_data_parsing(indexed_reports):
    client, _ = client_for(indexed_reports[0])
    response = client.post(PATH, headers=HEADERS, content=b'{"question":"private"}')
    assert response.status_code == 415
    assert 'private' not in response.text


def test_raw_byte_ceiling_counts_actual_stream_with_misleading_length(indexed_reports):
    client, service = client_for(indexed_reports[0])
    response = client.post(PATH, headers={**HEADERS, 'Content-Type': 'application/json', 'Content-Length': '1'},
        content=iter([b'{"question":"', b'x' * 8200, b'"}']))
    assert response.status_code == 413
    assert response.json() == {'detail': 'Statement request exceeds 8192 bytes.'}
    assert service.calls == []


def test_bearer_auth_precedes_malformed_body_validation(indexed_reports):
    client, _ = client_for(indexed_reports[0])
    assert client.post(PATH, content=b'not-json').status_code == 401
    assert client.post(PATH, headers={'Authorization': 'Bearer wrong'}, content=b'not-json').status_code == 401


def test_non_ascii_bearer_is_rejected_as_unauthorized():
    client, _ = client_for(enabled=False)
    response = client.post(PATH, headers=[(b'authorization', b'Bearer \xff')], content=b'not-json')
    assert response.status_code == 401


def test_unbuilt_and_replaced_indexes_are_rejected_at_startup(indexed_reports):
    for index in (IncomeIndex((), (), ()), replace(indexed_reports[0])):
        with pytest.raises(ValueError):
            client_for(index)


def test_statement_openapi_declares_request_and_dispositions(indexed_reports):
    client, _ = client_for(indexed_reports[0], docs=True)
    schema = client.get('/openapi.json').json()
    route = schema['paths'][PATH]['post']
    body = route['requestBody']['content']['application/json']['schema']
    if '$ref' in body:
        body = schema['components']['schemas'][body['$ref'].rsplit('/', 1)[1]]
    assert body['additionalProperties'] is False
    assert body['required'] == ['question']
    assert body['properties']['question']['maxLength'] == 500
    assert set(route['responses']) >= {'200', '401', '413', '415', '422', '503'}


def test_statement_index_does_not_change_the_existing_query_route(indexed_reports):
    client, service = client_for(indexed_reports[0])
    response = client.post('/v1/query', headers=HEADERS, json={'question': 'ordinary dense question'})
    assert response.status_code == 200 and response.json()['answer'] == ''
    assert response.json()['can_answer'] is False
    assert service.calls == ['ordinary dense question']
