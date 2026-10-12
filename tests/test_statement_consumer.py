"""The existing HTTP consumer must select its statement mode explicitly."""
from pathlib import Path
import runpy
import os
import subprocess
import sys

import httpx
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/query_client.py'
TOKEN = 'synthetic-consumer-token-' + 'x' * 32


def statement_consumer():
    function = runpy.run_path(str(EXAMPLE)).get('query_statement_candidates_api')
    assert callable(function), 'consumer must expose an explicit statement-candidates operation'
    return function


def test_consumer_transports_original_question_and_scope_without_inventing_fields():
    observed = []
    original = '  What was net income for the year ended January 26, 2025?  '
    expected = {'disposition': 'source_candidate', 'cells': [{'amount': '8650'}],
                'financial_correctness': 'unestablished'}
    def peer(request):
        observed.append(request)
        return httpx.Response(200, json=expected)
    with httpx.Client(base_url='https://example.com', transport=httpx.MockTransport(peer)) as client:
        result = statement_consumer()(client, token=TOKEN, question=original,
                                     document_ids=['nvidia-fy2025-annual'])
    assert result == expected
    assert len(observed) == 1 and observed[0].url.path == '/v1/statement-candidates'
    assert observed[0].headers['Authorization'] == 'Bearer ' + TOKEN
    import json
    assert json.loads(observed[0].content) == {'question': original, 'document_ids': ['nvidia-fy2025-annual']}


def test_global_scope_is_omitted_without_rewriting_the_question():
    observed = []
    def peer(request):
        observed.append(request)
        return httpx.Response(200, json={'disposition': 'clarify', 'cells': []})
    with httpx.Client(base_url='https://example.com', transport=httpx.MockTransport(peer)) as client:
        result = statement_consumer()(client, token=TOKEN, question='Original question', document_ids=None)
    import json
    assert json.loads(observed[0].content) == {'question': 'Original question'}
    assert result == {'disposition': 'clarify', 'cells': []}


def test_statement_operational_fault_is_not_converted_to_a_content_refusal():
    def peer(request):
        return httpx.Response(503, json={'disposition': 'unavailable', 'cells': []})
    with httpx.Client(base_url='https://example.com', transport=httpx.MockTransport(peer)) as client:
        with pytest.raises(httpx.HTTPStatusError) as error:
            statement_consumer()(client, token=TOKEN, question='Question', document_ids=None)
    assert error.value.response.status_code == 503


def test_cli_statement_mode_keeps_destination_validation_before_token_check():
    env = dict(os.environ)
    env.pop('NVIDIA_RAG_API_TOKEN', None)
    completed = subprocess.run([sys.executable, str(EXAMPLE), '--statement-candidates',
        '--base-url', 'http://example.com', '--question', 'Source question'],
        env=env, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 2
    assert 'remote API connections require HTTPS' in completed.stderr
    assert 'configure NVIDIA_RAG_API_TOKEN' not in completed.stderr
    assert completed.stdout == ''


def test_document_scope_cannot_silently_change_dense_query_mode():
    env = dict(os.environ)
    env.pop('NVIDIA_RAG_API_TOKEN', None)
    completed = subprocess.run([sys.executable, str(EXAMPLE), '--document-id', 'report-1',
        '--question', 'Source question'], env=env, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 2
    assert 'document IDs require --statement-candidates' in completed.stderr
    assert completed.stdout == ''
