"""Consumer contract tests use HTTP serialization, not a live network server."""
from pathlib import Path
import runpy
import os
import subprocess
import sys

import pytest
import httpx
from fastapi.testclient import TestClient

from nvidia_rag.api import create_app
from nvidia_rag.core import Chunk, ScoredChunk, build_grounded_context

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'query_client.py'
TOKEN = 'synthetic-test-token-' + 'x' * 32


def test_cli_rejects_remote_http_before_missing_token_check():
    environment = dict(os.environ)
    environment.pop('NVIDIA_RAG_API_TOKEN', None)
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE), '--base-url', 'http://example.com',
         '--question', 'Synthetic transport check'],
        env=environment, capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 2
    assert 'remote API connections require HTTPS' in completed.stderr
    assert 'configure NVIDIA_RAG_API_TOKEN' not in completed.stderr
    assert completed.stdout == ''


@pytest.mark.parametrize('url', [
    'http://example.com', 'http://192.0.2.15:8000',
    'http://0.0.0.0:8000', 'ftp://example.com',
    'https://user:secret@example.com', 'https://example.com/?token=secret',
    'https://example.com/#secret',
])
def test_consumer_rejects_unsafe_transport_destinations(url):
    validator = runpy.run_path(str(EXAMPLE)).get('validate_base_url')
    assert callable(validator), 'consumer must validate destination before sending credentials'
    with pytest.raises(ValueError):
        validator(url)


@pytest.mark.parametrize('url', [
    'http://127.0.0.1:8000', 'http://localhost:8000',
    'http://[::1]:8000', 'https://example.com',
])
def test_consumer_allows_loopback_http_and_remote_https(url):
    validator = runpy.run_path(str(EXAMPLE)).get('validate_base_url')
    assert callable(validator), 'consumer must validate destination before sending credentials'
    assert validator(url) == url


def test_learning_lab_separates_similarity_from_requested_period():
    lab = EXAMPLE.with_name('evidence_learning_lab.py')
    result = runpy.run_path(str(lab))['run_lab']()
    assert result['scores'] == [1.0, 0.8]
    assert result['selected_document'] == 'synthetic-fy2024'
    assert result['similarity_gate_passed'] is True
    assert result['requested_period_supported'] is False
    assert result['below_threshold_refused'] is True
    assert result['blank_input_encoder_calls'] == 0


def test_consumer_preserves_evidence_without_claiming_generated_answer():
    namespace = runpy.run_path(str(EXAMPLE))

    class EvidenceService:
        def query(self, question):
            candidate = ScoredChunk(Chunk('synthetic-chunk', 'synthetic-report', 2,
                                          'Synthetic revenue was $100 million.'), 0.9)
            return build_grounded_context(question, evidence=[candidate])

    client = TestClient(create_app(service=EvidenceService(), api_token=TOKEN))
    payload = namespace['query_api'](client, token=TOKEN, question='Revenue?')
    assert payload['evidence_passed'] is True
    assert payload['answer'] == ''
    assert payload['evidence'][0]['chunk_id'] == 'synthetic-chunk'
    assert payload['evidence'][0]['pdf_page'] == 2
    assert payload['citations'] == ['synthetic-report p.2']


def test_consumer_preserves_refusal_and_does_not_invent_answer():
    namespace = runpy.run_path(str(EXAMPLE))

    class EmptyService:
        def query(self, question):
            return build_grounded_context(question, evidence=[])

    client = TestClient(create_app(service=EmptyService(), api_token=TOKEN))
    payload = namespace['query_api'](client, token=TOKEN, question='Unavailable fact?')
    assert payload['can_answer'] is False
    assert payload['answer'] == ''
    assert payload['evidence'] == []
    assert payload['request_id']


def test_consumer_raises_for_service_fault_instead_of_treating_it_as_refusal():
    namespace = runpy.run_path(str(EXAMPLE))

    class BrokenService:
        def query(self, question):
            raise RuntimeError('synthetic failure')

    client = TestClient(create_app(service=BrokenService(), api_token=TOKEN))
    with pytest.raises(httpx.HTTPStatusError) as error:
        namespace['query_api'](client, token=TOKEN, question='Revenue?')
    assert getattr(error.value, 'response').status_code == 503
