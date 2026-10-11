"""Synthetic transport checks for operational failures versus evidence refusal.

The real adapter, service and HTTP boundary run together. MockTransport replaces
only the external model server; these are not model, latency or load results.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from nvidia_rag.api import create_app
from nvidia_rag.core import Chunk, ScoredChunk, build_grounded_context
from nvidia_rag.generation import GroundedRagService, OpenAICompatibleChatGenerator

TOKEN = "synthetic-api-token-" + "x" * 32
SECRET = "private-upstream-detail"
QUESTION = "What was synthetic revenue?"


class SyntheticEvidenceService:
    def query(self, question):
        evidence = () if question == "no evidence" else (
            ScoredChunk(Chunk("fixture-chunk", "fixture-report", 2,
                              "Synthetic revenue was $100 million."), 0.9),
        )
        return build_grounded_context(question, evidence=evidence)


def query_through_model_transport(handler, *, question=QUESTION):
    generator = OpenAICompatibleChatGenerator(
        base_url="http://127.0.0.1:8001", model_name="synthetic-model",
        api_token="synthetic-model-token-" + "x" * 16,
        transport=httpx.MockTransport(handler),
    )
    service = GroundedRagService(
        evidence_service=SyntheticEvidenceService(), generator=generator,
    )
    with TestClient(create_app(service=service, api_token=TOKEN)) as client:
        return client.post(
            "/v1/query", json={"question": question},
            headers={"Authorization": f"Bearer {TOKEN}",
                     "X-Request-ID": "generation-contract"},
        )


@pytest.mark.parametrize("timeout_type", [
    httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
])
def test_model_transport_timeout_is_504_with_gated_evidence(timeout_type):
    def handler(request):
        raise timeout_type(SECRET, request=request)

    response = query_through_model_transport(handler)
    assert response.status_code == 504
    body = response.json()
    assert body["error_code"] == "generation_timeout"
    assert body["can_answer"] is False
    assert body["evidence_passed"] is True
    assert body["answer"] == ""
    assert body["citations"] == ["fixture-report p.2"]
    assert body["evidence"][0]["chunk_id"] == "fixture-chunk"
    assert body["request_id"] == response.headers["X-Request-ID"] == "generation-contract"
    assert SECRET not in response.text


@pytest.mark.parametrize("failure", ["connect", "http_503", "malformed", "unfinished"])
def test_model_failure_is_503_and_never_releases_upstream_details(failure):
    def handler(request):
        if failure == "connect":
            raise httpx.ConnectError(SECRET, request=request)
        if failure == "http_503":
            return httpx.Response(503, text=SECRET)
        if failure == "malformed":
            return httpx.Response(200, json={"private": SECRET})
        return httpx.Response(200, json={"choices": [{
            "message": {"content": SECRET}, "finish_reason": "length",
        }]})

    response = query_through_model_transport(handler)
    assert response.status_code == 503
    body = response.json()
    assert body["error_code"] == "generation_unavailable"
    assert body["evidence_passed"] is True
    assert body["answer"] == ""
    assert body["citations"] == ["fixture-report p.2"]
    assert body["evidence"][0]["pdf_page"] == 2
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert SECRET not in response.text


@pytest.mark.parametrize("answer,can_answer", [
    ("The retrieved evidence is insufficient.", False),
    ("Synthetic revenue was $999 million [fixture-report p.2].", False),
    ("Synthetic revenue was $100 million [fixture-report p.2].", True),
])
def test_completed_answer_or_content_refusal_remains_200(answer, can_answer):
    response = query_through_model_transport(lambda request: httpx.Response(
        200, json={"choices": [{"message": {"content": answer}, "finish_reason": "stop"}]},
    ))
    assert response.status_code == 200
    body = response.json()
    assert body["can_answer"] is can_answer
    assert body["error_code"] is None
    assert body["answer"] == (answer if can_answer else "")
    assert body["evidence_passed"] is True


def test_missing_evidence_refusal_does_not_call_model_server():
    def forbidden_handler(request):
        pytest.fail("model server must not receive a prompt without evidence")

    response = query_through_model_transport(forbidden_handler, question="no evidence")
    assert response.status_code == 200
    body = response.json()
    assert body["error_code"] is None
    assert body["evidence_passed"] is False
    assert body["evidence"] == []
    assert body["answer"] == ""


def test_bare_generator_timeout_preserves_operational_failure_at_http_boundary():
    class Generator:
        def generate(self, prompt, *, max_new_tokens):
            raise TimeoutError(SECRET)

    service = GroundedRagService(
        evidence_service=SyntheticEvidenceService(), generator=Generator(),
    )
    with TestClient(create_app(service=service, api_token=TOKEN)) as client:
        response = client.post("/v1/query", json={"question": QUESTION},
                               headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 504
    assert response.json()["error_code"] == "generation_timeout"
    assert SECRET not in response.text


def test_uncaught_query_fault_is_503_with_machine_readable_code():
    class Service:
        def query(self, question):
            raise RuntimeError(SECRET)

    with TestClient(create_app(service=Service(), api_token=TOKEN)) as client:
        response = client.post("/v1/query", json={"question": QUESTION},
                               headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 503
    assert response.json()["error_code"] == "query_service_unavailable"
    assert response.json()["evidence_passed"] is False
    assert response.json()["evidence"] == []
    assert SECRET not in response.text


@pytest.mark.parametrize("counting_fails", [True, False])
def test_broken_token_counter_is_unavailable_but_prompt_budget_is_content_guard(counting_fails):
    class Generator:
        def generate(self, prompt, *, max_new_tokens):
            pytest.fail("token counting and token budget guards must precede generation")

    def count_tokens(prompt):
        if counting_fails:
            raise RuntimeError(SECRET)
        return 101

    service = GroundedRagService(
        evidence_service=SyntheticEvidenceService(), generator=Generator(),
        max_prompt_tokens=100, prompt_token_counter=count_tokens,
    )
    with TestClient(create_app(service=service, api_token=TOKEN)) as client:
        response = client.post("/v1/query", json={"question": QUESTION},
                               headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == (503 if counting_fails else 200)
    body = response.json()
    assert body["error_code"] == ("generation_unavailable" if counting_fails else None)
    assert body["evidence_passed"] is True
    assert body["answer"] == ""
    assert SECRET not in response.text


def test_openapi_documents_operational_status_codes():
    app = create_app(service=SyntheticEvidenceService(), api_token=TOKEN, expose_docs=True)
    responses = app.openapi()["paths"]["/v1/query"]["post"]["responses"]
    assert "503" in responses
    assert "504" in responses
