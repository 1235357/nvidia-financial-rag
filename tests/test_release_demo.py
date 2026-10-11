"""Generated transport demo checks; these are not financial quality labels."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def load_demo():
    path = Path(__file__).resolve().parents[1] / "examples" / "serve_demo.py"
    assert path.is_file(), "The portable generated-data demo entrypoint is missing"
    spec = spec_from_file_location("release_demo", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_demo_auth_validation_and_evidence_contract():
    demo = load_demo()
    token = "synthetic-contract-token-" + "x" * 24
    with TestClient(demo.create_demo_app({"NVIDIA_RAG_API_TOKEN": token})) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/docs").status_code == 404
        assert client.post("/v1/query", json={"question": "authentication"}).status_code == 401
        headers = {"Authorization": f"Bearer {token}", "X-Request-ID": "release-demo-contract"}
        assert client.post("/v1/query", headers=headers, json={"question": " "}).status_code == 422
        response = client.post("/v1/query", headers=headers, json={"question": "bearer authentication"})
        body = response.json()
        assert response.status_code == 200
        assert body["can_answer"] is True
        assert body["evidence_passed"] is True
        assert body["answer"] == ""
        assert body["error_code"] is None
        assert body["citations"] == ["synthetic-release-demo p.1"]
        assert body["evidence"][0]["document_id"] == "synthetic-release-demo"
        assert "Synthetic" in body["evidence"][0]["text"]
        assert body["request_id"] == response.headers["X-Request-ID"] == "release-demo-contract"


def test_unknown_vocabulary_refuses_without_invented_evidence():
    demo = load_demo()
    token = "synthetic-contract-token-" + "x" * 24
    with TestClient(demo.create_demo_app({"NVIDIA_RAG_API_TOKEN": token})) as client:
        response = client.post("/v1/query", headers={"Authorization": f"Bearer {token}"},
                               json={"question": "zxqv plmokn"})
        body = response.json()
        assert response.status_code == 200
        assert body["can_answer"] is False
        assert body["answer"] == ""
        assert body["evidence"] == []
        assert body["citations"] == []


@pytest.mark.parametrize("token", ["", "short"])
def test_token_preflight_precedes_demo_index_construction(monkeypatch, token):
    demo = load_demo()
    def forbidden(*args, **kwargs):
        raise AssertionError("Index construction must not precede token validation")
    monkeypatch.setattr(demo, "TfidfRetriever", forbidden)
    with pytest.raises(ValueError, match="at least 32"):
        demo.create_demo_app({"NVIDIA_RAG_API_TOKEN": token})
