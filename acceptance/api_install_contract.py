"""Verify an installed [api]-only wheel without checkout imports or models.

Run with the pristine virtual environment's Python -I. This acceptance uses
synthetic evidence and in-process HTTP serialization, not financial inference,
a listener, model readiness, a load test or a production deployment.
"""
from __future__ import annotations

from importlib import metadata
from importlib.util import find_spec
import json
from pathlib import Path
import secrets
import sys
import sysconfig


def main() -> None:
    assert sys.prefix != sys.base_prefix, "Use a dedicated virtual environment"
    assert sys.flags.isolated, "Use Python -I to exclude the checkout and user site"
    for module in ("gradio", "torch", "sentence_transformers", "transformers"):
        assert find_spec(module) is None, f"API-only acceptance must not preload {module}"

    # Import the actual installed consumer surface, not a stand-in dependency.
    import nvidia_rag.api as api
    from fastapi.testclient import TestClient
    from nvidia_rag.core import Chunk, ScoredChunk, build_grounded_context

    package_root = Path(api.__file__).resolve()
    installed_root = Path(sysconfig.get_paths()["purelib"]).resolve()
    assert package_root.is_relative_to(installed_root), "API was imported from a checkout"

    class SyntheticEvidenceService:
        def query(self, question):
            if question == "unanswerable fixture":
                return build_grounded_context(question, evidence=[])
            evidence = ScoredChunk(
                Chunk("fixture-chunk", "fixture-document", 2,
                      "Synthetic revenue was $100 million."),
                0.9,
            )
            return build_grounded_context(question, evidence=[evidence])

    token = secrets.token_urlsafe(32)
    app = api.create_app(service=SyntheticEvidenceService(), api_token=token)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.post("/v1/query", json={"question": "fixture"}).status_code == 401
        headers = {"Authorization": f"Bearer {token}", "X-Request-ID": "api-install-contract"}
        assert client.post("/v1/query", headers=headers,
                           json={"question": "   "}).status_code == 422
        response = client.post("/v1/query", headers=headers, json={"question": "fixture"})
        assert response.status_code == 200
        body = response.json()
        assert body["evidence_passed"] is True
        assert body["answer"] == ""
        assert body["evidence"][0]["chunk_id"] == "fixture-chunk"
        assert body["evidence"][0]["pdf_page"] == 2
        assert body["citations"] == ["fixture-document p.2"]
        assert body["request_id"] == response.headers["X-Request-ID"] == "api-install-contract"
        refusal = client.post("/v1/query", headers=headers,
                              json={"question": "unanswerable fixture"})
        assert refusal.status_code == 200
        assert refusal.json()["can_answer"] is False
        assert refusal.json()["evidence"] == []

    print(json.dumps({
        "status": "passed", "distribution": metadata.version("nvidia-financial-rag"),
        "python": sys.version.split()[0], "api_file": str(package_root),
        "httpx": metadata.version("httpx"), "fastapi": metadata.version("fastapi"),
        "uvicorn": metadata.version("uvicorn"),
        "absent": ["gradio", "torch", "sentence_transformers", "transformers"],
        "accepted": ["installed API import", "in-process 200/401/422",
                     "synthetic evidence-only response", "empty-evidence refusal",
                     "request ID serialization"],
        "not_accepted": ["real model/index", "financial quality", "network listener",
                         "runtime readiness", "load", "TLS", "Linux"],
    }, indent=2))


if __name__ == "__main__":
    main()
