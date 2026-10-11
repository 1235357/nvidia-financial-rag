"""Installed-wheel failure semantics; synthetic model transport, no model load.

Run with a fresh API-only environment from outside the checkout:
    <environment-python> -I -B <absolute-path-to-this-script>
"""

from hashlib import sha256
import argparse
from importlib import metadata, util
import json
from pathlib import Path
import secrets
import sys

import httpx
from fastapi.testclient import TestClient

from nvidia_rag import api, generation
from nvidia_rag.core import Chunk, ScoredChunk, build_grounded_context


def main(output: Path | None = None):
    package_file = Path(api.__file__).resolve()
    environment_root = Path(sys.prefix).resolve()
    assert package_file.is_relative_to(environment_root), "API must come from this environment"
    assert "site-packages" in package_file.parts
    for name in ("gradio", "torch", "transformers", "sentence_transformers"):
        assert util.find_spec(name) is None, f"unexpected heavyweight dependency: {name}"

    token = secrets.token_urlsafe(32)
    cases = []
    private_detail = "synthetic-private-upstream-detail"

    class EvidenceService:
        def query(self, question):
            evidence = () if question == "empty" else (
                ScoredChunk(Chunk("fixture-chunk", "fixture-report", 2,
                                  "Synthetic revenue was $100 million."), 0.9),
            )
            return build_grounded_context(question, evidence=evidence)

    scenarios = [
        ("timeout", 504, "generation_timeout", None),
        ("connection", 503, "generation_unavailable", None),
        ("upstream_503", 503, "generation_unavailable", None),
        ("invalid_shape", 503, "generation_unavailable", None),
        ("unfinished", 503, "generation_unavailable", None),
        ("abstention", 200, None, "The retrieved evidence is insufficient."),
        ("content_veto", 200, None, "Synthetic revenue was $999 million [fixture-report p.2]."),
        ("accepted", 200, None, "Synthetic revenue was $100 million [fixture-report p.2]."),
        ("empty", 200, None, None),
        ("token_counter_failure", 503, "generation_unavailable", None),
        ("prompt_budget", 200, None, None),
    ]
    for case, expected_status, expected_error, answer in scenarios:
        model_requests = []
        def handler(request):
            model_requests.append(request)
            if case in {"empty", "token_counter_failure", "prompt_budget"}:
                raise AssertionError("model must not run after a pre-generation guard")
            if case == "timeout":
                raise httpx.ReadTimeout(private_detail, request=request)
            if case == "connection":
                raise httpx.ConnectError(private_detail, request=request)
            if case == "upstream_503":
                return httpx.Response(503, text=private_detail)
            if case == "invalid_shape":
                return httpx.Response(200, json={"private": private_detail})
            return httpx.Response(200, json={"choices": [{
                "message": {"content": answer or private_detail},
                "finish_reason": "length" if case == "unfinished" else "stop",
            }]})

        generator = generation.OpenAICompatibleChatGenerator(
            base_url="http://127.0.0.1:8001", model_name="synthetic-model",
            api_token=secrets.token_urlsafe(24), transport=httpx.MockTransport(handler),
        )
        def token_count(prompt):
            if case == "token_counter_failure":
                raise RuntimeError(private_detail)
            return 101

        service = generation.GroundedRagService(
            evidence_service=EvidenceService(), generator=generator,
            **({"max_prompt_tokens": 100, "prompt_token_counter": token_count}
               if case in {"token_counter_failure", "prompt_budget"} else {}),
        )
        with TestClient(api.create_app(service=service, api_token=token, expose_docs=True)) as client:
            assert client.post("/v1/query", json={"question": "fixture"}).status_code == 401
            headers = {"Authorization": f"Bearer {token}", "X-Request-ID": "installed-failure-contract"}
            assert client.post("/v1/query", headers=headers, json={"question": " "}).status_code == 422
            response = client.post("/v1/query", headers=headers,
                                   json={"question": "empty" if case == "empty" else "Revenue?"})
            assert response.status_code == expected_status
            body = response.json()
            assert body["error_code"] == expected_error
            assert body["can_answer"] is (case == "accepted")
            assert body["answer"] == (answer if case == "accepted" else "")
            assert body["evidence_passed"] is (case != "empty")
            assert len(body["evidence"]) == (0 if case == "empty" else 1)
            assert body["citations"] == ([] if case == "empty" else ["fixture-report p.2"])
            assert body["request_id"] == response.headers["X-Request-ID"] == "installed-failure-contract"
            assert private_detail not in response.text
            openapi = client.get("/openapi.json").json()
            documented = openapi["paths"]["/v1/query"]["post"]["responses"]
            assert "503" in documented and "504" in documented
        assert len(model_requests) == (0 if case in {"empty", "token_counter_failure", "prompt_budget"} else 1)
        cases.append({"case": case, "http_status": expected_status, "error_code": expected_error})

    report = {
        "status": "passed", "python": sys.version.split()[0],
        "distribution": metadata.version("nvidia-financial-rag"),
        "api_file": str(package_file),
        "api_sha256": sha256(package_file.read_bytes()).hexdigest(),
        "generation_sha256": sha256(Path(generation.__file__).read_bytes()).hexdigest(),
        "httpx": metadata.version("httpx"), "fastapi": metadata.version("fastapi"),
        "cases": cases,
        "not_accepted": ["real model/index", "financial correctness", "runtime readiness",
                         "network listener", "load", "TLS", "Linux"],
    }
    if output is not None:
        # Library warnings may use stdout; write the JSON directly so the
        # receipt remains machine-readable without hiding those warnings.
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Create a new JSON receipt without overwriting")
    main(parser.parse_args().output)
