"""Check a running generated-data demo over real HTTP; no financial score."""
from __future__ import annotations

import argparse
import os
import time
from urllib.parse import urlsplit

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username or url.password or url.query or url.fragment):
        parser.error("This acceptance is restricted to a loopback HTTP demo")
    token = os.environ.get("NVIDIA_RAG_API_TOKEN", "")
    if len(token) < 32:
        parser.error("Supply an ephemeral NVIDIA_RAG_API_TOKEN of at least 32 characters")
    with httpx.Client(base_url=args.base_url, timeout=5, follow_redirects=False,
                      trust_env=False) as client:
        deadline = time.monotonic() + 30
        while True:
            try:
                response = client.get("/healthz")
                response.raise_for_status()
                break
            except httpx.HTTPError:
                if time.monotonic() >= deadline:
                    raise AssertionError("The local demo did not become live") from None
                time.sleep(0.2)
        assert client.get("/docs").status_code == 404
        assert client.post("/v1/query", json={"question": "authentication"}).status_code == 401
        headers = {"Authorization": f"Bearer {token}", "X-Request-ID": "generated-demo-http"}
        assert client.post("/v1/query", headers=headers, json={"question": " "}).status_code == 422
        response = client.post("/v1/query", headers=headers,
                               json={"question": "bearer authentication"})
        response.raise_for_status()
        body = response.json()
        assert body["can_answer"] is True and body["evidence_passed"] is True
        assert body["answer"] == "" and body["error_code"] is None
        assert body["evidence"][0]["document_id"] == "synthetic-release-demo"
        assert body["citations"] == ["synthetic-release-demo p.1"]
        assert body["request_id"] == response.headers["X-Request-ID"] == "generated-demo-http"
        response = client.post("/v1/query", headers=headers,
                               json={"question": "zxqv plmokn"})
        response.raise_for_status()
        body = response.json()
        assert body["can_answer"] is False and body["answer"] == ""
        assert body["evidence"] == [] and body["citations"] == []
    print("PASS: real HTTP generated-data demo, authentication, schema, evidence, refusal and request ID")
    print("Scope: evidence-only transport; no model, financial accuracy, load, TLS or readiness acceptance")


if __name__ == "__main__":
    main()
