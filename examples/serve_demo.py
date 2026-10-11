"""Run an evidence-only API over generated transport documentation.

No PDF, model, financial answer, remote service or download is involved at runtime.
Install the package's [api] extra and supply NVIDIA_RAG_API_TOKEN privately.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import os

from nvidia_rag.api import create_app, resolve_api_token
from nvidia_rag.core import Chunk
from nvidia_rag.retrieval import TfidfRetriever
from nvidia_rag.service import EvidenceService


DEMO_CHUNKS = (
    Chunk("synthetic-release-demo:p1:authentication", "synthetic-release-demo", 1,
          "Synthetic transport fixture. Bearer authentication is required for queries."),
    Chunk("synthetic-release-demo:p2:metadata", "synthetic-release-demo", 2,
          "Synthetic transport fixture. Responses contain evidence citations and request identifiers."),
)


def create_demo_app(environment: Mapping[str, str], *, expose_docs: bool = False):
    token = resolve_api_token(environment)
    service = EvidenceService(retriever=TfidfRetriever(DEMO_CHUNKS), threshold=0.2, top_k=1)
    return create_app(service=service, api_token=token, expose_docs=expose_docs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--expose-docs", action="store_true")
    args = parser.parse_args()
    app = create_demo_app(os.environ, expose_docs=args.expose_docs)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, access_log=False,
                proxy_headers=False, server_header=False, limit_concurrency=16,
                timeout_keep_alive=5)


if __name__ == "__main__":
    main()
