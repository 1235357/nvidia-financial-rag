"""Serve a verified local financial corpus as a lexical evidence-only API.

Source rights remain operator responsibilities. No PDF/model download, generated
answer, or natural-language reporting-period resolution is part of this recipe.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path

from nvidia_rag.api import create_app, resolve_api_token
from nvidia_rag.corpus import build_corpus_service, load_verified_corpus


def create_corpus_app(environment: Mapping[str, str], *, artifact_dir: Path,
                      source_registry: Path, threshold: float, top_k: int,
                      document_ids: frozenset[str] | None, expose_docs: bool = False):
    token = resolve_api_token(environment)
    corpus = load_verified_corpus(artifact_dir=artifact_dir, source_registry=source_registry)
    service = build_corpus_service(corpus, threshold=threshold, top_k=top_k, document_ids=document_ids)
    return create_app(service=service, api_token=token, expose_docs=expose_docs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--document-id", action="append", default=None)
    parser.add_argument("--threshold", type=float, required=True,
                        help="Operator setting; 0 is ungated lexical inspection, not calibrated financial acceptance")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--expose-docs", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    return args


def main() -> None:
    args = parse_args()
    try:
        app = create_corpus_app(os.environ, artifact_dir=args.artifact_dir,
            source_registry=args.source_registry, threshold=args.threshold, top_k=args.top_k,
            document_ids=None if args.document_id is None else frozenset(args.document_id),
            expose_docs=args.expose_docs)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False,
                proxy_headers=False, server_header=False, limit_concurrency=16, timeout_keep_alive=5)


if __name__ == "__main__":
    main()
