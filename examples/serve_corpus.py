"""Serve a verified local financial corpus as a lexical evidence-only API.

This compatibility recipe keeps its original loopback-only listen contract.
Use the installed nvidia-rag-serve-corpus CLI for an explicit container host.
"""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

from nvidia_rag.api import create_app, resolve_api_token
from nvidia_rag.corpus import build_corpus_service, load_verified_corpus
from nvidia_rag.serving import _create_corpus_app, _parse_args, _run_app


def create_corpus_app(environment: Mapping[str, str], *, artifact_dir: Path,
                      source_registry: Path, threshold: float, top_k: int,
                      document_ids: frozenset[str] | None, expose_docs: bool = False,
                      statement_source_dir: Path | None = None,
                      statement_fiscal_declarations: Path | None = None):
    return _create_corpus_app(environment, artifact_dir=artifact_dir,
        source_registry=source_registry, threshold=threshold, top_k=top_k,
        document_ids=document_ids, expose_docs=expose_docs,
        statement_source_dir=statement_source_dir,
        statement_fiscal_declarations=statement_fiscal_declarations,
        corpus_loader=load_verified_corpus, service_builder=build_corpus_service,
        app_factory=create_app)


def parse_args(argv=None):
    return _parse_args(argv, allow_host=False)


def main() -> None:
    args = parse_args()
    try:
        app = create_corpus_app(os.environ, artifact_dir=args.artifact_dir,
            source_registry=args.source_registry, threshold=args.threshold, top_k=args.top_k,
            document_ids=None if args.document_id is None else frozenset(args.document_id),
            expose_docs=args.expose_docs, statement_source_dir=args.statement_source_dir,
            statement_fiscal_declarations=args.statement_fiscal_declarations)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
    _run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
