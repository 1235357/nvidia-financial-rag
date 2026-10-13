"""Serve a verified local financial corpus through an installed API entry point.

The lexical query route returns source evidence. Optional reviewed fiscal
declarations enable a separate immutable statement source-candidate index;
neither route establishes financial correctness or acquires PDF/model files.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path

from .api import create_app, resolve_api_token
from .corpus import build_corpus_service, load_verified_corpus
from .sources import file_sha256

_CORPUS_FILES = ("manifest.json", "chunks.jsonl", "build-receipt.json")


def _absolute(path: Path, name: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{name} must be an absolute local path")
    return path.resolve(strict=True)


def _completion(directory: Path) -> None:
    if os.path.lexists(directory / "pending.json"):
        raise ValueError("corpus build is still pending")
    if os.path.lexists(directory / "failure.json"):
        raise ValueError("corpus has a failed build marker")


@dataclass(frozen=True, slots=True)
class _CorpusClosure:
    artifact_dir: Path
    source_registry: Path
    directory: Path
    registry: Path
    hashes: tuple[tuple[Path, str], ...]

    def verify_unchanged(self) -> None:
        if (self.artifact_dir.resolve(strict=True) != self.directory
                or self.source_registry.resolve(strict=True) != self.registry):
            raise ValueError("corpus configuration changed during startup")
        _completion(self.directory)
        for path, expected in self.hashes:
            if path != self.registry and (
                path.is_symlink() or path.resolve(strict=True).parent != self.directory
            ):
                raise ValueError("corpus artifact closure changed during startup")
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError("corpus input bytes changed during startup")
        _completion(self.directory)


def _capture_corpus_closure(*, artifact_dir: Path, source_registry: Path) -> _CorpusClosure:
    directory = _absolute(artifact_dir, "artifact_dir")
    registry = _absolute(source_registry, "source_registry")
    if not directory.is_dir() or not registry.is_file():
        raise ValueError("corpus configuration requires a local directory and registry file")
    _completion(directory)
    files = [registry]
    for name in _CORPUS_FILES:
        path = directory / name
        if path.is_symlink() or path.resolve(strict=True).parent != directory or not path.is_file():
            raise ValueError("corpus artifact must remain a local file inside its directory")
        files.append(path)
    closure = _CorpusClosure(artifact_dir, source_registry, directory, registry,
                             tuple((path, file_sha256(path)) for path in files))
    closure.verify_unchanged()
    return closure


def _paired_statement_options(source_dir: Path | None, declarations: Path | None) -> None:
    if (source_dir is None) != (declarations is None):
        raise ValueError("statement_source_dir and statement_fiscal_declarations must be supplied together")


def _create_corpus_app(environment: Mapping[str, str], *, artifact_dir: Path,
                       source_registry: Path, threshold: float, top_k: int,
                       document_ids: frozenset[str] | None, expose_docs: bool = False,
                       statement_source_dir: Path | None = None,
                       statement_fiscal_declarations: Path | None = None,
                       corpus_loader, service_builder, app_factory):
    """Shared assembly used by the package and the compatible loopback recipe."""
    token = resolve_api_token(environment)
    _paired_statement_options(statement_source_dir, statement_fiscal_declarations)
    closure = _capture_corpus_closure(artifact_dir=artifact_dir, source_registry=source_registry)
    corpus = corpus_loader(artifact_dir=artifact_dir, source_registry=source_registry)
    service = service_builder(corpus, threshold=threshold, top_k=top_k, document_ids=document_ids)
    snapshot = None
    if statement_source_dir is not None:
        from .statement_startup import load_statement_snapshot
        snapshot = load_statement_snapshot(source_registry=source_registry,
            source_dir=statement_source_dir, fiscal_declarations=statement_fiscal_declarations,
            document_ids=document_ids, expected_sources=corpus.sources)
    app = app_factory(service=service, api_token=token, expose_docs=expose_docs,
                      statement_index=None if snapshot is None else snapshot.index)
    if snapshot is not None:
        snapshot.verify_unchanged()
    closure.verify_unchanged()
    return app


def create_corpus_app(environment: Mapping[str, str], *, artifact_dir: Path,
                      source_registry: Path, threshold: float, top_k: int,
                      document_ids: frozenset[str] | None, expose_docs: bool = False,
                      statement_source_dir: Path | None = None,
                      statement_fiscal_declarations: Path | None = None):
    """Build an API only after exact corpus and optional statement startup checks.

Both statement options are required to enable statement candidates. Invalid
configuration raises instead of silently returning a lexical-only application.
Document scope remains the existing explicit lexical corpus scope.
"""
    return _create_corpus_app(environment, artifact_dir=artifact_dir,
        source_registry=source_registry, threshold=threshold, top_k=top_k,
        document_ids=document_ids, expose_docs=expose_docs,
        statement_source_dir=statement_source_dir,
        statement_fiscal_declarations=statement_fiscal_declarations,
        corpus_loader=load_verified_corpus, service_builder=build_corpus_service,
        app_factory=create_app)


def _parse_args(argv=None, *, allow_host: bool = True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--document-id", action="append", default=None)
    parser.add_argument("--threshold", type=float, required=True,
        help="Operator setting; 0 is ungated lexical inspection, not calibrated financial acceptance")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--expose-docs", action="store_true")
    parser.add_argument("--statement-source-dir", type=Path)
    parser.add_argument("--statement-fiscal-declarations", type=Path)
    if allow_host:
        parser.add_argument("--host", default="127.0.0.1",
            help="Listening address; select 0.0.0.0 explicitly for a container")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.top_k < 1:
        parser.error("top-k must be a positive integer")
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("threshold must be finite and between 0 and 1")
    if allow_host and (not args.host or args.host != args.host.strip()):
        parser.error("host must be a nonblank address without surrounding whitespace")
    try:
        _paired_statement_options(args.statement_source_dir, args.statement_fiscal_declarations)
    except ValueError as error:
        parser.error(str(error))
    return args


def parse_args(argv=None):
    """Parse the installed CLI, including an explicit container listen address."""
    return _parse_args(argv)


def _run_app(app, *, host: str, port: int) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, access_log=False,
                proxy_headers=False, server_header=False, limit_concurrency=16, timeout_keep_alive=5)


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
    _run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
