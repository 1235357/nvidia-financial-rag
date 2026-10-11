"""Launch the authenticated RAG API on loopback by default."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nvidia_rag.api import create_app, load_api_provenance, resolve_api_token
from nvidia_rag.embedding_store import load_passage_embedding_artifact
from nvidia_rag.generation import (
    GroundedRagService,
    OpenAICompatibleChatGenerator,
    TransformersChatGenerator,
)
from nvidia_rag.ingest import load_chunks
from nvidia_rag.sources import load_source_manifest
from nvidia_rag.runtime import (
    BGE_MODEL_NAME,
    BGE_MODEL_REVISION,
    build_bge_service,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("artifacts/annual-fy2022-fy2026-fixed-400-50/chunks.jsonl"),
    )
    parser.add_argument("--embedding-model", default=BGE_MODEL_NAME)
    parser.add_argument("--embedding-model-revision", default=BGE_MODEL_REVISION)
    parser.add_argument(
        "--passage-embeddings",
        type=Path,
        default=Path(
            "artifacts/annual-fy2022-fy2026-fixed-400-50/bge-small-en-v1.5.npy"
        ),
    )
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=Path(
            "artifacts/annual-fy2022-fy2026-fixed-400-50/embedding-manifest.json"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="BGE embedding device (use --device cpu on hosts without CUDA).",
    )
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--generator",
        choices=("none", "qwen-local", "openai-compatible"),
        default="none",
    )
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--generator-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--generator-base-url",
        default="http://127.0.0.1:8001",
        help="Loopback endpoint of an SSH-forwarded OpenAI-compatible server.",
    )
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--expose-docs", action="store_true")
    parser.add_argument(
        "--corpus-manifest", type=Path, default=None,
        help="Isolated corpus manifest; defaults to manifest.json beside chunks.",
    )
    parser.add_argument('--source-task', choices=('reported_results', 'management_commentary', 'third_party_view'),
                        default=None, help='Opt-in explicit source scope; disables annual-only routing.')
    parser.add_argument('--source-registry', type=Path, default=None,
                        help='Reviewed schema-v2 source registry matching the isolated corpus provenance.')
    args = parser.parse_args()
    if (args.source_task is None) != (args.source_registry is None):
        parser.error('--source-task and --source-registry must be supplied together')
    return args


def load_task_registry(registry_path: Path, *, corpus_manifest: Path, chunks_path: Path):
    """Require exact registration identity before loading GPU models or serving."""
    provenance = load_api_provenance(corpus_manifest, chunks_path=chunks_path)
    records = list(load_source_manifest(registry_path))
    registered = {record.document_id: record.provenance for record in records}
    if not provenance or registered != provenance:
        raise ValueError('source registry must exactly match reviewed corpus provenance')
    documents = json.loads(corpus_manifest.read_text(encoding='utf-8'))['documents']
    hashes = {row['document_id']: row.get('source_sha256') for row in documents}
    if any(hashes.get(record.document_id) != record.sha256 for record in records):
        raise ValueError('source registry hashes must match corpus source hashes')
    return records


if __name__ == "__main__":
    args = parse_args()
    # Reject missing credentials before corpus IO or expensive model loading.
    api_token = resolve_api_token(os.environ)
    # Optional serving/model dependencies are required only after configuration
    # is valid; importing this launcher must not eagerly load their runtimes.
    from sentence_transformers import SentenceTransformer
    import uvicorn

    corpus_manifest = args.corpus_manifest or args.chunks.parent / "manifest.json"
    source_provenance = load_api_provenance(corpus_manifest, chunks_path=args.chunks)
    source_records = None
    if args.source_task is not None:
        source_records = load_task_registry(args.source_registry,
            corpus_manifest=corpus_manifest, chunks_path=args.chunks)
    chunks = load_chunks(args.chunks)
    embedding_model = SentenceTransformer(
        args.embedding_model,
        revision=args.embedding_model_revision,
        device=args.device,
        local_files_only=not args.allow_model_download,
    )
    passage_embeddings = load_passage_embedding_artifact(
        chunks=chunks,
        chunks_path=args.chunks,
        embeddings_path=args.passage_embeddings,
        manifest_path=args.embedding_manifest,
        expected_model_name=args.embedding_model,
        expected_model_revision=args.embedding_model_revision,
    )
    evidence_service = build_bge_service(
        chunks=chunks,
        model=embedding_model,
        threshold=args.threshold,
        top_k=args.top_k,
        route_fiscal_year=args.source_task is None,
        passage_embeddings=passage_embeddings,
        source_task=args.source_task,
        source_records=source_records,
    )
    service = evidence_service
    if args.generator == "qwen-local":
        service = GroundedRagService(
            evidence_service=evidence_service,
            generator=TransformersChatGenerator(
                args.generator_model,
                device=args.generator_device,
                local_files_only=not args.allow_model_download,
            ),
        )
    elif args.generator == "openai-compatible":
        model_token = os.environ.get("NVIDIA_RAG_MODEL_TOKEN", "")
        service = GroundedRagService(
            evidence_service=evidence_service,
            generator=OpenAICompatibleChatGenerator(
                base_url=args.generator_base_url,
                model_name=args.generator_model,
                api_token=model_token,
            ),
        )
    app = create_app(
        service=service,
        api_token=api_token,
        expose_docs=args.expose_docs,
        source_provenance=source_provenance,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        access_log=False,
        proxy_headers=False,
        server_header=False,
    )
