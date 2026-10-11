"""Cached CUDA encoder plus read-only real index; not answer-quality acceptance."""
import argparse
from pathlib import Path

import nvidia_rag
from sentence_transformers import SentenceTransformer
from nvidia_rag.ingest import load_chunks
from nvidia_rag.embedding_store import load_passage_embedding_artifact
from nvidia_rag.runtime import BGE_MODEL_NAME, BGE_MODEL_REVISION, build_bge_service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--question', default='What was NVIDIA annual revenue in FY2025?')
    args = parser.parse_args()
    assert 'site-packages' in Path(nvidia_rag.__file__).resolve().parts
    path = args.artifact_dir.resolve(strict=True)
    chunks = load_chunks(path / 'chunks.jsonl')
    vectors = load_passage_embedding_artifact(
        chunks=chunks, chunks_path=path / 'chunks.jsonl',
        embeddings_path=path / 'bge-small-en-v1.5.npy',
        manifest_path=path / 'embedding-manifest.json',
        expected_model_name=BGE_MODEL_NAME, expected_model_revision=BGE_MODEL_REVISION)
    model = SentenceTransformer(BGE_MODEL_NAME, revision=BGE_MODEL_REVISION,
                                device='cuda', local_files_only=True)
    service = build_bge_service(chunks=chunks, model=model, threshold=0.78, top_k=5,
                                route_fiscal_year=True, passage_embeddings=vectors)
    trace = service.inspect(args.question)
    print(f'Device: {model.device}; similarity gate: {trace.context.can_answer}')
    for candidate in trace.retrieval.candidates:
        print(candidate.chunk.chunk_id, candidate.chunk.document_id, candidate.chunk.page_number,
              f'{candidate.score:.6f}')
    print('NOT VERIFIED: financial answer correctness, API transport or production quality')


if __name__ == '__main__':
    main()
