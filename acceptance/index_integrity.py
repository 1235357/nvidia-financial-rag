"""Read-only artifact acceptance; no model loading or answer-quality claim."""
import argparse
from pathlib import Path

import nvidia_rag
from nvidia_rag.embedding_store import load_passage_embedding_artifact
from nvidia_rag.ingest import load_chunks
from nvidia_rag.runtime import BGE_MODEL_NAME, BGE_MODEL_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', required=True, type=Path)
    args = parser.parse_args()
    assert 'site-packages' in Path(nvidia_rag.__file__).resolve().parts
    artifact = args.artifact_dir.resolve(strict=True)
    chunks_path = artifact / 'chunks.jsonl'
    chunks = load_chunks(chunks_path)
    matrix = load_passage_embedding_artifact(
        chunks=chunks,
        chunks_path=chunks_path,
        embeddings_path=artifact / 'bge-small-en-v1.5.npy',
        manifest_path=artifact / 'embedding-manifest.json',
        expected_model_name=BGE_MODEL_NAME,
        expected_model_revision=BGE_MODEL_REVISION,
    )
    print(f'PASS: {len(chunks)} chunks; matrix {matrix.shape}; dtype {matrix.dtype}')
    print(f'PASS: exact file hashes, ordered chunk IDs, model identity/revision and unit norms')
    print('NOT ACCEPTED: source truth, retrieval quality, runtime model loading or deployment')


if __name__ == '__main__':
    main()
