"""Disposable synthetic artifacts exercise the installed loader, not financial truth."""
import hashlib
import json

import numpy as np
import pytest

from nvidia_rag.core import Chunk
from nvidia_rag.embedding_store import load_passage_embedding_artifact


def fixture(tmp_path):
    chunks = [Chunk('a', 'synthetic', 1, 'alpha'), Chunk('b', 'synthetic', 2, 'beta')]
    source = tmp_path / 'chunks.jsonl'
    source.write_text('synthetic fixture bytes\n', encoding='utf-8')
    vectors = tmp_path / 'vectors.npy'
    np.save(vectors, np.eye(2, dtype=np.float32), allow_pickle=False)
    manifest = dict(schema_version=2, normalized=True, model_name='synthetic/model',
                    model_revision='fixture-revision', chunk_count=2,
                    chunk_ids_sha256=hashlib.sha256(b'a\nb').hexdigest(),
                    chunks_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    embeddings_sha256=hashlib.sha256(vectors.read_bytes()).hexdigest(),
                    shape=[2, 2], dtype='float32')
    metadata = tmp_path / 'manifest.json'
    metadata.write_text(json.dumps(manifest), encoding='utf-8')
    return chunks, source, vectors, metadata, manifest


def load(parts, **changes):
    chunks, source, vectors, metadata, _ = parts
    args = dict(chunks=chunks, chunks_path=source, embeddings_path=vectors,
                manifest_path=metadata, expected_model_name='synthetic/model',
                expected_model_revision='fixture-revision')
    args.update(changes)
    return load_passage_embedding_artifact(**args)


def test_valid_fixture(tmp_path):
    assert load(fixture(tmp_path)).shape == (2, 2)


@pytest.mark.parametrize('field,value,error', [
    ('expected_model_name', 'different/model', 'model identity'),
    ('expected_model_revision', 'different-revision', 'model revision'),
])
def test_wrong_model(tmp_path, field, value, error):
    with pytest.raises(ValueError, match=error):
        load(fixture(tmp_path), **{field: value})


def test_reordered_chunks(tmp_path):
    parts = fixture(tmp_path)
    with pytest.raises(ValueError, match='chunk_ids_sha256'):
        load(parts, chunks=list(reversed(parts[0])))


def test_changed_vector_bytes(tmp_path):
    parts = fixture(tmp_path)
    np.save(parts[2], np.array([[0, 1], [1, 0]], dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match='embeddings_sha256'):
        load(parts)


@pytest.mark.parametrize('matrix,error', [
    ([[2, 0], [0, 1]], 'unit norm'),
    ([[0, 0], [0, 1]], 'zero vector'),
    ([[float('nan'), 0], [0, 1]], 'non-finite'),
])
def test_matching_hash_is_not_numeric_validity(tmp_path, matrix, error):
    parts = fixture(tmp_path)
    np.save(parts[2], np.array(matrix, dtype=np.float32), allow_pickle=False)
    # Deliberately update the fixture manifest: the numeric contract must still fail.
    parts[4]['embeddings_sha256'] = hashlib.sha256(parts[2].read_bytes()).hexdigest()
    parts[3].write_text(json.dumps(parts[4]), encoding='utf-8')
    with pytest.raises(ValueError, match=error):
        load(parts)
