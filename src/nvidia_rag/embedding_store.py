"""Versioned, integrity-checked passage embedding artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
from typing import Protocol

import numpy as np

from .core import Chunk


class EmbeddingModel(Protocol):
    def encode(self, texts: Sequence[str], **kwargs): ...


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_ids_sha256(chunks: Sequence[Chunk]) -> str:
    payload = "\n".join(chunk.chunk_id for chunk in chunks).encode("utf-8")
    return sha256(payload).hexdigest()


def _validate_matrix(matrix: np.ndarray, chunk_count: int) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != chunk_count or matrix.shape[1] <= 0:
        raise ValueError("embedding matrix shape does not match the chunk artifact")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding matrix contains a non-finite value")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("embedding matrix contains a zero vector")
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("embedding matrix violates the unit norm contract")
    return matrix


def _build_runtime() -> dict[str, str]:
    packages = (
        "numpy",
        "sentence-transformers",
        "torch",
        "transformers",
    )
    return {
        "python": platform.python_version(),
        **{name: metadata.version(name) for name in packages},
    }


def build_passage_embedding_artifact(
    *,
    chunks: Sequence[Chunk],
    chunks_path: Path,
    embeddings_path: Path,
    manifest_path: Path,
    model: EmbeddingModel,
    model_name: str,
    model_revision: str,
) -> dict[str, object]:
    """Encode passages once and bind the matrix to exact chunks and model metadata."""
    if not chunks:
        raise ValueError("at least one chunk is required")
    chunks_path = chunks_path.resolve(strict=True)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(
        model.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    matrix = _validate_matrix(matrix, len(chunks))
    np.save(embeddings_path, matrix, allow_pickle=False)
    manifest: dict[str, object] = {
        "schema_version": 2,
        "model_name": model_name,
        "model_revision": model_revision,
        "build_runtime": _build_runtime(),
        "normalized": True,
        "chunk_count": len(chunks),
        "chunk_ids_sha256": _chunk_ids_sha256(chunks),
        "chunks_sha256": _sha256_file(chunks_path),
        "embeddings_sha256": _sha256_file(embeddings_path),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_passage_embedding_artifact(
    *,
    chunks: Sequence[Chunk],
    chunks_path: Path,
    embeddings_path: Path,
    manifest_path: Path,
    expected_model_name: str,
    expected_model_revision: str,
) -> np.ndarray:
    """Load only when file hashes, chunk order, shape, and model identity all match."""
    chunks_path = chunks_path.resolve(strict=True)
    embeddings_path = embeddings_path.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported embedding manifest schema")
    if manifest.get("normalized") is not True:
        raise ValueError("embedding normalization contract is missing or false")
    if manifest.get("model_name") != expected_model_name:
        raise ValueError("embedding model identity does not match the runtime model")
    if manifest.get("model_revision") != expected_model_revision:
        raise ValueError("embedding model revision does not match the runtime model")
    checks = {
        "chunk_count": len(chunks),
        "chunk_ids_sha256": _chunk_ids_sha256(chunks),
        "chunks_sha256": _sha256_file(chunks_path),
        "embeddings_sha256": _sha256_file(embeddings_path),
    }
    for key, observed in checks.items():
        if manifest.get(key) != observed:
            raise ValueError(f"embedding artifact integrity check failed: {key}")
    matrix = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    matrix = _validate_matrix(matrix, len(chunks))
    if list(matrix.shape) != manifest.get("shape") or str(matrix.dtype) != manifest.get(
        "dtype"
    ):
        raise ValueError("embedding matrix metadata does not match the manifest")
    return matrix
