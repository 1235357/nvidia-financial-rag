"""Separate local BGE artifacts bound to a completed read-only corpus.

Receipts check unsigned internal consistency. Injected models, actual cached
execution, installed wheel identity and financial quality are separate claims.
Model packages are imported only by the explicitly cached builder after preflight.
"""
from __future__ import annotations

from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sysconfig
import zipfile

import numpy as np

from .corpus import load_verified_corpus
from .embedding_store import (EmbeddingModel, build_passage_embedding_artifact,
                              load_passage_embedding_artifact)
from .runtime import BGE_MODEL_NAME, BGE_MODEL_REVISION
from .sources import file_sha256

_VECTOR = "bge-small-en-v1.5.npy"
_MANIFEST = "embedding-manifest.json"
_RECEIPT = "vector-build-receipt.json"
_PENDING = "pending.json"
_CORPUS_FILES = ("manifest.json", "chunks.jsonl", "build-receipt.json")
_RUNTIME = {"python", "numpy", "sentence-transformers", "torch", "transformers"}
_BOUND_MODULES = {"__init__.py", "core.py", "ingest.py", "sources.py", "corpus.py",
                  "embedding_store.py", "corpus_embeddings.py", "runtime.py",
                  "retrieval.py", "service.py"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, expected, label):
    _require(type(value) is dict and value.keys() == set(expected), f"invalid {label} fields")


def _digest(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "invalid SHA-256 digest")


def _integer(value, expected, label):
    _require(type(value) is int and value == expected, f"invalid {label} integer")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON member")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("non-finite JSON constant")


def _json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=_pairs, parse_constant=_constant)


def _absolute(path, label, *, exists=False):
    _require(isinstance(path, Path) and path.is_absolute(), f"{label} must be an absolute local path")
    return path.resolve(strict=exists)


def _file(directory, name):
    path = directory / name
    _require(not path.is_symlink() and path.resolve(strict=True).parent == directory and path.is_file(),
             "artifact file escapes its directory")
    return path


def _snapshot(directory, registry):
    _require(directory.is_dir() and registry.is_file(), "corpus directory/registry is absent")
    result = {"source_registry_sha256": file_sha256(registry)}
    for name in _CORPUS_FILES:
        result[name.replace(".jsonl", "").replace(".json", "").replace("-", "_") + "_sha256"] = file_sha256(_file(directory, name))
    _require(not os.path.lexists(directory / "failure.json"), "corpus has a failed build marker")
    _require(not os.path.lexists(directory / _PENDING), "corpus completion is pending")
    return result


def _unchanged(directory, registry, expected):
    _require(_snapshot(directory, registry) == expected, "read-only corpus closure changed")


def _load_corpus(directory, registry):
    before = _snapshot(directory, registry)
    corpus = load_verified_corpus(artifact_dir=directory, source_registry=registry)
    _unchanged(directory, registry, before)
    _require(corpus.chunks_sha256 == before["chunks_sha256"], "loaded chunks differ from corpus closure")
    return corpus, before


def _separate(vector, directory, registry):
    _require(not (vector.is_relative_to(directory) or directory.is_relative_to(vector)
                  or registry.is_relative_to(vector) or vector.is_relative_to(registry)),
             "vector output overlaps read-only corpus or source registry")


def _output_preflight(artifact_dir, source_registry, vector_dir):
    vector = _absolute(vector_dir, "vector_dir")
    _require(not os.path.lexists(vector_dir) and not os.path.lexists(vector),
             "vector directory already exists")
    directory = _absolute(artifact_dir, "artifact_dir", exists=True)
    registry = _absolute(source_registry, "source_registry", exists=True)
    _separate(vector, directory, registry)
    corpus, closure = _load_corpus(directory, registry)
    return vector, directory, registry, corpus, closure


def _chunk_ids(chunks):
    return sha256("\n".join(chunk.chunk_id for chunk in chunks).encode("utf-8")).hexdigest()


def _runtime(value):
    _keys(value, _RUNTIME, "embedding runtime")
    _require(all(type(item) is str and item == item.strip() and bool(item) for item in value.values()),
             "runtime versions must be nonblank strings")


def _manifest(value, corpus, vectors):
    _keys(value, {"schema_version", "model_name", "model_revision", "build_runtime", "normalized",
                  "chunk_count", "chunk_ids_sha256", "chunks_sha256", "embeddings_sha256", "shape", "dtype"},
          "embedding manifest")
    _integer(value["schema_version"], 2, "embedding schema")
    _require(value["model_name"] == BGE_MODEL_NAME and value["model_revision"] == BGE_MODEL_REVISION,
             "embedding model identity/revision mismatch")
    _runtime(value["build_runtime"])
    _require(value["normalized"] is True, "embedding normalization must be true")
    _integer(value["chunk_count"], len(corpus.chunks), "embedding chunk count")
    shape = value["shape"]
    _require(type(shape) is list and len(shape) == 2, "embedding shape must be an array")
    _integer(shape[0], len(corpus.chunks), "embedding row count")
    _integer(shape[1], 384, "embedding dimension")
    _require(value["dtype"] == "float32", "embedding dtype must be float32")
    for key, expected in (("chunk_ids_sha256", _chunk_ids(corpus.chunks)),
                          ("chunks_sha256", corpus.chunks_sha256),
                          ("embeddings_sha256", file_sha256(vectors))):
        _digest(value[key])
        _require(value[key] == expected, f"embedding {key} digest mismatch")


def _matrix(value, count):
    _require(value.dtype == np.dtype("float32"), "matrix dtype must be float32")
    _require(value.shape == (count, 384), "matrix shape must be [chunk_count,384]")
    _require(np.isfinite(value).all(), "matrix contains a non-finite value")
    norms = np.linalg.norm(value, axis=1)
    _require(np.all(norms > 0) and np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5),
             "matrix contains zero or non-unit vectors")


def _load_matrix(vector, directory, corpus):
    path, manifest_path = _file(vector, _VECTOR), _file(vector, _MANIFEST)
    vector_hash, manifest_hash = file_sha256(path), file_sha256(manifest_path)
    manifest = _json(manifest_path)
    _manifest(manifest, corpus, path)
    matrix = load_passage_embedding_artifact(chunks=corpus.chunks,
        chunks_path=directory / "chunks.jsonl", embeddings_path=path, manifest_path=manifest_path,
        expected_model_name=BGE_MODEL_NAME, expected_model_revision=BGE_MODEL_REVISION)
    _matrix(matrix, len(corpus.chunks))
    _require(file_sha256(_file(vector, _VECTOR)) == vector_hash
             and file_sha256(_file(vector, _MANIFEST)) == manifest_hash,
             "vector artifact changed while loading")
    return matrix, manifest, vector_hash, manifest_hash


def _module_hashes():
    directory = Path(__file__).resolve().parent
    return {name: file_sha256(_file(directory, name)) for name in sorted(_BOUND_MODULES)}


def _verify_installed_wheel(wheel):
    path = _absolute(wheel, "wheel", exists=True)
    _require(path.is_file() and path.name == "nvidia_financial_rag-0.1.0-py3-none-any.whl",
             "unexpected named wheel")
    package = Path(__file__).resolve().parent
    _require(package.parent == Path(sysconfig.get_paths()["purelib"]).resolve(),
             "run with the named wheel installed in interpreter purelib")
    before = file_sha256(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "duplicate wheel ZIP member")
        modules = {name for name in names if name.startswith("nvidia_rag/") and name.endswith(".py")}
        _require({"nvidia_rag/" + name for name in _BOUND_MODULES} <= modules,
                 "wheel vector interface modules are absent")
        _require(all(len(Path(name).parts) == 2 for name in modules), "unsupported wheel nested modules")
        _require(modules == {"nvidia_rag/" + item.name for item in package.glob("*.py")},
                 "wheel/installed module set mismatch")
        hashes = {Path(name).name: sha256(archive.read(name)).hexdigest() for name in sorted(modules)}
        for name, expected in hashes.items():
            _require(file_sha256(_file(package, name)) == expected, "installed module differs from named wheel")
    _require(file_sha256(path) == before, "wheel changed during identity verification")
    return {"wheel_basename": path.name, "wheel_bytes": path.stat().st_size,
            "wheel_sha256": before, "distribution_version": metadata.version("nvidia-financial-rag"),
            "module_sha256": hashes}


def _write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _failure(vector, error):
    kind = type(error).__name__
    if kind not in {"ValueError", "OSError", "RuntimeError", "FileNotFoundError", "PermissionError",
                    "ImportError", "ModuleNotFoundError"}:
        kind = "BuildError"
    try:
        _write(vector / "failure.json", {"schema_version": 1, "status": "failed", "error_type": kind})
    except OSError:
        pass


def _build(*, artifact_dir, source_registry, vector_dir, model_factory, cached, wheel):
    vector, directory, registry, corpus, closure = _output_preflight(artifact_dir, source_registry, vector_dir)
    identity = None if wheel is None else _verify_installed_wheel(wheel)
    modules = _module_hashes()
    _unchanged(directory, registry, closure)
    vector.mkdir(parents=True, exist_ok=False)
    try:
        # Refusal exists before any model work, even if later receipt/failure IO fails.
        with (vector / _PENDING).open("x"):
            pass
        model = model_factory()
        build_passage_embedding_artifact(chunks=corpus.chunks, chunks_path=directory / "chunks.jsonl",
            embeddings_path=vector / _VECTOR, manifest_path=vector / _MANIFEST,
            model=model, model_name=BGE_MODEL_NAME, model_revision=BGE_MODEL_REVISION)
        _unchanged(directory, registry, closure)
        matrix, manifest, vector_hash, manifest_hash = _load_matrix(vector, directory, corpus)
        _unchanged(directory, registry, closure)
        _require(_module_hashes() == modules, "builder module identities changed during build")
        if wheel is not None:
            _require(_verify_installed_wheel(wheel) == identity, "installed wheel identity changed during build")
        receipt = {"schema_version": 1, "status": "complete", "scope": "operator-local-bge-vector-build",
            "corpus": closure, "embeddings_sha256": vector_hash, "embedding_manifest_sha256": manifest_hash,
            "chunk_ids_sha256": _chunk_ids(corpus.chunks),
            "matrix": {"dtype": "float32", "shape": list(matrix.shape), "normalized": True},
            "model": {"name": BGE_MODEL_NAME, "revision": BGE_MODEL_REVISION,
                      "execution_origin": "cached-local-only" if cached else "operator-injected",
                      "device": "cpu" if cached else None, "local_files_only": True if cached else None},
            "runtime": dict(manifest["build_runtime"]), "module_sha256": modules, "wheel": identity,
            "post_build_verified": True, "generation_runs": 0, "human_acceptance": False}
        _validate_receipt(receipt, closure, corpus, vector_hash, manifest_hash, manifest)
        _require(not os.path.lexists(vector / "failure.json"), "vector builder left a failed marker")
        _write(vector / _RECEIPT, receipt)
        (vector / _PENDING).unlink()
        return receipt
    except Exception as error:
        _failure(vector, error)
        raise


def build_local_bge_vectors(*, artifact_dir: Path, source_registry: Path, vector_dir: Path,
                            model: EmbeddingModel, wheel: Path | None = None) -> dict[str, object]:
    """Build newly owned vectors with an injected model; no cached-run claim."""
    return _build(artifact_dir=artifact_dir, source_registry=source_registry, vector_dir=vector_dir,
                  model_factory=lambda: model, cached=False, wheel=wheel)


def _cached_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(BGE_MODEL_NAME, revision=BGE_MODEL_REVISION,
                               device="cpu", local_files_only=True)


def build_cached_local_bge_vectors(*, artifact_dir: Path, source_registry: Path, vector_dir: Path,
                                   wheel: Path) -> dict[str, object]:
    """Use installed-wheel identity and cached pinned CPU BGE after reservation."""
    _require(wheel is not None, "cached builder requires an identified wheel")
    return _build(artifact_dir=artifact_dir, source_registry=source_registry, vector_dir=vector_dir,
                  model_factory=_cached_model, cached=True, wheel=wheel)


def _validate_receipt(value, closure, corpus, vector_hash, manifest_hash, manifest):
    _keys(value, {"schema_version", "status", "scope", "corpus", "embeddings_sha256", "embedding_manifest_sha256",
                  "chunk_ids_sha256", "matrix", "model", "runtime", "module_sha256", "wheel",
                  "post_build_verified", "generation_runs", "human_acceptance"}, "vector build receipt")
    _integer(value["schema_version"], 1, "vector receipt schema")
    _require(value["status"] == "complete" and value["scope"] == "operator-local-bge-vector-build",
             "vector build is not complete")
    _keys(value["corpus"], closure.keys(), "corpus closure")
    for key, expected in closure.items():
        _digest(value["corpus"][key])
        _require(value["corpus"][key] == expected, "vector/corpus closure association mismatch")
    for key, expected in (("embeddings_sha256", vector_hash), ("embedding_manifest_sha256", manifest_hash),
                          ("chunk_ids_sha256", _chunk_ids(corpus.chunks))):
        _digest(value[key])
        _require(value[key] == expected, f"vector receipt {key} hash mismatch")
    _keys(value["matrix"], {"dtype", "shape", "normalized"}, "matrix contract")
    _require(value["matrix"]["dtype"] == "float32" and value["matrix"]["normalized"] is True,
             "receipt matrix must be normalized float32")
    shape = value["matrix"]["shape"]
    _require(type(shape) is list and len(shape) == 2, "receipt matrix shape must be an array")
    _integer(shape[0], len(corpus.chunks), "receipt row count")
    _integer(shape[1], 384, "receipt dimension")
    model = value["model"]
    _keys(model, {"name", "revision", "execution_origin", "device", "local_files_only"}, "model contract")
    _require(model["name"] == BGE_MODEL_NAME and model["revision"] == BGE_MODEL_REVISION, "receipt model mismatch")
    if model["execution_origin"] == "cached-local-only":
        _require(model["device"] == "cpu" and model["local_files_only"] is True and value["wheel"] is not None,
                 "cached model policy/wheel is absent")
    else:
        _require(model["execution_origin"] == "operator-injected" and model["device"] is None
                 and model["local_files_only"] is None, "injected model must retain unknown execution policy")
    _runtime(value["runtime"])
    _require(value["runtime"] == manifest["build_runtime"], "embedding runtime association mismatch")
    _keys(value["module_sha256"], _BOUND_MODULES, "builder module identities")
    for item in value["module_sha256"].values():
        _digest(item)
    wheel = value["wheel"]
    if wheel is not None:
        _keys(wheel, {"wheel_basename", "wheel_bytes", "wheel_sha256", "distribution_version", "module_sha256"},
              "wheel identity")
        _require(wheel["wheel_basename"] == "nvidia_financial_rag-0.1.0-py3-none-any.whl",
                 "receipt named wheel mismatch")
        _require(type(wheel["wheel_bytes"]) is int and wheel["wheel_bytes"] > 0, "invalid wheel byte count")
        _digest(wheel["wheel_sha256"])
        _require(type(wheel["distribution_version"]) is str and bool(wheel["distribution_version"].strip()),
                 "invalid wheel distribution version")
        wheel_modules = wheel["module_sha256"]
        _require(type(wheel_modules) is dict and _BOUND_MODULES <= wheel_modules.keys(),
                 "wheel receipt module set is incomplete")
        for name, item in wheel_modules.items():
            _require(type(name) is str and re.fullmatch(r"[a-z_]+\.py", name) is not None,
                     "invalid wheel module name")
            _digest(item)
            if name in _BOUND_MODULES:
                _require(item == value["module_sha256"][name], "wheel/builder module association mismatch")
    _require(value["post_build_verified"] is True and value["human_acceptance"] is False,
             "receipt verification/human boundary mismatch")
    _integer(value["generation_runs"], 0, "generation count")


def load_local_bge_vectors(*, artifact_dir: Path, source_registry: Path, vector_dir: Path) -> np.ndarray:
    """Return read-only vectors only for this exact completed corpus and receipt."""
    directory = _absolute(artifact_dir, "artifact_dir", exists=True)
    registry = _absolute(source_registry, "source_registry", exists=True)
    vector = _absolute(vector_dir, "vector_dir", exists=True)
    _separate(vector, directory, registry)
    _require(not os.path.lexists(vector / _PENDING), "vector build is incomplete: pending marker")
    _require(not os.path.lexists(vector / "failure.json"), "vector build has failed marker")
    corpus, closure = _load_corpus(directory, registry)
    receipt_path = _file(vector, _RECEIPT)
    receipt_hash = file_sha256(receipt_path)
    receipt = _json(receipt_path)
    matrix, manifest, vector_hash, manifest_hash = _load_matrix(vector, directory, corpus)
    _validate_receipt(receipt, closure, corpus, vector_hash, manifest_hash, manifest)
    _unchanged(directory, registry, closure)
    _require(file_sha256(_file(vector, _RECEIPT)) == receipt_hash
             and not os.path.lexists(vector / "failure.json")
             and not os.path.lexists(vector / _PENDING), "vector completion changed or pending while loading")
    return matrix
