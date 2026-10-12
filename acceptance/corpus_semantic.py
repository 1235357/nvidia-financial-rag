"""Installed-wheel cached BGE comparison for four exposed development queries.

This operator script verifies input identities and records two explicit retrieval
scopes. It runs no generator and computes no financial quality score. Corpus,
vector, model and dependency preparation are separate prerequisites.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import runpy
import time

from nvidia_rag.corpus import DocumentScopeRetriever, load_verified_corpus
from nvidia_rag.corpus_embeddings import (
    _verify_installed_wheel as verify_installed_wheel,
    load_local_bge_vectors,
)
from nvidia_rag.retrieval import DenseRetriever
from nvidia_rag.runtime import BGE_MODEL_NAME, BGE_MODEL_REVISION, BGE_QUERY_INSTRUCTION
from nvidia_rag.service import EvidenceService
from nvidia_rag.sources import file_sha256

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_SHA256 = "74379f0ecb5b131ca8562d2713a13147dc538047e7f2e2b03212ca6f2f7ed495"
THRESHOLD, TOP_K = 0.78, 5
CORPUS_FILES = ("manifest.json", "chunks.jsonl", "build-receipt.json")
VECTOR_FILES = ("bge-small-en-v1.5.npy", "embedding-manifest.json", "vector-build-receipt.json")
REQUIRED_CACHE_FILES = {"model.safetensors", "config.json", "modules.json", "tokenizer.json",
                        "tokenizer_config.json", "sentence_bert_config.json", "1_Pooling/config.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _absolute(path, label):
    require(isinstance(path, Path) and path.is_absolute(), f"{label} must be an absolute local path")
    return path.resolve(strict=True)


def _linked_directory(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def model_snapshot_identity(model_cache_dir: Path) -> dict:
    """Hash prepared pinned snapshot bodies, permitting only confined file links."""
    cache = _absolute(model_cache_dir, "model_cache_dir")
    require(cache.is_dir(), "prepared model cache must be a directory")
    snapshot = cache / "models--BAAI--bge-small-en-v1.5" / "snapshots" / BGE_MODEL_REVISION
    for directory in (snapshot.parent.parent, snapshot.parent, snapshot):
        require(directory.is_dir() and not _linked_directory(directory),
                "prepared snapshot directory is absent or linked")
    files = {}
    for directory, directories, filenames in os.walk(snapshot, followlinks=False):
        parent = Path(directory)
        for name in directories:
            require(not _linked_directory(parent / name), "linked model snapshot directory is unsupported")
        for name in filenames:
            path = parent / name
            resolved = path.resolve(strict=True)
            require(resolved.is_file() and resolved.is_relative_to(cache),
                    "model snapshot file escapes prepared cache")
            files[path.relative_to(snapshot).as_posix()] = {
                "bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)}
    require(REQUIRED_CACHE_FILES <= files.keys(), "required prepared model snapshot cache bodies are missing")
    require(all(row["bytes"] > 0 for row in files.values()), "prepared snapshot has an empty body")
    return {"model_name": BGE_MODEL_NAME, "model_revision": BGE_MODEL_REVISION,
            "files": dict(sorted(files.items()))}


def _load_cached_model(cache):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(BGE_MODEL_NAME, revision=BGE_MODEL_REVISION, device="cpu",
                               local_files_only=True, cache_folder=str(cache))


def _input_snapshot(inputs, corpus_dir, vector_dir):
    for directory in (corpus_dir, vector_dir):
        require(not os.path.lexists(directory / "pending.json"), "input completion is pending")
        require(not os.path.lexists(directory / "failure.json"), "input has a failed marker")
    result = {name: file_sha256(path) for name, path in inputs.items()}
    for directory in (corpus_dir, vector_dir):
        require(not os.path.lexists(directory / "pending.json")
                and not os.path.lexists(directory / "failure.json"), "input completion state changed")
    return result


class _RecordingRetriever:
    """Capture the sole call used by the service's threshold/context operation."""
    def __init__(self, retriever):
        self.retriever, self.candidates, self.calls = retriever, (), 0

    def retrieve(self, query, *, top_k):
        self.calls += 1
        require(self.calls == 1, "query unexpectedly retrieved more than once")
        self.candidates = tuple(self.retriever.retrieve(query, top_k=top_k))
        return list(self.candidates)


def _run_scope(corpus, matrix, model, questions, scope):
    def query_encoder(texts):
        return model.encode([BGE_QUERY_INSTRUCTION + text for text in texts],
                            normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    dense = DenseRetriever(corpus.chunks, query_encoder=query_encoder, precomputed_embeddings=matrix)
    scoped = DocumentScopeRetriever(dense, document_ids=scope)
    rows = []
    for question in questions:
        recorder = _RecordingRetriever(scoped)
        service = EvidenceService(retriever=recorder, threshold=THRESHOLD, top_k=TOP_K)
        started = time.perf_counter()
        context = service.query(question["query"])
        elapsed = (time.perf_counter() - started) * 1000
        require(recorder.calls == 1 and math.isfinite(elapsed) and elapsed >= 0,
                "query execution/clock contract mismatch")
        candidates = []
        for rank, item in enumerate(recorder.candidates, 1):
            require(math.isfinite(item.score), "non-finite observed candidate score")
            chunk = item.chunk
            candidates.append({"rank": rank, "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id, "physical_page": chunk.page_number,
                "score": item.score, "text_sha256": sha256(chunk.text.encode("utf-8")).hexdigest(),
                "below_threshold": item.score < THRESHOLD})
        rows.append({"question_id": question["question_id"], "query": question["query"],
            "original_question_record_sha256": question["original_question_record_sha256"],
            "elapsed_ms": elapsed, "candidates": candidates,
            "evidence_ids": [item.chunk.chunk_id for item in context.evidence],
            "citations": list(context.citations), "can_answer": context.can_answer,
            "message": context.message, "context_empty": context.context == "",
            "context_sha256": sha256(context.context.encode("utf-8")).hexdigest(),
            "answer": "", "generation_runs": 0,
            "development_exposed": True, "financial_field_support": "not_evaluated"})
    return {"document_scope": None if scope is None else sorted(scope), "records": rows}


def _write_exclusive(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _pending_path(output):
    return output.with_name(output.name + ".pending.json")


def load_completed_comparison(output: Path) -> dict:
    """Read completed local output only after its writer's pending guard cleared.

    This verifies operation state and read closure, not signed authenticity or
    correctness of the recorded financial evidence.
    """
    require(isinstance(output, Path) and output.is_absolute(), "result must be an absolute local path")
    pending = _pending_path(output)
    require(not os.path.lexists(pending), "semantic comparison is pending")
    require(output.is_file() and not output.is_symlink(), "comparison result must be an owned local file")
    body = output.read_bytes()
    result = json.loads(body)
    require(type(result) is dict and result.get("status") == "complete", "semantic comparison is not complete")
    require(result.get("generation_runs") == 0 and result.get("human_acceptance") is False
            and result.get("financial_field_support") == "not_evaluated", "result evidence scope changed")
    require(not os.path.lexists(pending) and file_sha256(output) == sha256(body).hexdigest(),
            "semantic completion changed or became pending while reading")
    return result


def run_comparison(*, artifact_dir: Path, source_registry: Path, vector_dir: Path,
                   candidates: Path, wheel: Path, model_cache_dir: Path, output: Path):
    require(isinstance(output, Path) and output.is_absolute() and not os.path.lexists(output),
            "comparison output must be absolute and not exist")
    pending = _pending_path(output)
    require(not os.path.lexists(pending), "semantic comparison pending marker already exists")
    # Overlap is refused before identities, model import or any query operation.
    for name, path in (("artifact_dir", artifact_dir), ("source_registry", source_registry),
                       ("vector_dir", vector_dir), ("candidates", candidates),
                       ("wheel", wheel), ("model_cache_dir", model_cache_dir)):
        require(isinstance(path, Path) and path.is_absolute(), f"{name} must be an absolute local path")
    resolved_output = output.resolve()
    require(not os.path.lexists(resolved_output), "resolved output already exists")
    require(not any(resolved_output.is_relative_to(path.resolve())
                    for path in (artifact_dir, vector_dir, model_cache_dir))
            and resolved_output not in {path.resolve() for path in (source_registry, candidates, wheel)},
            "comparison output overlaps a read-only input")
    directory, registry = _absolute(artifact_dir, "artifact_dir"), _absolute(source_registry, "source_registry")
    vector, candidate_file = _absolute(vector_dir, "vector_dir"), _absolute(candidates, "candidates")
    wheel_file, cache = _absolute(wheel, "wheel"), _absolute(model_cache_dir, "model_cache_dir")
    require(resolved_output.parent.is_dir(), "comparison output parent is absent")
    identity = verify_installed_wheel(wheel_file)
    inputs = {"registry": registry, "candidates": candidate_file, "wheel": wheel_file,
              **{"corpus/" + name: directory / name for name in CORPUS_FILES},
              **{"vectors/" + name: vector / name for name in VECTOR_FILES}}
    before = _input_snapshot(inputs, directory, vector)
    require(before["registry"] == REGISTRY_SHA256, "source registry differs from frozen eight-report recipe")
    corpus = load_verified_corpus(artifact_dir=directory, source_registry=registry)
    require(len(corpus.sources) == 8 and corpus.chunks_sha256 == before["corpus/chunks.jsonl"],
            "consumed corpus differs from eight-report input closure")
    matrix = load_local_bge_vectors(artifact_dir=directory, source_registry=registry, vector_dir=vector)
    vector_receipt = json.loads((vector / "vector-build-receipt.json").read_bytes())
    require(vector_receipt["model"]["execution_origin"] == "cached-local-only"
            and vector_receipt["wheel"] == identity, "vectors require cached origin and exact installed wheel identity")
    questions = runpy.run_path(str(ROOT / "acceptance/source_binding_replay.py"))["load_candidates"](candidate_file)
    require(before == _input_snapshot(inputs, directory, vector), "input closure changed while loading")
    snapshot = model_snapshot_identity(cache)
    require(before == _input_snapshot(inputs, directory, vector), "input closure changed during cache preflight")
    require(verify_installed_wheel(wheel_file) == identity, "installed wheel changed during preflight")

    def unchanged():
        require(before == _input_snapshot(inputs, directory, vector), "read-only input closure changed")
        require(model_snapshot_identity(cache) == snapshot, "prepared model cache changed during comparison")
        require(verify_installed_wheel(wheel_file) == identity, "installed wheel identity changed during comparison")

    established = {"installed_identity": identity, "input_sha256": before, "model_snapshot": snapshot}
    try:
        _write_exclusive(pending, {"schema_version": 1, "status": "pending",
                                  "scope": "operator-local-cached-bge-comparison"})
        unchanged()
        model_started = time.perf_counter()
        model = _load_cached_model(cache)
        initialization_ms = (time.perf_counter() - model_started) * 1000
        require(math.isfinite(initialization_ms) and initialization_ms >= 0, "model initialization clock mismatch")
        unchanged()
        comparisons = [_run_scope(corpus, matrix, model, questions, scope)
                       for scope in (None, frozenset({"nvidia-fy2025-q2"}))]
        unchanged()
        receipt = {"schema_version": 1, "status": "complete", "created_at_utc": datetime.now(UTC).isoformat(),
            "scope": "installed cached BGE development retrieval comparison; financial quality not evaluated",
            **established, "model": {"name": BGE_MODEL_NAME, "revision": BGE_MODEL_REVISION,
                "device": "cpu", "local_files_only": True, "initialization_ms": initialization_ms,
                "query_instruction": BGE_QUERY_INSTRUCTION, "query_encoding_calls": 8,
                "normalize_embeddings": True, "batch_size": 32, "annual_routing": False},
            "runtime": {"python": platform.python_version(),
                **{name: metadata.version(name) for name in ("numpy", "sentence-transformers", "torch", "transformers")}},
            "threshold": THRESHOLD, "top_k": TOP_K, "generation_runs": 0,
            "human_acceptance": False, "human_verdict": None, "development_exposed": True,
            "financial_field_support": "not_evaluated", "comparisons": comparisons}
        _write_exclusive(output, receipt)
        # The guard survives completed bytes if their writer/close or unlink fails.
        pending.unlink()
        return receipt
    except Exception as error:
        kind = type(error).__name__
        if kind not in {"ValueError", "OSError", "RuntimeError", "FileNotFoundError", "PermissionError",
                        "ImportError", "ModuleNotFoundError"}:
            kind = "ComparisonError"
        if not os.path.lexists(output):
            try:
                _write_exclusive(output, {"schema_version": 1, "status": "failed",
                    "scope": "cached BGE comparison failed; no completed semantic acceptance",
                    **established, "error_type": kind, "generation_runs": 0, "human_acceptance": False})
            except OSError:
                pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("artifact-dir", "source-registry", "vector-dir", "candidates", "wheel", "model-cache-dir", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    try:
        run_comparison(**vars(args))
    except (ValueError, OSError) as error:
        parser.error(type(error).__name__ + ": comparison refused; retain local failure evidence")
    print("PASS: cached BGE development comparison complete; financial quality not evaluated")


if __name__ == "__main__":
    main()
