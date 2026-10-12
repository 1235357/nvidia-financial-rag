"""Capture actual scoped cached BGE inspection and exact query arithmetic.

This local diagnostic runs no generator and assigns no financial verdict. The
frozen candidate labels are verified at preflight but never configure ranking.
Only the installed reviewed wheel, existing corpus/vectors and local model cache
are consumed; preparation and independent arithmetic/source evaluation remain
separate operations. Outputs are private local artifacts with a pending guard.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import platform

import numpy as np

from acceptance.expanded_cases import load_case_bundle, load_experiment
from acceptance.expanded_source_binding_replay import load_source_replay
from acceptance.corpus_semantic import model_snapshot_identity, _load_cached_model
from nvidia_rag.corpus import load_verified_corpus
from nvidia_rag.corpus_embeddings import (
    _verify_installed_wheel as verify_installed_wheel, load_local_bge_vectors,
)
from nvidia_rag.retrieval import DenseRetriever
from nvidia_rag.runtime import BGE_MODEL_NAME, BGE_MODEL_REVISION, BGE_QUERY_INSTRUCTION
from nvidia_rag.service import EvidenceService
from nvidia_rag.sources import file_sha256

OUTPUT_FILES = ("queries-raw.npy", "queries-normalized.npy", "query-index.json", "results.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _linked(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _local(path, label, *, directory=False):
    require(isinstance(path, Path) and path.is_absolute(), f"{label} must be an absolute local path")
    require(not any(_linked(p) for p in (path, *path.parents)), f"{label} has a linked local path")
    require(path.is_dir() if directory else path.is_file(), f"{label} is absent or has the wrong local type")
    return path.resolve(strict=True)


def _paths(*, artifact_dir, source_registry, vector_dir, cases, case_freeze,
           source_replay, experiment, wheel, model_cache_dir, output_dir):
    # Output ownership and overlap precede any input parsing or model/cache work.
    require(isinstance(output_dir, Path) and output_dir.is_absolute(), "output_dir must be an absolute local path")
    require(not os.path.lexists(output_dir), "comparison output already exists")
    _local(output_dir.parent, "output parent", directory=True)
    out = output_dir.resolve()
    require(not os.path.lexists(out), "resolved comparison output already exists")
    arguments = {"artifact_dir": artifact_dir, "source_registry": source_registry, "vector_dir": vector_dir,
        "cases": cases, "case_freeze": case_freeze, "source_replay": source_replay,
        "experiment": experiment, "wheel": wheel, "model_cache_dir": model_cache_dir}
    directories = {"artifact_dir", "vector_dir", "model_cache_dir"}
    for name, path in arguments.items():
        require(isinstance(path, Path) and path.is_absolute(), f"{name} must be an absolute local path")
        boundary = path.resolve() if name in directories else path.parent.resolve()
        require(not out.is_relative_to(boundary), "comparison output overlaps a read-only input directory")
    checked = {name: _local(path, name, directory=name in directories) for name, path in arguments.items()}
    return checked, out


def _input_files(paths):
    corpus, vector = paths["artifact_dir"], paths["vector_dir"]
    return {"registry": paths["source_registry"], "corpus_manifest": corpus / "manifest.json",
        "corpus_chunks": corpus / "chunks.jsonl", "vector_manifest": vector / "embedding-manifest.json",
        "vector_matrix": vector / "bge-small-en-v1.5.npy", "cases": paths["cases"],
        "case_freeze": paths["case_freeze"], "source_replay": paths["source_replay"],
        "wheel": paths["wheel"], "experiment": paths["experiment"],
        "corpus_build_receipt": corpus / "build-receipt.json",
        "vector_build_receipt": vector / "vector-build-receipt.json"}


def _input_guards(paths):
    for name in ("artifact_dir", "vector_dir"):
        directory = paths[name]
        require(not os.path.lexists(directory / "pending.json"), "input completion is pending")
        require(not os.path.lexists(directory / "failure.json"), "input has a failed marker")
    replay = paths["source_replay"]
    require(not os.path.lexists(replay.with_name(replay.name + ".pending.json")), "source replay is pending")
    require(not os.path.lexists(replay.with_name(replay.name + ".failure.json")), "source replay has a failure marker")


def _input_snapshot(files, paths):
    _input_guards(paths)
    result = {name: file_sha256(_local(path, name)) for name, path in files.items()}
    _input_guards(paths)
    return result


def _model_contract(cfg):
    model = cfg["model"]
    require(model["name"] == BGE_MODEL_NAME and model["revision"] == BGE_MODEL_REVISION
        and model["instruction"] == BGE_QUERY_INSTRUCTION and model["device"] == "cpu",
        "configured model revision/instruction differs from cached runtime")
    require(cfg["run_kind"] == "diagnostic" and tuple(cfg["threshold_grid"]) == (0.78,),
            "this producer requires the frozen initial diagnostic strategy")


def _corpus_contract(corpus, matrix, cfg):
    documents, sources = corpus.manifest["documents"], corpus.sources
    require(len(sources) == len(documents) == cfg["document_count"], "configured document count differs from corpus")
    require(len(corpus.chunks) == corpus.manifest["chunk_count"] == cfg["chunk_count"],
            "configured chunk count differs from corpus")
    for source, document, configured in zip(sources, documents, cfg["documents"], strict=True):
        require(source.document_id == document["document_id"] == configured["document_id"]
            and source.sha256 == document["source_sha256"] == configured["pdf_sha256"]
            and source.url == document["source_url"] == configured["issuer_url"]
            and document["page_count"] == configured["page_count"],
            "configured source identities/page count/order differ from verified corpus")
        if source.reporting_period is not None:
            require(source.reporting_period["fiscal_year"] == configured["report_fiscal_year"]
                and source.reporting_period["quarter"] == configured["report_fiscal_quarter"],
                "configured report period differs from registered source metadata")
    require(isinstance(matrix, np.ndarray) and matrix.dtype == np.dtype("float32")
            and matrix.shape == (cfg["chunk_count"], 384), "passage matrix dtype/shape differs from configuration")
    require(np.isfinite(matrix).all(), "passage matrix contains non-finite vectors")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    require(np.all(norms > 0) and np.isfinite(norms).all()
            and np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5), "passage matrix has invalid normalized vectors")
    ids = [chunk.chunk_id for chunk in corpus.chunks]
    require(len(ids) == len(set(ids)), "verified corpus chunk IDs are not unique")


def _scope_inventory(corpus, cfg):
    registered = {doc["document_id"] for doc in cfg["documents"]}
    result = {}
    for scope in cfg["scopes"]:
        doc_ids = scope["document_ids"]
        require(scope["id"] not in result, "duplicate scope ID")
        if doc_ids is not None:
            require(doc_ids and len(doc_ids) == len(set(doc_ids)) and set(doc_ids) <= registered,
                    "scope must contain distinct registered document IDs")
        indices = tuple(index for index, chunk in enumerate(corpus.chunks)
                        if doc_ids is None or chunk.document_id in doc_ids)
        require(indices, "configured scope contains no eligible chunks")
        result[scope["id"]] = (None if doc_ids is None else frozenset(doc_ids), indices)
    return result


class ScopedInspection:
    """Expose precisely one actual dense inspection in the operator's scope."""
    def __init__(self, dense, document_ids):
        self.dense, self.document_ids, self.calls = dense, document_ids, 0

    def inspect(self, query, *, top_k):
        self.calls += 1
        require(self.calls == 1, "query unexpectedly inspected more than once")
        return self.dense.inspect(query, top_k=top_k, document_ids=self.document_ids)


class CapturedEncoder:
    def __init__(self, model, query, unchanged):
        self.model, self.query, self.unchanged = model, query, unchanged
        self.calls, self.raw = 0, None

    def __call__(self, texts):
        self.calls += 1
        require(self.calls == 1 and texts == [self.query], "query encoder call/input contract differs")
        self.unchanged()
        raw = self.model.encode([BGE_QUERY_INSTRUCTION + self.query], normalize_embeddings=True,
                               batch_size=32, show_progress_bar=False)
        require(isinstance(raw, np.ndarray) and raw.dtype == np.dtype("float32")
                and raw.shape == (1, 384), "encoder must return exact float32 [1,384] vector shape")
        require(np.isfinite(raw).all(), "encoder returned non-finite vectors")
        norm = np.linalg.norm(raw.astype(np.float64), axis=1)
        require(np.isfinite(norm).all() and np.all(norm > 0), "encoder returned invalid zero/non-finite norms")
        self.raw = raw.copy()
        self.raw.setflags(write=False)
        self.unchanged()
        return raw


def _run_input(*, corpus, matrix, model, item, original_sha, row, scope, cfg, unchanged):
    normalized = item.query.strip()
    capture = CapturedEncoder(model, normalized, unchanged)
    dense = DenseRetriever(corpus.chunks, query_encoder=capture, precomputed_embeddings=matrix)
    document_ids, indices = scope
    adapter = ScopedInspection(dense, document_ids)
    service = EvidenceService(retriever=adapter, threshold=cfg["primary_threshold"], top_k=cfg["top_k"])
    inspected = service.inspect(item.query)
    unchanged()
    require(adapter.calls == capture.calls == 1 and capture.raw is not None, "actual encode/inspection count differs")
    require(inspected.normalized_query == normalized and inspected.threshold == cfg["primary_threshold"]
            and inspected.top_k == cfg["top_k"], "actual service query/gate contract differs")
    consumed = np.asarray(inspected.retrieval.query_vector, dtype=np.float64)
    expected = capture.raw.astype(np.float64)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    require(consumed.shape == (384,) and np.isfinite(consumed).all()
            and np.array_equal(consumed, expected[0]), "actual consumed query normalization differs from captured raw vector")
    scores = inspected.retrieval.considered_scores
    require(inspected.retrieval.considered_chunks == len(indices) == len(scores)
            and all(np.isfinite(score) for score in scores), "actual scope/considered score closure differs")
    by_id = {chunk.chunk_id: index for index, chunk in enumerate(corpus.chunks)}
    candidates = []
    for candidate in inspected.retrieval.candidates:
        chunk = candidate.chunk
        require(chunk.chunk_id in by_id, "actual candidate is absent from verified corpus")
        index = by_id[chunk.chunk_id]
        require(index in indices and chunk == corpus.chunks[index] and np.isfinite(candidate.score),
                "actual candidate escaped scope/corpus identity")
        candidates.append({"global_chunk_index": index, "chunk_id": chunk.chunk_id,
            "text_sha256": sha256(chunk.text.encode("utf-8")).hexdigest(), "score": float(candidate.score)})
    require(len(candidates) == min(len(indices), cfg["top_k"])
            and len({candidate["chunk_id"] for candidate in candidates}) == len(candidates),
            "actual candidate count or uniqueness differs")
    context = inspected.context
    index_row = {"row": row, "case_id": item.case_id, "scope_id": item.scope_id,
        "normalized_query_sha256": sha256(normalized.encode("utf-8")).hexdigest(),
        "instructed_query_sha256": sha256((BGE_QUERY_INSTRUCTION + normalized).encode("utf-8")).hexdigest(),
        "original_record_sha256": original_sha, "encoder_call_count": capture.calls}
    result_row = {"row": row, "case_id": item.case_id, "scope_id": item.scope_id,
        "scope_document_ids": None if item.document_ids is None else list(item.document_ids),
        "scoped_global_chunk_indices": list(indices), "considered_scores": [float(score) for score in scores],
        "top_k": candidates, "evidence_ids": [item.chunk.chunk_id for item in context.evidence],
        "citations": list(context.citations), "can_answer": context.can_answer, "message": context.message,
        "context_sha256": sha256(context.context.encode("utf-8")).hexdigest(), "context_empty": context.context == "",
        "threshold": inspected.threshold, "top_k_limit": inspected.top_k}
    return capture.raw[0].copy(), consumed.copy(), index_row, result_row


def _write_bytes_exclusive(path, body):
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256(body).hexdigest()


def _write_json_exclusive(path, value):
    body = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")
    return _write_bytes_exclusive(path, body)


def _write_array_exclusive(path, value):
    with io.BytesIO() as handle:
        np.save(handle, value, allow_pickle=False)
        body = handle.getvalue()
    return _write_bytes_exclusive(path, body)


def _check_outputs(directory, digests):
    require(all(file_sha256(_local(directory / name, "output file")) == digest
                for name, digest in digests.items()), "owned output digest changed during closure")


def run_comparison(*, artifact_dir: Path, source_registry: Path, vector_dir: Path,
                   cases: Path, case_freeze: Path, source_replay: Path, experiment: Path,
                   wheel: Path, model_cache_dir: Path, output_dir: Path):
    paths, output = _paths(artifact_dir=artifact_dir, source_registry=source_registry, vector_dir=vector_dir,
        cases=cases, case_freeze=case_freeze, source_replay=source_replay, experiment=experiment,
        wheel=wheel, model_cache_dir=model_cache_dir, output_dir=output_dir)
    cfg = load_experiment(paths["experiment"])
    _model_contract(cfg.data)
    bundle = load_case_bundle(paths["cases"], paths["case_freeze"], cfg)
    files = _input_files(paths)
    before = _input_snapshot(files, paths)
    require(before["experiment"] == cfg.sha256, "validated experiment changed before input snapshot")
    expected = dict(cfg.data["input_sha256"])
    require({name: before[name] for name in expected} == expected, "frozen input SHA-256 identity differs")
    identity = deepcopy(verify_installed_wheel(paths["wheel"]))
    require(identity["wheel_sha256"] == before["wheel"], "installed wheel body identity differs from frozen wheel")
    corpus = load_verified_corpus(artifact_dir=paths["artifact_dir"], source_registry=paths["source_registry"])
    matrix = load_local_bge_vectors(artifact_dir=paths["artifact_dir"], source_registry=paths["source_registry"],
                                    vector_dir=paths["vector_dir"])
    _corpus_contract(corpus, matrix, cfg.data)
    vector_receipt = json.loads(files["vector_build_receipt"].read_bytes())
    require(vector_receipt["model"]["execution_origin"] == "cached-local-only"
            and vector_receipt["model"]["device"] == "cpu" and vector_receipt["model"]["local_files_only"] is True
            and vector_receipt["wheel"] == identity, "vectors require cached CPU origin and exact installed wheel identity")
    load_source_replay(paths["source_replay"], bundle=bundle, artifact_dir=paths["artifact_dir"],
                       source_registry=paths["source_registry"])
    scopes = _scope_inventory(corpus, cfg.data)
    inputs = bundle.retrieval_inputs()
    require(inputs and len({(item.case_id, item.scope_id) for item in inputs}) == len(inputs),
            "frozen query/scope row order contains duplicate inputs")
    require(before == _input_snapshot(files, paths), "read-only input closure changed during preflight")
    cache = deepcopy(model_snapshot_identity(paths["model_cache_dir"]))
    require(cache["model_name"] == cfg.data["model"]["name"] and cache["model_revision"] == cfg.data["model"]["revision"],
            "prepared model cache identity differs from configured model")

    def input_unchanged():
        require(before == _input_snapshot(files, paths), "read-only input closure changed during comparison")
        require(model_snapshot_identity(paths["model_cache_dir"]) == cache, "prepared model cache changed during comparison")
        require(verify_installed_wheel(paths["wheel"]) == identity, "installed wheel identity changed during comparison")
        require(before == _input_snapshot(files, paths), "read-only input closure changed during cache/wheel verification")

    input_unchanged()
    pending, failure = output / "pending.json", output / "failure.json"
    output.mkdir()  # Never reuse a directory, including a concurrently created one.
    guard_digest = None
    try:
        guard_digest = _write_json_exclusive(pending, {"schema_version": 2, "status": "pending"})

        def unchanged():
            require(not os.path.lexists(failure) and file_sha256(_local(pending, "pending guard")) == guard_digest,
                    "owned comparison completion guard changed")
            input_unchanged()
            require(not os.path.lexists(failure) and file_sha256(_local(pending, "pending guard")) == guard_digest,
                    "owned comparison completion guard changed during input verification")

        unchanged()
        model = _load_cached_model(paths["model_cache_dir"])
        unchanged()
        original = {case.case_id: case.origin["original_record_sha256"] for case in bundle.cases}
        raw_rows, normalized_rows, index_rows, result_rows = [], [], [], []
        for row, item in enumerate(inputs):
            unchanged()
            raw, normalized, index, result = _run_input(corpus=corpus, matrix=matrix, model=model, item=item,
                original_sha=original[item.case_id], row=row, scope=scopes[item.scope_id], cfg=cfg.data, unchanged=unchanged)
            raw_rows.append(raw)
            normalized_rows.append(normalized)
            index_rows.append(index)
            result_rows.append(result)
        digests = {}
        for name, value, writer in (
            ("queries-raw.npy", np.stack(raw_rows), _write_array_exclusive),
            ("queries-normalized.npy", np.stack(normalized_rows), _write_array_exclusive),
            ("query-index.json", {"schema_version": 2, "rows": index_rows}, _write_json_exclusive),
            ("results.json", {"schema_version": 2, "rows": result_rows}, _write_json_exclusive),
        ):
            unchanged()
            digests[name] = writer(output / name, value)
            unchanged()
            _check_outputs(output, digests)
        receipt = {"schema_version": 2, "status": "completed", "experiment_sha256": cfg.sha256,
            "input_sha256": expected, "input_paths": {name: str(paths[name]) for name in ("cases", "source_replay", "wheel")},
            "producer": {"numpy_version": np.__version__, "python_version": platform.python_version(),
                         "platform": platform.platform()}, "query_count": len(inputs), "files": digests}
        unchanged()
        _check_outputs(output, digests)
        manifest_digest = _write_json_exclusive(output / "manifest.json", receipt)
        unchanged()
        _check_outputs(output, {**digests, "manifest.json": manifest_digest})
        pending.unlink()
        require(not any(os.path.lexists(path) for path in (pending, failure)),
                'comparison completion guard remains or appeared during closure')
        input_unchanged()
        _check_outputs(output, {**digests, "manifest.json": manifest_digest})
        require(not any(os.path.lexists(path) for path in (pending, failure)),
                'comparison completion guard appeared during final verification')
        return receipt
    except BaseException:
        if not os.path.lexists(pending):
            try:
                _write_json_exclusive(pending, {'schema_version': 2, 'status': 'pending'})
            except (OSError, ValueError):
                pass
        try:
            _write_json_exclusive(failure, {"schema_version": 2, "status": "failed", "code": "expanded_comparison_incomplete"})
        except (OSError, ValueError):
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-dir", "source-registry", "vector-dir", "cases", "case-freeze",
                 "source-replay", "experiment", "wheel", "model-cache-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    options = parser.parse_args()
    try:
        run_comparison(**vars(options))
    except Exception:
        parser.error("expanded comparison refused; retain pending guard and finite local diagnostics")
    print("PASS: cached BGE scoped traces and actual query vectors saved; financial quality not evaluated")


if __name__ == "__main__":
    main()
