"""Frozen local retrieval runs and portable, strictly checked result graphs.

The retained bodies establish unsigned byte and row consistency. They do not
authenticate a reviewer, every imported dependency, or financial correctness.
Optional semantic packages are imported only after the run is reserved.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

from .core import Chunk, ScoredChunk
from .evaluation import evaluate_retriever, validate_acceptance_preflight
from .evaluation_inputs import canonical_json, parse_json, parse_questions

MAX_OUTPUT_BYTES = 64 * 1024 * 1024
FREEZE_FILE = "freeze.json"
RESULTS_FILE = "results.json"
PACKET_FILE = "review-packet.json"
RECEIPT_FILE = "receipt.json"
PENDING_FILE = "pending.json"
FAILURE_FILE = "failure.json"
ATTEMPTS_FILE = "attempts.json"
CORPUS_FILES = ("manifest.json", "chunks.jsonl", "build-receipt.json")
VECTOR_FILES = ("bge-small-en-v1.5.npy", "embedding-manifest.json", "vector-build-receipt.json")
REQUIRED_MODEL_FILES = frozenset({"model.safetensors", "config.json", "modules.json", "tokenizer.json",
    "tokenizer_config.json", "sentence_bert_config.json", "1_Pooling/config.json"})
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
_CONFIG_FIELDS = {"mode", "backend", "top_k", "threshold", "document_ids", "routing_policy",
    "model_execution_origin", "model_name", "model_revision", "device", "local_files_only"}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, expected, label):
    _require(type(value) is dict and set(value) == set(expected), f"invalid {label} fields")


def _linked(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _path(path, label, *, directory=False, exists=True):
    _require(isinstance(path, Path) and path.is_absolute(), f"{label} must be an absolute local path")
    _require(not any(_linked(parent) for parent in (path, *path.parents)), f"{label} contains a linked path")
    if exists:
        _require(path.is_dir() if directory else path.is_file(), f"{label} is not the expected local entry")
    return path.resolve(strict=exists)


def _read(path, *, limit=MAX_OUTPUT_BYTES):
    _path(path, "input")
    _require(path.stat().st_size <= limit, "input exceeds byte budget")
    with path.open("rb") as stream:
        body = stream.read(limit + 1)
    _require(len(body) <= limit, "input exceeds byte budget")
    return body


def _identity(body):
    return {"bytes": len(body), "sha256": sha256(body).hexdigest()}


def _file_identity(path):
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _validate_identity(row):
    _require(type(row["bytes"]) is int and row["bytes"] >= 0, "invalid input byte count")
    _require(type(row["sha256"]) is str and _DIGEST.fullmatch(row["sha256"]) is not None,
             "invalid input digest")


def _complete_input(directory):
    _require(not os.path.lexists(directory / PENDING_FILE)
        and not os.path.lexists(directory / FAILURE_FILE), "input completion is pending or failed")


def _package_files():
    package = _path(Path(__file__).resolve().parent, "package", directory=True)
    result = {}
    for parent, directories, filenames in os.walk(package, followlinks=False):
        directories[:] = [name for name in directories if name != "__pycache__"]
        _require(not any(_linked(Path(parent) / name) for name in directories), "linked package directory")
        for name in filenames:
            if name.endswith(".py"):
                path = _path(Path(parent) / name, "package module")
                result[path.relative_to(package).as_posix()] = path
    _require(bool(result), "package Python bodies are absent")
    return dict(sorted(result.items()))


def model_snapshot_identity(model_cache_dir: Path) -> dict:
    """Inventory the complete selected pinned snapshot, including confined blobs."""
    cache = _path(model_cache_dir, "model_cache_dir", directory=True)
    snapshot = cache / "models--BAAI--bge-small-en-v1.5" / "snapshots" / _MODEL_REVISION
    for directory in (snapshot.parent.parent, snapshot.parent, snapshot):
        _path(directory, "prepared model snapshot", directory=True)
    files = {}
    for parent, directories, filenames in os.walk(snapshot, followlinks=False):
        _require(not any(_linked(Path(parent) / name) for name in directories), "linked model snapshot directory")
        for name in filenames:
            path = Path(parent) / name
            resolved = path.resolve(strict=True)
            _require(resolved.is_file() and resolved.is_relative_to(cache), "model snapshot file escapes cache")
            # Hugging Face snapshot files may be links to blobs inside this cache.
            _require(not any(_linked(p) for p in resolved.parents), "linked model blob directory")
            files[path.relative_to(snapshot).as_posix()] = _file_identity(resolved)
    _require(REQUIRED_MODEL_FILES <= files.keys(), "required prepared model snapshot bodies are missing")
    _require(all(row["bytes"] > 0 for row in files.values()), "prepared model snapshot contains an empty body")
    return {"model_name": _MODEL_NAME, "model_revision": _MODEL_REVISION, "files": dict(sorted(files.items()))}


def load_cached_model(cache: Path):
    """Construct only the selected pinned CPU model from this prepared cache."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME, revision=_MODEL_REVISION, device="cpu",
        local_files_only=True, cache_folder=str(cache))


def _settings(*, mode, backend, top_k, threshold, document_ids, routing_policy,
              vector_dir, model_cache_dir, statement_source_dir, statement_fiscal_declarations,
              reference, model_factory):
    _require(mode in ("development", "acceptance-candidate"), "unsupported evaluation mode")
    _require(backend in ("lexical", "cached-bge"), "unsupported retrieval backend")
    _require(type(top_k) is int and 1 <= top_k <= 1000, "top_k must be an integer from 1 through 1000")
    _require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 <= threshold <= 1,
             "threshold must be finite from zero through one")
    _require(routing_policy in ("none", "source-bound-period"), "unsupported routing policy")
    if document_ids is not None:
        _require(type(document_ids) is frozenset and bool(document_ids)
            and all(type(value) is str and value == value.strip() and bool(value) for value in document_ids),
            "document scope must be a nonempty frozenset of canonical strings")
    _require((statement_source_dir is None) == (statement_fiscal_declarations is None),
             "statement source and fiscal declarations must be paired")
    _require((routing_policy == "source-bound-period") == (statement_source_dir is not None),
             "source-bound-period requires its native source/declaration pair")
    if backend == "lexical":
        _require(vector_dir is None and model_cache_dir is None and model_factory is None
            and routing_policy == "none", "lexical backend has unsupported model or routing options")
    else:
        _require(vector_dir is not None and model_cache_dir is not None,
                 "cached-bge requires existing vectors and an exact prepared model cache")
        _require(model_factory is None or callable(model_factory), "model_factory must be callable")
    _require((mode == "acceptance-candidate") == (reference is not None),
             "acceptance-candidate requires a separate reference; development has no reference")


def _registrations(questions, reference, mode):
    if mode == "acceptance-candidate":
        validate_acceptance_preflight(questions, reference)
        _require(all(q.evaluation_usage is not None and q.evaluation_usage["frozen_on"] is not None
                     for q in reference), "acceptance reference must be separately frozen")
    else:
        _require(all(q.evaluation_usage is not None
            and q.evaluation_usage["purpose"] == "development"
            and q.evaluation_usage["development_exposed"] is True for q in questions),
            "development requires explicitly exposed development registrations")


def _output_path(output, inputs):
    target = _path(output, "output_dir", exists=False)
    _require(not os.path.lexists(output) and not os.path.lexists(target), "output directory already exists")
    _require(target.parent.is_dir(), "output parent directory is absent")
    protected = [*inputs, Path(__file__).resolve().parent]
    _require(not any(target.is_relative_to(value) or value.is_relative_to(target) for value in protected),
             "output overlaps a read-only input")
    return target


class _Writer:
    def __init__(self, directory):
        self.directory, self.written = directory, 0

    def body(self, name, body):
        _require(self.written + len(body) <= MAX_OUTPUT_BYTES, "complete output exceeds byte budget")
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(body)
        self.written += len(body)
        return _identity(body)

    def json(self, name, value):
        return self.body(name, canonical_json(value))

    def progress(self, value):
        """Atomically replace only this reserved run's mutable attempt journal."""
        body = canonical_json(value)
        path = self.directory / ATTEMPTS_FILE
        previous = path.stat().st_size if path.exists() else 0
        _require(self.written - previous + len(body) <= MAX_OUTPUT_BYTES,
                 "complete output exceeds byte budget")
        temporary = self.directory / 'attempts.pending'
        try:
            with temporary.open('xb') as stream:
                stream.write(body)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.written += len(body) - previous
        return _identity(body)


class CapturedRetriever:
    """Forward each exact query once and validate complete ordered candidates."""
    def __init__(self, retriever, *, questions, chunks, document_ids, top_k, attempts, persist=None):
        self.retriever, self.questions = retriever, tuple(questions)
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.document_ids, self.top_k, self.attempts = document_ids, top_k, attempts
        self.calls = []
        self.persist = persist or (lambda: None)

    def retrieve(self, query, *, top_k):
        index = len(self.calls)
        _require(index < len(self.questions) and query == self.questions[index].query
            and type(top_k) is int and top_k == self.top_k, "unexpected query execution contract")
        question = self.questions[index]
        attempt = {"question_id": question.question_id, "query_sha256": sha256(query.encode("utf-8")).hexdigest(),
                   "status": "started", "error_type": None}
        self.attempts.append(attempt)
        self.persist()
        try:
            values = self.retriever.retrieve(query, top_k=top_k)
            _require(type(values) in (list, tuple) and len(values) <= top_k, "invalid candidate count or collection")
            candidates, ids = [], set()
            for rank, item in enumerate(values, 1):
                _require(type(item) is ScoredChunk and type(item.chunk) is Chunk, "complete ScoredChunk required")
                chunk = item.chunk
                _require(chunk.chunk_id in self.chunks
                    and canonical_json(asdict(chunk)) == canonical_json(asdict(self.chunks[chunk.chunk_id])),
                    "candidate differs from verified corpus membership")
                _require(chunk.chunk_id not in ids and chunk.document_id in self.document_ids,
                         "duplicate or out-of-scope candidate")
                _require(type(item.score) in (int, float) and math.isfinite(item.score)
                    and -1 - 1e-12 <= item.score <= 1 + 1e-12, "candidate score is not a finite cosine")
                ids.add(chunk.chunk_id)
                candidates.append({"rank": rank, "chunk": asdict(chunk), "score": item.score,
                    "text_sha256": sha256(chunk.text.encode("utf-8")).hexdigest()})
            self.calls.append({"question_id": question.question_id, "query": query,
                "query_sha256": attempt["query_sha256"], "candidates": candidates})
            attempt["status"] = "complete"
            attempt["candidates"] = candidates
            self.persist()
            return list(values)
        except Exception as error:
            attempt["status"], attempt["error_type"] = "failed", _error_type(error)
            self.persist()
            raise


def _error_type(error):
    name = type(error).__name__
    return name if name in {"ValueError", "RuntimeError", "OSError", "FileNotFoundError", "PermissionError",
        "ImportError", "ModuleNotFoundError", "TypeError", "MemoryError"} else "EvaluationError"


def run_evaluation(*, artifact_dir, source_registry, questions, output_dir, mode, backend,
        top_k, threshold, reference=None, document_ids=None, vector_dir=None, model_cache_dir=None,
        routing_policy="none", statement_source_dir=None, statement_fiscal_declarations=None,
        model_factory=None):
    """Execute a single reserved retrieval run without generation or fallback.

    ``model_factory`` is an explicit Python injection hook whose origin remains
    operator-injected. The command line never exposes it as an option.
    """
    _settings(mode=mode, backend=backend, top_k=top_k, threshold=threshold, document_ids=document_ids,
        routing_policy=routing_policy, vector_dir=vector_dir, model_cache_dir=model_cache_dir,
        statement_source_dir=statement_source_dir, statement_fiscal_declarations=statement_fiscal_declarations,
        reference=reference, model_factory=model_factory)
    directory = _path(artifact_dir, "artifact_dir", directory=True)
    registry = _path(source_registry, "source_registry")
    question_path = _path(questions, "questions")
    reference_path = None if reference is None else _path(reference, "reference")
    _require(reference_path is None or reference_path != question_path, "reference must be a separate input")
    vector = None if vector_dir is None else _path(vector_dir, "vector_dir", directory=True)
    cache = None if model_cache_dir is None else _path(model_cache_dir, "model_cache_dir", directory=True)
    source = None if statement_source_dir is None else _path(statement_source_dir, "statement_source_dir", directory=True)
    declarations = None if statement_fiscal_declarations is None else _path(statement_fiscal_declarations, "fiscal declarations")
    original_paths = [p for p in (directory, registry, question_path, reference_path, vector, cache, source, declarations) if p is not None]
    output = _output_path(output_dir, original_paths)
    from .corpus import load_verified_corpus
    corpus = load_verified_corpus(artifact_dir=directory, source_registry=registry)
    available = frozenset(source.document_id for source in corpus.sources)
    scope = available if document_ids is None else document_ids
    _require(scope <= available, "document scope is not registered")
    question_body = _read(question_path, limit=1_048_576)
    registered = parse_questions(question_body)
    _require(bool(registered), "questions must be nonempty")
    reference_body = None if reference_path is None else _read(reference_path, limit=1_048_576)
    reference_rows = () if reference_body is None else parse_questions(reference_body)
    _registrations(registered, reference_rows, mode)
    for question in registered:
        _require(set(question.evidence_document_ids) <= available, "question labels refer to an unregistered document")
        pages = {doc["document_id"]: doc["page_count"] for doc in corpus.manifest["documents"]}
        _require(all(page <= pages[doc] for doc, page in zip(question.evidence_document_ids, question.evidence_pages)),
                 "question labels exceed the registered document page bounds")
    matrix, model_identity, snapshot, catalog = None, None, None, None
    if vector is not None:
        from .corpus_embeddings import load_local_bge_vectors
        matrix = load_local_bge_vectors(artifact_dir=directory, source_registry=registry, vector_dir=vector)
        model_identity = model_snapshot_identity(cache)
    if source is not None:
        from .statement_startup import load_statement_snapshot
        from .period_routing import build_report_catalog
        snapshot = load_statement_snapshot(source_registry=registry, source_dir=source,
            fiscal_declarations=declarations, document_ids=scope, expected_sources=corpus.sources,
            corpus_manifest=directory / "manifest.json", chunks_path=directory / "chunks.jsonl")
        catalog = build_report_catalog(snapshot, corpus_manifest=directory / "manifest.json", chunks_path=directory / "chunks.jsonl")
    config = {"mode": mode, "backend": backend, "top_k": top_k, "threshold": threshold,
        "document_ids": sorted(scope), "routing_policy": routing_policy,
        "model_execution_origin": None if backend == "lexical" else ("cached-local-only" if model_factory is None else "operator-injected"),
        "model_name": None if backend == "lexical" else _MODEL_NAME,
        "model_revision": None if backend == "lexical" else _MODEL_REVISION,
        "device": "cpu" if backend == "cached-bge" and model_factory is None else None,
        "local_files_only": True if backend == "cached-bge" and model_factory is None else None}
    config_body = canonical_json(config)
    paths = {"questions": question_path, "source_registry": registry,
        **{"corpus/" + name: _path(directory / name, "corpus input") for name in CORPUS_FILES},
        **{"package/" + name: path for name, path in _package_files().items()}}
    if reference_path is not None:
        paths["reference"] = reference_path
    if vector is not None:
        paths.update({"vectors/" + name: _path(vector / name, "vector input") for name in VECTOR_FILES})
    if snapshot is not None:
        paths["statement/fiscal-declarations.json"] = declarations
        for record in corpus.sources:
            if record.document_id in scope:
                paths["statement/" + record.filename] = _path(source / record.filename, "statement PDF")
    bodies = {name: _read(path) for name, path in paths.items()}
    _require(bodies["questions"] == question_body
        and (reference_body is None or bodies["reference"] == reference_body), "question inputs changed during preflight")
    # Pin the bytes that actually supplied the in-memory corpus/matrix. Later
    # closure checks alone cannot detect a mutation between loading and capture.
    verified_again = load_verified_corpus(artifact_dir=directory, source_registry=registry)
    _require(verified_again == corpus, "corpus changed before input capture")
    if vector is not None:
        import numpy as np
        matrix_again = load_local_bge_vectors(artifact_dir=directory, source_registry=registry, vector_dir=vector)
        _require(np.array_equal(matrix, matrix_again), "vectors changed before input capture")
    identities = {name: {"path": str(path), **_identity(bodies[name]), "snapshot": "inputs/" + name}
                  for name, path in paths.items()}
    identities["config"] = {"path": None, **_identity(config_body), "snapshot": "inputs/config"}
    if model_identity is not None:
        for name, identity in model_identity["files"].items():
            path = cache / "models--BAAI--bge-small-en-v1.5" / "snapshots" / _MODEL_REVISION / name
            identities["model/" + name] = {"path": str(path), **identity, "snapshot": None}

    def unchanged():
        _complete_input(directory)
        if vector is not None:
            _complete_input(vector)
        _require(set(_package_files()) == {name.removeprefix("package/") for name in paths if name.startswith("package/")},
                 "installed package module inventory changed")
        for name, path in paths.items():
            _path(path, "read-only input")
            _require(_file_identity(path) == _identity(bodies[name]), "read-only input closure changed")
        if model_identity is not None:
            _require(model_snapshot_identity(cache) == model_identity, "selected model snapshot closure changed")
        if snapshot is not None:
            snapshot.verify_unchanged()
            snapshot.verify_report_bindings()

    unchanged()
    output.mkdir(exist_ok=False)
    writer, attempts = _Writer(output), []
    # The guard exists before construction even if a later budget or write fails.
    with (output / PENDING_FILE).open("xb") as stream:
        pending_body = canonical_json({"schema_version": 1, "status": "pending"})
        stream.write(pending_body)
        writer.written += len(pending_body)
    try:
        for name, body in bodies.items():
            writer.body(identities[name]["snapshot"], body)
        writer.body(identities["config"]["snapshot"], config_body)
        freeze = {"schema_version": 1, "config": config, "inputs": identities,
            "questions": [asdict(question) for question in registered],
            "reference": [asdict(question) for question in reference_rows]}
        freeze_identity = writer.json(FREEZE_FILE, freeze)
        if backend == "lexical":
            from .corpus import DocumentScopeRetriever
            from .retrieval import TfidfRetriever
            retriever = DocumentScopeRetriever(TfidfRetriever(corpus.chunks), document_ids=scope)
        else:
            from .runtime import build_bge_retriever
            model = load_cached_model(cache) if model_factory is None else model_factory()
            retriever = build_bge_retriever(chunks=corpus.chunks, model=model,
                passage_embeddings=matrix, report_catalog=catalog)
            if routing_policy == "none":
                from .corpus import DocumentScopeRetriever
                retriever = DocumentScopeRetriever(retriever, document_ids=scope)
        unchanged()
        def persist_attempts():
            return writer.progress({"schema_version": 1, "attempts": attempts})
        persist_attempts()
        captured = CapturedRetriever(retriever, questions=registered, chunks=corpus.chunks,
            document_ids=scope, top_k=top_k, attempts=attempts, persist=persist_attempts)
        report = evaluate_retriever(retriever=captured, questions=registered, top_k=top_k,
                                   threshold=threshold, include_review_snapshot=True)
        _require(len(captured.calls) == len(registered), "retrieval did not complete every question")
        unchanged()
        attempts_identity = persist_attempts()
        result = {"schema_version": 1, "freeze_sha256": freeze_identity["sha256"], "report": report, "calls": captured.calls}
        from .evaluation_review import build_review_packet
        result_identity = writer.json(RESULTS_FILE, result)
        packet = build_review_packet(result, freeze_sha256=freeze_identity["sha256"], results_sha256=result_identity["sha256"])
        packet_identity = writer.json(PACKET_FILE, packet)
        # Read back the entire graph while the guard still prevents consumption.
        _validate_graph(output, allow_pending=True)
        unchanged()
        receipt = {"schema_version": 1, "status": "complete", "files": {
            FREEZE_FILE: freeze_identity, RESULTS_FILE: result_identity, PACKET_FILE: packet_identity,
            ATTEMPTS_FILE: attempts_identity},
            "generation_runs": 0, "financial_correctness": "unestablished"}
        writer.json(RECEIPT_FILE, receipt)
        (output / PENDING_FILE).unlink()
        return receipt
    except Exception as error:
        for name, value in ((ATTEMPTS_FILE, {"schema_version": 1, "attempts": attempts}),
                            (FAILURE_FILE, {"schema_version": 1, "status": "failed", "error_type": _error_type(error)})):
            if not os.path.lexists(output / name):
                with (output / name).open("xb") as stream:
                    stream.write(canonical_json(value))
        raise


def _canonical_file(directory, name):
    path = _path(directory / name, "run artifact")
    _require(path.is_relative_to(directory), "run artifact escapes directory")
    body = _read(path)
    value = parse_json(body, max_bytes=MAX_OUTPUT_BYTES)
    _require(canonical_json(value) == body, "run artifact is not canonical JSON")
    return value, body


def _inventory(inputs, config):
    """Reject omitted named bodies and unsupported provenance declarations."""
    expected = {"questions", "source_registry", "config",
                *("corpus/" + name for name in CORPUS_FILES),
                *("package/" + name for name in _package_files())}
    if config["mode"] == "acceptance-candidate":
        expected.add("reference")
    if config["backend"] == "lexical":
        _require(all(config[name] is None for name in (
            "model_execution_origin", "model_name", "model_revision", "device", "local_files_only")),
            "lexical run has unexpected model metadata")
    else:
        _require(config["model_name"] == _MODEL_NAME and config["model_revision"] == _MODEL_REVISION,
                 "invalid pinned model identity")
        origin = config["model_execution_origin"]
        _require(origin in ("cached-local-only", "operator-injected"), "invalid model execution origin")
        _require((config["device"] == "cpu" and config["local_files_only"] is True)
                 if origin == "cached-local-only" else
                 (config["device"] is None and config["local_files_only"] is None),
                 "model execution metadata is inconsistent")
        expected.update("vectors/" + name for name in VECTOR_FILES)
        models = {name for name in inputs if name.startswith("model/")}
        _require({"model/" + name for name in REQUIRED_MODEL_FILES} <= models,
                 "incomplete pinned model inventory")
        _require(all(inputs[name]["bytes"] > 0 for name in models), "empty model inventory body")
        expected.update(models)
    if config["routing_policy"] == "source-bound-period":
        expected.add("statement/fiscal-declarations.json")
        statements = {name for name in inputs if name.startswith("statement/") and name.endswith(".pdf")}
        _require(bool(statements), "missing statement PDF inventory")
        expected.update(statements)
    _require(set(inputs) == expected, "unsupported or incomplete input inventory")


def _validate_graph(directory, *, allow_pending=False):
    """Reconstruct metrics only from retained candidates; never execute a model."""
    if not allow_pending:
        _complete_input(directory)
    freeze, freeze_body = _canonical_file(directory, FREEZE_FILE)
    result, result_body = _canonical_file(directory, RESULTS_FILE)
    packet, packet_body = _canonical_file(directory, PACKET_FILE)
    _keys(freeze, {"schema_version", "config", "inputs", "questions", "reference"}, "freeze")
    _keys(result, {"schema_version", "freeze_sha256", "report", "calls"}, "results")
    _require(type(freeze["schema_version"]) is int and freeze["schema_version"] == 1
        and type(result["schema_version"]) is int and result["schema_version"] == 1, "unsupported run schema")
    _require(result["freeze_sha256"] == sha256(freeze_body).hexdigest(), "results freeze association mismatch")
    config, inputs = freeze["config"], freeze["inputs"]
    _keys(config, _CONFIG_FIELDS, "configuration")
    _require(type(inputs) is dict and {"questions", "source_registry", "config",
        *("corpus/" + name for name in CORPUS_FILES)} <= inputs.keys(), "incomplete input inventory")
    snapshots = {}
    for name, row in inputs.items():
        _require(type(name) is str and bool(name) and "\\" not in name and ":" not in name
                 and all(part not in ("", ".", "..") for part in name.split("/")), "invalid input name")
        _keys(row, {"path", "bytes", "sha256", "snapshot"}, "input identity")
        _validate_identity(row)
        _require(row["path"] is None if name == "config" else type(row["path"]) is str and (
            PurePosixPath(row["path"]).is_absolute() or PureWindowsPath(row["path"]).is_absolute()),
                 "invalid original input path")
        if name.startswith("model/"):
            _require(row["snapshot"] is None, "model inventory unexpectedly copies weights")
            continue
        _require(row["snapshot"] == "inputs/" + name, "input snapshot association mismatch")
        body = _read(_path(directory / row["snapshot"], "input snapshot"))
        _require(_identity(body) == {"bytes": row["bytes"], "sha256": row["sha256"]}, "input snapshot identity changed")
        snapshots[name] = body
    _inventory(inputs, config)
    expected_files = {row["snapshot"] for row in inputs.values() if row["snapshot"] is not None}
    expected_files.update({FREEZE_FILE, RESULTS_FILE, PACKET_FILE, ATTEMPTS_FILE,
                           PENDING_FILE if allow_pending else RECEIPT_FILE})
    actual_files = set()
    total_bytes = 0
    for parent, directories, filenames in os.walk(directory, followlinks=False):
        _require(not any(_linked(Path(parent) / name) for name in directories), "linked run directory")
        for name in filenames:
            path = _path(Path(parent) / name, "run inventory")
            actual_files.add(path.relative_to(directory).as_posix())
            total_bytes += path.stat().st_size
    _require(actual_files == expected_files, "unexpected or missing run file inventory")
    _require(total_bytes <= MAX_OUTPUT_BYTES, "complete output exceeds byte budget")
    _require(snapshots["config"] == canonical_json(config), "configuration snapshot mismatch")
    registered = parse_questions(snapshots["questions"])
    _require(bool(registered), "questions must be nonempty")
    reference = parse_questions(snapshots["reference"]) if "reference" in snapshots else ()
    _require(canonical_json(freeze["questions"]) == canonical_json([asdict(q) for q in registered])
        and canonical_json(freeze["reference"]) == canonical_json([asdict(q) for q in reference]), "registered rows differ from raw inputs")
    _registrations(registered, reference, config["mode"])
    _settings(mode=config["mode"], backend=config["backend"], top_k=config["top_k"], threshold=config["threshold"],
        document_ids=frozenset(config["document_ids"]) if type(config["document_ids"]) is list else config["document_ids"],
        routing_policy=config["routing_policy"], vector_dir=True if config["backend"] == "cached-bge" else None,
        model_cache_dir=True if config["backend"] == "cached-bge" else None,
        statement_source_dir=True if config["routing_policy"] == "source-bound-period" else None,
        statement_fiscal_declarations=True if config["routing_policy"] == "source-bound-period" else None,
        reference=True if "reference" in snapshots else None, model_factory=None)
    _require(type(config["document_ids"]) is list and config["document_ids"] == sorted(set(config["document_ids"])), "invalid frozen scope")
    from .corpus import load_verified_corpus
    corpus = load_verified_corpus(artifact_dir=directory / "inputs/corpus", source_registry=directory / "inputs/source_registry")
    _require(set(config["document_ids"]) <= {s.document_id for s in corpus.sources}, "frozen scope is not registered")
    page_counts = {row["document_id"]: row["page_count"] for row in corpus.manifest["documents"]}
    _require(all(doc in page_counts and page <= page_counts[doc] for question in registered
                 for doc, page in zip(question.evidence_document_ids, question.evidence_pages)),
                 "question labels exceed retained source bounds")
    if config["backend"] == "cached-bge":
        from .corpus_embeddings import load_local_bge_vectors
        load_local_bge_vectors(artifact_dir=directory / "inputs/corpus",
            source_registry=directory / "inputs/source_registry", vector_dir=directory / "inputs/vectors")
    if config["routing_policy"] == "source-bound-period":
        from .statement_startup import load_statement_snapshot
        from .period_routing import build_report_catalog
        source = directory / "inputs/statement"
        statement = load_statement_snapshot(source_registry=directory / "inputs/source_registry", source_dir=source,
            fiscal_declarations=source / "fiscal-declarations.json", document_ids=frozenset(config["document_ids"]),
            expected_sources=corpus.sources, corpus_manifest=directory / "inputs/corpus/manifest.json",
            chunks_path=directory / "inputs/corpus/chunks.jsonl")
        build_report_catalog(statement, corpus_manifest=directory / "inputs/corpus/manifest.json",
                             chunks_path=directory / "inputs/corpus/chunks.jsonl")
    calls = result["calls"]
    _require(type(calls) is list and len(calls) == len(registered), "incomplete call denominator")
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    # Validate the candidate schema and hashes before reconstruction.
    from .evaluation_review import build_review_packet
    expected_packet = build_review_packet(result, freeze_sha256=sha256(freeze_body).hexdigest(), results_sha256=sha256(result_body).hexdigest())
    _require(canonical_json(packet) == canonical_json(expected_packet), "review packet differs from result graph")
    class Replay:
        index = 0
        def retrieve(self, query, *, top_k):
            _require(self.index < len(calls), "unexpected replay call")
            call, question = calls[self.index], registered[self.index]
            self.index += 1
            _require(call["question_id"] == question.question_id and call["query"] == query, "call/question association mismatch")
            values = []
            for row in call["candidates"]:
                chunk = chunks.get(row["chunk"]["chunk_id"])
                _require(chunk is not None and canonical_json(row["chunk"]) == canonical_json(asdict(chunk)),
                         "recorded candidate differs from retained verified corpus")
                values.append(ScoredChunk(chunk, row["score"]))
            return values
    replay = CapturedRetriever(Replay(), questions=registered, chunks=corpus.chunks,
        document_ids=frozenset(config["document_ids"]), top_k=config["top_k"], attempts=[])
    expected_report = evaluate_retriever(retriever=replay, questions=registered, top_k=config["top_k"],
        threshold=config["threshold"], include_review_snapshot=True)
    _require(canonical_json(result["report"]) == canonical_json(expected_report), "report differs from recorded candidates")
    _require(canonical_json(calls) == canonical_json(replay.calls), "recorded calls differ from validated replay")
    attempts, attempts_body = _canonical_file(directory, ATTEMPTS_FILE)
    _require(canonical_json(attempts) == canonical_json({"schema_version": 1, "attempts": replay.attempts}),
             "attempt journal differs from completed calls")
    return {"freeze": freeze, "results": result, "packet": packet}, {
        FREEZE_FILE: _identity(freeze_body), RESULTS_FILE: _identity(result_body),
        PACKET_FILE: _identity(packet_body), ATTEMPTS_FILE: _identity(attempts_body)}


def load_completed_run(run_dir):
    """Validate a portable completed artifact graph without original input access."""
    directory = _path(run_dir, "run_dir", directory=True)
    _complete_input(directory)
    receipt, receipt_body = _canonical_file(directory, RECEIPT_FILE)
    _keys(receipt, {"schema_version", "status", "files", "generation_runs", "financial_correctness"}, "receipt")
    _require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
        and receipt["status"] == "complete" and type(receipt["generation_runs"]) is int
        and receipt["generation_runs"] == 0 and receipt["financial_correctness"] == "unestablished", "invalid completion scope")
    graph, identities = _validate_graph(directory)
    _require(canonical_json(receipt["files"]) == canonical_json(identities), "completion file identities changed")
    # Detect concurrent changes across all copied bodies and the terminal receipt.
    again, again_identities = _validate_graph(directory)
    _require(canonical_json(again) == canonical_json(graph) and again_identities == identities
        and _read(directory / RECEIPT_FILE) == receipt_body, "completed run changed during loading")
    _complete_input(directory)
    return {**graph, "receipt": receipt}
