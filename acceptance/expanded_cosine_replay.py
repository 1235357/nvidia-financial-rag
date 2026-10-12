"""Independent stored-vector arithmetic replay, with no model/ranking imports.

Matching unsigned local receipts establishes reproducible arithmetic only.
Development candidate labels are attached after ranking, never used to rank.
No PDF, encoder, retriever, service, generator or network operation runs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from datetime import date
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from types import SimpleNamespace

import numpy as np

from acceptance.expanded_cases import INPUTS, _read, load_case_bundle, load_experiment

ATOL = 1e-12
FILES = frozenset({"queries-raw.npy", "queries-normalized.npy", "query-index.json", "results.json"})
INDEX_FIELDS = frozenset({"row", "case_id", "scope_id", "normalized_query_sha256",
                         "instructed_query_sha256", "original_record_sha256", "encoder_call_count"})
ROW_FIELDS = frozenset({"row", "case_id", "scope_id", "scope_document_ids", "scoped_global_chunk_indices",
                       "considered_scores", "top_k", "evidence_ids", "citations", "can_answer", "message",
                       "context_sha256", "context_empty", "threshold", "top_k_limit"})
TOP_FIELDS = frozenset({"global_chunk_index", "chunk_id", "text_sha256", "score"})
REFUSAL = "I cannot answer from the provided reports because no evidence passed the retrieval gate."
MAX_BYTES = 256_000_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(body):
    return sha256(body).hexdigest()


def text_digest(text):
    return digest(text.encode("utf-8"))


def keys(value, fields, label):
    require(type(value) is dict and value.keys() == fields, f"{label} fields differ from schema")


def strict_json(body):
    """Bounded strict receipt JSON; score arrays may exceed case-parser limits."""
    require(type(body) is bytes and len(body) <= MAX_BYTES, "JSON byte budget exceeded")
    def pairs(items):
        result = {}
        for name, value in items:
            require(name not in result, "duplicate JSON member")
            result[name] = value
        return result
    def reject(_):
        raise ValueError("non-finite JSON value")
    def bounded(value, level=0):
        require(level <= 24, "JSON nesting budget exceeded")
        if type(value) is dict:
            require(len(value) <= 1024, "JSON object budget exceeded")
            for item in value.values():
                bounded(item, level + 1)
        elif type(value) is list:
            require(len(value) <= 10_000_000, "JSON list budget exceeded")
            for item in value:
                bounded(item, level + 1)
        elif type(value) is float:
            require(math.isfinite(value), "non-finite JSON number")
        elif type(value) is str:
            require(not any(0xD800 <= ord(c) <= 0xDFFF for c in value), "invalid JSON Unicode")
    try:
        text = body.decode("utf-8")
        require(not text.startswith("\ufeff"), "JSON BOM is unsupported")
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=reject)
        bounded(value)
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("invalid bounded UTF-8 JSON") from None


def vector(value, rows, dtype, label):
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype), f"{label} vector dtype differs")
    require(value.shape == (rows, 384), f"{label} vector shape differs")
    require(np.all(np.isfinite(value)), f"{label} vector must be finite")
    promoted = np.asarray(value, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(promoted, axis=1, keepdims=True)
    require(np.all(np.isfinite(norms)) and np.all(norms > 0), f"{label} vector norm must be finite and nonzero")
    return promoted / norms


def score_array(values, count, label):
    require(type(values) is list and len(values) == count and all(
        type(v) in (int, float) and -1 - ATOL <= v <= 1 + ATOL and math.isfinite(v) for v in values), f"invalid {label}")
    return np.asarray(values, dtype=np.float64)


def classify_boundaries(scores, *, top_k, threshold):
    require(isinstance(scores, np.ndarray) and scores.ndim == 1 and np.all(np.isfinite(scores)),
            "boundary scores must be finite")
    order = np.argsort(scores)[::-1]
    boundary = bool(len(order) > top_k and abs(float(scores[order[top_k - 1]] - scores[order[top_k]])) <= ATOL)
    near = np.flatnonzero(np.abs(scores - threshold) <= ATOL).tolist()
    tied = []
    if boundary:
        boundary_score = scores[order[top_k - 1]]
        tied = np.flatnonzero(np.abs(scores - boundary_score) <= ATOL).tolist()
    return {"top_k_boundary_indeterminate": boundary,
            "threshold_boundary_indeterminate": bool(near),
            "near_threshold_local_indices": near, "top_k_boundary_local_indices": tied}


def replay_rows(*, bundle, chunks, passages, raw_queries, normalized_queries, query_index, results):
    """Replay all rows from exact stored f32 inputs and consumed f64 queries."""
    cfg = bundle.experiment.data
    requests = bundle.retrieval_inputs()
    count = len(requests)
    require(len(chunks) == cfg["chunk_count"], "corpus chunk count differs")
    matrix = vector(passages, len(chunks), "float32", "passage")
    query_matrix = vector(raw_queries, count, "float32", "raw query")
    vector(normalized_queries, count, "float64", "normalized query")
    require(np.allclose(query_matrix, normalized_queries, atol=ATOL, rtol=0), "normalized query differs from raw float32 replay")
    for body, label in ((query_index, "query index"), (results, "results")):
        keys(body, {"schema_version", "rows"}, label)
        require(type(body["schema_version"]) is int and body["schema_version"] == 2
                and type(body["rows"]) is list and len(body["rows"]) == count, f"{label} row count/schema differs")
    origins = {case.case_id: case.origin["original_record_sha256"] for case in bundle.cases}
    observations = []
    for row, (request, index, recorded) in enumerate(zip(requests, query_index["rows"], results["rows"], strict=True)):
        keys(index, INDEX_FIELDS, "query index row")
        query = request.query.strip()
        expected_index = {"row": row, "case_id": request.case_id, "scope_id": request.scope_id,
                          "normalized_query_sha256": text_digest(query),
                          "instructed_query_sha256": text_digest(cfg["model"]["instruction"] + query),
                          "original_record_sha256": origins[request.case_id], "encoder_call_count": 1}
        require(type(index["row"]) is int and type(index["encoder_call_count"]) is int
                and index == expected_index, "query index order/instruction/identity differs")
        keys(recorded, ROW_FIELDS, "result row")
        require(type(recorded["row"]) is int and recorded["row"] == row
                and recorded["case_id"] == request.case_id and recorded["scope_id"] == request.scope_id,
                "result row identity/order differs")
        selected = [i for i, chunk in enumerate(chunks) if request.document_ids is None or chunk.document_id in request.document_ids]
        require(selected, "scope must contain registered corpus chunks")
        declared_scope = None if request.document_ids is None else list(request.document_ids)
        require(recorded["scope_document_ids"] == declared_scope
                and type(recorded["scoped_global_chunk_indices"]) is list
                and all(type(i) is int for i in recorded["scoped_global_chunk_indices"])
                and recorded["scoped_global_chunk_indices"] == selected, "scope global indices/documents differ")
        require(type(recorded["threshold"]) in (int, float) and recorded["threshold"] == cfg["primary_threshold"]
                and type(recorded["top_k_limit"]) is int and recorded["top_k_limit"] == cfg["top_k"], "gate parameters differ")
        scores = matrix[selected] @ query_matrix[row]
        saved = score_array(recorded["considered_scores"], len(selected), "considered scores")
        require(np.allclose(scores, saved, atol=ATOL, rtol=0), "considered scores differ from independent replay")
        boundaries = classify_boundaries(scores, top_k=cfg["top_k"], threshold=cfg["primary_threshold"])
        top = recorded["top_k"]
        require(type(top) is list and len(top) == min(cfg["top_k"], len(selected)), "Top-K count differs")
        local_by_global = {global_i: local_i for local_i, global_i in enumerate(selected)}
        producer_indices = []
        for entry in top:
            keys(entry, TOP_FIELDS, "Top-K row")
            global_i = entry["global_chunk_index"]
            require(type(global_i) is int and global_i in local_by_global and global_i not in producer_indices,
                    "Top-K index outside scope or duplicate")
            local_i = local_by_global[global_i]
            chunk = chunks[global_i]
            require(entry["chunk_id"] == chunk.chunk_id and entry["text_sha256"] == text_digest(chunk.text), "Top-K chunk/text identity differs")
            require(type(entry["score"]) in (int, float) and -1 - ATOL <= entry["score"] <= 1 + ATOL and math.isfinite(entry["score"])
                    and entry["score"] == float(saved[local_i]), "Top-K recorded score differs")
            scalar = math.fsum(float(a) * float(b) for a, b in zip(matrix[global_i], query_matrix[row], strict=True))
            require(math.isclose(scalar, entry["score"], abs_tol=ATOL, rel_tol=0), "Top-K scalar dot differs")
            producer_indices.append(global_i)
        # Preserve producer order in tied groups. A non-tied inversion or omitted
        # clearly larger score is still corruption, even when one boundary ties.
        expected_local = np.argsort(scores)[::-1][:cfg["top_k"]]
        for rank, global_i in enumerate(producer_indices):
            require(abs(float(scores[local_by_global[global_i]] - scores[expected_local[rank]])) <= ATOL,
                    "Top-K order/membership differs")
        accepted = [entry for entry in top if entry["score"] >= cfg["primary_threshold"]]
        accepted_ids = [entry["chunk_id"] for entry in accepted]
        require(type(recorded["evidence_ids"]) is list and recorded["evidence_ids"] == accepted_ids,
                "evidence gate IDs differ")
        citations = [f"{chunks[entry['global_chunk_index']].document_id} p.{chunks[entry['global_chunk_index']].page_number}" for entry in accepted]
        require(type(recorded["citations"]) is list and recorded["citations"] == citations, "citation order/content differs")
        context = "\n\n".join(f"[{citation}]\n{chunks[entry['global_chunk_index']].text}" for citation, entry in zip(citations, accepted, strict=True))
        require(type(recorded["can_answer"]) is bool and recorded["can_answer"] is bool(accepted)
                and type(recorded["context_empty"]) is bool and recorded["context_empty"] is (context == "")
                and recorded["context_sha256"] == text_digest(context)
                and recorded["message"] == (f"Evidence retrieved for: {query}" if accepted else REFUSAL),
                "context/gate/message differs")
        indeterminate = boundaries["top_k_boundary_indeterminate"] or boundaries["threshold_boundary_indeterminate"]
        observations.append({"row": row, "case_id": request.case_id, "scope_id": request.scope_id,
                             "status": "boundary_indeterminate" if indeterminate else "verified", **boundaries,
                             "producer_top_k_indices": producer_indices, "scalar_dot_controls": len(top),
                             "max_score_absolute_error": float(np.max(np.abs(scores - saved))),
                             "software_gate_can_answer": bool(accepted)})
    # Candidate labels enter only this reporting step, after every ranking replay.
    labels = {case.case_id: case.expected["outcome"] for case in bundle.cases}
    for observation in observations:
        negative = labels[observation["case_id"]] in ("refuse", "clarify")
        accepted = negative and observation["software_gate_can_answer"]
        observation["negative_candidate_accepted"] = accepted
        observation["negative_observation"] = ("negative candidate accepted by the software gate" if accepted
            else "negative candidate refused by the software gate" if negative else None)
    return {"schema_version": 2, "arithmetic_passed": True, "absolute_tolerance": ATOL, "relative_tolerance": 0,
            "status": "boundary_indeterminate" if any(o["status"] == "boundary_indeterminate" for o in observations) else "verified",
            "human_gold_count": 0, "generation_runs": 0, "financial_readiness": "not_established",
            "source_field_binding": "not_replayed_by_arithmetic_verifier",
            "query_count": count, "rows": observations,
            "replay_runtime": {"numpy_version": np.__version__, "python_version": platform.python_version(), "platform": platform.platform()}}


def plain(value):
    if isinstance(value, Mapping):
        return {name: plain(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def read_body(path):
    return _read(path, MAX_BYTES, "replay input")


def directory(path, label):
    require(isinstance(path, Path) and path.is_absolute() and path.is_dir(), f"{label} must be an absolute local directory")
    require(not any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction())
                    for p in (path, *path.parents)), f"{label} must be an unlinked local directory")
    return path.resolve(strict=True)


def completed(directories, source_replay):
    for path in directories:
        require(not os.path.lexists(path / "pending.json") and not os.path.lexists(path / "failure.json"),
                "input is pending or has a failure marker")
    for marker in ("pending", "failure"):
        require(not os.path.lexists(source_replay.with_name(source_replay.name + f".{marker}.json")),
                "source replay is pending or has a failure marker")


def integer(value, minimum=1):
    return type(value) is int and value >= minimum


def runtime_record(value, names, label):
    keys(value, names, label)
    require(all(type(v) is str and 0 < len(v) <= 128 and v == v.strip() for v in value.values()), f"{label} versions differ")


def module_record(value, names, label):
    keys(value, names, label)
    require(all(type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v) for v in value.values()), f"{label} SHA-256 differs")


def fixed_record(value, expected, label):
    keys(value, expected.keys(), label)
    require(all(type(value[name]) is type(wanted) and value[name] == wanted for name, wanted in expected.items()), f"{label} values differ")


def load_corpus_vectors(bodies, cfg):
    """Read frozen corpus and vectors without importing production modules."""
    registry = strict_json(bodies["registry"])
    keys(registry, {"schema_version", "sources"}, "registry")
    require(type(registry["schema_version"]) is int and registry["schema_version"] == 1
            and type(registry["sources"]) is list and len(registry["sources"]) == cfg["document_count"], "registry schema/count differs")
    manifest = strict_json(bodies["corpus_manifest"])
    required = {"schema_version", "chunks_sha256", "chunking", "documents", "document_count", "page_count", "nonempty_page_count", "chunk_count"}
    require(type(manifest) is dict and required <= manifest.keys() <= required | {"extraction_mode"}, "corpus manifest fields differ")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
            and manifest.get("extraction_mode", "plain") == "plain" and manifest["chunks_sha256"] == digest(bodies["corpus_chunks"]),
            "corpus schema/extraction/chunk identity differs")
    chunking = {"encoding_name": "cl100k_base", "max_tokens": 400, "overlap_tokens": 0, "strategy": "word_safe"}
    fixed_record(manifest["chunking"], chunking, "corpus chunking")
    require(manifest["chunking"] == chunking and all(integer(manifest[name]) for name in
            ("document_count", "page_count", "nonempty_page_count", "chunk_count"))
            and manifest["document_count"] == cfg["document_count"] and manifest["chunk_count"] == cfg["chunk_count"], "corpus counts/profile differ")
    documents = manifest["documents"]
    require(type(documents) is list and len(documents) == cfg["document_count"], "corpus document count differs")
    by_id = {}
    for reg, doc, wanted in zip(registry["sources"], documents, cfg["documents"], strict=True):
        reg_fields = {"document_id", "label", "url", "filename", "sha256"}
        require(type(reg) is dict and reg_fields <= reg.keys() <= reg_fields | {"reporting_period"}, "registered document fields differ")
        require(all(type(reg[name]) is str and reg[name].strip() == reg[name] and reg[name] for name in reg_fields)
                and not any(c in reg["filename"] for c in ("/", "\\", ":")), "registered document strings differ")
        doc_fields = {"document_id", "document_label", "source_url", "source_filename", "source_sha256", "page_count", "nonempty_page_count", "chunk_count"}
        require(type(doc) is dict and doc_fields <= doc.keys() <= doc_fields | {"reporting_period"}, "corpus document fields differ")
        require(doc["document_id"] == reg["document_id"] == wanted["document_id"] and reg["sha256"] == doc["source_sha256"] == wanted["pdf_sha256"]
                and reg["url"] == doc["source_url"] == wanted["issuer_url"] and reg["label"] == doc["document_label"]
                and reg["filename"] == doc["source_filename"] and doc["page_count"] == wanted["page_count"]
                and reg.get("reporting_period") == doc.get("reporting_period"), "corpus registry/config document identity/order differs")
        period = reg.get("reporting_period")
        require(('reporting_period' in reg) == ('reporting_period' in doc), 'registered period presence differs')
        if 'reporting_period' in reg:
            keys(period, {'fiscal_year', 'quarter', 'end_date', 'source_locator', 'reviewed_by'}, 'registered period')
            require(type(period['fiscal_year']) is int and 1900 <= period['fiscal_year'] <= 9999
                    and type(period['quarter']) is int and 1 <= period['quarter'] <= 4
                    and type(period['end_date']) is str and len(period['end_date']) == 10,
                    'registered period types differ')
            require(date.fromisoformat(period['end_date']).isoformat() == period['end_date']
                    and all(type(period[name]) is str and period[name].strip() for name in ('source_locator', 'reviewed_by')),
                    'registered period date or attribution differs')
        require(period is None or
                (type(period) is dict and period.get("fiscal_year") == wanted["report_fiscal_year"] and period.get("quarter") == wanted["report_fiscal_quarter"]),
                "registered report period differs")
        require(all(integer(doc[name]) for name in ("page_count", "nonempty_page_count", "chunk_count"))
                and doc["nonempty_page_count"] <= doc["page_count"] and doc["document_id"] not in by_id, "corpus document bounds differ")
        by_id[doc["document_id"]] = doc
    for name in ("page_count", "nonempty_page_count", "chunk_count"):
        require(sum(doc[name] for doc in documents) == manifest[name], "corpus count total differs")
    chunks, last, counts, pages, ids = [], (-1, 0), Counter(), {name: set() for name in by_id}, set()
    order = {name: i for i, name in enumerate(by_id)}
    for line in bodies["corpus_chunks"].split(b"\n"):
        if not line.strip():
            continue
        chunk = strict_json(line)
        keys(chunk, {"chunk_id", "document_id", "page_number", "text"}, "chunk")
        require(all(type(chunk[name]) is str and chunk[name].strip() for name in ("chunk_id", "document_id", "text"))
                and integer(chunk["page_number"]) and chunk["document_id"] in by_id, "chunk metadata differs")
        doc_id, page = chunk["document_id"], chunk["page_number"]
        prefix = f"{doc_id}:p{page}:w:"
        require(page <= by_id[doc_id]["page_count"] and chunk["chunk_id"].startswith(prefix)
                and re.fullmatch(r"[0-9a-f]{16}", chunk["chunk_id"][len(prefix):]) and chunk["chunk_id"] not in ids
                and (order[doc_id], page) >= last, "chunk identity/order/page differs")
        ids.add(chunk["chunk_id"])
        last = (order[doc_id], page)
        counts[doc_id] += 1
        pages[doc_id].add(page)
        chunks.append(SimpleNamespace(**chunk))
    require(len(chunks) == cfg["chunk_count"] and all(counts[name] == doc["chunk_count"]
            and len(pages[name]) == doc["nonempty_page_count"] for name, doc in by_id.items()), "actual corpus membership/count differs")
    receipt = strict_json(bodies["build_receipt"])
    keys(receipt, {"schema_version", "status", "scope", "profile", "source_registry_sha256", "manifest_sha256", "chunks_sha256",
                   "runtime", "module_sha256", "sources", "observed_counts", "human_acceptance"}, "corpus build receipt")
    require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1 and receipt["status"] == "complete"
            and receipt["scope"] == "operator-local-corpus-build" and receipt["human_acceptance"] is False
            and receipt["profile"] == {"name": "plain-word-safe-400-0-v1", "extraction_mode": "plain", **chunking}, "corpus completion/profile differs")
    fixed_record(receipt["profile"], {"name": "plain-word-safe-400-0-v1", "extraction_mode": "plain", **chunking}, "corpus profile")
    runtime_record(receipt["runtime"], {"python", "PyMuPDF", "numpy", "scikit-learn", "tiktoken"}, "corpus runtime")
    module_record(receipt["module_sha256"], {"core.py", "ingest.py", "sources.py", "corpus.py"}, "corpus module identities")
    closure = {"source_registry_sha256": digest(bodies["registry"]), "manifest_sha256": digest(bodies["corpus_manifest"]),
               "chunks_sha256": digest(bodies["corpus_chunks"]), "build_receipt_sha256": digest(bodies["build_receipt"])}
    require(all(receipt[name] == closure[name] for name in closure if name != "build_receipt_sha256")
            and receipt["observed_counts"] == {name: manifest[name] for name in ("document_count", "page_count", "nonempty_page_count", "chunk_count")}, "corpus build closure differs")
    require(type(receipt["sources"]) is list and len(receipt["sources"]) == len(documents), "corpus source receipt count differs")
    for source, wanted in zip(receipt["sources"], cfg["documents"], strict=True):
        keys(source, {"document_id", "source_sha256", "source_bytes"}, "corpus source receipt")
        require(source["document_id"] == wanted["document_id"] and source["source_sha256"] == wanted["pdf_sha256"] and integer(source["source_bytes"]), "corpus source receipt differs")
    embedding = strict_json(bodies["vector_manifest"])
    keys(embedding, {"schema_version", "model_name", "model_revision", "build_runtime", "normalized", "chunk_count", "chunk_ids_sha256",
                     "chunks_sha256", "embeddings_sha256", "shape", "dtype"}, "embedding manifest")
    ids_digest = text_digest("\n".join(chunk.chunk_id for chunk in chunks))
    runtime_record(embedding["build_runtime"], {"python", "numpy", "sentence-transformers", "torch", "transformers"}, "vector runtime")
    require(type(embedding["schema_version"]) is int and embedding["schema_version"] == 2 and embedding["model_name"] == cfg["model"]["name"]
            and embedding["model_revision"] == cfg["model"]["revision"] and embedding["normalized"] is True
            and type(embedding["chunk_count"]) is int and embedding["chunk_count"] == len(chunks)
            and embedding["chunk_ids_sha256"] == ids_digest and embedding["chunks_sha256"] == closure["chunks_sha256"]
            and embedding["embeddings_sha256"] == digest(bodies["vector_matrix"]) and embedding["shape"] == [len(chunks), 384]
            and embedding["dtype"] == "float32", "embedding manifest matrix/model/chunk closure differs")
    vector_receipt = strict_json(bodies["vector_receipt"])
    keys(vector_receipt, {"schema_version", "status", "scope", "corpus", "embeddings_sha256", "embedding_manifest_sha256", "chunk_ids_sha256",
                          "matrix", "model", "runtime", "module_sha256", "wheel", "post_build_verified", "generation_runs", "human_acceptance"}, "vector build receipt")
    require(type(vector_receipt["schema_version"]) is int and vector_receipt["schema_version"] == 1 and vector_receipt["status"] == "complete"
            and vector_receipt["scope"] == "operator-local-bge-vector-build" and vector_receipt["corpus"] == closure
            and vector_receipt["embeddings_sha256"] == embedding["embeddings_sha256"] and vector_receipt["embedding_manifest_sha256"] == digest(bodies["vector_manifest"])
            and vector_receipt["chunk_ids_sha256"] == ids_digest and vector_receipt["matrix"] == {"dtype": "float32", "normalized": True, "shape": [len(chunks), 384]}
            and vector_receipt["runtime"] == embedding["build_runtime"] and vector_receipt["post_build_verified"] is True
            and integer(vector_receipt["generation_runs"], 0) and vector_receipt["generation_runs"] == 0 and vector_receipt["human_acceptance"] is False,
            "vector completion/corpus closure differs")
    require(vector_receipt["model"] == {"name": cfg["model"]["name"], "revision": cfg["model"]["revision"], "device": "cpu",
                                        "execution_origin": "cached-local-only", "local_files_only": True}, "vector cached model profile differs")
    fixed_record(vector_receipt["model"], {"name": cfg["model"]["name"], "revision": cfg["model"]["revision"], "device": "cpu",
                                          "execution_origin": "cached-local-only", "local_files_only": True}, "vector model")
    modules = {"__init__.py", "core.py", "ingest.py", "sources.py", "corpus.py", "embedding_store.py",
               "corpus_embeddings.py", "runtime.py", "retrieval.py", "service.py"}
    module_record(vector_receipt["module_sha256"], modules, "vector module identities")
    wheel = vector_receipt["wheel"]
    keys(wheel, {"wheel_basename", "wheel_bytes", "wheel_sha256", "distribution_version", "module_sha256"}, "vector wheel identity")
    require(wheel["wheel_basename"] == "nvidia_financial_rag-0.1.0-py3-none-any.whl" and integer(wheel["wheel_bytes"])
            and wheel["wheel_sha256"] == cfg["input_sha256"]["wheel"] and type(wheel["distribution_version"]) is str
            and wheel["distribution_version"].strip(), "vector wheel identity differs from frozen experiment")
    require(type(wheel["module_sha256"]) is dict and modules <= wheel["module_sha256"].keys()
            and all(type(name) is str and re.fullmatch(r"[a-z_]+\.py", name) and type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)
                    for name, value in wheel["module_sha256"].items())
            and all(wheel["module_sha256"][name] == value for name, value in vector_receipt["module_sha256"].items()),
            "vector wheel/module identities differ")
    matrix = np.load(io.BytesIO(bodies["vector_matrix"]), allow_pickle=False)
    vector(matrix, len(chunks), "float32", "passage")
    require(np.allclose(np.linalg.norm(matrix, axis=1), 1., rtol=1e-4, atol=1e-5), "stored passage vectors are not normalized float32")
    return tuple(chunks), matrix


def run_replay(*, artifact_dir, source_registry, vector_dir, experiment, case_freeze, comparison_dir, output=None):
    """Close current input/output identities, then independently replay every row."""
    artifact_dir = directory(artifact_dir, "artifact_dir")
    vector_dir = directory(vector_dir, "vector_dir")
    comparison_dir = directory(comparison_dir, "comparison_dir")
    experiment_record = load_experiment(experiment)
    cfg = experiment_record.data
    manifest_path = comparison_dir / "manifest.json"
    manifest_body = read_body(manifest_path)
    manifest = strict_json(manifest_body)
    keys(manifest, {"schema_version", "status", "experiment_sha256", "input_sha256", "input_paths", "producer", "query_count", "files"}, "comparison manifest")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 2 and manifest["status"] == "completed"
            and manifest["experiment_sha256"] == experiment_record.sha256 and manifest["input_sha256"] == plain(cfg["input_sha256"]),
            "comparison completion/experiment/input closure differs")
    keys(manifest["input_paths"], {"cases", "source_replay", "wheel"}, "comparison input paths")
    keys(manifest["files"], FILES, "comparison output identities")
    keys(manifest["producer"], {"numpy_version", "python_version", "platform"}, "producer runtime")
    require(all(type(value) is str and 0 < len(value) <= 512 and value.strip() for value in manifest["producer"].values()), "producer runtime differs")
    for name in manifest["input_paths"]:
        require(type(manifest["input_paths"][name]) is str, "comparison input path type differs")
    paths = {"registry": source_registry, "experiment": experiment, "cases": Path(manifest["input_paths"]["cases"]),
             "case_freeze": case_freeze, "source_replay": Path(manifest["input_paths"]["source_replay"]), "wheel": Path(manifest["input_paths"]["wheel"]),
             "corpus_manifest": artifact_dir / "manifest.json", "corpus_chunks": artifact_dir / "chunks.jsonl", "build_receipt": artifact_dir / "build-receipt.json",
             "vector_manifest": vector_dir / "embedding-manifest.json", "vector_matrix": vector_dir / "bge-small-en-v1.5.npy", "vector_receipt": vector_dir / "vector-build-receipt.json",
             "comparison_manifest": manifest_path, **{name: comparison_dir / name for name in FILES}}
    completed((artifact_dir, vector_dir, comparison_dir), paths["source_replay"])
    bodies = {name: read_body(path) for name, path in paths.items()}
    identities = {name: digest(body) for name, body in bodies.items()}
    require(identities["comparison_manifest"] == digest(manifest_body) and identities["experiment"] == experiment_record.sha256, "comparison/experiment changed during loading")
    require(all(identities[name] == cfg["input_sha256"][name] for name in INPUTS), "current input SHA-256 differs from frozen experiment")
    require(all(identities[name] == manifest["files"][name] for name in FILES), "comparison output SHA-256 differs")
    bundle = load_case_bundle(paths["cases"], case_freeze, experiment_record)
    require(integer(manifest["query_count"]) and manifest["query_count"] == len(bundle.retrieval_inputs()), "comparison query count differs from frozen cases/scopes")
    chunks, passages = load_corpus_vectors(bodies, cfg)
    source = strict_json(bodies["source_replay"])
    keys(source, {"schema_version", "status", "scope", "experiment_binding_sha256", "input_sha256", "source_files", "ordered_case_ids", "counts", "corpus_profile",
                  "chunk_ids", "cells", "rationale_bindings", "source_binding_passed", "human_gold_count", "generation_runs", "joint_source_candidate_coverage", "limitations"}, "source replay receipt")
    bound = plain(cfg)
    bound["input_sha256"].pop("source_replay")
    binding_digest = digest(json.dumps(bound, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode())
    require(type(source["schema_version"]) is int and source["schema_version"] == 2 and source["status"] == "complete"
            and source["scope"] == "operator-local-source-candidate-replay" and source["experiment_binding_sha256"] == binding_digest
            and source["ordered_case_ids"] == list(cfg["expected_case_ids"]) and source["counts"] == plain(bundle.counts)
            and source["chunk_ids"] == [chunk.chunk_id for chunk in chunks] and type(source["source_binding_passed"]) is bool
            and integer(source["human_gold_count"], 0) and source["human_gold_count"] == 0
            and integer(source["generation_runs"], 0) and source["generation_runs"] == 0 and source["joint_source_candidate_coverage"] == "not_evaluated",
            "source replay case/config/evidence boundary differs")
    require(source["input_sha256"] == {name: identities[name] for name in ("registry", "cases", "case_freeze", "corpus_manifest", "corpus_chunks", "build_receipt")}, "source replay consumed inputs differ")
    require(type(source["source_files"]) is list and len(source["source_files"]) == cfg["document_count"], "source replay document count differs")
    for actual, doc in zip(source["source_files"], cfg["documents"], strict=True):
        keys(actual, {"document_id", "pdf_sha256", "source_bytes"}, "source replay file")
        require(actual["document_id"] == doc["document_id"] and actual["pdf_sha256"] == doc["pdf_sha256"] and integer(actual["source_bytes"]), "source replay file identity differs")
    result = replay_rows(bundle=bundle, chunks=chunks, passages=passages,
                         raw_queries=np.load(io.BytesIO(bodies["queries-raw.npy"]), allow_pickle=False),
                         normalized_queries=np.load(io.BytesIO(bodies["queries-normalized.npy"]), allow_pickle=False),
                         query_index=strict_json(bodies["query-index.json"]), results=strict_json(bodies["results.json"]))
    result.update(experiment_sha256=experiment_record.sha256, input_sha256=plain(cfg["input_sha256"]),
                  comparison_manifest_sha256=identities["comparison_manifest"], comparison_files_sha256=manifest["files"],
                  producer_runtime=manifest["producer"], source_replay_sha256=identities["source_replay"])
    def final_check():
        completed((artifact_dir, vector_dir, comparison_dir), paths["source_replay"])
        require(all(digest(read_body(path)) == identities[name] for name, path in paths.items()), "input/comparison changed during replay")
        completed((artifact_dir, vector_dir, comparison_dir), paths["source_replay"])
    final_check()
    if output is not None:
        require(isinstance(output, Path) and output.is_absolute(), "output must be an absolute local path")
        directory(output.parent, "output parent")
        require(not any(output.resolve().is_relative_to(path) for path in (artifact_dir, vector_dir, comparison_dir))
                and output.resolve() not in {path.resolve() for path in paths.values()}, "output overlaps a read-only input")
        pending, failure = (output.with_name(output.name + f".{kind}.json") for kind in ("pending", "failure"))
        require(not any(os.path.lexists(path) for path in (output, pending, failure)), "replay output already exists")
        output_body = (json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        output_digest = digest(output_body)
        try:
            with pending.open("xb") as handle:
                handle.write(b'{"status":"pending"}\n')
                handle.flush()
                os.fsync(handle.fileno())
            final_check()
            with output.open("xb") as handle:
                handle.write(output_body)
                handle.flush()
                os.fsync(handle.fileno())
            final_check()
            require(digest(read_body(output)) == output_digest and not os.path.lexists(failure),
                    'owned replay output changed before closure')
            pending.unlink()
            final_check()
            require(digest(read_body(output)) == output_digest, 'owned replay output changed during closure')
            final_check()
            require(not any(os.path.lexists(path) for path in (pending, failure)),
                    'replay completion guard remains or appeared during closure')
        except Exception:
            if not os.path.lexists(pending):
                try:
                    with pending.open('xb') as handle:
                        handle.write(b'{"status":"pending"}\n')
                        handle.flush()
                        os.fsync(handle.fileno())
                except OSError:
                    pass
            try:
                with failure.open("xb") as handle:
                    handle.write(b'{"status":"failed","code":"ARITHMETIC_REPLAY_INCOMPLETE"}\n')
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError:
                pass
            raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-dir", "source-registry", "vector-dir", "experiment", "case-freeze", "comparison-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    try:
        result = run_replay(**args)
    except (ValueError, OSError):
        parser.exit(1, "Independent arithmetic replay refused: input/output contract failed.\n")
    print(json.dumps({"status": result["status"], "query_count": result["query_count"], "human_gold_count": 0, "generation_runs": 0}))


if __name__ == "__main__":
    main()
