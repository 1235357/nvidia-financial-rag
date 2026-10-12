"""Operator-local corpus recipes with unsigned internal-consistency checks.

This opt-in boundary does not change legacy loaders, acquire data, or establish
source rights, financial correctness, or authenticity of adversarial rewrites.
"""
from __future__ import annotations

from collections import Counter
import base64
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
from types import FunctionType

from .core import Chunk
from .ingest import build_corpus_artifacts, load_chunks
from .sources import SourceRecord, file_sha256, load_source_manifest
from .service import EvidenceService

_TOKENIZER_FILENAME = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
_TOKENIZER_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
_TOKENIZER_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
_PROFILE = {"name": "plain-word-safe-400-0-v1", "extraction_mode": "plain",
            "strategy": "word_safe", "max_tokens": 400, "overlap_tokens": 0,
            "encoding_name": "cl100k_base"}
_CHUNKING = {key: _PROFILE[key] for key in
             ("encoding_name", "max_tokens", "overlap_tokens", "strategy")}
_COUNTS = {"document_count", "page_count", "nonempty_page_count", "chunk_count"}
_PACKAGES = ("PyMuPDF", "numpy", "scikit-learn", "tiktoken")
_MODULES = ("core.py", "ingest.py", "sources.py", "corpus.py")


@dataclass(frozen=True)
class VerifiedCorpus:
    chunks: tuple[Chunk, ...]
    manifest: dict[str, object]
    sources: tuple[SourceRecord, ...]
    chunks_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _keys(value, required, *, optional=(), label="object") -> None:
    _require(type(value) is dict, f"{label} must be an object")
    _require(set(required) <= value.keys() <= set(required) | set(optional),
             f"{label} has missing or unsupported fields")


def _string(value, label, *, canonical=False) -> None:
    _require(type(value) is str and bool(value.strip()), f"{label} must be a nonblank string")
    if canonical:
        _require(value == value.strip(), f"{label} must preserve canonical whitespace")


def _integer(value, label, *, minimum=1) -> None:
    _require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")


def _digest(value, label) -> None:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{label} must be a lowercase SHA-256 digest")


def _fixed(value, expected, label) -> None:
    _keys(value, expected, label=label)
    for key, wanted in expected.items():
        _require(type(value[key]) is type(wanted) and value[key] == wanted,
                 f"{label} differs from the fixed profile")


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON member")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("non-finite JSON constant is unsupported")


def _json(body: bytes | str):
    return json.loads(body, object_pairs_hook=_object_pairs, parse_constant=_nonfinite)


def _absolute(path: Path, label: str, *, exists=False) -> Path:
    _require(isinstance(path, Path) and path.is_absolute(), f"{label} must be an absolute local path")
    return path.resolve(strict=exists)


def _period(row):
    _keys(row, {"fiscal_year", "quarter", "end_date", "source_locator", "reviewed_by"},
          label="reporting_period")
    _integer(row["fiscal_year"], "fiscal_year", minimum=1900)
    _require(row["fiscal_year"] <= 9999, "fiscal_year outside supported range")
    _integer(row["quarter"], "quarter")
    _require(row["quarter"] <= 4, "quarter outside supported range")
    for key in ("end_date", "source_locator", "reviewed_by"):
        _string(row[key], key, canonical=True)


def _registry(path: Path):
    body = path.read_bytes()
    raw = _json(body)
    _keys(raw, {"schema_version", "sources"}, label="source registry")
    _require(type(raw["schema_version"]) is int and raw["schema_version"] == 1,
             "local recipe requires source schema 1")
    _require(type(raw["sources"]) is list and bool(raw["sources"]), "sources must be a nonempty array")
    for row in raw["sources"]:
        _keys(row, {"document_id", "label", "url", "filename", "sha256"},
              optional={"reporting_period"}, label="registered source")
        for field in ("document_id", "label", "url", "filename"):
            _string(row[field], field, canonical=True)
        _digest(row["sha256"], "registered source sha256")
        # Safe on both POSIX and Windows, before the legacy platform basename rule.
        _require(not any(c in row["filename"] for c in ("/", "\\", ":")), "unsafe source filename")
        if "reporting_period" in row:
            _period(row["reporting_period"])
    records = load_source_manifest(path)
    _require(path.read_bytes() == body, "source registry changed during validation")
    for record, row in zip(records, raw["sources"], strict=True):
        observed = {"document_id": record.document_id, "label": record.label, "url": record.url,
                    "filename": record.filename, "sha256": record.sha256}
        if record.reporting_period is not None:
            observed["reporting_period"] = dict(record.reporting_period)
        _require(observed == row, "source registration was normalized or metadata changed")
    return records, raw["sources"], sha256(body).hexdigest()


def _manifest(manifest, rows) -> dict[str, dict]:
    _keys(manifest, {"schema_version", "chunks_sha256", "chunking", "documents"} | _COUNTS,
          optional={"extraction_mode"}, label="corpus manifest")
    _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1,
             "unsupported corpus schema")
    _digest(manifest["chunks_sha256"], "manifest chunks_sha256")
    _fixed(manifest["chunking"], _CHUNKING, "manifest chunking")
    if "extraction_mode" in manifest:
        _require(manifest["extraction_mode"] == "plain", "unsupported extraction mode")
    for key in _COUNTS:
        _integer(manifest[key], key)
    documents = manifest["documents"]
    _require(type(documents) is list and len(documents) == len(rows), "document count mismatch")
    by_id = {}
    for doc, row in zip(documents, rows, strict=True):
        _keys(doc, {"document_id", "document_label", "source_url", "source_filename", "source_sha256",
                    "page_count", "nonempty_page_count", "chunk_count"},
              optional={"reporting_period"}, label="manifest document")
        association = {"document_id": "document_id", "document_label": "label", "source_url": "url",
                       "source_filename": "filename", "source_sha256": "sha256"}
        for key, source_key in association.items():
            _require(type(doc[key]) is str and doc[key] == row[source_key], "document registry metadata/order mismatch")
        _require(("reporting_period" in doc) == ("reporting_period" in row), "reporting period association mismatch")
        if "reporting_period" in doc:
            _period(doc["reporting_period"])
            _require(doc["reporting_period"] == row["reporting_period"], "reporting period metadata mismatch")
        for key in ("page_count", "nonempty_page_count", "chunk_count"):
            _integer(doc[key], key)
        _require(doc["nonempty_page_count"] <= doc["page_count"], "nonempty page count exceeds physical pages")
        by_id[doc["document_id"]] = doc
    _require(manifest["document_count"] == len(rows), "document count mismatch")
    for key in ("page_count", "nonempty_page_count", "chunk_count"):
        _require(manifest[key] == sum(doc[key] for doc in documents), f"manifest {key} total mismatch")
    return by_id


def _receipt(receipt, manifest, rows, registry_hash, manifest_hash, chunks_hash):
    _keys(receipt, {"schema_version", "status", "scope", "profile", "source_registry_sha256",
                    "manifest_sha256", "chunks_sha256", "runtime", "module_sha256", "sources",
                    "observed_counts", "human_acceptance"}, label="build receipt")
    _require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1,
             "unsupported build receipt schema")
    _require(receipt["status"] == "complete" and receipt["scope"] == "operator-local-corpus-build",
             "corpus build is not complete")
    _require(receipt["human_acceptance"] is False, "build receipt is not human acceptance")
    _fixed(receipt["profile"], _PROFILE, "receipt profile")
    for key, actual in (("source_registry_sha256", registry_hash), ("manifest_sha256", manifest_hash),
                        ("chunks_sha256", chunks_hash)):
        _digest(receipt[key], key)
        _require(receipt[key] == actual, f"{key} hash mismatch")
    _keys(receipt["runtime"], {"python", *_PACKAGES}, label="build runtime")
    for value in receipt["runtime"].values():
        _string(value, "runtime version", canonical=True)
    _keys(receipt["module_sha256"], _MODULES, label="module identities")
    for value in receipt["module_sha256"].values():
        _digest(value, "module sha256")
    _keys(receipt["observed_counts"], _COUNTS, label="observed counts")
    for key in _COUNTS:
        _integer(receipt["observed_counts"][key], key)
        _require(receipt["observed_counts"][key] == manifest[key], f"receipt {key} mismatch")
    _require(type(receipt["sources"]) is list and len(receipt["sources"]) == len(rows), "receipt sources mismatch")
    for source, row in zip(receipt["sources"], rows, strict=True):
        _keys(source, {"document_id", "source_sha256", "source_bytes"}, label="source receipt")
        _require(source["document_id"] == row["document_id"] and source["source_sha256"] == row["sha256"],
                 "source receipt identity/order mismatch")
        _integer(source["source_bytes"], "source_bytes")


def _validate_records(body: bytes, by_id):
    result = []
    ids = set()
    counts = Counter()
    pages = {key: set() for key in by_id}
    order = {key: index for index, key in enumerate(by_id)}
    last = (-1, 0)
    # JSONL uses LF delimiters; U+0085/U+2028/U+2029 remain legal string text.
    for line in body.decode("utf-8").split("\n"):
        if not line.strip():
            continue
        row = _json(line)
        _keys(row, {"chunk_id", "document_id", "page_number", "text"}, label="chunk record")
        for key in ("chunk_id", "document_id", "text"):
            _string(row[key], key)
        _integer(row["page_number"], "page_number")
        doc_id, page = row["document_id"], row["page_number"]
        _require(doc_id in by_id, "chunk document is not registered")
        _require(page <= by_id[doc_id]["page_count"], "chunk page exceeds document page bound")
        prefix = f"{doc_id}:p{page}:w:"
        _require(row["chunk_id"].startswith(prefix)
                 and re.fullmatch(r"[0-9a-f]{16}", row["chunk_id"][len(prefix):]) is not None,
                 "chunk ID/document/page association mismatch")
        _require(row["chunk_id"] not in ids, "duplicate chunk ID")
        current = (order[doc_id], page)
        _require(current >= last, "chunk document/page order mismatch")
        last = current
        ids.add(row["chunk_id"])
        counts[doc_id] += 1
        pages[doc_id].add(page)
        result.append(Chunk(**row))
    _require(bool(result), "chunk artifact is empty")
    for doc_id, doc in by_id.items():
        _require(counts[doc_id] == doc["chunk_count"], "actual document chunk_count mismatch")
        _require(len(pages[doc_id]) == doc["nonempty_page_count"], "actual represented page count mismatch")
    return tuple(result)


def _verified(manifest_body, chunks_body, receipt, sources, rows, registry_hash):
    manifest = _json(manifest_body)
    by_id = _manifest(manifest, rows)
    chunks_hash = sha256(chunks_body).hexdigest()
    _require(manifest["chunks_sha256"] == chunks_hash, "manifest chunks hash mismatch")
    _receipt(receipt, manifest, rows, registry_hash, sha256(manifest_body).hexdigest(), chunks_hash)
    chunks = _validate_records(chunks_body, by_id)
    _require(len(chunks) == manifest["chunk_count"], "actual corpus chunk_count mismatch")
    return VerifiedCorpus(chunks, manifest, sources, chunks_hash)


def _artifact_file(directory, filename):
    path = directory / filename
    _require(not path.is_symlink() and path.resolve(strict=True).parent == directory,
             "artifact file escapes its directory")
    _require(path.is_file(), "artifact entry must be a local file")
    return path


def load_verified_corpus(*, artifact_dir: Path, source_registry: Path) -> VerifiedCorpus:
    """Verify strict local schema, metadata, byte identities and actual memberships."""
    directory = _absolute(artifact_dir, "artifact_dir", exists=True)
    registry = _absolute(source_registry, "source_registry", exists=True)
    _require(not os.path.lexists(directory / "failure.json"), "corpus has a failed build marker")
    _require(not os.path.lexists(directory / "pending.json"), "corpus build is still pending")
    sources, rows, registry_hash = _registry(registry)
    manifest = _artifact_file(directory, "manifest.json")
    chunks = _artifact_file(directory, "chunks.jsonl")
    receipt = _artifact_file(directory, "build-receipt.json")
    manifest_body, chunks_body, receipt_body = manifest.read_bytes(), chunks.read_bytes(), receipt.read_bytes()
    before = {registry: registry_hash, manifest: sha256(manifest_body).hexdigest(),
              chunks: sha256(chunks_body).hexdigest(), receipt: sha256(receipt_body).hexdigest()}
    verified = _verified(manifest_body, chunks_body, _json(receipt_body),
                         sources, rows, registry_hash)
    # Reuse the legacy converter only after validation; no legacy caller changes.
    converted = tuple(load_chunks(chunks))
    _require(converted == verified.chunks, "chunks changed while loading")
    for path, expected in before.items():
        if path != registry:
            _artifact_file(directory, path.name)
        _require(file_sha256(path) == expected, "corpus closure changed while loading")
    _require(not os.path.lexists(directory / "failure.json"), "corpus changed to failed state while loading")
    _require(not os.path.lexists(directory / "pending.json"), "corpus changed to pending state while loading")
    return verified


def _write_json_exclusive(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _prepared_encoding(body: bytes):
    """Construct the installed cl100k config from owned verified BPE bytes only."""
    _require(sha256(body).hexdigest() == _TOKENIZER_SHA256, "tokenizer bytes hash mismatch")
    ranks = {}
    seen_ranks = set()
    for line in body.split(b"\n"):
        if not line.strip():
            continue
        fields = line.split()
        _require(len(fields) == 2 and re.fullmatch(rb"[0-9]+", fields[1]) is not None,
                 "malformed prepared BPE rank")
        try:
            token = base64.b64decode(fields[0], validate=True)
        except ValueError as error:
            raise ValueError("malformed prepared BPE token") from error
        rank = int(fields[1])
        _require(token and token not in ranks and rank not in seen_ranks, "duplicate or empty prepared BPE entry")
        ranks[token] = rank
        seen_ranks.add(rank)
    _require(bool(ranks), "prepared BPE table is empty")
    import tiktoken
    from tiktoken_ext.openai_public import cl100k_base
    _require(type(cl100k_base) is FunctionType and cl100k_base.__closure__ is None,
             "unsupported installed cl100k config factory")
    calls = 0
    def owned_ranks(url, expected_hash=None):
        nonlocal calls
        _require(url == _TOKENIZER_URL and
                 expected_hash == "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
                 "installed tokenizer factory requested unexpected BPE input")
        calls += 1
        return ranks
    config = FunctionType(cl100k_base.__code__, dict(cl100k_base.__globals__, load_tiktoken_bpe=owned_ranks),
                          cl100k_base.__name__, cl100k_base.__defaults__)()
    _require(calls == 1, "installed tokenizer factory did not use exactly one owned BPE input")
    _keys(config, {"name", "pat_str", "mergeable_ranks", "special_tokens"}, label="cl100k constructor config")
    _require(config["name"] == "cl100k_base" and config["mergeable_ranks"] is ranks,
             "installed tokenizer factory returned unexpected encoding identity")
    _string(config["pat_str"], "cl100k pattern")
    _require(type(config["special_tokens"]) is dict, "cl100k special tokens must be an object")
    for name, value in config["special_tokens"].items():
        _string(name, "special token")
        _integer(value, "special token ID", minimum=0)
    return tiktoken.Encoding(**config)


def build_local_corpus(*, source_registry: Path, source_dir: Path,
                       output_dir: Path, tokenizer_cache_dir: Path) -> dict[str, object]:
    """Build into a new owned directory from pinned local PDFs and prepared encoding.

The encoding is passed from already verified in-memory bytes; no downstream
cache lookup, fetch fallback, or caller environment mutation occurs.
"""
    output = _absolute(output_dir, "output_dir")
    _require(not os.path.lexists(output_dir), "output directory already exists")
    registry = _absolute(source_registry, "source_registry", exists=True)
    raw = _absolute(source_dir, "source_dir", exists=True)
    cache = _absolute(tokenizer_cache_dir, "tokenizer_cache_dir", exists=True)
    _require(raw.is_dir() and cache.is_dir(), "source and tokenizer cache must be directories")
    _require(not (output.is_relative_to(raw) or raw.is_relative_to(output)
                 or registry.is_relative_to(output) or cache.is_relative_to(output)),
             "output overlaps a read-only input")
    sources, rows, registry_hash = _registry(registry)
    encoding = cache / _TOKENIZER_FILENAME
    _require(encoding.is_file() and encoding.resolve(strict=True).parent == cache,
             "prepared tokenizer cache file is absent or escapes its directory")
    encoding_body = encoding.read_bytes()
    _require(sha256(encoding_body).hexdigest() == _TOKENIZER_SHA256, "tokenizer cache hash mismatch")
    prepared = _prepared_encoding(encoding_body)
    source_receipts = []
    for record in sources:
        pdf = (raw / record.filename).resolve(strict=True)
        _require(pdf.parent == raw and pdf.is_file(), "source PDF escapes source directory")
        observed = file_sha256(pdf)
        _require(observed == record.sha256, "source PDF hash mismatch")
        source_receipts.append(dict(document_id=record.document_id, source_sha256=observed,
                                    source_bytes=pdf.stat().st_size))
    output.mkdir(parents=True, exist_ok=False)
    try:
        # This owned marker precedes any builder output and survives every error.
        # Completion bytes alone are insufficient if closing/diagnostics fail.
        pending = output / "pending.json"
        _write_json_exclusive(pending, {"schema_version": 1, "status": "pending",
                                       "scope": "operator-local-corpus-build"})
        build_corpus_artifacts(sources=sources, source_dir=raw, output_dir=output,
                               strategy="word_safe", max_tokens=400, overlap_tokens=0,
                               encoding_name="cl100k_base", extraction_mode="plain", encoding=prepared)
        manifest_body = _artifact_file(output, "manifest.json").read_bytes()
        chunks_body = _artifact_file(output, "chunks.jsonl").read_bytes()
        manifest = _json(manifest_body)
        _require(encoding.resolve(strict=True).parent == cache and encoding.is_file(),
                 "tokenizer cache escapes directory after build")
        _require(file_sha256(registry) == registry_hash and file_sha256(encoding) == _TOKENIZER_SHA256,
                 "prepared input changed during build")
        for record in sources:
            path = (raw / record.filename).resolve(strict=True)
            _require(path.parent == raw and path.is_file(), "source PDF escapes directory after build")
            _require(file_sha256(path) == record.sha256, "source PDF changed during build")
        module_dir = Path(__file__).resolve().parent
        receipt = {"schema_version": 1, "status": "complete", "scope": "operator-local-corpus-build",
                   "profile": dict(_PROFILE), "source_registry_sha256": registry_hash,
                   "manifest_sha256": sha256(manifest_body).hexdigest(),
                   "chunks_sha256": sha256(chunks_body).hexdigest(),
                   "runtime": {"python": platform.python_version(),
                               **{name: metadata.version(name) for name in _PACKAGES}},
                   "module_sha256": {name: file_sha256(module_dir / name) for name in _MODULES},
                   "sources": source_receipts, "observed_counts": {key: manifest[key] for key in _COUNTS},
                   "human_acceptance": False}
        _verified(manifest_body, chunks_body, receipt, sources, rows, registry_hash)
        _require(not os.path.lexists(output / "failure.json"), "builder left a failed build marker")
        _write_json_exclusive(output / "build-receipt.json", receipt)
        # Remove only our marker, after the completed receipt writer closed.
        # An unlink failure leaves the marker and loader refuses the directory.
        pending.unlink()
        return receipt
    except Exception as error:
        kind = type(error).__name__
        if kind not in {"ValueError", "OSError", "RuntimeError", "FileNotFoundError", "PermissionError"}:
            kind = "BuildError"
        try:
            _write_json_exclusive(output / "failure.json", {"schema_version": 1, "status": "failed", "error_type": kind})
        except OSError:
            pass
        raise


def _scope(document_ids):
    if document_ids is None:
        return
    _require(type(document_ids) is frozenset and bool(document_ids), "document scope must be a nonempty frozenset")
    for doc_id in document_ids:
        _string(doc_id, "document scope ID", canonical=True)


class DocumentScopeRetriever:
    """Forward an explicit immutable document scope without global fallback."""
    def __init__(self, retriever, *, document_ids: frozenset[str] | None):
        _scope(document_ids)
        self._retriever = retriever
        self._document_ids = document_ids

    def retrieve(self, query: str, *, top_k: int):
        _integer(top_k, "top_k")
        return self._retriever.retrieve_from_documents(query, top_k=top_k,
            document_ids=None if self._document_ids is None else set(self._document_ids))


def build_corpus_service(corpus: VerifiedCorpus, *, threshold: float, top_k: int,
                         document_ids: frozenset[str] | None) -> EvidenceService:
    """Compose the verified local corpus into an explicitly scoped lexical service.

A zero threshold is an ungated inspection, including zero-score candidates.
It establishes no financial field support or calibrated answerability.
"""
    _require(type(corpus) is VerifiedCorpus, "service requires a verified corpus")
    _require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 <= threshold <= 1,
             "threshold must be a finite number from 0 through 1")
    _integer(top_k, "top_k")
    _scope(document_ids)
    available = {source.document_id for source in corpus.sources}
    _require(document_ids is None or document_ids <= available, "document scope is not registered")
    from .retrieval import TfidfRetriever
    retriever = DocumentScopeRetriever(TfidfRetriever(corpus.chunks), document_ids=document_ids)
    return EvidenceService(retriever=retriever, threshold=threshold, top_k=top_k)
