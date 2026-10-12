"""Local-artifact boundaries. Generated PDFs/encoding doubles are not financial acceptance."""
from __future__ import annotations

from copy import deepcopy
import base64
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path

import pymupdf
import pytest


def library():
    try:
        return importlib.import_module("nvidia_rag.corpus")
    except ModuleNotFoundError:
        pytest.fail("installed corpus build/validation interface is missing")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def source_record(document_id="report"):
    return {"document_id": document_id, "label": "Generated boundary fixture",
            "filename": document_id + ".pdf", "sha256": "0" * 64,
            "url": "https://s201.q4cdn.com/141608511/files/test/" + document_id + ".pdf"}


class Artifact:
    """Independent literal fixture, refreshed after mutations to exercise semantic checks."""
    def __init__(self, tmp_path, two=False):
        self.directory = tmp_path / "corpus"
        self.directory.mkdir()
        self.registry = tmp_path / "sources.json"
        self.sources = [source_record()]
        if two:
            self.sources.append(source_record("other"))
        self.records = [{"chunk_id": f"{r['document_id']}:p1:w:0123456789abcdef",
                         "document_id": r["document_id"], "page_number": 1,
                         "text": "Generated Revenue evidence"} for r in self.sources]
        self.manifest = {"schema_version": 1, "chunks_sha256": "0" * 64,
                         "document_count": len(self.sources), "page_count": 2 * len(self.sources),
                         "nonempty_page_count": len(self.sources), "chunk_count": len(self.sources),
                         "chunking": {"encoding_name": "cl100k_base", "max_tokens": 400,
                                      "overlap_tokens": 0, "strategy": "word_safe"},
                         "documents": [dict(document_id=r["document_id"], document_label=r["label"],
                                            source_url=r["url"], source_filename=r["filename"],
                                            source_sha256=r["sha256"], page_count=2,
                                            nonempty_page_count=1, chunk_count=1) for r in self.sources]}
        self.receipt = {"schema_version": 1, "status": "complete", "scope": "operator-local-corpus-build",
                        "profile": {"name": "plain-word-safe-400-0-v1", "extraction_mode": "plain",
                                    "strategy": "word_safe", "max_tokens": 400, "overlap_tokens": 0,
                                    "encoding_name": "cl100k_base"},
                        "source_registry_sha256": "0" * 64, "manifest_sha256": "0" * 64,
                        "chunks_sha256": "0" * 64, "human_acceptance": False,
                        "runtime": {"python": "3.13", "PyMuPDF": "1.28.2", "numpy": "2.5.4",
                                    "scikit-learn": "1.9.1", "tiktoken": "0.14.0"},
                        "module_sha256": {name: "0" * 64 for name in
                                          ("core.py", "ingest.py", "sources.py", "corpus.py")},
                        "sources": [dict(document_id=r["document_id"], source_sha256=r["sha256"],
                                         source_bytes=1) for r in self.sources],
                        "observed_counts": {}}
        self.sync()

    def sync(self):
        write_json(self.registry, {"schema_version": 1, "sources": self.sources})
        chunks = self.directory / "chunks.jsonl"
        chunks.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in self.records), encoding="utf-8")
        self.manifest["chunks_sha256"] = digest(chunks)
        write_json(self.directory / "manifest.json", self.manifest)
        self.receipt["source_registry_sha256"] = digest(self.registry)
        self.receipt["manifest_sha256"] = digest(self.directory / "manifest.json")
        self.receipt["chunks_sha256"] = digest(chunks)
        self.receipt["observed_counts"] = {k: self.manifest[k] for k in
                                           ("document_count", "page_count", "nonempty_page_count", "chunk_count")}
        write_json(self.directory / "build-receipt.json", self.receipt)

    def load(self):
        return library().load_verified_corpus(artifact_dir=self.directory, source_registry=self.registry)


def test_literal_artifact_load_preserves_order(tmp_path):
    a = Artifact(tmp_path, two=True)
    corpus = a.load()
    assert [c.document_id for c in corpus.chunks] == ["report", "other"]
    assert [s.document_id for s in corpus.sources] == ["report", "other"]
    assert corpus.chunks_sha256 == digest(a.directory / "chunks.jsonl")


@pytest.mark.parametrize("page", [True, "1", 1.0, 0, -1, 3])
def test_invalid_page_refused_even_with_matching_hashes(tmp_path, page):
    a = Artifact(tmp_path)
    a.records[0]["page_number"] = page
    a.sync()
    with pytest.raises(ValueError, match="page"):
        a.load()


@pytest.mark.parametrize("field,value", [
    ("chunk_id", ""), ("chunk_id", 12), ("document_id", None),
    ("document_id", "other"), ("chunk_id", "other:p1:w:0123456789abcdef"),
    ("chunk_id", "report:p2:w:0123456789abcdef"), ("text", "  "), ("text", 12),
])
def test_invalid_chunk_identity_refused(tmp_path, field, value):
    a = Artifact(tmp_path)
    a.records[0][field] = value
    a.sync()
    with pytest.raises(ValueError):
        a.load()


def test_duplicate_ids_refused(tmp_path):
    a = Artifact(tmp_path)
    a.records.append(deepcopy(a.records[0]))
    a.manifest["chunk_count"] = a.manifest["documents"][0]["chunk_count"] = 2
    a.sync()
    with pytest.raises(ValueError, match="duplicate"):
        a.load()


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("document_count", True),
    ("page_count", "2"), ("nonempty_page_count", 1.0), ("chunk_count", 2),
])
def test_manifest_schema_types_and_totals_refused(tmp_path, field, value):
    a = Artifact(tmp_path)
    a.manifest[field] = value
    a.sync()
    with pytest.raises(ValueError):
        a.load()


@pytest.mark.parametrize("field,value", [
    ("document_label", "different"), ("source_url", "https://example.com/report.pdf"),
    ("source_filename", "different.pdf"), ("source_sha256", "1" * 64),
    ("page_count", False), ("nonempty_page_count", 2), ("chunk_count", 2),
])
def test_manifest_document_metadata_and_actual_counts_refused(tmp_path, field, value):
    a = Artifact(tmp_path)
    a.manifest["documents"][0][field] = value
    if field in {"page_count", "nonempty_page_count", "chunk_count"}:
        a.manifest[field] = value
    a.sync()
    with pytest.raises(ValueError):
        a.load()


def test_document_and_jsonl_order_refused(tmp_path):
    a = Artifact(tmp_path, two=True)
    a.records.reverse()
    a.sync()
    with pytest.raises(ValueError, match="order"):
        a.load()
    a.records.reverse()
    a.manifest["documents"].reverse()
    a.sync()
    with pytest.raises(ValueError):
        a.load()


def test_pages_must_be_ordered_within_document(tmp_path):
    a = Artifact(tmp_path)
    a.records = [dict(a.records[0], chunk_id="report:p2:w:0123456789abcdef", page_number=2),
                 dict(a.records[0], chunk_id="report:p1:w:fedcba9876543210", page_number=1)]
    a.manifest.update(chunk_count=2, nonempty_page_count=2)
    a.manifest["documents"][0].update(chunk_count=2, nonempty_page_count=2)
    a.sync()
    with pytest.raises(ValueError, match="order"):
        a.load()


@pytest.mark.parametrize("section,key,value", [
    ("chunking", "max_tokens", True), ("chunking", "strategy", "fixed_token"),
    ("chunking", "overlap_tokens", 50), ("profile", "name", "other"),
    ("profile", "overlap_tokens", False), ("receipt", "human_acceptance", 0),
    ("receipt", "schema_version", True), ("receipt", "status", "failed"),
])
def test_fixed_profile_and_receipt_are_typed(tmp_path, section, key, value):
    a = Artifact(tmp_path)
    target = a.manifest["chunking"] if section == "chunking" else a.receipt["profile"] if section == "profile" else a.receipt
    target[key] = value
    a.sync()
    with pytest.raises(ValueError):
        a.load()


@pytest.mark.parametrize("target", ["receipt", "manifest", "record", "registry"])
def test_unknown_fields_refused(tmp_path, target):
    a = Artifact(tmp_path)
    obj = {"receipt": a.receipt, "manifest": a.manifest, "record": a.records[0], "registry": a.sources[0]}[target]
    obj["permission_status"] = "reviewed"
    a.sync()
    with pytest.raises(ValueError):
        a.load()


@pytest.mark.parametrize("field,value", [("document_id", 42), ("label", "  "),
                                         ("sha256", "A" * 64), ("filename", "../report.pdf")])
def test_registry_coercion_and_unsafe_names_refused(tmp_path, field, value):
    a = Artifact(tmp_path)
    a.sources[0][field] = value
    a.sync()
    with pytest.raises(ValueError):
        a.load()


def test_reporting_period_metadata_must_match(tmp_path):
    a = Artifact(tmp_path)
    period = {"fiscal_year": 2025, "quarter": 2, "end_date": "2024-07-28",
              "source_locator": "Generated location", "reviewed_by": "Development fixture"}
    a.sources[0]["reporting_period"] = period
    a.manifest["documents"][0]["reporting_period"] = dict(period)
    a.sync()
    assert a.load().sources[0].reporting_period["quarter"] == 2
    a.manifest["documents"][0]["reporting_period"]["quarter"] = 3
    a.sync()
    with pytest.raises(ValueError):
        a.load()


@pytest.mark.parametrize("filename", ["manifest.json", "chunks.jsonl", "build-receipt.json"])
def test_stale_hash_or_absent_completion_refused(tmp_path, filename):
    a = Artifact(tmp_path)
    p = a.directory / filename
    if filename == "build-receipt.json":
        p.unlink()
    else:
        p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises((ValueError, OSError)):
        a.load()


@pytest.mark.parametrize("raw", ['{"schema_version":1,"schema_version":1,"sources":[]}',
                                  '{"schema_version":NaN,"sources":[]}', '[]'])
def test_registry_duplicate_members_and_nonfinite_refused(tmp_path, raw):
    a = Artifact(tmp_path)
    a.registry.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        a.load()


def test_loader_refuses_relative_paths(tmp_path, monkeypatch):
    a = Artifact(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        library().load_verified_corpus(artifact_dir=Path("corpus"), source_registry=a.registry)


def test_existing_output_preserved_before_input_io(tmp_path):
    module = library()
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "retained"
    marker.write_text("retain me", encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError), match="exist"):
        module.build_local_corpus(source_registry=tmp_path / "absent.json", source_dir=tmp_path / "missing",
                                  output_dir=out, tokenizer_cache_dir=tmp_path / "cache")
    assert marker.read_text(encoding="utf-8") == "retain me"


@pytest.fixture
def build_inputs(tmp_path, monkeypatch):
    module = library()
    raw = tmp_path / "raw"
    raw.mkdir()
    pdf = raw / "report.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Generated Revenue evidence")
        document.new_page()
        document.save(pdf)
    row = source_record()
    row["sha256"] = digest(pdf)
    registry = tmp_path / "registry.json"
    write_json(registry, {"schema_version": 1, "sources": [row]})
    cache = tmp_path / "encoding-cache"
    cache.mkdir()
    body = b"".join(base64.b64encode(bytes([value])) + b" " + str(value).encode() + b"\n"
                    for value in range(256))
    (cache / "9b5ad71b2ce5302211f9c61530b329a4922fc6a4").write_bytes(body)
    monkeypatch.setattr(module, "_TOKENIZER_SHA256", sha256(body).hexdigest())
    return dict(source_registry=registry, source_dir=raw, output_dir=tmp_path / "new-corpus",
                tokenizer_cache_dir=cache)


def test_actual_generated_pdf_build_preserves_input_and_environment(build_inputs, monkeypatch):
    before = {p.name: digest(p) for p in build_inputs["source_dir"].iterdir()}
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "retained-process-value")
    receipt = library().build_local_corpus(**build_inputs)
    assert receipt["status"] == "complete"
    corpus = library().load_verified_corpus(artifact_dir=build_inputs["output_dir"],
                                            source_registry=build_inputs["source_registry"])
    assert corpus.manifest["page_count"] == 2
    assert corpus.manifest["nonempty_page_count"] == corpus.manifest["chunk_count"] == 1
    assert corpus.chunks[0].text == "Generated Revenue evidence"
    assert before == {p.name: digest(p) for p in build_inputs["source_dir"].iterdir()}
    assert os.environ["TIKTOKEN_CACHE_DIR"] == "retained-process-value"
    assert not any(str(build_inputs["source_dir"]) in str(value) for value in receipt.values())


@pytest.mark.parametrize("problem", ["missing-pdf", "hash", "missing-cache", "corrupt-cache", "overlap"])
def test_build_preflight_failure_does_not_create_output(build_inputs, problem):
    if problem == "missing-pdf":
        (build_inputs["source_dir"] / "report.pdf").unlink()
    elif problem == "hash":
        (build_inputs["source_dir"] / "report.pdf").write_bytes(b"different bytes")
    elif problem in {"missing-cache", "corrupt-cache"}:
        cached = next(build_inputs["tokenizer_cache_dir"].iterdir())
        cached.unlink() if problem == "missing-cache" else cached.write_bytes(b"wrong")
    else:
        build_inputs["output_dir"] = build_inputs["source_dir"] / "corpus"
    with pytest.raises((ValueError, OSError)):
        library().build_local_corpus(**build_inputs)
    assert not build_inputs["output_dir"].exists()


def test_builder_exception_has_bounded_failure_not_completion(build_inputs, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("private path and source contents must not enter receipt")
    monkeypatch.setattr("nvidia_rag.corpus.build_corpus_artifacts", fail)
    with pytest.raises(RuntimeError):
        library().build_local_corpus(**build_inputs)
    out = build_inputs["output_dir"]
    assert not (out / "build-receipt.json").exists()
    failure = json.loads((out / "failure.json").read_text("utf-8"))
    assert failure == {"schema_version": 1, "status": "failed", "error_type": "RuntimeError"}


def test_source_escape_symlink_refused(build_inputs, tmp_path):
    pdf = build_inputs["source_dir"] / "report.pdf"
    external = tmp_path / "external.pdf"
    pdf.replace(external)
    try:
        pdf.symlink_to(external)
    except OSError:
        pytest.skip("host does not grant symlink creation")
    with pytest.raises(ValueError):
        library().build_local_corpus(**build_inputs)
    assert not build_inputs["output_dir"].exists()


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_unicode_text_separators_are_not_jsonl_record_delimiters(tmp_path, separator):
    a = Artifact(tmp_path)
    a.records[0]["text"] = "Revenue" + separator + "evidence"
    a.sync()
    p = a.directory / "chunks.jsonl"
    p.write_text(json.dumps(a.records[0], ensure_ascii=False) + "\n", encoding="utf-8")
    a.manifest["chunks_sha256"] = digest(p)
    write_json(a.directory / "manifest.json", a.manifest)
    a.receipt["chunks_sha256"] = digest(p)
    a.receipt["manifest_sha256"] = digest(a.directory / "manifest.json")
    write_json(a.directory / "build-receipt.json", a.receipt)
    assert a.load().chunks[0].text == "Revenue" + separator + "evidence"


@pytest.mark.parametrize("filename", ["registry", "manifest.json", "build-receipt.json"])
def test_changed_closure_is_refused_before_return(tmp_path, monkeypatch, filename):
    a = Artifact(tmp_path)
    module = library()
    original = module.load_chunks
    path = a.registry if filename == "registry" else a.directory / filename
    def altered(chunks):
        result = original(chunks)
        path.write_bytes(path.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(module, "load_chunks", altered)
    with pytest.raises(ValueError, match="changed"):
        a.load()


@pytest.mark.parametrize("field", ["document_id", "label", "url", "filename"])
def test_registry_surrounding_whitespace_never_normalized(tmp_path, field):
    a = Artifact(tmp_path)
    a.sources[0][field] = " " + a.sources[0][field] + " "
    a.sync()
    with pytest.raises(ValueError):
        a.load()


@pytest.mark.parametrize("change", ["null", "unknown", "bool", "whitespace"])
def test_reporting_period_raw_schema_is_not_normalized(tmp_path, change):
    a = Artifact(tmp_path)
    period = {"fiscal_year": 2025, "quarter": 2, "end_date": "2024-07-28",
              "source_locator": "fixture", "reviewed_by": "fixture"}
    if change == "null":
        period = None
    elif change == "unknown":
        period["permission"] = "reviewed"
    elif change == "bool":
        period["quarter"] = True
    else:
        period["reviewed_by"] = " fixture "
    a.sources[0]["reporting_period"] = period
    a.sync()
    with pytest.raises(ValueError):
        a.load()


def test_partial_builder_failure_retains_incomplete_artifact(build_inputs, monkeypatch):
    module = library()
    original = module.build_corpus_artifacts
    def partial(**kwargs):
        original(**kwargs)
        raise RuntimeError("failed after manifest/chunks")
    monkeypatch.setattr(module, "build_corpus_artifacts", partial)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "retained-cache")
    with pytest.raises(RuntimeError):
        module.build_local_corpus(**build_inputs)
    assert (build_inputs["output_dir"] / "chunks.jsonl").is_file()
    assert not (build_inputs["output_dir"] / "build-receipt.json").exists()
    with pytest.raises(ValueError, match="failed"):
        module.load_verified_corpus(artifact_dir=build_inputs["output_dir"],
                                     source_registry=build_inputs["source_registry"])
    assert os.environ["TIKTOKEN_CACHE_DIR"] == "retained-cache"


def test_failure_marker_refuses_a_fully_written_completion(tmp_path):
    a = Artifact(tmp_path)
    write_json(a.directory / "failure.json", {"status": "failed"})
    with pytest.raises(ValueError, match="failed"):
        a.load()


def test_completion_writer_failure_cannot_be_loaded(build_inputs, monkeypatch):
    module = library()
    original = module._write_json_exclusive
    def failed_close(path, value):
        original(path, value)
        if path.name == "build-receipt.json":
            raise OSError("write/close failed after body")
    monkeypatch.setattr(module, "_write_json_exclusive", failed_close)
    with pytest.raises(OSError):
        module.build_local_corpus(**build_inputs)
    assert (build_inputs["output_dir"] / "failure.json").is_file()
    with pytest.raises(ValueError, match="failed"):
        module.load_verified_corpus(artifact_dir=build_inputs["output_dir"],
                                     source_registry=build_inputs["source_registry"])


@pytest.mark.parametrize("entry", ["source", "encoding"])
def test_post_build_source_and_cache_escape_refused(build_inputs, tmp_path, monkeypatch, entry):
    module = library()
    original = module.build_corpus_artifacts
    path = (build_inputs["source_dir"] / "report.pdf" if entry == "source" else
            next(build_inputs["tokenizer_cache_dir"].iterdir()))
    outside = tmp_path / ("outside-" + path.name)
    outside.write_bytes(path.read_bytes())
    probe = tmp_path / "probe-link"
    try:
        probe.symlink_to(outside)
    except OSError:
        pytest.skip("host does not grant symlink creation")
    probe.unlink()
    def replaced(**kwargs):
        result = original(**kwargs)
        path.unlink()
        path.symlink_to(outside)
        return result
    monkeypatch.setattr(module, "build_corpus_artifacts", replaced)
    with pytest.raises(ValueError, match="escape"):
        module.build_local_corpus(**build_inputs)
    assert not (build_inputs["output_dir"] / "build-receipt.json").exists()


@pytest.mark.parametrize("filename", ["manifest.json", "chunks.jsonl"])
def test_post_build_output_symlink_is_not_completed(build_inputs, tmp_path, monkeypatch, filename):
    module = library()
    original = module.build_corpus_artifacts
    outside = tmp_path / ("external-" + filename)
    probe = tmp_path / "probe-link"
    try:
        probe.symlink_to(outside)
    except OSError:
        pytest.skip("host does not grant symlink creation")
    probe.unlink()
    def replaced(**kwargs):
        result = original(**kwargs)
        path = build_inputs["output_dir"] / filename
        path.replace(outside)
        path.symlink_to(outside)
        return result
    monkeypatch.setattr(module, "build_corpus_artifacts", replaced)
    with pytest.raises(ValueError, match="escape"):
        module.build_local_corpus(**build_inputs)
    assert not (build_inputs["output_dir"] / "build-receipt.json").exists()


@pytest.mark.parametrize("entry", ["file", "empty-dir", "failed-dir", "dangling-symlink"])
def test_existing_output_kinds_are_never_reused(tmp_path, entry):
    out = tmp_path / "retained-output"
    if entry == "file":
        out.write_bytes(b"retained")
    elif entry == "dangling-symlink":
        try:
            out.symlink_to(tmp_path / "missing-target", target_is_directory=True)
        except OSError:
            pytest.skip("host does not grant symlink creation")
    else:
        out.mkdir()
        if entry == "failed-dir":
            (out / "failure.json").write_bytes(b"retained")
    with pytest.raises(ValueError, match="exist"):
        library().build_local_corpus(source_registry=tmp_path / "absent.json", source_dir=tmp_path / "absent",
                                    output_dir=out, tokenizer_cache_dir=tmp_path / "absent-cache")
    assert os.path.lexists(out)


def test_prepared_encoding_bypasses_all_upstream_cache_and_network(build_inputs, monkeypatch):
    import tiktoken
    import tiktoken.load
    import requests
    from tiktoken_ext import openai_public
    original_loader = openai_public.load_tiktoken_bpe
    def no_upstream(*args, **kwargs):
        pytest.fail("prepared corpus reached an upstream cache/download path")
    monkeypatch.setattr(tiktoken, "get_encoding", no_upstream)
    monkeypatch.setattr(tiktoken.load, "read_file", no_upstream)
    monkeypatch.setattr(requests, "get", no_upstream)
    receipt = library().build_local_corpus(**build_inputs)
    assert receipt["status"] == "complete"
    assert openai_public.load_tiktoken_bpe is original_loader


def test_deleted_cache_after_preparation_refuses_without_fetch(build_inputs, monkeypatch):
    import tiktoken.load
    import requests
    module = library()
    original = module.build_corpus_artifacts
    def no_fetch(*args, **kwargs):
        pytest.fail("deleted cache triggered an upstream fetch")
    monkeypatch.setattr(tiktoken.load, "read_file", no_fetch)
    monkeypatch.setattr(requests, "get", no_fetch)
    def deleted(**kwargs):
        next(build_inputs["tokenizer_cache_dir"].iterdir()).unlink()
        return original(**kwargs)
    monkeypatch.setattr(module, "build_corpus_artifacts", deleted)
    with pytest.raises((ValueError, OSError)):
        module.build_local_corpus(**build_inputs)
    assert not (build_inputs["output_dir"] / "build-receipt.json").exists()


def test_word_safe_encoding_dependency_preserves_legacy_default(monkeypatch):
    from nvidia_rag import core
    page = core.Chunk("report:page:1", "report", 1, "one two three")
    class Owned:
        name = "cl100k_base"
        def encode(self, text):
            return text.split()
    observed = []
    def get(name):
        observed.append(name)
        return Owned()
    monkeypatch.setattr(core.tiktoken, "get_encoding", get)
    legacy = core.chunk_pages_word_safe([page], max_tokens=2)
    try:
        injected = core.chunk_pages_word_safe([page], max_tokens=2, encoding=Owned())
    except TypeError:
        pytest.fail("word-safe encoding dependency injection is missing")
    assert [c.text for c in injected] == ["one two", "three"]
    assert legacy == injected
    assert observed == ["cl100k_base"]


@pytest.mark.parametrize("name", [None, 4, "other"])
def test_ingestion_refuses_encoding_mislabel_before_extraction(build_inputs, monkeypatch, name):
    from nvidia_rag import ingest
    from nvidia_rag.sources import load_source_manifest
    class Wrong:
        pass
    wrong = Wrong()
    if name is not None:
        wrong.name = name
    def no_extract(*args, **kwargs):
        pytest.fail("mislabeled encoding reached PDF extraction")
    monkeypatch.setattr(ingest, "extract_pdf_pages", no_extract)
    try:
        with pytest.raises(ValueError, match="encoding"):
            ingest.build_corpus_artifacts(sources=load_source_manifest(build_inputs["source_registry"]),
                source_dir=build_inputs["source_dir"], output_dir=build_inputs["output_dir"],
                strategy="word_safe", max_tokens=400, overlap_tokens=0, encoding=wrong)
    except TypeError:
        pytest.fail("ingestion encoding dependency injection is missing")


def test_ingestion_rejects_injection_for_another_strategy(build_inputs, monkeypatch):
    from nvidia_rag import ingest
    from nvidia_rag.sources import load_source_manifest
    class Named:
        name = "cl100k_base"
    def no_extract(*args, **kwargs):
        pytest.fail("unsupported injection reached extraction")
    monkeypatch.setattr(ingest, "extract_pdf_pages", no_extract)
    try:
        with pytest.raises(ValueError, match="word_safe"):
            ingest.build_corpus_artifacts(sources=load_source_manifest(build_inputs["source_registry"]),
                source_dir=build_inputs["source_dir"], output_dir=build_inputs["output_dir"],
                strategy="fixed_token", max_tokens=400, overlap_tokens=0, encoding=Named())
    except TypeError:
        pytest.fail("ingestion encoding dependency injection is missing")


@pytest.mark.parametrize("body", [b"", b"??? 0\n", b"YQ== -1\n", b"YQ== 0\nYQ== 1\n",
                                   b"YQ== 0\nYg== 0\n", b"YQ== x\n", b"YQ== 0 extra\n"])
def test_malformed_prepared_bpe_is_refused_without_factory(body, monkeypatch):
    module = library()
    monkeypatch.setattr(module, "_TOKENIZER_SHA256", sha256(body).hexdigest())
    with pytest.raises(ValueError):
        module._prepared_encoding(body)


@pytest.mark.parametrize("change", ["url", "hash", "closure", "missing-call", "config"])
def test_unexpected_installed_factory_refuses_without_fallback(build_inputs, monkeypatch, change):
    from tiktoken_ext import openai_public
    module = library()
    original_loader = openai_public.load_tiktoken_bpe
    if change == "closure":
        captured = "must refuse"
        def factory():
            return captured
    else:
        code = ("def factory():\n    load_tiktoken_bpe('https://example.com/encoding', expected_hash='wrong')\n"
                if change == "url" else
                "def factory():\n    load_tiktoken_bpe('https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken', expected_hash='wrong')\n"
                if change == "hash" else
                "def factory():\n    return {}\n" if change == "missing-call" else
                "def factory():\n    ranks=load_tiktoken_bpe('https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken', expected_hash='223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7')\n    return {'name':'other','pat_str':'x','mergeable_ranks':ranks,'special_tokens':{}}\n")
        namespace = {"load_tiktoken_bpe": original_loader}
        exec(code, namespace)
        factory = namespace["factory"]
    monkeypatch.setattr(openai_public, "cl100k_base", factory)
    body = next(build_inputs["tokenizer_cache_dir"].iterdir()).read_bytes()
    with pytest.raises(ValueError):
        module._prepared_encoding(body)
    assert openai_public.load_tiktoken_bpe is original_loader


def test_complete_receipt_and_failure_writer_errors_remain_pending(build_inputs, monkeypatch):
    module = library()
    original = module._write_json_exclusive
    calls = []
    def failed_close_and_diagnostic(path, value):
        if path.name == "failure.json":
            calls.append("diagnostic")
            raise OSError("diagnostic also failed")
        original(path, value)
        if path.name == "build-receipt.json":
            calls.append("completion")
            raise OSError("close failed after complete bytes")
    monkeypatch.setattr(module, "_write_json_exclusive", failed_close_and_diagnostic)
    with pytest.raises(OSError):
        module.build_local_corpus(**build_inputs)
    output = build_inputs["output_dir"]
    assert calls == ["completion", "diagnostic"]
    assert json.loads((output / "build-receipt.json").read_bytes())["status"] == "complete"
    assert not (output / "failure.json").exists()
    with pytest.raises(ValueError, match="pending"):
        module.load_verified_corpus(artifact_dir=output, source_registry=build_inputs["source_registry"])
    assert (output / "pending.json").is_file()


def test_pending_marker_is_written_before_builder_and_removed_after_closed_receipt(build_inputs, monkeypatch):
    module = library()
    original_builder = module.build_corpus_artifacts
    original_writer = module._write_json_exclusive
    observations = []
    def build(**kwargs):
        marker = kwargs["output_dir"] / "pending.json"
        assert marker.is_file(), "pending marker missing before builder"
        observations.append("builder guarded")
        return original_builder(**kwargs)
    def write(path, value):
        original_writer(path, value)
        if path.name == "build-receipt.json":
            assert (path.parent / "pending.json").is_file()
            observations.append("complete receipt closed while guarded")
    monkeypatch.setattr(module, "build_corpus_artifacts", build)
    monkeypatch.setattr(module, "_write_json_exclusive", write)
    result = module.build_local_corpus(**build_inputs)
    assert result["status"] == "complete"
    assert observations == ["builder guarded", "complete receipt closed while guarded"]
    assert not os.path.lexists(build_inputs["output_dir"] / "pending.json")
    assert module.load_verified_corpus(artifact_dir=build_inputs["output_dir"],
                                       source_registry=build_inputs["source_registry"]).chunks


def test_pending_creation_failure_precedes_builder(build_inputs, monkeypatch):
    module = library()
    original_writer = module._write_json_exclusive
    def failing_marker(path, value):
        if path.name == "pending.json":
            raise OSError("pending could not be created")
        original_writer(path, value)
    def forbidden_builder(**kwargs):
        pytest.fail("builder started without a successfully closed pending marker")
    monkeypatch.setattr(module, "_write_json_exclusive", failing_marker)
    monkeypatch.setattr(module, "build_corpus_artifacts", forbidden_builder)
    with pytest.raises(OSError, match="pending"):
        module.build_local_corpus(**build_inputs)
    assert not (build_inputs["output_dir"] / "build-receipt.json").exists()


def test_pending_removal_failure_preserves_valid_receipt_but_refuses_load(build_inputs, monkeypatch):
    module = library()
    original_unlink = Path.unlink
    def cannot_remove(path, *args, **kwargs):
        if path.name == "pending.json":
            raise OSError("pending unlink failed")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", cannot_remove)
    original_writer = module._write_json_exclusive
    def unavailable_diagnostic(path, value):
        if path.name == "failure.json":
            raise OSError("diagnostic unavailable")
        return original_writer(path, value)
    monkeypatch.setattr(module, "_write_json_exclusive", unavailable_diagnostic)
    with pytest.raises(OSError, match="pending"):
        module.build_local_corpus(**build_inputs)
    output = build_inputs["output_dir"]
    assert (output / "build-receipt.json").is_file()
    assert (output / "pending.json").is_file()
    assert not (output / "failure.json").exists()
    with pytest.raises(ValueError, match="pending"):
        module.load_verified_corpus(artifact_dir=output, source_registry=build_inputs["source_registry"])


@pytest.mark.parametrize("kind", ["file", "directory", "dangling-symlink"])
def test_existing_pending_marker_refuses_otherwise_complete_corpus(tmp_path, kind):
    artifact = Artifact(tmp_path)
    marker = artifact.directory / "pending.json"
    if kind == "file":
        marker.write_bytes(b"pending")
    elif kind == "directory":
        marker.mkdir()
    else:
        try:
            marker.symlink_to(tmp_path / "missing-marker-target")
        except OSError:
            pytest.skip("host does not grant symlink creation")
    with pytest.raises(ValueError, match="pending"):
        artifact.load()


def test_pending_marker_appearing_during_load_is_refused(tmp_path, monkeypatch):
    artifact = Artifact(tmp_path)
    module = library()
    original = module.load_chunks
    def changed(path):
        result = original(path)
        (artifact.directory / "pending.json").write_bytes(b"pending")
        return result
    monkeypatch.setattr(module, "load_chunks", changed)
    with pytest.raises(ValueError, match="pending"):
        artifact.load()

