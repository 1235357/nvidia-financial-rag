"""Synthetic semantic acceptance boundary tests, never actual BGE acceptance."""
from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from test_corpus_recipe import Artifact, digest, write_json
from test_corpus_embeddings import UnitModel

ROOT = Path(__file__).resolve().parents[1]


def script():
    path = ROOT / "acceptance/corpus_semantic.py"
    assert path.is_file(), "separate semantic acceptance source is missing"
    spec = importlib.util.spec_from_file_location("corpus_semantic_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case(tmp_path, monkeypatch):
    from nvidia_rag import corpus_embeddings, embedding_store
    a = Artifact(tmp_path)
    registry_body = (ROOT / "corpus/sources-annual-plus-fy2025-q2.json").read_bytes()
    a.sources = json.loads(registry_body)["sources"]
    a.records = [{"chunk_id": row["document_id"] + ":p1:w:0123456789abcdef",
                  "document_id": row["document_id"], "page_number": 1,
                  "text": "Synthetic period candidate " + row["document_id"]} for row in a.sources]
    a.manifest.update(document_count=8, page_count=16, nonempty_page_count=8, chunk_count=8,
        documents=[dict(document_id=r["document_id"], document_label=r["label"], source_url=r["url"],
                        source_filename=r["filename"], source_sha256=r["sha256"],
                        page_count=2, nonempty_page_count=1, chunk_count=1,
                        **({"reporting_period": r["reporting_period"]} if "reporting_period" in r else {}))
                   for r in a.sources])
    a.receipt["sources"] = [dict(document_id=r["document_id"], source_sha256=r["sha256"], source_bytes=1)
                            for r in a.sources]
    a.sync()
    a.registry.write_bytes(registry_body)
    a.receipt["source_registry_sha256"] = digest(a.registry)
    write_json(a.directory / "build-receipt.json", a.receipt)
    monkeypatch.setattr(embedding_store, "_build_runtime", lambda: {
        "python": "3.13.5", "numpy": np.__version__, "sentence-transformers": "synthetic",
        "torch": "synthetic", "transformers": "synthetic"})
    vector = tmp_path / "vectors"
    corpus_embeddings.build_local_bge_vectors(artifact_dir=a.directory, source_registry=a.registry,
                                             vector_dir=vector, model=UnitModel())
    package = Path(corpus_embeddings.__file__).parent
    identity = {"wheel_basename": "nvidia_financial_rag-0.1.0-py3-none-any.whl", "wheel_bytes": 1,
                "wheel_sha256": "0" * 64, "distribution_version": "0.1.0",
                "module_sha256": {p.name: digest(p) for p in package.glob("*.py")}}
    receipt_path = vector / "vector-build-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["model"].update(execution_origin="cached-local-only", device="cpu", local_files_only=True)
    receipt["wheel"] = identity
    write_json(receipt_path, receipt)
    cache = tmp_path / "cache"
    snapshot = cache / "models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    snapshot.mkdir(parents=True)
    for name in ("model.safetensors", "config.json", "modules.json", "tokenizer.json", "tokenizer_config.json",
                 "sentence_bert_config.json", "1_Pooling/config.json"):
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic body")
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_bytes((ROOT / "evaluation/source-candidates/fy2025-q2-income-duration.jsonl").read_bytes())
    wheel = tmp_path / identity["wheel_basename"]
    wheel.write_bytes(b"synthetic verifier double")
    return a, vector, cache, snapshot, candidates, wheel, identity


def prepare(tmp_path, monkeypatch, model=None):
    a, vector, cache, snapshot, candidates, wheel, identity = case(tmp_path, monkeypatch)
    module = script()
    monkeypatch.setattr(module, "verify_installed_wheel", lambda path: identity)
    monkeypatch.setattr(module, "_load_cached_model", lambda cache: model or UnitModel())
    options = dict(artifact_dir=a.directory, source_registry=a.registry, vector_dir=vector,
                   candidates=candidates, wheel=wheel, model_cache_dir=cache, output=tmp_path / "result.json")
    return module, options, a, snapshot


def test_same_four_queries_global_and_q2_use_actual_single_call_candidates(tmp_path, monkeypatch):
    model = UnitModel()
    module, options, a, snapshot = prepare(tmp_path, monkeypatch, model)
    before = {p: p.read_bytes() for p in a.directory.iterdir()}
    receipt = module.run_comparison(**options)
    assert receipt["status"] == "complete"
    assert receipt["threshold"] == 0.78 and receipt["top_k"] == 5
    assert receipt["generation_runs"] == 0 and receipt["human_acceptance"] is False
    assert receipt["financial_field_support"] == "not_evaluated"
    assert [r["document_scope"] for r in receipt["comparisons"]] == [None, ["nvidia-fy2025-q2"]]
    assert len(model.calls) == 8
    assert all(texts[0].startswith("Represent this sentence for searching relevant passages: ")
               and kwargs == {"normalize_embeddings": True, "batch_size": 32, "show_progress_bar": False}
               for texts, kwargs in model.calls)
    global_rows, q2_rows = [r["records"] for r in receipt["comparisons"]]
    assert len(global_rows) == len(q2_rows) == 4
    for row in global_rows:
        assert len(row["candidates"]) == 5 and row["elapsed_ms"] >= 0
        assert row["answer"] == "" and row["generation_runs"] == 0
    for row in q2_rows:
        assert {c["document_id"] for c in row["candidates"]} == {"nvidia-fy2025-q2"}
    assert {p: p.read_bytes() for p in a.directory.iterdir()} == before
    assert options["output"].read_text().find(str(tmp_path)) == -1


def test_existing_output_precedes_all_model_and_input_work(tmp_path, monkeypatch):
    module = script()
    output = tmp_path / "exists.json"
    output.write_text("retained")
    monkeypatch.setattr(module, "verify_installed_wheel", lambda p: pytest.fail("wheel before output refusal"))
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("model before output refusal"))
    with pytest.raises(ValueError, match="exist"):
        module.run_comparison(artifact_dir=tmp_path / "absent", source_registry=tmp_path / "absent.json",
            vector_dir=tmp_path / "absent-vectors", candidates=tmp_path / "absent-candidates", wheel=tmp_path / "absent.whl",
            model_cache_dir=tmp_path / "absent-cache", output=output)
    assert output.read_text() == "retained"


@pytest.mark.parametrize("input_name", ["artifact_dir", "vector_dir", "model_cache_dir"])
def test_output_inside_readonly_input_refused_before_model(tmp_path, monkeypatch, input_name):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    options["output"] = options[input_name] / "result.json"
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("overlap reached model"))
    with pytest.raises(ValueError, match="overlap|read-only"):
        module.run_comparison(**options)
    assert not options["output"].exists()


@pytest.mark.parametrize("input_name", ["artifact_dir", "vector_dir", "candidates"])
def test_changed_input_during_model_load_is_failed_receipt(tmp_path, monkeypatch, input_name):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    path = options[input_name]
    if path.is_dir():
        path = path / ("chunks.jsonl" if input_name == "artifact_dir" else "embedding-manifest.json")
    def changed(cache):
        path.write_bytes(path.read_bytes() + b"\n")
        return UnitModel()
    monkeypatch.setattr(module, "_load_cached_model", changed)
    with pytest.raises(ValueError, match="changed|closure"):
        module.run_comparison(**options)
    failed = json.loads(options["output"].read_text())
    assert failed["status"] == "failed" and "comparisons" not in failed


def test_injected_vector_receipt_cannot_claim_cached_semantic_acceptance(tmp_path, monkeypatch):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    path = options["vector_dir"] / "vector-build-receipt.json"
    value = json.loads(path.read_text())
    value["model"].update(execution_origin="operator-injected", device=None, local_files_only=None)
    write_json(path, value)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("injected vectors reached model"))
    with pytest.raises(ValueError, match="cached|origin"):
        module.run_comparison(**options)
    assert not options["output"].exists()


def test_changed_candidate_question_is_refused_before_model(tmp_path, monkeypatch):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    options["candidates"].write_bytes(options["candidates"].read_bytes().replace(b"three-month", b"two-month"))
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("changed question reached model"))
    with pytest.raises(ValueError, match="candidate|question|frozen"):
        module.run_comparison(**options)
    assert not options["output"].exists()


def test_cache_hash_records_bodies_without_paths_and_detects_change(tmp_path, monkeypatch):
    module, options, _, snapshot = prepare(tmp_path, monkeypatch)
    initial = module.model_snapshot_identity(options["model_cache_dir"])
    assert "model.safetensors" in initial["files"]
    assert initial["files"]["model.safetensors"] == {"bytes": 14, "sha256": sha256(b"synthetic body").hexdigest()}
    (snapshot / "model.safetensors").write_text("changed weight")
    assert module.model_snapshot_identity(options["model_cache_dir"]) != initial


def test_missing_cache_body_precedes_model(tmp_path, monkeypatch):
    module, options, _, snapshot = prepare(tmp_path, monkeypatch)
    (snapshot / "model.safetensors").unlink()
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("missing cache reached model"))
    with pytest.raises(ValueError, match="cache|snapshot|prepared"):
        module.run_comparison(**options)
    assert not options["output"].exists()


def test_cache_model_change_during_query_prevents_complete_receipt(tmp_path, monkeypatch):
    module, options, _, snapshot = prepare(tmp_path, monkeypatch)
    model = UnitModel(action=lambda: (snapshot / "model.safetensors").write_text("changed"))
    monkeypatch.setattr(module, "_load_cached_model", lambda p: model)
    with pytest.raises(ValueError, match="cache|changed"):
        module.run_comparison(**options)
    assert json.loads(options["output"].read_text())["status"] == "failed"


def test_model_failure_receipt_has_finite_error_without_raw_secret(tmp_path, monkeypatch):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    def failure(cache):
        raise RuntimeError("PRIVATE_TOKEN " + str(tmp_path))
    monkeypatch.setattr(module, "_load_cached_model", failure)
    with pytest.raises(RuntimeError):
        module.run_comparison(**options)
    body = options["output"].read_text()
    value = json.loads(body)
    assert value["status"] == "failed" and value["error_type"] == "RuntimeError"
    assert "PRIVATE_TOKEN" not in body and str(tmp_path) not in body


def test_cached_constructor_uses_fixed_cpu_local_revision_and_explicit_cache(tmp_path, monkeypatch):
    module = script()
    calls = []
    class Model:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))
    monkeypatch.setitem(sys.modules, "sentence_transformers", type("Module", (), {"SentenceTransformer": Model}))
    module._load_cached_model(tmp_path)
    assert calls == [("BAAI/bge-small-en-v1.5", {
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "device": "cpu",
        "local_files_only": True, "cache_folder": str(tmp_path)})]


def test_zero_score_candidates_are_preserved_as_below_threshold_without_quality_score(tmp_path, monkeypatch):
    matrix = np.zeros((1, 384), dtype=np.float32)
    matrix[0, 1] = 1
    module, options, _, _ = prepare(tmp_path, monkeypatch, UnitModel(matrix=matrix))
    receipt = module.run_comparison(**options)
    for comparison in receipt["comparisons"]:
        for row in comparison["records"]:
            assert row["evidence_ids"] == [] and row["can_answer"] is False
            assert all(c["score"] == 0 and c["below_threshold"] is True for c in row["candidates"])
    assert "accuracy" not in receipt and "recall" not in receipt and "refusal_rate" not in receipt


def test_cli_has_frozen_scope_and_no_threshold_generation_or_download_switch(monkeypatch):
    module = script()
    monkeypatch.setattr(sys, "argv", ["corpus_semantic.py", "--threshold", "0"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


def test_complete_json_close_failure_remains_pending_without_overwriting_bytes(tmp_path, monkeypatch):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    original = module._write_exclusive
    def failed_write(path, value):
        if value.get("status") == "complete":
            original(path, value)
            raise OSError("complete JSON close failed")
        if value.get("status") == "failed":
            raise OSError("diagnostic also failed")
        return original(path, value)
    monkeypatch.setattr(module, "_write_exclusive", failed_write)
    with pytest.raises(OSError, match="close"):
        module.run_comparison(**options)
    assert json.loads(options["output"].read_bytes())["status"] == "complete"
    assert options["output"].with_name(options["output"].name + ".pending.json").is_file()
    with pytest.raises(ValueError, match="pending"):
        module.load_completed_comparison(options["output"])


def test_completion_marker_unlink_failure_does_not_accept_receipt(tmp_path, monkeypatch):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    original = Path.unlink
    def failed_unlink(path, *args, **kwargs):
        if path.name == "result.json.pending.json":
            raise OSError("pending removal failed")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", failed_unlink)
    with pytest.raises(OSError, match="pending"):
        module.run_comparison(**options)
    with pytest.raises(ValueError, match="pending"):
        module.load_completed_comparison(options["output"])


def test_pending_marker_closed_before_model_and_removed_on_success(tmp_path, monkeypatch):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    marker = options["output"].with_name(options["output"].name + ".pending.json")
    def guarded(cache):
        assert marker.is_file()
        assert json.loads(marker.read_bytes())["status"] == "pending"
        return UnitModel()
    monkeypatch.setattr(module, "_load_cached_model", guarded)
    receipt = module.run_comparison(**options)
    assert not marker.exists()
    assert module.load_completed_comparison(options["output"]) == receipt


def test_preexisting_pending_marker_precedes_inputs_and_model(tmp_path, monkeypatch):
    module = script()
    output = tmp_path / "result.json"
    marker = output.with_name(output.name + ".pending.json")
    marker.write_text("retained pending")
    monkeypatch.setattr(module, "verify_installed_wheel", lambda p: pytest.fail("wheel before pending refusal"))
    with pytest.raises(ValueError, match="pending|exist"):
        module.run_comparison(artifact_dir=tmp_path / "absent", source_registry=tmp_path / "absent.json",
            vector_dir=tmp_path / "vectors", candidates=tmp_path / "candidates", wheel=tmp_path / "wheel",
            model_cache_dir=tmp_path / "cache", output=output)
    assert marker.read_text() == "retained pending" and not output.exists()


def test_pending_marker_creation_close_failure_prevents_model(tmp_path, monkeypatch):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    original = module._write_exclusive
    def failed_write(path, value):
        if value.get("status") == "pending":
            original(path, value)
            raise OSError("pending close failed")
        return original(path, value)
    monkeypatch.setattr(module, "_write_exclusive", failed_write)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("model after pending close failure"))
    with pytest.raises(OSError, match="pending"):
        module.run_comparison(**options)


def test_pending_marker_appearing_during_result_load_is_refused(tmp_path, monkeypatch):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    module.run_comparison(**options)
    original = Path.read_bytes
    marker = options["output"].with_name(options["output"].name + ".pending.json")
    def changed(path):
        result = original(path)
        if path == options["output"]:
            marker.write_text("pending")
        return result
    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="pending"):
        module.load_completed_comparison(options["output"])


def test_corrupted_input_during_cache_preflight_is_refused_before_model(tmp_path, monkeypatch):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    original = module.model_snapshot_identity
    path = options["artifact_dir"] / "chunks.jsonl"
    def changed(cache):
        result = original(cache)
        path.write_bytes(path.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(module, "model_snapshot_identity", changed)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("changed closure reached model"))
    with pytest.raises(ValueError, match="changed|closure"):
        module.run_comparison(**options)
    assert not options["output"].exists()


def test_failed_diagnostic_write_preserves_pending_and_original_model_error(tmp_path, monkeypatch):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    original = module._write_exclusive
    def failed_write(path, value):
        if value.get("status") == "failed":
            raise OSError("failed diagnostic writer")
        return original(path, value)
    def failed_model(cache):
        raise RuntimeError("original model error")
    monkeypatch.setattr(module, "_write_exclusive", failed_write)
    monkeypatch.setattr(module, "_load_cached_model", failed_model)
    with pytest.raises(RuntimeError, match="original"):
        module.run_comparison(**options)
    assert options["output"].with_name(options["output"].name + ".pending.json").exists()
    assert not options["output"].exists()


@pytest.mark.parametrize("target", ["confined-file", "escaping-file", "linked-directory"])
def test_snapshot_file_links_are_confined_and_directories_never_followed(tmp_path, monkeypatch, target):
    module, options, _, snapshot = prepare(tmp_path, monkeypatch)
    if target == "linked-directory":
        body = tmp_path / "outside-directory"
        body.mkdir()
        path = snapshot / "linked"
    else:
        body = (options["model_cache_dir"] / "blobs" / "body"
                if target == "confined-file" else tmp_path / "outside-body")
        body.parent.mkdir(parents=True, exist_ok=True)
        body.write_bytes(b"linked body")
        path = snapshot / "model.safetensors"
        path.unlink()
    try:
        path.symlink_to(body, target_is_directory=target == "linked-directory")
    except OSError as error:
        pytest.skip("host symlink privilege unavailable: " + type(error).__name__)
    if target == "confined-file":
        assert module.model_snapshot_identity(options["model_cache_dir"])["files"]["model.safetensors"] == {
            "bytes": 11, "sha256": sha256(b"linked body").hexdigest()}
    else:
        with pytest.raises(ValueError, match="escape|linked"):
            module.model_snapshot_identity(options["model_cache_dir"])


def test_vector_receipt_wheel_identity_cannot_be_reassigned(tmp_path, monkeypatch):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    receipt_path = options["vector_dir"] / "vector-build-receipt.json"
    value = json.loads(receipt_path.read_bytes())
    value["wheel"]["wheel_sha256"] = "1" * 64
    write_json(receipt_path, value)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("different wheel reached model"))
    with pytest.raises(ValueError, match="wheel|identity"):
        module.run_comparison(**options)
    assert not options["output"].exists()


def test_input_changed_when_pending_writer_closes_is_refused_before_model(tmp_path, monkeypatch):
    module, options, _, _ = prepare(tmp_path, monkeypatch)
    original = module._write_exclusive
    def changed(path, value):
        original(path, value)
        if value.get("status") == "pending":
            chunks = options["artifact_dir"] / "chunks.jsonl"
            chunks.write_bytes(chunks.read_bytes() + b"\n")
    monkeypatch.setattr(module, "_write_exclusive", changed)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("changed input reached model"))
    with pytest.raises(ValueError, match="changed|closure"):
        module.run_comparison(**options)
    assert json.loads(options["output"].read_bytes())["status"] == "failed"
