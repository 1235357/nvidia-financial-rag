"""Actual service arithmetic with injected encoders; no real model executes."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_corpus_semantic import case as old_fixture
from test_expanded_cases import case as candidate, experiment, write_inputs

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "acceptance/corpus_semantic_expanded.py"


def api():
    assert SCRIPT.is_file(), "expanded semantic producer must be implemented"
    spec = importlib.util.spec_from_file_location("corpus_semantic_expanded_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class Encoder:
    def __init__(self, matrix=None, action=None):
        self.matrix = np.zeros((1, 384), dtype=np.float32) if matrix is None else matrix
        if matrix is None:
            self.matrix[0, 0] = 4.0
            self.matrix[0, 1] = 3.0
        self.calls = []
        self.action = action

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        if self.action:
            self.action()
        return self.matrix


def fixture(tmp_path, monkeypatch, model=None):
    # Existing corpus/vector fixture exercises their real verified loaders.
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    a, vector, cache, snapshot, _, wheel, identity = old_fixture(inputs, monkeypatch)
    identity = {**identity, "wheel_sha256": digest(wheel), "wheel_bytes": wheel.stat().st_size}
    vector_receipt = vector / "vector-build-receipt.json"
    receipt = json.loads(vector_receipt.read_bytes())
    receipt["wheel"] = identity
    json_write(vector_receipt, receipt)
    cfg = experiment()
    cfg["model"]["revision"] = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    cfg["documents"] = [{"document_id": r["document_id"], "pdf_sha256": r["sha256"],
        "issuer_url": r["url"], "report_fiscal_year": r.get("reporting_period", {}).get("fiscal_year", 2025),
        "report_fiscal_quarter": r.get("reporting_period", {}).get("quarter"), "page_count": 2}
        for r in a.sources]
    cfg["document_count"], cfg["chunk_count"] = 8, 8
    cfg["scopes"] = [{"id": "all-reports", "document_ids": None},
                     {"id": "q2", "document_ids": ["nvidia-fy2025-q2"]}]
    row = candidate()
    row["query"] = "  What is the reported amount?  "
    row["request"]["report_document_id"] = "nvidia-fy2025-q2"
    row["source_cells"] = []
    row["expected"] = {"outcome": "refuse", "cell_ids": [],
                       "negative_reason": "A SECRET LABEL FORBIDDEN TO RETRIEVAL",
                       "rationale_binding_ids": ["inventory:1"]}
    row["scope_ids"] = ["all-reports", "q2"]
    freeze_dir = inputs / "freeze"
    freeze_dir.mkdir()
    replay = freeze_dir / "source-replay.json"
    replay.write_bytes(b'{"status":"synthetic-preflight-double"}\n')
    cfg["input_sha256"].update({key: digest(path) for key, path in {
        "registry": a.registry, "corpus_manifest": a.directory / "manifest.json",
        "corpus_chunks": a.directory / "chunks.jsonl",
        "vector_manifest": vector / "embedding-manifest.json",
        "vector_matrix": vector / "bge-small-en-v1.5.npy", "source_replay": replay,
        "wheel": wheel}.items()})
    cases, freeze, config = write_inputs(freeze_dir, [row], cfg)
    module = api()
    encoder = model or Encoder()
    monkeypatch.setattr(module, "verify_installed_wheel", lambda p: identity)
    monkeypatch.setattr(module, "load_source_replay", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_load_cached_model", lambda p: encoder)
    options = dict(artifact_dir=a.directory, source_registry=a.registry, vector_dir=vector,
        cases=cases, case_freeze=freeze, source_replay=replay, experiment=config,
        wheel=wheel, model_cache_dir=cache, output_dir=tmp_path / "comparison")
    return module, options, encoder, snapshot, identity


def failed(options):
    directory = options["output_dir"]
    assert (directory / "pending.json").exists()
    value = json.loads((directory / "failure.json").read_bytes())
    assert value == {"schema_version": 2, "status": "failed", "code": "expanded_comparison_incomplete"}
    return directory


def test_actual_single_encode_capture_scope_and_context_without_label_leakage(tmp_path, monkeypatch):
    module, options, model, _, _ = fixture(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in options["artifact_dir"].iterdir()}
    receipt = module.run_comparison(**options)
    out = options["output_dir"]
    assert receipt == json.loads((out / "manifest.json").read_bytes())
    assert receipt["status"] == "completed" and receipt["query_count"] == 2
    assert receipt.keys() == {"schema_version", "status", "experiment_sha256", "input_sha256",
                             "input_paths", "producer", "query_count", "files"}
    assert receipt["producer"] == {"numpy_version": np.__version__,
        "python_version": module.platform.python_version(), "platform": module.platform.platform()}
    assert not (out / "pending.json").exists() and not (out / "failure.json").exists()
    assert set(receipt["files"]) == {"queries-raw.npy", "queries-normalized.npy", "query-index.json", "results.json"}
    assert all(digest(out / name) == value for name, value in receipt["files"].items())
    raw = np.load(out / "queries-raw.npy", allow_pickle=False)
    consumed = np.load(out / "queries-normalized.npy", allow_pickle=False)
    assert raw.dtype == np.float32 and raw.shape == (2, 384)
    assert consumed.dtype == np.float64 and consumed.shape == (2, 384)
    assert np.array_equal(raw, np.repeat(model.matrix, 2, axis=0))
    expected = model.matrix.astype(np.float64)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.array_equal(consumed, np.repeat(expected, 2, axis=0))
    instruction = "Represent this sentence for searching relevant passages: "
    assert model.calls == [([instruction + "What is the reported amount?"], {
        "normalize_embeddings": True, "batch_size": 32, "show_progress_bar": False})] * 2
    index = json.loads((out / "query-index.json").read_bytes())
    rows = json.loads((out / "results.json").read_bytes())["rows"]
    assert [(r["row"], r["case_id"], r["scope_id"]) for r in rows] == [(0, "positive", "all-reports"), (1, "positive", "q2")]
    assert [r["scoped_global_chunk_indices"] for r in rows] == [list(range(8)), [4]]
    assert [r["scope_document_ids"] for r in rows] == [None, ["nvidia-fy2025-q2"]]
    assert all(r["encoder_call_count"] == 1 and r["original_record_sha256"] is None for r in index["rows"])
    for row in rows:
        assert len(row["considered_scores"]) == len(row["scoped_global_chunk_indices"])
        assert row["threshold"] == 0.78 and row["top_k_limit"] == 5
        assert row["context_empty"] is False and row["can_answer"] is True
        assert "SECRET LABEL" not in json.dumps(row) and "expected" not in row
    assert {p: p.read_bytes() for p in options["artifact_dir"].iterdir()} == before


def test_below_threshold_scores_are_kept_and_empty_context_digest_is_actual(tmp_path, monkeypatch):
    raw = np.zeros((1, 384), dtype=np.float32)
    raw[0, 1] = 1
    module, options, _, _, _ = fixture(tmp_path, monkeypatch, Encoder(raw))
    module.run_comparison(**options)
    for row in json.loads((options["output_dir"] / "results.json").read_bytes())["rows"]:
        assert row["considered_scores"] and all(score == 0 for score in row["considered_scores"])
        assert row["evidence_ids"] == row["citations"] == []
        assert row["can_answer"] is False and row["context_empty"] is True
        assert row["context_sha256"] == sha256(b"").hexdigest()


@pytest.mark.parametrize("input_key", ["registry", "corpus_manifest", "corpus_chunks", "vector_manifest",
    "vector_matrix", "cases", "case_freeze", "source_replay", "wheel"])
def test_changed_frozen_input_refuses_before_model(tmp_path, monkeypatch, input_key):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    paths = {"registry": options["source_registry"], "corpus_manifest": options["artifact_dir"] / "manifest.json",
        "corpus_chunks": options["artifact_dir"] / "chunks.jsonl", "vector_manifest": options["vector_dir"] / "embedding-manifest.json",
        "vector_matrix": options["vector_dir"] / "bge-small-en-v1.5.npy", **{k: options[k] for k in ("cases", "case_freeze", "source_replay", "wheel")}}
    path = paths[input_key]
    path.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("changed input reached model"))
    with pytest.raises(ValueError, match="SHA|hash|identity|closure|differs"):
        module.run_comparison(**options)
    assert not options["output_dir"].exists()


@pytest.mark.parametrize("input_name", ["artifact_dir", "vector_dir", "model_cache_dir", "cases", "source_replay", "wheel"])
def test_output_overlap_precedes_all_input_work(tmp_path, monkeypatch, input_name):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    path = options[input_name]
    options["output_dir"] = (path if path.is_dir() else path.parent) / "new-output"
    monkeypatch.setattr(module, "load_experiment", lambda p: pytest.fail("overlap reached input loader"))
    with pytest.raises(ValueError, match="overlap|read-only"):
        module.run_comparison(**options)
    assert not options["output_dir"].exists()


def test_existing_output_is_never_reused_or_overwritten(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    options["output_dir"].mkdir()
    (options["output_dir"] / "retained").write_bytes(b"original")
    monkeypatch.setattr(module, "load_experiment", lambda p: pytest.fail("existing output reached loader"))
    with pytest.raises(ValueError, match="exist"):
        module.run_comparison(**options)
    assert (options["output_dir"] / "retained").read_bytes() == b"original"


@pytest.mark.parametrize("mutation", ["empty", "unknown", "duplicate", "instruction", "revision"])
def test_bad_scope_and_model_contracts_never_reach_model(tmp_path, monkeypatch, mutation):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    cfg = json.loads(options["experiment"].read_bytes())
    if mutation == "empty": cfg["scopes"][1]["document_ids"] = []
    if mutation == "unknown": cfg["scopes"][1]["document_ids"] = ["unknown"]
    if mutation == "duplicate": cfg["scopes"][1]["document_ids"] *= 2
    if mutation == "instruction": cfg["model"]["instruction"] = "Different query instruction: "
    if mutation == "revision": cfg["model"]["revision"] = "b" * 40
    json_write(options["experiment"], cfg)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("bad scope/model reached model"))
    with pytest.raises(ValueError, match="scope|model|instruction|revision"):
        module.run_comparison(**options)
    assert not options["output_dir"].exists()


@pytest.mark.parametrize("directory_key", ["artifact_dir", "vector_dir", "source_replay"])
@pytest.mark.parametrize("guard", ["pending", "failure"])
def test_input_guard_prevents_model(tmp_path, monkeypatch, directory_key, guard):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    path = options[directory_key]
    marker = path / (guard + ".json") if path.is_dir() else path.with_name(path.name + "." + guard + ".json")
    marker.write_bytes(b"guard")
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("guard reached model"))
    with pytest.raises(ValueError, match="pending|failed|failure|completion"):
        module.run_comparison(**options)
    assert not options["output_dir"].exists()


def test_config_changed_after_validated_load_fails_before_model(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    original = module.load_experiment
    def changed(path):
        loaded = original(path)
        path.write_bytes(path.read_bytes() + b"\n")
        return loaded
    monkeypatch.setattr(module, "load_experiment", changed)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("changed config reached model"))
    with pytest.raises(ValueError, match="experiment|closure|changed"):
        module.run_comparison(**options)
    assert not options["output_dir"].exists()


@pytest.mark.parametrize("phase", ["model", "query", "source-preflight", "cache-preflight"])
def test_input_changes_across_execution_phases_never_complete(tmp_path, monkeypatch, phase):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    path = options["artifact_dir"] / "chunks.jsonl"
    def action(): path.write_bytes(path.read_bytes() + b"\n")
    if phase == "model":
        monkeypatch.setattr(module, "_load_cached_model", lambda p: (action(), Encoder())[1])
    if phase == "query":
        monkeypatch.setattr(module, "_load_cached_model", lambda p: Encoder(action=action))
    if phase == "source-preflight":
        monkeypatch.setattr(module, "load_source_replay", lambda *args, **kwargs: (action(), {})[1])
    if phase == "cache-preflight":
        original = module.model_snapshot_identity
        def changed(cache):
            result = original(cache)
            action()
            return result
        monkeypatch.setattr(module, "model_snapshot_identity", changed)
    with pytest.raises(ValueError, match="changed|closure"):
        module.run_comparison(**options)
    if phase in ("model", "query"): failed(options)
    else: assert not options["output_dir"].exists()


@pytest.mark.parametrize("kind", ["float64", "shape", "nan", "infinity", "zero", "list"])
def test_encoder_raw_output_must_be_exact_finite_nonzero_float32_384(tmp_path, monkeypatch, kind):
    raw = np.ones((1, 384), dtype=np.float32)
    if kind == "float64": raw = raw.astype(np.float64)
    if kind == "shape": raw = raw[:, :383]
    if kind == "nan": raw[0, 0] = np.nan
    if kind == "infinity": raw[0, 0] = np.inf
    if kind == "zero": raw[:] = 0
    if kind == "list": raw = raw.tolist()
    module, options, _, _, _ = fixture(tmp_path, monkeypatch, Encoder(raw))
    with pytest.raises(ValueError, match="encoder|float32|vector|shape|norm|finite"):
        module.run_comparison(**options)
    directory = failed(options)
    assert not (directory / "manifest.json").exists()


def test_actual_consumed_vector_corruption_is_refused(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    original = module.DenseRetriever
    class Corrupt(original):
        def inspect(self, *args, **kwargs):
            result = super().inspect(*args, **kwargs)
            return replace(result, query_vector=tuple([1.0] * 384))
    monkeypatch.setattr(module, "DenseRetriever", Corrupt)
    with pytest.raises(ValueError, match="consumed|normaliz"):
        module.run_comparison(**options)
    failed(options)


def test_captured_raw_copy_survives_encoder_reusing_its_buffer(tmp_path, monkeypatch):
    class SharedBuffer(Encoder):
        def encode(self, texts, **kwargs):
            if self.calls:
                self.matrix[0, 0], self.matrix[0, 1] = 3, 4
            return super().encode(texts, **kwargs)
    module, options, _, _, _ = fixture(tmp_path, monkeypatch, SharedBuffer())
    module.run_comparison(**options)
    raw = np.load(options["output_dir"] / "queries-raw.npy", allow_pickle=False)
    assert raw[0, :2].tolist() == [4, 3] and raw[1, :2].tolist() == [3, 4]


def test_cached_body_change_during_query_is_guarded(tmp_path, monkeypatch):
    module, options, _, snapshot, _ = fixture(tmp_path, monkeypatch)
    model = Encoder(action=lambda: (snapshot / "model.safetensors").write_bytes(b"changed weights"))
    monkeypatch.setattr(module, "_load_cached_model", lambda p: model)
    with pytest.raises(ValueError, match="cache|changed"):
        module.run_comparison(**options)
    failed(options)


def test_wheel_identity_change_during_query_is_guarded(tmp_path, monkeypatch):
    module, options, _, _, identity = fixture(tmp_path, monkeypatch)
    identity_changed = deepcopy(identity)
    model = Encoder(action=lambda: identity_changed.update(distribution_version="changed"))
    monkeypatch.setattr(module, "verify_installed_wheel", lambda p: identity_changed)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: model)
    with pytest.raises(ValueError, match="wheel|identity|changed"):
        module.run_comparison(**options)
    failed(options)


def test_model_failure_diagnostic_does_not_echo_private_message(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    def failure(path): raise RuntimeError("PRIVATE_TOKEN " + str(tmp_path))
    monkeypatch.setattr(module, "_load_cached_model", failure)
    with pytest.raises(RuntimeError): module.run_comparison(**options)
    directory = failed(options)
    assert "PRIVATE_TOKEN" not in (directory / "failure.json").read_text()
    assert str(tmp_path) not in (directory / "failure.json").read_text()


@pytest.mark.parametrize("file_name", ["pending.json", "queries-raw.npy", "queries-normalized.npy", "query-index.json", "results.json", "manifest.json"])
def test_output_close_failure_never_establishes_completion(tmp_path, monkeypatch, file_name):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    if file_name.endswith(".npy"):
        original = module._write_array_exclusive
        def failed_write(path, value):
            result = original(path, value)
            if path.name == file_name: raise OSError("owned array close failure")
            return result
        monkeypatch.setattr(module, "_write_array_exclusive", failed_write)
    else:
        original = module._write_json_exclusive
        def failed_write(path, value):
            result = original(path, value)
            if path.name == file_name: raise OSError("owned JSON close failure")
            return result
        monkeypatch.setattr(module, "_write_json_exclusive", failed_write)
    with pytest.raises(OSError, match="close"):
        module.run_comparison(**options)
    failed(options)


def test_changed_input_after_manifest_write_retains_pending(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    original = module._write_json_exclusive
    def changed(path, value):
        result = original(path, value)
        if path.name == "manifest.json":
            options["cases"].write_bytes(options["cases"].read_bytes() + b"\n")
        return result
    monkeypatch.setattr(module, "_write_json_exclusive", changed)
    with pytest.raises(ValueError, match="changed|closure"):
        module.run_comparison(**options)
    directory = failed(options)
    assert (directory / "manifest.json").exists()


def test_written_output_mutation_is_detected_before_manifest(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    original = module._write_json_exclusive
    def changed(path, value):
        result = original(path, value)
        if path.name == "results.json":
            raw = options["output_dir"] / "queries-raw.npy"
            raw.write_bytes(raw.read_bytes() + b"corruption")
        return result
    monkeypatch.setattr(module, "_write_json_exclusive", changed)
    with pytest.raises(ValueError, match="output|digest|changed"):
        module.run_comparison(**options)
    directory = failed(options)
    assert not (directory / "manifest.json").exists()


def test_pending_is_closed_before_model_creation(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    def guarded(path):
        assert json.loads((options["output_dir"] / "pending.json").read_bytes())["status"] == "pending"
        return Encoder()
    monkeypatch.setattr(module, "_load_cached_model", guarded)
    module.run_comparison(**options)


def test_unlink_failure_preserves_guard_even_with_final_manifest(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    original = Path.unlink
    def failure(path, *args, **kwargs):
        if path == options["output_dir"] / "pending.json": raise OSError("pending unlink failure")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", failure)
    with pytest.raises(OSError, match="unlink"):
        module.run_comparison(**options)
    directory = failed(options)
    assert json.loads((directory / "manifest.json").read_bytes())["status"] == "completed"


def test_failure_diagnostic_write_cannot_replace_original_error(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    original = module._write_json_exclusive
    def write(path, value):
        if path.name == "failure.json": raise OSError("diagnostic writer failed")
        return original(path, value)
    def failure(path): raise RuntimeError("original failure")
    monkeypatch.setattr(module, "_write_json_exclusive", write)
    monkeypatch.setattr(module, "_load_cached_model", failure)
    with pytest.raises(RuntimeError, match="original"):
        module.run_comparison(**options)
    assert (options["output_dir"] / "pending.json").exists()


def test_cli_does_not_accept_threshold_download_or_generation_switch(monkeypatch):
    module = api()
    monkeypatch.setattr(sys, "argv", ["corpus_semantic_expanded", "--threshold", "0"])
    with pytest.raises(SystemExit) as error: module.main()
    assert error.value.code == 2


@pytest.mark.parametrize("change", ["chunk_count", "page_count", "source_sha", "order", "report_period"])
def test_configured_corpus_identities_are_checked_before_model(tmp_path, monkeypatch, change):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    cfg = json.loads(options["experiment"].read_bytes())
    if change == "chunk_count": cfg["chunk_count"] += 1
    if change == "page_count": cfg["documents"][0]["page_count"] += 1
    if change == "source_sha": cfg["documents"][0]["pdf_sha256"] = "1" * 64
    if change == "order": cfg["documents"][0], cfg["documents"][1] = cfg["documents"][1], cfg["documents"][0]
    if change == "report_period": cfg["documents"][4]["report_fiscal_year"] += 1
    json_write(options["experiment"], cfg)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("different corpus reached model"))
    with pytest.raises(ValueError, match="corpus|count|source|period"):
        module.run_comparison(**options)
    assert not options["output_dir"].exists()


def test_registered_scope_with_no_actual_chunks_refuses_before_model(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    original = module.load_verified_corpus
    def corpus(**kwargs):
        result = original(**kwargs)
        chunks = tuple(replace(chunk, document_id="nvidia-fy2021-annual")
                       if chunk.document_id == "nvidia-fy2025-q2" else chunk for chunk in result.chunks)
        return replace(result, chunks=chunks)
    monkeypatch.setattr(module, "load_verified_corpus", corpus)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("empty scope reached model"))
    with pytest.raises(ValueError, match="scope|eligible"):
        module.run_comparison(**options)
    assert not options["output_dir"].exists()


def test_input_mutation_when_pending_closes_prevents_model_creation(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    original = module._write_json_exclusive
    def changed(path, value):
        result = original(path, value)
        if path.name == "pending.json":
            options["cases"].write_bytes(options["cases"].read_bytes() + b"\n")
        return result
    monkeypatch.setattr(module, "_write_json_exclusive", changed)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("pending close mutation reached model"))
    with pytest.raises(ValueError, match="changed|closure"):
        module.run_comparison(**options)
    failed(options)


def test_two_encoder_calls_cannot_be_logged_as_one(tmp_path, monkeypatch):
    module, options, model, _, _ = fixture(tmp_path, monkeypatch)
    original = module.DenseRetriever
    class Duplicate(original):
        def inspect(self, query, **kwargs):
            first = super().inspect(query, **kwargs)
            super().inspect(query, **kwargs)
            return first
    monkeypatch.setattr(module, "DenseRetriever", Duplicate)
    with pytest.raises(ValueError, match="encoder call"):
        module.run_comparison(**options)
    failed(options)
    assert len(model.calls) == 1


def test_snapshot_file_link_escape_never_reaches_model(tmp_path, monkeypatch):
    module, options, _, snapshot, _ = fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside-weight"
    outside.write_bytes(b"unbound outside weight")
    weight = snapshot / "model.safetensors"
    weight.unlink()
    try:
        weight.symlink_to(outside)
    except OSError as error:
        pytest.skip("host symlink privilege unavailable: " + type(error).__name__)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("escaping cache reached model"))
    with pytest.raises(ValueError, match="escape"):
        module.run_comparison(**options)
    assert not options["output_dir"].exists()


def test_vector_receipt_cannot_reassign_the_installed_wheel(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    path = options["vector_dir"] / "vector-build-receipt.json"
    receipt = json.loads(path.read_bytes())
    receipt["wheel"]["wheel_sha256"] = "1" * 64
    json_write(path, receipt)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("reassigned vector wheel reached model"))
    with pytest.raises(ValueError, match="wheel|identity"):
        module.run_comparison(**options)
    assert not options["output_dir"].exists()


def test_no_missing_source_replay_can_be_substituted_by_a_case_boolean(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    def refusal(*args, **kwargs): raise ValueError("source replay receipt does not verify")
    monkeypatch.setattr(module, "load_source_replay", refusal)
    monkeypatch.setattr(module, "_load_cached_model", lambda p: pytest.fail("source replay refusal reached model"))
    with pytest.raises(ValueError, match="source replay"):
        module.run_comparison(**options)
    assert not options["output_dir"].exists()


@pytest.mark.parametrize("mutation", ["no-op", "failure-marker", "body", "input"])
def test_changes_during_pending_unlink_never_return_completed(tmp_path, monkeypatch, mutation):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    original = Path.unlink
    def changed(path, *args, **kwargs):
        if path == options["output_dir"] / "pending.json":
            if mutation == "no-op": return None
            if mutation == "failure-marker":
                (options["output_dir"] / "failure.json").write_bytes(b"concurrent failure")
            if mutation == "body":
                raw = options["output_dir"] / "queries-raw.npy"
                raw.write_bytes(raw.read_bytes() + b"concurrent body mutation")
            if mutation == "input":
                options["cases"].write_bytes(options["cases"].read_bytes() + b"\n")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", changed)
    with pytest.raises(ValueError, match="guard|completion|output|closure|changed"):
        module.run_comparison(**options)
    directory = options["output_dir"]
    assert (directory / "pending.json").exists() or (directory / "failure.json").exists()


def test_post_unlink_failure_keeps_guard_when_diagnostic_writer_also_fails(tmp_path, monkeypatch):
    module, options, _, _, _ = fixture(tmp_path, monkeypatch)
    unlink = Path.unlink
    def changed(path, *args, **kwargs):
        if path == options["output_dir"] / "pending.json":
            raw = options["output_dir"] / "queries-raw.npy"
            raw.write_bytes(raw.read_bytes() + b"after-close corruption")
        return unlink(path, *args, **kwargs)
    write = module._write_json_exclusive
    def failed_diagnostic(path, value):
        if path.name == "failure.json": raise OSError("diagnostic write also failed")
        return write(path, value)
    monkeypatch.setattr(Path, "unlink", changed)
    monkeypatch.setattr(module, "_write_json_exclusive", failed_diagnostic)
    with pytest.raises(ValueError, match="output|digest|changed"):
        module.run_comparison(**options)
    assert (options["output_dir"] / "pending.json").exists()
