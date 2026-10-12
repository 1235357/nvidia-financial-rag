"""Synthetic vector boundary checks; no actual BGE or financial acceptance."""
from __future__ import annotations

from hashlib import sha256
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

from test_corpus_recipe import Artifact, digest, write_json


def library():
    try:
        return importlib.import_module("nvidia_rag.corpus_embeddings")
    except ModuleNotFoundError:
        pytest.fail("separate corpus vector interface is missing")


class UnitModel:
    def __init__(self, matrix=None, action=None):
        self.matrix, self.action, self.calls = matrix, action, []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        if self.action:
            self.action()
        if self.matrix is not None:
            return self.matrix
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        matrix[:, 0] = 1
        return matrix


@pytest.fixture(autouse=True)
def no_semantic_dependency_versions(monkeypatch):
    # Existing artifact construction queries metadata; do not import/run models.
    from nvidia_rag import embedding_store
    monkeypatch.setattr(embedding_store, "_build_runtime", lambda: {
        "python": "3.13.5", "numpy": np.__version__, "sentence-transformers": "synthetic",
        "torch": "synthetic", "transformers": "synthetic"})


def build(a, vector, model=None, **kwargs):
    return library().build_local_bge_vectors(artifact_dir=a.directory,
        source_registry=a.registry, vector_dir=vector, model=model or UnitModel(), **kwargs)


def load(a, vector):
    return library().load_local_bge_vectors(artifact_dir=a.directory,
        source_registry=a.registry, vector_dir=vector)


def closure(a):
    return {path: path.read_bytes() for path in [a.registry,
        *(a.directory / name for name in ("manifest.json", "chunks.jsonl", "build-receipt.json"))]}


def test_success_is_separate_float32_384_and_corpus_is_unchanged(tmp_path):
    a, model = Artifact(tmp_path, two=True), UnitModel()
    before = closure(a)
    vector = tmp_path / "new-vectors"
    receipt = build(a, vector, model)
    assert closure(a) == before
    assert set(path.name for path in vector.iterdir()) == {
        "bge-small-en-v1.5.npy", "embedding-manifest.json", "vector-build-receipt.json"}
    matrix = load(a, vector)
    assert matrix.shape == (2, 384) and matrix.dtype == np.dtype("float32")
    assert not matrix.flags.writeable
    assert receipt["status"] == "complete" and receipt["post_build_verified"] is True
    assert receipt["wheel"] is None and receipt["generation_runs"] == 0
    assert receipt["model"]["execution_origin"] == "operator-injected"
    assert receipt["model"]["device"] is None
    assert receipt["human_acceptance"] is False
    assert model.calls == [([record["text"] for record in a.records], {
        "normalize_embeddings": True, "batch_size": 32, "show_progress_bar": False})]


@pytest.mark.parametrize("existing", ["empty", "complete", "failed"])
def test_existing_output_precedes_inputs_model_and_builder(tmp_path, monkeypatch, existing):
    module = library()
    output = tmp_path / "existing"
    output.mkdir()
    if existing != "empty":
        (output / ("failure.json" if existing == "failed" else "vector-build-receipt.json")).write_text(existing)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    def trap(**kwargs):
        pytest.fail("embedding builder invoked before exclusive refusal")
    monkeypatch.setattr(module, "build_passage_embedding_artifact", trap)
    model = UnitModel(action=lambda: pytest.fail("model invoked before exclusive refusal"))
    with pytest.raises(ValueError, match="exist"):
        module.build_local_bge_vectors(artifact_dir=tmp_path / "absent",
            source_registry=tmp_path / "missing.json", vector_dir=output, model=model)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    assert not model.calls


@pytest.mark.parametrize("placement", ["equal", "inside", "contains", "registry", "registry-parent"])
def test_output_overlap_refused_without_model_or_corpus_change(tmp_path, placement):
    a = Artifact(tmp_path)
    if placement == "contains":
        nested = tmp_path / "container" / "corpus"
        nested.parent.mkdir()
        a.directory.rename(nested)
        a.directory = nested
        output = nested.parent
    else:
        output = {"equal": a.directory, "inside": a.directory / "vectors",
                  "registry": a.registry, "registry-parent": a.registry.parent}[placement]
    before = closure(a)
    model = UnitModel(action=lambda: pytest.fail("overlap reached model"))
    with pytest.raises(ValueError, match="exist|overlap|separate"):
        build(a, output, model)
    assert closure(a) == before and not model.calls


def test_incomplete_corpus_refused_before_reserving_output(tmp_path):
    a = Artifact(tmp_path)
    (a.directory / "build-receipt.json").unlink()
    vector = tmp_path / "vectors"
    model = UnitModel(action=lambda: pytest.fail("incomplete corpus reached model"))
    with pytest.raises((ValueError, OSError)):
        build(a, vector, model)
    assert not vector.exists() and not model.calls


@pytest.mark.parametrize("matrix", [np.ones((0, 384)), np.ones((1, 383)),
    np.full((1, 384), np.nan), np.zeros((1, 384)), np.ones((1, 384))])
def test_invalid_model_output_retains_failed_directory_without_complete_receipt(tmp_path, matrix):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    before = closure(a)
    with pytest.raises(ValueError):
        build(a, vector, UnitModel(matrix=matrix))
    assert vector.is_dir() and (vector / "failure.json").is_file()
    assert not (vector / "vector-build-receipt.json").exists()
    assert closure(a) == before
    with pytest.raises(ValueError, match="failed|complete"):
        load(a, vector)


def update_vector_receipt(vector):
    receipt_path = vector / "vector-build-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["embeddings_sha256"] = digest(vector / "bge-small-en-v1.5.npy")
    receipt["embedding_manifest_sha256"] = digest(vector / "embedding-manifest.json")
    write_json(receipt_path, receipt)


def test_float64_saved_artifact_refused_even_after_hash_metadata_rewrite(tmp_path):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    build(a, vector)
    path = vector / "bge-small-en-v1.5.npy"
    matrix = np.load(path, allow_pickle=False).astype(np.float64)
    np.save(path, matrix, allow_pickle=False)
    manifest = json.loads((vector / "embedding-manifest.json").read_text())
    manifest.update(dtype="float64", embeddings_sha256=digest(path))
    write_json(vector / "embedding-manifest.json", manifest)
    update_vector_receipt(vector)
    receipt = json.loads((vector / "vector-build-receipt.json").read_text())
    receipt["matrix"]["dtype"] = "float64"
    write_json(vector / "vector-build-receipt.json", receipt)
    with pytest.raises(ValueError, match="float32|dtype"):
        load(a, vector)


@pytest.mark.parametrize("change", ["bytes", "order", "text"])
def test_stale_vectors_refused_for_changed_completed_corpus(tmp_path, change):
    a, vector = Artifact(tmp_path, two=True), tmp_path / "vectors"
    build(a, vector)
    if change == "bytes":
        with (a.directory / "chunks.jsonl").open("a") as handle:
            handle.write("\n")
    elif change == "order":
        a.sources.reverse()
        a.manifest["documents"].reverse()
        a.receipt["sources"].reverse()
        a.records.reverse()
        a.sync()
    else:
        a.records[0]["text"] += " altered evidence"
        a.sync()
    with pytest.raises(ValueError, match="hash|digest|association|closure"):
        load(a, vector)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("chunk_count", True),
    ("shape", [True, 384]), ("shape", [1, 384.0]), ("normalized", 1),
    ("model_name", "different"), ("model_revision", "0" * 40), ("dtype", "float64")])
def test_typed_embedding_manifest_refused_with_fresh_receipt_hash(tmp_path, field, value):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    build(a, vector)
    path = vector / "embedding-manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    write_json(path, manifest)
    update_vector_receipt(vector)
    with pytest.raises(ValueError):
        load(a, vector)


def test_changed_corpus_during_encoding_leaves_no_completion(tmp_path):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    path = a.directory / "chunks.jsonl"
    model = UnitModel(action=lambda: path.write_bytes(path.read_bytes() + b"\n"))
    with pytest.raises(ValueError, match="changed|closure|hash"):
        build(a, vector, model)
    assert not (vector / "vector-build-receipt.json").exists()
    assert (vector / "failure.json").exists()


def test_failure_receipt_has_no_raw_exception_or_paths(tmp_path):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    def fail():
        raise RuntimeError("PRIVATE_MODEL_TOKEN at " + str(tmp_path))
    with pytest.raises(RuntimeError):
        build(a, vector, UnitModel(action=fail))
    body = (vector / "failure.json").read_text()
    assert "PRIVATE_MODEL_TOKEN" not in body and str(tmp_path) not in body
    assert json.loads(body) == {"schema_version": 1, "status": "failed", "error_type": "RuntimeError"}


def test_cached_path_preflight_precedes_semantic_model_import(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ValueError, match="exist"):
        library().build_cached_local_bge_vectors(artifact_dir=tmp_path / "absent",
            source_registry=tmp_path / "missing.json", vector_dir=output,
            wheel=tmp_path / "missing.whl")


def test_cached_constructor_is_pinned_cpu_and_local_only_after_reservation(tmp_path, monkeypatch):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    module = library()
    monkeypatch.setattr(module, "_verify_installed_wheel", lambda path: {
        "wheel_basename": "nvidia_financial_rag-0.1.0-py3-none-any.whl", "wheel_bytes": 1,
        "wheel_sha256": "0" * 64, "distribution_version": "0.1.0", "module_sha256": module._module_hashes()})
    calls = []
    class CachedModel(UnitModel):
        def __init__(self, name, **kwargs):
            assert vector.is_dir()
            calls.append((name, kwargs))
            super().__init__()
    monkeypatch.setitem(sys.modules, "sentence_transformers", type("Module", (), {"SentenceTransformer": CachedModel}))
    receipt = module.build_cached_local_bge_vectors(artifact_dir=a.directory,
        source_registry=a.registry, vector_dir=vector, wheel=tmp_path / "fixture.whl")
    assert calls == [("BAAI/bge-small-en-v1.5", {
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "device": "cpu", "local_files_only": True})]
    assert receipt["model"]["execution_origin"] == "cached-local-only"
    assert receipt["model"]["local_files_only"] is True


def test_cli_requires_wheel_and_no_download_or_host_switch(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/build_corpus_embeddings.py"
    assert path.is_file(), "cached vector CLI is missing"
    spec = importlib.util.spec_from_file_location("vector_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(path), "--artifact-dir", "/a", "--source-registry", "/b", "--vector-dir", "/c"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    source = path.read_text()
    assert "--allow-model-download" not in source and "--host" not in source


def test_changed_corpus_inside_loader_is_refused_before_model(tmp_path, monkeypatch):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    module = library()
    original = module.load_verified_corpus
    def replaced(**kwargs):
        corpus = original(**kwargs)
        a.records[0]["text"] += " changed after read"
        a.sync()
        return corpus
    monkeypatch.setattr(module, "load_verified_corpus", replaced)
    model = UnitModel(action=lambda: pytest.fail("changed corpus reached model"))
    with pytest.raises(ValueError, match="closure changed"):
        build(a, vector, model)
    assert not vector.exists() and not model.calls


def test_changed_corpus_inside_vector_loader_is_refused(tmp_path, monkeypatch):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    build(a, vector)
    module = library()
    original = module.load_passage_embedding_artifact
    def replaced(**kwargs):
        matrix = original(**kwargs)
        a.records[0]["text"] += " changed after matrix read"
        a.sync()
        return matrix
    monkeypatch.setattr(module, "load_passage_embedding_artifact", replaced)
    with pytest.raises(ValueError, match="closure changed"):
        load(a, vector)


@pytest.mark.parametrize("invalid", ["duplicate", "nonfinite", "unknown"])
def test_strict_embedding_json_rejected_before_matrix_loader(tmp_path, monkeypatch, invalid):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    build(a, vector)
    path = vector / "embedding-manifest.json"
    value = json.loads(path.read_text())
    if invalid == "duplicate":
        path.write_text('{"dtype":"float32",' + json.dumps(value)[1:])
    else:
        value["extra"] = float("nan") if invalid == "nonfinite" else "not allowed"
        write_json(path, value)
    update_vector_receipt(vector)
    module = library()
    monkeypatch.setattr(module, "load_passage_embedding_artifact", lambda **kw:
        pytest.fail("malformed manifest reached matrix loader"))
    with pytest.raises(ValueError):
        load(a, vector)


@pytest.mark.parametrize("mutation", ["schema", "shape-bool", "count-bool", "human", "completion", "model-policy"])
def test_receipt_types_and_execution_boundary_remain_required(tmp_path, mutation):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    build(a, vector)
    path = vector / "vector-build-receipt.json"
    value = json.loads(path.read_text())
    if mutation == "schema":
        value["schema_version"] = True
    elif mutation == "shape-bool":
        value["matrix"]["shape"][0] = True
    elif mutation == "count-bool":
        value["generation_runs"] = False
    elif mutation == "human":
        value["human_acceptance"] = 0
    elif mutation == "completion":
        value["post_build_verified"] = 1
    else:
        value["model"]["device"] = "cpu"
    write_json(path, value)
    with pytest.raises(ValueError):
        load(a, vector)


def test_completion_receipt_missing_is_not_accepted(tmp_path):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    build(a, vector)
    (vector / "vector-build-receipt.json").unlink()
    with pytest.raises((OSError, ValueError)):
        load(a, vector)


def synthetic_wheel(tmp_path, monkeypatch, *, mutate=None):
    module = library()
    package = Path(module.__file__).parent
    monkeypatch.setattr(module.sysconfig, "get_paths", lambda: {"purelib": str(package.parent)})
    wheel = tmp_path / "nvidia_financial_rag-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path in package.glob("*.py"):
            if mutate == "missing" and path.name == "corpus_embeddings.py":
                continue
            body = path.read_bytes()
            if mutate == "different" and path.name == "runtime.py":
                body += b"# different body\n"
            archive.writestr("nvidia_rag/" + path.name, body)
        if mutate == "extra":
            archive.writestr("nvidia_rag/extra.py", "")
    return wheel


def test_supplied_wheel_is_bound_but_injected_model_retains_unknown_execution_policy(tmp_path, monkeypatch):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    wheel = synthetic_wheel(tmp_path, monkeypatch)
    receipt = build(a, vector, wheel=wheel)
    assert receipt["wheel"]["wheel_sha256"] == digest(wheel)
    assert receipt["wheel"]["wheel_bytes"] == wheel.stat().st_size
    package = Path(library().__file__).parent
    assert receipt["wheel"].get("module_sha256") == {
        path.name: digest(path) for path in package.glob("*.py")}
    assert receipt["model"]["execution_origin"] == "operator-injected"
    assert receipt["model"]["device"] is None and receipt["model"]["local_files_only"] is None
    load(a, vector)


@pytest.mark.parametrize("mutation", ["missing", "different", "extra"])
def test_named_wheel_module_identity_mismatch_precedes_reservation_and_model(tmp_path, monkeypatch, mutation):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    wheel = synthetic_wheel(tmp_path, monkeypatch, mutate=mutation)
    model = UnitModel(action=lambda: pytest.fail("mismatched wheel reached model"))
    with pytest.raises(ValueError, match="module|interface"):
        build(a, vector, model, wheel=wheel)
    assert not vector.exists() and not model.calls


def test_wrong_named_wheel_refused_without_model(tmp_path):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    wheel = tmp_path / "different.whl"
    wheel.write_bytes(b"not a wheel")
    with pytest.raises(ValueError, match="named wheel"):
        build(a, vector, wheel=wheel)
    assert not vector.exists()


def test_cli_rejects_existing_output_without_importing_semantic_package(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/build_corpus_embeddings.py"
    spec = importlib.util.spec_from_file_location("vector_cli_existing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(sys, "argv", [str(path), "--artifact-dir", str(tmp_path / "absent"),
        "--source-registry", str(tmp_path / "absent.json"), "--vector-dir", str(output),
        "--wheel", str(tmp_path / "absent.whl")])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


def test_completion_write_and_failure_write_error_retains_pending_refusal(tmp_path, monkeypatch):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    module = library()
    original = module._write
    calls = []
    def failing(path, value):
        if path.name == "vector-build-receipt.json":
            calls.append("completion")
            original(path, value)
            raise OSError("completion close failure")
        if path.name == "failure.json":
            calls.append("diagnostic")
            raise OSError("failure marker write failure")
        return original(path, value)
    monkeypatch.setattr(module, "_write", failing)
    with pytest.raises(OSError, match="completion"):
        build(a, vector)
    assert calls == ["completion", "diagnostic"]
    assert json.loads((vector / "vector-build-receipt.json").read_bytes())["status"] == "complete"
    assert (vector / "pending.json").is_file() and not (vector / "failure.json").exists()
    with pytest.raises(ValueError, match="pending|complete"):
        load(a, vector)


def test_pending_marker_is_present_before_model_and_removed_after_closed_completion(tmp_path):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    observed = []
    model = UnitModel(action=lambda: observed.append((vector / "pending.json").is_file()))
    build(a, vector, model)
    assert observed == [True]
    assert not (vector / "pending.json").exists()
    load(a, vector)


def test_pending_unlink_failure_remains_unaccepted(tmp_path, monkeypatch):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    original = Path.unlink
    def failing(path, *args, **kwargs):
        if path.name == "pending.json" and path.parent == vector:
            raise OSError("pending unlink failure")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", failing)
    module = library()
    original_write = module._write
    def failed_diagnostic(path, value):
        if path.name == "failure.json":
            raise OSError("diagnostic unavailable")
        return original_write(path, value)
    monkeypatch.setattr(module, "_write", failed_diagnostic)
    with pytest.raises(OSError, match="pending"):
        build(a, vector)
    assert (vector / "pending.json").is_file()
    assert not (vector / "failure.json").exists()
    with pytest.raises(ValueError, match="pending|failed"):
        load(a, vector)


def test_pending_created_during_load_is_refused_after_closure_read(tmp_path, monkeypatch):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    build(a, vector)
    module = library()
    original = module.load_passage_embedding_artifact
    def pending(**kwargs):
        matrix = original(**kwargs)
        (vector / "pending.json").write_text("pending")
        return matrix
    monkeypatch.setattr(module, "load_passage_embedding_artifact", pending)
    with pytest.raises(ValueError, match="pending|completion"):
        load(a, vector)


def test_pending_creation_close_failure_precedes_vector_model_work(tmp_path, monkeypatch):
    from contextlib import contextmanager
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    original = Path.open
    @contextmanager
    def failed_close(path, *args, **kwargs):
        with original(path, *args, **kwargs) as handle:
            yield handle
        if path == vector / "pending.json":
            raise OSError("pending close failure")
    monkeypatch.setattr(Path, "open", failed_close)
    model = UnitModel(action=lambda: pytest.fail("model before pending closed"))
    with pytest.raises(OSError, match="pending"):
        build(a, vector, model)
    assert model.calls == [] and not (vector / "bge-small-en-v1.5.npy").exists()
    assert (vector / "pending.json").exists()


@pytest.mark.parametrize("kind", ["file", "directory", "dangling"])
def test_pending_entries_refuse_complete_vectors_before_loading(tmp_path, monkeypatch, kind):
    a, vector = Artifact(tmp_path), tmp_path / "vectors"
    build(a, vector)
    marker = vector / "pending.json"
    if kind == "file":
        marker.write_bytes(b"pending")
    elif kind == "directory":
        marker.mkdir()
    else:
        try:
            marker.symlink_to(tmp_path / "absent-marker-body")
        except OSError as error:
            pytest.skip("host symlink privilege unavailable: " + type(error).__name__)
    monkeypatch.setattr(library(), "_load_corpus", lambda *args: pytest.fail("pending reached loader"))
    with pytest.raises(ValueError, match="pending"):
        load(a, vector)
