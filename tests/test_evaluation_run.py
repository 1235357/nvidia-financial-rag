"""Executable retrieval boundary; fixtures and injected encoders are not native runs."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys

import pytest

from test_corpus_recipe import Artifact


def library():
    try:
        return importlib.import_module("nvidia_rag.evaluation_run")
    except ModuleNotFoundError:
        pytest.fail("source-bound evaluation runner interface is missing")


def rows(query="  Revenue?  ", *, acceptance=False):
    return [{"question_id": "q1", "query": query, "answerable": True,
             "evidence_pages": [1], "evidence_document_ids": ["report"],
             "evaluation_usage": {"purpose": "acceptance_candidate" if acceptance else "development",
                 "development_exposed": not acceptance, "frozen_on": "2026-10-14",
                 "registration_note": "Synthetic boundary fixture"}}]


def case(tmp_path, **changes):
    artifact = Artifact(tmp_path, two=True)
    path = tmp_path / "questions.json"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows()), encoding="utf-8")
    options = dict(artifact_dir=artifact.directory, source_registry=artifact.registry,
        questions=path, output_dir=tmp_path / "run", mode="development", backend="lexical",
        top_k=2, threshold=0.78)
    options.update(changes)
    return artifact, options


def test_lexical_complete_run_retains_exact_queries_candidates_and_input_bodies(tmp_path):
    module = library()
    artifact, options = case(tmp_path, document_ids=frozenset({"report"}))
    receipt = module.run_evaluation(**options)
    loaded = module.load_completed_run(options["output_dir"])
    assert receipt == loaded["receipt"]
    assert receipt["status"] == "complete" and receipt["generation_runs"] == 0
    assert receipt["financial_correctness"] == "unestablished"
    assert not (options["output_dir"] / "pending.json").exists()
    assert loaded["freeze"]["questions"][0]["query"] == "  Revenue?  "
    call = loaded["results"]["calls"][0]
    assert call["query"] == "  Revenue?  "
    assert call["query_sha256"] == sha256(call["query"].encode()).hexdigest()
    assert len(call["candidates"]) == 1
    assert set(call["candidates"][0]["chunk"]) == {
        "chunk_id", "document_id", "page_number", "text", "amount_bbox", "metric_bbox", "period_bbox"}
    assert call["candidates"][0]["chunk"]["document_id"] == "report"
    identity = loaded["freeze"]["inputs"]["questions"]
    assert (options["output_dir"] / identity["snapshot"]).read_bytes() == options["questions"].read_bytes()
    assert loaded["results"]["report"]["records"][0]["review_snapshot"]["generation_status"] == "not_run"


@pytest.mark.parametrize("changes", [
    {"mode": "other"}, {"backend": "other"}, {"top_k": True}, {"top_k": 0},
    {"threshold": True}, {"threshold": float("nan")}, {"threshold": 1.1},
    {"routing_policy": "annual"}, {"routing_policy": "source-bound-period"},
    {"vector_dir": Path("unused")}, {"model_cache_dir": Path("unused")},
    {"statement_source_dir": Path("unused")}, {"model_factory": lambda: None},
    {"document_ids": {"report"}}, {"document_ids": frozenset({"missing"})},
])
def test_unsupported_options_refused_before_constructor(tmp_path, monkeypatch, changes):
    module = library()
    _, options = case(tmp_path, **changes)
    monkeypatch.setattr("nvidia_rag.retrieval.TfidfRetriever", lambda *a, **k: pytest.fail("constructor ran"))
    with pytest.raises(ValueError):
        module.run_evaluation(**options)
    assert not options["output_dir"].exists()


def test_acceptance_overlap_rejected_before_constructor(tmp_path, monkeypatch):
    module = library()
    _, options = case(tmp_path, mode="acceptance-candidate")
    options["questions"].write_text("".join(json.dumps(row) + "\n" for row in rows(acceptance=True)))
    reference = tmp_path / "reference.json"
    reference.write_text("".join(json.dumps(row) + "\n" for row in rows()))
    monkeypatch.setattr("nvidia_rag.retrieval.TfidfRetriever", lambda *a, **k: pytest.fail("constructor ran"))
    with pytest.raises(ValueError, match="overlap"):
        module.run_evaluation(**options, reference=reference)


@pytest.mark.parametrize("failure", ["constructor", "query", "foreign", "altered", "duplicate", "nan", "count", "closure"])
def test_failure_keeps_reserved_pending_attempt_evidence_and_never_completes(tmp_path, monkeypatch, failure):
    from nvidia_rag.core import Chunk, ScoredChunk
    module = library()
    artifact, options = case(tmp_path, document_ids=frozenset({"report"}))
    chunk = artifact.load().chunks[0]
    class Backend:
        def __init__(self, chunks):
            assert (options["output_dir"] / "pending.json").exists()
            assert (options["output_dir"] / "freeze.json").exists()
            if failure == "constructor":
                raise RuntimeError("private constructor detail")
        def retrieve_from_documents(self, query, *, top_k, document_ids):
            assert query == "  Revenue?  " and document_ids == {"report"}
            if failure == "query":
                raise RuntimeError("private query detail")
            if failure == "foreign":
                return [ScoredChunk(Chunk("foreign", "other", 1, "x"), 0.5)]
            if failure == "altered":
                return [ScoredChunk(replace(chunk, text="changed"), 0.5)]
            if failure == "duplicate":
                return [ScoredChunk(chunk, 0.5)] * 2
            if failure == "nan":
                return [ScoredChunk(chunk, float("nan"))]
            if failure == "count":
                return [ScoredChunk(chunk, 0.5)] * 3
            if failure == "closure":
                options["questions"].write_bytes(b"[]")
            return [ScoredChunk(chunk, 0.5)]
    monkeypatch.setattr("nvidia_rag.retrieval.TfidfRetriever", Backend)
    with pytest.raises((ValueError, RuntimeError)):
        module.run_evaluation(**options)
    output = options["output_dir"]
    assert (output / "pending.json").exists() and (output / "failure.json").exists()
    assert (output / "attempts.json").exists()
    assert not (output / "receipt.json").exists() and not (output / "results.json").exists()
    assert b"private" not in (output / "failure.json").read_bytes()
    with pytest.raises(ValueError):
        module.load_completed_run(output)


@pytest.mark.parametrize("change", ["results", "pending", "failure", "unknown", "report", "candidate", "snapshot"])
def test_loader_refuses_mutated_or_inconsistent_graph_even_rehashed(tmp_path, change):
    module = library()
    _, options = case(tmp_path)
    module.run_evaluation(**options)
    output = options["output_dir"]
    if change in {"pending", "failure"}:
        (output / (change + ".json")).write_bytes(b"{}")
    elif change == "snapshot":
        freeze = json.loads((output / "freeze.json").read_bytes())
        (output / freeze["inputs"]["questions"]["snapshot"]).write_bytes(b"[]")
    else:
        path = output / "results.json"
        result = json.loads(path.read_bytes())
        if change == "unknown":
            result["extra"] = 1
        elif change == "report":
            result["report"]["records"][0]["query"] = "changed"
        elif change == "candidate":
            result["calls"][0]["candidates"][0]["chunk"]["text"] = "changed"
        else:
            result["freeze_sha256"] = "0" * 64
        from nvidia_rag.evaluation_inputs import canonical_json
        path.write_bytes(canonical_json(result))
        receipt_path = output / "receipt.json"
        receipt = json.loads(receipt_path.read_bytes())
        receipt["files"]["results.json"] = {"bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest()}
        receipt_path.write_bytes(canonical_json(receipt))
    with pytest.raises(ValueError):
        module.load_completed_run(output)


def test_existing_and_overlapping_output_refused(tmp_path):
    module = library()
    artifact, options = case(tmp_path)
    options["output_dir"] = artifact.directory / "new"
    with pytest.raises(ValueError, match="overlap"):
        module.run_evaluation(**options)
    options["output_dir"] = artifact.directory
    with pytest.raises(ValueError):
        module.run_evaluation(**options)


def test_package_overlap_refused_before_any_output_or_backend_work():
    module = library()
    package = Path(module.__file__).resolve().parent
    with pytest.raises(ValueError, match='overlap'):
        module._output_path(package / 'never-created-evaluation-output', [])


def test_complete_output_budget_refuses_without_truncation(tmp_path, monkeypatch):
    module = library()
    _, options = case(tmp_path)
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 100)
    with pytest.raises(ValueError, match="budget"):
        module.run_evaluation(**options)
    assert not (options["output_dir"] / "receipt.json").exists()


def test_module_import_is_lazy_for_semantic_packages():
    library()
    assert "sentence_transformers" not in sys.modules


def test_recording_wrapper_preserves_rectangles_and_rejects_noncosine_scores(tmp_path):
    from nvidia_rag.core import Chunk, ScoredChunk
    from nvidia_rag.evaluation import EvaluationQuestion
    module = library()
    chunk = Chunk("a", "report", 1, "evidence", amount_bbox=(1, 2, 3, 4))
    question = EvaluationQuestion("q", "  exact text  ", False, ())
    class Backend:
        def retrieve(self, query, *, top_k):
            assert query == question.query and top_k == 1
            return [ScoredChunk(chunk, 1.0 + 1e-13)]
    attempts = []
    recorder = module.CapturedRetriever(Backend(), questions=(question,), chunks=(chunk,),
        document_ids=frozenset({"report"}), top_k=1, attempts=attempts)
    assert recorder.retrieve(question.query, top_k=1)[0].chunk.amount_bbox == (1, 2, 3, 4)
    assert recorder.calls[0]["candidates"][0]["chunk"]["amount_bbox"] == (1, 2, 3, 4)
    with pytest.raises(ValueError):
        recorder.retrieve(question.query, top_k=1)


def test_pinned_snapshot_inventory_rejects_missing_bodies_and_tracks_extra_files(tmp_path):
    module = library()
    cache = tmp_path / "cache"
    snapshot = cache / "models--BAAI--bge-small-en-v1.5" / "snapshots" / "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    snapshot.mkdir(parents=True)
    with pytest.raises(ValueError):
        module.model_snapshot_identity(cache)
    for name in module.REQUIRED_MODEL_FILES:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"prepared body")
    initial = module.model_snapshot_identity(cache)
    (snapshot / "extra.json").write_bytes(b"extra")
    assert module.model_snapshot_identity(cache) != initial


def test_cached_constructor_uses_exact_pinned_cache_cpu_local_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = library()
    calls = []
    def constructor(*args, **kwargs):
        calls.append((args, kwargs))
        return object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    module.load_cached_model(tmp_path)
    assert calls == [(("BAAI/bge-small-en-v1.5",), {
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "device": "cpu",
        "local_files_only": True, "cache_folder": str(tmp_path)})]


def test_empty_questions_refused_before_constructor(tmp_path, monkeypatch):
    module = library()
    _, options = case(tmp_path)
    options["questions"].write_bytes(b"")
    monkeypatch.setattr("nvidia_rag.retrieval.TfidfRetriever", lambda *a, **k: pytest.fail("constructor ran"))
    with pytest.raises(ValueError, match="nonempty"):
        module.run_evaluation(**options)
    assert not options["output_dir"].exists()


def test_empty_retrieval_keeps_question_and_review_denominators(tmp_path, monkeypatch):
    module = library()
    _, options = case(tmp_path)
    class Empty:
        def __init__(self, chunks):
            pass
        def retrieve_from_documents(self, query, **kwargs):
            return []
    monkeypatch.setattr("nvidia_rag.retrieval.TfidfRetriever", Empty)
    module.run_evaluation(**options)
    loaded = module.load_completed_run(options["output_dir"])
    assert loaded["results"]["report"]["summary"]["question_count"] == 1
    assert loaded["packet"]["rows"][0]["candidates"] == []
    assert loaded["packet"]["rows"][0]["judgments"] == []


def test_call_attempt_is_durable_before_backend_query(tmp_path, monkeypatch):
    module = library()
    artifact, options = case(tmp_path)
    chunk = artifact.load().chunks[0]
    from nvidia_rag.core import ScoredChunk
    class Backend:
        def __init__(self, chunks):
            pass
        def retrieve_from_documents(self, query, **kwargs):
            progress = json.loads((options["output_dir"] / "attempts.json").read_bytes())
            assert progress["attempts"][0]["status"] == "started"
            return [ScoredChunk(chunk, 0.5)]
    monkeypatch.setattr("nvidia_rag.retrieval.TfidfRetriever", Backend)
    module.run_evaluation(**options)
    progress = json.loads((options["output_dir"] / "attempts.json").read_bytes())
    assert progress["attempts"][0]["status"] == "complete"
    assert progress["attempts"][0]["candidates"][0]["chunk"]["text"] == chunk.text


@pytest.mark.parametrize("change", ["origin", "extra-input", "missing-package", "extra-file"])
def test_loader_rejects_unknown_inventory_and_inconsistent_backend_metadata(tmp_path, change):
    module = library()
    _, options = case(tmp_path)
    module.run_evaluation(**options)
    output = options["output_dir"]
    from nvidia_rag.evaluation_inputs import canonical_json
    freeze = json.loads((output / "freeze.json").read_bytes())
    if change == "origin":
        freeze["config"]["model_execution_origin"] = "cached-local-only"
        body = canonical_json(freeze["config"])
        (output / "inputs/config").write_bytes(body)
        freeze["inputs"]["config"].update(bytes=len(body), sha256=sha256(body).hexdigest())
    elif change == "extra-input":
        freeze["inputs"]["unrecognized"] = dict(freeze["inputs"]["questions"], snapshot="inputs/unrecognized")
        (output / "inputs/unrecognized").write_bytes(options["questions"].read_bytes())
    elif change == "missing-package":
        del freeze["inputs"]["package/evaluation_run.py"]
    else:
        (output / "unrecognized.json").write_bytes(b"{}")
    freeze_body = canonical_json(freeze)
    (output / "freeze.json").write_bytes(freeze_body)
    result = json.loads((output / "results.json").read_bytes())
    result["freeze_sha256"] = sha256(freeze_body).hexdigest()
    result_body = canonical_json(result)
    (output / "results.json").write_bytes(result_body)
    from nvidia_rag.evaluation_review import build_review_packet
    packet = build_review_packet(result, freeze_sha256=sha256(freeze_body).hexdigest(), results_sha256=sha256(result_body).hexdigest())
    (output / "review-packet.json").write_bytes(canonical_json(packet))
    receipt = json.loads((output / "receipt.json").read_bytes())
    receipt["files"] = {name: {"bytes": (output / name).stat().st_size,
        "sha256": sha256((output / name).read_bytes()).hexdigest()} for name in receipt["files"]}
    (output / "receipt.json").write_bytes(canonical_json(receipt))
    with pytest.raises(ValueError):
        module.load_completed_run(output)


def test_cached_bge_injection_retains_factory_conventions_and_model_inventory(tmp_path, monkeypatch):
    import numpy as np
    from nvidia_rag import corpus_embeddings, embedding_store
    from test_corpus_embeddings import UnitModel
    module = library()
    artifact, options = case(tmp_path, backend="cached-bge", document_ids=frozenset({"report"}))
    monkeypatch.setattr(embedding_store, "_build_runtime", lambda: {
        "python": "3.13.5", "numpy": np.__version__, "sentence-transformers": "synthetic",
        "torch": "synthetic", "transformers": "synthetic"})
    vectors = tmp_path / "vectors"
    corpus_embeddings.build_local_bge_vectors(artifact_dir=artifact.directory, source_registry=artifact.registry,
        vector_dir=vectors, model=UnitModel())
    cache = tmp_path / "cache"
    snapshot = cache / "models--BAAI--bge-small-en-v1.5" / "snapshots" / "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    for name in module.REQUIRED_MODEL_FILES:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic prepared body")
    model = UnitModel()
    def factory():
        assert (options["output_dir"] / "pending.json").exists()
        assert (options["output_dir"] / "freeze.json").exists()
        return model
    module.run_evaluation(**options, vector_dir=vectors, model_cache_dir=cache, model_factory=factory)
    loaded = module.load_completed_run(options["output_dir"])
    assert model.calls == [(["Represent this sentence for searching relevant passages:   Revenue?  "], {
        "normalize_embeddings": True, "batch_size": 32, "show_progress_bar": False})]
    assert loaded["freeze"]["config"]["model_execution_origin"] == "operator-injected"
    assert loaded["freeze"]["inputs"]["model/model.safetensors"]["snapshot"] is None
    assert loaded["results"]["calls"][0]["candidates"][0]["chunk"]["document_id"] == "report"


def test_source_bound_backend_routes_native_quarter_and_verifies_retained_pdf_graph(tmp_path, monkeypatch):
    import runpy
    import numpy as np
    from nvidia_rag import corpus_embeddings, embedding_store
    from nvidia_rag.evaluation_inputs import canonical_json
    from test_corpus_embeddings import UnitModel
    module = library()
    fixture_builder = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'acceptance/serving_startup.py'))
    fixture = fixture_builder['create_fixture'](tmp_path / 'native')
    questions = rows('FY2025 Q2 net income for three months ended July 28, 2024')
    questions[0]['evidence_document_ids'] = ['synthetic-quarter']
    questions[0]['evidence_pages'] = [2]
    question_file = tmp_path / 'questions.jsonl'
    question_file.write_bytes(b''.join(canonical_json(row) for row in questions))
    monkeypatch.setattr(embedding_store, '_build_runtime', lambda: {
        'python': '3.13.5', 'numpy': np.__version__, 'sentence-transformers': 'synthetic',
        'torch': 'synthetic', 'transformers': 'synthetic'})
    vector = tmp_path / 'vectors'
    corpus_embeddings.build_local_bge_vectors(artifact_dir=fixture['artifact_dir'],
        source_registry=fixture['source_registry'], vector_dir=vector, model=UnitModel())
    cache = tmp_path / 'cache'
    snapshot = cache / 'models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
    for name in module.REQUIRED_MODEL_FILES:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic cache body')
    model = UnitModel()
    run_dir = tmp_path / 'run'
    module.run_evaluation(artifact_dir=fixture['artifact_dir'], source_registry=fixture['source_registry'],
        questions=question_file, output_dir=run_dir, mode='development', backend='cached-bge', top_k=2,
        threshold=.78, vector_dir=vector, model_cache_dir=cache, model_factory=lambda: model,
        routing_policy='source-bound-period', statement_source_dir=fixture['source_dir'],
        statement_fiscal_declarations=fixture['declarations'])
    loaded = module.load_completed_run(run_dir)
    assert [r['chunk']['document_id'] for r in loaded['results']['calls'][0]['candidates']] == ['synthetic-quarter']
    assert len(model.calls) == 1
    retained_pdf = run_dir / 'inputs/statement/quarter.pdf'
    retained_pdf.write_bytes(b'changed PDF')
    with pytest.raises(ValueError):
        module.load_completed_run(run_dir)


def test_acceptance_candidate_keeps_reference_and_does_not_certify_independence(tmp_path):
    module = library()
    _, options = case(tmp_path, mode='acceptance-candidate')
    options['questions'].write_text(''.join(json.dumps(row) + '\n' for row in rows(acceptance=True)))
    reference = tmp_path / 'reference.jsonl'
    reference.write_text(''.join(json.dumps(row) + '\n' for row in rows('Different development question')))
    module.run_evaluation(**options, reference=reference)
    loaded = module.load_completed_run(options['output_dir'])
    assert loaded['freeze']['reference'][0]['query'] == 'Different development question'
    assert loaded['receipt']['financial_correctness'] == 'unestablished'


def test_retained_run_accepts_other_os_absolute_provenance_paths(tmp_path):
    from nvidia_rag.evaluation_inputs import canonical_json
    from nvidia_rag.evaluation_review import build_review_packet
    module = library()
    _, options = case(tmp_path)
    module.run_evaluation(**options)
    run = options['output_dir']
    freeze = json.loads((run / 'freeze.json').read_bytes())
    for name, row in freeze['inputs'].items():
        if row['path'] is not None:
            row['path'] = '/original-host/' + name if sys.platform == 'win32' else 'C:\\original-host\\' + name.replace('/', '\\')
    freeze_body = canonical_json(freeze)
    (run / 'freeze.json').write_bytes(freeze_body)
    results = json.loads((run / 'results.json').read_bytes())
    results['freeze_sha256'] = sha256(freeze_body).hexdigest()
    results_body = canonical_json(results)
    (run / 'results.json').write_bytes(results_body)
    packet = build_review_packet(results, freeze_sha256=sha256(freeze_body).hexdigest(), results_sha256=sha256(results_body).hexdigest())
    (run / 'review-packet.json').write_bytes(canonical_json(packet))
    receipt = json.loads((run / 'receipt.json').read_bytes())
    receipt['files'] = {name: {'bytes': (run / name).stat().st_size,
        'sha256': sha256((run / name).read_bytes()).hexdigest()} for name in receipt['files']}
    (run / 'receipt.json').write_bytes(canonical_json(receipt))
    assert module.load_completed_run(run)['freeze'] == freeze
