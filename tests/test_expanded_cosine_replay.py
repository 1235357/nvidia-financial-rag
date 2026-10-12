"""Independent arithmetic and receipt tamper controls; no encoder executes."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "acceptance" / "expanded_cosine_replay.py"


def api():
    assert SCRIPT.exists(), "independent expanded cosine replay must be implemented"
    spec = importlib.util.spec_from_file_location("expanded_cosine_replay_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def h(text):
    return sha256(text.encode("utf-8")).hexdigest()


def fixture():
    """Known basis vectors make row, scope and gate errors observable."""
    chunks = tuple(SimpleNamespace(chunk_id=f"chunk-{i}", document_id="a" if i < 3 else "b",
                                   page_number=i + 1, text=f"source text {i}") for i in range(8))
    passages = np.zeros((8, 384), dtype=np.float32)
    passages[:, 0] = [1, .9, .8, .7, .6, .5, .4, .3]
    passages[:, 1] = np.sqrt(1 - passages[:, 0] ** 2)
    raw = np.zeros((2, 384), dtype=np.float32)
    raw[0, 0] = 2
    raw[1, 1] = 3
    normalized = raw.astype(np.float64) / np.linalg.norm(raw.astype(np.float64), axis=1, keepdims=True)
    scopes = [{"id": "all-reports", "document_ids": None}, {"id": "b-only", "document_ids": ("b",)}]
    cfg = {"model": {"instruction": "instruction: ", "dimension": 384},
           "top_k": 5, "primary_threshold": .78, "threshold_grid": (.78,),
           "chunk_count": len(chunks), "scopes": scopes}
    inputs = (SimpleNamespace(case_id="positive", scope_id="all-reports", query="  alpha\n", document_ids=None),
              SimpleNamespace(case_id="negative", scope_id="b-only", query="beta", document_ids=("b",)))
    cases = (SimpleNamespace(case_id="positive", origin={"original_record_sha256": None}, expected={"outcome": "evidence"}),
             SimpleNamespace(case_id="negative", origin={"original_record_sha256": None}, expected={"outcome": "refuse"}))
    bundle = SimpleNamespace(experiment=SimpleNamespace(data=cfg), cases=cases,
                             retrieval_inputs=lambda: inputs, human_gold_count=0)
    index = {"schema_version": 2, "rows": []}
    results = {"schema_version": 2, "rows": []}
    matrix = passages.astype(np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    for row, item in enumerate(inputs):
        scope = [i for i, chunk in enumerate(chunks) if item.document_ids is None or chunk.document_id in item.document_ids]
        scores = matrix[scope] @ normalized[row]
        order = np.argsort(scores)[::-1][:5]
        top = [{"global_chunk_index": scope[i], "chunk_id": chunks[scope[i]].chunk_id,
                "text_sha256": h(chunks[scope[i]].text), "score": float(scores[i])} for i in order]
        accepted = [entry for entry in top if entry["score"] >= .78]
        citations = [f"{chunks[entry['global_chunk_index']].document_id} p.{chunks[entry['global_chunk_index']].page_number}" for entry in accepted]
        context = "\n\n".join(f"[{citation}]\n{chunks[entry['global_chunk_index']].text}" for citation, entry in zip(citations, accepted))
        query = item.query.strip()
        index["rows"].append({"row": row, "case_id": item.case_id, "scope_id": item.scope_id,
                              "normalized_query_sha256": h(query), "instructed_query_sha256": h("instruction: " + query),
                              "original_record_sha256": None, "encoder_call_count": 1})
        results["rows"].append({"row": row, "case_id": item.case_id, "scope_id": item.scope_id,
                                "scope_document_ids": None if item.document_ids is None else list(item.document_ids),
                                "scoped_global_chunk_indices": scope, "considered_scores": scores.tolist(),
                                "top_k": top, "evidence_ids": [entry["chunk_id"] for entry in accepted],
                                "citations": citations, "can_answer": bool(accepted),
                                "message": f"Evidence retrieved for: {query}" if accepted else "I cannot answer from the provided reports because no evidence passed the retrieval gate.",
                                "context_sha256": h(context), "context_empty": not bool(context),
                                "threshold": .78, "top_k_limit": 5})
    return dict(bundle=bundle, chunks=chunks, passages=passages, raw_queries=raw,
                normalized_queries=normalized, query_index=index, results=results)


def test_independent_replay_reconstructs_scope_scores_gate_context_and_candidate_boundary():
    result = api().replay_rows(**fixture())
    assert result["arithmetic_passed"] is True
    assert result["human_gold_count"] == result["generation_runs"] == 0
    assert result["rows"][0]["status"] == "verified"
    assert result["rows"][1]["negative_candidate_accepted"] is True
    assert result["rows"][1]["negative_observation"] == "negative candidate accepted by the software gate"
    assert result["rows"][1]["scalar_dot_controls"] == 5


@pytest.mark.parametrize("mutation,match", [
    ("raw-mutated", "normalized query"), ("raw-reordered", "normalized query"),
    ("normalized-rounded", "normalized query"), ("passage-mutated", "considered scores"),
    ("wrong-instruction", "query index"), ("wrong-scope", "scope"),
    ("context-corrupt", "context"), ("citation-corrupt", "citation"),
    ("top-reordered", "Top-K"), ("score-rounded", "considered scores"),
    ("index-reordered", "query index"), ("label-injection", "result row fields"),
])
def test_rejects_mutation_and_wrong_scope_without_reencoding(mutation, match):
    data = fixture()
    if mutation == "raw-mutated":
        data["raw_queries"][0, 1] = .1
    elif mutation == "raw-reordered":
        data["raw_queries"] = data["raw_queries"][::-1].copy()
    elif mutation == "normalized-rounded":
        data["normalized_queries"][0, 0] -= 1e-7
    elif mutation == "passage-mutated":
        data["passages"][0, 1] = .3
    elif mutation == "wrong-instruction":
        data["query_index"]["rows"][0]["instructed_query_sha256"] = "0" * 64
    elif mutation == "wrong-scope":
        data["results"]["rows"][1]["scoped_global_chunk_indices"][0] = 0
    elif mutation == "context-corrupt":
        data["results"]["rows"][0]["context_sha256"] = "0" * 64
    elif mutation == "citation-corrupt":
        data["results"]["rows"][0]["citations"][0] = "b p.1"
    elif mutation == "top-reordered":
        data["results"]["rows"][0]["top_k"].reverse()
    elif mutation == "score-rounded":
        data["results"]["rows"][0]["considered_scores"][1] = .9
    elif mutation == "index-reordered":
        data["query_index"]["rows"].reverse()
    else:
        data["results"]["rows"][0]["expected_outcome"] = "evidence"
    with pytest.raises(ValueError, match=match):
        api().replay_rows(**data)


@pytest.mark.parametrize("field,kind", [("raw_queries", "zero"), ("raw_queries", "nan"),
                                       ("passages", "zero"), ("passages", "inf"),
                                       ("raw_queries", "dtype"), ("normalized_queries", "dtype"),
                                       ("raw_queries", "shape")])
def test_rejects_invalid_vector_shape_dtype_and_nonfinite_norm(field, kind):
    data = fixture()
    if kind == "zero":
        data[field][0] = 0
    elif kind == "nan":
        data[field][0, 0] = np.nan
    elif kind == "inf":
        data[field][0, 0] = np.inf
    elif kind == "dtype":
        data[field] = data[field].astype(np.float64 if field == "raw_queries" else np.float32)
    else:
        data[field] = data[field][:, :383]
    with pytest.raises(ValueError, match="vector|dtype|shape|norm|finite"):
        api().replay_rows(**data)


def test_exact_topk_boundary_tie_is_indeterminate_and_preserves_producer_order():
    data = fixture()
    # Five descending rows plus a sixth at exactly the same fifth score.
    data["passages"][5] = data["passages"][4]
    data = rebuild_receipt(data)
    receipt_order = [entry["global_chunk_index"] for entry in data["results"]["rows"][0]["top_k"]]
    result = api().replay_rows(**data)
    row = result["rows"][0]
    assert row["status"] == "boundary_indeterminate"
    assert row["top_k_boundary_indeterminate"] is True
    assert row["producer_top_k_indices"] == receipt_order


def rebuild_receipt(data):
    """Update numeric receipt after intentionally designing a new valid fixture."""
    matrix = data["passages"].astype(np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    for row, entry in enumerate(data["results"]["rows"]):
        scope = entry["scoped_global_chunk_indices"]
        scores = matrix[scope] @ data["normalized_queries"][row]
        entry["considered_scores"] = scores.tolist()
        entry["top_k"] = [{"global_chunk_index": scope[i], "chunk_id": data["chunks"][scope[i]].chunk_id,
                           "text_sha256": h(data["chunks"][scope[i]].text), "score": float(scores[i])}
                          for i in np.argsort(scores)[::-1][:5]]
        accepted = [item for item in entry["top_k"] if item["score"] >= .78]
        entry["evidence_ids"] = [item["chunk_id"] for item in accepted]
        entry["citations"] = [f"{data['chunks'][item['global_chunk_index']].document_id} p.{data['chunks'][item['global_chunk_index']].page_number}" for item in accepted]
        context = "\n\n".join(f"[{cite}]\n{data['chunks'][item['global_chunk_index']].text}" for cite, item in zip(entry["citations"], accepted))
        entry["context_sha256"] = h(context)
        entry["context_empty"] = not bool(context)
        entry["can_answer"] = bool(accepted)
    return data


def test_near_threshold_on_both_sides_is_explicit_and_cannot_certify_opposite_gate():
    data = fixture()
    # Exact integer norm sqrt(39^2+31^2+3^2+3^2)=50 yields 0.78.
    # Tiny representable f32 passage components straddle that actual gate.
    data["passages"][:2] = 0
    data["passages"][:2, 0] = 1
    data["passages"][:2, 1] = [1e-12, -1e-12]
    data["raw_queries"][0] = 0
    data["raw_queries"][0, :4] = [39, 31, 3, 3]
    q = data["raw_queries"][0].astype(np.float64)
    data["normalized_queries"][0] = q / np.linalg.norm(q)
    data = rebuild_receipt(data)
    scores = data["results"]["rows"][0]["considered_scores"]
    assert scores[0] > .78 > scores[1]
    row = api().replay_rows(**data)["rows"][0]
    assert row["status"] == "boundary_indeterminate"
    assert row["threshold_boundary_indeterminate"] is True
    assert row["near_threshold_local_indices"] == [0, 1]


def test_strict_receipt_json_rejects_duplicate_members_nonfinite_and_boolean_scores():
    module = api()
    for body in (b'{"rows":[],"rows":[]}', b'{"value":NaN}', b'{"value":1e999}'):
        with pytest.raises(ValueError, match="JSON|finite|duplicate"):
            module.strict_json(body)
    data = fixture()
    data["results"]["rows"][0]["considered_scores"][0] = True
    with pytest.raises(ValueError, match="scores"):
        module.replay_rows(**data)


def test_import_and_replay_have_no_retriever_service_or_model_dependencies():
    code = r'''
import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("sentence_transformers", "torch", "transformers", "nvidia_rag.retrieval", "nvidia_rag.service")):
            raise RuntimeError("forbidden dependency " + fullname)
sys.meta_path.insert(0, Block())
spec = importlib.util.spec_from_file_location("replay", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert callable(module.replay_rows)
'''
    result = subprocess.run([sys.executable, "-c", code, str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def completed_fixture(tmp_path):
    import test_expanded_cases as contracts
    data = fixture()
    cfg = contracts.experiment()
    cfg["model"]["instruction"] = "instruction: "
    docs = [deepcopy(cfg["documents"][0]), deepcopy(cfg["documents"][0])]
    docs[1].update(document_id="nvidia-fy2025-q2", report_fiscal_year=2025, report_fiscal_quarter=2)
    cfg.update(documents=docs, document_count=2, chunk_count=8,
               scopes=[{"id": "all-reports", "document_ids": None}, {"id": "b-only", "document_ids": [docs[1]["document_id"]]}])
    positive, negative = contracts.case(), contracts.case()
    positive.update(case_id="positive", query="  alpha\n", scope_ids=["all-reports"])
    negative.update(case_id="negative", query="beta", scope_ids=["b-only"], source_cells=[])
    negative["request"] = {name: None for name in negative["request"]}
    negative["request"].update(report_document_id=docs[1]["document_id"], report_fiscal_year=2025, report_fiscal_quarter=2)
    negative["expected"] = {"outcome": "refuse", "cell_ids": [], "negative_reason": "development candidate",
                             "rationale_binding_ids": ["inventory:1"]}
    cases_path, freeze_path, config_path = contracts.write_inputs(tmp_path, [positive, negative], config=cfg)
    corpus, vectors, comparison = [tmp_path / name for name in ("corpus", "vectors", "comparison")]
    for directory in (corpus, vectors, comparison):
        directory.mkdir()
    def write(path, value):
        body = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        path.write_bytes(body)
        return sha256(body).hexdigest()
    registry_rows = [{"document_id": doc["document_id"], "label": f"report {i}", "url": doc["issuer_url"],
                      "filename": f"report-{i}.pdf", "sha256": doc["pdf_sha256"],
                      "reporting_period": {"fiscal_year": doc["report_fiscal_year"], "quarter": doc["report_fiscal_quarter"],
                                           "end_date": "2026-01-01", "source_locator": "cover", "reviewed_by": "CodexAI"}}
                     for i, doc in enumerate(docs)]
    registry = tmp_path / "registry.json"
    cfg["input_sha256"]["registry"] = write(registry, {"schema_version": 1, "sources": registry_rows})
    manifest_docs = []
    for i, (reg, doc) in enumerate(zip(registry_rows, docs)):
        count = 3 if i == 0 else 5
        manifest_docs.append({"document_id": reg["document_id"], "document_label": reg["label"],
                              "source_url": reg["url"], "source_filename": reg["filename"], "source_sha256": reg["sha256"],
                              "reporting_period": reg["reporting_period"], "page_count": doc["page_count"],
                              "nonempty_page_count": count, "chunk_count": count})
    chunks = []
    for i, old in enumerate(data["chunks"]):
        document = docs[0 if i < 3 else 1]["document_id"]
        chunks.append(SimpleNamespace(chunk_id=f"{document}:p{i+1}:w:{i:016x}", document_id=document,
                                      page_number=i+1, text=old.text))
    data["chunks"] = tuple(chunks)
    body = b"".join((json.dumps(vars(chunk)) + "\n").encode() for chunk in chunks)
    (corpus / "chunks.jsonl").write_bytes(body)
    cfg["input_sha256"]["corpus_chunks"] = sha256(body).hexdigest()
    chunking = {"encoding_name": "cl100k_base", "max_tokens": 400, "overlap_tokens": 0, "strategy": "word_safe"}
    manifest = {"schema_version": 1, "chunks_sha256": cfg["input_sha256"]["corpus_chunks"], "chunking": chunking,
                "extraction_mode": "plain", "documents": manifest_docs,
                "document_count": 2, "page_count": 90, "nonempty_page_count": 8, "chunk_count": 8}
    cfg["input_sha256"]["corpus_manifest"] = write(corpus / "manifest.json", manifest)
    build = {"schema_version": 1, "status": "complete", "scope": "operator-local-corpus-build",
             "profile": {"name": "plain-word-safe-400-0-v1", "extraction_mode": "plain", **chunking},
             "source_registry_sha256": cfg["input_sha256"]["registry"], "manifest_sha256": cfg["input_sha256"]["corpus_manifest"],
             "chunks_sha256": cfg["input_sha256"]["corpus_chunks"], "runtime": {"python": "3", "PyMuPDF": "1", "numpy": "2", "scikit-learn": "1", "tiktoken": "1"},
             "module_sha256": {name: contracts.H for name in ("core.py", "ingest.py", "sources.py", "corpus.py")},
             "sources": [{"document_id": doc["document_id"], "source_sha256": doc["pdf_sha256"], "source_bytes": 10} for doc in docs],
             "observed_counts": {name: manifest[name] for name in ("document_count", "page_count", "nonempty_page_count", "chunk_count")},
             "human_acceptance": False}
    build_sha = write(corpus / "build-receipt.json", build)
    np.save(vectors / "bge-small-en-v1.5.npy", data["passages"], allow_pickle=False)
    cfg["input_sha256"]["vector_matrix"] = sha256((vectors / "bge-small-en-v1.5.npy").read_bytes()).hexdigest()
    ids_sha = h("\n".join(chunk.chunk_id for chunk in chunks))
    runtime = {name: "1" for name in ("numpy", "python", "sentence-transformers", "torch", "transformers")}
    vector_manifest = {"schema_version": 2, "build_runtime": runtime, "chunk_count": 8, "chunk_ids_sha256": ids_sha,
                       "chunks_sha256": cfg["input_sha256"]["corpus_chunks"], "dtype": "float32", "normalized": True,
                       "embeddings_sha256": cfg["input_sha256"]["vector_matrix"], "model_name": cfg["model"]["name"],
                       "model_revision": cfg["model"]["revision"], "shape": [8, 384]}
    cfg["input_sha256"]["vector_manifest"] = write(vectors / "embedding-manifest.json", vector_manifest)
    wheel = tmp_path / "nvidia_financial_rag-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"reviewed wheel test bytes")
    cfg["input_sha256"]["wheel"] = sha256(wheel.read_bytes()).hexdigest()
    vector_receipt = {"schema_version": 1, "status": "complete", "scope": "operator-local-bge-vector-build",
                      "corpus": {"source_registry_sha256": cfg["input_sha256"]["registry"], "manifest_sha256": cfg["input_sha256"]["corpus_manifest"],
                                 "chunks_sha256": cfg["input_sha256"]["corpus_chunks"], "build_receipt_sha256": build_sha},
                      "embedding_manifest_sha256": cfg["input_sha256"]["vector_manifest"], "embeddings_sha256": cfg["input_sha256"]["vector_matrix"],
                      "chunk_ids_sha256": ids_sha, "matrix": {"dtype": "float32", "normalized": True, "shape": [8, 384]},
                      "model": {"name": cfg["model"]["name"], "revision": cfg["model"]["revision"], "device": "cpu",
                                "execution_origin": "cached-local-only", "local_files_only": True},
                      "runtime": runtime,
                      "module_sha256": {name: contracts.H for name in ("__init__.py", "core.py", "ingest.py", "sources.py", "corpus.py",
                                                                        "embedding_store.py", "corpus_embeddings.py", "runtime.py", "retrieval.py", "service.py")},
                      "post_build_verified": True, "generation_runs": 0, "human_acceptance": False}
    vector_receipt["wheel"] = {"wheel_basename": wheel.name, "wheel_bytes": wheel.stat().st_size,
                               "wheel_sha256": cfg["input_sha256"]["wheel"], "distribution_version": "0.1.0",
                               "module_sha256": vector_receipt["module_sha256"].copy()}
    write(vectors / "vector-build-receipt.json", vector_receipt)
    source_replay = tmp_path / "source-replay.json"
    replay_receipt = {"schema_version": 2, "status": "complete", "scope": "operator-local-source-candidate-replay",
                      "input_sha256": {name: cfg["input_sha256"][name] for name in ("registry", "cases", "case_freeze", "corpus_manifest", "corpus_chunks")},
                      "ordered_case_ids": cfg["expected_case_ids"], "counts": cfg["case_counts"],
                      "source_files": [{"document_id": doc["document_id"], "pdf_sha256": doc["pdf_sha256"], "source_bytes": 10} for doc in docs],
                      "chunk_ids": [chunk.chunk_id for chunk in chunks], "corpus_profile": build["profile"], "cells": [], "rationale_bindings": [],
                      "source_binding_passed": False, "human_gold_count": 0, "generation_runs": 0,
                      "joint_source_candidate_coverage": "not_evaluated", "limitations": []}
    replay_receipt["input_sha256"]["build_receipt"] = build_sha
    binding_cfg = deepcopy(cfg)
    binding_cfg["input_sha256"].pop("source_replay")
    replay_receipt["experiment_binding_sha256"] = sha256(json.dumps(binding_cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    cfg["input_sha256"]["source_replay"] = write(source_replay, replay_receipt)
    write(config_path, cfg)
    contract_api = contracts.api()
    bundle = contract_api.load_case_bundle(cases_path, freeze_path, contract_api.load_experiment(config_path))
    data["bundle"] = bundle
    for row, record in enumerate(data["results"]["rows"]):
        record["scope_document_ids"] = None if row == 0 else [docs[1]["document_id"]]
    data = rebuild_receipt(data)
    files = {}
    for name, matrix in (("queries-raw.npy", data["raw_queries"]), ("queries-normalized.npy", data["normalized_queries"])):
        np.save(comparison / name, matrix, allow_pickle=False)
        files[name] = sha256((comparison / name).read_bytes()).hexdigest()
    for name, value in (("query-index.json", data["query_index"]), ("results.json", data["results"])):
        files[name] = write(comparison / name, value)
    completed = {"schema_version": 2, "status": "completed", "experiment_sha256": sha256(config_path.read_bytes()).hexdigest(),
                 "input_sha256": cfg["input_sha256"], "input_paths": {"cases": str(cases_path), "source_replay": str(source_replay), "wheel": str(wheel)},
                 "producer": {"numpy_version": np.__version__, "python_version": "test", "platform": "synthetic"}, "query_count": 2, "files": files}
    write(comparison / "manifest.json", completed)
    return {"artifact_dir": corpus, "source_registry": registry, "vector_dir": vectors,
            "experiment": config_path, "case_freeze": freeze_path, "comparison_dir": comparison}, completed


def test_completed_comparison_replay_binds_every_input_and_file_without_model(tmp_path):
    arguments, _ = completed_fixture(tmp_path)
    result = api().run_replay(**arguments)
    assert result["arithmetic_passed"] is True
    assert result["producer_runtime"]["platform"] == "synthetic"
    assert set(result["input_sha256"]) == {"registry", "corpus_manifest", "corpus_chunks", "vector_manifest", "vector_matrix", "cases", "case_freeze", "source_replay", "wheel"}


@pytest.mark.parametrize('period', ['absent', None, {'fiscal_year': 2025, 'quarter': 2}])
def test_optional_registered_period_is_checked_only_when_supplied(tmp_path, period):
    arguments, _ = completed_fixture(tmp_path)
    module = api()
    bodies = {name: path.read_bytes() for name, path in {
        'registry': arguments['source_registry'], 'corpus_manifest': arguments['artifact_dir'] / 'manifest.json',
        'corpus_chunks': arguments['artifact_dir'] / 'chunks.jsonl', 'build_receipt': arguments['artifact_dir'] / 'build-receipt.json',
        'vector_manifest': arguments['vector_dir'] / 'embedding-manifest.json',
        'vector_matrix': arguments['vector_dir'] / 'bge-small-en-v1.5.npy',
        'vector_receipt': arguments['vector_dir'] / 'vector-build-receipt.json'}.items()}
    def encoded(value):
        return json.dumps(value, sort_keys=True, allow_nan=False).encode()
    registry = json.loads(bodies['registry'])
    manifest = json.loads(bodies['corpus_manifest'])
    if period == 'absent':
        registry['sources'][0].pop('reporting_period')
        manifest['documents'][0].pop('reporting_period')
    else:
        registry['sources'][0]['reporting_period'] = period
        manifest['documents'][0]['reporting_period'] = period
    bodies['registry'], bodies['corpus_manifest'] = encoded(registry), encoded(manifest)
    build = json.loads(bodies['build_receipt'])
    build['source_registry_sha256'] = sha256(bodies['registry']).hexdigest()
    build['manifest_sha256'] = sha256(bodies['corpus_manifest']).hexdigest()
    bodies['build_receipt'] = encoded(build)
    receipt = json.loads(bodies['vector_receipt'])
    receipt['corpus'].update(source_registry_sha256=build['source_registry_sha256'],
        manifest_sha256=build['manifest_sha256'], build_receipt_sha256=sha256(bodies['build_receipt']).hexdigest())
    bodies['vector_receipt'] = encoded(receipt)
    cfg = json.loads(arguments['experiment'].read_bytes())
    if period == 'absent':
        chunks, matrix = module.load_corpus_vectors(bodies, cfg)
        assert len(chunks) == 8 and matrix.shape == (8, 384)
    else:
        with pytest.raises(ValueError, match='period'):
            module.load_corpus_vectors(bodies, cfg)


@pytest.mark.parametrize('marker', ['pending', 'failure'])
def test_guard_appearing_during_final_hash_reads_is_refused(tmp_path, monkeypatch, marker):
    arguments, _ = completed_fixture(tmp_path)
    module = api()
    original_replay, original_read = module.replay_rows, module.read_body
    after_arithmetic = [False]
    def replay(**kwargs):
        result = original_replay(**kwargs)
        after_arithmetic[0] = True
        return result
    def read(path):
        result = original_read(path)
        if after_arithmetic[0] and path == arguments['source_registry']:
            (arguments['comparison_dir'] / (marker + '.json')).write_text('{}', encoding='utf-8')
        return result
    monkeypatch.setattr(module, 'replay_rows', replay)
    monkeypatch.setattr(module, 'read_body', read)
    with pytest.raises(ValueError, match='pending|failure'):
        module.run_replay(**arguments)


@pytest.mark.parametrize('mutation', ['failure', 'pending-noop', 'input', 'output'])
def test_replay_unlink_close_mutation_never_returns_verified(tmp_path, monkeypatch, mutation):
    arguments, _ = completed_fixture(tmp_path)
    module = api()
    output = tmp_path / 'arithmetic.json'
    pending = output.with_name(output.name + '.pending.json')
    failure = output.with_name(output.name + '.failure.json')
    original = Path.unlink
    def changed(path, *args, **kwargs):
        if path == pending:
            if mutation == 'pending-noop':
                return None
            if mutation == 'failure':
                failure.write_text('{}', encoding='utf-8')
            elif mutation == 'input':
                target = arguments['source_registry']
                target.write_bytes(target.read_bytes() + b'\n')
            else:
                output.write_bytes(output.read_bytes() + b'changed')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', changed)
    with pytest.raises(ValueError):
        module.run_replay(**arguments, output=output)
    assert pending.exists() or failure.exists()


@pytest.mark.parametrize('mutation', ['pending', 'failure', 'input'])
def test_final_owned_output_read_cannot_hide_new_input_guard(tmp_path, monkeypatch, mutation):
    arguments, _ = completed_fixture(tmp_path)
    module = api()
    output = tmp_path / 'arithmetic.json'
    pending = output.with_name(output.name + '.pending.json')
    original = module.read_body
    def changed(path):
        result = original(path)
        if path == output and not pending.exists():
            if mutation == 'input':
                target = arguments['source_registry']
                target.write_bytes(target.read_bytes() + b'\n')
            else:
                (arguments['comparison_dir'] / (mutation + '.json')).write_text('{}', encoding='utf-8')
        return result
    monkeypatch.setattr(module, 'read_body', changed)
    with pytest.raises(ValueError):
        module.run_replay(**arguments, output=output)
    assert pending.exists() or output.with_name(output.name + '.failure.json').exists()


def test_full_completed_replay_runs_with_model_retriever_and_service_imports_blocked(tmp_path):
    arguments, _ = completed_fixture(tmp_path)
    code = r'''
import importlib.abc, json, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("sentence_transformers", "torch", "transformers", "nvidia_rag.retrieval", "nvidia_rag.service")):
            raise RuntimeError("forbidden dependency " + fullname)
sys.meta_path.insert(0, Block())
from acceptance.expanded_cosine_replay import run_replay
result = run_replay(**{name:pathlib.Path(path) for name,path in json.loads(sys.argv[1]).items()})
assert result['arithmetic_passed'] is True
'''
    run = subprocess.run([sys.executable, "-c", code, json.dumps({name: str(path) for name, path in arguments.items()})],
                         capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stderr


def test_rounded_consumed_query_from_float64_to_float32_is_refused():
    data = fixture()
    data["raw_queries"][0, :3] = [3, 5, 7]
    q = data["raw_queries"][0].astype(np.float64)
    data["normalized_queries"][0] = q / np.linalg.norm(q)
    data = rebuild_receipt(data)
    data["normalized_queries"] = data["normalized_queries"].astype(np.float32).astype(np.float64)
    with pytest.raises(ValueError, match="normalized query"):
        api().replay_rows(**data)


@pytest.mark.parametrize("location", ["considered_scores", "top_k"])
def test_huge_integer_score_is_bounded_value_error_instead_of_overflow(location):
    data = fixture()
    if location == "considered_scores":
        data["results"]["rows"][0][location][0] = 10 ** 400
    else:
        data["results"]["rows"][0][location][0]["score"] = 10 ** 400
    with pytest.raises(ValueError, match="score"):
        api().replay_rows(**data)


def test_comparison_changing_after_arithmetic_is_refused(tmp_path, monkeypatch):
    arguments, _ = completed_fixture(tmp_path)
    module = api()
    original = module.replay_rows
    def changed(**kwargs):
        result = original(**kwargs)
        path = arguments["comparison_dir"] / "query-index.json"
        path.write_bytes(path.read_bytes() + b" ")
        return result
    monkeypatch.setattr(module, "replay_rows", changed)
    with pytest.raises(ValueError, match="changed during replay"):
        module.run_replay(**arguments)


@pytest.mark.parametrize("kind", ["corpus-profile-boolean", "corpus-runtime", "vector-runtime", "vector-wheel-identity"])
def test_nested_receipt_contracts_reject_invalid_metadata(tmp_path, kind):
    arguments, _ = completed_fixture(tmp_path)
    module = api()
    cfg = module.load_experiment(arguments["experiment"]).data
    paths = {"registry": arguments["source_registry"], "corpus_manifest": arguments["artifact_dir"] / "manifest.json",
             "corpus_chunks": arguments["artifact_dir"] / "chunks.jsonl", "build_receipt": arguments["artifact_dir"] / "build-receipt.json",
             "vector_manifest": arguments["vector_dir"] / "embedding-manifest.json", "vector_matrix": arguments["vector_dir"] / "bge-small-en-v1.5.npy",
             "vector_receipt": arguments["vector_dir"] / "vector-build-receipt.json"}
    bodies = {name: path.read_bytes() for name, path in paths.items()}
    if kind == "corpus-profile-boolean":
        receipt = json.loads(bodies["build_receipt"])
        receipt["profile"]["overlap_tokens"] = False
        bodies["build_receipt"] = json.dumps(receipt).encode()
    elif kind == "corpus-runtime":
        receipt = json.loads(bodies["build_receipt"])
        receipt["runtime"]["python"] = False
        bodies["build_receipt"] = json.dumps(receipt).encode()
    elif kind == "vector-runtime":
        for name in ("vector_manifest", "vector_receipt"):
            receipt = json.loads(bodies[name])
            receipt["build_runtime" if name == "vector_manifest" else "runtime"]["numpy"] = False
            bodies[name] = json.dumps(receipt).encode()
        receipt = json.loads(bodies["vector_receipt"])
        receipt["embedding_manifest_sha256"] = sha256(bodies["vector_manifest"]).hexdigest()
        bodies["vector_receipt"] = json.dumps(receipt).encode()
    else:
        receipt = json.loads(bodies["vector_receipt"])
        receipt["wheel"]["wheel_sha256"] = "0" * 64
        bodies["vector_receipt"] = json.dumps(receipt).encode()
    with pytest.raises(ValueError):
        module.load_corpus_vectors(bodies, cfg)


@pytest.mark.parametrize("kind", ["comparison-pending", "comparison-failure", "corpus-pending", "vector-failure", "source-pending",
                                  "raw-file-mutated", "cases-mutated", "freeze-mutated", "wheel-mutated", "source-replay-mutated",
                                  "registry-mutated", "manifest-mutated", "matrix-mutated", "relative-input", "unknown-output-file"])
def test_completed_looking_comparison_refuses_bad_closure(tmp_path, kind):
    arguments, manifest = completed_fixture(tmp_path)
    comparison = arguments["comparison_dir"]
    if kind in ("comparison-pending", "comparison-failure", "corpus-pending", "vector-failure"):
        directory, marker = kind.split("-")
        directory = {"comparison": comparison, "corpus": arguments["artifact_dir"], "vector": arguments["vector_dir"]}[directory]
        (directory / (marker + ".json")).write_text("{}")
    elif kind == "source-pending":
        path = Path(manifest["input_paths"]["source_replay"])
        path.with_name(path.name + ".pending.json").write_text("{}")
    elif kind == "relative-input":
        manifest["input_paths"]["wheel"] = "reviewed.whl"
        (comparison / "manifest.json").write_text(json.dumps(manifest))
    elif kind == "unknown-output-file":
        manifest["files"]["extra.json"] = "a" * 64
        (comparison / "manifest.json").write_text(json.dumps(manifest))
    else:
        path = {"raw-file-mutated": comparison / "queries-raw.npy", "cases-mutated": Path(manifest["input_paths"]["cases"]),
                "freeze-mutated": arguments["case_freeze"], "wheel-mutated": Path(manifest["input_paths"]["wheel"]),
                "source-replay-mutated": Path(manifest["input_paths"]["source_replay"]), "registry-mutated": arguments["source_registry"],
                "manifest-mutated": arguments["artifact_dir"] / "manifest.json", "matrix-mutated": arguments["vector_dir"] / "bge-small-en-v1.5.npy"}[kind]
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        api().run_replay(**arguments)


def test_cli_rejects_relative_paths_and_output_overlap_and_preserves_existing_output(tmp_path):
    arguments, _ = completed_fixture(tmp_path)
    cli = [sys.executable, "-m", "acceptance.expanded_cosine_replay"]
    for name, path in arguments.items():
        cli += ["--" + name.replace("_", "-"), str(path)]
    output = tmp_path / "replay-output.json"
    run = subprocess.run(cli + ["--output", str(output)], capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stderr
    before = output.read_bytes()
    rerun = subprocess.run(cli + ["--output", str(output)], capture_output=True, text=True, cwd=ROOT)
    assert rerun.returncode != 0 and output.read_bytes() == before
    overlap = subprocess.run(cli + ["--output", str(arguments["comparison_dir"] / "new.json")], capture_output=True, text=True, cwd=ROOT)
    assert overlap.returncode != 0 and not (arguments["comparison_dir"] / "new.json").exists()
    relative = subprocess.run(cli + ["--output", "relative.json"], capture_output=True, text=True, cwd=ROOT)
    assert relative.returncode != 0
