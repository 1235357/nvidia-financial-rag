"""Explicit corpus scope and consumer boundaries; ASGI checks are not real HTTP acceptance."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import sys
import zipfile

from fastapi.testclient import TestClient
import pytest

from test_corpus_recipe import Artifact, library

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "generated-corpus-test-" + "x" * 32


def script(name):
    path = ROOT / name
    assert path.is_file(), f"consumer entry point missing: {name}"
    return runpy.run_path(str(path))


def service(corpus, **kwargs):
    module = library()
    build = getattr(module, "build_corpus_service", None)
    assert callable(build), "installed explicit corpus service interface is missing"
    return build(corpus, threshold=kwargs.get("threshold", 0.0), top_k=kwargs.get("top_k", 5),
                 document_ids=kwargs.get("document_ids"))


def test_explicit_scope_does_not_fall_back_to_other_document(tmp_path):
    a = Artifact(tmp_path, two=True)
    a.records[1]["text"] = "Generated Revenue evidence Revenue Revenue"
    a.sync()
    result = service(a.load(), document_ids=frozenset({"report"})).query("Revenue")
    assert [r.chunk.document_id for r in result.evidence] == ["report"]
    assert result.citations == ("report p.1",)


@pytest.mark.parametrize("scope", [frozenset(), frozenset({"missing"}), frozenset({" "}),
                                   frozenset({1}), {"report"}, "report"])
def test_invalid_scope_refused_before_fitting(tmp_path, scope, monkeypatch):
    corpus = Artifact(tmp_path).load()
    def unexpected(*args, **kwargs):
        pytest.fail("TF-IDF fitting happened before scope validation")
    monkeypatch.setattr("nvidia_rag.retrieval.TfidfRetriever", unexpected)
    with pytest.raises(ValueError):
        service(corpus, document_ids=scope)


@pytest.mark.parametrize("threshold,top_k", [(True, 5), (float("nan"), 5), (-0.1, 5),
                                           (1.1, 5), (0.0, True), (0.0, 0), (0.0, "5")])
def test_invalid_settings_refused_before_fitting(tmp_path, threshold, top_k, monkeypatch):
    corpus = Artifact(tmp_path).load()
    def unexpected(*args, **kwargs):
        pytest.fail("TF-IDF fitting happened before settings validation")
    monkeypatch.setattr("nvidia_rag.retrieval.TfidfRetriever", unexpected)
    with pytest.raises(ValueError):
        service(corpus, threshold=threshold, top_k=top_k)


def test_scope_delegation_uses_exact_immutable_selection():
    module = library()
    cls = getattr(module, "DocumentScopeRetriever", None)
    assert callable(cls), "installed document scope interface is missing"
    class Scoped:
        def retrieve_from_documents(self, query, *, top_k, document_ids):
            if query != "Revenue" or top_k != 2 or document_ids != {"report"}:
                raise RuntimeError("Wrong scope delegation")
            return ["scope-confirmed"]
    assert cls(Scoped(), document_ids=frozenset({"report"})).retrieve("Revenue", top_k=2) == ["scope-confirmed"]


def test_positive_gate_refuses_zero_similarity(tmp_path):
    corpus = Artifact(tmp_path).load()
    result = service(corpus, threshold=0.01).query("unrelatedword")
    assert result.can_answer is False
    assert result.evidence == ()


def test_launcher_rejects_token_before_corpus_io(tmp_path):
    namespace = script("examples/serve_corpus.py")
    with pytest.raises(ValueError, match="TOKEN"):
        namespace["create_corpus_app"]({}, artifact_dir=tmp_path / "absent", source_registry=tmp_path / "absent.json",
                                       threshold=0.0, top_k=5, document_ids=None)


def test_launcher_has_loopback_only_argument_contract():
    namespace = script("examples/serve_corpus.py")
    with pytest.raises(SystemExit) as error:
        namespace["parse_args"](["--host", "0.0.0.0"])
    assert error.value.code == 2


def test_real_app_serializes_registered_evidence_and_request_id(tmp_path):
    a = Artifact(tmp_path)
    ns = script("examples/serve_corpus.py")
    app = ns["create_corpus_app"]({"NVIDIA_RAG_API_TOKEN": TOKEN}, artifact_dir=a.directory,
                                  source_registry=a.registry, threshold=0.0, top_k=5, document_ids=None)
    client = TestClient(app)
    response = client.post("/v1/query", headers={"Authorization": "Bearer " + TOKEN,
                           "X-Request-ID": "corpus-control"}, json={"question": "Revenue"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["answer"] == "" and payload["error_code"] is None
    assert payload["citations"] == ["report p.1"]
    assert payload["evidence"][0]["chunk_id"] == "report:p1:w:0123456789abcdef"
    assert payload["evidence"][0]["text"] == "Generated Revenue evidence"
    assert payload["evidence"][0]["provenance"] is None
    assert payload["request_id"] == response.headers["X-Request-ID"] == "corpus-control"
    assert client.get("/readyz").status_code == 401
    assert client.get("/docs").status_code == 404


def test_comparison_rejects_response_wrong_scope_or_header(tmp_path):
    a = Artifact(tmp_path)
    context = service(a.load()).query("Revenue")
    ns = script("acceptance/corpus_consumer.py")
    compare = ns["compare_evidence_response"]
    payload = {"request_id": "case", "can_answer": True, "evidence_passed": True,
               "message": context.message, "answer": "", "citations": ["report p.1"], "error_code": None,
               "evidence": [{"rank": 1, "chunk_id": "report:p1:w:0123456789abcdef",
                             "document_id": "report", "pdf_page": 1, "score": context.evidence[0].score,
                             "text": "Generated Revenue evidence", "provenance": None}]}
    row = compare(payload, context, status_code=200, request_id_header="case", expected_request_id="case")
    assert row["answer_empty"] is True and row["evidence"][0]["text_sha256"] == sha256(b"Generated Revenue evidence").hexdigest()
    with pytest.raises(ValueError, match="request"):
        compare(payload, context, status_code=200, request_id_header="different", expected_request_id="case")
    payload["evidence"][0]["document_id"] = "other"
    with pytest.raises(ValueError):
        compare(payload, context, status_code=200, request_id_header="case", expected_request_id="case")


@pytest.fixture
def wheel_fixture(tmp_path, monkeypatch):
    pure = tmp_path / "purelib"
    package = pure / "nvidia_rag"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"VERSION = 'test'\n")
    (package / "corpus.py").write_bytes(b"VALUE = 1\n")
    wheel = tmp_path / "nvidia_financial_rag-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("nvidia_rag/__init__.py", b"VERSION = 'test'\n")
        z.writestr("nvidia_rag/corpus.py", b"VALUE = 1\n")
    ns = script("acceptance/corpus_consumer.py")
    monkeypatch.setattr(ns["nvidia_rag"], "__file__", str(package / "__init__.py"))
    monkeypatch.setattr(ns["sysconfig"], "get_paths", lambda: {"purelib": str(pure)})
    return wheel, package, ns


def test_wheel_identity_compares_actual_installed_modules(wheel_fixture):
    wheel, package, ns = wheel_fixture
    proof = ns["verify_installed_wheel"](wheel)
    assert proof["wheel_sha256"] == sha256(wheel.read_bytes()).hexdigest()
    assert proof["module_sha256"]["corpus.py"] == sha256(b"VALUE = 1\n").hexdigest()
    (package / "corpus.py").write_bytes(b"VALUE = 2\n")
    with pytest.raises(ValueError):
        ns["verify_installed_wheel"](wheel)


@pytest.mark.parametrize("change", ["extra-installed", "duplicate-wheel", "missing-wheel"])
def test_wheel_identity_rejects_member_set_mismatch(wheel_fixture, change):
    wheel, package, ns = wheel_fixture
    if change == "extra-installed":
        (package / "extra.py").write_bytes(b"EXTRA = 1\n")
    elif change == "duplicate-wheel":
        with zipfile.ZipFile(wheel, "a") as z, pytest.warns(UserWarning):
            z.writestr("nvidia_rag/corpus.py", b"VALUE = 1\n")
    else:
        with zipfile.ZipFile(wheel, "w") as z:
            z.writestr("nvidia_rag/__init__.py", b"VERSION = 'test'\n")
    with pytest.raises(ValueError):
        ns["verify_installed_wheel"](wheel)


def test_existing_comparison_output_precedes_child_creation(tmp_path, monkeypatch):
    ns = script("acceptance/corpus_consumer.py")
    target = tmp_path / "retained.json"
    target.write_text("retained", encoding="utf-8")
    def unexpected(*args, **kwargs):
        pytest.fail("Child started before existing-output refusal")
    monkeypatch.setattr(ns["subprocess"], "Popen", unexpected)
    with pytest.raises(ValueError, match="exist"):
        ns["run_comparison"](artifact_dir=tmp_path / "absent", source_registry=tmp_path / "absent.json",
                              candidates=tmp_path / "absent.jsonl", wheel=tmp_path / "absent.whl", output=target)
    assert target.read_text("utf-8") == "retained"


def test_comparison_output_cannot_mutate_read_only_corpus(tmp_path, monkeypatch):
    ns = script("acceptance/corpus_consumer.py")
    corpus = Artifact(tmp_path)
    def unexpected(*args, **kwargs):
        pytest.fail("wheel IO reached before read-only output validation")
    ns["run_comparison"].__globals__["verify_installed_wheel"] = unexpected
    with pytest.raises(ValueError, match="output.*overlap|output.*corpus"):
        ns["run_comparison"](artifact_dir=corpus.directory, source_registry=corpus.registry,
                              candidates=tmp_path / "absent.jsonl", wheel=tmp_path / "absent.whl",
                              output=corpus.directory / "comparison.json")
    assert not (corpus.directory / "comparison.json").exists()


def test_launch_corrupted_artifact_never_fits(tmp_path, monkeypatch):
    corpus = Artifact(tmp_path)
    (corpus.directory / "chunks.jsonl").write_text("corrupt", encoding="utf-8")
    namespace = script("examples/serve_corpus.py")
    def unexpected(*args, **kwargs):
        pytest.fail("retriever fitted corrupt corpus")
    monkeypatch.setattr("nvidia_rag.retrieval.TfidfRetriever", unexpected)
    with pytest.raises(ValueError):
        namespace["create_corpus_app"]({"NVIDIA_RAG_API_TOKEN": TOKEN}, artifact_dir=corpus.directory,
                                       source_registry=corpus.registry, threshold=0.0, top_k=5, document_ids=None)


def test_api_scope_fault_remains_operational_failure(tmp_path, monkeypatch):
    a = Artifact(tmp_path)
    ns = script("examples/serve_corpus.py")
    class Broken:
        def query(self, query):
            raise RuntimeError("test scope service unavailable")
    ns["create_corpus_app"].__globals__["build_corpus_service"] = lambda *args, **kwargs: Broken()
    client = TestClient(ns["create_corpus_app"]({"NVIDIA_RAG_API_TOKEN": TOKEN}, artifact_dir=a.directory,
                                              source_registry=a.registry, threshold=0.0, top_k=5, document_ids=None))
    response = client.post("/v1/query", headers={"Authorization": "Bearer " + TOKEN}, json={"question": "Revenue"})
    assert response.status_code == 503
    assert response.json()["error_code"] == "query_service_unavailable"
    assert response.json()["answer"] == ""


def test_teardown_refuses_a_nonterminal_owned_child(monkeypatch):
    ns = script("acceptance/corpus_consumer.py")
    verify = ns.get("verify_owned_teardown")
    assert callable(verify), "owned child/listener teardown verification is missing"
    class Running:
        def poll(self):
            return None
    with pytest.raises(ValueError, match="running|terminal"):
        verify(Running(), 12345)


def test_teardown_observes_closed_listener_without_stopping_others(monkeypatch):
    ns = script("acceptance/corpus_consumer.py")
    verify = ns.get("verify_owned_teardown")
    assert callable(verify), "owned child/listener teardown verification is missing"
    class Terminal:
        def poll(self):
            return 0
    def closed(address, timeout):
        assert address == ("127.0.0.1", 12345)
        raise OSError("closed controlled listener")
    monkeypatch.setattr(ns["socket"], "create_connection", closed)
    verify(Terminal(), 12345)
    class Reachable:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(ns["socket"], "create_connection", lambda *args, **kwargs: Reachable())
    with pytest.raises(ValueError, match="reachable"):
        verify(Terminal(), 12345)


@pytest.mark.parametrize("which", ["registry", "manifest", "candidates"])
def test_comparison_binds_input_snapshot_before_loading(tmp_path, monkeypatch, which):
    ns = script("acceptance/corpus_consumer.py")
    a = Artifact(tmp_path)
    candidates = tmp_path / "candidate.jsonl"
    candidates.write_text("generated candidate", encoding="utf-8")
    target = {"registry": a.registry, "manifest": a.directory / "manifest.json", "candidates": candidates}[which]
    real_load = library().load_verified_corpus
    def changing_load(**kwargs):
        corpus = real_load(**kwargs)
        target.write_bytes(target.read_bytes() + b"\n")
        return corpus
    globals_ = ns["run_comparison"].__globals__
    globals_["verify_installed_wheel"] = lambda path: {"identity": "controlled"}
    globals_["load_verified_corpus"] = changing_load
    monkeypatch.setattr(ns["runpy"], "run_path", lambda path: {"load_candidates": lambda p: []})
    with pytest.raises(ValueError, match="changed"):
        ns["run_comparison"](artifact_dir=a.directory, source_registry=a.registry, candidates=candidates,
                              wheel=tmp_path / "controlled.whl", output=tmp_path / "comparison.json")
    assert not (tmp_path / "comparison.json").exists()


@pytest.mark.parametrize("marker", ["pending.json", "failure.json"])
def test_comparison_rechecks_corpus_completion_state_after_query_work(tmp_path, monkeypatch, marker):
    from test_corpus_semantic import case
    a, _, _, _, candidates, wheel, identity = case(tmp_path, monkeypatch)
    ns = script("acceptance/corpus_consumer.py")
    globals_ = ns["run_comparison"].__globals__
    globals_["verify_installed_wheel"] = lambda path: identity
    frozen_questions = script("acceptance/source_binding_replay.py")["load_candidates"]
    monkeypatch.setattr(ns["runpy"], "run_path", lambda path: {
        "load_candidates": frozen_questions, "query_api": lambda *args, **kwargs: None})
    def controlled_scope(*args, **kwargs):
        (a.directory / marker).write_text("incomplete operation state")
        return {"records": []}
    globals_["_run_scope"] = controlled_scope
    output = tmp_path / "comparison.json"
    with pytest.raises(ValueError, match="pending|failed|completion"):
        ns["run_comparison"](artifact_dir=a.directory, source_registry=a.registry,
                              candidates=candidates, wheel=wheel, output=output)
    assert not output.exists()

