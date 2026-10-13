"""Installed startup assembly; generated corpus/PDF cases are not financial acceptance."""
from __future__ import annotations

from hashlib import sha256
import importlib
from pathlib import Path
import runpy
import sys
import tomllib

from fastapi.testclient import TestClient
import pytest

from test_corpus_recipe import Artifact, write_json
from test_structured_income import source_pdf

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "generated-serving-test-" + "x" * 32
HEADERS = {"Authorization": "Bearer " + TOKEN, "X-Request-ID": "serving-control"}
QUESTION = ("According to NVIDIA fiscal 2025 Q2 report, what was net income "
            "for the three months ended July 28, 2024, in millions of dollars?")


def serving():
    try:
        module = importlib.import_module("nvidia_rag.serving")
    except ModuleNotFoundError as error:
        if error.name == "nvidia_rag.serving":
            pytest.fail("installed corpus API startup interface is missing")
        raise
    assert callable(getattr(module, "create_corpus_app", None)), "corpus app factory is missing"
    return module


def create(a, **kwargs):
    return serving().create_corpus_app({"NVIDIA_RAG_API_TOKEN": TOKEN},
        artifact_dir=a.directory, source_registry=a.registry, threshold=0.0,
        top_k=5, document_ids=None, **kwargs)


def arguments(tmp_path):
    return ["--artifact-dir", str(tmp_path / "corpus"), "--source-registry",
            str(tmp_path / "sources.json"), "--threshold", "0"]


@pytest.fixture
def statement_corpus(tmp_path):
    a = Artifact(tmp_path)
    report = source_pdf(tmp_path)
    a.sources[0].update(document_id=report.document_id, filename=report.pdf_path.name,
                        sha256=report.pdf_sha256)
    a.records[0].update(document_id=report.document_id,
        chunk_id=report.document_id + ":p1:w:0123456789abcdef")
    a.manifest["documents"][0].update(document_id=report.document_id,
        source_filename=report.pdf_path.name, source_sha256=report.pdf_sha256)
    a.receipt["sources"][0].update(document_id=report.document_id,
        source_sha256=report.pdf_sha256, source_bytes=report.pdf_path.stat().st_size)
    a.sync()
    declarations = tmp_path / "fiscal.json"
    write_json(declarations, {"schema_version": 1,
        "source_registry_sha256": sha256(a.registry.read_bytes()).hexdigest(),
        "reports": [{"document_id": report.document_id, "pdf_sha256": report.pdf_sha256,
            "fiscal_year": 2025, "fiscal_quarter": 2,
            "reviewed_by": "synthetic boundary test", "source_locator": "generated page 2"}]})
    return a, report, declarations


def test_installed_factory_exposes_real_lexical_evidence_and_disabled_statement_route(tmp_path):
    app = create(Artifact(tmp_path))
    client = TestClient(app)
    lexical = client.post("/v1/query", headers=HEADERS, json={"question": "Revenue"})
    assert lexical.status_code == 200
    assert lexical.json()["answer"] == "" and lexical.json()["citations"] == ["report p.1"]
    assert lexical.json()["request_id"] == lexical.headers["X-Request-ID"] == "serving-control"
    statement = client.post("/v1/statement-candidates", headers=HEADERS, json={"question": QUESTION})
    assert statement.status_code == 503
    assert statement.json()["reason"] == "STATEMENT_INDEX_NOT_CONFIGURED"
    assert client.get("/docs").status_code == 404


def test_token_is_resolved_before_all_input_io(tmp_path, monkeypatch):
    module = serving()
    def unexpected(*args, **kwargs):
        pytest.fail("filesystem read occurred before token validation")
    monkeypatch.setattr(Path, "read_bytes", unexpected)
    monkeypatch.setattr(Path, "resolve", unexpected)
    with pytest.raises(ValueError, match="TOKEN"):
        module.create_corpus_app({}, artifact_dir=tmp_path / "absent",
            source_registry=tmp_path / "absent.json", threshold=0.0, top_k=5, document_ids=None,
            statement_source_dir=tmp_path / "absent")


@pytest.mark.parametrize("option", ["statement_source_dir", "statement_fiscal_declarations"])
def test_partial_statement_configuration_refuses_before_corpus_io(tmp_path, monkeypatch, option):
    module = serving()
    monkeypatch.setattr(module, "load_verified_corpus",
        lambda **kwargs: pytest.fail("corpus IO reached with incomplete statement configuration"))
    with pytest.raises(ValueError, match="together|paired|both"):
        module.create_corpus_app({"NVIDIA_RAG_API_TOKEN": TOKEN},
            artifact_dir=tmp_path / "absent", source_registry=tmp_path / "absent.json",
            threshold=0.0, top_k=5, document_ids=None, **{option: tmp_path})


def test_reviewed_statement_snapshot_is_installed_and_retained_after_startup(statement_corpus):
    a, report, declarations = statement_corpus
    client = TestClient(create(a, statement_source_dir=report.pdf_path.parent,
        statement_fiscal_declarations=declarations))
    first = client.post("/v1/statement-candidates", headers=HEADERS, json={"question": QUESTION})
    assert first.status_code == 200 and first.json()["disposition"] == "source_candidate"
    assert first.json()["cells"][0]["amount"] == "2860"
    assert first.json()["cells"][0]["pdf_sha256"] == report.pdf_sha256
    assert first.json()["financial_correctness"] == "unestablished"
    assert str(report.pdf_path) not in first.text
    report.pdf_path.write_bytes(b"changed after frozen startup")
    second = client.post("/v1/statement-candidates", headers=HEADERS, json={"question": QUESTION})
    assert second.json()["cells"] == first.json()["cells"]
    assert second.json()["index_sha256"] == first.json()["index_sha256"]


def test_declared_scope_is_shared_by_lexical_and_statement_services(statement_corpus, monkeypatch):
    a, report, declarations = statement_corpus
    module = serving()
    factory = importlib.import_module("nvidia_rag.statement_startup")
    observed = []
    actual = factory.load_statement_snapshot
    def observe(**kwargs):
        observed.append(kwargs)
        return actual(**kwargs)
    monkeypatch.setattr(factory, "load_statement_snapshot", observe)
    module.create_corpus_app({"NVIDIA_RAG_API_TOKEN": TOKEN}, artifact_dir=a.directory,
        source_registry=a.registry, threshold=0.0, top_k=5,
        document_ids=frozenset({report.document_id}),
        statement_source_dir=report.pdf_path.parent, statement_fiscal_declarations=declarations)
    assert observed[0]["document_ids"] == frozenset({report.document_id})
    assert tuple(r.document_id for r in observed[0]["expected_sources"]) == (report.document_id,)
    assert observed[0]["expected_sources"][0].sha256 == report.pdf_sha256


@pytest.mark.parametrize("scope", [frozenset(), frozenset({"missing"}), {"report"}])
def test_invalid_scope_is_rejected_before_statement_factory(tmp_path, monkeypatch, scope):
    module = serving()
    a = Artifact(tmp_path)
    def unexpected(*args, **kwargs):
        pytest.fail("retriever fit before scope validation")
    monkeypatch.setattr("nvidia_rag.retrieval.TfidfRetriever", unexpected)
    with pytest.raises(ValueError):
        module.create_corpus_app({"NVIDIA_RAG_API_TOKEN": TOKEN}, artifact_dir=a.directory,
            source_registry=a.registry, threshold=0.0, top_k=5, document_ids=scope,
            statement_source_dir=tmp_path, statement_fiscal_declarations=tmp_path / "absent")


@pytest.mark.parametrize("target", ["registry", "manifest.json", "chunks.jsonl", "build-receipt.json"])
@pytest.mark.parametrize("stage", ["load", "service", "app"])
def test_startup_rechecks_exact_corpus_bytes_after_all_construction(tmp_path, monkeypatch, target, stage):
    a = Artifact(tmp_path)
    module = serving()
    path = a.registry if target == "registry" else a.directory / target
    name = {"load": "load_verified_corpus", "service": "build_corpus_service", "app": "create_app"}[stage]
    actual = getattr(module, name)
    def mutate(*args, **kwargs):
        result = actual(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(module, name, mutate)
    with pytest.raises(ValueError, match="changed|closure"):
        create(a)


@pytest.mark.parametrize("marker", ["pending.json", "failure.json"])
def test_startup_rechecks_completion_markers_after_app_construction(tmp_path, monkeypatch, marker):
    a = Artifact(tmp_path)
    module = serving()
    actual = module.create_app
    def mark(*args, **kwargs):
        app = actual(*args, **kwargs)
        (a.directory / marker).write_bytes(b"not complete")
        return app
    monkeypatch.setattr(module, "create_app", mark)
    with pytest.raises(ValueError, match="pending|failed|marker"):
        create(a)


@pytest.mark.parametrize("target", ["PDF", "declarations"])
def test_statement_snapshot_is_rechecked_after_asgi_app_construction(statement_corpus, monkeypatch, target):
    a, report, declarations = statement_corpus
    module = serving()
    actual = module.create_app
    def mutate(*args, **kwargs):
        app = actual(*args, **kwargs)
        path = report.pdf_path if target == "PDF" else declarations
        path.write_bytes(path.read_bytes() + b"\n")
        return app
    monkeypatch.setattr(module, "create_app", mutate)
    with pytest.raises(ValueError, match="changed|mismatch"):
        create(a, statement_source_dir=report.pdf_path.parent,
               statement_fiscal_declarations=declarations)


def test_packaged_cli_has_explicit_host_and_paired_statement_arguments(tmp_path):
    module = serving()
    default = module.parse_args(arguments(tmp_path))
    assert default.host == "127.0.0.1" and default.port == 8000 and default.top_k == 5
    args = module.parse_args(arguments(tmp_path) + ["--host", "0.0.0.0", "--port", "9000",
        "--document-id", "report", "--statement-source-dir", str(tmp_path),
        "--statement-fiscal-declarations", str(tmp_path / "fiscal.json")])
    assert args.host == "0.0.0.0" and args.port == 9000 and args.document_id == ["report"]
    assert args.statement_source_dir == tmp_path
    assert args.statement_fiscal_declarations == tmp_path / "fiscal.json"


@pytest.mark.parametrize("extra", [["--port", "0"], ["--port", "65536"], ["--top-k", "0"],
    ["--threshold", "nan"], ["--threshold", "1.1"], ["--host", " "],
    ["--statement-source-dir", "."], ["--statement-fiscal-declarations", "fiscal.json"]])
def test_packaged_cli_rejects_invalid_launch_configuration(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        serving().parse_args(arguments(tmp_path) + extra)
    assert error.value.code == 2


def test_packaged_main_uses_bounded_runtime_and_passes_optional_configuration(statement_corpus, monkeypatch):
    a, report, declarations = statement_corpus
    module = serving()
    import uvicorn
    observed = []
    monkeypatch.setenv("NVIDIA_RAG_API_TOKEN", TOKEN)
    monkeypatch.setattr(sys, "argv", ["nvidia-rag-serve-corpus"] + arguments(a.registry.parent) +
        ["--host", "0.0.0.0", "--port", "9010", "--statement-source-dir", str(report.pdf_path.parent),
         "--statement-fiscal-declarations", str(declarations)])
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: observed.append((app, kwargs)))
    module.main()
    app, config = observed[0]
    assert config == {"host": "0.0.0.0", "port": 9010, "access_log": False,
        "proxy_headers": False, "server_header": False, "limit_concurrency": 16, "timeout_keep_alive": 5}
    response = TestClient(app).post("/v1/statement-candidates", headers=HEADERS, json={"question": QUESTION})
    assert response.status_code == 200 and response.json()["cells"][0]["amount"] == "2860"


def test_module_entrypoint_runs_same_bounded_server(tmp_path, monkeypatch):
    serving()
    a = Artifact(tmp_path)
    import uvicorn
    observed = []
    monkeypatch.setenv("NVIDIA_RAG_API_TOKEN", TOKEN)
    monkeypatch.setattr(sys, "argv", ["nvidia_rag.serving"] + arguments(a.registry.parent))
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: observed.append(kwargs))
    # Model a fresh interpreter's -m execution without leaving a second module
    # instance or triggering runpy's already-imported-module warning in pytest.
    monkeypatch.delitem(sys.modules, "nvidia_rag.serving")
    runpy.run_module("nvidia_rag.serving", run_name="__main__")
    assert observed[0]["host"] == "127.0.0.1" and observed[0]["limit_concurrency"] == 16


def test_console_script_is_installed_from_package_configuration():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    assert config["project"].get("scripts", {}).get("nvidia-rag-serve-corpus") == "nvidia_rag.serving:main"


def test_compatibility_recipe_keeps_loopback_and_statement_option_exports(tmp_path):
    namespace = runpy.run_path(str(ROOT / "examples" / "serve_corpus.py"))
    with pytest.raises(SystemExit) as error:
        namespace["parse_args"](arguments(tmp_path) + ["--host", "0.0.0.0"])
    assert error.value.code == 2
    args = namespace["parse_args"](arguments(tmp_path) + ["--statement-source-dir", str(tmp_path),
        "--statement-fiscal-declarations", str(tmp_path / "fiscal.json")])
    assert args.statement_source_dir == tmp_path
    assert callable(namespace["main"]) and callable(namespace["create_corpus_app"])
