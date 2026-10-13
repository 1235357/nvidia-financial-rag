"""Acceptance guards fail on fixture or transport tampering, not financial truth."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import runpy

import pytest


def acceptance():
    path = Path(__file__).resolve().parents[1] / "acceptance/serving_startup.py"
    assert path.is_file(), "installed serving startup acceptance is missing"
    return runpy.run_path(str(path))


def test_generated_fixture_replays_native_pdf_chunks_and_closure(tmp_path):
    ns = acceptance()
    fixture = ns["create_fixture"](tmp_path / "fixture")
    proof = ns["verify_fixture"](fixture["root"])
    assert len(proof["reports"]) == 2 and len(proof["chunks"]) == 2
    assert [row["amount"] for row in proof["reports"]] == ["9207", "4461"]
    assert all(row["physical_page"] == 2 for row in proof["reports"])
    assert proof["tokenizer_runs"] == 0


@pytest.mark.parametrize("target", ["sources.json", "fiscal-declarations.json",
                                  "corpus/chunks.jsonl", "pdfs/annual.pdf"])
def test_fixture_rejects_changed_input_before_starting_service(tmp_path, target):
    ns = acceptance()
    fixture = ns["create_fixture"](tmp_path / "fixture")
    path = fixture["root"] / target
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed|hash|fixture"):
        ns["verify_fixture"](fixture["root"])


def test_existing_fixture_directory_is_preserved(tmp_path):
    ns = acceptance()
    root = tmp_path / "retained"
    root.mkdir()
    marker = root / "keep"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(ValueError, match="exist"):
        ns["create_fixture"](root)
    assert marker.read_text("utf-8") == "unchanged"


def test_source_cell_verifier_rejects_wrong_literal_or_identity(tmp_path):
    ns = acceptance()
    fixture = ns["create_fixture"](tmp_path / "fixture")
    proof = ns["verify_fixture"](fixture["root"])
    from nvidia_rag.structured_income import ReportInput, build_income_index
    from dataclasses import asdict
    reports = [ReportInput(row["document_id"], fixture["source_dir"] / row["filename"],
                row["pdf_sha256"], 2025, row["fiscal_quarter"]) for row in proof["reports"]]
    index = build_income_index(tuple(reports))
    source = next(cell for cell in index.cells if cell.amount == "9207")
    cell = {**asdict(source), "period_kind": source.period_kind}
    expected = proof["reports"][0]
    ns["verify_source_cell"](cell, expected, fixture["source_dir"])
    for key, value in (("amount", "9999"), ("cell_id", "0" * 64),
                       ("pdf_sha256", "1" * 64), ("physical_page", 1)):
        altered = deepcopy(cell)
        altered[key] = value
        with pytest.raises(ValueError):
            ns["verify_source_cell"](altered, expected, fixture["source_dir"])
    altered = deepcopy(cell)
    name, span = next(pair for pair in altered["bindings"] if pair[0] == "value")
    span["start"] += 1
    with pytest.raises(ValueError, match="span|literal|identity"):
        ns["verify_source_cell"](altered, expected, fixture["source_dir"])


@pytest.mark.parametrize('change', ['parser-profile', 'parser-version', 'page-label', 'missing-bindings', 'unknown-field'])
def test_source_cell_verifier_rejects_rehashed_incomplete_provenance(tmp_path, change):
    import json
    from dataclasses import asdict
    from nvidia_rag.structured_income import ReportInput, build_income_index
    ns = acceptance()
    fixture = ns['create_fixture'](tmp_path / 'fixture')
    proof = ns['verify_fixture'](fixture['root'])
    reports = tuple(ReportInput(row['document_id'], fixture['source_dir'] / row['filename'],
        row['pdf_sha256'], 2025, row['fiscal_quarter']) for row in proof['reports'])
    source = next(c for c in build_income_index(reports).cells if c.amount == '9207')
    cell = {**asdict(source), 'period_kind': source.period_kind}
    if change == 'parser-profile':
        cell['parser_profile'] = 'wrong-parser-profile'
    elif change == 'parser-version':
        cell['pymupdf_version'] = 'wrong-version'
    elif change == 'page-label':
        cell['pdf_page_label'] = 'wrong-page-label'
    elif change == 'missing-bindings':
        cell['bindings'] = [pair for pair in cell['bindings'] if pair[0] == 'value']
    else:
        cell['unsupported'] = 'must reject'
    identity = {key: value for key, value in cell.items() if key not in {'cell_id', 'period_kind'}}
    cell['cell_id'] = sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError):
        ns['verify_source_cell'](cell, proof['reports'][0], fixture['source_dir'])


@pytest.mark.parametrize("url", ["https://example.com", "http://example.com",
                               "http://user:secret@127.0.0.1:8000", "http://127.0.0.1:8000/path",
                               "http://127.0.0.1:8000?query", "http://127.0.0.1:8000#fragment"])
def test_remote_acceptance_only_contacts_explicit_local_listener(url):
    ns = acceptance()
    with pytest.raises(ValueError, match="loopback"):
        ns["local_base_url"](url)


def test_output_never_overwrites_read_only_fixture(tmp_path):
    ns = acceptance()
    fixture = ns["create_fixture"](tmp_path / "fixture")
    with pytest.raises(ValueError, match="output"):
        ns["validate_output"](fixture["root"] / "new.json", fixture["root"])
    retained = tmp_path / "retained.json"
    retained.write_text("retained", encoding="utf-8")
    with pytest.raises(ValueError, match="exist"):
        ns["validate_output"](retained, fixture["root"])
    assert sha256(retained.read_bytes()).hexdigest() == sha256(b"retained").hexdigest()


def test_http_acceptance_rejects_a_stable_but_wrong_index_digest(tmp_path, monkeypatch):
    import httpx
    from fastapi.testclient import TestClient
    from nvidia_rag.serving import create_corpus_app
    ns = acceptance()
    fixture = ns['create_fixture'](tmp_path / 'fixture')
    token = 'synthetic-digest-control-' + 'x'*32
    app = create_corpus_app({'NVIDIA_RAG_API_TOKEN': token}, artifact_dir=fixture['artifact_dir'],
        source_registry=fixture['source_registry'], threshold=.01, top_k=5, document_ids=None,
        statement_source_dir=fixture['source_dir'], statement_fiscal_declarations=fixture['declarations'])
    actual_client = httpx.Client
    class CorruptingClient(TestClient):
        def post(self, url, **kwargs):
            response = super().post(url, **kwargs)
            if url == '/v1/statement-candidates' and response.status_code == 200:
                body = response.json()
                body['index_sha256'] = '0'*64
                return httpx.Response(response.status_code, json=body, headers=response.headers)
            return response
    def client(**kwargs):
        # TestClient's own ASGI transport still uses the real httpx constructor.
        monkeypatch.setattr(httpx, 'Client', actual_client)
        return CorruptingClient(app)
    monkeypatch.setattr(httpx, 'Client', client)
    with pytest.raises(ValueError, match='immutable-source-index'):
        ns['_http_checks']('http://127.0.0.1:8917', token, fixture['root'])
