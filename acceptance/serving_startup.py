"""Installed CLI plus real HTTP over native synthetic PDFs, never financial gold.

The fixture corpus is an unsigned consistency fixture assembled from native PDF
text under the loader's fixed profile, not a tokenizer/build performance result.
All children belong to this acceptance; external mode never stops its listeners.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata, util
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time
from urllib.parse import urlsplit
import zipfile


REPORTS = (
    {"document_id": "synthetic-annual", "filename": "annual.pdf", "fiscal_quarter": None,
     "amount": "9207", "value_literal": "9,207", "period_end": "2025-01-26",
     "period_kind": "fiscal_year", "duration_months": None,
     "question": "What was net income for the year ended January 26, 2025, in millions of dollars?"},
    {"document_id": "synthetic-quarter", "filename": "quarter.pdf", "fiscal_quarter": 2,
     "amount": "4461", "value_literal": "4,461", "period_end": "2024-07-28",
     "period_kind": "months", "duration_months": 3,
     "question": "What was net income for the three months ended July 28, 2024, in millions of dollars?"},
)
PROFILE = {"name": "plain-word-safe-400-0-v1", "extraction_mode": "plain",
           "strategy": "word_safe", "max_tokens": 400, "overlap_tokens": 0,
           "encoding_name": "cl100k_base"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _generated_pdf(path, *, annual):
    import pymupdf
    with pymupdf.open() as pdf:
        pdf.new_page()
        page = pdf.new_page()
        def text(x, y, value):
            page.insert_text((x, y), value, fontsize=8)
        title = "Consolidated Statements of Income" if annual else "Condensed Consolidated Statements of Income"
        text(175, 110, title)
        text(205, 130, "(In millions, except per share data)")
        glyphs, rights = ([300, 400, 500], [390, 490, 590]) if annual else ([240, 330, 420, 510], [315, 405, 495, 585])
        if annual:
            text(420, 150, "Year Ended")
            dates = ["Jan 26, 2025", "Jan 28, 2024", "Jan 29, 2023"]
        else:
            text(290, 150, "Three Months Ended")
            text(470, 150, "Six Months Ended")
            dates = ["Jul 28, 2024", "Jul 30, 2023"] * 2
        for x, value in zip(glyphs, dates, strict=True):
            text(x + 15, 180, value)
        def row(label, y, values, *, dollars=False):
            text(20, y, label)
            for x, right, value in zip(glyphs, rights, values, strict=True):
                if dollars:
                    text(x, y, "$")
                text(right - pymupdf.get_text_length(value, fontsize=8), y, value)
        row("Revenue", 210, ["14,907", "12,501", "10,777"] if annual else ["14,907", "12,501", "10,777", "8,501"], dollars=True)
        row("Net income", 250, ["9,207", "8,031", "7,024"] if annual else ["4,461", "3,502", "8,004", "6,913"], dollars=True)
        text(20, 300, "Net income per share:")
        row("Basic", 330, ["1.37", "1.08", "0.91"] if annual else ["1.37", "1.08", "0.91", "0.77"], dollars=True)
        row("Diluted", 350, ["1.29", "1.01", "0.89"] if annual else ["1.29", "1.01", "0.89", "0.75"], dollars=True)
        text(20, 400, "Weighted average shares used in per share computation:")
        row("Basic", 430, ["6,701", "6,612", "6,521"] if annual else ["6,701", "6,612", "6,521", "6,501"])
        row("Diluted", 450, ["7,137", "7,087", "7,001"] if annual else ["7,137", "7,087", "7,001", "6,981"])
        text(100, 500, "See accompanying Notes to the Consolidated Financial Statements." if annual else
             "See accompanying Notes to Condensed Consolidated Financial Statements.")
        pdf.save(path)


def create_fixture(root):
    """Create a new disposable native fixture without invoking a tokenizer/model."""
    import pymupdf
    import nvidia_rag
    require(root.is_absolute() and not os.path.lexists(root), "fixture must be absolute and not exist")
    root.mkdir()
    source_dir, artifact_dir = root / "pdfs", root / "corpus"
    source_dir.mkdir()
    artifact_dir.mkdir()
    sources, documents, chunks, report_rows = [], [], [], []
    for definition in REPORTS:
        path = source_dir / definition["filename"]
        _generated_pdf(path, annual=definition["fiscal_quarter"] is None)
        with pymupdf.open(path) as pdf:
            require(len(pdf) == 2 and not pdf[0].get_text().strip(), "native fixture physical pages differ")
            native = pdf[1].get_text("text", sort=True)
        start = re.search(r"\S", native).start()
        text = native[start:].rstrip()
        chunk_digest = sha256(f"{definition['document_id']}|2|word_safe|{start}|{text}".encode()).hexdigest()[:16]
        chunks.append({"chunk_id": definition["document_id"] + ":p2:w:" + chunk_digest,
                       "document_id": definition["document_id"], "page_number": 2, "text": text})
        source = {"document_id": definition["document_id"], "filename": definition["filename"],
                  "sha256": digest(path), "label": "Generated native transport fixture",
                  "url": "https://s201.q4cdn.com/141608511/files/generated/" + definition["filename"]}
        sources.append(source)
        documents.append({"document_id": source["document_id"], "document_label": source["label"],
            "source_filename": source["filename"], "source_url": source["url"], "source_sha256": source["sha256"],
            "page_count": 2, "nonempty_page_count": 1, "chunk_count": 1})
        report_rows.append({**definition, "pdf_sha256": source["sha256"], "physical_page": 2,
                            "page_text_sha256": sha256(native.encode()).hexdigest()})
    registry, declarations = root / "sources.json", root / "fiscal-declarations.json"
    write_json(registry, {"schema_version": 1, "sources": sources})
    write_json(declarations, {"schema_version": 1, "source_registry_sha256": digest(registry),
        "reports": [{"document_id": row["document_id"], "pdf_sha256": row["pdf_sha256"], "fiscal_year": 2025,
                     "fiscal_quarter": row["fiscal_quarter"], "reviewed_by": "Generated fixture author; no human financial review",
                     "source_locator": "Synthetic native PDF physical page 2"} for row in report_rows]})
    with (artifact_dir / "chunks.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in chunks:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    counts = {"document_count": 2, "page_count": 4, "nonempty_page_count": 2, "chunk_count": 2}
    manifest = {"schema_version": 1, "chunking": {key: PROFILE[key] for key in
        ("encoding_name", "max_tokens", "overlap_tokens", "strategy")}, "documents": documents,
        "chunks_sha256": digest(artifact_dir / "chunks.jsonl"), **counts}
    write_json(artifact_dir / "manifest.json", manifest)
    package = Path(nvidia_rag.__file__).resolve().parent
    write_json(artifact_dir / "build-receipt.json", {"schema_version": 1, "status": "complete",
        "scope": "operator-local-corpus-build", "profile": PROFILE, "human_acceptance": False,
        "source_registry_sha256": digest(registry), "manifest_sha256": digest(artifact_dir / "manifest.json"),
        "chunks_sha256": manifest["chunks_sha256"], "observed_counts": counts,
        "runtime": {"python": platform.python_version(), **{name: metadata.version(name) for name in
                    ("PyMuPDF", "numpy", "scikit-learn", "tiktoken")}},
        "module_sha256": {name: digest(package / name) for name in ("core.py", "ingest.py", "sources.py", "corpus.py")},
        "sources": [{"document_id": row["document_id"], "source_sha256": row["sha256"],
                     "source_bytes": (source_dir / row["filename"]).stat().st_size} for row in sources]})
    snapshot = {str(path.relative_to(root)).replace("\\", "/"): digest(path)
                for path in sorted(root.rglob("*")) if path.is_file()}
    write_json(root / "fixture-receipt.json", {"schema_version": 1, "input_sha256": snapshot,
        "reports": report_rows, "chunks": chunks, "tokenizer_runs": 0,
        "scope": "manually assembled unsigned native synthetic PDF consistency fixture; no tokenizer build"})
    return {"root": root, "artifact_dir": artifact_dir, "source_dir": source_dir,
            "source_registry": registry, "declarations": declarations}


def verify_fixture(root):
    import pymupdf
    require(root.is_absolute() and root.is_dir(), "fixture must be an absolute existing directory")
    proof = json.loads((root / "fixture-receipt.json").read_bytes())
    require(proof.get("schema_version") == 1 and proof.get("tokenizer_runs") == 0, "fixture schema differs")
    expected = proof["input_sha256"]
    actual_paths = {str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*")
                    if path.is_file() and path.name != "fixture-receipt.json"}
    require(set(expected) == actual_paths, "fixture input set changed")
    for relative, wanted in expected.items():
        path = root / relative
        require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), "unsafe fixture file")
        require(digest(path) == wanted, "fixture input hash changed: " + relative)
    require(len(proof["reports"]) == len(REPORTS) == len(proof["chunks"]), "fixture report/chunk count differs")
    for definition, report, chunk in zip(REPORTS, proof["reports"], proof["chunks"], strict=True):
        require(all(report[key] == wanted for key, wanted in definition.items()), "fixture expected literals changed")
        pdf_path = root / "pdfs" / report["filename"]
        require(digest(pdf_path) == report["pdf_sha256"], "native fixture PDF hash differs")
        with pymupdf.open(pdf_path) as pdf:
            require(len(pdf) == 2 and not pdf[0].get_text().strip(), "fixture physical pages differ")
            native = pdf[1].get_text("text", sort=True)
        require(sha256(native.encode()).hexdigest() == report["page_text_sha256"], "native page hash differs")
        require(chunk["text"] == native.strip() and chunk["document_id"] == report["document_id"] and
                chunk["page_number"] == report["physical_page"] == 2, "native chunk text or identity differs")
        start = re.search(r"\S", native).start()
        suffix = sha256(f"{report['document_id']}|2|word_safe|{start}|{chunk['text']}".encode()).hexdigest()[:16]
        require(chunk["chunk_id"] == report["document_id"] + ":p2:w:" + suffix, "native chunk ID differs")
    return proof


def verify_source_cell(cell, expected, source_dir):
    """Replay literals, character spans and geometry without calling the selector."""
    import pymupdf
    from dataclasses import fields
    from nvidia_rag.structured_income import ANNUAL_PROFILE, PROFILE, IncomeCell
    require(set(cell) == {field.name for field in fields(IncomeCell)} | {'period_kind'},
            'source cell DTO fields differ')
    require(cell['parser_profile'] == (ANNUAL_PROFILE if expected['fiscal_quarter'] is None else PROFILE)
            and cell['pymupdf_version'] == pymupdf.version[0], 'source parser provenance differs')
    for key in ("document_id", "pdf_sha256", "physical_page", "page_text_sha256", "fiscal_quarter",
                "period_end", "period_kind", "duration_months", "amount", "value_literal"):
        require(type(cell.get(key)) is type(expected[key]) and cell[key] == expected[key], "source cell differs: " + key)
    for key, value in (("fiscal_year", 2025), ("row_path", "monetary/net_income"), ("row_literal", "Net income"),
                       ("column_number", 1), ("column_role", "current"), ("scale", "1000000"),
                       ("unit_kind", "monetary"), ("currency_literal", "$"),
                       ("currency_normalization", "dollar-symbol;ISO-currency-unverified")):
        require(cell.get(key) == value, "source field differs: " + key)
    path = source_dir / expected["filename"]
    require(digest(path) == cell["pdf_sha256"], "source PDF hash changed")
    with pymupdf.open(path) as pdf:
        page = pdf[cell["physical_page"] - 1]
        require(cell['pdf_page_label'] == (page.get_label() or None), 'source PDF page label differs')
        native, words = page.get_text("text", sort=True), page.get_text("words", sort=True)
    require(sha256(native.encode()).hexdigest() == cell["page_text_sha256"], "source page text hash differs")
    bindings = cell["bindings"]
    require(len({pair[0] for pair in bindings}) == len(bindings), "duplicate source bindings")
    require({pair[0] for pair in bindings} == {'statement', 'unit', 'row', 'date', 'duration', 'value', 'currency'},
            'source mandatory bindings differ')
    for name, span in bindings:
        require(type(span["start"]) is int and type(span["end"]) is int and
                0 <= span["start"] < span["end"] <= len(native) and
                native[span["start"]:span["end"]] == span["literal"], "source character span mismatch")
        box = span["bbox"]
        require(len(box) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in box), "invalid source bbox")
        contained = [word for word in words if word[0] >= box[0] - .001 and word[1] >= box[1] - .001 and
                     word[2] <= box[2] + .001 and word[3] <= box[3] + .001]
        require(contained and " ".join(word[4] for word in contained) == " ".join(span["literal"].split()),
                "source literal geometry mismatch")
        bounds = (min(w[0] for w in contained), min(w[1] for w in contained),
                  max(w[2] for w in contained), max(w[3] for w in contained))
        require(all(abs(a - b) <= .001 for a, b in zip(box, bounds, strict=True)), "source bbox envelope mismatch")
    require(dict(bindings)["value"]["literal"] == expected["value_literal"], "source value literal differs")
    identity = {key: value for key, value in cell.items() if key not in {"cell_id", "period_kind"}}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    require(cell["cell_id"] == sha256(canonical).hexdigest(), "source cell identity differs")
    return {"cell_id": cell["cell_id"], "amount": cell["amount"], "physical_page": 2,
            "pdf_sha256": cell["pdf_sha256"], "page_text_sha256": cell["page_text_sha256"],
            "native_span_count": len(bindings), "native_geometry_replayed": True}


def installed_identity(wheel):
    import nvidia_rag
    require(sys.flags.isolated and sys.flags.dont_write_bytecode, "use Python -I -B")
    require(wheel.is_absolute() and wheel.is_file(), "wheel must be an absolute existing archive")
    package = Path(nvidia_rag.__file__).resolve().parent
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    require(package.parent == purelib and not package.is_relative_to(Path(__file__).resolve().parents[1]),
            "production package must come from installed purelib outside checkout")
    for name in ("torch", "transformers", "sentence_transformers", "gradio"):
        require(util.find_spec(name) is None, "unexpected heavyweight dependency: " + name)
    hashes, paths = {}, {}
    with zipfile.ZipFile(wheel) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "duplicate wheel members")
        names = {name for name in archive.namelist() if name.startswith("nvidia_rag/") and name.endswith(".py")}
        require("nvidia_rag/serving.py" in names, "installed startup module missing from wheel")
        installed = {"nvidia_rag/" + path.name for path in package.glob("*.py")}
        require(names == installed and all(len(Path(name).parts) == 2 for name in names), "wheel/installed module set differs")
        for name in sorted(names):
            path = package / Path(name).name
            expected = sha256(archive.read(name)).hexdigest()
            require(path.resolve().parent == package and digest(path) == expected, "installed production module differs from wheel")
            hashes[name] = expected
            paths[name] = str(path)
    return {"wheel_sha256": digest(wheel), "wheel_bytes": wheel.stat().st_size,
            "distribution": metadata.version("nvidia-financial-rag"), "purelib": str(purelib),
            "module_sha256": hashes, "module_paths": paths, "isolated": True,
            "heavyweight_dependencies_absent": True}


def local_base_url(value):
    url = urlsplit(value)
    require(url.scheme == "http" and url.hostname in {"127.0.0.1", "localhost", "::1"} and
            not url.username and not url.password and not url.query and not url.fragment and url.path in {"", "/"},
            "acceptance requires an explicit loopback HTTP listener")
    require(url.port is not None, "acceptance requires an explicit loopback port")
    return value.rstrip("/")


def validate_output(output, fixture_root):
    require(output.is_absolute() and output.parent.is_dir() and not os.path.lexists(output), "output must be absolute and not exist")
    require(not output.resolve().is_relative_to(fixture_root.resolve()), "output must not overlap read-only fixture")


def _http_checks(base_url, token, fixture_root, *, scoped=False, unconfigured=False, startup_timeout=60):
    import httpx
    proof = verify_fixture(fixture_root)
    expected_index_hash = None
    expected_cells = {}
    if not unconfigured:
        from dataclasses import asdict
        from nvidia_rag.structured_income import ReportInput, build_income_index
        reports = tuple(ReportInput(row['document_id'], fixture_root / 'pdfs' / row['filename'],
            row['pdf_sha256'], 2025, row['fiscal_quarter']) for row in proof['reports']
            if not scoped or row['document_id'] == 'synthetic-quarter')
        expected_index = build_income_index(reports)
        canonical = asdict(expected_index)
        canonical.pop('_construction_token')
        expected_index_hash = sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode() + b'\n').hexdigest()
        expected_cells = {cell.cell_id: json.loads(json.dumps({**asdict(cell), 'period_kind': cell.period_kind},
            allow_nan=False)) for cell in expected_index.cells}
    records = []
    def checked(name, condition, **extra):
        require(condition, name)
        records.append({"check": name, **extra})
    headers = {"Authorization": "Bearer " + token, "X-Request-ID": "installed-startup-http"}
    with httpx.Client(base_url=local_base_url(base_url), timeout=5, follow_redirects=False, trust_env=False) as client:
        deadline = time.monotonic() + startup_timeout
        while True:
            try:
                response = client.get("/readyz", headers=headers)
                if response.status_code == 200:
                    checked("authenticated-readiness", response.json() == {"status": "ready"}, http_status=200)
                    break
            except httpx.TransportError:
                pass
            require(time.monotonic() < deadline, "listener readiness observation deadline reached")
            time.sleep(.1)
        checked("liveness", client.get("/healthz").status_code == 200, http_status=200)
        checked("readiness-authentication", client.get("/readyz").status_code == 401, http_status=401)
        checked("docs-disabled", client.get("/docs").status_code == client.get("/openapi.json").status_code == 404, http_status=404)
        for path in ("/v1/query", "/v1/statement-candidates", "/v1/statement-fields"):
            response = client.post(path, content=b"not-json")
            checked(path + "-authentication-before-malformed-body", response.status_code == 401, http_status=401)
        if unconfigured:
            response = client.post("/v1/statement-candidates", headers=headers, json={"question": REPORTS[0]["question"]})
            body = response.json()
            checked("unconfigured-is-operational-503", response.status_code == 503 and body["disposition"] == "unavailable" and
                    body["reason"] == "STATEMENT_INDEX_NOT_CONFIGURED" and body["index_sha256"] is None and
                    body["cells"] == [] and body["error_code"] == "statement_selection_unavailable", http_status=503)
            typed = client.post('/v1/statement-fields', headers=headers, content=b'not-json')
            typed_body = typed.json()
            checked('unconfigured-typed-fields-operational-503', typed.status_code == 503 and
                typed_body['disposition'] == 'unavailable' and typed_body['cells'] == [] and
                typed_body['reason'] == 'STATEMENT_INDEX_NOT_CONFIGURED' and
                typed_body['selection_request'] is None and typed_body['selection_request_sha256'] is None and
                typed_body['index_sha256'] is None, http_status=503)
            return records
        indexes = set()
        for expected in proof["reports"]:
            if scoped and expected["document_id"] == "synthetic-annual":
                continue
            response = client.post("/v1/statement-candidates", headers=headers,
                json={"question": expected["question"], "document_ids": [expected["document_id"]]})
            body = response.json()
            checked(expected["period_kind"] + "-source-candidate", response.status_code == 200 and
                body["disposition"] == "source_candidate" and len(body["cells"]) == 1 and
                body["financial_correctness"] == "unestablished" and body["score_semantics"] == "exact-source-field-selection;no-cosine-score" and
                body["request_id"] == response.headers.get("X-Request-ID") == headers["X-Request-ID"], http_status=200,
                source=verify_source_cell(body["cells"][0], expected, fixture_root / "pdfs"))
            require(re.fullmatch(r"[0-9a-f]{64}", body["index_sha256"]) is not None, "index identity absent")
            indexes.add(body["index_sha256"])
            checked(expected['period_kind'] + '-complete-source-cell-transport',
                body['cells'][0] == expected_cells.get(body['cells'][0]['cell_id']))
        checked("one-immutable-source-index", indexes == {expected_index_hash}, index_sha256=expected_index_hash)
        for report in proof['reports']:
            if scoped and report['document_id'] == 'synthetic-annual':
                continue
            for kind, row_path in (('monetary', 'monetary/net_income'),
                                   ('per_share', 'per_share/basic'), ('count', 'count/basic')):
                native = next(cell for cell in expected_cells.values()
                    if cell['document_id'] == report['document_id'] and cell['row_path'] == row_path
                    and cell['column_number'] == 1)
                payload = {name: native[name] for name in ('row_path', 'period_kind', 'period_end',
                    'duration_months', 'unit_kind', 'scale', 'currency_literal', 'fiscal_year',
                    'fiscal_quarter', 'column_role')}
                payload['document_ids'] = [native['document_id']]
                typed = client.post('/v1/statement-fields', headers=headers, json=payload)
                typed_body = typed.json()
                request_hash = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode() + b'\n').hexdigest()
                checked(report['document_id'] + '-typed-' + kind + '-complete-native-cell',
                    typed.status_code == 200 and typed_body['disposition'] == 'source_candidate' and
                    typed_body['cells'] == [native] and typed_body['selection_request'] == payload and
                    typed_body['selection_request_sha256'] == request_hash and
                    typed_body['index_sha256'] == expected_index_hash and
                    typed_body['financial_correctness'] == 'unestablished' and
                    typed_body['score_semantics'] == 'exact-source-field-selection;no-cosine-score' and
                    typed_body['request_id'] == typed.headers.get('X-Request-ID') == headers['X-Request-ID'],
                    http_status=200, cell_id=native['cell_id'], unit_kind=kind)
        annual = REPORTS[0]["question"]
        control = REPORTS[1] if scoped else REPORTS[0]
        question = control["question"]
        variations = [(question.replace("January 26", "January 27").replace("July 28", "July 27"), "refuse"),
                      (question.replace("year ended", "annual three months ended").replace("three months ended", "annual three months ended") if scoped else
                       annual.replace("year ended", "annual three months ended"), "clarify"),
                      (question.replace("dollars", "USD"), "unsupported")]
        for change, disposition in variations:
            response = client.post("/v1/statement-candidates", headers=headers,
                json={"question": change, "document_ids": [control["document_id"]]})
            body = response.json()
            checked(disposition + "-completed-content-outcome", response.status_code == 200 and
                    body["disposition"] == disposition and body["cells"] == [] and body["error_code"] is None, http_status=200)
        duplicate = client.post("/v1/statement-candidates", headers={**headers, "Content-Type": "application/json"},
                                content=b'{"question":"first","question":"second"}')
        checked("duplicate-json-safe-422", duplicate.status_code == 422 and duplicate.json() == {"detail": "Invalid statement request."}, http_status=422)
        for query, evidence_wanted in (("net income", True), ("zxqvunmatchedword", False)):
            response = client.post("/v1/query", headers=headers, json={"question": query})
            body = response.json()
            checked("lexical-" + ("evidence" if evidence_wanted else "refusal"), response.status_code == 200 and
                    body["answer"] == "" and body["error_code"] is None and body["can_answer"] is evidence_wanted and
                    body["evidence_passed"] is evidence_wanted and body["request_id"] == response.headers.get("X-Request-ID"), http_status=200)
            if evidence_wanted:
                wanted = {row["document_id"]: row for row in proof["chunks"] if not scoped or row["document_id"] == "synthetic-quarter"}
                checked("lexical-scope-and-native-text", {row["document_id"] for row in body["evidence"]} == set(wanted) and
                    all(row["chunk_id"] == wanted[row["document_id"]]["chunk_id"] and row["pdf_page"] == 2 and
                        row["text"] == wanted[row["document_id"]]["text"] and row["provenance"] is None
                        for row in body["evidence"]), evidence_sha256=sha256(json.dumps(body["evidence"], sort_keys=True).encode()).hexdigest())
            else:
                checked("lexical-refusal-empty-evidence", body["evidence"] == [] and body["citations"] == [])
        if scoped:
            response = client.post("/v1/statement-candidates", headers=headers,
                json={"question": annual, "document_ids": ["synthetic-annual"]})
            checked("excluded-statement-source-refused", response.status_code == 200 and
                    response.json()["disposition"] == "refuse" and response.json()["cells"] == [], http_status=200)
    return records


def _stop_owned(process):
    if process.poll() is None:
        if os.name == 'nt':
            # Windows venv python.exe is a redirector. Stop only the exact owned
            # process tree so its interpreter cannot retain the listener/cwd.
            stopped = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                capture_output=True, timeout=10, check=False)
            require(stopped.returncode == 0, 'owned Windows process tree did not stop')
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    require(process.poll() is not None, "owned process did not terminate")


def _await_listener_closed(port):
    deadline = time.monotonic() + 5
    while True:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=.2):
                pass
        except OSError:
            return
        require(time.monotonic() < deadline,
                'owned listener remains reachable; unrelated listeners are never stopped')
        time.sleep(.05)


def _launch_scope(fixture_root, *, scoped=False, unconfigured=False):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [sys.executable, "-I", "-B", "-m", "nvidia_rag.serving",
        "--artifact-dir", str(fixture_root / "corpus"), "--source-registry", str(fixture_root / "sources.json"),
        "--threshold", "0.01", "--top-k", "5", "--port", str(port)]
    if not unconfigured:
        command.extend(["--statement-source-dir", str(fixture_root / "pdfs"),
                        "--statement-fiscal-declarations", str(fixture_root / "fiscal-declarations.json")])
    if scoped:
        command.extend(["--document-id", "synthetic-quarter"])
    token = secrets.token_urlsafe(32)
    environment = dict(os.environ, NVIDIA_RAG_API_TOKEN=token)
    environment.pop("NVIDIA_RAG_MODEL_TOKEN", None)
    with tempfile.TemporaryDirectory(prefix="installed-serving-cwd-") as cwd, tempfile.TemporaryFile() as log:
        require(not Path(cwd).resolve().is_relative_to(Path(__file__).resolve().parents[1]), "child cwd overlaps checkout")
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            records = _http_checks(f"http://127.0.0.1:{port}", token, fixture_root, scoped=scoped, unconfigured=unconfigured)
            require(process.poll() is None, "owned server exited unexpectedly")
        finally:
            _stop_owned(process)
            _await_listener_closed(port)
    return {"mode": "unconfigured" if unconfigured else "quarter-only" if scoped else "all-sources",
            "command": command, "cwd_outside_checkout": True, "pid": process.pid,
            "owned_process_stopped": True, "owned_listener_closed": True, "checks": records}


def token_first_preflight():
    environment = dict(os.environ)
    environment.pop("NVIDIA_RAG_API_TOKEN", None)
    with tempfile.TemporaryDirectory(prefix="installed-token-first-") as cwd:
        absent = Path(cwd) / "absent"
        command = [sys.executable, "-I", "-B", "-m", "nvidia_rag.serving", "--artifact-dir", str(absent),
                   "--source-registry", str(absent / "registry.json"), "--threshold", "0.01"]
        result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, timeout=30)
        require(result.returncode != 0 and "NVIDIA_RAG_API_TOKEN" in result.stderr and
                "FileNotFoundError" not in result.stderr and not absent.exists(), "token-first startup control failed")
    return {"check": "missing-token-before-corpus-io", "returncode": result.returncode,
            "no_input_created": True, "token_persisted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--create-fixture-dir", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--unconfigured-base-url")
    args = parser.parse_args()
    try:
        identity = installed_identity(args.wheel)
        if args.create_fixture_dir is not None:
            require(args.fixture_dir is None and args.base_url is None and args.output is None,
                    "fixture generation is a separate mode")
            create_fixture(args.create_fixture_dir)
            verify_fixture(args.create_fixture_dir)
            print("PASS: generated native synthetic PDFs and unsigned consistency fixture; no tokenizer/model runs")
            return
        require(args.output is not None, "--output is required for acceptance")
        require(args.base_url is None or args.fixture_dir is not None, "external HTTP mode requires --fixture-dir")
        require(args.unconfigured_base_url is None or args.base_url is not None, "unconfigured URL requires external mode")
        with tempfile.TemporaryDirectory(prefix="installed-serving-fixture-") as temp:
            root = args.fixture_dir or Path(temp) / "fixtures"
            validate_output(args.output, root)
            if args.fixture_dir is None:
                create_fixture(root)
            proof = verify_fixture(root)
            before = {**proof["input_sha256"], "fixture-receipt.json": digest(root / "fixture-receipt.json")}
            if args.base_url:
                token = os.environ.get("NVIDIA_RAG_API_TOKEN", "")
                require(len(token) >= 32 and token == token.strip(), "external mode requires environment-only ephemeral NVIDIA_RAG_API_TOKEN")
                results = [{"mode": "external-all-sources", "checks": _http_checks(args.base_url, token, root),
                            "owned_process_stopped": None, "scope": "external listener; caller owns teardown"}]
                if args.unconfigured_base_url:
                    results.append({"mode": "external-unconfigured", "checks": _http_checks(args.unconfigured_base_url, token, root, unconfigured=True),
                                    "owned_process_stopped": None, "scope": "external listener; caller owns teardown"})
                preflight = token_first_preflight()
            else:
                preflight = token_first_preflight()
                results = [_launch_scope(root), _launch_scope(root, scoped=True), _launch_scope(root, unconfigured=True)]
            verify_fixture(root)
            after = {**proof["input_sha256"], "fixture-receipt.json": digest(root / "fixture-receipt.json")}
            require(before == after, "read-only acceptance inputs changed")
            require(installed_identity(args.wheel) == identity, "installed wheel identity changed")
            receipt = {"schema_version": 1, "status": "passed", "created_at_utc": datetime.now(UTC).isoformat(),
                "python": platform.python_version(), "installed_identity": identity, "input_sha256": before,
                "input_unchanged": True, "startup_control": preflight, "transports": results,
                "scope": "installed CLI native synthetic PDF source selection and actual loopback HTTP",
                "fixture_construction": "manually assembled unsigned fixed-profile consistency fixture",
                "encoder_runs": 0, "generator_runs": 0, "tokenizer_runs": 0, "human_gold_count": 0,
                "financial_correctness": "unestablished", "not_accepted": ["financial answer quality", "original financial reports",
                    "public hosting", "load", "TLS", "model execution", "browser integration"]}
            write_json(args.output, receipt)
            require(json.loads(args.output.read_bytes()) == receipt, "saved acceptance receipt differs")
            print(json.dumps({"status": "passed", "production_modules": len(identity["module_sha256"]),
                "transport_modes": len(results), "checks": sum(len(row["checks"]) for row in results),
                "scope": receipt["scope"]}))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
