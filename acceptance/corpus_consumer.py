"""Actual installed-wheel lexical consumer comparison over operator financial files.

The four inputs are development-exposed assistant candidates. This verifies
installed modules, corpus identities and HTTP equivalence, not financial quality.
Each foreground child is exclusively owned; no existing listener is touched.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
import math
import os
from pathlib import Path
import runpy
import secrets
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time
import zipfile

import httpx
import nvidia_rag

from nvidia_rag.corpus import build_corpus_service, load_verified_corpus
from nvidia_rag.sources import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_installed_wheel(wheel: Path) -> dict:
    require(wheel.is_absolute() and wheel.is_file(), "--wheel must be an absolute existing wheel")
    require(wheel.name == "nvidia_financial_rag-0.1.0-py3-none-any.whl", "unexpected named wheel")
    package = Path(nvidia_rag.__file__).resolve().parent
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    require(package.parent == purelib, "run with the wheel installed in the consumer purelib")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "duplicate wheel ZIP member")
        modules = {name for name in names if name.startswith("nvidia_rag/") and name.endswith(".py")}
        require(modules and "nvidia_rag/corpus.py" in modules, "wheel corpus interface is absent")
        require(all(len(Path(name).parts) == 2 for name in modules), "unsupported nested package module")
        installed = {"nvidia_rag/" + path.name for path in package.glob("*.py")}
        require(installed == modules, "wheel/installed module set mismatch")
        hashes = {}
        for name in sorted(modules):
            expected = sha256(archive.read(name)).hexdigest()
            path = package / Path(name).name
            require(path.resolve().parent == package and file_sha256(path) == expected,
                    "installed module differs from named wheel")
            hashes[path.name] = expected
    return {"wheel_basename": wheel.name, "wheel_bytes": wheel.stat().st_size,
            "wheel_sha256": file_sha256(wheel), "module_sha256": hashes,
            "distribution_version": metadata.version("nvidia-financial-rag")}


def compare_evidence_response(payload, context, *, status_code, request_id_header, expected_request_id):
    require(status_code == 200 and isinstance(payload, dict), "query HTTP outcome mismatch")
    require(payload.get("request_id") == request_id_header == expected_request_id, "request ID/header mismatch")
    require(payload.get("can_answer") is context.can_answer and payload.get("evidence_passed") is context.can_answer,
            "evidence gate serialization mismatch")
    require(payload.get("answer") == "" and payload.get("error_code") is None, "unexpected generated/failed answer")
    require(payload.get("message") == context.message and payload.get("citations") == list(context.citations),
            "context/citation serialization mismatch")
    actual = payload.get("evidence")
    require(type(actual) is list and len(actual) == len(context.evidence), "evidence count mismatch")
    rows = []
    for rank, (item, expected) in enumerate(zip(actual, context.evidence, strict=True), 1):
        chunk = expected.chunk
        require(type(item) is dict and type(item.get("rank")) is int and item["rank"] == rank,
                "evidence rank mismatch")
        for key, wanted in (("chunk_id", chunk.chunk_id), ("document_id", chunk.document_id),
                            ("pdf_page", chunk.page_number), ("text", chunk.text)):
            require(type(item.get(key)) is type(wanted) and item[key] == wanted, "evidence source/text mismatch")
        score = item.get("score")
        require(type(score) in (int, float) and math.isfinite(score)
                and math.isclose(score, expected.score, rel_tol=1e-12, abs_tol=1e-12), "evidence score mismatch")
        require(item.get("provenance") is None, "legacy provenance must not be invented")
        rows.append({"rank": rank, "chunk_id": chunk.chunk_id, "document_id": chunk.document_id,
                     "physical_page": chunk.page_number, "score": score,
                     "zero_score_candidate": score == 0, "text_sha256": sha256(chunk.text.encode()).hexdigest()})
    return {"http_status": status_code, "request_id": expected_request_id,
            "evidence_passed": context.can_answer, "answer_empty": True,
            "citations": list(context.citations), "evidence": rows}


def _stop_owned(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        stopped = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                 capture_output=True, timeout=10, check=False)
        require(stopped.returncode == 0 or process.poll() is not None, "owned process tree did not stop")
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def verify_owned_teardown(process, port):
    require(process.poll() is not None, "owned child remains running instead of terminal")
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            require(False, "temporary listener remains reachable; no unrelated listener will be stopped")
    except OSError:
        pass


def _input_snapshot(inputs, artifact_dir):
    require(not os.path.lexists(artifact_dir / "pending.json"), "corpus completion is pending")
    require(not os.path.lexists(artifact_dir / "failure.json"), "corpus completion has failed")
    result = {name: file_sha256(path) for name, path in inputs.items()}
    require(not os.path.lexists(artifact_dir / "pending.json")
            and not os.path.lexists(artifact_dir / "failure.json"), "corpus completion state changed")
    return result


def _run_scope(corpus, *, artifact_dir, source_registry, questions, query_api, scope, startup_timeout):
    # Build the reference first, before starting any listener.
    service = build_corpus_service(corpus, threshold=0.0, top_k=5, document_ids=scope)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    token = secrets.token_urlsafe(32)
    environment = os.environ.copy()
    environment["NVIDIA_RAG_API_TOKEN"] = token
    environment.pop("NVIDIA_RAG_MODEL_TOKEN", None)
    command = [sys.executable, "-I", "-B", str(ROOT / "examples/serve_corpus.py"),
               "--artifact-dir", str(artifact_dir), "--source-registry", str(source_registry),
               "--threshold", "0.0", "--top-k", "5", "--port", str(port)]
    for doc_id in sorted(scope or ()):
        command.extend(["--document-id", doc_id])
    process = None
    records = []
    # Only a disposable cwd and the explicitly started child belong to this operation.
    with tempfile.TemporaryDirectory(prefix="nvidia-rag-corpus-consumer-") as cwd, tempfile.TemporaryFile() as log:
        try:
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT)
            observed = []
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5.0, follow_redirects=False,
                              trust_env=False, event_hooks={"response": [observed.append]}) as client:
                headers = {"Authorization": "Bearer " + token}
                deadline = time.monotonic() + startup_timeout
                while True:
                    require(process.poll() is None, "owned launcher exited before authenticated readiness")
                    try:
                        ready = client.get("/readyz", headers=headers)
                        if ready.status_code == 200:
                            require(ready.json() == {"status": "ready"}, "readiness response mismatch")
                            break
                    except httpx.TransportError:
                        pass
                    require(time.monotonic() < deadline, "launcher startup observation deadline reached")
                    time.sleep(0.1)
                require(client.get("/healthz").status_code == 200, "liveness check failed")
                require(client.get("/readyz").status_code == 401, "readiness authentication control failed")
                require(client.post("/v1/query", json={"question": "control"}).status_code == 401,
                        "query authentication control failed")
                require(client.post("/v1/query", headers=headers, json={"question": " "}).status_code == 422,
                        "query validation control failed")
                require(client.get("/docs").status_code == client.get("/openapi.json").status_code == 404,
                        "documentation should be disabled")
                for index, question in enumerate(questions, 1):
                    request_id = f"corpus-{'global' if scope is None else 'q2'}-{index}"
                    observed.clear()
                    client.headers["X-Request-ID"] = request_id
                    payload = query_api(client, token=token, question=question["query"])
                    require(len(observed) == 1, "unexpected query response count")
                    response = observed[0]
                    row = compare_evidence_response(payload, service.query(question["query"]),
                        status_code=response.status_code, request_id_header=response.headers.get("X-Request-ID"),
                        expected_request_id=request_id)
                    row.update(question_id=question["question_id"],
                               original_question_record_sha256=question["original_question_record_sha256"],
                               development_exposed=True, financial_field_support="not_evaluated")
                    records.append(row)
        finally:
            if process is not None:
                _stop_owned(process)
                verify_owned_teardown(process, port)
    return {"document_scope": None if scope is None else sorted(scope), "threshold": 0.0,
            "top_k": 5, "scope": "ungated lexical inspection including zero-score candidates",
            "transport_controls": {"liveness": 200, "readiness_unauthenticated": 401,
                                   "query_unauthenticated": 401, "query_blank": 422, "docs": 404},
            "records": records}


def run_comparison(*, artifact_dir: Path, source_registry: Path, candidates: Path,
                   wheel: Path, output: Path, startup_timeout: float = 60.0):
    require(output.is_absolute() and not os.path.lexists(output), "comparison output must be absolute and not exist")
    require(artifact_dir.is_absolute() and source_registry.is_absolute() and candidates.is_absolute()
            and wheel.is_absolute(), "comparison inputs must be absolute local paths")
    resolved_output = output.resolve()
    require(not resolved_output.is_relative_to(artifact_dir.resolve())
            and resolved_output not in {source_registry.resolve(), candidates.resolve(), wheel.resolve()},
            "comparison output overlaps read-only corpus/inputs")
    require(type(startup_timeout) in (int, float) and math.isfinite(startup_timeout) and startup_timeout > 0,
            "startup timeout must be finite and positive")
    require(sys.prefix != sys.base_prefix, "run comparison in the independent installed-wheel environment")
    identity = verify_installed_wheel(wheel)
    inputs = {"registry": source_registry, "candidates": candidates,
              **{name: artifact_dir / name for name in ("manifest.json", "chunks.jsonl", "build-receipt.json")}}
    before = _input_snapshot(inputs, artifact_dir)
    corpus = load_verified_corpus(artifact_dir=artifact_dir, source_registry=source_registry)
    replay = runpy.run_path(str(ROOT / "acceptance/source_binding_replay.py"))
    questions = replay["load_candidates"](candidates)
    require(before == _input_snapshot(inputs, artifact_dir),
            "comparison input changed during corpus/candidate loading")
    require(corpus.chunks_sha256 == before["chunks.jsonl"], "consumed corpus differs from input snapshot")
    require(len(corpus.sources) == 8 and "nvidia-fy2025-q2" in {row.document_id for row in corpus.sources},
            "this comparison requires the exact eight-document recipe")
    require(before["registry"] == "74379f0ecb5b131ca8562d2713a13147dc538047e7f2e2b03212ca6f2f7ed495",
            "comparison source registry differs from the frozen eight-report recipe")
    query_api = runpy.run_path(str(ROOT / "examples/query_client.py"))["query_api"]
    results = []
    try:
        for scope in (None, frozenset({"nvidia-fy2025-q2"})):
            results.append(_run_scope(corpus, artifact_dir=artifact_dir, source_registry=source_registry,
                questions=questions, query_api=query_api, scope=scope, startup_timeout=startup_timeout))
    finally:
        require(before == _input_snapshot(inputs, artifact_dir), "read-only comparison input changed")
        require(verify_installed_wheel(wheel) == identity, "installed wheel identity changed during comparison")
    receipt = {"schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
               "scope": "installed real-corpus lexical HTTP equivalence; no financial quality score",
               "installed_identity": identity, "input_sha256": before,
               "generation_runs": 0, "human_acceptance": False, "development_exposed": True,
               "financial_field_support": "not_evaluated", "comparisons": results}
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        import json
        json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("artifact-dir", "source-registry", "candidates", "wheel", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    args = parser.parse_args()
    try:
        run_comparison(**vars(args))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print("PASS: installed-wheel/corpus identity and actual lexical HTTP comparison; financial quality not evaluated")


if __name__ == "__main__":
    main()
