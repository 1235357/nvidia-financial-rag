"""Actual cached-CUDA launcher acceptance; no generation or quality acceptance.

Run with the independent environment's Python -I. The child receives an
ephemeral bearer token only through its environment, uses absolute read-only
artifact paths, and runs from a disposable working directory. Only this script's
own child process is stopped. Original artifacts are hashed before and after.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import nvidia_rag


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.startup_timeout <= 0:
        parser.error("startup timeout must be positive")

    root = Path(__file__).resolve().parents[1]
    package = Path(nvidia_rag.__file__).resolve().parent
    assert "site-packages" in package.parts, "Run against the installed wheel with -I"
    assert sys.prefix != sys.base_prefix, "Run in the independent virtual environment"
    print("Installed package:", package)
    print("Installed distribution version:", metadata.version("nvidia-financial-rag"))
    module_hashes = {}
    for source in sorted((root / "src" / "nvidia_rag").glob("*.py")):
        installed_hash = file_sha256(package / source.name)
        assert installed_hash == file_sha256(source), f"Installed/staged mismatch: {source.name}"
        module_hashes[source.name] = installed_hash
    print("Installed Python modules match staging SHA-256:", json.dumps(module_hashes, sort_keys=True))

    artifact = args.artifact_dir.resolve(strict=True)
    paths = {
        "--chunks": artifact / "chunks.jsonl",
        "--passage-embeddings": artifact / "bge-small-en-v1.5.npy",
        "--embedding-manifest": artifact / "embedding-manifest.json",
        "--corpus-manifest": artifact / "manifest.json",
    }
    for path in paths.values():
        assert path.is_file(), f"Missing reviewed artifact: {path.name}"
    corpus = json.loads(paths["--corpus-manifest"].read_text(encoding="utf-8"))
    assert not any("provenance" in row for row in corpus["documents"]), "This receipt is for the legacy annual corpus"
    before = {path.name: file_sha256(path) for path in paths.values()}

    # The launcher accepts a port number rather than an inherited socket. Close
    # this reservation just before spawning; a port race fails without touching
    # another process. Authenticated readiness verifies the ephemeral token.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    token = secrets.token_urlsafe(32)
    child_environment = os.environ.copy()
    child_environment["NVIDIA_RAG_API_TOKEN"] = token
    child_environment.pop("NVIDIA_RAG_MODEL_TOKEN", None)
    child_environment["HF_HUB_OFFLINE"] = "1"
    child_environment["TRANSFORMERS_OFFLINE"] = "1"
    command = [sys.executable, "-I", "-B", str(root / "run_api.py"),
               "--host", "127.0.0.1", "--port", str(port),
               "--device", "cuda", "--generator", "none"]
    for option, path in paths.items():
        command.extend([option, str(path)])

    process = None
    passed = False
    with tempfile.TemporaryDirectory(prefix="nvidia-rag-launcher-") as temporary_cwd:
        with tempfile.TemporaryFile() as child_log:
            try:
                process = subprocess.Popen(command, cwd=temporary_cwd,
                    env=child_environment, stdout=child_log, stderr=subprocess.STDOUT)
                print(f"Owned launcher PID: {process.pid}; temporary loopback port: {port}")
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5.0,
                                  follow_redirects=False, trust_env=False) as client:
                    headers = {"Authorization": f"Bearer {token}"}
                    deadline = time.monotonic() + args.startup_timeout
                    while True:
                        assert process.poll() is None, "Launcher exited before authenticated readiness"
                        try:
                            ready = client.get("/readyz", headers=headers)
                            if ready.status_code == 200:
                                assert ready.json() == {"status": "ready"}
                                break
                        except httpx.TransportError:
                            pass
                        assert time.monotonic() < deadline, "Launcher startup timed out"
                        time.sleep(0.1)
                    assert client.get("/healthz").status_code == 200
                    assert client.get("/readyz").status_code == 401
                    assert client.post("/v1/query", json={"question": "synthetic"}).status_code == 401
                    assert client.post("/v1/query", headers=headers, json={"question": "   "}).status_code == 422
                    request_headers = dict(headers, **{"X-Request-ID": "launcher-acceptance-real"})
                    response = client.post("/v1/query", headers=request_headers,
                        json={"question": "What was NVIDIA annual revenue in FY2025?"})
                    assert response.status_code == 200
                    payload = response.json()
                    assert payload["can_answer"] is True and payload["evidence_passed"] is True
                    assert payload["answer"] == ""
                    assert payload["evidence"][0]["chunk_id"] == "nvidia-fy2025-annual:p18:50cdb6ce95041a87"
                    assert all(item["provenance"] is None for item in payload["evidence"])
                    assert response.headers["X-Request-ID"] == payload["request_id"] == "launcher-acceptance-real"
                    assert float(response.headers["X-Process-Time-Ms"]) >= 0
                    assert client.get("/docs").status_code == 404
                    assert client.get("/openapi.json").status_code == 404
                print("PASS: actual run_api.py health 200, readiness auth/401, query 401/422/real 200, request ID and hidden docs")
                print("Observed real-index top chunk:", payload["evidence"][0]["chunk_id"])
                print("PASS: generation disabled; legacy evidence provenance is null")
                passed = True
            finally:
                if process is not None:
                    if process.poll() is None:
                        if os.name == "nt":
                            # A Windows venv python.exe can be a redirector whose
                            # child runs the actual interpreter. Terminating only
                            # the redirector leaks the listener and its cwd.
                            stopped = subprocess.run(
                                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=10, check=False,
                            )
                            assert stopped.returncode == 0, "Owned Windows process tree did not stop"
                            print("Teardown: forced stop of the owned Windows process tree")
                        else:
                            process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
                    print("Owned launcher exit status:", process.returncode)
                    assert process.poll() is not None, "Owned child still running"
                    child_log.seek(0)
                    log_text = child_log.read().decode("utf-8", errors="replace").replace(token, "[REDACTED]")
                    if log_text.strip():
                        print("Child log (ephemeral token redacted; last 8000 characters):")
                        print(log_text[-8000:].rstrip())
                after = {path.name: file_sha256(path) for path in paths.values()}
                assert before == after, "Original artifact bytes changed"
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        raise AssertionError("Temporary listener remains reachable; no unrelated process will be stopped")
                except OSError:
                    pass
                print("PASS: owned child stopped, temporary listener unreachable, original artifact SHA-256 unchanged")
    assert passed, "Actual launcher acceptance did not complete"
    print("NOT ACCEPTED: positive provenance attachment, generation, financial quality, cold model acquisition, TLS, load or Linux deployment")


if __name__ == "__main__":
    main()
