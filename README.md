# NVIDIA financial-report RAG API

An evidence-first Python library and authenticated FastAPI adapter for financial-document retrieval. Consumers can inject their own retriever and generator, keep source identities and citations, and receive distinct operational error codes. A small generated-data demo makes the installed API runnable on an ordinary CPU without downloading a model or supplying a financial corpus.

This is an independent development project licensed under **AGPL-3.0-or-later**. Review [LICENSE](LICENSE) and [third-party and data rights](THIRD_PARTY.md) before integration. The mandatory PyMuPDF/MuPDF dependency has AGPL/commercial terms; this release does not promise unrestricted proprietary relicensing of the combined runtime.

The [ten-report corpus guide](docs/ten-report-corpus-20261012.md) records 3,240 source-linked chunks, a pinned BGE passage matrix and 24 exposed development queries. Its [issuer source registry](corpus/sources-ten-reports.json) and [runtime metadata](docs/acceptance/local-ten-report-runtime-20261012.json) preserve exact provenance. Financial readiness failed: global retrieval accepts unrelated annual-report passages and the current similarity threshold rejects some correct statement pages. Use these recorded cases to reproduce and improve retrieval; they are not an accuracy claim.

The [expanded development comparison](docs/expanded-financial-comparator.md) adds six frozen refusal/clarification candidates and stores the actual query vectors for 35 case/scope requests. A separate no-model replay verified all 35 arithmetic traces and 175 scalar dot products. Accepted evidence still lacks complete same-cell financial field coverage; the similarity gate accepts 5/6 global and 2/5 explicitly scoped negative candidates. [The preserved metadata](docs/acceptance/expanded-development-runtime-20261012.json) reports those failures with human gold and generation both zero.

[Workflow 38205272181](https://github.com/1235357/nvidia-financial-rag/actions/runs/38205272181) passed Linux/Windows checkout, fresh installed API and nonroot generated-data container HTTP checks at `155758ca536354933a1f29075eb9cc48ada78932`. [The independent archive review](docs/acceptance/ci-api-acceptance-20261012.json) verifies both 362-test results and each rebuilt wheel's modules/RECORD. CI uses synthetic evidence and does not run the actual BGE experiment or establish financial readiness.

The expanded comparator release at `55a904b8ff018183531abf5b88e4d1deba5544bf` subsequently passed [workflow 38260699927](https://github.com/1235357/nvidia-financial-rag/actions/runs/38260699927), including 605 checkout tests on each platform, installed API checks and the Linux container job. That dated release is separate from later local changes.

The [source-bound statement selector](docs/structured-income-selection.md) keeps table headers, rows, columns and amounts together in an opt-in source index. Its initial quarterly run extracted 204 cells and compared the same 35 original question/scope rows. All 24 positive selections matched exposed source-field targets; eleven provisional negative rows were withheld. The later [annual extension](docs/annual-income-selection.md) adds 327 annual cells and retains fiscal-year periods without assuming twelve calendar months. Human gold and generation remain zero. The [explicit statement HTTP adapter](docs/statement-http-api.md) exposes this selector when an operator injects a source-built index at startup. It returns source candidates for review; the ordinary query service and browser demo remain separate.

## Run the generated-data API

Python package metadata permits Python 3.11 and later. The dated local acceptance used Windows/Python 3.13; the release workflow checks Python 3.13 on Linux and Windows. Read the actual workflow result before treating either platform as accepted for a particular commit.

From a fresh source checkout on Linux/macOS:

```bash
git clone https://github.com/1235357/nvidia-financial-rag.git
cd nvidia-financial-rag
python -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/api-demo.txt ".[api]"
export NVIDIA_RAG_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python examples/serve_demo.py
```

On PowerShell, after cloning and entering the repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -c constraints/api-demo.txt ".[api]"
$env:NVIDIA_RAG_API_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
python examples/serve_demo.py
```

Use a second terminal with the same privately configured token:

```text
python examples/query_client.py --question "bearer authentication"
```

The default listener is `127.0.0.1:8000`. The two passages are generated transport documentation under `synthetic-release-demo`; they contain no NVIDIA financial facts. TF-IDF supplies lexical retrieval, generation is disabled, and successful evidence responses have an empty `answer`. This demo exercises the install, HTTP adapter and source-citation flow. It is not a financial answer benchmark. `Ctrl+C` stops the server. Add `--expose-docs` for local interactive OpenAPI documentation.

The dependency constraints record a dated selection of direct core/API dependencies. Transitive dependencies and the container base tag still resolve at build time; the workflow archives the resolved API list. This is not a complete hash lockfile or a byte-reproducible build claim. Runtime uses no external service once dependencies are installed; installation and image building require package downloads.

## Integrate the installed library

The wheel contains `nvidia_rag`, including `nvidia_rag.api.create_app`, `EvidenceService`, lexical/dense retrieval and optional generation adapters. The clone also contains source-tree launchers and examples; it does not install a console command or bundle a trained index.

A consumer-owned retriever implements `retrieve(question, *, top_k)` and returns `ScoredChunk` records. The service enforces a selected evidence threshold before context construction:

```python
from nvidia_rag.api import create_app, resolve_api_token
from nvidia_rag.service import EvidenceService
import os

# Supply your own reviewed retriever; this variable is a consumer integration point.
service = EvidenceService(retriever=your_retriever, threshold=0.78, top_k=5)
app = create_app(service=service, api_token=resolve_api_token(os.environ))
```

For a complete runnable implementation, see [serve_demo.py](examples/serve_demo.py). A generator can be composed through `GroundedRagService`; enable it only after separately preparing its dependencies, model and source/evaluation contract. Dependency extras are `api`, `dev`, `semantic`, `generation` and `demo`. The API extra includes FastAPI, Uvicorn and HTTPX without the Gradio or model runtimes.

No bearer token belongs in browser JavaScript, committed files or command arguments. The provided client reads its token from the environment, refuses remote plain HTTP, disables redirect following and does not retry automatically. A shared token provides one authentication boundary; it is not tenant or per-document authorization.

## HTTP contract

`POST /v1/query` accepts `{"question":"..."}` with `Authorization: Bearer ...`. Questions must be nonblank and at most 500 characters; unknown JSON fields are refused. The response includes `request_id`, `can_answer`, `evidence_passed`, `message`, `answer`, `citations`, `evidence` and nullable `error_code`. Request IDs also appear in `X-Request-ID`.

| Outcome | HTTP | `error_code` |
| --- | ---: | --- |
| Evidence-only result, accepted generated result, evidence refusal, model abstention, validation veto or prompt-budget guard | 200 | `null` |
| Missing/incorrect bearer token | 401 | Authentication detail |
| Invalid question/request schema | 422 | Validation detail |
| Query service unavailable | 503 | `query_service_unavailable` |
| Generation unavailable, malformed/unfinished upstream output or tokenizer failure | 503 | `generation_unavailable` |
| Generation transport timeout | 504 | `generation_timeout` |

Operational failures return an empty answer and `can_answer=false`; established gated evidence is retained after a generation failure. Upstream error bodies and raw model diagnostics are withheld. A 200 response does not by itself establish a generated answer or financial correctness: inspect the outcome fields and evidence. The full completed-content outcome taxonomy remains future work.

`GET /healthz` is liveness. Authenticated `/readyz` currently returns a constant startup marker; it is not a live dependency probe. OpenAPI is off by default. [The dated failure contract](docs/api-failure-contract.md) explains the adapter's tested boundaries.

`POST /v1/statement-candidates` is an additional authenticated, explicitly configured interface. Supply `statement_index=build_income_index(reviewed_reports)` to `create_app`; without that index, authenticated calls return 503 with `STATEMENT_INDEX_NOT_CONFIGURED`. It accepts an original question and optional registered document IDs, then returns exact source-field candidates or a distinct clarification/refusal/unsupported outcome. It never calls the dense retriever or generator. See [the request, startup and consumer contract](docs/statement-http-api.md) before using this mode.

## CPU container demo

```bash
docker build -t financial-rag-demo .
docker run --rm --read-only --cap-drop ALL --security-opt no-new-privileges:true \
  --publish 127.0.0.1:8000:8000 --env NVIDIA_RAG_API_TOKEN financial-rag-demo
```

Supply `NVIDIA_RAG_API_TOKEN` privately in the invoking environment first. The image installs the built API wheel, runs as UID/GID `10001:10001`, serves only the generated-data evidence demo, and binds port 8000 inside the container. The host mapping above remains loopback. The health check probes liveness. The Docker build context is explicitly limited to package source, license/readme, the example and constraints; it excludes reports, indexes, receipts, credentials and runtime directories.

The demo has a 16-connection Uvicorn concurrency bound and a five-second keep-alive limit. These are demo settings, not load acceptance or a whole-request execution deadline. No public API infrastructure, GPU, paid model account or cloud subscription is required. Docker execution must be verified on a host with a working engine; the Linux workflow performs that gate.

## Real financial-corpus runtime

The [local financial-corpus recipe](docs/local-financial-corpus.md) adds an operator-local build and strict verified loader for the [eight pinned source registrations](corpus/sources-annual-plus-fy2025-q2.json), then exposes an explicitly scoped CPU TF-IDF evidence API through the installed `nvidia_rag.corpus` interfaces. PDFs, extracted chunks, vectors and models remain external. Build the reviewed wheel once, install that exact identified wheel, supply prepared local files/cache, and distinguish real-corpus HTTP equivalence from semantic retrieval, financial field support and generation acceptance. The recipe does not turn an older index count or a synthetic transport test into current financial acceptance.

The existing [run_api.py](run_api.py) is a separate model-backed source-tree launcher. Prepare approved chunks, corpus/embedding manifests, matching vectors and a cached pinned BGE model outside Git. It defaults to cached models, CUDA and generation disabled; use explicit CPU settings where appropriate:

```text
python -m pip install ".[semantic,api]"
python run_api.py --device cpu --generator none --chunks <chunks.jsonl> --passage-embeddings <vectors.npy> --embedding-manifest <embedding-manifest.json> --corpus-manifest <manifest.json>
```

The angle-bracket paths are operator-supplied inputs, not bundled files or pasteable literal values. The launcher verifies corpus/vector identity and validates its token before expensive startup. Model acquisition requires an explicit `--allow-model-download` choice; source acquisition and model licenses remain separate operator responsibilities. Generation choices `qwen-local` and `openai-compatible` need additional preparation. The default transport rejects public model endpoints and expects a private/loopback service.

The generated-data release demo and CI do not run this financial model path. Broader corporate use needs independently reviewed labels, corpus/model/config receipts, meaningful readiness, bounded work queues/deadlines, ingress request/rate limits, tenant authorization where relevant and observed load/latency outcomes. These require concrete deployment tests; neither a built wheel nor passing synthetic fixtures establishes them.

## Checks and CI

The [API workflow](.github/workflows/api.yml) runs checkout contracts on Linux and Windows/Python 3.13, builds a wheel, installs only `[api]` into a fresh environment, verifies installed-module identity and eleven synthetic failure cases, checks real loopback query and statement HTTP plus `pip check`, and uploads the wheel/test/dependency receipts. The statement check generates native PDF table fixtures and starts its own temporary loopback service. A dependent Linux job builds the nonroot image and checks actual generated-data HTTP. Actions use immutable commit references. No private corpus, raw report or model inference is part of pull-request CI.

Equivalent checkout commands:

```text
python -m pip install -c constraints/api-demo.txt ".[api,dev]" build
python -m pytest -o addopts= -q tests
python -m build --wheel --outdir dist
```

Install the resulting wheel with its `[api]` extra into a second fresh environment. From an external working directory, use that environment's Python with absolute script paths:

```text
python -I -B <repository>/acceptance/wheel_smoke.py
python -I -B <repository>/acceptance/api_failure_contract.py --output <new-receipt.json>
python -I -B <repository>/acceptance/http_loopback.py
python -I -B <repository>/acceptance/statement_http.py --output <absolute-new-statement-receipt.json>
python -m pip check
```

To check a running generated-data server or container, run `python -I -B acceptance/demo_http.py` with the same ephemeral test token. This is a real HTTP transport check and reports no financial score. Cold model acquisition, financial generation, TLS, concurrency distributions and deployment availability remain outside these checks. Local dated receipts establish local outcomes; only an actual completed Actions run establishes CI status.

For a collaborator's fresh checkout, clone `main`, install the chosen extras and run these commands. Existing clean checkouts can update with `git pull --ff-only`; preserve local changes before changing dependencies. The browser-native portfolio/demo is developed separately and does not host this Python process on GitHub Pages.

## Source review and evaluation boundaries

[The corpus/evaluation map](docs/evaluation-readiness-and-corpus-map-20261011.md) documents the existing data and its missing acceptance labels. Four portable FY2025-Q2 bindings under `evaluation/source-candidates/` are assistant-reviewed **development candidates**. They retain report/page/row/duration/date/unit/span identities and are not independently accepted human gold. The original PDF is supplied externally and verified by hash; it is not copied into this repository.

[The table experiment](docs/2026-10-11-table-aware-ingestion-experiment.md) keeps lexical presence separate from exact field selection. Its frozen profile controls and experimental cells are source-replay evidence, not retrieval recall, generation accuracy or a false-acceptance rate. No held-out financial accuracy number is claimed.

The October 11 API checkpoint passed 62 tests including token-counter failure mapping; the final dated API/source-binding checkpoint passed 74 after adding twelve source-binding guards. Later table and release-demo checks have separate scopes. These historical counts do not describe every future checkout. The previously accepted 73,357-byte wheel, SHA-256 `1f2e95aa8866e2f66fc4422583c257b9b07a6aa420db59f16ccc883fd2a5d8d6`, remains a local historical API artifact. Release metadata and licensing produce a distinct wheel; use each commit, wheel hash and actual CI receipt to identify what was accepted.
