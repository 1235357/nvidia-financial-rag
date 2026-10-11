# API failure contract — local acceptance, 2026-10-11

The local staged `/v1/query` endpoint now distinguishes a failed generation
operation from a completed request that refuses an answer. Previously the model
adapter lost timeout identity, and the RAG service returned a safe failure object
that the HTTP boundary serialized as 200. No raw model answer was released, but a
consumer could mistake an outage for a normal evidence refusal.

| Condition | HTTP | `error_code` | Evidence retained |
|---|---:|---|---|
| Model transport timeout, or generator `TimeoutError` | 504 | `generation_timeout` | Yes, after the evidence gate passed |
| Model connection/HTTP failure, malformed response, or unconfirmed termination | 503 | `generation_unavailable` | Yes |
| Prompt tokenizer/counting failure | 503 | `generation_unavailable` | Yes; model not called |
| Exception from the query service before a usable result | 503 | `query_service_unavailable` | No; evidence was not established |
| Empty evidence, exact model abstention, answer-validation veto, or prompt budget guard | 200 | `null` | According to the evidence gate |
| Evidence-only response or accepted generated answer | 200 | `null` | Yes |

`error_code` is an additive nullable response field. All operational failure
responses use `can_answer=false` and an empty `answer`. They retain the existing
response shape, a safe message, and the request ID in both the JSON body and
`X-Request-ID`. A generation failure after retrieval keeps the exact gated
citations and evidence. Upstream exception details, model response bodies and
diagnostic `raw_answer` are not returned. When enabled, OpenAPI documents the
503 and 504 response models.

Consumers should check HTTP status before presenting an answer. The existing
`examples/query_client.py` calls `raise_for_status()`, so these outages now reach
its error path. A 200 with an empty `answer` is still not a generated answer:
inspect `can_answer`, `evidence_passed` and the response message. A complete typed
taxonomy for evidence-only success, abstention and validation vetoes is future
work. This change adds no automatic retry policy.

## Reproducible local checks

The API checkout baseline was 45 tests. Seventeen API failure regressions
exercise the real adapter, RAG service and HTTP serialization while replacing
only the model transport or nondeterministic generator. The earlier API
checkpoint passed 62 tests, including token-counter failure classification;
that dated result is preserved in
`artifacts/api-failure-20261011/checkout-tests.xml` and
`candidate-integrity.json`.

The final staged API/source-binding checkpoint passed **74 tests**, with zero
failures, errors or skips, as recorded in
`artifacts/source-binding-fy2025-q2-20261011/checkout-tests-final.xml`.
Its additional twelve tests cover portable source-binding replay guards; they
do not change the accepted API wheel. Both checkpoints used Windows and Python
3.13.5 and retained the existing Starlette TestClient deprecation warning.
These are archived checkpoint totals, not a claim about every later experiment
added to the checkout. The bounded API/source-binding checks can be selected
explicitly:

```text
.venv-acceptance/Scripts/python.exe -m pytest -o addopts= -q tests/test_consumer_example.py tests/test_conversation.py tests/test_generation_failure_contract.py tests/test_index_integrity_contract.py tests/test_launcher_preflight.py tests/test_source_binding_replay.py
```

The final local wheel is
`artifacts/api-failure-20261011/final/nvidia_financial_rag-0.1.0-py3-none-any.whl`,
73,357 bytes, SHA-256
`1f2e95aa8866e2f66fc4422583c257b9b07a6aa420db59f16ccc883fd2a5d8d6`.
The version remains development `0.1.0`; use the hash to distinguish candidates.
The first wheel/receipt in the parent artifact directory is the intermediate
candidate before token-counter failure classification; it is preserved.

A new virtual environment without system-site-packages installed only the
wheel's `[api]` extra from the official PyPI index. The final wheel then replaced
the intermediate candidate with `--no-deps`; dependency declarations did not
change. From an external temporary working directory, the final installed package
passed the isolated `-I -B` acceptance script's eleven synthetic cases. The script
checks installed-module identity and verifies Gradio, Torch, Transformers and
sentence-transformers are absent. `pip check` found no broken requirements.

```text
<clean-environment-python> -I -B <absolute-path>/acceptance/api_failure_contract.py --output <new-receipt.json>
<clean-environment-python> -m pip check
```

The final receipt is
`artifacts/api-failure-20261011/installed-api-failure-contract-final.json`;
`environment-freeze-final.txt` records the final resolved dependencies and
`pip-check-final.txt` records the dependency check. These commands use explicit
local paths supplied by the operator. The clean environment resolved FastAPI
0.143.0 and HTTPX 0.28.1. PyMuPDF `fitz` and Starlette warnings remained.

## Enterprise acceptance still required

This is installed-package and synthetic failure-contract evidence. It does not
measure model inference, financial correctness, actual socket timeout behavior,
latency percentiles, concurrency, TLS, Linux deployment, or public availability.
The constant `/readyz` remains a startup marker rather than a dependency probe.
The single bearer token is not tenant authorization. Queue limits, cancellation,
whole-request deadlines and aggregate outcome/latency telemetry remain untested.
No migration to a different serving framework or vector database is implied.

Financial acceptance needs an independently reviewed, source-linked evaluation
set. Each label must bind the original report, page, requested period, field,
currency/unit and supporting span. Report retrieval recall/ranking separately
from answer correctness, faithfulness and correct refusal. Preserve frozen
corpus/model/config identities and distinguish seen development cases from new
acceptance cases. Citation and numeric-literal checks alone do not establish
claim entailment; existing nonnumeric support gaps remain.
