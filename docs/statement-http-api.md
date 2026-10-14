# Inspect a source-bound income cell over HTTP

`POST /v1/statement-candidates` lets a consumer inspect the source-bound income selector through an explicit authenticated route. An operator builds an immutable index from reviewed, hash-pinned PDF inputs and injects it at startup. Each request selects from that snapshot: it does not open a caller-supplied path, reload a PDF, retrieve vectors or generate an answer.

This interface supports the selector's existing monetary row and literal-period vocabulary. The [annual guide](annual-income-selection.md) explains fiscal-year periods, and the [quarterly guide](structured-income-selection.md) explains row, column and span bindings. EPS/share-count natural-language lookup, arithmetic, currency conversion and unseen financial accuracy remain outside the accepted scope. A returned cell remains a source candidate with `financial_correctness="unestablished"`.

Applications that already know the field and unit can use the separate
[`/v1/statement-fields` typed interface](statement-fields-api.md), including the
native per-share and share-count sections. It does not expand this route's
natural-language vocabulary or create a generated financial answer.

For a complete packaged startup path, use [the verified corpus startup guide](verified-corpus-startup.md). It documents the installed command and Python factory, strict registry/PDF-bound fiscal declarations, shared document scope, model-launcher options and read-only container composition. The custom application injection below remains supported.

## Configure a consumer-owned application

Install the selected wheel with `[api]`. The application continues to require a consumer-owned ordinary query service implementing `query(question)`; the statement route is an additional explicitly configured interface:

```python
import os
from pathlib import Path

from nvidia_rag.api import create_app, resolve_api_token
from nvidia_rag.structured_income import ReportInput, build_income_index

# Configure these from your reviewed source registry, not an HTTP request.
annual = ReportInput(
    document_id="nvidia-fy2025-annual",
    pdf_path=Path(os.environ["REVIEWED_ANNUAL_PDF"]).resolve(),
    pdf_sha256=os.environ["REVIEWED_ANNUAL_SHA256"],
    fiscal_year=2025,
    fiscal_quarter=None,
)
index = build_income_index((annual,))
app = create_app(
    service=your_query_service,  # Your application's existing injected service.
    api_token=resolve_api_token(os.environ),
    statement_index=index,
)
```

`your_query_service` is an integration point, not a bundled object. See [the runnable generated-data query demo](../examples/serve_demo.py) for a complete ordinary service implementation. For a quarterly source, supply its registered quarter rather than `None`. `ReportInput` carries the registered identifier, absolute source path and digest. `build_income_index` validates those inputs, checks the PDF bytes before parsing and retains native source identities; a forged or replaced `IncomeIndex` is rejected at application startup.

An omitted `statement_index` leaves the route registered but unavailable: after successful authentication it returns HTTP 503, `disposition="unavailable"`, `reason="STATEMENT_INDEX_NOT_CONFIGURED"`, an empty `cells` array and a null index hash. The generated-data server and container do not configure financial statement sources automatically.

The configured index is a startup snapshot. Subsequent disk changes do not update its cells or hash; build and review a new index to publish a new snapshot. The hash covers the canonical source-index metadata and cells, without the internal construction token. This in-process construction guard checks trusted application composition; it is not a sandbox for hostile Python code. HTTP output includes source identifiers and spans but no operator-local PDF path.

## Send the original question and explicit scope

The body is a JSON object:

```json
{
  "question": "What was net income for the year ended January 26, 2025, in millions of dollars?",
  "document_ids": ["nvidia-fy2025-annual"]
}
```

Use `Authorization: Bearer ...` configured privately in the caller's environment. The clone's consumer example supports the route explicitly:

```text
python examples/query_client.py --statement-candidates --document-id nvidia-fy2025-annual --question "What was net income for the year ended January 26, 2025, in millions of dollars?"
```

Run this command against your configured application; the default generated-data server has no statement index. Repeat `--document-id` for several registered reports, and use `--base-url` for a different server. The client preserves the original wording and explicit scope, refuses remote plain HTTP, disables redirect following and does not automatically retry or switch routes. A 503 raises the HTTP error path rather than being printed as a completed source result. The token is read from `NVIDIA_RAG_API_TOKEN`, never a command argument. A token must not be embedded in the public portfolio or browser JavaScript.

If `document_ids` is omitted or null, all registered reports form the operator scope. An explicit list must contain one to 128 unique identifiers, each one to 128 ASCII letters, digits, underscores, periods or hyphens. Unknown IDs withhold the whole requested scope with `OPERATOR_REPORT_UNAVAILABLE`; the adapter does not silently discard them. Overlapping source cells can require clarification rather than selecting the first match.

The original question must be a nonblank string of at most 500 characters. Unknown fields, coerced non-string values, duplicate JSON keys including escaped-key duplicates, nonfinite JSON constants, invalid UTF-8 and excessively nested JSON are rejected with a fixed 422 detail that does not echo the input. The media type must be `application/json` and the actual streamed body must be at most 8,192 bytes; the adapter does not rely on the declared Content-Length. Authentication runs before the configured route reads that body. This byte ceiling is not a whole-request execution deadline or load acceptance.

## Interpret the completed result

Every completed source selection returns HTTP 200 with `schema_version=1`, `request_id`, `disposition`, `reason`, `parsed_request`, `index_sha256`, `cells`, `financial_correctness`, `score_semantics` and a null `error_code`:

| Disposition | Meaning | Cells |
| --- | --- | --- |
| `source_candidate` | Exactly one source cell matches the supported literal request and scope | Exactly one |
| `clarify` | The supported fields or source scope need disambiguation | Empty |
| `refuse` | The requested supported field tuple is absent, or a requested source is unavailable | Empty |
| `unsupported` | The operation or wording falls outside this selector's supported contract | Empty |

`parsed_request` retains the original query, its UTF-8 SHA-256, operator document IDs, literal mentions with character offsets, and parsing issues. A candidate retains its cell/report/PDF/page-text identities, physical page and nullable PDF label, section-qualified row, column role and number, period end, unit, scale, amount string and native text/rectangle bindings. Annual candidates have `period_kind="fiscal_year"`, null `duration_months` and null report quarter; quarterly candidates retain their month duration. Exact amount strings avoid converting the printed decimal value through a binary float. Rectangle coordinates remain finite JSON numbers.

`score_semantics="exact-source-field-selection;no-cosine-score"` identifies this method. There is no invented vector score, generated answer or `can_answer` field in this envelope. Dollar symbols do not prove an ISO currency. The caller must distinguish physical PDF pages from printed numbers and preserve source identity when displaying the candidate.

| Failure | HTTP | Body |
| --- | ---: | --- |
| Missing or invalid bearer token | 401 | Authentication detail and Bearer challenge |
| Actual body exceeds the byte ceiling | 413 | Fixed size detail |
| Unsupported media type | 415 | Fixed media-type detail |
| Invalid JSON or strict request schema | 422 | `{"detail":"Invalid statement request."}` |
| Missing startup index, selection/validation/serialization fault | 503 | `unavailable` envelope, empty cells, null parsed request, `statement_selection_unavailable` |

Result faults, including an unexpected parser/selector `HTTPException`, are operational failures. Their private exception contents and partial candidate data are withheld. An established snapshot hash is retained in the failure envelope, while a missing startup index has a null hash. `X-Request-ID` follows the existing adapter policy and matches the envelope for source outcomes; `X-Process-Time-Ms` is one observed request duration, not a latency distribution. Existing `/v1/query` composition and generation failure semantics continue through their own service.

## Reproduce the installed transport check

Build a wheel, install that exact wheel with `[api]` into a fresh virtual environment, then run the fresh environment's Python from outside the checkout:

```text
python -I -B <absolute-repository>/acceptance/statement_http.py --output <absolute-new-receipt.json>
```

Replace both angle-bracket paths with actual absolute paths. The receipt's parent directory must already exist and the output must be new. Python isolated mode and API-only imports prevent the checkout or heavyweight model packages from supplying this run. The script generates two native PDF fixtures, builds 42 source cells, starts an ephemeral loopback Uvicorn listener, sends real HTTP requests, checks annual/quarterly exact source fields and withholding outcomes, verifies authentication and malformed/oversized input, exercises the ordinary query service, and stops its temporary server. Fixture amounts are synthetic transport data, not financial reference facts.

The [release workflow](../.github/workflows/api.yml) performs this check on Linux and Windows after installing each newly built wheel, and archives `installed-statement-contract.json` alongside the checkout XML, existing installed API contract, resolved dependencies and wheel. Local successful receipts and completed CI runs identify their own commit and package; an encoded source candidate alone does not establish public API hosting, browser integration, load/TLS behavior, human-reviewed financial accuracy or generation acceptance.

The [dated local development metadata](acceptance/statement-http-development-20261012.json) binds 711 passing checkout tests and a separate fresh API-only wheel with 28 real loopback checks. Complete HTTP candidate records match their native synthetic index cells, and every returned text span is replayed against the generated PDF page. An independent coordinate-corruption probe was initially accepted by a weaker check; the strengthened check rejects it and closes the temporary listener without writing a successful receipt. Those historical observations remain separate from later CI artifacts and original financial-source acceptance.

The source release `1fb5a50dc48645dbbf4fc50159d2986bcf12d687` subsequently completed [workflow 38292236298](https://github.com/1235357/nvidia-financial-rag/actions/runs/38292236298) successfully on Linux, Windows and the nonroot generated-data container. The [actual archive review](acceptance/statement-http-ci-publication-20261012.json) verifies 711 unique passing checkout identities on each platform, all 18 production modules against that commit, all 24 wheel RECORD rows, the eleven existing synthetic model-transport cases and 28 installed statement checks over 42 generated source cells. The downloaded container log confirms the existing generated-data query HTTP check; it does not configure a financial statement index or run a model. The two rebuilt wheels have distinct archive identities recorded in that metadata.

## Check the original-source snapshot separately

A later local run installed the identified source-release wheel in an API-only environment, rebuilt the index from all ten hash-pinned original reports, and sent the already exposed 35 original and 42 source-derived annual requests through a real temporary loopback service. The rebuilt canonical 531-cell index matched both earlier index artifacts exactly. All 77 HTTP outcomes retained their prior dispositions, reasons, original parses and complete selected-cell records, with the additional derived period kind. Sixty requests returned a source candidate; the others retained their clarification, refusal or unsupported outcomes. This is transport equivalence over exposed development material, not an unseen financial benchmark.

The [sanitized original-source observation](acceptance/statement-http-original-sources-20261012.json) binds that installed wheel, runner, complete private receipt and source index. The finite replay checked 438 returned text spans and corresponding native word rectangles on nine source pages. All 88 watched source, package, test and input files remained unchanged, the ordinary query service was never called, and the temporary service and listener stopped. A separate 619-check completed-record reload verified the saved requests, source cells, outcome counts and input closure without starting another HTTP service or model. The recorded whole-sequence interval includes source replay and file-closure checks; it is not a request-latency distribution. Human gold, held-out cases, encoder and generation runs are zero. Complete source bodies, local paths and question responses remain in operator-local ignored artifacts.
