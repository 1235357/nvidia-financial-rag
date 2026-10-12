# Source-qualified development comparison

The new acceptance tools freeze financial requests and replay their source coordinates before retrieval. They preserve the original eight-report scripts and the ten-report corpus, vectors and reviewed wheel. A scoped producer saves the actual raw and consumed query vectors; a separate no-model verifier reconstructs their arithmetic. The real expanded run and its outcome denominators are recorded only after these tools close their inputs and outputs successfully.

## Why a page or number hit is insufficient

A question such as “three-month net income ended July 28, 2024” refers to one row and one dated column in a particular report. The same table also has a six-month column with the same date, comparison-year columns, printed dollar glyphs and a scale heading. Finding `16,599`, or even the correct statement page, does not establish that all requested fields were retained together.

`acceptance/expanded_cases.py` validates exact case bytes, ordered IDs, counts, report fiscal identity, period end and duration, metric/unit distinctions, provisional currency normalization and source-cell coordinates. A cell ID includes the page-text digest and extraction profile defining its offsets. Retrieved inputs expose only the original question and operator-selected scope; expected amounts, cells and provisional refusal labels are kept outside that input.

`acceptance/expanded_source_binding_replay.py` first checks registry, corpus and source PDF identities. It then verifies the declared PyMuPDF version, exact page-text spans, rectangular source occurrences, frozen dated columns and metric/value row alignment. Repeated dates and numbers must identify their selected occurrence. Column ranges cannot overlap, and an enlarged row band cannot join an amount to another metric row.

Source review and corpus extraction have separate coordinates. The current review uses sorted, unstripped page text; the prepared corpus uses unsorted, stripped text. The tool re-extracts both profiles and maps exact source occurrences into the corpus profile. Each same-page chunk must occur exactly once to acquire a range. Missing or repeated matches remain unresolved. Coverage uses the union of accepted ranges from that same source page and cell; other pages cannot supply its missing headers or amount.

## Current local evidence

The finite local freeze contains **18 exposed development cases**: twelve preserved positive candidates, four provisional refusal candidates and two provisional clarification candidates. The twelve prior original record strings and their digests were retained. Before any new model query, the negative candidates were tied to frozen source-column inventory, registry absence, declared currency/FX capability or a declared contradictory duration request.

The actual source replay checked ten registered PDF hashes and twelve selected financial cells. All twelve source-coordinate bindings passed, and all required roles mapped into the actual corpus extraction profile. A separate program that does not import the new replay implementation completed **477 actual-file, page-span, clipped-box, source-row and chunk-range checks**. An independent review also verified **358 source-body, span, box and corpus-range assertions** without importing the new implementation. These checks establish mechanical candidate provenance. USD normalization remains provisional; a dollar glyph alone does not prove the normalized currency, and no independent human adjudication is claimed.

Applying these mappings to the previously saved 24 retrieval traces gives:

| Stored query scope | Requests | Software gate accepts | Complete same-cell field coverage in Top-5 | Complete same-cell field coverage in accepted evidence |
| --- | ---: | ---: | ---: | ---: |
| All ten reports | 12 | 12 | 1 | 0 |
| Explicit source report | 12 | 6 | 10 | 0 |

This retrospective computation verifies candidate identities, text hashes and threshold filtering against the stored traces and current unchanged corpus. It does not rerun encoders. The old run did not save query vector values, so its cosine arithmetic cannot be independently reproduced from those records. The six new negative questions have not yet been queried. No new model, query or generation action ran, and human gold remains zero.

The latest focused case/source suite passed **120 tests**, including 67 case-contract controls, 41 new source-replay controls and the twelve original source-replay controls. Independent review reproduced the focused result after the configuration-snapshot and late completion-marker fixes. Public CI has not yet checked this new increment.

## Actual expanded diagnostic

The new producer then loaded the existing cached CPU BGE model once and encoded all **35 frozen case/scope requests**, one encode per row. It saved raw float32 `[35,384]` and consumed float64 `[35,384]` query matrices. The independent verifier reconstructed every row with fixed `atol=1e-12`, `rtol=0`, checked all **175 Top-5 scalar dot products**, and reproduced candidate/evidence identities, citations, gate messages and context digests. Maximum score absolute error was zero in this observed runtime; none of the rows had an indeterminate boundary. A second independent reviewer repeated the actual arithmetic with model, retriever, service and PDF imports blocked.

The expanded result is:

| Scope | Positive requests | Positive software gate accepts | Positive Top-5 joint field coverage | Positive accepted-evidence joint coverage | Provisional negative requests | Negative candidates accepted by gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All ten reports | 12 | 12 | 1 | 0 | 6 | 5 |
| Explicit requested report | 12 | 6 | 10 | 0 | 5 | 2 |

The unavailable FY2035 report has only a global scope, so the negative denominators intentionally differ. The other five negative candidates run in both scopes. There are four refusal and two clarification cases, all frozen before querying; similarity acceptance of any of them remains a provisional failure observation rather than an independently measured financial false-accept rate.

The positive observations reproduce the earlier stored-trace breakdown. Explicit report scope improves Top-5 source-coordinate coverage but does not automatically preserve every field after threshold filtering. No positive accepted evidence set jointly covers all required statement, date, duration, metric, column, amount, currency and unit source roles. The fixed threshold also accepts negative candidates. Financial readiness remains **FAILED**; generation and independent human gold remain zero.

The final local checkout suite passed **605 tests**, with no failures/errors/skips and the existing Starlette/httpx deprecation warning. The producer/baseline focused suite passed 103 controls; the independent arithmetic suite passed 62. These software checks and the actual local model observation are separate from CI. [Sanitized runtime metadata](acceptance/expanded-development-runtime-20261012.json) retains exact hashes, runtime, all 35 row observations and the failure denominators, without report text, tensors, weights or private paths.

## Run the source replay

Use a source checkout and the already reviewed installed corpus runtime. Source acquisition, source rights, model preparation and corpus/vector construction are separate steps. This command opens local PDFs and does not download files or run an encoder.

Supply absolute local paths for the source registry, completed corpus, schema-v2 cases, case freeze, pre-replay experiment and frozen binding packet directory. Association and referenced rationale packets are named by their exact SHA-256, with a `.json` suffix. The experiment's temporary `source_replay` hash is excluded from its canonical binding digest to avoid a hash cycle; all other configuration fields remain bound. After successful replay, freeze a new final experiment containing the exact replay hash.

```powershell
python -m acceptance.expanded_source_binding_replay `
  --source-dir C:\operator\sources `
  --source-registry C:\operator\sources-ten-reports.json `
  --artifact-dir C:\operator\completed-corpus `
  --cases C:\operator\development\cases.jsonl `
  --case-freeze C:\operator\development\case-freeze.json `
  --experiment C:\operator\development\experiment.pre-replay.json `
  --binding-dir C:\operator\development\bindings `
  --output C:\operator\outputs\source-replay.json
```

The output parent must exist and the result path must be fresh and outside read-only input directories. Writes are exclusive. Pending/failure markers prevent a partially closed result from loading, including after a failed final input check or write. Inputs and referenced packet hashes are rechecked while closing. `load_source_replay` validates the frozen result against the final configuration, case order, source-cell identities, essential roles and actual corpus chunk inventory, then checks completion markers again before returning. Every explicit scope for a request naming a registered report must include that report; a different-report operator configuration is rejected before it can enter a financial outcome denominator.

Tests use synthetic boundary fixtures separately from actual issuer-PDF observations:

```powershell
python -m pytest -q tests/test_expanded_cases.py tests/test_expanded_source_binding_replay.py tests/test_source_binding_replay.py
```

## Capture actual queries and independently replay their arithmetic

Use the same installed wheel recorded by the vector-build receipt. Equal production module contents do not make two ZIP archives interchangeable for a frozen wheel digest. The current local vectors bind the original `dist/corpus-recipe-20261011-01` wheel; a later release-preparation wheel has a different archive digest. The final experiment was frozen again, and source replay was repeated, before any expanded model query.

The source checkout provides the acceptance modules; the reviewed runtime provides `nvidia_rag`. Run from the checkout without installing it over that runtime. All paths below are absolute examples. The output directory must be fresh, outside the read-only corpus, vectors, model cache and input-file directories.

```powershell
python -m acceptance.corpus_semantic_expanded `
  --artifact-dir C:\operator\completed-corpus `
  --source-registry C:\operator\sources-ten-reports.json `
  --vector-dir C:\operator\completed-vectors `
  --cases C:\operator\development\cases.jsonl `
  --case-freeze C:\operator\development\case-freeze.json `
  --source-replay C:\operator\outputs\source-replay.json `
  --experiment C:\operator\development\experiment.json `
  --wheel C:\operator\reviewed-wheel\nvidia_financial_rag-0.1.0-py3-none-any.whl `
  --model-cache-dir C:\operator\prepared-cache `
  --output-dir C:\operator\comparisons\run-01
```

Preflight verifies all nine frozen input digests, the installed wheel's production modules, completed corpus/vector/source receipts, configuration and every scope before loading the pinned cached CPU BGE model. No network model fallback exists. For each ordered case/scope, `EvidenceService.inspect` performs one actual inspection. The capture wrapper stores a copy of the exact float32 encoder return; the inspection exposes the exact float64 normalized vector used by `DenseRetriever`. Expected amounts and provisional outcomes never enter the query encoder or ranking inputs.

`queries-raw.npy` and `queries-normalized.npy` bind to `query-index.json`. `results.json` retains the full scoped chunk indices and considered scores, including scores below the gate, Top-K identities, evidence IDs, citations, service disposition and context digest. `manifest.json` is written last and binds every output file. Inputs, cached weights, wheel modules and output hashes are rechecked through closure; a failed write, check or guard removal keeps pending/failure evidence. Neither a manifest alone nor a return value bypasses those completion markers.

The verifier consumes these saved files and needs NumPy, not an encoder or source PDF parser:

```powershell
python -m acceptance.expanded_cosine_replay `
  --artifact-dir C:\operator\completed-corpus `
  --source-registry C:\operator\sources-ten-reports.json `
  --vector-dir C:\operator\completed-vectors `
  --experiment C:\operator\development\experiment.json `
  --case-freeze C:\operator\development\case-freeze.json `
  --comparison-dir C:\operator\comparisons\run-01 `
  --output C:\operator\reviews\arithmetic-run-01.json
```

It independently promotes stored passage/query vectors to float64, normalizes them, calculates every scoped dot product and checks the producer's scores with fixed absolute tolerance `1e-12` and relative tolerance zero. Every recorded Top-K result also gets a scalar `math.fsum` dot-product control. It reconstructs evidence filtering, citation order, service message and context SHA without importing `DenseRetriever`, `EvidenceService` or model packages. It checks source-replay identity but does not independently redo PDF field/geometry verification.

Ties at the Top-K boundary and scores within tolerance of the threshold are reported as `boundary_indeterminate`. Producer ordering is retained; a tolerance does not authorize certifying an opposite gate decision. Arithmetic consistency can verify a bad retrieval just as precisely as a good one. Apply the separate source-coordinate coverage check to Top-K and accepted evidence, then report provisional negative acceptance separately. These operations still establish no human gold or generated financial correctness.

## Remaining comparison gates

Keep `0.78` and Top-5 fixed for the first diagnostic. The new negative denominator is measured only when those requests actually run. The [published candidate metadata](../evaluation/source-candidates/ten-report-v2/README.md) preserves case bytes and exposure; it does not distribute the private prepared corpus, weights or tensors.

Use that evidence to compare report routing and header/row/column-preserving retrieval on the same frozen development set. Development calibration needs its own declared operating points. Independently authored and adjudicated unseen cases are still required before a financial-readiness claim. A successful source replay of a failed retrieval remains a failed retrieval.
