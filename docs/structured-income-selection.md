# Literal source-bound income-statement selection

`nvidia_rag.structured_income` adds an opt-in source-bound table index and a narrow literal-query selector. It reads pinned local PDF bytes and keeps each amount with its statement, row, date, duration, column role, unit section and printed currency occurrence. It returns a `source_candidate`, `clarify`, `refuse` or `unsupported` result. A source candidate is ready for source review; it is not an accepted financial answer.

The later [annual-statement extension](annual-income-selection.md) adds the six registered annual layouts while preserving the original quarterly cell identities. The dated 204-cell/35-query observation below remains the initial quarterly checkpoint at `224d02d`; it does not describe the broader annual index.

The existing dense retriever, evidence threshold, generator and HTTP service remain separate. This increment does not insert a table candidate into their results, manufacture a cosine score or silently fall back to dense retrieval. Consumers must explicitly choose and review this interface before composing it into an application.

## What the source index retains

`build_income_index` accepts `ReportInput` records with an absolute local PDF path, its exact SHA-256 and an explicit fiscal-year/quarter declaration. The fiscal declaration is operator metadata and must be reviewed separately; calendar dates do not determine the report fiscal identity. The builder sees no evaluation labels, queries or expected values.

It scans exact standalone statement headings, preserves the three-month and six-month column groups, distinguishes current and prior-comparison columns, and binds sorted native text spans to source rectangles. A cell identity hashes all its source fields and bindings. Negative parentheses and decimal EPS values use exact decimal construction. Ambiguous or duplicated row/column occurrences are omitted with a reason; a dash is not converted to zero.

Monetary, earnings-per-share and weighted-average-share rows have separate section paths. Printed `$` is recorded as a dollar symbol with **ISO currency unverified**. It does not establish USD or authorize currency conversion. The literal-query interface currently selects monetary rows only; EPS and share-count lookup need a section-qualified adapter. The index is an in-process object built by this factory; this release provides no serialized-index loader or Python security sandbox.

Quarterly layouts with two or four dated columns are supported. The later annual parser supports three `Year Ended` columns with complete or split native dates and retains a null month duration. At the initial quarterly checkpoint, registered annual reports retained their hash and scope identity but recorded `ANNUAL_LAYOUT_NOT_IMPLEMENTED`. A known report whose layout is unsupported must not be described as missing source data. Other layouts require further parser development and actual source review.

## Use the installed library

Install the chosen source release with its core dependencies. Then supply an externally prepared and independently reviewed PDF and its exact hash:

```python
from pathlib import Path
from nvidia_rag.structured_income import (
    ReportInput, build_income_index, parse_statement_request,
    select_statement_candidates,
)

report = ReportInput(
    document_id="nvidia-fy2027-q2",
    pdf_path=Path("/operator/sources/fy2027-q2.pdf"),
    pdf_sha256=reviewed_pdf_sha256,  # Supply the reviewed exact local digest.
    fiscal_year=2027,
    fiscal_quarter=2,
)
index = build_income_index((report,))
request = parse_statement_request(
    original_question,
    operator_document_ids=(report.document_id,),
    index=index,
)
selection = select_statement_candidates(index, request)
```

`reviewed_pdf_sha256` and `original_question` are caller-supplied values. Use an absolute path appropriate to your operating system. This example neither downloads the report nor supplies a financial verdict. Inspect the disposition, reason and full cell bindings before presenting a candidate.

The parser preserves every recognized fiscal, quarter, duration, date, currency, scale and monetary-row mention. Conflicting fields require clarification. Unknown material words and arithmetic symbols are withheld. Multiple metrics, general NLP, negation, arithmetic, ratios, growth, FX and scale conversion are outside this grammar. Requests with an explicit ISO currency cannot be fulfilled from a dollar glyph alone. A report-scope conflict and an unavailable report have distinct outcomes. The selector re-parses the exact original query before using its fields.

## Reproduce the development comparison

The source-only comparison CLI is a source-checkout tool under `acceptance/`; it is not installed in the wheel. Install this checkout so its new production module is available. Supply all ten pinned local PDFs in the source registry, including the six annual reports. Their filenames and hashes must match the registry. The committed [development packet](../evaluation/source-candidates/ten-report-v2/README.md) provides the exact eighteen exposed cases, case freeze and experiment used here.

Run from the checkout, replacing `C:\operator` with existing absolute input paths and a fresh output directory whose parent exists:

```powershell
python -m acceptance.structured_income_comparison `
  --source-dir C:\operator\sources `
  --source-registry C:\checkout\corpus\sources-ten-reports.json `
  --cases C:\checkout\evaluation\source-candidates\ten-report-v2\cases.jsonl `
  --case-freeze C:\checkout\evaluation\source-candidates\ten-report-v2\case-freeze.json `
  --experiment C:\checkout\evaluation\source-candidates\ten-report-v2\experiment.json `
  --output-dir C:\operator\comparisons\structured-run-01
```

The CLI verifies registry and case-file identities, builds and closes the source-only index, then uses the same **35 original question/scope rows** as the dense diagnostic. Global scope retains all ten registered report identities. Expected cells and provisional labels enter only a post-selection comparison after every selection has completed. No corpus vectors, encoder, retriever or generator run on this path.

Both JSON outputs use exclusive creation and completion guards. PDF, input, implementation and runner identities are checked during closure. A failed late check retains failure evidence. A reused output path is refused without invalidating its previous completed receipt. Preserve failed and superseded runs separately.

## Initial quarterly checkpoint at 224d02d on October 12

The final local run for the initial quarterly checkpoint at `224d02d` registered ten pinned reports and extracted **204 source cells from four quarterly statements**: 156 monetary, 24 EPS and 24 share-count cells. The six annual layouts were explicitly recorded as unsupported at that checkpoint. An independent program reopened the four actual PDFs and completed **8,531 source-value, column, period, section, span, rectangle and identity checks** on the earlier identical cell payload; its finite audit is mechanical provenance evidence.

| Original query scope | Requests | Source candidates | Refuse | Clarify | Unsupported |
| --- | ---: | ---: | ---: | ---: | ---: |
| All ten registered reports | 18 | 12 | 3 | 2 | 1 |
| Explicit requested report | 17 | 12 | 2 | 2 | 1 |
| Total | 35 | 24 | 5 | 4 | 2 |

All 24 positive candidate selections matched the exposed expected source-field tuples, including the report hash, page, date, duration, row, column role, amount, scale and printed currency. This is a mechanical comparison against already exposed development targets, not a held-out accuracy result. The eleven provisional negative request rows were withheld; their appropriate financial outcome still needs independent adjudication. Human gold, encoder runs and generation runs were zero.

The [earlier dense diagnostic](expanded-financial-comparator.md) remains preserved: positive joint Top-5 source coverage was 1/12 globally and 10/12 with explicit report scope, while accepted evidence jointly covered 0/12 in both scopes. Its `0.78` gate accepted 5/6 global and 2/5 scoped negative candidates. These methods have different output semantics; a literal source-field match is not a replacement cosine score or financial-readiness certificate.

The initial quarterly checkout at `224d02d` passed **646 tests** with no failures, errors or skips. Its new module and comparison boundaries account for 41 tests, including synthetic PDF geometry, decimal precision, unknown operations/currencies/symbols, parser ownership and late output-closure controls. The actual PDF observation is separate from synthetic tests and from CI. [Historical sanitized metadata](acceptance/structured-income-development-20261012.json) records hashes and outcomes without distributing report bodies or complete extracted pages. The broader annual index and its later checks are documented in the [annual extension](annual-income-selection.md).

Before integrating this path into the full RAG UI/API, extend other financial statements, add section-qualified input for EPS/share counts, compare unseen independently reviewed requests, and evaluate generated outputs and observed operational performance. The annual extension is recorded separately above. Financial readiness remains unestablished.
