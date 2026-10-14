# Select a native statement field with an explicit unit contract

`POST /v1/statement-fields` is the typed counterpart to the literal-question
selector. An application already holding a field, period and unit can submit
those values directly. The adapter selects from the operator's existing native
PDF snapshot. It does not turn the request into a sentence, run vector search,
open a request-supplied path, perform arithmetic or generate an answer.

This contract makes a practical distinction that a bare row name cannot express:
`per_share/basic` is Basic earnings per share; `count/basic` is Basic weighted
average shares. Both rows may be printed as **Basic** on the same source page.
They have different units, scales and currency applicability.

## Configure the source snapshot

Use the same `statement_index` supplied to `create_app`, or the optional paired
source/declaration inputs of the [installed corpus launcher](verified-corpus-startup.md).
The native extraction and fiscal declaration contracts remain unchanged.
Unconfigured statement mode returns an authenticated HTTP 503 rather than
falling back to semantic retrieval. Each deployed operator chooses and verifies
its own PDFs and registered report IDs.

## Submit the field and its units explicitly

The following request illustrates a source lookup; it is meaningful only if the
operator's registered report contains the stated field and period:

```json
{
  "document_ids": ["nvidia-fy2025-q2"],
  "fiscal_year": 2025,
  "fiscal_quarter": 2,
  "row_path": "per_share/basic",
  "column_role": "current",
  "period_kind": "months",
  "period_end": "2024-07-28",
  "duration_months": 3,
  "unit_kind": "per_share",
  "scale": "1",
  "currency_literal": "$"
}
```

The seven field/period/unit members `row_path`, `period_kind`, `period_end`,
`duration_months`, `unit_kind`, `scale` and `currency_literal` are required.
`duration_months` and `currency_literal` accept null where appropriate but must
still be present. Fiscal identity, document scope and column role are optional
additional constraints, not inferred defaults. In particular, omitting a column
role does not mean current.

Row paths are bounded to 256 characters with a lowercase section and an
underscore-separated lowercase row key. Document IDs use one to 128 ASCII
letters, digits, underscores, periods or hyphens; explicit scopes contain one to
128 unique IDs. An optional fiscal year is an integer from 1900 to 9999 and an
optional quarter is 1 through 4. A month duration is an integer from 1 through
12. Booleans and string-to-number coercion are not accepted.

| Source quantity | Section-qualified row example | Unit kind | Source scale | Printed currency |
| --- | --- | --- | --- | --- |
| Statement monetary amount | `monetary/net_income` | `monetary` | `"1000000"` | `"$"` |
| Earnings per share | `per_share/basic` | `per_share` | `"1"` | `"$"` |
| Weighted average share count | `count/basic` | `count` | `"1000000"` | `null` |

These are the native profiles currently extracted by this project. The source
scale is a string representing the retained multiplier, not an instruction to
convert amounts. A printed dollar sign remains ISO-unverified. `USD`, another
currency, a different scale or a section/unit conflict cannot receive a guessed
or converted value.

`period_kind="fiscal_year"` requires `duration_months=null` and matches an actual
annual source profile. `period_kind="months"` requires an integer month duration;
it does not treat a fiscal year as twelve months. The date is a real canonical
`YYYY-MM-DD` source-column end date. A comparison column may predate the fiscal
identity of the containing report.

An omitted or null `document_ids` uses the configured snapshot's reports. An
explicit scope contains unique registered identifiers. Unknown identifiers
withhold the whole scope; the selector never silently intersects away a bad ID.
A fiscal restriction outside an otherwise valid scope requires clarification.
If more than one exact source cell remains, the result also asks for
clarification rather than choosing the first report.

## Interpret the source outcome

A completed selection uses HTTP 200 with one of `source_candidate`, `clarify`,
`refuse` or `unsupported`. Only `source_candidate` contains a cell, and it
contains exactly one. Withheld outcomes contain an empty list. The response
retains the typed `selection_request` and its canonical
`selection_request_sha256`, plus `index_sha256` and the ordinary request ID.
It has no fabricated natural-language parse or mention offsets.

The request digest uses the eleven typed members, including null optional
defaults and the input document list or null. It uses UTF-8, sorted JSON keys,
compact separators, no nonfinite numbers and one final newline. The requested
scope is not replaced by an inferred list in this digest. `index_sha256` binds
the complete accepted in-memory reports, cells and omissions separately.

| Outcome | Reason | Meaning |
| --- | --- | --- |
| `source_candidate` | `EXACT_TYPED_FIELDS_MATCH` | Exactly one native cell matches all supplied constraints |
| `refuse` | `OPERATOR_REPORT_UNAVAILABLE` | At least one explicit scoped ID is unregistered |
| `refuse` | `REPORT_UNAVAILABLE` | No report satisfies the fiscal restrictions |
| `clarify` | `REPORT_SCOPE_CONFLICT` | Fiscal restrictions and the otherwise valid document scope do not overlap |
| `clarify` | `REQUEST_CONFLICT` | A fiscal-year period also specifies a quarter |
| `refuse` | `NO_INDEXED_CELL_MATCH` | No retained native cell has the complete requested field identity |
| `clarify` | `MULTIPLE_SOURCE_MATCHES` | More than one native source cell remains |
| `unsupported` | `REQUESTED_UNIT_KIND_MISMATCH` | Row section and requested quantity kind conflict |
| `unsupported` | `REQUESTED_SCALE_CONVERSION_UNSUPPORTED` | The requested scale differs from the native quantity profile |
| `unsupported` | `CURRENCY_NOT_APPLICABLE` | A share-count request supplies currency |
| `unsupported` | `REQUESTED_CURRENCY_MISMATCH` | A monetary or per-share request omits its printed dollar symbol |
| `unsupported` | `ISO_CURRENCY_UNVERIFIED_NO_FX` | A monetary or per-share request supplies a currency other than the printed dollar symbol |

Syntax validation precedes these outcomes. For a well-formed request the selector
checks scope/report conflicts, then unit semantics, then exact native cells.
The reason reports the first failed boundary; it is not a list of every possible
problem in the request.

The complete native cell retains its PDF and page-text hashes, physical page and
printed page label, row/column identity, period kind/date/duration, printed and
normalized amounts, section-qualified unit, currency status, native text spans
and rectangles. The same cell identity survives selection; the adapter does not
replace it with a quote assembled from a model response.

`financial_correctness="unestablished"` and
`score_semantics="exact-source-field-selection;no-cosine-score"` apply to every
outcome. The returned amount is a source candidate for a consuming application
to inspect. It is not a reviewed financial answer, calibrated probability or
admission to generation.

## Separate malformed requests from unavailable selection

Bearer authentication runs before reading the body. Requests require JSON and
are limited by actual streamed bytes to 8192 bytes. Unknown fields, duplicate
decoded JSON members, nonfinite values, type coercion, missing required nulls,
invalid dates and inconsistent period shapes are rejected without reflecting
private input values. Authentication, body and operational errors retain their
separate HTTP statuses.

| Condition | HTTP status | Response |
| --- | ---: | --- |
| Missing or invalid bearer | 401 | Authentication refusal before reading body |
| Actual streamed body exceeds 8192 bytes | 413 | Fixed size-error message |
| Non-JSON content type | 415 | Fixed media-type message |
| Malformed or out-of-contract JSON | 422 | `{"detail":"Invalid statement field request."}` |
| Index absent or processing unavailable | 503 | Typed unavailable envelope, empty cells |

An absent startup index or a selection/validation/serialization fault returns
HTTP 503 with no cells and null typed-request metadata. An established snapshot
hash remains available. Private exception messages and partial candidates are
withheld. Request handling uses the accepted in-memory snapshot; changing a PDF
on disk later neither refreshes its cells nor proves the snapshot current.

The existing `/v1/statement-candidates` literal-question interface and
`/v1/query` semantic/lexical interface retain their own contracts. A typed
selection is not evidence that an open-ended question was interpreted correctly.
The [field embedding diagnostic](source-field-bge-diagnostic.md) continues to
record the observed dense retrieval gap.

## Verification boundary

The [dated local acceptance record](acceptance/typed-statement-fields-20261014.json)
binds 1038 passing checkout tests, a fresh 24-module wheel with 31 wheel RECORD
rows, and two separate installed executions. Independent generated native PDFs
exercise six real HTTP modes with 254 assertions, including source withholding
and injected operational faults. The installed corpus launcher exercises three
startup modes with 57 assertions, including the typed monetary, per-share and
count paths. Assertion counts are not counts of distinct HTTP requests.

A separate original-source replay rebuilds the same ten-report snapshot: 531
cells and six omissions. It sends 531 source-derived typed requests and four
negative requests over actual loopback HTTP; every complete candidate equals
its native source record. The 411 monetary, 60 per-share and 60 count fields
retain their existing identities. Original PDF, raw-corpus, vector and index
inputs remain unchanged. None of these checks executes an encoder or generator.

Synthetic native-PDF unit tests and installed real-HTTP checks verify the named
selection, transport, source-retention and refusal behavior. Requests derived
from the retained ten-report source index are development replay, not unseen
questions. Independent human financial labels, held-out financial accuracy,
generation quality and hosted-service behavior require their own evidence.
