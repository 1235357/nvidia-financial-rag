> Portable-path note: <original-project>, <staged-project> and <browser-project> identify operator-supplied checkout locations. Substitute your own paths; dated local evidence and hashes below are retained. Local receipt artifacts remain excluded from the source release.

# Evaluation readiness and corpus map — local audit, 2026-10-11

This audit connects existing local financial data to the staged API work. It
reads the original project without changing its code, sources, labels or reports.
Counts and hashes below were checked against files during this audit. No new
retrieval quality score, model run, paid request or human adjudication was made.

Local roots used here:

- **Original:** `<original-project>`
- **Staged:** `<staged-project>`

The staged repository contains API/package code, synthetic acceptance tests and
four exported source-reviewed development candidates. It contains no exported
financial PDF or corpus JSONL. The candidate package preserves original question
bytes and source bindings; an operator supplies the exact registered PDF locally.
Passing the installed API failure contract or source-binding replay does not
close financial quality, independent adjudication or full corpus portability.
See [the separate API contract](api-failure-contract.md).

## Portable four-case development candidate package

`Staged/evaluation/source-candidates/fy2025-q2-income-duration.jsonl` contains
exactly the four existing quarter-income diagnostic questions. Dataset SHA-256:
`9cf18ce2edd0a0348df794e4b5526c6155eef869525e9b22385b007f16deb21d`.
Each row is explicitly `label_status=assistant-source-reviewed-candidate`,
`reviewer=CodexAI`, `human_acceptance=false`, `human_verdict=null`, and
`freeze_purpose=development-replay`. No negative financial question was added.

Each record binds the report URL and PDF hash, physical/printed page, statement,
row, duration, end date, source currency/unit qualification, value, named text
spans and source-cell bounding box. The extraction profile specifies PyMuPDF
1.28.2, sorted text, no normalization and half-open Unicode code-point offsets.
Original JSONL records, including their terminating LF, are retained verbatim
with per-record SHA-256 values. Concatenating the four records reconstructs the
original question file's bytes and hash. The original chunk ID/hash/value spans
are also retained as provenance metadata.

`Staged/acceptance/source_binding_replay.py` runs without model or retrieval
imports and requires `--pdf` to be an absolute local path. It checks source bytes
before opening the PDF, exact extraction version/hash, headers, row/duration/date
column bindings, value spans and value bounding boxes. It reconstructs and checks
the original question bytes. It validates provisional review/development status
and rejects metadata that promotes these candidates into human acceptance.
Original chunk metadata is retained; this PDF-only script does not independently
reopen or verify the original corpus JSONL. Output receipts are created with
exclusive write mode and are never overwritten.

Run from the staged repository with a new receipt path:

```powershell
& '.\.venv-acceptance\Scripts\python.exe' -I -B '.\acceptance\source_binding_replay.py' --pdf '<original-project>\data\raw\nvidia-fy2025-q2.pdf' --output '<absolute-new-receipt-path.json>'
```

The actual local run passed all four source bindings, with zero model runs.
Its receipt is
`Staged/artifacts/source-binding-fy2025-q2-20261011/source-binding-receipt.json`.
This closes portability of four candidate labels and their replay, while requiring
the operator's own exact local PDF. No PDF was copied or exported. The receipt
does not contain a model quality score or an independent human verdict.

## Actual local corpus coverage

`Original/corpus/sources-annual-plus-fy2025-q2-experiment.json` contains eight
source registrations. Each local PDF's SHA-256 matched that registry in this
audit. Opening all eight PDFs confirmed **1,153 physical pages**. The structured
comparison artifact at `Original/artifacts/structured-label-clean-20260929/`
has **4,126 actual JSONL chunks**, with the following distribution:

| Registered document | Local PDF under `data/raw/` | Actual physical pages | Actual chunks in structured comparison |
|---|---|---:|---:|
| `nvidia-fy2021-annual` | `nvidia-fy2021-10k.pdf` | 93 | 431 |
| `nvidia-fy2022-annual` | `nvidia-fy2022-annual.pdf` | 207 | 700 |
| `nvidia-fy2023-annual` | `nvidia-fy2023-annual.pdf` | 169 | 623 |
| `nvidia-fy2024-annual` | `nvidia-fy2024-annual.pdf` | 187 | 656 |
| `nvidia-fy2025-annual` | `nvidia-fy2025-annual.pdf` | 181 | 682 |
| `nvidia-fy2026-annual` | `nvidia-fy2026-annual.pdf` | 175 | 647 |
| `nvidia-fy2025-q2` | `nvidia-fy2025-q2.pdf` | 80 | 226 |
| `nvidia-fy2027-q2` | `nvidia-fy2027-q2.pdf` | 61 | 161 |

The 4,126 chunks include **1,101 structured table candidates**: 1,085 `:table:`
and 16 `:quarter-table:` chunks. The manifest declares 1,137 nonempty pages;
that specific nonempty-page count was not independently recomputed here.
This is an experimental corpus inventory, not a claim that the live default
service uses this artifact or that all extracted cells are financially correct.

Other concrete comparison artifacts remain available:

| Directory under `Original/artifacts/` | Documents | Manifest chunk count | Purpose/boundary |
|---|---:|---:|---|
| `nvidia-fy2025-q2` | 1 | 218 | Original 400-token/50-overlap quarter baseline; actual JSONL count checked |
| `annual-fy2022-fy2026-fixed-400-50` | 5 | 2,399 | Annual fixed-token comparison |
| `annual-fy2022-fy2026-recursive-500-50` | 5 | 1,999 | Annual recursive comparison |
| `annual-fy2022-fy2026-fixed-400-0` | 5 | 2,265 | Annual no-overlap comparison |
| `annual-fy2022-fy2026-word-safe-400-0` | 5 | 2,268 | Annual word-boundary comparison |
| `annual-recursive-word-fallback-400-50` | 5 | 2,418 | Recursive word-fallback comparison |
| `annual-plus-two-quarters-fixed-400-50` | 7 | 2,770 | Mixed annual/quarter comparison |
| `annual-plus-two-quarters-structured-400-50` | 7 | 3,687 | Mixed structured comparison |
| `period-bound-fixed-400-0` | 8 | 2,862 | Explicit period-routing comparison |
| `period-bound-fixed-400-50` | 8 | 3,025 | Explicit period-routing comparison |
| `period-bound-word-safe-400-0` | 8 | 2,865 | Quarter diagnostic source; actual count and manifest hash checked |

These are variants over overlapping reports. Adding their counts does not yield
a larger independent corpus. Hash-linked source acquisition and local extraction
are distinct from permission to redistribute files or a third-party data review.

## Existing evaluation coverage and exposure

There are **15 JSONL suites**, **109 file memberships** and **104 distinct
question IDs** under `Original/evaluation/`. The memberships contain 77 positive
and 32 negative labels; those numbers include overlapping versions and are not
an independent sample-size claim. Three FY2021 IDs appear in both prospective and
adjudicated files; two balance-sheet IDs appear in v4 and its field-label migration.
Version selection must preserve changed labels rather than silently deduplicate.

| File under `evaluation/` | Positive | Negative | Label/exposure boundary |
|---|---:|---:|---|
| `questions.jsonl` | 8 | 2 | FY2025 Q2 seed; page-only labels, no explicit document IDs |
| `heldout_questions.jsonl` | 8 | 4 | Historical quarter heldout; now inspect exposure before reuse; page-only labels |
| `annual_multiyear_seed.jsonl` | 8 | 3 | Five annual reports; expanded development file |
| `annual_multiyear_heldout.jsonl` | 6 | 4 | Every row now explicitly registered as development-exposed |
| `financial_table_frozen_v1.jsonl` | 5 | 5 | Existing parser/scope development; filename does not establish independence |
| `balance_sheet_prospective_v1.jsonl` | 6 | 3 | Earlier balance-sheet candidate version |
| `balance_sheet_validation_v2.jsonl` | 6 | 3 | Separate version; source-note/PDF-hash fields present |
| `balance_sheet_prospective_v3.jsonl` | 6 | 4 | Another field/date/unit candidate version |
| `balance_sheet_prospective_v4.jsonl` | 6 | 4 | Adds explicit table, row, as-of date, currency and unit fields |
| `balance_sheet_field_review_v1.jsonl` | 2 | 0 | Development migration of two v4 IDs; exclude duplicate versions from combined counts |
| `table_layout_prospective.jsonl` | 3 | 0 | Three annual table-layout candidates |
| `fy2021_10k_prospective.jsonl` | 3 | 0 | Earlier FY2021 candidate page labels |
| `fy2021_10k_adjudicated.jsonl` | 3 | 0 | Explicitly post-retrieval correction; no reviewer identity authenticating independent human adjudication |
| `fy2027_q2_prospective.jsonl` | 3 | 0 | Revenue, Data Center and gross-margin candidates |
| `quarter_income_regression_20260930.jsonl` | 4 | 0 | Repeated revenue/net-income duration diagnostic; explicitly not independent heldout |

Fourteen memberships have `financial_ground_truth`. Their reviewer strings all
identify Codex/assistant source review. None of the 109 memberships has both an
explicit acceptance-purpose registration and a nonempty freeze date. Absence of
those fields does not prove that a question was never seen; it leaves independence
unestablished. This audit does not rename any candidate label as human gold.

The original artifact inventory has **26 retrieval evaluation JSON files** and
one two-record generation-review JSON file. They span different corpus variants,
question versions and gates; they are not 26 independent quality studies. For
example, the annual seed is now 11 rows while older annual seed reports contain
eight records. Some old retrieval reports have no input-hash provenance block.
Do not attach their metrics to today's edited labels without matching bytes.

`Original/artifacts/annual-fy2022-fy2026-fixed-400-50/generation-review-two-controls-20260930.json`
has one completed generation rejected for missing citation and one no-evidence
case with generation not run. Both `human_verdict` fields are null. It provides
no independently adjudicated answer-accuracy denominator.

## FY2025 Q2 source-linked candidate cells

Original source: `Original/data/raw/nvidia-fy2025-q2.pdf`.
Registered publisher URL:
<https://s201.q4cdn.com/141608511/files/doc_financials/2025/q2/78501ce3-7816-4c4d-8688-53dd140df456.pdf>.
The local PDF is **80 pages**, SHA-256
`35d94221f0ed30f974b4fe6536fa54ff40945265d8b354b0ae17066c8856a861`.
No live remote-byte verification was performed in this subtask.

Page 1 says the quarterly period ended July 28, 2024. Page 12, Note 5, explicitly
uses second quarter/first half of fiscal year 2025. Physical page 3, also printed
page 3, was rendered and inspected in this audit. Its income statement has two
duration groups, each with current and prior-year columns. The unit heading is
`In millions, except per share data`; the revenue and net-income cells use `$`.
Existing labels normalize these as USD millions. Page 3 itself prints `$` rather
than `USD`; that currency normalization remains part of candidate review.

| Existing diagnostic ID suffix | Statement/field | Duration and column end | Existing candidate value | Sorted page-text value span | Original quarter-chunk value span |
|---|---|---|---|---|---|
| `revenue_three_months` | Condensed Consolidated Statements of Income / Revenue | Three months / Jul 28, 2024 | 30,040 million dollars | `[766,772)` | `[302,308)` |
| `revenue_six_months` | Same statement / Revenue | Six months / Jul 28, 2024 | 56,084 million dollars | `[804,810)` | `[322,328)` |
| `net_income_three_months` | Same statement / Net income | Three months / Jul 28, 2024 | 16,599 million dollars | `[2594,2600)` | `[926,932)` |
| `net_income_six_months` | Same statement / Net income | Six months / Jul 28, 2024 | 31,480 million dollars | `[2632,2638)` | `[945,951)` |

All four IDs have prefix `q2_regression_`. Spans are zero-based, half-open Unicode
code-point offsets, not tokens or UTF-16 offsets. Page spans use PyMuPDF **1.28.2**,
`page.get_text('text', sort=True)` without further normalization. Page-text SHA-256:
`224e3c92eb9d84d69dff20d606f4b3e3286aff01719fbc114f29855179954d3d`.
Header spans are statement `[197,240)`, unit `[309,343)`, three-month heading
`[492,510)`, six-month heading `[530,546)`, current-date columns `[619,631)` and
`[661,673)`, Revenue `[696,703)`, and Net income `[2523,2533)`.

The original chunk containing all four values is
`nvidia-fy2025-q2:p3:bc9bbf8286dc3763`, in
`Original/artifacts/nvidia-fy2025-q2/chunks.jsonl`. These value spans must be read
with the table headers and row/column geometry. Presence of the number in a chunk
does not establish the requested field or duration.

The structured comparison contains eight FY2025 Q2 `:quarter-table:` candidates:
Revenue and Operating income, for both durations and both year columns. It
contains **zero structured FY2025 Q2 Net income candidates**. Raw page chunks
still contain net income. The four-question diagnostic therefore tests coverage
that a revenue/operating-income-only structured scope cannot fully supply.

## Existing result replay and the smallest next evaluation

The four questions are in
`Original/evaluation/quarter_income_regression_20260930.jsonl`, SHA-256
`4027ca9d287894f940d9836467547adc0950082be3536b0c1cbb795247c10ee6`.
The quarter-only chunks hash is
`33bac1bf14149c1538e003dee78cdab12e71818687961480c1e7c7b061d18e8a`.
The eight-document word-safe chunks hash is
`bb026279f7b9f41df339d252c404baabd5b403071cbb1a858a4303322c89c331`.

`Original/artifacts/period-bound-word-safe-400-0/evaluation-quarter-income-regression-null-negative-rate.json`
preserves an existing BGE diagnostic with k=5, gate=0.78, four positive questions
and no negative questions. Its historical gate accepted three of four, while the
single-page-3 labels report a page hit for two of four. The existing AI source
review identifies support on alternate pages 23 and 4. Those are different
measurements; the AI review does not independently establish general correctness.
The report correctly records negative false-accept rate as **null**, with no
negative denominator. The earlier uncorrected report records 0.0 for that same
empty denominator and must not be used as refusal-success evidence.

The source review is
`Original/artifacts/period-bound-word-safe-400-0/source-review/field-review.json`.
It explicitly identifies Codex AI visual review and says independent human
confirmation is required. Its reviewed three-of-four accepted candidates must
not be converted into independently judged answer accuracy. No generated answers
were evaluated by that file.

The smallest defensible next run is a **four-record source-binding replay**:
verify the PDF, questions and chunk hashes; bind each ID to the table, duration,
end date, unit and exact value span above; inspect the existing retrieved candidate
and preserve its historical score/gate result. This needs no model or new corpus.
Keep page-label diagnostics and provisional field-support judgments separate.
Do not change historical gold pages or tune the gate during this replay.

The following read-only preparation command checks the existing source/value
bindings and prints the four frozen result records. It is a reproducible next-run
recipe, not a new retrieval benchmark or a human review:

```powershell
@'
import json
from hashlib import sha256
from pathlib import Path
import pymupdf
root = Path(r'<original-project>')
pdf = root / 'data/raw/nvidia-fy2025-q2.pdf'
assert sha256(pdf.read_bytes()).hexdigest() == '35d94221f0ed30f974b4fe6536fa54ff40945265d8b354b0ae17066c8856a861'
with pymupdf.open(pdf) as doc:
    text = doc[2].get_text('text', sort=True)
assert sha256(text.encode()).hexdigest() == '224e3c92eb9d84d69dff20d606f4b3e3286aff01719fbc114f29855179954d3d'
for value, start, end in [('30,040',766,772),('56,084',804,810),('16,599',2594,2600),('31,480',2632,2638)]:
    assert text[start:end] == value
questions = root / 'evaluation/quarter_income_regression_20260930.jsonl'
assert sha256(questions.read_bytes()).hexdigest() == '4027ca9d287894f940d9836467547adc0950082be3536b0c1cbb795247c10ee6'
chunks = root / 'artifacts/period-bound-word-safe-400-0/chunks.jsonl'
assert sha256(chunks.read_bytes()).hexdigest() == 'bb026279f7b9f41df339d252c404baabd5b403071cbb1a858a4303322c89c331'
report = root / 'artifacts/period-bound-word-safe-400-0/evaluation-quarter-income-regression-null-negative-rate.json'
assert sha256(report.read_bytes()).hexdigest() == 'c361dc7074ab5e9624a246d6bc465884bb2dc502c1fd5d23993059c6e482ff6a'
records = json.loads(report.read_text(encoding='utf-8'))['records']
assert len(records) == 4 and all(row['answerable'] for row in records)
print(json.dumps({'scope':'frozen source-binding replay; no fresh retrieval or human verdict','records':records}, indent=2))
'@ | & '<staged-project>\.venv-acceptance\Scripts\python.exe' -
```

The documented replay was executed during this audit: its source hashes, value
spans and four-record checks passed. It read the existing report and made no new
retrieval or model call.

After that replay, independent review must confirm field/column support and
register new duration, wrong-year, wrong-unit and unsupported-field controls
before their results are inspected. Freeze source/question/index/model/config
identities. Evaluate retrieval ranking, source-cell support, correct refusal and
generated-answer faithfulness separately. Record source reviewer, date, decision,
supporting span and disagreements. Page hit, number match, a valid citation,
synthetic HTTP tests and a reviewer string each establish narrower facts.

This audit makes no claim about independent human quality acceptance, fresh BGE
execution, generation accuracy, performance percentiles, larger enterprise data,
current service corpus, public deployment, or the user's personal mastery.
