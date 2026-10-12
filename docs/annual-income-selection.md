# Annual income statements retain their fiscal-year period

The source-bound income index now parses the six registered annual statements as well as the four quarterly statements. Annual cells retain the literal `Year Ended` header, dated column, row, amount, section and source coordinates. Their `duration_months` and report `fiscal_quarter` are null. `IncomeCell.period_kind` is a derived property returning `fiscal_year` for the annual extraction profile and `months` for the preserved quarterly profile.

This distinction follows the source. The FY2021 Form 10-K's Fiscal Year note says fiscal 2021 contains 53 weeks while fiscal 2020 and 2019 contain 52. The FY2026 annual report says fiscal 2026, 2025 and 2024 contain 52 weeks. A `Year Ended` cell therefore must not silently satisfy a literal twelve-calendar-month question. These source notes qualify the period convention; this release does not calculate each cell's week count or period-start date.

## Two date layouts, separate native spans

FY2021, FY2022 and FY2023 print a month/day group above a separate year group. The parser binds these as `date` and `date_year`, retaining each actual text slice and rectangle. It normalizes their joint meaning to an ISO end date without inventing one contiguous date quotation. FY2024, FY2025 and FY2026 print a complete `Jan DD, YYYY` group and need only the complete `date` binding.

Each annual statement must have exactly three dated columns within distinct revenue-currency column bands. Dates descend from current to prior comparisons; `column_number` distinguishes all three positions. The current column has role `current`, while both comparison columns have role `prior_comparison`. An explicit report fiscal-year declaration continues to identify the report, including its printed comparison cells. A prior calendar date does not retarget that source to a different report.

Both `Basic` and `Diluted` occur in EPS and share-count sections. Section-qualified row paths keep them separate. EPS scale is one; monetary and share-count values use the printed millions heading. Printed `$` remains an unverified dollar symbol rather than a proven ISO currency. Natural-language EPS and share-count lookup still need a later section-qualified input adapter.

## Preserve numbers beside missing values

Three annual Acquisition termination cost rows contain six printed em-dash placeholders and three numeric values in neighboring columns. The annual parser keeps each numeric cell whose column and provenance are unambiguous, while recording an omission for each unsupported column. It does not translate a dash to zero. A geometric ambiguity or duplicate row still prevents unsafe row construction. The earlier quarterly row-atomic behavior is preserved.

An independent source program inspected all six original annual PDFs before the final source-only build. The source index subsequently contained **531 cells**: the original 204 quarterly cells plus 327 annual cells. Six unsupported-value omissions correspond to the six actual annual placeholders. An independent replay reopened all ten pinned PDFs without importing the production parser and checked every cell's source metadata, literal spans, word rectangles, dates, sections and identity. All 204 quarterly serialized cell bodies and IDs remained identical to the prior quarterly index.

| Registered annual report | Physical PDF statement page | Annual cells | Date layout |
| --- | ---: | ---: | --- |
| FY2021 | 47 | 54 | Separate month/day and year |
| FY2022 | 167 | 54 | Separate month/day and year |
| FY2023 | 135 | 55 | Separate month/day and year |
| FY2024 | 150 | 55 | Complete date group |
| FY2025 | 144 | 55 | Complete date group |
| FY2026 | 139 | 54 | Complete date group |

Physical pages remain distinct from printed page numbers and nullable PDF labels. For example, the FY2026 statement is physical PDF page 139 and prints page 51; its PDF label is null in the inspected source.

## Ask an explicitly annual question

Use the same installed interfaces documented in [source-bound income selection](structured-income-selection.md), with `ReportInput.fiscal_quarter=None` for an annual report. Supply the exact pinned PDF hash and absolute local path. The literal parser recognizes `annual` or `year ended`, plus an exact date and monetary row:

```python
request = parse_statement_request(
    "According to NVIDIA fiscal 2025 annual report, what was net income "
    "for the year ended January 26, 2025, in millions of dollars?",
    operator_document_ids=("nvidia-fy2025-annual",),
    index=index,
)
selection = select_statement_candidates(index, request)
```

The caller must first build `index` from reviewed `ReportInput` sources. A combined annual and month-duration request requires clarification. A twelve-month request alone does not match a fiscal-year cell. Unknown wording, arithmetic, multiple metrics, FX and ISO-currency requests retain the prior withholding boundaries. Overlapping annual dates in several reports require a sufficiently narrow source scope; the selector does not pick the first convenient report.

The same original 35 question/scope rows were rerun against the broader source index and retained their previous outcomes: 24 source candidates, five refusals, four clarifications and two unsupported requests. This checks preservation of the exposed quarterly comparison; it does not add an annual financial-accuracy denominator.

A separate question freeze derived eighteen annual questions from the independently inspected source dates, each with global and explicit report scope, then added six withholding controls. The actual 42 requests produced 36 source candidates, two refusals, three clarifications and one unsupported result. The source-only index was built before consuming this bank, and no expected amount entered selection. These questions are source-derived, exposed development material: held-out count and human gold are zero. Returning a source candidate is not an independently accepted financial answer.

The existing [comparison command](structured-income-selection.md#reproduce-the-development-comparison) now builds both annual and quarterly layouts from all ten pinned PDF inputs. The observed source-derived annual question bank has its own frozen bytes and local run receipt; it is separate from the committed eighteen-case/35-row diagnostic. Model encoding and generation remain absent from this source-only path. Default API and browser composition, unseen human-reviewed questions and operational performance remain separate implementation and acceptance gates.

The final local checkout passed **661 tests**, with no failures, errors or skips. Six new comparison-closure controls cover a valid annual replay, files changing between validation and the initial snapshot, index bytes changing before their hash is captured, and pending/failure markers appearing after selection. The runner compares its initial snapshot with the already validated identities, binds the index's exact canonical bytes, and repeatedly reads the completed index during receipt closure. Finite independent probes also inserted markers after the comparison writer returned; those runs were rejected and retained a comparison failure guard.

[Sanitized annual development metadata](acceptance/annual-income-development-20261012.json) binds the final implementation, runner, source index, test XML and fresh 35/42-request receipts. The fresh index bytes and all request result rows remain identical to the earlier independently reviewed payloads. The earlier native replay's 14,466 source checks and annual question review's 925 checks keep their original scope and hashes; this byte/result equivalence check does not claim they were rerun. Metadata contains no original report bodies, complete extracted pages, model weights or private local paths. These are local source and integrity observations; the later CI and financial acceptance gates must be reported separately.

The original [quarterly observation metadata](acceptance/structured-income-development-20261012.json) and its source-pinned blog figure describe the earlier `224d02d` checkpoint. They remain preserved historical evidence of that implementation and are not silently rewritten to claim the annual parser existed then.
