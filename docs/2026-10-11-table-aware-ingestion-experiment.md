> Portable-path note: <original-project>, <staged-project> and <browser-project> identify operator-supplied checkout locations. Substitute your own paths; dated local evidence and hashes below are retained. Local receipt artifacts remain excluded from the source release.

# FY2025 Q2 net-income cell ingestion: plan and bounded increments 01–02

> **Current status, 2026-10-11:** Bounded increment 01 extracts the source-bound cells, and increment 02 compares explicit-field selection with raw-source coverage. Both are implemented, source-replayed and independently reviewed. The remaining retrieval, independent acceptance-set, package-adapter and browser-lineage work in steps 6–8 is pending. The original plan is retained below so that proposed work is distinguishable from executed work. No embedding/model run, retrieval integration or human acceptance is established by these Python increments. The separately recorded browser native v4 runs are described below.

## Implemented increment 01

The standalone [extractor](../acceptance/table_cells_experiment.py) and [21 focused tests](../tests/test_table_cells_experiment.py) now derive the four Net income cells from the frozen original FY2025 Q2 PDF, physical page 3, with PyMuPDF 1.28.2 sorted text and word geometry. Neither expected-answer strings nor candidate labels are inputs to cell discovery. Each record binds its statement, row, duration, date, unit, raw amount, printed page, text spans and bounding boxes to the exact source/profile. Ambiguous or missing headers, crossed columns, inconsistent text/geometry, wrong source hash and unsupported input geometry are refused.

| Source duration | Source column date | Source amount |
|---|---|---:|
| Three months ended | July 28, 2024 | 16,599 |
| Three months ended | July 30, 2023 | 6,188 |
| Six months ended | July 28, 2024 | 31,480 |
| Six months ended | July 30, 2023 | 8,232 |

The exclusively created output is [artifacts/table-net-income-experiment-20261011-01/cells.jsonl](../artifacts/table-net-income-experiment-20261011-01/cells.jsonl), 10,634 bytes, SHA-256 `3f6c35f8d7fcbf0b960be74fc168b0e97ff77e558b5ac63b3ff94fe3ac70d37f`. The [source/digest receipt](../artifacts/table-net-income-experiment-20261011-01/receipt.json) records the PDF/page/profile and extractor hash. The [preservation and independent-review receipt](../artifacts/table-net-income-experiment-20261011-01/preservation-review.json) records 21 passing tests, an independent SPEC then QUALITY review without a blocking finding, a second 21-test run and two actual-PDF re-extraction/deep-comparison replays. All ten monitored original/corpus/package/candidate/wheel files remained byte-identical; no combined corpus was written.

These records are `AI-source-reviewed-experimental`, reviewer `CodexAI`, `human_acceptance=false`, with zero embedding/model runs and `retrieval_integration=not-run`. Currency is the literal source glyph `$`; normalized currency remains unset. The sidecar establishes a bounded source-cell contract, not human gold, general table parsing, retrieval quality or faithful generation. Its PyMuPDF Unicode-code-point offsets and PDF-point geometry remain separate from browser PDF.js UTF-16 bindings; there is no browser lineage bridge.

Reproduce the immutable output's source replay from the staged Python repository:

```powershell
& '.\.venv-acceptance\Scripts\python.exe' -I -B '.\acceptance\table_cells_experiment.py' --pdf '<original-project>\data\raw\nvidia-fy2025-q2.pdf' --output-dir '<staged-project>\artifacts\table-net-income-experiment-20261011-01' --verify-existing
```

**Goal:** Add conservative, source-bound Net income cells to one FY2025 Q2 income-statement experiment, then measure field coverage separately from retrieval ranking and generated-answer validity.

**Architecture:** Keep the existing raw text corpus and the accepted API wheel intact. A standalone experimental extractor should emit immutable cell records with exact row, duration, date-column, unit, value and source bindings, plus a compatible retrieval-text sidecar. Reject ambiguous geometry instead of guessing a column or repairing a model answer. Promote an adapter into the package only after this isolated comparison and independent review.

**Tech stack:** Existing PyMuPDF 1.28.2, Python standard-library JSON/SHA-256 and the source-candidate replay profile. Initial extraction and field lookup require no embedding or language model. BGE, browser MiniLM and Qwen comparisons remain separate later gates.

## Implemented increment 02: explicit fields and source coverage

The standalone [comparison](../acceptance/table_field_coverage_experiment.py) uses four frozen question IDs with explicitly declared `metric`, `duration_months`, `period_end` and `unit_literal` profiles. It does not parse their natural-language questions or select using expected-answer strings. It first checks pinned raw/candidate/cell/extractor identities, freshly replays the four source candidates against the original PDF, re-extracts and deep-compares increment 01, and verifies the original raw chunk value spans. Candidate amounts are assertions after cell derivation and selection. No corpus is merged or re-embedded.

| Frozen development profile | Raw bound value | Raw lexical row/duration/date/unit/value present | Net income-only sidecar exact selection |
|---|---:|---|---|
| Revenue / 3 months / 2024-07-28 | 30,040 | Yes | `NO_EXACT_CELL`; Revenue is outside this sidecar |
| Revenue / 6 months / 2024-07-28 | 56,084 | Yes | `NO_EXACT_CELL`; Revenue is outside this sidecar |
| Net income / 3 months / 2024-07-28 | 16,599 | Yes | Unique source-derived cell, value `16,599` |
| Net income / 6 months / 2024-07-28 | 31,480 | Yes | Unique source-derived cell, value `31,480` |

Thus raw bound-value and lexical coverage are each 4/4, and exact profile selection from the Net income-only sidecar is 2/4 across the four development questions. Raw lexical presence is explicitly separate from semantic field entailment: the raw chunk contains two `Jul 28, 2024` occurrences, and this diagnostic does not choose a raw column from that repeated text. It makes no raw-cell selection, retrieval rank/cosine, NLU, accuracy or generation claim. The fresh page-source replay and the cell geometry provide separately identified bindings.

Nine synthetic contract controls passed: unsupported duration, uncovered year, unsupported row, uncovered unit, missing duration, missing unit, duplicate agreeing cells, conflicting cells and a missing cell-unit binding. These controlled record/query mutations are not independently adjudicated financial negative questions. The financial negative count stays zero and the false-accept rate stays null. All question candidates and controls remain development-exposed, assistant/source-reviewed and `human_acceptance=false`.

The exclusively created [comparison.json](../artifacts/table-field-coverage-experiment-20261011-01/comparison.json) is 15,775 bytes, SHA-256 `38576f1d7529ffeea3eadd7dbbe61f1b2ee1c2b0f4d70000f4b89fac96ebdfec`. Its [digest/code receipt](../artifacts/table-field-coverage-experiment-20261011-01/receipt.json) and [permanent review/preservation receipt](../artifacts/table-field-coverage-experiment-20261011-01/review-preservation.json) identify the execution and review separately. The [24 new focused tests](../tests/test_table_field_coverage_experiment.py) plus the 21 extraction tests passed together (45/45); an independent SPEC then QUALITY review found no blocker within the frozen scope, independently reran the 24 comparison tests and reproduced the actual-PDF/raw-corpus `--verify-existing` comparison. All ten preserved inputs/package/wheel files remained byte-identical.

```powershell
& '.\.venv-acceptance\Scripts\python.exe' -I -B '.\acceptance\table_field_coverage_experiment.py' --pdf '<original-project>\data\raw\nvidia-fy2025-q2.pdf' --raw-corpus '<original-project>\artifacts\nvidia-fy2025-q2\chunks.jsonl' --output-dir '<staged-project>\artifacts\table-field-coverage-experiment-20261011-01' --verify-existing
```

This closes only the exact-field/source-coverage part of step 6. Steps 6–8 remain pending for actual retrieval, an independently frozen financial negative/acceptance set, package integration and a separate browser geometry/lineage plan. No PyMuPDF offset is substituted for a PDF.js UTF-16 binding.

## What the observed failures establish

The site preserves four native revenue runs on the same reader-selected financial excerpt. They diagnose output behavior; none closes the structured Net income ingestion gap:

| Version/configuration | Actual prompt/output IDs | Actual elapsed | Termination and raw behavior | Accepted financial draft |
|---|---:|---:|---|---|
| v1, top 3 E1/E4/E2 | 1551 / 128 | 49,749.6 ms | length; fenced/truncated object copied evidence metadata | No |
| v2, same complete top 3 JSON | 1664 / 5 | 38,928 ms | EOS; example text `brief answer from evidence` | No |
| v3, same top 3 JSON | 1627 / 27 | 37,737.7 ms | EOS; plain revenue sentence without required JSON | No |
| v3, top 1 E1 | 794 / 11 | 29,729.6 ms | EOS; object with only `answer`, missing citations/insufficient_evidence | No |

The receipts are under `<browser-project>/docs/acceptance-2026-10-11/`, named `native-financial-baseline.json`, `native-financial-v2.json`, `native-financial-v3.json`, and `native-financial-v3-top1.json`. Top 1 changes selected evidence and prompt length, so it is not the same full-input test as top 3. Each timing is one attempt, not a latency distribution or isolated GPU measurement. These development-exposed questions must not become an unseen acceptance set.

v4 initially passed a separately reviewed framing check: v3 system unchanged, full escaped JSON preserved between frozen task prefix/suffix, 1726 measured prompt IDs plus the same 128 reserve in a 2048 window on that original baseline input. Actual browser native v4 runs have since been recorded and freshly replicated on all four frozen development questions. The [canonical four-case replay](../../zhentong-ye-site/docs/acceptance-2026-10-11/financial-native-v4-four-case-canonical-replay.json) verifies actual disclosed inputs, EOS, exact JSON/schema and selected citation identities for all four drafts; the [old/new comparison](../../zhentong-ye-site/docs/acceptance-2026-10-11/financial-native-v4-old-new-comparison.md) preserves the separate attempts and source profiles. These fresh questions have 1744 prompt IDs for Revenue and 1745 for Net income, rather than the earlier baseline's 1726; each retains the same 128 output reserve.

The fresh native outputs were `30,040` for both Revenue durations and `16,599` for both Net income durations. The three-month bare amounts agree with the assistant-reviewed candidates, while both six-month questions select the three-month amount despite the requested six-month values appearing in the cited evidence. All four answers omit explicit metric/period/unit wording. This is a four-positive development diagnostic, not heldout accuracy; financial correctness and human acceptance remain unestablished. E4 contains the Net income row on the Comprehensive Income statement and was independently checked against original physical page 4, so it is alternate Net income evidence; it contains no Revenue row. Formatting acceptance has therefore progressed while duration selection still fails. These browser runs did not consume this Python sidecar, and improving formatting does not add a structured field that ingestion never emitted.

## The specific ingestion omission

The staged `src/nvidia_rag/table_structure.py:294-365` only permits quarterly metrics `Revenue` and `Operating income`; `src/nvidia_rag/ingest.py:212` requests those two metrics. The existing structured FY2025 Q2 comparison therefore contains eight cells: two metrics × two durations × two dated columns. It contains zero structured Net income cells, although the raw page/chunk contains the row and values.

The current quarterly candidate text infers the duration group from column index and emits a `Chunk` without its row, period or amount boxes. Merely adding `Net income` to the allowlist would add coverage, but it would not prove that the three-month/six-month headings were geometrically bound to the selected columns or preserve the full source-cell record. This experiment should resolve those two shortcomings before expanding to more tables.

The smallest scope is **one registered PDF, physical page 3, one income statement, the Net income row and its four dated columns**. Current-year three- and six-month cells are already source-bound development candidates. Prior-year cells may be extraction candidates but require their own source review before being used as labels. The experiment is not a general table parser, OCR system, cross-page merger or arithmetic service.

## Frozen source and baseline identities

| Input/reference | Verified identity |
|---|---|
| Original local PDF | `<original-project>/data/raw/nvidia-fy2025-q2.pdf`; 80 physical pages; SHA-256 `35d94221f0ed30f974b4fe6536fa54ff40945265d8b354b0ae17066c8856a861` |
| Physical page 3 sorted text | PyMuPDF 1.28.2, `get_text('text', sort=True)`, no normalization; SHA-256 `224e3c92eb9d84d69dff20d606f4b3e3286aff01719fbc114f29855179954d3d` |
| Portable development candidates | `evaluation/source-candidates/fy2025-q2-income-duration.jsonl`; SHA-256 `9cf18ce2edd0a0348df794e4b5526c6155eef869525e9b22385b007f16deb21d`; four positives; CodexAI source review, `human_acceptance=false` |
| Original raw quarter corpus | Original `artifacts/nvidia-fy2025-q2/chunks.jsonl`; SHA-256 `33bac1bf14149c1538e003dee78cdab12e71818687961480c1e7c7b061d18e8a`; target chunk `nvidia-fy2025-q2:p3:bc9bbf8286dc3763` |
| Original eight-document word-safe corpus | Original `artifacts/period-bound-word-safe-400-0/chunks.jsonl`; SHA-256 `bb026279f7b9f41df339d252c404baabd5b403071cbb1a858a4303322c89c331` |
| Accepted API wheel to preserve | `artifacts/api-failure-20261011/final/nvidia_financial_rag-0.1.0-py3-none-any.whl`; SHA-256 `1f2e95aa8866e2f66fc4422583c257b9b07a6aa420db59f16ccc883fd2a5d8d6` |

This author independently reopened the exact local PDF and verified its hash, PyMuPDF version, page-text hash, four candidate records and all named source spans. No PDF, question or corpus was modified. The source-binding script's PDF-only replay does not independently verify raw-corpus bytes; retain that distinction when using its receipt.

The two existing Net income candidate bindings are:

| Candidate ID | Duration / end | Raw value | Sorted page-text value span | Source value box in PDF points |
|---|---|---|---|---|
| `q2_regression_net_income_three_months` | 3 months / 2024-07-28 | `16,599` | `[2594,2600)` | `[328.9570007324219,357.16741943359375,353.726806640625,366.21661376953125]` |
| `q2_regression_net_income_six_months` | 6 months / 2024-07-28 | `31,480` | `[2632,2638)` | `[486.4007568359375,357.16741943359375,511.1705322265625,366.21661376953125]` |

These are source-review candidates, not independently adjudicated gold or investment advice. Offsets are zero-based half-open Unicode code-point positions under the declared extraction profile. Browser PDF.js UTF-16 spans, its two-page excerpt hash and its physical pages are separate identities; never copy these offsets/page numbers into browser evidence or equate them without a verified lineage mapping.

## Candidate record and refusal contract

A new experimental cell record should retain the following fields, all derived from exact source bytes and measured geometry. No amount, period or label should be constructed from an expected answer:

| Record field | Required evidence |
|---|---|
| `cell_id`, extraction profile/version | Stable digest of source hash, physical page, table, row, duration, date-column and raw value binding; changing parser/profile must change identity |
| Source/document identity | Registered document ID, PDF SHA-256, physical page and separately printed label |
| Table and metric | Exact statement-heading and Net income row text, named text spans and boxes |
| Duration and end date | Exact Three/Six Months Ended header and the selected date header, plus geometry showing that the date column belongs to that group |
| Value | Original lexeme, exact page-text span and positive-area value box; parsed numeric value is a separate representation |
| Unit/currency | Exact unit-heading text and span; source currency literal `$`; any USD normalization remains explicitly reviewed/candidate status |
| Retrieval representation | Deterministic text derived from the record; never replace raw evidence; retain a binding back to `cell_id` |
| Review status | Development exposure, assistant/source reviewer identity, `human_acceptance=false` and separate human verdict |

Extraction returns a cell only if one row, one duration group, one date column and one value can be uniquely associated. Return a fixed rejection reason for absent row/header/unit, multiple matches, crossed date groups, duplicate cell identity, value outside the row or conflicting source hashes. Do not choose the largest value, nearest year or most similar text to resolve ambiguity. Cross-page tables and unsupported layouts return a scope rejection; do not silently splice them.

## Original implementation sequence and remaining gates

The initial proposal named `experiments/fy2025_q2_net_income_v1.py`, `experiments/test_fy2025_q2_net_income_v1.py`, and `artifacts/table-net-income-development-v1/`. Those historical names were not created. The approved implementation uses the `acceptance/`, `tests/` and `artifacts/table-net-income-experiment-20261011-01/` paths linked above. `src/`, `pyproject.toml`, API schemas, original PDFs/corpora and historical label files remain outside this experiment.

1. [x] Freeze the experiment's own PDF/page/profile input receipt and refuse changed hashes/version; re-extract and deep-compare all emitted cells against the actual PDF. The four existing candidate replay remains a separately named baseline, and the historical fresh-receipt recipe below was not executed under that filename.
2. [x] Add controlled failing tests for the exact Net income row, duration/date geometry, missing/duplicate headers, value position, duplicate identity and text/geometry consistency; 21 focused tests passed. Cross-page input is outside the frozen page-3 extraction scope, rather than a supported merge capability.
3. [x] Implement the isolated extractor and exclusive four-cell output with source/profile/code/digest receipt. Discovery does not read expected answers or candidate labels.
4. [x] Replay all emitted bindings by actual-PDF re-extraction/deep comparison. The two existing current-year bindings match exactly; four cells were emitted. Prior-year records are source-reviewed experimental cells, not newly created answer labels.
5. [x] Preserve separately identified raw and old structured baselines plus the new standalone sidecar. All ten monitored files, including the accepted API wheel, stayed byte-identical. No combined retrieval corpus or adapter was created, and overlapping variant counts are not added together.
6. [ ] Compare exact field coverage first, then retrieve on the same four frozen positive questions with fixed k/gate/model/config. Report row/duration/date/unit/value support separately from page hit, rank and cosine. No new unsupported-field/negative denominator exists in those four records; negative false-accept rate remains null.
7. [ ] Before inspecting new acceptance results, independently source-review and freeze wrong-duration, wrong-year, missing-unit, unsupported-row and conflicting-column controls. Label exposure and reviewer roles explicitly. A new human-reviewed unseen set must be created separately rather than renaming the exposed four-case diagnostic.
8. [ ] Only after the isolated cell/source contract passes, propose a package adapter or browser lineage bridge. Revalidate installed-wheel/API contracts and benchmark generation independently. Table coverage success must not alter the strict output grammar, increase model budgets or promote assistant-reviewed labels into human gold.

For step 1, run from `<staged-project>` with a fresh output path whose parent directory already exists:

```powershell
& '.\.venv-acceptance\Scripts\python.exe' -I -B '.\acceptance\source_binding_replay.py' --pdf '<original-project>\data\raw\nvidia-fy2025-q2.pdf' --output '<staged-project>\artifacts\source-binding-fy2025-q2-20261011\net-income-experiment-input-receipt-v1.json'
```

This is a historical proposed recipe; it was not executed to create that specifically named receipt. Increment 01 used its own source/digest and preservation/review receipts linked above. Do not reuse an existing output filename or overwrite historical reports to make a replay appear fresh.

## Completed contracts and remaining acceptance gates

Increment 01 passed the bounded cell/source contract: four source-derived records and both existing current-year candidate bindings replay, geometry/text perturbations refuse ambiguity, and raw/historical records plus the staged API wheel remain intact. Increment 02 passed the explicit-field and raw-source coverage comparison with independently reproduced tests and source replay. These establish local source/selection contracts. Steps 6–8 still require retrieval evaluation, a separately frozen financial acceptance set, package integration and a separate browser lineage plan. Neither increment proves better heldout retrieval, faithful generation, enterprise scale, public source redistribution rights or learner mastery.
