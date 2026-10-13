# Route model retrieval by the source report

The configured model launcher uses a source-bound report catalog to distinguish annual reports from quarterly reports before dense ranking. This repairs the observed case where a question about the fiscal 2025 second-quarter report was routed to the fiscal 2025 annual overview merely because it mentioned the same fiscal year. The [preserved model diagnostic](current-source-model-http.md) records that earlier behavior.

This is a report-scope check. Selecting the requested report does not prove that a retrieved passage contains the correct metric, column, unit or answer.

## Choose the intended entry point

| Entry point | Configured behavior |
| --- | --- |
| Source-tree `run_api.py` with all three statement options | Builds a statement snapshot and its report catalog before importing the model/server runtimes. BGE ranks within the selected source-report scope. |
| Source-tree `run_api.py` without statement options | Retains legacy annual routing by default, or the existing explicit source-task policy when configured. Neither uses the new report catalog. |
| Installed `nvidia-rag-serve-corpus` / `python -m nvidia_rag.serving` | Uses the verified lexical corpus. Its optional statement selector and repeatable `--document-id` scope remain independent of the BGE routing change. |

For the model path, supply prepared corpus/vector artifacts and the complete trio `--statement-source-registry`, `--statement-source-dir`, `--statement-fiscal-declarations`. The [startup guide](verified-corpus-startup.md) describes their schemas and local-path requirements. The model launcher has no `--document-id` startup option. Its catalog contains the reports actually declared and built into the statement snapshot; registered reports omitted from that catalog do not enter configured model ranking.

`--source-task` and its paired `--source-registry` represent a different explicit source policy. Combining them with the statement report catalog is refused during argument parsing, before model/server imports. Direct Python composition likewise rejects a catalog combined with an explicit source task or the old annual-only router before passage encoding.

## Bind the catalog before ranking

`build_report_catalog` accepts a source-built statement snapshot and the exact model corpus manifest and chunks. It rechecks retained inputs, full registry/corpus membership and hashes, selected fiscal declarations and selected original PDF identities. Duplicate fiscal report identities, unknown report owners and mismatched inputs fail startup. Report years and quarters come from those declarations, including explicit annual/null-quarter identities; document names and expected evaluation answers do not supply them.

The immutable catalog retains each report's source-derived `(period_end, duration_months)` pairs. Annual columns retain null duration with their native Year Ended semantics. Unsupported table layouts retain their declared report and omissions but add no invented column periods. Both runtime composition and the direct scoped retriever require exact equality with the catalog's bound chunk inputs.

## Keep report identity separate from column identity

With a matching registered report and retained source columns, these bounded literal forms select its Q2 scope:

- `fiscal 2025 second-quarter report`
- `FY2025 Q2`
- `FY2025Q2`

The original question reaches BGE with the existing query instruction; routing does not rewrite it. A request for three or six months ended July 28, 2024 must match a retained date/duration pair in the selected report. A request naming the fiscal 2025 Q2 report can also use its July 30, 2023 comparison column when that exact pair exists. The comparison date does not change the report's fiscal identity. Dates and durations from different columns cannot be combined into an unobserved pair.

A malformed, unavailable, conflicting or ambiguous literal restriction yields no unrelated fallback candidates. Inspection distinguishes unknown reports, unavailable columns, conflicting period restrictions, multiple requested reports and ambiguous report scope. Calendar-year/quarter requests and an unqualified `quarter ended ...` phrase are unsupported by this fiscal grammar. A fiscal year alone can be ambiguous when annual and quarterly reports share it. Multi-report field selection requires clarification in this increment.

A question without literal period restrictions may rank across the configured catalog, including when it is a subset of the full corpus. It cannot rank omitted reports. Inspection retains the allowed document IDs, disposition, reason, considered chunk count and dense arithmetic. Refusal/clarification inspections still encode the unchanged question for the existing trace contract, but score no unrelated chunks.

## Interpret the result at its actual scope

The comparison settings remain threshold `0.78`, Top K `5` and the pinned BGE revision. These settings are not calibrated financial accuracy. The v1 HTTP response keeps its eight fields and legacy behavior: with generation disabled, `can_answer` and `evidence_passed` describe the nonempty similarity-gated passage result while `answer` remains empty. Routing reasons are inspection metadata; this change does not add new HTTP fields or turn those flags into financial verdicts. The independent statement endpoint continues returning source candidates, clarification, refusal or unsupported outcomes under its own contract.

At this documentation stage, the post-change native model comparison is **pending**. It must reuse the same exposed questions, original source/corpus/vector bytes, model revision and threshold in a new frozen attempt. A quarterly positive case must demonstrate correct source scope; a below-threshold result must remain visible rather than being repaired by lowering the gate. Source-field support, generation, independent human evaluation, held-out quality, hosting and load acceptance remain separate gates.
