# Build source-bound income-field passages

`nvidia_rag.source_passages` creates a separate field-level artifact from the native income-statement snapshot. Each record keeps the complete original `IncomeCell` and a deterministic readable `Chunk` projection. It puts a row, its dated duration, comparison role, value and unit together for a later retrieval experiment. The text is assembled from source fields; it is not a contiguous PDF quotation or an accepted financial answer.

This source-only step does not encode passages, run a model or query, assign similarity scores, generate an answer or change either v1 API. The original 3,240 plain-text chunks and their BGE vectors remain a separate baseline. New passage text requires its own new vector artifact and acceptance before any model/API integration.

## Prepare and build

Install the selected wheel and its core dependencies in the intended Python environment. Keep the matching checkout's example script separately: `examples/build_source_passages.py` is a source-tree CLI that imports the installed package; it is not an installed console command. The package requires Python 3.11 or later.

Prepare the original registered PDFs, the exact source registry and explicit fiscal declarations as described in [verified source startup](verified-corpus-startup.md#prepare-inputs-once). Fiscal declarations bind report year/quarter to PDF and registry hashes. They are operator assertions; the builder does not infer a fiscal year from the PDF filename or column date. No report is downloaded by this command.

The following PowerShell example uses illustrative absolute operator paths. Replace them with the actual installed interpreter, matching script and reviewed inputs. Create the parent output directory first; the final `field-passages-01` directory must not exist.

```powershell
& 'C:\operator\field-env\Scripts\python.exe' -I -B `
  'C:\operator\checkout\examples\build_source_passages.py' `
  --source-registry 'C:\operator\inputs\sources.json' `
  --source-dir 'C:\operator\inputs\pdfs' `
  --fiscal-declarations 'C:\operator\inputs\fiscal-declarations.json' `
  --output-dir 'C:\operator\outputs\field-passages-01'
```

Repeat `--document-id` to build an explicitly selected report subset. IDs must be unique and registered, and the selection must contain declared statement reports. Unselected source PDFs are not needed for that subset. With no selection, the snapshot builds all declared reports. Place output outside the original PDF directory and any bound optional corpus directory. Existing output, linked paths and input/output overlap are refused.

The command prints the completion status, passage count and passage-body hash, with zero model and embedding runs. A successful command establishes the source-only artifact boundary described below; it makes no claim about later retrieval scores.

## Read through the installed library

The same source snapshot API supports construction and strict reading. All paths below are operator-supplied absolute local paths; the output's parent must already exist.

```python
from pathlib import Path

from nvidia_rag.source_passages import (
    build_source_passage_artifact,
    load_source_passage_artifact,
)
from nvidia_rag.statement_startup import load_statement_snapshot

inputs = dict(
    source_registry=Path("C:/operator/inputs/sources.json"),
    source_dir=Path("C:/operator/inputs/pdfs"),
    fiscal_declarations=Path("C:/operator/inputs/fiscal-declarations.json"),
)
output = Path("C:/operator/outputs/field-passages-01")

snapshot = load_statement_snapshot(**inputs)
receipt = build_source_passage_artifact(snapshot=snapshot, output_dir=output)

# Reading requires original inputs and a fresh source-built snapshot.
fresh_snapshot = load_statement_snapshot(**inputs)
passages = load_source_passage_artifact(
    snapshot=fresh_snapshot, artifact_dir=output
)
first = passages[0] if passages else None
```

`derive_source_passages(snapshot)` provides the same immutable tuple in memory. Each `SourcePassage` has `serializer_profile`, `cell` and `chunk`. The `cell` remains the original frozen native record, including every ordered text span and rectangle. The chunk ID separately binds that cell ID, serializer profile and assembled text. It does not redefine the original cell identity.

## Interpret fields and units

The readable projection records the source report's fiscal identity separately from the financial column's end date and current/prior-comparison role. A FY2025 Q2 report can legitimately contain a July 2023 comparison column; it must not be relabeled as a different report. Annual columns retain the native `Year Ended` header and null month duration rather than an assumed twelve-month period.

| Native unit kind | Preserved meaning |
| --- | --- |
| `monetary` | Printed value and source scale `1000000`; printed `$` remains ISO-currency-unverified. |
| `per_share` | Native per-share section plus Basic/Diluted row, scale `1` and printed currency; the amount is not in millions. |
| `count` | Native weighted-average-shares section plus Basic/Diluted row, scale `1000000`; currency is null and not applicable. |

Negative parentheses and decimal values stay in `value_literal`; normalized `amount` remains an exact decimal string. The sidecar makes no USD inference, currency conversion or financial verdict. Including EPS and count records does not extend the existing monetary-only natural-language statement selector.

The `Chunk` carries three actual source rectangles: `amount_bbox` from the value span, `metric_bbox` from the row span and `period_bbox` from the date span. The full cell retains any section, unit, duration, currency and split `date_year` bindings as well. A split annual date remains multiple source spans; the projected date rectangle is not presented as a contiguous quotation of the combined date. Physical PDF page and nullable PDF page label retain their separate meanings.

## Verify the artifact boundary

A complete directory contains exactly `passages.jsonl`, `manifest.json` and `build-receipt.json`. The separately named schema and serializer profile distinguish it from the fixed plain-word-safe corpus recipe. Each JSONL record contains its schema/profile, complete native cell and chunk projection. The manifest binds passage bytes, selected reports, omissions, dynamically counted unit kinds and retained input byte identities. The build receipt binds the manifest, passages and implementation hashes. These are unsigned consistency records, not source-rights or authenticity certificates.

Construction and reading verify the snapshot's registry, declarations and selected original PDF identities, then rebuild the native index and compare all reports, cells and omissions through canonical bytes. Optional paired corpus inputs, when present in the supplied snapshot, must also replay their source registration and chunk binding. The reader regenerates all expected records and exact artifact bytes; recomputing a hash after changing an amount, unit, text or rectangle does not make it acceptable. It checks artifact and source closure again before returning.

The builder uses a new exclusive output directory and keeps `pending.json` through all fallible completion checks. A failure leaves refusal state; `failure.json`, a pending marker, missing/extra entries, malformed JSON or changed source inputs prevent reading. The owned pending marker is checked immediately before its final removal. Preserve failed output separately and use a fresh path for another build. Limits include 128 reports, 65,536 retained cells, a 512 MiB retained-input budget and a 64 MiB artifact-read budget, alongside the statement loader's existing PDF/JSON limits.

## Completed source-only acceptance

The [dated acceptance record](acceptance/source-field-passages-20261014.json) reports a fresh native replay of the ten original PDFs: **531 passages, six omissions, 411 monetary cells, 60 per-share cells and 60 count cells**. The native index equals the preserved source index; source PDFs, raw corpus, original vectors and that index remain byte-identical. A separate installed-wheel CLI and fresh reader reproduced the three output bodies exactly, with 22 installed package modules and 29 wheel RECORD rows verified against the wheel and source. These actual-PDF and installed checks were performed by the coordinating acceptance run; the author-side focused tests use synthetic native PDF fixtures.

The fresh API/dev environment's full suite passed **941 tests** with zero failures/skips and one existing Starlette/httpx deprecation warning. The acceptance record also preserves two earlier unsuccessful full-suite attempts and their distinct environment/import causes. None of these checks ran a field encoder or generator.

Next, build a separately owned BGE field-vector artifact and compare actual source-qualified candidates at the original `0.78` threshold and Top K `5`, retaining the original questions and raw-vector baseline. Report field/column support separately from page hit, cosine and accepted coverage. API/browser consumption, generation, independent financial adjudication, human gold and held-out accuracy require their own evidence. The existing [report-routing diagnostic](source-bound-period-routing.md) continues to record the six-month question's below-threshold raw-passage result.
