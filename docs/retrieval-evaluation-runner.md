# Run retrieval evaluation from an installed package

`nvidia-rag-evaluate` records what an explicitly selected retriever returned for a frozen question file. It retains every ranked candidate, its source text and score, then creates a separate packet for supplied relevance judgments. The command never generates an answer. A complete run establishes an auditable retrieval record; it does not establish financial correctness or independent human gold.

The workflow has three steps: prepare reviewed inputs, execute a new run, and review a copy of its packet. Retrieval results remain unchanged when judgments are imported.

## Prepare the corpus and questions

Install the selected wheel into a virtual environment. Lexical execution, help and review import need only the wheel's core dependencies. The optional `[semantic]` extra is needed for an actual cached BGE run. Original PDFs, corpus artifacts, vectors and model weights remain external to the wheel.

Use [the local corpus recipe](local-financial-corpus.md) to prepare a completed corpus containing `manifest.json`, `chunks.jsonl` and `build-receipt.json`. Supply the exact source registry used to build it. The runner checks the corpus against that registry before constructing a retriever.

Questions are UTF-8 JSONL: one JSON object on each physical line. The following is an illustrative row for the generated synthetic acceptance fixture, whose second PDF page contains its source passage. It is not a financial reference question:

```json
{"question_id":"synthetic-positive","query":"  Net income in the annual report?  ","answerable":true,"evidence_pages":[2],"evidence_document_ids":["synthetic-annual"],"evaluation_usage":{"purpose":"development","development_exposed":true,"frozen_on":"2026-10-14","registration_note":"Generated synthetic transport fixture; no financial gold"}}
```

Preserve each query exactly, including its leading and trailing whitespace. The backend receives that query and the selected top K. Page labels, expected answers, financial annotations and later relevance judgments are not backend inputs. Each answerable question needs aligned `evidence_document_ids` and one-based `evidence_pages`; those documents and pages must exist in the verified corpus. An unanswerable row uses empty page/document arrays and cannot carry an expected answer.

The four required fields are `question_id`, `query`, `answerable` and `evidence_pages`. Optional fields are `evidence_document_ids`, `expected_answer`, `attribution_ground_truth`, `financial_ground_truth` and `evaluation_usage`; their exact schemas are enforced by [`EvaluationQuestion`](../src/nvidia_rag/evaluation.py) and the [strict parser](../src/nvidia_rag/evaluation_inputs.py). This runner also requires registration metadata for its selected mode. Do not use a date or a field named `ground_truth` as a substitute for independent source adjudication.

Input is limited to 1 MiB and 1,000 questions, with at most 500 decoded characters per query. Blank lines are ignored. Invalid UTF-8, duplicate decoded JSON keys, nonfinite numbers, unknown fields, duplicate question IDs or document/page pairs, and coerced types are refused.

## Execute a new run

All paths must be absolute local paths. The output parent must already exist, while the output directory must be new and outside the inputs and installed package. Symlinks and junctions in these paths are refused. The angle-bracket paths below are operator-supplied placeholders:

```text
nvidia-rag-evaluate run --artifact-dir <absolute-corpus-dir> --source-registry <absolute-sources.json> --questions <absolute-questions.jsonl> --output-dir <absolute-new-run-dir> --mode development --backend lexical --top-k 5 --threshold 0.01
```

`python -I -B -m nvidia_rag.evaluation_cli` accepts the same subcommands. Help is available through `nvidia-rag-evaluate --help`, `run --help` and `import-review --help`. Successful execution prints one bounded JSON status with the retained receipt's SHA-256. A failed command returns nonzero and a bounded error type, without echoing question text or raw dependency diagnostics.

The parent command parses arguments and starts a separate worker with the same Python interpreter and package location. That worker's standard output and error handles point to the null device, including native library output. It writes a small status file in a temporary directory; the parent bounds and validates that record and independently hashes the reported artifact before emitting its terminal JSON. A missing, inconsistent or interrupted worker status produces `WorkerError`. Run progress and failure evidence remain in the requested run directory. This process boundary does not authenticate the host, sandbox model code or provide a model execution deadline.

The threshold is explicit and must be finite between zero and one; top K is an integer from 1 through 1,000. The example threshold is a lexical inspection setting, not calibrated financial acceptance. Complete ranking is retained before the evidence gate. Repeat `--document-id` to limit retrieval to a nonempty registered run-wide document scope; omitting it selects all registered documents. A page label does not silently choose the retrieval scope.

Two modes are available:

| Mode | Required registration and reference | Interpretation |
| --- | --- | --- |
| `development` | Every question uses `purpose: "development"` and `development_exposed: true`; no reference option | Reproducible work on explicitly exposed development inputs |
| `acceptance-candidate` | Every question uses `purpose: "acceptance_candidate"`, a real `frozen_on` date and `development_exposed: false`; supply a separate nonempty `--reference` JSONL whose registrations are also frozen | A supplied candidate registration that passes exact overlap checks against the reference |

Acceptance preflight checks duplicate IDs, exact query overlap and exact financial-target overlap before model construction. These checks are narrower than independence: whitespace/case variants, paraphrases, common source exposure or an incorrect supplied label can still exist. Mode selection does not certify a held-out financial benchmark.

For a prepared semantic run, explicitly select the cached backend:

```text
nvidia-rag-evaluate run --artifact-dir <absolute-corpus-dir> --source-registry <absolute-sources.json> --questions <absolute-questions.jsonl> --output-dir <absolute-new-run-dir> --mode development --backend cached-bge --vector-dir <absolute-vector-dir> --model-cache-dir <absolute-model-cache-dir> --top-k 5 --threshold 0.78
```

The vector directory must contain the verified `bge-small-en-v1.5.npy`, `embedding-manifest.json` and `vector-build-receipt.json`. The model cache must already contain `BAAI/bge-small-en-v1.5` at revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The command inventories the complete selected snapshot, constructs on CPU with `local_files_only=True`, and never downloads a model or falls back to lexical retrieval. A missing or changed cache fails the requested run.

The default `--routing-policy none` applies only the run-wide document scope. `--routing-policy source-bound-period` additionally requires both `--statement-source-dir` and `--statement-fiscal-declarations`, using the [same native PDF/declaration format as installed startup](verified-corpus-startup.md). It independently constructs the source-bound report catalog and uses the shared production BGE retriever factory. These statement options are refused for lexical execution. Synthetic injected-model tests exercise the construction contract; they are not an actual cached BGE experiment.

## Inspect the retained run

The runner reserves the output directory before constructing a backend. It copies the exact selected input bodies, writes its freeze, and records each attempted retrieval. It verifies the named original inputs after construction and after all questions. Each copied input or output body is limited to 64 MiB; the complete retained run is limited to 256 MiB. Exceeding either bound fails the run without truncating source bodies or candidates.

Known corpus, registry, question/reference, package, declaration and vector paths receive a size-only preflight before corpus/vector loading. After the verified corpus determines the selected PDF scope, the full input set is checked again before snapshot allocation, native PDF parsing or model construction. Bounded reads still check actual bytes, and the completed loader checks its entire file inventory before reading its artifact bodies. These checks do not replace the byte/hash closure checks.

The 256 MiB bound covers the logical retained run, including input snapshots, attempts, results, review packet and completion receipt. It is not a RAM or peak-disk limit: in-memory copies, parsed objects, model weights and the temporary attempt-journal replacement can consume additional resources. A failed reserved directory may also retain a small failure diagnostic alongside its pending marker. The independent review JSON limit remains 64 MiB; question/reference input remains 1 MiB, 1,000 rows and 500 decoded characters per query. The limits do not promise that every allowed question/top-K combination will fit.

| Artifact | Retained evidence |
| --- | --- |
| `freeze.json` | Raw input identities, canonical configuration, supplied question/reference rows, and selected model snapshot inventory |
| `inputs/` | Exact question/reference, configuration, corpus/registry, installed package Python bodies, optional vectors, and selected native PDF/declaration bodies |
| `attempts.json` | Per-question start/completion state and complete returned candidates, or a bounded failure type |
| `results.json` | Metrics against supplied page labels and complete ordered calls: exact query, rank, score, full chunk fields and source text hash |
| `review-packet.json` | The same complete candidates plus one initially null judgment slot per candidate |
| `receipt.json` | Completion status and byte/hash identities for the freeze, results, packet and attempts; generation remains zero and financial correctness unestablished |

Large model weights are inventoried rather than copied. The receipt is written last, followed by removal of `pending.json`. A failure after reservation retains pending/failure and available attempt evidence; an incomplete directory is never accepted as a completed report. Preserve failed directories and choose a new output for a deliberate retry.

From Python, `nvidia_rag.evaluation_run.load_completed_run(Path(<absolute-run-dir>))` checks the complete graph: canonical bodies, input snapshots, source/candidate membership, order, metrics reconstructed from stored candidates, attempt journal and receipt. It does not execute a new backend search, encode a query or construct a model. It refuses missing, altered, pending, failed, unknown or inconsistent artifacts.

Original Windows or POSIX input paths are provenance metadata when a completed run is loaded; the original inputs need not remain at those paths. Loading currently requires the installed package to have the same Python module-name layout as the captured layout. Preserve the exact wheel alongside an archived run rather than assuming future releases can load every prior schema. The named input/module inventory provides unsigned consistency evidence, not a signature, reviewer authentication or a capture of every imported runtime dependency.

## Import supplied relevance judgments

Copy `review-packet.json` outside the immutable run. Edit only `grade`, `reviewer` and `source_check` within each existing judgment slot. Keep every question, candidate and slot in its original order, including questions with no candidates.

| Value | Meaning and provenance |
| --- | --- |
| `grade: null` | Unjudged; keep both provenance fields null, or supply both as nonblank strings |
| `grade: 0` | Explicitly judged irrelevant; requires nonblank reviewer and source check |
| Positive integer grade | Supplied relevance grade; requires nonblank reviewer and source check |

Boolean, negative or fractional grades are refused. A positive grade records a supplied relevance judgment; the importer does not independently verify its truth or turn it into answer correctness. Record the source location actually inspected in `source_check` and the actual reviewer/process in `reviewer`.

```text
nvidia-rag-evaluate import-review --run-dir <absolute-completed-run-dir> --review-path <absolute-reviewed-copy.json> --output-path <absolute-new-review-import.json>
```

Import verifies the completed run before and immediately before writing. It binds the raw review body and source run hashes, rejects changed or transplanted query/text/rank/score/IDs, and writes a fresh exclusive result outside the run. It neither rewrites the run nor reruns retrieval. The imported result retains all supplied judgments and the complete candidate denominator.

The summary reports judged relevant, judged irrelevant, unjudged, empty retrieval and candidate-weighted coverage. Null does not count as zero. When no candidates exist, coverage is null rather than an invented success rate. Page recall/MRR and gate metrics in the original run are against supplied page labels; judgment coverage is a different measure. Neither alone establishes generated-answer faithfulness, financial value/unit/period correctness, human gold or held-out independence.

## Verify an installed consumer

From a working directory outside the checkout, use a fresh core-only environment containing the exact wheel:

```text
python -I -B <absolute-repository>/acceptance/evaluation_consumer.py --wheel <absolute-built-wheel> --output <absolute-new-consumer-receipt.json>
```

The harness compares installed production module bodies to the wheel, creates native synthetic PDF inputs, invokes the actual console command, checks all four candidates for two questions, imports both all-null and explicitly synthetic grades, and refuses an altered candidate rank. It rechecks source/run closures and records zero encoder, generation, human-gold and held-out runs. Its generated values and grades are installation fixtures, not reviewed financial evidence. The API workflow also runs this check in its fresh Linux/Windows API consumer environments and retains the result; workflow wiring alone is not a completed CI result.

An actual financial-model attempt needs its own frozen question/reference, corpus/vector/cache and configuration record, retained successes and failures, and separately supplied source review. This installed-command increment does not alter the earlier failed financial-readiness findings or establish generation quality, hosting, load or deployment acceptance.
