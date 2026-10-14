# Source-bound Evaluation Runner Implementation Plan

> **For agentic workers:** Use subagent-driven-development with disjoint implementation ownership and independent review. Preserve failed attempts and install/run the finished command outside the checkout.

**Goal:** Make retrieval evaluation executable from an installed package, preserving frozen inputs, complete ordered candidates and a separate result-bound relevance review.

**Architecture:** A strict input boundary feeds the existing evaluation metrics. The CLI uses a verified corpus and explicitly selected lexical or cached BGE backend; BGE construction shares production's public retriever factory. Completed immutable run artifacts precede editable relevance judgments. Review import validates the full artifact graph and never reruns retrieval or modifies its source run.

**Tech Stack:** Python standard library, existing corpus/vector verification, NumPy/scikit-learn, optional cached SentenceTransformers, pytest and the installed console entry point.

Official reference inspected in the Codex browser: [Anyscale RAG evaluation](https://docs.anyscale.com/rag/evaluation). Retrieval metrics require supplied relevance labels; generation correctness, faithfulness, relevance and completeness are separate. This increment runs retrieval only. Registrations and hashes do not establish independent financial gold, reviewer identity or whole-runtime code authenticity.

## 1. Input and execution contract

- [x] Create `evaluation_inputs.py`: `parse_questions(body: bytes) -> tuple[EvaluationQuestion, ...]`. Reject invalid UTF-8, duplicate decoded JSON members, nonfinite numbers, unknown fields, coercion, duplicate IDs, invalid nested labels and page/document associations. Preserve query text exactly. Required fields are question_id/query/answerable/evidence_pages; the existing dataclass's other fields are optional. Limit raw input to 1 MiB, questions to 1000, query length to 500, and require document-qualified page labels on answerable rows for this multi-document runner.
- [x] Development mode accepts only explicitly development-exposed registrations; acceptance-candidate mode requires a nonempty separately frozen reference and the existing `validate_acceptance_preflight`. Exact overlap checks remain narrower than independence.
- [x] Extract public `build_bge_retriever` from `runtime.py`; existing `build_bge_service` becomes a wrapper, preserving encoder conventions and routing conflict checks. Never access service private attributes or evaluate already gated HTTP evidence as raw ranking.
- [x] Create `evaluation_run.py` with `run_evaluation(*, artifact_dir, source_registry, questions, output_dir, mode, backend, top_k, threshold, reference=None, document_ids=None, vector_dir=None, model_cache_dir=None, routing_policy='none', statement_source_dir=None, statement_fiscal_declarations=None, model_factory=None)` and `load_completed_run(run_dir)`.

The runtime receives query and top K only:

```python
report = evaluate_retriever(retriever=captured_retriever,
    questions=registered_questions, top_k=top_k, threshold=threshold,
    include_review_snapshot=True)
```

`captured_retriever.retrieve(query, *, top_k)` forwards the exact query once; validates each complete ScoredChunk against verified corpus membership, unique IDs, allowed run-wide scope, count and finite cosine bounds (allowing 1e-12 roundoff); and retains complete dataclass chunk fields plus actual scores/order. No gold pages, expected amounts, annotations or relevance judgments reach the backend.

- [x] CLI backend `lexical` uses TfidfRetriever plus explicit run-wide scope. `cached-bge` requires verified existing vectors and the exact pinned local BGE cache revision, CPU/local-files-only construction and no download; no lexical fallback. `routing_policy=none` means explicit run-wide scope only. `source-bound-period` requires the native source/declaration pair and independently built report catalog, and uses the shared production factory. Reject unsupported option combinations before model work.
- [x] Capture exact raw question/reference and canonical configuration bodies, verified corpus files/registry, optional vectors/receipt/manifest, complete selected model snapshot file inventory and installed package Python bodies. Recheck closure after construction and after all queries. Inventory claims are limited to these named bodies, not all Torch imports.
- [x] Refuse existing or overlapping/symlink output paths. Reserve an exclusive output directory with pending marker before backend construction. Save per-call attempt evidence; failed construction/query/validation/closure leaves pending/failure evidence and never a completed report or a fallback run. The original release bounded the combined graph at 64 MiB; the separately tested budget follow-up below preserves 64 MiB per body and allows 256 MiB for the retained graph. Refuse rather than truncate.

Completed artifact graph, with UTF-8 canonical JSON (`sort_keys=True`, compact separators, `allow_nan=False`, trailing newline):

```python
freeze = {"schema_version": 1, "config": config, "inputs": input_identities,
          "questions": registered_question_rows, "reference": reference_rows}
results = {"schema_version": 1, "freeze_sha256": freeze_digest,
           "report": report, "calls": complete_ordered_call_records}
packet = build_review_packet(results, freeze_sha256=freeze_digest,
                             results_sha256=results_digest)
receipt = {"schema_version": 1, "status": "complete",
           "files": identities_for_freeze_results_packet_attempts,
           "generation_runs": 0, "financial_correctness": "unestablished"}
```

Write freeze before execution; write results and packet after closure checks; write receipt last, then remove pending. `load_completed_run` rejects pending/failure, missing or changed artifacts and inconsistent graph/row/candidate associations. Source chunks retain all fields, including optional rectangles. Source snapshots and candidate ranks remain distinguishable from relevance labels.

## 2. Separate review contract

- [x] Create `evaluation_review.py`: `build_review_packet(results, *, freeze_sha256, results_sha256) -> dict` and `import_review(*, run_dir, review_path, output_path) -> dict`. Do not import semantic runtimes or execute a model during import.

```python
packet = {"schema_version": 1, "kind": "retrieval-relevance-review",
          "freeze_sha256": freeze_sha256, "results_sha256": results_sha256,
          "rows": [{"question_id": question_id, "query": query,
                    "query_sha256": query_digest,
                    "candidates": complete_ranked_candidates,
                    "judgments": [{"chunk_id": chunk_id, "grade": None,
                                   "reviewer": None, "source_check": None}]}]}
```

Each immutable candidate contains rank, complete chunk fields, score and exact text hash. Keep the complete ordered run and candidate denominator, including empty retrieval. A supplied packet must retain every immutable field and every judgment slot; partial review is expressed by null grades. Integer zero is explicitly irrelevant; positive integers are supplied relevance grades. Reject boolean/negative/noninteger grades, foreign/duplicate/reordered IDs, changed text/rank/score/query, transplanted run hashes and unknown keys. Require nonblank reviewer/source_check for a supplied grade; an unknown grade may retain either both null or both nonblank provenance fields.

- [x] Normalize only editable judgment values back to the expected null template and compare canonical bytes to the reconstructed baseline; Python's bool/int or float/int equality must not hide type changes. Aggregate nonnull grades with `summarize_judgment_coverage_rows`, retain nulls and full denominators in the imported record, and bind raw review body plus source run hashes. Write only a fresh exclusive output outside the immutable run. Unknowns remain unjudged. No generation-review summaries or financial accuracy rates are fabricated.

## 3. Installed command, tests and delivery

- [x] Create `evaluation_cli.py` and register `nvidia-rag-evaluate = "nvidia_rag.evaluation_cli:main"`. Provide `run` and `import-review` commands with explicit paths/configuration. Help and review import work without SentenceTransformers. Print a bounded terminal status and artifact identity; failures return nonzero without logging question text or credentials.
- [x] Author meaningful tests before implementation and retain RED output. Cover exact query isolation, complete source records, acceptance overlap before construction, strict inputs, all routing/backend options, shared BGE conventions, failure/no-fallback, mutation closure, output refusal, review transplant/type/unknown handling and empty retrieval denominator. Port self-contained legacy metric/preflight tests into the staged package suite.
- [x] Run focused tests, relevant neighboring regression and full staged suite. Build a fresh wheel, install in a fresh external consumer environment and execute help, a generated-PDF lexical run and result-bound review import from outside the checkout. Exercise cached BGE only through an explicitly frozen separate native attempt; synthetic model tests are not a native run.
- [x] Independently inspect implementation and installed artifacts. Publish the finished reviewed increment and inspect exact CI outputs; add installed command smoke/acceptance to CI. Then write a concise reader-facing record with source and evaluation boundaries. The latest October 14 user instruction requires profile, lab, standalone journal, every blog and all figures to share the LIGHT theme, superseding the former dark-journal exception.
- [x] Update active-goal todo and small memory correction note. Full portfolio goal stays ACTIVE until independent financial/generation, hosted integration, load, Dot handoff, physical-device and aesthetic requirements are established.

## Delivery checkpoint (2026-10-14)

Implementation48daf59 / official38583572671 completed all three jobs; downloaded package and container artifacts were independently audited. Current local1330PASS and fresh external consumer07 bind the installed command. Reader section published at site d5c4b5e / official38584263104, with actual public IAB review in the shared LIGHT theme. Full financial evaluation remains separate: ten original PDFs total168446328B, exceeding this release combined64MiB snapshot cap. The next budget/integration change needs its own tests and reviewed frozen native attempt; see .runtime/evaluation-runner-20261014/next-financial-run-preparation.md. No new financial model run or human gold is established here.

## Complete-corpus budget follow-up

- [x] Independently measure the actual intended fixed inputs: ten PDFs 168,446,328 bytes; corpus 5,219,294; vectors 4,981,388; registry/declarations 9,252; original 30-module package 485,190. This 179,141,452-byte subtotal excludes dynamic run artifacts. The largest copied PDF is 49,970,589 bytes. The 133,466,304-byte model weight is inventoried, not copied.
- [x] Separate the 64 MiB body bound from the 256 MiB complete retained-run bound. Check known input sizes before loading, then selected PDFs before capture/native parsing, while retaining actual-byte reads and unchanged-input closure. Enforce both bounds for writer bodies and atomic attempt updates; preflight the loader inventory before JSON/PDF reads.
- [x] Preserve RED evidence for the legacy aggregate cap and three early-read regressions. Verify a real complete synthetic graph above 64 MiB through run, portable load and all-unknown review import. Cover exact/refused boundaries, dynamic packet failure and pre-read loader refusal without executing a native model.
- [ ] Complete independent review, full regression, a fresh wheel and an external installed consumer; publish and inspect the exact official checks.
- [ ] Register and independently review a separate frozen financial attempt and cleanup harness before any native query encoding. The earlier failed BGE result remains unchanged; human gold, held-out validation, financial/generation quality, hosted integration and Dot handoff remain open.

Local follow-up checkpoint: 119 focused checks passed; independent probes confirmed no full corpus read and no native PDF parser/model call on oversize inputs. The complete staged suite passed 1,342 tests in 276.12 seconds with one existing Starlette/httpx deprecation warning. Fresh core-only external consumer and wheel/RECORD audit passed for wheel SHA-256 `5487ddbe064ab826194aabf181c2d54e32a47842d22d151a57844ca9c24c444c` (163,806 bytes, 30 matching Python modules, 37 RECORD rows). Its two questions/four candidates and relevance grades are synthetic installation fixtures; encoder/generation/human-gold/held-out counts remain zero. Official follow-up publication is not established by this local checkpoint.
