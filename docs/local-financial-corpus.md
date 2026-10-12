# Build and consume an operator-local financial corpus

This recipe connects the installed library to real operator-supplied financial
reports. It publishes metadata and code; the wheel and repository contain no PDF,
extracted corpus, vector matrix or model weights. Source permissions remain an
operator responsibility; the legacy schema-v1 registry is not a permission or
redistribution approval. See [data and dependency rights](../THIRD_PARTY.md).

The [eight-report registry](../corpus/sources-annual-plus-fy2025-q2.json) preserves
the original source metadata and SHA-256 pins. Its 3,274 bytes have SHA-256
`74379f0ecb5b131ca8562d2713a13147dc538047e7f2e2b03212ca6f2f7ed495`.
The reports span annual material from FY2021 through FY2026 and FY2025/FY2027 Q2.
Obtain the exact approved files externally and place them in a dedicated local
directory with those filenames. The builder reads them in place; it downloads
or repairs no file, and it rejects a mismatched pin. A filename or official URL
does not independently authenticate source bytes or license their use.

The new fixed profile is `plain-word-safe-400-0-v1`: PyMuPDF plain page text,
`cl100k_base`, a 400-token limit, and whitespace-boundary packing with no overlap.
One-based physical PDF pages remain attached to chunks. A unit longer than the
token limit is refused rather than truncated. This profile is not a table-cell
extractor, semantic index, financial field parser or OCR pipeline. Older 2,865-
and 4,126-chunk experiments used different extraction/runtime configurations;
fresh counts and output digests must be observed rather than copied from them.

## Prepare one identified installed wheel

Use Python 3.13 for this dated dependency selection. The package metadata permits
Python >=3.11, but the constraints/CI selection does not establish compatibility
of that exact selection with Python 3.11. `constraints/api-demo.txt` fixes direct
core/API dependencies; it is not a complete transitive hash lockfile.

Finish build, consumer and vector source changes and their independent reviews
before building the shared wheel. Use an
absent output directory, record the resulting wheel name, size and SHA-256, then
install **that exact wheel** in a separate environment. Do not install the source
project a second time into that consumer environment: its module identity would
no longer be established by the recorded wheel. Installation/build preparation
may download dependencies; the later corpus and lexical runtime are local.

From a fresh clone on PowerShell, after verifying these three owned paths are
absent (retain any previous environment/artifact instead of removing it):

```powershell
$corpusRecipeRepo = (Get-Location).Path
$corpusRecipeBuildEnv = Join-Path $corpusRecipeRepo '.runtime/corpus-build-venv-20261011'
$corpusRecipeConsumerEnv = Join-Path $corpusRecipeRepo '.runtime/corpus-consumer-venv-20261011'
$corpusRecipeWheelDir = Join-Path $corpusRecipeRepo 'dist/corpus-recipe-20261011-01'
foreach ($corpusRecipeNewPath in @($corpusRecipeBuildEnv, $corpusRecipeConsumerEnv, $corpusRecipeWheelDir)) {
  if (Test-Path -LiteralPath $corpusRecipeNewPath) { throw "Owned path already exists: $corpusRecipeNewPath" }
}
# Use an available Python 3.13 executable; verify python --version first.
python -m venv $corpusRecipeBuildEnv
$corpusRecipeBuildPython = Join-Path $corpusRecipeBuildEnv 'Scripts/python.exe'
& $corpusRecipeBuildPython -m pip install build
& $corpusRecipeBuildPython -m build --wheel --outdir $corpusRecipeWheelDir $corpusRecipeRepo
$corpusRecipeWheel = Join-Path $corpusRecipeWheelDir 'nvidia_financial_rag-0.1.0-py3-none-any.whl'
Get-FileHash -LiteralPath $corpusRecipeWheel -Algorithm SHA256
python -m venv $corpusRecipeConsumerEnv
$corpusRecipePython = Join-Path $corpusRecipeConsumerEnv 'Scripts/python.exe'
& $corpusRecipePython -m pip install -c (Join-Path $corpusRecipeRepo 'constraints/api-demo.txt') "$corpusRecipeWheel[api,dev]"
& $corpusRecipePython -m pip check
& $corpusRecipePython -m pip freeze
```

Stop at any nonzero native command exit (`$LASTEXITCODE`). Keep the new wheel
identity and resolved dependency list with the subsequent acceptance receipt.
The examples/acceptance scripts belong to the clone; they call the installed
`nvidia_rag` package with `-I`. The wheel installs no console command and bundles
no registry data.

On POSIX, use two separately owned Python 3.13 virtual environments under ignored
`.runtime/`, build once with the builder Python, and use the consumer Python:

```bash
corpus_recipe_repo="$PWD"
corpus_recipe_builder="$PWD/.runtime/corpus-build-venv-20261011"
corpus_recipe_consumer="$PWD/.runtime/corpus-consumer-venv-20261011"
corpus_recipe_wheel_dir="$PWD/dist/corpus-recipe-20261011-01"
for corpus_recipe_path in "$corpus_recipe_builder" "$corpus_recipe_consumer" "$corpus_recipe_wheel_dir"; do
  if [ -e "$corpus_recipe_path" ] || [ -L "$corpus_recipe_path" ]; then
    printf 'Owned path already exists: %s\n' "$corpus_recipe_path" >&2
    exit 1
  fi
done
python3.13 -m venv "$corpus_recipe_builder"
"$corpus_recipe_builder/bin/python" -m pip install build
"$corpus_recipe_builder/bin/python" -m build --wheel --outdir "$corpus_recipe_wheel_dir" "$corpus_recipe_repo"
corpus_recipe_wheel="$corpus_recipe_wheel_dir/nvidia_financial_rag-0.1.0-py3-none-any.whl"
sha256sum "$corpus_recipe_wheel"
python3.13 -m venv "$corpus_recipe_consumer"
corpus_recipe_python="$corpus_recipe_consumer/bin/python"
"$corpus_recipe_python" -m pip install -c "$PWD/constraints/api-demo.txt" "$corpus_recipe_wheel[api,dev]"
"$corpus_recipe_python" -m pip check
"$corpus_recipe_python" -m pip freeze
```

Stop on any failing command; these manual preparation snippets do not silently
recover or reuse an old wheel. Consumer checks should run from an external cwd
with absolute clone/script/input paths so the source checkout cannot satisfy
an installed-module check accidentally.

## Build the corpus from prepared local inputs

Prepare `cl100k_base` encoding data before the build. The supplied cache directory
must contain filename `9b5ad71b2ce5302211f9c61530b329a4922fc6a4`, SHA-256
`223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`.
This is the encoding resource pinned by tiktoken, not a report or trained model.
The builder checks its bytes even when an encoding is already warm in memory.
Missing/corrupt prepared cache is refused before extraction. It constructs the
encoding from those verified in-memory BPE bytes using the installed cl100k
configuration and an owned exact-input guard, then explicitly supplies that
encoding to the existing word-safe chunker. Downstream has no cache lookup or
download/delete fallback, and the caller's environment remains unchanged. Other
legacy callers omit this optional encoding and retain their original behavior.

Provide `NVIDIA_RAG_SOURCE_DIR` and `NVIDIA_RAG_TOKENIZER_CACHE_DIR` privately as
absolute local directories. The following PowerShell command uses a new output
path; any existing file/directory there is refused without modification:

```powershell
$corpusRecipeRegistry = Join-Path $corpusRecipeRepo 'corpus/sources-annual-plus-fy2025-q2.json'
$corpusRecipeOutput = Join-Path $corpusRecipeRepo 'artifacts/portable-word-safe-20261011-01'
& $corpusRecipePython -I -B (Join-Path $corpusRecipeRepo 'examples/build_corpus.py') `
  --source-registry $corpusRecipeRegistry --source-dir $env:NVIDIA_RAG_SOURCE_DIR `
  --output-dir $corpusRecipeOutput --tokenizer-cache-dir $env:NVIDIA_RAG_TOKENIZER_CACHE_DIR
```

On POSIX:

```bash
"$corpus_recipe_python" -I -B "$corpus_recipe_repo/examples/build_corpus.py" \
  --source-registry "$corpus_recipe_repo/corpus/sources-annual-plus-fy2025-q2.json" \
  --source-dir "${NVIDIA_RAG_SOURCE_DIR:?supply an absolute prepared PDF directory}" \
  --output-dir "$corpus_recipe_repo/artifacts/portable-word-safe-20261011-01" \
  --tokenizer-cache-dir "${NVIDIA_RAG_TOKENIZER_CACHE_DIR:?supply an absolute prepared encoding cache}"
```

The builder first writes and closes `pending.json` inside its exclusively created
directory, before extraction or chunk output. The directory then receives
`chunks.jsonl`, `manifest.json`, and the last-written `build-receipt.json`, which
binds registry/source bytes, observed counts, profile, builder runtime and executing
module hashes. Only after the completed receipt writer closes successfully does
the builder remove its own pending marker. A completion-write/close error or a
pending-removal error preserves pending state; a bounded `failure.json` is a
best-effort diagnostic and is not the only rejection signal. The loader refuses
any pending or failure entry before and after reading, including a directory or
dangling symlink with that name. It therefore rejects an otherwise valid receipt
even when writing the failure diagnostic also fails. Partial evidence is retained;
do not remove markers to turn a failed build into a completion. Resolve the cause
and use another owned output path.

`load_verified_corpus` checks strict schema/types, exact profile and registry
metadata/order, manifest totals, JSONL digest, unique IDs, document/page/ID
association, record order, actual per-document counts and represented pages.
It rereads closure identities before returning. Legacy loaders retain their
old behavior. These unsigned hashes demonstrate internal consistency and ordinary
input-change detection; an adversary who rewrites the entire registry/artifact
set is not authenticated by them. The loader does not re-extract the PDFs or
judge whether a number is under the correct row, duration, date or unit.

## Serve an explicit lexical scope

Configure `NVIDIA_RAG_API_TOKEN` privately (at least 32 characters). It belongs in
the process environment, never in command arguments, committed files, browser
JavaScript or a receipt. The server binds `127.0.0.1`; no remote host option or
generation/model-download switch exists:

```powershell
& $corpusRecipePython -I -B (Join-Path $corpusRecipeRepo 'examples/serve_corpus.py') `
  --artifact-dir $corpusRecipeOutput --source-registry $corpusRecipeRegistry `
  --document-id nvidia-fy2025-q2 --threshold 0.0 --top-k 5 --port 8000
```

Repeat `--document-id` for a declared group, or omit it for the explicitly global
corpus. Unknown/empty scope is refused rather than falling back globally.
`DocumentScopeRetriever` and `build_corpus_service` live in the installed
`nvidia_rag.corpus` module. They make no natural-language period decision. This
avoids using the old annual-only launcher as proof of mixed-period retrieval.

Threshold **0.0** is deliberately **ungated lexical inspection**: TF-IDF may return
zero-score candidates and the existing evidence service can then report
`can_answer=true`. That means the selected evidence passed this configured gate,
not that a field, duration or financial answer is supported. For a meaningful
deployment gate, separately evaluate and choose settings; do not transfer BGE's
0.78 threshold to TF-IDF. Generation remains disabled and `answer` remains empty.

Use the existing client in another terminal with the same privately configured
token. Stop only this foreground server with `Ctrl+C`:

```powershell
& $corpusRecipePython -I -B (Join-Path $corpusRecipeRepo 'examples/query_client.py') `
  --question "According to NVIDIA's fiscal 2025 second-quarter report, what was net income for the three-month period ended July 28, 2024, in millions of dollars?"
```

## Actual installed consumer comparison

The acceptance script requires the exact recorded `--wheel`, verifies its entire
Python-module set against the consumer installation, and runs both global and
explicit FY2025-Q2 configurations. It owns one temporary foreground child at a
time, an ephemeral loopback port, a private test token, and a startup observation
deadline. It stops only those owned children. It compares actual HTTP payload,
ranked source/page IDs, score, exact evidence text digest, citations, empty answer,
error code, and request-ID/header against the same installed in-process service.
It snapshots input identities before loading corpus/questions, checks them again
immediately afterward and finally, and verifies its child terminated and its
temporary loopback listener is unreachable after teardown. A remaining listener
is reported without stopping an unrelated process.
The resulting small receipt contains no raw corpus text, PDF, token or local
absolute path. An existing output or one inside the read-only corpus is refused.

```powershell
& $corpusRecipePython -I -B (Join-Path $corpusRecipeRepo 'acceptance/corpus_consumer.py') `
  --artifact-dir $corpusRecipeOutput --source-registry $corpusRecipeRegistry `
  --wheel $corpusRecipeWheel `
  --candidates (Join-Path $corpusRecipeRepo 'evaluation/source-candidates/fy2025-q2-income-duration.jsonl') `
  --output (Join-Path $corpusRecipeRepo 'artifacts/portable-corpus-http-comparison-20261011-01.json')
```

The four original questions are development-exposed assistant source candidates,
not independent held-out labels or human gold. The script verifies their original
question identity and compares transport/lexical execution only. Source-cell replay,
retrieval field support, calibrated refusals, financial generation, independently
reviewed answer accuracy and load/latency distributions are separate gates.
Passing an ASGI test, building a wheel or having a valid citation replaces none
of those gates. The unchanged source-binding replay can separately inspect the
operator's exact FY2025-Q2 PDF with PyMuPDF 1.28.2.

The expanded **BGE semantic** gate remains required before claiming current
model-backed retrieval. It must use the pinned cached BGE revision and new vectors
bound to the rebuilt JSONL bytes/order, in another exclusively created vector
directory. The completed corpus remains read-only; vectors must satisfy float32
`[chunk_count,384]`, normalization and existing manifest identity checks. A later
semantic source increment is included in the one shared reviewed wheel. No BGE, generation,
financial-quality, real eight-PDF or actual HTTP result is claimed by this recipe
document alone.

## Prepare the separate cached BGE vector gate

After all three source increments pass their review gates, one shared package
wheel can contain the local corpus, lexical consumer and vector interfaces.
Use that **same recorded wheel file and SHA-256** for the following separate
semantic environment. If semantic source changes after the recorded wheel was
built, create a new owned build directory and identity; do not silently replace
the wheel or attach its old digest to the changed code. These are operator
instructions for the next acceptance gate, not evidence that it has run.

Create a previously absent `.runtime/corpus-semantic-venv-20261011` environment
with a verified Python 3.13 executable. Install the exact shared wheel with its
`[api,dev,semantic]` extras and the same dated core/API constraints. The additional
semantic dependencies still resolve during preparation; save their actual list
and notices. This is not a complete semantic lockfile. The following preparation
can acquire dependencies; it does not acquire model weights:

```powershell
$corpusRecipeSemanticEnv = Join-Path $corpusRecipeRepo '.runtime/corpus-semantic-venv-20261011'
if (Test-Path -LiteralPath $corpusRecipeSemanticEnv) { throw 'Semantic environment already exists; retain it and choose a new owned path' }
python -m venv $corpusRecipeSemanticEnv
$corpusRecipeSemanticPython = Join-Path $corpusRecipeSemanticEnv 'Scripts/python.exe'
& $corpusRecipeSemanticPython -m pip install -c (Join-Path $corpusRecipeRepo 'constraints/api-demo.txt') "$corpusRecipeWheel[api,dev,semantic]"
& $corpusRecipeSemanticPython -m pip check
& $corpusRecipeSemanticPython -m pip freeze
```

Verify `python --version` before environment creation, stop on any nonzero native
exit, and retain the named wheel's size/digest and resolved dependency identities.
Do not install the source checkout into this semantic environment a second time.
An existing environment, failed vector directory or completed vector directory
is retained rather than reused or overwritten.

The CLI fixes **`BAAI/bge-small-en-v1.5`**, revision
**`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`**, device **CPU**, and
**`local_files_only=True`**. Prepare that model cache separately under the operator's
model/dependency permissions before this step. There is no model-download switch
or fallback in this CLI; incomplete cache is a failure to record. Model-cache
existence, an import, a synthetic encoder or a passing unit test does not establish
actual BGE initialization or inference.

Choose a new vector directory disjoint from the completed corpus and registry.
The CLI refuses an existing output before model initialization, then exclusively
creates its owned directory. The completed corpus remains read-only, and the
builder verifies corpus identity before/after vector work. From the same clone
with the previously recorded variables:

```powershell
$corpusRecipeVectorDir = Join-Path $corpusRecipeRepo 'artifacts/portable-word-safe-bge-vectors-20261011-01'
& $corpusRecipeSemanticPython -I -B (Join-Path $corpusRecipeRepo 'examples/build_corpus_embeddings.py') `
  --wheel $corpusRecipeWheel --artifact-dir $corpusRecipeOutput `
  --source-registry $corpusRecipeRegistry --vector-dir $corpusRecipeVectorDir
```

On POSIX, create the separately absent semantic environment and install the
same recorded wheel before calling the same four required CLI arguments:

```bash
corpus_recipe_semantic="$corpus_recipe_repo/.runtime/corpus-semantic-venv-20261011"
if [ -e "$corpus_recipe_semantic" ] || [ -L "$corpus_recipe_semantic" ]; then
  printf 'Semantic environment already exists: %s\n' "$corpus_recipe_semantic" >&2
  exit 1
fi
python3.13 -m venv "$corpus_recipe_semantic"
corpus_recipe_semantic_python="$corpus_recipe_semantic/bin/python"
"$corpus_recipe_semantic_python" -m pip install -c "$corpus_recipe_repo/constraints/api-demo.txt" "$corpus_recipe_wheel[api,dev,semantic]"
"$corpus_recipe_semantic_python" -m pip check
"$corpus_recipe_semantic_python" -m pip freeze
"$corpus_recipe_semantic_python" -I -B "$corpus_recipe_repo/examples/build_corpus_embeddings.py" \
  --wheel "$corpus_recipe_wheel" \
  --artifact-dir "$corpus_recipe_repo/artifacts/portable-word-safe-20261011-01" \
  --source-registry "$corpus_recipe_repo/corpus/sources-annual-plus-fy2025-q2.json" \
  --vector-dir "$corpus_recipe_repo/artifacts/portable-word-safe-bge-vectors-20261011-01"
```

The separate output contains `bge-small-en-v1.5.npy`, `embedding-manifest.json`
and the last-written `vector-build-receipt.json`. Schema-2 embedding identities
bind the exact JSONL bytes, ordered chunk IDs, vector bytes and model revision;
the wrapper additionally requires **float32 `[chunk_count,384]`**, finite/nonzero
unit-normalized rows, an identified installed wheel and successful post-build
validation. An exclusively created `pending.json` precedes model work and remains
until `vector-build-receipt.json` has been written and closed successfully. The
loader rejects any pending entry before and after reads. Writer/close or pending
removal failure preserves partial or completed bytes and keeps the directory
unacceptable, even when best-effort `failure.json` cannot be written.
Old vectors cannot be attached to a newly rebuilt JSONL or reordered chunks.
Nothing in this receipt grants rights to redistribute PDF text or vectors.

### Freeze the subsequent semantic comparison before running it

The separate script, `acceptance/corpus_semantic.py`, provides required
absolute-path arguments `--artifact-dir`,
`--source-registry`, `--vector-dir`, `--candidates`, `--wheel`, `--model-cache-dir`
and `--output`; `--model-cache-dir` identifies the prepared Hugging Face hub root,
whose BGE snapshot must match the fixed revision. It permits no threshold/top-k
override. Run it only after independent source review, exact-wheel installation,
and actual corpus/vector prerequisites pass. Source and synthetic tests are
provided; actual execution and its receipt remain separate acceptance gates.
A command shown below is not evidence of a completed model run.
Before it runs, freeze the named wheel/module identities, complete
corpus and vector receipts/hashes, pinned BGE/cache/runtime/device, and the exact
four original candidate-question records in
[fy2025-q2-income-duration.jsonl](../evaluation/source-candidates/fy2025-q2-income-duration.jsonl).
Use **threshold 0.78 and top_k 5** for this declared development comparison;
preserve the settings, questions, wrong-period results and refusals after seeing
the output. That frozen diagnostic threshold is not a calibrated enterprise
financial-accuracy claim.

Compare two explicitly declared scopes: all eight reports with
`route_fiscal_year=False`, and `frozenset({"nvidia-fy2025-q2"})` selected **before**
ranking through `DocumentScopeRetriever`. Annual-only routing is disabled for
both, so a quarter question does not silently select an annual report. Passage
vectors use BGE's normalized passage encoding; query encoding uses the existing
`Represent this sentence for searching relevant passages: ` instruction and
normalization. Use `load_local_bge_vectors` to validate the separate vector output
against the completed corpus rather than loading a bare `.npy` without its receipt.

The small receipt identifies cached-model initialization and
query execution separately from artifact validation. Keep ranked source/page/ID
records, observed scores, selected scope, per-query elapsed time and preserved
input identities; avoid raw vectors, report text, model weights, tokens or local
absolute paths in public summaries. A single elapsed time is not a percentile or
throughput benchmark. These four assistant-reviewed, exposed development questions
do not establish held-out recall, independent human answer accuracy or refusal
reliability. Generation remains disabled; exact financial field support,
generated-answer correctness and human adjudication remain separate unfinished
gates. The script hashes every prepared pinned snapshot body, permits ordinary
Hugging Face file links only within the supplied cache root, and refuses linked
snapshot directories. It verifies corpus/vector completion, input body hashes
and installed wheel identity immediately before model initialization, immediately
after initialization, and after query work. A recording adapter captures the
single retrieval call used by each threshold/context decision; the eight rows
are eight query encodings, with no second hidden ranking call.

From the prepared semantic environment, with a new result pathname:

```powershell
$corpusRecipeModelCache = 'C:\prepared-model-cache\hub'
$corpusRecipeSemanticResult = Join-Path $corpusRecipeRepo 'artifacts/portable-word-safe-bge-semantic-20261011-01.json'
& $corpusRecipeSemanticPython -I -B (Join-Path $corpusRecipeRepo 'acceptance/corpus_semantic.py') `
  --wheel $corpusRecipeWheel --artifact-dir $corpusRecipeOutput `
  --source-registry $corpusRecipeRegistry --vector-dir $corpusRecipeVectorDir `
  --candidates (Join-Path $corpusRecipeRepo 'evaluation/source-candidates/fy2025-q2-income-duration.jsonl') `
  --model-cache-dir $corpusRecipeModelCache --output $corpusRecipeSemanticResult
```

`--model-cache-dir` is an operator-supplied absolute local Hugging Face hub root;
adapt the example to the prepared machine. There is no model acquisition,
listener, generation, annual routing or threshold override in this command.
Run from an external working directory to verify the installed-wheel boundary,
and preserve the terminal exit and exact result digest in acceptance evidence.
Only after the command exits successfully should the result be read using
`load_completed_comparison` from this acceptance script.

The comparison exclusively creates `<output filename>.pending.json` before
model work. Success removes only that owned sidecar after the result JSON writer
closes; the completed reader rejects any pending entry before and after its
read/hash closure. A failed model/query/close/unlink operation retains the
sidecar and any diagnostic or full result bytes, and never overwrites a previous
output. Full JSON text alone is insufficient evidence of completed comparison.
An existing result or pending sidecar is refused before wheel/model work. As
with the corpus/vector guards, this is a local operation-state check, not signed
authenticity, hostile filesystem race resistance or a power-loss durability
guarantee.
