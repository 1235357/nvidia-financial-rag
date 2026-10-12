# Installed local corpus and BGE runtime — 12 October 2026

One reviewed wheel was installed into separate consumer and semantic environments.
The consumer built a portable corpus from eight operator-supplied NVIDIA reports
and completed eight lexical HTTP queries on loopback. The semantic environment
encoded all 2,855 chunks with a cached BGE model and recorded eight evidence-only
query results. This establishes the dated local execution described below.
Financial field support, independent human gold, production capacity and a public
hosted API remain unverified. Publication CI for this increment has not yet run.

The [public acceptance summary](acceptance/local-corpus-runtime-20261012.json)
contains the exact source, package, input and output identities. The underlying
operator archives remain local: this repository publishes code, metadata and
this bounded summary, without PDFs, extracted chunks, vector bodies, model
weights, credentials or raw logs containing private machine paths. The two new
summary files sit outside the frozen implementation; they do not alter the wheel.

## Reproduce the same sequence

Follow [the operator-local corpus recipe](local-financial-corpus.md), in order:

1. [Build one identified wheel and install that wheel](local-financial-corpus.md#prepare-one-identified-installed-wheel)
   into a separate consumer environment.
2. [Build the corpus from prepared inputs](local-financial-corpus.md#build-the-corpus-from-prepared-local-inputs).
   Supply the pinned reports and verified `cl100k_base` encoding resource locally.
3. [Run the installed consumer comparison](local-financial-corpus.md#actual-installed-consumer-comparison).
   It starts owned loopback children for the global and explicit FY2025-Q2 scopes.
4. [Prepare and run the separate BGE gate](local-financial-corpus.md#prepare-the-separate-cached-bge-vector-gate),
   then use the [frozen semantic comparison](local-financial-corpus.md#freeze-the-subsequent-semantic-comparison-before-running-it).

The recipe includes PowerShell and POSIX commands, input checks and failure
behavior. Use new owned output paths and retain a failed attempt's evidence.
Do not delete completion markers, overwrite an old output or silently replace a
recorded wheel. Dependency preparation can use the network; corpus extraction,
lexical queries and this cached semantic run use prepared local inputs. A new
reader still has to prepare those inputs and dependencies.

The actual wheel was `nvidia_financial_rag-0.1.0-py3-none-any.whl`, 105,190 bytes,
with SHA-256
`76246d9638a6ddfb30e0abeace884fa67115fa8e38a4187590efeb52f86701e9`.
All 16 Python modules in that wheel matched the reviewed source and the installed
consumer and semantic packages. The source review recorded 239 focused and 362
repository tests passing, with one warning in each run. Those controlled tests
used fixtures and model doubles; the actual corpus, HTTP and cached-model records
provide separate runtime evidence. The public summary identifies each scope.

## What the corpus contains

The profile is `plain-word-safe-400-0-v1`: plain page text from PyMuPDF, packing at
whitespace boundaries with a maximum of 400 `cl100k_base` tokens, and no overlap.
It retains one-based physical PDF pages. It is not a table-cell parser or OCR
pipeline. The eight files total 167,056,483 bytes and were already registered
before this portability increment; this increment adds no new source.

| Source | Physical pages | Nonempty pages | Chunks |
| --- | ---: | ---: | ---: |
| FY2021 annual | 93 | 93 | 227 |
| FY2022 annual | 207 | 203 | 491 |
| FY2023 annual | 169 | 164 | 414 |
| FY2024 annual | 187 | 180 | 456 |
| FY2025 annual | 181 | 181 | 465 |
| FY2026 annual | 175 | 175 | 442 |
| FY2025 Q2 | 80 | 80 | 211 |
| FY2027 Q2 | 61 | 61 | 149 |
| **Total** | **1,153** | **1,137** | **2,855** |

The current `chunks.jsonl` is 4,589,830 bytes, SHA-256
`2360ddf7959f455da88a4d8005342f267c321cbe8a2b13ecefd9652ec4b25ae5`.
These fresh counts belong to this profile and these exact bytes. Older experiments
used different extraction and runtime configurations and cannot supply its count.

Strict loaders verify schemas, source metadata, record order, counts, page
association, output hashes and the absence of pending or failure markers before
and after reading. An independent consumer audit checked the installed package,
source bytes, loader and archived HTTP evidence in 27 bounded checks. Such checks
establish internal consistency and ordinary input-change detection; unsigned
hashes do not authenticate an externally supplied document or establish its
redistribution rights. See [data and dependency rights](../THIRD_PARTY.md).

## Lexical HTTP: transport works, duration evidence is unresolved

The consumer issued the same four FY2025-Q2 development questions twice: once
over all eight documents and once with explicit scope `nvidia-fy2025-q2`. The
questions ask for revenue and net income over three-month and six-month periods
ending July 28, 2024, in millions of dollars. Two owned loopback service children
produced eight HTTP records, with Top-5 retrieval and threshold **0.0**. Generation
was disabled and every answer remained empty.

Threshold 0.0 deliberately permits lexical inspection. A reported evidence gate
pass therefore does not mean that a financial number is supported. In both
scopes, the three-month and six-month revenue questions led with page 29; the
three-month and six-month net-income questions led with page 31. Page 3 was absent
from all eight Top-5 lists. The period collision remains visible instead of being
counted as financial correctness. BGE's 0.78 threshold is a separate setting and
must not be transferred to TF-IDF.

## BGE: actual encoding and a useful failure case

The semantic environment used `BAAI/bge-small-en-v1.5`, revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, on CPU with local-files-only loading.
The completed vector matrix has shape **2,855 × 384**, dtype `float32`, finite
values and row norms close to one. The strict loader exposes it read-only. Its
SHA-256 is
`a159cf92ec65912d4ffb14da2994a8e321b75d861240ba081aae88667d2e49d3`.

The actual comparison recorded eight query encodings: the original four questions
in each of the same two scopes. Top-5 produced 40 candidate rows at a fixed
threshold of **0.78**. There was no automatic annual routing, generation or human
adjudication. Scores below are archived observations; a cosine score measures
embedding similarity, not financial truth.

| Scope | FY2025-Q2 question | First candidate | Physical page | Cosine | Configured gate |
| --- | --- | --- | ---: | ---: | --- |
| Global | Revenue, three months | FY2023 annual | 149 | 0.807568 | Pass |
| Global | Revenue, six months | FY2023 annual | 149 | 0.811966 | Pass |
| Global | Net income, three months | FY2026 annual | 158 | 0.811011 | Pass |
| Global | Net income, six months | FY2023 annual | 149 | 0.807054 | Pass |
| FY2025 Q2 only | Revenue, three months | FY2025 Q2 | 23 | 0.786375 | Pass |
| FY2025 Q2 only | Revenue, six months | FY2025 Q2 | 23 | 0.782769 | Pass |
| FY2025 Q2 only | Net income, three months | FY2025 Q2 | 4 | 0.787178 | Pass |
| FY2025 Q2 only | Net income, six months | FY2025 Q2 | 3 | 0.778378 | **Refuse** |

All four global leaders are other-year annual reports even though each question
names FY2025 Q2. Passing 0.78 did not enforce the requested source or period. The
explicit Q2 scope confines the document, but its two revenue questions select
the same page-23 passage. The six-month net-income question's exact first score,
`0.7783780214668117`, is below the threshold: the service selects no evidence and
returns an empty context. All eight generated-answer fields are empty, including
the rows whose evidence gate passes. No financial accuracy rate is assigned.

An independent archive audit verified 51 frozen file identities, strict loaders,
the exact source/wheel/cache/vector binding, eight result records and 40 candidate
bindings, and reconstructed citation, context and refusal identities. It did not
load model weights or repeat queries. Query vectors were not retained in the
comparison receipt, so it did not independently recompute cosine values or
observe model-call counts. Its scope is archive closure, separate from the
operator's actual model run.

## Token coverage and the remaining gates

A separate tokenizer-only recount used the cached BGE tokenizer with special
tokens and with truncation disabled for counting. All 2,855 passage lengths were
between 3 and 471 tokens, below the installed model limit of 512. The four original
queries, including the retrieval instruction and special tokens, measured
46, 46, 47 and 47 tokens. This exact corpus has no measured input above the limit.
It does not establish preservation of table relationships, dates, units or
financial fields; truncation remains configured for future longer inputs.

The operator recorded one vector-build CLI wall time of about 123.65 seconds and
one comparison CLI wall time of about 9.05 seconds, including their setup and
closure work. These two observations are not latency distributions, throughput
benchmarks or scale acceptance. Installation and diagnostic histories were
retained; an incomplete first install log is not retrospectively labelled a
terminal failure or success.

This increment still needs publication CI and independent reproduction from a
public clone. Financial field and duration support, threshold calibration,
independent human gold, hosted API acceptance and Dot reproduction remain open.
The four exposed development questions provide a regression investigation, not
independent acceptance data. The next quality work must address the wrong-year
global candidates and the duration collision, then evaluate support against
independently reviewed evidence before making answer-quality claims. These local
runtime records do not establish learner mastery or completion of the full RAG
project.
