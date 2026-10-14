# Source fields preserve relationships; this BGE experiment did not fix retrieval

The source-only builder preserved 531 native income-statement fields from ten
reports. The next local experiment encoded those exact fields with cached BGE
and asked the same three exposed development questions. **The native attempt
failed its final loaded-module-origin audit. Separate posthoc checks validated
the retained arithmetic and source bindings, while preserving that failure.**

The results do not justify replacing the original raw-passage retriever. At the
unchanged Top K 5 and cosine threshold 0.78, only the requested annual Revenue
field survived. The three-month Net income field ranked eleventh; the six-month
field ranked third but remained below threshold. Source integrity and dense
retrieval quality are separate properties.

The [sanitized diagnostic](diagnostics/source-field-bge-20261014.json) records
the exact values, input and observation digests, the terminal failure, and the
limits of the later no-model checks. This is a diagnostic, not an accepted
field-vector bundle or a financial-accuracy evaluation.

## Hold the comparison fixed

The experiment used source revision `7039b98daea7fd25c3f3d0f2e87284211605637a`,
whose 22 package modules matched the separately audited wheel. An isolated
source snapshot ran with historical environment dependencies; this was not an
installed-wheel model run. It used `BAAI/bge-small-en-v1.5` revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, CPU, one thread, normalized float32
outputs and batch size 32. The original 3,240 raw chunks and their vectors
remained unchanged.

The source-only field snapshot and raw-bound report catalog were loaded
separately and compared against the same original registry, fiscal declarations,
PDFs and native index. The pure report decision selected the annual report or
Q2 report before actual `DenseRetriever.inspect` ranked all fields in that
report. Candidate populations were 55 / 72 / 72. No metric, date, duration,
comparison-role or expected-amount filter removed fields before ranking.

The sole new passage encode consumed all 531 field texts in accepted order.
Three query encodes used the original questions with the original BGE instruction
prefix exactly once. The retained concrete-function observations contain one
model constructor, four encodes, twenty preprocessing batches and twenty-one
tokenizer calls, including one complete untruncated preflight. All 534 effective
strings fit the 512-token limit; the maximum was 156. Every native masked token
sequence equaled its corresponding untruncated preflight sequence. The actual
encode bodies observed inference mode at entry and return.

## Read the requested field, not just the highest score

| Requested source field | Rank among all scoped fields | Cosine | Survives Top5 and 0.78 |
| --- | ---: | ---: | --- |
| FY2025 annual Revenue, current column | 4 of 55 | 0.7844318683 | Yes |
| FY2025 Q2 Net income, three months ended July 28, 2024 | 11 of 72 | 0.7781030011 | No |
| FY2025 Q2 Net income, six months ended July 28, 2024 | 3 of 72 | 0.7745978725 | No |

The annual top candidate was a prior-comparison **Cost of revenue** field,
not the requested current Revenue field. The three-month question's Top5 had
other metrics and columns, even though all five exceeded 0.78. Its software
evidence gate was true while the requested field was absent. The six-month
question's requested field was present before the gate but no candidate passed.
An application must not display a verified financial answer from these flags.

A separate NumPy-only process recomputed every scoped field score, query vector,
Top5 order, threshold decision and complete context from the retained arrays.
Maximum score and query-vector differences were zero. A fresh native PDF replay
then checked all fifteen Top5 candidates, including those below threshold:
complete cells, source PDF and page hashes, every text binding and the three
projected rectangles matched. These checks establish correspondence to native
source records; they do not make a wrong metric or comparison column relevant.

Using those same captured query vectors with the unchanged raw passage matrix
gave highest scores of 0.8072567675 / 0.7871780289 / 0.7783779982 and three / one /
zero gated passages. This was arithmetic replay, with no new raw encode or HTTP
request. Its score differences from the earlier raw native run were below
`9e-8`, and are retained rather than described as byte-identical. Raw candidate
counts and scores are not a spatial table-support adjudication.

## Preserve the terminal failure

After all encodes and queries, the final origin collector tried to resolve
Torch's virtual `torch.ops` namespace metadata, `__file__="_ops.py"`, as a real
path under the attempt directory. The file did not exist there. The installed
definition file exists; Torch intentionally registers a ModuleType singleton
with that placeholder metadata. The collector needed an explicit virtual-origin
classification, rather than a fallback that ignores unknown paths.

The attempt therefore remains failed, with its pending guard and failure record.
Its missing final loaded-module inventory cannot be reconstructed by a later
process. The parent rechecked all 85 protected inputs, 22 source-snapshot modules,
eight harness bodies, 6,090 dependency Python/metadata files and dependency
versions. Both owned processes exited, no forced termination was needed, and no
listener or generator ran. These file and process checks do not replace the
missing actual loaded-origin closure.

The posthoc diagnostic used a separate directory and preserved every original
attempt file. It made zero new model, tokenizer, retrieval, HTTP or generation
calls. Passing those diagnostic checks does not retroactively accept the failed
attempt. The retained vectors are not exposed through a production loader or API.

## What follows from this result

Keep exact source-field selection and semantic passage ranking separately typed.
The existing statement-candidate route supports explicit field selection; a
dense cosine score cannot substitute for the row, column, duration and unit
contract. Any later hybrid or reranking design needs its own scoring policy and
evaluation; a rank-fusion score must not be put through a cosine threshold.

This run used three already exposed development questions, with zero independent
human-gold or held-out questions. Printed dollar symbols remain ISO-unverified.
No financial correctness, calibrated accuracy, generation quality, installed-wheel
model execution, API integration, load performance or hosted service is claimed.
