# Ten-report corpus: expansion and retrieval failures

The local Python corpus now contains ten NVIDIA issuer reports and 3,240 source-linked chunks. Two first-quarter Form 10-Q reports were appended to the preserved eight-report baseline. The installed reviewed wheel built the corpus, a pinned cached BGE encoder built a separate passage matrix, and 24 diagnostic queries completed. The expansion is operationally reproducible at the recorded local scope. Financial answer readiness failed: retrieval still selects the wrong reporting period or rejects the intended statement page, and no generator or independent human gold evaluation ran.

The later [35-request expanded diagnostic](expanded-financial-comparator.md) preserves this corpus and passage matrix, adds pre-query refusal/clarification candidates and saves actual query vectors for independent arithmetic replay. Its additional source-span coverage and negative observations retain the failed financial-readiness result; the 24-request record below remains the original historical experiment.

This document records a local Python experiment for consumers of the source repository. The GitHub Pages browser demo uses the reader's selected PDF; the ten-report corpus is a separate prepared Python artifact. No public API hosts these artifacts, and CI did not independently run this real model experiment.

## What was added

| Report | Period ending | Whole PDF physical pages | Nonempty pages | Added chunks | Issuer PDF SHA-256 |
|---|---|---:|---:|---:|---|
| [NVIDIA fiscal 2025 Q1 Form 10-Q](https://s201.q4cdn.com/141608511/files/doc_financials/2025/q1/NVIDIA-10Q-20242905.pdf) | April 28, 2024 | 126 | 126 | 282 | `4459a2a6dbbf8fc737aa869deffdceea95b5161ec7f261d1b444c00b927fee8b` |
| [NVIDIA fiscal 2027 Q1 Form 10-Q](https://s201.q4cdn.com/141608511/files/doc_financials/2027/q1/927dc2d6-a76c-4006-9f34-8769b2c665fb.pdf) | April 26, 2026 | 45 | 45 | 103 | `9b48d36e503ba4dd3c2fb1414924ee5c7db9cd45f342cf7b280112d0ca6055f2` |

The ten source bodies total 168,446,328 bytes. Their 1,324 physical pages contain 1,308 nonempty pages. The FY2025 Q1 count includes the downloaded PDF's attachments and exhibits; it is not a claim that its core financial statement has 126 pages. The original eight source registry objects, eight manifest document records, and 2,855 chunk objects remain in their original order. The old JSONL file is an exact byte prefix of the new JSONL file.

The chunking profile is `word_safe`, using `cl100k_base`, a 400-token ceiling, and zero overlap. The BGE passage matrix is normalized `float32[3240,384]`, built on CPU from the prepared cache for `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The original matrix file remains unchanged. Re-encoding the preserved passage prefix into the new matrix is not bit-identical: its maximum observed absolute element difference is `2.086162567138672e-07`. This receipt records the difference rather than presenting two separately built matrices as identical.

## What the queries show

Twelve assistant-exposed positive development cases were frozen before querying: eight current/prior comparison cells from the two new Q1 reports and the four original Q2 controls. Each case ran once against all ten reports and once against an operator-provided explicit report scope, at Top-5 and threshold `0.78`. These 24 requests each recorded one encoder call. There is no automatic fiscal router and no negative or held-out denominator.

| Scope | Requests | Software evidence gate accepts | Intended document and page anywhere in Top-5 | Intended document and page in accepted evidence | Expected numeric literal in accepted evidence |
|---|---:|---:|---:|---:|---:|
| All ten reports | 12 | 12 | 1 | 0 | 1 |
| Explicit source report | 12 | 6 | 10 | 0 | 4 |

For example, the fiscal 2025 Q1 current revenue question globally ranks a fiscal 2024 annual-report passage on physical page 179 first, with similarity about `0.7920`. Restricting it to its actual report ranks page 26 first at about `0.7826`; the intended income-statement page 3 appears below the gate. The fiscal 2027 Q1 revenue questions rank the intended page 3 first within the correct report, but scores about `0.7664` and `0.7597` are below `0.78` and cause refusal. These are concrete period-selection and threshold failures.

`can_answer` in this experiment means that at least one passage passed the similarity evidence gate. It is not a financial answer verdict. A numeric literal or page hit does not prove the requested date, duration, metric row, comparison column, amount, currency, and unit jointly match. This run generated zero answers and established zero independent human gold judgments. Its rows are development diagnostics, not accuracy or generalization scores.

## Reproduce and inspect the local corpus

[The ten-source registry](../corpus/sources-ten-reports.json) contains only official issuer URLs, expected filenames, labels, reporting-period observations, and source hashes. It is byte-identical to the locally executed registry, with SHA-256 `4c570d9c8a8dd47afde9ad0a8c2245bd875c23c98bf409c0b3d828a775d3ebc5`. Obtain and review the original issuer documents separately, preserve their expected bytes, and use fresh absolute output paths. [The local corpus recipe](local-financial-corpus.md) documents wheel and tokenizer preparation, permissions, failure markers, and strict loaders.

`examples/build_corpus.py` accepts `--source-registry`, `--source-dir`, `--output-dir`, and `--tokenizer-cache-dir`. `examples/build_corpus_embeddings.py` accepts `--artifact-dir`, `--source-registry`, `--vector-dir`, and `--wheel`; it uses the separately prepared local model cache and requires the semantic dependencies. These commands create new artifact directories. They do not download source PDFs or prepare model weights. Existing completed output directories must be inspected and retained instead of rerun after an observation timeout.

`examples/serve_corpus.py` remains a loopback lexical evidence-only API recipe. Its explicit `--document-id` option scopes documents; its `--threshold` is an operator setting. The semantic experiment here used a separate local harness. The preserved original `acceptance/corpus_consumer.py` and `acceptance/corpus_semantic.py` comparators are still pinned to their eight-report baseline and four Q2 cases. Passing the ten-report registry to those original comparators is not a supported expansion command.

## Evidence identities and remaining gates

| Local artifact | Bytes | SHA-256 |
|---|---:|---|
| Ten-report `chunks.jsonl` | 5,209,819 | `094b3b59570b04c61d3bf54eb2d7468c73d2562a056a4924267f8dba3dbbe0e2` |
| Ten-report `manifest.json` | 6,440 | `77a9dae5f669caaf61d75d956e6559cf4104d32a4d106f4ffa8b3c6e7081c59f` |
| BGE matrix `.npy` | 4,976,768 | `6969739fb3744cea6fb9a1a62c5bbd2982c566f8a6fb5fc58443b99a48d9c591` |
| Saved development comparison | 106,660 | `b1901e5b82fbfcaf7de729069a6070f30396518431ca0891bb1735f06f2b8b55` |
| Installed reviewed wheel | 105,190 | `76246d9638a6ddfb30e0abeace884fa67115fa8e38a4187590efeb52f86701e9` |

[The public metadata projection](acceptance/local-ten-report-runtime-20261012.json) retains result identities, exact top-result observations, limitations, and source hashes without redistributing PDF bodies, extracted report text, tensors, model weights, or private machine paths. The source commit audited here is `155758ca536354933a1f29075eb9cc48ada78932`. That commit supersedes the historical freeze for `tests/test_corpus_consumer.py` and `tests/test_corpus_semantic.py`; the other fourteen frozen source entries, sixteen production modules, original wheel, and both installed module copies remain unchanged. Historical runtime receipts remain historical records.

An independent read-only review recomputed 364 artifact identity and result binding checks: ten source hashes, the preserved chunk prefix, matrix shape and normalization, wheel and installed module identities, all 120 candidate chunk bindings, threshold-filtered evidence IDs, context digests, and page/literal observations. Query vector values were not stored. The review therefore does not independently reproduce cosine arithmetic or encoder execution and ran no models, queries, builds, or new wheels.

The next quality gate requires source-qualified unseen cases and independent field-level judgments covering report identity, date, duration, metric row, column, amount, currency, and unit, including negative/refusal cases. A verified fiscal 2025 Q3 Form 10-Q URL remains pending. Public PDF redistribution permission has not been established. Larger storage, successful CI, or more accepted similarity scores cannot close these remaining financial readiness gates.
