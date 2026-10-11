# Source, dependencies and data rights

Original project code and the release examples: Copyright (C) 2026 Zhentong Ye.
This published Python variant is licensed under **AGPL-3.0-or-later**. The complete,
unaltered license text is in [LICENSE](LICENSE), obtained from the official
[PyMuPDF 1.28.2 COPYING file](https://raw.githubusercontent.com/pymupdf/PyMuPDF/1.28.2/COPYING).
It is the Free Software Foundation's AGPL version 3 text; the local file is
34,520 bytes with SHA-256 `57c8ff33c9c0cfc3ef00e650a1cc910d7ee479a8bc509f6c9209a7c2a11399d6`.
Third-party dependencies and external data retain their own licenses and rights.

## Mandatory Python runtime

PyMuPDF and its MuPDF runtime are available under open-source AGPL or an Artifex
commercial license. This variant uses the open-source dependency. Selecting a
project license does not replace a dependency's terms. See the official
[PyMuPDF license and copyright documentation](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)
and the installed distribution's license files. Source availability and applicable
AGPL obligations must be preserved when distributing or operating modified
versions. This repository does not grant unrestricted proprietary relicensing
of the combined runtime or provide an Artifex commercial license.

The accepted API environment also used NumPy (BSD/MIT and bundled notices),
scikit-learn (BSD-3-Clause), tiktoken (MIT), FastAPI (MIT), HTTPX (BSD-3-Clause)
and Uvicorn (BSD-3-Clause). This is an attribution overview, not a complete
transitive-license inventory. Keep the installed dependency notices in any
distributed runtime; record its resolved dependency list. The optional demo,
semantic and generation extras add dependencies with separate terms.

## Models, reports and evaluation records

No model weights, model cache, embedding index, raw PDF, course record or mentor
material is bundled. BGE and Qwen runtimes are opt-in and remain subject to their
model and dependency licenses. Acquiring model weights is a separate operator
action; the generated-data release demo needs none.

NVIDIA financial filings are external source documents. Their official URLs,
report hashes and a small set of development candidate bindings may be retained
for attribution and reproducibility; this software license does not license or
grant redistribution rights to NVIDIA reports or other publishers' material.
Review source permissions before acquiring, indexing, displaying or redistributing
data. A source-registry declaration is not independent legal permission.

`examples/serve_demo.py` uses two newly generated transport-document passages,
identified as `synthetic-release-demo`. They contain no financial facts. Synthetic
transport checks and assistant-reviewed financial candidates are not human gold,
held-out benchmarks, NVIDIA answer accuracy or production acceptance.

NVIDIA, BGE, Qwen and other product names identify external technologies or source
material. This is an independent project; those organizations do not endorse it.
