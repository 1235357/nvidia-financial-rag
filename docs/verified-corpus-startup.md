# Launch the installed API from verified local sources

The installed `nvidia-rag-serve-corpus` command serves an existing verified lexical corpus. Supplying a local PDF directory and explicit fiscal declarations additionally configures `/v1/statement-candidates` and the explicit [`/v1/statement-fields` interface](statement-fields-api.md). Consumers can use the same startup composition from Python. Neither path downloads PDFs or models, generates an answer, infers a fiscal year, or establishes financial correctness.

The command is installed through `[project.scripts]`, following the [PyPA packaging guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#creating-executable-scripts). The module form is useful with isolated Python or an explicit container command.

## Prepare inputs once

Use [the local corpus recipe](local-financial-corpus.md) to prepare a completed fixed-profile corpus with `manifest.json`, `chunks.jsonl` and `build-receipt.json`. The exact source registry used to build it must accompany that corpus. Failed or pending build markers prevent startup; stale hashes, source associations or counts also prevent startup.

Statement mode needs the original registered PDFs in a separate local directory and a separate declaration file. Corpus schema 1 remains unchanged; its quarterly `reporting_period` field does not represent annual reports. The independent declaration schema is:

```json
{
  "schema_version": 1,
  "source_registry_sha256": "<exact-lowercase-SHA256-of-the-source-registry-bytes>",
  "reports": [
    {
      "document_id": "nvidia-fy2025-annual",
      "pdf_sha256": "<exact-registered-lowercase-PDF-SHA256>",
      "fiscal_year": 2025,
      "fiscal_quarter": null,
      "reviewed_by": "<person-or-process-that-reviewed-this-declaration>",
      "source_locator": "<specific-source-location-used-to-check-the-fiscal-identity>"
    }
  ]
}
```

Replace the bracketed strings with reviewed values from your own registry and original report. The displayed year and identifier are an example, not permission to assign those identities to another PDF. A quarterly declaration uses an integer quarter from 1 through 4. Annual declarations require an explicit year and null quarter; the loader never reads the year from a filename, URL or calendar end date. A declaration contradicting existing quarterly registry metadata is rejected.

Only the displayed fields are accepted. Duplicate JSON members, nonfinite constants, malformed or mismatched hashes, unknown reports, duplicate declarations, blank/noncanonical reviewer or locator text and invalid year/quarter types are refused. JSON inputs are limited to 1 MiB and registration/declarations to 128 reports. Selected PDFs retain the existing 256 MB and 2048-page limits. Absolute local paths and registered basenames are required; PDF paths containing symlinks, junctions or directory escapes are rejected before extraction. No arbitrary path from an HTTP request enters startup.

The registry/PDF hashes bind byte identity. An unsigned declaration remains an operator assertion and does not prove authenticity, source rights, human gold or an independently correct fiscal label.

## Start with the installed command

Install the selected wheel with `[api]`, then configure a strong `NVIDIA_RAG_API_TOKEN` privately in the invoking environment. All example paths below are operator-supplied absolute paths, not bundled data:

```text
python -I -B -m nvidia_rag.serving --artifact-dir <absolute-corpus-dir> --source-registry <absolute-source-registry.json> --statement-source-dir <absolute-PDF-dir> --statement-fiscal-declarations <absolute-fiscal-declarations.json> --threshold 0.01 --top-k 5
```

`nvidia-rag-serve-corpus` accepts the same arguments. The threshold is an explicit lexical inspection setting; the example value is not calibrated financial acceptance. Defaults are loopback `127.0.0.1:8000`, generation disabled, hidden docs, 16 simultaneous connections and five-second keep-alive. Connection bounds do not provide a complete request deadline or observed load capacity. Use `--expose-docs` only when you explicitly want the local API documentation.

Repeat `--document-id` to constrain both the lexical query service and the statement snapshot. Every selected ID must be registered. Declarations may cover more registered reports than the selected scope, but only selected declared PDFs are read and indexed. A scope with no declared statement reports fails startup; it never silently falls back to all sources.

Both statement options must be supplied together. With neither option, lexical mode runs and authenticated statement calls retain HTTP 503 `STATEMENT_INDEX_NOT_CONFIGURED`. If statement mode was explicitly requested but its inputs are invalid, the process exits before listening. Resolve the token before source reads, fitting, native extraction or optional server imports. All corpus/registry/declaration/PDF identities are rechecked after app construction. After startup, HTTP requests use the immutable snapshot without reloading disk files.

The source-tree `examples/serve_corpus.py` remains a loopback-only compatibility wrapper. Existing ordinary query and statement response schemas are unchanged; use [the statement HTTP guide](statement-http-api.md) to interpret source candidate, refusal, clarification and unsupported outcomes.

## Embed the factory

```python
import os
from pathlib import Path
from nvidia_rag.serving import create_corpus_app

app = create_corpus_app(
    os.environ,
    artifact_dir=Path(os.environ["REVIEWED_CORPUS_DIR"]),
    source_registry=Path(os.environ["REVIEWED_SOURCE_REGISTRY"]),
    threshold=0.01,
    top_k=5,
    document_ids=frozenset({"nvidia-fy2025-annual"}),
    statement_source_dir=Path(os.environ["REVIEWED_PDF_DIR"]),
    statement_fiscal_declarations=Path(os.environ["REVIEWED_FISCAL_DECLARATIONS"]),
)
```

The model-backed source-tree `run_api.py` additionally accepts the explicit trio `--statement-source-registry`, `--statement-source-dir` and `--statement-fiscal-declarations`. All three are required together. That path binds the optional statement registration to its configured corpus manifest and actual chunk identities/counts before importing the model/server runtimes. It retains the original BGE and generation configuration, then rechecks statement startup inputs after model/app construction. This integration does not replace preparation or acceptance of model artifacts.

## Run the existing image with read-only external inputs

The image still defaults to the generated-data demo. Override its command for this installed corpus mode; supply the token through the environment and mount your entire reviewed input directory read-only:

```bash
docker run --rm --read-only --cap-drop ALL --security-opt no-new-privileges:true \
  --publish 127.0.0.1:8000:8000 --env NVIDIA_RAG_API_TOKEN \
  --mount type=bind,source=<absolute-reviewed-input-dir>,target=/inputs,readonly \
  financial-rag-demo python -I -B -m nvidia_rag.serving --host 0.0.0.0 --port 8000 \
  --artifact-dir /inputs/corpus --source-registry /inputs/sources.json --threshold 0.01 \
  --statement-source-dir /inputs/pdfs --statement-fiscal-declarations /inputs/fiscal-declarations.json
```

The source mount must permit reads by UID/GID `10001:10001`. The explicit internal bind does not alter the example host loopback mapping. This is a deployment recipe; only a completed container test proves execution on its named host. No private source files, declaration secrets, trained indexes or models are copied into the image.

## Verify the startup boundary

From outside the checkout, run a fresh API-only environment's Python against the exact wheel:

```text
python -I -B <absolute-repository>/acceptance/serving_startup.py --wheel <absolute-built-wheel> --output <absolute-new-receipt.json>
```

The script checks all installed production module bytes against the wheel, rejects heavyweight model/UI dependencies, creates independent native synthetic annual/quarterly PDFs and a manually assembled unsigned corpus fixture, then launches the installed module in three owned child processes: configured, quarter-only and unconfigured. It uses actual loopback HTTP, compares complete source-cell transport and the canonical expected index hash, independently replays native spans/geometry, checks ordinary query text/scope, distinct content outcomes, authentication and the missing-index 503. Inputs remain unchanged; owned children and listeners must stop before the successful receipt is written. The synthetic values differ from unit-test source fixtures and are not financial reference facts. Windows teardown accounts for the virtual-environment interpreter redirector by stopping only each owned process tree.

The corrected [official run 38428973799](https://github.com/1235357/nvidia-financial-rag/actions/runs/38428973799) completed successfully for source revision `9687491a8c006a91ae2902689c1594a28cfe3e2c`. Its downloaded Linux and Windows archives each contain 842 passing tests, 20 exact installed production modules, 27 valid wheel records and 44 actual installed startup HTTP checks across three modes. The separate container archive records 26 HTTP checks across configured and unconfigured services; both used UID/GID `10001:10001`, a read-only root, read-only fixture mount, dropped capabilities and no-new-privileges, and both exited with code 0 after the owned stop. Docker healthcheck status during stop is not claimed as deployment health acceptance. [Sanitized release metadata](acceptance/enterprise-startup-release-20261013.json) binds the archive identities and scope. The earlier container permissions failure at run `38427779437` remains part of the development history.

Original-report transport replay, financial adjudication, model execution, public hosting, TLS, performance, browser-to-API integration and tenant authorization remain separate gates. This installed API-only fixture acceptance does not establish model execution or financial answer quality.

A later [current-source CPU model HTTP diagnostic](current-source-model-http.md) exercises the separate `run_api.py` path with the optional statement snapshot. It preserves a failed observation attempt, records a repaired model/HTTP run and identifies unsupported quarterly evidence from the retained annual router. It is separate from this installed API-only acceptance and does not establish financial quality.

The configured model path now has a [source-bound annual/quarter routing guide](source-bound-period-routing.md). With the statement trio, it builds an immutable report catalog from the retained source declarations and original PDF columns before model/server imports, scopes BGE ranking to that catalog, and refuses incompatible explicit source-task configuration. The installed lexical launcher retains its own behavior. The completed exposed same-question model comparison selects Q2 for both quarter questions; its six-month candidate remains below the original `0.78` gate. The guide preserves failed and successful attempts, independent vector arithmetic and separate installed release checks. Correct report scope and the unchanged v1 similarity-gate flags do not establish financial answer quality.
