# Typed Statement Fields Implementation Plan

> **For agentic workers:** Use subagent-driven-development for implementation and independent acceptance review. Preserve every failed attempt and validate the final installed interface outside the checkout.

**Goal:** Let an integrating application select an existing native income-statement field with explicit period and unit semantics, including per-share amounts and share counts.

**Architecture:** Add a pure typed selector and a separate authenticated HTTP route over the existing operator-built immutable index. Reuse complete source-cell output records. Keep natural-language statement parsing, semantic ranking, raw artifacts and generation contracts separate.

**Tech Stack:** Python dataclasses, existing native PyMuPDF index, Pydantic/FastAPI, pytest and actual loopback HTTP from an installed wheel.

## Evidence and design decision

The three exposed BGE probes did not justify promoting source-field embeddings: Net income ranked eleventh for three months and remained below the gate for six months. `docs/source-field-bge-diagnostic.md` preserves the failed native attempt and separate posthoc diagnostics. Do not lower the gate or claim a semantic retrieval repair from this interface.

The current literal selector already supports indexed monetary labels. Its parser deliberately excludes `per_share/` and `count/`, leaving 120 of the 531 retained source fields inaccessible through that vocabulary. The new interface accepts the actual section-qualified row path and an explicit unit contract; it does not invent a natural-language parse.

Alternatives considered: continue serializer tuning against the three exposed questions; add general query rewriting/reranking; or provide typed exact selection while evaluating semantic ranking separately. The last directly supports enterprise/UI integrations with known fields and makes the important unit distinction inspectable. It does not address open-ended natural-language quality, calculations or unseen accuracy by itself.

Official design reference: [Anyscale exact-match lookups](https://docs.anyscale.com/rag/quality-improvement/retrieval-strategies#exact-match-lookups). The [structured-data guide](https://docs.anyscale.com/rag/structured-data) concerns natively structured data; this project's source is PDF-extracted native cells, so it does not inherit a database-accuracy claim from that page.

## 1. Pure selection contract

- [x] Create `src/nvidia_rag/statement_fields.py` with a frozen request value and selection function. Required fields: operator scope, row path, period kind, exact ISO date, nullable duration, unit kind, scale and nullable printed currency. Optional report fiscal identity and column role constrain matches without defaulting to current.
- [x] Validate exact Python types, bounded strings/scope, canonical real dates and annual/month duration consistency before selection. No paths, amounts, source cells or quality labels are request inputs.
- [x] Match all source identity predicates. Reject unknown scope in full; clarify multiple source matches; refuse no match. Annual means native fiscal-year profile and null month duration, never assumed twelve months.
- [x] Keep source unit contracts distinct: monetary/source millions/printed dollar; per-share/one per-share unit/printed dollar; count/source millions of shares/no currency. Conflicting contracts, ISO conversion and scale conversion withhold cells.
- [x] Write meaningful synthetic native-PDF tests first, retain the failing run, then exercise all three unit kinds, prior columns, signs, annual and month periods, ambiguous source scopes and rejected contracts. Existing source index and cell bytes remain unchanged.

## 2. HTTP adapter

- [x] Create `src/nvidia_rag/statement_fields_api.py`; attach `/v1/statement-fields` in `api.py` with the same startup index and bearer dependency.
- [x] Strict JSON requires explicit nulls for nullable required fields and forbids unknown fields, duplicate decoded members, nonfinite values and coercion. Authenticate before reading the bounded 8192-byte stream; return generic input errors without reflection.
- [x] Return a separate response with typed `selection_request`, canonical request digest, index digest and complete native source candidates. Do not fabricate `parsed_request`, original question or mention offsets. No cosine score, generated answer or financial acceptance flag is added.
- [x] Complete domain selection, response validation and JSON rendering inside the operational guard. Missing index or processing fault returns 503, empty cells and null request metadata; established index identity remains available. Preserve existing query and literal-statement behavior.

## 3. Installed and original-source acceptance

- [x] Add `acceptance/statement_fields_http.py` with independent generated native PDF examples and a real owned loopback service outside the checkout. Check exact installed module/wheel identities, field values, source spans, request metadata, auth/bounds/withholding and shutdown. Its fixture outcome is transport and contract evidence, not human financial gold.
- [x] Run focused tests and full existing suite in the dedicated API/dev environment. Build a fresh wheel; install in a new consumer environment so earlier installed evidence remains reproducible.
- [x] Rebuild the ten-report native snapshot from retained original PDF identities; compare the existing 531-cell index and six omissions. Exercise all retained cells through typed requests and verify complete results plus selected negative cases. Label these source-derived development checks; they are not independent held-out evaluation.
- [ ] Publish the independently reviewed implementation and installed evidence through CI, then inspect the exact current run. Official installed checks now include the new route; current remote execution remains pending.

## 4. Reader-facing record and limits

- [x] Document the request and disposition contract with a clear example showing why per-share Basic differs from count Basic. Update startup and HTTP guides with an accurate link.
- [ ] Add a concise course-linked blog increment after implementation and publication evidence exist. Keep detailed operational records behind a link/disclosure and retain the shared LIGHT profile/lab and DARK standalone journal.
- [ ] Update task progress and memory with observed results, preserved failures and remaining work. Full portfolio/RAG goal stays active while independent financial gold/held-out evaluation, generation quality and deployment/handoff requirements remain incomplete.
