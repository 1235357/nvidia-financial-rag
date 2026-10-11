"""Frozen explicit-field/source-coverage diagnostic; no NLU or model ranking.

Candidate values are assertions in source replay, never selection inputs.
The Net income-only sidecar is compared with original raw lexical/value coverage.
Synthetic controls exercise contracts, not financial false-accept reliability.
"""
import argparse
from copy import deepcopy
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).resolve().parents[1]
CELL_DIR = ROOT / "artifacts" / "table-net-income-experiment-20261011-01"
CANDIDATES = ROOT / "evaluation" / "source-candidates" / "fy2025-q2-income-duration.jsonl"
CELL_SHA256 = "3f6c35f8d7fcbf0b960be74fc168b0e97ff77e558b5ac63b3ff94fe3ac70d37f"
CELL_RECEIPT_SHA256 = "9169e4a03ea943c39d106ca3dc3c53663bf45a5b80ec86914b8de2d2f9042a7b"
PRESERVATION_SHA256 = "0daf235b54df83b1c7f6cba65fe580ed139d673920487ec730f40ed4ae59412c"
EXTRACTOR_SHA256 = "ead976a3076536be1c81bd0b09e1f967e013dbf836b4995444221133c696ddac"
CANDIDATES_SHA256 = "9cf18ce2edd0a0348df794e4b5526c6155eef869525e9b22385b007f16deb21d"
RAW_SHA256 = "33bac1bf14149c1538e003dee78cdab12e71818687961480c1e7c7b061d18e8a"
UNIT = "In millions, except per share data"
PROFILE_KEYS = {"metric", "duration_months", "period_end", "unit_literal"}
QUERY_PROFILES = {
    "q2_regression_revenue_three_months": {"metric": "Revenue", "duration_months": 3,
                                          "period_end": "2024-07-28", "unit_literal": UNIT},
    "q2_regression_revenue_six_months": {"metric": "Revenue", "duration_months": 6,
                                        "period_end": "2024-07-28", "unit_literal": UNIT},
    "q2_regression_net_income_three_months": {"metric": "Net income", "duration_months": 3,
                                             "period_end": "2024-07-28", "unit_literal": UNIT},
    "q2_regression_net_income_six_months": {"metric": "Net income", "duration_months": 6,
                                           "period_end": "2024-07-28", "unit_literal": UNIT},
}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def digest_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_digest(path, expected):
    actual = digest_file(path)
    require(actual == expected, "INPUT_DIGEST_MISMATCH")
    return actual


def valid_profile(profile):
    if not isinstance(profile, dict) or set(profile) != PROFILE_KEYS:
        return False
    if not (isinstance(profile["metric"], str) and 0 < len(profile["metric"]) <= 80
            and type(profile["duration_months"]) is int and 0 < profile["duration_months"] <= 120
            and isinstance(profile["unit_literal"], str) and 0 < len(profile["unit_literal"]) <= 120
            and isinstance(profile["period_end"], str)):
        return False
    try:
        return date.fromisoformat(profile["period_end"]).isoformat() == profile["period_end"]
    except ValueError:
        return False


def valid_cell(cell):
    try:
        profile = {key: cell[key] for key in PROFILE_KEYS}
        bindings = cell["bindings"]
        period = date.fromisoformat(cell["period_end"])
        duration = {3: "Three Months Ended", 6: "Six Months Ended"}[cell["duration_months"]]
        raw = cell["value_text"]
        return (valid_profile(profile) and isinstance(cell["cell_id"], str) and bool(cell["cell_id"])
                and re.fullmatch(r"\(?\d{1,3}(?:,\d{3})*\)?", raw) is not None
                and type(cell["value"]) is int
                and cell["value"] == int(raw.strip("()").replace(",", "")) * (-1 if raw.startswith("(") else 1)
                and bindings["value"]["text"] == raw
                and bindings["row"]["text"] == cell["metric"]
                and bindings["unit"]["text"] == cell["unit_literal"]
                and bindings["duration"]["text"] == duration
                and period.month == 7
                and bindings["period_end"]["text"] == f"Jul {period.day}, {period.year}")
    except (KeyError, TypeError, ValueError):
        return False


def select_cell(cells, profile):
    """Exact tuple lookup only. Authority is supplied by the separate PDF replay."""
    if not valid_profile(profile):
        return {"status": "refused", "reason": "INVALID_QUERY_PROFILE"}
    if not isinstance(cells, list) or len(cells) > 10000 or not all(valid_cell(c) for c in cells):
        return {"status": "refused", "reason": "INVALID_CELL_FIELDS"}
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        return {"status": "refused", "reason": "DUPLICATE_CELL_ID"}
    matches = [cell for cell in cells if all(cell[key] == profile[key] for key in PROFILE_KEYS)]
    if len(matches) == 0:
        return {"status": "refused", "reason": "NO_EXACT_CELL"}
    if len(matches) != 1:
        return {"status": "refused", "reason": "AMBIGUOUS_EXACT_CELL"}
    cell = matches[0]
    return {"status": "selected", "cell_id": cell["cell_id"],
            "value_text": cell["value_text"], "value": cell["value"]}


def raw_chunk_coverage(chunk, candidate, profile, corpus_digest):
    """Validate the reviewed raw value span; measure lexical presence separately."""
    source = candidate["source"]
    require(valid_profile(profile) and profile["metric"] == source["row"]
            and profile["duration_months"] == source["duration_months"]
            and profile["period_end"] == source["period_end"] and profile["unit_literal"] == UNIT,
            "QUERY_SOURCE_PROFILE_MISMATCH")
    binding = candidate["original_chunk_binding"]
    require(binding["corpus_sha256"] == corpus_digest and chunk["chunk_id"] == binding["chunk_id"]
            and chunk["document_id"] == source["document_id"]
            and chunk["page_number"] == source["physical_page"], "RAW_SOURCE_IDENTITY_MISMATCH")
    require(binding["offset_unit"] == "unicode_code_points" and binding["interval"] == "half_open",
            "RAW_BINDING_PROFILE_MISMATCH")
    text, span = chunk["text"], binding["value_span"]
    require(type(span["start"]) is int and type(span["end"]) is int
            and 0 <= span["start"] < span["end"] <= len(text)
            and text[span["start"]:span["end"]] == span["text"] == source["value_text"],
            "RAW_VALUE_SPAN_MISMATCH")
    duration = {3: "Three Months Ended", 6: "Six Months Ended"}[profile["duration_months"]]
    end = date.fromisoformat(profile["period_end"])
    lexemes = {"row": profile["metric"], "duration": duration,
               "period_end": f"Jul {end.day}, {end.year}", "unit": UNIT, "value": span["text"]}
    lines = [line.strip(" \t\u00a0") for line in text.splitlines()]
    occurrences = {name: (text.count(phrase) if name == "unit" else lines.count(phrase))
                   for name, phrase in lexemes.items()}
    return {"chunk_id": chunk["chunk_id"], "value_span_verified": True,
            "value_span": span, "lexical_presence": {key: count > 0 for key, count in occurrences.items()},
            "lexical_occurrences": occurrences, "semantic_field_entailment_established_here": False,
            "deterministic_cell_selection": "not-evaluated"}


def controlled_checks(cells):
    """Synthetic contract mutations; no financial negative labels are constructed."""
    base = QUERY_PROFILES["q2_regression_net_income_three_months"]
    cases = []

    def add(name, rows, query, expected):
        outcome = select_cell(rows, query)
        cases.append({"control_id": name, "kind": "synthetic-contract-control",
                      "expected_refusal_reason": expected, "outcome": outcome,
                      "passed": outcome == {"status": "refused", "reason": expected},
                      "human_acceptance": False, "development_exposed": True})

    for name, changes in [("unsupported-duration", {"duration_months": 12}),
                          ("uncovered-year", {"period_end": "2025-07-28"}),
                          ("unsupported-row", {"metric": "Adjusted net income"}),
                          ("uncovered-unit", {"unit_literal": "thousands"})]:
        add(name, cells, {**base, **changes}, "NO_EXACT_CELL")
    missing = {key: value for key, value in base.items() if key != "duration_months"}
    add("missing-duration", cells, missing, "INVALID_QUERY_PROFILE")
    missing = {key: value for key, value in base.items() if key != "unit_literal"}
    add("missing-unit", cells, missing, "INVALID_QUERY_PROFILE")
    target = next(cell for cell in cells if all(cell[key] == base[key] for key in PROFILE_KEYS))
    duplicate = deepcopy(target)
    duplicate["cell_id"] = "synthetic-duplicate:" + target["cell_id"]
    add("duplicate-agreeing-cell", cells + [duplicate], base, "AMBIGUOUS_EXACT_CELL")
    conflicting = deepcopy(duplicate)
    conflicting["cell_id"] = "synthetic-conflict:" + target["cell_id"]
    conflicting["value"] += 1
    conflicting["value_text"] = format(conflicting["value"], ",")
    conflicting["bindings"]["value"]["text"] = conflicting["value_text"]
    add("conflicting-cell", cells + [conflicting], base, "AMBIGUOUS_EXACT_CELL")
    broken = deepcopy(cells)
    broken[0]["bindings"].pop("unit")
    add("cell-missing-unit-binding", broken, base, "INVALID_CELL_FIELDS")
    require(all(case["passed"] for case in cases), "CONTROL_CONTRACT_FAILURE")
    return cases


def build_comparison(pdf_path, raw_path, cell_dir=CELL_DIR, candidates_path=CANDIDATES):
    require(all(Path(path).is_absolute() for path in (pdf_path, raw_path, cell_dir, candidates_path)),
            "INPUT_PATH_MUST_BE_ABSOLUTE")
    checked_digest(raw_path, RAW_SHA256)
    checked_digest(candidates_path, CANDIDATES_SHA256)
    checked_digest(Path(cell_dir) / "cells.jsonl", CELL_SHA256)
    checked_digest(Path(cell_dir) / "receipt.json", CELL_RECEIPT_SHA256)
    preservation_path = Path(cell_dir) / "preservation-review.json"
    checked_digest(preservation_path, PRESERVATION_SHA256)
    extractor_path = ROOT / "acceptance" / "table_cells_experiment.py"
    checked_digest(extractor_path, EXTRACTOR_SHA256)
    preservation = json.loads(preservation_path.read_text(encoding="utf-8"))
    frozen_files = preservation["preservation"]["before_sha256"]
    before = {path: checked_digest(path, expected) for path, expected in frozen_files.items()}
    dependencies = {"table_cells_experiment.py": EXTRACTOR_SHA256,
                    "source_binding_replay.py": digest_file(ROOT / "acceptance" / "source_binding_replay.py")}
    extractor = runpy.run_path(str(extractor_path))
    extractor["verify_artifacts"](cell_dir, pdf_path)
    replay = runpy.run_path(str(ROOT / "acceptance" / "source_binding_replay.py"))
    candidate_replay = replay["replay"](Path(pdf_path), Path(candidates_path))
    candidates = replay["load_candidates"](Path(candidates_path))
    cells = [json.loads(line) for line in (Path(cell_dir) / "cells.jsonl").read_text(encoding="utf-8").splitlines()]
    wanted = {record["original_chunk_binding"]["chunk_id"] for record in candidates}
    chunks = {}
    with Path(raw_path).open(encoding="utf-8") as stream:
        for line in stream:
            chunk = json.loads(line)
            if chunk["chunk_id"] in wanted:
                require(chunk["chunk_id"] not in chunks, "DUPLICATE_RAW_CHUNK_ID")
                chunks[chunk["chunk_id"]] = chunk
    require(set(chunks) == wanted, "MISSING_RAW_CHUNK")
    outcomes = []
    for candidate in candidates:
        identifier = candidate["question_id"]
        profile = QUERY_PROFILES[identifier]
        raw = raw_chunk_coverage(chunks[candidate["original_chunk_binding"]["chunk_id"]],
                                 candidate, profile, RAW_SHA256)
        selected = select_cell(cells, profile)
        if selected["status"] == "selected":
            # Candidate source bindings are assertions after actual cell derivation.
            require(selected["value_text"] == candidate["source"]["value_text"], "SOURCE_VALUE_ASSERTION_MISMATCH")
        outcomes.append({"question_id": identifier, "query_text_sha256": sha256(candidate["query"].encode()).hexdigest(),
                         "explicit_query_profile": profile, "profile_resolution": "frozen-explicit-fields-not-NLU",
                         "source_page_binding_replayed": True, "raw_baseline": raw,
                         "net_income_sidecar_selection": selected,
                         "candidate_status": candidate["label_status"], "human_acceptance": False,
                         "development_exposed": True})
    controls = controlled_checks(cells)
    after = {path: digest_file(path) for path in frozen_files}
    require(before == after, "PRESERVED_INPUT_CHANGED")
    input_files = {str(Path(raw_path)): RAW_SHA256, str(Path(candidates_path)): CANDIDATES_SHA256,
                   str(Path(cell_dir) / "cells.jsonl"): CELL_SHA256,
                   str(Path(cell_dir) / "receipt.json"): CELL_RECEIPT_SHA256,
                   str(preservation_path): PRESERVATION_SHA256}
    require(all(digest_file(path) == expected for path, expected in input_files.items()), "INPUT_CHANGED_DURING_RUN")
    require(dependencies["source_binding_replay.py"] == digest_file(ROOT / "acceptance" / "source_binding_replay.py")
            and digest_file(extractor_path) == EXTRACTOR_SHA256, "DEPENDENCY_CHANGED_DURING_RUN")
    selected_count = sum(row["net_income_sidecar_selection"]["status"] == "selected" for row in outcomes)
    return {"schema_version": 1, "scope": "frozen-explicit-field-and-source-coverage-development-comparison-v1",
            "status": "AI-source-reviewed-experimental", "reviewer": "CodexAI", "human_acceptance": False,
            "human_verdict": None, "development_exposed": True, "model_runs": 0, "embedding_runs": 0,
            "network_runs": 0, "retrieval_integration": "not-run", "input_sha256": input_files,
            "dependency_sha256": dependencies, "query_profile_sha256": sha256(json.dumps(
                QUERY_PROFILES, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "candidate_source_replay": {"passed": candidate_replay["source_binding_passed"],
                "candidate_count": candidate_replay["candidate_count"], "pdf_sha256": candidate_replay["source_pdf_sha256"],
                "page_text_sha256": candidate_replay["page_text_sha256"],
                "profile": candidate_replay["extraction_profile"]},
            "sidecar_source_replay_passed": True, "cases": outcomes, "synthetic_controls": controls,
            "coverage_counts": {"development_profiles": len(outcomes),
                "raw_bound_values": sum(row["raw_baseline"]["value_span_verified"] for row in outcomes),
                "raw_complete_lexical_presence": sum(all(row["raw_baseline"]["lexical_presence"].values()) for row in outcomes),
                "sidecar_exact_profile_matches": selected_count, "sidecar_uncovered_profiles": len(outcomes) - selected_count,
                "synthetic_contract_controls": len(controls), "synthetic_contract_controls_passed": len(controls)},
            "financial_negative_count": 0, "financial_false_accept_rate": None,
            "preservation": {"checked_files": len(before), "unchanged": True, "before_sha256": before, "after_sha256": after},
            "not_evaluated": ["natural-language parsing", "retrieval ranking or cosine", "raw deterministic cell selection",
                "financial negative/refusal reliability", "generation", "human gold or heldout accuracy", "browser lineage"],
            "value_policy": "Expected-answer strings are not selector inputs. Existing source-candidate replay verifies frozen label bookkeeping; raw/source spans and derived cells are compared separately."}


def write_artifacts(directory, report):
    body = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    receipt = {"schema_version": 1, "status": "AI-source-reviewed-experimental", "human_acceptance": False,
               "comparison_sha256": sha256(body).hexdigest(), "comparison_bytes": len(body),
               "script_sha256": digest_file(__file__), "model_runs": 0, "embedding_runs": 0,
               "retrieval_integration": "not-run", "created_at_utc": datetime.now(timezone.utc).isoformat()}
    directory = Path(directory)
    directory.mkdir(parents=False, exist_ok=False)
    with (directory / "comparison.json").open("xb") as target:
        target.write(body)
    with (directory / "receipt.json").open("x", encoding="utf-8", newline="\n") as target:
        json.dump(receipt, target, indent=2, sort_keys=True)
        target.write("\n")
    return receipt


def verify_report_digest(directory):
    directory = Path(directory)
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    body = (directory / "comparison.json").read_bytes()
    require(sha256(body).hexdigest() == receipt["comparison_sha256"]
            and len(body) == receipt["comparison_bytes"], "REPORT_DIGEST_MISMATCH")
    require(digest_file(__file__) == receipt["script_sha256"], "BENCHMARK_CODE_MISMATCH")
    return json.loads(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--raw-corpus", type=Path, required=True)
    parser.add_argument("--cell-dir", type=Path, default=CELL_DIR)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    require(args.output_dir.is_absolute() and args.output_dir.resolve().is_relative_to(
        (ROOT / "artifacts").resolve()) and args.output_dir.resolve() != (ROOT / "artifacts").resolve(),
        "OUTPUT_OUTSIDE_STAGED_ARTIFACTS")
    report = build_comparison(args.pdf, args.raw_corpus, args.cell_dir, args.candidates)
    if args.verify_existing:
        require(verify_report_digest(args.output_dir) == report, "COMPARISON_SOURCE_REPLAY_MISMATCH")
        receipt = json.loads((args.output_dir / "receipt.json").read_text(encoding="utf-8"))
    else:
        receipt = write_artifacts(args.output_dir, report)
    print(json.dumps({"status": report["status"], "coverage_counts": report["coverage_counts"],
                      "comparison_sha256": receipt["comparison_sha256"],
                      "source_replay": bool(args.verify_existing), "human_acceptance": False,
                      "model_runs": 0, "embedding_runs": 0, "financial_false_accept_rate": None}, indent=2))


if __name__ == "__main__":
    main()
