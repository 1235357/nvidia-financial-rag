"""Synthetic contract controls, not financial correctness or human gold."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "acceptance" / "expanded_cases.py"
H = "a" * 64
COUNTS = {"cases": 1, "evidence": 1, "refuse": 0, "clarify": 0,
          "negative": 0, "development": 1, "heldout": 0, "human_gold": 0}


def api():
    assert SCRIPT.exists(), "strict expanded case loader must be implemented"
    spec = importlib.util.spec_from_file_location("expanded_cases_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def experiment():
    return {
        "schema_version": 2, "run_kind": "diagnostic",
        "input_sha256": {name: H for name in (
            "registry", "corpus_manifest", "corpus_chunks", "vector_manifest",
            "vector_matrix", "cases", "case_freeze", "source_replay", "wheel")},
        "documents": [{"document_id": "nvidia-fy2027-q1", "report_fiscal_year": 2027,
                       "report_fiscal_quarter": 1, "pdf_sha256": H,
                       "issuer_url": "https://example.org/issuer/report.pdf", "page_count": 45}],
        "document_count": 1, "chunk_count": 42,
        "expected_case_ids": ["positive"], "case_counts": deepcopy(COUNTS),
        "model": {"name": "BAAI/bge-small-en-v1.5", "revision": "b" * 40,
                  "instruction": "Represent this sentence for searching relevant passages: ",
                  "device": "cpu", "dimension": 384,
                  "raw_dtype": "float32", "normalized_dtype": "float64"},
        "query_normalization": "strip", "top_k": 5, "primary_threshold": 0.78,
        "threshold_grid": [0.78], "supported_currencies": ["USD"],
        "supported_unit_kinds": ["monetary", "per_share", "percent", "count"],
        "metric_unit_kinds": {"revenue": "monetary", "net_income": "monetary", "basic_eps": "per_share"},
        "rationale_bindings": [{"binding_id": "inventory:1", "kind": "source-column-inventory", "receipt_sha256": H}],
        "scopes": [{"id": "all-reports", "document_ids": None},
                   {"id": "q1", "document_ids": ["nvidia-fy2027-q1"]}],
        "limits": {"max_cases": 64, "max_cells_per_case": 16,
                   "max_source_text_chars": 100000, "max_case_file_bytes": 1000000},
    }


def cell():
    texts = {"statement": "Income", "unit": "In millions", "currency": "$",
             "duration": "Three Months Ended", "date": "Apr 27, 2025",
             "row": "Revenue", "value": "44,062"}
    spans, offset = {}, 0
    for key, text in texts.items():
        spans[key] = {"start": offset, "end": offset + len(text), "text": text,
                      "bbox": [10, 20, 60, 30]}
        offset += len(text) + 1
    return {
        "cell_id": "", "document_id": "nvidia-fy2027-q1", "pdf_sha256": H,
        "issuer_url": "https://example.org/issuer/report.pdf", "physical_page": 3,
        "printed_page": "3", "report_fiscal_year": 2027, "report_fiscal_quarter": 1,
        "statement": "Income", "metric": "revenue", "row": "Revenue",
        "column_role": "prior_comparison", "period_kind": "duration",
        "period_end": "2025-04-27", "duration_months": 3,
        "currency_literal": "$", "unit_literal": "In millions",
        "currency": "USD", "unit_kind": "monetary", "scale": "1000000",
        "normalization_status": "assistant-source-reviewed-candidate",
        "amount": "44062", "value_text": "44,062",
        "bindings": {"status": "candidate-complete", "page_text_sha256": H,
                     "page_text_chars": 1000,
                     "extraction_profile": {"library": "PyMuPDF", "version": "1.28.2",
                         "mode": "text", "sort": True, "normalization": "none",
                         "text_encoding": "utf-8", "offset_unit": "unicode_code_points",
                         "interval": "half_open"},
                     "spans": spans, "association_receipt_sha256": H},
    }


def case():
    source = cell()
    # The test computes the canonical identity independently of the loader.
    identity = {key: source[key] for key in (
        "document_id", "pdf_sha256", "physical_page", "statement", "metric", "row",
        "column_role", "period_kind", "period_end", "duration_months")}
    identity["text_coordinates"] = {key: source["bindings"][key] for key in
                                    ("page_text_sha256", "extraction_profile")}
    identity["value_location"] = {key: source["bindings"]["spans"]["value"][key]
                                  for key in ("start", "end", "bbox")}
    source["cell_id"] = "cell:" + sha256(json.dumps(identity, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return {
        "case_id": "positive", "query": "What was prior comparison revenue?",
        "origin": {"kind": "new-candidate", "original_record_utf8": None,
                   "original_record_sha256": None},
        "exposure": {"split": "development", "development_exposed": True,
                     "first_exposed_at_utc": "2026-10-12T00:00:00Z", "freeze_purpose": "development-replay"},
        "review": {"status": "assistant-source-reviewed-candidate", "reviewer": "CodexAI",
                   "human_acceptance": False, "human_verdict": None,
                   "adjudication_receipt_sha256": None},
        "request": {"report_document_id": "nvidia-fy2027-q1", "report_fiscal_year": 2027,
                    "report_fiscal_quarter": 1, "metric": "revenue", "period_kind": "duration",
                    "period_end": "2025-04-27", "duration_months": 3,
                    "currency": "USD", "unit_kind": "monetary", "scale": "1000000"},
        "expected": {"outcome": "evidence", "cell_ids": [source["cell_id"]],
                     "negative_reason": None, "rationale_binding_ids": []},
        "scope_ids": ["all-reports", "q1"], "source_cells": [source],
    }


def write_inputs(tmp_path, records=None, config=None, freeze_edit=None, raw=None):
    records = [case()] if records is None else records
    config = experiment() if config is None else config
    body = raw if raw is not None else b"".join(
        (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode() for row in records)
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_bytes(body)
    outcomes = [row["expected"]["outcome"] for row in records]
    counts = {"cases": len(records), "evidence": outcomes.count("evidence"),
              "refuse": outcomes.count("refuse"), "clarify": outcomes.count("clarify"),
              "negative": len(records) - outcomes.count("evidence"),
              "development": len(records), "heldout": 0, "human_gold": 0}
    freeze = {"schema_version": 2, "cases_sha256": sha256(body).hexdigest(),
              "ordered_case_ids": [r["case_id"] for r in records], "counts": counts,
              "frozen_at_utc": "2026-10-12T00:01:00Z", "freeze_purpose": "development-replay",
              "case_status": [{"case_id": r["case_id"], "outcome": r["expected"]["outcome"],
                  "review_status": r["review"]["status"], "development_exposed": r["exposure"]["development_exposed"],
                  "original_record_sha256": r["origin"]["original_record_sha256"]} for r in records]}
    if freeze_edit:
        freeze_edit(freeze)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    config["input_sha256"]["cases"] = sha256(body).hexdigest()
    config["input_sha256"]["case_freeze"] = sha256(freeze_path.read_bytes()).hexdigest()
    config["expected_case_ids"] = [r["case_id"] for r in records]
    config["case_counts"] = counts.copy()
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return cases_path, freeze_path, config_path


def load(paths):
    module = api()
    return module.load_case_bundle(paths[0], paths[1], module.load_experiment(paths[2]))


@pytest.mark.parametrize('outcome', ['evidence', 'refuse', 'clarify'])
def test_registered_report_request_cannot_use_other_report_scope(tmp_path, outcome):
    row, cfg = case(), experiment()
    other = deepcopy(cfg['documents'][0])
    other.update(document_id='nvidia-fy2025-q2', report_fiscal_year=2025, report_fiscal_quarter=2)
    cfg['documents'].append(other)
    cfg['document_count'] = 2
    cfg['scopes'].append({'id': 'wrong-existing-report', 'document_ids': ['nvidia-fy2025-q2']})
    row['scope_ids'] = ['wrong-existing-report']
    if outcome != 'evidence':
        row['expected'] = {'outcome': outcome, 'cell_ids': [], 'negative_reason': 'provisional control',
                           'rationale_binding_ids': ['inventory:1']}
    with pytest.raises(ValueError, match='scope'):
        load(write_inputs(tmp_path, [row], cfg))


@pytest.mark.parametrize('field', ['threshold', 'grid', 'bbox'])
def test_huge_integer_numeric_fields_raise_validation_error(tmp_path, field):
    row, cfg = case(), experiment()
    if field == 'threshold':
        cfg['primary_threshold'] = 10 ** 400
    elif field == 'grid':
        cfg['threshold_grid'] = [0.78, 10 ** 400]
    else:
        row['source_cells'][0]['bindings']['spans']['value']['bbox'][2] = 10 ** 400
    with pytest.raises(ValueError):
        load(write_inputs(tmp_path, [row], cfg))


def test_prior_calendar_year_belongs_to_its_report_and_labels_are_not_query_input(tmp_path):
    bundle = load(write_inputs(tmp_path))
    assert bundle.cases[0].request["report_fiscal_year"] == 2027
    assert bundle.cases[0].source_cells[0]["period_end"] == "2025-04-27"
    assert bundle.cases[0].source_cells[0]["binding_complete"] is True
    assert bundle.human_gold_count == 0
    assert bundle.source_binding_passed is False
    assert bundle.joint_source_candidate_coverage == "not_evaluated"
    rows = bundle.retrieval_inputs()
    assert len(rows) == 2 and rows[0].document_ids is None
    assert asdict(rows[0]) == {"case_id": "positive", "query": case()["query"],
                            "scope_id": "all-reports", "document_ids": None}
    with pytest.raises(TypeError):
        bundle.cases[0].request["period_end"] = "2026-04-26"


def test_incomplete_q1_geometry_remains_unreplayed_candidate(tmp_path):
    row = case()
    row["source_cells"][0]["bindings"]["spans"]["date"] = None
    row["source_cells"][0]["bindings"]["association_receipt_sha256"] = None
    row["source_cells"][0]["bindings"]["status"] = "incomplete"
    bundle = load(write_inputs(tmp_path, [row]))
    assert bundle.cases[0].source_cells[0]["binding_complete"] is False
    assert bundle.source_binding_passed is False


@pytest.mark.parametrize("outcome", ["refuse", "clarify"])
def test_negative_labels_allow_absent_or_unsupported_request_qualifiers(tmp_path, outcome):
    row = case()
    row["case_id"] = "negative"
    row["request"].update(report_document_id="not-in-corpus", report_fiscal_year=2035,
                          duration_months=None, currency="EUR", metric=None)
    row["expected"] = {"outcome": outcome, "cell_ids": [],
        "negative_reason": "provisional capability control", "rationale_binding_ids": ["inventory:1"]}
    row["source_cells"] = []
    row["scope_ids"] = ["all-reports"]
    bundle = load(write_inputs(tmp_path, [row]))
    assert bundle.counts["negative"] == 1
    assert bundle.retrieval_inputs()[0].query == row["query"]


@pytest.mark.parametrize("change", [
    "report_year", "missing_duration", "impossible_date", "money_eps", "boolean_page",
    "zero_box", "offset", "amount", "scale", "currency", "pdf_hash", "issuer_url",
    "false_complete", "cell_id", "exposed_heldout", "human", "unknown", "unknown_nested",
    "unknown_scope", "empty_scopes", "unknown_cell", "bad_timestamp", "unit_inference",
])
def test_invalid_source_or_case_claims_fail_closed(tmp_path, change):
    row = case()
    source = row["source_cells"][0]
    if change == "report_year": source["report_fiscal_year"] = 2026
    elif change == "missing_duration": source["duration_months"] = None
    elif change == "impossible_date": source["period_end"] = "2025-02-30"
    elif change == "money_eps": source["metric"] = "basic_eps"
    elif change == "boolean_page": source["physical_page"] = True
    elif change == "zero_box": source["bindings"]["spans"]["value"]["bbox"] = [1, 1, 1, 2]
    elif change == "offset": source["bindings"]["spans"]["row"]["start"] = 9999
    elif change == "amount": source["amount"] = "44063"
    elif change == "scale": source["scale"] = "NaN"
    elif change == "currency": source["currency"] = "EUR"
    elif change == "pdf_hash": source["pdf_sha256"] = "c" * 64
    elif change == "issuer_url": source["issuer_url"] = "file:///C:/private.pdf"
    elif change == "false_complete": source["bindings"]["spans"]["date"] = None
    elif change == "cell_id": source["cell_id"] = "cell:" + H
    elif change == "exposed_heldout": row["exposure"]["split"] = "heldout"
    elif change == "human": row["review"].update(human_acceptance=True, human_verdict="passed")
    elif change == "unknown": row["run_model"] = True
    elif change == "unknown_nested": row["request"]["expected_answer"] = "44062"
    elif change == "unknown_scope": row["scope_ids"] = ["missing"]
    elif change == "empty_scopes": row["scope_ids"] = []
    elif change == "unknown_cell": row["expected"]["cell_ids"] = ["cell:" + H]
    elif change == "bad_timestamp": row["exposure"]["first_exposed_at_utc"] = "2026-10-12"
    else: source["normalization_status"] = "verified-usd-by-dollar-symbol"
    with pytest.raises(ValueError):
        load(write_inputs(tmp_path, [row]))


def test_changed_original_record_byte_cannot_be_migrated(tmp_path):
    row = case()
    raw = json.dumps({"question_id": row["case_id"], "query": row["query"]}) + "\n"
    row["origin"] = {"kind": "migrated", "original_record_utf8": raw,
                     "original_record_sha256": sha256(raw.encode()).hexdigest()}
    assert load(write_inputs(tmp_path, [row])).cases[0].origin["original_record_utf8"] == raw
    row["origin"]["original_record_utf8"] += " "
    with pytest.raises(ValueError, match="original"):
        load(write_inputs(tmp_path, [row]))


@pytest.mark.parametrize("kind", ["duplicate_key", "nan", "infinity", "blank_line", "bom"])
def test_strict_jsonl_bytes(tmp_path, kind):
    row = case()
    raw = json.dumps(row)
    if kind == "duplicate_key": raw = raw.replace('"case_id": "positive"', '"case_id": "positive", "case_id": "other"')
    elif kind == "nan": raw = raw.replace('"physical_page": 3', '"physical_page": NaN')
    elif kind == "infinity": raw = raw.replace('"physical_page": 3', '"physical_page": Infinity')
    elif kind == "blank_line": raw += "\n"
    else: raw = "\ufeff" + raw
    with pytest.raises(ValueError):
        load(write_inputs(tmp_path, [row], raw=(raw + "\n").encode()))


def test_duplicate_case_ids_and_changed_negative_denominator_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        load(write_inputs(tmp_path, [case(), case()]))
    with pytest.raises(ValueError, match="count"):
        load(write_inputs(tmp_path, freeze_edit=lambda f: f["counts"].update(negative=1)))


@pytest.mark.parametrize("kind", ["empty_scope", "unregistered_scope", "duplicate_scope", "boolean_count",
    "nan_threshold", "changed_primary", "empty_grid", "unknown_input", "unknown_field", "budget"])
def test_experiment_preflight_controls(tmp_path, kind):
    cfg = experiment()
    if kind == "empty_scope": cfg["scopes"][1]["document_ids"] = []
    elif kind == "unregistered_scope": cfg["scopes"][1]["document_ids"] = ["missing"]
    elif kind == "duplicate_scope": cfg["scopes"][1]["id"] = "all-reports"
    elif kind == "boolean_count": cfg["document_count"] = True
    elif kind == "nan_threshold": cfg["primary_threshold"] = float("nan")
    elif kind == "changed_primary": cfg["primary_threshold"] = 0.65
    elif kind == "empty_grid": cfg["threshold_grid"] = []
    elif kind == "unknown_input": cfg["input_sha256"]["private_pdf"] = H
    elif kind == "unknown_field": cfg["execute_generation"] = True
    else: cfg["limits"]["max_case_file_bytes"] = 2 ** 60
    paths = write_inputs(tmp_path, config=cfg)
    with pytest.raises(ValueError):
        load(paths)


def test_case_and_freeze_exact_byte_identities_and_order_are_bound(tmp_path):
    paths = write_inputs(tmp_path)
    module = api()
    cfg = module.load_experiment(paths[2])
    paths[0].write_bytes(paths[0].read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA"):
        module.load_case_bundle(paths[0], paths[1], cfg)
    paths = write_inputs(tmp_path)
    cfg = module.load_experiment(paths[2])
    paths[1].write_bytes(paths[1].read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA"):
        module.load_case_bundle(paths[0], paths[1], cfg)


def test_loader_has_no_model_pdf_or_runtime_imports_and_requires_absolute_paths(tmp_path):
    assert SCRIPT.exists(), "strict expanded case loader must be implemented"
    code = "import runpy,sys; runpy.run_path(sys.argv[1]); assert not ({'torch','numpy','pymupdf','sentence_transformers','nvidia_rag'} & set(sys.modules))"
    result = subprocess.run([sys.executable, "-I", "-c", code, str(SCRIPT)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    with pytest.raises(ValueError, match="absolute"):
        api().load_experiment(Path("relative.json"))


def test_oversized_case_file_refused_before_json_decode(tmp_path):
    cfg = experiment()
    cfg["limits"]["max_case_file_bytes"] = 8
    with pytest.raises(ValueError, match="budget"):
        load(write_inputs(tmp_path, config=cfg))


def test_negative_rationale_cannot_reference_an_unregistered_receipt(tmp_path):
    row = case()
    row["expected"] = {"outcome": "refuse", "cell_ids": [], "negative_reason": "unknown capability",
                       "rationale_binding_ids": ["missing:receipt"]}
    with pytest.raises(ValueError, match="rationale"):
        load(write_inputs(tmp_path, [row]))


def test_freeze_boolean_cannot_be_integer(tmp_path):
    with pytest.raises(ValueError, match="status"):
        load(write_inputs(tmp_path, freeze_edit=lambda f: f["case_status"][0].update(development_exposed=1)))


@pytest.mark.parametrize("field", ["unit_kind", "metric", "period_end", "scale"])
def test_unknown_nested_request_types_are_finite_validation_errors(tmp_path, field):
    row = case()
    row["request"][field] = {"unexpected": "object"}
    with pytest.raises(ValueError):
        load(write_inputs(tmp_path, [row]))


def test_immutable_query_records_have_no_mutable_instance_dictionary(tmp_path):
    row = load(write_inputs(tmp_path)).retrieval_inputs()[0]
    assert not hasattr(row, "__dict__")


def test_declared_known_eps_capability_cannot_be_monetary(tmp_path):
    cfg = experiment()
    cfg["metric_unit_kinds"]["basic_eps"] = "monetary"
    with pytest.raises(ValueError, match="metric"):
        load(write_inputs(tmp_path, config=cfg))


def test_escaped_lone_surrogate_is_rejected_before_query_encoding(tmp_path):
    row = case()
    row["query"] = "invalid\ud800question"
    # The escaped JSON itself is valid UTF-8; only the decoded question is not.
    raw = (json.dumps(row, ensure_ascii=True) + "\n").encode("utf-8")
    with pytest.raises(ValueError, match="query"):
        load(write_inputs(tmp_path, [row], raw=raw))


def test_overflowing_json_number_in_original_provenance_is_rejected(tmp_path):
    row = case()
    original = json.dumps({"question_id": row["case_id"], "query": row["query"]})
    raw = original[:-1] + ',"legacy_numeric_metadata":1e999}'
    row["origin"] = {"kind": "migrated", "original_record_utf8": raw,
                     "original_record_sha256": sha256(raw.encode("utf-8")).hexdigest()}
    with pytest.raises(ValueError, match="non-finite"):
        load(write_inputs(tmp_path, [row]))


def test_conflicting_legacy_case_ids_cannot_claim_migrated_provenance(tmp_path):
    row = case()
    raw = json.dumps({"case_id": row["case_id"], "question_id": "a-different-record",
                      "query": row["query"]})
    row["origin"] = {"kind": "migrated", "original_record_utf8": raw,
                     "original_record_sha256": sha256(raw.encode("utf-8")).hexdigest()}
    with pytest.raises(ValueError, match="original"):
        load(write_inputs(tmp_path, [row]))


@pytest.mark.parametrize("coordinate", ["page_text_sha256", "extraction_profile"])
def test_source_cell_id_binds_the_coordinate_system_of_its_text_offsets(coordinate):
    source = cell()
    changed = deepcopy(source)
    if coordinate == "page_text_sha256":
        changed["bindings"][coordinate] = "c" * 64
    else:
        changed["bindings"][coordinate]["sort"] = False
    assert api().source_cell_id(source) != api().source_cell_id(changed)


def test_cell_identity_can_be_recomputed_from_an_immutable_loaded_record(tmp_path):
    bundle = load(write_inputs(tmp_path))
    source = bundle.cases[0].source_cells[0]
    assert api().source_cell_id(source) == source["cell_id"]
