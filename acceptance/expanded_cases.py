"""Strict development-case and experiment contracts for future comparisons.

Only standard-library parsing runs here. Source bindings are declared candidates:
no PDF is opened, offset replay performed, corpus loaded or model queried. A
valid bundle cannot establish field coverage, human gold or financial accuracy.
All errors describe contract fields rather than echoing private input contents.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlsplit

INPUTS = frozenset({"registry", "corpus_manifest", "corpus_chunks", "vector_manifest",
                   "vector_matrix", "cases", "case_freeze", "source_replay", "wheel"})
COUNT_FIELDS = frozenset({"cases", "evidence", "refuse", "clarify", "negative",
                          "development", "heldout", "human_gold"})
UNITS = frozenset({"monetary", "per_share", "percent", "count"})
SPAN_NAMES = frozenset({"statement", "unit", "currency", "duration", "date", "row", "value"})
REQUEST_FIELDS = frozenset({"report_document_id", "report_fiscal_year", "report_fiscal_quarter",
    "metric", "period_kind", "period_end", "duration_months", "currency", "unit_kind", "scale"})
LIMITS = {"max_cases": 256, "max_cells_per_case": 64,
          "max_source_text_chars": 1_000_000, "max_case_file_bytes": 16_000_000}
CANDIDATE = "assistant-source-reviewed-candidate"
HEX = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value, fields, label):
    require(type(value) is dict and value.keys() == fields, f"{label} fields differ from schema")
    return value


def _string(value, label, maximum=2000):
    require(type(value) is str and 0 < len(value) <= maximum and value.strip()
            and not any((ord(c) < 32 and c not in "\t\n\r") or 0xD800 <= ord(c) <= 0xDFFF
                        for c in value), f"invalid {label} string")
    return value


def _id(value, label):
    require(type(value) is str and IDENTIFIER.fullmatch(value) is not None, f"invalid {label} ID")
    return value


def _int(value, label, minimum=1, maximum=10_000_000):
    require(type(value) is int and minimum <= value <= maximum, f"invalid {label} integer")
    return value


def _hash(value, label):
    require(type(value) is str and HEX.fullmatch(value) is not None, f"invalid {label} SHA-256")
    return value


def _date(value, label):
    require(type(value) is str and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None,
            f"invalid {label} ISO date")
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"invalid {label} ISO date") from None


def _utc(value, label):
    require(type(value) is str and re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)", value) is not None,
        f"invalid {label} UTC timestamp")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(stamp.utcoffset() == timezone.utc.utcoffset(stamp), f"invalid {label} UTC timestamp")
    except ValueError:
        raise ValueError(f"invalid {label} UTC timestamp") from None
    return stamp


def _decimal(value, label, positive=False):
    require(type(value) is str and len(value) <= 64 and DECIMAL.fullmatch(value) is not None,
            f"invalid {label} exact Decimal string")
    number = Decimal(value)
    require(number.is_finite() and len(number.as_tuple().digits) <= 28
            and (not positive or number > 0), f"invalid {label} exact Decimal value")
    return number


def _url(value):
    _string(value, "issuer URL", 2048)
    try:
        parts = urlsplit(value)
        require(parts.scheme == "https" and parts.hostname is not None
                and parts.username is None and parts.password is None and not parts.fragment
                and parts.port in (None, 443), "issuer URL must be credential-free HTTPS")
    except ValueError:
        raise ValueError("issuer URL must be credential-free HTTPS") from None


def _list(value, label, maximum=256, empty=False, identifiers=True):
    require(type(value) is list and (empty or bool(value)) and len(value) <= maximum,
            f"invalid {label} list")
    for entry in value:
        (_id if identifiers else _string)(entry, label)
    require(len(set(value)) == len(value), f"duplicate {label} entries")
    return tuple(value)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _depth(value, level=0):
    require(level <= 24, "JSON nesting exceeds budget")
    if type(value) is dict:
        require(len(value) <= 1024, "JSON object exceeds budget")
        for item in value.values():
            _depth(item, level + 1)
    elif type(value) is list:
        require(len(value) <= 1024, "JSON array exceeds budget")
        for item in value:
            _depth(item, level + 1)
    elif type(value) is float:
        require(math.isfinite(value), "non-finite JSON number")


def _json(body):
    def reject_constant(_):
        raise ValueError("non-finite JSON number")
    try:
        text = body.decode("utf-8") if type(body) is bytes else body
        require(not text.startswith("\ufeff"), "UTF-8 BOM is unsupported")
        result = json.loads(text, object_pairs_hook=_pairs, parse_constant=reject_constant)
        _depth(result)
        return result
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("invalid bounded UTF-8 JSON") from None


def _read(path: Path, limit: int, label: str) -> bytes:
    require(isinstance(path, Path) and path.is_absolute(), f"{label} must be an absolute local path")
    require(path.is_file() and not path.is_symlink()
            and not any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction())
                        for p in path.parents), f"{label} must be an unlinked local file")
    require(path.stat().st_size <= limit, f"{label} exceeds byte budget")
    with path.open("rb") as handle:
        body = handle.read(limit + 1)
    require(len(body) <= limit, f"{label} exceeds byte budget")
    return body


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class Experiment:
    sha256: str
    data: Mapping


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    query: str
    origin: Mapping
    exposure: Mapping
    review: Mapping
    request: Mapping
    expected: Mapping
    scope_ids: tuple[str, ...]
    source_cells: tuple[Mapping, ...]


@dataclass(frozen=True, slots=True)
class RetrievalInput:
    """Only operator scope and actual question reach the ranking path."""
    case_id: str
    query: str
    scope_id: str
    document_ids: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class CaseBundle:
    cases_sha256: str
    freeze_sha256: str
    experiment: Experiment
    cases: tuple[Case, ...]
    counts: Mapping
    human_gold_count: int = 0
    source_binding_passed: bool = False
    joint_source_candidate_coverage: str = "not_evaluated"

    def retrieval_inputs(self) -> tuple[RetrievalInput, ...]:
        scopes = {row["id"]: row["document_ids"] for row in self.experiment.data["scopes"]}
        return tuple(RetrievalInput(case.case_id, case.query, scope, scopes[scope])
                     for case in self.cases for scope in case.scope_ids)


def _counts(value):
    _object(value, COUNT_FIELDS, "count")
    for key, number in value.items():
        _int(number, "count " + key, 0, LIMITS["max_cases"])
    require(value["cases"] > 0 and value["cases"] == value["evidence"] + value["refuse"] + value["clarify"]
            and value["negative"] == value["refuse"] + value["clarify"]
            and value["development"] == value["cases"] and value["heldout"] == value["human_gold"] == 0,
            "count denominator or development evidence boundary changed")


def load_experiment(path: Path) -> Experiment:
    """Load finite configuration metadata; external bodies need later replay."""
    body = _read(path, 512_000, "experiment")
    cfg = _json(body)
    _object(cfg, {"schema_version", "run_kind", "input_sha256", "documents", "document_count",
        "chunk_count", "expected_case_ids", "case_counts", "model", "query_normalization", "top_k",
        "primary_threshold", "threshold_grid", "supported_currencies", "supported_unit_kinds",
        "metric_unit_kinds", "rationale_bindings", "scopes", "limits"}, "experiment")
    require(type(cfg["schema_version"]) is int and cfg["schema_version"] == 2, "unsupported experiment schema")
    require(cfg["run_kind"] in ("diagnostic", "development-calibration"), "invalid run kind")
    _object(cfg["input_sha256"], INPUTS, "input identity")
    for key, value in cfg["input_sha256"].items():
        _hash(value, key)
    _object(cfg["limits"], LIMITS.keys(), "limits")
    for key, maximum in LIMITS.items():
        _int(cfg["limits"][key], "budget " + key, 1, maximum)
    ids = _list(cfg["expected_case_ids"], "case IDs", cfg["limits"]["max_cases"])
    _counts(cfg["case_counts"])
    require(cfg["case_counts"]["cases"] == len(ids), "configured case count differs from ordered IDs")
    require(type(cfg["documents"]) is list and 0 < len(cfg["documents"]) <= 1000,
            "invalid registered document list")
    document_ids = []
    for doc in cfg["documents"]:
        _object(doc, {"document_id", "report_fiscal_year", "report_fiscal_quarter", "pdf_sha256",
                      "issuer_url", "page_count"}, "document")
        document_ids.append(_id(doc["document_id"], "document"))
        _int(doc["report_fiscal_year"], "report fiscal year", 1900, 9999)
        if doc["report_fiscal_quarter"] is not None:
            _int(doc["report_fiscal_quarter"], "report fiscal quarter", 1, 4)
        _hash(doc["pdf_sha256"], "PDF")
        _url(doc["issuer_url"])
        _int(doc["page_count"], "physical page count", 1, 10000)
    _int(cfg["document_count"], "document count", 1, 1000)
    require(len(document_ids) == len(set(document_ids)) == cfg["document_count"], "document count/IDs mismatch")
    _int(cfg["chunk_count"], "chunk count", 1, 10_000_000)
    currencies = _list(cfg["supported_currencies"], "currency", 32)
    require(all(re.fullmatch(r"[A-Z]{3}", item) for item in currencies), "invalid currency code")
    units = _list(cfg["supported_unit_kinds"], "unit", 4)
    require(set(units) <= UNITS, "unsupported unit kind")
    require(type(cfg["metric_unit_kinds"]) is dict and 0 < len(cfg["metric_unit_kinds"]) <= 128,
            "invalid metric capability map")
    for key, value in cfg["metric_unit_kinds"].items():
        _id(key, "metric")
        require(value in units, "metric unit is not a declared capability")
        known = {"revenue": "monetary", "net_income": "monetary",
                 "basic_eps": "per_share", "diluted_eps": "per_share"}
        require(key not in known or value == known[key], "known metric unit capability differs")
    require(type(cfg["rationale_bindings"]) is list and len(cfg["rationale_bindings"]) <= 256,
            "rationale bindings exceed budget")
    rationale_ids = []
    for binding in cfg["rationale_bindings"]:
        _object(binding, {"binding_id", "kind", "receipt_sha256"}, "rationale binding")
        rationale_ids.append(_id(binding["binding_id"], "rationale binding"))
        require(binding["kind"] in ("source-column-inventory", "registry-absence", "experiment-capability",
                                     "request-contradiction"), "invalid rationale binding kind")
        _hash(binding["receipt_sha256"], "rationale receipt")
    require(len(rationale_ids) == len(set(rationale_ids)), "duplicate rationale binding IDs")
    model = _object(cfg["model"], {"name", "revision", "instruction", "device", "dimension",
                                  "raw_dtype", "normalized_dtype"}, "model")
    require(model["name"] == "BAAI/bge-small-en-v1.5" and type(model["revision"]) is str
            and re.fullmatch(r"[0-9a-f]{40}", model["revision"]), "invalid pinned BGE model identity")
    _string(model["instruction"], "query instruction", 500)
    require(model["device"] == "cpu" and type(model["dimension"]) is int and model["dimension"] == 384
            and model["raw_dtype"] == "float32" and model["normalized_dtype"] == "float64",
            "model arithmetic profile changed")
    require(cfg["query_normalization"] == "strip" and type(cfg["top_k"]) is int and cfg["top_k"] == 5,
            "initial diagnostic query/Top-K contract changed")
    threshold = cfg["primary_threshold"]
    require(type(threshold) in (int, float) and threshold == 0.78 and math.isfinite(threshold),
            "initial diagnostic primary threshold changed")
    grid = cfg["threshold_grid"]
    require(type(grid) is list and 0 < len(grid) <= 21
            and all(type(v) in (int, float) and 0 <= v <= 1 and math.isfinite(v) for v in grid)
            and len(set(grid)) == len(grid) and threshold in grid, "invalid finite threshold grid")
    require(cfg["run_kind"] != "diagnostic" or grid == [0.78], "diagnostic grid changed before calibration")
    require(type(cfg["scopes"]) is list and 0 < len(cfg["scopes"]) <= 64, "invalid scope list")
    scope_ids = []
    for scope in cfg["scopes"]:
        _object(scope, {"id", "document_ids"}, "scope")
        scope_ids.append(_id(scope["id"], "scope"))
        if scope["document_ids"] is None:
            require(scope["id"] == "all-reports", "only all-reports has global scope")
        else:
            selected = _list(scope["document_ids"], "scope document", 1000)
            require(scope["id"] != "all-reports" and set(selected) <= set(document_ids),
                    "scope contains unregistered documents")
    require(len(scope_ids) == len(set(scope_ids)) and "all-reports" in scope_ids, "duplicate/missing global scope")
    return Experiment(sha256(body).hexdigest(), _freeze(cfg))


def _request(req):
    _object(req, REQUEST_FIELDS, "request")
    for key in ("report_document_id", "metric"):
        if req[key] is not None:
            _id(req[key], key)
    for key, low, high in (("report_fiscal_year", 1900, 9999), ("report_fiscal_quarter", 1, 4),
                           ("duration_months", 1, 120)):
        if req[key] is not None:
            _int(req[key], key, low, high)
    require(req["period_kind"] in (None, "duration", "instant"), "invalid request period kind")
    if req["period_end"] is not None:
        _date(req["period_end"], "request period end")
    if req["currency"] is not None:
        require(type(req["currency"]) is str and re.fullmatch(r"[A-Z]{3}", req["currency"]), "invalid request currency")
    require(req["unit_kind"] is None or (type(req["unit_kind"]) is str and req["unit_kind"] in UNITS),
            "invalid request unit kind")
    if req["scale"] is not None:
        _decimal(req["scale"], "request scale", positive=True)
    # Absent/contradictory qualifiers and unsupported currencies belong in the
    # measured negative set. They do not configure a production retrieval veto.


def _bbox(box):
    require(type(box) is list and len(box) == 4
            and all(type(n) in (int, float) and abs(n) <= 100000 and math.isfinite(n) for n in box)
            and box[0] < box[2] and box[1] < box[3], "invalid finite nonzero source box")


def source_cell_id(cell: Mapping) -> str:
    """Bind a selected row/column and the coordinate system of its location."""
    identity = {key: cell[key] for key in ("document_id", "pdf_sha256", "physical_page", "statement",
        "metric", "row", "column_role", "period_kind", "period_end", "duration_months")}
    binding = cell["bindings"]
    identity["text_coordinates"] = {"page_text_sha256": binding["page_text_sha256"],
                                    "extraction_profile": dict(binding["extraction_profile"])}
    value = binding["spans"]["value"]
    identity["value_location"] = None if value is None else {key: value[key] for key in ("start", "end", "bbox")}
    return "cell:" + sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _cell(cell, cfg):
    _object(cell, {"cell_id", "document_id", "pdf_sha256", "issuer_url", "physical_page", "printed_page",
        "report_fiscal_year", "report_fiscal_quarter", "statement", "metric", "row", "column_role",
        "period_kind", "period_end", "duration_months", "currency_literal", "unit_literal", "currency",
        "unit_kind", "scale", "normalization_status", "amount", "value_text", "bindings"}, "source cell")
    _id(cell["document_id"], "source document")
    docs = {row["document_id"]: row for row in cfg["documents"]}
    require(cell["document_id"] in docs, "source document is not registered")
    doc = docs[cell["document_id"]]
    for key in ("pdf_sha256", "issuer_url", "report_fiscal_year", "report_fiscal_quarter"):
        require(type(cell[key]) is type(doc[key]) and cell[key] == doc[key], "registered source identity differs")
    _hash(cell["pdf_sha256"], "source PDF")
    _url(cell["issuer_url"])
    _int(cell["physical_page"], "physical page", 1, doc["page_count"])
    if cell["printed_page"] is not None:
        _string(cell["printed_page"], "printed page", 64)
    for key in ("statement", "row", "unit_literal", "value_text"):
        _string(cell[key], "source " + key, 1000)
    _id(cell["metric"], "source metric")
    require(cell["metric"] in cfg["metric_unit_kinds"] and cell["unit_kind"] == cfg["metric_unit_kinds"][cell["metric"]],
            "source metric/unit capability mismatch")
    require(cell["column_role"] in ("current", "prior_comparison", "annual_comparison"), "invalid source column role")
    require(cell["period_kind"] in ("duration", "instant"), "source period kind is missing")
    _date(cell["period_end"], "source period end")
    if cell["period_kind"] == "duration":
        _int(cell["duration_months"], "source duration", 1, 120)
    else:
        require(cell["duration_months"] is None, "instant source facts cannot have duration")
    if cell["unit_kind"] in ("monetary", "per_share"):
        require(cell["currency"] in cfg["supported_currencies"], "source currency is not a declared capability")
        _string(cell["currency_literal"], "printed currency", 64)
    else:
        require(cell["currency"] is None and cell["currency_literal"] is None,
                "nonmonetary source unit cannot become currency")
    scale = _decimal(cell["scale"], "source scale", positive=True)
    require(cell["unit_kind"] != "per_share" or scale == 1, "per-share source scale must be one")
    require(cell["normalization_status"] == CANDIDATE, "source normalization must remain provisional")
    amount = _decimal(cell["amount"], "source amount")
    literal = cell["value_text"]
    if cell["currency_literal"] and literal.startswith(cell["currency_literal"]):
        literal = literal[len(cell["currency_literal"]):].strip()
    if cell["unit_kind"] == "percent" and literal.endswith("%"):
        literal = literal[:-1]
    if literal.startswith("(") and literal.endswith(")"):
        literal = "-" + literal[1:-1]
    require(re.fullmatch(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", literal) is not None,
            "source printed amount is not a supported exact literal")
    require(Decimal(literal.replace(",", "")) == amount, "source amount differs from printed literal")
    binding = _object(cell["bindings"], {"status", "page_text_sha256", "page_text_chars",
        "extraction_profile", "spans", "association_receipt_sha256"}, "source bindings")
    _hash(binding["page_text_sha256"], "source page text")
    _int(binding["page_text_chars"], "source page text length", 1, cfg["limits"]["max_source_text_chars"])
    profile = _object(binding["extraction_profile"], {"library", "version", "mode", "sort", "normalization",
        "text_encoding", "offset_unit", "interval"}, "extraction profile")
    require(profile["library"] == "PyMuPDF" and type(profile["version"]) is str
            and re.fullmatch(r"\d+\.\d+\.\d+", profile["version"])
            and profile["mode"] == "text" and type(profile["sort"]) is bool
            and profile["normalization"] in ("none", "strip") and profile["text_encoding"] == "utf-8"
            and profile["offset_unit"] == "unicode_code_points" and profile["interval"] == "half_open",
            "source extraction profile is incomplete or unsupported")
    spans = _object(binding["spans"], SPAN_NAMES, "source spans")
    for name, span in spans.items():
        if span is None:
            continue
        _object(span, {"start", "end", "text", "bbox"}, "source span")
        start = _int(span["start"], "span start", 0, binding["page_text_chars"])
        end = _int(span["end"], "span end", 1, binding["page_text_chars"])
        _string(span["text"], "span text", cfg["limits"]["max_source_text_chars"])
        require(start < end and end - start == len(span["text"]), "source span offset/text length differs")
        if span["bbox"] is not None:
            _bbox(span["bbox"])
    expected = {"statement": cell["statement"], "unit": cell["unit_literal"],
                "currency": cell["currency_literal"], "row": cell["row"], "value": cell["value_text"]}
    for name, text in expected.items():
        require(spans[name] is None or spans[name]["text"] == text, "source span field text differs")
    require(spans["value"] is not None, "selected amount location is required even for incomplete candidates")
    receipt = binding["association_receipt_sha256"]
    if receipt is not None:
        _hash(receipt, "geometry association receipt")
    essential = {"statement", "unit", "date", "row", "value"}
    if cell["unit_kind"] in ("monetary", "per_share"):
        essential.add("currency")
    if cell["period_kind"] == "duration":
        essential.add("duration")
    else:
        require(spans["duration"] is None, "instant facts cannot bind duration headers")
    complete = receipt is not None and all(spans[name] is not None and spans[name]["bbox"] is not None
                                         for name in essential)
    require(binding["status"] == ("candidate-complete" if complete else "incomplete"),
            "declared source binding completeness differs from structure")
    require(cell["cell_id"] == source_cell_id(cell), "source cell ID does not bind row/column/location")
    # Structural completeness never asserts matching live page bytes/geometry.
    return _freeze({**cell, "binding_complete": complete})


def _case(row, cfg):
    _object(row, {"case_id", "query", "origin", "exposure", "review", "request", "expected",
                  "scope_ids", "source_cells"}, "case")
    _id(row["case_id"], "case")
    _string(row["query"], "query", 500)
    origin = _object(row["origin"], {"kind", "original_record_utf8", "original_record_sha256"}, "origin")
    require(origin["kind"] in ("migrated", "new-candidate"), "invalid case origin")
    if origin["kind"] == "migrated":
        raw = _string(origin["original_record_utf8"], "original record", 128000)
        _hash(origin["original_record_sha256"], "original record")
        require(sha256(raw.encode("utf-8")).hexdigest() == origin["original_record_sha256"],
                "original record SHA-256 differs")
        original = _json(raw)
        require(type(original) is dict, "migrated original record must be an object")
        original_ids = [original[key] for key in ("case_id", "question_id") if key in original]
        require(original.get("query") == row["query"]
                and original_ids and all(type(value) is str and value == row["case_id"]
                                         for value in original_ids),
                "migrated original query/ID changed")
    else:
        require(origin["original_record_utf8"] is None and origin["original_record_sha256"] is None,
                "new candidate cannot claim original record provenance")
    exposure = _object(row["exposure"], {"split", "development_exposed", "first_exposed_at_utc", "freeze_purpose"}, "exposure")
    require(exposure["split"] == "development" and exposure["development_exposed"] is True
            and exposure["freeze_purpose"] == "development-replay", "exposed cases must remain development material")
    if exposure["first_exposed_at_utc"] is not None:
        _utc(exposure["first_exposed_at_utc"], "first exposure")
    review = _object(row["review"], {"status", "reviewer", "human_acceptance", "human_verdict",
                                     "adjudication_receipt_sha256"}, "review")
    _string(review["reviewer"], "reviewer", 128)
    require(review["status"] == CANDIDATE and review["human_acceptance"] is False
            and review["human_verdict"] is None and review["adjudication_receipt_sha256"] is None,
            "human adjudication requires a separate verified ledger; human gold is zero")
    _request(row["request"])
    expected = _object(row["expected"], {"outcome", "cell_ids", "negative_reason", "rationale_binding_ids"}, "expected label")
    require(expected["outcome"] in ("evidence", "refuse", "clarify"), "invalid provisional outcome")
    cell_ids = _list(expected["cell_ids"], "expected source cell", 64, empty=True)
    rationale = _list(expected["rationale_binding_ids"], "rationale binding", 64, empty=True)
    require(set(rationale) <= {binding["binding_id"] for binding in cfg["rationale_bindings"]},
            "expected rationale refers to an unregistered receipt")
    scopes = _list(row["scope_ids"], "case scope", 64)
    registered = {s["id"]: s["document_ids"] for s in cfg["scopes"]}
    require(set(scopes) <= registered.keys(), "case requests an unknown scope")
    requested_report = row['request']['report_document_id']
    if requested_report in {d['document_id'] for d in cfg['documents']}:
        require(all(registered[scope] is None or requested_report in registered[scope] for scope in scopes),
                'case scope excludes the requested registered report')
    require(type(row["source_cells"]) is list and len(row["source_cells"]) <= cfg["limits"]["max_cells_per_case"],
            "source cells exceed configured budget")
    cells = tuple(_cell(source, cfg) for source in row["source_cells"])
    by_id = {source["cell_id"]: source for source in cells}
    require(len(by_id) == len(cells) and set(cell_ids) <= by_id.keys(), "duplicate/unknown source cell reference")
    if expected["outcome"] == "evidence":
        require(cell_ids and expected["negative_reason"] is None, "positive candidate requires source cells")
        req = row["request"]
        require(all(req[key] is not None for key in REQUEST_FIELDS - {"report_fiscal_quarter", "duration_months", "currency"}),
                "positive request is missing financial dimensions")
        for cell_id in cell_ids:
            source = by_id[cell_id]
            for key in REQUEST_FIELDS:
                source_key = "document_id" if key == "report_document_id" else key
                require(req[key] == source[source_key], "positive request/source financial tuple differs")
    else:
        require(not cell_ids and rationale, "negative candidate requires capability rationale without positive cell labels")
        _string(expected["negative_reason"], "negative reason", 2000)
    if row["request"]["report_document_id"] not in {d["document_id"] for d in cfg["documents"]}:
        require(scopes == ("all-reports",), "unavailable report requires global diagnostic scope")
    return Case(row["case_id"], row["query"], _freeze(origin), _freeze(exposure), _freeze(review),
                _freeze(row["request"]), _freeze(expected), scopes, cells)


def load_case_bundle(cases_path: Path, freeze_path: Path, experiment: Experiment) -> CaseBundle:
    """Freeze exact bytes/order/counts without passing labels to retrieval."""
    require(type(experiment) is Experiment, "validated experiment is required")
    cfg = experiment.data
    body = _read(cases_path, cfg["limits"]["max_case_file_bytes"], "case file")
    frozen = _read(freeze_path, 512_000, "case freeze")
    case_hash, freeze_hash = sha256(body).hexdigest(), sha256(frozen).hexdigest()
    require(case_hash == cfg["input_sha256"]["cases"] and freeze_hash == cfg["input_sha256"]["case_freeze"],
            "case/freeze SHA-256 differs from experiment")
    require(body.endswith(b"\n") and b"\r" not in body and len(body.splitlines()) <= cfg["limits"]["max_cases"],
            "case JSONL newline/count budget differs")
    lines = body.split(b"\n")[:-1]
    require(lines and all(line.strip() and len(line) <= 256_000 for line in lines), "case JSONL record exceeds budget or is blank")
    cases = tuple(_case(_json(line), cfg) for line in lines)
    ids = tuple(case.case_id for case in cases)
    require(len(ids) == len(set(ids)) and ids == cfg["expected_case_ids"], "case IDs/order differ from frozen experiment")
    freeze = _object(_json(frozen), {"schema_version", "cases_sha256", "ordered_case_ids", "counts",
        "frozen_at_utc", "freeze_purpose", "case_status"}, "case freeze")
    require(type(freeze["schema_version"]) is int and freeze["schema_version"] == 2
            and freeze["cases_sha256"] == case_hash and freeze["freeze_purpose"] == "development-replay",
            "case freeze schema/identity/purpose differs")
    require(tuple(_list(freeze["ordered_case_ids"], "frozen case IDs", cfg["limits"]["max_cases"])) == ids,
            "frozen case IDs/order differs")
    freeze_time = _utc(freeze["frozen_at_utc"], "case freeze")
    _counts(freeze["counts"])
    outcomes = [case.expected["outcome"] for case in cases]
    counts = {"cases": len(cases), "evidence": outcomes.count("evidence"), "refuse": outcomes.count("refuse"),
              "clarify": outcomes.count("clarify"), "negative": len(cases) - outcomes.count("evidence"),
              "development": len(cases), "heldout": 0, "human_gold": 0}
    require(freeze["counts"] == dict(cfg["case_counts"]) == counts, "frozen count denominators differ from cases")
    require(type(freeze["case_status"]) is list and len(freeze["case_status"]) == len(cases), "frozen status count differs")
    for case, status in zip(cases, freeze["case_status"]):
        _object(status, {"case_id", "outcome", "review_status", "development_exposed", "original_record_sha256"}, "frozen case status")
        require(type(status["development_exposed"]) is bool, "frozen case status exposure must be boolean")
        require(status == {"case_id": case.case_id, "outcome": case.expected["outcome"],
            "review_status": case.review["status"], "development_exposed": True,
            "original_record_sha256": case.origin["original_record_sha256"]}, "frozen case status differs")
        if case.exposure["first_exposed_at_utc"] is not None:
            require(_utc(case.exposure["first_exposed_at_utc"], "first exposure") <= freeze_time,
                    "freeze precedes recorded development exposure")
    return CaseBundle(case_hash, freeze_hash, experiment, cases, _freeze(counts))
