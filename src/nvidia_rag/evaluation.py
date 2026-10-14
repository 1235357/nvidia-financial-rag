"""Ground-truth evaluation for retrieval and evidence-gate behavior."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
import math
from time import perf_counter
from typing import Protocol, Sequence

from .core import ScoredChunk, build_grounded_context, select_evidence
from .prompt import build_grounded_prompt


def summarize_judgment_coverage(
    retrieved_ids: Sequence[str], judgments: dict[str, int]
) -> dict[str, object]:
    """Audit explicit per-query relevance judgments, not answer correctness.

    IDs must refer to the same assessment unit (documents, pages, or chunks).
    Zero is explicitly irrelevant; positive grades are relevant; absent keys
    are unjudged. Deduplicate retrieval IDs so repeated chunks cannot inflate
    coverage when the assessment unit is a document or page. No gold-page
    labels are automatically converted into relevance judgments.
    """
    def valid_id(value):
        return isinstance(value, str) and bool(value.strip())
    if isinstance(retrieved_ids, (str, bytes)) or any(not valid_id(value) for value in retrieved_ids):
        raise ValueError("retrieved_ids must contain nonblank string IDs")
    if not isinstance(judgments, dict) or any(
        not valid_id(key) or type(grade) is not int or grade < 0
        for key, grade in judgments.items()
    ):
        raise ValueError("judgments require nonblank IDs and nonnegative integer grades")
    unique_ids = list(dict.fromkeys(retrieved_ids))
    unjudged = [key for key in unique_ids if key not in judgments]
    relevant = sum(judgments[key] > 0 for key in unique_ids if key in judgments)
    irrelevant = sum(judgments[key] == 0 for key in unique_ids if key in judgments)
    count = len(unique_ids)
    return {
        "scope": "explicit relevance judgment coverage; not retrieval quality or answer truth",
        "retrieved_unique_count": count,
        "judged_relevant_count": relevant,
        "judged_irrelevant_count": irrelevant,
        "unjudged_count": len(unjudged),
        "unjudged_ratio": len(unjudged) / count if count else None,
        "judgment_coverage": (relevant + irrelevant) / count if count else None,
        "unjudged_ids": unjudged,
    }


def summarize_judgment_coverage_rows(rows: list[dict]) -> dict[str, object]:
    """Batch explicit per-query judgments, with candidate-weighted coverage.

    Assessment units are caller-supplied assertions, not verified provenance.
    Missing queries cannot be discovered from this input alone.
    """
    if not isinstance(rows, list):
        raise ValueError("coverage input must be a JSON list")
    required = {"question_id", "assessment_unit", "retrieved_ids", "judgments"}
    seen = set()
    unit = None
    summaries = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("coverage rows require exact question_id, assessment_unit, retrieved_ids, judgments fields")
        question_id = row["question_id"]
        if not isinstance(question_id, str) or not question_id.strip() or question_id in seen:
            raise ValueError("coverage question IDs must be nonblank and unique")
        seen.add(question_id)
        assessment_unit = row["assessment_unit"]
        if assessment_unit not in ("document", "page", "chunk") or (unit is not None and assessment_unit != unit):
            raise ValueError("coverage rows require one consistent assessment unit")
        unit = assessment_unit
        if not isinstance(row["retrieved_ids"], list):
            raise ValueError("retrieved_ids must be a JSON list")
        summary = summarize_judgment_coverage(row["retrieved_ids"], row["judgments"])
        summaries.append({"question_id": question_id, "assessment_unit": unit, **summary})
    total = sum(row["retrieved_unique_count"] for row in summaries)
    unjudged = sum(row["unjudged_count"] for row in summaries)
    return {
        "scope": "candidate-weighted explicit judgment coverage; not answer accuracy or independence",
        "assessment_unit": unit, "question_count": len(summaries),
        "empty_retrieval_count": sum(row["retrieved_unique_count"] == 0 for row in summaries),
        "retrieved_unique_count": total,
        "judged_relevant_count": sum(row["judged_relevant_count"] for row in summaries),
        "judged_irrelevant_count": sum(row["judged_irrelevant_count"] for row in summaries),
        "unjudged_count": unjudged,
        "judgment_coverage": (total - unjudged) / total if total else None,
        "unjudged_ratio": unjudged / total if total else None,
        "rows": summaries,
    }


def measure_serial_calls(operation, inputs: Sequence, *, repeats: int = 5,
                         warmup_cycles: int = 1, clock=perf_counter) -> dict:
    """Measure completed synchronous calls, excluding startup and warmup.

    CUDA callers must synchronize inside the operation before returning.
    Error messages and input values are intentionally not retained.
    Percentiles use nearest rank over successful calls only.
    """
    if (not inputs or type(repeats) is not int or repeats < 1
            or type(warmup_cycles) is not int or warmup_cycles < 0):
        raise ValueError('nonempty inputs, positive repeats and nonnegative warmup cycles required')
    for _ in range(warmup_cycles):
        for value in inputs:
            operation(value)  # Abort on warmup failure; do not hide it.
    samples = []
    for cycle in range(repeats):
        for input_index, value in enumerate(inputs):
            error_type = None
            start = clock()
            try:
                operation(value)
            except Exception as exc:
                error_type = type(exc).__name__
            elapsed = (clock() - start) * 1000
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('clock must produce finite nonnegative durations')
            samples.append({'cycle': cycle, 'input_index': input_index,
                            'latency_ms': elapsed, 'error_type': error_type})
    successes = sorted(row['latency_ms'] for row in samples if row['error_type'] is None)
    def percentile(fraction):
        return successes[math.ceil(fraction * len(successes)) - 1] if successes else None
    return {
        'scope': 'serial in-process calls; not concurrent or end-to-end service throughput',
        'warmup_call_count': len(inputs) * warmup_cycles,
        'attempt_count': len(samples), 'success_count': len(successes),
        'failure_count': len(samples) - len(successes),
        'percentile_method': 'nearest-rank, successful calls only',
        'success_latency_ms': {'p50': percentile(.5), 'p95': percentile(.95)},
        'samples': samples,
    }


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    """One retrieval question with page-level ground truth."""

    question_id: str
    query: str
    answerable: bool
    evidence_pages: tuple[int, ...]
    evidence_document_ids: tuple[str, ...] = ()
    expected_answer: str | None = None
    attribution_ground_truth: dict[str, str] | None = None
    financial_ground_truth: dict[str, str] | None = None
    evaluation_usage: dict[str, object] | None = None

    def __post_init__(self) -> None:
        usage = self.evaluation_usage
        if usage is not None:
            required = {"purpose", "frozen_on", "development_exposed", "registration_note"}
            if not isinstance(usage, dict) or set(usage) != required:
                raise ValueError("evaluation_usage requires exact registration fields")
            if usage["purpose"] not in ("development", "acceptance_candidate"):
                raise ValueError("evaluation_usage purpose is not an independence certificate")
            if type(usage["development_exposed"]) is not bool:
                raise ValueError("evaluation_usage development_exposed must be boolean")
            if not isinstance(usage["registration_note"], str) or not usage["registration_note"].strip():
                raise ValueError("evaluation_usage requires a registration_note")
            frozen = usage["frozen_on"]
            if frozen is not None:
                if not isinstance(frozen, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", frozen):
                    raise ValueError("evaluation_usage frozen_on must be an ISO date")
                try:
                    date.fromisoformat(frozen)
                except ValueError as exc:
                    raise ValueError("evaluation_usage frozen_on must be a real date") from exc
            if usage["purpose"] == "acceptance_candidate" and (
                frozen is None or usage["development_exposed"] is not False
            ):
                raise ValueError("evaluation_usage acceptance candidate must be frozen and unexposed")
            # Registration is a supplied assertion, not independent verification.
        fields = self.financial_ground_truth
        if fields is not None:
            required = {"document_id", "statement", "metric", "period", "unit", "value",
                        "source_locator", "reviewer", "source_check"}
            if not isinstance(fields, dict) or set(fields) != required:
                raise ValueError("financial_ground_truth requires exact financial fields and review provenance")
            if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
                raise ValueError("financial_ground_truth values must be nonblank strings")
            if self.answerable is not True:
                raise ValueError("financial_ground_truth is only allowed on answerable questions")
            # Supplied annotations are not proof of PDF truth or generated-answer correctness.
        labels = self.attribution_ground_truth
        if labels is not None:
            required = {"source_type", "actor", "claim_type", "source_locator"}
            if not isinstance(labels, dict) or set(labels) != required:
                raise ValueError("attribution requires source_type, actor, claim_type and source_locator")
            if any(not isinstance(value, str) or not value.strip() for value in labels.values()):
                raise ValueError("attribution values must be nonblank strings")
            if labels["source_type"] not in {"official_filing", "earnings_call_transcript", "third_party_analysis"}:
                raise ValueError("invalid attribution source_type")
            if labels["claim_type"] not in {"reported_result", "forecast", "question", "opinion"}:
                raise ValueError("invalid attribution claim_type")
        if type(self.answerable) is not bool:
            raise ValueError("answerable must be a boolean")
        if not self.question_id.strip():
            raise ValueError("question_id must not be blank")
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if self.answerable and not self.evidence_pages:
            raise ValueError("answerable questions require at least one evidence page")
        if not self.answerable and self.evidence_pages:
            raise ValueError("unanswerable questions must not list evidence pages")
        if any(type(page) is not int or page <= 0 for page in self.evidence_pages):
            raise ValueError("evidence pages must use positive one-based integer page numbers")
        if self.evidence_document_ids:
            if len(self.evidence_document_ids) != len(self.evidence_pages):
                raise ValueError(
                    'evidence_document_ids must align one-to-one with evidence_pages'
                )
            if any(not document_id.strip() for document_id in self.evidence_document_ids):
                raise ValueError('evidence document IDs must not be blank')
        if self.expected_answer is not None and not self.answerable:
            raise ValueError("unanswerable questions cannot have an expected answer")


def load_questions(path: Path) -> list[EvaluationQuestion]:
    """Load the same page-labeled JSONL for CLI and live teaching evaluation."""
    questions: list[EvaluationQuestion] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(
            EvaluationQuestion(
                question_id=str(row["question_id"]),
                query=str(row["query"]),
                answerable=row["answerable"],
                evidence_pages=tuple(row["evidence_pages"]),
                evidence_document_ids=tuple(
                    str(document_id)
                    for document_id in row.get("evidence_document_ids", [])
                ),
                expected_answer=row.get("expected_answer"),
                attribution_ground_truth=row.get("attribution_ground_truth"),
                financial_ground_truth=row.get("financial_ground_truth"),
                evaluation_usage=row.get("evaluation_usage"),
            )
        )
    _require_unique_question_ids(questions)
    return questions


def _require_unique_question_ids(questions: Sequence[EvaluationQuestion]) -> None:
    seen: set[str] = set()
    for question in questions:
        if question.question_id in seen:
            raise ValueError(f"duplicate question_id: {question.question_id}")
        seen.add(question.question_id)


def validate_acceptance_preflight(
    candidates: Sequence[EvaluationQuestion], reference: Sequence[EvaluationQuestion]
) -> None:
    """Reject known registration/overlap problems, not certify independence.

    Reference rows count as exposed regardless of legacy registration. Exact
    string cell identity does not discover semantic aliases or missing labels.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    _require_unique_question_ids(candidates)
    _require_unique_question_ids(reference)
    identity_fields = ("document_id", "statement", "metric", "period", "unit")
    reference_queries = {q.query for q in reference}
    reference_cells = {
        tuple(q.financial_ground_truth[field] for field in identity_fields)
        for q in reference if q.financial_ground_truth is not None
    }
    for question in candidates:
        usage = question.evaluation_usage
        if (usage is None or usage["purpose"] != "acceptance_candidate"
                or usage["development_exposed"] is not False or usage["frozen_on"] is None):
            raise ValueError(f"registered acceptance candidate required: {question.question_id}")
        if question.query in reference_queries:
            raise ValueError(f"acceptance/reference exact query overlap: {question.question_id}")
        if question.financial_ground_truth is not None:
            identity = tuple(question.financial_ground_truth[field] for field in identity_fields)
            if identity in reference_cells:
                raise ValueError(f"acceptance/reference financial cell overlap: {question.question_id}")
    if not reference:
        raise ValueError("reference must not be empty")


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]: ...


def review_context_digest(row: dict[str, object]) -> str:
    """Bind bookkeeping to this replay, not authenticate its truth or reviewer."""
    payload = {field: row.get(field) for field in (
        "question_id", "generation_status", "generated_answer", "prompt", "review_context")}
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _context_bound_review(row: dict[str, object], verdict: dict[str, object]) -> bool:
    if "review_context" not in row:
        return False  # Legacy answer-only records remain explicitly distinguishable.
    if not isinstance(row["review_context"], dict) or not row["review_context"]:
        raise ValueError("review_context must be a nonempty object")
    digest = review_context_digest(row)
    if row.get("review_context_sha256") != digest or verdict.get("context_sha256") != digest:
        raise ValueError("context_sha256 must match the exact reviewed replay context")
    return True


def summarize_attribution_reviews(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Aggregate supplied human source checks, not automated attribution truth."""
    groups: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    for row in rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip() or question_id in seen:
            raise ValueError("question_id must be nonblank and unique")
        seen.add(question_id)
        source_type = row.get("source_type")
        if source_type not in {"official_filing", "earnings_call_transcript", "third_party_analysis"}:
            raise ValueError("invalid attribution source_type")
        # Legacy attribution-only records may omit status; an explicit state
        # must use the same vocabulary as generation review records.
        if ("generation_status" in row
                and row["generation_status"] not in ("not_run", "failed", "completed")):
            raise ValueError("invalid generation_status")
        group = groups.setdefault(source_type, {"question_count": 0, "reviewed_count": 0,
                                                "correct_count": 0, "context_bound_reviewed_count": 0})
        group["question_count"] += 1
        verdict = row.get("human_verdict")
        if row.get("generation_status") in ("not_run", "failed"):
            if row.get("generated_answer") is not None or verdict is not None:
                raise ValueError("uncompleted generation cannot have an answer or verdict")
        if verdict is None:
            continue
        if not isinstance(verdict, dict) or type(verdict.get("attribution_correct")) is not bool:
            raise ValueError("human_verdict requires boolean attribution_correct")
        for field in ("reviewer", "source_check"):
            if not isinstance(verdict.get(field), str) or not verdict[field].strip():
                raise ValueError(f"human_verdict requires nonblank {field}")
        answer = row.get("generated_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("attribution review requires nonblank generated_answer")
        if verdict.get("answer_sha256") != sha256(answer.encode("utf-8")).hexdigest():
            raise ValueError("answer_sha256 must match the exact reviewed answer")
        group["context_bound_reviewed_count"] += int(_context_bound_review(row, verdict))
        group["reviewed_count"] += 1
        group["correct_count"] += int(verdict["attribution_correct"])
    for group in groups.values():
        group["review_coverage"] = group["reviewed_count"] / group["question_count"]
        group["human_attribution_correct_rate"] = (
            group["correct_count"] / group["reviewed_count"] if group["reviewed_count"] else None
        )
        group["scope"] = "supplied human judgments only; not independent validation"
    return groups


def summarize_generation_reviews(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Summarize explicit human judgments, never infer truth from retrieval.

    This validates bookkeeping only; reviewer identity and source checks are
    supplied assertions, not authenticated proof or an automated truth judge.
    """
    counts = {"not_run_count": 0, "failed_count": 0,
              "unreviewed_count": 0, "reviewed_count": 0, "supported_count": 0,
              "context_bound_reviewed_count": 0, "field_reviewed_count": 0}
    field_judgments = {field: dict(true_count=0, false_count=0, unknown_count=0)
                       for field in ("metric", "period", "unit", "value", "citation")}
    claim_judgments = dict(supported=0, contradicted=0, not_established=0)
    complete_claim_judgments = dict(supported=0, contradicted=0, not_established=0)
    complete_claim_reviewed_count = 0
    claim_reviewed_count = 0
    response_judgments = dict(true_count=0, false_count=0, unknown_count=0)
    response_reviewed_count = 0
    supported_and_responsive_count = 0
    validator_counts = dict(pass_supported=0, pass_unsupported=0, reject_supported=0,
                            reject_unsupported=0, pass_unreviewed=0, reject_unreviewed=0,
                            validation_unknown=0, exact_abstention=0)
    seen: set[str] = set()
    for row in rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip() or question_id in seen:
            raise ValueError("question_id must be nonblank and unique")
        seen.add(question_id)
        status = row.get("generation_status")
        answer = row.get("generated_answer")
        verdict = row.get("human_verdict")
        if status not in ("not_run", "failed", "completed"):
            raise ValueError("invalid generation_status")
        if status != "completed":
            if answer is not None or verdict is not None:
                raise ValueError("uncompleted generation cannot have an answer or verdict")
            counts["not_run_count" if status == "not_run" else "failed_count"] += 1
            continue
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("completed generation requires a nonblank answer")
        validation_known = 'validation_error' in row
        validation_error = row.get('validation_error')
        if validation_known and validation_error is not None and (
                not isinstance(validation_error, str) or not validation_error.strip()):
            raise ValueError('validation_error must be null or a nonblank string')
        exact_abstention = answer == 'The retrieved evidence is insufficient.'
        validation_bucket = None
        if exact_abstention:
            validator_counts['exact_abstention'] += 1
        elif not validation_known:
            validator_counts['validation_unknown'] += 1
        else:
            validation_bucket = 'pass' if validation_error is None else 'reject'
        if verdict is None:
            counts["unreviewed_count"] += 1
            if validation_bucket is not None:
                validator_counts[validation_bucket + '_unreviewed'] += 1
            continue
        if not isinstance(verdict, dict) or type(verdict.get("supported")) is not bool:
            raise ValueError("human_verdict requires a boolean supported judgment")
        for field in ("reviewer", "source_check"):
            if not isinstance(verdict.get(field), str) or not verdict[field].strip():
                raise ValueError(f"human_verdict requires nonblank {field}")
        if verdict.get("answer_sha256") != sha256(answer.encode("utf-8")).hexdigest():
            raise ValueError("answer_sha256 must match the exact reviewed answer")
        counts["context_bound_reviewed_count"] += int(_context_bound_review(row, verdict))
        if 'answers_question' in verdict:
            responsive = verdict['answers_question']
            if responsive is not None and type(responsive) is not bool:
                raise ValueError('answers_question must be boolean or null')
            query = row.get('query')
            if (not isinstance(query, str) or not query.strip()
                    or verdict.get('query_sha256') != sha256(query.encode('utf-8')).hexdigest()):
                raise ValueError('query_sha256 must match the exact reviewed nonblank query')
            response_reviewed_count += 1
            response_judgments['unknown_count' if responsive is None else (
                'true_count' if responsive else 'false_count')] += 1
            supported_and_responsive_count += int(verdict['supported'] and responsive is True)
        if "claim_checks" in verdict or "claim_coverage_complete" in verdict:
            claims = verdict.get("claim_checks")
            complete = verdict.get("claim_coverage_complete")
            if not isinstance(claims, list) or not claims or type(complete) is not bool:
                raise ValueError("claim_checks requires nonempty list and boolean claim_coverage_complete")
            covered = set()
            judgments = []
            for claim in claims:
                if not isinstance(claim, dict):
                    raise ValueError("claim must be an object")
                start, end = claim.get("start"), claim.get("end")
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(answer)):
                    raise ValueError("claim offsets must be valid integer character positions")
                if claim.get("text") != answer[start:end] or not answer[start:end].strip():
                    raise ValueError("claim text must match exact answer span")
                positions = set(range(start, end))
                if covered & positions:
                    raise ValueError("claim spans must not overlap")
                covered.update(positions)
                judgment = claim.get("judgment")
                if not isinstance(judgment, str) or judgment not in claim_judgments:
                    raise ValueError("claim judgment must be supported/contradicted/not_established")
                if not isinstance(claim.get("source_check"), str) or not claim["source_check"].strip():
                    raise ValueError("claim source_check must be nonblank")
                judgments.append(judgment)
            # Character coverage prevents omission, not semantic claim segmentation.
            if complete and any(not char.isspace() and i not in covered
                                for i, char in enumerate(answer)):
                raise ValueError("complete claim review must cover all nonwhitespace answer characters")
            if verdict["supported"] and (not complete or any(j != "supported" for j in judgments)):
                raise ValueError("supported review conflicts with incomplete or unsupported claim checks")
            for judgment in judgments:
                claim_judgments[judgment] += 1
                if complete:
                    complete_claim_judgments[judgment] += 1
            claim_reviewed_count += 1
            complete_claim_reviewed_count += int(complete)
        if "field_checks" in verdict:
            checks = verdict["field_checks"]
            required = {"metric", "period", "unit", "value", "citation"}
            if (not isinstance(checks, dict) or set(checks) != required
                    or any(value is not None and type(value) is not bool for value in checks.values())):
                raise ValueError("field_checks requires metric/period/unit/value/citation with boolean or null judgments")
            if verdict["supported"] and not all(value is True for value in checks.values()):
                raise ValueError("supported review conflicts with false or unknown field_checks")
            counts["field_reviewed_count"] += 1
            for field, judgment in checks.items():
                bucket = "unknown_count" if judgment is None else (
                    "true_count" if judgment else "false_count")
                field_judgments[field][bucket] += 1
        counts["reviewed_count"] += 1
        counts["supported_count"] += int(verdict["supported"])
        if validation_bucket is not None:
            validator_counts[validation_bucket + ('_supported' if verdict['supported'] else '_unsupported')] += 1
    for totals in field_judgments.values():
        judged = totals["true_count"] + totals["false_count"]
        totals["judged_count"] = judged
        totals["supplied_true_rate"] = totals["true_count"] / judged if judged else None
    return {
        "question_count": len(rows), **counts,
        "question_response_judgments": response_judgments,
        "question_response_reviewed_count": response_reviewed_count,
        "question_response_missing_count": len(rows) - response_reviewed_count,
        "supported_and_responsive_count": supported_and_responsive_count,
        "question_response_scope": (
            'supplied answers_question judgments bound to exact query and answer hashes; '
            'source support remains separate; absent reviews are missing, null judgments unknown; '
            'not authenticated completeness, correctness or independent answer accuracy'
        ),
        "validator_review_counts": validator_counts,
        "validator_review_scope": (
            'completed non-abstention outputs only; explicit validation_error crossed with supplied '
            'support judgments; missing validation remains unknown; not authenticated truth, '
            'question completeness or independently calibrated validator performance'
        ),
        "field_judgments": field_judgments,
        "claim_judgments": claim_judgments,
        "complete_claim_judgments": complete_claim_judgments,
        "complete_supplied_claim_support_rate": (
            complete_claim_judgments["supported"] / sum(complete_claim_judgments.values())
            if sum(complete_claim_judgments.values()) else None
        ),
        "claim_rate_scope": (
            "supplied judgments from character-complete reviews only; unknown claims remain "
            "in the denominator; not authenticated review, semantic coverage or answer accuracy"
        ),
        "claim_reviewed_count": claim_reviewed_count,
        "complete_claim_reviewed_count": complete_claim_reviewed_count,
        "partial_claim_reviewed_count": claim_reviewed_count - complete_claim_reviewed_count,
        "complete_claim_review_coverage": complete_claim_reviewed_count / len(rows) if rows else None,
        "human_supported_rate": (
            counts["supported_count"] / counts["reviewed_count"]
            if counts["reviewed_count"] else None
        ),
        "outcome_counts": {
            "supported": counts["supported_count"],
            "unsupported": counts["reviewed_count"] - counts["supported_count"],
            "unreviewed": counts["unreviewed_count"],
            "failed": counts["failed_count"],
            "not_run": counts["not_run_count"],
        },
        "supported_question_yield": counts["supported_count"] / len(rows) if rows else None,
        "supported_question_yield_scope": (
            "supplied supported judgments divided by all input questions, including "
            "failed, unreviewed and not-run outcomes; not independent answer accuracy "
            "or relevance/completeness validation"
        ),
        "review_coverage": counts["reviewed_count"] / len(rows) if rows else None,
        "field_review_coverage": counts["field_reviewed_count"] / len(rows) if rows else None,
        "scope": "supplied human judgments only; not independent validation",
    }


def evaluate_retriever(
    *,
    retriever: Retriever,
    questions: Sequence[EvaluationQuestion],
    top_k: int,
    threshold: float,
    include_review_snapshot: bool = False,
) -> dict[str, object]:
    """Evaluate page retrieval and the evidence gate without generating answers."""
    if not questions:
        raise ValueError("at least one evaluation question is required")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    _require_unique_question_ids(questions)

    records: list[dict[str, object]] = []
    answerable_hits = 0
    page_recall_sum = 0.0
    gold_page_chunk_precision_sum = 0.0
    reciprocal_rank_sum = 0.0
    page_slot_precision_sum = 0.0
    page_slot_f1_sum = 0.0
    unique_page_ap_sum = 0.0
    answerable_gate_accepts = 0
    gold_page_gate_hits = 0
    answerable_non_gold_gate_accepts = 0
    top_evidence_gold_hits = 0
    top_evidence_non_gold_accepts = 0
    unanswerable_false_accepts = 0

    for question in questions:
        candidates = retriever.retrieve(question.query, top_k=top_k)
        retrieved_pages = [item.chunk.page_number for item in candidates]
        retrieved_locations = [
            (item.chunk.document_id, item.chunk.page_number) for item in candidates
        ]
        scores = [item.score for item in candidates]
        selected = select_evidence(candidates, threshold=threshold, top_k=top_k)
        gate_accepted = bool(selected)
        number_pattern = re.compile(
            r"(?<![\d.])([+\-−]?)\s*\$?\s*([+\-−]?)"
            r"(\d[\d,]*(?:\.\d+)?)(?!\d|\.\d)"
        )
        expected_numbers = list(number_pattern.finditer(question.expected_answer or ""))
        # Free-text answers can include years, amounts, ratios or comparisons.
        # Without a typed target, the first numeral is not an answer-value label.
        expected_number = expected_numbers[0] if len(expected_numbers) == 1 else None
        # Parentheses can encode accounting signs; unsupported is not positive.
        if expected_number and (
            expected_number.string[:expected_number.start()].rstrip().endswith("(")
            or expected_number.string[expected_number.end():].lstrip().startswith(")")
        ):
            expected_number = None
        # A matching number is a teaching diagnostic, not field/column validation.
        expected_number_in_top_evidence = (
            any(
                not (match[1] and match[2])
                and not match.string[:match.start()].rstrip().endswith("(")
                and not match.string[match.end():].lstrip().startswith(")")
                and not (expected_number[1] and expected_number[2])
                and ("-" if (match[1] or match[2]) in {"-", "−"} else "+")
                == ("-" if (expected_number[1] or expected_number[2]) in {"-", "−"} else "+")
                and match[3].replace(",", "") == expected_number[3].replace(",", "")
                for match in number_pattern.finditer(selected[0].chunk.text)
            )
            if selected and expected_number else None
        )

        first_gold_rank: int | None = None
        page_hit: bool | None = None
        gate_gold_page_hit: bool | None = None
        top_evidence_gold_hit: bool | None = None
        page_recall: float | None = None
        gold_page_chunk_precision: float | None = None
        page_slot_precision: float | None = None
        page_slot_f1: float | None = None
        unique_page_ap: float | None = None
        if question.answerable:
            if question.evidence_document_ids:
                gold_locations = set(
                    zip(
                        question.evidence_document_ids,
                        question.evidence_pages,
                        strict=True,
                    )
                )
                page_recall = len(gold_locations.intersection(retrieved_locations)) / len(
                    gold_locations
                )
                gold_page_chunk_precision = sum(
                    location in gold_locations for location in retrieved_locations
                ) / top_k
                first_gold_rank = next(
                    (
                        rank
                        for rank, location in enumerate(retrieved_locations, start=1)
                        if location in gold_locations
                    ),
                    None,
                )
                gate_gold_page_hit = any(
                    (item.chunk.document_id, item.chunk.page_number)
                    in gold_locations
                    for item in selected
                )
                top_evidence_gold_hit = bool(
                    selected
                    and (selected[0].chunk.document_id, selected[0].chunk.page_number)
                    in gold_locations
                )
            else:
                gold_pages = set(question.evidence_pages)
                page_recall = len(gold_pages.intersection(retrieved_pages)) / len(
                    gold_pages
                )
                gold_page_chunk_precision = sum(
                    page in gold_pages for page in retrieved_pages
                ) / top_k
                first_gold_rank = next(
                    (
                        rank
                        for rank, page in enumerate(retrieved_pages, start=1)
                        if page in gold_pages
                    ),
                    None,
                )
                gate_gold_page_hit = any(
                    item.chunk.page_number in gold_pages for item in selected
                )
                top_evidence_gold_hit = bool(
                    selected and selected[0].chunk.page_number in gold_pages
                )
            # A page is a labeled retrieval unit, not every chunk on that page.
            # Keep original chunk ranks: repeated pages consume slots but earn
            # no additional credit. Missing slots remain in the k denominator.
            gold_units = gold_locations if question.evidence_document_ids else gold_pages
            ranked_units = retrieved_locations if question.evidence_document_ids else retrieved_pages
            seen_gold = set()
            precision_sum = 0.0
            for rank, unit in enumerate(ranked_units, start=1):
                if unit in gold_units and unit not in seen_gold:
                    seen_gold.add(unit)
                    precision_sum += len(seen_gold) / rank
            page_slot_precision = len(seen_gold) / top_k
            page_slot_f1 = (
                2 * page_slot_precision * page_recall / (page_slot_precision + page_recall)
                if page_slot_precision + page_recall else 0.0
            )
            unique_page_ap = precision_sum / len(gold_units)
            page_slot_precision_sum += page_slot_precision
            page_slot_f1_sum += page_slot_f1
            unique_page_ap_sum += unique_page_ap
            page_hit = first_gold_rank is not None
            answerable_hits += int(page_hit)
            page_recall_sum += page_recall
            gold_page_chunk_precision_sum += gold_page_chunk_precision
            if first_gold_rank is not None:
                reciprocal_rank_sum += 1.0 / first_gold_rank
            answerable_gate_accepts += int(gate_accepted)
            gold_page_gate_hits += int(gate_gold_page_hit)
            answerable_non_gold_gate_accepts += int(
                gate_accepted and not gate_gold_page_hit
            )
            top_evidence_gold_hits += int(top_evidence_gold_hit)
            top_evidence_non_gold_accepts += int(
                gate_accepted and not top_evidence_gold_hit
            )
        else:
            unanswerable_false_accepts += int(gate_accepted)

        # Diagnostic depends on supplied page labels, not claim entailment.
        if question.answerable:
            if not page_hit:
                page_gate_diagnostic = "gold_page_missing_from_candidates"
            elif not gate_gold_page_hit:
                page_gate_diagnostic = "gold_page_removed_by_gate"
            else:
                page_gate_diagnostic = "gold_page_retained_answer_unverified"
        else:
            page_gate_diagnostic = (
                "negative_gate_accepted" if gate_accepted else "negative_gate_rejected"
            )

        records.append(
            {
                "question_id": question.question_id,
                "evaluation_usage": (
                    dict(question.evaluation_usage)
                    if question.evaluation_usage is not None else None
                ),
                "financial_ground_truth": (
                    dict(question.financial_ground_truth)
                    if question.financial_ground_truth is not None else None
                ),
                "attribution_ground_truth": (
                    dict(question.attribution_ground_truth)
                    if question.attribution_ground_truth is not None else None
                ),
                "query": question.query,
                "answerable": question.answerable,
                "evidence_pages": list(question.evidence_pages),
                "evidence_document_ids": list(question.evidence_document_ids),
                "retrieved_pages": retrieved_pages,
                "retrieved_chunk_ids": [item.chunk.chunk_id for item in candidates],
                "retrieved_locations": [
                    {'document_id': document_id, 'page_number': page_number}
                    for document_id, page_number in retrieved_locations
                ],
                "scores": scores,
                "page_hit": page_hit,
                "page_recall_at_k": page_recall,
                "gold_page_chunk_precision_at_k": gold_page_chunk_precision,
                "page_slot_precision_at_k": page_slot_precision,
                "page_slot_f1_at_k": page_slot_f1,
                "unique_page_average_precision_at_k": unique_page_ap,
                "first_gold_rank": first_gold_rank,
                "gate_accepted": gate_accepted,
                "gate_gold_page_hit": gate_gold_page_hit,
                "page_gate_diagnostic": page_gate_diagnostic,
                "top_evidence_gold_hit": top_evidence_gold_hit,
                "top_evidence_chunk_id": (
                    selected[0].chunk.chunk_id if selected else None
                ),
                "expected_answer": question.expected_answer,
                "expected_number_in_top_evidence": expected_number_in_top_evidence,
            }
        )
        if include_review_snapshot:
            context = build_grounded_context(question.query, evidence=selected)
            def snapshot(items: Sequence[ScoredChunk]) -> list[dict[str, object]]:
                return [
                    {"chunk_id": item.chunk.chunk_id,
                     "document_id": item.chunk.document_id,
                     "page_number": item.chunk.page_number,
                     "score": item.score, "text": item.chunk.text}
                    for item in items
                ]
            records[-1]["review_snapshot"] = {
                "candidates": snapshot(candidates),
                "selected_evidence": snapshot(selected),
                "prompt": (build_grounded_prompt(query=question.query, context=context)
                           if context.can_answer else None),
                "generation_status": "not_run",
                "generated_answer": None,
                "human_verdict": None,
            }

    answerable_count = sum(question.answerable for question in questions)
    unanswerable_count = len(questions) - answerable_count

    # Exact supplied cell identities only: this does not authenticate annotations,
    # normalize financial terminology, or establish statistical independence.
    identity_fields = ("document_id", "statement", "metric", "period", "unit")
    financial_cells: dict[tuple[str, ...], list[EvaluationQuestion]] = {}
    for question in questions:
        if question.financial_ground_truth is not None:
            key = tuple(question.financial_ground_truth[field] for field in identity_fields)
            financial_cells.setdefault(key, []).append(question)
    financial_conflicts = []
    development_acceptance_overlaps = []
    for key, members in financial_cells.items():
        development_ids = [q.question_id for q in members if q.evaluation_usage is not None
                           and q.evaluation_usage["purpose"] == "development"]
        candidate_ids = [q.question_id for q in members if q.evaluation_usage is not None
                         and q.evaluation_usage["purpose"] == "acceptance_candidate"]
        if development_ids and candidate_ids:
            development_acceptance_overlaps.append({
                "identity": dict(zip(identity_fields, key)),
                "development_question_ids": development_ids,
                "acceptance_candidate_question_ids": candidate_ids,
            })
        values = sorted({q.financial_ground_truth["value"] for q in members})
        if len(values) > 1:
            financial_conflicts.append({
                "identity": dict(zip(identity_fields, key)),
                "question_ids": [q.question_id for q in members],
                "values": values,
            })

    # Exact query equality also catches overlaps without financial labels,
    # including refusal questions. It is a warning, not an independence proof.
    query_groups: dict[str, list[EvaluationQuestion]] = {}
    for question in questions:
        query_groups.setdefault(question.query, []).append(question)
    query_overlaps = []
    for query, members in query_groups.items():
        development_ids = [q.question_id for q in members if q.evaluation_usage is not None
                           and q.evaluation_usage["purpose"] == "development"]
        candidate_ids = [q.question_id for q in members if q.evaluation_usage is not None
                         and q.evaluation_usage["purpose"] == "acceptance_candidate"]
        if development_ids and candidate_ids:
            query_overlaps.append({
                "query": query, "development_question_ids": development_ids,
                "acceptance_candidate_question_ids": candidate_ids,
            })

    summary = {
        "question_count": len(questions),
        "dataset_registration": {
            "unregistered_count": sum(q.evaluation_usage is None for q in questions),
            "development_count": sum(
                q.evaluation_usage is not None
                and q.evaluation_usage["purpose"] == "development" for q in questions
            ),
            "acceptance_candidate_count": sum(
                q.evaluation_usage is not None
                and q.evaluation_usage["purpose"] == "acceptance_candidate" for q in questions
            ),
            "unique_query_count": len({q.query for q in questions}),
            "financial_label_count": sum(q.financial_ground_truth is not None for q in questions),
            "unique_financial_cell_count": len(financial_cells),
            "repeated_financial_cell_question_count": sum(len(group) - 1 for group in financial_cells.values()),
            "conflicting_financial_cell_count": len(financial_conflicts),
            "financial_cell_conflicts": financial_conflicts,
            "development_acceptance_cell_overlap_count": len(development_acceptance_overlaps),
            "development_acceptance_cell_overlaps": development_acceptance_overlaps,
            "development_acceptance_query_overlap_count": len(query_overlaps),
            "development_acceptance_query_overlaps": query_overlaps,
            "independently_verified_count": None,
            "scope": "supplied registrations only; exact query/cell equality is not semantic deduplication, source verification or independence verification",
        },
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        # Historical reports call this recall, but it counts any gold-page hit.
        # Keep the key for compatibility and expose the exact metrics below.
        "recall_at_k": answerable_hits / answerable_count if answerable_count else None,
        "page_hit_rate_at_k": (
            answerable_hits / answerable_count if answerable_count else None
        ),
        "page_recall_at_k": (
            page_recall_sum / answerable_count if answerable_count else None
        ),
        "gold_page_chunk_precision_at_k": (
            gold_page_chunk_precision_sum / answerable_count
            if answerable_count else None
        ),
        "mean_reciprocal_rank": (
            reciprocal_rank_sum / answerable_count if answerable_count else None
        ),
        "page_slot_precision_at_k": page_slot_precision_sum / answerable_count if answerable_count else None,
        "page_slot_f1_at_k": page_slot_f1_sum / answerable_count if answerable_count else None,
        "unique_page_mean_average_precision_at_k": unique_page_ap_sum / answerable_count if answerable_count else None,
        "page_ranking_metric_scope": "supplied gold pages; first occurrence earns credit at original chunk rank; repeated pages and missing slots are penalized; page-only legacy labels cannot distinguish documents; not field-level relevance or answer correctness",
        "answerable_gate_accept_rate": (
            answerable_gate_accepts / answerable_count if answerable_count else None
        ),
        "gold_page_gate_hit_rate": (
            gold_page_gate_hits / answerable_count if answerable_count else None
        ),
        "answerable_non_gold_gate_rate": (
            answerable_non_gold_gate_accepts / answerable_count
            if answerable_count
            else None
        ),
        "top_evidence_gold_rate": (
            top_evidence_gold_hits / answerable_count if answerable_count else None
        ),
        "top_evidence_non_gold_accept_rate": (
            top_evidence_non_gold_accepts / answerable_count
            if answerable_count
            else None
        ),
        "unanswerable_false_accept_rate": (
            unanswerable_false_accepts / unanswerable_count
            if unanswerable_count
            else None
        ),
    }
    source_groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        labels = record["attribution_ground_truth"]
        source_type = labels["source_type"] if labels is not None else "unlabeled"
        source_groups.setdefault(source_type, []).append(record)
    source_summary = {}
    for source_type, group in source_groups.items():
        positives = [row for row in group if row["answerable"]]
        negatives = [row for row in group if not row["answerable"]]
        source_summary[source_type] = {
            "question_count": len(group),
            "answerable_count": len(positives),
            "unanswerable_count": len(negatives),
            "page_hit_rate_at_k": (
                sum(bool(row["page_hit"]) for row in positives) / len(positives)
                if positives else None
            ),
            "unanswerable_false_accept_rate": (
                sum(bool(row["gate_accepted"]) for row in negatives) / len(negatives)
                if negatives else None
            ),
            "attribution_accuracy": None,
            "scope": "retrieval only; source labels supplied by annotator, attribution not scored",
        }
    return {"summary": summary, "records": records,
            "source_retrieval_summary": source_summary}
