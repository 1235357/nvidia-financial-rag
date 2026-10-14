"""Self-contained legacy metric/preflight regressions, brought into package CI.

Synthetic labels establish metric semantics, never financial gold.
Legacy source SHA256: 6b998b94bb51d19ccc3dc49cc5460fd1f33ce856b733f47128bfe7d7e5fd67c4
"""
import pytest
from nvidia_rag.core import Chunk, ScoredChunk
from nvidia_rag.evaluation import EvaluationQuestion, evaluate_retriever

def _acceptance_preflight_question(question_id, query, *, labels=None, registered=True):
    from nvidia_rag.evaluation import EvaluationQuestion
    usage = {"purpose": "acceptance_candidate", "frozen_on": "2026-10-02",
             "development_exposed": False, "registration_note": "synthetic fixture"}
    return EvaluationQuestion(question_id, query, True, (3,),
                              financial_ground_truth=labels,
                              evaluation_usage=usage if registered else None)

def test_acceptance_preflight_rejects_unregistered_candidate():
    from nvidia_rag import evaluation
    validate = getattr(evaluation, "validate_acceptance_preflight", None)
    assert callable(validate), "acceptance preflight is missing"
    with pytest.raises(ValueError, match="registered acceptance"):
        validate([_acceptance_preflight_question("new", "new query", registered=False)], [])

@pytest.mark.parametrize("empty_side", ["candidates", "reference"])
def test_acceptance_preflight_rejects_empty_inputs(empty_side):
    from nvidia_rag.evaluation import validate_acceptance_preflight
    candidates = [_acceptance_preflight_question("new", "new query")]
    reference = [_acceptance_preflight_question("old", "old query")]
    if empty_side == "candidates":
        candidates = []
    else:
        reference = []
    with pytest.raises(ValueError, match=f"{empty_side} must not be empty"):
        validate_acceptance_preflight(candidates, reference)

def test_acceptance_preflight_rejects_exact_query_overlap_with_legacy_reference():
    from nvidia_rag import evaluation
    validate = getattr(evaluation, "validate_acceptance_preflight", None)
    assert callable(validate), "acceptance preflight is missing"
    candidate = _acceptance_preflight_question("new", "same query")
    reference = evaluation.EvaluationQuestion("old", "same query", True, (3,))
    with pytest.raises(ValueError, match="query overlap"):
        validate([candidate], [reference])

def test_acceptance_preflight_rejects_cell_overlap_despite_different_question():
    from nvidia_rag import evaluation
    validate = getattr(evaluation, "validate_acceptance_preflight", None)
    assert callable(validate), "acceptance preflight is missing"
    labels = financial_labels()
    candidate = _acceptance_preflight_question("new", "paraphrase", labels=labels)
    reference = evaluation.EvaluationQuestion("old", "original", True, (3,),
                                             financial_ground_truth=labels)
    with pytest.raises(ValueError, match="financial cell overlap"):
        validate([candidate], [reference])

def test_acceptance_preflight_allows_other_period_without_certifying_independence():
    from nvidia_rag import evaluation
    validate = getattr(evaluation, "validate_acceptance_preflight", None)
    assert callable(validate), "acceptance preflight is missing"
    labels = financial_labels()
    candidate = _acceptance_preflight_question("new", "other year", labels={
        **labels, "period": "Year ended 2024-01-28"})
    reference = evaluation.EvaluationQuestion("old", "original", True, (3,),
                                             financial_ground_truth=labels)
    assert validate([candidate], [reference]) is None

def financial_labels():
    return {"document_id": "fixture-report", "statement": "Income statement",
            "metric": "Net income", "period": "Year ended 2025-01-26",
            "unit": "USD million", "value": "72,880", "source_locator": "PDF page 1, row 10, column 2",
            "reviewer": "synthetic fixture", "source_check": "Synthetic labels, not a PDF verification"}

@pytest.mark.parametrize("locations,precision,recall,f1,ap", [
    ([("a", 1), ("a", 1), ("b", 1)], .5, 1.0, 2/3, (1 + 2/3)/2),
    ([("wrong", 1), ("a", 1)], .25, .5, 1/3, .25),
    ([], 0.0, 0.0, 0.0, 0.0),
])
def test_unique_page_ranking_metrics_do_not_reward_duplicate_chunks(
    locations, precision, recall, f1, ap
):
    class RankedFixture:
        def retrieve(self, query, *, top_k):
            return [ScoredChunk(Chunk(str(i), doc, page, "synthetic ranking"), .9)
                    for i, (doc, page) in enumerate(locations)]

    questions = [EvaluationQuestion("positive", "synthetic", True, (1, 1),
                                    evidence_document_ids=("a", "b")),
                 EvaluationQuestion("negative", "synthetic negative", False, ())]
    report = evaluate_retriever(retriever=RankedFixture(), questions=questions,
                                top_k=4, threshold=.68)
    row, negative = report["records"]
    assert row["page_slot_precision_at_k"] == pytest.approx(precision)
    assert row["page_recall_at_k"] == pytest.approx(recall)
    assert row["page_slot_f1_at_k"] == pytest.approx(f1)
    assert row["unique_page_average_precision_at_k"] == pytest.approx(ap)
    assert negative["unique_page_average_precision_at_k"] is None
    assert report["summary"]["unique_page_mean_average_precision_at_k"] == pytest.approx(ap)
    assert report["summary"]["page_slot_f1_at_k"] == pytest.approx(f1)

def test_optional_review_snapshot_preserves_exact_candidates_and_gated_prompt() -> None:
    question = EvaluationQuestion("review", "answerable hit", True, (3,))
    report = evaluate_retriever(
        retriever=StubRetriever(), questions=[question], top_k=2,
        threshold=0.78, include_review_snapshot=True,
    )
    review = report["records"][0]["review_snapshot"]
    assert [row["text"] for row in review["candidates"]] == ["noise", "answer"]
    assert [row["chunk_id"] for row in review["selected_evidence"]] == ["noise"]
    assert "noise" in review["prompt"]
    assert "[doc p.9]" in review["prompt"]
    assert review["generation_status"] == "not_run"
    assert review["generated_answer"] is None
    assert review["human_verdict"] is None

def test_multi_document_ground_truth_matches_document_and_page() -> None:
    class MultiDocumentRetriever:
        def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
            return [
                ScoredChunk(
                    Chunk('wrong-doc', 'nvidia-fy2026-annual', 144, 'Wrong report.'),
                    0.90,
                ),
                ScoredChunk(
                    Chunk('right-doc', 'nvidia-fy2025-annual', 144, 'Revenue table.'),
                    0.80,
                ),
            ][:top_k]

    questions = [
        EvaluationQuestion(
            question_id='fy2025-revenue',
            query='What was fiscal 2025 revenue?',
            answerable=True,
            evidence_pages=(144,),
            evidence_document_ids=('nvidia-fy2025-annual',),
        )
    ]

    report = evaluate_retriever(
        retriever=MultiDocumentRetriever(),
        questions=questions,
        top_k=2,
        threshold=0.70,
    )

    assert report['summary']['recall_at_k'] == 1.0
    assert report['summary']['mean_reciprocal_rank'] == 0.5
    assert report['records'][0]['retrieved_locations'][0] == {
        'document_id': 'nvidia-fy2026-annual',
        'page_number': 144,
    }

def test_wrong_report_above_gate_does_not_count_as_gold_page_hit() -> None:
    """Catch the false claim that accepting any page proves correct evidence."""
    class MixedRetriever:
        def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
            return [
                ScoredChunk(
                    Chunk('wrong', 'nvidia-fy2026-annual', 144, 'Wrong year.'),
                    0.90,
                ),
                ScoredChunk(
                    Chunk('gold', 'nvidia-fy2025-annual', 144, 'Correct year.'),
                    0.80,
                ),
            ][:top_k]

    report = evaluate_retriever(
        retriever=MixedRetriever(),
        questions=[
            EvaluationQuestion(
                'fy2025',
                'What was fiscal 2025 revenue?',
                True,
                (144,),
                ('nvidia-fy2025-annual',),
            )
        ],
        top_k=2,
        threshold=0.85,
    )

    assert report['summary']['recall_at_k'] == 1.0
    assert report['summary']['answerable_gate_accept_rate'] == 1.0
    assert report['summary']['gold_page_gate_hit_rate'] == 0.0
    assert report['summary']['answerable_non_gold_gate_rate'] == 1.0
    assert report['records'][0]['gate_gold_page_hit'] is False

def test_top_accepted_evidence_can_be_wrong_while_gold_also_passes() -> None:
    """Catch a misleading all-green gate when the first cited page is wrong."""
    class WrongFirstRetriever:
        def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
            return [
                ScoredChunk(Chunk('wrong', 'report', 170, 'Not the table.'), 0.83),
                ScoredChunk(Chunk('gold', 'report', 150, 'Correct table.'), 0.81),
            ][:top_k]

    report = evaluate_retriever(
        retriever=WrongFirstRetriever(),
        questions=[EvaluationQuestion('q', 'Research expenses?', True, (150,), ('report',))],
        top_k=2,
        threshold=0.78,
    )

    assert report['summary']['gold_page_gate_hit_rate'] == 1.0
    assert report['summary']['top_evidence_gold_rate'] == 0.0
    assert report['summary']['top_evidence_non_gold_accept_rate'] == 1.0
    assert report['records'][0]['top_evidence_gold_hit'] is False
    assert report['records'][0]['retrieved_chunk_ids'] == ['wrong', 'gold']
    assert report['records'][0]['top_evidence_chunk_id'] == 'wrong'

def test_positive_metrics_are_unmeasured_without_positive_questions() -> None:
    class EmptyRetriever:
        def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
            return []

    report = evaluate_retriever(
        retriever=EmptyRetriever(),
        questions=[EvaluationQuestion('negative', 'Unsupported question?', False, (), ())],
        top_k=2,
        threshold=0.78,
    )
    assert report['summary']['answerable_count'] == 0
    for key in (
        'recall_at_k', 'page_hit_rate_at_k', 'page_recall_at_k',
        'gold_page_chunk_precision_at_k', 'mean_reciprocal_rank',
        'answerable_gate_accept_rate', 'gold_page_gate_hit_rate',
        'answerable_non_gold_gate_rate', 'top_evidence_gold_rate',
        'top_evidence_non_gold_accept_rate',
    ):
        assert report['summary'][key] is None, key
    assert report['summary']['unanswerable_false_accept_rate'] == 0.0

def test_page_hit_rate_is_not_page_recall_when_multiple_gold_pages_exist() -> None:
    class OneOfTwoPagesRetriever:
        def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
            return [
                ScoredChunk(Chunk('a', 'fy2025', 18, 'Relevant page.'), 0.90),
                ScoredChunk(Chunk('b', 'fy2025', 18, 'Same relevant page.'), 0.85),
            ][:top_k]

    report = evaluate_retriever(
        retriever=OneOfTwoPagesRetriever(),
        questions=[EvaluationQuestion('q', 'Revenue?', True, (18, 144), ('fy2025', 'fy2025'))],
        top_k=2,
        threshold=0.78,
    )

    assert report['summary']['page_hit_rate_at_k'] == 1.0
    assert report['summary']['page_recall_at_k'] == 0.5
    assert report['records'][0]['page_recall_at_k'] == 0.5
    # Two hits on the same gold page fill both result slots, even though one
    # of the two distinct labeled pages is still missing.
    assert report['summary']['gold_page_chunk_precision_at_k'] == 1.0
    assert report['records'][0]['gold_page_chunk_precision_at_k'] == 1.0
    # Legacy output key remains available for historical reports but denotes hit rate.
    assert report['summary']['recall_at_k'] == 1.0


class StubRetriever:
    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        candidates = {
            "answerable hit": [
                ScoredChunk(Chunk("noise", "doc", 9, "noise"), 0.80),
                ScoredChunk(Chunk("gold", "doc", 3, "answer"), 0.75),
            ],
            "answerable miss": [
                ScoredChunk(Chunk("noise", "doc", 8, "noise"), 0.72),
            ],
            "unanswerable": [
                ScoredChunk(Chunk("weak", "doc", 10, "weak"), 0.61),
            ],
        }
        return candidates[query][:top_k]

@pytest.mark.parametrize('query,answerable,threshold,expected', [
    ('answerable hit', True, .78, 'gold_page_removed_by_gate'),
    ('answerable hit', True, .68, 'gold_page_retained_answer_unverified'),
    ('answerable miss', True, .68, 'gold_page_missing_from_candidates'),
    ('unanswerable', False, .68, 'negative_gate_rejected'),
    ('unanswerable', False, .60, 'negative_gate_accepted'),
])
def test_page_gate_diagnostic_localizes_stage_without_claiming_answer_truth(
    query, answerable, threshold, expected,
):
    question = EvaluationQuestion('diagnostic', query, answerable, (3,) if answerable else ())
    row = evaluate_retriever(retriever=StubRetriever(), questions=[question],
                             top_k=2, threshold=threshold)['records'][0]
    assert row.get('page_gate_diagnostic') == expected

def test_false_accept_rate_is_unmeasured_without_negative_questions():
    report = evaluate_retriever(
        retriever=StubRetriever(),
        questions=[EvaluationQuestion('positive', 'answerable hit', True, (3,))],
        top_k=2, threshold=0.68,
    )
    assert report['summary']['unanswerable_count'] == 0
    assert report['summary']['unanswerable_false_accept_rate'] is None
