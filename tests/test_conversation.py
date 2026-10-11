import pytest

from nvidia_rag.conversation import resolve_demo_query


@pytest.mark.parametrize("previous", ["", "What was NVIDIA's annual revenue in FY2025?"])
@pytest.mark.parametrize("followup", ["How did that change?", "The previous year?"])
def test_unresolved_reference_requires_clarification(previous: str, followup: str) -> None:
    with pytest.raises(ValueError, match="clarif"):
        resolve_demo_query(previous, followup)


def test_explicit_followup_rewrites_one_supported_antecedent() -> None:
    result = resolve_demo_query(
        "What was NVIDIA's annual revenue in FY2025?",
        "What about FY2024?",
    )
    assert result.query == "What was NVIDIA's annual revenue in FY2024?"
    assert result.used_previous is True
    assert result.previous_fiscal_year == 2025
    assert result.target_fiscal_year == 2024


@pytest.mark.parametrize(
    ("previous", "followup"),
    [
        ("", "What about FY2024?"),
        ("What were revenue and net income in FY2025?", "What about FY2024?"),
        ("What was NVIDIA's annual revenue in FY2025?", "What about it?"),
        ("What was NVIDIA's annual revenue in FY2025?", "Last year?"),
        ("What was NVIDIA's annual revenue in FY2025?", "What about FY2027?"),
        ("What was NVIDIA's annual revenue in FY2025?", "What about FY2021?"),
    ],
)
def test_followup_requires_clarification_when_not_in_safe_contract(
    previous: str, followup: str
) -> None:
    with pytest.raises(ValueError, match="clarif"):
        resolve_demo_query(previous, followup)


def test_standalone_question_is_not_silently_rewritten() -> None:
    result = resolve_demo_query(
        "What was NVIDIA's annual revenue in FY2025?",
        "What was NVIDIA's net income in FY2024?",
    )
    assert result.query == "What was NVIDIA's net income in FY2024?"
    assert result.used_previous is False


def test_and_year_followup_preserves_revenue_intent() -> None:
    result = resolve_demo_query(
        "What was NVIDIA's annual revenue in FY2025?", "And FY2024?"
    )
    assert result.query == "What was NVIDIA's annual revenue in FY2024?"
    assert result.used_previous is True
    assert result.previous_fiscal_year == 2025
    assert result.target_fiscal_year == 2024


@pytest.mark.parametrize("followup", ["And FY2024?", "And FY2027?", "And last year?", "And it?"])
def test_and_followup_does_not_retrieve_without_clear_supported_context(followup: str) -> None:
    with pytest.raises(ValueError, match="clarif"):
        resolve_demo_query("What were revenue and net income in FY2025?", followup)
