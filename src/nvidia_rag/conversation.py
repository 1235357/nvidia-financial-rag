"""Conservative, transparent follow-up rewriting for the teaching demo.

This is not a general conversational agent. It never infers a metric or year
from retrieved passages, and it does not change the stateless production API.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_REVENUE_ANTECEDENT = re.compile(
    r"What was NVIDIA's annual revenue in FY(20\d{2})\?",
    re.IGNORECASE,
)
_EXPLICIT_YEAR_FOLLOWUP = re.compile(
    r"(?:(?:What|How) about|And) FY(20\d{2})\?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ResolvedDemoQuery:
    query: str
    used_previous: bool
    previous_fiscal_year: int | None = None
    target_fiscal_year: int | None = None


def resolve_demo_query(
    previous_query: str,
    current_query: str,
    *,
    available_fiscal_years: range = range(2022, 2027),
) -> ResolvedDemoQuery:
    """Rewrite explicit revenue/year follow-ups; clarify recognized ambiguity.

    This bounded rule set does not detect every possible conversational reference.

    ``previous_query`` must be the last *resolved* question, not a generated
    answer or a retrieved passage. The caller controls per-session state.
    """
    current = current_query.strip()
    previous = previous_query.strip()
    if not current or len(current) > 500:
        raise ValueError("Please clarify the current question (1–500 characters).")
    followup = _EXPLICIT_YEAR_FOLLOWUP.fullmatch(current)
    if followup:
        antecedent = _REVENUE_ANTECEDENT.fullmatch(previous)
        if antecedent is None:
            raise ValueError(
                "Please clarify the metric and fiscal year in a standalone question."
            )
        target_year = int(followup.group(1))
        if target_year not in available_fiscal_years:
            raise ValueError(
                "Please clarify with a fiscal year available in this demo corpus."
            )
        return ResolvedDemoQuery(
            query=f"What was NVIDIA's annual revenue in FY{target_year}?",
            used_previous=True,
            previous_fiscal_year=int(antecedent.group(1)),
            target_fiscal_year=target_year,
        )
    unresolved_reference = re.fullmatch(
        r"(?:How did that change|The previous year)\?", current, re.IGNORECASE
    )
    if unresolved_reference or re.match(
        r"(?:what|how) about\b|last year\b|and\b", current, re.IGNORECASE
    ):
        raise ValueError(
            "Please clarify the metric and fiscal year in a standalone question."
        )
    return ResolvedDemoQuery(query=current, used_previous=False)
