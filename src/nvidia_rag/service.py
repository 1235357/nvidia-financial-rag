"""Bounded application service for evidence retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .core import GroundedContext, ScoredChunk, build_grounded_context, select_evidence
from .retrieval import DenseInspection


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]: ...


@dataclass(frozen=True, slots=True)
class EvidenceInspection:
    context: GroundedContext
    retrieval: DenseInspection
    threshold: float
    top_k: int
    normalized_query: str


class EvidenceService:
    """Validate queries, retrieve candidates, and enforce the evidence gate."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        threshold: float,
        top_k: int,
        max_query_chars: int = 500,
    ) -> None:
        if max_query_chars <= 0:
            raise ValueError("max_query_chars must be positive")
        # Validate threshold and top_k once, before accepting traffic.
        select_evidence([], threshold=threshold, top_k=top_k)
        self._retriever = retriever
        self._threshold = threshold
        self._top_k = top_k
        self._max_query_chars = max_query_chars

    def query(self, query: str) -> GroundedContext:
        normalized = self._validate_query(query)
        candidates = self._retriever.retrieve(normalized, top_k=self._top_k)
        evidence = select_evidence(
            candidates,
            threshold=self._threshold,
            top_k=self._top_k,
        )
        return build_grounded_context(normalized, evidence=evidence)

    def _validate_query(self, query: str) -> str:
        normalized = query.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        if len(normalized) > self._max_query_chars:
            raise ValueError(
                f"query must be at most {self._max_query_chars} characters"
            )
        return normalized

    def inspect(self, query: str) -> EvidenceInspection:
        """Trace the same deterministic path used by the evidence-only demo."""
        normalized = self._validate_query(query)
        trace_method = getattr(self._retriever, "inspect", None)
        if trace_method is None:
            raise TypeError("retriever does not expose a dense inspection")
        retrieval = trace_method(normalized, top_k=self._top_k)
        evidence = select_evidence(
            retrieval.candidates,
            threshold=self._threshold,
            top_k=self._top_k,
        )
        return EvidenceInspection(
            context=build_grounded_context(normalized, evidence=evidence),
            retrieval=retrieval,
            threshold=self._threshold,
            top_k=self._top_k,
            normalized_query=normalized,
        )
