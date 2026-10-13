"""Interchangeable lexical and dense retrieval baselines."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .core import Chunk, ScoredChunk
from .sources import SourceRecord, select_task_sources

Encoder = Callable[[list[str]], Sequence[Sequence[float]]]


@dataclass(frozen=True, slots=True)
class DenseInspection:
    """Actual query-side arithmetic from one dense retrieval call."""

    candidates: tuple[ScoredChunk, ...]
    query_vector: tuple[float, ...]
    top_passage_vector: tuple[float, ...] | None
    considered_chunks: int
    considered_scores: tuple[float, ...]
    routed_document_id: str | None = None
    routed_document_ids: tuple[str, ...] = ()
    route_disposition: str | None = None
    route_reason: str | None = None


def _validate_chunks(chunks: Sequence[Chunk]) -> tuple[Chunk, ...]:
    materialized = tuple(chunks)
    if not materialized:
        raise ValueError("at least one chunk is required")
    return materialized


def _validate_top_k(top_k: int) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be positive")


class ScopedRetriever(Protocol):
    def retrieve_from_documents(
        self, query: str, *, top_k: int, document_ids: set[str] | None,
    ) -> list[ScoredChunk]: ...


class TaskSourceRetriever:
    """Opt-in explicit source task: scope before ranking, never fall back.

    Source declarations must originate from a separately validated manifest.
    This does not classify queries, verify licenses, or establish claim truth.
    """

    def __init__(self, retriever: ScopedRetriever, *, sources: list[SourceRecord], task: str) -> None:
        self._retriever = retriever
        self._document_ids = {record.document_id for record in select_task_sources(sources, task=task)}

    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        _validate_top_k(top_k)
        if not self._document_ids:
            return []
        return self._retriever.retrieve_from_documents(
            query, top_k=top_k, document_ids=set(self._document_ids),
        )


class TfidfRetriever:
    """Lexical control baseline; it is not presented as semantic retrieval."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = _validate_chunks(chunks)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(chunk.text for chunk in self._chunks)

    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        return self.retrieve_from_documents(query, top_k=top_k, document_ids=None)

    def retrieve_from_documents(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: set[str] | None,
    ) -> list[ScoredChunk]:
        """Rank lexical candidates within the same trusted scope as dense retrieval."""
        _validate_top_k(top_k)
        if document_ids is not None and (
            not document_ids or any(not item.strip() for item in document_ids)
        ):
            raise ValueError("document_ids must contain at least one non-blank ID")
        indices = [
            index for index, chunk in enumerate(self._chunks)
            if document_ids is None or chunk.document_id in document_ids
        ]
        if not indices:
            return []
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix[indices])[0]
        order = np.argsort(scores)[::-1][:top_k]
        return [
            ScoredChunk(chunk=self._chunks[indices[local_index]], score=float(scores[local_index]))
            for local_index in order
        ]


def reciprocal_rank_fusion(
    dense: Sequence[ScoredChunk],
    lexical: Sequence[ScoredChunk],
    *,
    top_k: int,
    rank_constant: int = 60,
) -> list[ScoredChunk]:
    """Fuse two ranked lists by rank, never by incomparable similarity scores.

    This is an isolated retrieval experiment. RRF values are not cosine
    similarities and must not enter the dense evidence threshold gate.
    """
    _validate_top_k(top_k)
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in (dense, lexical):
        seen_in_ranking: set[str] = set()
        for rank, candidate in enumerate(ranking, start=1):
            chunk_id = candidate.chunk.chunk_id
            if chunk_id in seen_in_ranking:
                raise ValueError("duplicate chunk ID within a ranking")
            seen_in_ranking.add(chunk_id)
            if chunk_id in chunks and chunks[chunk_id] != candidate.chunk:
                raise ValueError("conflicting chunk data across rankings")
            chunks[chunk_id] = candidate.chunk
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rank_constant + rank)
    order = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))[:top_k]
    return [ScoredChunk(chunk=chunks[chunk_id], score=fused[chunk_id]) for chunk_id in order]


class DenseRetriever:
    """Cosine retriever with an injected encoder for local or hosted embeddings."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        encoder: Encoder | None = None,
        query_encoder: Encoder | None = None,
        precomputed_embeddings: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> None:
        self._chunks = _validate_chunks(chunks)
        if query_encoder is None and encoder is None:
            raise ValueError("an encoder or query_encoder is required")
        self._encoder = encoder
        self._query_encoder = query_encoder or encoder
        if precomputed_embeddings is None:
            if encoder is None:
                raise ValueError(
                    "encoder is required when precomputed_embeddings are absent"
                )
            raw_matrix = encoder([chunk.text for chunk in self._chunks])
        else:
            raw_matrix = precomputed_embeddings
        self._matrix = self._normalize(
            np.asarray(raw_matrix, dtype=float)
        )
        if self._matrix.ndim != 2 or self._matrix.shape[0] != len(self._chunks):
            raise ValueError("encoder must return one fixed-length vector per text")

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2:
            raise ValueError("encoder output must be a two-dimensional matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("encoder vectors must contain only finite values")
        with np.errstate(over="ignore"):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)):
            raise ValueError("encoder vectors must have finite norms")
        if np.any(norms == 0):
            raise ValueError("encoder returned a zero vector")
        return matrix / norms

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """Immutable ranking inputs for source-bound composition validation."""
        return self._chunks

    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        return self.retrieve_from_documents(query, top_k=top_k, document_ids=None)

    def retrieve_from_documents(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: set[str] | None,
    ) -> list[ScoredChunk]:
        """Retrieve globally or from an explicit, trusted document subset."""
        return list(
            self.inspect(query, top_k=top_k, document_ids=document_ids).candidates
        )

    def inspect(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: set[str] | None = None,
        candidate_filter: Callable[[Chunk], bool] | None = None,
    ) -> DenseInspection:
        """Expose the exact normalized vectors and ranking used by retrieval."""
        _validate_top_k(top_k)
        query_matrix = self._normalize(
            np.asarray(self._query_encoder([query]), dtype=float)  # type: ignore[misc]
        )
        if query_matrix.shape[1] != self._matrix.shape[1]:
            raise ValueError("query and document embeddings must have the same dimension")
        if document_ids is None and candidate_filter is None:
            candidate_indices = np.arange(len(self._chunks))
            candidate_matrix = self._matrix
        else:
            if document_ids is not None and any(not item.strip() for item in document_ids):
                raise ValueError("document_ids must contain only non-blank IDs")
            candidate_indices = np.asarray(
                [
                    index
                    for index, chunk in enumerate(self._chunks)
                    if (document_ids is None or chunk.document_id in document_ids)
                    and (candidate_filter is None or candidate_filter(chunk))
                ],
                dtype=int,
            )
            if not len(candidate_indices):
                return DenseInspection(
                    candidates=(),
                    query_vector=tuple(float(value) for value in query_matrix[0]),
                    top_passage_vector=None,
                    considered_chunks=0,
                    considered_scores=(),
                )
            candidate_matrix = self._matrix[candidate_indices]
        scores = candidate_matrix @ query_matrix[0]
        local_order = np.argsort(scores)[::-1][:top_k]
        order = candidate_indices[local_order]
        ranked = tuple(
            ScoredChunk(chunk=self._chunks[index], score=float(scores[local_index]))
            for local_index, index in zip(local_order, order, strict=True)
        )
        return DenseInspection(
            candidates=ranked,
            query_vector=tuple(float(value) for value in query_matrix[0]),
            top_passage_vector=(
                tuple(float(value) for value in self._matrix[order[0]])
                if len(order)
                else None
            ),
            considered_chunks=len(candidate_indices),
            considered_scores=tuple(float(value) for value in scores),
        )


class FiscalYearRoutingRetriever:
    """Route explicit single-fiscal-year queries before dense ranking.

    The router only uses a caller-provided mapping derived from trusted corpus
    metadata. Queries with no fiscal year or multiple known fiscal years use
    global retrieval; an explicit unknown fiscal year returns no candidates.
    """

    _FISCAL_YEAR = re.compile(r"\b(?:fiscal\s*(?:year\s*)?|fy\s*)(20\d{2})\b", re.IGNORECASE)
    _TABLE_YEAR = re.compile(r"\bFor fiscal (20\d{2})\b", re.IGNORECASE)
    _REPORT_YEAR = re.compile(
        r"\b(?:fiscal\s+(?:year\s+)?|FY\s*)(20\d{2})\s+annual\s+report\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        retriever: DenseRetriever,
        *,
        year_to_document_id: Mapping[int, str],
    ) -> None:
        self._retriever = retriever
        self._year_to_document_id = {
            int(year): document_id
            for year, document_id in year_to_document_id.items()
            if document_id.strip()
        }

    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        return list(self.inspect(query, top_k=top_k).candidates)

    def _route_document_id(self, query: str) -> str | None:
        years = {int(match) for match in self._FISCAL_YEAR.findall(query)}
        if len(years) == 1:
            return self._year_to_document_id.get(next(iter(years)))
        return None

    def _has_unavailable_fiscal_year(self, query: str) -> bool:
        years = {int(match) for match in self._FISCAL_YEAR.findall(query)}
        return bool(years - self._year_to_document_id.keys())

    def inspect(self, query: str, *, top_k: int) -> DenseInspection:
        document_id = self._route_document_id(query)
        years = {int(match) for match in self._FISCAL_YEAR.findall(query)}
        requested_year = next(iter(years)) if len(years) == 1 else None
        # An explicit report year selects a document, not a financial column.
        # The downstream field filter must still resolve dates/rows/units.
        report_years = self._REPORT_YEAR.findall(query)
        report_scope = len(report_years) == 1 and int(report_years[0]) == requested_year

        def keep_matching_table_cell(chunk: Chunk) -> bool:
            if report_scope or requested_year is None or ":table:" not in chunk.chunk_id:
                return True
            match = self._TABLE_YEAR.search(chunk.text)
            return match is None or int(match.group(1)) == requested_year

        inspection = self._retriever.inspect(
            query,
            top_k=top_k,
            document_ids=(
                set() if self._has_unavailable_fiscal_year(query)
                else {document_id} if document_id is not None else None
            ),
            candidate_filter=keep_matching_table_cell if document_id is not None else None,
        )
        return replace(inspection, routed_document_id=document_id)


class FiscalPeriodRoutingRetriever:
    """Isolated annual/quarter routing experiment for a mixed-period corpus."""

    _FISCAL_YEAR = FiscalYearRoutingRetriever._FISCAL_YEAR
    _QUARTER = re.compile(r"\bq([1-4])\b", re.IGNORECASE)

    def __init__(
        self,
        retriever: DenseRetriever | TfidfRetriever,
        *,
        annual_documents: Mapping[int, str],
        quarterly_documents: Mapping[tuple[int, int], str],
        quarter_end_dates: Mapping[str, str],
    ) -> None:
        self._retriever = retriever
        self._annual_documents = dict(annual_documents)
        self._quarterly_documents = dict(quarterly_documents)
        self._quarter_end_dates = dict(quarter_end_dates)

    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        years = {int(value) for value in self._FISCAL_YEAR.findall(query)}
        quarters = {int(value) for value in self._QUARTER.findall(query)}
        if len(years) == 1 and len(quarters) == 1:
            document_id = self._quarterly_documents.get(
                (next(iter(years)), next(iter(quarters)))
            )
            if document_id is None:
                return []
            explicit_dates = re.findall(
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b",
                query, re.IGNORECASE,
            )
            known_dates = {date.casefold(): doc for date, doc in self._quarter_end_dates.items()}
            if any(known_dates.get(date.casefold()) != document_id for date in explicit_dates):
                return []
            return self._retriever.retrieve_from_documents(
                query, top_k=top_k, document_ids={document_id}
            )
        if re.search(r"\b(?:quarter|(?:three|six)[ -]months?)\b", query, re.IGNORECASE):
            date_documents = {
                document_id for date, document_id in self._quarter_end_dates.items()
                if re.search(r"\b" + re.escape(date) + r"\b", query, re.IGNORECASE)
            }
            if len(date_documents) == 1:
                return self._retriever.retrieve_from_documents(
                    query, top_k=top_k, document_ids=date_documents
                )
            return []
        if quarters:
            return []
        if len(years) == 1 and not quarters:
            document_id = self._annual_documents.get(next(iter(years)))
            if document_id is None:
                return []
            return self._retriever.retrieve_from_documents(
                query, top_k=top_k, document_ids={document_id}
            )
        return self._retriever.retrieve(query, top_k=top_k)


class _Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]: ...


class QuarterRevenueScopeRetriever:
    """Opt-in veto: quarterly table cells cannot answer a different metric.

    This does not verify the surviving evidence or change the default demo.
    Only explicit single-metric queries may use these experimental cells.
    """

    _TOTAL_REVENUE = re.compile(
        r"\btotal\s+revenue\b|"
        r"\brevenue\s+for\s+the\s+(?:three|six)\s+months?\s+ended\b",
        re.IGNORECASE,
    )
    _OTHER_REVENUE_SCOPE = re.compile(
        r"\b(?:data\s+center|gaming|automotive|compute\s+and\s+networking|"
        r"segment|product)\b",
        re.IGNORECASE,
    )
    _OPERATING_INCOME = re.compile(r"\boperating\s+income\b", re.IGNORECASE)
    _OTHER_METRIC = re.compile(
        r"\b(?:gross\s+margin|gross\s+profit|net\s+income)\b",
        re.IGNORECASE,
    )
    _EXPLICIT_PERIOD = re.compile(
        r"\b(three|six)\s+months?\s+ended\s+(?:jul|july)\s+"
        r"(\d{1,2}),?\s+(20\d{2})\b",
        re.IGNORECASE,
    )
    _CANDIDATE_PREFIX = "NVIDIA Condensed Consolidated Statements of Income in "

    def __init__(self, retriever: _Retriever, *, candidate_pool_size: int = 50) -> None:
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        self._retriever = retriever
        self._candidate_pool_size = candidate_pool_size

    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        _validate_top_k(top_k)
        candidates = self._retriever.retrieve(
            query, top_k=max(top_k, self._candidate_pool_size)
        )
        wants_revenue = bool(self._TOTAL_REVENUE.search(query))
        wants_operating = bool(self._OPERATING_INCOME.search(query))
        if self._OTHER_REVENUE_SCOPE.search(query) or self._OTHER_METRIC.search(query):
            allowed_metric = None
        elif wants_revenue != wants_operating:
            allowed_metric = "Revenue" if wants_revenue else "Operating income"
        else:
            allowed_metric = None
        period_match = self._EXPLICIT_PERIOD.search(query)
        expected_period = (
            f"{period_match.group(1).title()} Months Ended "
            f"Jul {int(period_match.group(2))}, {period_match.group(3)}, "
            if period_match else None
        )
        return [
            item for item in candidates
            if not (
                ":quarter-table:" in item.chunk.chunk_id
                and item.chunk.text.startswith(self._CANDIDATE_PREFIX)
                and (
                    ", Revenue was $" in item.chunk.text
                    and allowed_metric != "Revenue"
                    or ", Operating income was $" in item.chunk.text
                    and allowed_metric != "Operating income"
                    or expected_period is not None
                    and expected_period not in item.chunk.text
                )
            )
        ][:top_k]


class FinancialTableScopeRetriever:
    """Experimental, conservative corpus-scope veto for annual USD tables.

    This only rejects requests known to be outside the supplied corpus. Passing
    this check does *not* establish that the retrieved row, column, or value is
    correct. Keep it out of the default demo until independently evaluated.
    """

    _FISCAL_YEAR = FiscalYearRoutingRetriever._FISCAL_YEAR
    _NON_USD = re.compile(
        r"(?:\b(?:eur(?:os?)?|gbp|pounds?|jpy|yen|rmb|cny|yuan|cad|Canadian dollars?)\b|€|£|¥)",
        re.IGNORECASE,
    )
    # Only a terminal exclusion following an explicit USD request is handled.
    # This is not general negation parsing; mixed/ambiguous requests stay vetoed.
    _USD_EXCLUSION = re.compile(
        rf"(\bUSD)\s*,?\s*not\s+{_NON_USD.pattern}\s*(?=[?.!]*$)",
        re.IGNORECASE,
    )
    _MONTHLY = re.compile(
        r"\b(?:monthly|per[- ]month)\b|"
        r"\b(?:during|in|for)\s+(?:the\s+month\s+of\s+)?"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\.?\s+20\d{2}\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        retriever: _Retriever,
        *,
        available_fiscal_years: set[int],
    ) -> None:
        if not available_fiscal_years:
            raise ValueError("available_fiscal_years must not be empty")
        self._retriever = retriever
        self._available_fiscal_years = frozenset(available_fiscal_years)

    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        _validate_top_k(top_k)
        years = {int(year) for year in self._FISCAL_YEAR.findall(query)}
        currency_query = self._USD_EXCLUSION.sub(r"\1", query)
        if (
            not years.issubset(self._available_fiscal_years)
            or self._NON_USD.search(currency_query)
            or self._MONTHLY.search(query)
        ):
            return []
        return self._retriever.retrieve(query, top_k=top_k)


class BalanceSheetFieldRetriever:
    """Opt-in exact-field filter for explicit annual balance-sheet questions.

    A candidate must come from the experimental structured table extractor and
    match the requested report, row and as-of date. This is a veto over dense
    retrieval, not independent verification of the PDF parser or of an answer.
    """

    _DATE = re.compile(r"\bas of\s+([A-Za-z]+)\s+(\d{1,2}),?\s+(20\d{2})\b", re.IGNORECASE)
    _BALANCE_SHEET_TITLE = re.compile(r"\bbalance sheets?\b", re.IGNORECASE)
    _CALENDAR_DATE = re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{1,2},?\s+20\d{2}\b",
        re.IGNORECASE,
    )
    _USD_UNIT = re.compile(
        r"\bin\s+(?:millions?\s+of\s+(?:(?:U\.S\.|US)\s+)?dollars?|"
        r"USD\s+millions?|millions?\s+(?:of\s+)?USD)\b",
        re.IGNORECASE,
    )
    _OTHER_USD_SCALE = re.compile(
        r"\b(?:USD\s+(?:billions?|thousands?)|"
        r"(?:billions?|thousands?)\s+(?:(?:of\s+)?USD|"
        r"of\s+(?:(?:U\.S\.|US)\s+)?dollars?))\b",
        re.IGNORECASE,
    )
    _CANDIDATE = re.compile(
        r"^NVIDIA Consolidated Balance Sheets in (nvidia-fy20\d{2}-annual)\. "
        r"As of ([A-Za-z]+ \d{1,2}, 20\d{2}), (.+?) was "
        r"\(?\$[\d,]+\)? million USD\.$"
    )

    def __init__(self, retriever: _Retriever, *, candidate_pool_size: int = 50) -> None:
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        self._retriever = retriever
        self._candidate_pool_size = candidate_pool_size

    @staticmethod
    def _date_key(value: str) -> str | None:
        for format_string in ("%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(value, format_string).date().isoformat()
            except ValueError:
                continue
        return None

    @staticmethod
    def _words(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        _validate_top_k(top_k)
        if self._BALANCE_SHEET_TITLE.search(query) is None:
            # A fiscal-year-plus-calendar-date lookup can match a cash-flow
            # change with the same row label and amount. Without an explicit
            # statement title, this opt-in filter cannot establish scope.
            if (
                FiscalYearRoutingRetriever._FISCAL_YEAR.search(query)
                and self._CALENDAR_DATE.search(query)
            ):
                return []
            return self._retriever.retrieve(query, top_k=top_k)

        if (
            self._USD_UNIT.search(query) is None
            or self._OTHER_USD_SCALE.search(query)
            or FinancialTableScopeRetriever._NON_USD.search(query)
        ):
            return []

        years = {int(year) for year in FiscalYearRoutingRetriever._FISCAL_YEAR.findall(query)}
        date_matches = list(self._DATE.finditer(query))
        if len(years) != 1 or len(date_matches) != 1:
            return []
        date_match = date_matches[0]
        requested_date = self._date_key(" ".join(date_match.groups()[:2]) + ", " + date_match.group(3))
        if requested_date is None:
            return []
        document_id = f"nvidia-fy{next(iter(years))}-annual"
        candidates = self._retriever.retrieve(
            query, top_k=max(top_k, self._candidate_pool_size)
        )
        normalized_query = f" {self._words(query)} "
        matching_labels: set[str] = set()
        matching: list[ScoredChunk] = []
        for candidate in candidates:
            chunk = candidate.chunk
            if chunk.document_id != document_id or ":table:" not in chunk.chunk_id:
                continue
            parsed = self._CANDIDATE.fullmatch(chunk.text)
            if parsed is None or parsed.group(1) != chunk.document_id:
                continue
            if self._date_key(parsed.group(2)) != requested_date:
                continue
            label = self._words(parsed.group(3).rstrip(" $"))
            if label and f" {label} " in normalized_query:
                matching_labels.add(label)
                matching.append(candidate)

        # Multiple plausible rows or repeated cells require manual review.
        if len(matching_labels) != 1 or len(matching) != 1:
            return []
        return matching[:top_k]


class AnnualStatementFieldRetriever:
    """Opt-in annual income/cash-flow row-and-column candidate filter.

    This matches fields serialized by the experimental PDF table extractor.
    It does not validate the extractor against the source PDF and must not be
    interpreted as answer correctness.
    """

    # This branch returns million-USD cells, not conversions or multi-unit answers.
    _OTHER_USD_SCALE = BalanceSheetFieldRetriever._OTHER_USD_SCALE
    _PER_SHARE = re.compile(r"\b(?:per[\s-]+share|EPS)\b", re.IGNORECASE)
    _REPORT = re.compile(
        r"\b(?:fiscal\s+(?:year\s+)?|FY\s*)(20\d{2})\s+annual\s+report\b",
        re.IGNORECASE,
    )
    _STATEMENT = re.compile(
        r"\bConsolidated Statements of (Income|Cash Flows)\b", re.IGNORECASE
    )
    _DATE = re.compile(
        r"\byear ended\s+([A-Za-z]+)\s+(\d{1,2}),?\s+(20\d{2})\b",
        re.IGNORECASE,
    )
    _CANDIDATE = re.compile(
        r"^NVIDIA Consolidated Statements of (Income|Cash Flows) in "
        r"(nvidia-fy20\d{2}-annual)\. For fiscal 20\d{2} "
        r"\(year ended ([A-Za-z]+ \d{1,2}, 20\d{2})\), "
        r"(.+?) was \(?\$[\d,]+\)? million USD\.$",
        re.IGNORECASE,
    )

    def __init__(self, retriever: _Retriever, *, candidate_pool_size: int = 50) -> None:
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        self._retriever = retriever
        self._candidate_pool_size = candidate_pool_size

    def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        _validate_top_k(top_k)
        reports = list(self._REPORT.finditer(query))
        statements = list(self._STATEMENT.finditer(query))
        dates = list(self._DATE.finditer(query))
        currency_query = FinancialTableScopeRetriever._USD_EXCLUSION.sub(r"\1", query)
        if (
            len(reports) != 1 or len(statements) != 1 or len(dates) != 1
            or BalanceSheetFieldRetriever._USD_UNIT.search(query) is None
            or FinancialTableScopeRetriever._NON_USD.search(currency_query)
            or self._OTHER_USD_SCALE.search(query)
            or self._PER_SHARE.search(query)
        ):
            return []
        requested_date = BalanceSheetFieldRetriever._date_key(
            " ".join(dates[0].groups()[:2]) + ", " + dates[0].group(3)
        )
        if requested_date is None:
            return []
        document_id = f"nvidia-fy{reports[0].group(1)}-annual"
        statement = statements[0].group(1).lower()
        normalized_query = f" {BalanceSheetFieldRetriever._words(query)} "
        candidates = self._retriever.retrieve(
            query, top_k=max(top_k, self._candidate_pool_size)
        )
        matches: list[tuple[str, ScoredChunk]] = []
        for candidate in candidates:
            chunk = candidate.chunk
            if chunk.document_id != document_id or ":table:" not in chunk.chunk_id:
                continue
            parsed = self._CANDIDATE.fullmatch(chunk.text)
            if (
                parsed is None or parsed.group(1).lower() != statement
                or parsed.group(2) != document_id
                or BalanceSheetFieldRetriever._date_key(parsed.group(3)) != requested_date
            ):
                continue
            label = BalanceSheetFieldRetriever._words(parsed.group(4))
            if label and f" {label} " in normalized_query:
                matches.append((label, candidate))
        # Prefer the most specific complete row label when a shorter row name
        # is only a subphrase (e.g. "Revenue" inside "Cost of revenue").
        # Duplicate cells or equally specific competing labels still refuse.
        if not matches:
            return []
        labels = {label for label, _ in matches}
        maximal_labels = {
            label for label in labels
            if not any(label != other and f" {label} " in f" {other} "
                       for other in labels)
        }
        if len(maximal_labels) != 1:
            # This branch returns one field, never a partial multi-field answer.
            return []
        longest_label = max(len(label.split()) for label, _ in matches)
        most_specific = [
            candidate
            for label, candidate in matches
            if len(label.split()) == longest_label
        ]
        return most_specific[:top_k] if len(most_specific) == 1 else []
