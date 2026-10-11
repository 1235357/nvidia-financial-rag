"""Small, testable primitives for an evidence-first RAG baseline."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
import math
from typing import Iterable, Literal, Sequence

import fitz
import tiktoken


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    page_number: int
    text: str
    amount_bbox: tuple[float, float, float, float] | None = None
    metric_bbox: tuple[float, float, float, float] | None = None
    period_bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        for name in ('amount_bbox', 'metric_bbox', 'period_bbox'):
            bbox = getattr(self, name)
            if bbox is None:
                continue
            if (len(bbox) != 4 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 for value in bbox
            ) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
                raise ValueError(f'{name} must be a finite, nonnegative, positive-area rectangle')


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class GroundedContext:
    can_answer: bool
    message: str
    context: str
    citations: tuple[str, ...]
    evidence: tuple[ScoredChunk, ...]


def extract_pdf_pages(
    pdf_path: Path,
    *,
    document_id: str,
    extraction_mode: Literal["plain", "coordinate_sorted"] = "plain",
) -> list[Chunk]:
    """Extract text page-by-page while retaining one-based PDF page numbers."""
    if extraction_mode not in {"plain", "coordinate_sorted"}:
        raise ValueError("unsupported PDF extraction mode")
    pages: list[Chunk] = []
    with fitz.open(pdf_path) as document:
        for zero_based_page, page in enumerate(document):
            page_number = zero_based_page + 1
            text = page.get_text(
                "text", sort=extraction_mode == "coordinate_sorted"
            ).strip()
            pages.append(
                Chunk(
                    chunk_id=f"{document_id}:page:{page_number}",
                    document_id=document_id,
                    page_number=page_number,
                    text=text,
                )
            )
    return pages


def chunk_pages(
    pages: Iterable[Chunk],
    *,
    max_tokens: int,
    overlap_tokens: int = 0,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Split each page into deterministic token windows without losing provenance."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be in [0, max_tokens)")

    encoding = tiktoken.get_encoding(encoding_name)
    stride = max_tokens - overlap_tokens
    chunks: list[Chunk] = []

    for page in pages:
        tokens = encoding.encode(page.text)
        for start in range(0, len(tokens), stride):
            window = tokens[start : start + max_tokens]
            if not window:
                continue
            text = encoding.decode(window).strip()
            if not text:
                continue
            digest = sha256(
                f"{page.document_id}|{page.page_number}|{start}|{text}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=f"{page.document_id}:p{page.page_number}:{digest}",
                    document_id=page.document_id,
                    page_number=page.page_number,
                    text=text,
                )
            )
            if start + max_tokens >= len(tokens):
                break
    return chunks


def chunk_pages_word_safe(
    pages: Iterable[Chunk], *, max_tokens: int,
    encoding_name: str = 'cl100k_base',
) -> list[Chunk]:
    """Experimental whitespace-boundary packing; no overlap or sentence guarantee.

    Re-encode exact source substrings instead of decoding partial BPE windows.
    A whitespace-delimited unit that cannot fit is refused, never truncated.
    Existing token-window strategies and frozen indexes remain unchanged.
    """
    if max_tokens <= 0:
        raise ValueError('max_tokens must be positive')
    encoding = tiktoken.get_encoding(encoding_name)
    chunks: list[Chunk] = []
    for page in pages:
        words = list(re.finditer(r'\S+', page.text))
        start = 0
        while start < len(words):
            end = start
            while end < len(words):
                candidate = page.text[words[start].start():words[end].end()]
                if len(encoding.encode(candidate)) > max_tokens:
                    break
                end += 1
            if end == start:
                raise ValueError(f'word exceeds token budget in {page.document_id} page {page.page_number}')
            text = page.text[words[start].start():words[end - 1].end()]
            digest = sha256(
                f'{page.document_id}|{page.page_number}|word_safe|{words[start].start()}|{text}'.encode('utf-8')
            ).hexdigest()[:16]
            chunks.append(Chunk(f'{page.document_id}:p{page.page_number}:w:{digest}',
                                page.document_id, page.page_number, text))
            start = end
    return chunks


def chunk_pages_sentence_safe(
    pages: Iterable[Chunk], *, max_tokens: int,
    encoding_name: str = 'cl100k_base',
) -> list[Chunk]:
    """Experimental punctuation-unit packing, without token-window fallback.

    A detected unit ends at .!? with optional closing quotes/brackets followed
    by whitespace, or at the page end.
    Common titles and dotted abbreviations are kept together conservatively.
    This is a heuristic, not linguistic sentence recognition: abbreviations,
    extracted tables and sentences spanning PDF pages need separate review.
    Exact source substrings are retained; oversized units fail explicitly.
    """
    if max_tokens <= 0:
        raise ValueError('max_tokens must be positive')
    encoding = tiktoken.get_encoding(encoding_name)
    chunks: list[Chunk] = []
    for page in pages:
        spans: list[tuple[int, int]] = []
        start = 0
        for boundary in re.finditer(r'''[.!?]["'”’)\]]*(?P<space>\s+)''', page.text):
            preceding = page.text[:boundary.start() + 1].rstrip().split()[-1]
            if (preceding.lower() in {'dr.', 'mr.', 'mrs.', 'ms.', 'prof.',
                                      'corp.', 'inc.', 'ltd.', 'co.'}
                    or re.fullmatch(r'(?:[A-Za-z]\.){2,}', preceding)):
                continue
            spans.append((start, boundary.start('space')))
            start = boundary.end()
        spans.append((start, len(page.text)))
        packed_start: int | None = None
        packed_end = 0

        def emit() -> None:
            if packed_start is None:
                return
            text = page.text[packed_start:packed_end].strip()
            digest = sha256(
                f'{page.document_id}|{page.page_number}|sentence_safe|{packed_start}|{text}'.encode('utf-8')
            ).hexdigest()[:16]
            chunks.append(Chunk(f'{page.document_id}:p{page.page_number}:s:{digest}',
                                page.document_id, page.page_number, text))

        for unit_start, unit_end in spans:
            unit = page.text[unit_start:unit_end].strip()
            if not unit:
                continue
            if len(encoding.encode(unit)) > max_tokens:
                raise ValueError(
                    f'sentence unit exceeds token budget in {page.document_id} page {page.page_number}'
                )
            if packed_start is not None:
                candidate = page.text[packed_start:unit_end].strip()
                if len(encoding.encode(candidate)) > max_tokens:
                    emit()
                    packed_start = None
            if packed_start is None:
                packed_start = unit_start
            packed_end = unit_end
        emit()
    return chunks


def _recursive_units(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    encoding,
    document_id: str,
    page_number: int,
) -> list[str]:
    """Split paragraphs, sentences, then complete whitespace-delimited words."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r'\n\s*\n+', text)
        if paragraph.strip()
    ]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(encoding.encode(paragraph)) <= max_tokens:
            units.append(paragraph)
            continue

        sentences = [
            sentence.strip()
            for sentence in re.split(r'(?<=[.!?])\s+', paragraph)
            if sentence.strip()
        ]
        for sentence in sentences:
            tokens = encoding.encode(sentence)
            if len(tokens) <= max_tokens:
                units.append(sentence)
                continue
            # Never decode a partial BPE word/Unicode sequence. An oversized
            # sentence may still be split between words; callers requiring
            # complete sentences must use sentence_safe instead. This fallback
            # does not promise token-exact overlap within the sentence.
            units.extend(
                chunk.text for chunk in chunk_pages_word_safe(
                    [Chunk("fallback", document_id, page_number, sentence)],
                    max_tokens=max_tokens, encoding_name=encoding.name,
                )
            )
    return units


def chunk_pages_recursive(
    pages: Iterable[Chunk],
    *,
    max_tokens: int,
    overlap_tokens: int = 0,
    encoding_name: str = 'cl100k_base',
) -> list[Chunk]:
    """Pack paragraph/sentence units without crossing PDF page boundaries.

    Normal prose is kept whole whenever it fits the budget. A single oversized
    sentence or table-like block falls back to complete-word packing. A word
    exceeding the budget is refused. This is not sentence/table-safe and does
    not guarantee token-exact overlap for oversized sentence fallback.
    """
    if max_tokens <= 0:
        raise ValueError('max_tokens must be positive')
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError('overlap_tokens must be in [0, max_tokens)')

    encoding = tiktoken.get_encoding(encoding_name)
    chunks: list[Chunk] = []

    def token_count(units: Sequence[str]) -> int:
        return len(encoding.encode('\n\n'.join(units)))

    for page in pages:
        units = _recursive_units(
            page.text,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            encoding=encoding,
            document_id=page.document_id,
            page_number=page.page_number,
        )
        current: list[str] = []
        chunk_ordinal = 0

        def emit(materialized: Sequence[str]) -> None:
            nonlocal chunk_ordinal
            text = '\n\n'.join(materialized).strip()
            if not text:
                return
            digest = sha256(
                (
                    f'{page.document_id}|{page.page_number}|recursive|'
                    f'{chunk_ordinal}|{text}'
                ).encode('utf-8')
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=f'{page.document_id}:p{page.page_number}:r:{digest}',
                    document_id=page.document_id,
                    page_number=page.page_number,
                    text=text,
                )
            )
            chunk_ordinal += 1

        for unit in units:
            proposed = [*current, unit]
            if not current or token_count(proposed) <= max_tokens:
                current = proposed
                continue

            previous = list(current)
            emit(previous)
            overlap: list[str] = []
            if overlap_tokens:
                for candidate in reversed(previous):
                    proposed_overlap = [candidate, *overlap]
                    if token_count(proposed_overlap) > overlap_tokens:
                        break
                    overlap = proposed_overlap
            current = [*overlap, unit]
            if token_count(current) > max_tokens:
                current = [unit]

        if current:
            emit(current)

    return chunks


def select_evidence(
    candidates: Iterable[ScoredChunk], *, threshold: float, top_k: int
) -> list[ScoredChunk]:
    """Return only sufficiently similar evidence, ordered by score."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    eligible = (candidate for candidate in candidates if candidate.score >= threshold)
    return sorted(eligible, key=lambda candidate: candidate.score, reverse=True)[:top_k]


def build_grounded_context(
    query: str, *, evidence: Sequence[ScoredChunk]
) -> GroundedContext:
    """Build a citation-bearing context or refuse when retrieval produced no evidence."""
    if not evidence:
        return GroundedContext(
            can_answer=False,
            message="I cannot answer from the provided reports because no evidence passed the retrieval gate.",
            context="",
            citations=(),
            evidence=(),
        )

    citations = tuple(
        f"{item.chunk.document_id} p.{item.chunk.page_number}" for item in evidence
    )
    blocks = [
        f"[{citation}]\n{item.chunk.text}"
        for item, citation in zip(evidence, citations, strict=True)
    ]
    context = "\n\n".join(blocks)
    return GroundedContext(
        can_answer=True,
        message=f"Evidence retrieved for: {query}",
        context=context,
        citations=citations,
        evidence=tuple(evidence),
    )
