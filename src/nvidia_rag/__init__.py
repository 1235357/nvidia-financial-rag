"""Evidence-first NVIDIA financial-report RAG components."""

from .core import (
    Chunk,
    GroundedContext,
    ScoredChunk,
    build_grounded_context,
    chunk_pages,
    extract_pdf_pages,
    select_evidence,
)
from .retrieval import DenseRetriever, TfidfRetriever

__all__ = [
    "Chunk",
    "GroundedContext",
    "ScoredChunk",
    "build_grounded_context",
    "chunk_pages",
    "extract_pdf_pages",
    "select_evidence",
    "DenseRetriever",
    "TfidfRetriever",
]
