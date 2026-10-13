"""Runtime configuration for the loopback evidence demo."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .core import Chunk
from .retrieval import DenseRetriever, FiscalYearRoutingRetriever, TaskSourceRetriever
from .period_routing import ReportCatalog, SourceBoundPeriodRetriever, validate_report_catalog
from .sources import SourceRecord
from .service import EvidenceService

BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingModel(Protocol):
    def encode(self, texts: Sequence[str], **kwargs): ...


def build_bge_service(
    *,
    chunks: Sequence[Chunk],
    model: EmbeddingModel,
    threshold: float,
    top_k: int,
    route_fiscal_year: bool = False,
    passage_embeddings: Sequence[Sequence[float]] | np.ndarray | None = None,
    source_task: str | None = None,
    source_records: list[SourceRecord] | None = None,
    report_catalog: ReportCatalog | None = None,
) -> EvidenceService:
    """Wire BGE's passage/query conventions into the generic dense retriever."""

    if source_task is not None and (route_fiscal_year or report_catalog is not None):
        raise ValueError('source task cannot be combined with automatic report routing')
    if report_catalog is not None:
        if route_fiscal_year:
            raise ValueError('source-bound period routing cannot be combined with annual-only routing')
        validate_report_catalog(report_catalog, chunks=chunks)
    if source_task is not None and source_records is None:
        raise ValueError('source task requires reviewed source records')

    def encode_passages(texts: list[str]):
        return model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

    def encode_queries(texts: list[str]):
        return model.encode(
            [BGE_QUERY_INSTRUCTION + text for text in texts],
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

    dense_retriever = DenseRetriever(
        chunks,
        encoder=encode_passages,
        query_encoder=encode_queries,
        precomputed_embeddings=passage_embeddings,
    )
    if source_task is not None:
        retriever = TaskSourceRetriever(dense_retriever, sources=source_records, task=source_task)
    elif report_catalog is not None:
        retriever = SourceBoundPeriodRetriever(dense_retriever, report_catalog=report_catalog)
    elif route_fiscal_year:
        year_to_document_id: dict[int, str] = {}
        for chunk in chunks:
            match = re.fullmatch(
                r"nvidia-fy(20\d{2})-annual",
                chunk.document_id,
            )
            if match:
                year_to_document_id[int(match.group(1))] = chunk.document_id
        retriever = FiscalYearRoutingRetriever(
            dense_retriever,
            year_to_document_id=year_to_document_id,
        )
    else:
        retriever = dense_retriever
    return EvidenceService(
        retriever=retriever,
        threshold=threshold,
        top_k=top_k,
    )
