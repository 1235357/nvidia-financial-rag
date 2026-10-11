"""Create deterministic, inspectable artifacts from one trusted PDF."""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from .core import Chunk, chunk_pages, chunk_pages_recursive, chunk_pages_word_safe, chunk_pages_sentence_safe, extract_pdf_pages
from .sources import SourceRecord, file_sha256, require_source_operation
from .table_structure import (
    extract_quarter_income_revenue_chunks,
    extract_table_candidate_chunks,
)

ChunkStrategy = Literal['fixed_token', 'recursive', 'word_safe', 'sentence_safe']
ExtractionMode = Literal['plain', 'coordinate_sorted', 'table_structured']


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_url(source_url: str) -> None:
    parsed = urlparse(source_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("source_url must be an absolute HTTPS URL")


def _chunk_extracted_pages(
    pages: Sequence[Chunk],
    *,
    strategy: ChunkStrategy,
    max_tokens: int,
    overlap_tokens: int,
    encoding_name: str,
) -> list[Chunk]:
    if strategy == 'sentence_safe':
        if overlap_tokens != 0:
            raise ValueError('sentence_safe does not support overlap; use overlap_tokens=0')
        return chunk_pages_sentence_safe(pages, max_tokens=max_tokens, encoding_name=encoding_name)
    if strategy == 'word_safe':
        if overlap_tokens != 0:
            raise ValueError('word_safe does not support overlap; use overlap_tokens=0')
        return chunk_pages_word_safe(pages, max_tokens=max_tokens, encoding_name=encoding_name)
    if strategy == 'fixed_token':
        chunker = chunk_pages
    elif strategy == 'recursive':
        chunker = chunk_pages_recursive
    else:
        raise ValueError(f'unsupported chunk strategy: {strategy}')
    return chunker(
        pages,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        encoding_name=encoding_name,
    )


def _write_chunks(path: Path, chunks: Sequence[Chunk]) -> None:
    with path.open('w', encoding='utf-8', newline='\n') as target:
        for chunk in chunks:
            target.write(
                json.dumps(
                    {
                        'chunk_id': chunk.chunk_id,
                        'document_id': chunk.document_id,
                        'page_number': chunk.page_number,
                        'text': chunk.text,
                        **({'amount_bbox': chunk.amount_bbox} if chunk.amount_bbox is not None else {}),
                        **({'metric_bbox': chunk.metric_bbox} if chunk.metric_bbox is not None else {}),
                        **({'period_bbox': chunk.period_bbox} if chunk.period_bbox is not None else {}),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + '\n'
            )


def build_index_artifacts(
    *,
    pdf_path: Path,
    output_dir: Path,
    document_id: str,
    document_label: str,
    source_url: str,
    max_tokens: int,
    overlap_tokens: int,
    encoding_name: str = "cl100k_base",
    strategy: ChunkStrategy = 'fixed_token',
) -> dict[str, object]:
    """Extract and chunk one PDF, then write a manifest and JSONL evidence store."""
    pdf_path = pdf_path.resolve(strict=True)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("pdf_path must point to a PDF")
    if not document_id.strip() or not document_label.strip():
        raise ValueError("document_id and document_label must be non-empty")
    _validate_source_url(source_url)

    pages = extract_pdf_pages(pdf_path, document_id=document_id)
    chunks = _chunk_extracted_pages(
        pages,
        strategy=strategy,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        encoding_name=encoding_name,
    )
    if not chunks:
        raise ValueError(f'{document_id}: no searchable chunks; inspect extraction or OCR requirements')
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / "chunks.jsonl"
    _write_chunks(chunks_path, chunks)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "document_id": document_id,
        "document_label": document_label,
        "source_url": source_url,
        "source_filename": pdf_path.name,
        "source_sha256": _file_sha256(pdf_path),
        "page_count": len(pages),
        "nonempty_page_count": sum(bool(page.text) for page in pages),
        "chunk_count": len(chunks),
        "chunking": {
            "encoding_name": encoding_name,
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
            "strategy": strategy,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_corpus_artifacts(
    *,
    sources: Sequence[SourceRecord],
    source_dir: Path,
    output_dir: Path,
    strategy: ChunkStrategy,
    max_tokens: int,
    overlap_tokens: int,
    encoding_name: str = 'cl100k_base',
    extraction_mode: ExtractionMode = 'plain',
) -> dict[str, object]:
    """Build one deterministic, multi-document evidence store."""
    if not sources:
        raise ValueError('at least one pinned source is required')
    for source in sources:
        require_source_operation(source, 'index')
    document_ids: set[str] = set()
    for source in sources:
        if source.document_id in document_ids:
            raise ValueError(f'duplicate document_id: {source.document_id}')
        document_ids.add(source.document_id)
    if any(source.sha256 is None for source in sources):
        raise ValueError('all corpus sources must have a pinned sha256')
    if extraction_mode not in {'plain', 'coordinate_sorted', 'table_structured'}:
        raise ValueError('unsupported PDF extraction mode')
    protected_default = Path(__file__).resolve().parents[2] / 'artifacts' / 'annual-fy2022-fy2026-fixed-400-50'
    if output_dir.resolve() == protected_default.resolve() and any(
        source.provenance is not None
        and source.provenance['source_type'] != 'official_filing'
        for source in sources
    ):
        raise ValueError('third-party sources cannot overwrite the default demo index')
    if extraction_mode != 'plain' and output_dir.resolve() == protected_default.resolve():
        raise ValueError('experimental extraction cannot overwrite the default demo index')

    all_chunks: list[Chunk] = []
    documents: list[dict[str, object]] = []
    total_pages = 0
    total_nonempty_pages = 0
    for source in sources:
        pdf_path = (source_dir / source.filename).resolve(strict=True)
        observed_hash = file_sha256(pdf_path)
        if observed_hash != source.sha256:
            raise ValueError(f'source hash mismatch for {source.document_id}')
        pages = extract_pdf_pages(
            pdf_path,
            document_id=source.document_id,
            extraction_mode='plain' if extraction_mode == 'table_structured' else extraction_mode,
        )
        chunks = _chunk_extracted_pages(
            pages,
            strategy=strategy,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            encoding_name=encoding_name,
        )
        table_candidate_count = 0
        if extraction_mode == 'table_structured':
            candidates = extract_table_candidate_chunks(
                pdf_path, document_id=source.document_id
            )
            candidates.extend(
                extract_quarter_income_revenue_chunks(
                    pdf_path,
                    document_id=source.document_id,
                    metrics=("Revenue", "Operating income"),
                )
            )
            table_candidate_count = len(candidates)
            chunks.extend(candidates)
        if not chunks:
            raise ValueError(f'{source.document_id}: no searchable chunks; inspect extraction or OCR requirements')
        all_chunks.extend(chunks)
        nonempty_pages = sum(bool(page.text) for page in pages)
        total_pages += len(pages)
        total_nonempty_pages += nonempty_pages
        documents.append(
            {
                'document_id': source.document_id,
                'document_label': source.label,
                'source_url': source.url,
                'source_filename': source.filename,
                'source_sha256': observed_hash,
                **({'provenance': dict(source.provenance)} if source.provenance is not None else {}),
                **({'reporting_period': dict(source.reporting_period)} if source.reporting_period is not None else {}),
                'page_count': len(pages),
                'nonempty_page_count': nonempty_pages,
                'chunk_count': len(chunks),
                **({'table_candidate_count': table_candidate_count} if extraction_mode == 'table_structured' else {}),
            }
        )

    if len({chunk.chunk_id for chunk in all_chunks}) != len(all_chunks):
        raise ValueError('chunk identifiers must be unique across the corpus')
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_chunks(output_dir / 'chunks.jsonl', all_chunks)
    manifest: dict[str, object] = {
        'schema_version': 1,
        'chunks_sha256': file_sha256(output_dir / 'chunks.jsonl'),
        'document_count': len(documents),
        'page_count': total_pages,
        'nonempty_page_count': total_nonempty_pages,
        'chunk_count': len(all_chunks),
        'chunking': {
            'encoding_name': encoding_name,
            'max_tokens': max_tokens,
            'overlap_tokens': overlap_tokens,
            'strategy': strategy,
        },
        'documents': documents,
    }
    if extraction_mode != 'plain':
        manifest['extraction_mode'] = extraction_mode
    if extraction_mode == 'table_structured':
        manifest['table_candidate_count'] = sum(
            int(document['table_candidate_count']) for document in documents
        )
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return manifest


def load_chunks(path: Path) -> list[Chunk]:
    """Load a previously generated JSONL evidence store."""
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                chunk = Chunk(
                    chunk_id=str(record["chunk_id"]),
                    document_id=str(record["document_id"]),
                    page_number=int(record["page_number"]),
                    text=str(record["text"]),
                    amount_bbox=tuple(record['amount_bbox']) if record.get('amount_bbox') is not None else None,
                    metric_bbox=tuple(record['metric_bbox']) if record.get('metric_bbox') is not None else None,
                    period_bbox=tuple(record['period_bbox']) if record.get('period_bbox') is not None else None,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid chunk record at line {line_number}") from error
            chunks.append(chunk)
    if not chunks:
        raise ValueError("chunk artifact is empty")
    return chunks
