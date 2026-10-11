"""Small, read-only Gradio surface for inspecting retrieval evidence."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
from math import isfinite, sqrt
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Protocol

import fitz
import gradio as gr
import tiktoken

from .core import Chunk, GroundedContext, ScoredChunk, chunk_pages
from .conversation import resolve_demo_query
from .generation import GeneratedResponse, build_grounded_prompt, numeric_support_trace
from .runtime import BGE_QUERY_INSTRUCTION
from .service import EvidenceInspection
from .table_structure import extract_table_candidate_chunks
from .sources import _reviewed_provenance


class QueryService(Protocol):
    def query(self, query: str) -> GroundedContext | GeneratedResponse: ...


class LexicalRetriever(Protocol):
    def retrieve_from_documents(
        self, query: str, *, top_k: int, document_ids: set[str] | None
    ) -> list[ScoredChunk]: ...


def _display_provenance(metadata: Mapping[str, object]) -> Mapping[str, object]:
    row = dict(metadata)
    if isinstance(row.get('permitted_operations'), tuple):
        row['permitted_operations'] = list(row['permitted_operations'])
    reviewed = _reviewed_provenance(row)
    if 'display' not in reviewed['permitted_operations']:
        raise ValueError('source display permission missing')
    if 'index' not in reviewed['permitted_operations']:
        raise ValueError('source index permission missing')
    return reviewed


def load_source_provenance(manifest_path: Path) -> dict[str, Mapping[str, object]]:
    """Validate declared display use before launching the bounded demo."""
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    result = {}
    for record in manifest['documents']:
        if 'provenance' in record:
            result[str(record['document_id'])] = _display_provenance(record['provenance'])
    return result


def load_source_documents(
    manifest_path: Path, *, source_root: Path
) -> dict[str, tuple[Path, str]]:
    """Resolve only basename PDF references from the local frozen manifest."""
    load_source_provenance(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources: dict[str, tuple[Path, str]] = {}
    for record in manifest["documents"]:
        filename = str(record["source_filename"])
        if (filename != Path(filename).name or "\\" in filename
                or not filename.lower().endswith(".pdf")):
            raise ValueError("source filename must be a local PDF basename")
        document_id = str(record["document_id"])
        expected_hash = str(record["source_sha256"])
        sources[document_id] = (source_root / filename, expected_hash)
    return sources


@dataclass(frozen=True, slots=True)
class IngestionReplay:
    """One source-page replay, never a claim that indexing ran during this query."""

    verified: bool
    reason: str
    chunk_id: str
    page_number: int
    page_token_count: int = 0
    page_chunk_count: int = 0
    window_start: int = 0
    window_end: int = 0
    overlap_tokens: int = 0
    overlap_text: str = ""
    source_sha256: str = ""
    source_excerpt: str = ""
    window_excerpt: str = ""
    page_windows: tuple[tuple[str, int, int], ...] = ()
    shared_boundary_text: str = ""
    replay_kind: str = "text-window"
    table_candidate_text: str = ""
    table_page_image: str = ""
    period_highlighted: bool = False


def replay_ingestion_window(
    top_chunk: Chunk,
    *,
    pdf_path: Path,
    expected_sha256: str,
    indexed_chunks: Sequence[Chunk],
) -> IngestionReplay:
    """Verify the frozen top chunk against its hashed PDF and exact page split."""
    def refuse(reason: str) -> IngestionReplay:
        return IngestionReplay(False, reason, top_chunk.chunk_id, top_chunk.page_number)

    if not expected_sha256 or len(expected_sha256) != 64:
        return refuse("source manifest hash is unavailable")
    try:
        digest = sha256()
        with pdf_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        observed_hash = digest.hexdigest()
        if observed_hash != expected_sha256.lower():
            return refuse("source PDF hash differs from the frozen manifest")
        with fitz.open(pdf_path) as document:
            if not 1 <= top_chunk.page_number <= len(document):
                return refuse("PDF page number is outside the source document")
            page_text = document[top_chunk.page_number - 1].get_text(
                "text", sort=False
            ).strip()
    except (OSError, RuntimeError, ValueError):
        return refuse("source PDF could not be read for a verified replay")

    if ":table:" in top_chunk.chunk_id:
        if top_chunk not in indexed_chunks:
            return refuse("table candidate is absent from the frozen index")
        try:
            extracted = extract_table_candidate_chunks(
                pdf_path,
                document_id=top_chunk.document_id,
                page_numbers={top_chunk.page_number},
            )
        except (OSError, RuntimeError, ValueError):
            return refuse("table candidate could not be re-extracted")
        if top_chunk not in extracted:
            legacy_match = any(
                candidate.chunk_id == top_chunk.chunk_id
                and candidate.document_id == top_chunk.document_id
                and candidate.page_number == top_chunk.page_number
                and candidate.text == top_chunk.text
                and all(
                    getattr(top_chunk, name) is None
                    or getattr(top_chunk, name) == getattr(candidate, name)
                    for name in ('amount_bbox', 'metric_bbox', 'period_bbox')
                )
                for candidate in extracted
            )
            if legacy_match and (
                top_chunk.amount_bbox is None or top_chunk.metric_bbox is None
                or top_chunk.period_bbox is None
            ):
                return refuse(
                    "table candidate coordinate metadata is missing; "
                    "rebuild the structured index from the hashed PDF"
                )
            return refuse("table candidate differs from the hashed PDF re-extraction")
        table_page_image = ""
        if top_chunk.amount_bbox is not None and top_chunk.metric_bbox is not None:
            with fitz.open(pdf_path) as document:
                source_page = document[top_chunk.page_number - 1]
                for bbox, color in (
                    (top_chunk.amount_bbox, (0.9, 0.25, 0.1)),
                    (top_chunk.metric_bbox, (0.1, 0.35, 0.9)),
                    (top_chunk.period_bbox, (0.1, 0.65, 0.2)),
                ):
                    if bbox is None:
                        continue
                    rectangle = fitz.Rect(bbox)
                    if not source_page.rect.contains(rectangle):
                        return refuse("table candidate coordinates are outside the source page")
                    source_page.draw_rect(rectangle, color=color, width=1.5)
                # Draw using the extraction coordinate system before expanding
                # the viewport. Some annual PDFs hide the header in CropBox.
                # This changes only the in-memory replay, never the source file.
                source_page.set_cropbox(source_page.mediabox)
                png = source_page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False).tobytes("png")
                table_page_image = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        return IngestionReplay(
            True,
            "source hash and table candidate match the frozen index",
            top_chunk.chunk_id,
            top_chunk.page_number,
            page_chunk_count=len(extracted),
            source_sha256=observed_hash,
            source_excerpt=page_text[:300],
            replay_kind="table-candidate",
            table_candidate_text=top_chunk.text,
            table_page_image=table_page_image,
            period_highlighted=bool(table_page_image and top_chunk.period_bbox is not None),
        )

    page = Chunk(
        f"{top_chunk.document_id}:page:{top_chunk.page_number}",
        top_chunk.document_id,
        top_chunk.page_number,
        page_text,
    )
    replayed = chunk_pages([page], max_tokens=400, overlap_tokens=50)
    frozen = [
        chunk for chunk in indexed_chunks
        if chunk.document_id == top_chunk.document_id
        and chunk.page_number == top_chunk.page_number
    ]
    if replayed != frozen or top_chunk not in replayed:
        return refuse("replayed windows differ from the frozen chunk index")

    encoding = tiktoken.get_encoding("cl100k_base")
    page_tokens = encoding.encode(page_text)
    ordinal = replayed.index(top_chunk)
    start = ordinal * 350
    end = min(start + 400, len(page_tokens))
    overlap = 50 if ordinal else 0
    spans = tuple(
        (chunk.chunk_id, index * 350, min(index * 350 + 400, len(page_tokens)))
        for index, chunk in enumerate(replayed)
    )
    boundary_start = ordinal * 350 if ordinal else 350
    shared_boundary = (
        encoding.decode(page_tokens[boundary_start : boundary_start + 50])
        if len(replayed) > 1 else ""
    )
    return IngestionReplay(
        True,
        "source hash, page text, and all page chunks match the frozen index",
        top_chunk.chunk_id,
        top_chunk.page_number,
        len(page_tokens),
        len(replayed),
        start,
        end,
        overlap,
        encoding.decode(page_tokens[start : start + overlap]) if overlap else "",
        observed_hash,
        page_text[:300],
        top_chunk.text[:300],
        spans,
        shared_boundary,
    )


def replay_top_candidate(
    inspected: EvidenceInspection,
    *,
    source_documents: Mapping[str, tuple[Path, str]],
    indexed_chunks: Sequence[Chunk],
) -> IngestionReplay | None:
    """Choose the trusted PDF for this query's actual first ranked candidate."""
    if not inspected.retrieval.candidates:
        return None
    top_chunk = inspected.retrieval.candidates[0].chunk
    source = source_documents.get(top_chunk.document_id)
    if source is None:
        return IngestionReplay(
            False, "no trusted source PDF entry for this candidate",
            top_chunk.chunk_id, top_chunk.page_number,
        )
    return replay_ingestion_window(
        top_chunk,
        pdf_path=source[0],
        expected_sha256=source[1],
        indexed_chunks=indexed_chunks,
    )


_LAB_STYLE = """
<style>
.rag-lab{font-family:Georgia,serif;color:#243027;background:#f3f1e9;border:1px solid #c7c9b9;border-radius:13px;padding:22px;box-shadow:0 12px 30px #18251b16}
.rag-lab *{box-sizing:border-box}.rag-lab p,.rag-lab b,.rag-lab strong,.rag-lab .body,.rag-lab code,.rag-lab pre,.rag-lab td,.rag-lab th,.rag-lab .source,.rag-lab .score{color:#243027!important}.rag-lab .eyebrow{font:700 11px/1.4 ui-monospace,monospace;letter-spacing:.18em;color:#3f6b3b!important;text-transform:uppercase}
.rag-lab h2{font:700 29px/1.1 Georgia,serif;margin:7px 0 9px;color:#243027!important}.rag-lab .lede{color:#566252;margin:0 0 16px;font-size:14px;line-height:1.5}
.rag-lab h3,.rag-lab li{color:#243027!important}
.rag-lab details{border-top:1px solid #c7c9b9;padding:12px 0}.rag-lab summary{cursor:pointer;font-weight:700;font-size:16px;list-style:none;display:flex;align-items:center;gap:10px;color:#243027!important}.rag-lab summary::-webkit-details-marker{display:none}
.rag-lab .num{font:700 12px ui-monospace,monospace;background:#20362b;color:#eaf0d9!important;padding:5px 7px;border-radius:4px}.rag-lab .body{padding:9px 0 3px 37px;font-size:14px;line-height:1.55}
.rag-lab .metric-grid{display:flex;flex-wrap:wrap;gap:8px}.rag-lab .metric{background:#fffef9;border:1px solid #d5d8c8;border-radius:7px;padding:8px 11px;min-width:125px}.rag-lab .metric b{display:block;font:700 18px ui-monospace,monospace;color:#214329!important}.rag-lab .metric small{color:#627064!important}
.rag-lab code,.rag-lab pre{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.rag-lab code{overflow-wrap:anywhere}.rag-lab pre{white-space:pre-wrap;background:#e8ecdf;padding:11px;border-left:3px solid #7ca646;border-radius:3px;font-size:12px}
.rag-lab .row{display:grid;grid-template-columns:minmax(150px,1fr) 70px 65px;gap:10px;align-items:center;padding:7px 0;border-bottom:1px dashed #cbd0c0}.rag-lab .row .source{overflow-wrap:anywhere}.rag-lab .score{font:700 13px ui-monospace,monospace;text-align:right}.rag-lab .pass{color:#316e36!important}.rag-lab .fail{color:#9f4d32!important}
.rag-lab .candidate-excerpt{font:12px/1.45 ui-monospace,monospace;color:#566252!important;margin:4px 0;max-width:75ch}
.rag-lab .bar{height:7px;background:#dce3d5;border-radius:9px;margin-top:4px}.rag-lab .bar i{display:block;background:#83aa4e;height:100%;border-radius:9px}.rag-lab .note{background:#e8ecdf;border-left:3px solid #b87741;padding:9px 11px;color:#485247}
.rag-lab .hist-bar{display:block;width:100%;height:10px;background:#dce3d5;border-radius:9px}.rag-lab .hist-bar i{display:block;height:100%;background:#698f68;border-radius:9px}.rag-lab .histogram td:nth-child(3){width:42%}
.rag-lab table{width:100%;border-collapse:collapse;font:12px ui-monospace,monospace}.rag-lab th,.rag-lab td{text-align:right;border-bottom:1px solid #d4d9ca;padding:5px}.rag-lab th:first-child,.rag-lab td:first-child{text-align:left}
.rag-lab .practice{font:700 12px/1.5 ui-monospace,monospace;color:#3f6b3b!important;letter-spacing:.02em}.rag-lab .prompt{margin-top:10px;background:#e8ecdf;border-left:3px solid #83aa4e;padding:9px 11px}.rag-lab .matrix-ledger{margin-top:12px;border:1px solid #c7c9b9;border-radius:7px;padding:9px}.rag-lab .matrix-ledger summary{font-size:13px}.rag-lab .ledger-scroll{max-height:310px;overflow:auto;margin-top:8px}.rag-lab .excerpt{max-height:130px;overflow:auto}.rag-lab details[open]>summary{color:#214329!important}
.rag-lab #evaluation-suite .ledger-scroll table{min-width:820px}.rag-lab #evaluation-suite th,.rag-lab #evaluation-suite td{white-space:nowrap}.rag-lab #evaluation-suite th:first-child,.rag-lab #evaluation-suite td:first-child{min-width:215px}
.rag-lab #evaluation-suite th:nth-child(2),.rag-lab #evaluation-suite td:nth-child(2){white-space:normal;min-width:160px;max-width:210px;overflow-wrap:anywhere;text-align:left}
.rag-lab .source-replay{margin-top:12px;border:1px solid #bbc9b5;border-radius:7px;background:#fffef9;padding:10px}.rag-lab .source-replay summary{font-size:13px}.rag-lab .source-replay pre{max-height:145px;overflow:auto}.rag-lab .source-replay .verified{color:#316e36!important;font-weight:700}
@media(max-width:600px){.rag-lab{padding:14px}.rag-lab .body{padding-left:0}.rag-lab .row{grid-template-columns:minmax(110px,1fr) 60px 45px}}
</style>"""


def _render_evaluation_suite(
    report: Mapping[str, object] | None,
    *,
    dataset_sha256: str,
    dataset_label: str,
    top_k: int,
    threshold: float,
) -> str:
    """Explain an at-launch development-suite run without implying answer accuracy."""
    if report is None:
        return '<p class="note">No at-launch retrieval evaluation was configured for this run.</p>'
    summary = report.get("summary")
    records = report.get("records")
    if not isinstance(summary, Mapping) or not isinstance(records, list):
        raise ValueError("evaluation report lacks summary or records")
    if summary.get("question_count") != len(records) or not records:
        raise ValueError("evaluation report question count does not match records")
    if len(dataset_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in dataset_sha256):
        raise ValueError("evaluation report requires the benchmark file SHA-256")

    answerable = [item for item in records if isinstance(item, Mapping) and item.get("answerable") is True]
    unanswerable = [item for item in records if isinstance(item, Mapping) and item.get("answerable") is False]
    if len(answerable) + len(unanswerable) != len(records):
        raise ValueError("evaluation report contains an invalid question record")
    if summary.get("answerable_count") != len(answerable) or summary.get("unanswerable_count") != len(unanswerable):
        raise ValueError("evaluation report class counts do not match records")

    page_hits = sum(item.get("page_hit") is True for item in answerable)
    top_gold = sum(item.get("top_evidence_gold_hit") is True for item in answerable)
    false_accepts = sum(item.get("gate_accepted") is True for item in unanswerable)
    recalls = [item.get("page_recall_at_k") for item in answerable]
    precisions = [item.get("gold_page_chunk_precision_at_k") for item in answerable]
    ranks = [item.get("first_gold_rank") for item in answerable]
    if any(not isinstance(recall, (float, int)) or not isfinite(recall) or not 0 <= recall <= 1 for recall in recalls):
        raise ValueError("evaluation report contains an invalid page recall")
    if any(not isinstance(precision, (float, int)) or not isfinite(precision) or not 0 <= precision <= 1 for precision in precisions):
        raise ValueError("evaluation report contains an invalid gold-page chunk precision")
    if any(rank is not None and (not isinstance(rank, int) or not 1 <= rank <= top_k) for rank in ranks):
        raise ValueError("evaluation report contains an invalid first gold rank")
    no_positives = 'Not measured · no answerable questions'
    page_recall = f'{sum(recalls) / len(recalls):.3f}' if recalls else no_positives
    page_chunk_precision = f'{sum(precisions) / len(precisions):.3f}' if precisions else no_positives
    mrr = f'{sum(1 / rank for rank in ranks if rank is not None) / len(ranks):.3f}' if ranks else no_positives
    page_hit_display = f'{page_hits} / {len(answerable)}' if answerable else no_positives
    top_gold_display = f'{top_gold} / {len(answerable)}' if answerable else no_positives
    false_accept_display = (
        f'{false_accepts} / {len(unanswerable)} false accepts'
        if unanswerable else 'Not measured · no unsupported questions'
    )
    rows = "".join(
        '<tr><td><code>' + escape(str(item.get("question_id", ""))) + '</code></td>'
        + '<td><code>' + escape(str(item.get("page_gate_diagnostic", "not recorded"))) + '</code></td>'
        + '<td>' + ("answerable" if item["answerable"] else "unsupported") + '</td>'
        + '<td>' + (str(item.get("first_gold_rank") or "—") if item["answerable"] else "—") + '</td>'
        + '<td>' + ("PASS" if item.get("gate_accepted") is True else "REFUSE") + '</td>'
        + '<td>' + (
            "yes" if item.get("top_evidence_gold_hit") is True else
            "no" if item["answerable"] else "n/a"
        ) + '</td>'
        + '<td>' + (
            "yes · number only" if item.get("expected_number_in_top_evidence") is True else
            "no" if item.get("expected_number_in_top_evidence") is False else
            "n/a"
        ) + '</td></tr>'
        for item in records
    )
    false_accept_example = next(
        (item for item in unanswerable if item.get("gate_accepted") is True), None
    )
    missed_example = next(
        (item for item in answerable if item.get("top_evidence_gold_hit") is False), None
    )
    if false_accept_example is not None:
        practice = (
            'Investigate <code>'
            + escape(str(false_accept_example.get("question_id", "")))
            + '</code>: why did an unsupported question pass the similarity gate? '
            'Check scope and the original PDF before proposing a fix.'
        )
    elif missed_example is not None:
        practice = (
            'Investigate <code>'
            + escape(str(missed_example.get("question_id", "")))
            + '</code>: compare the gold-page ranking with the first accepted evidence.'
        )
    else:
        practice = (
            'Choose one question above and verify its field, fiscal-year column, '
            'unit, and amount in the original PDF.'
        )
    return (
        '<details class="matrix-ledger" id="evaluation-suite"><summary>'
        f'At-launch retrieval development suite · {len(records)} questions</summary>'
        '<p class="note">This is a small development suite already used in error analysis, '
        'not an independently held-out estimate. Labels are at PDF-page level; '
        'the optional number-occurrence diagnostic is not a field, year-column, '
        'unit, or answer-correctness check; generation correctness is not measured. '
        'Gate labels describe retrieval selection, not generated answers. '
        'Gold-page chunk precision counts duplicate chunks from one labeled page '
        'as separate hits and is not field-level entailment. '
        'The similarity threshold was not '
        'independently calibrated. This table was recomputed when this service started, '
        'not by the individual query shown above.</p>'
        f'<p>Question file: <code>{escape(dataset_label)}</code> · '
        f'SHA-256 <code>{dataset_sha256.lower()}</code> · Top-{top_k} · threshold {threshold:.2f}.</p>'
        '<div class="metric-grid">'
        f'<div class="metric"><b>{page_hit_display}</b><small>gold-page hit@{top_k} (answerable)</small></div>'
        f'<div class="metric"><b>{page_recall}</b><small>mean page recall@{top_k}</small></div>'
        f'<div class="metric"><b>{page_chunk_precision}</b><small>gold-page chunk precision@{top_k}</small></div>'
        f'<div class="metric"><b>{mrr}</b><small>first-gold-page MRR</small></div>'
        f'<div class="metric"><b>{top_gold_display}</b><small>top accepted evidence on gold page</small></div>'
        f'<div class="metric"><b>{false_accept_display}</b>'
        '<small>unsupported questions</small></div></div>'
        '<div class="ledger-scroll"><table><tr><th>question ID</th><th>page / gate diagnostic</th><th>label</th>'
        '<th>first gold rank</th><th>gate</th><th>top accepted on gold page?</th>'
        '<th>expected number in top evidence?</th></tr>'
        + rows + '</table></div>'
        '<p class="practice">Page-level evidence can still contain the wrong field, '
        'date column, unit, or amount. ' + practice + '</p></details>'
    )


def render_pipeline_walkthrough(
    inspected: EvidenceInspection,
    *,
    corpus_documents: int,
    corpus_chunks: int,
    indexed_pages: int = 0,
    table_cell_chunks: int = 0,
    ingestion_replay: IngestionReplay | None = None,
    evaluation_report: Mapping[str, object] | None = None,
    evaluation_dataset_sha256: str = "",
    evaluation_dataset_label: str = "annual_multiyear_heldout.jsonl",
    lexical_candidates: Sequence[ScoredChunk] | None = None,
    generation_enabled: bool = False,
    source_provenance: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    """Render only values computed by the live dense retrieval inspection."""
    if not 0 <= table_cell_chunks <= corpus_chunks:
        raise ValueError("table_cell_chunks must be within the stored chunk count")
    retrieval = inspected.retrieval
    reviewed_sources = {document_id: _display_provenance(metadata)
                        for document_id, metadata in (source_provenance or {}).items()}
    provenance_rows = []
    for document_id in dict.fromkeys(item.chunk.document_id for item in inspected.context.evidence):
        metadata = reviewed_sources.get(document_id)
        description = (
            ' | '.join(f'{key}: {metadata[key]}' for key in (
                'source_type', 'publisher', 'attribution', 'financial_period',
                'event_date', 'published_date', 'source_locator'))
            if metadata is not None else 'legacy: no additional source review metadata registered'
        )
        provenance_rows.append(f'<li><b>{escape(document_id)}</b>: {escape(description)}</li>')
    provenance_html = ('<p>Source attribution (declarations, not a fact check):</p><ul>'
                       + ''.join(provenance_rows) + '</ul>') if provenance_rows else ''
    vector = retrieval.query_vector
    passage = retrieval.top_passage_vector
    query_norm = sqrt(sum(value * value for value in vector))
    route = retrieval.routed_document_id or "global search (no single trusted FY route)"
    vector_rows = "".join(
        f"<tr><td>{index}</td><td>{value:.4f}</td>"
        f"<td>{passage[index]:.4f}</td><td>{value * passage[index]:.4f}</td></tr>"
        for index, value in enumerate(vector[:6])
    ) if passage is not None else ""
    ledger_rows: list[str] = []
    running_sum = 0.0
    if passage is not None:
        if len(vector) != len(passage):
            raise ValueError("query and passage vectors have different dimensions")
        for index, (query_value, passage_value) in enumerate(zip(vector, passage)):
            product = query_value * passage_value
            running_sum += product
            ledger_rows.append(
                f"<tr><td>{index}</td><td>{query_value:.6f}</td>"
                f"<td>{passage_value:.6f}</td><td>{product:.6f}</td>"
                f"<td>{running_sum:.6f}</td></tr>"
            )
    selected_ids = {item.chunk.chunk_id for item in inspected.context.evidence}
    ranks = "".join(
        f'<div class="row" data-score="{item.score!r}" '
        f'data-threshold="{inspected.threshold!r}" '
        f'data-selected="{str(item.chunk.chunk_id in selected_ids).lower()}"><div class="source">'
        f"<b>{index}. {escape(item.chunk.document_id)}</b> · PDF page {item.chunk.page_number}"
        f'<div class="candidate-excerpt">{escape(item.chunk.text[:210])}</div>'
        f'<div class="bar"><i style="width:{max(0, min(100, item.score * 100)):.1f}%"></i></div>'
        f'</div><span class="score" title="Unrounded score: {item.score!r}; threshold: {inspected.threshold!r}">{item.score:.6f}</span>'
        f'<span class="{ "pass" if item.score >= inspected.threshold else "fail" }">'
        f'{"PASS" if item.score >= inspected.threshold else "OUT"}</span></div>'
        for index, item in enumerate(retrieval.candidates, start=1)
    ) or '<p>No candidate chunks were returned.</p>'
    candidate_full_html = ''.join(
        f'<details class="matrix-ledger" id="candidate-full-{rank}"><summary>'
        f'Inspect exact candidate {rank} · PDF page {item.chunk.page_number}</summary>'
        f'<p>Chunk ID: <code>{escape(item.chunk.chunk_id)}</code>; '
        f'document: {escape(item.chunk.document_id)}; cosine {item.score:.6f}. '
        + ('Selected by the similarity gate' if item.chunk.chunk_id in selected_ids
           else 'Not selected by the similarity gate')
        + ' — neither status establishes factual support.</p>'
        f'<pre class="ledger-scroll">{escape(item.chunk.text)}</pre></details>'
        for rank, item in enumerate(retrieval.candidates, start=1)
    )
    if lexical_candidates is None:
        lexical_html = '<p class="note">No lexical baseline was configured for this run.</p>'
    else:
        if len(lexical_candidates) > inspected.top_k or any(
            not isfinite(item.score)
            or (retrieval.routed_document_id is not None
                and item.chunk.document_id != retrieval.routed_document_id)
            for item in lexical_candidates
        ):
            raise ValueError("lexical comparison does not match the dense retrieval scope")
        dense_ids = {item.chunk.chunk_id for item in retrieval.candidates}
        lexical_ids = {item.chunk.chunk_id for item in lexical_candidates}
        if len(lexical_ids) != len(lexical_candidates):
            raise ValueError("lexical comparison has duplicate chunk IDs")
        lexical_rows = "".join(
            '<tr><td>' + str(rank) + '</td><td><code>'
            + escape(item.chunk.chunk_id) + '</code></td><td>'
            + str(item.chunk.page_number) + '</td><td>'
            + f'{item.score:.4f}' + '</td></tr>'
            for rank, item in enumerate(lexical_candidates, start=1)
        )
        lexical_html = (
            '<details class="matrix-ledger" id="lexical-contrast"><summary>'
            'Compare this query with the TF-IDF lexical baseline</summary>'
            '<p>TF-IDF and dense BGE rank the same candidate scope'
            + (f' ({escape(retrieval.routed_document_id)})' if retrieval.routed_document_id else ' (global)')
            + f'. {len(dense_ids & lexical_ids)} of the displayed Top-{inspected.top_k} '
            'chunk IDs overlap. Their cosine scores come from different vector spaces '
            'and are not numerically comparable; the 0.78 BGE gate is not applied to TF-IDF. '
            'Neither ranking proves the requested row, date, unit, or value.</p>'
            '<div class="ledger-scroll"><table><tr><th>TF-IDF rank</th><th>chunk ID</th>'
            '<th>PDF page</th><th>TF-IDF cosine</th></tr>'
            + lexical_rows + '</table></div>'
            + (
                '<details class="matrix-ledger" id="lexical-top-text"><summary>'
                'Open the exact top TF-IDF passage</summary>'
                f'<pre class="ledger-scroll">{escape(lexical_candidates[0].chunk.text)}</pre>'
                '</details>'
                if lexical_candidates else ''
            )
            + '<p class="practice">Find a chunk returned by only one method. Is it '
            'actually relevant in the original PDF, or merely lexically/semantically similar?</p>'
            '</details>'
        )
    scores = retrieval.considered_scores
    if len(scores) != retrieval.considered_chunks:
        raise ValueError("score distribution does not match the considered chunk count")
    if scores:
        low, high = min(scores), max(scores)
        bin_count = 10 if high > low else 1
        counts = [0] * bin_count
        for score in scores:
            bin_index = min(bin_count - 1, int((score - low) / (high - low) * bin_count)) if high > low else 0
            counts[bin_index] += 1
        largest_bin = max(counts)
        histogram_rows = "".join(
            '<tr class="histogram-row">'
            f'<td>{low + (high - low) * index / bin_count:.3f}–'
            f'{low + (high - low) * (index + 1) / bin_count:.3f}'
            f'{" inclusive" if index == bin_count - 1 else ""}</td>'
            f'<td>{count}</td><td><span class="hist-bar">'
            f'<i style="width:{count / largest_bin * 100:.1f}%"></i>'
            '</span></td></tr>'
            for index, count in enumerate(counts)
        )
        above_gate = sum(score >= inspected.threshold for score in scores)
        histogram_html = (
            '<details class="matrix-ledger" id="score-distribution"><summary>'
            f'Score distribution for all {len(scores)} scored chunks</summary>'
            f'<p>{above_gate} of {len(scores)} scores at or above the gate; '
            f'only the Top-{inspected.top_k} candidates continue to the evidence selector. '
            f'Observed cosine range: {low:.4f}–{high:.4f}. '
            'These bars count scores, not relevant or correct answers. Bin labels are rounded; '
            'the gate count uses unrounded scores.</p>'
            '<table class="histogram"><tr><th>cosine interval</th><th>count</th><th>bar vs tallest bin</th></tr>'
            + histogram_rows + '</table></details>'
        )
    else:
        histogram_html = '<p>No chunk scores are available for this route.</p>'
    citations = ", ".join(escape(value) for value in inspected.context.citations) or "none"
    top_score = retrieval.candidates[0].score if retrieval.candidates else None
    total_dot = f"{running_sum:.4f}" if passage is not None else "n/a"
    top_chunk = retrieval.candidates[0].chunk if retrieval.candidates else None
    if inspected.context.can_answer and inspected.context.evidence:
        full_prompt = build_grounded_prompt(
            query=inspected.normalized_query, context=inspected.context,
        )
        prompt_preview_html = (
            '<details class="matrix-ledger" id="grounded-prompt-preview"><summary>'
            'Open the complete grounded prompt (preview)</summary>'
            f'<p>Full prompt size: {len(full_prompt)} characters; '
            f'{len(tiktoken.get_encoding("cl100k_base").encode(full_prompt))} '
            'cl100k_base tokens. Includes instructions, query and evidence; '
            'excludes chat-template overhead and output budget. '
            'Not the generator tokenizer or a model-window fit check.</p>'
            '<p>Preview only; no model call is made by this panel. '
            'This is the prompt constructed from the current gated context, '
            'not a completed generation or a verified answer.</p>'
            f'<pre class="ledger-scroll">{escape(full_prompt)}</pre></details>'
        )
    else:
        prompt_preview_html = '<p>No grounded prompt assembled: the evidence gate refused.</p>'
    assembled_context_html = (
        '<details class="matrix-ledger" id="assembled-context"><summary>'
        'Open the exact assembled context passed by the similarity gate</summary>'
        f'<p>Context size: {len(inspected.context.context)} characters; '
        f'{len(tiktoken.get_encoding("cl100k_base").encode(inspected.context.context))} '
        'cl100k_base tokens. Excludes the user query, instructions and output budget. '
        'Not the generator tokenizer or a model-window fit check.</p>'
        '<p>These are the full selected passages in rank order, with the exact '
        'citation prefixes built by the service. No deduplication or '
        'token-budget truncation is implemented here.</p>'
        f'<pre class="ledger-scroll">{escape(inspected.context.context)}</pre>'
        '</details>'
        if inspected.context.evidence else
        '<p class="note">No retrieval candidates were returned. '
        'No passage was assembled; the context is empty. '
        'This is not a similarity-threshold rejection.</p>'
        if not retrieval.candidates else
        '<p class="note">No passage was assembled: every retrieved candidate '
        'fell below the similarity gate. The context is empty.</p>'
    )
    selected_chunks_html = ''.join(
        f'<li>rank {index}: <code>{escape(item.chunk.chunk_id)}</code> · '
        f'{escape(item.chunk.document_id)} PDF page {item.chunk.page_number} · '
        f'cosine {item.score:.4f}</li>'
        for index, item in enumerate(inspected.context.evidence, start=1)
    )
    if ingestion_replay is None:
        source_replay_html = '<p class="note">Source-page replay was not configured for this run.</p>'
    elif not ingestion_replay.verified:
        source_replay_html = (
            '<p class="note">Source-page replay unavailable: '
            + escape(ingestion_replay.reason) + '. No extraction result is claimed.</p>'
        )
    elif ingestion_replay.replay_kind == "table-candidate":
        source_replay_html = (
            '<details class="source-replay" id="source-table-replay"><summary>'
            'Verified candidate replay · inspect PDF-derived candidate</summary>'
            '<p class="verified">The frozen PDF SHA-256 matches. Re-running the '
            'conservative table parser on this page produced this exact stored candidate.</p>'
            f'<p>PDF page {ingestion_replay.page_number} · '
            f'{ingestion_replay.page_chunk_count} unambiguous candidates re-extracted on this page. '
            'The parser did not claim to recover every table cell.</p>'
            f'<p>Frozen chunk ID: <code>{escape(ingestion_replay.chunk_id)}</code><br>'
            f'Source SHA-256: <code>{escape(ingestion_replay.source_sha256)}</code></p>'
            '<p>Structured row/period/value candidate:</p>'
            f'<pre>{escape(ingestion_replay.table_candidate_text)}</pre>'
            + (
                '<figure><img style="max-width:100%;height:auto" '
                'alt="Source PDF page: amount outlined in orange, metric label in blue" '
                f'src="{escape(ingestion_replay.table_page_image, quote=True)}">'
                '<figcaption>Orange: amount; blue: metric label. '
                + ('Green: period header. Highlighted coordinates do not independently verify period interpretation. '
                   if ingestion_replay.period_highlighted else
                   'No period header coordinates are available for this candidate. ')
                + 'Units are not independently highlighted or verified.</figcaption></figure>'
                if ingestion_replay.table_page_image else
                '<p>No coordinate-backed page image is available for this candidate.</p>'
            )
            +
            '<p>Extracted page opening:</p>'
            f'<pre>{escape(ingestion_replay.source_excerpt)}</pre>'
            '<p class="practice">Indexing happened earlier. This replay verifies that '
            'the parser can reproduce the same candidate from the unchanged PDF; '
            'it does not independently prove that the row/column interpretation is correct. '
            'Check the cited PDF table visually.</p></details>'
        )
    else:
        page_windows_html = ''.join(
            '<tr>'
            f'<td>{index + 1}</td><td>{start}–{end}</td>'
            f'<td><code>{escape(chunk_id)}</code>'
            + (' ← retrieved top candidate' if chunk_id == ingestion_replay.chunk_id else '')
            + '</td></tr>'
            for index, (chunk_id, start, end) in enumerate(ingestion_replay.page_windows)
        )
        source_replay_html = (
            '<details class="source-replay" id="source-window-replay"><summary>'
            'Verified source-page replay · inspect extracted text windows</summary>'
            '<p>This is not a visual PDF viewer. Check the original PDF separately '
            'for table row, date-column, unit and amount alignment.</p>'
            '<p class="verified">Frozen PDF hash and every chunk on this page match.</p>'
            f'<p>PDF page {ingestion_replay.page_number} → '
            f'{ingestion_replay.page_token_count} cl100k_base tokens → '
            f'{ingestion_replay.page_chunk_count} stored page windows. '
            f'This selected window covers tokens {ingestion_replay.window_start}–'
            f'{ingestion_replay.window_end} (end exclusive); '
            f'{ingestion_replay.overlap_tokens}-token overlap with the preceding window.</p>'
            f'<p>Frozen chunk ID: <code>{escape(ingestion_replay.chunk_id)}</code><br>'
            f'Source SHA-256: <code>{escape(ingestion_replay.source_sha256)}</code></p>'
            f'<p>Extracted page opening:</p><pre>{escape(ingestion_replay.source_excerpt)}</pre>'
            f'<p>Selected stored window opening:</p><pre>{escape(ingestion_replay.window_excerpt)}</pre>'
            '<p>Every stored window on this PDF page (token end is exclusive):</p>'
            '<div class="ledger-scroll"><table><tr><th>window</th><th>token span</th><th>chunk ID</th></tr>'
            + page_windows_html + '</table></div>'
            + (f'<p>Shared 50-token boundary between neighboring windows:</p>'
               f'<pre>{escape(ingestion_replay.shared_boundary_text)}</pre>'
               if ingestion_replay.shared_boundary_text else '')
            + '<p class="practice">Indexing happened earlier; this replays it from the '
            'unchanged source PDF and verifies the stored result. Token windows can split '
            'table rows even when the page and hash are right.</p></details>'
        )
    evaluation_html = _render_evaluation_suite(
        evaluation_report,
        dataset_sha256=evaluation_dataset_sha256,
        dataset_label=evaluation_dataset_label,
        top_k=inspected.top_k,
        threshold=inspected.threshold,
    )
    return (
        _LAB_STYLE
        + '<section class="rag-lab"><div class="eyebrow">LIVE TRACE · EVIDENCE LAB</div>'
        + '<h2>Follow one question through the pipeline</h2>'
        + '<p class="lede">The index was built earlier. Steps 01–04 show this query. '
        + 'Cell 05 reports generation status and evaluation boundaries; it does not imply generation ran. '
        + 'Scores are cosine similarity, not a fact check.</p>'
        + '<p class="practice">Notebook mode · open cells 00 → 05 in order. '
        + 'Only the first cell starts open.</p>'
        + '<details id="rag-step-00" open><summary><span class="num">00</span> Precomputed corpus &amp; index</summary><div class="body">'
        + f'<div class="metric-grid"><div class="metric"><b>{corpus_documents}</b><small>source PDFs</small></div>'
        + f'<div class="metric"><b>{indexed_pages or "—"}</b><small>indexed PDF pages</small></div>'
        + f'<div class="metric"><b>{corpus_chunks}</b><small>stored chunks</small></div>'
        + (f'<div class="metric"><b>{table_cell_chunks}</b><small>structured table cells</small></div>'
           if table_cell_chunks else '')
        + f'<div class="metric"><b>{len(vector)}</b><small>embedding dimensions</small></div></div>'
        + '<p>The counts describe the corpus supplied to this view. '
        + 'Check the corpus and embedding manifests for the extraction strategy, '
        + 'chunk size, overlap, model revision and vector artifact; these cannot '
        + 'be inferred from a displayed vector or a replay image. '
        + 'No ingestion rerun or Chroma/ANN index is established by this panel.</p>'
        + source_replay_html
        + '<p class="prompt">Explain: What does the 50-token overlap preserve, and what can it duplicate?</p>'
        + '</div></details>'
        + '<details id="rag-step-01"><summary><span class="num">01</span> Input &amp; routing</summary><div class="body">'
        + f'<pre>{escape(inspected.normalized_query)}</pre><p>Route: <code>{escape(route)}</code>; '
        + f'{retrieval.considered_chunks} chunks scored in this route.</p>'
        + f'<p>Query encoder input = <code>{escape(BGE_QUERY_INSTRUCTION + inspected.normalized_query)}</code></p>'
        + '<p class="prompt">Predict: Would removing “FY2025” expand or narrow this candidate pool?</p>'
        + '</div></details>'
        + '<details id="rag-step-02"><summary><span class="num">02</span> Query vector &amp; cosine arithmetic</summary><div class="body">'
        + f'<p>Actual query vector: {len(vector)} dimensions; L2 norm = {query_norm:.4f}. '
        + 'Normalized query · normalized passage = cosine score.</p>'
        + ('<table><tr><th>dimension</th><th>query</th><th>top passage</th><th>product</th></tr>'
           + vector_rows + '</table><p>The six rows above are a readable preview; '
           + f'the full-vector dot product is <b>{total_dot}</b> '
           + f'(retriever score <b>{top_score:.4f}</b>).</p>'
           + '<details class="matrix-ledger"><summary>Open the complete '
           + f'{len(vector)}-dimension multiplication ledger and running sum</summary>'
           + '<p>Each displayed row is rounded; the running sum uses unrounded values. '
           + 'The final sum is the cosine score because both vectors have L2 norm 1.</p>'
           + '<div class="ledger-scroll"><table id="full-vector-ledger">'
           + '<tr><th>dimension</th><th>query</th><th>passage</th><th>product</th><th>running sum</th></tr>'
           + ''.join(ledger_rows) + '</table></div></details>'
           + '<p class="prompt">Calculate one product by hand. Why does the first-six-row sum not equal the full score?</p>'
           if passage is not None else '<p>No top passage vector is available.</p>')
        + '</div></details>'
        + '<details id="rag-step-03"><summary><span class="num">03</span> Top-k ranking &amp; similarity gate</summary><div class="body">'
        + f'<p>Actual top {inspected.top_k} requested; threshold = <b>{inspected.threshold:.2f}</b>. '
        + 'The bars encode positive score magnitude, not correctness.</p>'
        + ranks
        + candidate_full_html
        + histogram_html
        + lexical_html
        + '<p class="prompt">Which candidate is below the gate? Would a higher threshold also reject some useful evidence?</p>'
        + '</div></details>'
        + '<details id="rag-step-04"><summary><span class="num">04</span> Context assembly &amp; citations</summary><div class="body">'
        + f'<p>Evidence passed: <b>{len(inspected.context.evidence)}</b>. Citations: {citations}.</p>'
        + (f'<p>Top-ranked candidate chunk ID: <code>{escape(top_chunk.chunk_id)}</code> '
           '(ranking alone does not mean it passed the gate).</p>' if top_chunk is not None else '')
        + (f'<ol>{selected_chunks_html}</ol>' if selected_chunks_html else '')
        + provenance_html
        + assembled_context_html
        + '<p>A page citation shows provenance, not that the selected row/date/value '
        'answers the question.</p>'
        + ('<p>The gated context is passed to the optional generator below.</p>'
           if generation_enabled else '<p>No LLM sees this context in evidence-only mode.</p>')
        + '<p class="prompt">Open the original PDF page. Which row, date, unit, and amount actually support the claim?</p>'
        + '</div></details>'
        + '<details id="rag-step-05"><summary><span class="num">05</span> Generation &amp; evaluation boundary</summary><div class="body">'
        + prompt_preview_html
        + ('<p><b>Optional generation result follows in cell 06.</b> '
           'A model response is not automatically a verified financial answer.</p>'
           if generation_enabled else '<p><b>Generation not run.</b> Earlier local models failed factual acceptance; '
           'this evidence-only demo does not synthesize a financial answer.</p>')
        + '<p class="note">For evaluation, inspect the original PDF row and compare '
        + 'gold-labeled page/field evidence. A retrieval PASS is not a verified answer.</p>'
        + evaluation_html
        + '<p class="prompt">Name one retrieval metric and one different test needed before releasing generated answers.</p>'
        + '</div></details></section>'
    )


def format_result(result: GroundedContext) -> tuple[str, str]:
    """Convert a grounded response to status plus plain, non-upload evidence text."""
    gate = "PASS" if result.can_answer else "REFUSE"
    status = (
        f"### Similarity gate: {gate} (retrieval score only; answer not verified)"
        f"\n\n{result.message}"
    )
    blocks = [
        (
            f"rank={rank} score={item.score:.4f} "
            f"document={item.chunk.document_id} PDF_page={item.chunk.page_number}\n"
            f"{item.chunk.text}"
        )
        for rank, item in enumerate(result.evidence, start=1)
    ]
    return status, "\n\n".join(blocks)


def _format_evidence(evidence: tuple[ScoredChunk, ...]) -> str:
    return "\n\n".join(
        (
            f"rank={rank} score={item.score:.4f} "
            f"document={item.chunk.document_id} PDF_page={item.chunk.page_number}\n"
            f"{item.chunk.text}"
        )
        for rank, item in enumerate(evidence, start=1)
    )


def format_demo_result(
    result: GroundedContext | GeneratedResponse,
) -> tuple[str, str, str]:
    """Expose retrieval, generation, and raw evidence as separate UI regions."""
    if isinstance(result, GeneratedResponse):
        retrieval_gate = "PASS" if result.evidence_passed else "REFUSE"
        generation_gate = "PASS" if result.can_answer else "REFUSE"
        if (not result.can_answer and result.generation_error is None
                and result.raw_answer == 'The retrieved evidence is insufficient.'):
            generation_gate = "N/A — MODEL ABSTAINED"
        status = (
            f"### Similarity gate: {retrieval_gate} · Citation-format gate: "
            f"{generation_gate} (factual correctness not verified)"
            f"\n\n{result.message}"
        )
        answer = result.answer or "No generated answer was released."
        return status, answer, _format_evidence(result.evidence)

    status, evidence = format_result(result)
    return (
        status,
        "Generation is disabled for this run; inspect the gated evidence below.",
        evidence,
    )


def render_generation_walkthrough(*, query, context, result: GeneratedResponse) -> str:
    """Show diagnostic output separately from the released answer, without another call."""
    raw = result.raw_answer
    prompt = (build_grounded_prompt(query=query, context=context)
              if raw is not None else "No completed model output; no prompt replay claimed.")
    rows = numeric_support_trace(raw, evidence=result.evidence) if raw is not None else []
    items = ''.join(
        '<li><code>' + escape(str(row['literal'])) + '</code> — '
        + ('literal veto passed' if row['literal_veto_passed'] else 'literal veto failed')
        + ': ' + escape(str(row['error'] or 'not factual entailment')) + '</li>'
        for row in rows
    )
    label = 'ACCEPTED BY AUTOMATIC CHECKS' if result.can_answer else 'REJECTED / NOT RELEASED'
    if (not result.can_answer and result.generation_error is None
            and raw == 'The retrieved evidence is insufficient.'):
        label = 'MODEL ABSTAINED / NOT RELEASED'
    if raw is None:
        label = ('NO COMPLETED OUTPUT' if result.generation_error == 'the language model request failed'
                 else 'NOT RUN — PRE-GENERATION GATE')
    abstention_note = (
        '<p class="note"><b>EXACT ABSTENTION</b> — this may be a correct refusal or false abstention. '
        'Check whether the question has supporting evidence; correctness is not determined by citation presence. '
        'This diagnostic does not release an answer.</p>'
        if raw == 'The retrieved evidence is insufficient.' else ''
    )
    return (
        '<section class="rag-lab"><div class="eyebrow">06 · GENERATION</div>'
        '<h2>From retrieved evidence to a model response</h2>'
        '<p class="note">Literal checks are not factual entailment. '
        'Check the original financial row, period and unit yourself.</p>'
        '<details><summary>Prompt from this same gated context</summary>'
        f'<pre>{escape(prompt)}</pre></details><p>{escape(result.message)}</p>'
        f'<h3>{label}</h3><p>Raw model output — diagnostic only, not a verified answer.</p>'
        f'<pre>{escape(raw if raw is not None else "No completed model output.")}</pre>'
        f'{abstention_note}'
        '<details><summary>Per-number literal veto trace (explicitly cited chunks only)</summary>'
        f'<ul>{items}</ul></details></section>'
    )


def build_demo(
    *,
    service: QueryService,
    corpus_documents: int = 0,
    corpus_chunks: int = 0,
    indexed_pages: int = 0,
    table_cell_chunks: int = 0,
    source_documents: Mapping[str, tuple[Path, str]] | None = None,
    indexed_chunks: Sequence[Chunk] = (),
    evaluation_report: Mapping[str, object] | None = None,
    evaluation_dataset_sha256: str = "",
    evaluation_dataset_label: str = "annual_multiyear_heldout.jsonl",
    lexical_retriever: LexicalRetriever | None = None,
    source_provenance: Mapping[str, Mapping[str, object]] | None = None,
) -> gr.Blocks:
    """Build a fixed-corpus demo; intentionally exposes no file or URL ingestion."""
    reviewed_sources = {document_id: _display_provenance(metadata)
                        for document_id, metadata in (source_provenance or {}).items()}

    def run_query(query: str, previous_query: str) -> tuple[str, str, str, str, str]:
        try:
            resolved = resolve_demo_query(previous_query, query)
            query = resolved.query
            rewrite_note = (
                f"**Follow-up rewrite (not a fact check):** FY{resolved.previous_fiscal_year} "
                f"→ FY{resolved.target_fiscal_year}; searched `{resolved.query}`.\n\n"
                if resolved.used_previous else ""
            )
            inspect = getattr(service, "inspect", None)
            if inspect is not None:
                inspected = inspect(query)
                generate_context = getattr(service, "generate_context", None)
                result = (
                    generate_context(query, inspected.context)
                    if generate_context is not None else inspected.context
                )
                status, answer, evidence = format_demo_result(result)
                ingestion_replay = replay_top_candidate(
                    inspected,
                    source_documents=source_documents or {},
                    indexed_chunks=indexed_chunks,
                )
                lexical_candidates = (
                    lexical_retriever.retrieve_from_documents(
                        inspected.normalized_query,
                        top_k=inspected.top_k,
                        document_ids=(
                            {inspected.retrieval.routed_document_id}
                            if inspected.retrieval.routed_document_id else None
                        ),
                    )
                    if lexical_retriever is not None else None
                )
                walkthrough = render_pipeline_walkthrough(
                    inspected,
                    corpus_documents=corpus_documents,
                    corpus_chunks=corpus_chunks,
                    indexed_pages=indexed_pages,
                    table_cell_chunks=table_cell_chunks,
                    ingestion_replay=ingestion_replay,
                    evaluation_report=evaluation_report,
                    evaluation_dataset_sha256=evaluation_dataset_sha256,
                    evaluation_dataset_label=evaluation_dataset_label,
                    lexical_candidates=lexical_candidates,
                    generation_enabled=generate_context is not None,
                    source_provenance=reviewed_sources,
                )
                if generate_context is not None:
                    walkthrough += render_generation_walkthrough(
                        query=query, context=inspected.context, result=result,
                    )
                next_previous = query if inspected.context.can_answer else ""
                return rewrite_note + status, answer, evidence, walkthrough, next_previous
            result = service.query(query)
            status, answer, evidence = format_demo_result(result)
            return (
                rewrite_note + status, answer, evidence,
                _LAB_STYLE + '<section class="rag-lab"><p class="note">'
                'Detailed dense arithmetic is unavailable for this optional service mode.'
                '</p></section>',
                query if result.can_answer else "",
            )
        except ValueError as error:
            return f"### Request rejected\n\n{error}", "", "", "", ""

    with gr.Blocks(
        title="Evidence-first NVIDIA Financial RAG",
        analytics_enabled=False,
        theme=gr.themes.Soft(primary_hue="green", neutral_hue="stone"),
        fill_width=True,
        css="""
        .gradio-container {max-width: 1800px !important; padding: 24px !important;}
        #rag-header h1 {font-family: Georgia, serif; font-size: 30px; margin-bottom: 8px;}
        #rag-controls {position: sticky; top: 16px; align-self: flex-start;
          border: 1px solid #8b9b87; border-radius: 12px; padding: 20px;}
        #rag-trace {max-height: calc(100vh - 145px); overflow-y: auto;
          padding: 0 12px 12px; scrollbar-gutter: stable;}
        #rag-trace .rag-lab {margin-top: 0;}
        #rag-trace pre {overflow-x: auto;}
        @media(max-width: 1050px) {
          #rag-controls {position: static;}
          #rag-trace {max-height: none; overflow: visible;}
        }
        """,
    ) as demo:
        gr.Markdown(
            "# NVIDIA Financial RAG · Evidence notebook\n"
            "1. Ask one question — specify the fiscal year and the metric.\n"
            "2. Read the answer or outcome on the left, then inspect the evidence on the right.\n"
            "3. Check the original report — open the cited PDF page and verify the row, "
            "date column, amount, and unit yourself.\n"
            "**Generation OFF by default · similarity is not factual verification.**",
            elem_id="rag-header",
        )
        with gr.Row():
          with gr.Column(scale=1, min_width=320, elem_id="rag-controls"):
            gr.Markdown("### 01 · Ask & inspect\nSpecify the fiscal year and financial metric.")
            query = gr.Textbox(label="Question", placeholder="What was NVIDIA's annual revenue in FY2025?", lines=3, max_lines=5)
            previous_query = gr.State("")
            run = gr.Button("Run retrieval trace", variant="primary")
            reset = gr.Button("Clear follow-up context", variant="secondary")
            status = gr.Markdown()
            answer = gr.Textbox(label="Answer / outcome (not fact-verified)",
                                value="No question run yet.", lines=5,
                                interactive=False, elem_id="rag-answer")
            gr.Markdown("### Source-check checklist\nFiscal period → metric row → date column → amount → unit.")
            gr.Markdown(r"### Similarity, not truth" + "\n" + r"$$s(q,d)=\frac{q\cdot d}{\lVert q\rVert\lVert d\rVert}$$")
            with gr.Accordion("Advanced lab · constrained follow-up", open=False):
                gr.Markdown("After a successful `What was NVIDIA's annual revenue in FY2025?`, try `What about FY2024?` or `And FY2024?`. Other ambiguous follow-ups require clarification. This is not general chat memory.")
          with gr.Column(scale=3, min_width=600, elem_id="rag-trace"):
            walkthrough = gr.HTML(
                _LAB_STYLE + '<section class="rag-lab"><div class="eyebrow">EVIDENCE NOTEBOOK · READY</div>'
                '<h2>Follow one request through the pipeline.</h2>'
                '<p class="lede">Run a question on the left. This panel shows text, vectors, '
                'scores, selection and context from the actual inspection. The index was built offline.</p>'
                f'<div class="metric-grid"><div class="metric"><b>{corpus_documents or "—"}</b><small>source PDFs</small></div>'
                f'<div class="metric"><b>{indexed_pages or "—"}</b><small>indexed pages</small></div>'
                f'<div class="metric"><b>{corpus_chunks or "—"}</b><small>stored chunks</small></div></div></section>'
            )
            with gr.Accordion("Source passages · audit", open=False):
                evidence = gr.Textbox(label="Retrieved passages (similarity-filtered, not fact-checked)", lines=22, interactive=False)
        # Layout only: preserve the same callback, state and inspection policy.
        # Keep one stable, named API for automated mentor-demo acceptance checks.
        # The Enter-key event reuses the same handler but is intentionally hidden
        # from the public Gradio API to avoid two drifting endpoint names.
        run.click(
            run_query,
            inputs=[query, previous_query],
            outputs=[status, answer, evidence, walkthrough, previous_query],
            api_name="query",
        )
        query.submit(
            run_query,
            inputs=[query, previous_query],
            outputs=[status, answer, evidence, walkthrough, previous_query],
            api_name=False,
        )
        reset.click(lambda: "", outputs=previous_query, api_name=False)
    return demo.queue(default_concurrency_limit=2, max_size=8)
