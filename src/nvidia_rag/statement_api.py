"""Explicit HTTP source selection; no dense fallback, source IO or generation.

The operator injects a source-built immutable index at startup. HTTP requests
contain only a literal question and optional registered document identifiers.
The returned cells remain source candidates with unestablished correctness.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
import json
import logging
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from .structured_income import (
    IncomeIndex, ParsedRequest, Selection, parse_statement_request,
    select_statement_candidates, validate_income_index,
)

MAX_BODY_BYTES = 8192
_INVALID_REQUEST = 'Invalid statement request.'
_SCORE_SEMANTICS = 'exact-source-field-selection;no-cosine-score'
DocumentId = Annotated[str, Field(min_length=1, max_length=128,
                                 pattern=r'^[A-Za-z0-9_.-]+$')]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class StatementRequest(StrictModel):
    question: str = Field(min_length=1, max_length=500)
    document_ids: list[DocumentId] | None = Field(default=None, min_length=1, max_length=128)

    @field_validator('question')
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('blank question')
        return value

    @field_validator('document_ids')
    @classmethod
    def unique_scope(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError('duplicate document scope')
        return value


class SourceSpanView(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    literal: str
    bbox: tuple[float, float, float, float]


class MentionView(StrictModel):
    field: str
    value: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class ParsedRequestView(StrictModel):
    query: str
    query_sha256: str
    operator_document_ids: tuple[str, ...]
    mentions: tuple[MentionView, ...]
    issues: tuple[str, ...]


class SourceCandidate(StrictModel):
    cell_id: str
    document_id: str
    pdf_sha256: str
    physical_page: int = Field(ge=1)
    pdf_page_label: str | None
    fiscal_year: int
    fiscal_quarter: int | None
    parser_profile: str
    pymupdf_version: str
    page_text_sha256: str
    row_path: str
    row_literal: str
    column_number: int = Field(ge=1)
    column_role: Literal['current', 'prior_comparison']
    period_end: str
    duration_months: int | None
    period_kind: Literal['fiscal_year', 'months']
    amount: str
    value_literal: str
    unit_kind: Literal['monetary', 'per_share', 'count']
    scale: str
    currency_literal: str | None
    currency_normalization: str
    bindings: tuple[tuple[str, SourceSpanView], ...]


class StatementResponse(StrictModel):
    schema_version: Literal[1] = 1
    request_id: str
    disposition: Literal['source_candidate', 'clarify', 'refuse', 'unsupported', 'unavailable']
    reason: str
    parsed_request: ParsedRequestView | None = None
    index_sha256: str | None = None
    cells: list[SourceCandidate] = Field(default_factory=list)
    financial_correctness: Literal['unestablished'] = 'unestablished'
    score_semantics: Literal['exact-source-field-selection;no-cosine-score'] = _SCORE_SEMANTICS
    error_code: Literal['statement_selection_unavailable'] | None = None

    @model_validator(mode='after')
    def coherent_outcome(self) -> StatementResponse:
        if self.disposition == 'source_candidate':
            if len(self.cells) != 1:
                raise ValueError('a candidate outcome requires exactly one source cell')
        elif self.cells:
            raise ValueError('a withheld outcome cannot expose source cells')
        if self.disposition == 'unavailable':
            if self.error_code is None or self.parsed_request is not None:
                raise ValueError('invalid operational failure')
        elif self.error_code is not None or self.parsed_request is None:
            raise ValueError('invalid completed source outcome')
        return self


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite JSON constant')


async def _read_statement_request(request: Request) -> StatementRequest:
    media_type = request.headers.get('content-type', '').split(';', 1)[0].strip().lower()
    if media_type != 'application/json':
        raise HTTPException(status_code=415, detail='Statement requests require application/json.')
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail='Statement request exceeds 8192 bytes.')
        body.extend(chunk)
    try:
        data = json.loads(body.decode('utf-8'), object_pairs_hook=_unique_pairs,
                          parse_constant=_reject_constant)
        return StatementRequest.model_validate(data)
    except (UnicodeDecodeError, ValueError, ValidationError, RecursionError):
        # Do not reflect input values, unknown key names, loc or exception ctx.
        raise HTTPException(status_code=422, detail=_INVALID_REQUEST) from None


def _completed_response(index: IncomeIndex, payload: StatementRequest,
                        request_id: str, index_sha: str) -> JSONResponse:
    scope = (tuple(row[0] for row in index.reports) if payload.document_ids is None
             else tuple(payload.document_ids))
    parsed: ParsedRequest = parse_statement_request(
        payload.question, operator_document_ids=scope, index=index,
    )
    selected: Selection = select_statement_candidates(index, parsed)
    response = StatementResponse(
        request_id=request_id, disposition=selected.disposition, reason=selected.reason,
        parsed_request=ParsedRequestView.model_validate(asdict(parsed)), index_sha256=index_sha,
        cells=[SourceCandidate.model_validate({**asdict(cell), 'period_kind': cell.period_kind})
               for cell in selected.cells],
        financial_correctness=selected.financial_correctness, score_semantics=selected.score_semantics,
    )
    # Both typed validation and actual JSON rendering occur inside the route's
    # safe operational boundary, before any candidate response is returned.
    return JSONResponse(content=response.model_dump(mode='json'))


def _unavailable(request_id: str, *, reason: str, index_sha: str | None) -> JSONResponse:
    # Keep the failure envelope independent of a broken result DTO serializer.
    return JSONResponse(status_code=503, content={
        'schema_version': 1, 'request_id': request_id, 'disposition': 'unavailable',
        'reason': reason, 'parsed_request': None, 'index_sha256': index_sha, 'cells': [],
        'financial_correctness': 'unestablished', 'score_semantics': _SCORE_SEMANTICS,
        'error_code': 'statement_selection_unavailable',
    })


def attach_statement_route(app: FastAPI, *, index: IncomeIndex | None,
                           require_bearer: Callable, logger: logging.Logger) -> None:
    """Register a separate authenticated route over an operator-owned snapshot."""
    index_sha = None
    if index is not None:
        validate_income_index(index)
        value = asdict(index)
        value.pop('_construction_token')
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'
        index_sha = sha256(canonical).hexdigest()

    @app.post(
        '/v1/statement-candidates', response_model=StatementResponse,
        dependencies=[Depends(require_bearer)],
        openapi_extra={'requestBody': {'required': True, 'content': {
            'application/json': {'schema': StatementRequest.model_json_schema()}}}},
        responses={
            401: {'description': 'Missing or invalid bearer token'},
            413: {'description': 'Actual request body exceeds 8192 bytes'},
            415: {'description': 'Unsupported request media type'},
            422: {'description': 'Invalid statement request; input details are withheld'},
            503: {'model': StatementResponse, 'description': 'Statement selection unavailable'},
        },
    )
    async def statement_candidates(request: Request) -> JSONResponse:
        # No declared framework body parameter: authentication runs before this
        # route reads or parses the stream. Existing /v1/query parsing is intact.
        request_id = request.state.request_id
        if index is None:
            return _unavailable(request_id, reason='STATEMENT_INDEX_NOT_CONFIGURED', index_sha=None)
        try:
            payload = await _read_statement_request(request)
        except HTTPException:
            raise
        except Exception:
            logger.error('statement_selection_unavailable request_id=%s', request_id)
            return _unavailable(request_id, reason='STATEMENT_SELECTION_UNAVAILABLE', index_sha=index_sha)
        try:
            return await run_in_threadpool(_completed_response, index, payload, request_id, index_sha)
        except Exception:
            # Only the request reader owns intentional HTTP validation errors.
            # Result exceptions, including HTTPException, remain service faults.
            logger.error('statement_selection_unavailable request_id=%s', request_id)
            return _unavailable(request_id, reason='STATEMENT_SELECTION_UNAVAILABLE', index_sha=index_sha)
