"""Independent typed HTTP source selection over an operator-owned snapshot."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
import json
import logging
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import Field, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool

from .statement_api import DocumentId, MAX_BODY_BYTES, SourceCandidate, StrictModel
from .statement_fields import StatementFieldRequest, select_statement_fields, validate_field_request
from .structured_income import IncomeIndex, validate_income_index

_INVALID_REQUEST = 'Invalid statement field request.'
_SCORE_SEMANTICS = 'exact-source-field-selection;no-cosine-score'


class StatementFieldPayload(StrictModel):
    row_path: str = Field(min_length=1, max_length=256)
    period_kind: Literal['fiscal_year', 'months']
    period_end: str = Field(min_length=10, max_length=10)
    duration_months: int | None
    unit_kind: Literal['monetary', 'per_share', 'count']
    scale: str = Field(min_length=1, max_length=32)
    currency_literal: str | None = Field(min_length=1, max_length=32)
    document_ids: list[DocumentId] | None = Field(default=None, min_length=1, max_length=128)
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    column_role: Literal['current', 'prior_comparison'] | None = None

    def domain_request(self) -> StatementFieldRequest:
        values = self.model_dump()
        values['document_ids'] = None if self.document_ids is None else tuple(self.document_ids)
        return StatementFieldRequest(**values)

    @model_validator(mode='after')
    def strict_native_fields(self) -> StatementFieldPayload:
        validate_field_request(self.domain_request())
        return self


class StatementFieldResponse(StrictModel):
    schema_version: Literal[1] = 1
    request_id: str
    disposition: Literal['source_candidate', 'clarify', 'refuse', 'unsupported', 'unavailable']
    reason: str
    selection_request: StatementFieldPayload | None = None
    selection_request_sha256: str | None = None
    index_sha256: str | None = None
    cells: list[SourceCandidate] = Field(default_factory=list)
    financial_correctness: Literal['unestablished'] = 'unestablished'
    score_semantics: Literal['exact-source-field-selection;no-cosine-score'] = _SCORE_SEMANTICS
    error_code: Literal['statement_selection_unavailable'] | None = None

    @model_validator(mode='after')
    def coherent_outcome(self) -> StatementFieldResponse:
        if self.disposition == 'source_candidate':
            if len(self.cells) != 1:
                raise ValueError('a candidate outcome requires exactly one source cell')
        elif self.cells:
            raise ValueError('a withheld outcome cannot expose source cells')
        if self.disposition == 'unavailable':
            if self.error_code is None or self.selection_request is not None or self.selection_request_sha256 is not None:
                raise ValueError('invalid operational failure')
        elif self.error_code is not None or self.selection_request is None or self.selection_request_sha256 is None:
            raise ValueError('invalid completed source outcome')
        return self


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON key')
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError('nonfinite JSON constant')


async def _read_field_request(request: Request) -> StatementFieldPayload:
    media_type = request.headers.get('content-type', '').split(';', 1)[0].strip().lower()
    if media_type != 'application/json':
        raise HTTPException(status_code=415, detail='Statement field requests require application/json.')
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail='Statement field request exceeds 8192 bytes.')
        body.extend(chunk)
    try:
        value = json.loads(body.decode('utf-8'), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
        return StatementFieldPayload.model_validate(value)
    except (UnicodeDecodeError, ValueError, ValidationError, RecursionError):
        raise HTTPException(status_code=422, detail=_INVALID_REQUEST) from None


def _completed_response(index: IncomeIndex, payload: StatementFieldPayload,
                        request_id: str, index_sha: str) -> JSONResponse:
    selected = select_statement_fields(index, payload.domain_request())
    response = StatementFieldResponse(
        request_id=request_id, disposition=selected.disposition, reason=selected.reason,
        selection_request=payload, selection_request_sha256=sha256(_canonical(payload.model_dump(mode='json'))).hexdigest(),
        index_sha256=index_sha,
        cells=[SourceCandidate.model_validate({**asdict(cell), 'period_kind': cell.period_kind})
               for cell in selected.cells],
        financial_correctness=selected.financial_correctness, score_semantics=selected.score_semantics,
    )
    # Typed validation and actual JSON rendering share the safe result boundary.
    return JSONResponse(content=response.model_dump(mode='json'))


def _unavailable(request_id: str, *, reason: str, index_sha: str | None) -> JSONResponse:
    return JSONResponse(status_code=503, content={
        'schema_version': 1, 'request_id': request_id, 'disposition': 'unavailable',
        'reason': reason, 'selection_request': None, 'selection_request_sha256': None,
        'index_sha256': index_sha, 'cells': [], 'financial_correctness': 'unestablished',
        'score_semantics': _SCORE_SEMANTICS, 'error_code': 'statement_selection_unavailable',
    })


def attach_statement_fields_route(app: FastAPI, *, index: IncomeIndex | None,
                                  require_bearer: Callable, logger: logging.Logger) -> None:
    """Register a strict typed route, independent of question parsing and query."""
    index_sha = None
    if index is not None:
        validate_income_index(index)
        value = asdict(index)
        value.pop('_construction_token')
        index_sha = sha256(_canonical(value)).hexdigest()

    @app.post(
        '/v1/statement-fields', response_model=StatementFieldResponse,
        dependencies=[Depends(require_bearer)],
        openapi_extra={'requestBody': {'required': True, 'content': {
            'application/json': {'schema': StatementFieldPayload.model_json_schema()}}}},
        responses={
            401: {'description': 'Missing or invalid bearer token'},
            413: {'description': 'Actual request body exceeds 8192 bytes'},
            415: {'description': 'Unsupported request media type'},
            422: {'description': 'Invalid statement field request; input details are withheld'},
            503: {'model': StatementFieldResponse, 'description': 'Statement field selection unavailable'},
        },
    )
    async def statement_fields(request: Request) -> JSONResponse:
        request_id = request.state.request_id
        if index is None:
            return _unavailable(request_id, reason='STATEMENT_INDEX_NOT_CONFIGURED', index_sha=None)
        try:
            payload = await _read_field_request(request)
        except HTTPException:
            raise
        except Exception:
            logger.error('statement_field_selection_unavailable request_id=%s', request_id)
            return _unavailable(request_id, reason='STATEMENT_SELECTION_UNAVAILABLE', index_sha=index_sha)
        try:
            return await run_in_threadpool(_completed_response, index, payload, request_id, index_sha)
        except Exception:
            logger.error('statement_field_selection_unavailable request_id=%s', request_id)
            return _unavailable(request_id, reason='STATEMENT_SELECTION_UNAVAILABLE', index_sha=index_sha)
