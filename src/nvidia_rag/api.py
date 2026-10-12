"""Authenticated FastAPI surface for a production-like RAG service boundary."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
import secrets
import time
from typing import Literal, Protocol
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .core import GroundedContext
from .generation import GeneratedResponse
from .sources import _reviewed_provenance
from .structured_income import IncomeIndex
from .statement_api import attach_statement_route

LOGGER = logging.getLogger("nvidia_rag.api")
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}\Z")


def load_api_provenance(
    manifest_path: Path, *, chunks_path: Path
) -> dict[str, Mapping[str, object]]:
    """Bind reviewed attribution to exact local corpus bytes, not a filename."""
    from .ingest import load_chunks

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    if not any("provenance" in record for record in documents):
        return {}  # Legacy official corpus: do not invent provenance.
    observed_hash = sha256(chunks_path.read_bytes()).hexdigest()
    if manifest.get("chunks_sha256") != observed_hash:
        raise ValueError("corpus chunks hash missing or mismatched")
    chunks = load_chunks(chunks_path)
    actual_counts = Counter(chunk.document_id for chunk in chunks)
    declared_counts = {record["document_id"]: record["chunk_count"] for record in documents}
    if (len(declared_counts) != len(documents) or declared_counts != dict(actual_counts)
            or manifest.get("document_count") != len(documents)
            or manifest.get("chunk_count") != len(chunks)):
        raise ValueError("corpus document identities or counts mismatched")
    result: dict[str, Mapping[str, object]] = {}
    for record in documents:
        if "provenance" not in record:
            continue
        reviewed = _reviewed_provenance(record["provenance"])
        if "display" not in reviewed["permitted_operations"]:
            raise ValueError("source display permission missing")
        if "index" not in reviewed["permitted_operations"]:
            raise ValueError("source index permission missing")
        result[record["document_id"]] = reviewed
    return result


def _request_id_from_header(value: str | None) -> str:
    if value is not None and _SAFE_REQUEST_ID.fullmatch(value):
        return value
    return str(uuid4())


class QueryService(Protocol):
    def query(self, query: str) -> GroundedContext | GeneratedResponse: ...


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=500)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class EvidenceItem(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    pdf_page: int
    score: float
    text: str
    provenance: dict[str, str] | None = None


class QueryResponse(BaseModel):
    request_id: str
    can_answer: bool
    evidence_passed: bool
    message: str
    answer: str
    citations: list[str]
    evidence: list[EvidenceItem]
    error_code: Literal[
        "generation_timeout", "generation_unavailable", "query_service_unavailable"
    ] | None = None


def resolve_api_token(environment: Mapping[str, str]) -> str:
    """Fail closed when a production API token is absent or trivially weak."""
    token = environment.get("NVIDIA_RAG_API_TOKEN", "")
    if len(token) < 32:
        raise ValueError("NVIDIA_RAG_API_TOKEN must contain at least 32 characters")
    return token


def create_app(
    *,
    service: QueryService,
    api_token: str,
    expose_docs: bool = False,
    source_provenance: Mapping[str, Mapping[str, object]] | None = None,
    statement_index: IncomeIndex | None = None,
) -> FastAPI:
    """Build an injected, testable API without loading models at import time."""
    if len(api_token) < 32:
        raise ValueError("api_token must contain at least 32 characters")

    public_provenance: dict[str, dict[str, str]] = {}
    for document_id, metadata in (source_provenance or {}).items():
        row = dict(metadata)
        if isinstance(row.get("permitted_operations"), tuple):
            row["permitted_operations"] = list(row["permitted_operations"])
        reviewed = _reviewed_provenance(row)
        if "display" not in reviewed["permitted_operations"]:
            raise ValueError("source display permission missing")
        if "index" not in reviewed["permitted_operations"]:
            raise ValueError("source index permission missing")
        public_provenance[document_id] = {
            key: str(reviewed[key]) for key in (
                "source_type", "publisher", "attribution", "published_date",
                "event_date", "financial_period", "source_locator",
            )
        }

    app = FastAPI(
        title="NVIDIA Financial RAG API",
        version="0.1.0",
        docs_url="/docs" if expose_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if expose_docs else None,
    )

    def require_bearer(authorization: str | None = Header(default=None)) -> None:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        supplied = authorization.removeprefix("Bearer ")
        try:
            matches = secrets.compare_digest(supplied, api_token)
        except TypeError:
            # A non-ASCII header must be an authentication refusal, not a 500.
            matches = False
        if not matches:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.middleware("http")
    async def add_request_metadata(request: Request, call_next):
        request_id = _request_id_from_header(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        started = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        LOGGER.info(
            "request_complete method=%s path=%s status=%s request_id=%s latency_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            request_id,
            elapsed_ms,
        )
        return response

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False, dependencies=[Depends(require_bearer)])
    def readyz() -> dict[str, str]:
        return {"status": "ready"}

    @app.post(
        "/v1/query",
        response_model=QueryResponse,
        dependencies=[Depends(require_bearer)],
        responses={
            503: {"model": QueryResponse, "description": "Query or generation service unavailable"},
            504: {"model": QueryResponse, "description": "Generation request timed out"},
        },
    )
    def query(payload: QueryRequest, request: Request) -> QueryResponse | JSONResponse:
        try:
            result = service.query(payload.question)
        except Exception:
            LOGGER.error("query_service_unavailable request_id=%s", request.state.request_id)
            unavailable = QueryResponse(
                request_id=request.state.request_id,
                can_answer=False,
                evidence_passed=False,
                message="The query service is unavailable. Please try again later.",
                answer="",
                citations=[],
                evidence=[],
                error_code="query_service_unavailable",
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=unavailable.model_dump(),
            )
        evidence = [
            EvidenceItem(
                rank=rank,
                chunk_id=item.chunk.chunk_id,
                document_id=item.chunk.document_id,
                pdf_page=item.chunk.page_number,
                score=item.score,
                text=item.chunk.text,
                provenance=public_provenance.get(item.chunk.document_id),
            )
            for rank, item in enumerate(result.evidence, start=1)
        ]
        if isinstance(result, GeneratedResponse):
            operational_failure = result.generation_failure
            response = QueryResponse(
                request_id=request.state.request_id,
                can_answer=False if operational_failure else result.can_answer,
                evidence_passed=result.evidence_passed,
                message=(
                    "Evidence passed, but the generation request timed out. Please try again later."
                    if operational_failure == "timeout" else
                    "Evidence passed, but the generation service is unavailable. Please try again later."
                    if operational_failure == "unavailable" else result.message
                ),
                answer="" if operational_failure else result.answer,
                citations=list(result.citations),
                evidence=evidence,
                error_code=(
                    "generation_timeout" if operational_failure == "timeout" else
                    "generation_unavailable" if operational_failure == "unavailable" else None
                ),
            )
            if operational_failure is not None:
                LOGGER.error("%s request_id=%s", response.error_code, request.state.request_id)
                return JSONResponse(
                    status_code=(status.HTTP_504_GATEWAY_TIMEOUT
                                 if operational_failure == "timeout"
                                 else status.HTTP_503_SERVICE_UNAVAILABLE),
                    content=response.model_dump(),
                )
            return response
        return QueryResponse(
            request_id=request.state.request_id,
            can_answer=result.can_answer,
            evidence_passed=result.can_answer,
            message=result.message,
            answer="",
            citations=list(result.citations),
            evidence=evidence,
        )

    attach_statement_route(
        app, index=statement_index, require_bearer=require_bearer, logger=LOGGER,
    )
    return app
