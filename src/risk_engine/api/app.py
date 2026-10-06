"""Versioned offline Risk Signal HTTP boundary."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Annotated, Literal, cast
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime, field_validator, model_validator
from starlette.exceptions import HTTPException

from risk_engine.api.dependencies import (
    AnalysisRecord,
    AppContainer,
    ArtifactNotFoundError,
    SnapshotIds,
    VersionConflictError,
)
from risk_engine.data.duckdb_store import SnapshotManifestError, SnapshotNotFoundError
from risk_engine.domain import DomainModel, NonEmptyString, RiskSignal, SourceItem


class AnalyzeSignalsRequest(DomainModel):
    snapshot_ids: SnapshotIds
    as_of: AwareDatetime

    @field_validator("as_of", mode="before")
    @classmethod
    def explicit_timestamp(cls, value: object) -> object:
        if not isinstance(value, str | datetime) or (isinstance(value, str) and "T" not in value):
            raise ValueError("as_of requires an explicit ISO 8601 timestamp")
        return value

    @model_validator(mode="after")
    def unique_snapshot_ids(self) -> AnalyzeSignalsRequest:
        if len(set(self.snapshot_ids)) != len(self.snapshot_ids):
            raise ValueError("snapshot IDs must be unique")
        return self


class ErrorResponse(DomainModel):
    code: NonEmptyString
    message: NonEmptyString
    correlation_id: NonEmptyString


class HealthResponse(DomainModel):
    status: Literal["ok"]


def get_container(request: Request) -> AppContainer:
    return cast(AppContainer, request.app.state.container)


ContainerDependency = Annotated[AppContainer, Depends(get_container)]


def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    record = ErrorResponse(code=code, message=message, correlation_id=request.state.correlation_id)
    return JSONResponse(status_code=status, content=record.model_dump(mode="json"))


def create_app(container: AppContainer) -> FastAPI:
    """Compose caller-supplied local ports; construction performs no processing."""
    app = FastAPI(title="Financial Risk Engine", version="0.1.0")
    app.state.container = container

    @app.middleware("http")
    async def correlate(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.correlation_id = str(uuid4())
        try:
            response = await call_next(request)
        except Exception:
            response = _error(request, 500, "internal_error", "Request processing failed")
        response.headers["X-Correlation-ID"] = request.state.correlation_id
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        return _error(request, 422, "invalid_request", "Invalid request")

    @app.exception_handler(ValueError)
    async def invalid_domain(request: Request, error: ValueError) -> JSONResponse:
        return _error(request, 422, "invalid_domain", "Invalid domain data")

    @app.exception_handler(SnapshotNotFoundError)
    async def snapshot_missing(request: Request, error: SnapshotNotFoundError) -> JSONResponse:
        return _error(request, 404, "snapshot_not_found", "Snapshot not found")

    @app.exception_handler(ArtifactNotFoundError)
    async def artifact_missing(request: Request, error: ArtifactNotFoundError) -> JSONResponse:
        return _error(request, 404, "artifact_not_found", "Artifact not found")

    @app.exception_handler(VersionConflictError)
    @app.exception_handler(SnapshotManifestError)
    async def version_conflict(request: Request, error: Exception) -> JSONResponse:
        return _error(request, 409, "version_conflict", "Immutable identity or version conflict")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        if error.status_code == 404:
            return _error(request, 404, "not_found", "Route not found")
        if error.status_code == 405:
            return _error(request, 405, "method_not_allowed", "Method not allowed")
        return _error(request, error.status_code, "http_error", "Request rejected")

    errors: dict[int | str, dict[str, object]] = {
        404: {"model": ErrorResponse, "description": "Requested immutable snapshot not found"},
        409: {"model": ErrorResponse, "description": "Immutable identity or version conflict"},
        422: {"model": ErrorResponse, "description": "Invalid request or domain data"},
        500: {"model": ErrorResponse, "description": "Request processing failed"},
    }

    @app.post("/v1/signals/analyze", response_model=list[RiskSignal], responses=errors)
    def analyze_signals(
        analysis: AnalyzeSignalsRequest, dependencies: ContainerDependency
    ) -> list[RiskSignal]:
        items: list[SourceItem] = []
        for snapshot_id in analysis.snapshot_ids:
            replayed = dependencies.data.replay(snapshot_id)
            for original in replayed:
                item = SourceItem.model_validate(original.model_dump(mode="python"))
                if item.snapshot_id != snapshot_id:
                    raise ValueError("replayed Source Item snapshot identity mismatch")
                items.append(item)
        signals = dependencies.risk_engine.analyze(items, analysis.as_of)
        record = AnalysisRecord(
            snapshot_ids=analysis.snapshot_ids, as_of=analysis.as_of, signals=tuple(signals)
        )
        dependencies.signals.publish(record)
        return list(record.signals)

    @app.get("/v1/signals", response_model=list[RiskSignal], responses=errors)
    def list_signals(dependencies: ContainerDependency) -> list[RiskSignal]:
        return dependencies.signals.list()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    return app
