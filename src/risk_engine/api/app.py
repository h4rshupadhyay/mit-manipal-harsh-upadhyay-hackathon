"""Versioned offline Risk Signal HTTP boundary."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import datetime
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from typing import Annotated, Literal, cast
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime, ModelWrapValidatorHandler, field_validator, model_validator
from starlette.exceptions import HTTPException

from risk_engine.api.dependencies import (
    AnalysisRecord,
    AppContainer,
    ArtifactNotFoundError,
    SnapshotIds,
    StoredSignal,
    VersionConflictError,
)
from risk_engine.data.duckdb_store import SnapshotManifestError, SnapshotNotFoundError
from risk_engine.domain import (
    BacktestReport,
    DomainModel,
    MarketSnapshot,
    NonEmptyString,
    Portfolio,
    PortfolioMateriality,
    RiskSignal,
    SourceItem,
    StressResult,
    StressScenario,
)
from risk_engine.risk.policy import ManualOverride, TriggerDecision
from risk_engine.stress.shocks import normalize_scenario


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


class StressTestRequest(DomainModel):
    """Explicit immutable lookups plus an optional new analyst scenario."""

    signal_id: NonEmptyString
    portfolio_id: NonEmptyString
    portfolio_version: NonEmptyString
    market_snapshot_id: NonEmptyString
    scenario_id: NonEmptyString
    calibration_version: NonEmptyString
    scenario: StressScenario | None = None
    manual_override: ManualOverride | None = None

    @model_validator(mode="after")
    def manual_scenario_requires_audit(self) -> StressTestRequest:
        scenario, audit = self.scenario, self.manual_override
        if (scenario is None) != (audit is None):
            raise ValueError("a new scenario and manual audit are required together")
        if (
            scenario is not None
            and audit is not None
            and (
                scenario.scenario_id == self.scenario_id
                or not scenario.analyst_override
                or scenario.override_reason != audit.reason
            )
        ):
            raise ValueError("manual scenario requires a new identity and matching override reason")
        return self


def _arithmetic_context() -> Context:
    return Context(
        prec=28,
        rounding=ROUND_HALF_EVEN,
        Emin=-999999,
        Emax=999999,
        capitals=1,
        clamp=0,
        flags=[],
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )


def _materiality(result: StressResult) -> PortfolioMateriality:
    # A gain must not produce economic-loss eligibility on a short-value book.
    positive_loss = result.absolute_loss > 0
    return PortfolioMateriality(
        absolute_loss=result.absolute_loss if positive_loss else Decimal(0),
        percentage_loss=max(0, result.percentage_loss) if positive_loss else 0,
        currency=result.valuation_currency,
    )


ExecutionStatus = Literal["automatic", "manual", "blocked"]


def _execution_status(decision: TriggerDecision) -> ExecutionStatus:
    if decision.manual_override is not None:
        return "manual"
    return "automatic" if decision.automatic_trigger else "blocked"


class StressTestResponse(DomainModel):
    """Read-only valuation preview and explicit production action permission."""

    execution_status: ExecutionStatus
    signal_id: NonEmptyString
    source_item_id: NonEmptyString
    snapshot_ids: SnapshotIds
    as_of: AwareDatetime
    portfolio_id: NonEmptyString
    portfolio_version: NonEmptyString
    market_snapshot_id: NonEmptyString
    original_scenario: StressScenario
    scenario: StressScenario
    stress_result: StressResult
    trigger_decision: TriggerDecision

    @model_validator(mode="wrap")
    @classmethod
    def fixed_decimal_validation(
        cls, value: object, handler: ModelWrapValidatorHandler[StressTestResponse]
    ) -> StressTestResponse:
        # Protect nested canonical validators before they perform arithmetic.
        with localcontext(_arithmetic_context()):
            return handler(value)

    @model_validator(mode="after")
    def response_preserves_lineage(self) -> StressTestResponse:
        # FastAPI response validation may execute after the route's context exits.
        with localcontext(_arithmetic_context()):
            result = StressResult.model_validate(self.stress_result.model_dump(mode="python"))
            decision = TriggerDecision.model_validate(
                self.trigger_decision.model_dump(mode="python")
            )
            if (
                decision.signal_id != self.signal_id
                or decision.source_item_id != self.source_item_id
                or decision.as_of != self.as_of
                or decision.portfolio_materiality != _materiality(result)
                or self.execution_status != _execution_status(decision)
                or self.scenario.risk_signal_id != self.signal_id
                or self.original_scenario.risk_signal_id != self.signal_id
                or self.scenario.calibration_version != self.original_scenario.calibration_version
                or result.scenario_id != self.scenario.scenario_id
                or result.portfolio_id != self.portfolio_id
                or result.versions.portfolio_version != self.portfolio_version
                or result.versions.scenario_version != self.scenario.calibration_version
                or json.loads(result.versions.market_version).get("snapshot_id")
                != self.market_snapshot_id
            ):
                raise ValueError("Stress Test response lineage disagrees")
            audit = decision.manual_override
            if audit is None:
                if self.scenario != self.original_scenario or self.scenario.analyst_override:
                    raise ValueError("unaudited manual scenario")
            elif (
                self.scenario.scenario_id == self.original_scenario.scenario_id
                or not self.scenario.analyst_override
                or self.scenario.override_reason != audit.reason
            ):
                raise ValueError("manual scenario lineage disagrees")
        return self


def _validate_source_lineage(stored: StoredSignal, dependencies: AppContainer) -> None:
    matched: list[SourceItem] = []
    for snapshot_id in stored.snapshot_ids:
        for original in dependencies.data.replay(snapshot_id):
            item = SourceItem.model_validate(original.model_dump(mode="python"))
            if item.snapshot_id != snapshot_id:
                raise VersionConflictError("Source Item snapshot identity disagrees")
            if item.source_item_id == stored.signal.source_item_id:
                matched.append(item)
    if len(matched) != 1:
        raise VersionConflictError("stored Source Item lineage is missing or ambiguous")
    if matched[0].published_at > stored.as_of or matched[0].retrieved_at > stored.as_of:
        raise ValueError("Source Item was unavailable at the stored analysis cutoff")


def _stress_test(analysis: StressTestRequest, dependencies: AppContainer) -> StressTestResponse:
    inputs, engine, policy = (
        dependencies.stress_inputs,
        dependencies.stress_engine,
        dependencies.trigger_policy,
    )
    if inputs is None or engine is None or policy is None:
        raise ArtifactNotFoundError("Stress Test ports are not configured")
    with localcontext(_arithmetic_context()):
        stored = StoredSignal.model_validate(
            dependencies.signals.get(analysis.signal_id).model_dump(mode="python")
        )
        if stored.signal.signal_id != analysis.signal_id:
            raise VersionConflictError("stored Risk Signal identity disagrees")
        _validate_source_lineage(stored, dependencies)
        portfolio = Portfolio.model_validate(
            inputs.portfolio(analysis.portfolio_id, analysis.portfolio_version).model_dump(
                mode="python"
            )
        )
        market = MarketSnapshot.model_validate(
            inputs.market(analysis.market_snapshot_id).model_dump(mode="python")
        )
        original = StressScenario.model_validate(
            inputs.scenario(analysis.scenario_id, analysis.calibration_version).model_dump(
                mode="python"
            )
        )
        if (
            (portfolio.portfolio_id, portfolio.version)
            != (analysis.portfolio_id, analysis.portfolio_version)
            or market.snapshot_id != analysis.market_snapshot_id
            or (original.scenario_id, original.calibration_version)
            != (analysis.scenario_id, analysis.calibration_version)
            or original.risk_signal_id != analysis.signal_id
            or original.calibration_version != stored.signal.impact.calibration_version
        ):
            raise VersionConflictError("immutable Stress Test input bindings disagree")
        if portfolio.as_of != market.as_of or market.as_of > stored.as_of:
            raise ValueError("valuation inputs disagree with the stored analysis cutoff")
        if original.analyst_override:
            raise ValueError("original scenario requires its own retained analyst audit")
        scenario = analysis.scenario if analysis.scenario is not None else original
        if (
            scenario.risk_signal_id != analysis.signal_id
            or scenario.calibration_version != original.calibration_version
        ):
            raise VersionConflictError("manual scenario bindings disagree")
        normalized = normalize_scenario(scenario)  # Reject all shocks before valuation.
        result = StressResult.model_validate(
            engine.run(portfolio, market, normalized).model_dump(mode="python")
        )
        expected_market_version = json.dumps(
            {
                "snapshot_id": market.snapshot_id,
                "schema": market.schema_version,
                "normalization": market.normalization_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if (
            result.scenario_id != scenario.scenario_id
            or result.portfolio_id != portfolio.portfolio_id
            or result.valuation_currency != portfolio.valuation_currency
            or result.versions.portfolio_version != portfolio.version
            or result.versions.scenario_version != scenario.calibration_version
            or result.versions.market_version != expected_market_version
        ):
            raise VersionConflictError("Stress Result bindings disagree with valuation inputs")
        materiality = _materiality(result)
        decision = TriggerDecision.model_validate(
            policy.evaluate(
                stored.signal, materiality, manual_override=analysis.manual_override
            ).model_dump(mode="python")
        )
        if (
            decision.risk_signal != stored.signal
            or decision.as_of != stored.as_of
            or decision.portfolio_materiality != materiality
            or decision.manual_override != analysis.manual_override
        ):
            raise VersionConflictError("TriggerDecision bindings disagree with requested inputs")
        return StressTestResponse(
            execution_status=_execution_status(decision),
            signal_id=analysis.signal_id,
            source_item_id=stored.signal.source_item_id,
            snapshot_ids=stored.snapshot_ids,
            as_of=stored.as_of,
            portfolio_id=portfolio.portfolio_id,
            portfolio_version=portfolio.version,
            market_snapshot_id=market.snapshot_id,
            original_scenario=original,
            scenario=scenario,
            stress_result=result,
            trigger_decision=decision,
        )


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

    @app.post("/v1/stress-tests", response_model=StressTestResponse, responses=errors)
    def stress_tests(
        analysis: StressTestRequest, dependencies: ContainerDependency
    ) -> StressTestResponse:
        return _stress_test(analysis, dependencies)

    @app.get("/v1/backtests/latest", response_model=BacktestReport, responses=errors)
    def latest_backtest(dependencies: ContainerDependency) -> BacktestReport:
        if dependencies.backtests is None:
            raise ArtifactNotFoundError("Backtest reader is not configured")
        return BacktestReport.model_validate(
            dependencies.backtests.latest().model_dump(mode="python")
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    return app
