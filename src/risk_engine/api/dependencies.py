"""Explicit local ports for the HTTP composition boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Annotated, Protocol

from pydantic import AwareDatetime, Field, model_validator

from risk_engine.backtest.module import (
    ARTIFACT_PATHS,
    ArtifactEnvelope,
    CutpointsPayload,
    DevelopmentPayload,
    FinalArtifact,
    SelectedPayload,
    UnresolvedPayload,
)
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
from risk_engine.stress.interfaces import NormalizedScenario

SnapshotIds = Annotated[tuple[NonEmptyString, ...], Field(min_length=1)]


class ArtifactNotFoundError(RuntimeError):
    """Requested offline evidence is missing or explicitly unresolved."""


class VersionConflictError(RuntimeError):
    """An immutable identity is already bound to different content or versions."""


class SnapshotReplayPort(Protocol):
    def replay(self, snapshot_id: str) -> list[SourceItem]: ...


class RiskEnginePort(Protocol):
    def analyze(self, items: Sequence[SourceItem], as_of: datetime) -> list[RiskSignal]: ...


class AnalysisRecord(DomainModel):
    snapshot_ids: SnapshotIds
    as_of: AwareDatetime
    signals: tuple[RiskSignal, ...]

    @model_validator(mode="after")
    def unique_identities(self) -> AnalysisRecord:
        if len(set(self.snapshot_ids)) != len(self.snapshot_ids):
            raise ValueError("snapshot IDs must be unique")
        ids = [signal.signal_id for signal in self.signals]
        if len(set(ids)) != len(ids):
            raise ValueError("Risk Signal IDs must be unique")
        return self


class StoredSignal(DomainModel):
    snapshot_ids: SnapshotIds
    as_of: AwareDatetime
    signal: RiskSignal


class SignalRepositoryPort(Protocol):
    """Publish a complete batch atomically; preserve immutable signal IDs."""

    def publish(self, analysis: AnalysisRecord) -> None: ...
    def list(self) -> list[RiskSignal]: ...
    def get(self, signal_id: str) -> StoredSignal: ...


class InMemorySignalRepository:
    """Process-local immutable records; inject durable storage in deployment."""

    def __init__(self) -> None:
        self._records: dict[str, StoredSignal] = {}
        self._lock = RLock()

    def publish(self, analysis: AnalysisRecord) -> None:
        analysis = AnalysisRecord.model_validate(analysis.model_dump(mode="python"))
        pending = {
            signal.signal_id: StoredSignal(
                snapshot_ids=analysis.snapshot_ids, as_of=analysis.as_of, signal=signal
            )
            for signal in analysis.signals
        }
        with self._lock:
            if any(
                key in self._records and self._records[key] != value
                for key, value in pending.items()
            ):
                raise VersionConflictError("immutable Risk Signal identity conflict")
            self._records.update(pending)

    def list(self) -> list[RiskSignal]:
        with self._lock:
            return [record.signal for record in self._records.values()]

    def get(self, signal_id: str) -> StoredSignal:
        with self._lock:
            try:
                return self._records[signal_id]
            except KeyError as error:
                raise ArtifactNotFoundError("Risk Signal not found") from error


class StressEnginePort(Protocol):
    def run(
        self,
        portfolio: Portfolio,
        base_market: MarketSnapshot,
        scenario: StressScenario | NormalizedScenario,
    ) -> StressResult: ...


class TriggerPolicyPort(Protocol):
    def evaluate(
        self,
        signal: RiskSignal,
        portfolio_materiality: PortfolioMateriality,
        *,
        manual_override: ManualOverride | None = None,
    ) -> TriggerDecision: ...


class StressInputsPort(Protocol):
    """Read explicit immutable identities; missing/conflicting records raise safe errors."""

    def portfolio(self, portfolio_id: str, version: str) -> Portfolio: ...
    def market(self, snapshot_id: str) -> MarketSnapshot: ...
    def scenario(self, scenario_id: str, calibration_version: str) -> StressScenario: ...


class BacktestArtifactReaderPort(Protocol):
    def latest(self) -> BacktestReport: ...


class LocalBacktestArtifactReader:
    """Read and verify configured artifacts only, without evaluating a Backtest."""

    def __init__(self, *, root: Path, final_path: Path | None) -> None:
        self._root = root
        self._final_path = final_path

    def latest(self) -> BacktestReport:
        try:
            if self._final_path is not None:
                final = FinalArtifact.model_validate_json(self._final_path.read_bytes())
                if final.status != "complete" or final.report is None:
                    raise ArtifactNotFoundError("complete final Backtest unavailable")
                return final.report
            return self._development_report()
        except FileNotFoundError as error:
            raise ArtifactNotFoundError("Backtest artifact not found") from error
        except (ValueError, OSError) as error:
            raise VersionConflictError("Backtest artifact validation conflict") from error

    def _development_report(self) -> BacktestReport:
        # Same module-record/receipt checks as the Task 27 CLI loader, without
        # importing an unpackaged scripts module into the installed API.
        envelopes = tuple(
            ArtifactEnvelope.model_validate_json((self._root / path).read_bytes())
            for path in ARTIFACT_PATHS
        )
        first = envelopes[0]
        for envelope, role in zip(
            envelopes, ("selected-config", "impact-cutpoints", "development-backtest"), strict=True
        ):
            if (
                envelope.artifact_set_hash != first.artifact_set_hash
                or envelope.payload_hashes != first.payload_hashes
                or envelope.payload.artifact_type != role
            ):
                raise ValueError("Backtest artifact receipt or role conflict")
        if first.status != "ready":
            if any(not isinstance(e.payload, UnresolvedPayload) for e in envelopes):
                raise ValueError("inconsistent unresolved artifact roles")
            raise ArtifactNotFoundError("Backtest evidence unresolved")
        selected, cutpoints, development = (e.payload for e in envelopes)
        if (
            not isinstance(selected, SelectedPayload)
            or not isinstance(cutpoints, CutpointsPayload)
            or not isinstance(development, DevelopmentPayload)
        ):
            raise ValueError("ready artifact payload roles conflict")
        lock = selected.locked
        if (
            development.audit.locked != lock
            or (cutpoints.fit_hash, cutpoints.training_hash, cutpoints.identity)
            != (lock.fitted.manifest_hash, lock.fitted.training_hash, lock.fitted.impact)
            or any(e.evidence_kind != lock.evidence_kind for e in envelopes)
        ):
            raise ValueError("Backtest artifact selected lock or fit conflict")
        return development.report


@dataclass(frozen=True, kw_only=True)
class AppContainer:
    data: SnapshotReplayPort
    risk_engine: RiskEnginePort
    signals: SignalRepositoryPort
    stress_engine: StressEnginePort | None = None
    trigger_policy: TriggerPolicyPort | None = None
    stress_inputs: StressInputsPort | None = None
    backtests: BacktestArtifactReaderPort | None = None
