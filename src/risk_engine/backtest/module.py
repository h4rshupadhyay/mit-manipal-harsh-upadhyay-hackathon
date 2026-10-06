"""Offline chronological backtests. Composition ports are trusted, never score providers.

Synthetic replay proves mechanics only. Policy evidence and local artefact identities
are checked, but this module cannot authenticate a deliberately dishonest closure.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeVar

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    TypeAdapter,
    field_validator,
    model_validator,
)

from risk_engine.backtest import metrics as m
from risk_engine.backtest.splits import (
    ChronologicalSplitSpec,
    FoldGroup,
    OuterFold,
    nested_chronological_splits,
)
from risk_engine.config import ClusteringConfig, PolicyConfig, SelectedTriggerPolicy
from risk_engine.data.clustering import StoryCluster, cluster_stories
from risk_engine.domain import (
    BacktestReport,
    CandidateConfiguration,
    ConfidenceTarget,
    ConfigurationScalar,
    CurrencyCode,
    EventClass,
    FactorShock,
    MarketSnapshot,
    NonEmptyString,
    Portfolio,
    PortfolioMateriality,
    RiskSignal,
    SensitivityResult,
    Sha256Hex,
    SourceItem,
    StressResult,
    StressScenario,
)
from risk_engine.impact.module import ImpactAudit
from risk_engine.risk.module import (
    CalibrationManifest,
    ConfidenceCalibrationIdentity,
    ModelIdentity,
    ModelManifest,
    SnapshotManifest,
)
from risk_engine.risk.policy import TriggerDecision, TriggerPolicy
from risk_engine.stress.module import StressEngine

UTC = timezone.utc  # noqa: UP017 -- inherited verification runtime is Python 3.10
HASH_DOMAIN = "backtest-canonical-json-v1"
EvidenceKind = Literal["synthetic", "empirical"]
NonemptyTerms = Annotated[tuple[NonEmptyString, ...], Field(min_length=1)]


def _utc(value: object) -> object:
    if isinstance(value, BaseModel):
        return _utc(value.model_dump(mode="python", warnings=False))
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)
    if isinstance(value, dict):
        return {k: _utc(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return tuple(_utc(v) for v in value)
    return value


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="python", warnings=False))
    if isinstance(value, datetime):
        instant = _utc(value)
        assert isinstance(instant, datetime)
        return instant.isoformat().replace("+00:00", "Z")
    if isinstance(value, timedelta):
        return TypeAdapter(timedelta).dump_python(value, mode="json")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("nonfinite Decimal")
        return str(value)
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(v) for v in value]
    return value


def canonical_bytes(value: object) -> bytes:
    """Declared UTF-8 JSON hash domain, exact Decimal strings, UTC timestamps."""
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _decimal_context() -> Context:
    return Context(prec=28, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)


class Record(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    @model_validator(mode="wrap")
    @classmethod
    def deterministic_validation(
        cls, value: object, handler: ModelWrapValidatorHandler[Record]
    ) -> Record:
        with localcontext(_decimal_context()):
            return handler(value)

    @model_validator(mode="before")
    @classmethod
    def normalize_instants(cls, value: object) -> object:
        return _utc(value)


R = TypeVar("R", bound=BaseModel)


def _copy(record: R) -> R:
    with localcontext(_decimal_context()):
        return type(record).model_validate(record.model_dump(mode="python", warnings=False))


class SnapshotReference(Record):
    snapshot_id: NonEmptyString
    content_hash: Sha256Hex
    source_terms: NonemptyTerms


class ReplayInput(Record):
    case_id: NonEmptyString
    cluster: StoryCluster
    as_of: AwareDatetime
    portfolio: Portfolio
    market: MarketSnapshot
    market_available_at: AwareDatetime
    market_source: SnapshotReference

    @model_validator(mode="after")
    def historical_inputs(self) -> ReplayInput:
        representative = self.cluster.items[0]
        if representative.retrieved_at > self.as_of:
            raise ValueError("representative Source Item unavailable by historical as_of")
        if self.portfolio.as_of > self.as_of:
            raise ValueError("portfolio unavailable by historical as_of")
        if self.market.as_of > self.as_of or self.market_available_at > self.as_of:
            raise ValueError("market unavailable by historical as_of")
        if self.market_available_at < self.market.as_of:
            raise ValueError("market availability precedes observed snapshot")
        if (
            self.market_source.snapshot_id != self.market.snapshot_id
            or self.market_source.content_hash != content_hash(self.market)
        ):
            raise ValueError("market snapshot content hash mismatch")
        return self


class ReferenceLoss(Record):
    basket_hash: Sha256Hex
    loss: Decimal
    currency: CurrencyCode
    derivation_reference: NonEmptyString
    derivation_hash: Sha256Hex
    available_at: AwareDatetime


class RealizedValuation(Record):
    pnl: Decimal
    currency: CurrencyCode
    horizon_days: Annotated[int, Field(strict=True, gt=0)]
    comparison_scope: NonEmptyString
    baseline_market_hash: Sha256Hex
    position_ids: tuple[NonEmptyString, ...]
    gross_values: tuple[tuple[NonEmptyString, Annotated[Decimal, Field(ge=0)]], ...]

    @model_validator(mode="after")
    def scope_is_canonical(self) -> RealizedValuation:
        if self.position_ids != tuple(sorted(set(self.position_ids))):
            raise ValueError("valuation position IDs must be sorted and unique")
        ids = tuple(k for k, _ in self.gross_values)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("gross value IDs must be sorted and unique")
        return self


class ObservedOutcome(Record):
    case_id: NonEmptyString
    actual_event_class: EventClass
    actual_entity_id: NonEmptyString
    label_available_at: AwareDatetime
    outcome_available_at: AwareDatetime
    source_terms: NonemptyTerms
    evidence_reference: NonEmptyString
    evidence_hash: Sha256Hex
    realized_shocks: tuple[FactorShock, ...] | None
    scenario_absence_reason: NonEmptyString | None
    valuation: RealizedValuation | None
    valuation_absence_reason: NonEmptyString | None
    reference_losses: tuple[ReferenceLoss, ...]
    reference_loss_absence_reason: NonEmptyString | None
    material_event: bool | None
    material_event_absence_reason: NonEmptyString | None

    @model_validator(mode="after")
    def explicit_missing_evidence(self) -> ObservedOutcome:
        for evidence, reason in (
            (self.realized_shocks, self.scenario_absence_reason),
            (self.valuation, self.valuation_absence_reason),
            (self.reference_losses or None, self.reference_loss_absence_reason),
            (self.material_event, self.material_event_absence_reason),
        ):
            if (evidence is None) != (reason is not None):
                raise ValueError("missing evidence requires exactly one absence reason")
        if self.realized_shocks is not None and not self.realized_shocks:
            raise ValueError("realized Joint Shock Vector cannot be empty")
        if len({r.basket_hash for r in self.reference_losses}) != len(self.reference_losses):
            raise ValueError("duplicate Reference Basket observation")
        payload = self.model_dump(exclude={"evidence_hash"})
        if self.evidence_hash != content_hash(payload):
            raise ValueError("outcome evidence content hash mismatch")
        return self


class BacktestDataset(Record):
    schema_version: Literal["backtest-dataset-v1"]
    snapshot_id: NonEmptyString
    content_hash: Sha256Hex
    evidence_kind: EvidenceKind
    provenance: NonEmptyString
    source_terms: NonemptyTerms
    source_snapshots: tuple[SnapshotReference, ...]
    clustering: ClusteringConfig
    cases: Annotated[tuple[ReplayInput, ...], Field(min_length=1)]
    outcomes: Annotated[tuple[ObservedOutcome, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def complete_frozen_membership(self) -> BacktestDataset:
        ids = tuple(c.case_id for c in self.cases)
        if len(set(ids)) != len(ids) or set(ids) != {o.case_id for o in self.outcomes}:
            raise ValueError("dataset case/outcome membership mismatch")
        if len(self.outcomes) != len(ids):
            raise ValueError("duplicate outcome identity")
        clusters = tuple(c.cluster for c in self.cases)
        items = tuple(i for c in clusters for i in c.items)
        if len({i.source_item_id for i in items}) != len(items):
            raise ValueError("Source Item membership overlaps clusters")
        for item in items:
            if hashlib.sha256(item.text.encode("utf-8")).hexdigest() != item.content_hash:
                raise ValueError("Source Item content hash mismatch")
        rebuilt = tuple(cluster_stories(items, self.clustering))
        if tuple(sorted(clusters, key=lambda c: (c.event_time, c.cluster_id))) != rebuilt:
            raise ValueError("frozen clustering contradicts deterministic clustering")
        snapshots = {s.snapshot_id: s for s in self.source_snapshots}
        if len(snapshots) != len(self.source_snapshots):
            raise ValueError("duplicate source snapshot reference")
        grouped: dict[str, list[SourceItem]] = {}
        for item in items:
            grouped.setdefault(item.snapshot_id, []).append(item)
        if set(grouped) != set(snapshots):
            raise ValueError("source snapshot membership mismatch")
        for snapshot_id, members in grouped.items():
            payload = tuple(sorted(members, key=lambda i: i.source_item_id))
            if snapshots[snapshot_id].content_hash != content_hash(payload):
                raise ValueError("source snapshot content hash mismatch")
        if self.content_hash != content_hash(self.model_dump(exclude={"content_hash"})):
            raise ValueError("dataset content hash mismatch")
        return self


class CandidateSpec(Record):
    candidate_id: NonEmptyString
    version: NonEmptyString
    parameters: dict[NonEmptyString, ConfigurationScalar]
    complexity_dimensions: NonemptyTerms
    complexity: tuple[Annotated[int, Field(strict=True, ge=0)], ...]
    event_window_days: Annotated[int, Field(strict=True, gt=0)]
    basket_convention: NonEmptyString
    matching_convention: NonEmptyString

    @model_validator(mode="after")
    def declared_complexity(self) -> CandidateSpec:
        if len(self.complexity) != len(self.complexity_dimensions) or len(
            set(self.complexity_dimensions)
        ) != len(self.complexity_dimensions):
            raise ValueError("candidate complexity dimensions mismatch")
        return self


class EvaluationPeriod(Record):
    start: AwareDatetime
    end: AwareDatetime
    reporting_cutoff: AwareDatetime

    @model_validator(mode="after")
    def chronological(self) -> EvaluationPeriod:
        if not self.start < self.end <= self.reporting_cutoff:
            raise ValueError("evaluation period requires start < end <= reporting cutoff")
        return self


class DevelopmentConfiguration(Record):
    schema_version: Literal["backtest-development-v1"]
    version: NonEmptyString
    frozen_at: AwareDatetime
    candidates: Annotated[tuple[CandidateSpec, ...], Field(min_length=1)]
    split: ChronologicalSplitSpec
    objective: Literal["event-class-macro-f1"]
    direction: Literal["maximize"]
    sensitivity_parameters: tuple[NonEmptyString, ...]
    reliability_bin_edges: tuple[float, ...]
    nominal_interval_coverage: Annotated[float, Field(gt=0, le=1)]
    alert_budget: Annotated[int, Field(strict=True, ge=0)]
    alert_window: EvaluationPeriod
    final_fit_cutoff: AwareDatetime
    reduction_rule: Literal["greatest-confidence-then-signal-id-v1"]
    projection_rule: Literal["defined-finite-scalars-v1"]
    final_choice_rule: Literal["last-outer-inner-winner-v1"]

    @field_validator("split", mode="before")
    @classmethod
    def restore_declared_durations(cls, value: object) -> object:
        if isinstance(value, dict):
            payload = dict(value)
            for name in ("embargo", "longest_event_window"):
                duration = payload.get(name)
                if isinstance(duration, str):
                    payload[name] = TypeAdapter(timedelta).validate_json(json.dumps(duration))
            return payload
        return value

    @model_validator(mode="after")
    def preregistered(self) -> DevelopmentConfiguration:
        if len({c.candidate_id for c in self.candidates}) != len(self.candidates):
            raise ValueError("duplicate candidate identity")
        if any(
            c.complexity_dimensions != self.candidates[0].complexity_dimensions
            for c in self.candidates
        ):
            raise ValueError("candidate complexity dimensions must match")
        if len(set(self.sensitivity_parameters)) != len(self.sensitivity_parameters):
            raise ValueError("duplicate sensitivity parameter")
        if any(p not in c.parameters for p in self.sensitivity_parameters for c in self.candidates):
            raise ValueError("sensitivity parameter missing from candidate")
        edges = self.reliability_bin_edges
        if (
            len(edges) < 2
            or edges[0] != 0
            or edges[-1] != 1
            or any(a >= b for a, b in pairwise(edges))
        ):
            raise ValueError("reliability bin edges must increase from 0 to 1")
        if any(
            c.event_window_days * 86400 > self.split.longest_event_window.total_seconds()
            for c in self.candidates
        ):
            raise ValueError("candidate event window exceeds registered longest event window")
        return self


class TrainingPartition(Record):
    groups: Annotated[tuple[FoldGroup, ...], Field(min_length=1)]
    cases: tuple[ReplayInput, ...]
    outcomes: tuple[ObservedOutcome, ...]
    cutoff: AwareDatetime
    snapshot_id: NonEmptyString
    dataset_hash: Sha256Hex
    evidence_kind: EvidenceKind
    source_terms: NonemptyTerms
    membership_hash: Sha256Hex

    @model_validator(mode="after")
    def train_only(self) -> TrainingPartition:
        expected = tuple(_group(c) for c in self.cases)
        if (
            expected != self.groups
            or {o.case_id for o in self.outcomes} != {c.case_id for c in self.cases}
            or len(self.outcomes) != len(self.cases)
        ):
            raise ValueError("training membership mismatch")
        if self.membership_hash != content_hash(self.groups):
            raise ValueError("training membership hash mismatch")
        if any(t > self.cutoff for t in self.availability):
            raise ValueError("training evidence unavailable by fit cutoff")
        return self

    @property
    def availability(self) -> tuple[datetime, ...]:
        return tuple(
            t
            for c in self.cases
            for t in (
                c.as_of,
                c.market_available_at,
                c.portfolio.as_of,
                *(i.retrieved_at for i in c.cluster.items),
            )
        ) + tuple(
            t
            for o in self.outcomes
            for t in (
                o.label_available_at,
                o.outcome_available_at,
                *(r.available_at for r in o.reference_losses),
            )
        )


class LocalArtifact(Record):
    identity: NonEmptyString
    path: NonEmptyString
    sha256: Sha256Hex
    available_at: AwareDatetime

    def verify(self, cutoff: datetime) -> None:
        if self.available_at > cutoff:
            raise ValueError("local fitted artifact unavailable by cutoff")
        if hashlib.sha256(Path(self.path).read_bytes()).hexdigest() != self.sha256:
            raise ValueError("local fitted artifact content hash mismatch")


class ImpactFitIdentity(Record):
    evidence_kind: EvidenceKind
    calibration_version: NonEmptyString
    calibration_hash: Sha256Hex
    reference_basket_version: NonEmptyString
    reference_basket_hash: Sha256Hex
    matching_version: NonEmptyString
    training_event_ids: tuple[NonEmptyString, ...]
    cutpoints: Annotated[tuple[Decimal, ...], Field(min_length=9, max_length=9)]
    currency: CurrencyCode
    horizon_days: Annotated[int, Field(strict=True, gt=0)]
    quantile_convention: Literal["nearest-rank-lower-ties"]
    frozen_at: AwareDatetime

    @model_validator(mode="after")
    def ordered_cutpoints(self) -> ImpactFitIdentity:
        if any(a > b for a, b in pairwise(self.cutpoints)):
            raise ValueError("Impact cutpoints must be nondecreasing")
        if len(set(self.training_event_ids)) != len(self.training_event_ids):
            raise ValueError("duplicate Impact training identity")
        return self


class SyntheticImpactAudit(Record):
    schema_version: Literal["synthetic-impact-fit-v1"]
    fit: ImpactFitIdentity


class PolicyTemplate(Record):
    version: NonEmptyString
    validation_status: Literal["chronologically_validated"]
    evidence_kind: Literal["empirical"]
    selection_protocol: Literal["nested_chronological_development"]
    development_start: AwareDatetime
    development_end: AwareDatetime
    frozen_at: AwareDatetime
    validation_evidence: NonEmptyString
    evidence_hash: Sha256Hex
    snapshot_id: NonEmptyString
    source_terms: NonEmptyString
    confidence_target: Literal[ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS]
    confidence_threshold: Annotated[Decimal, Field(gt=0, le=1)]
    economic_floor: Annotated[Decimal, Field(gt=0)]
    materiality_tolerance: Annotated[Decimal, Field(ge=0)]
    currency: CurrencyCode

    @model_validator(mode="after")
    def validate_production_policy_fields(self) -> PolicyTemplate:
        SelectedTriggerPolicy.model_validate(
            {**self.model_dump(), "calibration_version": "per-signal-composite-binding-v1"}
        )
        return self


class PolicyEvidence(Record):
    schema_version: Literal["backtest-policy-binding-v1"]
    candidate_hash: Sha256Hex
    dataset_hash: Sha256Hex
    training_hash: Sha256Hex
    stable_fit_hash: Sha256Hex
    template: PolicyTemplate

    @model_validator(mode="after")
    def bind_complete_selection_evidence(self) -> PolicyEvidence:
        payload = self.model_dump()
        payload["template"].pop("evidence_hash")
        if self.template.evidence_hash != content_hash(payload):
            raise ValueError("policy selection evidence content hash mismatch")
        return self


class FittedManifest(Record):
    schema_version: Literal["backtest-fit-v1"]
    manifest_hash: Sha256Hex
    candidate: CandidateSpec
    configuration_hash: Sha256Hex
    dataset_hash: Sha256Hex
    evidence_kind: EvidenceKind
    training_groups: tuple[FoldGroup, ...]
    training_hash: Sha256Hex
    maximum_evidence_available_at: AwareDatetime
    fit_cutoff: AwareDatetime
    model_identity: ModelIdentity
    model_lock: LocalArtifact
    artifacts: Annotated[tuple[LocalArtifact, ...], Field(min_length=1)]
    confidence: ConfidenceCalibrationIdentity
    confidence_training_source_ids: tuple[NonEmptyString, ...]
    impact: ImpactFitIdentity
    source_terms: NonemptyTerms
    policy: PolicyEvidence | None
    policy_absence_reason: NonEmptyString | None

    @property
    def stable_fit_hash(self) -> str:
        return content_hash(
            {
                "candidate": self.candidate,
                "dataset_hash": self.dataset_hash,
                "training_hash": self.training_hash,
                "models": self.model_identity,
                "model_lock_hash": self.model_lock.sha256,
                "confidence": self.confidence,
                "impact": self.impact,
            }
        )

    @model_validator(mode="after")
    def identities_and_chronology(self) -> FittedManifest:
        if self.training_hash != content_hash(self.training_groups):
            raise ValueError("fit training membership hash mismatch")
        sources = tuple(sorted(s for g in self.training_groups for s in g.source_item_ids))
        if self.confidence_training_source_ids != sources:
            raise ValueError("Confidence training membership mismatch")
        if tuple(g.cluster_id for g in self.training_groups) != self.impact.training_event_ids:
            raise ValueError("Impact training membership mismatch")
        if self.maximum_evidence_available_at > self.fit_cutoff or any(
            t > self.fit_cutoff
            for t in (
                self.confidence.frozen_at,
                self.impact.frozen_at,
                self.model_lock.available_at,
                *(a.available_at for a in self.artifacts),
            )
        ):
            raise ValueError("fitted evidence unavailable by fit cutoff")
        if self.confidence.evidence_kind != self.evidence_kind or (
            self.impact.evidence_kind != self.evidence_kind
        ):
            raise ValueError("fit evidence kind mismatch")
        if self.confidence.score_definition.event_model_version != (
            self.model_identity.event_model_version
        ) or self.confidence.score_definition.entity_linker_version != (
            self.model_identity.entity_linker_version
        ):
            raise ValueError("Confidence/model identity mismatch")
        if (self.policy is None) != (self.policy_absence_reason is not None):
            raise ValueError("unselected policy needs an explicit reason")
        if self.policy is not None:
            if self.evidence_kind == "synthetic":
                raise ValueError("synthetic fit cannot carry empirical policy selection")
            p = self.policy
            if (p.candidate_hash, p.dataset_hash, p.training_hash, p.stable_fit_hash) != (
                content_hash(self.candidate),
                self.dataset_hash,
                self.training_hash,
                self.stable_fit_hash,
            ):
                raise ValueError("policy selection evidence authenticates a different fit")
            if p.template.frozen_at > self.fit_cutoff:
                raise ValueError("policy selection unavailable by fit cutoff")
        if self.manifest_hash != content_hash(self.model_dump(exclude={"manifest_hash"})):
            raise ValueError("fitted manifest content hash mismatch")
        return self

    def verify_artifacts(self) -> None:
        self.model_lock.verify(self.fit_cutoff)
        for artifact in self.artifacts:
            artifact.verify(self.fit_cutoff)


class ScenarioEvidence(Record):
    scenario: StressScenario
    joint_samples: Annotated[tuple[tuple[FactorShock, ...], ...], Field(min_length=1)]
    fit_hash: Sha256Hex
    calibration_hash: Sha256Hex
    available_at: AwareDatetime
    source_terms: NonemptyTerms
    support_reference: NonEmptyString
    support_hash: Sha256Hex


class RiskEnginePort(Protocol):
    def analyze(self, source_items: Sequence[SourceItem], as_of: datetime) -> list[RiskSignal]: ...


class FittedRuntime(Protocol):
    @property
    def manifest(self) -> FittedManifest: ...
    @property
    def risk_engine(self) -> RiskEnginePort: ...
    def scenarios(
        self, signal: RiskSignal, case: ReplayInput, *, as_of: datetime
    ) -> ScenarioEvidence: ...


class CandidateFitter(Protocol):
    def fit(
        self, candidate: CandidateSpec, training: TrainingPartition, *, as_of: datetime
    ) -> FittedRuntime: ...
    def restore(self, locked: FittedManifest) -> FittedRuntime: ...


def _group(case: ReplayInput) -> FoldGroup:
    return FoldGroup(
        cluster_id=case.cluster.cluster_id,
        event_time=case.cluster.event_time,
        source_item_ids=case.cluster.source_item_ids,
    )


class UndefinedMetric(Record):
    metric: NonEmptyString
    reason: NonEmptyString
    applicable_count: Annotated[int, Field(strict=True, ge=0)]


class MetricEvidence(Record):
    classification: m.ClassificationMetrics
    entity_link_accuracy: float
    confidence: m.ConfidenceMetrics | None
    severity: m.SeverityMetrics | None
    interval: m.IntervalMetrics | None
    scenario: m.ScenarioMetrics | None
    valuation: m.ValuationMetrics | None
    alerts: m.AlertMetrics | None
    undefined: tuple[UndefinedMetric, ...]
    scalar_projection: dict[NonEmptyString, float]
    monetary_projection_units: tuple[tuple[NonEmptyString, NonEmptyString], ...]


class PolicyBindingAudit(Record):
    rule: Literal["per-signal-composite-binding-v1"]
    template_hash: Sha256Hex | None
    absence_reason: NonEmptyString | None
    stable_fit_hash: Sha256Hex
    composite_hash: Sha256Hex
    composite: NonEmptyString
    concrete_policy: PolicyConfig


class CaseEvaluation(Record):
    case_id: NonEmptyString
    cluster_id: NonEmptyString
    member_source_ids: tuple[NonEmptyString, ...]
    representative_source_id: NonEmptyString
    as_of: AwareDatetime
    market_snapshot_id: NonEmptyString
    market_hash: Sha256Hex
    fit_hash: Sha256Hex
    impact_fit: ImpactFitIdentity
    portfolio_id: NonEmptyString
    portfolio_gross_values: tuple[tuple[NonEmptyString, Decimal], ...]
    returned_signal_ids: tuple[NonEmptyString, ...]
    chosen_signal_id: NonEmptyString | None
    abstention_reason: NonEmptyString | None
    chosen_signal: RiskSignal | None
    scenario: ScenarioEvidence | None
    stress: StressResult | None
    policy_binding: PolicyBindingAudit | None
    decision: TriggerDecision | None
    observed: ObservedOutcome

    @model_validator(mode="after")
    def consistent_reduction(self) -> CaseEvaluation:
        if len(set(self.returned_signal_ids)) != len(self.returned_signal_ids):
            raise ValueError("duplicate returned signal identity")
        if self.chosen_signal is None:
            if (
                self.returned_signal_ids
                or self.chosen_signal_id is not None
                or (self.abstention_reason is None)
                or any(
                    v is not None
                    for v in (
                        self.scenario,
                        self.stress,
                        self.policy_binding,
                        self.decision,
                    )
                )
            ):
                raise ValueError("abstention evidence mismatch")
        elif (
            self.chosen_signal_id != self.chosen_signal.signal_id
            or self.chosen_signal_id not in self.returned_signal_ids
            or self.abstention_reason is not None
            or any(
                v is None for v in (self.scenario, self.stress, self.policy_binding, self.decision)
            )
        ):
            raise ValueError("chosen signal audit mismatch")
        if self.chosen_signal is not None:
            signal = self.chosen_signal
            assert self.decision is not None and self.scenario is not None
            assert self.stress is not None and self.policy_binding is not None
            materiality = PortfolioMateriality(
                absolute_loss=max(Decimal(0), self.stress.absolute_loss),
                percentage_loss=max(0, self.stress.percentage_loss),
                currency=self.stress.valuation_currency,
            )
            if self.decision.portfolio_materiality != materiality:
                raise ValueError("decision materiality contradicts retained StressResult")
            if self.decision.manual_override is not None:
                raise ValueError("backtest replay cannot contain a manual override")
            binding = self.policy_binding
            if (
                self.decision.risk_signal != signal
                or self.decision.as_of != self.as_of
                or (self.decision.policy_config != binding.concrete_policy)
                or signal.source_item_id != self.representative_source_id
                or (self.scenario.scenario.risk_signal_id != signal.signal_id)
                or self.stress.scenario_id != self.scenario.scenario.scenario_id
                or (self.stress.portfolio_id != self.portfolio_id)
                or self.scenario.fit_hash != self.fit_hash
                or (self.scenario.calibration_hash != self.impact_fit.calibration_hash)
                or binding.composite != signal.versions.calibration_version
                or (
                    binding.composite_hash != hashlib.sha256(binding.composite.encode()).hexdigest()
                )
            ):
                raise ValueError("per-signal audit records contradict production inputs")
        if self.observed.case_id != self.case_id:
            raise ValueError("case outcome identity mismatch")
        return self


class ScoredPartition(Record):
    partition_id: NonEmptyString
    fit: FittedManifest
    cases: tuple[CaseEvaluation, ...]
    metrics: MetricEvidence


class InnerCandidateAudit(Record):
    candidate: CandidateSpec
    validations: tuple[ScoredPartition, ...]
    development_result: m.CandidateDevelopmentResult


class OuterEvaluation(Record):
    fold: OuterFold
    inner_candidates: tuple[InnerCandidateAudit, ...]
    selection: m.SelectionResult
    test: ScoredPartition


class SensitivityObservation(Record):
    candidate_id: NonEmptyString
    parameter: NonEmptyString
    value: ConfigurationScalar
    other_parameters: dict[NonEmptyString, ConfigurationScalar]
    metrics: dict[NonEmptyString, float]
    scope: Literal["retained-inner-development-descriptive-v1"]


class DevelopmentLock(Record):
    schema_version: Literal["backtest-development-lock-v1"]
    lock_hash: Sha256Hex
    evidence_kind: EvidenceKind
    dataset_hash: Sha256Hex
    configuration: DevelopmentConfiguration
    configuration_hash: Sha256Hex
    model_hash: Sha256Hex
    final_groups: Annotated[tuple[FoldGroup, ...], Field(min_length=1)]
    final_period: EvaluationPeriod
    first_final_as_of: AwareDatetime
    omitted_final_boundary_groups: tuple[FoldGroup, ...]
    fitted: FittedManifest
    selection: m.SelectionResult
    development_report_hash: Sha256Hex
    candidate_configurations: tuple[CandidateConfiguration, ...]
    sensitivity_results: tuple[SensitivityResult, ...]
    sensitivity_observations: tuple[SensitivityObservation, ...]

    @model_validator(mode="after")
    def lock_claims_match(self) -> DevelopmentLock:
        if self.configuration_hash != content_hash(self.configuration):
            raise ValueError("lock configuration content hash mismatch")
        if (
            self.fitted.dataset_hash,
            self.fitted.configuration_hash,
            self.fitted.evidence_kind,
            self.fitted.model_lock.sha256,
        ) != (
            self.dataset_hash,
            self.configuration_hash,
            self.evidence_kind,
            self.model_hash,
        ):
            raise ValueError("lock fitted identities mismatch")
        if self.fitted.candidate.candidate_id != self.selection.selected_candidate_id:
            raise ValueError("lock selected convention mismatch")
        declared = {c.candidate_id: c for c in self.configuration.candidates}
        if declared.get(self.fitted.candidate.candidate_id) != self.fitted.candidate:
            raise ValueError("lock selected candidate was not declared")
        for candidate in self.selection.candidates:
            if (
                candidate.candidate_id not in declared
                or (candidate.configuration_version != self.configuration.version)
                or candidate.complexity != declared[candidate.candidate_id].complexity
            ):
                raise ValueError("lock selection evidence contradicts declarations")
        if {c.candidate_id for c in self.selection.candidates} != set(declared):
            raise ValueError("lock lacks all candidate evidence")
        selected = tuple(c.candidate_id for c in self.candidate_configurations if c.selected)
        if selected != (self.fitted.candidate.candidate_id,) or {
            c.candidate_id for c in self.candidate_configurations
        } != set(declared):
            raise ValueError("lock report candidate selection mismatch")
        for c in self.candidate_configurations:
            if c.parameters != declared[c.candidate_id].parameters:
                raise ValueError("lock report parameters mismatch")
        if self.fitted.fit_cutoff != self.configuration.final_fit_cutoff or (
            self.fitted.fit_cutoff > self.first_final_as_of
        ):
            raise ValueError("lock final fit cutoff mismatch")
        final_sources = {s for g in self.final_groups for s in g.source_item_ids}
        if any(s in final_sources for g in self.fitted.training_groups for s in g.source_item_ids):
            raise ValueError("lock final membership leaked into fit")
        if any(
            g.event_time + self.configuration.split.embargo > self.final_groups[0].event_time
            for g in self.fitted.training_groups
        ):
            raise ValueError("lock final embargo violated")
        if self.lock_hash != content_hash(self.model_dump(exclude={"lock_hash"})):
            raise ValueError("development lock content hash mismatch")
        return self


class FinalConfiguration(Record):
    schema_version: Literal["backtest-final-v1"]
    locked: DevelopmentLock


class BacktestAudit(Record):
    schema_version: Literal["backtest-audit-v1"]
    mode: Literal["development", "final"]
    evidence_kind: EvidenceKind
    dataset_hash: Sha256Hex
    configuration_hash: Sha256Hex
    model_hash: Sha256Hex
    report_hash: Sha256Hex
    folds: tuple[OuterFold, ...]
    outer_results: tuple[OuterEvaluation, ...]
    final_result: ScoredPartition | None
    selected_final_manifest: FittedManifest
    locked: DevelopmentLock
    metrics: MetricEvidence
    sensitivity_observations: tuple[SensitivityObservation, ...]

    @model_validator(mode="after")
    def coherent_claims(self) -> BacktestAudit:
        if (
            self.dataset_hash,
            self.configuration_hash,
            self.model_hash,
            self.evidence_kind,
            self.selected_final_manifest,
        ) != (
            self.locked.dataset_hash,
            self.locked.configuration_hash,
            self.locked.model_hash,
            self.locked.evidence_kind,
            self.locked.fitted,
        ):
            raise ValueError("audit lock identity mismatch")
        if self.mode == "development":
            if (
                not self.outer_results
                or self.final_result is not None
                or self.folds != tuple(o.fold for o in self.outer_results)
                or self.locked.selection != self.outer_results[-1].selection
            ):
                raise ValueError("audit chronological selection mismatch")
            if self.report_hash != self.locked.development_report_hash:
                raise ValueError("audit development report hash mismatch")
        elif self.outer_results or self.folds or self.final_result is None:
            raise ValueError("final audit must contain only final observations")
        config = self.locked.configuration
        snapshot_id = self.locked.selection.candidates[0].snapshot_id
        all_cases: tuple[CaseEvaluation, ...]
        if self.mode == "development":
            for outer in self.outer_results:
                if outer.selection.candidates != tuple(
                    c.development_result for c in outer.inner_candidates
                ):
                    raise ValueError("audit selection is not bound to retained candidate evidence")
                if outer.test.fit.candidate.candidate_id != outer.selection.selected_candidate_id:
                    raise ValueError("outer fit differs from inner-selected convention")
                for candidate in outer.inner_candidates:
                    if candidate.candidate not in config.candidates or len(
                        candidate.validations
                    ) != len(outer.fold.inner_folds):
                        raise ValueError("audit declared candidate/fold membership mismatch")
                    for scored, inner, objective in zip(
                        candidate.validations,
                        outer.fold.inner_folds,
                        candidate.development_result.fold_results,
                        strict=True,
                    ):
                        _verify_scored(
                            scored,
                            inner.validation,
                            inner.train,
                            config,
                            self.dataset_hash,
                            snapshot_id,
                        )
                        if (
                            objective.fold != inner
                            or objective.score != scored.metrics.classification.macro_f1
                        ):
                            raise ValueError("audit objective contradicts metric evidence")
                _verify_scored(
                    outer.test,
                    outer.fold.test,
                    outer.fold.train,
                    config,
                    self.dataset_hash,
                    snapshot_id,
                )
            all_cases = tuple(c for outer in self.outer_results for c in outer.test.cases)
            candidates, sensitivity, observations = BacktestModule._development_projections(
                self.outer_results, config, self.locked.selection.selected_candidate_id
            )
            if (candidates, sensitivity, observations) != (
                self.locked.candidate_configurations,
                self.locked.sensitivity_results,
                self.sensitivity_observations,
            ):
                raise ValueError("development projection/sensitivity contradicts inner evidence")
        else:
            assert self.final_result is not None
            _verify_scored(
                self.final_result,
                self.locked.final_groups,
                self.locked.fitted.training_groups,
                config,
                self.dataset_hash,
                snapshot_id,
            )
            if self.final_result.fit != self.locked.fitted:
                raise ValueError("final audit fit differs from locked fit")
            all_cases = self.final_result.cases
        if self.metrics != _metrics(all_cases, config, snapshot_id):
            raise ValueError("audit metric evidence contradicts retained cases")
        if self.sensitivity_observations != self.locked.sensitivity_observations:
            raise ValueError("audit sensitivity provenance mismatch")
        return self


def _verify_scored(
    scored: ScoredPartition,
    expected_groups: tuple[FoldGroup, ...],
    training: tuple[FoldGroup, ...],
    config: DevelopmentConfiguration,
    dataset_hash: str,
    snapshot_id: str,
) -> None:
    if (
        scored.fit.training_groups != training
        or scored.fit.configuration_hash != content_hash(config)
        or (scored.fit.dataset_hash != dataset_hash)
        or tuple((c.cluster_id, c.member_source_ids) for c in scored.cases)
        != tuple((g.cluster_id, g.source_item_ids) for g in expected_groups)
    ):
        raise ValueError("audit scored/training membership mismatch")
    for case in scored.cases:
        if case.fit_hash != scored.fit.manifest_hash or case.impact_fit != scored.fit.impact:
            raise ValueError("audit case fitted identity mismatch")
        if case.decision is not None:
            assert case.chosen_signal is not None and case.policy_binding is not None
            if case.policy_binding.stable_fit_hash != scored.fit.stable_fit_hash:
                raise ValueError("audit policy binding differs from stable fit identity")
    if scored.metrics != _metrics(scored.cases, config, snapshot_id):
        raise ValueError("audit metric evidence contradicts scored cases")


def _as_float(value: float | Decimal) -> float:
    projected = float(value)
    if not math.isfinite(projected):
        raise ValueError("Decimal scalar projection overflow")
    return projected


def _signature(shocks: Sequence[FactorShock]) -> tuple[tuple[object, ...], ...]:
    return tuple((s.factor_id, s.shock_type, s.unit, s.horizon_days) for s in shocks)


def _metrics(
    rows: tuple[CaseEvaluation, ...], config: DevelopmentConfiguration, snapshot_id: str
) -> MetricEvidence:
    classification = m.classification_metrics(
        [r.observed.actual_event_class for r in rows],
        [r.chosen_signal.event_class if r.chosen_signal else None for r in rows],
    )
    entity_accuracy = m.entity_link_accuracy(
        [r.observed.actual_entity_id for r in rows],
        [r.chosen_signal.entity.entity_id if r.chosen_signal else None for r in rows],
    )
    projected = {
        "classification.macro_f1": classification.macro_f1,
        "classification.non_abstention_coverage": classification.non_abstention_coverage,
        "entity_link_accuracy": entity_accuracy,
    }
    undefined: list[UndefinedMetric] = []
    units: list[tuple[str, str]] = []

    def missing(metric: str, reason: str, count: int = 0) -> None:
        undefined.append(UndefinedMetric(metric=metric, reason=reason, applicable_count=count))

    def scalar(
        name: str, value: float | Decimal | None, reason: str, count: int, unit: str | None = None
    ) -> None:
        if value is None:
            missing(name, reason, count)
        else:
            projected[name] = _as_float(value)
            if isinstance(value, Decimal):
                assert unit is not None
                units.append((name, unit))

    predicted = tuple(r for r in rows if r.chosen_signal is not None)
    confidence = None
    if predicted:
        correct, probabilities = [], []
        for r in predicted:
            assert r.chosen_signal is not None
            signal = r.chosen_signal
            correct.append(
                signal.event_class == r.observed.actual_event_class
                and signal.entity.entity_id == r.observed.actual_entity_id
            )
            probabilities.append(signal.confidence)
        if any(
            (y and p == 0) or (not y and p == 1)
            for y, p in zip(correct, probabilities, strict=True)
        ):
            projected["confidence.brier_score"] = math.fsum(
                (p - y) ** 2 / len(correct) for p, y in zip(probabilities, correct, strict=True)
            )
            missing(
                "confidence.log_loss",
                "wrong certain probabilities have undefined log loss",
                len(predicted),
            )
            missing(
                "confidence.reliability",
                "confidence family has undefined finite log loss",
                len(predicted),
            )
        else:
            confidence = m.confidence_metrics(correct, probabilities, config.reliability_bin_edges)
            projected.update(
                {
                    "confidence.brier_score": confidence.brier_score,
                    "confidence.log_loss": confidence.log_loss,
                }
            )
    else:
        missing("confidence", "empty post-abstention population")

    severity_rows: list[tuple[CaseEvaluation, ReferenceLoss]] = []
    for row in predicted:
        ref = next(
            (
                r
                for r in row.observed.reference_losses
                if r.basket_hash == row.impact_fit.reference_basket_hash
            ),
            None,
        )
        if ref is None:
            missing(
                f"case.{row.case_id}.severity",
                row.observed.reference_loss_absence_reason
                or "no realized loss for fitted Reference Basket",
            )
        else:
            if ref.currency != row.impact_fit.currency:
                raise ValueError("Reference Basket loss currency mismatch")
            severity_rows.append((row, ref))
    severity = interval = None
    if severity_rows:
        if len({r.currency for _, r in severity_rows}) != 1:
            missing("severity", "heterogeneous Reference Basket currencies", len(severity_rows))
            missing("interval", "heterogeneous Reference Basket currencies", len(severity_rows))
        else:
            scores, deciles, losses, lowers, uppers = [], [], [], [], []
            for row, ref in severity_rows:
                assert row.chosen_signal is not None
                if row.chosen_signal.impact.loss_currency != ref.currency:
                    raise ValueError("predicted/realized Reference Basket currency mismatch")
                scores.append(row.chosen_signal.impact.impact_score)
                deciles.append(1 + sum(ref.loss > c for c in row.impact_fit.cutpoints))
                losses.append(_as_float(ref.loss))
                lowers.append(_as_float(row.chosen_signal.impact.loss_lower_bound))
                uppers.append(_as_float(row.chosen_signal.impact.loss_upper_bound))
            severity = m.severity_metrics(scores, deciles, losses, severity_rows[0][1].currency)
            interval = m.interval_metrics(
                lowers, uppers, losses, severity.loss_unit, config.nominal_interval_coverage
            )
            scalar(
                "severity.rank_correlation",
                severity.rank_correlation,
                "constant ranks or insufficient population",
                len(severity_rows),
            )
            scalar(
                "severity.bucket_monotonicity",
                severity.bucket_monotonicity,
                "fewer than two occupied Impact Score buckets",
                len(severity_rows),
            )
            projected.update(
                {
                    "severity.ordinal_mae": severity.ordinal_mae,
                    "interval.coverage": interval.coverage,
                    "interval.mean_width": interval.mean_width,
                }
            )
    else:
        missing("severity", "no matched Reference Basket outcome evidence")
        missing("interval", "no matched Reference Basket outcome evidence")

    scenarios, valuations, alerts = [], [], []
    for row in predicted:
        assert row.chosen_signal is not None and row.scenario is not None and row.stress is not None
        observed = row.observed
        if observed.realized_shocks is not None:
            scenarios.append(
                m.ScenarioObservation(
                    case_id=row.case_id,
                    occurred_at=row.as_of,
                    snapshot_id=snapshot_id,
                    configuration_version=config.version,
                    realized=tuple(sorted(observed.realized_shocks, key=lambda s: s.factor_id)),
                    joint_samples=row.scenario.joint_samples,
                )
            )
        else:
            missing(f"case.{row.case_id}.scenario", observed.scenario_absence_reason or "missing")
        valuation = observed.valuation
        valuation_reason = None
        if valuation is not None:
            if valuation.baseline_market_hash != row.market_hash:
                raise ValueError("valuation baseline market hash mismatch")
            if valuation.currency != row.stress.valuation_currency or (
                valuation.horizon_days != row.impact_fit.horizon_days
            ):
                raise ValueError("valuation currency/horizon mismatch")
            gross = dict(valuation.gross_values)
            all_ids = {p for p in gross}
            supported_ids = all_ids - set(row.stress.unsupported_position_ids)
            # Explicit outcome scope must equal the production engine's supported subset.
            if gross != dict(row.portfolio_gross_values):
                valuation_reason = (
                    "gross support evidence does not match complete portfolio position values"
                )
            elif tuple(sorted(supported_ids)) != valuation.position_ids:
                valuation_reason = "comparison scope differs from supported subset"
            elif not gross or sum(gross.values()) <= 0:
                valuation_reason = "positive total gross support evidence absent"
            else:
                valuations.append(
                    m.ValuationObservation(
                        case_id=row.case_id,
                        occurred_at=row.as_of,
                        comparison_scope=valuation.comparison_scope,
                        snapshot_id=snapshot_id,
                        configuration_version=config.version,
                        predicted_pnl=row.stress.stressed_value - row.stress.base_value,
                        realized_pnl=valuation.pnl,
                        currency=valuation.currency,
                        horizon_days=valuation.horizon_days,
                        supported_gross_value=sum((gross[k] for k in supported_ids), Decimal(0)),
                        total_gross_value=sum(gross.values(), Decimal(0)),
                    )
                )
        else:
            valuation_reason = observed.valuation_absence_reason or "missing"
        if valuation_reason is not None:
            missing(f"case.{row.case_id}.valuation", valuation_reason)
        if row.chosen_signal.action_priority is None:
            missing(f"case.{row.case_id}.alerts", "Action Priority unavailable; no fabricated zero")
        elif observed.material_event is None:
            missing(
                f"case.{row.case_id}.alerts",
                observed.material_event_absence_reason or "material-event outcome unavailable",
            )
        elif valuation_reason is not None:
            missing(
                f"case.{row.case_id}.alerts",
                f"realized portfolio loss unavailable: {valuation_reason}",
            )
        else:
            assert row.decision is not None and valuation is not None
            alerts.append(
                m.AlertObservation(
                    case_id=row.case_id,
                    occurred_at=row.as_of,
                    is_alert=row.decision.automatic_trigger,
                    material_event=observed.material_event,
                    action_priority=row.chosen_signal.action_priority,
                    realized_loss=max(Decimal(0), -valuation.pnl),
                )
            )
    scenario_metric = valuation_metric = alert_metric = None
    if scenarios:
        if len({_signature(s.realized) for s in scenarios}) == 1:
            scenario_metric = m.scenario_metrics(scenarios)
            for factor in scenario_metric.factors:
                projected[f"scenario.{factor.factor_id}.direction_accuracy"] = (
                    factor.direction_accuracy
                )
                projected[f"scenario.{factor.factor_id}.crps"] = factor.crps
        else:
            missing("scenario", "heterogeneous factor signatures/horizons", len(scenarios))
    else:
        missing("scenario", "empty applicable scenario population")
    if valuations:
        if len({(v.currency, v.comparison_scope, v.horizon_days) for v in valuations}) == 1:
            valuation_metric = m.valuation_metrics(valuations)
            scalar(
                "valuation.pnl_mae",
                valuation_metric.pnl_mae,
                "",
                len(valuations),
                valuation_metric.currency,
            )
            scalar(
                "valuation.supported_value_share",
                valuation_metric.supported_value_share,
                "",
                len(valuations),
                "fraction",
            )
        else:
            missing("valuation", "heterogeneous comparison scope/currency/horizon", len(valuations))
    else:
        missing("valuation", "empty applicable valuation population")
    if alerts:
        currencies = {
            r.observed.valuation.currency
            for r in predicted
            if r.observed.valuation is not None and r.case_id in {a.case_id for a in alerts}
        }
        if len(currencies) != 1:
            missing("alerts", "heterogeneous realized loss currencies", len(alerts))
        else:
            alert_metric = m.alert_metrics(
                alerts,
                m.AlertWindow(
                    start=config.alert_window.start,
                    end=config.alert_window.end,
                    currency=next(iter(currencies)),
                    budget=config.alert_budget,
                    snapshot_id=snapshot_id,
                    configuration_version=config.version,
                ),
            )
            for name, value in (
                ("precision", alert_metric.precision),
                ("recall", alert_metric.recall),
                ("alerts_per_day", alert_metric.alerts_per_day),
                ("loss_captured_at_budget", alert_metric.loss_captured_at_budget),
            ):
                scalar(
                    f"alerts.{name}",
                    value,
                    "undefined alert ratio denominator",
                    len(alerts),
                    "fraction" if name == "loss_captured_at_budget" else None,
                )
    else:
        missing("alerts", "no applicable Action Priority and material-event outcome evidence")
    return MetricEvidence(
        classification=classification,
        entity_link_accuracy=entity_accuracy,
        confidence=confidence,
        severity=severity,
        interval=interval,
        scenario=scenario_metric,
        valuation=valuation_metric,
        alerts=alert_metric,
        undefined=tuple(undefined),
        scalar_projection=projected,
        monetary_projection_units=tuple(units),
    )


def decode_impact_audit(composite: str) -> ImpactAudit:
    payload = json.loads(composite)
    if not isinstance(payload, dict):
        raise ValueError("Impact audit must be a JSON record")
    digest = payload.pop("audit_sha256", None)
    # Task 22 uses its existing ASCII JSON hash domain for this nested audit.
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != expected:
        raise ValueError("Impact audit content hash mismatch")
    return ImpactAudit.model_validate(payload)


def _verify_calibration(signal: RiskSignal, manifest: FittedManifest, as_of: datetime) -> None:
    calibration = CalibrationManifest.model_validate_json(signal.versions.calibration_version)
    if (
        calibration.confidence_calibration != manifest.confidence
        or (calibration.confidence.probability != signal.confidence)
        or calibration.confidence.fitted_artifact_hash != manifest.confidence.fitted_artifact_hash
        or (calibration.confidence.evidence_hash != manifest.confidence.evidence_hash)
        or calibration.confidence.evidence_kind != manifest.evidence_kind
        or (calibration.confidence.calibration_version != manifest.confidence.calibration_version)
        or calibration.confidence.score_definition != manifest.confidence.score_definition
        or (calibration.confidence.transform_version != manifest.confidence.transform_version)
        or calibration.confidence.target != manifest.confidence.target
        or (signal.confidence_target != manifest.confidence.target)
    ):
        raise ValueError("Risk Signal Confidence fit identity mismatch")
    if (
        calibration.impact.calibration_version != signal.impact.calibration_version
        or (calibration.impact.reference_basket_version != manifest.impact.reference_basket_version)
        or signal.impact.reference_basket_version != manifest.impact.reference_basket_version
    ):
        raise ValueError("Risk Signal Impact/Reference Basket identity mismatch")
    impact = manifest.impact
    if manifest.evidence_kind == "synthetic":
        audit = SyntheticImpactAudit.model_validate_json(signal.impact.calibration_version)
        if audit.fit != impact:
            raise ValueError("synthetic Impact training identity mismatch")
    else:
        empirical = decode_impact_audit(signal.impact.calibration_version)
        if (
            (
                empirical.calibration_sha256,
                empirical.calibration.reference_basket_sha256,
                empirical.matching_version,
                empirical.cutpoints,
                tuple(r.event_id for r in empirical.training_losses),
            )
            != (
                impact.calibration_hash,
                impact.reference_basket_hash,
                impact.matching_version,
                impact.cutpoints,
                impact.training_event_ids,
            )
            or any(r.available_at > manifest.fit_cutoff for r in empirical.training_losses)
            or (empirical.calibration.calibrated_at > as_of)
        ):
            raise ValueError("empirical Impact calibration/training identity mismatch")


def bind_trigger_policy(
    signal: RiskSignal, manifest: FittedManifest
) -> tuple[PolicyConfig, PolicyBindingAudit]:
    """Bind a frozen verified template to Task 24's exact per-signal composite."""
    signal, manifest = _copy(signal), _copy(manifest)
    _verify_calibration(signal, manifest, manifest.fit_cutoff)
    if manifest.policy is None:
        policy = PolicyConfig(selected=None)
        template_hash = None
    else:
        payload = manifest.policy.template.model_dump()
        payload["calibration_version"] = signal.versions.calibration_version
        policy = PolicyConfig(selected=SelectedTriggerPolicy.model_validate(payload))
        template_hash = content_hash(manifest.policy)
    binding = PolicyBindingAudit(
        rule="per-signal-composite-binding-v1",
        template_hash=template_hash,
        absence_reason=manifest.policy_absence_reason,
        stable_fit_hash=manifest.stable_fit_hash,
        composite_hash=hashlib.sha256(signal.versions.calibration_version.encode()).hexdigest(),
        composite=signal.versions.calibration_version,
        concrete_policy=policy,
    )
    return policy, binding


class BacktestModule:
    def __init__(self, fitter: CandidateFitter, stress_engine: StressEngine) -> None:
        self._fitter = fitter
        self._stress = stress_engine
        self._audit: BacktestAudit | None = None

    @property
    def audit(self) -> BacktestAudit:
        if self._audit is None:
            raise ValueError("no successful evaluation audit is available")
        return _copy(self._audit)

    def evaluate(
        self,
        historical_snapshots: BacktestDataset,
        frozen_configuration: DevelopmentConfiguration | FinalConfiguration,
        evaluation_period: EvaluationPeriod,
    ) -> BacktestReport:
        self._audit = None
        with localcontext(_decimal_context()):
            return self._evaluate(historical_snapshots, frozen_configuration, evaluation_period)

    def _evaluate(
        self,
        historical_snapshots: BacktestDataset,
        frozen_configuration: DevelopmentConfiguration | FinalConfiguration,
        evaluation_period: EvaluationPeriod,
    ) -> BacktestReport:
        dataset, config, period = (
            _copy(historical_snapshots),
            _copy(frozen_configuration),
            _copy(evaluation_period),
        )
        if isinstance(config, FinalConfiguration):
            report, audit = self._final(dataset, config.locked, period)
        else:
            report, audit = self._development(dataset, config, period)
        if content_hash(report) != audit.report_hash:
            raise ValueError("report/audit content hash mismatch")
        self._audit = _copy(audit)
        return _copy(report)

    def _training(
        self, dataset: BacktestDataset, groups: tuple[FoldGroup, ...], cutoff: datetime
    ) -> TrainingPartition:
        by_cluster = {c.cluster.cluster_id: c for c in dataset.cases}
        outcomes = {o.case_id: o for o in dataset.outcomes}
        cases = tuple(by_cluster[g.cluster_id] for g in groups)
        return TrainingPartition(
            groups=groups,
            cases=cases,
            outcomes=tuple(outcomes[c.case_id] for c in cases),
            cutoff=cutoff,
            snapshot_id=dataset.snapshot_id,
            dataset_hash=dataset.content_hash,
            evidence_kind=dataset.evidence_kind,
            source_terms=dataset.source_terms,
            membership_hash=content_hash(groups),
        )

    def _fit(
        self,
        candidate: CandidateSpec,
        training: TrainingPartition,
        config: DevelopmentConfiguration,
    ) -> FittedRuntime:
        runtime = self._fitter.fit(_copy(candidate), _copy(training), as_of=training.cutoff)
        manifest = _copy(runtime.manifest)
        if (
            manifest.candidate,
            manifest.training_groups,
            manifest.training_hash,
            manifest.dataset_hash,
            manifest.configuration_hash,
            manifest.fit_cutoff,
            manifest.maximum_evidence_available_at,
            manifest.evidence_kind,
        ) != (
            candidate,
            training.groups,
            training.membership_hash,
            training.dataset_hash,
            content_hash(config),
            training.cutoff,
            max(training.availability),
            training.evidence_kind,
        ):
            raise ValueError(
                "fitter returned mismatched candidate/training/configuration identities"
            )
        manifest.verify_artifacts()
        return runtime

    def _verify_signal(
        self, signal: RiskSignal, case: ReplayInput, manifest: FittedManifest
    ) -> None:
        item = case.cluster.items[0]
        if signal.source_item_id != item.source_item_id:
            raise ValueError("Risk Signal source membership mismatch")
        if manifest.fit_cutoff > case.as_of:
            raise ValueError("fitted calibration unavailable by historical as_of")
        models = ModelManifest.model_validate_json(signal.versions.model_version)
        if models.models != manifest.model_identity:
            raise ValueError("Risk Signal model identity mismatch")
        snapshot = SnapshotManifest.model_validate_json(signal.versions.snapshot_version)
        if len(snapshot.source_items) != 1 or (
            snapshot.source_items[0].model_dump() != item.model_dump(exclude={"text"})
        ):
            raise ValueError("Risk Signal Source Item snapshot provenance mismatch")
        _verify_calibration(signal, manifest, case.as_of)

    def _replay(
        self,
        dataset: BacktestDataset,
        config: DevelopmentConfiguration,
        groups: tuple[FoldGroup, ...],
        runtime: FittedRuntime,
        period: EvaluationPeriod,
        partition_id: str,
    ) -> ScoredPartition:
        manifest = _copy(runtime.manifest)
        by_cluster = {c.cluster.cluster_id: c for c in dataset.cases}
        outcomes = {o.case_id: o for o in dataset.outcomes}
        rows = []
        for group in groups:
            case = _copy(by_cluster[group.cluster_id])
            if not period.start <= case.cluster.event_time < period.end:
                raise ValueError("scored case lies outside evaluation period")
            observed = outcomes[case.case_id]
            if (
                max(
                    observed.label_available_at,
                    observed.outcome_available_at,
                    *(r.available_at for r in observed.reference_losses),
                )
                > period.reporting_cutoff
            ):
                raise ValueError("scored outcomes unavailable by reporting cutoff")
            signals = tuple(
                _copy(s) for s in runtime.risk_engine.analyze((case.cluster.items[0],), case.as_of)
            )
            if len({s.signal_id for s in signals}) != len(signals):
                raise ValueError("duplicate Risk Signal identity")
            for signal in signals:
                self._verify_signal(signal, case, manifest)
            chosen = min(signals, key=lambda s: (-s.confidence, s.signal_id)) if signals else None
            scenario = stress = binding = decision = None
            if chosen is not None:
                scenario = _copy(runtime.scenarios(chosen, _copy(case), as_of=case.as_of))
                shocks = tuple(sorted(scenario.scenario.shocks, key=lambda s: s.factor_id))
                if (
                    scenario.scenario.risk_signal_id != chosen.signal_id
                    or (scenario.fit_hash != manifest.manifest_hash)
                    or scenario.calibration_hash != manifest.impact.calibration_hash
                    or (scenario.available_at > case.as_of)
                    or scenario.scenario.calibration_version != chosen.impact.calibration_version
                    or (scenario.scenario.analyst_override)
                    or any(
                        _signature(sample) != _signature(shocks)
                        for sample in scenario.joint_samples
                    )
                    or (len({s.horizon_days for s in shocks}) != 1)
                    or shocks[0].horizon_days != manifest.impact.horizon_days
                ):
                    raise ValueError(
                        "ScenarioEvidence fit/calibration/factor/horizon/as_of mismatch"
                    )
                stress = self._stress.run(case.portfolio, case.market, scenario.scenario)
                materiality = PortfolioMateriality(
                    absolute_loss=max(Decimal(0), stress.absolute_loss),
                    percentage_loss=max(0, stress.percentage_loss),
                    currency=stress.valuation_currency,
                )
                if (
                    chosen.portfolio_materiality is not None
                    and chosen.portfolio_materiality != materiality
                ):
                    raise ValueError("Risk Signal materiality differs from production StressResult")
                policy, binding = bind_trigger_policy(chosen, manifest)
                decision = TriggerPolicy(policy, as_of=case.as_of).evaluate(chosen, materiality)
            rows.append(
                CaseEvaluation(
                    case_id=case.case_id,
                    cluster_id=group.cluster_id,
                    member_source_ids=group.source_item_ids,
                    representative_source_id=case.cluster.representative_source_item_id,
                    as_of=case.as_of,
                    market_snapshot_id=case.market.snapshot_id,
                    market_hash=content_hash(case.market),
                    fit_hash=manifest.manifest_hash,
                    impact_fit=manifest.impact,
                    portfolio_id=case.portfolio.portfolio_id,
                    portfolio_gross_values=tuple(
                        sorted((p.position_id, abs(p.notional)) for p in case.portfolio.positions)
                    ),
                    returned_signal_ids=tuple(s.signal_id for s in signals),
                    chosen_signal_id=chosen.signal_id if chosen else None,
                    abstention_reason=None
                    if chosen
                    else "production Risk Engine returned no signals",
                    chosen_signal=chosen,
                    scenario=scenario,
                    stress=stress,
                    policy_binding=binding,
                    decision=decision,
                    observed=observed,
                )
            )
        evaluated = tuple(rows)
        return ScoredPartition(
            partition_id=partition_id,
            fit=manifest,
            cases=evaluated,
            metrics=_metrics(evaluated, config, dataset.snapshot_id),
        )

    def _development(
        self, dataset: BacktestDataset, config: DevelopmentConfiguration, period: EvaluationPeriod
    ) -> tuple[BacktestReport, BacktestAudit]:
        if config.frozen_at > min(c.cluster.event_time for c in dataset.cases):
            raise ValueError("configuration must be frozen before historical evaluation schedule")
        folds = tuple(
            nested_chronological_splits(tuple(c.cluster for c in dataset.cases), config.split)
        )
        if any(len(f.inner_folds) < 2 for f in folds):
            raise ValueError("every outer fold requires at least two complete inner folds")
        by_cluster = {c.cluster.cluster_id: c for c in dataset.cases}
        groups = tuple(
            sorted((_group(c) for c in dataset.cases), key=lambda g: (g.event_time, g.cluster_id))
        )
        group_map = {g.cluster_id: g for g in groups}
        for fold in folds:
            partitions = fold.train + fold.embargo + fold.test + fold.final_holdout
            if any(group_map.get(g.cluster_id) != g for g in partitions):
                raise ValueError("split membership mismatch")
            if any(
                g.event_time + config.split.embargo > fold.test[0].event_time for g in fold.train
            ):
                raise ValueError("outer training embargo violated")
        final_groups = folds[0].final_holdout
        first_final = by_cluster[final_groups[0].cluster_id]
        if config.final_fit_cutoff > first_final.as_of:
            raise ValueError("final fit cutoff exceeds first final historical as_of")
        eligible = tuple(
            g
            for g in groups[: -config.split.final_holdout_groups]
            if g.event_time + config.split.embargo <= first_final.cluster.event_time
        )
        omitted = tuple(
            g for g in groups[: -config.split.final_holdout_groups] if g not in eligible
        )
        # Preflight all fitting partitions before calling any production inference.
        final_training = self._training(dataset, eligible, config.final_fit_cutoff)
        training_schedule = {}
        for fold in folds:
            cutoff = by_cluster[fold.test[0].cluster_id].as_of
            training_schedule[fold.fold_id] = self._training(dataset, fold.train, cutoff)
            for inner in fold.inner_folds:
                start = by_cluster[inner.validation[0].cluster_id].as_of
                if any(
                    g.event_time + config.split.embargo > inner.validation[0].event_time
                    for g in inner.train
                ):
                    raise ValueError("inner training embargo violated")
                training_schedule[inner.fold_id] = self._training(dataset, inner.train, start)
        outer_results = []
        for fold in folds:
            inner_candidates = []
            for candidate in sorted(config.candidates, key=lambda c: c.candidate_id):
                validations = []
                fold_results = []
                for inner in fold.inner_folds:
                    runtime = self._fit(candidate, training_schedule[inner.fold_id], config)
                    # UTC event-time successor defines an exclusive endpoint without
                    # assuming any delay between publication and historical replay.
                    inner_period = EvaluationPeriod(
                        start=inner.validation[0].event_time,
                        end=inner.validation[-1].event_time + timedelta(microseconds=1),
                        reporting_cutoff=period.reporting_cutoff,
                    )
                    scored = self._replay(
                        dataset, config, inner.validation, runtime, inner_period, inner.fold_id
                    )
                    validations.append(scored)
                    fold_results.append(
                        m.InnerDevelopmentResult(
                            fold=inner, score=scored.metrics.classification.macro_f1
                        )
                    )
                evidence = m.CandidateDevelopmentResult(
                    candidate_id=candidate.candidate_id,
                    configuration_version=config.version,
                    convention_version="one-standard-error-v1",
                    evaluation_scope="inner-development",
                    snapshot_id=dataset.snapshot_id,
                    split_version=config.split.version,
                    outer_fold_id=fold.fold_id,
                    objective_name=config.objective,
                    objective_unit="fraction",
                    direction=config.direction,
                    complexity_dimensions=candidate.complexity_dimensions,
                    complexity=candidate.complexity,
                    fold_results=tuple(fold_results),
                )
                inner_candidates.append(
                    InnerCandidateAudit(
                        candidate=candidate,
                        validations=tuple(validations),
                        development_result=evidence,
                    )
                )
            selection = m.select_within_one_standard_error(
                tuple(c.development_result for c in inner_candidates)
            )
            selected = next(
                c for c in config.candidates if c.candidate_id == selection.selected_candidate_id
            )
            runtime = self._fit(selected, training_schedule[fold.fold_id], config)
            test = self._replay(dataset, config, fold.test, runtime, period, fold.fold_id)
            outer_results.append(
                OuterEvaluation(
                    fold=fold,
                    inner_candidates=tuple(inner_candidates),
                    selection=selection,
                    test=test,
                )
            )
        final_selection = outer_results[-1].selection
        selected = next(
            c for c in config.candidates if c.candidate_id == final_selection.selected_candidate_id
        )
        fitted = _copy(self._fit(selected, final_training, config).manifest)
        outer_rows = tuple(c for o in outer_results for c in o.test.cases)
        metrics = _metrics(outer_rows, config, dataset.snapshot_id)
        candidates, sensitivity, observations = self._development_projections(
            tuple(outer_results), config, selected.candidate_id
        )
        report = self._report(
            dataset, config, period, metrics, outer_rows, candidates, sensitivity, "development"
        )
        lock_payload = dict(
            schema_version="backtest-development-lock-v1",
            evidence_kind=dataset.evidence_kind,
            dataset_hash=dataset.content_hash,
            configuration=config,
            configuration_hash=content_hash(config),
            model_hash=fitted.model_lock.sha256,
            final_groups=final_groups,
            first_final_as_of=first_final.as_of,
            final_period=EvaluationPeriod(
                start=first_final.cluster.event_time,
                end=final_groups[-1].event_time + timedelta(microseconds=1),
                reporting_cutoff=period.reporting_cutoff,
            ),
            omitted_final_boundary_groups=omitted,
            fitted=fitted,
            selection=final_selection,
            development_report_hash=content_hash(report),
            candidate_configurations=candidates,
            sensitivity_results=sensitivity,
            sensitivity_observations=observations,
        )
        lock = DevelopmentLock.model_validate(
            {**lock_payload, "lock_hash": content_hash(lock_payload)}
        )
        audit = BacktestAudit(
            schema_version="backtest-audit-v1",
            mode="development",
            evidence_kind=dataset.evidence_kind,
            dataset_hash=dataset.content_hash,
            configuration_hash=content_hash(config),
            model_hash=fitted.model_lock.sha256,
            report_hash=content_hash(report),
            folds=folds,
            outer_results=tuple(outer_results),
            final_result=None,
            selected_final_manifest=fitted,
            locked=lock,
            metrics=metrics,
            sensitivity_observations=observations,
        )
        return report, audit

    @staticmethod
    def _development_projections(
        outer: tuple[OuterEvaluation, ...], config: DevelopmentConfiguration, selected_id: str
    ) -> tuple[
        tuple[CandidateConfiguration, ...],
        tuple[SensitivityResult, ...],
        tuple[SensitivityObservation, ...],
    ]:
        candidates, sensitivity, observations = [], [], []
        for candidate in sorted(config.candidates, key=lambda c: c.candidate_id):
            projected = {}
            scores = []
            for result in outer:
                evidence = next(
                    c
                    for c in result.inner_candidates
                    if c.candidate.candidate_id == candidate.candidate_id
                )
                for validation in evidence.validations:
                    scores.append(validation.metrics.classification.macro_f1)
                    for key, value in validation.metrics.scalar_projection.items():
                        projected[f"{validation.partition_id}.{key}"] = value
            projected["inner.mean_macro_f1"] = math.fsum(s / len(scores) for s in scores)
            candidates.append(
                CandidateConfiguration(
                    candidate_id=candidate.candidate_id,
                    parameters=dict(candidate.parameters),
                    metrics=projected,
                    selected=candidate.candidate_id == selected_id,
                )
            )
            for parameter in config.sensitivity_parameters:
                descriptive = {"inner.mean_macro_f1": projected["inner.mean_macro_f1"]}
                sensitivity.append(
                    SensitivityResult(
                        parameter=parameter,
                        value=candidate.parameters[parameter],
                        metrics=descriptive,
                    )
                )
                observations.append(
                    SensitivityObservation(
                        candidate_id=candidate.candidate_id,
                        parameter=parameter,
                        value=candidate.parameters[parameter],
                        other_parameters={
                            k: v for k, v in candidate.parameters.items() if k != parameter
                        },
                        metrics=descriptive,
                        scope="retained-inner-development-descriptive-v1",
                    )
                )
        return tuple(candidates), tuple(sensitivity), tuple(observations)

    @staticmethod
    def _report(
        dataset: BacktestDataset,
        config: DevelopmentConfiguration,
        period: EvaluationPeriod,
        metrics: MetricEvidence,
        rows: tuple[CaseEvaluation, ...],
        candidates: tuple[CandidateConfiguration, ...],
        sensitivity: tuple[SensitivityResult, ...],
        mode: str,
    ) -> BacktestReport:
        identity = content_hash(
            (
                dataset.content_hash,
                content_hash(config),
                period,
                mode,
                tuple(r.case_id for r in rows),
                metrics.scalar_projection,
            )
        )
        return BacktestReport(
            report_id=f"{dataset.evidence_kind}:{mode}:{identity}",
            configuration_version=config.version,
            evaluation_start=period.start,
            evaluation_end=period.end,
            metrics=dict(metrics.scalar_projection),
            sensitivity_results=sensitivity,
            candidate_configurations=candidates,
            historical_case_ids=tuple(r.case_id for r in rows),
            schema_version="backtest-report-v1",
        )

    def _final(
        self, dataset: BacktestDataset, lock: DevelopmentLock, period: EvaluationPeriod
    ) -> tuple[BacktestReport, BacktestAudit]:
        if dataset.content_hash != lock.dataset_hash or dataset.evidence_kind != lock.evidence_kind:
            raise ValueError("locked dataset identity mismatch")
        if period != lock.final_period:
            raise ValueError("locked final period cannot be changed")
        groups = tuple(
            sorted((_group(c) for c in dataset.cases), key=lambda g: (g.event_time, g.cluster_id))
        )
        first_final_case = next(
            c for c in dataset.cases if c.cluster.cluster_id == lock.final_groups[0].cluster_id
        )
        if first_final_case.as_of != lock.first_final_as_of:
            raise ValueError("locked first final historical as_of mismatch")
        if groups[-len(lock.final_groups) :] != lock.final_groups:
            raise ValueError("locked final group/source membership mismatch")
        eligible = tuple(
            g
            for g in groups[: -len(lock.final_groups)]
            if g.event_time + lock.configuration.split.embargo <= lock.final_groups[0].event_time
        )
        omitted = tuple(g for g in groups[: -len(lock.final_groups)] if g not in eligible)
        if eligible != lock.fitted.training_groups or omitted != lock.omitted_final_boundary_groups:
            raise ValueError("locked final training/embargo membership mismatch")
        training = self._training(dataset, eligible, lock.fitted.fit_cutoff)
        if max(training.availability) != lock.fitted.maximum_evidence_available_at:
            raise ValueError("locked final training availability mismatch")
        lock.fitted.verify_artifacts()
        runtime = self._fitter.restore(_copy(lock.fitted))
        if _copy(runtime.manifest) != lock.fitted:
            raise ValueError("restored runtime differs from locked fitted identity")
        result = self._replay(
            dataset, lock.configuration, lock.final_groups, runtime, period, "locked-final"
        )
        report = self._report(
            dataset,
            lock.configuration,
            period,
            result.metrics,
            result.cases,
            lock.candidate_configurations,
            lock.sensitivity_results,
            "final",
        )
        audit = BacktestAudit(
            schema_version="backtest-audit-v1",
            mode="final",
            evidence_kind=dataset.evidence_kind,
            dataset_hash=dataset.content_hash,
            configuration_hash=lock.configuration_hash,
            model_hash=lock.model_hash,
            report_hash=content_hash(report),
            folds=(),
            outer_results=(),
            final_result=result,
            selected_final_manifest=lock.fitted,
            locked=lock,
            metrics=result.metrics,
            sensitivity_observations=lock.sensitivity_observations,
        )
        return report, audit


ARTIFACT_PATHS = (
    "data/calibration/selected-config.json",
    "data/calibration/impact-cutpoints.json",
    "data/results/development-backtest.json",
)
ArtifactType = Literal["selected-config", "impact-cutpoints", "development-backtest"]


class SelectedPayload(Record):
    artifact_type: Literal["selected-config"]
    locked: DevelopmentLock


class CutpointsPayload(Record):
    artifact_type: Literal["impact-cutpoints"]
    status: Literal["empirical-trained", "synthetic-only"]
    fit_hash: Sha256Hex
    training_hash: Sha256Hex
    identity: ImpactFitIdentity

    @model_validator(mode="after")
    def evidence_status(self) -> CutpointsPayload:
        if (self.status == "synthetic-only") != (self.identity.evidence_kind == "synthetic"):
            raise ValueError("cutpoint evidence status mismatch")
        return self


class DevelopmentPayload(Record):
    artifact_type: Literal["development-backtest"]
    report: BacktestReport
    audit: BacktestAudit

    @model_validator(mode="after")
    def report_claims_match_audit(self) -> DevelopmentPayload:
        if self.audit.mode != "development" or self.audit.report_hash != content_hash(self.report):
            raise ValueError("development report/audit hash mismatch")
        if (
            self.report.metrics != self.audit.metrics.scalar_projection
            or (self.report.candidate_configurations != self.audit.locked.candidate_configurations)
            or self.report.sensitivity_results != self.audit.locked.sensitivity_results
            or (
                self.report.historical_case_ids
                != tuple(c.case_id for o in self.audit.outer_results for c in o.test.cases)
            )
            or not self.report.report_id.startswith(f"{self.audit.evidence_kind}:development:")
        ):
            raise ValueError("development report claims contradict audit")
        return self


class UnresolvedPayload(Record):
    artifact_type: ArtifactType
    reasons: Annotated[tuple[NonEmptyString, ...], Field(min_length=1)]
    dataset_hash: None
    model_hash: None
    selected_candidate: None
    cutpoints: None
    report: None
    unresolved_model_declaration_hash: Sha256Hex | None


class PayloadDigest(Record):
    path: Literal[
        "data/calibration/selected-config.json",
        "data/calibration/impact-cutpoints.json",
        "data/results/development-backtest.json",
    ]
    digest: Sha256Hex


class ArtifactEnvelope(Record):
    schema_version: Literal["backtest-artifact-envelope-v1"]
    hash_domain: Literal["backtest-canonical-json-v1"]
    status: Literal["ready", "unresolved"]
    evidence_kind: EvidenceKind | None
    artifact_set_id: NonEmptyString
    authored_at: AwareDatetime
    source_terms: NonemptyTerms
    payload: SelectedPayload | CutpointsPayload | DevelopmentPayload | UnresolvedPayload
    payload_hash: Sha256Hex
    payload_hashes: Annotated[tuple[PayloadDigest, ...], Field(min_length=3, max_length=3)]
    artifact_set_hash: Sha256Hex

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema_version": self.schema_version,
                "hash_domain": self.hash_domain,
                "status": self.status,
                "evidence_kind": self.evidence_kind,
                "artifact_set_id": self.artifact_set_id,
                "authored_at": self.authored_at,
                "source_terms": self.source_terms,
                "payload_hashes": self.payload_hashes,
            }
        )

    @model_validator(mode="after")
    def envelope_claims_match(self) -> ArtifactEnvelope:
        if isinstance(self.payload, UnresolvedPayload):
            if self.status != "unresolved" or self.evidence_kind is not None:
                raise ValueError(
                    "unresolved artifact cannot claim ready empirical/synthetic evidence"
                )
        elif self.status != "ready" or self.evidence_kind is None:
            raise ValueError("ready artifact requires explicit evidence kind")
        if tuple(d.path for d in self.payload_hashes) != ARTIFACT_PATHS:
            raise ValueError("artifact receipts must contain all three canonical ordered paths")
        if self.payload_hash != content_hash(self.payload):
            raise ValueError("artifact payload content hash mismatch")
        index = ("selected-config", "impact-cutpoints", "development-backtest").index(
            self.payload.artifact_type
        )
        if self.payload_hashes[index].digest != self.payload_hash:
            raise ValueError("artifact cross-file payload hash mismatch")
        if self.artifact_set_hash != self.receipt_hash:
            raise ValueError("artifact-set receipt content hash mismatch")
        return self


class FinalAttemptReceipt(Record):
    schema_version: Literal["backtest-final-attempt-v1"]
    status: Literal["consumed"]
    lock_hash: Sha256Hex
    artifact_set_hash: Sha256Hex
    output_path: NonEmptyString
    evaluation_period: EvaluationPeriod


class FinalArtifact(Record):
    schema_version: Literal["backtest-final-artifact-v1"]
    status: Literal["reserved", "failed", "complete"]
    hash_domain: Literal["backtest-canonical-json-v1"]
    content_hash: Sha256Hex
    evidence_kind: EvidenceKind
    dataset_hash: Sha256Hex
    model_hash: Sha256Hex
    configuration_hash: Sha256Hex
    selection_hash: Sha256Hex
    fit_hash: Sha256Hex
    lock_hash: Sha256Hex
    artifact_set_hash: Sha256Hex
    holdout_groups: tuple[FoldGroup, ...]
    evaluation_period: EvaluationPeriod
    report: BacktestReport | None
    audit: BacktestAudit | None
    failure_reason: NonEmptyString | None

    @model_validator(mode="after")
    def final_claims_match(self) -> FinalArtifact:
        if self.content_hash != content_hash(self.model_dump(exclude={"content_hash"})):
            raise ValueError("final artifact content hash mismatch")
        if self.status == "complete":
            if self.report is None or self.audit is None or self.failure_reason is not None:
                raise ValueError("complete final artifact requires report/audit")
            audit, report = self.audit, self.report
            if (
                audit.mode != "final"
                or audit.report_hash != content_hash(report)
                or (report.metrics != audit.metrics.scalar_projection)
                or not report.report_id.startswith(f"{self.evidence_kind}:final:")
                or (audit.locked.lock_hash != self.lock_hash)
                or (
                    self.dataset_hash,
                    self.model_hash,
                    self.configuration_hash,
                    self.fit_hash,
                    self.selection_hash,
                    self.holdout_groups,
                    self.evaluation_period,
                )
                != (
                    audit.dataset_hash,
                    audit.model_hash,
                    audit.configuration_hash,
                    audit.selected_final_manifest.manifest_hash,
                    content_hash(audit.locked.selection),
                    audit.locked.final_groups,
                    audit.locked.final_period,
                )
            ):
                raise ValueError("final artifact claims contradict locked audit")
            assert audit.final_result is not None
            if report.historical_case_ids != tuple(c.case_id for c in audit.final_result.cases):
                raise ValueError("final report case membership mismatch")
        elif (
            self.report is not None
            or self.audit is not None
            or ((self.status == "failed") != (self.failure_reason is not None))
        ):
            raise ValueError("incomplete final artifact cannot claim successful report")
        return self
