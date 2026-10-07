"""Preregistered local runtime definitions and partition-bound evidence readers."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, Field, model_validator

from risk_engine.backtest.module import (
    CandidateSpec,
    LocalArtifact,
    NonemptyTerms,
    Record,
    TrainingPartition,
    content_hash,
)
from risk_engine.calibration.event_study import EventWindow
from risk_engine.config import AnalogueConfig
from risk_engine.domain import CurrencyCode, EventClass, MarketSnapshot, NonEmptyString, Sha256Hex
from risk_engine.impact.analogues import HistoricalAnalogue
from risk_engine.impact.module import FactorBinding
from risk_engine.impact.reference_basket import ReferenceBasketSpec, ReferenceReturnRow
from risk_engine.nlp.interfaces import ModelLock


def _absolute(artifact: LocalArtifact) -> None:
    if not Path(artifact.path).is_absolute():
        raise ValueError("LocalArtifact path must be absolute")


def _verified_bytes(artifact: LocalArtifact, cutoff: datetime) -> bytes:
    _absolute(artifact)
    if artifact.available_at > cutoff:
        raise ValueError("local artifact unavailable by cutoff")
    data = Path(artifact.path).read_bytes()
    if hashlib.sha256(data).hexdigest() != artifact.sha256:
        raise ValueError("local artifact byte hash mismatch")
    return data


def _canonical(values: tuple[str, ...], description: str, *, populated: bool = True) -> None:
    if (populated and not values) or values != tuple(sorted(set(values))):
        raise ValueError(f"{description} must be sorted, unique and complete")


def _normalize_utc(value: object) -> object:
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            current = getattr(value, field_name)
            normalized = _normalize_utc(current)
            if normalized is not current:
                object.__setattr__(value, field_name, normalized)
        return value
    if isinstance(value, tuple):
        return tuple(_normalize_utc(item) for item in value)
    if isinstance(value, list):
        return [_normalize_utc(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_utc(item) for key, item in value.items()}
    return value


class RuntimeRecord(Record):
    @model_validator(mode="after")
    def utc_instants(self) -> RuntimeRecord:
        _normalize_utc(self)
        return self


class PolicySelectionSpec(RuntimeRecord):
    schema_version: Literal["production-policy-selection-spec-v1"]
    version: NonEmptyString
    confidence_thresholds: tuple[Decimal, ...] = Field(min_length=1)
    economic_floors: tuple[Decimal, ...] = Field(min_length=1)
    materiality_tolerance: Decimal = Field(ge=0)
    currency: CurrencyCode
    train_groups: Annotated[int, Field(strict=True, gt=0)]
    validation_groups: Annotated[int, Field(strict=True, gt=0)]
    minimum_folds: Annotated[int, Field(strict=True, gt=0)]
    minimum_cases: Annotated[int, Field(strict=True, gt=0)]
    minimum_positive_cases: Annotated[int, Field(strict=True, gt=0)]
    minimum_negative_cases: Annotated[int, Field(strict=True, gt=0)]
    alert_budget: Annotated[int, Field(strict=True, ge=0)]
    objective: Literal["material-event-f1-v1"]
    tie_rule: Literal["fewest-alerts-highest-confidence-highest-floor-v1"]

    @model_validator(mode="after")
    def valid_grid(self) -> PolicySelectionSpec:
        if (
            any(value <= 0 or value > 1 for value in self.confidence_thresholds)
            or tuple(sorted(set(self.confidence_thresholds))) != self.confidence_thresholds
        ):
            raise ValueError(
                "Confidence threshold grid must be positive, bounded, sorted and unique"
            )
        if (
            any(value <= 0 for value in self.economic_floors)
            or tuple(sorted(set(self.economic_floors))) != self.economic_floors
            or self.materiality_tolerance >= self.economic_floors[0]
        ):
            raise ValueError("economic floor grid/tolerance is invalid")
        return self


class WindowConvention(RuntimeRecord):
    key: NonEmptyString
    window: EventWindow


class BasketConvention(RuntimeRecord):
    key: NonEmptyString
    spec: ReferenceBasketSpec
    returns: tuple[ReferenceReturnRow, ...] = Field(min_length=1)
    base_market: MarketSnapshot
    factor_bindings: tuple[FactorBinding, ...] = Field(min_length=1)
    available_at: AwareDatetime
    source_terms: NonemptyTerms
    source_hashes: tuple[Sha256Hex, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def historically_available(self) -> BasketConvention:
        _canonical(self.source_terms, "basket source terms")
        _canonical(self.source_hashes, "basket source hashes")
        if (
            self.spec.calibration_date > self.available_at
            or self.base_market.as_of > self.available_at
            or any(row.observed_at > self.spec.calibration_date for row in self.returns)
            or any(row.observed_at > self.available_at for row in self.returns)
            or any(row.observed_at > self.available_at for row in self.base_market.observations)
        ):
            raise ValueError("Reference Basket resource unavailable at declared time")
        source_ids = tuple(row.historical_factor_id for row in self.factor_bindings)
        target_ids = tuple(row.valuation_factor_id for row in self.factor_bindings)
        if len(set(source_ids)) != len(source_ids) or len(set(target_ids)) != len(target_ids):
            raise ValueError("duplicate Reference Basket factor binding")
        return self


class MatchingConvention(RuntimeRecord):
    key: NonEmptyString
    config: AnalogueConfig

    @model_validator(mode="after")
    def bootstrap_only(self) -> MatchingConvention:
        if self.config.validation_status != "bootstrap_unvalidated":
            raise ValueError("matching requires a separately bound validation artifact")
        return self


class RuntimeDefinition(RuntimeRecord):
    schema_version: Literal["production-runtime-definition-v1"]
    version: NonEmptyString
    content_hash: Sha256Hex
    frozen_at: AwareDatetime
    source_terms: NonemptyTerms
    signal_schema_version: NonEmptyString
    model_lock: LocalArtifact
    catalogue: LocalArtifact
    windows: tuple[WindowConvention, ...] = Field(min_length=1)
    baskets: tuple[BasketConvention, ...] = Field(min_length=1)
    matching: tuple[MatchingConvention, ...] = Field(min_length=1)
    policy_selection: PolicySelectionSpec | None

    @model_validator(mode="after")
    def preregistered(self) -> RuntimeDefinition:
        _canonical(self.source_terms, "definition source terms")
        for artifact in (self.model_lock, self.catalogue):
            _absolute(artifact)
            if artifact.available_at > self.frozen_at:
                raise ValueError("definition resource newer than frozen_at")
        for registry in (self.windows, self.baskets, self.matching):
            _canonical(tuple(row.key for row in registry), "registry keys")
        if any(row.available_at > self.frozen_at for row in self.baskets) or any(
            row.config.frozen_at > self.frozen_at for row in self.matching
        ):
            raise ValueError("definition resource newer than frozen_at")
        if self.content_hash != content_hash(self.model_dump(exclude={"content_hash"})):
            raise ValueError("runtime definition content hash mismatch")
        return self


class CaseEvidenceReference(RuntimeRecord):
    case_id: NonEmptyString
    cluster_id: NonEmptyString
    artifact: LocalArtifact

    @model_validator(mode="after")
    def absolute_descriptor(self) -> CaseEvidenceReference:
        _absolute(self.artifact)
        return self


class RuntimeEvidenceIndex(RuntimeRecord):
    schema_version: Literal["production-runtime-evidence-index-v1"]
    content_hash: Sha256Hex
    dataset_hash: Sha256Hex
    source_terms: NonemptyTerms
    cases: tuple[CaseEvidenceReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def bound_index(self) -> RuntimeEvidenceIndex:
        _canonical(self.source_terms, "index source terms")
        _canonical(tuple(row.case_id for row in self.cases), "index case IDs")
        if len({row.cluster_id for row in self.cases}) != len(self.cases):
            raise ValueError("duplicate cluster descriptor")
        if len({row.artifact.path for row in self.cases}) != len(self.cases):
            raise ValueError("duplicate case descriptor path")
        if self.content_hash != content_hash(self.model_dump(exclude={"content_hash"})):
            raise ValueError("runtime evidence index content hash mismatch")
        return self


class SourceEvidenceBinding(RuntimeRecord):
    source_item_id: NonEmptyString
    content_hash: Sha256Hex


class InterpretationTarget(RuntimeRecord):
    source_item_id: NonEmptyString
    content_hash: Sha256Hex
    event_id: NonEmptyString
    interpretation_version: Literal["interpret-v1"]
    actual_entity_ids: tuple[NonEmptyString, ...]
    actual_event_class: EventClass
    label_available_at: AwareDatetime
    label_provenance: NonEmptyString
    label_hash: Sha256Hex
    source_terms: NonemptyTerms

    @model_validator(mode="after")
    def independent_label(self) -> InterpretationTarget:
        _canonical(self.actual_entity_ids, "actual entity IDs", populated=False)
        _canonical(self.source_terms, "target source terms")
        if self.label_hash != content_hash(self.model_dump(exclude={"label_hash"})):
            raise ValueError("independent target label hash mismatch")
        return self


class TrainingCaseEvidence(RuntimeRecord):
    schema_version: Literal["production-training-case-v1"]
    case_id: NonEmptyString
    cluster_id: NonEmptyString
    outcome_hash: Sha256Hex
    source_items: tuple[SourceEvidenceBinding, ...] = Field(min_length=1)
    targets: tuple[InterpretationTarget, ...] = Field(min_length=1)
    analogue: HistoricalAnalogue
    source_terms: NonemptyTerms

    @model_validator(mode="after")
    def complete_case(self) -> TrainingCaseEvidence:
        _canonical(self.source_terms, "case source terms")
        _canonical(tuple(row.source_item_id for row in self.source_items), "source descriptors")
        bindings = {row.source_item_id: row.content_hash for row in self.source_items}
        target_keys = [(row.source_item_id, row.event_id) for row in self.targets]
        if len(set(target_keys)) != len(target_keys):
            raise ValueError("duplicate independent interpretation target")
        if any(bindings.get(row.source_item_id) != row.content_hash for row in self.targets):
            raise ValueError("interpretation target refers to unbound Source Item hash")
        if {row.source_item_id for row in self.targets} != set(bindings):
            raise ValueError("source descriptor lacks independent interpretation target")
        if self.analogue.event_id != self.cluster_id or self.analogue.evidence_kind != "observed":
            raise ValueError("empirical case requires an observed cluster analogue")
        return self


class LocalModelLocations(RuntimeRecord):
    sentiment_snapshot: Path
    event_snapshot: Path
    device: Literal["cpu", "cuda"]


class RuntimeLocationConfig(RuntimeRecord):
    schema_version: Literal["production-runtime-locations-v1"]
    definition_path: Path
    evidence_index_path: Path | None
    artifact_root: Path
    models: LocalModelLocations


class CandidateRuntimeSpec(RuntimeRecord):
    candidate: CandidateSpec
    definition_hash: Sha256Hex
    window: WindowConvention
    basket: BasketConvention
    matching: MatchingConvention
    confidence_fit: Literal["binary-temperature-logit-clip1e-12-v1"]
    cutpoint_rule: Literal["nearest-rank-lower-ties"]
    scenario_choice: Literal["nearest-median-reference-loss-event-id-v1"]
    policy_mode: Literal["held-out-material-event-f1-v1", "unselected"]


def load_runtime_definition(path: Path) -> RuntimeDefinition:
    definition = RuntimeDefinition.model_validate_json(path.read_bytes())
    lock_bytes = _verified_bytes(definition.model_lock, definition.frozen_at)
    _verified_bytes(definition.catalogue, definition.frozen_at)
    ModelLock.model_validate_json(lock_bytes)
    return definition


def load_runtime_evidence_index(path: Path) -> RuntimeEvidenceIndex:
    return RuntimeEvidenceIndex.model_validate_json(path.read_bytes())


def load_training_evidence(
    index: RuntimeEvidenceIndex, training: TrainingPartition
) -> tuple[TrainingCaseEvidence, ...]:
    index = RuntimeEvidenceIndex.model_validate(index.model_dump(mode="python"))
    training = TrainingPartition.model_validate(training.model_dump(mode="python"))
    if training.evidence_kind != "empirical":
        raise ValueError("production training requires empirical TrainingPartition evidence")
    if index.dataset_hash != training.dataset_hash:
        raise ValueError("runtime evidence index dataset hash mismatch")
    case_ids = tuple(case.case_id for case in training.cases)
    cluster_ids = tuple(case.cluster.cluster_id for case in training.cases)
    source_ids = tuple(
        item.source_item_id for case in training.cases for item in case.cluster.items
    )
    outcome_ids = tuple(outcome.case_id for outcome in training.outcomes)
    if any(
        len(values) != len(set(values))
        for values in (case_ids, cluster_ids, source_ids, outcome_ids)
    ):
        raise ValueError("duplicate training case, cluster, Source Item or outcome membership")
    references = {row.case_id: row for row in index.cases}
    outcomes = {row.case_id: row for row in training.outcomes}
    result: list[TrainingCaseEvidence] = []
    for case in training.cases:
        outcome = outcomes[case.case_id]
        descriptor = references.get(case.case_id)
        if descriptor is None or descriptor.cluster_id != case.cluster.cluster_id:
            raise ValueError("missing or mismatched training case descriptor")
        evidence = TrainingCaseEvidence.model_validate_json(
            _verified_bytes(descriptor.artifact, training.cutoff)
        )
        if evidence.case_id != case.case_id or evidence.cluster_id != case.cluster.cluster_id:
            raise ValueError("training sidecar case/cluster identity mismatch")
        if evidence.outcome_hash != content_hash(outcome):
            raise ValueError("training outcome hash mismatch")
        expected_sources = tuple(
            sorted((item.source_item_id, item.content_hash) for item in case.cluster.items)
        )
        actual_sources = tuple(
            (row.source_item_id, row.content_hash) for row in evidence.source_items
        )
        if actual_sources != expected_sources:
            raise ValueError("training Source Item identity/hash membership mismatch")
        if any(
            hashlib.sha256(item.text.encode("utf-8")).hexdigest() != item.content_hash
            for item in case.cluster.items
        ):
            raise ValueError("training Source Item text hash mismatch")
        if any(
            target.label_available_at > outcome.label_available_at for target in evidence.targets
        ):
            raise ValueError("independent label later than observed outcome label availability")
        if evidence.analogue.available_at > outcome.outcome_available_at:
            raise ValueError("analogue evidence later than observed outcome availability")
        outcome_resources = (
            case.market.as_of,
            case.market_available_at,
            *(row.observed_at for row in case.market.observations),
            *(item.published_at for item in case.cluster.items),
            *(item.retrieved_at for item in case.cluster.items),
            *(row.available_at for row in outcome.reference_losses),
        )
        if any(available_at > outcome.outcome_available_at for available_at in outcome_resources):
            raise ValueError("source, market or reference evidence exceeds outcome availability")
        latest = max(
            outcome.outcome_available_at,
            outcome.label_available_at,
            evidence.analogue.available_at,
            *(target.label_available_at for target in evidence.targets),
        )
        if latest > descriptor.artifact.available_at or latest > training.cutoff:
            raise ValueError("case evidence descriptor predates underlying availability")
        result.append(evidence)
    return tuple(result)


def resolve_candidate(
    candidate: CandidateSpec, definition: RuntimeDefinition
) -> CandidateRuntimeSpec:
    candidate = CandidateSpec.model_validate(candidate.model_dump(mode="python"))
    definition = RuntimeDefinition.model_validate(definition.model_dump(mode="python"))
    parameters = candidate.parameters
    keys = {
        "runtime_definition_sha256",
        "event_window",
        "confidence_fit",
        "cutpoint_rule",
        "scenario_choice",
        "policy_mode",
    }
    if set(parameters) != keys or any(type(value) is not str for value in parameters.values()):
        raise ValueError("candidate requires exactly six string runtime parameter keys")
    string_parameters = {key: value for key, value in parameters.items() if isinstance(value, str)}
    if string_parameters["runtime_definition_sha256"] != definition.content_hash:
        raise ValueError("candidate runtime definition hash mismatch")
    windows = {row.key: row for row in definition.windows}
    baskets = {row.key: row for row in definition.baskets}
    matching = {row.key: row for row in definition.matching}
    try:
        window = windows[string_parameters["event_window"]]
        basket = baskets[candidate.basket_convention]
        match = matching[candidate.matching_convention]
    except KeyError as error:
        raise ValueError("candidate references an unknown runtime convention") from error
    if candidate.event_window_days != window.window.end - window.window.start + 1:
        raise ValueError("candidate event window horizon mismatch")
    if (
        string_parameters["policy_mode"] == "held-out-material-event-f1-v1"
        and definition.policy_selection is None
    ):
        raise ValueError("held-out policy mode requires a PolicySelectionSpec")
    return CandidateRuntimeSpec(
        candidate=candidate,
        definition_hash=definition.content_hash,
        window=window,
        basket=basket,
        matching=match,
        confidence_fit=string_parameters["confidence_fit"],
        cutpoint_rule=string_parameters["cutpoint_rule"],
        scenario_choice=string_parameters["scenario_choice"],
        policy_mode=string_parameters["policy_mode"],
    )
