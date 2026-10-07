"""Offline orchestration from immutable Source Items to complete Risk Signals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Protocol

from pydantic import AwareDatetime, Field, model_validator

from risk_engine.domain import (
    ConfidenceTarget,
    DomainModel,
    EntityLink,
    ImpactEstimate,
    InterpretedEvent,
    NonEmptyString,
    PortfolioMateriality,
    ProvenanceMethod,
    RiskSignal,
    Sha256Hex,
    SignalFlag,
    SourceItem,
    SourceType,
    VersionMetadata,
)
from risk_engine.nlp.interfaces import EventModel, SentimentModel
from risk_engine.nlp.interpret import Matcher, interpret
from risk_engine.risk.confidence import (
    ConfidenceCalibrator,
    ConfidenceEstimate,
    ConfidenceScoreDefinition,
    RawConfidenceScore,
)

ACTION_PRIORITY_METHOD: Literal[
    "impact-confidence-percentage-materiality-product-v1"
] = "impact-confidence-percentage-materiality-product-v1"


class ImpactEstimatorPort(Protocol):
    """The existing ImpactEstimator scoring seam, without its construction details."""

    def estimate(self, event: InterpretedEvent, as_of: datetime) -> ImpactEstimate: ...


class PortfolioMaterialityProvider(Protocol):
    """Optional current-portfolio calculation supplied by an explicit caller."""

    def measure(
        self,
        event: InterpretedEvent,
        entity: EntityLink,
        estimate: ImpactEstimate,
        *,
        as_of: datetime,
    ) -> AuditedPortfolioMateriality | None: ...


class ModelIdentity(DomainModel):
    event_model_version: NonEmptyString
    sentiment_model_version: NonEmptyString
    entity_linker_version: NonEmptyString


class ActionPriorityCandidate(DomainModel):
    """Requirement-driven attention candidate; not a selected trigger policy."""

    method: Literal[
        "impact-confidence-percentage-materiality-product-v1"
    ] = ACTION_PRIORITY_METHOD
    validation_status: Literal["unvalidated"] = "unvalidated"
    selection_basis: Literal["requirement-driven"] = "requirement-driven"

    def calculate(
        self,
        impact: ImpactEstimate,
        confidence: ConfidenceEstimate,
        materiality: PortfolioMateriality,
    ) -> float:
        return impact.impact_score * confidence.probability * materiality.percentage_loss


class PortfolioMaterialityAudit(DomainModel):
    provider_version: NonEmptyString
    calculation_version: NonEmptyString
    portfolio_id: NonEmptyString
    portfolio_version: NonEmptyString
    market_snapshot_id: NonEmptyString
    scenario_id: NonEmptyString
    scenario_version: NonEmptyString
    valuation_rule_version: NonEmptyString
    as_of: AwareDatetime


class AuditedPortfolioMateriality(DomainModel):
    materiality: PortfolioMateriality
    audit: PortfolioMaterialityAudit


class ModelManifest(DomainModel):
    manifest_version: Literal["risk-engine-model-manifest-v1"] = (
        "risk-engine-model-manifest-v1"
    )
    models: ModelIdentity
    action_priority: ActionPriorityCandidate
    portfolio_materiality: AuditedPortfolioMateriality | None


class ImpactCalibrationIdentity(DomainModel):
    calibration_version: NonEmptyString
    reference_basket_version: NonEmptyString


class ConfidenceCalibrationIdentity(DomainModel):
    target: ConfidenceTarget
    calibration_version: NonEmptyString
    evidence_kind: Literal["synthetic", "empirical"]
    calibration_evidence_snapshot_id: NonEmptyString
    calibration_evidence_snapshot_hash: Sha256Hex
    development_start: AwareDatetime
    development_end: AwareDatetime
    frozen_at: AwareDatetime
    score_definition: ConfidenceScoreDefinition
    fit_version: NonEmptyString
    transform_version: NonEmptyString
    temperature: float = Field(ge=0.05, le=20)
    evidence_hash: Sha256Hex
    fitted_artifact_hash: Sha256Hex
    source_terms: tuple[NonEmptyString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def source_terms_are_canonical(self) -> ConfidenceCalibrationIdentity:
        if self.source_terms != tuple(sorted(set(self.source_terms))):
            raise ValueError("Confidence calibration source terms must be sorted and unique")
        return self


def confidence_calibration_identity(
    calibrator: ConfidenceCalibrator,
) -> ConfidenceCalibrationIdentity:
    """Expose the exact fitted Confidence identity emitted in signal manifests."""
    calibrator = ConfidenceCalibrator.model_validate(calibrator.model_dump(mode="python"))
    return ConfidenceCalibrationIdentity(
        target=calibrator.target,
        calibration_version=calibrator.evidence.calibration_version,
        evidence_kind=calibrator.evidence.evidence_kind,
        calibration_evidence_snapshot_id=calibrator.evidence.snapshot_id,
        calibration_evidence_snapshot_hash=calibrator.evidence.snapshot_hash,
        development_start=calibrator.evidence.development_start,
        development_end=calibrator.evidence.development_end,
        frozen_at=calibrator.evidence.frozen_at,
        score_definition=calibrator.evidence.score_definition,
        fit_version=calibrator.fit_version,
        transform_version=calibrator.transform_version,
        temperature=calibrator.temperature,
        evidence_hash=calibrator.evidence_hash,
        fitted_artifact_hash=calibrator.fitted_artifact_hash,
        source_terms=tuple(sorted({row.source_terms for row in calibrator.evidence.rows})),
    )


class CalibrationManifest(DomainModel):
    manifest_version: Literal["risk-engine-calibration-manifest-v1"] = (
        "risk-engine-calibration-manifest-v1"
    )
    confidence: ConfidenceEstimate
    confidence_calibration: ConfidenceCalibrationIdentity
    impact: ImpactCalibrationIdentity


class SourceItemIdentity(DomainModel):
    source_item_id: NonEmptyString
    source_type: SourceType
    provider: NonEmptyString
    source_reference: NonEmptyString
    content_hash: Sha256Hex
    snapshot_id: NonEmptyString
    published_at: AwareDatetime
    retrieved_at: AwareDatetime
    provenance: NonEmptyString
    license: NonEmptyString


class SnapshotManifest(DomainModel):
    manifest_version: Literal["risk-engine-snapshot-manifest-v1"] = (
        "risk-engine-snapshot-manifest-v1"
    )
    source_items: tuple[SourceItemIdentity, ...] = Field(min_length=1)


class _EngineConfiguration(DomainModel):
    model_identity: ModelIdentity
    schema_version: NonEmptyString
    action_priority: ActionPriorityCandidate


def _canonical_json(record: DomainModel) -> str:
    return json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _source_identity(item: SourceItem) -> SourceItemIdentity:
    return SourceItemIdentity(
        source_item_id=item.source_item_id,
        source_type=item.source_type,
        provider=item.provider,
        source_reference=item.source_reference,
        content_hash=item.content_hash,
        snapshot_id=item.snapshot_id,
        published_at=item.published_at,
        retrieved_at=item.retrieved_at,
        provenance=item.provenance,
        license=item.license,
    )


def _unknown_entity(event: InterpretedEvent) -> EntityLink:
    return EntityLink(
        entity_id=f"unknown:{event.event_id}",
        canonical_name="Unknown entity",
        confidence=0.0,
        evidence=event.rationale,
        ambiguous=True,
        candidate_entity_ids=(),
    )


def _signal_id(item: SourceItem, event: InterpretedEvent, entity: EntityLink) -> str:
    identity = json.dumps(
        {
            "candidate_entity_ids": entity.candidate_entity_ids,
            "content_hash": item.content_hash,
            "entity_evidence": entity.evidence,
            "entity_id": entity.entity_id,
            "event_id": event.event_id,
            "source_item_id": item.source_item_id,
            "version": "risk-signal-id-v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "signal:" + hashlib.sha256(identity.encode()).hexdigest()


def _flags(
    event: InterpretedEvent, entity: EntityLink, impact: ImpactEstimate
) -> tuple[SignalFlag, ...]:
    flags: list[SignalFlag] = []
    unknown_or_ambiguous = (
        not event.eligible_for_automatic_stress
        or entity.ambiguous
        or entity.entity_id.casefold().startswith(("unknown:", "ambiguous:"))
    )
    if unknown_or_ambiguous:
        flags.append(SignalFlag.AMBIGUOUS_ENTITY)
    if impact.method is ProvenanceMethod.HYPOTHETICAL:
        flags.extend((SignalFlag.THIN_HISTORY, SignalFlag.HYPOTHETICAL_SCENARIO))
    return tuple(flags)


class RiskEngine:
    """Compose deterministic local ports and return only fully built Risk Signals.

    TriggerPolicy remains a downstream consumer because its recorded TriggerDecision
    cannot be represented by this interface's declared ``list[RiskSignal]`` result.
    """

    def __init__(
        self,
        *,
        matcher: Matcher,
        sentiment_model: SentimentModel,
        event_model: EventModel,
        impact_estimator: ImpactEstimatorPort,
        confidence_calibrator: ConfidenceCalibrator,
        model_identity: ModelIdentity,
        schema_version: str,
        materiality_provider: PortfolioMaterialityProvider | None,
        action_priority: ActionPriorityCandidate,
    ) -> None:
        configuration = _EngineConfiguration(
            model_identity=model_identity,
            schema_version=schema_version,
            action_priority=action_priority,
        )
        calibrator = ConfidenceCalibrator.model_validate(
            confidence_calibrator.model_dump(mode="python")
        )
        definition = calibrator.evidence.score_definition
        if (
            configuration.model_identity.event_model_version != definition.event_model_version
            or configuration.model_identity.entity_linker_version
            != definition.entity_linker_version
        ):
            raise ValueError("model identity differs from Confidence score definition")
        self._matcher = matcher
        self._sentiment_model = sentiment_model
        self._event_model = event_model
        self._impact_estimator = impact_estimator
        self._confidence_calibrator = calibrator
        self._confidence_calibration = confidence_calibration_identity(calibrator)
        self._configuration = configuration
        self._materiality_provider = materiality_provider

    def analyze(self, items: Sequence[SourceItem], as_of: datetime) -> list[RiskSignal]:
        """Build the whole batch locally; any validation or port failure raises atomically."""
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("analysis as_of must be timezone-aware")
        source_items = tuple(
            SourceItem.model_validate(item.model_dump(mode="python")) for item in items
        )
        for item in source_items:
            if item.published_at > as_of or item.retrieved_at > as_of:
                raise ValueError(
                    f"Source Item {item.source_item_id} is unavailable by analysis as_of"
                )

        completed: list[RiskSignal] = []
        for item in source_items:
            events = interpret(
                item,
                self._matcher,
                self._sentiment_model,
                self._event_model,
            )
            for event in events:
                if event.source_item_id != item.source_item_id:
                    raise ValueError("interpreted event Source Item identity mismatch")
                if event.rationale not in item.text or any(
                    evidence not in item.text for evidence in event.evidence
                ):
                    raise ValueError("Risk Signal rationale and evidence must be extractive")
                impact = ImpactEstimate.model_validate(
                    self._impact_estimator.estimate(event, as_of).model_dump(mode="python")
                )
                entities = event.entity_links or (_unknown_entity(event),)
                for entity in entities:
                    confidence = self._confidence_calibrator.transform(
                        RawConfidenceScore(
                            definition=self._confidence_calibrator.evidence.score_definition,
                            entity_link_confidence=entity.confidence,
                            classification_confidence=event.classification_confidence,
                        ),
                        as_of=as_of,
                    )
                    audited_materiality = self._measure_materiality(
                        event, entity, impact, as_of=as_of
                    )
                    materiality = (
                        audited_materiality.materiality
                        if audited_materiality is not None
                        else None
                    )
                    priority = (
                        self._configuration.action_priority.calculate(
                            impact, confidence, materiality
                        )
                        if materiality is not None
                        else None
                    )
                    completed.append(
                        RiskSignal(
                            signal_id=_signal_id(item, event, entity),
                            source_item_id=item.source_item_id,
                            entity=entity,
                            sentiment=event.sentiment,
                            event_class=event.event_class,
                            impact=impact,
                            confidence=confidence.probability,
                            confidence_target=confidence.target,
                            rationale=event.rationale,
                            evidence=event.evidence,
                            flags=_flags(event, entity, impact),
                            versions=self._versions(
                                item, impact, confidence, audited_materiality
                            ),
                            portfolio_materiality=materiality,
                            action_priority=priority,
                        )
                    )
        return completed

    def _measure_materiality(
        self,
        event: InterpretedEvent,
        entity: EntityLink,
        impact: ImpactEstimate,
        *,
        as_of: datetime,
    ) -> AuditedPortfolioMateriality | None:
        if self._materiality_provider is None:
            return None
        result = self._materiality_provider.measure(event, entity, impact, as_of=as_of)
        if result is None:
            return None
        audited = AuditedPortfolioMateriality.model_validate(
            result.model_dump(mode="python")
        )
        if audited.audit.as_of != as_of:
            raise ValueError("materiality audit as_of must equal analysis as_of")
        return audited

    def _versions(
        self,
        item: SourceItem,
        impact: ImpactEstimate,
        confidence: ConfidenceEstimate,
        materiality: AuditedPortfolioMateriality | None,
    ) -> VersionMetadata:
        return VersionMetadata(
            schema_version=self._configuration.schema_version,
            model_version=_canonical_json(
                ModelManifest(
                    models=self._configuration.model_identity,
                    action_priority=self._configuration.action_priority,
                    portfolio_materiality=materiality,
                )
            ),
            calibration_version=_canonical_json(
                CalibrationManifest(
                    confidence=confidence,
                    confidence_calibration=self._confidence_calibration,
                    impact=ImpactCalibrationIdentity(
                        calibration_version=impact.calibration_version,
                        reference_basket_version=impact.reference_basket_version,
                    ),
                )
            ),
            snapshot_version=_canonical_json(
                SnapshotManifest(source_items=(_source_identity(item),))
            ),
        )


__all__ = [
    "ACTION_PRIORITY_METHOD",
    "ActionPriorityCandidate",
    "AuditedPortfolioMateriality",
    "ConfidenceCalibrationIdentity",
    "ImpactEstimatorPort",
    "ModelIdentity",
    "PortfolioMaterialityAudit",
    "PortfolioMaterialityProvider",
    "RiskEngine",
]
