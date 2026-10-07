"""Content-addressed frozen production state, without refitting authority."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from risk_engine.backtest.module import (
    CandidateSpec,
    LocalArtifact,
    NonemptyTerms,
    canonical_bytes,
    content_hash,
)
from risk_engine.backtest.policy_selection import PolicySelectionAudit
from risk_engine.backtest.runtime_inputs import RuntimeRecord, _verified_bytes
from risk_engine.backtest.splits import FoldGroup
from risk_engine.config import AnalogueConfig
from risk_engine.domain import MarketSnapshot, NonEmptyString, Sha256Hex
from risk_engine.impact.module import FrozenImpactCalibration
from risk_engine.nlp.interfaces import ModelLock
from risk_engine.risk.confidence import ConfidenceCalibrator
from risk_engine.risk.module import ModelIdentity


class FrozenRuntimeState(RuntimeRecord):
    schema_version: Literal["production-runtime-state-v1"]
    content_hash: Sha256Hex
    candidate: CandidateSpec
    configuration_hash: Sha256Hex
    runtime_definition_hash: Sha256Hex
    dataset_hash: Sha256Hex
    training_groups: tuple[FoldGroup, ...] = Field(min_length=1)
    maximum_evidence_available_at: AwareDatetime
    fit_cutoff: AwareDatetime
    model_identity: ModelIdentity
    model_lock: LocalArtifact
    catalogue: LocalArtifact
    source_terms: NonemptyTerms
    training_evidence: tuple[LocalArtifact, ...] = Field(min_length=1)
    confidence_calibrator: ConfidenceCalibrator
    impact_calibration: FrozenImpactCalibration
    matching_config: AnalogueConfig
    base_market: MarketSnapshot
    scenario_choice: Literal["nearest-median-reference-loss-event-id-v1"]
    policy_audit: PolicySelectionAudit | None
    policy_absence_reason: NonEmptyString | None

    @model_validator(mode="after")
    def cross_identities(self) -> FrozenRuntimeState:
        candidate = self.candidate
        parameters = candidate.parameters
        if set(parameters) != {
            "runtime_definition_sha256",
            "event_window",
            "confidence_fit",
            "cutpoint_rule",
            "scenario_choice",
            "policy_mode",
        } or any(type(value) is not str for value in parameters.values()):
            raise ValueError("state candidate requires six string runtime semantics")
        if (
            parameters["runtime_definition_sha256"] != self.runtime_definition_hash
            or parameters["scenario_choice"] != self.scenario_choice
            or parameters["confidence_fit"] != self.confidence_calibrator.transform_version
            or parameters["cutpoint_rule"] != self.impact_calibration.quantile_convention
            or parameters["policy_mode"] not in {"unselected", "held-out-material-event-f1-v1"}
        ):
            raise ValueError("state candidate/definition/calibration identity mismatch")
        confidence = self.confidence_calibrator.evidence
        impact = self.impact_calibration
        definition = confidence.score_definition
        if (
            confidence.evidence_kind != "empirical"
            or confidence.snapshot_hash != self.dataset_hash
            or definition.event_model_version != self.model_identity.event_model_version
            or definition.entity_linker_version != self.model_identity.entity_linker_version
            or self.model_identity.entity_linker_version
            != "entity-matcher-v1:" + self.catalogue.sha256
        ):
            raise ValueError("state dataset/model/Confidence identity mismatch")
        sources = tuple(
            source for group in self.training_groups for source in group.source_item_ids
        )
        ids = tuple(group.cluster_id for group in self.training_groups)
        if (
            len(set(ids)) != len(ids)
            or len(set(sources)) != len(sources)
            or ids != tuple(row.analogue.event_id for row in impact.training)
            or set(sources) != {row.source_item_id for row in confidence.rows}
            or len(self.training_evidence) != len(self.training_groups)
            or len({row.path for row in self.training_evidence}) != len(self.training_evidence)
        ):
            raise ValueError("state calibration/membership identity mismatch")
        event_times = {g.cluster_id: g.event_time for g in self.training_groups}
        if any(
            event_times[row.analogue.event_id] != row.analogue.event_at for row in impact.training
        ):
            raise ValueError("state analogue event time differs from training group")
        latest = self.maximum_evidence_available_at
        if (
            impact.training_end != latest + timedelta(microseconds=1)
            or impact.calibrated_at != latest + timedelta(microseconds=2)
            or confidence.frozen_at != impact.calibrated_at
            or confidence.development_end != impact.training_end
            or impact.calibrated_at >= self.fit_cutoff
            or candidate.event_window_days != impact.window.end - impact.window.start + 1
            or any(row.base_market != self.base_market for row in impact.training)
            or any(row.analogue.available_at > latest for row in impact.training)
            or any(row.label_available_at > latest for row in confidence.rows)
            or self.matching_config.frozen_at > latest
        ):
            raise ValueError("state frozen calibration chronology/window/base market mismatch")
        for artifact in (self.model_lock, self.catalogue, *self.training_evidence):
            if not Path(artifact.path).is_absolute() or artifact.available_at > latest:
                raise ValueError("state descriptor availability/path mismatch")
        if self.source_terms != tuple(sorted(set(self.source_terms))):
            raise ValueError("state source terms must be canonical")
        audit = self.policy_audit
        if audit is None:
            if self.policy_absence_reason is None:
                raise ValueError("state policy absence needs an explicit reason")
        else:
            if (
                audit.configuration_hash != self.configuration_hash
                or audit.runtime_definition_hash != self.runtime_definition_hash
                or audit.parent_training_hash != content_hash(self.training_groups)
                or audit.absence_reason != self.policy_absence_reason
            ):
                raise ValueError("state policy audit identity/absence mismatch")
        if self.content_hash != content_hash(
            self.model_dump(mode="json", exclude={"content_hash"})
        ):
            raise ValueError("runtime state content hash mismatch")
        return self


def write_runtime_state(state: FrozenRuntimeState, artifact_root: Path) -> LocalArtifact:
    state = FrozenRuntimeState.model_validate(state.model_dump(mode="python"))
    data = canonical_bytes(state.model_dump(mode="json"))
    artifact_root.mkdir(parents=True, exist_ok=True)
    path = artifact_root.absolute() / f"{state.content_hash}.json"
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError("immutable existing state bytes differ") from None
    return LocalArtifact(
        identity=state.schema_version,
        path=str(path),
        sha256=hashlib.sha256(data).hexdigest(),
        available_at=state.impact_calibration.calibrated_at,
    )


def load_runtime_state(artifact: LocalArtifact, *, cutoff: datetime) -> FrozenRuntimeState:
    artifact = LocalArtifact.model_validate(artifact.model_dump(mode="python"))
    if artifact.identity != "production-runtime-state-v1" or not Path(artifact.path).is_absolute():
        raise ValueError("invalid runtime state descriptor identity/path")
    if artifact.available_at > cutoff:
        raise ValueError("runtime state unavailable by cutoff")
    data = Path(artifact.path).read_bytes()
    if hashlib.sha256(data).hexdigest() != artifact.sha256:
        raise ValueError("runtime state byte hash mismatch")
    state = FrozenRuntimeState.model_validate_json(data)
    if (
        data != canonical_bytes(state.model_dump(mode="json"))
        or Path(artifact.path).name != f"{state.content_hash}.json"
        or artifact.available_at != state.impact_calibration.calibrated_at
        or state.fit_cutoff > cutoff
    ):
        raise ValueError("runtime state descriptor/canonical identity mismatch")
    lock = ModelLock.model_validate_json(_verified_bytes(state.model_lock, cutoff))
    _verified_bytes(state.catalogue, cutoff)
    if state.model_identity != ModelIdentity(
        event_model_version=f"{lock.event.model_id}@{lock.event.revision}",
        sentiment_model_version=f"{lock.sentiment.model_id}@{lock.sentiment.revision}",
        entity_linker_version=f"entity-matcher-v1:{state.catalogue.sha256}",
    ):
        raise ValueError("runtime state pinned model identity mismatch")
    return state
