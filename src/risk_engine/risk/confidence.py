"""Offline joint interpretation Confidence, independent of severity and support.

The minimum of two raw model scores is a versioned ranking candidate, not an
independence estimate, probability bound, or empirically optimal combination.
Temperature scaling learns against entity AND Event Class correctness. Synthetic
fixtures prove calculation behavior only; deployment requires real chronological
development labels and untouched evaluation. This module grants no stress eligibility.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictBool, model_validator
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

from risk_engine.domain import (
    ConfidenceTarget,
    DomainModel,
    EventClass,
    NonEmptyString,
    Probability,
    Sha256Hex,
)


class ConfidenceScoreDefinition(DomainModel):
    version: Literal["min-entity-class-raw-v1"] = "min-entity-class-raw-v1"
    event_model_version: NonEmptyString
    entity_linker_version: NonEmptyString


class RawConfidenceScore(DomainModel):
    definition: ConfidenceScoreDefinition
    entity_link_confidence: Probability
    classification_confidence: Probability

    @property
    def value(self) -> float:
        """Uncalibrated minimum, reproduced for each linked entity-event pair."""
        return min(self.entity_link_confidence, self.classification_confidence)


class CalibrationRow(DomainModel):
    row_id: NonEmptyString
    source_item_id: NonEmptyString
    event_id: NonEmptyString
    entity_id: NonEmptyString
    event_class: EventClass
    content_hash: Sha256Hex
    source_terms: NonEmptyString
    label_provenance: NonEmptyString
    source_available_at: AwareDatetime
    label_available_at: AwareDatetime
    split: Literal["development", "test", "final"]
    raw_score: RawConfidenceScore
    entity_correct: StrictBool
    event_class_correct: StrictBool

    @property
    def joint_correct(self) -> bool:
        return self.entity_correct and self.event_class_correct


class CalibrationEvidence(DomainModel):
    calibration_version: NonEmptyString
    evidence_kind: Literal["synthetic", "empirical"]
    snapshot_id: NonEmptyString
    snapshot_hash: Sha256Hex
    score_definition: ConfidenceScoreDefinition
    development_start: AwareDatetime
    development_end: AwareDatetime
    frozen_at: AwareDatetime
    rows: Annotated[tuple[CalibrationRow, ...], Field(min_length=2)]

    @model_validator(mode="after")
    def development_evidence_is_consistent(self) -> CalibrationEvidence:
        if not self.development_start <= self.development_end <= self.frozen_at:
            raise ValueError("development window must end by frozen_at")
        if len({row.row_id for row in self.rows}) != len(self.rows):
            raise ValueError("duplicate calibration row identity")
        pairs = {(row.source_item_id, row.event_id, row.entity_id) for row in self.rows}
        if len(pairs) != len(self.rows):
            raise ValueError("duplicate labeled entity-event interpretation")
        for row in self.rows:
            if row.split != "development":
                raise ValueError("fit requires development-only rows")
            if row.raw_score.definition != self.score_definition:
                raise ValueError("raw score definition differs from calibration evidence")
            if not self.development_start <= row.source_available_at <= self.development_end:
                raise ValueError("Source Item outside frozen development window")
            if not row.source_available_at <= row.label_available_at <= self.frozen_at:
                raise ValueError("labels must be available after source and by frozen_at")
        if len({row.joint_correct for row in self.rows}) != 2:
            raise ValueError("temperature fitting requires both joint outcome classes")
        if all(row.raw_score.value == 0.5 for row in self.rows):
            raise ValueError("temperature is unidentified for all-neutral raw scores")
        return self


class ConfidenceEstimate(DomainModel):
    probability: Probability
    evidence_kind: Literal["synthetic", "empirical"]
    target: Literal[ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS]
    calibration_version: NonEmptyString
    evidence_hash: Sha256Hex
    fitted_artifact_hash: Sha256Hex
    score_definition: ConfidenceScoreDefinition
    transform_version: Literal["binary-temperature-logit-clip1e-12-v1"]


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-12), 1 - 1e-12)
    return math.log(clipped) - math.log1p(-clipped)


def _evidence_hash(evidence: CalibrationEvidence) -> str:
    return hashlib.sha256(evidence.model_dump_json().encode()).hexdigest()


def _fitted_artifact_hash(
    *,
    target: ConfidenceTarget,
    transform_version: str,
    fit_version: str,
    temperature: float,
    evidence_hash: str,
) -> str:
    identity = {
        "temperature": temperature,
        "target": target.value,
        "fit_version": fit_version,
        "transform_version": transform_version,
        "evidence_hash": evidence_hash,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class ConfidenceCalibrator(DomainModel):
    """Frozen fitted state; JSON round trips perform no fitting or acquisition.

    v1 minimizes mean binary log loss over T in [0.05, 20], clipping raw endpoints
    to [1e-12, 1-1e-12] before logit/T. Bounded scalar fitting uses xatol=1e-10,
    maxiter=1000; endpoints and T=1 are compared, ties choose smaller T. These are
    declared project conventions, not conclusions about optimal production policy.
    """

    target: Literal[ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS]
    transform_version: Literal["binary-temperature-logit-clip1e-12-v1"]
    fit_version: Literal["bounded-logloss-T0.05-20-xatol1e-10-maxiter1000-v1"]
    temperature: Annotated[float, Field(ge=0.05, le=20)]
    evidence: CalibrationEvidence
    evidence_hash: Sha256Hex
    fitted_artifact_hash: Sha256Hex

    @model_validator(mode="after")
    def evidence_identity_matches(self) -> ConfidenceCalibrator:
        self._validate_fitted_identity()
        return self

    def _validate_fitted_identity(self) -> None:
        if tuple(sorted(self.evidence.rows, key=lambda row: row.row_id)) != self.evidence.rows:
            raise ValueError("calibration evidence must be in canonical row order")
        if _evidence_hash(self.evidence) != self.evidence_hash:
            raise ValueError("calibration evidence hash mismatch")
        expected_artifact_hash = _fitted_artifact_hash(
            target=self.target,
            transform_version=self.transform_version,
            fit_version=self.fit_version,
            temperature=self.temperature,
            evidence_hash=self.evidence_hash,
        )
        if expected_artifact_hash != self.fitted_artifact_hash:
            raise ValueError("fitted artifact hash mismatch")

    @classmethod
    def fit(cls, evidence: CalibrationEvidence) -> ConfidenceCalibrator:
        # Revalidate even model_copy-created records; frozen state cannot bless invalid rows.
        validated = CalibrationEvidence.model_validate(evidence.model_dump(mode="python"))
        ordered = validated.model_copy(
            update={"rows": tuple(sorted(validated.rows, key=lambda row: row.row_id))}
        )
        logits = tuple(_logit(row.raw_score.value) for row in ordered.rows)
        labels = tuple(int(row.joint_correct) for row in ordered.rows)

        def loss(temperature: float) -> float:
            terms = []
            for logit, label in zip(logits, labels, strict=True):
                scaled = logit / temperature
                terms.append(max(scaled, 0) - label * scaled + math.log1p(math.exp(-abs(scaled))))
            return math.fsum(terms) / len(terms)

        fitted = minimize_scalar(
            loss,
            bounds=(0.05, 20.0),
            method="bounded",
            options={"xatol": 1e-10, "maxiter": 1000},
        )
        if not fitted.success or not math.isfinite(float(fitted.x)):
            raise ValueError("temperature optimizer failed")
        temperature = min((0.05, 1.0, 20.0, float(fitted.x)), key=lambda t: (loss(t), t))
        target = ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS
        transform_version = "binary-temperature-logit-clip1e-12-v1"
        fit_version = "bounded-logloss-T0.05-20-xatol1e-10-maxiter1000-v1"
        evidence_hash = _evidence_hash(ordered)
        return cls(
            target=target,
            transform_version=transform_version,
            fit_version=fit_version,
            temperature=temperature,
            evidence=ordered,
            evidence_hash=evidence_hash,
            fitted_artifact_hash=_fitted_artifact_hash(
                target=target,
                transform_version=transform_version,
                fit_version=fit_version,
                temperature=temperature,
                evidence_hash=evidence_hash,
            ),
        )

    def transform(self, raw_score: RawConfidenceScore, *, as_of: datetime) -> ConfidenceEstimate:
        self._validate_fitted_identity()
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("scoring as_of must be timezone aware")
        if as_of < self.evidence.frozen_at:
            raise ValueError("scoring precedes calibrator frozen_at")
        score = RawConfidenceScore.model_validate(raw_score.model_dump(mode="python"))
        if score.definition != self.evidence.score_definition:
            raise ValueError("raw score definition does not match fitted calibrator")
        scaled = _logit(score.value) / self.temperature
        if scaled >= 0:
            probability = 1 / (1 + math.exp(-scaled))
        else:
            weight = math.exp(scaled)
            probability = weight / (1 + weight)
        return ConfidenceEstimate(
            probability=probability,
            target=self.target,
            evidence_kind=self.evidence.evidence_kind,
            calibration_version=self.evidence.calibration_version,
            evidence_hash=self.evidence_hash,
            fitted_artifact_hash=self.fitted_artifact_hash,
            score_definition=score.definition,
            transform_version=self.transform_version,
        )


class ConfidenceObservation(DomainModel):
    probability: Probability
    entity_correct: StrictBool
    event_class_correct: StrictBool

    @property
    def joint_correct(self) -> bool:
        return self.entity_correct and self.event_class_correct


class ReliabilityBin(DomainModel):
    lower: Probability
    upper: Probability
    count: Annotated[int, Field(ge=0)]
    mean_confidence: Probability | None
    joint_accuracy: Probability | None


class ConfidenceMetrics(DomainModel):
    target: Literal[ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS]
    brier_version: Literal["binary-mean-squared-joint-error-v1"]
    bin_version: Literal["equal-width-left-closed-last-right-closed-v1"]
    observations: tuple[ConfidenceObservation, ...]
    brier_score: Probability
    bins: tuple[ReliabilityBin, ...]


def confidence_metrics(
    observations: tuple[ConfidenceObservation, ...], *, bin_count: int
) -> ConfidenceMetrics:
    """Report actual labeled probabilities; no training, held-out claims, or silent empty bins.

    Binary Brier uses [0,1] scaling. Bins are [i/n,(i+1)/n), except the last
    includes 1. Empty bins carry count zero and no mean or observed accuracy.
    """
    if not observations:
        raise ValueError("Confidence metrics require labeled observations")
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count < 1:
        raise ValueError("bin_count must be a positive integer")
    rows = tuple(ConfidenceObservation.model_validate(row.model_dump()) for row in observations)
    bins = []
    for index in range(bin_count):
        lower, upper = index / bin_count, (index + 1) / bin_count
        members = tuple(
            row
            for row in rows
            if lower <= row.probability < upper or (index == bin_count - 1 and row.probability == 1)
        )
        bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=len(members),
                mean_confidence=math.fsum(row.probability for row in members) / len(members)
                if members
                else None,
                joint_accuracy=sum(row.joint_correct for row in members) / len(members)
                if members
                else None,
            )
        )
    return ConfidenceMetrics(
        target=ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS,
        brier_version="binary-mean-squared-joint-error-v1",
        bin_version="equal-width-left-closed-last-right-closed-v1",
        observations=rows,
        brier_score=math.fsum((row.probability - int(row.joint_correct)) ** 2 for row in rows)
        / len(rows),
        bins=tuple(bins),
    )
