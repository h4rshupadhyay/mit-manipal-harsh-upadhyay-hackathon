"""Synthetic calculation fixtures; these do not establish production calibration."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from risk_engine.domain import ConfidenceTarget, EventClass
from risk_engine.risk.confidence import (
    CalibrationEvidence,
    CalibrationRow,
    ConfidenceCalibrator,
    ConfidenceObservation,
    ConfidenceScoreDefinition,
    RawConfidenceScore,
    confidence_metrics,
)

TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)
DEFINITION = ConfidenceScoreDefinition(
    event_model_version="event-v1", entity_linker_version="link-v1"
)


def score(entity: float = 0.9, event: float = 0.95) -> RawConfidenceScore:
    return RawConfidenceScore(
        entity_link_confidence=entity, classification_confidence=event, definition=DEFINITION
    )


def evidence() -> CalibrationEvidence:
    rows = tuple(
        CalibrationRow(
            row_id=f"row-{index}",
            source_item_id=f"source-{index}",
            event_id=f"event-{index}",
            entity_id="entity-1",
            event_class=EventClass.CREDIT_DEFAULT,
            content_hash=f"{index:064x}",
            source_terms="project-authored synthetic fixture",
            label_provenance="synthetic joint correctness fixture",
            split="development",
            source_available_at=TIME,
            label_available_at=TIME + timedelta(days=1),
            raw_score=score(),
            entity_correct=index != 6,
            event_class_correct=index < 7,
        )
        for index in range(10)
    )
    return CalibrationEvidence(
        calibration_version="synthetic-calibration-v1",
        evidence_kind="synthetic",
        snapshot_id="synthetic-snapshot-v1",
        snapshot_hash="a" * 64,
        score_definition=DEFINITION,
        development_start=TIME,
        development_end=TIME + timedelta(days=1),
        frozen_at=TIME + timedelta(days=2),
        rows=rows,
    )


def test_fit_joint_outcomes_reduces_overconfidence_and_declares_target() -> None:
    # A class-only label or an unfitted temperature fails the hand-derived 6/10 outcome.
    data = evidence()
    calibrator = ConfidenceCalibrator.fit(data)
    result = calibrator.transform(score(), as_of=data.frozen_at)
    assert result.probability == pytest.approx(0.6, abs=1e-6)
    assert calibrator.temperature == pytest.approx(5.41902258, abs=1e-5)
    assert result.target == ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS
    assert result.calibration_version == data.calibration_version
    assert result.evidence_kind == "synthetic"
    # A different entity score must matter even with the same class score.
    assert calibrator.transform(score(0.1), as_of=data.frozen_at).probability < 0.5


@pytest.mark.parametrize("split", ["test", "final"])
def test_fit_rejects_non_development_rows(split: str) -> None:
    data = evidence().model_dump(mode="python")
    data["rows"][0]["split"] = split
    with pytest.raises(ValueError, match="development"):
        ConfidenceCalibrator.fit(CalibrationEvidence.model_validate(data))


@pytest.mark.parametrize("field", ["source_available_at", "label_available_at"])
def test_fit_rejects_future_evidence(field: str) -> None:
    data = evidence().model_dump(mode="python")
    data["rows"][0][field] = TIME + timedelta(days=3)
    with pytest.raises(ValueError):
        ConfidenceCalibrator.fit(CalibrationEvidence.model_validate(data))


def test_transform_rejects_future_calibrator_and_different_score_models() -> None:
    data = evidence()
    calibrator = ConfidenceCalibrator.fit(data)
    with pytest.raises(ValueError, match="frozen"):
        calibrator.transform(score(), as_of=TIME)
    changed = score().model_dump(mode="python")
    changed["definition"]["event_model_version"] = "event-v2"
    with pytest.raises(ValueError, match="definition"):
        calibrator.transform(RawConfidenceScore.model_validate(changed), as_of=data.frozen_at)
    with pytest.raises(ValueError, match="aware"):
        calibrator.transform(score(), as_of=datetime(2025, 1, 3))


def test_serialization_replays_identically_and_binds_evidence() -> None:
    data = evidence()
    original = ConfidenceCalibrator.fit(data)
    reordered = ConfidenceCalibrator.fit(
        data.model_copy(update={"rows": tuple(reversed(data.rows))})
    )
    assert reordered.model_dump_json() == original.model_dump_json()
    restored = ConfidenceCalibrator.model_validate_json(original.model_dump_json())
    assert restored.transform(score(), as_of=data.frozen_at) == original.transform(
        score(), as_of=data.frozen_at
    )
    changed = original.model_dump(mode="python")
    changed["evidence"]["rows"][0]["label_provenance"] = "different labels"
    with pytest.raises(ValueError, match="hash"):
        ConfidenceCalibrator.model_validate(changed)


def test_serialized_temperature_cannot_change_without_fitted_artifact_identity() -> None:
    calibrator = ConfidenceCalibrator.fit(evidence())
    serialized = calibrator.model_dump(mode="python")
    serialized["temperature"] = 1.0

    with pytest.raises(ValueError, match="fitted artifact hash"):
        ConfidenceCalibrator.model_validate(serialized)


def test_transform_rejects_copied_temperature_with_stale_fitted_identity() -> None:
    data = evidence()
    calibrator = ConfidenceCalibrator.fit(data).model_copy(update={"temperature": 1.0})

    with pytest.raises(ValueError, match="fitted artifact hash"):
        calibrator.transform(score(), as_of=data.frozen_at)


def test_confidence_estimates_identify_distinct_fitted_artifacts() -> None:
    first_evidence = evidence()
    second_data = first_evidence.model_dump(mode="python")
    second_data["calibration_version"] = "synthetic-calibration-v2"
    second_evidence = CalibrationEvidence.model_validate(second_data)

    first = ConfidenceCalibrator.fit(first_evidence).transform(
        score(), as_of=first_evidence.frozen_at
    )
    second = ConfidenceCalibrator.fit(second_evidence).transform(
        score(), as_of=second_evidence.frozen_at
    )

    assert first.evidence_hash != second.evidence_hash
    assert first.fitted_artifact_hash != second.fitted_artifact_hash


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf"), float("nan")])
def test_invalid_raw_probabilities_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        score(value)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_transform_boundary_probabilities_are_finite_and_bounded(value: float) -> None:
    data = evidence()
    probability = (
        ConfidenceCalibrator.fit(data)
        .transform(score(value, value), as_of=data.frozen_at)
        .probability
    )
    assert 0.0 <= probability <= 1.0
    if value == 0.5:
        assert probability == 0.5


def test_degenerate_and_duplicate_evidence_is_rejected() -> None:
    for change in ("one-class", "uninformative", "duplicate", "empty", "missing-label"):
        data = evidence().model_dump(mode="python")
        if change == "empty":
            data["rows"] = ()
        elif change == "duplicate":
            data["rows"] = (*data["rows"], data["rows"][0])
        elif change == "missing-label":
            del data["rows"][0]["entity_correct"]
        else:
            for row in data["rows"]:
                if change == "one-class":
                    row["entity_correct"] = row["event_class_correct"] = True
                else:
                    row["raw_score"]["entity_link_confidence"] = 0.5
        with pytest.raises(ValueError):
            ConfidenceCalibrator.fit(CalibrationEvidence.model_validate(data))


def test_renamed_duplicate_interpretation_cannot_reweight_fit() -> None:
    data = evidence().model_dump(mode="python")
    for field in ("source_item_id", "event_id", "entity_id"):
        data["rows"][1][field] = data["rows"][0][field]
    with pytest.raises(ValueError, match="duplicate"):
        ConfidenceCalibrator.fit(CalibrationEvidence.model_validate(data))


@pytest.mark.parametrize("field", ["event_model_version", "entity_linker_version", "version"])
def test_score_definition_changes_cannot_reuse_calibration(field: str) -> None:
    data = evidence()
    changed = score().model_dump(mode="python")
    changed["definition"][field] = "different-version"
    with pytest.raises(ValueError):
        ConfidenceCalibrator.fit(data).transform(
            RawConfidenceScore.model_validate(changed), as_of=data.frozen_at
        )


@pytest.mark.parametrize("field", ["entity_link_confidence", "classification_confidence"])
def test_copied_invalid_score_is_revalidated_at_transform(field: str) -> None:
    data = evidence()
    with pytest.raises(ValueError):
        ConfidenceCalibrator.fit(data).transform(
            score().model_copy(update={field: float("nan")}), as_of=data.frozen_at
        )


def test_temperature_boundary_solution_is_serializable_and_finite() -> None:
    data = evidence().model_dump(mode="python")
    for row in data["rows"]:
        row["raw_score"]["entity_link_confidence"] = 1.0 if row["event_class_correct"] else 0.0
        row["raw_score"]["classification_confidence"] = row["raw_score"]["entity_link_confidence"]
        row["entity_correct"] = True
    frozen = CalibrationEvidence.model_validate(data)
    calibrator = ConfidenceCalibrator.fit(frozen)
    assert calibrator.temperature == 0.05
    restored = ConfidenceCalibrator.model_validate_json(calibrator.model_dump_json())
    assert restored.transform(score(0, 0), as_of=frozen.frozen_at).probability == pytest.approx(
        1e-240, rel=1e-9, abs=0
    )
    assert restored.transform(score(1, 1), as_of=frozen.frozen_at).probability == 1


def test_metrics_report_hand_calculated_brier_and_boundary_bins() -> None:
    # Wrong joint labels, class-only labels, or shifting boundary bins breaks these literals.
    observations = tuple(
        ConfidenceObservation(probability=p, entity_correct=e, event_class_correct=c)
        for p, e, c in [
            (0, False, True),
            (0.25, True, False),
            (0.5, True, True),
            (0.75, False, True),
            (1, True, True),
        ]
    )
    report = confidence_metrics(observations, bin_count=4)
    assert report.brier_score == pytest.approx(0.175)
    assert [bin.count for bin in report.bins] == [1, 1, 1, 2]
    assert [bin.mean_confidence for bin in report.bins] == [0, 0.25, 0.5, 0.875]
    assert [bin.joint_accuracy for bin in report.bins] == [0, 0, 1, 0.5]
    empty_bins = confidence_metrics(observations, bin_count=10).bins
    assert empty_bins[1].count == 0
    assert empty_bins[1].mean_confidence is None
    assert empty_bins[1].joint_accuracy is None
    with pytest.raises(ValueError):
        confidence_metrics((), bin_count=4)
    with pytest.raises(ValueError):
        confidence_metrics(observations, bin_count=0)
