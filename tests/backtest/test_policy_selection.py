"""Held-out policy scoring with project-authored contract simulations."""

import json
from datetime import timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from risk_engine.backtest import policy_selection
from risk_engine.backtest.module import (
    CandidateSpec,
    FittedManifest,
    ImpactFitIdentity,
    ObservedOutcome,
    RealizedValuation,
    content_hash,
    decode_impact_audit,
)
from risk_engine.backtest.policy_selection import (
    PolicyExclusion,
    PolicyObservation,
    PolicySelectionAudit,
)
from risk_engine.backtest.runtime_inputs import PolicySelectionSpec
from risk_engine.backtest.splits import FoldGroup, InnerFold
from risk_engine.domain import (
    ConfidenceTarget,
    FactorShock,
    PortfolioMateriality,
    RiskSignal,
    ShockType,
    ShockUnit,
    StressScenario,
    VersionMetadata,
)
from risk_engine.impact.module import ImpactEstimator
from risk_engine.risk.confidence import ConfidenceEstimate, ConfidenceScoreDefinition
from risk_engine.risk.module import (
    CalibrationManifest,
    ConfidenceCalibrationIdentity,
    ImpactCalibrationIdentity,
    ModelIdentity,
)
from risk_engine.stress.module import StressEngine
from tests.backtest.runtime_fixtures import TERMS, artifact, seal, simulated_case
from tests.impact.test_analogues import AS_OF, event
from tests.impact.test_impact_score import calibration, market, repository
from tests.risk.test_policy import signal as base_signal


def _spec(**changes) -> PolicySelectionSpec:
    payload = dict(
        schema_version="production-policy-selection-spec-v1",
        version="contract-simulation-v1",
        confidence_thresholds=(Decimal("0.5"), Decimal("0.8")),
        economic_floors=(Decimal("50"), Decimal("100")),
        materiality_tolerance=Decimal("0"),
        currency="USD",
        train_groups=10,
        validation_groups=4,
        minimum_folds=1,
        minimum_cases=4,
        minimum_positive_cases=1,
        minimum_negative_cases=1,
        alert_budget=4,
        objective="material-event-f1-v1",
        tie_rule="fewest-alerts-highest-confidence-highest-floor-v1",
    )
    payload.update(changes)
    return PolicySelectionSpec.model_validate(payload)


def _sample(
    tmp_path,
    *,
    losses=(120, 120, 40, 40),
    confidences=(0.9, 0.6, 0.9, 0.6),
    labels=(True, False, True, False),
):
    # Independently authored contract records exercise empirical-schema binding;
    # no synthetic dataset or fitted record is relabeled as empirical evidence.
    impact = ImpactEstimator(repository(80), calibration(source_terms=TERMS[0]), market()).estimate(
        event(), AS_OF
    )
    assert impact.impact_score == 8
    impact_audit = decode_impact_audit(impact.calibration_version)
    train_groups = tuple(
        FoldGroup(
            cluster_id=loss.event_id,
            event_time=AS_OF - timedelta(days=90 - i),
            source_item_ids=(f"training-source-{i:02}",),
        )
        for i, loss in enumerate(impact_audit.training_losses)
    )
    fit_time = AS_OF - timedelta(days=1)
    descriptor_path = tmp_path / "contract-state.json"
    descriptor_path.write_text(TERMS[0])
    descriptor = artifact(descriptor_path, "contract-state", available_at=fit_time)
    confidence = ConfidenceCalibrationIdentity(
        target=ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS,
        calibration_version="contract-confidence-v1",
        evidence_kind="empirical",
        calibration_evidence_snapshot_id="contract-simulation",
        calibration_evidence_snapshot_hash="a" * 64,
        development_start=train_groups[0].event_time,
        development_end=fit_time - timedelta(hours=1),
        frozen_at=fit_time,
        score_definition=ConfidenceScoreDefinition(
            event_model_version="contract-event-v1",
            entity_linker_version="contract-link-v1",
        ),
        fit_version="contract-fit-v1",
        transform_version="binary-temperature-logit-clip1e-12-v1",
        temperature=1,
        evidence_hash="d" * 64,
        fitted_artifact_hash="e" * 64,
        source_terms=TERMS,
    )
    candidate = CandidateSpec(
        candidate_id="contract-candidate",
        version="v1",
        parameters={
            "runtime_definition_sha256": "b" * 64,
            "event_window": "session-zero",
            "confidence_fit": "binary-temperature-logit-clip1e-12-v1",
            "cutpoint_rule": "nearest-rank-lower-ties",
            "scenario_choice": "nearest-median-reference-loss-event-id-v1",
            "policy_mode": "held-out-material-event-f1-v1",
        },
        complexity_dimensions=("rules",),
        complexity=(1,),
        event_window_days=1,
        basket_convention="contract-basket",
        matching_convention="contract-matching",
    )
    manifest = seal(
        FittedManifest,
        dict(
            schema_version="backtest-fit-v1",
            candidate=candidate,
            configuration_hash="f" * 64,
            dataset_hash="a" * 64,
            evidence_kind="empirical",
            training_groups=train_groups,
            training_hash=content_hash(train_groups),
            maximum_evidence_available_at=fit_time,
            fit_cutoff=fit_time,
            model_identity=ModelIdentity(
                event_model_version="contract-event-v1",
                sentiment_model_version="contract-sentiment-v1",
                entity_linker_version="contract-link-v1",
            ),
            model_lock=descriptor,
            artifacts=(descriptor,),
            confidence=confidence,
            confidence_training_source_ids=tuple(
                sorted(s for g in train_groups for s in g.source_item_ids)
            ),
            impact=ImpactFitIdentity(
                evidence_kind="empirical",
                calibration_version="impact-v1",
                calibration_hash=impact_audit.calibration_sha256,
                reference_basket_version=impact.reference_basket_version,
                reference_basket_hash=impact_audit.calibration.reference_basket_sha256,
                matching_version=impact_audit.matching_version,
                training_event_ids=tuple(g.cluster_id for g in train_groups),
                cutpoints=impact_audit.cutpoints,
                currency="USD",
                horizon_days=1,
                quantile_convention="nearest-rank-lower-ties",
                frozen_at=impact_audit.calibration.calibrated_at,
            ),
            source_terms=TERMS,
            policy=None,
            policy_absence_reason="contract subfit without policy",
        ),
        "manifest_hash",
    )
    entries = tuple(simulated_case(i + 10) for i in range(len(losses)))
    chosen_cases = tuple(entry[0] for entry in entries)
    chosen_outcomes = tuple(entry[1] for entry in entries)
    groups = tuple(
        FoldGroup(
            cluster_id=c.cluster.cluster_id,
            event_time=AS_OF + timedelta(days=i),
            source_item_ids=(c.cluster.items[0].source_item_id,),
        )
        for i, c in enumerate(chosen_cases)
    )
    fold = InnerFold(fold_id="policy-inner-0001", train=train_groups, embargo=(), validation=groups)
    observations = []
    for i, (case, original, amount, probability, label) in enumerate(
        zip(chosen_cases, chosen_outcomes, losses, confidences, labels, strict=True)
    ):
        observed_payload = original.model_dump(exclude={"evidence_hash"})
        observed_payload.update(
            label_available_at=AS_OF + timedelta(days=i, hours=3),
            outcome_available_at=AS_OF + timedelta(days=i + 1),
            material_event=label,
            material_event_absence_reason=None,
            valuation=RealizedValuation(
                pnl=-Decimal(amount),
                currency="USD",
                horizon_days=1,
                comparison_scope="complete-equity-contract-simulation",
                baseline_market_hash=content_hash(case.market),
                position_ids=("equity",),
                gross_values=(("equity", Decimal("1000")),),
            ),
            valuation_absence_reason=None,
        )
        observed = seal(ObservedOutcome, observed_payload, "evidence_hash")
        signal_calibration = CalibrationManifest(
            confidence=ConfidenceEstimate(
                probability=probability,
                evidence_kind="empirical",
                target=ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS,
                calibration_version=manifest.confidence.calibration_version,
                evidence_hash=manifest.confidence.evidence_hash,
                fitted_artifact_hash=manifest.confidence.fitted_artifact_hash,
                score_definition=manifest.confidence.score_definition,
                transform_version=manifest.confidence.transform_version,
            ),
            confidence_calibration=manifest.confidence,
            impact=ImpactCalibrationIdentity(
                calibration_version=impact.calibration_version,
                reference_basket_version=impact.reference_basket_version,
            ),
        )
        item = base_signal()
        chosen = RiskSignal.model_validate(
            item.model_dump()
            | {
                "signal_id": f"signal-{i}",
                "source_item_id": case.cluster.items[0].source_item_id,
                "confidence": probability,
                "impact": impact,
                "versions": VersionMetadata.model_validate(
                    item.versions.model_dump()
                    | {
                        "calibration_version": json.dumps(
                            signal_calibration.model_dump(mode="json"),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                ),
            }
        )
        scenario = StressScenario(
            scenario_id=f"scenario-{i}",
            risk_signal_id=chosen.signal_id,
            shocks=(
                FactorShock(
                    factor_id="EQUITY-US",
                    shock_type=ShockType.RELATIVE,
                    value=-amount / 1000,
                    unit=ShockUnit.DECIMAL,
                    horizon_days=1,
                ),
            ),
            method=chosen.impact.method,
            reference_event_ids=chosen.impact.analogue_ids,
            calibration_version=chosen.impact.calibration_version,
        )
        stress = StressEngine().run(case.portfolio, case.market, scenario)
        assert stress.absolute_loss == Decimal(amount)
        materiality = PortfolioMateriality(
            absolute_loss=Decimal(amount),
            percentage_loss=stress.percentage_loss,
            currency="USD",
        )
        observations.append(
            PolicyObservation(
                fold_id=fold.fold_id,
                case_id=case.case_id,
                cluster_id=case.cluster.cluster_id,
                as_of=AS_OF + timedelta(days=i, hours=2),
                subfit_manifest=manifest,
                signal=chosen,
                stress=stress,
                materiality=materiality,
                material_event=label,
                outcome_hash=content_hash(observed),
                label_available_at=observed.label_available_at,
                outcome_available_at=observed.outcome_available_at,
                source_terms=TERMS,
                observed=observed,
                market_hash=content_hash(case.market),
                portfolio_id=case.portfolio.portfolio_id,
                portfolio_gross_values=(("equity", Decimal("1000")),),
            )
        )
    return tuple(observations), (fold,), _spec(), manifest.configuration_hash


def _select(rows, folds, spec, configuration_hash, exclusions=()):
    return policy_selection.select_policy(
        rows,
        exclusions,
        spec,
        folds=folds,
        parent_training_hash="a" * 64,
        runtime_definition_hash="b" * 64,
        configuration_hash=configuration_hash,
    )


def test_policy_grid_uses_same_eligible_population_and_exact_f1(tmp_path) -> None:
    rows, folds, spec, config_hash = _sample(tmp_path)
    audit = _select(rows, folds, spec, config_hash)
    assert tuple(
        (r.true_positives, r.false_positives, r.false_negatives, r.true_negatives)
        for r in audit.results
    ) == (
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 0, 1, 2),
        (1, 0, 1, 2),
    )
    assert tuple(r.f1 for r in audit.results) == (0.5, 0.5, 2 / 3, 2 / 3)
    assert audit.selected == audit.results[-1].criteria
    assert PolicySelectionAudit.model_validate_json(audit.model_dump_json()) == audit


def test_policy_ties_use_alerts_confidence_then_floor(tmp_path) -> None:
    rows, folds, spec, config_hash = _sample(tmp_path)
    result = _select(rows, folds, spec, config_hash)
    assert result.selected == result.results[-1].criteria
    assert result.results[-1].alert_count < result.results[0].alert_count


def test_policy_budget_zero_cannot_select_all_suppressed(tmp_path) -> None:
    rows, folds, spec, config_hash = _sample(tmp_path)
    audit = _select(rows, folds, spec.model_copy(update={"alert_budget": 0}), config_hash)
    assert audit.selected is None
    assert audit.absence_reason == "all candidates exceed declared alert budget"


def test_policy_insufficient_population_is_unselected(tmp_path) -> None:
    rows, folds, spec, config_hash = _sample(tmp_path)
    for change, expected in (
        ({"minimum_cases": 5}, "insufficient eligible cases"),
        ({"minimum_folds": 2}, "insufficient eligible folds"),
        ({"minimum_positive_cases": 3}, "insufficient positive cases"),
        ({"minimum_negative_cases": 3}, "insufficient negative cases"),
    ):
        audit = _select(rows, folds, spec.model_copy(update=change), config_hash)
        assert audit.selected is None
        assert audit.absence_reason == expected


def test_policy_audit_revalidation_recomputes_counts_and_winner(tmp_path) -> None:
    rows, folds, spec, config_hash = _sample(tmp_path)
    audit = _select(rows, folds, spec, config_hash)
    for change in (
        {
            "results": (
                audit.results[0].model_copy(
                    update={
                        "true_positives": 2,
                        "false_negatives": 0,
                        "alert_count": 3,
                    }
                ),
                *audit.results[1:],
            )
        },
        {"results": (audit.results[0].model_copy(update={"f1": 0.9}), *audit.results[1:])},
        {"selected": audit.results[0].criteria},
    ):
        payload = audit.model_dump(exclude={"content_hash"}) | change
        payload["content_hash"] = content_hash(payload)
        with pytest.raises(ValidationError):
            PolicySelectionAudit.model_validate(payload)


def test_policy_rejects_duplicate_scope_and_missing_coverage(tmp_path) -> None:
    rows, folds, spec, config_hash = _sample(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        _select((*rows, rows[0]), folds, spec, config_hash)
    with pytest.raises(ValueError, match="omits"):
        _select(rows[:-1], folds, spec, config_hash)
    exclusion = PolicyExclusion(
        fold_id=folds[0].fold_id,
        case_id=rows[0].case_id,
        cluster_id=rows[0].cluster_id,
        reason="missing label",
        source_terms=TERMS,
    )
    with pytest.raises(ValueError, match="contradict"):
        _select(rows, folds, spec, config_hash, (exclusion,))


@pytest.mark.parametrize(
    "field,value",
    [
        ("outcome_hash", "0" * 64),
        ("material_event", None),
        ("market_hash", "0" * 64),
        ("portfolio_gross_values", (("equity", Decimal("900")),)),
    ],
)
def test_policy_observation_rejects_unverified_outcomes_and_valuation(
    tmp_path, field, value
) -> None:
    rows, _, _, _ = _sample(tmp_path)
    with pytest.raises(ValidationError):
        PolicyObservation.model_validate(rows[0].model_dump() | {field: value})


def test_missing_label_is_explicit_exclusion_not_negative(tmp_path) -> None:
    rows, folds, spec, config_hash = _sample(tmp_path)
    excluded = rows[-1]
    omission = PolicyExclusion(
        fold_id=excluded.fold_id,
        case_id=excluded.case_id,
        cluster_id=excluded.cluster_id,
        reason="independent label unavailable",
        source_terms=TERMS,
    )
    audit = _select(rows[:-1], folds, spec, config_hash, (omission,))
    assert all(
        r.true_positives + r.false_positives + r.false_negatives + r.true_negatives == 3
        for r in audit.results
    )


def test_first_validation_as_of_is_valid_subfit_cutoff(tmp_path) -> None:
    rows, folds, spec, config_hash = _sample(tmp_path)
    payload = rows[0].subfit_manifest.model_dump(exclude={"manifest_hash"})
    payload["fit_cutoff"] = rows[0].as_of
    manifest = seal(FittedManifest, payload, "manifest_hash")
    later = tuple(row.model_copy(update={"subfit_manifest": manifest}) for row in rows)
    assert _select(later, folds, spec, config_hash).selected is not None


def test_outcome_chronology_rejects_pre_replay_availability(tmp_path) -> None:
    rows, _, _, _ = _sample(tmp_path)
    first = rows[0]
    outcome_payload = first.observed.model_dump(exclude={"evidence_hash"})
    outcome_payload["label_available_at"] = first.as_of - timedelta(hours=1)
    outcome_payload["outcome_available_at"] = first.as_of - timedelta(minutes=1)
    outcome = seal(ObservedOutcome, outcome_payload, "evidence_hash")
    tampered = first.model_dump() | {
        "observed": outcome,
        "outcome_hash": content_hash(outcome),
        "label_available_at": outcome.label_available_at,
        "outcome_available_at": outcome.outcome_available_at,
    }
    with pytest.raises(ValidationError, match="chronology"):
        PolicyObservation.model_validate(tampered)


def test_runtime_definition_hash_binds_observation_candidate(tmp_path) -> None:
    rows, folds, spec, config_hash = _sample(tmp_path)
    with pytest.raises(ValueError, match="runtime definition"):
        policy_selection.select_policy(
            rows,
            (),
            spec,
            folds=folds,
            parent_training_hash="a" * 64,
            runtime_definition_hash="c" * 64,
            configuration_hash=config_hash,
        )


@pytest.mark.parametrize(
    "field,change",
    [
        ("material_event", False),
        ("label_available_at", AS_OF),
        ("outcome_available_at", AS_OF),
        ("portfolio_id", "different-portfolio"),
        ("portfolio_gross_values", (("equity", Decimal("-1")),)),
        ("portfolio_gross_values", (("equity", Decimal("1000")), ("equity", Decimal("1000")))),
    ],
)
def test_observation_retained_summaries_match_independent_evidence(tmp_path, field, change):
    rows, _, _, _ = _sample(tmp_path)
    with pytest.raises(ValidationError):
        PolicyObservation.model_validate(rows[0].model_dump() | {field: change})


@pytest.mark.parametrize(
    "change",
    [
        {"position_ids": ()},
        {"position_ids": ("other",)},
        {"currency": "INR"},
        {"horizon_days": 2},
        {"gross_values": (("equity", Decimal("900")),)},
    ],
)
def test_observation_rejects_incomplete_or_mismatched_valuation(tmp_path, change):
    rows, _, _, _ = _sample(tmp_path)
    row = rows[0]
    payload = row.observed.model_dump(exclude={"evidence_hash"})
    payload["valuation"].update(change)
    outcome = seal(ObservedOutcome, payload, "evidence_hash")
    with pytest.raises(ValidationError):
        PolicyObservation.model_validate(
            row.model_dump()
            | {
                "observed": outcome,
                "outcome_hash": content_hash(outcome),
            }
        )


def test_observation_rejects_mismatched_calibration_and_materiality(tmp_path):
    rows, _, _, _ = _sample(tmp_path)
    row = rows[0]
    altered_signal = row.signal.model_copy(update={"confidence": 0.7})
    altered_materiality = row.materiality.model_copy(update={"absolute_loss": Decimal("1")})
    for change in ({"signal": altered_signal}, {"materiality": altered_materiality}):
        with pytest.raises(ValidationError):
            PolicyObservation.model_validate(row.model_dump() | change)


def test_audit_binds_exact_fold_train_candidate_dataset_and_configuration(tmp_path):
    rows, folds, spec, config_hash = _sample(tmp_path)
    for field, value in (("configuration_hash", "c" * 64), ("dataset_hash", "c" * 64)):
        payload = rows[0].subfit_manifest.model_dump(exclude={"manifest_hash"})
        payload[field] = value
        manifest = seal(FittedManifest, payload, "manifest_hash")
        with pytest.raises(ValueError):
            _select(
                (rows[0].model_copy(update={"subfit_manifest": manifest}), *rows[1:]),
                folds,
                spec,
                config_hash,
            )
    with pytest.raises(ValueError):
        _select(
            (rows[0].model_copy(update={"cluster_id": "outside"}), *rows[1:]),
            folds,
            spec,
            config_hash,
        )
    changed_group = folds[0].train[0].model_copy(update={"event_time": AS_OF - timedelta(days=100)})
    altered_fold = folds[0].model_copy(update={"train": (changed_group, *folds[0].train[1:])})
    with pytest.raises(ValueError):
        _select(rows, (altered_fold,), spec, config_hash)


def test_duplicate_source_membership_across_validation_folds_is_rejected(tmp_path):
    rows, folds, spec, config_hash = _sample(tmp_path)
    second = folds[0].model_copy(update={"fold_id": "second"})
    with pytest.raises(ValueError, match="duplicate"):
        _select(rows, (*folds, second), spec, config_hash)


def test_whole_fold_exclusion_has_empty_eligible_population(tmp_path):
    _, folds, spec, config_hash = _sample(tmp_path)
    exclusion = PolicyExclusion(
        fold_id=folds[0].fold_id,
        case_id=None,
        cluster_id=None,
        reason="insufficient calibration evidence",
        source_terms=TERMS,
    )
    audit = _select((), folds, spec, config_hash, (exclusion,))
    assert audit.selected is None
    assert all(r.f1 is None and r.alert_count == 0 for r in audit.results)
    assert audit.source_terms == TERMS
    empty = _select((), (), spec, config_hash)
    assert empty.source_terms == () and empty.selected is None


def test_budget_equality_and_zero_true_positives(tmp_path):
    rows, folds, spec, config_hash = _sample(tmp_path)
    audit = _select(rows, folds, spec.model_copy(update={"alert_budget": 1}), config_hash)
    assert audit.results[-1].feasible and audit.results[-1].alert_count == 1
    suppressed = _select(
        rows,
        folds,
        spec.model_copy(
            update={
                "economic_floors": (Decimal("200"),),
                "alert_budget": 0,
            }
        ),
        config_hash,
    )
    assert suppressed.selected is None
    assert all(r.f1 == 0 and r.reason == "no true positives" for r in suppressed.results)


def test_f1_tie_chooses_fewest_alerts_before_confidence_and_floor(tmp_path):
    rows, folds, spec, config_hash = _sample(
        tmp_path,
        losses=(120, 120, 120, 120),
        confidences=(0.9, 0.6, 0.6, 0.6),
        labels=(True, True, False, False),
    )
    audit = _select(rows, folds, spec, config_hash)
    assert tuple(r.f1 for r in audit.results) == (2 / 3,) * 4
    assert tuple(r.alert_count for r in audit.results) == (4, 4, 1, 1)
    assert audit.selected == audit.results[-1].criteria


def test_audit_rejects_rehashed_unchecked_nested_record_copies(tmp_path):
    rows, folds, spec, config_hash = _sample(tmp_path)
    audit = _select(rows, folds, spec, config_hash)
    bad_observed = rows[0].observed.model_copy(update={"actual_entity_id": "forged"})
    bad_row = rows[0].model_copy(
        update={
            "observed": bad_observed,
            "outcome_hash": content_hash(bad_observed),
        }
    )
    payload = audit.model_dump(exclude={"content_hash"}) | {
        "observations": (bad_row, *rows[1:]),
    }
    payload["content_hash"] = content_hash(payload)
    with pytest.raises(ValidationError):
        PolicySelectionAudit.model_validate(payload)


def test_policy_records_reject_coerced_f1_and_summary_booleans(tmp_path):
    rows, folds, spec, config_hash = _sample(tmp_path)
    audit = _select(rows, folds, spec, config_hash)
    from risk_engine.backtest.policy_selection import PolicyCandidateResult

    with pytest.raises(ValidationError):
        PolicyCandidateResult.model_validate(audit.results[0].model_dump() | {"f1": "0.5"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("currency", "INR"),
        ("frozen_at", AS_OF - timedelta(days=2)),
        ("calibration_version", "different-impact-version"),
    ],
)
def test_observation_verifies_complete_frozen_impact_identity(tmp_path, field, value):
    rows, _, _, _ = _sample(tmp_path)
    row = rows[0]
    payload = row.subfit_manifest.model_dump(exclude={"manifest_hash"})
    payload["impact"][field] = value
    manifest = seal(FittedManifest, payload, "manifest_hash")
    with pytest.raises(ValidationError, match="Impact"):
        PolicyObservation.model_validate(row.model_dump() | {"subfit_manifest": manifest})


@pytest.mark.parametrize(
    "grid_field,values",
    [
        (
            "confidence_thresholds",
            (
                Decimal("0.80000000000000000000000000001"),
                Decimal("0.80000000000000000000000000002"),
            ),
        ),
        (
            "economic_floors",
            (
                Decimal("100.000000000000000000000000001"),
                Decimal("100.000000000000000000000000002"),
            ),
        ),
    ],
)
def test_policy_tie_order_preserves_all_decimal_digits(tmp_path, grid_field, values):
    rows, folds, spec, config_hash = _sample(tmp_path)
    audit = _select(rows, folds, spec.model_copy(update={grid_field: values}), config_hash)
    assert audit.selected is not None
    selected_field = (
        "confidence_threshold" if grid_field == "confidence_thresholds" else "economic_floor"
    )
    assert getattr(audit.selected, selected_field) == values[-1]
    assert PolicySelectionAudit.model_validate_json(audit.model_dump_json()) == audit


@pytest.mark.parametrize("precision", [1, 2, 28, 60])
def test_policy_selection_and_observation_ignore_ambient_decimal_precision(tmp_path, precision):
    from decimal import localcontext

    rows, folds, spec, config_hash = _sample(tmp_path)
    spec = spec.model_copy(
        update={
            "confidence_thresholds": (Decimal("0.811"), Decimal("0.812")),
            "economic_floors": (Decimal("101"), Decimal("102")),
        }
    )
    expected = _select(rows, folds, spec, config_hash)
    with localcontext() as caller:
        caller.prec = precision
        actual = _select(rows, folds, spec, config_hash)
        assert actual == expected
        assert PolicySelectionAudit.model_validate_json(expected.model_dump_json()) == expected
        assert PolicyObservation.model_validate(rows[0].model_dump()) == rows[0]


@pytest.mark.parametrize(
    "impact_changes",
    [
        {"impact_score": 7},
        {"expected_reference_loss": Decimal("79"), "loss_lower_bound": Decimal("70")},
        {"loss_lower_bound": Decimal("70")},
        {"loss_upper_bound": Decimal("90")},
        {"analogue_ids": ("different-1", "different-2", "different-3")},
    ],
)
def test_policy_observation_rejects_impact_output_different_from_audit(tmp_path, impact_changes):
    rows, _, _, _ = _sample(tmp_path)
    payload = rows[0].model_dump()
    payload["signal"]["impact"].update(impact_changes)
    with pytest.raises(ValidationError, match="Impact"):
        PolicyObservation.model_validate(payload)


@pytest.mark.parametrize(
    "impact_changes",
    [
        {"impact_score": 7},
        {"expected_reference_loss": Decimal("79"), "loss_lower_bound": Decimal("70")},
    ],
)
def test_rehashed_policy_audit_rejects_inconsistent_nonalerting_impact(tmp_path, impact_changes):
    rows, folds, spec, config_hash = _sample(tmp_path)
    audit = _select(rows, folds, spec, config_hash)
    payload = audit.model_dump(exclude={"content_hash"})
    # This case is below every economic floor, so grid counts/winner stay unchanged.
    payload["observations"][2]["signal"]["impact"].update(impact_changes)
    payload["content_hash"] = content_hash(payload)
    with pytest.raises(ValidationError, match="Impact"):
        PolicySelectionAudit.model_validate(payload)


def _fold_training(count=8, times=None):
    from datetime import timezone

    from risk_engine.backtest.module import TrainingPartition
    from tests.backtest.runtime_fixtures import training_pair

    base, _ = training_pair()
    cases, outcomes = [], []
    for n in range(count):
        case, outcome, _ = simulated_case(n + 1)
        instant = times[n] if times else case.cluster.event_time + timedelta(days=n)
        item = case.cluster.items[0].model_copy(
            update={"published_at": instant, "retrieved_at": instant + timedelta(hours=1)}
        )
        cluster = case.cluster.model_copy(update={"event_time": instant, "items": (item,)})
        case = case.model_copy(update={"cluster": cluster, "as_of": instant + timedelta(hours=2)})
        cases.append(case)
        outcomes.append(outcome)
    # Membership canonical ordering uses UTC instant then cluster ID.
    cases.sort(key=lambda c: (c.cluster.event_time.astimezone(timezone.utc), c.cluster.cluster_id))
    groups = tuple(
        FoldGroup(
            cluster_id=c.cluster.cluster_id,
            event_time=c.cluster.event_time,
            source_item_ids=c.cluster.source_item_ids,
        )
        for c in cases
    )
    return TrainingPartition.model_validate(
        base.model_dump()
        | {
            "cases": tuple(cases),
            "outcomes": tuple(outcomes),
            "groups": groups,
            "membership_hash": content_hash(groups),
            "cutoff": base.cutoff + timedelta(days=count + 1),
        }
    )


def test_policy_folds_use_last_eligible_groups_and_disjoint_validation():
    training = _fold_training()
    spec = _spec(train_groups=2, validation_groups=2)
    folds = policy_selection.policy_folds(training, spec, embargo=timedelta(days=2))
    assert tuple(f.fold_id for f in folds) == ("policy-inner-0001", "policy-inner-0002")
    assert tuple((f.train, f.embargo, f.validation) for f in folds) == (
        (training.groups[:2], training.groups[2:3], training.groups[3:5]),
        (training.groups[2:4], training.groups[4:5], training.groups[5:7]),
    )
    assert not policy_selection.policy_folds(_fold_training(3), spec, embargo=timedelta(days=2))


def test_policy_folds_normalize_timezones_and_revalidate_naive_records():
    from datetime import timezone

    training = _fold_training()
    zones = [
        g.event_time.astimezone(timezone(timedelta(hours=5, minutes=30))) for g in training.groups
    ]
    shifted = _fold_training(times=zones)
    folds = policy_selection.policy_folds(
        shifted, _spec(train_groups=2, validation_groups=2), embargo=timedelta(days=2)
    )
    assert all(
        g.event_time.utcoffset() == timedelta(0) for f in folds for g in f.train + f.validation
    )
    bad = training.groups[0].model_copy(
        update={"event_time": training.groups[0].event_time.replace(tzinfo=None)}
    )
    with pytest.raises(ValueError):
        policy_selection.policy_folds(
            training.model_copy(update={"groups": (bad, *training.groups[1:])}),
            _spec(train_groups=2, validation_groups=2),
            embargo=timedelta(days=2),
        )
    with pytest.raises(ValueError):
        policy_selection.policy_folds(training, _spec(), embargo=timedelta(0))


def test_policy_folds_repeated_times_keep_canonical_order():
    from tests.impact.test_analogues import PAST

    training = _fold_training(times=[PAST + timedelta(days=n // 2) for n in range(8)])
    folds = policy_selection.policy_folds(
        training, _spec(train_groups=2, validation_groups=2), embargo=timedelta(days=1)
    )
    assert folds[0].train == training.groups[:2]
    assert folds[0].validation == training.groups[2:4]
    assert folds[1].train == training.groups[2:4]
    assert folds[1].validation == training.groups[4:6]
