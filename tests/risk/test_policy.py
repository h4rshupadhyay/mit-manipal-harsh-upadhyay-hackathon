"""Synthetic policy fixtures exercise gates, not empirical policy selection."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext

import pytest
from pydantic import ValidationError

from risk_engine.config import AppConfig, PolicyConfig
from risk_engine.domain import (
    ConfidenceTarget,
    EntityLink,
    EventClass,
    ImpactEstimate,
    PortfolioMateriality,
    ProvenanceMethod,
    RiskSignal,
    SignalFlag,
    VersionMetadata,
)
from risk_engine.risk.policy import (
    CandidateTriggerEvaluation,
    ManualOverride,
    TriggerCriteria,
    TriggerDecision,
    TriggerPolicy,
    evaluate_candidate_trigger,
)

TIME = datetime(2025, 1, 4, tzinfo=timezone.utc)


def criteria() -> TriggerCriteria:
    return TriggerCriteria(
        confidence_threshold=Decimal("0.85"),
        economic_floor=Decimal("100.00"),
        materiality_tolerance=Decimal("0.01"),
        currency="USD",
        confidence_target=ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS,
    )


def test_candidate_trigger_has_only_substantive_gates() -> None:
    result = evaluate_candidate_trigger(signal(), materiality("99.99"), criteria())
    assert tuple(g.name for g in result.gates) == (
        "entity_identity",
        "confidence_binding",
        "confidence_threshold",
        "impact_threshold",
        "supported_risk",
        "materiality_currency",
        "economic_floor",
    )
    assert result.would_trigger == all(g.passed for g in result.gates)
    assert result.would_trigger
    assert not evaluate_candidate_trigger(signal(), materiality("99.989"), criteria()).would_trigger
    assert not evaluate_candidate_trigger(
        signal(), materiality(currency="INR"), criteria()
    ).would_trigger
    for item in (
        signal().model_copy(
            update={"entity": signal().entity.model_copy(update={"entity_id": "unknown:bank"})}
        ),
        signal().model_copy(update={"flags": (SignalFlag.UNSUPPORTED_EXPOSURE,)}),
        signal().model_copy(
            update={"impact": signal().impact.model_copy(update={"impact_score": 7})}
        ),
    ):
        assert not evaluate_candidate_trigger(item, materiality(), criteria()).would_trigger
    with pytest.raises(ValidationError):
        CandidateTriggerEvaluation(gates=result.gates[:-1], would_trigger=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence_threshold", Decimal("0")),
        ("confidence_threshold", Decimal("NaN")),
        ("economic_floor", Decimal("Infinity")),
        ("economic_floor", Decimal("0")),
        ("materiality_tolerance", Decimal("100")),
        ("materiality_tolerance", Decimal("-1")),
        ("confidence_target", "other"),
    ],
)
def test_candidate_rejects_invalid_copied_criteria(field: str, value: object) -> None:
    invalid = criteria().model_copy(update={field: value})
    with pytest.raises(ValidationError):
        evaluate_candidate_trigger(signal(), materiality(), invalid)


def test_existing_decision_bytes_are_stable() -> None:
    examples = (
        (
            policy().evaluate(signal(), materiality()),
            "8c94399d1ae5425d3bc5611ecd9a107edfccff48829da4f502e99f7e90e382f1",
        ),
        (
            TriggerPolicy(PolicyConfig(), as_of=TIME).evaluate(signal(), materiality()),
            "5cd7250149ecccb7c09cc5cc4a562edf9541c6b973426df05741a84834aa83ae",
        ),
        (
            TriggerPolicy(PolicyConfig(), as_of=TIME).evaluate(
                signal(),
                materiality(),
                manual_override=ManualOverride(analyst_id="analyst-1", reason="Run sensitivity"),
            ),
            "5ec9986ddea63a3c9b32b4cccef57a1d0ec30f54f4f4e02d445700ca3365c66e",
        ),
    )
    for decision, expected in examples:
        assert hashlib.sha256(decision.model_dump_json().encode()).hexdigest() == expected


def selected_payload() -> dict:
    return {
        "version": "policy-fixture-v1",
        "validation_status": "chronologically_validated",
        "evidence_kind": "empirical",
        "selection_protocol": "nested_chronological_development",
        "development_start": TIME - timedelta(days=3),
        "development_end": TIME - timedelta(days=2),
        "frozen_at": TIME - timedelta(days=1),
        "validation_evidence": "fixture reference only; not real empirical evidence",
        "evidence_hash": "a" * 64,
        "snapshot_id": "fixture-snapshot-v1",
        "source_terms": "project-authored synthetic fixture",
        "confidence_target": ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS,
        "calibration_version": "confidence-fixture-v1",
        "confidence_threshold": "0.85",
        "economic_floor": "100.00",
        "materiality_tolerance": "0.01",
        "currency": "USD",
    }


def policy(*, as_of: datetime = TIME) -> TriggerPolicy:
    return TriggerPolicy(PolicyConfig.model_validate({"selected": selected_payload()}), as_of=as_of)


def signal() -> RiskSignal:
    return RiskSignal(
        signal_id="signal-1",
        source_item_id="source-1",
        entity=EntityLink(
            entity_id="bank-1",
            canonical_name="Bank",
            confidence=0.95,
            evidence="Bank",
            ambiguous=False,
        ),
        sentiment=-0.5,
        event_class=EventClass.CREDIT_DEFAULT,
        impact=ImpactEstimate(
            impact_score=8,
            expected_reference_loss=Decimal("500"),
            loss_lower_bound=Decimal("400"),
            loss_upper_bound=Decimal("600"),
            loss_currency="USD",
            analogue_count=1,
            analogue_ids=("event-1",),
            backoff_level="exact",
            method=ProvenanceMethod.EMPIRICAL,
            calibration_version="impact-v1",
            reference_basket_version="basket-v1",
        ),
        confidence=0.85,
        confidence_target=ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS,
        rationale="Credit default",
        evidence=("Bank",),
        flags=(),
        versions=VersionMetadata(
            schema_version="1",
            model_version="model-v1",
            calibration_version="confidence-fixture-v1",
            snapshot_version="snap-1",
        ),
        action_priority=7.5,
    )


def materiality(loss: str = "100", currency: str = "USD") -> PortfolioMateriality:
    return PortfolioMateriality(
        absolute_loss=Decimal(loss), percentage_loss=0.01, currency=currency
    )


def test_all_gates_pass_at_inclusive_thresholds_and_decision_round_trips() -> None:
    result = policy().evaluate(signal(), materiality())
    assert result.automatic_trigger is True
    assert result.triggered is True
    assert result.manual_override is None
    assert {gate.name for gate in result.gates} == {
        "policy_selection",
        "policy_chronology",
        "entity_identity",
        "confidence_binding",
        "confidence_threshold",
        "impact_threshold",
        "supported_risk",
        "materiality_currency",
        "economic_floor",
    }
    assert all(gate.passed and gate.reason for gate in result.gates)
    assert result.as_of == TIME
    assert result.policy_config.selected is not None
    assert result.policy_config.selected.evidence_hash == "a" * 64
    assert result.impact_score == 8
    assert result.confidence == 0.85
    assert result.portfolio_materiality == materiality()
    assert result.action_priority == 7.5
    assert TriggerDecision.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(("score", "expected"), [(7, False), (8, True), (10, True)])
def test_impact_gate_is_requirement_driven(score: int, expected: bool) -> None:
    item = signal().model_copy(
        update={"impact": signal().impact.model_copy(update={"impact_score": score})}
    )
    assert policy().evaluate(item, materiality()).automatic_trigger is expected


@pytest.mark.parametrize(
    ("probability", "expected"), [(0.849999, False), (0.85, True), (1.0, True)]
)
def test_injected_confidence_threshold(probability: float, expected: bool) -> None:
    item = signal().model_copy(update={"confidence": probability})
    assert policy().evaluate(item, materiality()).automatic_trigger is expected


@pytest.mark.parametrize(
    ("loss", "expected"),
    [("0", False), ("0.01", False), ("99.989", False), ("99.99", True), ("100", True)],
)
def test_decimal_floor_uses_explicit_absolute_tolerance(loss: str, expected: bool) -> None:
    assert policy().evaluate(signal(), materiality(loss)).automatic_trigger is expected


@pytest.mark.parametrize("damage", ["unknown", "ambiguous_prefix", "declared", "flag"])
def test_unknown_or_ambiguous_entity_never_auto_triggers(damage: str) -> None:
    item = signal()
    if damage in {"unknown", "ambiguous_prefix"}:
        prefix = "UNKNOWN:" if damage == "unknown" else "ambiguous:"
        item = item.model_copy(
            update={"entity": item.entity.model_copy(update={"entity_id": prefix + "bank"})}
        )
    elif damage == "declared":
        item = item.model_copy(
            update={"entity": item.entity.model_copy(update={"ambiguous": True})}
        )
    else:
        item = item.model_copy(update={"flags": (SignalFlag.AMBIGUOUS_ENTITY,)})
    result = policy().evaluate(item, materiality())
    assert not result.automatic_trigger
    assert not next(g for g in result.gates if g.name == "entity_identity").passed


@pytest.mark.parametrize("damage", ["calibration", "unsupported", "currency", "future"])
def test_failed_gates_are_recorded_without_short_circuiting(damage: str) -> None:
    item, loss, engine = signal(), materiality(), policy()
    gate = {
        "calibration": "confidence_binding",
        "unsupported": "supported_risk",
        "currency": "materiality_currency",
        "future": "policy_chronology",
    }[damage]
    if damage == "calibration":
        item = item.model_copy(
            update={"versions": item.versions.model_copy(update={"calibration_version": "other"})}
        )
    elif damage == "unsupported":
        item = item.model_copy(update={"flags": (SignalFlag.UNSUPPORTED_EXPOSURE,)})
    elif damage == "currency":
        loss = materiality(currency="INR")
    else:
        engine = policy(as_of=TIME - timedelta(days=2))
    result = engine.evaluate(item, loss)
    assert not result.automatic_trigger
    assert len(result.gates) == 9
    assert not next(g for g in result.gates if g.name == gate).passed
    assert all(g.reason for g in result.gates)


def test_default_abstains_and_manual_override_does_not_reclassify_automatic_gates() -> None:
    from pathlib import Path

    engine = TriggerPolicy(AppConfig.load(Path("config/default.toml")).policy, as_of=TIME)
    automatic = engine.evaluate(signal(), materiality())
    assert not automatic.triggered
    assert not next(g for g in automatic.gates if g.name == "policy_selection").passed
    override = ManualOverride(analyst_id="analyst-1", reason="Investigate unknown bank exposure")
    manual = engine.evaluate(signal(), materiality(), manual_override=override)
    assert manual.triggered
    assert not manual.automatic_trigger
    assert manual.manual_override == override
    assert manual.gates == automatic.gates


def test_manual_override_requires_identity_and_reason() -> None:
    with pytest.raises(ValidationError):
        ManualOverride(analyst_id="analyst-1", reason=" ")


def test_replay_cutoff_must_be_explicit_and_timezone_aware() -> None:
    with pytest.raises(ValidationError):
        TriggerPolicy(PolicyConfig(), as_of=datetime(2025, 1, 1))


@pytest.mark.parametrize("field", ["confidence", "absolute_loss", "percentage_loss"])
def test_unvalidated_negative_input_is_rejected_not_clamped(field: str) -> None:
    item, loss = signal(), materiality()
    if field == "confidence":
        item = item.model_copy(update={field: -0.1})
    else:
        loss = loss.model_copy(update={field: Decimal("-1") if field == "absolute_loss" else -1})
    with pytest.raises(ValidationError):
        policy().evaluate(item, loss)


def test_decision_retains_complete_gate_inputs_for_offline_audit() -> None:
    item = signal().model_copy(update={"flags": (SignalFlag.UNSUPPORTED_EXPOSURE,)})
    result = policy().evaluate(item, materiality())
    assert result.risk_signal == item
    assert result.risk_signal.impact.calibration_version == "impact-v1"
    assert result.risk_signal.entity.entity_id == "bank-1"


def test_manual_unknown_entity_override_preserves_failed_automatic_gate() -> None:
    item = signal().model_copy(
        update={"entity": signal().entity.model_copy(update={"entity_id": "unknown:bank"})}
    )
    result = policy().evaluate(
        item,
        materiality(),
        manual_override=ManualOverride(
            analyst_id="analyst-1", reason="Run hypothetical sensitivity for unidentified bank"
        ),
    )
    assert result.triggered
    assert not result.automatic_trigger
    assert not next(g for g in result.gates if g.name == "entity_identity").passed


def test_percentage_and_priority_do_not_substitute_for_absolute_economic_floor() -> None:
    item = signal().model_copy(update={"action_priority": 1000.0})
    loss = materiality("1").model_copy(update={"percentage_loss": 99.0})
    result = policy().evaluate(item, loss)
    assert not result.automatic_trigger
    assert result.impact_score == 8
    assert result.confidence == 0.85


def test_floor_boundary_is_independent_of_ambient_decimal_precision() -> None:
    with localcontext() as context:
        context.prec = 2
        assert policy().evaluate(signal(), materiality("99.99")).automatic_trigger
        assert not policy().evaluate(signal(), materiality("99.989")).automatic_trigger


def test_floor_boundary_is_independent_of_ambient_decimal_exponent_limits() -> None:
    with localcontext() as context:
        context.Emax = 0
        assert policy().evaluate(signal(), materiality("99.99")).automatic_trigger
        assert not policy().evaluate(signal(), materiality("99.989")).automatic_trigger


@pytest.mark.parametrize(
    "damage",
    [
        "missing_gates",
        "partial_gates",
        "duplicate_gate",
        "gate_order",
        "gate_result",
        "gate_reason",
        "unselected_policy",
        "future_policy",
        "ambiguous_input",
        "calibration_input",
        "materiality_input",
        "currency_input",
        "signal_id",
        "source_item_id",
        "signal_versions",
        "impact_score",
        "confidence",
        "action_priority",
        "automatic_status",
        "combined_status",
    ],
)
def test_restored_decision_rejects_corrupted_gate_evidence_or_summaries(damage: str) -> None:
    payload = json.loads(policy().evaluate(signal(), materiality()).model_dump_json())
    if damage == "missing_gates":
        payload["gates"] = []
    elif damage == "partial_gates":
        payload["gates"].pop()
    elif damage == "duplicate_gate":
        payload["gates"][-1] = payload["gates"][0]
    elif damage == "gate_order":
        payload["gates"].reverse()
    elif damage == "gate_result":
        payload["gates"][0]["passed"] = False
    elif damage == "gate_reason":
        payload["gates"][0]["reason"] = "Automatic policy is unselected"
    elif damage == "unselected_policy":
        payload["policy_config"]["selected"] = None
    elif damage == "future_policy":
        payload["as_of"] = "2025-01-02T00:00:00Z"
    elif damage == "ambiguous_input":
        payload["risk_signal"]["entity"]["ambiguous"] = True
    elif damage == "calibration_input":
        payload["risk_signal"]["versions"]["calibration_version"] = "other"
        payload["signal_versions"]["calibration_version"] = "other"
    elif damage == "materiality_input":
        payload["portfolio_materiality"]["absolute_loss"] = "1"
    elif damage == "currency_input":
        payload["portfolio_materiality"]["currency"] = "INR"
    elif damage in {"signal_id", "source_item_id"}:
        payload[damage] = "other"
    elif damage == "signal_versions":
        payload["signal_versions"]["snapshot_version"] = "other"
    elif damage == "impact_score":
        payload["impact_score"] = 1
    elif damage == "confidence":
        payload["confidence"] = 0.1
    elif damage == "action_priority":
        payload["action_priority"] = None
    elif damage == "automatic_status":
        payload["automatic_trigger"] = False
    else:
        payload["triggered"] = False
    with pytest.raises(ValidationError):
        TriggerDecision.model_validate_json(json.dumps(payload))


def test_restored_decision_rejects_fabricated_automatic_and_manual_statuses() -> None:
    result = TriggerPolicy(PolicyConfig(), as_of=TIME).evaluate(
        signal(),
        materiality(),
        manual_override=ManualOverride(analyst_id="analyst-1", reason="Run sensitivity"),
    )
    assert TriggerDecision.model_validate_json(result.model_dump_json()) == result
    payload = json.loads(result.model_dump_json())
    payload["automatic_trigger"] = True
    with pytest.raises(ValidationError):
        TriggerDecision.model_validate_json(json.dumps(payload))
    payload["automatic_trigger"] = False
    payload["manual_override"] = None
    with pytest.raises(ValidationError):
        TriggerDecision.model_validate_json(json.dumps(payload))


def test_restored_failed_gate_rejects_false_success_reason() -> None:
    result = TriggerPolicy(PolicyConfig(), as_of=TIME).evaluate(signal(), materiality())
    assert TriggerDecision.model_validate_json(result.model_dump_json()) == result
    payload = json.loads(result.model_dump_json())
    payload["gates"][0]["reason"] = (
        "Empirical chronological development selection is declared with evidence metadata"
    )
    with pytest.raises(ValidationError):
        TriggerDecision.model_validate_json(json.dumps(payload))
