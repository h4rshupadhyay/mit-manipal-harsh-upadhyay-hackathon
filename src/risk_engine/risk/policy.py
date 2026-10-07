"""Recorded deterministic stress proposals from explicitly frozen policy inputs.

No automatic thresholds are inferred from a numeric Confidence alone. Callers
supply the replay cutoff and governed empirical selection. RiskSignal currently
binds Confidence by target/version; full fitted-artifact provenance is upstream.
"""

from __future__ import annotations

from datetime import datetime
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal, localcontext
from typing import Literal

from pydantic import AwareDatetime, model_validator

from risk_engine.config import PolicyConfig
from risk_engine.domain import (
    ConfidenceTarget,
    CurrencyCode,
    DomainModel,
    ImpactScore,
    NonEmptyString,
    PortfolioMateriality,
    Probability,
    RiskSignal,
    SignalFlag,
    VersionMetadata,
)

GateName = Literal[
    "policy_selection",
    "policy_chronology",
    "entity_identity",
    "confidence_binding",
    "confidence_threshold",
    "impact_threshold",
    "supported_risk",
    "materiality_currency",
    "economic_floor",
]


class TriggerGate(DomainModel):
    name: GateName
    passed: bool
    reason: NonEmptyString


_SUBSTANTIVE_NAMES = (
    "entity_identity",
    "confidence_binding",
    "confidence_threshold",
    "impact_threshold",
    "supported_risk",
    "materiality_currency",
    "economic_floor",
)


class TriggerCriteria(DomainModel):
    confidence_threshold: Decimal
    economic_floor: Decimal
    materiality_tolerance: Decimal
    currency: CurrencyCode
    confidence_target: Literal[ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS]

    @model_validator(mode="after")
    def valid_thresholds(self) -> TriggerCriteria:
        if not all(
            v.is_finite()
            for v in (self.confidence_threshold, self.economic_floor, self.materiality_tolerance)
        ) or not (
            0 < self.confidence_threshold <= 1
            and self.economic_floor > 0
            and 0 <= self.materiality_tolerance < self.economic_floor
        ):
            raise ValueError("invalid finite candidate trigger thresholds")
        return self


class CandidateTriggerEvaluation(DomainModel):
    gates: tuple[TriggerGate, ...]
    would_trigger: bool

    @model_validator(mode="after")
    def ordered_results(self) -> CandidateTriggerEvaluation:
        if tuple(g.name for g in self.gates) != _SUBSTANTIVE_NAMES:
            raise ValueError("candidate trigger requires complete ordered substantive gates")
        if self.would_trigger != all(g.passed for g in self.gates):
            raise ValueError("candidate trigger status differs from retained gate results")
        return self


class ManualOverride(DomainModel):
    analyst_id: NonEmptyString
    reason: NonEmptyString


class TriggerDecision(DomainModel):
    risk_signal: RiskSignal
    signal_id: NonEmptyString
    source_item_id: NonEmptyString
    signal_versions: VersionMetadata
    as_of: AwareDatetime
    policy_config: PolicyConfig
    impact_score: ImpactScore
    confidence: Probability
    confidence_target: ConfidenceTarget
    portfolio_materiality: PortfolioMateriality
    action_priority: float | None
    gates: tuple[TriggerGate, ...]
    automatic_trigger: bool
    manual_override: ManualOverride | None
    triggered: bool

    @model_validator(mode="after")
    def audit_record_is_consistent(self) -> TriggerDecision:
        signal = self.risk_signal
        if (
            self.signal_id,
            self.source_item_id,
            self.signal_versions,
            self.impact_score,
            self.confidence,
            self.confidence_target,
            self.action_priority,
        ) != (
            signal.signal_id,
            signal.source_item_id,
            signal.versions,
            signal.impact.impact_score,
            signal.confidence,
            signal.confidence_target,
            signal.action_priority,
        ):
            raise ValueError("Decision summaries must match the retained Risk Signal")
        expected_gates = TriggerPolicy._evaluate_gates(
            signal, self.portfolio_materiality, self.policy_config, self.as_of
        )
        if self.gates != expected_gates:
            raise ValueError(
                "Decision requires complete ordered gates matching input results/reasons"
            )
        automatic = all(gate.passed for gate in expected_gates)
        if self.automatic_trigger != automatic:
            raise ValueError("Automatic status must match the complete gate evidence")
        if self.triggered != (automatic or self.manual_override is not None):
            raise ValueError(
                "Combined status must match automatic eligibility or explicit override"
            )
        return self


class _PolicyContext(DomainModel):
    config: PolicyConfig
    as_of: AwareDatetime


def _meets_floor(loss: Decimal, floor: Decimal, tolerance: Decimal) -> bool:
    """Inclusive floor-minus-tolerance comparison with an explicit arithmetic context."""
    precision = (
        max(floor.adjusted(), tolerance.adjusted())
        - min(int(str(floor.as_tuple().exponent)), int(str(tolerance.as_tuple().exponent)))
        + 3
    )
    with localcontext(Context(prec=precision, Emax=MAX_EMAX, Emin=MIN_EMIN)):
        return loss >= floor - tolerance


def _substantive_gates(
    signal: RiskSignal,
    loss: PortfolioMateriality,
    *,
    confidence_threshold: Decimal | None,
    economic_floor: Decimal | None,
    materiality_tolerance: Decimal | None,
    currency: CurrencyCode | None,
    confidence_bound: bool,
    impact_threshold: int,
) -> tuple[TriggerGate, ...]:
    known_entity = (
        not signal.entity.ambiguous
        and SignalFlag.AMBIGUOUS_ENTITY not in signal.flags
        and not signal.entity.entity_id.casefold().startswith(("unknown:", "ambiguous:"))
    )
    gates: list[TriggerGate] = []

    def gate(name: GateName, passed: bool, success: str, failure: str) -> None:
        gates.append(TriggerGate(name=name, passed=passed, reason=success if passed else failure))

    gate(
        "entity_identity",
        known_entity,
        "Entity identity is known and unambiguous",
        "Unknown or ambiguous entity identity cannot automatically trigger stress",
    )
    gate(
        "confidence_binding",
        confidence_bound,
        "Confidence target and calibration version match selected policy",
        "Selected policy is absent or Confidence target/calibration version does not match",
    )
    gate(
        "confidence_threshold",
        confidence_threshold is not None
        and Decimal(str(signal.confidence)) >= confidence_threshold,
        "Confidence meets the selected inclusive threshold",
        "Confidence threshold is unselected or calibrated probability is below it",
    )
    gate(
        "impact_threshold",
        signal.impact.impact_score >= impact_threshold,
        "Impact Score meets the requirement-driven inclusive threshold of 8",
        "Impact Score is below the requirement-driven threshold of 8",
    )
    gate(
        "supported_risk",
        SignalFlag.UNSUPPORTED_EXPOSURE not in signal.flags,
        "No unsupported exposure is declared",
        "Unsupported exposure cannot be treated as zero risk",
    )
    gate(
        "materiality_currency",
        currency is not None and loss.currency == currency,
        "Portfolio Materiality currency matches the selected economic floor",
        "Economic floor currency is unselected or differs; no implicit FX conversion",
    )
    gate(
        "economic_floor",
        economic_floor is not None
        and materiality_tolerance is not None
        and _meets_floor(loss.absolute_loss, economic_floor, materiality_tolerance),
        "Absolute Portfolio Materiality meets the inclusive floor within declared tolerance",
        "Economic floor is unselected or absolute loss is below floor minus tolerance",
    )
    return tuple(gates)


def evaluate_candidate_trigger(
    signal: RiskSignal, materiality: PortfolioMateriality, criteria: TriggerCriteria
) -> CandidateTriggerEvaluation:
    signal = RiskSignal.model_validate(signal.model_dump(mode="python"))
    materiality = PortfolioMateriality.model_validate(materiality.model_dump(mode="python"))
    criteria = TriggerCriteria.model_validate(criteria.model_dump(mode="python"))
    gates = _substantive_gates(
        signal,
        materiality,
        confidence_threshold=criteria.confidence_threshold,
        economic_floor=criteria.economic_floor,
        materiality_tolerance=criteria.materiality_tolerance,
        currency=criteria.currency,
        confidence_bound=signal.confidence_target == criteria.confidence_target,
        impact_threshold=8,
    )
    return CandidateTriggerEvaluation(gates=gates, would_trigger=all(g.passed for g in gates))


class TriggerPolicy:
    def __init__(self, config: PolicyConfig, *, as_of: datetime) -> None:
        # Revalidate copied records rather than blessing unchecked model_copy updates.
        self._context = _PolicyContext(
            config=PolicyConfig.model_validate(config.model_dump(mode="python")), as_of=as_of
        )

    def evaluate(
        self,
        signal: RiskSignal,
        portfolio_materiality: PortfolioMateriality,
        *,
        manual_override: ManualOverride | None = None,
    ) -> TriggerDecision:
        signal = RiskSignal.model_validate(signal.model_dump(mode="python"))
        loss = PortfolioMateriality.model_validate(portfolio_materiality.model_dump(mode="python"))
        if manual_override is not None:
            manual_override = ManualOverride.model_validate(
                manual_override.model_dump(mode="python")
            )
        config = self._context.config
        gates = self._evaluate_gates(signal, loss, config, self._context.as_of)
        automatic = all(g.passed for g in gates)
        return TriggerDecision(
            risk_signal=signal,
            signal_id=signal.signal_id,
            source_item_id=signal.source_item_id,
            signal_versions=signal.versions,
            as_of=self._context.as_of,
            policy_config=config,
            impact_score=signal.impact.impact_score,
            confidence=signal.confidence,
            confidence_target=signal.confidence_target,
            portfolio_materiality=loss,
            action_priority=signal.action_priority,
            gates=gates,
            automatic_trigger=automatic,
            manual_override=manual_override,
            triggered=automatic or manual_override is not None,
        )

    @staticmethod
    def _evaluate_gates(
        signal: RiskSignal,
        loss: PortfolioMateriality,
        config: PolicyConfig,
        as_of: datetime,
    ) -> tuple[TriggerGate, ...]:
        """One gate definition shared by creation and restored-record validation."""
        selected = config.selected
        gates: list[TriggerGate] = []

        def gate(name: GateName, passed: bool, success: str, failure: str) -> None:
            gates.append(
                TriggerGate(name=name, passed=passed, reason=success if passed else failure)
            )

        gate(
            "policy_selection",
            selected is not None,
            "Empirical chronological development selection is declared with evidence metadata",
            "Automatic policy is unselected; no validated Confidence threshold or economic floor",
        )
        gate(
            "policy_chronology",
            selected is not None and selected.frozen_at <= as_of,
            "Selected policy was frozen by the explicit replay cutoff",
            "Selected policy is absent or was frozen after the replay cutoff",
        )
        gates.extend(
            _substantive_gates(
                signal,
                loss,
                confidence_threshold=selected.confidence_threshold if selected else None,
                economic_floor=selected.economic_floor if selected else None,
                materiality_tolerance=selected.materiality_tolerance if selected else None,
                currency=selected.currency if selected else None,
                confidence_bound=selected is not None
                and signal.confidence_target == selected.confidence_target
                and signal.versions.calibration_version == selected.calibration_version,
                impact_threshold=config.impact_threshold,
            )
        )
        return tuple(gates)


__all__ = [
    "CandidateTriggerEvaluation",
    "ManualOverride",
    "TriggerCriteria",
    "TriggerDecision",
    "TriggerGate",
    "TriggerPolicy",
    "evaluate_candidate_trigger",
]
