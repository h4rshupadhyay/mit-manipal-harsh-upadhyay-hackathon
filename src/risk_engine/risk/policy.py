"""Recorded deterministic stress proposals from explicitly frozen policy inputs.

No automatic thresholds are inferred from a numeric Confidence alone. Callers
supply the replay cutoff and governed empirical selection. RiskSignal currently
binds Confidence by target/version; full fitted-artifact provenance is upstream.
"""

from __future__ import annotations

from datetime import datetime
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal, localcontext
from typing import Literal

from pydantic import AwareDatetime

from risk_engine.config import PolicyConfig
from risk_engine.domain import (
    ConfidenceTarget,
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
        selected = config.selected
        known_entity = (
            not signal.entity.ambiguous
            and SignalFlag.AMBIGUOUS_ENTITY not in signal.flags
            and not signal.entity.entity_id.casefold().startswith(("unknown:", "ambiguous:"))
        )
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
            selected is not None and selected.frozen_at <= self._context.as_of,
            "Selected policy was frozen by the explicit replay cutoff",
            "Selected policy is absent or was frozen after the replay cutoff",
        )
        gate(
            "entity_identity",
            known_entity,
            "Entity identity is known and unambiguous",
            "Unknown or ambiguous entity identity cannot automatically trigger stress",
        )
        gate(
            "confidence_binding",
            selected is not None
            and signal.confidence_target == selected.confidence_target
            and signal.versions.calibration_version == selected.calibration_version,
            "Confidence target and calibration version match selected policy",
            "Selected policy is absent or Confidence target/calibration version does not match",
        )
        gate(
            "confidence_threshold",
            selected is not None
            and Decimal(str(signal.confidence)) >= selected.confidence_threshold,
            "Confidence meets the selected inclusive threshold",
            "Confidence threshold is unselected or calibrated probability is below it",
        )
        gate(
            "impact_threshold",
            signal.impact.impact_score >= config.impact_threshold,
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
            selected is not None and loss.currency == selected.currency,
            "Portfolio Materiality currency matches the selected economic floor",
            "Economic floor currency is unselected or differs; no implicit FX conversion",
        )
        gate(
            "economic_floor",
            selected is not None
            and _meets_floor(
                loss.absolute_loss, selected.economic_floor, selected.materiality_tolerance
            ),
            "Absolute Portfolio Materiality meets the inclusive floor within declared tolerance",
            "Economic floor is unselected or absolute loss is below floor minus tolerance",
        )
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
            gates=tuple(gates),
            automatic_trigger=automatic,
            manual_override=manual_override,
            triggered=automatic or manual_override is not None,
        )


__all__ = ["ManualOverride", "TriggerDecision", "TriggerGate", "TriggerPolicy"]
