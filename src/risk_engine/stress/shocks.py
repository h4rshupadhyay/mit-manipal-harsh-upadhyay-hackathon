"""Single offline, atomic normalization boundary for stress valuation."""

from risk_engine.domain import ShockType, ShockUnit, StressScenario
from risk_engine.stress.interfaces import (
    FACTOR_REGISTRY,
    NORMALIZATION_VERSION,
    NormalizedScenario,
    NormalizedShock,
)


def normalize_scenario(scenario: StressScenario) -> NormalizedScenario:
    """Validate the entire scenario before returning any normalized shocks.

    Direction is signed for market moves: rallies, tightening, and volatility
    declines are legitimate joint observations. Relative spot losses below
    -100% and default fractions outside [0, 1] are economically invalid.
    Structural validity, finite numbers, and shock-type/unit compatibility
    belong to the canonical domain records and are revalidated here.
    """
    validated = StressScenario.model_validate(scenario.model_dump())
    horizon = validated.shocks[0].horizon_days
    normalized: list[NormalizedShock] = []
    for shock in validated.shocks:
        definition = FACTOR_REGISTRY.get(shock.factor_id)
        if definition is None:
            raise ValueError(f"unknown factor: {shock.factor_id}")
        if shock.shock_type not in definition.shock_types:
            raise ValueError(f"incompatible shock type/unit for factor: {shock.factor_id}")
        if shock.shock_type is ShockType.ABSOLUTE and shock.unit is not definition.native_unit:
            raise ValueError(f"incompatible native unit for factor: {shock.factor_id}")
        if shock.horizon_days != horizon:
            raise ValueError("all scenario shocks must have the same horizon")
        value = shock.value
        unit = shock.unit
        if unit in {ShockUnit.PERCENT, ShockUnit.VOLATILITY_POINT}:
            value /= 100
            unit = ShockUnit.DECIMAL
        elif unit is ShockUnit.BASIS_POINT:
            value /= 10_000
            unit = ShockUnit.DECIMAL
        if shock.shock_type is ShockType.RELATIVE and value < -1:
            raise ValueError("relative shock cannot be below -100%")
        if shock.shock_type is ShockType.DEFAULT and not 0 <= value <= 1:
            raise ValueError("default shock must be a fraction in [0, 1]")
        normalized.append(NormalizedShock(
            factor_id=shock.factor_id,
            shock_type=shock.shock_type,
            value=value,
            unit=unit,
            horizon_days=horizon,
        ))
    return NormalizedScenario(
        scenario_id=validated.scenario_id,
        risk_signal_id=validated.risk_signal_id,
        shocks=tuple(normalized),
        horizon_days=horizon,
        method=validated.method,
        reference_event_ids=validated.reference_event_ids,
        calibration_version=validated.calibration_version,
        analyst_override=validated.analyst_override,
        override_reason=validated.override_reason,
        normalization_version=NORMALIZATION_VERSION,
    )
