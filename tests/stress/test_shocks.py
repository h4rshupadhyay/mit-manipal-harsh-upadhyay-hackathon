"""The valuation boundary converts complete Joint Shock Vectors atomically."""

import pytest

from risk_engine.domain import FactorShock, ProvenanceMethod, ShockType, ShockUnit, StressScenario
from risk_engine.stress.shocks import normalize_scenario


def shock(
    factor: str, kind: ShockType, value: float, unit: ShockUnit, horizon: int = 5
) -> FactorShock:
    return FactorShock(
        factor_id=factor, shock_type=kind, value=value, unit=unit, horizon_days=horizon
    )


def scenario(*shocks: FactorShock) -> StressScenario:
    return StressScenario(
        scenario_id="scenario-1",
        risk_signal_id="signal-1",
        shocks=shocks,
        method=ProvenanceMethod.EMPIRICAL,
        reference_event_ids=("event-1",),
        calibration_version="calibration-1",
        analyst_override=True,
        override_reason="Analyst approved",
    )


@pytest.mark.parametrize(
    ("factor", "kind", "value", "unit", "expected", "expected_unit"),
    [
        ("EQUITY-INDIA", ShockType.RELATIVE, -12, ShockUnit.PERCENT, -.12, ShockUnit.DECIMAL),
        ("USD-INR", ShockType.RELATIVE, .08, ShockUnit.DECIMAL, .08, ShockUnit.DECIMAL),
        ("USD-INR", ShockType.ABSOLUTE, 2.5, ShockUnit.CURRENCY, 2.5, ShockUnit.CURRENCY),
        ("EQUITY-INDIA", ShockType.ABSOLUTE, -25, ShockUnit.INDEX_POINT,
         -25, ShockUnit.INDEX_POINT),
        ("COMMODITY", ShockType.ABSOLUTE, -7, ShockUnit.ABSOLUTE, -7, ShockUnit.ABSOLUTE),
        ("CREDIT-SPREAD", ShockType.BASIS_POINT, 125, ShockUnit.BASIS_POINT,
         .0125, ShockUnit.DECIMAL),
        ("RATE-USD", ShockType.BASIS_POINT, -25, ShockUnit.BASIS_POINT,
         -.0025, ShockUnit.DECIMAL),
        ("VOLATILITY", ShockType.VOLATILITY, 5, ShockUnit.VOLATILITY_POINT,
         .05, ShockUnit.DECIMAL),
        ("DEFAULT", ShockType.DEFAULT, 75, ShockUnit.PERCENT, .75, ShockUnit.DECIMAL),
    ],
)
def test_converts_shocks_to_explicit_adapter_units(
    factor: str, kind: ShockType, value: float, unit: ShockUnit,
    expected: float, expected_unit: ShockUnit,
) -> None:
    result = normalize_scenario(scenario(shock(factor, kind, value, unit)))
    converted = result.shocks[0]
    assert converted.value == pytest.approx(expected)
    assert converted.unit is expected_unit
    assert converted.factor_id == factor
    assert converted.shock_type is kind
    assert converted.horizon_days == 5


def test_preserves_scenario_provenance_and_input() -> None:
    original = scenario(
        shock("EQUITY-INDIA", ShockType.RELATIVE, -10, ShockUnit.PERCENT),
        shock("CREDIT-SPREAD", ShockType.BASIS_POINT, 50, ShockUnit.BASIS_POINT),
    )
    before = original.model_dump()
    result = normalize_scenario(original)
    assert result.scenario_id == "scenario-1"
    assert result.risk_signal_id == "signal-1"
    assert result.method is ProvenanceMethod.EMPIRICAL
    assert result.reference_event_ids == ("event-1",)
    assert result.calibration_version == "calibration-1"
    assert result.analyst_override is True
    assert result.override_reason == "Analyst approved"
    assert result.horizon_days == 5
    assert result.normalization_version
    assert [item.value for item in result.shocks] == pytest.approx([-.1, .005])
    assert original.model_dump() == before


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (shock("UNKNOWN", ShockType.RELATIVE, -.1, ShockUnit.DECIMAL), "unknown factor"),
        (shock("USD-INR", ShockType.RELATIVE, .1, ShockUnit.DECIMAL, 10), "horizon"),
        (shock("EQUITY-INDIA", ShockType.RELATIVE, -101, ShockUnit.PERCENT), "relative"),
        (shock("DEFAULT", ShockType.DEFAULT, -.1, ShockUnit.DECIMAL), "default"),
        (shock("DEFAULT", ShockType.DEFAULT, 101, ShockUnit.PERCENT), "default"),
        (shock("VOLATILITY", ShockType.RELATIVE, .1, ShockUnit.DECIMAL), "incompatible"),
        (shock("CREDIT-SPREAD", ShockType.ABSOLUTE, 10, ShockUnit.CURRENCY), "incompatible"),
        (shock("EQUITY-INDIA", ShockType.ABSOLUTE, 10, ShockUnit.CURRENCY), "native unit"),
        (shock("USD-INR", ShockType.ABSOLUTE, 10, ShockUnit.INDEX_POINT), "native unit"),
    ],
)
def test_rejects_entire_scenario_without_mutation(invalid: FactorShock, message: str) -> None:
    original = scenario(shock("COMMODITY", ShockType.RELATIVE, -.1, ShockUnit.DECIMAL), invalid)
    before = original.model_dump()
    with pytest.raises(ValueError, match=message):
        normalize_scenario(original)
    assert original.model_dump() == before


@pytest.mark.parametrize("value", [-1, 0, 1])
def test_accepts_valid_relative_and_default_boundaries(value: float) -> None:
    result = normalize_scenario(scenario(
        shock("EQUITY-INDIA", ShockType.RELATIVE, value, ShockUnit.DECIMAL),
        shock("DEFAULT", ShockType.DEFAULT, abs(value), ShockUnit.DECIMAL),
    ))
    assert result.shocks[0].value == value
    assert result.shocks[1].value == abs(value)


def test_revalidates_type_unit_compatibility_at_boundary() -> None:
    incompatible = shock("COMMODITY", ShockType.RELATIVE, .1, ShockUnit.DECIMAL).model_copy(
        update={"unit": ShockUnit.BASIS_POINT}
    )
    with pytest.raises(ValueError, match="incompatible"):
        normalize_scenario(scenario(incompatible))


def test_retains_signed_volatility_declines() -> None:
    result = normalize_scenario(scenario(
        shock("VOLATILITY", ShockType.VOLATILITY, -3, ShockUnit.VOLATILITY_POINT),
    ))
    assert result.shocks[0].value == pytest.approx(-.03)
