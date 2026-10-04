"""Hand-calculated values and signed Stress Test invariants."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from risk_engine.domain import (
    AssetType,
    FactorShock,
    MarketObservation,
    MarketSnapshot,
    Position,
    ProvenanceMethod,
    ShockType,
    ShockUnit,
    StressScenario,
)
from risk_engine.stress.interfaces import FACTOR_REGISTRY, NormalizedScenario
from risk_engine.stress.shocks import normalize_scenario
from risk_engine.stress.valuation import LinearDerivativeValuationAdapter, SpotValuationAdapter


def position(
    exposures: dict[str, float], *, derivative: bool = False, notional: str = "10000"
) -> Position:
    return Position(
        position_id="position-1",
        asset_type=AssetType.DERIVATIVE if derivative else AssetType.OTHER,
        notional=Decimal(notional),
        currency="USD",
        sector="Synthetic",
        region="Global",
        obligor="Synthetic obligor",
        factor_exposures=exposures,
    )


def market(factor: str, level: float = 100) -> MarketSnapshot:
    as_of = datetime(2026, 10, 4, tzinfo=timezone.utc)
    return MarketSnapshot(
        snapshot_id="market-1",
        as_of=as_of,
        observations=(
            MarketObservation(
                factor_id=factor,
                observed_at=as_of,
                value=level,
                unit=FACTOR_REGISTRY[factor].native_unit,
                provider="synthetic",
                vintage="v1",
            ),
        ),
        provider="synthetic",
        schema_version="v1",
        normalization_version="market-v1",
    )


def scenario(
    factor: str,
    value: float,
    kind: ShockType = ShockType.RELATIVE,
    unit: ShockUnit = ShockUnit.DECIMAL,
) -> NormalizedScenario:
    return normalize_scenario(
        StressScenario(
            scenario_id="scenario-1",
            risk_signal_id="signal-1",
            shocks=(
                FactorShock(
                    factor_id=factor, value=value, shock_type=kind, unit=unit, horizon_days=5
                ),
            ),
            method=ProvenanceMethod.HYPOTHETICAL,
            calibration_version="calibration-v1",
        )
    )


@pytest.mark.parametrize("factor", ["EQUITY-US", "COMMODITY", "EUR-USD"])
@pytest.mark.parametrize(
    ("notional", "weight", "expected"),
    [
        ("10000", 1, "-1000"),
        ("-10000", 1, "1000"),
        ("10000", -0.5, "500"),
    ],
)
def test_spot_signed_market_value_times_relative_shock(
    factor: str,
    notional: str,
    weight: float,
    expected: str,
) -> None:
    adapter = SpotValuationAdapter()
    holding = position({factor: weight}, notional=notional)
    snapshot = market(factor)
    assert adapter.value(holding, snapshot) == Decimal(notional)
    pnl = adapter.stressed_pnl(holding, snapshot, scenario(factor, -10, unit=ShockUnit.PERCENT))
    assert isinstance(pnl, Decimal)
    assert pnl == Decimal(expected)
    assert adapter.stressed_value(holding, snapshot, scenario(factor, -0.1)) == (
        Decimal(notional) + pnl
    )


@pytest.mark.parametrize(
    ("factor", "level", "move", "unit", "expected"),
    [
        ("EQUITY-US", 200, -20, ShockUnit.INDEX_POINT, "-1000"),
        ("COMMODITY", 80, -4, ShockUnit.ABSOLUTE, "-500"),
        ("EUR-USD", 1.25, 0.125, ShockUnit.CURRENCY, "1000"),
    ],
)
def test_spot_absolute_shock_uses_snapshot_level(
    factor: str,
    level: float,
    move: float,
    unit: ShockUnit,
    expected: str,
) -> None:
    assert SpotValuationAdapter().stressed_pnl(
        position({factor: 1}),
        market(factor, level),
        scenario(factor, move, ShockType.ABSOLUTE, unit),
    ) == Decimal(expected)


@pytest.mark.parametrize(
    ("sensitivity", "kind", "move", "unit", "expected"),
    [
        ("delta:EQUITY-US", ShockType.RELATIVE, -0.1, ShockUnit.DECIMAL, "-500"),
        ("delta:EQUITY-US", ShockType.ABSOLUTE, -10, ShockUnit.INDEX_POINT, "-500"),
        ("dv01:RATE-USD", ShockType.BASIS_POINT, 25, ShockUnit.BASIS_POINT, "1250"),
        ("dv01:RATE-USD", ShockType.BASIS_POINT, -25, ShockUnit.BASIS_POINT, "-1250"),
    ],
)
def test_linear_delta_and_signed_dv01(
    sensitivity: str,
    kind: ShockType,
    move: float,
    unit: ShockUnit,
    expected: str,
) -> None:
    factor = sensitivity.split(":")[1]
    holding = position({sensitivity: 50}, derivative=True, notional="75")
    adapter = LinearDerivativeValuationAdapter()
    assert adapter.value(holding, market(factor)) == Decimal("75")
    pnl = adapter.stressed_pnl(holding, market(factor), scenario(factor, move, kind, unit))
    assert isinstance(pnl, Decimal)
    assert pnl == Decimal(expected)


@pytest.mark.parametrize("sensitivity", ["EQUITY-US", "delta:EQUITY-US", "dv01:RATE-USD"])
@given(exposure=st.integers(-10000, 10000), notional=st.integers(-100000, 100000))
def test_zero_shock_preserves_base_value(sensitivity: str, exposure: int, notional: int) -> None:
    derivative = ":" in sensitivity
    factor = sensitivity.split(":")[-1]
    adapter = LinearDerivativeValuationAdapter() if derivative else SpotValuationAdapter()
    kind = ShockType.BASIS_POINT if sensitivity.startswith("dv01:") else ShockType.RELATIVE
    unit = ShockUnit.BASIS_POINT if kind is ShockType.BASIS_POINT else ShockUnit.DECIMAL
    holding = position({sensitivity: exposure}, derivative=derivative, notional=str(notional))
    snapshot = market(factor)
    zero = scenario(factor, 0, kind, unit)
    assert adapter.stressed_pnl(holding, snapshot, zero) == Decimal(0)
    assert adapter.stressed_value(holding, snapshot, zero) == adapter.value(holding, snapshot)


@pytest.mark.parametrize("sensitivity", ["EQUITY-US", "delta:EQUITY-US", "dv01:RATE-USD"])
@given(exposure=st.integers(1, 10000), first=st.integers(-100, 100), increment=st.integers(0, 100))
def test_positive_exposure_value_is_monotone_in_factor_move(
    sensitivity: str,
    exposure: int,
    first: int,
    increment: int,
) -> None:
    derivative = ":" in sensitivity
    factor = sensitivity.split(":")[-1]
    adapter = LinearDerivativeValuationAdapter() if derivative else SpotValuationAdapter()
    kind = ShockType.BASIS_POINT if sensitivity.startswith("dv01:") else ShockType.RELATIVE
    unit = ShockUnit.BASIS_POINT if kind is ShockType.BASIS_POINT else ShockUnit.PERCENT
    holding = position({sensitivity: exposure}, derivative=derivative)
    snapshot = market(factor)
    assert adapter.stressed_pnl(holding, snapshot, scenario(factor, first, kind, unit)) <= (
        adapter.stressed_pnl(holding, snapshot, scenario(factor, first + increment, kind, unit))
    )


def test_unshocked_factor_has_no_pnl() -> None:
    assert SpotValuationAdapter().stressed_pnl(
        position({"EQUITY-US": 1}),
        market("EQUITY-US"),
        scenario("COMMODITY", -0.1),
    ) == Decimal(0)


@pytest.mark.parametrize(
    "adapter,exposures,derivative",
    [
        (SpotValuationAdapter(), {}, False),
        (LinearDerivativeValuationAdapter(), {"EQUITY-US": 50}, True),
        (LinearDerivativeValuationAdapter(), {"delta:RATE-USD": 50}, True),
        (SpotValuationAdapter(), {"RATE-USD": 1}, False),
    ],
)
def test_rejects_missing_or_ambiguous_sensitivity(
    adapter: SpotValuationAdapter | LinearDerivativeValuationAdapter,
    exposures: dict[str, float],
    derivative: bool,
) -> None:
    with pytest.raises(ValueError):
        adapter.value(position(exposures, derivative=derivative), market("EQUITY-US"))


def test_rejects_missing_market_factor_and_incompatible_unit() -> None:
    adapter = SpotValuationAdapter()
    holding = position({"EQUITY-US": 1})
    with pytest.raises(ValueError, match="missing market factor"):
        adapter.value(holding, market("COMMODITY"))
    snapshot = market("EQUITY-US")
    bad = snapshot.model_copy(
        update={
            "observations": (
                snapshot.observations[0].model_copy(update={"unit": ShockUnit.CURRENCY}),
            )
        }
    )
    with pytest.raises(ValueError, match="unit"):
        adapter.value(holding, bad)


def test_rejects_nonpositive_spot_level() -> None:
    with pytest.raises(ValueError, match="positive"):
        SpotValuationAdapter().value(position({"COMMODITY": 1}), market("COMMODITY", 0))


def test_absolute_delta_preserves_native_move_precision() -> None:
    assert LinearDerivativeValuationAdapter().stressed_pnl(
        position({"delta:COMMODITY": 1}, derivative=True),
        market("COMMODITY", 3),
        scenario("COMMODITY", 1, ShockType.ABSOLUTE, ShockUnit.ABSOLUTE),
    ) == Decimal("1")


def test_negative_dv01_loses_value_when_rates_rise() -> None:
    assert LinearDerivativeValuationAdapter().stressed_pnl(
        position({"dv01:RATE-USD": -50}, derivative=True),
        market("RATE-USD"),
        scenario("RATE-USD", 25, ShockType.BASIS_POINT, ShockUnit.BASIS_POINT),
    ) == Decimal("-1250")


def test_combined_linear_exposures_preserve_inputs() -> None:
    adapter = LinearDerivativeValuationAdapter()
    holding = position({"delta:EQUITY-US": 50, "dv01:RATE-USD": -20}, derivative=True)
    snapshot = market("EQUITY-US").model_copy(
        update={
            "observations": (
                *market("EQUITY-US").observations,
                *market("RATE-USD").observations,
            )
        }
    )
    joint = scenario("EQUITY-US", -0.1).model_copy(
        update={
            "shocks": (
                *scenario("EQUITY-US", -0.1).shocks,
                *scenario("RATE-USD", 25, ShockType.BASIS_POINT, ShockUnit.BASIS_POINT).shocks,
            )
        }
    )
    before = (holding.model_dump(), snapshot.model_dump(), joint.model_dump())
    assert adapter.stressed_pnl(holding, snapshot, joint) == Decimal("-1000")
    assert adapter.stressed_value(holding, snapshot, joint) == Decimal("9000")
    assert before == (holding.model_dump(), snapshot.model_dump(), joint.model_dump())


@pytest.mark.parametrize("derivative", [False, True])
def test_rejects_shocks_implying_negative_spot_levels(derivative: bool) -> None:
    adapter = LinearDerivativeValuationAdapter() if derivative else SpotValuationAdapter()
    exposure = "delta:COMMODITY" if derivative else "COMMODITY"
    with pytest.raises(ValueError, match="negative market level"):
        adapter.stressed_pnl(
            position({exposure: 1}, derivative=derivative),
            market("COMMODITY", 10),
            scenario("COMMODITY", -11, ShockType.ABSOLUTE, ShockUnit.ABSOLUTE),
        )


def test_rejects_corrupt_normalized_unit() -> None:
    normalized = scenario("EQUITY-US", -0.1)
    bad = normalized.model_copy(
        update={"shocks": (normalized.shocks[0].model_copy(update={"unit": ShockUnit.PERCENT}),)}
    )
    with pytest.raises(ValueError, match="normalized shock unit"):
        SpotValuationAdapter().stressed_pnl(
            position({"EQUITY-US": 1}),
            market("EQUITY-US"),
            bad,
        )
