"""Portfolio orchestration with hand-calculated losses and explicit coverage."""

import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from risk_engine.domain import (
    AssetType,
    AttributionDimension,
    FactorShock,
    MarketSnapshot,
    Portfolio,
    Position,
    ProvenanceMethod,
    ShockType,
    ShockUnit,
    StressScenario,
)
from risk_engine.stress import module
from risk_engine.stress.interfaces import NormalizedScenario
from risk_engine.stress.module import StressEngine
from tests.stress.test_linear_valuation import market, position, scenario


def portfolio(*holdings: Position) -> Portfolio:
    return Portfolio.model_validate(
        {
            "portfolio_id": "portfolio-1",
            "positions": holdings,
            "valuation_currency": "USD",
            "as_of": market("EQUITY-US").as_of,
            "version": "portfolio-v1",
        }
    )


def joint(*parts: NormalizedScenario) -> NormalizedScenario:
    return parts[0].model_copy(update={"shocks": tuple(p.shocks[0] for p in parts)})


def snapshot(*factors: str) -> MarketSnapshot:
    return market(factors[0]).model_copy(
        update={
            "observations": tuple(market(f).observations[0] for f in factors),
        }
    )


def test_mixed_credit_rules_count_base_once_and_attribute_each_factor() -> None:
    holding = position(
        {
            "duration:RATE-USD": 5,
            "convexity:RATE-USD": 20,
            "cs01:CREDIT-SPREAD": -50,
            "ead": 12000,
            "lgd": 0.4,
        }
    ).model_copy(update={"asset_type": AssetType.BOND})
    result = StressEngine().run(
        portfolio(holding),
        snapshot("RATE-USD", "CREDIT-SPREAD"),
        joint(
            scenario("RATE-USD", 100, ShockType.BASIS_POINT, ShockUnit.BASIS_POINT),
            scenario("CREDIT-SPREAD", 25, ShockType.BASIS_POINT, ShockUnit.BASIS_POINT),
            scenario("DEFAULT", 0.25, ShockType.DEFAULT),
        ),
    )
    assert (result.base_value, result.stressed_value, result.absolute_loss) == (
        Decimal("10000"),
        Decimal("7060"),
        Decimal("2940"),
    )
    assert result.percentage_loss == 0.294
    assert result.valuation_coverage == 1
    assert result.unsupported_position_ids == ()
    factors = {
        r.label: r.loss for r in result.attribution if r.dimension is AttributionDimension.FACTOR
    }
    assert factors == {
        "RATE-USD": Decimal("490"),
        "CREDIT-SPREAD": Decimal("1250"),
        "DEFAULT": Decimal("1200"),
    }
    for dimension in (
        AttributionDimension.ASSET,
        AttributionDimension.SECTOR,
        AttributionDimension.REGION,
        AttributionDimension.OBLIGOR,
        AttributionDimension.FACTOR,
    ):
        assert sum(r.loss for r in result.attribution if r.dimension is dimension) == 2940
    versions = json.loads(result.versions.valuation_rule_version)
    assert set(versions["factors"]) == {"RATE-USD", "CREDIT-SPREAD", "DEFAULT"}
    assert all(versions["factors"].values())
    assert result.versions.portfolio_version == "portfolio-v1"
    assert result.versions.scenario_version == "calibration-v1"
    assert "market-1" in result.versions.market_version


@pytest.mark.parametrize("missing", ["delta:EQUITY-US", "gamma:EQUITY-US", "vega:VOLATILITY"])
def test_incomplete_nonlinear_position_is_disclosed_without_zero_loss(missing: str) -> None:
    exposures = {"delta:EQUITY-US": 50, "gamma:EQUITY-US": 2, "vega:VOLATILITY": 10}
    del exposures[missing]
    unsupported = position(exposures, derivative=True, notional="-30000").model_copy(
        update={"position_id": "unsupported"}
    )
    result = StressEngine().run(
        portfolio(position({"EQUITY-US": 1}), unsupported),
        snapshot("EQUITY-US", "VOLATILITY"),
        joint(
            scenario("EQUITY-US", -0.1),
            scenario("VOLATILITY", 2, ShockType.VOLATILITY, ShockUnit.VOLATILITY_POINT),
        ),
    )
    assert result.valuation_coverage == 0.25
    assert result.unsupported_position_ids == ("unsupported",)
    assert result.base_value == 10000
    assert result.stressed_value == 9000
    assert all(r.label != "unsupported" for r in result.attribution)


def test_complete_nonlinear_uses_native_delta_gamma_and_per_point_vega() -> None:
    result = StressEngine().run(
        portfolio(
            position(
                {
                    "delta:EQUITY-US": 50,
                    "gamma:EQUITY-US": 2,
                    "vega:VOLATILITY": 10,
                },
                derivative=True,
                notional="75",
            )
        ),
        snapshot("EQUITY-US", "VOLATILITY"),
        joint(
            scenario("EQUITY-US", -0.1),
            scenario("VOLATILITY", 2, ShockType.VOLATILITY, ShockUnit.VOLATILITY_POINT),
        ),
    )
    # -500 delta + 100 gamma + 20 vega, independent of base value 75.
    assert result.absolute_loss == 380
    assert result.stressed_value == -305
    assert result.valuation_coverage == 1
    assert {
        r.label: r.loss for r in result.attribution if r.dimension is AttributionDimension.FACTOR
    } == {"EQUITY-US": Decimal("400"), "VOLATILITY": Decimal("-20")}


def test_linear_derivative_without_nonlinear_keys_remains_supported() -> None:
    result = StressEngine().run(
        portfolio(position({"delta:EQUITY-US": 50}, derivative=True)),
        market("EQUITY-US"),
        scenario("EQUITY-US", -0.1),
    )
    assert result.absolute_loss == 500
    assert result.valuation_coverage == 1


def test_explicit_zero_gamma_and_vega_are_complete() -> None:
    result = StressEngine().run(
        portfolio(
            position(
                {
                    "delta:EQUITY-US": 50,
                    "gamma:EQUITY-US": 0,
                    "vega:VOLATILITY": 0,
                },
                derivative=True,
            )
        ),
        snapshot("EQUITY-US", "VOLATILITY"),
        scenario("EQUITY-US", -0.1),
    )
    assert result.absolute_loss == 500
    assert result.valuation_coverage == 1


def test_currency_mismatch_rejects_instead_of_implicit_conversion() -> None:
    with pytest.raises(ValueError, match="currency"):
        StressEngine().run(
            portfolio(position({"EQUITY-US": 1}).model_copy(update={"currency": "INR"})),
            market("EQUITY-US"),
            scenario("EQUITY-US", -0.1),
        )


def test_missing_market_input_rejects_instead_of_reclassifying_supported_position() -> None:
    with pytest.raises(ValueError, match="missing market factor"):
        StressEngine().run(
            portfolio(position({"EQUITY-US": 1})), market("COMMODITY"), scenario("EQUITY-US", -0.1)
        )


def test_no_supported_positions_rejects_without_invented_attribution() -> None:
    with pytest.raises(ValueError, match="supported"):
        StressEngine().run(
            portfolio(position({}, derivative=True)),
            market("EQUITY-US"),
            scenario("EQUITY-US", -0.1),
        )


def test_unknown_sensitivity_is_not_silently_dropped() -> None:
    bad = position({"delta:EQUITY-US": 50, "mystery:EQUITY-US": 3}, derivative=True)
    result = StressEngine().run(
        portfolio(position({"EQUITY-US": 1}), bad.model_copy(update={"position_id": "bad"})),
        market("EQUITY-US"),
        scenario("EQUITY-US", -0.1),
    )
    assert result.unsupported_position_ids == ("bad",)
    assert result.valuation_coverage == 0.5


def test_temporal_mismatch_and_corrupt_normalization_reject() -> None:
    holdings = portfolio(position({"EQUITY-US": 1}))
    with pytest.raises(ValueError, match="as_of"):
        StressEngine().run(
            holdings.model_copy(update={"as_of": datetime(2025, 1, 1, tzinfo=timezone.utc)}),
            market("EQUITY-US"),
            scenario("EQUITY-US", -0.1),
        )
    with pytest.raises(ValueError, match="normalization"):
        StressEngine().run(
            holdings,
            market("EQUITY-US"),
            scenario("EQUITY-US", -0.1).model_copy(update={"normalization_version": "unknown"}),
        )


def test_raw_scenario_normalizes_to_identical_offline_result() -> None:
    holdings = portfolio(position({"EQUITY-US": 1}))
    raw = StressScenario(
        scenario_id="scenario-1",
        risk_signal_id="signal-1",
        shocks=(
            FactorShock(
                factor_id="EQUITY-US",
                value=-10,
                shock_type=ShockType.RELATIVE,
                unit=ShockUnit.PERCENT,
                horizon_days=5,
            ),
        ),
        method=ProvenanceMethod.HYPOTHETICAL,
        calibration_version="calibration-v1",
    )
    assert StressEngine().run(holdings, market("EQUITY-US"), raw) == (
        StressEngine().run(holdings, market("EQUITY-US"), scenario("EQUITY-US", -0.1))
    )


def test_missing_registered_factor_versions_reject_even_when_factor_is_unexposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "FACTOR_RULE_VERSIONS", {})
    with pytest.raises(ValueError, match="registered rule version"):
        StressEngine().run(
            portfolio(position({"EQUITY-US": 1})), market("EQUITY-US"), scenario("COMMODITY", -0.1)
        )


def test_normalized_provenance_must_remain_consistent() -> None:
    with pytest.raises(ValueError, match="override"):
        StressEngine().run(
            portfolio(position({"EQUITY-US": 1})),
            market("EQUITY-US"),
            scenario("EQUITY-US", -0.1).model_copy(update={"analyst_override": True}),
        )
    with pytest.raises(ValueError, match="empirical"):
        StressEngine().run(
            portfolio(position({"EQUITY-US": 1})),
            market("EQUITY-US"),
            scenario("EQUITY-US", -0.1).model_copy(update={"method": ProvenanceMethod.EMPIRICAL}),
        )


def test_zero_value_unsupported_still_discloses_incomplete_coverage() -> None:
    result = StressEngine().run(
        portfolio(
            position({"EQUITY-US": 1}),
            position({}, derivative=True, notional="0").model_copy(update={"position_id": "zero"}),
        ),
        market("EQUITY-US"),
        scenario("EQUITY-US", -0.1),
    )
    assert result.unsupported_position_ids == ("zero",)
    assert result.valuation_coverage < 1
    assert result.absolute_loss == 1000


def test_all_zero_value_coverage_uses_count_share() -> None:
    result = StressEngine().run(
        portfolio(
            position({"EQUITY-US": 1}, notional="0"),
            position({}, derivative=True, notional="0").model_copy(update={"position_id": "zero"}),
        ),
        market("EQUITY-US"),
        scenario("EQUITY-US", -0.1),
    )
    assert result.valuation_coverage == 0.5
    assert result.percentage_loss == 0


def test_nonlinear_requires_gamma_for_every_supplied_spot_factor() -> None:
    bad = position(
        {"delta:EQUITY-US": 50, "gamma:EQUITY-US": 2, "delta:COMMODITY": 10, "vega:VOLATILITY": 10},
        derivative=True,
    ).model_copy(update={"position_id": "incomplete"})
    result = StressEngine().run(
        portfolio(position({"EQUITY-US": 1}), bad),
        snapshot("EQUITY-US", "COMMODITY", "VOLATILITY"),
        scenario("EQUITY-US", -0.1),
    )
    assert result.unsupported_position_ids == ("incomplete",)
    assert result.absolute_loss == 1000


def test_invalid_default_inputs_reject_without_partial_result() -> None:
    holding = position(
        {"duration:RATE-USD": 5, "convexity:RATE-USD": 20, "ead": 12000, "lgd": 2}
    ).model_copy(update={"asset_type": AssetType.LOAN})
    with pytest.raises(ValueError, match="LGD"):
        StressEngine().run(
            portfolio(holding),
            market("RATE-USD"),
            scenario("RATE-USD", 100, ShockType.BASIS_POINT, ShockUnit.BASIS_POINT),
        )


def test_nonlinear_absolute_delta_retains_native_precision() -> None:
    holding = position(
        {"delta:COMMODITY": 1, "gamma:COMMODITY": 0, "vega:VOLATILITY": 0}, derivative=True
    )
    result = StressEngine().run(
        portfolio(holding),
        snapshot("COMMODITY", "VOLATILITY"),
        scenario("COMMODITY", 1, ShockType.ABSOLUTE, ShockUnit.ABSOLUTE),
    )
    assert result.absolute_loss == -1
