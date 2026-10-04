"""Hand-calculated, separate risk-free, credit spread and accounting losses."""

from decimal import Decimal

import pytest

from risk_engine.domain import AssetType, ShockType, ShockUnit
from risk_engine.stress import valuation
from risk_engine.stress.interfaces import NormalizedScenario
from tests.stress.test_linear_valuation import market, position, scenario


@pytest.mark.parametrize("asset_type", [AssetType.BOND, AssetType.LOAN])
@pytest.mark.parametrize(
    "move,notional,expected",
    [(100, "10000", "-490"), (-100, "10000", "510"), (100, "-10000", "490")],
)
def test_duration_convexity_uses_decimal_risk_free_yield(
    asset_type: AssetType, move: float, notional: str, expected: str
) -> None:
    adapter = valuation.DurationConvexityValuationAdapter()
    holding = position(
        {"duration:RATE-USD": 5, "convexity:RATE-USD": 20}, notional=notional
    ).model_copy(update={"asset_type": asset_type})
    snapshot = market("RATE-USD")
    shock = scenario("RATE-USD", move, ShockType.BASIS_POINT, ShockUnit.BASIS_POINT)
    assert adapter.value(holding, snapshot) == Decimal(notional)
    pnl = adapter.stressed_pnl(holding, snapshot, shock)
    assert isinstance(pnl, Decimal)
    assert pnl == Decimal(expected)
    assert adapter.stressed_value(holding, snapshot, shock) == Decimal(notional) + pnl


@pytest.mark.parametrize("move,expected", [(25, "-1250"), (-25, "1250"), (0, "0")])
def test_signed_cs01_is_currency_per_basis_point(move: float, expected: str) -> None:
    adapter = valuation.CreditSpreadValuationAdapter()
    holding = position({"cs01:CREDIT-SPREAD": -50}, notional="75").model_copy(
        update={"asset_type": AssetType.BOND}
    )
    shock = scenario("CREDIT-SPREAD", move, ShockType.BASIS_POINT, ShockUnit.BASIS_POINT)
    assert adapter.stressed_pnl(holding, market("CREDIT-SPREAD"), shock) == Decimal(expected)
    assert adapter.stressed_value(holding, market("CREDIT-SPREAD"), shock) == (
        Decimal("75") + Decimal(expected)
    )


@pytest.mark.parametrize("fraction,expected", [(0, "0"), (0.25, "-1200"), (1, "-4800")])
def test_default_is_accounting_loss_from_explicit_ead_lgd(fraction: float, expected: str) -> None:
    adapter = valuation.DefaultValuationAdapter()
    holding = position({"ead": 12000, "lgd": 0.4}, notional="10000").model_copy(
        update={"asset_type": AssetType.LOAN}
    )
    shock = scenario("DEFAULT", fraction, ShockType.DEFAULT)
    # Default fractions need no market price observation.
    snapshot = market("DEFAULT").model_copy(update={"observations": ()})
    assert adapter.value(holding, snapshot) == Decimal("10000")
    assert adapter.stressed_pnl(holding, snapshot, shock) == Decimal(expected)
    assert adapter.stressed_value(holding, snapshot, shock) == Decimal("10000") + Decimal(expected)


@pytest.mark.parametrize(
    "name,exposures,factor",
    [
        (
            "DurationConvexityValuationAdapter",
            {"duration:RATE-USD": 5, "convexity:RATE-USD": 20},
            "RATE-USD",
        ),
        ("CreditSpreadValuationAdapter", {"cs01:CREDIT-SPREAD": -50}, "CREDIT-SPREAD"),
        ("DefaultValuationAdapter", {"ead": 12000, "lgd": 0.4}, "DEFAULT"),
    ],
)
def test_joint_shocks_keep_valuation_rules_separate(
    name: str, exposures: dict[str, float], factor: str
) -> None:
    holding = position(exposures).model_copy(update={"asset_type": AssetType.BOND})
    shock_parts = (
        scenario("RATE-USD", 100, ShockType.BASIS_POINT, ShockUnit.BASIS_POINT),
        scenario("CREDIT-SPREAD", 25, ShockType.BASIS_POINT, ShockUnit.BASIS_POINT),
        scenario("DEFAULT", 0.25, ShockType.DEFAULT),
    )
    joint = shock_parts[0].model_copy(
        update={"shocks": tuple(part.shocks[0] for part in shock_parts)}
    )
    expected = {"RATE-USD": "-490", "CREDIT-SPREAD": "-1250", "DEFAULT": "-1200"}
    snapshot = market(factor)
    before = holding.model_dump(), snapshot.model_dump(), joint.model_dump()
    assert getattr(valuation, name)().stressed_pnl(holding, snapshot, joint) == Decimal(
        expected[factor]
    )
    assert before == (holding.model_dump(), snapshot.model_dump(), joint.model_dump())


@pytest.mark.parametrize(
    "name,exposures",
    [
        ("DurationConvexityValuationAdapter", {}),
        ("DurationConvexityValuationAdapter", {"duration:RATE-USD": 5}),
        ("DurationConvexityValuationAdapter", {"convexity:RATE-USD": 20}),
        ("DurationConvexityValuationAdapter", {"duration:RATE-USD": 5, "convexity:RATE-EUR": 20}),
        (
            "DurationConvexityValuationAdapter",
            {"duration:CREDIT-SPREAD": 5, "convexity:CREDIT-SPREAD": 20},
        ),
        ("DurationConvexityValuationAdapter", {"duration:RATE-FAKE": 5, "convexity:RATE-FAKE": 20}),
        ("CreditSpreadValuationAdapter", {}),
        ("CreditSpreadValuationAdapter", {"cs01:RATE-USD": -50}),
        ("CreditSpreadValuationAdapter", {"CREDIT-SPREAD": -50}),
        ("DefaultValuationAdapter", {"ead": 12000}),
        ("DefaultValuationAdapter", {"lgd": 0.4}),
        ("DefaultValuationAdapter", {"ead": -1, "lgd": 0.4}),
        ("DefaultValuationAdapter", {"ead": 12000, "lgd": -0.1}),
        ("DefaultValuationAdapter", {"ead": 12000, "lgd": 1.1}),
        ("DefaultValuationAdapter", {"ead": 12000, "lgd": 0.4, "recovery": 0.6}),
    ],
)
def test_rejects_incomplete_or_inconsistent_credit_inputs(
    name: str, exposures: dict[str, float]
) -> None:
    holding = position(exposures).model_copy(update={"asset_type": AssetType.BOND})
    with pytest.raises(ValueError):
        getattr(valuation, name)().value(holding, market("RATE-USD"))


@pytest.mark.parametrize(
    "name",
    [
        "DurationConvexityValuationAdapter",
        "CreditSpreadValuationAdapter",
        "DefaultValuationAdapter",
    ],
)
def test_credit_adapters_reject_other_asset_types(name: str) -> None:
    with pytest.raises(ValueError):
        getattr(valuation, name)().value(position({"ead": 1, "lgd": 1}), market("DEFAULT"))


@pytest.mark.parametrize(
    "factor,value,unit,kind",
    [
        ("DEFAULT", -0.1, ShockUnit.DECIMAL, ShockType.DEFAULT),
        ("DEFAULT", 1.1, ShockUnit.DECIMAL, ShockType.DEFAULT),
        ("DEFAULT", 0.1, ShockUnit.PERCENT, ShockType.DEFAULT),
        ("DEFAULT", 0.1, ShockUnit.DECIMAL, ShockType.RELATIVE),
    ],
)
def test_rejects_corrupt_normalized_default_shocks(
    factor: str, value: float, unit: ShockUnit, kind: ShockType
) -> None:
    valid = scenario("DEFAULT", 0.1, ShockType.DEFAULT)
    corrupt: NormalizedScenario = valid.model_copy(
        update={
            "shocks": (
                valid.shocks[0].model_copy(
                    update={"factor_id": factor, "value": value, "unit": unit, "shock_type": kind}
                ),
            )
        }
    )
    holding = position({"ead": 100, "lgd": 0.4}).model_copy(update={"asset_type": AssetType.LOAN})
    with pytest.raises(ValueError):
        valuation.DefaultValuationAdapter().stressed_pnl(
            holding, market("DEFAULT"), corrupt
        )
