"""Stress Test accounting and reproducibility properties."""

from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from risk_engine.domain import AttributionDimension
from risk_engine.stress.module import StressEngine
from tests.stress.test_linear_valuation import market, position, scenario
from tests.stress.test_stress_module import portfolio


@given(
    first=st.integers(-10000, 10000), second=st.integers(-10000, 10000), move=st.integers(-100, 100)
)
def test_signed_accounting_and_each_attribution_dimension_reconcile(
    first: int,
    second: int,
    move: int,
) -> None:
    holdings = portfolio(
        position({"EQUITY-US": 1}, notional=str(first)),
        position({"EQUITY-US": 0.5}, notional=str(second)).model_copy(
            update={"position_id": "second"}
        ),
    )
    shock = scenario("EQUITY-US", move / 100)
    result = StressEngine().run(holdings, market("EQUITY-US"), shock)
    assert result.base_value == first + second
    assert result.absolute_loss == result.base_value - result.stressed_value
    for dimension in (
        AttributionDimension.ASSET,
        AttributionDimension.SECTOR,
        AttributionDimension.REGION,
        AttributionDimension.OBLIGOR,
        AttributionDimension.FACTOR,
    ):
        assert sum(r.loss for r in result.attribution if r.dimension is dimension) == (
            result.absolute_loss
        )
    assert result.percentage_loss == (
        float(result.absolute_loss / result.base_value) if result.base_value else 0
    )
    reversed_holdings = holdings.model_copy(update={"positions": holdings.positions[::-1]})
    assert result == StressEngine().run(reversed_holdings, market("EQUITY-US"), shock)


@given(notional=st.integers(-100000, 100000), weight=st.integers(-100, 100))
def test_zero_shock_and_immutable_inputs(notional: int, weight: int) -> None:
    holdings = portfolio(position({"EQUITY-US": weight}, notional=str(notional)))
    snapshot = market("EQUITY-US")
    shock = scenario("EQUITY-US", 0)
    before = holdings.model_dump(), snapshot.model_dump(), shock.model_dump()
    result = StressEngine().run(holdings, snapshot, shock)
    assert result.base_value == result.stressed_value == Decimal(notional)
    assert result.absolute_loss == 0
    assert result.valuation_coverage == 1
    assert before == (holdings.model_dump(), snapshot.model_dump(), shock.model_dump())
