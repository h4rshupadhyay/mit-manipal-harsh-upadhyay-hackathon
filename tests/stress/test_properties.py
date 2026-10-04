"""Stress Test accounting and reproducibility properties."""

from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Decimal,
    Inexact,
    getcontext,
    localcontext,
)

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


def test_replay_is_independent_of_caller_decimal_precision_and_rounding() -> None:
    holdings = portfolio(
        position({"delta:EQUITY-US": 0.12345678901234567}, derivative=True, notional="0")
    )
    snapshot = market("EQUITY-US")
    shock = scenario("EQUITY-US", 0.12345678901234567)
    results = []
    for precision, rounding in ((6, ROUND_FLOOR), (28, ROUND_HALF_EVEN), (60, ROUND_CEILING)):
        with localcontext() as caller:
            caller.prec = precision
            caller.rounding = rounding
            result = StressEngine().run(holdings, snapshot, shock)
            assert getcontext().prec == precision
            assert getcontext().rounding == rounding
            results.append(result)
    assert len({r.stress_result_id for r in results}) == 1
    assert len({r.model_dump_json() for r in results}) == 1


def test_stress_result_validation_and_flags_use_only_the_local_context() -> None:
    holdings = portfolio(position({"delta:EQUITY-US": 50.123456}, derivative=True))
    snapshot = market("EQUITY-US")
    shock = scenario("EQUITY-US", -0.1)
    expected = StressEngine().run(holdings, snapshot, shock)
    with localcontext() as caller:
        caller.prec = 6
        caller.rounding = ROUND_FLOOR
        caller.traps[Inexact] = True
        caller.clear_flags()
        before = caller.copy()
        result = StressEngine().run(holdings, snapshot, shock)
        assert result.model_dump_json() == expected.model_dump_json()
        assert caller.prec == before.prec
        assert caller.rounding == before.rounding
        assert caller.traps == before.traps
        assert caller.flags == before.flags
