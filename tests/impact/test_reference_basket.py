"""Frozen Reference Basket construction over synthetic factor returns."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from risk_engine.domain import (
    AssetType,
    MarketObservation,
    MarketSnapshot,
    Portfolio,
    Position,
    ShockUnit,
)
from risk_engine.impact.reference_basket import (
    BasketConstruction,
    ReferenceBasketBuilder,
    ReferenceBasketSpec,
    ReferenceConstituent,
    ReferenceReturnRow,
)
from risk_engine.stress.module import StressEngine
from tests.stress.test_linear_valuation import scenario

CALIBRATION = datetime(2025, 1, 10, tzinfo=timezone.utc)


def constituent(position_id: str, factor: str) -> ReferenceConstituent:
    return ReferenceConstituent(
        factor_proxy_id=factor,
        position=Position(
            position_id=position_id,
            asset_type=AssetType.EQUITY,
            notional=Decimal("100"),
            currency="USD",
            sector="test-sector",
            region="test-region",
            obligor="test-obligor",
            factor_exposures={factor: 1.0},
        ),
    )


def spec(construction: BasketConstruction) -> ReferenceBasketSpec:
    return ReferenceBasketSpec(
        basket_id="reference-v1",
        basket_version="v1",
        construction=construction,
        constituents=(constituent("us", "EQUITY-US"), constituent("india", "EQUITY-INDIA")),
        return_snapshot_id="synthetic-returns-v1",
        return_source_terms="project-authored synthetic data",
        calibration_date=CALIBRATION,
        lookback_observations=3,
        covariance_estimator="sample-ddof-1",
        constraint="long-only-fully-invested",
        recalibration_rule="frozen-until-new-version",
        total_notional=Decimal("1000"),
        valuation_currency="USD",
    )


def rows() -> tuple[ReferenceReturnRow, ...]:
    return tuple(
        ReferenceReturnRow(
            observed_at=CALIBRATION - timedelta(days=3 - i),
            factor_returns={"EQUITY-US": us, "EQUITY-INDIA": india},
        )
        for i, (us, india) in enumerate(((-0.01, -0.02), (0.0, 0.0), (0.01, 0.02)))
    )


def test_equal_notional_allocates_equal_positions_and_freezes_covariance() -> None:
    basket = ReferenceBasketBuilder().build(rows(), spec(BasketConstruction.EQUAL_NOTIONAL))
    assert basket.portfolio_id == "reference-v1"
    assert basket.as_of == CALIBRATION
    assert [p.notional for p in basket.positions] == [Decimal("500"), Decimal("500")]
    manifest = json.loads(basket.version)
    assert manifest["basket_version"] == "v1"
    assert manifest["factor_proxy_ids"] == ["EQUITY-US", "EQUITY-INDIA"]
    assert manifest["covariance"] == [[0.0001, 0.0002], [0.0002, 0.0004]]
    assert manifest["lookback_observations"] == 3
    assert manifest["return_snapshot_id"] == "synthetic-returns-v1"
    assert manifest["return_source_terms"] == "project-authored synthetic data"


def test_inverse_volatility_allocates_more_to_less_volatile_proxy() -> None:
    basket = ReferenceBasketBuilder().build(rows(), spec(BasketConstruction.INVERSE_VOLATILITY))
    assert [float(p.notional) for p in basket.positions] == pytest.approx(
        [1000 * 2 / 3, 1000 / 3]
    )


def test_equal_risk_contribution_equalizes_correlated_proxy_risks() -> None:
    candidate = spec(BasketConstruction.EQUAL_RISK_CONTRIBUTION).model_copy(
        update={
            "constituents": (
                constituent("us", "EQUITY-US"),
                constituent("india", "EQUITY-INDIA"),
                constituent("europe", "EQUITY-EUROPE"),
            ),
            "lookback_observations": 4,
        }
    )
    observations = tuple(
        ReferenceReturnRow(
            observed_at=CALIBRATION - timedelta(days=4 - i),
            factor_returns={
                "EQUITY-US": a,
                "EQUITY-INDIA": b,
                "EQUITY-EUROPE": a + b,
            },
        )
        for i, (a, b) in enumerate(((-0.01, -0.02), (-0.01, 0.02), (0.01, -0.02), (0.01, 0.02)))
    )
    basket = ReferenceBasketBuilder().build(observations, candidate)
    weights = [float(p.notional / candidate.total_notional) for p in basket.positions]
    # Hand-derived covariance is 4/3 times [[.0001, 0, .0001],
    # [0, .0004, .0004], [.0001, .0004, .0005]].
    base = ((1, 0, 1), (0, 4, 4), (1, 4, 5))
    marginal = [sum(base[i][j] * weights[j] for j in range(3)) for i in range(3)]
    contributions = [weights[i] * marginal[i] for i in range(3)]
    assert max(contributions) - min(contributions) < 1e-5
    assert all(weight > 0 for weight in weights)


def test_version_fingerprint_changes_with_constituent_or_return_data() -> None:
    builder = ReferenceBasketBuilder()
    candidate = spec(BasketConstruction.EQUAL_NOTIONAL)
    baseline = builder.build(rows(), candidate)
    changed_return = list(rows())
    changed_return[0] = changed_return[0].model_copy(
        update={"factor_returns": {"EQUITY-US": -0.03, "EQUITY-INDIA": -0.02}}
    )
    assert builder.build(tuple(changed_return), candidate).version != baseline.version
    changed_position = candidate.constituents[0].position.model_copy(update={"sector": "other"})
    changed_constituent = candidate.constituents[0].model_copy(
        update={"position": changed_position}
    )
    changed_spec = candidate.model_copy(
        update={"constituents": (changed_constituent, candidate.constituents[1])}
    )
    assert builder.build(rows(), changed_spec).version != baseline.version


def test_allocations_scale_only_currency_valued_sensitivities() -> None:
    candidate = spec(BasketConstruction.EQUAL_NOTIONAL)
    derivative = candidate.constituents[0].position.model_copy(
        update={
            "asset_type": AssetType.DERIVATIVE,
            "factor_exposures": {"delta:EQUITY-US": 10.0, "dv01:RATE-USD": -2.0},
        }
    )
    bond = candidate.constituents[1].position.model_copy(
        update={
            "asset_type": AssetType.BOND,
            "factor_exposures": {
                "duration:RATE-USD": 5.0,
                "convexity:RATE-USD": 20.0,
                "cs01:CREDIT-SPREAD": -2.0,
                "ead": 80.0,
                "lgd": 0.4,
            },
        }
    )
    candidate = candidate.model_copy(
        update={
            "constituents": (
                candidate.constituents[0].model_copy(update={"position": derivative}),
                candidate.constituents[1].model_copy(update={"position": bond}),
            )
        }
    )
    basket = ReferenceBasketBuilder().build(rows(), candidate)
    assert basket.positions[0].factor_exposures == {
        "delta:EQUITY-US": 50.0,
        "dv01:RATE-USD": -10.0,
    }
    assert basket.positions[1].factor_exposures == {
        "duration:RATE-USD": 5.0,
        "convexity:RATE-USD": 20.0,
        "cs01:CREDIT-SPREAD": -10.0,
        "ead": 400.0,
        "lgd": 0.4,
    }
    assert derivative.factor_exposures["delta:EQUITY-US"] == 10.0


def test_reference_basket_is_stress_engine_portfolio_independent_of_synthetic_holdings() -> None:
    basket = ReferenceBasketBuilder().build(rows(), spec(BasketConstruction.EQUAL_NOTIONAL))
    synthetic = Portfolio(
        portfolio_id="synthetic-portfolio",
        positions=(
            constituent("synthetic-position", "EQUITY-US").position.model_copy(
                update={"notional": Decimal("999999")}
            ),
        ),
        valuation_currency="USD",
        as_of=CALIBRATION,
        version="synthetic-v1",
    )
    market = MarketSnapshot(
        snapshot_id="market-v1",
        as_of=CALIBRATION,
        observations=tuple(
            MarketObservation(
                factor_id=factor,
                observed_at=CALIBRATION,
                value=100,
                unit=ShockUnit.INDEX_POINT,
                provider="synthetic",
                vintage="v1",
            )
            for factor in ("EQUITY-US", "EQUITY-INDIA")
        ),
        provider="synthetic",
        schema_version="v1",
        normalization_version="v1",
    )
    shock = scenario("EQUITY-US", -0.1)
    reference_loss = StressEngine().run(basket, market, shock).absolute_loss
    synthetic_loss = StressEngine().run(synthetic, market, shock).absolute_loss
    assert reference_loss == Decimal("50")
    assert synthetic_loss == Decimal("99999.9")
    assert ReferenceBasketBuilder().build(rows(), spec(BasketConstruction.EQUAL_NOTIONAL)) == basket


def test_future_return_row_is_rejected_before_freezing_version() -> None:
    observations = list(rows())
    observations[-1] = observations[-1].model_copy(
        update={"observed_at": CALIBRATION + timedelta(days=1)}
    )
    with pytest.raises(ValueError, match="future return row"):
        ReferenceBasketBuilder().build(tuple(observations), spec(BasketConstruction.EQUAL_NOTIONAL))


def test_equal_notional_preserves_zero_variance_proxy_without_dividing_by_volatility() -> None:
    observations = tuple(
        row.model_copy(update={"factor_returns": {**row.factor_returns, "EQUITY-US": 0.01}})
        for row in rows()
    )
    basket = ReferenceBasketBuilder().build(observations, spec(BasketConstruction.EQUAL_NOTIONAL))
    assert [p.notional for p in basket.positions] == [Decimal("500"), Decimal("500")]
    assert json.loads(basket.version)["covariance"][0][0] == 0
