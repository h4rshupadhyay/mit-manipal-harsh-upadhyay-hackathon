"""Offline deterministic valuation in each Position's currency."""

from abc import ABC, abstractmethod
from decimal import Decimal

from risk_engine.domain import AssetType, MarketSnapshot, Position, ShockType, ShockUnit
from risk_engine.stress.interfaces import (
    FACTOR_REGISTRY,
    NORMALIZATION_VERSION,
    NormalizedScenario,
    NormalizedShock,
)

_SPOT_FACTORS = frozenset(
    factor
    for factor, definition in FACTOR_REGISTRY.items()
    if ShockType.RELATIVE in definition.shock_types
)


def _market_level(market: MarketSnapshot, factor: str) -> Decimal:
    """Require an explicit catalogue-native level, never refresh a snapshot."""
    for observation in market.observations:
        if observation.factor_id == factor:
            if observation.unit is not FACTOR_REGISTRY[factor].native_unit:
                raise ValueError(f"incompatible market unit for factor: {factor}")
            level = Decimal(str(observation.value))
            if factor in _SPOT_FACTORS and level <= 0:
                raise ValueError(f"spot market level must be positive: {factor}")
            return level
    raise ValueError(f"missing market factor: {factor}")


def _shocks(scenario: NormalizedScenario) -> dict[str, NormalizedShock]:
    if not isinstance(scenario, NormalizedScenario):
        raise TypeError("valuation requires a NormalizedScenario")
    validated = NormalizedScenario.model_validate(scenario.model_dump())
    if validated.normalization_version != NORMALIZATION_VERSION:
        raise ValueError("unsupported scenario normalization version")
    result: dict[str, NormalizedShock] = {}
    for shock in validated.shocks:
        definition = FACTOR_REGISTRY.get(shock.factor_id)
        if definition is None or shock.shock_type not in definition.shock_types:
            raise ValueError(f"unsupported normalized shock: {shock.factor_id}")
        expected = (
            definition.native_unit if shock.shock_type is ShockType.ABSOLUTE else ShockUnit.DECIMAL
        )
        if shock.unit is not expected:
            raise ValueError(f"incompatible normalized shock unit: {shock.factor_id}")
        if shock.factor_id in result or shock.horizon_days != validated.horizon_days:
            raise ValueError("normalized shocks require unique factors and consistent horizon")
        if shock.shock_type is ShockType.RELATIVE and shock.value < -1:
            raise ValueError("relative shock cannot be below -100%")
        if shock.shock_type is ShockType.DEFAULT and not 0 <= shock.value <= 1:
            raise ValueError("default shock must be a fraction in [0, 1]")
        result[shock.factor_id] = shock
    return result


def _relative_move(shock: NormalizedShock, level: Decimal) -> Decimal:
    move = Decimal(str(shock.value))
    if shock.shock_type is ShockType.ABSOLUTE:
        move /= level
    if move < -1:
        raise ValueError("spot shock cannot imply a negative market level")
    return move


class SpotValuationAdapter:
    """Equity, commodity and FX spot approximation.

    Position.notional is signed base market value, in Position.currency.
    factor_exposures maps spot factor IDs to signed dimensionless weights;
    P&L is notional * weight * relative move, summed across exposed factors.
    Absolute moves become relative moves using the snapshot's native level.
    No quantity, live price, currency conversion, or implicit exposure is inferred.
    """

    def value(self, position: Position, market: MarketSnapshot) -> Decimal:
        position = Position.model_validate(position.model_dump())
        market = MarketSnapshot.model_validate(market.model_dump())
        if position.asset_type not in {AssetType.EQUITY, AssetType.OTHER}:
            raise ValueError("spot adapter requires an equity or other spot position")
        if not position.factor_exposures:
            raise ValueError("spot position requires explicit factor exposures")
        for factor in position.factor_exposures:
            if factor not in _SPOT_FACTORS:
                raise ValueError(f"unsupported spot factor: {factor}")
            _market_level(market, factor)
        return position.notional

    def stressed_pnl(
        self,
        position: Position,
        market: MarketSnapshot,
        scenario: NormalizedScenario,
    ) -> Decimal:
        base = self.value(position, market)
        shocks = _shocks(scenario)
        pnl = Decimal(0)
        for factor, weight in position.factor_exposures.items():
            if factor in shocks:
                move = _relative_move(shocks[factor], _market_level(market, factor))
                pnl += base * Decimal(str(weight)) * move
        return pnl

    def stressed_value(
        self,
        position: Position,
        market: MarketSnapshot,
        scenario: NormalizedScenario,
    ) -> Decimal:
        return self.value(position, market) + self.stressed_pnl(position, market, scenario)


class LinearDerivativeValuationAdapter:
    """First-order supplied sensitivities, with signed base value in notional.

    factor_exposures keys are delta:<spot factor> or dv01:<rate factor>.
    Delta is signed Position.currency per native factor unit; multiply it by
    the absolute level change (relative shocks use the snapshot level).
    DV01 is signed Position.currency per +1 basis point; multiply it by the
    normalized decimal rate move * 10,000. A conventional long-duration
    exposure therefore supplies a negative DV01. Sensitivities already include
    position size and sign, and are never multiplied by notional again.
    No foreign-currency conversion or unsupported sensitivity is inferred.
    """

    def value(self, position: Position, market: MarketSnapshot) -> Decimal:
        position = Position.model_validate(position.model_dump())
        market = MarketSnapshot.model_validate(market.model_dump())
        if position.asset_type is not AssetType.DERIVATIVE:
            raise ValueError("linear adapter requires a derivative position")
        if not position.factor_exposures:
            raise ValueError("linear position requires explicit sensitivities")
        for key in position.factor_exposures:
            kind, separator, factor = key.partition(":")
            if (
                not separator
                or (kind == "delta" and factor not in _SPOT_FACTORS)
                or (kind == "dv01" and not factor.startswith("RATE-"))
                or kind not in {"delta", "dv01"}
                or factor not in FACTOR_REGISTRY
            ):
                raise ValueError(f"unsupported or ambiguous linear sensitivity: {key}")
            _market_level(market, factor)
        return position.notional

    def stressed_pnl(
        self,
        position: Position,
        market: MarketSnapshot,
        scenario: NormalizedScenario,
    ) -> Decimal:
        self.value(position, market)
        shocks = _shocks(scenario)
        pnl = Decimal(0)
        for key, sensitivity in position.factor_exposures.items():
            kind, _, factor = key.partition(":")
            shock = shocks.get(factor)
            if shock is None:
                continue
            move = Decimal(str(shock.value))
            if kind == "dv01":
                move *= 10_000
            else:
                level = _market_level(market, factor)
                # Validate the stressed level for both relative and absolute moves.
                _relative_move(shock, level)
                if shock.shock_type is ShockType.RELATIVE:
                    move *= level
            pnl += Decimal(str(sensitivity)) * move
        return pnl

    def stressed_value(
        self,
        position: Position,
        market: MarketSnapshot,
        scenario: NormalizedScenario,
    ) -> Decimal:
        return self.value(position, market) + self.stressed_pnl(position, market, scenario)


class _FixedIncomeValuationAdapter(ABC):
    def value(self, position: Position, market: MarketSnapshot) -> Decimal:
        position = Position.model_validate(position.model_dump())
        MarketSnapshot.model_validate(market.model_dump())
        if position.asset_type not in {AssetType.BOND, AssetType.LOAN}:
            raise ValueError("fixed-income adapter requires a bond or loan position")
        return position.notional

    @abstractmethod
    def stressed_pnl(
        self, position: Position, market: MarketSnapshot, scenario: NormalizedScenario
    ) -> Decimal:
        """Calculate one explicit rule's contribution in Position.currency."""

    def stressed_value(
        self, position: Position, market: MarketSnapshot, scenario: NormalizedScenario
    ) -> Decimal:
        return self.value(position, market) + self.stressed_pnl(position, market, scenario)


class DurationConvexityValuationAdapter(_FixedIncomeValuationAdapter):
    """Risk-free yield approximation for fixed-rate bonds and loans.

    Notional is signed base market value in Position.currency. Each rate
    factor requires both duration:<RATE-factor> in years and
    convexity:<RATE-factor> in years squared, including explicit zero values.
    P&L = notional * (-duration * dy + 0.5 * convexity * dy squared), summed
    across factors; dy is the normalized decimal risk-free yield change.
    Credit spread and accounting default losses are separate adapter rules.
    """

    def value(self, position: Position, market: MarketSnapshot) -> Decimal:
        base = super().value(position, market)
        exposures = position.factor_exposures
        if not exposures:
            raise ValueError("duration/convexity requires explicit paired sensitivities")
        for key in exposures:
            kind, _, factor = key.partition(":")
            if (
                kind not in {"duration", "convexity"}
                or not factor.startswith("RATE-")
                or factor not in FACTOR_REGISTRY
                or f"duration:{factor}" not in exposures
                or f"convexity:{factor}" not in exposures
            ):
                raise ValueError(f"unsupported or unpaired duration/convexity sensitivity: {key}")
            _market_level(market, factor)
        return base

    def stressed_pnl(
        self, position: Position, market: MarketSnapshot, scenario: NormalizedScenario
    ) -> Decimal:
        base = self.value(position, market)
        shocks = _shocks(scenario)
        pnl = Decimal(0)
        for key, sensitivity in position.factor_exposures.items():
            kind, _, factor = key.partition(":")
            if kind != "duration" or factor not in shocks:
                continue
            dy = Decimal(str(shocks[factor].value))
            duration = Decimal(str(sensitivity))
            convexity = Decimal(str(position.factor_exposures[f"convexity:{factor}"]))
            pnl += base * (-duration * dy + Decimal("0.5") * convexity * dy * dy)
        return pnl


class CreditSpreadValuationAdapter(_FixedIncomeValuationAdapter):
    """Signed supplied CS01, independently of base market value.

    The sole key cs01:CREDIT-SPREAD is signed Position.currency per +1 bp;
    a conventional long credit exposure supplies negative CS01. Sensitivity
    includes position size and sign: P&L = CS01 * normalized spread * 10,000.
    """

    def value(self, position: Position, market: MarketSnapshot) -> Decimal:
        base = super().value(position, market)
        if set(position.factor_exposures) != {"cs01:CREDIT-SPREAD"}:
            raise ValueError("credit spread requires explicit cs01:CREDIT-SPREAD only")
        _market_level(market, "CREDIT-SPREAD")
        return base

    def stressed_pnl(
        self, position: Position, market: MarketSnapshot, scenario: NormalizedScenario
    ) -> Decimal:
        self.value(position, market)
        shocks = _shocks(scenario)
        shock = shocks.get("CREDIT-SPREAD")
        if shock is None:
            return Decimal(0)
        return (
            Decimal(str(position.factor_exposures["cs01:CREDIT-SPREAD"]))
            * Decimal(str(shock.value))
            * 10_000
        )


class DefaultValuationAdapter(_FixedIncomeValuationAdapter):
    """Accounting default loss, independently of yield and spread movements.

    Exact keys ead (nonnegative Position.currency) and lgd (fraction [0, 1])
    explicitly supply exposure at default and loss given default net of
    recovery/collateral. No recovery or collateral assumption is inferred.
    P&L = -EAD * LGD * normalized DEFAULT fraction. EAD already includes
    exposure size; notional remains base value and does not scale the loss.
    Default valuation needs no market price observation.
    """

    def value(self, position: Position, market: MarketSnapshot) -> Decimal:
        base = super().value(position, market)
        exposures = position.factor_exposures
        if set(exposures) != {"ead", "lgd"}:
            raise ValueError("default valuation requires explicit ead and lgd only")
        if exposures["ead"] < 0 or not 0 <= exposures["lgd"] <= 1:
            raise ValueError("EAD must be nonnegative and LGD must be in [0, 1]")
        return base

    def stressed_pnl(
        self, position: Position, market: MarketSnapshot, scenario: NormalizedScenario
    ) -> Decimal:
        self.value(position, market)
        shocks = _shocks(scenario)
        shock = shocks.get("DEFAULT")
        if shock is None:
            return Decimal(0)
        return (
            -Decimal(str(position.factor_exposures["ead"]))
            * Decimal(str(position.factor_exposures["lgd"]))
            * Decimal(str(shock.value))
        )
