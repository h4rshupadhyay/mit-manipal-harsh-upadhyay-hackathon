"""Immutable, normalized Joint Shock Vectors consumed by valuation adapters."""

from types import MappingProxyType
from typing import Annotated

from pydantic import Field

from risk_engine.domain import (
    DomainModel,
    NonEmptyString,
    PositiveDays,
    ProvenanceMethod,
    ShockType,
    ShockUnit,
)


class FactorDefinition(DomainModel):
    """Explicit factor forms and native units in the initial versioned catalogue."""

    native_unit: ShockUnit
    shock_types: tuple[ShockType, ...]


NORMALIZATION_VERSION = "stress-normalization-v1"
_EQUITY = FactorDefinition(
    native_unit=ShockUnit.INDEX_POINT, shock_types=(ShockType.RELATIVE, ShockType.ABSOLUTE)
)
_FX = FactorDefinition(
    native_unit=ShockUnit.CURRENCY, shock_types=(ShockType.RELATIVE, ShockType.ABSOLUTE)
)
_BASIS_POINT = FactorDefinition(
    native_unit=ShockUnit.BASIS_POINT, shock_types=(ShockType.BASIS_POINT,)
)
# Extend and version this explicit registry as new calibration factors are added.
FACTOR_REGISTRY = MappingProxyType({
    "EQUITY-INDIA": _EQUITY,
    "EQUITY-US": _EQUITY,
    "EQUITY-EUROPE": _EQUITY,
    "EQUITY-ASIA": _EQUITY,
    "USD-INR": _FX,
    "EUR-USD": _FX,
    "COMMODITY": FactorDefinition(
        native_unit=ShockUnit.ABSOLUTE, shock_types=(ShockType.RELATIVE, ShockType.ABSOLUTE)
    ),
    "RATE-USD": _BASIS_POINT,
    "RATE-EUR": _BASIS_POINT,
    "RATE-INR": _BASIS_POINT,
    "CREDIT-SPREAD": _BASIS_POINT,
    "VOLATILITY": FactorDefinition(
        native_unit=ShockUnit.VOLATILITY_POINT, shock_types=(ShockType.VOLATILITY,)
    ),
    "DEFAULT": FactorDefinition(native_unit=ShockUnit.DECIMAL, shock_types=(ShockType.DEFAULT,)),
})


class NormalizedShock(DomainModel):
    """Relative, default, rate, spread, and volatility moves use decimal units.

    A basis-point move of 100 becomes 0.01; a volatility-point move of 1
    becomes 0.01. Absolute moves retain their explicit native unit. The
    original shock type distinguishes sensitivity conventions (e.g. CS01
    adapters multiply a decimal spread move by 10,000).
    """

    factor_id: NonEmptyString
    shock_type: ShockType
    value: float
    unit: ShockUnit
    horizon_days: PositiveDays


class NormalizedScenario(DomainModel):
    """Validated scenario with preserved provenance and conversion version."""

    scenario_id: NonEmptyString
    risk_signal_id: NonEmptyString
    shocks: Annotated[tuple[NormalizedShock, ...], Field(min_length=1)]
    horizon_days: PositiveDays
    method: ProvenanceMethod
    reference_event_ids: tuple[NonEmptyString, ...]
    calibration_version: NonEmptyString
    analyst_override: bool
    override_reason: NonEmptyString | None
    normalization_version: NonEmptyString
