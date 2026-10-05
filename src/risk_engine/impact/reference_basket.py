"""Deterministic, versioned Reference Baskets for portfolio-independent severity."""

import hashlib
import json
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from pydantic import AwareDatetime, Field, model_validator
from scipy.optimize import minimize  # type: ignore[import-untyped]

from risk_engine.domain import (
    AssetType,
    CurrencyCode,
    DomainModel,
    NonEmptyString,
    Portfolio,
    Position,
    ShockType,
    ShockUnit,
)
from risk_engine.stress.interfaces import FACTOR_REGISTRY


class BasketConstruction(str, Enum):
    EQUAL_RISK_CONTRIBUTION = "equal-risk-contribution"
    EQUAL_NOTIONAL = "equal-notional"
    INVERSE_VOLATILITY = "inverse-volatility"


class ReferenceConstituent(DomainModel):
    """Position prototype and dimensionless return proxy for one constituent."""

    factor_proxy_id: NonEmptyString
    position: Position


class ReferenceReturnRow(DomainModel):
    """Simultaneous dimensionless proxy returns observed at one aware instant."""

    observed_at: AwareDatetime
    unit: Literal[ShockUnit.DECIMAL]
    factor_returns: dict[NonEmptyString, float]


class ReferenceBasketSpec(DomainModel):
    basket_id: NonEmptyString
    basket_version: NonEmptyString
    construction: BasketConstruction
    constituents: Annotated[tuple[ReferenceConstituent, ...], Field(min_length=2)]
    return_snapshot_id: NonEmptyString
    return_source_terms: NonEmptyString
    calibration_date: AwareDatetime
    lookback_observations: Annotated[int, Field(ge=3)]
    covariance_estimator: NonEmptyString
    constraint: NonEmptyString
    recalibration_rule: NonEmptyString
    total_notional: Annotated[Decimal, Field(gt=0)]
    valuation_currency: CurrencyCode

    @model_validator(mode="after")
    def validate_constituents(self) -> "ReferenceBasketSpec":
        ids = [c.position.position_id for c in self.constituents]
        proxies = [c.factor_proxy_id for c in self.constituents]
        if len(set(ids)) != len(ids) or len(set(proxies)) != len(proxies):
            raise ValueError("constituent position IDs and factor proxies must be unique")
        if self.covariance_estimator != "sample-ddof-1":
            raise ValueError("unsupported covariance estimator")
        if self.constraint != "long-only-fully-invested":
            raise ValueError("unsupported basket constraint")
        if self.recalibration_rule != "frozen-until-new-version":
            raise ValueError("unsupported recalibration rule")
        if any(c.position.currency != self.valuation_currency for c in self.constituents):
            raise ValueError("constituent currency must match basket valuation currency")
        return self


class ReferenceBasketBuilder:
    """Build a Portfolio from a declared, frozen return window."""

    def build(
        self, returns: tuple[ReferenceReturnRow, ...], spec: ReferenceBasketSpec
    ) -> Portfolio:
        spec = ReferenceBasketSpec.model_validate(spec.model_dump())
        rows = tuple(ReferenceReturnRow.model_validate(row.model_dump()) for row in returns)
        if len(rows) != spec.lookback_observations:
            raise ValueError("returns must exactly match the declared lookback")
        if any(row.observed_at > spec.calibration_date for row in rows):
            raise ValueError("future return row after calibration date")
        if list(row.observed_at for row in rows) != sorted(row.observed_at for row in rows):
            raise ValueError("return rows must be chronologically ordered")
        if len({row.observed_at for row in rows}) != len(rows):
            raise ValueError("return row timestamps must be unique")
        proxies = [c.factor_proxy_id for c in spec.constituents]
        if any(set(row.factor_returns) != set(proxies) for row in rows):
            raise ValueError("return row proxies must exactly match constituents")
        values = np.array([[row.factor_returns[factor] for factor in proxies] for row in rows])
        covariance = np.cov(values, rowvar=False, ddof=1)
        if spec.construction is BasketConstruction.INVERSE_VOLATILITY:
            volatilities = np.sqrt(np.diag(covariance))
            if np.any(volatilities <= 0):
                raise ValueError("inverse-volatility construction requires positive volatility")
            inverse_volatilities = 1 / volatilities
            weights = inverse_volatilities / inverse_volatilities.sum()
        elif spec.construction is BasketConstruction.EQUAL_RISK_CONTRIBUTION:
            if np.any(np.diag(covariance) <= 0):
                raise ValueError("equal-risk construction requires positive volatility")
            weights = _equal_risk_weights(covariance)
        else:
            weights = np.full(len(proxies), 1 / len(proxies))
        allocations = [spec.total_notional * Decimal(str(weight)) for weight in weights[:-1]]
        allocations.append(spec.total_notional - sum(allocations, Decimal(0)))
        positions = tuple(
            _scaled_position(c.position, allocation)
            for c, allocation in zip(spec.constituents, allocations, strict=True)
        )
        rows_json = [row.model_dump(mode="json") for row in rows]
        row_bytes = json.dumps(rows_json, sort_keys=True, separators=(",", ":")).encode()
        spec_bytes = json.dumps(
            spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        manifest = {
            "basket_version": spec.basket_version,
            "construction": spec.construction.value,
            "factor_proxy_ids": proxies,
            "constituent_ids": [c.position.position_id for c in spec.constituents],
            "covariance": covariance.tolist(),
            "covariance_estimator": spec.covariance_estimator,
            "lookback_observations": spec.lookback_observations,
            "lookback_start": rows[0].observed_at.isoformat(),
            "lookback_end": rows[-1].observed_at.isoformat(),
            "calibration_date": spec.calibration_date.isoformat(),
            "constraint": spec.constraint,
            "recalibration_rule": spec.recalibration_rule,
            "return_snapshot_id": spec.return_snapshot_id,
            "return_source_terms": spec.return_source_terms,
            "return_rows_sha256": hashlib.sha256(row_bytes).hexdigest(),
            "construction_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
            "total_notional": str(spec.total_notional),
            "valuation_currency": spec.valuation_currency,
        }
        return Portfolio(
            portfolio_id=spec.basket_id,
            positions=positions,
            valuation_currency=spec.valuation_currency,
            as_of=spec.calibration_date,
            version=json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        )


def _equal_risk_weights(covariance: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Solve long-only equal variance contributions with deterministic SLSQP."""
    size = len(covariance)

    def objective(weights: npt.NDArray[np.float64]) -> float:
        marginal = covariance @ weights
        contributions = weights * marginal
        total_variance = float(weights @ marginal)
        if total_variance <= 0:
            return 1e12
        return float(np.square(contributions / total_variance - 1 / size).sum())

    start = 1 / np.sqrt(np.diag(covariance))
    start /= start.sum()
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(1e-10, 1.0)] * size,
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1)},
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    weights = np.asarray(result.x, dtype=float)
    if not result.success or np.any(weights <= 0) or abs(weights.sum() - 1) > 1e-8:
        raise ValueError("equal-risk-contribution solver did not converge")
    if objective(weights) > 1e-8:
        raise ValueError("equal-risk-contribution tolerance was not reached")
    return weights


def _scaled_position(prototype: Position, allocation: Decimal) -> Position:
    """Scale only sensitivities measured in currency; retain unitless parameters."""
    if prototype.notional <= 0:
        raise ValueError("long-only constituent prototype requires positive notional")
    if not prototype.factor_exposures:
        raise ValueError("constituent requires explicit supported factor exposures")
    spot = {
        factor
        for factor, definition in FACTOR_REGISTRY.items()
        if ShockType.RELATIVE in definition.shock_types
    }
    rate = {factor for factor in FACTOR_REGISTRY if factor.startswith("RATE-")}
    exposures = prototype.factor_exposures
    keys = set(exposures)
    if prototype.asset_type in {AssetType.EQUITY, AssetType.OTHER}:
        if not keys <= spot:
            raise ValueError("unsupported spot constituent exposure")
        dimensionful: set[str] = set()
    elif prototype.asset_type is AssetType.DERIVATIVE:
        delta_factors = {key.partition(":")[2] for key in keys if key.startswith("delta:")}
        nonlinear = any(key.startswith(("gamma:", "vega:")) for key in keys)
        if nonlinear:
            expected = {
                f"{kind}:{factor}" for factor in delta_factors for kind in ("delta", "gamma")
            } | {"vega:VOLATILITY"}
            if not delta_factors or not delta_factors <= spot or keys != expected:
                raise ValueError("unsupported or incomplete nonlinear constituent")
        elif not keys <= ({f"delta:{factor}" for factor in spot} | {f"dv01:{f}" for f in rate}):
            raise ValueError("unsupported linear derivative constituent")
        dimensionful = keys
    else:
        rate_keys = {
            f"{kind}:{factor}" for factor in rate for kind in ("duration", "convexity")
        }
        allowed = rate_keys | {"cs01:CREDIT-SPREAD", "ead", "lgd"}
        if not keys <= allowed:
            raise ValueError("unsupported fixed-income constituent exposure")
        for factor in rate:
            if (f"duration:{factor}" in keys) != (f"convexity:{factor}" in keys):
                raise ValueError("unpaired duration/convexity constituent")
        if ("ead" in keys) != ("lgd" in keys):
            raise ValueError("unpaired ead/lgd constituent")
        if "ead" in keys and (exposures["ead"] < 0 or not 0 <= exposures["lgd"] <= 1):
            raise ValueError("invalid ead/lgd constituent")
        dimensionful = keys & {"cs01:CREDIT-SPREAD", "ead"}
    ratio = allocation / prototype.notional
    scaled = {
        key: float(Decimal(str(value)) * ratio) if key in dimensionful else value
        for key, value in exposures.items()
    }
    copy = prototype.model_copy(update={"notional": allocation, "factor_exposures": scaled})
    return Position.model_validate(copy.model_dump())
