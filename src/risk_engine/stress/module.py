"""Deterministic offline Stress Engine with disclosed supported-value coverage."""

import hashlib
import json
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from types import MappingProxyType
from typing import Protocol

from risk_engine.domain import (
    AssetType,
    AttributionDimension,
    DomainModel,
    MarketSnapshot,
    NonEmptyString,
    Portfolio,
    Position,
    ProvenanceMethod,
    ShockType,
    StressAttribution,
    StressResult,
    StressResultVersions,
    StressScenario,
)
from risk_engine.stress.interfaces import FACTOR_REGISTRY, NormalizedScenario
from risk_engine.stress.shocks import normalize_scenario
from risk_engine.stress.valuation import (
    CreditSpreadValuationAdapter,
    DefaultValuationAdapter,
    DurationConvexityValuationAdapter,
    LinearDerivativeValuationAdapter,
    SpotValuationAdapter,
    _market_level,
    _relative_move,
    _shocks,
)


class _Adapter(Protocol):
    def value(self, position: Position, market: MarketSnapshot) -> Decimal: ...

    def stressed_pnl(
        self, position: Position, market: MarketSnapshot, scenario: NormalizedScenario
    ) -> Decimal: ...


class FactorRuleVersion(DomainModel):
    """Registered rule identity, never an inferred version default."""

    rule: NonEmptyString
    version: NonEmptyString


_SPOT = frozenset(f for f, d in FACTOR_REGISTRY.items() if ShockType.RELATIVE in d.shock_types)
_RATES = frozenset(f for f in FACTOR_REGISTRY if f.startswith("RATE-"))
FACTOR_RULE_VERSIONS = MappingProxyType(
    {
        factor: tuple(
            FactorRuleVersion(rule=rule, version=f"{rule}-v1")
            for rule in (
                ("spot", "linear", "delta-gamma-vega")
                if factor in _SPOT
                else ("duration-convexity", "linear")
                if factor in _RATES
                else ("credit-spread",)
                if factor == "CREDIT-SPREAD"
                else ("default",)
                if factor == "DEFAULT"
                else ("delta-gamma-vega",)
            )
        )
        for factor in FACTOR_REGISTRY
    }
)
ENGINE_VERSION = "stress-engine-v2"


def _view(position: Position, exposures: dict[str, float]) -> Position:
    return Position.model_validate(position.model_copy(update={"factor_exposures": exposures}))


def _rules(position: Position) -> list[tuple[_Adapter, Position]] | None:
    """Reject unknown/incomplete key sets before composing strict adapter views."""
    exposures = position.factor_exposures
    if not exposures:
        return None
    if position.asset_type in {AssetType.EQUITY, AssetType.OTHER}:
        if not set(exposures) <= _SPOT:
            return None
        return [(SpotValuationAdapter(), position)]
    if position.asset_type is AssetType.DERIVATIVE:
        if any(k.startswith(("gamma:", "vega:")) for k in exposures):
            return None  # Complete nonlinear sets are handled as one rule below.
        allowed = {f"delta:{f}" for f in _SPOT} | {f"dv01:{f}" for f in _RATES}
        if not set(exposures) <= allowed:
            return None
        return [(LinearDerivativeValuationAdapter(), position)]
    allowed = {f"{kind}:{f}" for f in _RATES for kind in ("duration", "convexity")} | {
        "cs01:CREDIT-SPREAD",
        "ead",
        "lgd",
    }
    if not set(exposures) <= allowed:
        return None
    rate_keys = {k for k in exposures if k.startswith(("duration:", "convexity:"))}
    if any(
        f"{kind}:{key.partition(':')[2]}" not in rate_keys
        for key in rate_keys
        for kind in ("duration", "convexity")
    ):
        return None
    if ("ead" in exposures) != ("lgd" in exposures):
        return None
    rules: list[tuple[_Adapter, Position]] = []
    if rate_keys:
        rules.append(
            (
                DurationConvexityValuationAdapter(),
                _view(position, {k: exposures[k] for k in rate_keys}),
            )
        )
    if "cs01:CREDIT-SPREAD" in exposures:
        rules.append(
            (
                CreditSpreadValuationAdapter(),
                _view(position, {"cs01:CREDIT-SPREAD": exposures["cs01:CREDIT-SPREAD"]}),
            )
        )
    if "ead" in exposures:
        rules.append(
            (DefaultValuationAdapter(), _view(position, {k: exposures[k] for k in ("ead", "lgd")}))
        )
    return rules


def _nonlinear(
    position: Position,
    market: MarketSnapshot,
    scenario: NormalizedScenario,
) -> dict[str, Decimal] | None:
    """Delta/gamma per native spot unit and vega per +1 volatility point.

    All Greeks are signed, position-sized currency sensitivities. Gamma uses
    currency/native-unit squared. No cross-gamma or volatility surface is inferred.
    Every supplied spot factor requires both delta and gamma, plus explicit
    vega:VOLATILITY; explicit zero sensitivities count as complete inputs.
    """
    exposures = position.factor_exposures
    spots = {k.partition(":")[2] for k in exposures if k.startswith(("delta:", "gamma:"))}
    required = {f"{kind}:{f}" for f in spots for kind in ("delta", "gamma")}
    if not spots or not spots <= _SPOT or set(exposures) != required | {"vega:VOLATILITY"}:
        return None
    for factor in sorted(spots | {"VOLATILITY"}):
        _market_level(market, factor)
    contributions: dict[str, Decimal] = {}
    for shock in sorted(scenario.shocks, key=lambda s: s.factor_id):
        factor = shock.factor_id
        move = Decimal(str(shock.value))
        if factor in spots:
            level = _market_level(market, factor)
            _relative_move(shock, level)
            if shock.shock_type is ShockType.RELATIVE:
                move *= level
            contributions[factor] = (
                Decimal(str(exposures[f"delta:{factor}"])) * move
                + Decimal("0.5") * Decimal(str(exposures[f"gamma:{factor}"])) * move * move
            )
        elif factor == "VOLATILITY":
            contributions[factor] = Decimal(str(exposures["vega:VOLATILITY"])) * move * 100
        else:
            contributions[factor] = Decimal(0)
    return contributions


class StressEngine:
    """Run a Stress Test on the supported subset, without imputing unsupported risk.

    Coverage is the supported share of gross absolute supplied base value.
    If all notionals are zero, the supported position-count share is used.
    Base/stressed value and attribution describe only the supported subset.
    Empty and entirely unsupported portfolios reject rather than invent totals.
    The version string is a canonical JSON manifest of engine and factor rules.
    """

    def run(
        self,
        portfolio: Portfolio,
        base_market: MarketSnapshot,
        scenario: StressScenario | NormalizedScenario,
    ) -> StressResult:
        # Specify every context setting rather than inheriting caller or DefaultContext.
        # Include canonical record validation: it also performs Decimal arithmetic.
        with localcontext(
            Context(
                prec=28,
                rounding=ROUND_HALF_EVEN,
                Emin=-999999,
                Emax=999999,
                capitals=1,
                clamp=0,
                flags=[],
                traps=[InvalidOperation, DivisionByZero, Overflow],
            )
        ):
            return self._run(portfolio, base_market, scenario)

    def _run(
        self,
        portfolio: Portfolio,
        base_market: MarketSnapshot,
        scenario: StressScenario | NormalizedScenario,
    ) -> StressResult:
        portfolio = Portfolio.model_validate(portfolio.model_dump())
        market = MarketSnapshot.model_validate(base_market.model_dump())
        normalized = (
            normalize_scenario(scenario) if isinstance(scenario, StressScenario) else scenario
        )
        _shocks(normalized)  # Validate atomically even if no adapter consumes a factor.
        normalized = NormalizedScenario.model_validate(normalized.model_dump())
        if normalized.analyst_override is not (normalized.override_reason is not None):
            raise ValueError("override_reason is required exactly for an analyst override")
        if normalized.method is ProvenanceMethod.EMPIRICAL and not normalized.reference_event_ids:
            raise ValueError("empirical scenarios require reference_event_ids")
        if len(set(normalized.reference_event_ids)) != len(normalized.reference_event_ids):
            raise ValueError("reference_event_ids must be unique")
        if market.as_of != portfolio.as_of:
            raise ValueError("portfolio and market as_of must agree")
        if any(p.currency != portfolio.valuation_currency for p in portfolio.positions):
            raise ValueError("position currency must match portfolio valuation currency")
        manifest: dict[str, dict[str, str]] = {}
        for shock in sorted(normalized.shocks, key=lambda s: s.factor_id):
            registered = FACTOR_RULE_VERSIONS.get(shock.factor_id)
            if not registered:
                raise ValueError(f"missing registered rule version: {shock.factor_id}")
            manifest[shock.factor_id] = {r.rule: r.version for r in registered}
        versions = StressResultVersions(
            scenario_version=normalized.calibration_version,
            market_version=json.dumps(
                {
                    "snapshot_id": market.snapshot_id,
                    "schema": market.schema_version,
                    "normalization": market.normalization_version,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            portfolio_version=portfolio.version,
            valuation_rule_version=json.dumps(
                {
                    "engine": ENGINE_VERSION,
                    "normalization": normalized.normalization_version,
                    "factors": manifest,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        totals: dict[tuple[AttributionDimension, str], Decimal] = {}
        unsupported: list[str] = []
        base = gross_supported = Decimal(0)
        supported_count = 0
        for position in sorted(portfolio.positions, key=lambda p: p.position_id):
            nonlinear = position.asset_type is AssetType.DERIVATIVE and any(
                k.startswith(("gamma:", "vega:")) for k in position.factor_exposures
            )
            contributions: dict[str, Decimal] | None
            if nonlinear:
                contributions = _nonlinear(position, market, normalized)
            else:
                rules = _rules(position)
                contributions = None
                if rules:
                    # Run each adapter's complete-input validation before any contributions.
                    for adapter, view in rules:
                        adapter.value(view, market)
                    contributions = {
                        shock.factor_id: sum(
                            (
                                adapter.stressed_pnl(
                                    view, market, normalized.model_copy(update={"shocks": (shock,)})
                                )
                                for adapter, view in rules
                            ),
                            Decimal(0),
                        )
                        for shock in sorted(normalized.shocks, key=lambda s: s.factor_id)
                    }
            if contributions is None:
                unsupported.append(position.position_id)
                continue
            supported_count += 1
            base += position.notional
            gross_supported += abs(position.notional)
            loss = -sum(contributions.values(), Decimal(0))
            for dimension, label in (
                (AttributionDimension.ASSET, position.position_id),
                (AttributionDimension.SECTOR, position.sector),
                (AttributionDimension.REGION, position.region),
                (AttributionDimension.OBLIGOR, position.obligor),
            ):
                key = (dimension, label)
                totals[key] = totals.get(key, Decimal(0)) + loss
            for factor, pnl in sorted(contributions.items()):
                key = (AttributionDimension.FACTOR, factor)
                totals[key] = totals.get(key, Decimal(0)) - pnl
        if not supported_count:
            raise ValueError("Stress Test requires at least one supported position")
        gross = sum((abs(p.notional) for p in portfolio.positions), Decimal(0))
        coverage = (
            float(gross_supported / gross)
            if gross
            else (supported_count / len(portfolio.positions))
        )
        # Zero-value unsupported positions still require visibly incomplete coverage.
        if unsupported and coverage == 1:
            coverage = min(coverage, supported_count / len(portfolio.positions))
        loss = sum(
            (v for (d, _), v in totals.items() if d is AttributionDimension.ASSET), Decimal(0)
        )
        identity = json.dumps(
            {
                "portfolio": portfolio.model_dump(mode="json"),
                "market": market.model_dump(mode="json"),
                "scenario": normalized.model_dump(mode="json"),
                "versions": versions.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        # Canonical input order is part of replay identity, independent of tuple iteration order.
        identity_data = json.loads(identity)
        identity_data["portfolio"]["positions"].sort(key=lambda p: p["position_id"])
        identity_data["market"]["observations"].sort(key=lambda o: o["factor_id"])
        identity_data["scenario"]["shocks"].sort(key=lambda s: s["factor_id"])
        digest = hashlib.sha256(
            json.dumps(identity_data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return StressResult(
            stress_result_id=f"stress-{digest}",
            scenario_id=normalized.scenario_id,
            portfolio_id=portfolio.portfolio_id,
            base_value=base,
            stressed_value=base - loss,
            absolute_loss=loss,
            percentage_loss=float(loss / base) if base else 0,
            valuation_currency=portfolio.valuation_currency,
            attribution=tuple(
                StressAttribution(dimension=d, label=label, loss=value)
                for (d, label), value in sorted(
                    totals.items(), key=lambda item: (item[0][0].value, item[0][1])
                )
            ),
            valuation_coverage=coverage,
            unsupported_position_ids=tuple(unsupported),
            versions=versions,
        )
