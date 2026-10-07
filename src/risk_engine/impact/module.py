"""Frozen training-only Reference Basket severity; offline and holdings-independent."""

import hashlib
import json
from bisect import bisect_left
from datetime import datetime
from decimal import Decimal, localcontext
from math import ceil
from statistics import median
from typing import Literal

import numpy as np
from pydantic import AwareDatetime, Field, model_validator

from risk_engine.calibration.event_study import EventWindow, MeasurementDimension
from risk_engine.domain import (
    CurrencyCode,
    DomainModel,
    FactorShock,
    ImpactEstimate,
    InterpretedEvent,
    MarketSnapshot,
    NonEmptyString,
    Portfolio,
    ProvenanceMethod,
    Sha256Hex,
    ShockType,
    ShockUnit,
    StressResult,
    StressScenario,
)
from risk_engine.impact.analogues import AnalogueRepository, HistoricalAnalogue
from risk_engine.impact.reference_basket import (
    ALLOCATION_ARITHMETIC_VERSION,
    BasketConstruction,
    _equal_risk_weights,
    reference_allocation_context,
)
from risk_engine.stress.interfaces import FACTOR_REGISTRY
from risk_engine.stress.module import StressEngine
from risk_engine.stress.shocks import normalize_scenario


def _json(record: DomainModel) -> str:
    return json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _hash(record: DomainModel) -> str:
    return hashlib.sha256(_json(record).encode()).hexdigest()


class FactorBinding(DomainModel):
    historical_factor_id: NonEmptyString
    valuation_factor_id: NonEmptyString
    measurement_dimension: MeasurementDimension
    source_unit: ShockUnit
    shock_type: ShockType
    valuation_unit: ShockUnit
    value_multiplier: Decimal = Field(gt=0)

    @model_validator(mode="after")
    def registered_conversion(self) -> "FactorBinding":
        definition = FACTOR_REGISTRY.get(self.valuation_factor_id)
        if definition is None or self.shock_type not in definition.shock_types:
            raise ValueError("unknown factor or incompatible registered shock type")
        target_dimension = (
            MeasurementDimension.YIELD
            if self.valuation_factor_id.startswith("RATE-")
            else MeasurementDimension.SPREAD
            if self.valuation_factor_id == "CREDIT-SPREAD"
            else MeasurementDimension.VOLATILITY
            if self.valuation_factor_id == "VOLATILITY"
            else MeasurementDimension.CURRENCY
            if self.valuation_factor_id in {"USD-INR", "EUR-USD"}
            else MeasurementDimension.PRICE
            if self.valuation_factor_id == "COMMODITY"
            else MeasurementDimension.RETURN
        )
        relative_return = (
            self.shock_type is ShockType.RELATIVE
            and target_dimension in {MeasurementDimension.CURRENCY, MeasurementDimension.PRICE}
            and self.measurement_dimension is MeasurementDimension.RETURN
        )
        if self.measurement_dimension is not target_dimension and not relative_return:
            raise ValueError("measurement dimension disagrees with valuation factor")
        scales = {
            ShockUnit.DECIMAL: Decimal(1),
            ShockUnit.PERCENT: Decimal("0.01"),
            ShockUnit.BASIS_POINT: Decimal("0.0001"),
        }
        if self.source_unit == self.valuation_unit:
            expected = Decimal(1)
        elif self.source_unit in scales and self.valuation_unit in scales:
            expected = scales[self.source_unit] / scales[self.valuation_unit]
        else:
            raise ValueError("unsupported explicit unit conversion")
        if self.value_multiplier != expected:
            raise ValueError("unit conversion multiplier disagrees with declared units")
        normalize_scenario(
            StressScenario(
                scenario_id="binding-validation",
                risk_signal_id="binding-validation",
                shocks=(
                    FactorShock(
                        factor_id=self.valuation_factor_id,
                        shock_type=self.shock_type,
                        unit=self.valuation_unit,
                        value=0,
                        horizon_days=1,
                    ),
                ),
                method=ProvenanceMethod.HYPOTHETICAL,
                calibration_version="binding-validation",
            )
        )
        return self


class TrainingEvidence(DomainModel):
    analogue: HistoricalAnalogue
    base_market: MarketSnapshot
    partition: Literal["training"]


class FrozenImpactCalibration(DomainModel):
    """Input population and conventions, never selected on scoring observations."""

    version: NonEmptyString
    calibrated_at: AwareDatetime
    training_start: AwareDatetime
    training_end: AwareDatetime
    reference_basket: Portfolio
    window: EventWindow
    factor_bindings: tuple[FactorBinding, ...] = Field(min_length=1)
    quantile_convention: Literal["nearest-rank-lower-ties"]
    uncertainty_convention: Literal["observed-min-max"]
    source_terms: NonEmptyString
    training: tuple[TrainingEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def frozen_training_only(self) -> "FrozenImpactCalibration":
        if not self.training_start < self.training_end < self.calibrated_at:
            raise ValueError("training interval must end before calibration")
        _basket_manifest(self.reference_basket)
        if self.reference_basket.as_of > self.training_start:
            raise ValueError("Reference Basket must be frozen before training begins")
        ids = [row.analogue.event_id for row in self.training]
        if len(set(ids)) != len(ids):
            raise ValueError("training event IDs must be unique")
        for row in self.training:
            if row.analogue.evidence_kind != "observed":
                raise ValueError("empirical cutpoints require observed training evidence")
            if not self.training_start <= row.analogue.event_at < self.training_end:
                raise ValueError("training event outside training interval")
            if row.analogue.available_at >= self.training_end:
                raise ValueError("training evidence unavailable before training end")
            _market_available(row.base_market, row.analogue.event_at)
            if _hash(row.base_market) != _hash(self.training[0].base_market):
                raise ValueError("training must share one frozen base market")
        source_ids = [item.historical_factor_id for item in self.factor_bindings]
        target_ids = [item.valuation_factor_id for item in self.factor_bindings]
        if len(set(source_ids)) != len(source_ids) or len(set(target_ids)) != len(target_ids):
            raise ValueError("factor bindings must be unique on both sides")
        return self


class BasketManifest(DomainModel):
    basket_version: NonEmptyString
    allocation_arithmetic_version: NonEmptyString
    construction: BasketConstruction
    factor_proxy_ids: tuple[NonEmptyString, ...]
    constituent_ids: tuple[NonEmptyString, ...]
    covariance: tuple[tuple[float, ...], ...]
    covariance_estimator: Literal["sample-ddof-1"]
    lookback_observations: int = Field(ge=3)
    lookback_start: AwareDatetime
    lookback_end: AwareDatetime
    calibration_date: AwareDatetime
    constraint: Literal["long-only-fully-invested"]
    recalibration_rule: Literal["frozen-until-new-version"]
    return_snapshot_id: NonEmptyString
    return_source_terms: NonEmptyString
    return_rows_sha256: Sha256Hex
    construction_spec_sha256: Sha256Hex
    total_notional: Decimal = Field(gt=0)
    valuation_currency: CurrencyCode


def _basket_manifest(basket: Portfolio) -> BasketManifest:
    try:
        manifest = BasketManifest.model_validate_json(basket.version)
        if manifest.calibration_date != basket.as_of:
            raise ValueError("Reference Basket calibration timestamp mismatch")
        if manifest.constituent_ids != tuple(p.position_id for p in basket.positions):
            raise ValueError("Reference Basket constituent mismatch")
        if not manifest.lookback_start < manifest.lookback_end <= basket.as_of:
            raise ValueError("Reference Basket lookback unavailable at calibration")
        if manifest.valuation_currency != basket.valuation_currency:
            raise ValueError("Reference Basket currency mismatch")
        if not basket.positions or any(p.notional <= 0 for p in basket.positions):
            raise ValueError("Reference Basket must be long-only and nonempty")
        size = len(basket.positions)
        if (
            len(manifest.factor_proxy_ids) != size
            or len(set(manifest.factor_proxy_ids)) != size
            or len(manifest.covariance) != size
            or any(len(row) != size for row in manifest.covariance)
        ):
            raise ValueError("Reference Basket covariance/proxy dimensions mismatch")
        with localcontext(reference_allocation_context(manifest.allocation_arithmetic_version)):
            if sum((p.notional for p in basket.positions), Decimal(0)) != manifest.total_notional:
                raise ValueError("Reference Basket total notional mismatch")
            covariance = np.array(manifest.covariance, dtype=float)
            if manifest.construction is BasketConstruction.EQUAL_NOTIONAL:
                weights = np.full(size, 1 / size)
            elif manifest.construction is BasketConstruction.INVERSE_VOLATILITY:
                variance = np.diag(covariance)
                if np.any(variance <= 0):
                    raise ValueError("Reference Basket requires positive volatility")
                inverse_volatility = 1 / np.sqrt(variance)
                weights = inverse_volatility / inverse_volatility.sum()
            else:
                if np.any(np.diag(covariance) <= 0):
                    raise ValueError("Reference Basket requires positive volatility")
                weights = _equal_risk_weights(covariance)
            # Match the builder's currency allocation and final residual conventions.
            allocations = [manifest.total_notional * Decimal(str(w)) for w in weights[:-1]]
            allocations.append(manifest.total_notional - sum(allocations, Decimal(0)))
            if tuple(p.notional for p in basket.positions) != tuple(allocations):
                raise ValueError("Reference Basket construction allocation mismatch")
        return manifest
    except ValueError as error:
        raise ValueError("invalid Reference Basket manifest") from error


def _market_available(market: MarketSnapshot, cutoff: datetime) -> None:
    if market.as_of > cutoff or any(row.observed_at > cutoff for row in market.observations):
        raise ValueError("base market is unavailable at required timestamp")


class GovernedHypothetical(DomainModel):
    scenario: StressScenario
    base_market: MarketSnapshot
    window: EventWindow
    available_at: AwareDatetime
    governance_version: NonEmptyString
    source_terms: NonEmptyString
    source_sha256: Sha256Hex

    @model_validator(mode="after")
    def hypothetical_only(self) -> "GovernedHypothetical":
        if self.scenario.method is not ProvenanceMethod.HYPOTHETICAL:
            raise ValueError("fallback must be explicitly hypothetical")
        if self.scenario.reference_event_ids:
            raise ValueError("hypothetical fallback cannot claim historical analogue IDs")
        if any(
            s.horizon_days != self.window.end - self.window.start + 1 for s in self.scenario.shocks
        ):
            raise ValueError("fallback horizon disagrees with registered window")
        _market_available(self.base_market, self.available_at)
        return self


class LossEvidence(DomainModel):
    event_id: NonEmptyString
    loss: Decimal
    evidence_sha256: Sha256Hex
    available_at: AwareDatetime
    stress_result_id: NonEmptyString
    valuation_rule_version: NonEmptyString


class CalibrationAudit(DomainModel):
    version: NonEmptyString
    calibrated_at: AwareDatetime
    training_start: AwareDatetime
    training_end: AwareDatetime
    reference_basket_sha256: Sha256Hex
    base_market_sha256: Sha256Hex
    base_market_snapshot_id: NonEmptyString
    window: EventWindow
    factor_bindings: tuple[FactorBinding, ...]
    quantile_convention: NonEmptyString
    uncertainty_convention: NonEmptyString
    source_terms: NonEmptyString


class ImpactAudit(DomainModel):
    calibration: CalibrationAudit
    calibration_sha256: Sha256Hex
    cutpoints: tuple[Decimal, ...]
    training_losses: tuple[LossEvidence, ...]
    selected_losses: tuple[LossEvidence, ...]
    support_count: int = Field(ge=0)
    matching_version: NonEmptyString
    matching_validation_status: NonEmptyString
    cohort_sha256: Sha256Hex
    fallback: GovernedHypothetical | None
    hypothetical_loss: LossEvidence | None


class ImpactCalibrationSummary(DomainModel):
    calibration_version: NonEmptyString
    calibration_hash: Sha256Hex
    reference_basket_version: NonEmptyString
    reference_basket_hash: Sha256Hex
    matching_version: NonEmptyString
    training_event_ids: tuple[NonEmptyString, ...]
    cutpoints: tuple[Decimal, ...] = Field(min_length=9, max_length=9)
    currency: CurrencyCode
    horizon_days: int = Field(strict=True, gt=0)
    quantile_convention: Literal["nearest-rank-lower-ties"]
    frozen_at: AwareDatetime

    @model_validator(mode="after")
    def valid_population(self) -> "ImpactCalibrationSummary":
        if len(set(self.training_event_ids)) != len(self.training_event_ids):
            raise ValueError("duplicate Impact training event identity")
        if any(a > b for a, b in zip(self.cutpoints, self.cutpoints[1:], strict=False)):
            raise ValueError("Impact cutpoints must be nondecreasing")
        return self


class ImpactEstimator:
    """Small public scoring seam; no current Synthetic Portfolio input exists."""

    def __init__(
        self,
        repository: AnalogueRepository,
        calibration: FrozenImpactCalibration,
        base_market: MarketSnapshot,
        hypothetical: GovernedHypothetical | None = None,
    ) -> None:
        self._repository = repository
        self._calibration = FrozenImpactCalibration.model_validate(calibration.model_dump())
        self._market = MarketSnapshot.model_validate(base_market.model_dump())
        if _hash(self._market) != _hash(self._calibration.training[0].base_market):
            raise ValueError("scoring must share the training frozen base market")
        self._fallback = (
            GovernedHypothetical.model_validate(hypothetical.model_dump()) if hypothetical else None
        )
        self._training_losses = tuple(
            self._loss(row.analogue, row.base_market, "training")
            for row in self._calibration.training
        )
        population = sorted(row.loss for row in self._training_losses)
        self._cutpoints = tuple(
            population[ceil(len(population) * k / 10) - 1] for k in range(1, 10)
        )

    @property
    def calibration_summary(self) -> ImpactCalibrationSummary:
        """Return a fresh validated view of the actual frozen Impact fit."""
        calibration = self._calibration
        return ImpactCalibrationSummary(
            calibration_version=calibration.version,
            calibration_hash=_hash(calibration),
            reference_basket_version=calibration.reference_basket.version,
            reference_basket_hash=_hash(calibration.reference_basket),
            matching_version=self._repository._config.version,
            training_event_ids=tuple(row.analogue.event_id for row in calibration.training),
            cutpoints=self._cutpoints,
            currency=calibration.reference_basket.valuation_currency,
            horizon_days=calibration.window.end - calibration.window.start + 1,
            quantile_convention=calibration.quantile_convention,
            frozen_at=calibration.calibrated_at,
        )

    def _stress(self, market: MarketSnapshot, scenario: StressScenario) -> StressResult:
        required = set()
        for position in self._calibration.reference_basket.positions:
            for key in position.factor_exposures:
                required.add(
                    key.partition(":")[2]
                    if ":" in key
                    else "DEFAULT"
                    if key in {"ead", "lgd"}
                    else key
                )
        if not required <= {shock.factor_id for shock in scenario.shocks}:
            raise ValueError("incomplete Reference Basket factor coverage")
        result = StressEngine().run(self._calibration.reference_basket, market, scenario)
        if result.valuation_coverage != 1 or result.unsupported_position_ids:
            raise ValueError("incomplete Reference Basket valuation coverage")
        return result

    def _loss(
        self, analogue: HistoricalAnalogue, market: MarketSnapshot, query_id: str
    ) -> LossEvidence:
        with localcontext(reference_allocation_context(ALLOCATION_ARITHMETIC_VERSION)):
            return self._loss_in_context(analogue, market, query_id)

    def _loss_in_context(
        self, analogue: HistoricalAnalogue, market: MarketSnapshot, query_id: str
    ) -> LossEvidence:
        scenario = analogue.scenario
        if scenario is None:
            raise ValueError("historical analogue lacks Joint Scenario")
        windows = [row for row in scenario.windows if row.window == self._calibration.window]
        if len(windows) != 1:
            raise ValueError("missing registered event window")
        bindings = {item.historical_factor_id: item for item in self._calibration.factor_bindings}
        if {shock.factor_id for shock in windows[0].shocks} != bindings.keys():
            raise ValueError("factor bindings do not cover the entire joint vector")
        shocks = []
        for historical in windows[0].shocks:
            binding = bindings[historical.factor_id]
            if historical.unit != binding.source_unit or (
                historical.measurement_dimension != binding.measurement_dimension
            ):
                raise ValueError("historical unit/dimension disagrees with registered binding")
            shocks.append(
                FactorShock(
                    factor_id=binding.valuation_factor_id,
                    shock_type=binding.shock_type,
                    unit=binding.valuation_unit,
                    value=float(Decimal(str(historical.value)) * binding.value_multiplier),
                    horizon_days=self._calibration.window.end - self._calibration.window.start + 1,
                )
            )
        result = self._stress(
            market,
            StressScenario(
                scenario_id=f"reference-{analogue.event_id}",
                risk_signal_id=query_id,
                shocks=tuple(shocks),
                method=ProvenanceMethod.EMPIRICAL,
                reference_event_ids=(analogue.event_id,),
                calibration_version=self._calibration.version,
            ),
        )
        return LossEvidence(
            event_id=analogue.event_id,
            loss=result.absolute_loss,
            evidence_sha256=_hash(analogue),
            available_at=analogue.available_at,
            stress_result_id=result.stress_result_id,
            valuation_rule_version=result.versions.valuation_rule_version,
        )

    def estimate(self, event: InterpretedEvent, as_of: datetime) -> ImpactEstimate:
        with localcontext(reference_allocation_context(ALLOCATION_ARITHMETIC_VERSION)):
            return self._estimate(event, as_of)

    def _estimate(self, event: InterpretedEvent, as_of: datetime) -> ImpactEstimate:
        if as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        calibration = self._calibration
        if calibration.calibrated_at >= as_of:
            raise ValueError("calibration unavailable before as_of")
        if event.event_id in {row.analogue.event_id for row in calibration.training}:
            raise ValueError("scoring event overlaps frozen training population")
        _market_available(self._market, as_of)
        cohort = self._repository.match(event, as_of)
        selected = tuple(self._loss(row, self._market, event.event_id) for row in cohort.analogues)
        fallback = None
        hypothetical_loss = None
        losses: tuple[Decimal, ...]
        if cohort.method is ProvenanceMethod.HYPOTHETICAL:
            fallback = self._fallback
            if fallback is None:
                raise ValueError("inadequate history: governed hypothetical fallback required")
            if fallback.available_at >= as_of or fallback.window != calibration.window:
                raise ValueError("hypothetical fallback unavailable or window mismatch")
            if fallback.scenario.risk_signal_id != event.event_id:
                raise ValueError("hypothetical fallback must be governed for scoring event")
            if _hash(fallback.base_market) != _hash(self._market):
                raise ValueError("hypothetical fallback must share the frozen base market")
            result = self._stress(fallback.base_market, fallback.scenario)
            hypothetical_loss = LossEvidence(
                event_id=fallback.scenario.scenario_id,
                loss=result.absolute_loss,
                evidence_sha256=_hash(fallback),
                available_at=fallback.available_at,
                stress_result_id=result.stress_result_id,
                valuation_rule_version=result.versions.valuation_rule_version,
            )
            losses = (result.absolute_loss,)
        else:
            losses = tuple(row.loss for row in selected)
        expected = Decimal(median(losses))
        audit = ImpactAudit(
            calibration=CalibrationAudit(
                version=calibration.version,
                calibrated_at=calibration.calibrated_at,
                training_start=calibration.training_start,
                training_end=calibration.training_end,
                reference_basket_sha256=_hash(calibration.reference_basket),
                base_market_sha256=_hash(self._market),
                base_market_snapshot_id=self._market.snapshot_id,
                window=calibration.window,
                factor_bindings=calibration.factor_bindings,
                quantile_convention=calibration.quantile_convention,
                uncertainty_convention=calibration.uncertainty_convention,
                source_terms=calibration.source_terms,
            ),
            calibration_sha256=_hash(calibration),
            cutpoints=self._cutpoints,
            training_losses=self._training_losses,
            selected_losses=selected,
            support_count=cohort.support_count,
            matching_version=cohort.matching_version,
            matching_validation_status=cohort.validation_status,
            cohort_sha256=_hash(cohort),
            fallback=fallback,
            hypothetical_loss=hypothetical_loss,
        )
        audit_data = audit.model_dump(mode="json")
        audit_data["audit_sha256"] = _hash(audit)
        return ImpactEstimate(
            impact_score=1 + bisect_left(self._cutpoints, expected),
            expected_reference_loss=expected,
            loss_lower_bound=min(losses),
            loss_upper_bound=max(losses),
            loss_currency=calibration.reference_basket.valuation_currency,
            analogue_count=len(selected),
            analogue_ids=tuple(row.event_id for row in selected),
            backoff_level=cohort.backoff_level,
            method=cohort.method,
            calibration_version=json.dumps(audit_data, sort_keys=True, separators=(",", ":")),
            reference_basket_version=calibration.reference_basket.version,
        )
