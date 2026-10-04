from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from risk_engine.domain import (
    AssetType,
    AttributionDimension,
    BacktestReport,
    CandidateConfiguration,
    ConfidenceTarget,
    EntityLink,
    EventClass,
    FactorShock,
    ImpactEstimate,
    InterpretedEvent,
    MarketObservation,
    MarketSnapshot,
    Portfolio,
    PortfolioMateriality,
    Position,
    ProvenanceMethod,
    RiskSignal,
    SensitivityResult,
    ShockType,
    ShockUnit,
    SignalFlag,
    SourceItem,
    SourceType,
    StressAttribution,
    StressResult,
    StressResultVersions,
    StressScenario,
    VersionMetadata,
)

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


def make_source_item(**overrides: object) -> SourceItem:
    values: dict[str, object] = {
        "source_item_id": "source-1",
        "source_type": SourceType.NEWS,
        "provider": "demo-provider",
        "text": "Issuer spreads widened after the announcement.",
        "published_at": NOW,
        "retrieved_at": NOW,
        "source_reference": "https://example.test/story/1",
        "content_hash": "a" * 64,
        "snapshot_id": "snapshot-1",
        "provenance": "project-authored fixture",
        "license": "MIT",
    }
    values.update(overrides)
    return SourceItem.model_validate(values)


def make_entity_link() -> EntityLink:
    return EntityLink(
        entity_id="entity-1",
        canonical_name="Example Bank",
        confidence=0.9,
        evidence="Example Bank",
        ambiguous=False,
        candidate_entity_ids=("entity-1",),
    )


def make_interpreted_event(**overrides: object) -> InterpretedEvent:
    values: dict[str, object] = {
        "event_id": "event-1",
        "source_item_id": "source-1",
        "entity_links": (make_entity_link(),),
        "event_class": EventClass.CREDIT_DEFAULT,
        "sentiment": -0.7,
        "classification_confidence": 0.85,
        "rationale": "The text describes deterioration in credit conditions.",
        "evidence": ("spreads widened",),
        "eligible_for_automatic_stress": True,
    }
    values.update(overrides)
    return InterpretedEvent.model_validate(values)


def make_impact(**overrides: object) -> ImpactEstimate:
    values: dict[str, object] = {
        "impact_score": 8,
        "expected_reference_loss": Decimal("125000.00"),
        "loss_lower_bound": Decimal("100000.00"),
        "loss_upper_bound": Decimal("150000.00"),
        "loss_currency": "USD",
        "analogue_count": 2,
        "analogue_ids": ("analogue-1", "analogue-2"),
        "backoff_level": "event_class+region",
        "method": ProvenanceMethod.EMPIRICAL,
        "calibration_version": "calibration-v1",
        "reference_basket_version": "basket-v1",
    }
    values.update(overrides)
    return ImpactEstimate.model_validate(values)


def make_versions(**overrides: object) -> VersionMetadata:
    values: dict[str, object] = {
        "schema_version": "1.0.0",
        "model_version": "models-v1",
        "calibration_version": "calibration-v1",
        "snapshot_version": "snapshot-v1",
    }
    values.update(overrides)
    return VersionMetadata.model_validate(values)


def make_required_attribution() -> tuple[StressAttribution, ...]:
    return (
        StressAttribution(
            dimension=AttributionDimension.ASSET,
            label="loan",
            loss=Decimal("20"),
        ),
        StressAttribution(
            dimension=AttributionDimension.SECTOR,
            label="Financials",
            loss=Decimal("20"),
        ),
        StressAttribution(
            dimension=AttributionDimension.REGION,
            label="India",
            loss=Decimal("20"),
        ),
        StressAttribution(
            dimension=AttributionDimension.OBLIGOR,
            label="Example Bank",
            loss=Decimal("20"),
        ),
        StressAttribution(
            dimension=AttributionDimension.FACTOR,
            label="CREDIT-SPREAD",
            loss=Decimal("20"),
        ),
    )


def test_event_class_is_the_frozen_eight_value_taxonomy() -> None:
    assert {event_class.value for event_class in EventClass} == {
        "Geopolitical",
        "Macroeconomic/Monetary",
        "Credit/Default",
        "Regulatory/Legal",
        "Operational/Cyber",
        "Corporate Action",
        "Climate/Natural Disaster",
        "Other/Uncertain",
    }


@pytest.mark.parametrize("field", ["published_at", "retrieved_at"])
def test_source_item_rejects_naive_timestamps(field: str) -> None:
    with pytest.raises(ValidationError):
        make_source_item(**{field: datetime(2026, 10, 4, 12, 0)})


def test_source_item_rejects_retrieval_before_publication() -> None:
    with pytest.raises(ValidationError, match="retrieved_at"):
        make_source_item(retrieved_at=NOW - timedelta(seconds=1))


@pytest.mark.parametrize("sentiment", [-1.0001, 1.0001])
def test_interpreted_event_rejects_sentiment_outside_closed_unit_interval(
    sentiment: float,
) -> None:
    with pytest.raises(ValidationError):
        make_interpreted_event(sentiment=sentiment)


@pytest.mark.parametrize("confidence", [-0.0001, 1.0001])
def test_entity_link_rejects_confidence_outside_probability_range(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        EntityLink(
            entity_id="entity-1",
            canonical_name="Example Bank",
            confidence=confidence,
            evidence="Example Bank",
            ambiguous=False,
        )


@pytest.mark.parametrize("impact_score", [0, 11])
def test_impact_estimate_rejects_scores_outside_deciles(impact_score: int) -> None:
    with pytest.raises(ValidationError):
        make_impact(impact_score=impact_score)


def test_factor_shock_requires_explicit_supported_type_and_unit() -> None:
    shock = FactorShock(
        factor_id="USD-INR",
        shock_type=ShockType.RELATIVE,
        value=0.03,
        unit=ShockUnit.DECIMAL,
        horizon_days=1,
    )

    assert shock.shock_type is ShockType.RELATIVE
    assert shock.unit is ShockUnit.DECIMAL

    with pytest.raises(ValidationError):
        FactorShock.model_validate(
            {
                "factor_id": "USD-INR",
                "shock_type": "invented",
                "value": 0.03,
                "unit": "unknown",
                "horizon_days": 1,
            }
        )


def test_market_snapshot_rejects_duplicate_factor_ids() -> None:
    observation = MarketObservation(
        factor_id="USD-INR",
        observed_at=NOW,
        value=83.25,
        unit=ShockUnit.ABSOLUTE,
        provider="fixture",
        vintage="2026-10-04",
    )

    with pytest.raises(ValidationError, match="factor_id"):
        MarketSnapshot(
            snapshot_id="market-1",
            as_of=NOW,
            observations=(observation, observation),
            provider="fixture",
            schema_version="1.0.0",
            normalization_version="normalization-v1",
        )


def test_market_snapshot_rejects_observations_after_as_of_time() -> None:
    observation = MarketObservation(
        factor_id="USD-INR",
        observed_at=NOW + timedelta(seconds=1),
        value=83.25,
        unit=ShockUnit.ABSOLUTE,
        provider="fixture",
        vintage="2026-10-04",
    )

    with pytest.raises(ValidationError, match="as_of"):
        MarketSnapshot(
            snapshot_id="market-1",
            as_of=NOW,
            observations=(observation,),
            provider="fixture",
            schema_version="1.0.0",
            normalization_version="normalization-v1",
        )


def test_portfolio_rejects_duplicate_position_ids_and_keeps_money_as_decimal() -> None:
    position = Position(
        position_id="position-1",
        asset_type=AssetType.LOAN,
        notional=Decimal("1000000.01"),
        currency="USD",
        sector="Financials",
        region="India",
        obligor="Example Bank",
        factor_exposures={"USD-INR": 0.25},
    )

    with pytest.raises(ValidationError, match="position_id"):
        Portfolio(
            portfolio_id="portfolio-1",
            positions=(position, position),
            valuation_currency="USD",
            as_of=NOW,
            version="portfolio-v1",
        )

    assert isinstance(position.notional, Decimal)


def test_stress_scenario_rejects_duplicate_factor_shocks() -> None:
    shock = FactorShock(
        factor_id="CREDIT-SPREAD",
        shock_type=ShockType.BASIS_POINT,
        value=75.0,
        unit=ShockUnit.BASIS_POINT,
        horizon_days=1,
    )

    with pytest.raises(ValidationError, match="factor_id"):
        StressScenario(
            scenario_id="scenario-1",
            risk_signal_id="signal-1",
            shocks=(shock, shock),
            method=ProvenanceMethod.EMPIRICAL,
            reference_event_ids=("event-1",),
            calibration_version="calibration-v1",
        )


def test_interpreted_event_rejects_automatic_stress_for_ambiguous_entity() -> None:
    ambiguous_link = make_entity_link().model_copy(update={"ambiguous": True})

    with pytest.raises(ValidationError, match="ambiguous"):
        make_interpreted_event(entity_links=(ambiguous_link,))


@pytest.mark.parametrize(
    ("shock_type", "unit"),
    [
        (ShockType.RELATIVE, ShockUnit.BASIS_POINT),
        (ShockType.BASIS_POINT, ShockUnit.DECIMAL),
        (ShockType.VOLATILITY, ShockUnit.CURRENCY),
    ],
)
def test_factor_shock_rejects_incompatible_type_and_unit(
    shock_type: ShockType,
    unit: ShockUnit,
) -> None:
    with pytest.raises(ValidationError, match="unit"):
        FactorShock(
            factor_id="factor-1",
            shock_type=shock_type,
            value=1.0,
            unit=unit,
            horizon_days=1,
        )


def test_stress_scenario_requires_signal_and_consistent_override_provenance() -> None:
    values = {
        "scenario_id": "scenario-1",
        "shocks": (
            {
                "factor_id": "CREDIT-SPREAD",
                "shock_type": ShockType.BASIS_POINT,
                "value": 75.0,
                "unit": ShockUnit.BASIS_POINT,
                "horizon_days": 1,
            },
        ),
        "method": ProvenanceMethod.EMPIRICAL,
        "reference_event_ids": ("event-1",),
        "calibration_version": "calibration-v1",
        "analyst_override": True,
        "override_reason": "Analyst widened the credit-spread shock.",
    }

    with pytest.raises(ValidationError):
        StressScenario.model_validate(values)

    scenario = StressScenario.model_validate({**values, "risk_signal_id": "signal-1"})
    assert scenario.method is ProvenanceMethod.EMPIRICAL
    assert scenario.analyst_override is True

    with pytest.raises(ValidationError, match="override_reason"):
        StressScenario.model_validate(
            {**values, "risk_signal_id": "signal-1", "override_reason": None}
        )


def test_required_evidence_and_joint_shock_collections_cannot_be_empty() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        make_interpreted_event(evidence=())

    with pytest.raises(ValidationError, match="evidence"):
        RiskSignal(
            signal_id="signal-1",
            source_item_id="source-1",
            entity=make_entity_link(),
            sentiment=-0.7,
            event_class=EventClass.CREDIT_DEFAULT,
            impact=make_impact(),
            confidence=0.8,
            confidence_target=ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS,
            rationale="Credit conditions deteriorated.",
            evidence=(),
            flags=(),
            versions=make_versions(),
        )

    with pytest.raises(ValidationError, match="shocks"):
        StressScenario(
            scenario_id="scenario-1",
            risk_signal_id="signal-1",
            shocks=(),
            method=ProvenanceMethod.HYPOTHETICAL,
            calibration_version="calibration-v1",
        )


def test_impact_estimate_rejects_incoherent_range_and_empirical_zero_support() -> None:
    with pytest.raises(ValidationError, match="loss"):
        make_impact(
            loss_lower_bound=Decimal("200"),
            expected_reference_loss=Decimal("100"),
            loss_upper_bound=Decimal("50"),
        )

    with pytest.raises(ValidationError, match="analogue"):
        make_impact(analogue_count=0, analogue_ids=())


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("analogue_ids", ("analogue-1", "analogue-1")),
        ("candidate_entity_ids", ("entity-1", "entity-1")),
    ],
)
def test_evidence_id_collections_reject_duplicates(
    field: str,
    values: tuple[str, ...],
) -> None:
    if field == "analogue_ids":
        with pytest.raises(ValidationError, match="analogue_ids"):
            make_impact(analogue_ids=values)
    else:
        with pytest.raises(ValidationError, match="candidate_entity_ids"):
            EntityLink(
                entity_id="entity-1",
                canonical_name="Example Bank",
                confidence=0.8,
                evidence="Example Bank",
                ambiguous=True,
                candidate_entity_ids=values,
            )


def test_backtest_report_rejects_reversed_evaluation_period() -> None:
    with pytest.raises(ValidationError, match="evaluation_end"):
        BacktestReport(
            report_id="backtest-1",
            configuration_version="config-v1",
            evaluation_start=NOW,
            evaluation_end=NOW - timedelta(days=1),
            metrics={},
            sensitivity_results=(),
            candidate_configurations=(),
            historical_case_ids=(),
            schema_version="1.0.0",
        )


@pytest.mark.parametrize(
    "missing_field",
    ["schema_version", "model_version", "calibration_version", "snapshot_version"],
)
def test_risk_signal_rejects_incomplete_version_metadata(missing_field: str) -> None:
    versions = make_versions().model_dump()
    versions.pop(missing_field)

    with pytest.raises(ValidationError):
        RiskSignal(
            signal_id="signal-1",
            source_item_id="source-1",
            entity=make_entity_link(),
            sentiment=-0.7,
            event_class=EventClass.CREDIT_DEFAULT,
            impact=make_impact(),
            confidence=0.8,
            confidence_target=ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS,
            rationale="Credit conditions deteriorated.",
            evidence=("spreads widened",),
            flags=(),
            versions=versions,
        )


def test_risk_signal_carries_typed_materiality_confidence_target_and_flags() -> None:
    materiality = PortfolioMateriality(
        absolute_loss=Decimal("50000"),
        percentage_loss=0.05,
        currency="USD",
    )
    signal = RiskSignal(
        signal_id="signal-1",
        source_item_id="source-1",
        entity=make_entity_link(),
        sentiment=-0.7,
        event_class=EventClass.CREDIT_DEFAULT,
        impact=make_impact(),
        confidence=0.8,
        confidence_target=ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS,
        rationale="Credit conditions deteriorated.",
        evidence=("spreads widened",),
        flags=(SignalFlag.THIN_HISTORY,),
        versions=make_versions(),
        portfolio_materiality=materiality,
    )

    assert signal.portfolio_materiality == materiality
    assert signal.confidence_target is ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS
    assert signal.flags == (SignalFlag.THIN_HISTORY,)


def test_remaining_cross_module_records_are_constructible() -> None:
    scenario = StressScenario(
        scenario_id="scenario-1",
        risk_signal_id="signal-1",
        shocks=(
            FactorShock(
                factor_id="EQUITY-INDIA",
                shock_type=ShockType.RELATIVE,
                value=-0.05,
                unit=ShockUnit.DECIMAL,
                horizon_days=1,
            ),
        ),
        method=ProvenanceMethod.HYPOTHETICAL,
        calibration_version="calibration-v1",
    )
    stress_result = StressResult(
        stress_result_id="result-1",
        scenario_id=scenario.scenario_id,
        portfolio_id="portfolio-1",
        base_value=Decimal("1000000"),
        stressed_value=Decimal("950000"),
        absolute_loss=Decimal("50000"),
        percentage_loss=0.05,
        valuation_currency="USD",
        attribution=make_required_attribution(),
        valuation_coverage=0.95,
        unsupported_position_ids=("position-9",),
        versions=StressResultVersions(
            scenario_version="scenario-v1",
            market_version="market-v1",
            portfolio_version="portfolio-v1",
            valuation_rule_version="valuation-v1",
        ),
    )
    report = BacktestReport(
        report_id="backtest-1",
        configuration_version="config-v1",
        evaluation_start=NOW,
        evaluation_end=NOW,
        metrics={"macro_f1": 0.65},
        sensitivity_results=(
            SensitivityResult(
                parameter="event_window",
                value="[0,+1]",
                metrics={"macro_f1": 0.65},
            ),
        ),
        candidate_configurations=(
            CandidateConfiguration(
                candidate_id="candidate-1",
                parameters={"event_window": "[0,+1]"},
                metrics={"macro_f1": 0.65},
                selected=True,
            ),
        ),
        historical_case_ids=("case-1",),
        schema_version="1.0.0",
    )

    assert stress_result.absolute_loss == Decimal("50000")
    assert report.metrics["macro_f1"] == 0.65


def make_stress_result_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "stress_result_id": "result-1",
        "scenario_id": "scenario-1",
        "portfolio_id": "portfolio-1",
        "base_value": Decimal("100"),
        "stressed_value": Decimal("80"),
        "absolute_loss": Decimal("20"),
        "percentage_loss": 0.20,
        "valuation_currency": "USD",
        "attribution": make_required_attribution(),
        "valuation_coverage": 0.8,
        "unsupported_position_ids": ("position-1",),
        "versions": {
            "scenario_version": "scenario-v1",
            "market_version": "market-v1",
            "portfolio_version": "portfolio-v1",
            "valuation_rule_version": "valuation-v1",
        },
    }
    values.update(overrides)
    return values


def test_stress_result_rejects_inconsistent_loss_arithmetic() -> None:
    with pytest.raises(ValidationError, match="absolute_loss"):
        StressResult.model_validate(make_stress_result_values(absolute_loss=Decimal("10")))


def test_stress_result_rejects_duplicate_unsupported_position_ids() -> None:
    with pytest.raises(ValidationError, match="unsupported_position_ids"):
        StressResult.model_validate(
            make_stress_result_values(unsupported_position_ids=("position-1", "position-1"))
        )


def test_stress_result_rejects_full_coverage_with_unsupported_positions() -> None:
    with pytest.raises(ValidationError, match="valuation_coverage"):
        StressResult.model_validate(make_stress_result_values(valuation_coverage=1.0))


def test_stress_result_requires_all_attribution_dimensions() -> None:
    with pytest.raises(ValidationError, match="attribution"):
        StressResult.model_validate(
            make_stress_result_values(
                attribution=(
                    StressAttribution(
                        dimension=AttributionDimension.REGION,
                        label="India",
                        loss=Decimal("20"),
                    ),
                )
            )
        )


def test_stress_scenario_rejects_duplicate_reference_event_ids() -> None:
    shock = FactorShock(
        factor_id="CREDIT-SPREAD",
        shock_type=ShockType.BASIS_POINT,
        value=75.0,
        unit=ShockUnit.BASIS_POINT,
        horizon_days=1,
    )

    with pytest.raises(ValidationError, match="reference_event_ids"):
        StressScenario(
            scenario_id="scenario-1",
            risk_signal_id="signal-1",
            shocks=(shock,),
            method=ProvenanceMethod.EMPIRICAL,
            reference_event_ids=("event-1", "event-1"),
            calibration_version="calibration-v1",
        )


def test_hashes_and_currency_codes_use_domain_formats() -> None:
    with pytest.raises(ValidationError, match="content_hash"):
        make_source_item(content_hash="not-a-sha256")

    with pytest.raises(ValidationError, match="currency"):
        Position(
            position_id="position-1",
            asset_type=AssetType.LOAN,
            notional=Decimal("100"),
            currency="usd",
            sector="Financials",
            region="India",
            obligor="Example Bank",
            factor_exposures={},
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_financial_floats_reject_non_finite_values(value: float) -> None:
    with pytest.raises(ValidationError):
        MarketObservation(
            factor_id="USD-INR",
            observed_at=NOW,
            value=value,
            unit=ShockUnit.ABSOLUTE,
            provider="fixture",
            vintage="2026-10-04",
        )

    with pytest.raises(ValidationError):
        FactorShock(
            factor_id="USD-INR",
            shock_type=ShockType.RELATIVE,
            value=value,
            unit=ShockUnit.DECIMAL,
            horizon_days=1,
        )
