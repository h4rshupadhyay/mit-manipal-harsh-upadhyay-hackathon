from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from risk_engine.domain import (
    AssetType,
    BacktestReport,
    EntityLink,
    EventClass,
    FactorShock,
    ImpactEstimate,
    InterpretedEvent,
    MarketObservation,
    MarketSnapshot,
    Portfolio,
    Position,
    ProvenanceMethod,
    RiskSignal,
    ShockType,
    ShockUnit,
    SourceItem,
    SourceType,
    StressResult,
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
        "analogue_count": 4,
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
            rationale="Credit conditions deteriorated.",
            evidence=("spreads widened",),
            versions=versions,
        )


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
        attribution={"region:India": Decimal("50000")},
        valuation_coverage=0.95,
        unsupported_position_ids=("position-9",),
        scenario_version="scenario-v1",
        market_version="market-v1",
        portfolio_version="portfolio-v1",
        valuation_rule_version="valuation-v1",
    )
    report = BacktestReport(
        report_id="backtest-1",
        configuration_version="config-v1",
        evaluation_start=NOW,
        evaluation_end=NOW,
        metrics={"macro_f1": 0.65},
        sensitivity_results=(),
        historical_case_ids=("case-1",),
        schema_version="1.0.0",
    )

    assert stress_result.absolute_loss == Decimal("50000")
    assert report.metrics["macro_f1"] == 0.65
