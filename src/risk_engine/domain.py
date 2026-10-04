"""Canonical records shared across the risk-engine module boundaries."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, TypeVar

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
SentimentScore = Annotated[float, Field(ge=-1.0, le=1.0)]
ImpactScore = Annotated[int, Field(ge=1, le=10)]
PositiveDays = Annotated[int, Field(ge=1)]
NonNegativeCount = Annotated[int, Field(ge=0)]


class DomainModel(BaseModel):
    """Strict, immutable base for records that cross module seams."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EventClass(str, Enum):
    """Frozen top-level taxonomy used by every classifier and report."""

    GEOPOLITICAL = "Geopolitical"
    MACROECONOMIC_MONETARY = "Macroeconomic/Monetary"
    CREDIT_DEFAULT = "Credit/Default"
    REGULATORY_LEGAL = "Regulatory/Legal"
    OPERATIONAL_CYBER = "Operational/Cyber"
    CORPORATE_ACTION = "Corporate Action"
    CLIMATE_NATURAL_DISASTER = "Climate/Natural Disaster"
    OTHER_UNCERTAIN = "Other/Uncertain"


class SourceType(str, Enum):
    NEWS = "news"
    SOCIAL = "social"


class ShockType(str, Enum):
    RELATIVE = "relative"
    ABSOLUTE = "absolute"
    BASIS_POINT = "basis_point"
    DEFAULT = "default"
    VOLATILITY = "volatility"


class ShockUnit(str, Enum):
    DECIMAL = "decimal"
    PERCENT = "percent"
    BASIS_POINT = "basis_point"
    ABSOLUTE = "absolute"
    CURRENCY = "currency"
    INDEX_POINT = "index_point"
    VOLATILITY_POINT = "volatility_point"


class ProvenanceMethod(str, Enum):
    EMPIRICAL = "empirical"
    HYPOTHETICAL = "hypothetical"
    ANALYST_OVERRIDE = "analyst_override"


class AssetType(str, Enum):
    LOAN = "loan"
    BOND = "bond"
    DERIVATIVE = "derivative"
    EQUITY = "equity"
    OTHER = "other"


class VersionMetadata(DomainModel):
    schema_version: NonEmptyString
    model_version: NonEmptyString
    calibration_version: NonEmptyString
    snapshot_version: NonEmptyString


class SourceItem(DomainModel):
    source_item_id: NonEmptyString
    source_type: SourceType
    provider: NonEmptyString
    text: NonEmptyString
    published_at: AwareDatetime
    retrieved_at: AwareDatetime
    source_reference: NonEmptyString
    content_hash: NonEmptyString
    snapshot_id: NonEmptyString
    provenance: NonEmptyString
    license: NonEmptyString


class MarketObservation(DomainModel):
    factor_id: NonEmptyString
    observed_at: AwareDatetime
    value: float
    unit: ShockUnit
    provider: NonEmptyString
    vintage: NonEmptyString


_Record = TypeVar("_Record")


def _require_unique_ids(records: tuple[_Record, ...], attribute: str) -> None:
    values = [getattr(record, attribute) for record in records]
    if len(values) != len(set(values)):
        raise ValueError(f"{attribute} values must be unique")


class MarketSnapshot(DomainModel):
    snapshot_id: NonEmptyString
    as_of: AwareDatetime
    observations: tuple[MarketObservation, ...]
    provider: NonEmptyString
    schema_version: NonEmptyString
    normalization_version: NonEmptyString

    @model_validator(mode="after")
    def observations_have_unique_factor_ids(self) -> MarketSnapshot:
        _require_unique_ids(self.observations, "factor_id")
        return self


class EntityLink(DomainModel):
    entity_id: NonEmptyString
    canonical_name: NonEmptyString
    confidence: Probability
    evidence: NonEmptyString
    ambiguous: bool
    candidate_entity_ids: tuple[NonEmptyString, ...] = ()


class InterpretedEvent(DomainModel):
    event_id: NonEmptyString
    source_item_id: NonEmptyString
    entity_links: tuple[EntityLink, ...]
    event_class: EventClass
    sentiment: SentimentScore
    classification_confidence: Probability
    rationale: NonEmptyString
    evidence: tuple[NonEmptyString, ...]
    eligible_for_automatic_stress: bool


class ImpactEstimate(DomainModel):
    impact_score: ImpactScore
    expected_reference_loss: Decimal
    loss_lower_bound: Decimal
    loss_upper_bound: Decimal
    analogue_count: NonNegativeCount
    analogue_ids: tuple[NonEmptyString, ...]
    backoff_level: NonEmptyString
    method: ProvenanceMethod
    calibration_version: NonEmptyString
    reference_basket_version: NonEmptyString


class RiskSignal(DomainModel):
    signal_id: NonEmptyString
    source_item_id: NonEmptyString
    entity: EntityLink
    sentiment: SentimentScore
    event_class: EventClass
    impact: ImpactEstimate
    confidence: Probability
    rationale: NonEmptyString
    evidence: tuple[NonEmptyString, ...]
    versions: VersionMetadata
    portfolio_materiality: Decimal | None = None
    action_priority: float | None = None


class FactorShock(DomainModel):
    factor_id: NonEmptyString
    shock_type: ShockType
    value: float
    unit: ShockUnit
    horizon_days: PositiveDays


class StressScenario(DomainModel):
    scenario_id: NonEmptyString
    risk_signal_id: NonEmptyString | None = None
    shocks: tuple[FactorShock, ...]
    method: ProvenanceMethod
    reference_event_ids: tuple[NonEmptyString, ...] = ()
    calibration_version: NonEmptyString
    analyst_override: bool = False

    @model_validator(mode="after")
    def shocks_have_unique_factor_ids(self) -> StressScenario:
        _require_unique_ids(self.shocks, "factor_id")
        return self


class Position(DomainModel):
    position_id: NonEmptyString
    asset_type: AssetType
    notional: Decimal
    currency: NonEmptyString
    sector: NonEmptyString
    region: NonEmptyString
    obligor: NonEmptyString
    factor_exposures: dict[NonEmptyString, float]
    attributes: dict[str, Any] = Field(default_factory=dict)


class Portfolio(DomainModel):
    portfolio_id: NonEmptyString
    positions: tuple[Position, ...]
    valuation_currency: NonEmptyString
    as_of: AwareDatetime
    version: NonEmptyString

    @model_validator(mode="after")
    def positions_have_unique_ids(self) -> Portfolio:
        _require_unique_ids(self.positions, "position_id")
        return self


class StressResult(DomainModel):
    stress_result_id: NonEmptyString
    scenario_id: NonEmptyString
    portfolio_id: NonEmptyString
    base_value: Decimal
    stressed_value: Decimal
    absolute_loss: Decimal
    percentage_loss: float
    attribution: dict[NonEmptyString, Decimal]
    valuation_coverage: Probability
    unsupported_position_ids: tuple[NonEmptyString, ...]
    scenario_version: NonEmptyString
    market_version: NonEmptyString
    portfolio_version: NonEmptyString
    valuation_rule_version: NonEmptyString


class BacktestReport(DomainModel):
    report_id: NonEmptyString
    configuration_version: NonEmptyString
    evaluation_start: AwareDatetime
    evaluation_end: AwareDatetime
    metrics: dict[NonEmptyString, float]
    sensitivity_results: tuple[dict[str, Any], ...]
    historical_case_ids: tuple[NonEmptyString, ...]
    schema_version: NonEmptyString


__all__ = [
    "AssetType",
    "BacktestReport",
    "EntityLink",
    "EventClass",
    "FactorShock",
    "ImpactEstimate",
    "InterpretedEvent",
    "MarketObservation",
    "MarketSnapshot",
    "Portfolio",
    "Position",
    "ProvenanceMethod",
    "RiskSignal",
    "ShockType",
    "ShockUnit",
    "SourceItem",
    "SourceType",
    "StressResult",
    "StressScenario",
    "VersionMetadata",
]
