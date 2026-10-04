"""Canonical records shared across the risk-engine module boundaries."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Annotated, TypeAlias, TypeVar

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
SentimentScore = Annotated[float, Field(ge=-1.0, le=1.0)]
ImpactScore = Annotated[int, Field(ge=1, le=10)]
PositiveDays = Annotated[int, Field(ge=1)]
NonNegativeCount = Annotated[int, Field(ge=0)]
NonNegativeDecimal = Annotated[Decimal, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
ConfigurationScalar: TypeAlias = str | int | float | bool


class DomainModel(BaseModel):
    """Strict, immutable base for records that cross module seams."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)


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


class AssetType(str, Enum):
    LOAN = "loan"
    BOND = "bond"
    DERIVATIVE = "derivative"
    EQUITY = "equity"
    OTHER = "other"


class ConfidenceTarget(str, Enum):
    JOINT_ENTITY_AND_EVENT_CLASS = "entity_and_event_class_joint_correctness"


class SignalFlag(str, Enum):
    AMBIGUOUS_ENTITY = "ambiguous_entity"
    LOW_CONFIDENCE = "low_confidence"
    THIN_HISTORY = "thin_history"
    HYPOTHETICAL_SCENARIO = "hypothetical_scenario"
    UNSUPPORTED_EXPOSURE = "unsupported_exposure"


class AttributionDimension(str, Enum):
    ASSET = "asset"
    SECTOR = "sector"
    REGION = "region"
    COUNTRY = "country"
    OBLIGOR = "obligor"
    FACILITY = "facility"
    FACTOR = "factor"


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
    content_hash: Sha256Hex
    snapshot_id: NonEmptyString
    provenance: NonEmptyString
    license: NonEmptyString

    @model_validator(mode="after")
    def retrieval_is_not_before_publication(self) -> SourceItem:
        if self.retrieved_at < self.published_at:
            raise ValueError("retrieved_at cannot be before published_at")
        return self


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


def _require_unique_values(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique")


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
        if any(observation.observed_at > self.as_of for observation in self.observations):
            raise ValueError("observations cannot be later than snapshot as_of")
        return self


class EntityLink(DomainModel):
    entity_id: NonEmptyString
    canonical_name: NonEmptyString
    confidence: Probability
    evidence: NonEmptyString
    ambiguous: bool
    candidate_entity_ids: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def candidates_have_unique_ids(self) -> EntityLink:
        _require_unique_values(self.candidate_entity_ids, "candidate_entity_ids")
        return self


class InterpretedEvent(DomainModel):
    event_id: NonEmptyString
    source_item_id: NonEmptyString
    entity_links: tuple[EntityLink, ...]
    event_class: EventClass
    sentiment: SentimentScore
    classification_confidence: Probability
    rationale: NonEmptyString
    evidence: Annotated[tuple[NonEmptyString, ...], Field(min_length=1)]
    eligible_for_automatic_stress: bool

    @model_validator(mode="after")
    def ambiguous_or_missing_entities_cannot_auto_stress(self) -> InterpretedEvent:
        if self.eligible_for_automatic_stress and (
            not self.entity_links or any(link.ambiguous for link in self.entity_links)
        ):
            raise ValueError(
                "ambiguous or missing entities cannot be eligible for automatic stress"
            )
        return self


class ImpactEstimate(DomainModel):
    impact_score: ImpactScore
    expected_reference_loss: Decimal
    loss_lower_bound: Decimal
    loss_upper_bound: Decimal
    loss_currency: CurrencyCode
    analogue_count: NonNegativeCount
    analogue_ids: tuple[NonEmptyString, ...]
    backoff_level: NonEmptyString
    method: ProvenanceMethod
    calibration_version: NonEmptyString
    reference_basket_version: NonEmptyString

    @model_validator(mode="after")
    def range_and_support_are_consistent(self) -> ImpactEstimate:
        if not (self.loss_lower_bound <= self.expected_reference_loss <= self.loss_upper_bound):
            raise ValueError("loss bounds must contain expected_reference_loss")
        _require_unique_values(self.analogue_ids, "analogue_ids")
        if self.analogue_count != len(self.analogue_ids):
            raise ValueError("analogue_count must equal the number of analogue_ids")
        if self.method is ProvenanceMethod.EMPIRICAL and self.analogue_count == 0:
            raise ValueError("empirical impact requires at least one historical analogue")
        return self


class PortfolioMateriality(DomainModel):
    absolute_loss: NonNegativeDecimal
    percentage_loss: NonNegativeFloat
    currency: CurrencyCode


class RiskSignal(DomainModel):
    signal_id: NonEmptyString
    source_item_id: NonEmptyString
    entity: EntityLink
    sentiment: SentimentScore
    event_class: EventClass
    impact: ImpactEstimate
    confidence: Probability
    confidence_target: ConfidenceTarget
    rationale: NonEmptyString
    evidence: Annotated[tuple[NonEmptyString, ...], Field(min_length=1)]
    flags: tuple[SignalFlag, ...]
    versions: VersionMetadata
    portfolio_materiality: PortfolioMateriality | None = None
    action_priority: float | None = None


class FactorShock(DomainModel):
    factor_id: NonEmptyString
    shock_type: ShockType
    value: float
    unit: ShockUnit
    horizon_days: PositiveDays

    @model_validator(mode="after")
    def type_matches_unit(self) -> FactorShock:
        compatible_units = {
            ShockType.RELATIVE: {ShockUnit.DECIMAL, ShockUnit.PERCENT},
            ShockType.ABSOLUTE: {
                ShockUnit.ABSOLUTE,
                ShockUnit.CURRENCY,
                ShockUnit.INDEX_POINT,
            },
            ShockType.BASIS_POINT: {ShockUnit.BASIS_POINT},
            ShockType.DEFAULT: {ShockUnit.DECIMAL, ShockUnit.PERCENT},
            ShockType.VOLATILITY: {ShockUnit.VOLATILITY_POINT},
        }
        if self.unit not in compatible_units[self.shock_type]:
            raise ValueError(f"unit {self.unit.value} is incompatible with {self.shock_type.value}")
        return self


class StressScenario(DomainModel):
    scenario_id: NonEmptyString
    risk_signal_id: NonEmptyString
    shocks: Annotated[tuple[FactorShock, ...], Field(min_length=1)]
    method: ProvenanceMethod
    reference_event_ids: tuple[NonEmptyString, ...] = ()
    calibration_version: NonEmptyString
    analyst_override: bool = False
    override_reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def shocks_have_unique_factor_ids(self) -> StressScenario:
        _require_unique_ids(self.shocks, "factor_id")
        _require_unique_values(self.reference_event_ids, "reference_event_ids")
        if self.analyst_override is not (self.override_reason is not None):
            raise ValueError("override_reason is required exactly when analyst_override is true")
        if self.method is ProvenanceMethod.EMPIRICAL and not self.reference_event_ids:
            raise ValueError("empirical scenarios require reference_event_ids")
        return self


class Position(DomainModel):
    position_id: NonEmptyString
    asset_type: AssetType
    notional: Decimal
    currency: CurrencyCode
    sector: NonEmptyString
    region: NonEmptyString
    obligor: NonEmptyString
    factor_exposures: dict[NonEmptyString, float]


class Portfolio(DomainModel):
    portfolio_id: NonEmptyString
    positions: tuple[Position, ...]
    valuation_currency: CurrencyCode
    as_of: AwareDatetime
    version: NonEmptyString

    @model_validator(mode="after")
    def positions_have_unique_ids(self) -> Portfolio:
        _require_unique_ids(self.positions, "position_id")
        return self


class StressAttribution(DomainModel):
    dimension: AttributionDimension
    label: NonEmptyString
    loss: Decimal


class StressResultVersions(DomainModel):
    scenario_version: NonEmptyString
    market_version: NonEmptyString
    portfolio_version: NonEmptyString
    valuation_rule_version: NonEmptyString


class StressResult(DomainModel):
    stress_result_id: NonEmptyString
    scenario_id: NonEmptyString
    portfolio_id: NonEmptyString
    base_value: Decimal
    stressed_value: Decimal
    absolute_loss: Decimal
    percentage_loss: float
    valuation_currency: CurrencyCode
    attribution: tuple[StressAttribution, ...]
    valuation_coverage: Probability
    unsupported_position_ids: tuple[NonEmptyString, ...]
    versions: StressResultVersions

    @model_validator(mode="after")
    def losses_and_ids_are_consistent(self) -> StressResult:
        _require_unique_values(self.unsupported_position_ids, "unsupported_position_ids")
        attribution_keys = tuple(f"{row.dimension.value}:{row.label}" for row in self.attribution)
        _require_unique_values(attribution_keys, "attribution dimension and label")
        dimensions = {row.dimension for row in self.attribution}
        required_dimensions = {
            AttributionDimension.ASSET,
            AttributionDimension.SECTOR,
            AttributionDimension.REGION,
            AttributionDimension.FACTOR,
        }
        has_obligor_or_facility = bool(
            dimensions & {AttributionDimension.OBLIGOR, AttributionDimension.FACILITY}
        )
        if not required_dimensions <= dimensions or not has_obligor_or_facility:
            raise ValueError(
                "attribution must include asset, sector, region, obligor/facility, and factor"
            )
        if self.unsupported_position_ids and self.valuation_coverage >= 1.0:
            raise ValueError("valuation_coverage must be below 1 with unsupported positions")
        if self.base_value - self.stressed_value != self.absolute_loss:
            raise ValueError("absolute_loss must equal base_value minus stressed_value")
        if self.base_value == 0:
            if self.percentage_loss != 0:
                raise ValueError("percentage_loss must be zero when base_value is zero")
        else:
            expected_percentage = self.absolute_loss / self.base_value
            supplied_percentage = Decimal(str(self.percentage_loss))
            if abs(expected_percentage - supplied_percentage) > Decimal("0.000000001"):
                raise ValueError("percentage_loss must equal absolute_loss divided by base_value")
        return self


class SensitivityResult(DomainModel):
    parameter: NonEmptyString
    value: ConfigurationScalar
    metrics: dict[NonEmptyString, float]


class CandidateConfiguration(DomainModel):
    candidate_id: NonEmptyString
    parameters: dict[NonEmptyString, ConfigurationScalar]
    metrics: dict[NonEmptyString, float]
    selected: bool


class BacktestReport(DomainModel):
    report_id: NonEmptyString
    configuration_version: NonEmptyString
    evaluation_start: AwareDatetime
    evaluation_end: AwareDatetime
    metrics: dict[NonEmptyString, float]
    sensitivity_results: tuple[SensitivityResult, ...]
    candidate_configurations: tuple[CandidateConfiguration, ...]
    historical_case_ids: tuple[NonEmptyString, ...]
    schema_version: NonEmptyString

    @model_validator(mode="after")
    def evaluation_period_is_chronological(self) -> BacktestReport:
        if self.evaluation_end < self.evaluation_start:
            raise ValueError("evaluation_end cannot be before evaluation_start")
        _require_unique_ids(self.candidate_configurations, "candidate_id")
        _require_unique_values(self.historical_case_ids, "historical_case_ids")
        return self


__all__ = [
    "AssetType",
    "AttributionDimension",
    "BacktestReport",
    "CandidateConfiguration",
    "ConfidenceTarget",
    "EntityLink",
    "EventClass",
    "FactorShock",
    "ImpactEstimate",
    "InterpretedEvent",
    "MarketObservation",
    "MarketSnapshot",
    "Portfolio",
    "PortfolioMateriality",
    "Position",
    "ProvenanceMethod",
    "RiskSignal",
    "SensitivityResult",
    "ShockType",
    "ShockUnit",
    "SignalFlag",
    "SourceItem",
    "SourceType",
    "StressAttribution",
    "StressResult",
    "StressResultVersions",
    "StressScenario",
    "VersionMetadata",
]
