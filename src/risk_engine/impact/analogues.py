"""Deterministic historical analogue retrieval with explicit evidence provenance."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime, field_validator, model_validator

from risk_engine.calibration.event_study import EventWindow, MeasurementDimension
from risk_engine.calibration.scenarios import HistoricalFactorShock, JointScenario
from risk_engine.config import AnalogueConfig, AnalogueLevelConfig, MatchingField
from risk_engine.domain import (
    DomainModel,
    EventClass,
    InterpretedEvent,
    NonEmptyString,
    NonNegativeCount,
    ProvenanceMethod,
    Sha256Hex,
    ShockUnit,
)

_ROLE_DIMENSIONS = {
    "equity": {MeasurementDimension.RETURN},
    "rate": {MeasurementDimension.YIELD},
    "spread": {MeasurementDimension.SPREAD},
    "fx": {MeasurementDimension.RETURN, MeasurementDimension.CURRENCY},
    "commodity": {MeasurementDimension.RETURN, MeasurementDimension.PRICE},
    "volatility": {MeasurementDimension.VOLATILITY},
}
_DIMENSION_UNITS = {
    MeasurementDimension.RETURN: {ShockUnit.DECIMAL, ShockUnit.PERCENT},
    MeasurementDimension.YIELD: {ShockUnit.DECIMAL, ShockUnit.PERCENT, ShockUnit.BASIS_POINT},
    MeasurementDimension.SPREAD: {ShockUnit.DECIMAL, ShockUnit.PERCENT, ShockUnit.BASIS_POINT},
    MeasurementDimension.CURRENCY: {
        ShockUnit.DECIMAL,
        ShockUnit.PERCENT,
        ShockUnit.ABSOLUTE,
        ShockUnit.CURRENCY,
    },
    MeasurementDimension.PRICE: {
        ShockUnit.DECIMAL,
        ShockUnit.PERCENT,
        ShockUnit.ABSOLUTE,
        ShockUnit.INDEX_POINT,
    },
    MeasurementDimension.VOLATILITY: {ShockUnit.VOLATILITY_POINT},
}


class ScaleValue(DomainModel):
    name: NonEmptyString
    value: float
    unit: NonEmptyString


class MatchingAttributes(DomainModel):
    """Caller-supplied evidence; None explicitly means unknown, never inferred."""

    event_id: NonEmptyString
    version: NonEmptyString
    available_at: AwareDatetime
    subtype: NonEmptyString | None
    region: NonEmptyString | None
    sector: NonEmptyString | None
    exposure_type: NonEmptyString | None
    scales: tuple[ScaleValue, ...]

    @model_validator(mode="after")
    def unique_scales(self) -> MatchingAttributes:
        if len({scale.name for scale in self.scales}) != len(self.scales):
            raise ValueError("duplicate scale indicator")
        return self


class FactorRole(DomainModel):
    factor_id: NonEmptyString
    role: Literal["equity", "rate", "spread", "fx", "commodity", "volatility"]


class WindowEvidence(DomainModel):
    window: EventWindow
    sessions: tuple[date, ...]
    completed_at: AwareDatetime


class SnapshotEvidence(DomainModel):
    provider: NonEmptyString
    snapshot_id: NonEmptyString
    series_version: NonEmptyString
    sha256: Sha256Hex
    source_terms: NonEmptyString
    available_at: AwareDatetime


class HistoricalAnalogue(DomainModel):
    """An immutable source bundle, retaining every contemporaneous factor vector.

    Evidence kind is a caller assertion about source origin; synthetic test
    fixtures must remain labeled synthetic in application data. Source hashes
    identify externally retained snapshots, not an implicit acquisition service.
    """

    event_id: NonEmptyString
    event_class: EventClass
    event_at: AwareDatetime
    available_at: AwareDatetime
    market_timezone: NonEmptyString
    attributes: MatchingAttributes
    evidence_kind: Literal["observed", "synthetic", "metadata_only"]
    scenario: JointScenario | None
    factor_roles: tuple[FactorRole, ...]
    window_evidence: tuple[WindowEvidence, ...]
    source_evidence: tuple[SnapshotEvidence, ...]

    @field_validator("market_timezone")
    @classmethod
    def require_iana_market_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError) as error:
            raise ValueError("market_timezone must name a valid IANA timezone") from error
        return value

    @model_validator(mode="after")
    def complete_contemporaneous_evidence(self) -> HistoricalAnalogue:
        if self.attributes.event_id != self.event_id:
            raise ValueError("matching attributes disagree with event ID")
        if not self.source_evidence:
            raise ValueError("source evidence is required even for metadata")
        times = [self.event_at, self.attributes.available_at]
        times.extend(source.available_at for source in self.source_evidence)
        if self.evidence_kind == "metadata_only":
            if self.scenario is not None or self.factor_roles or self.window_evidence:
                raise ValueError("metadata-only evidence cannot claim a joint vector")
        else:
            times.extend(self._validate_vector())
        if any(timestamp > self.available_at for timestamp in times):
            raise ValueError("historical bundle availability precedes its evidence")
        return self

    def _validate_vector(self) -> list[datetime]:
        scenario = self.scenario
        if scenario is None:
            raise ValueError("observed/synthetic evidence requires a complete Joint Scenario")
        if scenario.event_id != self.event_id:
            raise ValueError("Joint Scenario disagrees with event ID")
        clock = scenario.clock_decision
        if clock.event_local_time != self.event_at:
            raise ValueError("Joint Scenario event clock disagrees with event timestamp")
        roles = {binding.factor_id: binding.role for binding in self.factor_roles}
        if len(roles) != len(self.factor_roles) or set(roles.values()) != set(_ROLE_DIMENSIONS):
            raise ValueError("complete six-role factor mapping is required")
        windows = {item.window: item for item in self.window_evidence}
        if (
            len(windows) != len(self.window_evidence)
            or set(windows) != {item.window for item in scenario.windows}
            or len(windows) != len(scenario.windows)
        ):
            raise ValueError("registered windows require unique complete timing evidence")
        all_sessions: set[date] = set()
        market_zone = ZoneInfo(self.market_timezone)
        session_by_offset: dict[int, date] = {}
        for timing in windows.values():
            sessions = timing.sessions
            if (
                len(sessions) != timing.window.end - timing.window.start + 1
                or tuple(sorted(set(sessions))) != sessions
                or sessions[-timing.window.start] != clock.session_date
            ):
                raise ValueError("window sessions disagree with registered offsets/event clock")
            if timing.completed_at.astimezone(market_zone).date() < sessions[-1]:
                raise ValueError("window completion precedes final session")
            for offset, session in enumerate(sessions, start=timing.window.start):
                if session_by_offset.setdefault(offset, session) != session:
                    raise ValueError("registered windows disagree on shared session offset")
            all_sessions.update(sessions)
        manifests = {
            (source.provider, source.snapshot_id, source.series_version): source
            for source in self.source_evidence
        }
        if len(manifests) != len(self.source_evidence):
            raise ValueError("duplicate source manifest identity")
        used_sources: set[tuple[str, str, str]] = set()
        times: list[datetime] = [clock.observation_cutoff]
        factor_history: dict[str, HistoricalFactorShock] = {}
        for item in scenario.windows:
            shocks = {shock.factor_id: shock for shock in item.shocks}
            if len(shocks) != len(item.shocks) or shocks.keys() != roles.keys():
                raise ValueError("every window must retain all mapped factors exactly once")
            timing = windows[item.window]
            times.append(timing.completed_at)
            for shock in shocks.values():
                previous = factor_history.setdefault(shock.factor_id, shock)
                if (
                    previous.unit != shock.unit
                    or previous.measurement_dimension != shock.measurement_dimension
                    or previous.source_observations != shock.source_observations
                    or previous.source_benchmark_observations != shock.source_benchmark_observations
                ):
                    raise ValueError("factor units/source history differ across windows")
                if (
                    shock.measurement_dimension not in _ROLE_DIMENSIONS[roles[shock.factor_id]]
                    or shock.unit not in _DIMENSION_UNITS[shock.measurement_dimension]
                ):
                    raise ValueError("factor role, unit, and measurement dimension disagree")
                for observations, is_factor in (
                    (shock.source_observations, True),
                    (shock.source_benchmark_observations, False),
                ):
                    if (
                        len(observations) != len(all_sessions)
                        or {row.session_date for row in observations} != all_sessions
                        or len({row.series_id for row in observations}) != 1
                    ):
                        raise ValueError("factor/benchmark must cover the same complete sessions")
                    if not is_factor and (
                        len({row.unit for row in observations}) != 1
                        or len({row.measurement_dimension for row in observations}) != 1
                    ):
                        raise ValueError("benchmark history must use one unit and dimension")
                    for row in observations:
                        if is_factor and (
                            row.series_id != shock.factor_id
                            or row.unit != shock.unit
                            or row.measurement_dimension != shock.measurement_dimension
                        ):
                            raise ValueError("factor source disagrees with shock identity/unit")
                        if row.unit not in _DIMENSION_UNITS[row.measurement_dimension]:
                            raise ValueError("unsupported observation unit/dimension")
                        local_time = row.observed_at.astimezone(market_zone)
                        if local_time.date() != row.session_date:
                            raise ValueError("observation timestamp disagrees with session date")
                        if local_time.time().replace(tzinfo=None) > clock.close_time or (
                            row.session_date == clock.session_date
                            and row.observed_at > clock.observation_cutoff
                        ):
                            raise ValueError("observation exceeds event-clock close/cutoff")
                        key = (row.provider, row.snapshot_id, row.series_version)
                        if key not in manifests or manifests[key].available_at < row.observed_at:
                            raise ValueError("observation lacks an available bound source manifest")
                        used_sources.add(key)
                        times.append(row.observed_at)
                        if (
                            row.session_date in timing.sessions
                            and row.observed_at > timing.completed_at
                        ):
                            raise ValueError("window completion precedes its observations")
        if used_sources != manifests.keys():
            raise ValueError("source manifest is not bound to any factor/benchmark evidence")
        return times


class LevelSupport(DomainModel):
    name: NonEmptyString
    fields: tuple[MatchingField, ...]
    scales: tuple[str, ...]
    support_count: NonNegativeCount
    candidate_ids: tuple[NonEmptyString, ...]
    missing_query_attributes: tuple[str, ...]

    @model_validator(mode="after")
    def ids_match_support(self) -> LevelSupport:
        if self.support_count != len(self.candidate_ids) or len(set(self.candidate_ids)) != len(
            self.candidate_ids
        ):
            raise ValueError("level support must equal unique candidate count")
        return self


class AnalogueCohort(DomainModel):
    """Effective observed cohort; inadequate support returns no invented scenario.

    support_count is the eligible pool size at the selected level; analogue_count
    is the selected nearest-neighbour count. Hypothetical cohorts are abstentions
    here: a downstream separately governed fallback must supply its own vector.
    """

    analogues: tuple[HistoricalAnalogue, ...]
    distances: tuple[float, ...]
    as_of: AwareDatetime
    query_event_id: NonEmptyString
    query_event_class: EventClass
    matching_attributes: MatchingAttributes | None
    matching_spec: AnalogueConfig
    support_count: NonNegativeCount
    backoff_level: NonEmptyString
    method: ProvenanceMethod
    matching_version: str
    validation_status: Literal["bootstrap_unvalidated", "chronologically_validated"]
    unknown_attributes: tuple[str, ...]
    level_support: tuple[LevelSupport, ...]

    @model_validator(mode="after")
    def cohort_metadata_agrees_with_evidence(self) -> AnalogueCohort:
        if (
            self.matching_version != self.matching_spec.version
            or self.validation_status != self.matching_spec.validation_status
        ):
            raise ValueError("matching version/status disagrees with frozen spec")
        if not self.level_support or (
            self.support_count != self.level_support[-1].support_count
            or self.backoff_level != self.level_support[-1].name
        ):
            raise ValueError("effective support/back-off disagrees with attempted levels")
        if len(self.distances) != len(self.analogues) or any(value < 0 for value in self.distances):
            raise ValueError("distances must correspond to every selected analogue")
        ids = tuple(row.event_id for row in self.analogues)
        if ids != self.level_support[-1].candidate_ids[: len(ids)]:
            raise ValueError("selected IDs must retain the declared nearest-neighbour order")
        if self.method is ProvenanceMethod.EMPIRICAL and (
            len(ids) < self.matching_spec.minimum_support
            or len(ids) > self.matching_spec.nearest_neighbors
            or any(row.evidence_kind != "observed" for row in self.analogues)
        ):
            raise ValueError("empirical cohort requires sufficient observed neighbours")
        if self.method is ProvenanceMethod.HYPOTHETICAL and self.analogues:
            raise ValueError("hypothetical abstention cannot claim an effective historical cohort")
        return self

    @property
    def analogue_count(self) -> int:
        return len(self.analogues)


class AnalogueRepository:
    def __init__(
        self,
        records: tuple[HistoricalAnalogue, ...],
        config: AnalogueConfig,
        attributes: tuple[MatchingAttributes, ...],
    ) -> None:
        self._config = AnalogueConfig.model_validate(config.model_dump())
        self._records = tuple(
            HistoricalAnalogue.model_validate(row.model_dump()) for row in records
        )
        self._attributes = {
            row.event_id: MatchingAttributes.model_validate(row.model_dump()) for row in attributes
        }
        if len({row.event_id for row in self._records}) != len(self._records):
            raise ValueError("duplicate historical event ID")
        if len(self._attributes) != len(attributes):
            raise ValueError("duplicate injected matching event ID")
        for row in (*self._attributes.values(), *(item.attributes for item in self._records)):
            self._validate_units(row)

    def _validate_units(self, attributes: MatchingAttributes) -> None:
        units = {scale.name: scale.unit for scale in self._config.scale_distances}
        if any(
            scale.name in units and scale.unit != units[scale.name] for scale in attributes.scales
        ):
            raise ValueError("scale indicator unit disagrees with frozen matching spec")

    @staticmethod
    def _missing(
        attributes: MatchingAttributes | None,
        fields: tuple[MatchingField, ...],
        scales: tuple[str, ...],
    ) -> tuple[str, ...]:
        known_scales = {row.name for row in attributes.scales} if attributes is not None else set()
        return tuple(
            field
            for field in fields
            if field != "event_class" and (attributes is None or getattr(attributes, field) is None)
        ) + tuple(f"scale:{name}" for name in scales if name not in known_scales)

    def _distance(
        self,
        row: HistoricalAnalogue,
        event: InterpretedEvent,
        attributes: MatchingAttributes | None,
        level: AnalogueLevelConfig,
    ) -> float | None:
        if self._missing(row.attributes, level.fields, level.scales):
            return None
        for field in level.fields:
            query_value = (
                event.event_class if field == "event_class" else getattr(attributes, field)
            )
            historical_value = (
                row.event_class if field == "event_class" else getattr(row.attributes, field)
            )
            if historical_value != query_value:
                return None
        query_scales = (
            {scale.name: scale.value for scale in attributes.scales} if attributes else {}
        )
        historical_scales = {scale.name: scale.value for scale in row.attributes.scales}
        distances = {scale.name: scale for scale in self._config.scale_distances}
        return sum(
            distances[name].weight
            * abs(historical_scales[name] - query_scales[name])
            / distances[name].normalizer
            for name in level.scales
        )

    @staticmethod
    def _available(row: HistoricalAnalogue, as_of: datetime) -> bool:
        times = [row.event_at, row.available_at, row.attributes.available_at]
        times.extend(source.available_at for source in row.source_evidence)
        times.extend(window.completed_at for window in row.window_evidence)
        if row.scenario is not None:
            times.extend(
                (
                    row.scenario.clock_decision.event_local_time,
                    row.scenario.clock_decision.observation_cutoff,
                )
            )
            times.extend(
                observation.observed_at
                for window in row.scenario.windows
                for shock in window.shocks
                for observation in shock.source_observations + shock.source_benchmark_observations
            )
        return all(timestamp < as_of for timestamp in times)

    def match(self, event: InterpretedEvent, as_of: datetime) -> AnalogueCohort:
        if as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        if self._config.frozen_at >= as_of:
            raise ValueError("matching configuration is not available before as_of")
        candidates = tuple(
            row
            for row in self._records
            if row.event_id != event.event_id
            and row.evidence_kind == "observed"
            and row.scenario is not None
            and self._available(row, as_of)
        )
        attributes = self._attributes.get(event.event_id)
        if attributes is not None and attributes.available_at >= as_of:
            attributes = None
        unknown = self._missing(
            attributes,
            tuple(dict.fromkeys(field for level in self._config.levels for field in level.fields)),
            tuple(scale.name for scale in self._config.scale_distances),
        )
        support: list[LevelSupport] = []
        selected: tuple[HistoricalAnalogue, ...] = ()
        selected_distances: tuple[float, ...] = ()
        for level in self._config.levels:
            missing = self._missing(attributes, level.fields, level.scales)
            ranked = []
            if not missing:
                for row in candidates:
                    distance = self._distance(row, event, attributes, level)
                    if distance is not None:
                        ranked.append((distance, row.event_id, row))
            ranked.sort(key=lambda item: (item[0], item[1]))
            support.append(
                LevelSupport(
                    name=level.name,
                    fields=level.fields,
                    scales=level.scales,
                    support_count=len(ranked),
                    candidate_ids=tuple(item[1] for item in ranked),
                    missing_query_attributes=missing,
                )
            )
            if len(ranked) >= self._config.minimum_support:
                nearest = ranked[: self._config.nearest_neighbors]
                selected = tuple(item[2] for item in nearest)
                selected_distances = tuple(item[0] for item in nearest)
                break
        return AnalogueCohort(
            analogues=selected,
            distances=selected_distances,
            as_of=as_of,
            matching_spec=self._config,
            query_event_id=event.event_id,
            query_event_class=event.event_class,
            matching_attributes=attributes,
            support_count=support[-1].support_count,
            backoff_level=support[-1].name,
            method=ProvenanceMethod.EMPIRICAL if selected else ProvenanceMethod.HYPOTHETICAL,
            matching_version=self._config.version,
            validation_status=self._config.validation_status,
            unknown_attributes=unknown,
            level_support=tuple(support),
        )
