"""Leakage-safe retrieval over explicit, frozen matching inputs."""

from copy import deepcopy
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from risk_engine.calibration.event_study import (
    EventReaction,
    EventWindow,
    MeasurementDimension,
    ReturnObservation,
    WindowReaction,
)
from risk_engine.calibration.market_calendar import EventClockDecision
from risk_engine.calibration.scenarios import (
    HistoricalFactorShock,
    JointScenario,
    JointWindowScenario,
    build_joint_scenario,
)
from risk_engine.config import AnalogueConfig, AppConfig
from risk_engine.domain import EventClass, InterpretedEvent, ProvenanceMethod, ShockUnit
from risk_engine.impact.analogues import (
    AnalogueRepository,
    FactorRole,
    HistoricalAnalogue,
    MatchingAttributes,
    ScaleValue,
    SnapshotEvidence,
    WindowEvidence,
)

PAST = datetime(2025, 6, 2, 12, tzinfo=timezone.utc)
AS_OF = datetime(2026, 1, 1, tzinfo=timezone.utc)
ROLES = (
    ("equity", MeasurementDimension.RETURN, ShockUnit.DECIMAL),
    ("rate", MeasurementDimension.YIELD, ShockUnit.BASIS_POINT),
    ("spread", MeasurementDimension.SPREAD, ShockUnit.BASIS_POINT),
    ("fx", MeasurementDimension.CURRENCY, ShockUnit.PERCENT),
    ("commodity", MeasurementDimension.PRICE, ShockUnit.PERCENT),
    ("volatility", MeasurementDimension.VOLATILITY, ShockUnit.VOLATILITY_POINT),
)


def config():
    result = AppConfig.load(Path("config/default.toml")).analogues
    assert result is not None
    return result


def attributes(event_id: str, *, size: float = 1_000_000, **changes):
    return MatchingAttributes.model_validate(
        {
            "event_id": event_id,
            "version": "attributes-v1",
            "available_at": PAST,
            "subtype": "conflict",
            "region": "Europe",
            "sector": "energy",
            "exposure_type": "bond",
            "scales": (ScaleValue(name="size", value=size, unit="USD"),),
            **changes,
        }
    )


def historical(event_id: str, *, evidence_kind: str = "observed", **changes):
    observed_at = PAST + timedelta(hours=4)
    window = EventWindow(start=0, end=0)
    clock = EventClockDecision(
        event_local_time=PAST,
        session_date=PAST.date(),
        open_time=time(9),
        close_time=time(17),
        calendar_id="test",
        calendar_version="v1",
        calendar_source="fixture",
        observation_cutoff=observed_at,
        reason="intraday",
    )
    shocks = []
    for role, dimension, unit in ROLES:
        factor = ReturnObservation(
            series_id=role,
            session_date=PAST.date(),
            value=0.1,
            observed_at=observed_at,
            provider="fixture",
            snapshot_id="snapshot",
            series_version="v1",
            unit=unit,
            measurement_dimension=dimension,
        )
        benchmark = factor.model_copy(update={"series_id": f"benchmark-{role}"})
        shocks.append(
            HistoricalFactorShock(
                factor_id=role,
                value=-0.1,
                unit=unit,
                measurement_dimension=dimension,
                source_observations=(factor,),
                source_benchmark_observations=(benchmark,),
            )
        )
    scenario = JointScenario(
        event_id=event_id,
        clock_decision=clock,
        spec_version="study-v1",
        windows=(JointWindowScenario(window=window, shocks=tuple(shocks)),),
    )
    return HistoricalAnalogue.model_validate(
        {
            "event_id": event_id,
            "event_class": EventClass.GEOPOLITICAL,
            "event_at": PAST,
            "available_at": observed_at + timedelta(hours=1),
            "market_timezone": "UTC",
            "attributes": attributes(event_id),
            "evidence_kind": evidence_kind,
            "scenario": None if evidence_kind == "metadata_only" else scenario,
            "factor_roles": ()
            if evidence_kind == "metadata_only"
            else tuple(FactorRole(factor_id=role, role=role) for role, _, _ in ROLES),
            "window_evidence": ()
            if evidence_kind == "metadata_only"
            else (
                WindowEvidence(window=window, sessions=(PAST.date(),), completed_at=observed_at),
            ),
            "source_evidence": (
                SnapshotEvidence(
                    provider="fixture",
                    snapshot_id="snapshot",
                    series_version="v1",
                    sha256="a" * 64,
                    source_terms="project-authored test data",
                    available_at=observed_at,
                ),
            ),
            **changes,
        }
    )


def event() -> InterpretedEvent:
    return InterpretedEvent(
        event_id="query",
        source_item_id="source",
        entity_links=(),
        event_class=EventClass.GEOPOLITICAL,
        sentiment=-0.5,
        classification_confidence=0.9,
        rationale="test",
        evidence=("evidence",),
        eligible_for_automatic_stress=False,
    )


def test_empty_repository_discloses_unknown_attributes_and_hypothetical_support() -> None:
    cohort = AnalogueRepository((), config(), ()).match(
        event(), datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    assert cohort.method is ProvenanceMethod.HYPOTHETICAL
    assert cohort.support_count == 0
    assert cohort.analogues == ()
    assert cohort.backoff_level == "global"
    assert set(cohort.unknown_attributes) == {
        "subtype",
        "region",
        "sector",
        "exposure_type",
        "scale:size",
    }


def test_naive_scoring_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        AnalogueRepository((), config(), ()).match(event(), datetime(2026, 1, 1))


def test_only_complete_observed_available_history_counts_as_support() -> None:
    records = (
        historical("observed"),
        historical("synthetic", evidence_kind="synthetic"),
        historical("metadata", evidence_kind="metadata_only"),
        historical("unavailable", available_at=AS_OF),
        historical("query"),
    )
    cohort = AnalogueRepository(records, config(), (attributes("query"),)).match(event(), AS_OF)
    assert cohort.method is ProvenanceMethod.HYPOTHETICAL
    assert cohort.support_count == 1
    assert cohort.analogues == ()
    assert cohort.backoff_level == "global"
    assert cohort.level_support[-1].candidate_ids == ("observed",)
    assert cohort.validation_status == "bootstrap_unvalidated"


@pytest.mark.parametrize(
    "damage",
    [
        "missing_scenario",
        "missing_role",
        "duplicate_factor",
        "wrong_dimension",
        "wrong_unit",
        "missing_window",
        "incomplete_sessions",
        "wrong_window_session",
        "missing_source",
        "wrong_source",
        "early_availability",
        "early_completion",
        "wrong_event",
        "wrong_attribute_id",
        "wrong_clock",
        "missing_factor_observation",
        "wrong_observation_series",
        "wrong_observation_unit",
        "future_session_date",
    ],
)
def test_historical_evidence_rejects_incomplete_or_inconsistent_vectors(damage: str) -> None:
    payload = historical("damaged").model_dump()
    window = payload["scenario"]["windows"][0]
    shock = window["shocks"][0]
    if damage == "missing_scenario":
        payload["scenario"] = None
    elif damage == "missing_role":
        payload["factor_roles"] = payload["factor_roles"][:-1]
    elif damage == "duplicate_factor":
        window["shocks"] = (*window["shocks"], shock)
    elif damage == "wrong_dimension":
        shock["measurement_dimension"] = MeasurementDimension.YIELD
    elif damage == "wrong_unit":
        shock["unit"] = ShockUnit.BASIS_POINT
    elif damage == "missing_window":
        payload["window_evidence"] = ()
    elif damage == "incomplete_sessions":
        payload["window_evidence"][0]["sessions"] = ()
    elif damage == "wrong_window_session":
        payload["window_evidence"][0]["sessions"] = (PAST.date() + timedelta(days=1),)
    elif damage == "missing_source":
        payload["source_evidence"] = ()
    elif damage == "wrong_source":
        payload["source_evidence"][0]["snapshot_id"] = "unbound"
    elif damage == "early_availability":
        payload["available_at"] = PAST
    elif damage == "early_completion":
        payload["window_evidence"][0]["completed_at"] = PAST
    elif damage == "wrong_event":
        payload["scenario"]["event_id"] = "another"
    elif damage == "wrong_attribute_id":
        payload["attributes"]["event_id"] = "another"
    elif damage == "wrong_clock":
        payload["scenario"]["clock_decision"]["event_local_time"] = PAST + timedelta(hours=1)
    elif damage == "missing_factor_observation":
        shock["source_observations"] = ()
    elif damage == "wrong_observation_series":
        shock["source_observations"][0]["series_id"] = "another"
    elif damage == "wrong_observation_unit":
        shock["source_observations"][0]["unit"] = ShockUnit.BASIS_POINT
    elif damage == "future_session_date":
        shock["source_observations"][0]["session_date"] = AS_OF.date()
    with pytest.raises(ValueError):
        HistoricalAnalogue.model_validate(payload)


def test_frozen_nearest_neighbors_use_normalized_distance_then_event_id() -> None:
    matching = config().model_copy(update={"minimum_support": 2, "nearest_neighbors": 2})
    records = tuple(
        historical(key, attributes=attributes(key, size=size))
        for key, size in (
            ("a", 1_300_000),
            ("c", 1_100_000),
            ("b", 900_000),
            ("d", 2_000_000),
        )
    )
    cohort = AnalogueRepository(records, matching, (attributes("query"),)).match(event(), AS_OF)
    assert cohort.method is ProvenanceMethod.EMPIRICAL
    assert cohort.validation_status == "bootstrap_unvalidated"
    assert cohort.backoff_level == "exact"
    assert cohort.support_count == 4
    assert cohort.analogue_count == 2
    assert [row.event_id for row in cohort.analogues] == ["b", "c"]
    assert cohort.distances == pytest.approx((0.1, 0.1))
    assert cohort.level_support[0].candidate_ids == ("b", "c", "a", "d")
    assert cohort.unknown_attributes == ()
    assert cohort.analogues[0].scenario == records[2].scenario


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"subtype": "different"}, "class_region_sector_exposure"),
        ({"sector": "different"}, "class_region_exposure"),
        ({"region": "different"}, "class_exposure"),
        ({"exposure_type": "different"}, "class"),
        ({"event_class": EventClass.CREDIT_DEFAULT}, "global"),
    ],
)
def test_declared_backoff_stops_at_first_supported_level(changes, expected: str) -> None:
    attributes_changes = {key: value for key, value in changes.items() if key != "event_class"}
    records = tuple(
        historical(
            str(i),
            attributes=attributes(str(i), **attributes_changes),
            **({"event_class": changes["event_class"]} if "event_class" in changes else {}),
        )
        for i in range(3)
    )
    cohort = AnalogueRepository(records, config(), (attributes("query"),)).match(event(), AS_OF)
    assert cohort.backoff_level == expected
    assert cohort.support_count == 3
    assert cohort.method is ProvenanceMethod.EMPIRICAL
    assert [level.name for level in cohort.level_support] == [
        level.name for level in config().levels[: len(cohort.level_support)]
    ]
    assert all(level.support_count == 0 for level in cohort.level_support[:-1])


def test_unknown_scale_skips_distance_levels_and_is_disclosed() -> None:
    records = tuple(historical(str(i)) for i in range(3))
    cohort = AnalogueRepository(records, config(), (attributes("query", scales=()),)).match(
        event(), AS_OF
    )
    assert cohort.backoff_level == "class"
    assert cohort.unknown_attributes == ("scale:size",)
    assert cohort.level_support[0].missing_query_attributes == ("scale:size",)
    assert cohort.level_support[-1].scales == ()


def test_cohort_carries_scoring_time_and_complete_frozen_spec() -> None:
    cohort = AnalogueRepository((), config(), ()).match(event(), AS_OF)
    assert cohort.as_of == AS_OF
    assert cohort.matching_spec == config()


def test_configured_fields_and_multiple_weighted_native_scale_distances() -> None:
    payload = config().model_dump()
    payload.update(
        {
            "minimum_support": 1,
            "nearest_neighbors": 1,
            "scale_distances": (
                {"name": "size", "unit": "USD", "normalizer": 100_000, "weight": 2},
                {"name": "leverage", "unit": "ratio", "normalizer": 0.1, "weight": 0.5},
            ),
            "levels": (
                {
                    "name": "region",
                    "fields": ("event_class", "region"),
                    "scales": ("size", "leverage"),
                },
                {"name": "global", "fields": (), "scales": ()},
            ),
        }
    )
    matching = AnalogueConfig.model_validate(payload)
    query = attributes(
        "query",
        scales=(
            ScaleValue(name="size", unit="USD", value=1_000_000),
            ScaleValue(name="leverage", unit="ratio", value=0.1),
        ),
    )
    candidate = attributes(
        "candidate",
        sector="different",
        scales=(
            ScaleValue(name="size", unit="USD", value=1_200_000),
            ScaleValue(name="leverage", unit="ratio", value=0.3),
        ),
    )
    cohort = AnalogueRepository(
        (historical("candidate", attributes=candidate),), matching, (query,)
    ).match(event(), AS_OF)
    assert cohort.backoff_level == "region"
    assert cohort.distances == pytest.approx((5.0,))


@pytest.mark.parametrize("as_of", [PAST, PAST + timedelta(hours=4), PAST + timedelta(hours=5)])
def test_history_at_event_data_or_bundle_availability_boundary_is_excluded(as_of) -> None:
    cohort = AnalogueRepository((historical("history"),), config(), (attributes("query"),)).match(
        event(), as_of
    )
    assert cohort.support_count == 0
    assert cohort.method is ProvenanceMethod.HYPOTHETICAL


def test_future_query_attributes_cannot_constrain_historical_matching() -> None:
    records = tuple(historical(str(i)) for i in range(3))
    query = attributes("query", available_at=AS_OF, region="unavailable knowledge")
    cohort = AnalogueRepository(records, config(), (query,)).match(event(), AS_OF)
    assert cohort.backoff_level == "class"
    assert "region" in cohort.unknown_attributes


@pytest.mark.parametrize("target", ["frozen_at", "fit_as_of"])
def test_future_configuration_is_rejected(target: str) -> None:
    payload = config().model_dump()
    payload[target] = AS_OF
    with pytest.raises(ValueError):
        matching = AnalogueConfig.model_validate(payload)
        AnalogueRepository((), matching, ()).match(event(), AS_OF)


def test_distance_overflow_is_rejected_instead_of_silently_backing_off() -> None:
    matching = config().model_copy(update={"minimum_support": 1, "nearest_neighbors": 1})
    row = historical("huge", attributes=attributes("huge", size=1e308))
    with pytest.raises(ValueError, match="distance"):
        AnalogueRepository((row,), matching, (attributes("query", size=-1e308),)).match(
            event(), AS_OF
        )


def test_multiple_windows_cannot_change_factor_units_or_source_history() -> None:
    payload = historical("multi").model_dump()
    first = payload["scenario"]["windows"][0]
    for shock in first["shocks"]:
        for name in ("source_observations", "source_benchmark_observations"):
            original = shock[name][0]
            previous = {
                **original,
                "session_date": PAST.date() - timedelta(days=1),
                "observed_at": original["observed_at"] - timedelta(days=1),
            }
            shock[name] = (previous, original)
    second = deepcopy(first)
    second["window"] = {"start": -1, "end": 0}
    second["shocks"][0]["unit"] = ShockUnit.PERCENT
    for name in ("source_observations", "source_benchmark_observations"):
        for row in second["shocks"][0][name]:
            row["unit"] = ShockUnit.PERCENT
    payload["scenario"]["windows"] = (first, second)
    timing = deepcopy(payload["window_evidence"][0])
    timing["window"] = second["window"]
    timing["sessions"] = (PAST.date() - timedelta(days=1), PAST.date())
    payload["window_evidence"] = (*payload["window_evidence"], timing)
    with pytest.raises(ValueError, match="across windows"):
        HistoricalAnalogue.model_validate(payload)


def test_dst_sessions_survive_json_round_trip_with_market_local_closes() -> None:
    market_zone = ZoneInfo("America/New_York")
    event_at = datetime(2025, 10, 31, 12, tzinfo=market_zone)
    sessions = (datetime(2025, 10, 31).date(), datetime(2025, 11, 3).date())
    close_times = (
        datetime(2025, 10, 31, 17, tzinfo=market_zone),
        datetime(2025, 11, 3, 17, tzinfo=market_zone),
    )
    payload = historical("dst").model_dump()
    payload["event_at"] = event_at
    payload["market_timezone"] = "America/New_York"
    payload["scenario"]["clock_decision"].update(
        event_local_time=event_at,
        session_date=sessions[0],
        observation_cutoff=datetime(2025, 10, 31, 23, tzinfo=market_zone),
    )
    payload["scenario"]["windows"][0]["window"] = {"start": 0, "end": 1}
    for shock in payload["scenario"]["windows"][0]["shocks"]:
        for source_name in ("source_observations", "source_benchmark_observations"):
            original = shock[source_name][0]
            shock[source_name] = tuple(
                {
                    **original,
                    "session_date": session,
                    "observed_at": close.astimezone(timezone.utc),
                }
                for session, close in zip(sessions, close_times, strict=True)
            )
    payload["window_evidence"] = (
        {
            "window": {"start": 0, "end": 1},
            "sessions": sessions,
            "completed_at": close_times[-1],
        },
    )
    payload["available_at"] = datetime(2025, 11, 3, 18, tzinfo=market_zone)
    payload["source_evidence"][0]["available_at"] = close_times[-1]
    record = HistoricalAnalogue.model_validate(payload)
    replayed = HistoricalAnalogue.model_validate_json(record.model_dump_json())
    assert replayed.scenario == record.scenario


@pytest.mark.parametrize("damage", ["unit", "dimension"])
def test_benchmark_history_rejects_mixed_units_or_dimensions(damage: str) -> None:
    payload = historical("benchmark-mix").model_dump()
    session = PAST.date() + timedelta(days=1)
    completed = PAST + timedelta(days=1, hours=4)
    window_data = payload["scenario"]["windows"][0]
    window_data["window"] = {"start": 0, "end": 1}
    for shock in window_data["shocks"]:
        for source_name in ("source_observations", "source_benchmark_observations"):
            original = shock[source_name][0]
            second = {
                **original,
                "session_date": session,
                "observed_at": original["observed_at"] + timedelta(days=1),
            }
            if source_name == "source_benchmark_observations" and shock["factor_id"] == "equity":
                if damage == "unit":
                    second["unit"] = ShockUnit.PERCENT
                else:
                    second["measurement_dimension"] = MeasurementDimension.PRICE
                    second["unit"] = ShockUnit.ABSOLUTE
            shock[source_name] = (original, second)
    payload["window_evidence"] = (
        {
            "window": {"start": 0, "end": 1},
            "sessions": (PAST.date(), session),
            "completed_at": completed,
        },
    )
    payload["source_evidence"][0]["available_at"] = completed
    payload["available_at"] = completed + timedelta(hours=1)
    with pytest.raises(ValueError, match="benchmark history"):
        HistoricalAnalogue.model_validate(payload)


def test_uniform_benchmark_unit_may_differ_from_factor_unit() -> None:
    payload = historical("benchmark-unit").model_dump()
    shock = payload["scenario"]["windows"][0]["shocks"][0]
    benchmark = dict(shock["source_benchmark_observations"][0])
    benchmark["unit"] = ShockUnit.PERCENT
    shock["source_benchmark_observations"] = (benchmark,)
    record = HistoricalAnalogue.model_validate(payload)
    assert record.scenario is not None


@pytest.mark.parametrize("timezone_value", [None, "+05:30"])
def test_market_timezone_must_be_explicit_and_iana(timezone_value: str | None) -> None:
    payload = historical("timezone").model_dump()
    if timezone_value is None:
        payload.pop("market_timezone")
    else:
        payload["market_timezone"] = timezone_value
    with pytest.raises(ValueError):
        HistoricalAnalogue.model_validate(payload)


def test_cohort_retains_query_attribute_version_and_rejects_inconsistent_counts() -> None:
    from risk_engine.impact.analogues import AnalogueCohort

    query = attributes("query")
    cohort = AnalogueRepository((), config(), (query,)).match(event(), AS_OF)
    assert cohort.matching_attributes == query
    assert cohort.query_event_id == "query"
    assert cohort.query_event_class == EventClass.GEOPOLITICAL
    with pytest.raises(ValueError):
        AnalogueCohort.model_validate({**cohort.model_dump(), "support_count": -1})


def test_sparse_exact_pool_backs_off_and_discloses_both_support_counts() -> None:
    records = (
        historical("a"),
        historical("b"),
        historical("c", attributes=attributes("c", subtype="different")),
    )
    cohort = AnalogueRepository(records, config(), (attributes("query"),)).match(event(), AS_OF)
    assert cohort.backoff_level == "class_region_sector_exposure"
    assert [level.support_count for level in cohort.level_support] == [2, 3]


@pytest.mark.parametrize("field", ["support_count", "matching_version", "distances"])
def test_public_cohort_rejects_metadata_inconsistent_with_its_evidence(field: str) -> None:
    from risk_engine.impact.analogues import AnalogueCohort

    matching = config().model_copy(update={"minimum_support": 1, "nearest_neighbors": 1})
    cohort = AnalogueRepository((historical("a"),), matching, (attributes("query"),)).match(
        event(), AS_OF
    )
    payload = cohort.model_dump()
    payload[field] = {"support_count": 9, "matching_version": "unrelated", "distances": ()}[field]
    with pytest.raises(ValueError):
        AnalogueCohort.model_validate(payload)


def test_overlapping_windows_require_consistent_shared_session_offsets() -> None:
    payload = historical("overlap").model_dump()
    first = payload["scenario"]["windows"][0]
    first["window"] = {"start": 0, "end": 1}
    for shock in first["shocks"]:
        for name in ("source_observations", "source_benchmark_observations"):
            original = shock[name][0]
            shock[name] = tuple(
                {
                    **original,
                    "session_date": original["session_date"] + timedelta(days=i),
                    "observed_at": original["observed_at"] + timedelta(days=i),
                }
                for i in range(4)
            )
    second = deepcopy(first)
    second["window"] = {"start": 0, "end": 2}
    payload["scenario"]["windows"] = (first, second)
    completed = PAST + timedelta(days=3, hours=4)
    payload["window_evidence"] = (
        {
            "window": first["window"],
            "sessions": (PAST.date(), PAST.date() + timedelta(days=1)),
            "completed_at": completed,
        },
        {
            "window": second["window"],
            "sessions": (
                PAST.date(),
                PAST.date() + timedelta(days=2),
                PAST.date() + timedelta(days=3),
            ),
            "completed_at": completed,
        },
    )
    payload["source_evidence"][0]["available_at"] = completed
    payload["available_at"] = completed + timedelta(hours=1)
    with pytest.raises(ValueError, match="shared session offset"):
        HistoricalAnalogue.model_validate(payload)


def test_public_scenario_producer_retains_full_union_and_distinct_benchmark_sources() -> None:
    original = historical("producer")
    assert original.scenario is not None
    windows = (EventWindow(start=0, end=0), EventWindow(start=0, end=1))
    reactions = []
    for shock in original.scenario.windows[0].shocks:
        factor = shock.source_observations[0]
        benchmark = shock.source_benchmark_observations[0].model_copy(
            update={
                "provider": "benchmark-provider",
                "snapshot_id": "benchmark-snapshot",
                "series_version": "benchmark-v1",
            }
        )

        def following(row):
            return row.model_copy(
                update={
                    "session_date": row.session_date + timedelta(days=1),
                    "observed_at": row.observed_at + timedelta(days=1),
                }
            )

        reactions.append(
            EventReaction(
                event_id="producer",
                spec_version="study-v1",
                clock_decision=original.scenario.clock_decision,
                method="market_adjusted",
                alpha=0,
                alpha_unit=shock.unit,
                beta=1,
                factor_unit=shock.unit,
                benchmark_unit=shock.unit,
                factor_dimension=shock.measurement_dimension,
                benchmark_dimension=shock.measurement_dimension,
                abnormal_returns=(),
                windows=tuple(
                    WindowReaction(
                        window=window,
                        car=shock.value,
                        unit=shock.unit,
                        measurement_dimension=shock.measurement_dimension,
                    )
                    for window in windows
                ),
                estimation_factor_observations=(),
                estimation_benchmark_observations=(),
                event_factor_observations=(factor, following(factor)),
                event_benchmark_observations=(benchmark, following(benchmark)),
            )
        )
    scenario = build_joint_scenario(reactions)
    complete = PAST + timedelta(days=1, hours=4)
    record = historical(
        "producer",
        scenario=scenario,
        available_at=complete + timedelta(hours=1),
        window_evidence=(
            WindowEvidence(
                window=windows[0], sessions=(PAST.date(),), completed_at=PAST + timedelta(hours=4)
            ),
            WindowEvidence(
                window=windows[1], sessions=(PAST.date(), complete.date()), completed_at=complete
            ),
        ),
        source_evidence=(
            original.source_evidence[0].model_copy(update={"available_at": complete}),
            SnapshotEvidence(
                provider="benchmark-provider",
                snapshot_id="benchmark-snapshot",
                series_version="benchmark-v1",
                sha256="b" * 64,
                source_terms="project-authored test data",
                available_at=complete,
            ),
        ),
    )
    matching = config().model_copy(update={"minimum_support": 1, "nearest_neighbors": 1})
    cohort = AnalogueRepository((record,), matching, (attributes("query"),)).match(event(), AS_OF)
    assert cohort.analogues[0].scenario == scenario
    assert len(cohort.analogues[0].scenario.windows[0].shocks[0].source_observations) == 2
