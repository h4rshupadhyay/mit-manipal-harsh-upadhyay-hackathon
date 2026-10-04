"""Complete, contemporaneous historical shock vectors."""

from datetime import date, datetime, time, timezone

import pytest

from risk_engine.calibration.event_study import (
    EventReaction,
    EventWindow,
    MeasurementDimension,
    ReturnObservation,
    WindowReaction,
)
from risk_engine.calibration.market_calendar import EventClockDecision
from risk_engine.calibration.scenarios import build_joint_scenario
from risk_engine.domain import ShockUnit

WINDOWS = (EventWindow(start=0, end=0), EventWindow(start=0, end=1))
DECISION = EventClockDecision(
    event_local_time=datetime(2026, 10, 5, 14, tzinfo=timezone.utc),
    session_date=date(2026, 10, 5),
    open_time=time(9, 30),
    close_time=time(16),
    calendar_id="TEST",
    calendar_version="v1",
    calendar_source="synthetic",
    observation_cutoff=datetime(2026, 10, 5, 16, tzinfo=timezone.utc),
    reason="intraday",
)


def reaction(
    series_id: str,
    values: tuple[float, float],
    unit: ShockUnit,
    dimension: MeasurementDimension,
) -> EventReaction:
    observations = (
        ReturnObservation(
            series_id=series_id,
            session_date=date(2026, 10, 5),
            value=values[0],
            observed_at=datetime(2026, 10, 5, 15, tzinfo=timezone.utc),
            provider="synthetic",
            snapshot_id="snapshot-1",
            series_version="v1",
            unit=unit,
            measurement_dimension=dimension,
        ),
    )
    benchmark_observations = (
        observations[0].model_copy(
            update={
                "series_id": f"benchmark-{series_id}",
                "provider": "benchmark-source",
                "snapshot_id": "benchmark-snapshot-2",
                "series_version": "benchmark-v2",
            }
        ),
    )
    return EventReaction(
        event_id="event-1",
        spec_version="spec-1",
        clock_decision=DECISION,
        method="market_adjusted",
        alpha=0.0,
        alpha_unit=unit,
        beta=1.0,
        factor_unit=unit,
        benchmark_unit=unit,
        factor_dimension=dimension,
        benchmark_dimension=dimension,
        abnormal_returns=(),
        windows=tuple(
            WindowReaction(window=window, car=value, unit=unit, measurement_dimension=dimension)
            for window, value in zip(WINDOWS, values, strict=True)
        ),
        estimation_factor_observations=(),
        estimation_benchmark_observations=(),
        event_factor_observations=observations,
        event_benchmark_observations=benchmark_observations,
    )


def six_reactions() -> tuple[EventReaction, ...]:
    return (
        reaction("equity", (-0.03, -0.05), ShockUnit.DECIMAL, MeasurementDimension.RETURN),
        reaction("rate", (12.0, 15.0), ShockUnit.BASIS_POINT, MeasurementDimension.YIELD),
        reaction("spread", (35.0, 40.0), ShockUnit.BASIS_POINT, MeasurementDimension.SPREAD),
        reaction("fx", (0.02, 0.01), ShockUnit.DECIMAL, MeasurementDimension.RETURN),
        reaction("commodity", (-4.0, -6.0), ShockUnit.PERCENT, MeasurementDimension.RETURN),
        reaction(
            "volatility", (2.0, 3.0), ShockUnit.VOLATILITY_POINT, MeasurementDimension.VOLATILITY
        ),
    )


def test_every_registered_window_retains_a_complete_native_joint_vector():
    result = build_joint_scenario(six_reactions())

    assert result.event_id == "event-1"
    assert result.clock_decision == DECISION
    assert result.spec_version == "spec-1"
    assert [item.window for item in result.windows] == list(WINDOWS)
    assert result.windows[0].shocks[0].source_observations[0].snapshot_id == "snapshot-1"
    assert [
        [
            (shock.factor_id, shock.value, shock.unit, shock.measurement_dimension)
            for shock in item.shocks
        ]
        for item in result.windows
    ] == [
        [
            ("equity", -0.03, ShockUnit.DECIMAL, MeasurementDimension.RETURN),
            ("rate", 12.0, ShockUnit.BASIS_POINT, MeasurementDimension.YIELD),
            ("spread", 35.0, ShockUnit.BASIS_POINT, MeasurementDimension.SPREAD),
            ("fx", 0.02, ShockUnit.DECIMAL, MeasurementDimension.RETURN),
            ("commodity", -4.0, ShockUnit.PERCENT, MeasurementDimension.RETURN),
            ("volatility", 2.0, ShockUnit.VOLATILITY_POINT, MeasurementDimension.VOLATILITY),
        ],
        [
            ("equity", -0.05, ShockUnit.DECIMAL, MeasurementDimension.RETURN),
            ("rate", 15.0, ShockUnit.BASIS_POINT, MeasurementDimension.YIELD),
            ("spread", 40.0, ShockUnit.BASIS_POINT, MeasurementDimension.SPREAD),
            ("fx", 0.01, ShockUnit.DECIMAL, MeasurementDimension.RETURN),
            ("commodity", -6.0, ShockUnit.PERCENT, MeasurementDimension.RETURN),
            ("volatility", 3.0, ShockUnit.VOLATILITY_POINT, MeasurementDimension.VOLATILITY),
        ],
    ]


def test_every_shock_retains_factor_and_benchmark_source_observations():
    reactions = six_reactions()

    result = build_joint_scenario(reactions)

    for window in result.windows:
        for shock, source in zip(window.shocks, reactions, strict=True):
            assert shock.source_observations == source.event_factor_observations
            assert shock.source_benchmark_observations == source.event_benchmark_observations
            assert shock.source_observations[0].snapshot_id == "snapshot-1"
            assert shock.source_benchmark_observations[0].series_id == (
                f"benchmark-{shock.factor_id}"
            )
            assert shock.source_benchmark_observations[0].provider == "benchmark-source"
            assert shock.source_benchmark_observations[0].snapshot_id == "benchmark-snapshot-2"
            assert shock.source_benchmark_observations[0].series_version == "benchmark-v2"


def test_missing_benchmark_observations_cannot_produce_an_auditable_shock():
    source = six_reactions()[0]
    missing = source.model_copy(update={"event_benchmark_observations": ()})

    with pytest.raises(ValueError, match="source_benchmark_observations"):
        build_joint_scenario((missing,))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"event_id": "event-2"}, "event ID"),
        ({"spec_version": "spec-2"}, "spec version"),
        ({"clock_decision": DECISION.model_copy(update={"reason": "after_close"})}, "clock"),
    ],
)
def test_reactions_must_have_identical_event_identity(change, message):
    first, second = six_reactions()[:2]
    with pytest.raises(ValueError, match=message):
        build_joint_scenario((first, second.model_copy(update=change)))


def test_reactions_must_have_identical_window_definitions():
    first, second = six_reactions()[:2]
    changed = second.windows[1].model_copy(update={"window": EventWindow(start=-1, end=1)})
    with pytest.raises(ValueError, match="window set"):
        build_joint_scenario(
            (first, second.model_copy(update={"windows": (second.windows[0], changed)}))
        )


def test_reactions_cannot_mix_horizons_even_if_window_count_matches():
    first, second = six_reactions()[:2]
    changed = second.windows[1].model_copy(update={"window": EventWindow(start=0, end=2)})
    with pytest.raises(ValueError, match="window set"):
        build_joint_scenario(
            (first, second.model_copy(update={"windows": (second.windows[0], changed)}))
        )


def test_matching_window_set_can_arrive_in_another_order():
    first, second = six_reactions()[:2]
    reordered = second.model_copy(update={"windows": tuple(reversed(second.windows))})

    result = build_joint_scenario((first, reordered))

    assert [item.shocks[1].value for item in result.windows] == [12.0, 15.0]


def test_cannot_splice_each_factors_independent_tail_from_different_windows():
    first, second = six_reactions()[:2]
    tail_only = first.model_copy(update={"windows": (first.windows[1],)})
    other_tail = second.model_copy(update={"windows": (second.windows[0],)})
    with pytest.raises(ValueError, match="window set"):
        build_joint_scenario((tail_only, other_tail))


def test_rejects_duplicate_factor_series_and_empty_input():
    first = six_reactions()[0]
    with pytest.raises(ValueError, match="at least one"):
        build_joint_scenario(())
    with pytest.raises(ValueError, match="duplicate factor"):
        build_joint_scenario((first, first))


def test_rejects_ambiguous_factor_identity_and_unsupported_dimension_unit():
    first = six_reactions()[0]
    with pytest.raises(ValueError, match="factor series"):
        build_joint_scenario((first.model_copy(update={"event_factor_observations": ()}),))
    bad = first.model_copy(
        update={
            "factor_unit": ShockUnit.CURRENCY,
            "event_factor_observations": tuple(
                row.model_copy(update={"unit": ShockUnit.CURRENCY})
                for row in first.event_factor_observations
            ),
            "windows": tuple(
                item.model_copy(update={"unit": ShockUnit.CURRENCY}) for item in first.windows
            ),
        }
    )
    with pytest.raises(ValueError, match="unit"):
        build_joint_scenario((bad,))
