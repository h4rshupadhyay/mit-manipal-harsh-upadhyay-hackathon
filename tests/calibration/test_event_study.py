"""Known-alpha/beta reactions and explicit data-boundary behavior."""

import csv
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from risk_engine.calibration.event_study import (
    EventStudyEvent,
    EventStudySpec,
    EventWindow,
    MeasurementDimension,
    ReturnObservation,
    compute_event_reaction,
)
from risk_engine.calibration.market_calendar import (
    MarketCalendar,
    map_event_to_session,
)
from risk_engine.domain import ShockUnit

NY = ZoneInfo("America/New_York")
FIXTURE = Path(__file__).parents[1] / "fixtures" / "market-series.csv"


def inputs():
    calendar = MarketCalendar(
        calendar_id="XNYS",
        version="2026.1",
        source="synthetic exchange schedule",
        timezone="America/New_York",
        open_time=time(9, 30),
        holidays=(),
    )
    decision = map_event_to_session(datetime(2026, 10, 5, 14, tzinfo=NY), calendar, time(16))
    event = EventStudyEvent(event_id="event-1", clock_decision=decision)
    spec = EventStudySpec(
        calendar=calendar,
        estimation_sessions=4,
        estimation_gap_sessions=2,
        minimum_estimation_pairs=3,
        windows=(
            EventWindow(start=0, end=0),
            EventWindow(start=0, end=1),
            EventWindow(start=-1, end=1),
            EventWindow(start=-2, end=2),
        ),
        version="event-study-v1",
    )
    rows = list(csv.DictReader(FIXTURE.open(encoding="utf-8", newline="")))
    series = {
        kind: tuple(
            ReturnObservation(
                series_id=kind,
                session_date=date.fromisoformat(row["session_date"]),
                value=float(row["value"]),
                observed_at=datetime.fromisoformat(row["observed_at"]),
                provider=row["provider"],
                snapshot_id=row["snapshot_id"],
                series_version=row["series_version"],
                unit=ShockUnit(row["unit"]),
                measurement_dimension=row["measurement_dimension"],
            )
            for row in rows
            if row["series"] == kind
        )
        for kind in ("factor", "benchmark")
    }
    return event, series["factor"], series["benchmark"], spec


def test_known_alpha_beta_and_all_registered_windows():
    event, factor, benchmark, spec = inputs()

    result = compute_event_reaction(event, factor, benchmark, spec)

    assert result.method == "market_model"
    assert result.alpha == pytest.approx(0.002)
    assert result.beta == pytest.approx(1.5)
    assert result.alpha_unit == ShockUnit.DECIMAL
    assert result.factor_dimension == "return"
    assert result.benchmark_dimension == "return"
    assert result.abnormal_returns[0].value == pytest.approx(0.001)
    assert result.abnormal_returns[2].value == pytest.approx(0.01)
    assert [(item.window.start, item.window.end) for item in result.windows] == [
        (0, 0), (0, 1), (-1, 1), (-2, 2)
    ]
    assert [item.car for item in result.windows] == pytest.approx(
        [0.01, 0.006, 0.008, 0.012]
    )
    assert result.estimation_factor_observations == factor[:4]
    assert result.estimation_benchmark_observations == benchmark[:4]
    assert result.event_factor_observations == factor[4:]
    assert result.event_benchmark_observations == benchmark[4:]
    assert result.clock_decision == event.clock_decision
    assert result.spec_version == "event-study-v1"


def test_missing_estimation_data_uses_declared_market_adjusted_fallback():
    event, factor, benchmark, spec = inputs()

    result = compute_event_reaction(event, factor[4:], benchmark[4:], spec)

    assert result.method == "market_adjusted"
    assert result.alpha == 0.0
    assert result.beta == 1.0
    assert result.windows[0].car == pytest.approx(0.022)
    assert result.estimation_factor_observations == ()


@pytest.mark.parametrize(
    ("dimension", "unit"),
    [
        (MeasurementDimension.RETURN, ShockUnit.VOLATILITY_POINT),
        (MeasurementDimension.YIELD, ShockUnit.CURRENCY),
    ],
)
def test_observation_rejects_incompatible_measurement_unit(dimension, unit):
    _, factor, _, _ = inputs()

    with pytest.raises(ValueError, match="unit.*measurement dimension"):
        ReturnObservation(
            **(factor[0].model_dump() | {"measurement_dimension": dimension, "unit": unit})
        )


@pytest.mark.parametrize(
    ("dimension", "unit"),
    [
        (MeasurementDimension.RETURN, ShockUnit.VOLATILITY_POINT),
        (MeasurementDimension.YIELD, ShockUnit.CURRENCY),
    ],
)
@pytest.mark.parametrize("invalid_series", ["factor", "benchmark", "both"])
def test_computation_rejects_copied_incompatible_measurement_unit(
    dimension, unit, invalid_series
):
    event, factor, benchmark, spec = inputs()
    update = {"measurement_dimension": dimension, "unit": unit}
    if invalid_series in {"factor", "both"}:
        factor = tuple(row.model_copy(update=update) for row in factor)
    if invalid_series in {"benchmark", "both"}:
        benchmark = tuple(row.model_copy(update=update) for row in benchmark)

    with pytest.raises(ValueError, match="unit.*measurement dimension"):
        compute_event_reaction(event, factor, benchmark, spec)


def test_fallback_rejects_matching_copied_incompatible_measurement_units():
    event, factor, benchmark, spec = inputs()
    update = {"unit": ShockUnit.VOLATILITY_POINT}
    factor = tuple(row.model_copy(update=update) for row in factor[4:])
    benchmark = tuple(row.model_copy(update=update) for row in benchmark[4:])

    with pytest.raises(ValueError, match="unit.*measurement dimension"):
        compute_event_reaction(event, factor, benchmark, spec)


@pytest.mark.parametrize(
    ("dimension", "unit", "scale"),
    [
        (MeasurementDimension.RETURN, ShockUnit.PERCENT, 100),
        (MeasurementDimension.YIELD, ShockUnit.BASIS_POINT, 10_000),
        (MeasurementDimension.VOLATILITY, ShockUnit.VOLATILITY_POINT, 1),
    ],
)
@pytest.mark.parametrize(
    ("has_estimation", "method", "event_car"),
    [(True, "market_model", 0.01), (False, "market_adjusted", 0.022)],
)
def test_compatible_native_units_preserve_reactions(
    dimension, unit, scale, has_estimation, method, event_car
):
    event, factor, benchmark, spec = inputs()
    if not has_estimation:
        factor, benchmark = factor[4:], benchmark[4:]
    factor, benchmark = (
        tuple(
            ReturnObservation.model_validate(
                row.model_dump()
                | {"measurement_dimension": dimension, "unit": unit, "value": row.value * scale}
            )
            for row in series
        )
        for series in (factor, benchmark)
    )

    result = compute_event_reaction(event, factor, benchmark, spec)

    assert result.method == method
    assert result.factor_unit == result.benchmark_unit == unit
    assert result.factor_dimension == result.benchmark_dimension == dimension
    assert result.windows[0].car == pytest.approx(event_car * scale)
    assert all(row.unit == unit for row in result.abnormal_returns)
    assert all(row.measurement_dimension == dimension for row in result.abnormal_returns)


def test_partial_estimation_data_remains_visible_in_fallback():
    event, factor, benchmark, spec = inputs()
    incomplete = tuple(row for row in factor if row.session_date != date(2026, 9, 29))

    result = compute_event_reaction(event, incomplete, benchmark, spec)

    assert result.method == "market_adjusted"
    assert len(result.estimation_factor_observations) == 3
    assert len(result.estimation_benchmark_observations) == 4


@pytest.mark.parametrize("series", ["factor", "benchmark"])
def test_missing_event_window_observation_is_rejected(series):
    event, factor, benchmark, spec = inputs()
    if series == "factor":
        factor = tuple(row for row in factor if row.session_date != date(2026, 10, 6))
    else:
        benchmark = tuple(row for row in benchmark if row.session_date != date(2026, 10, 6))

    with pytest.raises(ValueError, match="event-window observation"):
        compute_event_reaction(event, factor, benchmark, spec)


def test_gap_sessions_are_excluded_from_estimation():
    event, factor, benchmark, spec = inputs()
    disturbed = tuple(
        row.model_copy(update={"value": 0.9})
        if row.session_date in {date(2026, 10, 1), date(2026, 10, 2)}
        else row
        for row in factor
    )

    result = compute_event_reaction(event, disturbed, benchmark, spec)

    assert result.alpha == pytest.approx(0.002)
    assert result.beta == pytest.approx(1.5)
    assert all(
        row.session_date <= date(2026, 9, 30)
        for row in result.estimation_factor_observations
    )


def test_pre_cutoff_observation_is_rejected():
    event, factor, benchmark, spec = inputs()
    event = event.model_copy(
        update={
            "clock_decision": map_event_to_session(
                datetime(2026, 10, 5, 14, tzinfo=NY),
                spec.calendar,
                time(16),
                evaluation_end=datetime(2026, 10, 5, 15, tzinfo=NY),
            )
        }
    )

    with pytest.raises(ValueError, match="observation cutoff"):
        compute_event_reaction(event, factor, benchmark, spec)


def test_estimation_gap_cannot_overlap_pre_event_window():
    _, _, _, spec = inputs()

    with pytest.raises(ValueError, match="overlap"):
        EventStudySpec.model_validate(spec.model_dump() | {"estimation_gap_sessions": 1})


def test_mixed_factor_series_is_rejected():
    event, factor, benchmark, spec = inputs()
    mixed = factor[:6] + (factor[6].model_copy(update={"series_id": "different"}),) + factor[7:]

    with pytest.raises(ValueError, match="mixed series"):
        compute_event_reaction(event, mixed, benchmark, spec)


def test_native_basis_point_units_survive_market_model_and_car():
    event, factor, benchmark, spec = inputs()
    factor_bps = tuple(
        ReturnObservation.model_validate(
            row.model_dump() | {
                "value": row.value * 10_000,
                "unit": ShockUnit.BASIS_POINT,
                "measurement_dimension": "yield",
            }
        )
        for row in factor
    )
    benchmark_bps = tuple(
        ReturnObservation.model_validate(
            row.model_dump() | {
                "value": row.value * 10_000,
                "unit": ShockUnit.BASIS_POINT,
                "measurement_dimension": "yield",
            }
        )
        for row in benchmark
    )

    result = compute_event_reaction(event, factor_bps, benchmark_bps, spec)

    assert result.method == "market_model"
    assert result.alpha == pytest.approx(20)
    assert result.alpha_unit == ShockUnit.BASIS_POINT
    assert result.factor_unit == ShockUnit.BASIS_POINT
    assert result.benchmark_unit == ShockUnit.BASIS_POINT
    assert result.factor_dimension == "yield"
    assert result.benchmark_dimension == "yield"
    assert result.event_factor_observations[2].value == pytest.approx(420)
    assert result.event_factor_observations[2].unit == ShockUnit.BASIS_POINT
    assert result.abnormal_returns[2].value == pytest.approx(100)
    assert result.abnormal_returns[2].unit == ShockUnit.BASIS_POINT
    assert result.abnormal_returns[2].measurement_dimension == "yield"
    assert [item.car for item in result.windows] == pytest.approx([100, 60, 80, 120])
    assert all(item.unit == ShockUnit.BASIS_POINT for item in result.windows)
    assert all(item.measurement_dimension == "yield" for item in result.windows)


def test_market_adjusted_fallback_rejects_incomparable_numeric_units():
    event, factor, benchmark, spec = inputs()
    factor_bps = tuple(
        ReturnObservation.model_validate(
            row.model_dump() | {
                "value": row.value * 10_000,
                "unit": ShockUnit.BASIS_POINT,
                "measurement_dimension": "yield",
            }
        )
        for row in factor[4:]
    )

    with pytest.raises(ValueError, match="market-adjusted.*unit"):
        compute_event_reaction(event, factor_bps, benchmark[4:], spec)


def test_market_adjusted_fallback_rejects_ambiguous_absolute_units():
    event, factor, benchmark, spec = inputs()
    factor_absolute = tuple(
        ReturnObservation.model_validate(
            row.model_dump() | {"unit": ShockUnit.ABSOLUTE, "measurement_dimension": "price"}
        )
        for row in factor[4:]
    )
    benchmark_absolute = tuple(
        ReturnObservation.model_validate(
            row.model_dump() | {"unit": ShockUnit.ABSOLUTE, "measurement_dimension": "price"}
        )
        for row in benchmark[4:]
    )

    with pytest.raises(ValueError, match="market-adjusted.*unit"):
        compute_event_reaction(event, factor_absolute, benchmark_absolute, spec)


def test_market_adjusted_fallback_rejects_equal_bps_of_different_dimensions():
    event, factor, benchmark, spec = inputs()
    factor_yield = tuple(
        ReturnObservation.model_validate(
            row.model_dump() | {
                "value": row.value * 10_000,
                "unit": ShockUnit.BASIS_POINT,
                "measurement_dimension": "yield",
            }
        )
        for row in factor[4:]
    )
    benchmark_spread = tuple(
        ReturnObservation.model_validate(
            row.model_dump() | {
                "value": row.value * 10_000,
                "unit": ShockUnit.BASIS_POINT,
                "measurement_dimension": "spread",
            }
        )
        for row in benchmark[4:]
    )

    with pytest.raises(ValueError, match="market-adjusted.*dimension"):
        compute_event_reaction(event, factor_yield, benchmark_spread, spec)


def test_market_model_retains_distinct_factor_and_benchmark_dimensions():
    event, factor, benchmark, spec = inputs()
    factor_yield = tuple(
        ReturnObservation.model_validate(
            row.model_dump() | {
                "value": row.value * 10_000,
                "unit": ShockUnit.BASIS_POINT,
                "measurement_dimension": "yield",
            }
        )
        for row in factor
    )
    benchmark_spread = tuple(
        ReturnObservation.model_validate(
            row.model_dump() | {
                "value": row.value * 10_000,
                "unit": ShockUnit.BASIS_POINT,
                "measurement_dimension": "spread",
            }
        )
        for row in benchmark
    )

    result = compute_event_reaction(event, factor_yield, benchmark_spread, spec)

    assert result.method == "market_model"
    assert result.factor_dimension == "yield"
    assert result.benchmark_dimension == "spread"
