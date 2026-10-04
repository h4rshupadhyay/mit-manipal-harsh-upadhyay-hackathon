"""Deterministic, auditable abnormal returns for historical market events."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, Field, model_validator

from risk_engine.calibration.market_calendar import EventClockDecision, MarketCalendar
from risk_engine.domain import DomainModel, NonEmptyString


class ReturnObservation(DomainModel):
    """One session's decimal return, with the raw source identity preserved."""

    series_id: NonEmptyString
    session_date: date
    return_decimal: float
    observed_at: AwareDatetime
    provider: NonEmptyString
    snapshot_id: NonEmptyString
    series_version: NonEmptyString
    unit: Literal["decimal_return"]


class EventStudyEvent(DomainModel):
    event_id: NonEmptyString
    clock_decision: EventClockDecision


class EventWindow(DomainModel):
    start: int
    end: int

    @model_validator(mode="after")
    def includes_event_session(self) -> EventWindow:
        if self.start > 0 or self.end < 0:
            raise ValueError("event window must include session zero")
        return self


class EventStudySpec(DomainModel):
    calendar: MarketCalendar
    estimation_sessions: int = Field(gt=0)
    estimation_gap_sessions: int = Field(ge=0)
    minimum_estimation_pairs: int = Field(ge=3)
    windows: tuple[EventWindow, ...] = Field(min_length=1)
    version: NonEmptyString

    @model_validator(mode="after")
    def estimation_and_windows_are_consistent(self) -> EventStudySpec:
        if self.minimum_estimation_pairs > self.estimation_sessions:
            raise ValueError("minimum_estimation_pairs exceeds estimation_sessions")
        keys = [(window.start, window.end) for window in self.windows]
        if len(keys) != len(set(keys)):
            raise ValueError("event windows must be unique")
        if self.estimation_gap_sessions < -min(window.start for window in self.windows):
            raise ValueError("estimation window and pre-event window overlap")
        return self


class AbnormalReturn(DomainModel):
    session_offset: int
    session_date: date
    return_decimal: float


class WindowReaction(DomainModel):
    window: EventWindow
    car_decimal: float


class EventReaction(DomainModel):
    event_id: NonEmptyString
    spec_version: NonEmptyString
    clock_decision: EventClockDecision
    method: Literal["market_model", "market_adjusted"]
    alpha: float
    beta: float
    abnormal_returns: tuple[AbnormalReturn, ...]
    windows: tuple[WindowReaction, ...]
    estimation_factor_observations: tuple[ReturnObservation, ...]
    estimation_benchmark_observations: tuple[ReturnObservation, ...]
    event_factor_observations: tuple[ReturnObservation, ...]
    event_benchmark_observations: tuple[ReturnObservation, ...]


def _session(day: date, offset: int, calendar: MarketCalendar) -> date:
    holidays = set(calendar.holidays)
    direction = 1 if offset >= 0 else -1
    remaining = abs(offset)
    while remaining:
        day += timedelta(days=direction)
        if day.weekday() < 5 and day not in holidays:
            remaining -= 1
    return day


def _index_series(
    observations: tuple[ReturnObservation, ...],
    calendar: MarketCalendar,
    decision: EventClockDecision,
) -> dict[date, ReturnObservation]:
    by_date: dict[date, ReturnObservation] = {}
    zone = ZoneInfo(calendar.timezone)
    series_ids = {row.series_id for row in observations}
    if len(series_ids) > 1:
        raise ValueError("mixed series IDs in market observations")
    for row in observations:
        if row.session_date in by_date:
            raise ValueError(f"duplicate market observation on {row.session_date}")
        local_observed_at = row.observed_at.astimezone(zone)
        if local_observed_at.date() != row.session_date:
            raise ValueError("observation timestamp disagrees with session date")
        if local_observed_at.time().replace(tzinfo=None) > decision.close_time:
            raise ValueError("observation is after market session close")
        if (
            row.session_date == decision.session_date
            and row.observed_at > decision.observation_cutoff
        ):
            raise ValueError("observation exceeds event-clock observation cutoff")
        by_date[row.session_date] = row
    return by_date


def compute_event_reaction(
    event: EventStudyEvent,
    factor_series: tuple[ReturnObservation, ...],
    benchmark_series: tuple[ReturnObservation, ...],
    spec: EventStudySpec,
) -> EventReaction:
    """Fit normal returns before the gap, then compute all registered AR/CAR windows.

    A missing or degenerate estimation sample selects market adjustment. Every
    event-window pair is mandatory; no return is imputed or refreshed.
    """
    decision = event.clock_decision
    calendar = spec.calendar
    if (
        decision.calendar_id != calendar.calendar_id
        or decision.calendar_version != calendar.version
        or decision.calendar_source != calendar.source
        or decision.open_time != calendar.open_time
    ):
        raise ValueError("event-clock decision and study calendar disagree")

    factor = _index_series(factor_series, calendar, decision)
    benchmark = _index_series(benchmark_series, calendar, decision)
    minimum_offset = min(window.start for window in spec.windows)
    maximum_offset = max(window.end for window in spec.windows)
    event_offsets = range(minimum_offset, maximum_offset + 1)
    event_dates = tuple(
        _session(decision.session_date, offset, calendar) for offset in event_offsets
    )
    for day in event_dates:
        if day not in factor or day not in benchmark:
            raise ValueError(f"missing event-window observation on {day}")

    estimation_offsets = range(
        -spec.estimation_gap_sessions - spec.estimation_sessions,
        -spec.estimation_gap_sessions,
    )
    estimation_dates = tuple(
        _session(decision.session_date, offset, calendar) for offset in estimation_offsets
    )
    complete_estimation = all(day in factor and day in benchmark for day in estimation_dates)
    estimation_factor = tuple(factor[day] for day in estimation_dates if day in factor)
    estimation_benchmark = tuple(benchmark[day] for day in estimation_dates if day in benchmark)
    distinct_market_returns = {row.return_decimal for row in estimation_benchmark}
    if (
        complete_estimation
        and len(estimation_factor) >= spec.minimum_estimation_pairs
        and len(distinct_market_returns) >= 2
    ):
        import statsmodels.api as sm  # type: ignore[import-untyped]

        explanatory = sm.add_constant(
            [row.return_decimal for row in estimation_benchmark], has_constant="add"
        )
        fit = sm.OLS([row.return_decimal for row in estimation_factor], explanatory).fit()
        alpha, beta = float(fit.params[0]), float(fit.params[1])
        method: Literal["market_model", "market_adjusted"] = "market_model"
    else:
        alpha, beta = 0.0, 1.0
        method = "market_adjusted"

    abnormal = tuple(
        AbnormalReturn(
            session_offset=offset,
            session_date=day,
            return_decimal=(
                factor[day].return_decimal - (alpha + beta * benchmark[day].return_decimal)
            ),
        )
        for offset, day in zip(event_offsets, event_dates, strict=True)
    )
    windows = tuple(
        WindowReaction(
            window=window,
            car_decimal=sum(
                row.return_decimal
                for row in abnormal
                if window.start <= row.session_offset <= window.end
            ),
        )
        for window in spec.windows
    )
    return EventReaction(
        event_id=event.event_id,
        spec_version=spec.version,
        clock_decision=decision,
        method=method,
        alpha=alpha,
        beta=beta,
        abnormal_returns=abnormal,
        windows=windows,
        estimation_factor_observations=estimation_factor,
        estimation_benchmark_observations=estimation_benchmark,
        event_factor_observations=tuple(factor[day] for day in event_dates),
        event_benchmark_observations=tuple(benchmark[day] for day in event_dates),
    )
