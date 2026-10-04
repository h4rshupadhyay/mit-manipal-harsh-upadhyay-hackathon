"""Map timestamped events to explicit, leakage-safe market sessions."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime, field_validator

from risk_engine.domain import DomainModel


class MarketCalendar(DomainModel):
    """A caller-supplied weekday calendar with explicit holidays and timezone."""

    calendar_id: str
    version: str
    source: str
    timezone: str
    holidays: tuple[date, ...]

    @field_validator("timezone")
    @classmethod
    def timezone_must_exist(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"unknown IANA timezone: {value}") from error
        return value

    @field_validator("holidays")
    @classmethod
    def holidays_must_be_unique(cls, value: tuple[date, ...]) -> tuple[date, ...]:
        if len(value) != len(set(value)):
            raise ValueError("holidays must be unique")
        return value


class EventClockDecision(DomainModel):
    """Auditable event-to-session mapping and its permissible data cutoff."""

    event_local_time: AwareDatetime
    session_date: date
    close_time: time
    calendar_id: str
    calendar_version: str
    calendar_source: str
    observation_cutoff: AwareDatetime
    reason: Literal["intraday", "after_close", "closed_day"]

    @field_validator("observation_cutoff")
    @classmethod
    def cutoff_must_be_aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("observation_cutoff must be timezone-aware")
        return value


def map_event_to_session(
    timestamp: datetime,
    calendar: MarketCalendar,
    close_time: time,
    evaluation_end: datetime | None = None,
) -> EventClockDecision:
    """Map an event to the next eligible close, capped at evaluation_end.

    The calendar explicitly models weekdays plus its supplied holiday dates;
    callers are responsible for supplying the correct exchange holiday set.
    """
    if timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    if close_time.tzinfo is not None:
        raise ValueError("close_time must be a local wall-clock time without timezone")
    if evaluation_end is not None and evaluation_end.utcoffset() is None:
        raise ValueError("evaluation_end must be timezone-aware")

    zone = ZoneInfo(calendar.timezone)
    local_timestamp = timestamp.astimezone(zone)
    holidays = set(calendar.holidays)

    def is_open(day: date) -> bool:
        return day.weekday() < 5 and day not in holidays

    if is_open(local_timestamp.date()) and local_timestamp.time().replace(tzinfo=None) < close_time:
        session_day = local_timestamp.date()
        reason: Literal["intraday", "after_close", "closed_day"] = "intraday"
    else:
        is_after_close = is_open(local_timestamp.date()) and (
            local_timestamp.time().replace(tzinfo=None) >= close_time
        )
        session_day = local_timestamp.date() + timedelta(days=1)
        while not is_open(session_day):
            session_day += timedelta(days=1)
        reason = "after_close" if is_after_close else "closed_day"

    session_close = datetime.combine(session_day, close_time, tzinfo=zone)
    cutoff = session_close
    if evaluation_end is not None:
        local_evaluation_end = evaluation_end.astimezone(zone)
        if local_evaluation_end.date() < session_day:
            raise ValueError("no eligible market observation within evaluation window")
        cutoff = min(cutoff, local_evaluation_end)
        if cutoff < local_timestamp:
            raise ValueError("no eligible market observation within evaluation window")

    return EventClockDecision(
        event_local_time=local_timestamp,
        session_date=session_day,
        close_time=close_time,
        calendar_id=calendar.calendar_id,
        calendar_version=calendar.version,
        calendar_source=calendar.source,
        observation_cutoff=cutoff,
        reason=reason,
    )
