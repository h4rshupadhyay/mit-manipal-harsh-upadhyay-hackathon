from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from risk_engine.calibration.market_calendar import MarketCalendar, map_event_to_session

NY = ZoneInfo("America/New_York")


def test_intraday_event_maps_to_same_market_session() -> None:
    calendar = MarketCalendar(timezone="America/New_York", holidays=())

    decision = map_event_to_session(
        datetime(2026, 10, 5, 14, 0, tzinfo=NY),
        calendar,
        time(16, 0),
        evaluation_end=datetime(2026, 10, 5, 16, 0, tzinfo=NY),
    )

    assert decision.session_date == date(2026, 10, 5)
    assert decision.observation_cutoff == datetime(2026, 10, 5, 16, 0, tzinfo=NY)
    assert decision.reason == "intraday"


def test_after_close_friday_maps_to_next_open_session() -> None:
    calendar = MarketCalendar(timezone="America/New_York", holidays=())

    decision = map_event_to_session(datetime(2026, 10, 2, 17, 0, tzinfo=NY), calendar, time(16, 0))

    assert decision.session_date == date(2026, 10, 5)
    assert decision.reason == "after_close"


def test_weekend_maps_to_next_weekday_session() -> None:
    calendar = MarketCalendar(timezone="America/New_York", holidays=())

    decision = map_event_to_session(datetime(2026, 10, 3, 12, 0, tzinfo=NY), calendar, time(16, 0))

    assert decision.session_date == date(2026, 10, 5)
    assert decision.reason == "closed_day"


def test_holiday_maps_to_next_nonholiday_session() -> None:
    calendar = MarketCalendar(timezone="America/New_York", holidays=(date(2026, 10, 5),))

    decision = map_event_to_session(datetime(2026, 10, 2, 17, 0, tzinfo=NY), calendar, time(16, 0))

    assert decision.session_date == date(2026, 10, 6)
    assert decision.reason == "after_close"


def test_event_timestamp_is_converted_to_calendar_timezone() -> None:
    calendar = MarketCalendar(timezone="America/New_York", holidays=())

    decision = map_event_to_session(
        datetime(2026, 10, 5, 19, 30, tzinfo=timezone.utc), calendar, time(16, 0)
    )

    assert decision.session_date == date(2026, 10, 5)
    assert decision.event_local_time == datetime(2026, 10, 5, 15, 30, tzinfo=NY)
    assert decision.reason == "intraday"


def test_observation_cutoff_cannot_exceed_evaluation_window() -> None:
    calendar = MarketCalendar(timezone="America/New_York", holidays=())

    decision = map_event_to_session(
        datetime(2026, 10, 5, 14, 0, tzinfo=NY),
        calendar,
        time(16, 0),
        evaluation_end=datetime(2026, 10, 5, 15, 0, tzinfo=NY),
    )

    assert decision.observation_cutoff == datetime(2026, 10, 5, 15, 0, tzinfo=NY)


@pytest.mark.parametrize(
    "timestamp, close_time",
    [
        (datetime(2026, 10, 5, 10, 0), time(16, 0)),
        (datetime(2026, 10, 5, 10, 0, tzinfo=NY), time(16, 0, tzinfo=NY)),
    ],
)
def test_rejects_naive_or_timezone_aware_market_close(
    timestamp: datetime, close_time: time
) -> None:
    calendar = MarketCalendar(timezone="America/New_York", holidays=())

    with pytest.raises(ValueError):
        map_event_to_session(timestamp, calendar, close_time)
