"""Preserve complete, contemporaneous historical factor movements."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from risk_engine.calibration.event_study import (
    EventReaction,
    EventWindow,
    MeasurementDimension,
    ReturnObservation,
    WindowReaction,
)
from risk_engine.calibration.market_calendar import EventClockDecision
from risk_engine.domain import DomainModel, NonEmptyString, ShockUnit


class HistoricalFactorShock(DomainModel):
    """A factor's observed abnormal movement in its native measurement unit."""

    factor_id: NonEmptyString
    value: float
    unit: ShockUnit
    measurement_dimension: MeasurementDimension
    source_observations: tuple[ReturnObservation, ...] = Field(min_length=1)


class JointWindowScenario(DomainModel):
    """All factor movements observed for one registered event window."""

    window: EventWindow
    shocks: tuple[HistoricalFactorShock, ...] = Field(min_length=1)


class JointScenario(DomainModel):
    """One event's complete joint vectors over every registered window."""

    event_id: NonEmptyString
    clock_decision: EventClockDecision
    spec_version: NonEmptyString
    windows: tuple[JointWindowScenario, ...] = Field(min_length=1)


_SUPPORTED_UNITS = {
    MeasurementDimension.RETURN: frozenset({ShockUnit.DECIMAL, ShockUnit.PERCENT}),
    MeasurementDimension.YIELD: frozenset(
        {ShockUnit.DECIMAL, ShockUnit.PERCENT, ShockUnit.BASIS_POINT}
    ),
    MeasurementDimension.SPREAD: frozenset(
        {ShockUnit.DECIMAL, ShockUnit.PERCENT, ShockUnit.BASIS_POINT}
    ),
    MeasurementDimension.VOLATILITY: frozenset({ShockUnit.VOLATILITY_POINT}),
    MeasurementDimension.PRICE: frozenset(
        {ShockUnit.DECIMAL, ShockUnit.PERCENT, ShockUnit.ABSOLUTE, ShockUnit.INDEX_POINT}
    ),
    MeasurementDimension.CURRENCY: frozenset(
        {ShockUnit.DECIMAL, ShockUnit.PERCENT, ShockUnit.ABSOLUTE, ShockUnit.CURRENCY}
    ),
}


def _factor_id(reaction: EventReaction) -> str:
    observations = reaction.event_factor_observations
    if not observations or len({row.series_id for row in observations}) != 1:
        raise ValueError("factor series must have one explicit identity")
    factor_id = observations[0].series_id
    if any(
        row.unit != reaction.factor_unit
        or row.measurement_dimension != reaction.factor_dimension
        for row in observations
    ):
        raise ValueError("factor series unit or measurement dimension disagrees with reaction")
    return factor_id


def _windows(reaction: EventReaction) -> dict[EventWindow, WindowReaction]:
    if not reaction.windows:
        raise ValueError("reaction must contain at least one registered window")
    by_window = {item.window: item for item in reaction.windows}
    if len(by_window) != len(reaction.windows):
        raise ValueError("reaction has duplicate window definitions")
    for item in reaction.windows:
        if (
            item.unit != reaction.factor_unit
            or item.measurement_dimension != reaction.factor_dimension
        ):
            raise ValueError("window unit or measurement dimension disagrees with factor series")
        if item.unit not in _SUPPORTED_UNITS[item.measurement_dimension]:
            raise ValueError("unsupported unit for measurement dimension")
    return by_window


def build_joint_scenario(reactions: Sequence[EventReaction]) -> JointScenario:
    """Group matching Event Reactions without selecting factors' tails independently."""
    if not reactions:
        raise ValueError("at least one Event Reaction is required")
    first = reactions[0]
    registered = _windows(first)
    factor_ids: set[str] = set()
    per_factor: list[tuple[str, EventReaction, dict[EventWindow, WindowReaction]]] = []
    for reaction in reactions:
        if reaction.event_id != first.event_id:
            raise ValueError("event ID differs between reactions")
        if reaction.clock_decision != first.clock_decision:
            raise ValueError("event clock decision differs between reactions")
        if reaction.spec_version != first.spec_version:
            raise ValueError("event-study spec version differs between reactions")
        windows = _windows(reaction)
        if windows.keys() != registered.keys():
            raise ValueError("registered window set differs between reactions")
        factor_id = _factor_id(reaction)
        if factor_id in factor_ids:
            raise ValueError(f"duplicate factor series: {factor_id}")
        factor_ids.add(factor_id)
        per_factor.append((factor_id, reaction, windows))

    return JointScenario(
        event_id=first.event_id,
        clock_decision=first.clock_decision,
        spec_version=first.spec_version,
        windows=tuple(
            JointWindowScenario(
                window=window,
                shocks=tuple(
                    HistoricalFactorShock(
                        factor_id=factor_id,
                        value=by_window[window].car,
                        unit=by_window[window].unit,
                        measurement_dimension=by_window[window].measurement_dimension,
                        source_observations=reaction.event_factor_observations,
                    )
                    for factor_id, reaction, by_window in per_factor
                ),
            )
            for window in registered
        ),
    )
