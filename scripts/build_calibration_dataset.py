"""Build a deterministic, offline calibration evidence export from local CSV inputs.

Input event descriptions must be project-authored metadata or paraphrases. Raw
third-party text is never an output column. Factor movements must already be
derived locally; this builder does not infer movements from event references.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from risk_engine.calibration.event_study import MeasurementDimension
from risk_engine.domain import EventClass, ShockUnit

VERSION = "calibration-dataset-v1"
EVENT_INPUT_FIELDS = (
    "event_id", "event_date", "date_precision", "event_class", "region", "description",
    "source_reference", "source_terms", "permission", "evidence_kind", "source_version",
)
FACTOR_INPUT_FIELDS = (
    "event_id", "factor_id", "window_start", "window_end", "value", "unit",
    "measurement_dimension", "observed_at", "provider", "snapshot_id", "series_version",
    "source_terms", "permission", "evidence_kind",
)
EVENT_OUTPUT_FIELDS = (*EVENT_INPUT_FIELDS, "market_support", "missing_support_reason")
FACTOR_OUTPUT_FIELDS = FACTOR_INPUT_FIELDS
REQUIRED_DIMENSIONS = frozenset(MeasurementDimension)


class EventInput(BaseModel):
    """Locally recorded event evidence with explicit redistribution permission."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    event_date: str = Field(min_length=1)
    date_precision: Literal["day", "month"]
    event_class: EventClass
    region: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)
    source_terms: str = Field(min_length=1)
    permission: Literal["derived_metadata", "restricted"]
    evidence_kind: Literal["official_metadata", "synthetic"]
    source_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def date_matches_precision(self) -> EventInput:
        try:
            if self.date_precision == "day":
                if len(self.event_date) != 10:
                    raise ValueError("day precision requires YYYY-MM-DD")
                date.fromisoformat(self.event_date)
            else:
                if len(self.event_date) != 7:
                    raise ValueError("month precision requires YYYY-MM")
                date.fromisoformat(self.event_date + "-01")
        except ValueError as exc:
            raise ValueError("event_date disagrees with date_precision") from exc
        return self


class FactorInput(BaseModel):
    """One already-derived factor movement in native units and a fixed window."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    factor_id: str = Field(min_length=1)
    window_start: int
    window_end: int
    value: float
    unit: ShockUnit
    measurement_dimension: MeasurementDimension
    observed_at: datetime
    provider: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    series_version: str = Field(min_length=1)
    source_terms: str = Field(min_length=1)
    permission: Literal["derived_export", "restricted"]
    evidence_kind: Literal["observed", "synthetic"]

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("factor movement must be finite")
        return value

    @model_validator(mode="after")
    def valid_window_and_timestamp(self) -> FactorInput:
        if self.window_start > 0 or self.window_end < 0 or self.window_start > self.window_end:
            raise ValueError("factor window must include event session")
        if self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must include timezone")
        return self


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or tuple(reader.fieldnames) != fields:
            raise ValueError(f"{path} must have columns {fields}")
        return list(reader)


def _csv_bytes(rows: list[dict[str, str]], fields: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _empirical_support(shocks: list[FactorInput]) -> bool:
    """Require one complete contemporaneous observed vector for an event."""
    dimensions: dict[tuple[int, int], set[MeasurementDimension]] = defaultdict(set)
    for shock in shocks:
        if shock.evidence_kind == "observed":
            dimensions[(shock.window_start, shock.window_end)].add(shock.measurement_dimension)
    return any(found >= REQUIRED_DIMENSIONS for found in dimensions.values())


def build_dataset(
    events_path: Path,
    output_dir: Path,
    *,
    factor_shocks_path: Path | None = None,
) -> None:
    """Export permitted metadata and derived shocks, with coverage and byte hashes.

    The caller supplies all inputs locally. Existing exports are immutable; use
    a fresh directory for a new version. No network access occurs here.
    """
    event_bytes = events_path.read_bytes()
    event_inputs = [
        EventInput.model_validate(row) for row in _read_csv(events_path, EVENT_INPUT_FIELDS)
    ]
    event_ids = [event.event_id for event in event_inputs]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("duplicate event_id in event metadata")

    factor_bytes: bytes | None = None
    factor_inputs: list[FactorInput] = []
    if factor_shocks_path is not None:
        factor_bytes = factor_shocks_path.read_bytes()
        factor_inputs = [
            FactorInput.model_validate(row)
            for row in _read_csv(factor_shocks_path, FACTOR_INPUT_FIELDS)
        ]
    if any(factor.event_id not in event_ids for factor in factor_inputs):
        raise ValueError("factor shock references an unknown event_id")
    factor_keys = [
        (factor.event_id, factor.factor_id, factor.window_start, factor.window_end)
        for factor in factor_inputs
    ]
    if len(factor_keys) != len(set(factor_keys)):
        raise ValueError("duplicate factor shock for event, factor and window")

    permitted_events = sorted(
        (event for event in event_inputs if event.permission == "derived_metadata"),
        key=lambda event: event.event_id,
    )
    permitted_ids = {event.event_id for event in permitted_events}
    permitted_factors = sorted(
        (
            factor for factor in factor_inputs
            if factor.permission == "derived_export" and factor.event_id in permitted_ids
        ),
        key=lambda factor: (
            factor.event_id, factor.window_start, factor.window_end, factor.factor_id
        ),
    )
    by_event: dict[str, list[FactorInput]] = defaultdict(list)
    for factor in permitted_factors:
        by_event[factor.event_id].append(factor)

    event_rows: list[dict[str, str]] = []
    strata: Counter[tuple[str, str]] = Counter()
    supported_strata: Counter[tuple[str, str]] = Counter()
    missing_reasons: Counter[str] = Counter()
    for event in permitted_events:
        shocks = by_event[event.event_id]
        supported = event.evidence_kind != "synthetic" and _empirical_support(shocks)
        if supported:
            reason = ""
        elif not shocks:
            reason = "no_local_factor_shocks"
        elif all(shock.evidence_kind == "synthetic" for shock in shocks):
            reason = "synthetic_only"
        else:
            reason = "incomplete_joint_vector"
        if reason:
            missing_reasons[reason] += 1
        stratum = (event.event_class.value, event.region)
        strata[stratum] += 1
        if supported:
            supported_strata[stratum] += 1
        event_rows.append({
            **event.model_dump(mode="json"),
            "market_support": "empirical" if supported else "missing",
            "missing_support_reason": reason,
        })

    factor_rows = [
        {
            **factor.model_dump(mode="json"),
            "value": str(factor.value),
            "observed_at": factor.observed_at.isoformat(),
        }
        for factor in permitted_factors
    ]
    coverage = {
        "schema_version": VERSION,
        "included_events": len(permitted_events),
        "excluded_by_permission": len(event_inputs) - len(permitted_events),
        "included_factor_shocks": len(permitted_factors),
        "excluded_factor_shocks_by_permission": len(factor_inputs) - len(permitted_factors),
        "empirical_supported_events": sum(supported_strata.values()),
        "missing_market_support_events": sum(missing_reasons.values()),
        "synthetic_only_events": missing_reasons["synthetic_only"],
        "missing_support_reasons": dict(sorted(missing_reasons.items())),
        "strata": [
            {
                "event_class": event_class,
                "region": region,
                "events": count,
                "empirical_supported": supported_strata[(event_class, region)],
            }
            for (event_class, region), count in sorted(strata.items())
        ],
    }
    outputs = {
        "events.csv": _csv_bytes(event_rows, EVENT_OUTPUT_FIELDS),
        "factor_shocks.csv": _csv_bytes(factor_rows, FACTOR_OUTPUT_FIELDS),
        "coverage.json": _json_bytes(coverage),
    }
    sources = [{"role": "event_metadata", "sha256": _sha256(event_bytes)}]
    if factor_bytes is not None:
        sources.append({"role": "derived_factor_shocks", "sha256": _sha256(factor_bytes)})
    outputs["manifest.json"] = _json_bytes({
        "schema_version": VERSION,
        "sources": sources,
        "outputs": {name: _sha256(payload) for name, payload in outputs.items()},
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in outputs):
        raise FileExistsError("calibration export already exists; use a new output directory")
    for name, payload in outputs.items():
        (output_dir / name).write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factor-shocks", type=Path)
    args = parser.parse_args()
    build_dataset(args.events, args.output_dir, factor_shocks_path=args.factor_shocks)


if __name__ == "__main__":
    main()
