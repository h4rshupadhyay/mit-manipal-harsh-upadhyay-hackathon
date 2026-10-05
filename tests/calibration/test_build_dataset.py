"""Contract tests for reproducible, licensed calibration evidence exports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_calibration_dataset import build_dataset

FIXTURE = Path(__file__).parents[1] / "fixtures" / "historical-events.csv"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_fixture_exports_official_metadata_without_inventing_market_shocks(tmp_path: Path) -> None:
    """Catches a fabricated price reaction or an invented exact IL&FS default day."""
    build_dataset(FIXTURE, tmp_path)

    events = _rows(tmp_path / "events.csv")
    assert [(row["event_id"], row["event_date"], row["date_precision"]) for row in events] == [
        ("ilfs-takeover-2018-10", "2018-10", "month"),
        ("rbi-repo-2022-09-30", "2022-09-30", "day"),
    ]
    assert {row["event_class"] for row in events} == {
        "Credit/Default", "Macroeconomic/Monetary"
    }
    assert all(row["region"] == "India" and row["market_support"] == "missing" for row in events)
    assert all(row["missing_support_reason"] == "no_local_factor_shocks" for row in events)
    assert all(row["evidence_kind"] == "official_metadata" for row in events)
    assert _rows(tmp_path / "factor_shocks.csv") == []

    coverage = json.loads((tmp_path / "coverage.json").read_text())
    assert coverage["empirical_supported_events"] == 0
    assert coverage["missing_market_support_events"] == 2
    assert coverage["strata"] == [
        {"event_class": "Credit/Default", "region": "India", "events": 1, "empirical_supported": 0},
        {
            "event_class": "Macroeconomic/Monetary", "region": "India",
            "events": 1, "empirical_supported": 0,
        },
    ]


def test_identical_inputs_produce_identical_bytes_and_hash_actual_local_input(
    tmp_path: Path,
) -> None:
    """Catches unstable output or a manifest that hashes a remote URL instead of local bytes."""
    first, second = tmp_path / "first", tmp_path / "second"
    build_dataset(FIXTURE, first)
    build_dataset(FIXTURE, second)

    names = ("events.csv", "factor_shocks.csv", "coverage.json", "manifest.json")
    assert {name: (first / name).read_bytes() for name in names} == {
        name: (second / name).read_bytes() for name in names
    }
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["schema_version"] == "calibration-dataset-v2"
    assert manifest["sources"] == [
        {"role": "event_metadata", "sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest()}
    ]
    assert manifest["outputs"] == {
        name: hashlib.sha256((first / name).read_bytes()).hexdigest() for name in names[:-1]
    }


def test_restricted_event_is_excluded_and_counted(tmp_path: Path) -> None:
    """Catches accidental export of a source whose derived metadata permission is absent."""
    rows = _rows(FIXTURE)
    restricted = dict(rows[0], event_id="restricted-event", permission="restricted")
    source = tmp_path / "input.csv"
    _write_rows(source, [*rows, restricted])

    build_dataset(source, tmp_path / "out")

    assert {row["event_id"] for row in _rows(tmp_path / "out" / "events.csv")} == {
        "rbi-repo-2022-09-30", "ilfs-takeover-2018-10"
    }
    coverage = json.loads((tmp_path / "out" / "coverage.json").read_text())
    assert coverage["excluded_by_permission"] == 1


def test_synthetic_factor_shocks_do_not_count_as_empirical_support(tmp_path: Path) -> None:
    """Catches treating a synthetic test movement as an observed Historical Analogue."""
    factor = tmp_path / "local-factors.csv"
    _write_rows(factor, [{
        "event_id": "rbi-repo-2022-09-30", "factor_id": "INR_10Y",
        "window_start": "0", "window_end": "1", "value": "12.5", "unit": "basis_point",
        "measurement_dimension": "yield", "observed_at": "2022-10-03T16:00:00+05:30",
        "provider": "synthetic-test", "snapshot_id": "synthetic-snapshot",
        "series_version": "test-v1", "source_terms": "synthetic test fixture",
        "permission": "derived_export", "evidence_kind": "synthetic",
        "asset_role": "rate", "event_clock_id": "clock-rbi-v1",
        "event_session_date": "2022-09-30", "event_study_spec_version": "study-v1",
        "window_start_session": "2022-09-30", "window_end_session": "2022-10-03",
    }])

    build_dataset(FIXTURE, tmp_path / "out", factor_shocks_path=factor)

    shocks = _rows(tmp_path / "out" / "factor_shocks.csv")
    assert len(shocks) == 1
    assert shocks[0]["value"] == "12.5" and shocks[0]["evidence_kind"] == "synthetic"
    events = {row["event_id"]: row for row in _rows(tmp_path / "out" / "events.csv")}
    assert events["rbi-repo-2022-09-30"]["market_support"] == "missing"
    assert events["rbi-repo-2022-09-30"]["missing_support_reason"] == "synthetic_only"
    coverage = json.loads((tmp_path / "out" / "coverage.json").read_text())
    assert coverage["empirical_supported_events"] == 0
    assert coverage["synthetic_only_events"] == 1


def test_restricted_factor_shock_is_not_exported(tmp_path: Path) -> None:
    """Catches an unlicensed local factor movement passing through to a derived file."""
    factor = tmp_path / "local-factors.csv"
    _write_rows(factor, [{
        "event_id": "rbi-repo-2022-09-30", "factor_id": "INR_10Y",
        "window_start": "0", "window_end": "1", "value": "12.5", "unit": "basis_point",
        "measurement_dimension": "yield", "observed_at": "2022-10-03T16:00:00+05:30",
        "provider": "local", "snapshot_id": "local-1", "series_version": "v1",
        "source_terms": "local source; derived export restricted", "permission": "restricted",
        "evidence_kind": "observed",
        "asset_role": "rate", "event_clock_id": "clock-rbi-v1",
        "event_session_date": "2022-09-30", "event_study_spec_version": "study-v1",
        "window_start_session": "2022-09-30", "window_end_session": "2022-10-03",
    }])

    build_dataset(FIXTURE, tmp_path / "out", factor_shocks_path=factor)

    assert _rows(tmp_path / "out" / "factor_shocks.csv") == []
    coverage = json.loads((tmp_path / "out" / "coverage.json").read_text())
    assert coverage["excluded_factor_shocks_by_permission"] == 1


def test_incomplete_source_metadata_is_rejected(tmp_path: Path) -> None:
    """Catches silent defaults for required source terms and date precision."""
    rows = _rows(FIXTURE)
    rows[0]["source_terms"] = ""
    source = tmp_path / "incomplete.csv"
    _write_rows(source, rows)

    with pytest.raises(ValueError, match="source_terms"):
        build_dataset(source, tmp_path / "out")


def _complete_factor_rows() -> list[dict[str, str]]:
    """A six-role, one-window test vector, with distinct factor source identities."""
    cases = (
        ("equity", "NIFTY", "return", "decimal", "-0.03"),
        ("rate", "INR_10Y", "yield", "basis_point", "12.5"),
        ("spread", "INDIA_IG", "spread", "basis_point", "35"),
        ("fx", "USD_INR", "return", "decimal", "0.02"),
        ("commodity", "BRENT", "return", "percent", "-4"),
        ("volatility", "INDIA_VIX", "volatility", "volatility_point", "2"),
    )
    return [
        {
            "event_id": "rbi-repo-2022-09-30", "factor_id": factor_id,
            "window_start": "0", "window_end": "1", "value": value, "unit": unit,
            "measurement_dimension": dimension, "observed_at": "2022-10-03T16:00:00+05:30",
            "provider": f"provider-{role}", "snapshot_id": f"snapshot-{role}",
            "series_version": f"version-{role}", "source_terms": "derived export allowed",
            "permission": "derived_export", "evidence_kind": "observed", "asset_role": role,
            "event_clock_id": "clock-rbi-v1", "event_session_date": "2022-09-30",
            "event_study_spec_version": "study-v1", "window_start_session": "2022-09-30",
            "window_end_session": "2022-10-03",
        }
        for role, factor_id, dimension, unit, value in cases
    ]


def test_complete_six_role_vector_with_common_clock_and_spec_is_empirical(tmp_path: Path) -> None:
    """Catches rejecting a genuine joint window because providers differ by factor."""
    factor = tmp_path / "factors.csv"
    _write_rows(factor, _complete_factor_rows())

    build_dataset(FIXTURE, tmp_path / "out", factor_shocks_path=factor)

    events = {row["event_id"]: row for row in _rows(tmp_path / "out" / "events.csv")}
    assert events["rbi-repo-2022-09-30"]["market_support"] == "empirical"
    assert events["rbi-repo-2022-09-30"]["missing_support_reason"] == ""
    assert len(_rows(tmp_path / "out" / "factor_shocks.csv")) == 6
    coverage = json.loads((tmp_path / "out" / "coverage.json").read_text())
    assert coverage["empirical_supported_events"] == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("asset_role", "fx"),
        ("event_clock_id", "different-clock"),
        ("event_study_spec_version", "different-spec"),
        ("event_session_date", "2022-10-03"),
        ("window_end_session", "2022-10-04"),
    ],
)
def test_mixed_role_or_event_study_identity_is_not_empirical(
    tmp_path: Path, field: str, replacement: str,
) -> None:
    """Catches splicing six unrelated rows into one historical joint vector."""
    rows = _complete_factor_rows()
    rows[4 if field == "asset_role" else -1][field] = replacement
    if field == "event_session_date":
        rows[-1]["window_start_session"] = "2022-10-03"
        rows[-1]["window_end_session"] = "2022-10-04"
        rows[-1]["observed_at"] = "2022-10-04T16:00:00+05:30"
    factor = tmp_path / "factors.csv"
    _write_rows(factor, rows)

    build_dataset(FIXTURE, tmp_path / "out", factor_shocks_path=factor)

    events = {row["event_id"]: row for row in _rows(tmp_path / "out" / "events.csv")}
    assert events["rbi-repo-2022-09-30"]["market_support"] == "missing"
    assert events["rbi-repo-2022-09-30"]["missing_support_reason"] == (
        "incomplete_joint_vector"
    )


def test_incompatible_dimension_unit_is_rejected(tmp_path: Path) -> None:
    """Catches exporting a spread movement measured in volatility points."""
    rows = _complete_factor_rows()
    rows[2]["unit"] = "volatility_point"
    factor = tmp_path / "factors.csv"
    _write_rows(factor, rows)

    with pytest.raises(ValueError, match="unsupported unit for measurement dimension"):
        build_dataset(FIXTURE, tmp_path / "out", factor_shocks_path=factor)


def test_month_precision_event_cannot_claim_empirical_reaction(tmp_path: Path) -> None:
    """Catches treating IL&FS month-only chronology as an exact event-study clock."""
    rows = _complete_factor_rows()
    for row in rows:
        row["event_id"] = "ilfs-takeover-2018-10"
        row["event_session_date"] = "2018-10-01"
        row["window_start_session"] = "2018-10-01"
        row["window_end_session"] = "2018-10-02"
        row["observed_at"] = "2018-10-02T16:00:00+05:30"
    factor = tmp_path / "factors.csv"
    _write_rows(factor, rows)

    build_dataset(FIXTURE, tmp_path / "out", factor_shocks_path=factor)

    events = {row["event_id"]: row for row in _rows(tmp_path / "out" / "events.csv")}
    assert events["ilfs-takeover-2018-10"]["market_support"] == "missing"
    assert events["ilfs-takeover-2018-10"]["missing_support_reason"] == "imprecise_event_date"


def test_zero_window_with_broader_sessions_cannot_claim_empirical_support(tmp_path: Path) -> None:
    """Catches six same-window rows claiming support with dishonest session dates."""
    rows = _complete_factor_rows()
    for row in rows:
        row["window_end"] = "0"
        row["window_start_session"] = "2022-09-29"
        row["window_end_session"] = "2022-10-03"
    factor = tmp_path / "factors.csv"
    _write_rows(factor, rows)

    with pytest.raises(ValueError, match="window session dates disagree with offsets"):
        build_dataset(FIXTURE, tmp_path / "out", factor_shocks_path=factor)


@pytest.mark.parametrize(
    ("start", "end", "start_session", "end_session"),
    [
        ("0", "1", "2022-09-29", "2022-10-03"),
        ("-1", "0", "2022-09-29", "2022-10-03"),
        ("-1", "1", "2022-09-30", "2022-10-03"),
        ("0", "1", "2022-09-30", "2022-09-30"),
    ],
)
def test_window_session_dates_must_follow_offset_direction(
    tmp_path: Path, start: str, end: str, start_session: str, end_session: str,
) -> None:
    """Catches zero offsets that drift and nonzero offsets with no session separation."""
    rows = _complete_factor_rows()[:1]
    rows[0].update({
        "window_start": start,
        "window_end": end,
        "window_start_session": start_session,
        "window_end_session": end_session,
    })
    factor = tmp_path / "factors.csv"
    _write_rows(factor, rows)

    with pytest.raises(ValueError, match="window session dates disagree with offsets"):
        build_dataset(FIXTURE, tmp_path / "out", factor_shocks_path=factor)
