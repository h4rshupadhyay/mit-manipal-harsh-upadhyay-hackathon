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
