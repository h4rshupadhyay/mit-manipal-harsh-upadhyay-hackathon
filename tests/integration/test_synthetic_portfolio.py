"""Offline Synthetic Portfolio replay, audit metadata, and production valuation."""

import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from risk_engine.domain import (
    AssetType,
    AttributionDimension,
    FactorShock,
    MarketObservation,
    MarketSnapshot,
    Portfolio,
    ProvenanceMethod,
    ShockType,
    ShockUnit,
    StressScenario,
)
from risk_engine.stress.interfaces import FACTOR_REGISTRY
from risk_engine.stress.module import StressEngine
from scripts.generate_synthetic_portfolio import (
    generate_synthetic_portfolio,
    load_artifact,
    load_portfolio,
    write_portfolio,
)

ARTIFACT = Path(__file__).resolve().parents[2] / "data/portfolio/synthetic_portfolio.csv"


def test_generation_replays_exact_bytes_and_preserves_metadata(tmp_path: Path) -> None:
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    write_portfolio(first, generate_synthetic_portfolio())
    write_portfolio(second, generate_synthetic_portfolio())
    assert first.read_bytes() == second.read_bytes()
    loaded = load_artifact(first)
    assert loaded == generate_synthetic_portfolio()
    assert loaded.content_hash == hashlib.sha256(first.read_bytes()).hexdigest()
    portfolio = load_portfolio(first)
    assert isinstance(portfolio, Portfolio)
    assert len(portfolio.positions) == 60
    assert len({p.position_id for p in portfolio.positions}) == 60
    assert Counter(p.asset_type for p in portfolio.positions) == {
        AssetType.LOAN: 20,
        AssetType.BOND: 20,
        AssetType.DERIVATIVE: 20,
    }
    assert Counter(p.region for p in portfolio.positions) == {
        "United States": 15,
        "Europe": 15,
        "India": 15,
        "Asia-Pacific": 15,
    }
    assert len({r.sector for r in loaded.rows}) >= 5
    assert len({r.rating for r in loaded.rows}) >= 5
    assert len({r.maturity_date for r in loaded.rows}) >= 5
    assert all(r.maturity_date > r.as_of.date() for r in loaded.rows)
    assert all(r.evidence_kind == "synthetic" for r in loaded.rows)
    assert all(r.source_terms == "MIT; project-authored synthetic data" for r in loaded.rows)
    assert all(
        r.as_of == portfolio.as_of and r.portfolio_version == portfolio.version for r in loaded.rows
    )
    assert all(r.seed == 20261004 for r in loaded.rows)
    assert all(r.notional_unit == "USD base market value" for r in loaded.rows)


def test_all_six_factor_roles_value_with_production_stress_engine(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.csv"
    write_portfolio(path, generate_synthetic_portfolio())
    artifact = load_artifact(path)
    portfolio = load_portfolio(path)
    for row in artifact.rows:
        assert set(row.factor_exposures) == set(row.sensitivity_units)
        assert all(FACTOR_REGISTRY[f].native_unit == unit for f, unit in row.factor_units.items())
    factors = {f for row in artifact.rows for f in row.factor_units}
    assert {
        "EQUITY-INDIA",
        "RATE-INR",
        "CREDIT-SPREAD",
        "USD-INR",
        "COMMODITY",
        "VOLATILITY",
    } <= factors
    native_levels = {
        "EQUITY-US": 100,
        "EQUITY-EUROPE": 100,
        "EQUITY-INDIA": 100,
        "EQUITY-ASIA": 100,
        "USD-INR": 80,
        "EUR-USD": 1.1,
        "COMMODITY": 100,
        "RATE-USD": 400,
        "RATE-EUR": 300,
        "RATE-INR": 600,
        "CREDIT-SPREAD": 150,
        "VOLATILITY": 20,
        "DEFAULT": 0,
    }
    market = MarketSnapshot(
        snapshot_id="synthetic-test-market-v1",
        as_of=portfolio.as_of,
        observations=tuple(
            MarketObservation(
                factor_id=f,
                observed_at=portfolio.as_of,
                value=native_levels[f],
                unit=FACTOR_REGISTRY[f].native_unit,
                provider="project-authored synthetic test",
                vintage="synthetic-v1",
            )
            for f in sorted(factors)
        ),
        provider="project-authored synthetic test",
        schema_version="market-v1",
        normalization_version="native-units-v1",
    )
    scenario = StressScenario(
        scenario_id="synthetic-six-role-scenario-v1",
        risk_signal_id="synthetic-test-signal",
        shocks=tuple(
            FactorShock(
                factor_id=f,
                shock_type=kind,
                value=value,
                unit=unit,
                horizon_days=5,
            )
            for f, kind, value, unit in (
                ("EQUITY-INDIA", ShockType.RELATIVE, -10, ShockUnit.PERCENT),
                ("RATE-INR", ShockType.BASIS_POINT, 100, ShockUnit.BASIS_POINT),
                ("CREDIT-SPREAD", ShockType.BASIS_POINT, 50, ShockUnit.BASIS_POINT),
                ("USD-INR", ShockType.RELATIVE, 5, ShockUnit.PERCENT),
                ("COMMODITY", ShockType.RELATIVE, -15, ShockUnit.PERCENT),
                ("VOLATILITY", ShockType.VOLATILITY, 5, ShockUnit.VOLATILITY_POINT),
            )
        ),
        method=ProvenanceMethod.HYPOTHETICAL,
        calibration_version="synthetic-test-v1",
    )
    result = StressEngine().run(portfolio, market, scenario)
    assert result == StressEngine().run(portfolio, market, scenario)
    assert result.valuation_coverage == 1
    assert result.unsupported_position_ids == ()
    assert result.base_value == sum(p.notional for p in portfolio.positions)
    assert result.base_value - result.stressed_value == result.absolute_loss
    assert all(
        v.is_finite() for v in (result.base_value, result.stressed_value, result.absolute_loss)
    )
    assert math.isfinite(result.percentage_loss)
    factor_losses = {
        a.label: a.loss for a in result.attribution if a.dimension is AttributionDimension.FACTOR
    }
    assert set(factor_losses) == {s.factor_id for s in scenario.shocks}
    assert all(loss != 0 for loss in factor_losses.values())
    for dimension in (
        AttributionDimension.ASSET,
        AttributionDimension.REGION,
        AttributionDimension.SECTOR,
        AttributionDimension.OBLIGOR,
        AttributionDimension.FACTOR,
    ):
        assert sum(a.loss for a in result.attribution if a.dimension is dimension) == (
            result.absolute_loss
        )


def test_existing_artifact_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.csv"
    write_portfolio(path, generate_synthetic_portfolio())
    original = path.read_bytes()
    write_portfolio(path, generate_synthetic_portfolio())
    assert path.read_bytes() == original
    path.write_bytes(b"immutable different version\n")
    with pytest.raises(FileExistsError):
        write_portfolio(path, generate_synthetic_portfolio())
    assert path.read_bytes() == b"immutable different version\n"


def corrupt_csv(path: Path, field: str, value: str) -> None:
    reader = csv.DictReader(io.StringIO(path.read_text()))
    rows = list(reader)
    assert reader.fieldnames is not None
    rows[0][field] = value
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_terms", ""),
        ("as_of", "2026-10-04T00:00:00"),
        ("portfolio_version", ""),
        ("snapshot_id", ""),
        ("seed", ""),
        ("factor_units", "{}"),
        ("sensitivity_units", "{}"),
        ("notional_unit", "EUR"),
        ("rating", ""),
        ("maturity_date", "2020-01-01"),
        ("factor_exposures", '{"duration:RATE-USD":"NaN"}'),
        ("factor_exposures", '{"duration:RATE-USD":"1","duration:RATE-USD":"2"}'),
        ("content_hash", "0" * 64),
        ("notional", "999.01"),
    ],
)
def test_loader_rejects_missing_inconsistent_or_corrupt_audit_data(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    path = tmp_path / "bad.csv"
    write_portfolio(path, generate_synthetic_portfolio())
    corrupt_csv(path, field, value)
    with pytest.raises(ValueError):
        load_artifact(path)
    with pytest.raises(ValueError):
        load_portfolio(path)


def test_loader_rejects_extra_columns_duplicate_positions_and_truncated_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.csv"
    write_portfolio(path, generate_synthetic_portfolio())
    original = path.read_text()
    lines = original.splitlines(keepends=True)
    for malformed in (
        original.replace("position_id,", "extra,position_id,", 1),
        original + lines[1],
        lines[0] + "truncated\n",
        lines[0],
    ):
        path.write_text(malformed)
        with pytest.raises(ValueError):
            load_artifact(path)


def write_rehashed_default_exposures(
    path: Path,
    exposures: dict[str, str],
    units: dict[str, str],
) -> None:
    """Make canonical hash-valid input so rejection must be semantic, not checksum."""
    reader = csv.DictReader(io.StringIO(ARTIFACT.read_text()))
    rows = list(reader)
    assert reader.fieldnames is not None
    row = rows[0]
    for field, value in (
        ("factor_exposures", exposures),
        ("factor_units", {"DEFAULT": "decimal"}),
        ("sensitivity_units", units),
    ):
        row[field] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    payload: dict[str, object] = {key: value for key, value in row.items() if key != "content_hash"}
    for field in ("factor_exposures", "factor_units", "sensitivity_units"):
        payload[field] = json.loads(row[field])
    payload["seed"] = int(row["seed"])
    row["content_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue())


@pytest.mark.parametrize(
    ("exposures", "units"),
    [
        (
            {"ead:DEFAULT": "100", "lgd:DEFAULT": "0.4"},
            {"ead:DEFAULT": "USD", "lgd:DEFAULT": "decimal"},
        ),
        (
            {"ead:DEFAULT": "-1", "lgd:DEFAULT": "2"},
            {"ead:DEFAULT": "USD", "lgd:DEFAULT": "decimal"},
        ),
        ({"ead:DEFAULT": "100", "lgd": "0.4"}, {"ead:DEFAULT": "USD", "lgd": "decimal"}),
        ({"ead": "100", "lgd:DEFAULT": "0.4"}, {"ead": "USD", "lgd:DEFAULT": "decimal"}),
    ],
)
def test_hash_valid_prefixed_default_keys_are_rejected(
    tmp_path: Path,
    exposures: dict[str, str],
    units: dict[str, str],
) -> None:
    path = tmp_path / "prefixed.csv"
    write_rehashed_default_exposures(path, exposures, units)
    with pytest.raises(ValueError):
        load_artifact(path)
    with pytest.raises(ValueError):
        load_portfolio(path)


@pytest.mark.parametrize(
    ("exposures", "units"),
    [
        ({"ead": "-1", "lgd": "0.4"}, {"ead": "USD", "lgd": "decimal"}),
        ({"ead": "100", "lgd": "-0.1"}, {"ead": "USD", "lgd": "decimal"}),
        ({"ead": "100", "lgd": "1.1"}, {"ead": "USD", "lgd": "decimal"}),
        ({"ead": "100"}, {"ead": "USD"}),
        ({"lgd": "0.4"}, {"lgd": "decimal"}),
    ],
)
def test_hash_valid_bare_default_keys_require_a_paired_valid_range(
    tmp_path: Path,
    exposures: dict[str, str],
    units: dict[str, str],
) -> None:
    path = tmp_path / "invalid-default.csv"
    write_rehashed_default_exposures(path, exposures, units)
    with pytest.raises(ValueError, match="EAD/LGD"):
        load_artifact(path)


@pytest.mark.parametrize(("ead", "lgd"), [("100", "0.4"), ("0", "0"), ("100", "1")])
def test_hash_valid_bare_default_keys_preserve_valid_values(
    tmp_path: Path,
    ead: str,
    lgd: str,
) -> None:
    path = tmp_path / "valid-default.csv"
    write_rehashed_default_exposures(
        path,
        {"ead": ead, "lgd": lgd},
        {"ead": "USD", "lgd": "decimal"},
    )
    row = load_artifact(path).rows[0]
    assert str(row.factor_exposures["ead"]) == ead
    assert str(row.factor_exposures["lgd"]) == lgd
    assert load_portfolio(path).positions[0].factor_exposures == {
        "ead": float(ead),
        "lgd": float(lgd),
    }


def test_committed_csv_is_the_canonical_reproducible_artifact(tmp_path: Path) -> None:
    path = tmp_path / "generated.csv"
    write_portfolio(path, generate_synthetic_portfolio())
    assert ARTIFACT.read_bytes() == path.read_bytes()
    assert load_portfolio(ARTIFACT) == load_portfolio(path)
    rows = list(csv.DictReader(io.StringIO(ARTIFACT.read_text())))
    assert all(isinstance(json.loads(row["factor_exposures"]), dict) for row in rows)
