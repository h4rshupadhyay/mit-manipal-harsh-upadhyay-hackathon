"""Create an immutable, project-authored Synthetic Portfolio for offline replay.

All economics, obligors, ratings, maturities, and sensitivities are fictional.
USD values are supplied base market values, not face principal or trade notional.
Regional FX/rate exposures are explicit synthetic sensitivities; no conversion,
market estimation, rating inference, or empirical calibration is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from risk_engine.domain import (
    AssetType,
    CurrencyCode,
    DomainModel,
    NonEmptyString,
    Portfolio,
    Position,
    Sha256Hex,
    ShockType,
    ShockUnit,
)
from risk_engine.stress.interfaces import FACTOR_REGISTRY

SEED = 20261004
AS_OF = datetime(2026, 10, 4, tzinfo=timezone.utc)  # noqa: UP017 (offline Python 3.10 runner)
REGIONS = ("United States", "Europe", "India", "Asia-Pacific")
SECTORS = ("Financials", "Technology", "Energy", "Industrials", "Infrastructure", "Healthcare")
RATINGS = ("AAA", "AA", "A", "BBB", "BB", "B")
PORTFOLIO_FIELDS = (
    "portfolio_id",
    "portfolio_version",
    "valuation_currency",
    "as_of",
    "snapshot_id",
    "schema_version",
    "generator_version",
    "seed",
    "source_reference",
    "source_terms",
    "evidence_kind",
)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _exposure_metadata(
    exposures: dict[str, Decimal],
) -> tuple[dict[str, ShockUnit], dict[str, str]]:
    """Declare adapter-native conventions; this never supplies missing CSV metadata."""
    factors: dict[str, ShockUnit] = {}
    sensitivities: dict[str, str] = {}
    for key in exposures:
        kind, _, factor = key.partition(":")
        if kind in {"ead", "lgd"} and key not in {"ead", "lgd"}:
            raise ValueError("EAD/LGD keys must be exactly ead or lgd")
        if key in {"ead", "lgd"}:
            factor = "DEFAULT"
        definition = FACTOR_REGISTRY.get(factor)
        if definition is None:
            raise ValueError(f"unregistered exposure factor: {key}")
        native = definition.native_unit.value
        units = {
            "duration": "years",
            "convexity": "years_squared",
            "cs01": "USD/basis_point",
            "dv01": "USD/basis_point",
            "delta": f"USD/{native}",
            "gamma": f"USD/{native}^2",
            "vega": "USD/volatility_point",
            "ead": "USD",
            "lgd": "decimal",
        }
        if kind not in units:
            raise ValueError(f"unregistered sensitivity convention: {key}")
        if kind in {"duration", "convexity", "dv01"} and not factor.startswith("RATE-"):
            raise ValueError("rate sensitivity requires a registered rate factor")
        if kind == "cs01" and factor != "CREDIT-SPREAD":
            raise ValueError("CS01 requires CREDIT-SPREAD")
        if kind == "vega" and factor != "VOLATILITY":
            raise ValueError("vega requires VOLATILITY")
        if kind in {"delta", "gamma"} and ShockType.RELATIVE not in definition.shock_types:
            raise ValueError("delta/gamma requires a registered spot factor")
        factors[factor] = definition.native_unit
        sensitivities[key] = units[kind]
    return factors, sensitivities


class SyntheticPosition(DomainModel):
    """Required artifact data retained beyond the canonical Position contract.

    Rating denotes the fictional obligor/counterparty rating. Maturity is an
    authored contractual date, independent of supplied risk sensitivities.
    Decimal sensitivities are preserved exactly here and in CSV, then converted
    to the canonical Position float contract only at the valuation boundary.
    """

    portfolio_id: NonEmptyString
    portfolio_version: NonEmptyString
    valuation_currency: CurrencyCode
    as_of: AwareDatetime
    snapshot_id: NonEmptyString
    schema_version: Literal["synthetic-portfolio-csv-v1"]
    generator_version: Literal["synthetic-portfolio-generator-v1"]
    seed: Literal[20261004]
    source_reference: Literal["scripts/generate_synthetic_portfolio.py"]
    source_terms: Literal["MIT; project-authored synthetic data"]
    evidence_kind: Literal["synthetic"]
    position_id: NonEmptyString
    asset_type: Literal[AssetType.LOAN, AssetType.BOND, AssetType.DERIVATIVE]
    notional: Annotated[Decimal, Field(gt=0)]
    notional_unit: Literal["USD base market value"]
    currency: Literal["USD"]
    sector: NonEmptyString
    region: Literal["United States", "Europe", "India", "Asia-Pacific"]
    obligor: NonEmptyString
    rating: Literal["AAA", "AA", "A", "BBB", "BB", "B"]
    maturity_date: date
    factor_exposures: dict[NonEmptyString, Decimal]
    factor_units: dict[NonEmptyString, ShockUnit]
    sensitivity_units: dict[NonEmptyString, NonEmptyString]

    @model_validator(mode="after")
    def validate_economics_and_units(self) -> SyntheticPosition:
        if self.currency != self.valuation_currency:
            raise ValueError("all base values and sensitivities must use valuation currency USD")
        if self.maturity_date <= self.as_of.date():
            raise ValueError("maturity_date must follow as_of")
        if not self.factor_exposures:
            raise ValueError("explicit factor exposures are required")
        factors, sensitivities = _exposure_metadata(self.factor_exposures)
        if self.factor_units != factors or self.sensitivity_units != sensitivities:
            raise ValueError("explicit factor and sensitivity units must match adapter conventions")
        exposures = self.factor_exposures
        if self.asset_type is AssetType.DERIVATIVE:
            kinds = {key.partition(":")[0] for key in exposures}
            if kinds & {"gamma", "vega"}:
                spots = {
                    key.partition(":")[2]
                    for key in exposures
                    if key.startswith(("delta:", "gamma:"))
                }
                expected = {f"{kind}:{factor}" for factor in spots for kind in ("delta", "gamma")}
                if not spots or set(exposures) != expected | {"vega:VOLATILITY"}:
                    raise ValueError("nonlinear exposures require complete delta/gamma/vega")
            elif not kinds <= {"delta", "dv01"}:
                raise ValueError("unsupported linear derivative sensitivities")
        else:
            if any(
                key.partition(":")[0] not in {"duration", "convexity", "cs01", "ead", "lgd"}
                for key in exposures
            ):
                raise ValueError("unsupported fixed-income sensitivities")
            for key in exposures:
                kind, _, factor = key.partition(":")
                if kind in {"duration", "convexity"} and any(
                    f"{paired}:{factor}" not in exposures for paired in ("duration", "convexity")
                ):
                    raise ValueError("duration/convexity must be paired")
            if ("ead" in exposures) != ("lgd" in exposures):
                raise ValueError("EAD/LGD must be paired")
            if "ead" in exposures and (exposures["ead"] < 0 or not 0 <= exposures["lgd"] <= 1):
                raise ValueError("invalid EAD/LGD")
        return self

    def to_position(self) -> Position:
        return Position(
            position_id=self.position_id,
            asset_type=self.asset_type,
            notional=self.notional,
            currency=self.currency,
            sector=self.sector,
            region=self.region,
            obligor=self.obligor,
            factor_exposures={key: float(value) for key, value in self.factor_exposures.items()},
        )


class AuditedPosition(SyntheticPosition):
    """SHA-256 of the canonical JSON row payload excluding content_hash itself."""

    content_hash: Sha256Hex

    @model_validator(mode="after")
    def verify_content_hash(self) -> AuditedPosition:
        if self.content_hash != _hash(self.model_dump(mode="json", exclude={"content_hash"})):
            raise ValueError("row content_hash mismatch")
        return self


CSV_FIELDS = tuple(AuditedPosition.model_fields)
JSON_FIELDS = frozenset({"factor_exposures", "factor_units", "sensitivity_units"})


class SyntheticPortfolioArtifact(DomainModel):
    """Typed audit wrapper; retain this when consuming rating/maturity/provenance."""

    rows: Annotated[tuple[AuditedPosition, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def verify_portfolio_identity(self) -> SyntheticPortfolioArtifact:
        identity = self.rows[0].model_dump(include=set(PORTFOLIO_FIELDS))
        if any(row.model_dump(include=set(PORTFOLIO_FIELDS)) != identity for row in self.rows):
            raise ValueError("inconsistent portfolio metadata across CSV rows")
        self.to_portfolio()  # Canonical uniqueness and finite-value validation.
        return self

    def to_portfolio(self) -> Portfolio:
        first = self.rows[0]
        return Portfolio(
            portfolio_id=first.portfolio_id,
            positions=tuple(r.to_position() for r in self.rows),
            valuation_currency=first.valuation_currency,
            as_of=first.as_of,
            version=first.portfolio_version,
        )

    @property
    def content_hash(self) -> str:
        """SHA-256 of the canonical CSV bytes (no self-referential file hash column)."""
        return hashlib.sha256(self.to_csv_bytes()).hexdigest()

    def to_csv_bytes(self) -> bytes:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in self.rows:
            data = row.model_dump(mode="json")
            writer.writerow(
                {
                    key: _json(value) if key in JSON_FIELDS else str(value)
                    for key, value in data.items()
                }
            )
        return stream.getvalue().encode("utf-8")


def generate_synthetic_portfolio() -> SyntheticPortfolioArtifact:
    """Generate exactly 60 positions with a local fixed RNG and frozen timestamp."""
    rng = random.Random(SEED)
    rows: list[AuditedPosition] = []
    for index in range(60):
        region_index = (index // 3) % 4
        region = REGIONS[region_index]
        asset = (AssetType.LOAN, AssetType.BOND, AssetType.DERIVATIVE)[index % 3]
        rate = ("RATE-USD", "RATE-EUR", "RATE-INR", "RATE-USD")[region_index]
        equity = ("EQUITY-US", "EQUITY-EUROPE", "EQUITY-INDIA", "EQUITY-ASIA")[region_index]
        fx = "EUR-USD" if region_index == 1 else "USD-INR"
        base_value = Decimal(rng.randint(1000000, 15000000)).quantize(Decimal("0.01"))
        maturity_years = rng.randint(1, 10)
        if asset is AssetType.DERIVATIVE:
            base_value = (base_value / 10).quantize(Decimal("0.01"))
            exposures = {
                f"delta:{equity}": Decimal(rng.randint(500, 5000)),
                f"delta:{fx}": Decimal(rng.randint(1000, 10000)),
                "delta:COMMODITY": Decimal(rng.randint(500, 2500)),
            }
            if (index // 12) % 2:
                exposures.update(
                    {
                        f"gamma:{equity}": Decimal("0.50"),
                        f"gamma:{fx}": Decimal("0.25"),
                        "gamma:COMMODITY": Decimal("0.10"),
                        "vega:VOLATILITY": Decimal(rng.randint(1000, 5000)),
                    }
                )
            else:
                exposures[f"dv01:{rate}"] = -Decimal(rng.randint(100, 1000))
        else:
            duration = Decimal(rng.randint(10, maturity_years * 10)) / 10
            exposures = {
                f"duration:{rate}": duration,
                f"convexity:{rate}": duration * duration + duration,
                "cs01:CREDIT-SPREAD": -Decimal(rng.randint(200, 2500)),
                "ead": base_value,
                "lgd": Decimal(rng.randint(25, 65)) / 100,
            }
        factors, units = _exposure_metadata(exposures)
        record = SyntheticPosition(
            portfolio_id="synthetic-wholesale-global-v1",
            portfolio_version="synthetic-portfolio-v1",
            valuation_currency="USD",
            as_of=AS_OF,
            snapshot_id="synthetic-portfolio-20261004-v1",
            schema_version="synthetic-portfolio-csv-v1",
            generator_version="synthetic-portfolio-generator-v1",
            seed=SEED,
            source_reference="scripts/generate_synthetic_portfolio.py",
            source_terms="MIT; project-authored synthetic data",
            evidence_kind="synthetic",
            position_id=f"SYN-{index + 1:03d}",
            asset_type=asset,
            notional=base_value,
            notional_unit="USD base market value",
            currency="USD",
            sector=SECTORS[(index // 3 + index % 3) % len(SECTORS)],
            region=region,
            obligor=f"Synthetic {region} Obligor {index + 1:03d}",
            rating=RATINGS[(index // 3 + index % 3) % len(RATINGS)],
            maturity_date=date(AS_OF.year + maturity_years, AS_OF.month, AS_OF.day),
            factor_exposures=exposures,
            factor_units=factors,
            sensitivity_units=units,
        )
        rows.append(
            AuditedPosition(
                **record.model_dump(),
                content_hash=_hash(record.model_dump(mode="json")),
            )
        )
    return SyntheticPortfolioArtifact(rows=tuple(rows))


def write_portfolio(path: Path, artifact: SyntheticPortfolioArtifact) -> None:
    """Exclusively create CSV, or verify an existing identical byte artifact.

    A changed artifact requires a new path/version; existing bytes are never
    replaced. Exclusive creation also prevents clobbering concurrent writers.
    """
    validated = SyntheticPortfolioArtifact.model_validate(artifact.model_dump())
    payload = validated.to_csv_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise FileExistsError("immutable portfolio differs; use a new version path") from None


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in CSV metadata")
        result[key] = value
    return result


def load_artifact(path: Path) -> SyntheticPortfolioArtifact:
    """Read strict canonical CSV, checking provenance, units, identity, and hashes."""
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""), strict=True)
    if reader.fieldnames is None or tuple(reader.fieldnames) != CSV_FIELDS:
        raise ValueError("CSV must contain the exact versioned columns")
    rows: list[AuditedPosition] = []
    for data in reader:
        if set(data) != set(CSV_FIELDS) or any(
            value is None or value == "" for value in data.values()
        ):
            raise ValueError("CSV rows require every column, without missing or extra cells")
        parsed: dict[str, object] = dict(data)
        for key in JSON_FIELDS:
            value = json.loads(data[key], object_pairs_hook=_unique_json_object)
            if not isinstance(value, dict):
                raise ValueError("CSV sensitivity/unit metadata must be JSON objects")
            if any(not isinstance(item, str) for item in value.values()):
                raise ValueError("CSV exposure Decimals and unit metadata require explicit strings")
            parsed[key] = value
        parsed["seed"] = int(data["seed"])
        rows.append(AuditedPosition.model_validate(parsed))
    artifact = SyntheticPortfolioArtifact(rows=tuple(rows))
    if artifact.to_csv_bytes() != payload:
        raise ValueError("CSV bytes must use the canonical versioned representation")
    return artifact


def load_portfolio(path: Path) -> Portfolio:
    """Canonical valuation seam; use load_artifact to retain all audit metadata."""
    return load_artifact(path).to_portfolio()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_portfolio(args.output, generate_synthetic_portfolio())


if __name__ == "__main__":
    main()
