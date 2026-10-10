"""Replay the frozen offline demo; use --manual-exercise for fictional Stress Tests.

The default full replay refuses while verified empirical/model prerequisites are
unavailable. This command never refreshes providers or writes artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Literal

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import AwareDatetime

from risk_engine.domain import DomainModel, NonEmptyString, Sha256Hex, SourceItem, StressResult
from risk_engine.stress.module import StressEngine
from scripts.generate_synthetic_portfolio import load_artifact
from scripts.prepare_demo_snapshot import (
    MANIFEST_PATH,
    PORTFOLIO_PATH,
    DemoReadiness,
    GovernedHypotheticalScenario,
    HypotheticalEvent,
    load_demo_snapshot,
)

OUTPUT_SCHEMA = "offline-demo-output-v1"


class SnapshotIdentity(DomainModel):
    snapshot_id: NonEmptyString
    schema_version: NonEmptyString
    generator_version: NonEmptyString
    manifest_sha256: Sha256Hex
    manifest_content_hash: Sha256Hex
    authored_at: AwareDatetime
    source_terms: NonEmptyString
    evidence_kind: NonEmptyString


class PortfolioIdentity(DomainModel):
    portfolio_id: NonEmptyString
    version: NonEmptyString
    artifact_sha256: Sha256Hex
    as_of: AwareDatetime
    valuation_currency: NonEmptyString


class ExerciseEvidence(DomainModel):
    source_items: tuple[SourceItem, ...]
    events: tuple[HypotheticalEvent, ...]
    signal_status: NonEmptyString
    market_evidence_kind: NonEmptyString
    calibration_directory_status: NonEmptyString
    readiness: DemoReadiness


class ScenarioExercise(DomainModel):
    governance: GovernedHypotheticalScenario
    stress_result: StressResult


class ManualExerciseOutput(DomainModel):
    schema_version: Literal["offline-demo-output-v1"]
    mode: Literal["manual-exercise"]
    snapshot: SnapshotIdentity
    portfolio: PortfolioIdentity
    evidence: ExerciseEvidence
    scenarios: tuple[ScenarioExercise, ...]


class FullReplayRefusal(DomainModel):
    schema_version: Literal["offline-demo-output-v1"]
    mode: Literal["full-replay"]
    status: Literal["unavailable"]
    snapshot_id: NonEmptyString
    missing_prerequisites: tuple[NonEmptyString, ...]


class InvalidInputRefusal(DomainModel):
    schema_version: Literal["offline-demo-output-v1"]
    status: Literal["invalid-input"]
    reason: NonEmptyString


def _json_bytes(record: DomainModel) -> bytes:
    return (
        json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root containing the immutable demo snapshot and Synthetic Portfolio.",
    )
    parser.add_argument(
        "--manual-exercise",
        action="store_true",
        help="Run the fictional manual Stress Test; without this flag, request full replay.",
    )
    arguments = parser.parse_args(argv)
    root: Path = arguments.root
    try:
        snapshot = load_demo_snapshot(root)
        artifact = load_artifact(root / PORTFOLIO_PATH)
        if artifact.content_hash != snapshot.manifest.portfolio.sha256:
            raise ValueError("Synthetic Portfolio byte hash disagrees with verified snapshot")
        portfolio = artifact.to_portfolio()
        if portfolio.portfolio_id != snapshot.manifest.portfolio.portfolio_id or (
            portfolio.version != snapshot.manifest.portfolio.portfolio_version
            or portfolio.as_of != snapshot.manifest.portfolio.as_of
            or portfolio.valuation_currency != snapshot.manifest.portfolio.valuation_currency
        ):
            raise ValueError("Synthetic Portfolio disagrees with verified snapshot")
        manifest_bytes = (root / MANIFEST_PATH).read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if manifest_bytes != snapshot.artifact_bytes()[MANIFEST_PATH]:
            raise ValueError("snapshot manifest changed during verification")
        if not arguments.manual_exercise:
            refusal = FullReplayRefusal(
                schema_version=OUTPUT_SCHEMA,
                mode="full-replay",
                status="unavailable",
                snapshot_id=snapshot.manifest.snapshot_id,
                missing_prerequisites=snapshot.manifest.readiness.missing_prerequisites,
            )
            sys.stdout.buffer.write(_json_bytes(refusal))
            return 2
        scenarios = tuple(
            ScenarioExercise(
                governance=governed,
                stress_result=StressEngine().run(
                    portfolio, snapshot.manifest.market_snapshot, governed.scenario
                ),
            )
            for governed in snapshot.manifest.scenarios
        )
        result = ManualExerciseOutput(
            schema_version=OUTPUT_SCHEMA,
            mode="manual-exercise",
            snapshot=SnapshotIdentity(
                snapshot_id=snapshot.manifest.snapshot_id,
                schema_version=snapshot.manifest.schema_version,
                generator_version=snapshot.manifest.generator_version,
                manifest_sha256=manifest_sha256,
                manifest_content_hash=snapshot.manifest.content_hash,
                authored_at=snapshot.manifest.authored_at,
                source_terms=snapshot.manifest.source_terms,
                evidence_kind=snapshot.manifest.evidence_kind,
            ),
            portfolio=PortfolioIdentity(
                portfolio_id=portfolio.portfolio_id,
                version=portfolio.version,
                artifact_sha256=snapshot.manifest.portfolio.sha256,
                as_of=portfolio.as_of,
                valuation_currency=portfolio.valuation_currency,
            ),
            evidence=ExerciseEvidence(
                source_items=snapshot.source_items,
                events=snapshot.events,
                signal_status=snapshot.manifest.signal_status,
                market_evidence_kind=snapshot.manifest.market_evidence_kind,
                calibration_directory_status=snapshot.manifest.calibration_directory_status,
                readiness=snapshot.manifest.readiness,
            ),
            scenarios=scenarios,
        )
    except (ValueError, OSError) as error:
        invalid = InvalidInputRefusal(
            schema_version=OUTPUT_SCHEMA,
            status="invalid-input",
            reason=str(error).replace(str(root), "<root>"),
        )
        sys.stderr.buffer.write(_json_bytes(invalid))
        return 2
    sys.stdout.buffer.write(_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
