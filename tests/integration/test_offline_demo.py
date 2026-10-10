"""Offline CLI replay uses frozen fictional inputs and real production valuation."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from risk_engine.domain import AttributionDimension, SourceType, StressResult
from risk_engine.stress.module import StressEngine
from scripts import run_demo as demo
from scripts.generate_synthetic_portfolio import (
    AuditedPosition,
    SyntheticPortfolioArtifact,
    load_artifact,
    load_portfolio,
)
from scripts.prepare_demo_snapshot import OUTPUT_PATHS, PORTFOLIO_PATH, load_demo_snapshot

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/golden/demo-output.json"


def _frozen_root(tmp_path: Path) -> Path:
    for relative in (*OUTPUT_PATHS, PORTFOLIO_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


def _run(tmp_path: Path, root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    # A fresh interpreter exercises the direct launcher from another working
    # directory. sitecustomize denies outbound connections and provider/model
    # imports throughout that interpreter's life, including startup.
    guard = tmp_path / "offline-guard"
    guard.mkdir(exist_ok=True)
    (guard / "sitecustomize.py").write_text(
        "import builtins\n"
        "import socket\n"
        "def forbidden(*args, **kwargs):\n"
        "    raise AssertionError('offline demo attempted a network connection')\n"
        "socket.create_connection = forbidden\n"
        "socket.socket.connect = forbidden\n"
        "original_import = builtins.__import__\n"
        "def guarded_import(name, *args, **kwargs):\n"
        "    if name == 'transformers' or name.startswith('transformers.') or "
        "name == 'torch' or name.startswith('torch.') or "
        "name in {'risk_engine.nlp.local_models', 'risk_engine.data.gdelt', "
        "'risk_engine.backtest.runtime'}:\n"
        "        raise AssertionError('offline demo attempted provider/model load: ' + name)\n"
        "    return original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded_import\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_demo.py"), "--root", str(root), *arguments],
        cwd=tmp_path,
        env=dict(
            os.environ, PYTHONPATH=os.pathsep.join((str(guard), str(ROOT / "src"), str(ROOT)))
        ),
        capture_output=True,
        check=False,
    )


def test_manual_exercise_is_byte_stable_and_uses_real_stress_results(tmp_path: Path) -> None:
    root = _frozen_root(tmp_path)
    before = {
        relative: (root / relative).read_bytes() for relative in (*OUTPUT_PATHS, PORTFOLIO_PATH)
    }
    first = _run(tmp_path, root, "--manual-exercise")
    second = _run(tmp_path, root, "--manual-exercise")
    assert first.returncode == second.returncode == 0, first.stderr.decode()
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout == GOLDEN.read_bytes()
    assert before == {
        relative: (root / relative).read_bytes() for relative in (*OUTPUT_PATHS, PORTFOLIO_PATH)
    }

    document = json.loads(first.stdout)
    snapshot = load_demo_snapshot(root)
    portfolio = load_portfolio(root / PORTFOLIO_PATH)
    assert document["schema_version"] == "offline-demo-output-v1"
    assert document["mode"] == "manual-exercise"
    assert document["snapshot"]["snapshot_id"] == snapshot.manifest.snapshot_id
    assert (
        document["snapshot"]["manifest_sha256"]
        == hashlib.sha256((root / OUTPUT_PATHS[-1]).read_bytes()).hexdigest()
    )
    assert document["snapshot"]["manifest_content_hash"] == snapshot.manifest.content_hash
    assert (
        document["portfolio"]["artifact_sha256"]
        == hashlib.sha256((root / PORTFOLIO_PATH).read_bytes()).hexdigest()
    )
    assert document["portfolio"]["portfolio_id"] == portfolio.portfolio_id
    assert {item["source_type"] for item in document["evidence"]["source_items"]} == {
        SourceType.NEWS.value,
        SourceType.SOCIAL.value,
    }
    assert {item["source_reference"] for item in document["evidence"]["source_items"]} == {
        item.source_reference for item in snapshot.source_items
    }
    assert document["evidence"]["signal_status"] == snapshot.manifest.signal_status
    assert document["evidence"]["readiness"] == snapshot.manifest.readiness.model_dump(mode="json")
    assert document["evidence"]["market_evidence_kind"] == "project-authored-hypothetical-levels"
    assert "risk_signals" not in document
    assert "trigger_decisions" not in document
    assert "automatic_actions" not in document

    assert [row["governance"]["scenario"]["scenario_id"] for row in document["scenarios"]] == [
        "demo-rbi-policy-v1",
        "demo-rbi-policy-override-v1",
        "demo-indian-credit-v1",
    ]
    for row, governed in zip(document["scenarios"], snapshot.manifest.scenarios, strict=True):
        assert row["governance"] == governed.model_dump(mode="json")
        actual = StressResult.model_validate(row["stress_result"])
        expected = StressEngine().run(
            portfolio, snapshot.manifest.market_snapshot, governed.scenario
        )
        assert actual == expected
        assert actual.base_value - actual.stressed_value == actual.absolute_loss
        assert actual.valuation_coverage == 1
        assert actual.unsupported_position_ids == ()
        for dimension in (
            AttributionDimension.ASSET,
            AttributionDimension.SECTOR,
            AttributionDimension.REGION,
            AttributionDimension.OBLIGOR,
            AttributionDimension.FACTOR,
        ):
            assert (
                sum(
                    (part.loss for part in actual.attribution if part.dimension is dimension),
                    Decimal(0),
                )
                == actual.absolute_loss
            )
    original, override, credit = (row["governance"] for row in document["scenarios"])
    assert original["parent_scenario_id"] is None
    assert override["parent_scenario_id"] == original["scenario"]["scenario_id"]
    assert override["scenario"]["scenario_id"] != original["scenario"]["scenario_id"]
    assert override["scenario"]["override_reason"]
    assert override["source_item_id"] == original["source_item_id"]
    assert credit["event_id"] != original["event_id"]
    assert all(
        row["governance"]["origin_kind"] == "manual-source-reference"
        for row in document["scenarios"]
    )


def test_default_refuses_unavailable_full_replay_without_stress_results(tmp_path: Path) -> None:
    root = _frozen_root(tmp_path)
    before = {
        relative: (root / relative).read_bytes() for relative in (*OUTPUT_PATHS, PORTFOLIO_PATH)
    }
    process = _run(tmp_path, root)
    assert process.returncode != 0
    assert process.stderr == b""
    refusal = json.loads(process.stdout)
    assert refusal["schema_version"] == "offline-demo-output-v1"
    assert refusal["mode"] == "full-replay"
    assert refusal["status"] == "unavailable"
    assert refusal["snapshot_id"] == "project-authored-demo-20261007-v1"
    assert refusal["missing_prerequisites"] == list(
        load_demo_snapshot(root).manifest.readiness.missing_prerequisites
    )
    assert "scenarios" not in refusal
    assert "risk_signals" not in refusal
    assert "traceback" not in process.stdout.decode().lower()
    assert before == {
        relative: (root / relative).read_bytes() for relative in (*OUTPUT_PATHS, PORTFOLIO_PATH)
    }


def test_manual_exercise_preserves_both_source_types_for_each_case(tmp_path: Path) -> None:
    root = _frozen_root(tmp_path)
    process = _run(tmp_path, root, "--manual-exercise")
    assert process.returncode == 0, process.stderr.decode()
    document = json.loads(process.stdout)
    source_types = {
        item["source_item_id"]: item["source_type"] for item in document["evidence"]["source_items"]
    }
    events = document["evidence"]["events"]
    assert {event["case_kind"] for event in events} == {
        "hypothetical-rbi-policy",
        "fictional-indian-credit-stress",
    }
    assert {source_id for event in events for source_id in event["source_item_ids"]} == set(
        source_types
    )
    for event in events:
        assert {source_types[source_id] for source_id in event["source_item_ids"]} == {
            SourceType.NEWS.value,
            SourceType.SOCIAL.value,
        }
        assert event["source_terms"] == "MIT; project-authored fictional exercise data"
        assert event["assumption_version"] == "hypothetical-exercise-assumptions-v1"
    by_id = {event["event_id"]: event for event in events}
    for row in document["scenarios"]:
        governed = row["governance"]
        assert governed["source_item_id"] in by_id[governed["event_id"]]["source_item_ids"]


def test_portfolio_replacement_between_verification_and_valuation_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    root = _frozen_root(tmp_path)
    original_snapshot_load = demo.load_demo_snapshot
    original_run = StressEngine.run
    valued_scenarios: list[str] = []

    def replace_after_verification(snapshot_root: Path) -> object:
        snapshot = original_snapshot_load(snapshot_root)
        artifact = load_artifact(root / PORTFOLIO_PATH)
        first = artifact.rows[0]
        changed = first.model_dump(mode="json", exclude={"content_hash"})
        exposures = changed["factor_exposures"]
        factor = next(iter(exposures))
        exposures[factor] = str(Decimal(exposures[factor]) + Decimal(1))
        changed_hash = hashlib.sha256(
            json.dumps(changed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        changed_row = AuditedPosition.model_validate(dict(changed, content_hash=changed_hash))
        replacement = SyntheticPortfolioArtifact(rows=(changed_row, *artifact.rows[1:]))
        assert replacement.content_hash != snapshot.manifest.portfolio.sha256
        (root / PORTFOLIO_PATH).write_bytes(replacement.to_csv_bytes())
        return snapshot

    def observed_run(
        engine: StressEngine, portfolio: object, market: object, scenario: object
    ) -> StressResult:
        valued_scenarios.append(scenario.scenario_id)
        return original_run(engine, portfolio, market, scenario)

    monkeypatch.setattr(demo, "load_demo_snapshot", replace_after_verification)
    monkeypatch.setattr(StressEngine, "run", observed_run)
    status = demo.main(["--root", str(root), "--manual-exercise"])
    output, error = capfdbinary.readouterr()
    assert status != 0
    assert output == b""
    assert b"portfolio" in error.lower()
    assert valued_scenarios == []


@pytest.mark.parametrize("failure", ["missing", "malformed", "tampered", "portfolio"])
def test_manual_exercise_refuses_unverified_inputs_without_output(
    tmp_path: Path, failure: str
) -> None:
    root = _frozen_root(tmp_path)
    target = root / (PORTFOLIO_PATH if failure == "portfolio" else OUTPUT_PATHS[0])
    if failure == "missing":
        target.unlink()
    elif failure == "malformed":
        target.write_bytes(b"{invalid json")
    else:
        target.write_bytes(target.read_bytes() + b"\n")
    process = _run(tmp_path, root, "--manual-exercise")
    assert process.returncode != 0
    assert process.stdout == b""
    assert b"Traceback" not in process.stderr
    assert (
        b"snapshot" in process.stderr.lower()
        or b"artifact" in process.stderr.lower()
        or b"portfolio" in process.stderr.lower()
    )


def test_help_identifies_modes_and_explicit_root(tmp_path: Path) -> None:
    process = _run(tmp_path, tmp_path, "--help")
    assert process.returncode == 0
    assert b"--manual-exercise" in process.stdout
    assert b"--root" in process.stdout
    assert b"full replay" in process.stdout.lower()
