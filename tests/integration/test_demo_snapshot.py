"""Frozen, fictional inputs are auditable without becoming historical evidence."""

import csv
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from risk_engine.domain import AttributionDimension, ProvenanceMethod, SourceType
from risk_engine.stress.module import StressEngine
from scripts import prepare_demo_snapshot as api
from scripts.generate_synthetic_portfolio import load_portfolio

ROOT = Path(__file__).resolve().parents[2]
PORTFOLIO = "data/portfolio/synthetic_portfolio.csv"
MANIFEST = "data/manifests/demo-snapshot.json"


def _root(tmp_path: Path) -> Path:
    destination = tmp_path / PORTFOLIO
    destination.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / PORTFOLIO, destination)
    return tmp_path


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("demo input preparation must remain offline")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def test_prepare_freezes_exact_bytes_and_honest_readiness(tmp_path: Path) -> None:
    root = _root(tmp_path)
    snapshot = api.prepare_demo_snapshot(root)
    loaded = api.load_demo_snapshot(root)
    assert loaded == snapshot
    assert snapshot.manifest.readiness.illustrative_inputs_complete is True
    assert snapshot.manifest.readiness.local_models_available is False
    assert snapshot.manifest.readiness.empirical_calibration_available is False
    assert snapshot.manifest.readiness.selected_policy_available is False
    assert snapshot.manifest.readiness.final_evaluation_available is False
    assert snapshot.manifest.readiness.actual_historical_cases_available is False
    assert snapshot.manifest.readiness.full_replay_available is False
    assert snapshot.manifest.readiness.missing_prerequisites
    assert {item.source_type for item in snapshot.source_items} == {
        SourceType.NEWS,
        SourceType.SOCIAL,
    }
    assert len(snapshot.source_items) == 4
    for item in snapshot.source_items:
        assert item.provider == "project-authored-demo"
        assert "fictional" in item.text.lower()
        assert "hypothetical" in item.text.lower()
        assert item.published_at.isoformat() == "2026-10-07T00:00:00+00:00"
        assert item.retrieved_at == item.published_at
        assert item.content_hash == hashlib.sha256(item.text.encode()).hexdigest()
        assert "MIT" in item.license
        assert "project-authored" in item.provenance
    assert {event.case_kind for event in snapshot.events} == {
        "hypothetical-rbi-policy",
        "fictional-indian-credit-stress",
    }
    assert all(event.evidence_kind == "hypothetical" for event in snapshot.events)
    for artifact in snapshot.manifest.artifacts:
        assert artifact.sha256 == hashlib.sha256((root / artifact.path).read_bytes()).hexdigest()
    before = {p: (root / p).read_bytes() for p in api.OUTPUT_PATHS}
    api.prepare_demo_snapshot(root)
    assert before == {p: (root / p).read_bytes() for p in api.OUTPUT_PATHS}
    assert api.build_demo_snapshot(root) == snapshot


def test_frozen_complete_vectors_use_production_units_and_valuation(tmp_path: Path) -> None:
    root = _root(tmp_path)
    snapshot = api.prepare_demo_snapshot(root)
    portfolio = load_portfolio(root / PORTFOLIO)
    assert snapshot.manifest.market_snapshot.as_of == portfolio.as_of
    assert snapshot.manifest.authored_at > portfolio.as_of
    # These complete simultaneous assumptions are frozen before any valuation.
    expected = {
        "demo-rbi-policy-v1": (-2, 25, 10, 1, -1, 2),
        "demo-rbi-policy-override-v1": (-3, 50, 20, 2, -2, 3),
        "demo-indian-credit-v1": (-4, -10, 100, 2, -3, 4),
    }
    factors = ("EQUITY-INDIA", "RATE-INR", "CREDIT-SPREAD", "USD-INR", "COMMODITY", "VOLATILITY")
    for governed in snapshot.manifest.scenarios:
        scenario = governed.scenario
        assert scenario.method is ProvenanceMethod.HYPOTHETICAL
        assert governed.origin_kind == "manual-source-reference"
        assert scenario.risk_signal_id == "manual-exercise:" + governed.source_item_id
        assert scenario.reference_event_ids == ()
        by_factor = {shock.factor_id: shock for shock in scenario.shocks}
        assert set(by_factor) == set(factors)
        assert tuple(by_factor[f].value for f in factors) == expected[scenario.scenario_id]
        assert all(shock.horizon_days == 1 for shock in scenario.shocks)
        result = StressEngine().run(portfolio, snapshot.manifest.market_snapshot, scenario)
        assert result == StressEngine().run(portfolio, snapshot.manifest.market_snapshot, scenario)
        assert result.valuation_coverage == 1
        assert result.unsupported_position_ids == ()
        assert result.base_value - result.stressed_value == result.absolute_loss
        assert result.absolute_loss.is_finite()
        for dimension in AttributionDimension:
            rows = [a.loss for a in result.attribution if a.dimension is dimension]
            if rows:
                assert sum(rows) == result.absolute_loss
    override = snapshot.manifest.scenarios[1]
    assert override.scenario.analyst_override
    assert override.scenario.override_reason
    assert override.parent_scenario_id == snapshot.manifest.scenarios[0].scenario.scenario_id


@pytest.mark.parametrize(
    "path",
    [
        "data/demo/news.json",
        "data/demo/social.json",
        "data/calibration/events.csv",
        "data/calibration/factor_shocks.csv",
        MANIFEST,
        PORTFOLIO,
    ],
)
def test_missing_and_tampered_inputs_refused(tmp_path: Path, path: str) -> None:
    root = _root(tmp_path)
    api.prepare_demo_snapshot(root)
    target = root / path
    original = target.read_bytes()
    target.write_bytes(original + b"tampered")
    with pytest.raises(ValueError):
        api.load_demo_snapshot(root)
    target.unlink()
    with pytest.raises(ValueError):
        api.load_demo_snapshot(root)


def test_publish_preflights_all_paths_and_refuses_partial_or_different_set(tmp_path: Path) -> None:
    root = _root(tmp_path)
    path = root / api.OUTPUT_PATHS[-2]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing immutable other version")
    with pytest.raises(FileExistsError):
        api.prepare_demo_snapshot(root)
    assert path.read_bytes() == b"existing immutable other version"
    assert not (root / "data/demo/news.json").exists()
    path.unlink()
    api.prepare_demo_snapshot(root)
    (root / "data/demo/news.json").unlink()
    with pytest.raises(FileExistsError):
        api.prepare_demo_snapshot(root)


def test_committed_inputs_match_deterministic_preparation() -> None:
    expected = api.build_demo_snapshot(ROOT)
    assert api.load_demo_snapshot(ROOT) == expected
    for path, body in expected.artifact_bytes().items():
        assert (ROOT / path).read_bytes() == body


def test_missing_versions_and_inconsistent_metadata_rejected(tmp_path: Path) -> None:
    root = _root(tmp_path)
    snapshot = api.prepare_demo_snapshot(root)
    body = snapshot.manifest.model_dump(mode="json")
    del body["schema_version"]
    with pytest.raises(ValueError):
        api.DemoManifest.model_validate(body)
    body = snapshot.manifest.model_dump(mode="json")
    body["readiness"]["empirical_calibration_available"] = True
    with pytest.raises(ValueError):
        api.DemoManifest.model_validate(body)
    body = snapshot.manifest.model_dump(mode="json")
    body["market_snapshot"]["observations"][0]["unit"] = "currency"
    _sign_manifest(body)
    with pytest.raises(ValueError):
        api.DemoManifest.model_validate(body)
    source = snapshot.news.model_dump(mode="json")
    source["items"][0]["content_hash"] = "a" * 64
    with pytest.raises(ValueError):
        api.SourceBundle.model_validate(source)
    duplicate = '{"schema_version":"one","schema_version":"two"}'
    (root / MANIFEST).write_text(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        api.load_demo_snapshot(root)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def _sign_manifest(body: dict[str, object]) -> None:
    unsigned = {key: value for key, value in body.items() if key != "content_hash"}
    body["content_hash"] = hashlib.sha256(_canonical(unsigned)).hexdigest()


def _refresh_manifest(root: Path, changed_path: str) -> None:
    # Independent checksums ensure these refusals are semantic, not byte-hash failures.
    manifest = json.loads((root / MANIFEST).read_bytes())
    for ref in manifest["artifacts"]:
        if ref["path"] == changed_path:
            ref["sha256"] = hashlib.sha256((root / changed_path).read_bytes()).hexdigest()
    _sign_manifest(manifest)
    (root / MANIFEST).write_bytes(_canonical(manifest))


@pytest.mark.parametrize("mutation", ["derivation", "case-source-lineage"])
def test_hash_valid_event_metadata_must_match_its_case(tmp_path: Path, mutation: str) -> None:
    root = _root(tmp_path)
    api.prepare_demo_snapshot(root)
    path = "data/calibration/events.csv"
    reader = csv.DictReader(io.StringIO((root / path).read_text()))
    fields = reader.fieldnames
    assert fields is not None
    rows = list(reader)
    if mutation == "derivation":
        rows[0]["derivation_reference"] = "scripts/prepare_demo_snapshot.py#unrelated-case"
    else:
        for key in ("case_kind", "event_class", "description"):
            rows[0][key], rows[1][key] = rows[1][key], rows[0][key]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (root / path).write_text(output.getvalue())
    _refresh_manifest(root, path)
    with pytest.raises(ValueError, match="case"):
        api.load_demo_snapshot(root)


def test_hash_valid_scenarios_cannot_be_reassigned_to_another_case(tmp_path: Path) -> None:
    root = _root(tmp_path)
    api.prepare_demo_snapshot(root)
    path = "data/calibration/factor_shocks.csv"
    reader = csv.DictReader(io.StringIO((root / path).read_text()))
    fields = reader.fieldnames
    assert fields is not None
    rows = list(reader)
    for row in rows:
        if row["scenario_id"].startswith("demo-rbi-policy-"):
            row["event_id"] = "demo-event-credit-v1"
            row["source_item_id"] = "demo-news-credit-v1"
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (root / path).write_text(output.getvalue())

    manifest = json.loads((root / MANIFEST).read_bytes())
    for governed in manifest["scenarios"]:
        if governed["scenario"]["scenario_id"].startswith("demo-rbi-policy-"):
            governed["event_id"] = "demo-event-credit-v1"
            governed["source_item_id"] = "demo-news-credit-v1"
            governed["scenario"]["risk_signal_id"] = "manual-exercise:demo-news-credit-v1"
    for ref in manifest["artifacts"]:
        if ref["path"] == path:
            ref["sha256"] = hashlib.sha256((root / path).read_bytes()).hexdigest()
    _sign_manifest(manifest)
    (root / MANIFEST).write_bytes(_canonical(manifest))

    with pytest.raises(ValueError, match="scenario.*case"):
        api.load_demo_snapshot(root)


def test_hash_valid_source_reference_must_identify_its_source_item(tmp_path: Path) -> None:
    root = _root(tmp_path)
    api.prepare_demo_snapshot(root)
    path = "data/demo/news.json"
    body = json.loads((root / path).read_bytes())
    body["items"][0]["source_reference"] = "project-authored:demo-news-credit-v1"
    (root / path).write_bytes(_canonical(body))
    _refresh_manifest(root, path)

    with pytest.raises(ValueError, match="source reference"):
        api.load_demo_snapshot(root)


def test_hash_valid_readiness_cannot_omit_required_prerequisites(tmp_path: Path) -> None:
    root = _root(tmp_path)
    snapshot = api.prepare_demo_snapshot(root)
    body = snapshot.manifest.model_dump(mode="json")
    body["readiness"]["missing_prerequisites"] = ["Models unavailable"] * 6
    _sign_manifest(body)
    with pytest.raises(ValueError):
        api.DemoManifest.model_validate(body)


def test_direct_script_launch_prepares_the_explicit_root(tmp_path: Path) -> None:
    root = _root(tmp_path)
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_demo_snapshot.py"), "--root", str(root)],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == api.load_demo_snapshot(root).manifest.readiness.model_dump(
        mode="json"
    )
