"""HTTP contracts over explicit local snapshots and immutable Risk Signals."""

import hashlib
import json
import runpy
import socket
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from risk_engine.api.app import AnalyzeSignalsRequest, create_app
from risk_engine.api.dependencies import (
    AppContainer,
    ArtifactNotFoundError,
    InMemorySignalRepository,
    LocalBacktestArtifactReader,
    VersionConflictError,
)
from risk_engine.data.duckdb_store import DuckDbSnapshotStore
from risk_engine.data.interfaces import ProviderRequest, RawEnvelope
from risk_engine.data.module import DataModule
from risk_engine.domain import RiskSignal, SourceItem

TIME = datetime(2026, 1, 10, tzinfo=timezone.utc)
EXPECTED_SIGNAL = {
    "signal_id": "signal:synthetic-1",
    "source_item_id": "source-1",
    "entity": {
        "entity_id": "bank:alpha",
        "canonical_name": "Alpha Bank",
        "confidence": 0.9,
        "evidence": "Alpha Bank failed",
        "ambiguous": False,
        "candidate_entity_ids": [],
    },
    "sentiment": -0.75,
    "event_class": "Credit/Default",
    "impact": {
        "impact_score": 8,
        "expected_reference_loss": "100",
        "loss_lower_bound": "60",
        "loss_upper_bound": "140",
        "loss_currency": "USD",
        "analogue_count": 1,
        "analogue_ids": ["synthetic-analogue-1"],
        "backoff_level": "event-class-region",
        "method": "empirical",
        "calibration_version": "synthetic-impact-v1",
        "reference_basket_version": "reference-basket-v1",
    },
    "confidence": 0.8,
    "confidence_target": "entity_and_event_class_joint_correctness",
    "rationale": "Alpha Bank failed",
    "evidence": ["Alpha Bank failed"],
    "flags": [],
    "versions": {
        "schema_version": "risk-signal-schema-v1",
        "model_version": "synthetic-model-v1",
        "calibration_version": "synthetic-calibration-v1",
        "snapshot_version": "synthetic-v1",
    },
    "portfolio_materiality": {"absolute_loss": "20000", "percentage_loss": 0.02, "currency": "USD"},
    "action_priority": 0.128,
}


class LocalRiskEngine:
    """Deterministic test port; real snapshot storage/replay and API stay active."""

    def __init__(self) -> None:
        self.error: Exception | None = None
        self.changed = False
        self.calls: list[tuple[tuple[SourceItem, ...], datetime]] = []

    def analyze(self, items: Sequence[SourceItem], as_of: datetime) -> list[RiskSignal]:
        self.calls.append((tuple(items), as_of))
        if self.error is not None:
            raise self.error
        result = []
        for item in items:
            value = {
                **EXPECTED_SIGNAL,
                "source_item_id": item.source_item_id,
                "signal_id": "signal:" + item.source_item_id,
            }
            if item.source_item_id == "source-1":
                value["signal_id"] = "signal:synthetic-1"
            if self.changed:
                value["confidence"] = 0.7
            result.append(RiskSignal.model_validate(value))
        return result


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = DuckDbSnapshotStore(tmp_path / "snapshots.duckdb")
    snapshots = []
    for source_id in ("source-1", "source-2"):
        envelope = RawEnvelope.from_response(
            provider="synthetic",
            request=ProviderRequest(provider="synthetic", parameters={}),
            retrieved_at=TIME,
            response_body=b"Alpha Bank failed",
            response_metadata={"source_id": source_id},
            source_terms_url="synthetic:MIT",
        )
        snapshots.append(store.write(envelope).snapshot_id)

    def normalize(envelope: RawEnvelope) -> list[SourceItem]:
        return [
            SourceItem(
                source_item_id=envelope.response_metadata["source_id"],
                source_type="news",
                provider="synthetic",
                text="Alpha Bank failed",
                published_at=TIME,
                retrieved_at=TIME,
                source_reference="synthetic:story",
                content_hash=envelope.content_hash,
                snapshot_id=hashlib.sha256(envelope.canonical_manifest_bytes()).hexdigest(),
                provenance="project-authored synthetic API fixture",
                license="MIT",
            )
        ]

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("implicit refresh or network access")

    monkeypatch.setattr(DataModule, "refresh", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    data = DataModule(store=store, normalizers={"synthetic": normalize})
    engine = LocalRiskEngine()
    repository = InMemorySignalRepository()
    app = create_app(AppContainer(data=data, risk_engine=engine, signals=repository))
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, engine, repository, snapshots


def request(snapshot_ids: list[str]) -> dict[str, object]:
    return {"snapshot_ids": snapshot_ids, "as_of": "2026-01-10T00:00:00Z"}


def assert_safe_error(response, status: int, code: str) -> None:
    assert response.status_code == status
    value = response.json()
    assert set(value) == {"code", "message", "correlation_id"}
    assert value["code"] == code
    assert value["correlation_id"] == response.headers["X-Correlation-ID"]
    UUID(value["correlation_id"])
    assert all(secret not in response.text for secret in ("/home/", "secret", "traceback", "input"))


def test_exact_risk_signal_json_is_stored_without_processing_get(setup) -> None:
    client, engine, repository, snapshots = setup
    assert client.get("/v1/signals").json() == []
    response = client.post("/v1/signals/analyze", json=request(snapshots[:1]))
    assert response.status_code == 200
    assert response.json() == [EXPECTED_SIGNAL]
    UUID(response.headers["X-Correlation-ID"])
    assert client.get("/v1/signals").json() == [EXPECTED_SIGNAL]
    assert len(engine.calls) == 1
    assert engine.calls[0][1] == TIME
    assert engine.calls[0][0][0].snapshot_id == snapshots[0]
    stored = repository.get("signal:synthetic-1")
    assert stored.snapshot_ids == tuple(snapshots[:1])
    assert stored.as_of == TIME


def test_multi_snapshot_replay_preserves_explicit_order(setup) -> None:
    client, engine, _, snapshots = setup
    response = client.post("/v1/signals/analyze", json=request(list(reversed(snapshots))))
    assert response.status_code == 200
    assert [signal["source_item_id"] for signal in response.json()] == ["source-2", "source-1"]
    assert [item.snapshot_id for item in engine.calls[0][0]] == list(reversed(snapshots))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"as_of": "2026-01-10T00:00:00Z"},
        {"snapshot_ids": ["snapshot"]},
        {"snapshot_ids": [], "as_of": "2026-01-10T00:00:00Z"},
        {"snapshot_ids": ["  "], "as_of": "2026-01-10T00:00:00Z"},
        {"snapshot_ids": ["one", "one"], "as_of": "2026-01-10T00:00:00Z"},
        {"snapshot_ids": ["one"], "as_of": "2026-01-10T00:00:00"},
        {"snapshot_ids": ["one"], "as_of": "/home/secret/token"},
        {"snapshot_ids": ["one"], "as_of": 1768003200},
        {"snapshot_ids": ["one"], "as_of": "2026-01-10T00:00:00Z", "/home/secret": "secret"},
    ],
)
def test_incomplete_or_invalid_requests_are_safe_and_do_not_analyze(setup, payload) -> None:
    client, engine, _, _ = setup
    assert_safe_error(client.post("/v1/signals/analyze", json=payload), 422, "invalid_request")
    assert engine.calls == []
    assert client.get("/v1/signals").json() == []


def test_malformed_json_never_echoes_raw_body(setup) -> None:
    client, engine, _, _ = setup
    response = client.post(
        "/v1/signals/analyze",
        content='{"secret":"/home/token"',
        headers={"Content-Type": "application/json"},
    )
    assert_safe_error(response, 422, "invalid_request")
    assert engine.calls == []


def test_missing_snapshot_is_404_and_publishes_no_partial_batch(setup) -> None:
    client, engine, _, snapshots = setup
    response = client.post("/v1/signals/analyze", json=request([snapshots[0], "/home/secret"]))
    assert_safe_error(response, 404, "snapshot_not_found")
    assert engine.calls == []
    assert client.get("/v1/signals").json() == []


@pytest.mark.parametrize(
    "error,status,code",
    [
        (ValueError("/home/secret invalid calibration"), 422, "invalid_domain"),
        (RuntimeError("/home/secret model token"), 500, "internal_error"),
    ],
)
def test_failure_preserves_successful_records_with_safe_correlation(setup, error, status, code):
    client, engine, _, snapshots = setup
    assert client.post("/v1/signals/analyze", json=request(snapshots[:1])).status_code == 200
    engine.error = error
    assert_safe_error(client.post("/v1/signals/analyze", json=request(snapshots[1:])), status, code)
    assert client.get("/v1/signals").json() == [EXPECTED_SIGNAL]


def test_exact_replay_is_idempotent_and_conflict_never_overwrites(setup) -> None:
    client, engine, _, snapshots = setup
    payload = request(snapshots[:1])
    for _ in range(2):
        assert client.post("/v1/signals/analyze", json=payload).json() == [EXPECTED_SIGNAL]
    assert client.get("/v1/signals").json() == [EXPECTED_SIGNAL]
    engine.changed = True
    response = client.post("/v1/signals/analyze", json=request(snapshots))
    assert_safe_error(response, 409, "version_conflict")
    assert client.get("/v1/signals").json() == [EXPECTED_SIGNAL]


def test_health_does_no_processing_and_ids_are_generated_not_reflected(setup) -> None:
    client, engine, _, _ = setup
    first = client.get("/health", headers={"X-Correlation-ID": "/home/secret"})
    second = client.get("/health")
    assert first.json() == {"status": "ok"}
    UUID(first.headers["X-Correlation-ID"])
    assert first.headers["X-Correlation-ID"] != second.headers["X-Correlation-ID"]
    assert engine.calls == []
    assert_safe_error(client.get("/missing/secret"), 404, "not_found")
    assert_safe_error(client.post("/health"), 405, "method_not_allowed")


def test_request_record_is_frozen() -> None:
    record = AnalyzeSignalsRequest(snapshot_ids=("one",), as_of=TIME)
    with pytest.raises(ValidationError):
        record.as_of = TIME


def test_openapi_describes_canonical_json_and_safe_errors(setup) -> None:
    client, _, _, _ = setup
    schema = client.get("/openapi.json").json()
    assert {"/health", "/v1/signals", "/v1/signals/analyze"} <= set(schema["paths"])
    analyze = schema["paths"]["/v1/signals/analyze"]["post"]
    assert analyze["responses"]["200"]["content"]["application/json"]["schema"] == {
        "type": "array",
        "items": {"$ref": "#/components/schemas/RiskSignal"},
        "title": "Response Analyze Signals V1 Signals Analyze Post",
    }
    assert {"404", "409", "422", "500"} <= set(analyze["responses"])
    assert "HTTPValidationError" not in schema["components"]["schemas"]
    required = schema["components"]["schemas"]["AnalyzeSignalsRequest"]
    assert required["required"] == ["snapshot_ids", "as_of"]
    assert required["additionalProperties"] is False
    # Freeze only Task 28's surface; Task 29 may add independent endpoints and records.
    names = (
        "AnalyzeSignalsRequest",
        "HealthResponse",
        "ErrorResponse",
        "RiskSignal",
        "EntityLink",
        "EventClass",
        "ImpactEstimate",
        "ProvenanceMethod",
        "PortfolioMateriality",
        "ConfidenceTarget",
        "SignalFlag",
        "VersionMetadata",
    )
    snapshot = {
        "openapi": schema["openapi"],
        "info": schema["info"],
        "paths": {
            path: schema["paths"][path]
            for path in ("/health", "/v1/signals", "/v1/signals/analyze")
        },
        "schemas": {name: schema["components"]["schemas"][name] for name in names},
    }
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode())
    assert digest.hexdigest() == "ca8ce3aff66ff1b73707c377e027d461e516a63c9e4542ac333b9757611f341b"


@pytest.mark.parametrize("kind", ["missing", "unresolved", "corrupt"])
def test_local_backtest_reader_classifies_unavailable_artifacts(tmp_path: Path, kind: str) -> None:
    root = tmp_path if kind != "unresolved" else Path.cwd()
    if kind == "corrupt":
        path = root / "data/calibration/selected-config.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"secret":"/home/private"}')
    reader = LocalBacktestArtifactReader(root=root, final_path=None)
    expected = VersionConflictError if kind == "corrupt" else ArtifactNotFoundError
    with pytest.raises(expected):
        reader.latest()


def test_explicit_analysis_instant_cannot_rebind_stored_signal(setup) -> None:
    client, _, _, snapshots = setup
    payload = request(snapshots[:1])
    assert client.post("/v1/signals/analyze", json=payload).status_code == 200
    payload["as_of"] = "2026-01-10T01:00:00Z"
    assert_safe_error(client.post("/v1/signals/analyze", json=payload), 409, "version_conflict")
    assert client.get("/v1/signals").json() == [EXPECTED_SIGNAL]


def test_unknown_stored_signal_is_missing_artifact(setup) -> None:
    _, _, repository, _ = setup
    with pytest.raises(ArtifactNotFoundError):
        repository.get("missing")


def test_ready_backtest_reader_verifies_real_synthetic_development_and_final(tmp_path: Path):
    # Reuse Task 27's synthetic fitted-runtime builders, keeping the actual CLI
    # publication, canonical records, cross-file receipts, and API reader real.
    fixture = runpy.run_path(str(Path(__file__).parents[1] / "backtest/test_backtest_module.py"))
    _, _, fitter, args = fixture["write_cli_inputs"](tmp_path)
    runner = fixture["cli"]()
    assert runner.main(args, fitter=fitter) == 0
    root = tmp_path / "selection"
    reader = LocalBacktestArtifactReader(root=root, final_path=None)
    report = reader.latest()
    assert report.report_id.startswith("synthetic:development:")
    fits_before = len(fitter.fits)
    assert reader.latest() == report
    assert len(fitter.fits) == fits_before
    final_path = tmp_path / "final.json"
    assert (
        runner.main(
            [
                "--final",
                "--dataset",
                str(tmp_path / "dataset.json"),
                "--lock-root",
                str(root),
                "--output-file",
                str(final_path),
            ],
            fitter=fitter,
        )
        == 0
    )
    final_reader = LocalBacktestArtifactReader(root=root, final_path=final_path)
    final = final_reader.latest()
    assert final.report_id.startswith("synthetic:final:")
    assert final.historical_case_ids == ("case-10", "case-11")
    final_path.unlink()
    with pytest.raises(ArtifactNotFoundError):
        final_reader.latest()  # A configured final never silently falls back to development.
    selected = root / "data/calibration/selected-config.json"
    tampered = json.loads(selected.read_text())
    tampered["artifact_set_hash"] = "a" * 64
    selected.write_text(json.dumps(tampered))
    with pytest.raises(VersionConflictError):
        reader.latest()
