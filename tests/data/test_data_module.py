import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest

from risk_engine.data.duckdb_store import DuckDbSnapshotStore, SnapshotNotFoundError
from risk_engine.data.interfaces import ProviderRequest, RawEnvelope
from risk_engine.data.module import DataModule
from risk_engine.domain import SourceItem, SourceType

RETRIEVED_AT = datetime(2026, 10, 4, 12, 30, tzinfo=timezone.utc)
PUBLISHED_AT = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


def make_request(query: str = "bank risk") -> ProviderRequest:
    return ProviderRequest(provider="gdelt", parameters={"query": query})


def make_envelope(request: ProviderRequest | None = None) -> RawEnvelope:
    effective_request = request or make_request()
    return RawEnvelope.from_response(
        provider="gdelt",
        request=effective_request,
        retrieved_at=RETRIEVED_AT,
        response_body=b'{"articles":[]}',
        response_metadata={"status_code": 200},
        source_terms_url="https://www.gdeltproject.org/about.html",
    )


def normalize_test_envelope(envelope: RawEnvelope) -> list[SourceItem]:
    snapshot_id = hashlib.sha256(envelope.canonical_manifest_bytes()).hexdigest()
    text = "Bank funding conditions tighten"
    return [
        SourceItem(
            source_item_id="gdelt-test-story",
            source_type=SourceType.NEWS,
            provider=envelope.provider,
            text=text,
            published_at=PUBLISHED_AT,
            retrieved_at=envelope.retrieved_at,
            source_reference="https://example.test/story",
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            snapshot_id=snapshot_id,
            provenance="Test provider metadata",
            license="Test fixture license",
        )
    ]


class RecordingProvider:
    def __init__(
        self,
        *,
        envelope: RawEnvelope | None = None,
        error: Exception | None = None,
    ) -> None:
        self.envelope = envelope
        self.error = error
        self.requests: list[ProviderRequest] = []

    def fetch(self, request: ProviderRequest) -> RawEnvelope:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.envelope is None:
            raise AssertionError("test provider has no envelope")
        return self.envelope


class RecordingNormalizer:
    def __init__(
        self,
        delegate: Callable[[RawEnvelope], list[SourceItem]],
    ) -> None:
        self.delegate = delegate
        self.envelopes: list[RawEnvelope] = []

    def __call__(self, envelope: RawEnvelope) -> list[SourceItem]:
        self.envelopes.append(envelope)
        return self.delegate(envelope)


def test_refresh_fetches_only_when_explicitly_called_and_persists_raw_response(
    tmp_path: Path,
) -> None:
    store = DuckDbSnapshotStore(tmp_path / "snapshots.duckdb")
    request = make_request()
    effective_request = ProviderRequest(
        provider="gdelt",
        parameters={**request.parameters, "format": "json"},
    )
    envelope = make_envelope(effective_request)
    provider = RecordingProvider(envelope=envelope)
    module = DataModule(store=store, normalizers={})

    assert provider.requests == []

    reference = module.refresh(provider, request)

    assert provider.requests == [request]
    assert store.load(reference.snapshot_id) == envelope


def test_refresh_rejects_a_response_for_an_unrelated_request(tmp_path: Path) -> None:
    store = DuckDbSnapshotStore(tmp_path / "snapshots.duckdb")
    request = make_request("requested query")
    unrelated_envelope = make_envelope(make_request("different query"))
    provider = RecordingProvider(envelope=unrelated_envelope)
    module = DataModule(store=store, normalizers={})
    unrelated_snapshot_id = hashlib.sha256(
        unrelated_envelope.canonical_manifest_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="does not preserve the explicit request"):
        module.refresh(provider, request)

    with pytest.raises(SnapshotNotFoundError):
        store.load(unrelated_snapshot_id)


def test_replay_loads_and_normalizes_an_explicit_snapshot(tmp_path: Path) -> None:
    store = DuckDbSnapshotStore(tmp_path / "snapshots.duckdb")
    envelope = make_envelope()
    reference = store.write(envelope)
    normalizer = RecordingNormalizer(normalize_test_envelope)
    module = DataModule(store=store, normalizers={"gdelt": normalizer})

    items = module.replay(reference.snapshot_id)

    assert items == normalize_test_envelope(envelope)
    assert normalizer.envelopes == [envelope]


def test_failed_refresh_preserves_an_existing_snapshot(tmp_path: Path) -> None:
    store = DuckDbSnapshotStore(tmp_path / "snapshots.duckdb")
    existing = make_envelope()
    existing_reference = store.write(existing)
    failure = RuntimeError("provider unavailable")
    provider = RecordingProvider(error=failure)
    module = DataModule(store=store, normalizers={})

    with pytest.raises(RuntimeError, match="provider unavailable"):
        module.refresh(provider, make_request("new query"))

    assert store.load(existing_reference.snapshot_id) == existing


def test_replay_never_fetches_from_the_provider(tmp_path: Path) -> None:
    store = DuckDbSnapshotStore(tmp_path / "snapshots.duckdb")
    request = make_request()
    provider = RecordingProvider(envelope=make_envelope(request))
    module = DataModule(
        store=store,
        normalizers={"gdelt": normalize_test_envelope},
    )
    reference = module.refresh(provider, request)

    provider.error = AssertionError("replay contacted the provider")
    items = module.replay(reference.snapshot_id)

    assert len(items) == 1
    assert provider.requests == [request]
