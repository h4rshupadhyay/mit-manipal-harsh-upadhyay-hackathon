import operator
from collections.abc import MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import duckdb
import pytest
from pydantic import ValidationError

from risk_engine.data.duckdb_store import (
    DuckDbSnapshotStore,
    SnapshotAlreadyExistsError,
    SnapshotManifestError,
)
from risk_engine.data.interfaces import ProviderRequest, RawEnvelope

RETRIEVED_AT = datetime(2026, 10, 4, 12, 30, tzinfo=timezone.utc)
RESPONSE_BODY = b'{"articles":[]}'
RESPONSE_SHA256 = "35c74c2e22b3ccd222b09b8b63f16f3c9c1312e50fbdaff47ef8af563af95791"


def make_envelope() -> RawEnvelope:
    return RawEnvelope.from_response(
        provider="gdelt",
        request=ProviderRequest(
            provider="gdelt",
            parameters={"query": "bank risk", "max_records": 10, "format": "json"},
        ),
        retrieved_at=RETRIEVED_AT,
        response_body=RESPONSE_BODY,
        response_metadata={"status_code": 200, "content_type": "application/json"},
        source_terms_url="https://www.gdeltproject.org/about.html",
    )


def test_provider_request_serializes_parameters_canonically() -> None:
    first = ProviderRequest(
        provider="gdelt",
        parameters={"query": "bank risk", "max_records": 10, "format": "json"},
    )
    reordered = ProviderRequest(
        provider="gdelt",
        parameters={"format": "json", "query": "bank risk", "max_records": 10},
    )

    assert first.canonical_bytes() == (
        b'{"parameters":{"format":"json","max_records":10,"query":"bank risk"},'
        b'"provider":"gdelt"}'
    )
    assert reordered.canonical_bytes() == first.canonical_bytes()


def test_request_and_response_metadata_are_deeply_immutable() -> None:
    request = ProviderRequest(provider="gdelt", parameters={"query": "bank risk"})
    envelope = make_envelope()

    with pytest.raises(TypeError):
        operator.setitem(
            cast(MutableMapping[str, object], request.parameters),
            "query",
            "mutated",
        )
    with pytest.raises(TypeError):
        operator.setitem(
            cast(MutableMapping[str, object], envelope.response_metadata),
            "status_code",
            500,
        )

    assert request.canonical_bytes() == (
        b'{"parameters":{"query":"bank risk"},"provider":"gdelt"}'
    )
    assert envelope.response_metadata["status_code"] == 200


def test_metadata_keys_must_not_require_whitespace_normalization() -> None:
    with pytest.raises(ValidationError, match="whitespace"):
        ProviderRequest(
            provider="gdelt",
            parameters={"q": "first", " q ": "second"},
        )

    with pytest.raises(ValidationError, match="whitespace"):
        RawEnvelope.from_response(
            provider="gdelt",
            request=ProviderRequest(provider="gdelt", parameters={"q": "risk"}),
            retrieved_at=RETRIEVED_AT,
            response_body=RESPONSE_BODY,
            response_metadata={"status_code": 200, " status_code ": 500},
            source_terms_url="https://www.gdeltproject.org/about.html",
        )


def test_raw_envelope_derives_and_validates_response_sha256() -> None:
    envelope = make_envelope()

    assert envelope.content_hash == RESPONSE_SHA256

    with pytest.raises(ValidationError, match="content_hash"):
        RawEnvelope(
            provider=envelope.provider,
            request=envelope.request,
            retrieved_at=envelope.retrieved_at,
            response_body=envelope.response_body,
            response_metadata=envelope.response_metadata,
            source_terms_url=envelope.source_terms_url,
            content_hash="0" * 64,
            schema_version=envelope.schema_version,
        )


def test_store_replays_exact_bytes_after_reopening(tmp_path: Path) -> None:
    database_path = tmp_path / "snapshots.duckdb"
    envelope = make_envelope()
    reference = DuckDbSnapshotStore(database_path).write(envelope)

    replayed = DuckDbSnapshotStore(database_path).load(reference.snapshot_id)

    assert replayed.response_body == RESPONSE_BODY
    assert replayed == envelope
    assert reference.provider == "gdelt"
    assert reference.content_hash == RESPONSE_SHA256
    assert len(reference.manifest_hash) == 64


def test_store_is_append_only_for_an_existing_snapshot(tmp_path: Path) -> None:
    store = DuckDbSnapshotStore(tmp_path / "snapshots.duckdb")
    envelope = make_envelope()
    reference = store.write(envelope)

    with pytest.raises(SnapshotAlreadyExistsError, match=reference.snapshot_id):
        store.write(envelope)

    assert store.load(reference.snapshot_id) == envelope


def test_load_rejects_an_incomplete_manifest(tmp_path: Path) -> None:
    database_path = tmp_path / "snapshots.duckdb"
    DuckDbSnapshotStore(database_path)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            INSERT INTO source_snapshots (snapshot_id, manifest_json, response_body)
            VALUES (?, ?, ?)
            """,
            ["broken-snapshot", '{"schema_version":"1.0.0"}', b"payload"],
        )

    with pytest.raises(SnapshotManifestError, match="broken-snapshot"):
        DuckDbSnapshotStore(database_path).load("broken-snapshot")
