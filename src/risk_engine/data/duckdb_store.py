"""Append-only DuckDB persistence for exact provider response envelopes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
from pydantic import ValidationError

from risk_engine.data.interfaces import ProviderRequest, RawEnvelope, SnapshotRef


class SnapshotStoreError(RuntimeError):
    """Base error for immutable snapshot persistence."""


class SnapshotAlreadyExistsError(SnapshotStoreError):
    """Raised when a caller attempts to replace an existing snapshot."""


class SnapshotNotFoundError(SnapshotStoreError):
    """Raised when a requested snapshot does not exist."""


class SnapshotManifestError(SnapshotStoreError):
    """Raised when stored snapshot metadata is incomplete or inconsistent."""


class DuckDbSnapshotStore:
    """Persist raw response bytes alongside a validated canonical manifest."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS source_snapshots (
                    snapshot_id VARCHAR PRIMARY KEY,
                    manifest_json VARCHAR NOT NULL,
                    response_body BLOB NOT NULL
                )
                """
            )

    def write(self, envelope: RawEnvelope) -> SnapshotRef:
        manifest_bytes = envelope.canonical_manifest_bytes()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        snapshot_id = manifest_hash
        try:
            with duckdb.connect(str(self.database_path)) as connection:
                connection.execute(
                    """
                    INSERT INTO source_snapshots (snapshot_id, manifest_json, response_body)
                    VALUES (?, ?, ?)
                    """,
                    [snapshot_id, manifest_bytes.decode("utf-8"), envelope.response_body],
                )
        except duckdb.ConstraintException as error:
            raise SnapshotAlreadyExistsError(
                f"snapshot {snapshot_id} already exists"
            ) from error
        return SnapshotRef(
            snapshot_id=snapshot_id,
            provider=envelope.provider,
            retrieved_at=envelope.retrieved_at,
            content_hash=envelope.content_hash,
            manifest_hash=manifest_hash,
        )

    def load(self, snapshot_id: str) -> RawEnvelope:
        with duckdb.connect(str(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT manifest_json, response_body
                FROM source_snapshots
                WHERE snapshot_id = ?
                """,
                [snapshot_id],
            ).fetchone()
        if row is None:
            raise SnapshotNotFoundError(f"snapshot {snapshot_id} does not exist")

        manifest_json, response_body = row
        try:
            manifest = json.loads(manifest_json)
            request = ProviderRequest.model_validate(manifest["request"])
            envelope = RawEnvelope.model_validate(
                {
                    **manifest,
                    "request": request,
                    "response_body": bytes(response_body),
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as error:
            raise SnapshotManifestError(
                f"snapshot {snapshot_id} has an invalid manifest"
            ) from error

        expected_id = hashlib.sha256(envelope.canonical_manifest_bytes()).hexdigest()
        if expected_id != snapshot_id:
            raise SnapshotManifestError(
                f"snapshot {snapshot_id} manifest hash does not match its identifier"
            )
        return envelope


__all__ = [
    "DuckDbSnapshotStore",
    "SnapshotAlreadyExistsError",
    "SnapshotManifestError",
    "SnapshotNotFoundError",
    "SnapshotStoreError",
]
