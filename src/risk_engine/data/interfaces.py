"""Provider and immutable-snapshot interfaces for source ingestion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Protocol, TypeAlias

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
MetadataKey = Annotated[str, StringConstraints(min_length=1)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RequestScalar: TypeAlias = str | int | float | bool | None


def _require_exact_mapping_keys(value: object) -> object:
    if isinstance(value, Mapping):
        for key in value:
            if isinstance(key, str) and key != key.strip():
                raise ValueError("mapping keys cannot contain surrounding whitespace")
    return value


def _freeze_mapping(
    value: Mapping[str, RequestScalar],
) -> Mapping[str, RequestScalar]:
    return MappingProxyType(dict(value))


class SnapshotModel(BaseModel):
    """Strict immutable base for provider and snapshot records."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)


class ProviderRequest(SnapshotModel):
    """A provider request with deterministic serialization."""

    provider: NonEmptyString
    parameters: Mapping[MetadataKey, RequestScalar]

    _validate_parameter_keys = field_validator("parameters", mode="before")(
        _require_exact_mapping_keys
    )
    _freeze_parameters = field_validator("parameters")(_freeze_mapping)

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {"parameters": dict(self.parameters), "provider": self.provider},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


class RawEnvelope(SnapshotModel):
    """The exact provider response and the metadata needed to replay it."""

    provider: NonEmptyString
    request: ProviderRequest
    retrieved_at: AwareDatetime
    response_body: bytes
    response_metadata: Mapping[MetadataKey, RequestScalar]
    source_terms_url: NonEmptyString
    content_hash: Sha256Hex
    schema_version: NonEmptyString = "1.0.0"

    _validate_response_metadata_keys = field_validator(
        "response_metadata", mode="before"
    )(_require_exact_mapping_keys)
    _freeze_response_metadata = field_validator("response_metadata")(_freeze_mapping)

    @classmethod
    def from_response(
        cls,
        *,
        provider: str,
        request: ProviderRequest,
        retrieved_at: datetime,
        response_body: bytes,
        response_metadata: Mapping[str, RequestScalar],
        source_terms_url: str,
        schema_version: str = "1.0.0",
    ) -> RawEnvelope:
        return cls(
            provider=provider,
            request=request,
            retrieved_at=retrieved_at,
            response_body=response_body,
            response_metadata=response_metadata,
            source_terms_url=source_terms_url,
            content_hash=hashlib.sha256(response_body).hexdigest(),
            schema_version=schema_version,
        )

    @model_validator(mode="after")
    def provider_and_hash_match_payload(self) -> RawEnvelope:
        if self.provider != self.request.provider:
            raise ValueError("provider must match request.provider")
        expected_hash = hashlib.sha256(self.response_body).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError("content_hash must match response_body SHA-256")
        return self

    def canonical_manifest_bytes(self) -> bytes:
        manifest = {
            "content_hash": self.content_hash,
            "provider": self.provider,
            "request": json.loads(self.request.canonical_bytes()),
            "response_metadata": dict(self.response_metadata),
            "retrieved_at": self.retrieved_at.isoformat(),
            "schema_version": self.schema_version,
            "source_terms_url": self.source_terms_url,
        }
        return json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


class SnapshotRef(SnapshotModel):
    snapshot_id: Sha256Hex
    provider: NonEmptyString
    retrieved_at: AwareDatetime
    content_hash: Sha256Hex
    manifest_hash: Sha256Hex


class SourceProvider(Protocol):
    def fetch(self, request: ProviderRequest) -> RawEnvelope: ...


__all__ = [
    "ProviderRequest",
    "RawEnvelope",
    "SnapshotRef",
    "SourceProvider",
]
