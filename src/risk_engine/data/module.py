"""Explicit source refresh and network-free snapshot replay orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from risk_engine.data.duckdb_store import DuckDbSnapshotStore
from risk_engine.data.interfaces import (
    ProviderRequest,
    RawEnvelope,
    SnapshotRef,
    SourceProvider,
)
from risk_engine.domain import SourceItem

Normalizer = Callable[[RawEnvelope], list[SourceItem]]


class NormalizerNotFoundError(RuntimeError):
    """Raised when replay has no normalizer for a stored provider envelope."""


class ProviderResponseMismatchError(ValueError):
    """Raised when a provider response changes the caller's explicit request."""


class DataModule:
    """Coordinate explicit acquisition and pure replay through stored envelopes."""

    def __init__(
        self,
        *,
        store: DuckDbSnapshotStore,
        normalizers: Mapping[str, Normalizer],
    ) -> None:
        self._store = store
        self._normalizers = dict(normalizers)

    def refresh(
        self,
        provider: SourceProvider,
        request: ProviderRequest,
    ) -> SnapshotRef:
        envelope = provider.fetch(request)
        response_request = envelope.request
        preserves_request = response_request.provider == request.provider and all(
            name in response_request.parameters
            and response_request.parameters[name] == value
            for name, value in request.parameters.items()
        )
        if not preserves_request:
            raise ProviderResponseMismatchError(
                "provider response does not preserve the explicit request"
            )
        return self._store.write(envelope)

    def replay(self, snapshot_id: str) -> list[SourceItem]:
        envelope = self._store.load(snapshot_id)
        try:
            normalizer = self._normalizers[envelope.provider]
        except KeyError as error:
            raise NormalizerNotFoundError(
                f"no normalizer registered for provider {envelope.provider!r}"
            ) from error
        return normalizer(envelope)


__all__ = [
    "DataModule",
    "Normalizer",
    "NormalizerNotFoundError",
    "ProviderResponseMismatchError",
]
