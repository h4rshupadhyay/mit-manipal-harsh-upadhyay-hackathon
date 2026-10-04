"""Explicit GDELT DOC API refresh and pure snapshot normalization."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Annotated

import httpx
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from risk_engine.data.interfaces import ProviderRequest, RawEnvelope, RequestScalar
from risk_engine.domain import SourceItem, SourceType

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_TERMS_URL = "https://www.gdeltproject.org/about.html"
_PROVIDER = "gdelt"
_NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_REQUIRED_PARAMETERS: dict[str, RequestScalar] = {
    "format": "json",
    "mode": "ArtList",
    "sort": "HybridRel",
}


class GdeltError(RuntimeError):
    """Base error for the GDELT adapter."""


class GdeltRequestError(GdeltError):
    """Raised when a request does not satisfy the GDELT adapter contract."""


class GdeltRateLimitError(GdeltError):
    """Raised when GDELT rejects a refresh because of rate limiting."""


class GdeltTransportError(GdeltError):
    """Raised when a refresh cannot obtain a complete HTTP response."""


class GdeltResponseError(GdeltError):
    """Raised for non-success responses other than rate limits."""


class GdeltSchemaError(GdeltError):
    """Raised when a stored response no longer matches the expected schema."""


class _GdeltArticle(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    url: _NonEmptyString
    title: _NonEmptyString
    seendate: _NonEmptyString
    domain: _NonEmptyString
    language: _NonEmptyString
    sourcecountry: _NonEmptyString


class _GdeltDocumentResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    articles: tuple[object, ...]


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)  # noqa: UP017 -- test runtime is Python 3.10


class GdeltProvider:
    """Fetch raw GDELT responses only when explicitly called by a refresh path."""

    def __init__(
        self,
        *,
        client: httpx.Client,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.client = client
        self.clock = clock

    def fetch(self, request: ProviderRequest) -> RawEnvelope:
        if request.provider != _PROVIDER:
            raise GdeltRequestError("GDELT requests must use provider='gdelt'")
        query = request.parameters.get("query")
        if not isinstance(query, str) or not query.strip():
            raise GdeltRequestError("GDELT requests require a non-empty query")

        for name, required_value in _REQUIRED_PARAMETERS.items():
            supplied_value = request.parameters.get(name)
            if name in request.parameters and supplied_value != required_value:
                raise GdeltRequestError(
                    f"GDELT parameter {name!r} must be {required_value!r}"
                )
        parameters: dict[str, RequestScalar] = dict(request.parameters)
        parameters.update(_REQUIRED_PARAMETERS)
        effective_request = ProviderRequest(provider=_PROVIDER, parameters=parameters)
        try:
            response = self.client.get(GDELT_DOC_ENDPOINT, params=parameters)
        except httpx.TimeoutException as error:
            raise GdeltTransportError("GDELT refresh timed out") from error
        except httpx.RequestError as error:
            raise GdeltTransportError("GDELT refresh failed") from error

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            retry_after = response.headers.get("retry-after", "unspecified")
            raise GdeltRateLimitError(
                f"GDELT rate limit reached; retry-after={retry_after}"
            )
        if not response.is_success:
            raise GdeltResponseError(
                f"GDELT returned HTTP {response.status_code}"
            )

        return RawEnvelope.from_response(
            provider=_PROVIDER,
            request=effective_request,
            retrieved_at=self.clock(),
            response_body=response.content,
            response_metadata={
                "content_type": response.headers.get("content-type", ""),
                "status_code": response.status_code,
            },
            source_terms_url=GDELT_TERMS_URL,
        )


def _parse_seen_at(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S%z")
    except ValueError as error:
        raise GdeltSchemaError(f"invalid GDELT seendate: {value}") from error


def normalize_gdelt(envelope: RawEnvelope) -> list[SourceItem]:
    """Normalize a stored response without accessing the network."""

    if envelope.provider != _PROVIDER:
        raise GdeltSchemaError("cannot normalize a non-GDELT envelope")
    try:
        document = _GdeltDocumentResponse.model_validate_json(envelope.response_body)
    except (ValidationError, ValueError) as error:
        raise GdeltSchemaError("GDELT response does not match the article schema") from error

    snapshot_id = hashlib.sha256(envelope.canonical_manifest_bytes()).hexdigest()
    normalized: list[SourceItem] = []
    for index, raw_article in enumerate(document.articles):
        try:
            article = _GdeltArticle.model_validate(raw_article)
            published_at = _parse_seen_at(article.seendate)
            source_item_id = "gdelt-" + hashlib.sha256(
                article.url.encode("utf-8")
            ).hexdigest()
            content_hash = hashlib.sha256(article.title.encode("utf-8")).hexdigest()
            item = SourceItem(
                source_item_id=source_item_id,
                source_type=SourceType.NEWS,
                provider=_PROVIDER,
                text=article.title,
                published_at=published_at,
                retrieved_at=envelope.retrieved_at,
                source_reference=article.url,
                content_hash=content_hash,
                snapshot_id=snapshot_id,
                provenance=(
                    f"GDELT DOC API metadata (terms: {envelope.source_terms_url}); "
                    f"publisher={article.domain}; language={article.language}; "
                    f"source_country={article.sourcecountry}"
                ),
                license=(
                    "Article rights remain with the originating publisher; "
                    "GDELT metadata terms apply."
                ),
            )
        except (GdeltSchemaError, ValidationError) as error:
            raise GdeltSchemaError(
                f"GDELT article {index} violates the normalized schema"
            ) from error
        normalized.append(item)
    return normalized


__all__ = [
    "GdeltError",
    "GdeltProvider",
    "GdeltRateLimitError",
    "GdeltRequestError",
    "GdeltResponseError",
    "GdeltSchemaError",
    "GdeltTransportError",
    "normalize_gdelt",
]
