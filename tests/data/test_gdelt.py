import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from risk_engine.data.gdelt import (
    GdeltProvider,
    GdeltRateLimitError,
    GdeltRequestError,
    GdeltSchemaError,
    GdeltTransportError,
    normalize_gdelt,
)
from risk_engine.data.interfaces import ProviderRequest, RawEnvelope
from risk_engine.domain import SourceType

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "gdelt-response.json"
RETRIEVED_AT = datetime(2026, 10, 4, 12, 30, tzinfo=timezone.utc)
ARTICLE_TITLE = "Reserve Bank reviews liquidity conditions after bank funding pressure"


def make_request() -> ProviderRequest:
    return ProviderRequest(
        provider="gdelt",
        parameters={
            "query": "India bank liquidity",
            "maxrecords": 10,
            "timespan": "1d",
        },
    )


def make_envelope(response_body: bytes) -> RawEnvelope:
    return RawEnvelope.from_response(
        provider="gdelt",
        request=make_request(),
        retrieved_at=RETRIEVED_AT,
        response_body=response_body,
        response_metadata={"status_code": 200, "content_type": "application/json"},
        source_terms_url="https://www.gdeltproject.org/about.html",
    )


def test_fetch_uses_canonical_doc_api_parameters_and_preserves_response() -> None:
    fixture_bytes = FIXTURE_PATH.read_bytes()

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v2/doc/doc"
        assert dict(request.url.params) == {
            "format": "json",
            "maxrecords": "10",
            "mode": "ArtList",
            "query": "India bank liquidity",
            "sort": "HybridRel",
            "timespan": "1d",
        }
        return httpx.Response(
            200,
            content=fixture_bytes,
            headers={"content-type": "application/json"},
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    envelope = GdeltProvider(client=client, clock=lambda: RETRIEVED_AT).fetch(
        make_request()
    )

    assert envelope.response_body == fixture_bytes
    assert envelope.retrieved_at == RETRIEVED_AT
    assert envelope.response_metadata == {
        "content_type": "application/json",
        "status_code": 200,
    }
    assert envelope.request.parameters == {
        "format": "json",
        "maxrecords": 10,
        "mode": "ArtList",
        "query": "India bank liquidity",
        "sort": "HybridRel",
        "timespan": "1d",
    }
    assert envelope.content_hash == hashlib.sha256(fixture_bytes).hexdigest()


def test_fetch_rejects_conflicting_reserved_parameters_before_network_access() -> None:
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"unexpected network request: {request.url}")

    request = ProviderRequest(
        provider="gdelt",
        parameters={"query": "India bank liquidity", "sort": "DateDesc"},
    )
    client = httpx.Client(transport=httpx.MockTransport(unexpected_request))

    with pytest.raises(GdeltRequestError, match="sort"):
        GdeltProvider(client=client, clock=lambda: RETRIEVED_AT).fetch(request)


def test_normalize_gdelt_produces_canonical_news_with_provenance() -> None:
    envelope = make_envelope(FIXTURE_PATH.read_bytes())

    first = normalize_gdelt(envelope)
    replayed = normalize_gdelt(envelope)

    assert first == replayed
    assert len(first) == 1
    item = first[0]
    assert item.source_type is SourceType.NEWS
    assert item.provider == "gdelt"
    assert item.text == ARTICLE_TITLE
    assert item.published_at == datetime(2026, 10, 4, 12, 15, tzinfo=timezone.utc)
    assert item.published_at.tzinfo is timezone.utc
    assert item.retrieved_at == RETRIEVED_AT
    assert item.source_reference == (
        "https://publisher.example/markets/rbi-liquidity-review"
    )
    assert item.content_hash == hashlib.sha256(ARTICLE_TITLE.encode("utf-8")).hexdigest()
    assert item.source_item_id == (
        "gdelt-"
        + hashlib.sha256(item.source_reference.encode("utf-8")).hexdigest()
    )
    assert item.snapshot_id == hashlib.sha256(
        envelope.canonical_manifest_bytes()
    ).hexdigest()
    assert "publisher.example" in item.provenance
    assert "GDELT DOC API" in item.provenance
    assert "https://www.gdeltproject.org/about.html" in item.provenance
    assert "originating publisher" in item.license


def test_fetch_reports_rate_limit_with_retry_metadata() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "60"})

    client = httpx.Client(transport=httpx.MockTransport(respond))

    with pytest.raises(GdeltRateLimitError, match="60"):
        GdeltProvider(client=client, clock=lambda: RETRIEVED_AT).fetch(make_request())


def test_fetch_wraps_http_timeout_without_returning_partial_data() -> None:
    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("GDELT did not respond", request=request)

    client = httpx.Client(transport=httpx.MockTransport(time_out))

    with pytest.raises(GdeltTransportError, match="timed out") as raised:
        GdeltProvider(client=client, clock=lambda: RETRIEVED_AT).fetch(make_request())

    assert isinstance(raised.value.__cause__, httpx.ReadTimeout)


@pytest.mark.parametrize(
    "response_body",
    [
        b"not-json",
        b'{"unexpected":[]}',
        (
            b'{"articles":[{"url":"https://publisher.example/story",'
            b'"seendate":"20261004T121500Z"}]}'
        ),
    ],
    ids=["invalid-json", "missing-articles", "incomplete-article"],
)
def test_normalize_gdelt_rejects_provider_schema_errors(response_body: bytes) -> None:
    with pytest.raises(GdeltSchemaError):
        normalize_gdelt(make_envelope(response_body))


@pytest.mark.parametrize(
    "seen_date",
    ["not-a-date", "20261004T124500Z"],
    ids=["invalid-date", "publication-after-retrieval"],
)
def test_normalize_gdelt_wraps_article_timestamp_errors(seen_date: str) -> None:
    response_body = json.dumps(
        {
            "articles": [
                {
                    "url": "https://publisher.example/story",
                    "title": "A complete article title",
                    "seendate": seen_date,
                    "domain": "publisher.example",
                    "language": "English",
                    "sourcecountry": "India",
                }
            ]
        }
    ).encode("utf-8")

    with pytest.raises(GdeltSchemaError, match="article 0"):
        normalize_gdelt(make_envelope(response_body))


def test_normalize_gdelt_identifies_article_with_invalid_provider_fields() -> None:
    response_body = json.dumps(
        {
            "articles": [
                {
                    "url": "https://publisher.example/story",
                    "title": "A complete article title",
                    "seendate": "20261004T121500Z",
                    "domain": " ",
                    "language": "English",
                    "sourcecountry": "India",
                }
            ]
        }
    ).encode("utf-8")

    with pytest.raises(GdeltSchemaError, match="article 0"):
        normalize_gdelt(make_envelope(response_body))
