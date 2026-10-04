import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from risk_engine.data.gdelt import normalize_gdelt
from risk_engine.data.interfaces import ProviderRequest, RawEnvelope
from risk_engine.data.social_csv import (
    SocialCsvEncodingError,
    SocialCsvProvider,
    SocialCsvRequestError,
    SocialCsvSchemaError,
    normalize_social_csv,
)
from risk_engine.domain import SourceItem, SourceType

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures"
FIXTURE_PATH = FIXTURE_ROOT / "social.csv"
RETRIEVED_AT = datetime(2026, 10, 4, 12, 30, tzinfo=timezone.utc)
PROJECT_TERMS_URL = (
    "https://github.com/h4rshupadhyay/"
    "mit-manipal-harsh-upadhyay-hackathon/blob/main/LICENSE"
)


def make_request(path: Path = FIXTURE_PATH) -> ProviderRequest:
    return ProviderRequest(
        provider="social_csv",
        parameters={
            "license": "MIT",
            "path": str(path),
            "provenance": "project-authored synthetic demonstration post",
            "source_terms_url": PROJECT_TERMS_URL,
        },
    )


def make_envelope(response_body: bytes) -> RawEnvelope:
    return RawEnvelope.from_response(
        provider="social_csv",
        request=ProviderRequest(
            provider="social_csv",
            parameters={
                "license": "MIT",
                "path": "in-memory-social.csv",
                "provenance": "project-authored synthetic demonstration post",
                "source_terms_url": PROJECT_TERMS_URL,
            },
        ),
        retrieved_at=RETRIEVED_AT,
        response_body=response_body,
        response_metadata={"encoding": "utf-8", "file_size": len(response_body)},
        source_terms_url=PROJECT_TERMS_URL,
    )


def test_fetch_reads_only_the_requested_local_csv_bytes() -> None:
    fixture_bytes = FIXTURE_PATH.read_bytes()

    envelope = SocialCsvProvider(clock=lambda: RETRIEVED_AT).fetch(make_request())

    assert envelope.response_body == fixture_bytes
    assert envelope.request == make_request()
    assert envelope.retrieved_at == RETRIEVED_AT
    assert envelope.response_metadata == {
        "encoding": "utf-8",
        "file_size": len(fixture_bytes),
    }
    assert envelope.source_terms_url == PROJECT_TERMS_URL
    assert envelope.content_hash == hashlib.sha256(fixture_bytes).hexdigest()


@pytest.mark.parametrize(
    "path",
    [
        "https://example.test/restricted-social.csv",
        "//fileserver.example/restricted/social.csv",
    ],
    ids=["https-url", "network-authority"],
)
def test_fetch_rejects_network_paths_without_downloading(path: str) -> None:
    request = ProviderRequest(
        provider="social_csv",
        parameters={"path": path},
    )

    with pytest.raises(SocialCsvRequestError, match="local path"):
        SocialCsvProvider(clock=lambda: RETRIEVED_AT).fetch(request)


def test_normalize_social_csv_preserves_unicode_provenance_and_stable_hashes() -> None:
    envelope = make_envelope(FIXTURE_PATH.read_bytes())

    first = normalize_social_csv(envelope)
    replayed = normalize_social_csv(envelope)

    assert first == replayed
    assert len(first) == 2
    india_post = first[0]
    assert india_post.source_item_id == "social-social-india-001"
    assert india_post.source_type is SourceType.SOCIAL
    assert india_post.provider == "social_csv"
    assert india_post.text == (
        "भारतीय बैंक ने तरलता दबाव के बाद फंडिंग योजना की समीक्षा की"
    )
    assert india_post.published_at == datetime(
        2026, 10, 4, 11, 45, tzinfo=timezone.utc
    )
    assert india_post.published_at.tzinfo is not None
    assert india_post.published_at.utcoffset() == timedelta(0)
    assert india_post.retrieved_at == RETRIEVED_AT
    assert india_post.source_reference == "project://social/social-india-001"
    assert india_post.content_hash == hashlib.sha256(
        india_post.text.encode("utf-8")
    ).hexdigest()
    assert india_post.snapshot_id == hashlib.sha256(
        envelope.canonical_manifest_bytes()
    ).hexdigest()
    assert india_post.provenance == "project-authored synthetic demonstration post"
    assert india_post.license == "MIT"


@pytest.mark.parametrize(
    "response_body, message",
    [
        (
            b"published_at,text,source_reference,provenance,license\n"
            b"2026-10-04T12:00:00+00:00,text,project://row,project-authored,MIT\n",
            "source_row_id",
        ),
        (
            b"source_row_id,published_at,text,source_reference,provenance,license\n"
            b",2026-10-04T12:00:00+00:00,text,project://row,project-authored,MIT\n",
            "row 2",
        ),
        (
            b"source_row_id,published_at,text,source_reference,provenance,license\n"
            b"row-1,not-a-date,text,project://row,project-authored,MIT\n",
            "row 2",
        ),
    ],
    ids=["missing-header", "missing-row-id", "invalid-timestamp"],
)
def test_normalize_social_csv_rejects_malformed_rows(
    response_body: bytes,
    message: str,
) -> None:
    with pytest.raises(SocialCsvSchemaError, match=message):
        normalize_social_csv(make_envelope(response_body))


@pytest.mark.parametrize(
    "header, message",
    [
        (
            "source_row_id,source_row_id,published_at,text,source_reference,"
            "provenance,license\n",
            "duplicate.*source_row_id",
        ),
        (
            "source_row_id,published_at,text,source_reference,provenance,license,"
            "unexpected\n",
            "unexpected.*unexpected",
        ),
    ],
    ids=["duplicate-column", "unexpected-column"],
)
def test_normalize_social_csv_rejects_ambiguous_headers(
    header: str,
    message: str,
) -> None:
    with pytest.raises(SocialCsvSchemaError, match=message):
        normalize_social_csv(make_envelope(header.encode("utf-8")))


def test_normalize_social_csv_reports_physical_line_after_blank_lines() -> None:
    response_body = (
        b"source_row_id,published_at,text,source_reference,provenance,license\n"
        b"\n"
        b",2026-10-04T12:00:00+00:00,text,project://row,"
        b"project-authored synthetic demonstration post,MIT\n"
    )

    with pytest.raises(SocialCsvSchemaError, match="row 3"):
        normalize_social_csv(make_envelope(response_body))


def test_normalize_social_csv_reports_line_for_malformed_quoting() -> None:
    response_body = (
        b"source_row_id,published_at,text,source_reference,provenance,license\n"
        b'row-1,2026-10-04T12:00:00+00:00,"unterminated\n'
    )

    with pytest.raises(SocialCsvSchemaError, match="line 2"):
        normalize_social_csv(make_envelope(response_body))


def test_normalize_social_csv_wraps_malformed_header_quoting() -> None:
    response_body = b'source_row_id,published_at,"unterminated\n'

    with pytest.raises(SocialCsvSchemaError, match="line 1"):
        normalize_social_csv(make_envelope(response_body))


def test_normalize_social_csv_reports_current_line_for_mid_file_parser_error() -> None:
    response_body = (
        b"source_row_id,published_at,text,source_reference,provenance,license\n"
        b'row-1,2026-10-04T11:00:00+00:00,"bad quote"x,project://1,'
        b"project-authored synthetic demonstration post,MIT\n"
        b"row-2,2026-10-04T12:00:00+00:00,valid,project://2,"
        b"project-authored synthetic demonstration post,MIT\n"
    )

    with pytest.raises(SocialCsvSchemaError, match="line 2"):
        normalize_social_csv(make_envelope(response_body))


def test_normalize_social_csv_requires_unique_source_row_ids() -> None:
    response_body = (
        b"source_row_id,published_at,text,source_reference,provenance,license\n"
        b"row-1,2026-10-04T11:00:00+00:00,first,project://1,"
        b"project-authored synthetic demonstration post,MIT\n"
        b"row-1,2026-10-04T12:00:00+00:00,second,project://2,"
        b"project-authored synthetic demonstration post,MIT\n"
    )

    with pytest.raises(SocialCsvSchemaError, match="duplicate source_row_id.*row-1"):
        normalize_social_csv(make_envelope(response_body))


def test_normalize_social_csv_rejects_terms_that_disagree_with_request() -> None:
    response_body = (
        b"source_row_id,published_at,text,source_reference,provenance,license\n"
        b"row-1,2026-10-04T11:00:00+00:00,text,project://1,"
        b"project-authored synthetic demonstration post,CC-BY\n"
    )

    with pytest.raises(SocialCsvSchemaError, match="license"):
        normalize_social_csv(make_envelope(response_body))


def test_normalize_social_csv_rejects_non_utf8_bytes() -> None:
    with pytest.raises(SocialCsvEncodingError, match="UTF-8"):
        normalize_social_csv(make_envelope(b"source_row_id,text\nrow-1,\xff\n"))


def test_news_and_social_normalizers_emit_the_same_source_item_contract() -> None:
    social_item = normalize_social_csv(make_envelope(FIXTURE_PATH.read_bytes()))[0]
    gdelt_bytes = (FIXTURE_ROOT / "gdelt-response.json").read_bytes()
    gdelt_envelope = RawEnvelope.from_response(
        provider="gdelt",
        request=ProviderRequest(provider="gdelt", parameters={"query": "bank"}),
        retrieved_at=RETRIEVED_AT,
        response_body=gdelt_bytes,
        response_metadata={"status_code": 200},
        source_terms_url="https://www.gdeltproject.org/about.html",
    )
    news_item = normalize_gdelt(gdelt_envelope)[0]

    assert type(social_item) is type(news_item) is SourceItem
    assert social_item.model_fields.keys() == news_item.model_fields.keys()
