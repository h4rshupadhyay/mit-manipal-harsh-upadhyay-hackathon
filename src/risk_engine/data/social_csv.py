"""Local-only replay of licensed historical social text from UTF-8 CSV."""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
)

from risk_engine.data.interfaces import ProviderRequest, RawEnvelope
from risk_engine.domain import SourceItem, SourceType

PROJECT_LICENSE_URL = (
    "https://github.com/h4rshupadhyay/"
    "mit-manipal-harsh-upadhyay-hackathon/blob/main/LICENSE"
)
_PROVIDER = "social_csv"
_NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_REQUIRED_COLUMNS = {
    "license",
    "provenance",
    "published_at",
    "source_reference",
    "source_row_id",
    "text",
}


class SocialCsvError(RuntimeError):
    """Base error for historical social CSV replay."""


class SocialCsvRequestError(SocialCsvError):
    """Raised when a replay request is not a valid local-file request."""


class SocialCsvFileError(SocialCsvError):
    """Raised when the requested local file cannot be read."""


class SocialCsvEncodingError(SocialCsvError):
    """Raised when a replay file is not valid UTF-8."""


class SocialCsvSchemaError(SocialCsvError):
    """Raised when a replay file violates the social-row contract."""


class _SocialRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_row_id: _NonEmptyString
    published_at: AwareDatetime
    text: _NonEmptyString
    source_reference: _NonEmptyString
    provenance: _NonEmptyString
    license: _NonEmptyString


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)  # noqa: UP017 -- test runtime is Python 3.10


def _required_request_text(request: ProviderRequest, name: str) -> str:
    value = request.parameters.get(name)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SocialCsvRequestError(
            f"social CSV requests require canonical {name!r} metadata"
        )
    return value


def _csv_error_line(reader: csv.DictReader[str], text: str, error: csv.Error) -> int:
    if "unexpected end of data" in str(error):
        return max(reader.line_num, len(text.splitlines()))
    return reader.line_num + 1


class SocialCsvProvider:
    """Read an explicitly named local CSV into an immutable raw envelope."""

    def __init__(self, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self.clock = clock

    def fetch(self, request: ProviderRequest) -> RawEnvelope:
        if request.provider != _PROVIDER:
            raise SocialCsvRequestError(
                "social CSV requests must use provider='social_csv'"
            )
        path_value = request.parameters.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            raise SocialCsvRequestError("social CSV requests require a local path")
        parsed_path = urlsplit(path_value)
        if parsed_path.scheme or parsed_path.netloc:
            raise SocialCsvRequestError("social CSV path must be a local path, not a URL")
        source_terms_url = _required_request_text(request, "source_terms_url")
        _required_request_text(request, "provenance")
        _required_request_text(request, "license")

        path = Path(path_value)
        try:
            response_body = path.read_bytes()
        except OSError as error:
            raise SocialCsvFileError(f"cannot read social CSV: {path}") from error

        return RawEnvelope.from_response(
            provider=_PROVIDER,
            request=request,
            retrieved_at=self.clock(),
            response_body=response_body,
            response_metadata={
                "encoding": "utf-8",
                "file_size": len(response_body),
            },
            source_terms_url=source_terms_url,
        )


def normalize_social_csv(envelope: RawEnvelope) -> list[SourceItem]:
    """Normalize a stored social CSV without network access or file writes."""

    if envelope.provider != _PROVIDER:
        raise SocialCsvSchemaError("cannot normalize a non-social-CSV envelope")
    try:
        expected_provenance = _required_request_text(envelope.request, "provenance")
        expected_license = _required_request_text(envelope.request, "license")
        expected_terms_url = _required_request_text(
            envelope.request, "source_terms_url"
        )
    except SocialCsvRequestError as error:
        raise SocialCsvSchemaError(
            "social CSV envelope is missing dataset terms metadata"
        ) from error
    if envelope.source_terms_url != expected_terms_url:
        raise SocialCsvSchemaError(
            "social CSV envelope terms URL does not match its canonical request"
        )
    try:
        text = envelope.response_body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SocialCsvEncodingError("social CSV must be valid UTF-8") from error

    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    try:
        fieldnames = reader.fieldnames or []
    except csv.Error as error:
        error_line = _csv_error_line(reader, text, error)
        raise SocialCsvSchemaError(
            f"social CSV contains malformed quoting at line {error_line}"
        ) from error
    duplicate_columns = sorted(
        name for name, count in Counter(fieldnames).items() if count > 1
    )
    if duplicate_columns:
        raise SocialCsvSchemaError(
            "social CSV has duplicate columns: " + ", ".join(duplicate_columns)
        )
    fieldname_set = set(fieldnames)
    missing_columns = sorted(_REQUIRED_COLUMNS - fieldname_set)
    if missing_columns:
        raise SocialCsvSchemaError(
            "social CSV is missing required columns: " + ", ".join(missing_columns)
        )
    unexpected_columns = sorted(fieldname_set - _REQUIRED_COLUMNS)
    if unexpected_columns:
        raise SocialCsvSchemaError(
            "social CSV has unexpected columns: " + ", ".join(unexpected_columns)
        )

    snapshot_id = hashlib.sha256(envelope.canonical_manifest_bytes()).hexdigest()
    seen_row_ids: set[str] = set()
    normalized: list[SourceItem] = []
    try:
        for raw_row in reader:
            row_number = reader.line_num
            try:
                row = _SocialRow.model_validate(raw_row)
                if row.source_row_id in seen_row_ids:
                    raise SocialCsvSchemaError(
                        f"duplicate source_row_id {row.source_row_id!r} at row {row_number}"
                    )
                if row.provenance != expected_provenance:
                    raise SocialCsvSchemaError(
                        f"social CSV row {row_number} provenance does not match request"
                    )
                if row.license != expected_license:
                    raise SocialCsvSchemaError(
                        f"social CSV row {row_number} license does not match request"
                    )
                item = SourceItem(
                    source_item_id=f"social-{row.source_row_id}",
                    source_type=SourceType.SOCIAL,
                    provider=_PROVIDER,
                    text=row.text,
                    published_at=row.published_at,
                    retrieved_at=envelope.retrieved_at,
                    source_reference=row.source_reference,
                    content_hash=hashlib.sha256(row.text.encode("utf-8")).hexdigest(),
                    snapshot_id=snapshot_id,
                    provenance=row.provenance,
                    license=row.license,
                )
            except SocialCsvSchemaError:
                raise
            except ValidationError as error:
                raise SocialCsvSchemaError(
                    f"social CSV row {row_number} violates the source contract"
                ) from error
            seen_row_ids.add(row.source_row_id)
            normalized.append(item)
    except csv.Error as error:
        error_line = _csv_error_line(reader, text, error)
        raise SocialCsvSchemaError(
            f"social CSV contains malformed quoting at line {error_line}"
        ) from error
    return normalized


__all__ = [
    "SocialCsvEncodingError",
    "SocialCsvError",
    "SocialCsvFileError",
    "SocialCsvProvider",
    "SocialCsvRequestError",
    "SocialCsvSchemaError",
    "normalize_social_csv",
]
