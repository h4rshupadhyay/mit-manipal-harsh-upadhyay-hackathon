"""Deterministic duplicate and revised-story clustering."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import (
    AnyHttpUrl,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from risk_engine.config import ClusteringConfig
from risk_engine.domain import SourceItem

_NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_TRACKING_PARAMETERS = {"fbclid", "gclid"}
_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)


def _publication_key(item: SourceItem) -> tuple[datetime, str]:
    return item.published_at.astimezone(UTC), item.source_item_id


class StoryCluster(BaseModel):
    """One deterministic component of duplicate or revised source stories."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cluster_id: _NonEmptyString
    event_time: AwareDatetime
    representative_source_item_id: _NonEmptyString
    source_item_ids: Annotated[tuple[_NonEmptyString, ...], Field(min_length=1)]
    items: Annotated[tuple[SourceItem, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def membership_is_consistent(self) -> StoryCluster:
        item_ids = tuple(item.source_item_id for item in self.items)
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("cluster items must have unique source_item_id values")
        canonical_source_item_ids = tuple(sorted(item_ids))
        if self.source_item_ids != canonical_source_item_ids:
            raise ValueError("source_item_ids must be unique, complete, and sorted")
        canonical_items = tuple(
            sorted(self.items, key=_publication_key)
        )
        if self.items != canonical_items:
            raise ValueError("cluster items must use canonical publication and ID order")
        representative = canonical_items[0]
        if self.representative_source_item_id != representative.source_item_id:
            raise ValueError("representative must be the earliest canonical source item")
        if self.event_time.astimezone(UTC) != representative.published_at.astimezone(UTC):
            raise ValueError("event_time must be the earliest publication time")
        if self.cluster_id != _cluster_id(canonical_source_item_ids):
            raise ValueError("cluster_id must be derived from canonical source_item_ids")
        return self


def _is_tracking_parameter(name: str) -> bool:
    normalized = name.casefold()
    return normalized.startswith("utm_") or normalized in _TRACKING_PARAMETERS


def _normalize_http_url(parsed: SplitResult) -> str | None:
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return None

    try:
        raw_hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if not raw_hostname or parsed.username is not None or parsed.password is not None:
        return None
    hostname = raw_hostname.casefold()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    default_port = (parsed.scheme.casefold(), port) in {("http", 80), ("https", 443)}
    authority_host = f"[{hostname}]" if ":" in hostname else hostname
    authority = (
        authority_host
        if port is None or default_port
        else f"{authority_host}:{port}"
    )
    path = parsed.path.rstrip("/") or "/"
    query = urlencode(
        sorted(
            (name, parameter_value)
            for name, parameter_value in parse_qsl(
                parsed.query,
                keep_blank_values=True,
            )
            if not _is_tracking_parameter(name)
        )
    )
    return urlunsplit((parsed.scheme.casefold(), authority, path, query, ""))


def _reference_key(item: SourceItem) -> tuple[str, ...] | None:
    value = item.source_reference.strip()
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme.casefold() in {"http", "https"}:
        if not value.casefold().startswith(f"{parsed.scheme.casefold()}://"):
            return None
        try:
            validated_url = _HTTP_URL_ADAPTER.validate_python(value)
        except ValidationError:
            return None
        normalized_url = _normalize_http_url(urlsplit(str(validated_url)))
        return ("url", normalized_url) if normalized_url is not None else None
    return ("opaque", item.source_type.value, item.provider, value)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(text.casefold()))


def _jaccard_similarity(first: frozenset[str], second: frozenset[str]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def _cluster_id(source_item_ids: tuple[str, ...]) -> str:
    canonical_ids = json.dumps(
        source_item_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical_ids).hexdigest()
    return f"story-{digest}"


def cluster_stories(
    items: Sequence[SourceItem],
    config: ClusteringConfig,
) -> list[StoryCluster]:
    """Cluster stories using deterministic graph components and tie-breaks."""

    ordered_items = sorted(items, key=lambda item: item.source_item_id)
    item_ids = [item.source_item_id for item in ordered_items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("source_item_id values must be unique before clustering")
    reference_keys = [_reference_key(item) for item in ordered_items]
    token_sets = [_tokens(item.text) for item in ordered_items]

    parents = list(range(len(ordered_items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    reference_representatives: dict[tuple[str, ...], int] = {}
    content_representatives: dict[str, int] = {}
    for index, item in enumerate(ordered_items):
        reference_key = reference_keys[index]
        if reference_key is not None:
            previous_reference = reference_representatives.setdefault(
                reference_key, index
            )
            union(previous_reference, index)
        previous_content = content_representatives.setdefault(item.content_hash, index)
        union(previous_content, index)

    time_ordered_indices = sorted(
        range(len(ordered_items)),
        key=lambda index: _publication_key(ordered_items[index]),
    )
    max_delta_seconds = config.max_time_delta_hours * 3600
    for position, first_index in enumerate(time_ordered_indices):
        first = ordered_items[first_index]
        for second_position in range(position + 1, len(time_ordered_indices)):
            second_index = time_ordered_indices[second_position]
            second = ordered_items[second_index]
            delta_seconds = (
                second.published_at.astimezone(UTC) - first.published_at.astimezone(UTC)
            ).total_seconds()
            if delta_seconds > max_delta_seconds:
                break
            similarity = _jaccard_similarity(
                token_sets[first_index],
                token_sets[second_index],
            )
            if similarity >= config.similarity_threshold:
                union(first_index, second_index)

    components: dict[int, list[SourceItem]] = defaultdict(list)
    for index, item in enumerate(ordered_items):
        components[find(index)].append(item)

    clusters: list[StoryCluster] = []
    for members in components.values():
        ordered_members = tuple(
            sorted(members, key=_publication_key)
        )
        source_item_ids = tuple(sorted(item.source_item_id for item in members))
        representative = ordered_members[0]
        clusters.append(
            StoryCluster(
                cluster_id=_cluster_id(source_item_ids),
                event_time=representative.published_at,
                representative_source_item_id=representative.source_item_id,
                source_item_ids=source_item_ids,
                items=ordered_members,
            )
        )
    return sorted(
        clusters, key=lambda cluster: (cluster.event_time.astimezone(UTC), cluster.cluster_id)
    )


__all__ = ["StoryCluster", "cluster_stories"]
