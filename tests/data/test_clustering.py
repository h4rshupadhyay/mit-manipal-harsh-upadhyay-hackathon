import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from risk_engine.config import ClusteringConfig
from risk_engine.data import clustering as clustering_module
from risk_engine.data.clustering import StoryCluster, cluster_stories
from risk_engine.domain import SourceItem, SourceType

BASE_TIME = datetime(2026, 10, 4, 9, 0, tzinfo=timezone.utc)
RETRIEVED_AT = datetime(2026, 10, 6, 9, 0, tzinfo=timezone.utc)
CONFIG = ClusteringConfig(similarity_threshold=0.6, max_time_delta_hours=6)


def make_item(
    source_item_id: str,
    *,
    text: str,
    published_at: datetime = BASE_TIME,
    source_reference: str | None = None,
    provider: str = "fixture",
) -> SourceItem:
    return SourceItem(
        source_item_id=source_item_id,
        source_type=SourceType.NEWS,
        provider=provider,
        text=text,
        published_at=published_at,
        retrieved_at=RETRIEVED_AT,
        source_reference=source_reference or f"https://publisher.test/{source_item_id}",
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        snapshot_id="snapshot-1",
        provenance="project-authored fixture",
        license="MIT",
    )


def test_normalized_urls_cluster_revised_versions_and_retain_sources() -> None:
    first = make_item(
        "source-a",
        text="Initial report about bank funding pressure.",
        source_reference=(
            "HTTPS://WWW.Publisher.Test:443/story/?utm_source=feed&b=2&a=1#latest"
        ),
    )
    revision = make_item(
        "source-b",
        text="Revised report adds details that make its words entirely different.",
        published_at=BASE_TIME + timedelta(hours=10),
        source_reference="https://publisher.test/story?a=1&b=2",
    )

    clusters = cluster_stories([revision, first], CONFIG)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert isinstance(cluster, StoryCluster)
    assert cluster.event_time == BASE_TIME
    assert cluster.representative_source_item_id == "source-a"
    assert cluster.source_item_ids == ("source-a", "source-b")
    assert {item.source_item_id for item in cluster.items} == {
        "source-a",
        "source-b",
    }


def test_exact_content_clusters_across_distinct_source_urls() -> None:
    text = "The same syndicated story appears at two publishers."
    items = [
        make_item("source-a", text=text),
        make_item(
            "source-b",
            text=text,
            source_reference="https://second-publisher.test/syndicated-story",
        ),
    ]

    assert cluster_stories(items, CONFIG)[0].source_item_ids == (
        "source-a",
        "source-b",
    )


def test_exact_content_match_bypasses_the_fuzzy_time_window() -> None:
    text = "Exact syndicated copy with a stable content hash."
    first = make_item("source-a", text=text)
    much_later = make_item(
        "source-b",
        text=text,
        published_at=BASE_TIME + timedelta(days=1),
        source_reference="https://archive.test/copy",
    )

    assert len(cluster_stories([first, much_later], CONFIG)) == 1


def test_similarity_requires_both_frozen_threshold_and_time_window() -> None:
    near = make_item(
        "near-a",
        text="central bank raises policy rate after inflation surprise",
    )
    revision = make_item(
        "near-b",
        text="central bank raises policy rate after inflation data surprise",
        published_at=BASE_TIME + timedelta(hours=1),
    )
    outside_window = make_item(
        "late",
        text="central bank raises policy rate after inflation outlook surprise",
        published_at=BASE_TIME + timedelta(hours=8),
    )
    below_threshold = make_item(
        "unrelated",
        text="company launches unrelated consumer product",
        published_at=BASE_TIME + timedelta(hours=1),
    )

    clusters = cluster_stories(
        [outside_window, below_threshold, revision, near],
        CONFIG,
    )

    assert [cluster.source_item_ids for cluster in clusters] == [
        ("near-a", "near-b"),
        ("unrelated",),
        ("late",),
    ]


def test_similarity_and_time_boundaries_are_inclusive() -> None:
    first = make_item("source-a", text="alpha beta gamma delta")
    boundary = make_item(
        "source-b",
        text="alpha beta gamma epsilon",
        published_at=BASE_TIME + timedelta(hours=6),
    )

    assert len(cluster_stories([first, boundary], CONFIG)) == 1


def test_similarity_components_are_transitive() -> None:
    first = make_item("source-a", text="alpha beta gamma delta")
    bridge = make_item(
        "source-b",
        text="alpha beta gamma epsilon",
        published_at=BASE_TIME + timedelta(hours=1),
    )
    third = make_item(
        "source-c",
        text="alpha beta epsilon zeta",
        published_at=BASE_TIME + timedelta(hours=2),
    )

    clusters = cluster_stories([third, first, bridge], CONFIG)

    assert len(clusters) == 1
    assert clusters[0].source_item_ids == ("source-a", "source-b", "source-c")


def test_cluster_order_and_identity_do_not_depend_on_input_order() -> None:
    items = [
        make_item(
            "later",
            text="later independent story",
            published_at=BASE_TIME + timedelta(days=1),
        ),
        make_item("early-b", text="matching exact content"),
        make_item("early-a", text="matching exact content"),
    ]

    forward = cluster_stories(items, CONFIG)
    reversed_input = cluster_stories(list(reversed(items)), CONFIG)

    assert forward == reversed_input
    assert [cluster.event_time for cluster in forward] == sorted(
        cluster.event_time for cluster in forward
    )
    assert forward[0].representative_source_item_id == "early-a"


def test_clustering_rejects_duplicate_source_item_ids() -> None:
    first = make_item("duplicate", text="first")
    second = make_item("duplicate", text="second")

    with pytest.raises(ValueError, match="source_item_id"):
        cluster_stories([first, second], CONFIG)


def test_opaque_references_are_qualified_by_provider() -> None:
    first = make_item(
        "provider-a-row",
        text="bank funding conditions tightened",
        source_reference="1",
        provider="provider-a",
    )
    second = make_item(
        "provider-b-row",
        text="commodity exporter announces dividend",
        source_reference="1",
        provider="provider-b",
    )

    assert len(cluster_stories([first, second], CONFIG)) == 2


def test_malformed_urls_do_not_abort_or_form_reference_matches() -> None:
    malformed = make_item(
        "malformed",
        text="bank funding conditions tightened",
        source_reference="https://[invalid-ipv6",
    )
    other = make_item(
        "other",
        text="commodity exporter announces dividend",
        source_reference="https://[invalid-ipv6",
    )

    assert len(cluster_stories([malformed, other], CONFIG)) == 2


@pytest.mark.parametrize(
    "source_reference",
    [
        "https:/broken",
        "http:story",
        "https://:443/story",
        "https://exa mple.test/story",
        "https://%zz.test/story",
    ],
)
def test_parseable_invalid_http_references_do_not_match(
    source_reference: str,
) -> None:
    first = make_item(
        "first",
        text="bank funding conditions tightened",
        source_reference=source_reference,
    )
    second = make_item(
        "second",
        text="commodity exporter announces dividend",
        source_reference=source_reference,
    )

    assert len(cluster_stories([first, second], CONFIG)) == 2


def test_ipv6_hosts_preserve_brackets_and_port_boundaries() -> None:
    explicit_port = make_item(
        "explicit-port",
        text="bank funding conditions tightened",
        source_reference="https://[::1]:80/story",
    )
    address_suffix = make_item(
        "address-suffix",
        text="commodity exporter announces dividend",
        source_reference="https://[::1:80]/story",
    )

    assert len(cluster_stories([explicit_port, address_suffix], CONFIG)) == 2


@pytest.mark.parametrize(
    "violation",
    [
        "duplicate-source-ids",
        "unsorted-source-ids",
        "unsorted-items",
        "non-earliest-representative",
        "forged-cluster-id",
    ],
)
def test_story_cluster_rejects_noncanonical_public_state(violation: str) -> None:
    first = make_item("source-a", text="matching content")
    second = make_item(
        "source-b",
        text="matching content",
        published_at=BASE_TIME + timedelta(minutes=5),
    )
    valid = cluster_stories([second, first], CONFIG)[0]
    values = valid.model_dump()

    if violation == "duplicate-source-ids":
        values["source_item_ids"] = ("source-a", "source-a", "source-b")
    elif violation == "unsorted-source-ids":
        values["source_item_ids"] = ("source-b", "source-a")
    elif violation == "unsorted-items":
        values["items"] = tuple(reversed(valid.items))
    elif violation == "non-earliest-representative":
        values["representative_source_item_id"] = "source-b"
    else:
        values["cluster_id"] = "story-" + "0" * 64

    with pytest.raises(ValidationError):
        StoryCluster.model_validate(values)


def test_clustering_precomputes_reference_and_token_features_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [
        make_item(
            f"source-{index}",
            text=f"independent story number {index}",
            published_at=BASE_TIME + timedelta(minutes=index),
        )
        for index in range(4)
    ]
    calls = {"references": 0, "tokens": 0}
    original_reference_key = clustering_module._reference_key
    original_tokens = clustering_module._tokens

    def counted_reference_key(item: SourceItem) -> tuple[str, ...] | None:
        calls["references"] += 1
        return original_reference_key(item)

    def counted_tokens(text: str) -> frozenset[str]:
        calls["tokens"] += 1
        return original_tokens(text)

    monkeypatch.setattr(clustering_module, "_reference_key", counted_reference_key)
    monkeypatch.setattr(clustering_module, "_tokens", counted_tokens)

    cluster_stories(items, CONFIG)

    assert calls == {"references": len(items), "tokens": len(items)}
