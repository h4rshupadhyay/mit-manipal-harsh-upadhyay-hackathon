import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from risk_engine.backtest.splits import (
    ChronologicalSplitSpec,
    FoldGroup,
    InnerFold,
    OuterFold,
    nested_chronological_splits,
)
from risk_engine.config import ClusteringConfig
from risk_engine.data.clustering import StoryCluster, cluster_stories
from risk_engine.domain import SourceItem, SourceType

BASE_TIME = datetime(2020, 1, 1, tzinfo=timezone.utc)
CLUSTERING_CONFIG = ClusteringConfig(
    similarity_threshold=1.0,
    max_time_delta_hours=1,
)


def make_item(source_item_id: str, day: int) -> SourceItem:
    text = f"Unique project-authored story {source_item_id}"
    published_at = BASE_TIME + timedelta(days=day)
    return SourceItem(
        source_item_id=source_item_id,
        source_type=SourceType.NEWS,
        provider="fixture",
        text=text,
        published_at=published_at,
        retrieved_at=published_at + timedelta(hours=1),
        source_reference=f"https://publisher.test/{source_item_id}",
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        snapshot_id="snapshot-1",
        provenance="project-authored fixture",
        license="MIT",
    )


def make_cluster(source_item_id: str, day: int) -> StoryCluster:
    return cluster_stories([make_item(source_item_id, day)], CLUSTERING_CONFIG)[0]


def make_cluster_at(source_item_id: str, event_time: datetime) -> StoryCluster:
    cluster = make_cluster(source_item_id, 0)
    item = SourceItem.model_validate(
        cluster.items[0].model_copy(
            update={
                "published_at": event_time,
                "retrieved_at": event_time.astimezone(timezone.utc) + timedelta(hours=1),
            }
        ).model_dump()
    )
    return StoryCluster.model_validate(
        cluster.model_copy(update={"event_time": event_time, "items": (item,)}).model_dump()
    )


def repeated_hour_clusters() -> list[StoryCluster]:
    new_york = ZoneInfo("America/New_York")
    times = (
        datetime(2020, 11, 1, 0, 0, tzinfo=new_york),
        datetime(2020, 11, 1, 0, 30, tzinfo=new_york),
        datetime(2020, 11, 1, 1, 30, fold=1, tzinfo=new_york),
        datetime(2020, 11, 1, 1, 45, fold=0, tzinfo=new_york),
        datetime(2020, 11, 1, 2, 30, tzinfo=new_york),
        datetime(2020, 11, 1, 3, 30, tzinfo=new_york),
    )
    return [make_cluster_at(f"source-{index}", time) for index, time in enumerate(times)]


def dst_spec(embargo: timedelta = timedelta(minutes=5)) -> ChronologicalSplitSpec:
    return standard_spec(
        initial_outer_train_groups=2,
        outer_test_groups=1,
        inner_train_groups=1,
        inner_validation_groups=1,
        final_holdout_groups=1,
        embargo=embargo,
        longest_event_window=embargo,
    )


def make_cluster_with_items(source_item_ids: tuple[str, ...], day: int) -> StoryCluster:
    text = f"Common project-authored story for day {day}"
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    items = [
        make_item(source_item_id, day).model_copy(
            update={"text": text, "content_hash": content_hash}
        )
        for source_item_id in source_item_ids
    ]
    return cluster_stories(items, CLUSTERING_CONFIG)[0]


def memberships(groups: tuple[FoldGroup, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(group.source_item_ids for group in groups)


def standard_spec(**updates: object) -> ChronologicalSplitSpec:
    values = {
        "version": "nested-grouped-chronological-v1",
        "initial_outer_train_groups": 3,
        "outer_test_groups": 2,
        "inner_train_groups": 2,
        "inner_validation_groups": 1,
        "final_holdout_groups": 2,
        "embargo": timedelta(days=2),
        "longest_event_window": timedelta(days=2),
    }
    values.update(updates)
    return ChronologicalSplitSpec(**values)  # type: ignore[arg-type]


def test_builds_expanding_outer_and_rolling_inner_folds_at_exact_boundary() -> None:
    clusters = [make_cluster(f"source-{index}", index) for index in range(11)]
    spec = standard_spec()

    folds = nested_chronological_splits(clusters, spec)

    assert [fold.fold_id for fold in folds] == ["outer-0001", "outer-0002"]
    assert memberships(folds[0].train) == (
        ("source-0",),
        ("source-1",),
        ("source-2",),
        ("source-3",),
    )
    assert memberships(folds[0].embargo) == (("source-4",),)
    assert memberships(folds[0].test) == (("source-5",), ("source-6",))
    assert memberships(folds[1].train) == (
        ("source-0",),
        ("source-1",),
        ("source-2",),
        ("source-3",),
        ("source-4",),
        ("source-5",),
    )
    assert memberships(folds[1].embargo) == (("source-6",),)
    assert memberships(folds[1].test) == (("source-7",), ("source-8",))
    assert memberships(folds[0].final_holdout) == (
        ("source-9",),
        ("source-10",),
    )
    assert folds[0].final_holdout == folds[1].final_holdout

    first_inner = folds[0].inner_folds[0]
    assert first_inner.fold_id == "outer-0001-inner-0001"
    assert memberships(first_inner.train) == (("source-0",), ("source-1",))
    assert memberships(first_inner.embargo) == (("source-2",),)
    assert memberships(first_inner.validation) == (("source-3",),)

    assert [fold.fold_id for fold in folds[1].inner_folds] == [
        "outer-0002-inner-0001",
        "outer-0002-inner-0002",
        "outer-0002-inner-0003",
    ]
    assert [memberships(fold.train) for fold in folds[1].inner_folds] == [
        (("source-0",), ("source-1",)),
        (("source-1",), ("source-2",)),
        (("source-2",), ("source-3",)),
    ]
    assert [memberships(fold.embargo) for fold in folds[1].inner_folds] == [
        (("source-2",),),
        (("source-3",),),
        (("source-4",),),
    ]
    assert [memberships(fold.validation) for fold in folds[1].inner_folds] == [
        (("source-3",),),
        (("source-4",),),
        (("source-5",),),
    ]


def test_canonical_input_order_produces_stable_replay() -> None:
    clusters = [make_cluster(f"source-{index}", index) for index in range(11)]

    canonical = nested_chronological_splits(clusters, standard_spec())
    replay = nested_chronological_splits(
        [clusters[index] for index in (10, 2, 8, 0, 6, 4, 1, 9, 3, 7, 5)],
        standard_spec(),
    )

    assert replay == canonical
    assert memberships(replay[0].train) == (
        ("source-0",),
        ("source-1",),
        ("source-2",),
        ("source-3",),
    )


def test_multi_source_story_cluster_remains_one_complete_fold_group() -> None:
    clusters = [make_cluster(f"source-{index}", index) for index in range(11)]
    clusters[5] = make_cluster_with_items(("source-5a", "source-5b"), 5)

    folds = nested_chronological_splits(clusters, standard_spec())

    assert memberships(folds[0].test) == (
        ("source-5a", "source-5b"),
        ("source-6",),
    )
    assert memberships(folds[1].train)[-1] == ("source-5a", "source-5b")


def test_embargo_partition_is_empty_when_no_group_falls_inside_the_gap() -> None:
    clusters = [make_cluster(f"source-{index}", index) for index in range(8)]
    spec = standard_spec(
        final_holdout_groups=1,
        embargo=timedelta(hours=1),
        longest_event_window=timedelta(hours=1),
    )

    folds = nested_chronological_splits(clusters, spec)

    assert [memberships(fold.embargo) for fold in folds] == [(), ()]
    assert memberships(folds[0].test) == (("source-3",), ("source-4",))
    assert all(inner.embargo == () for fold in folds for inner in fold.inner_folds)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"initial_outer_train_groups": 0}, "greater than 0"),
        ({"outer_test_groups": True}, "valid integer"),
        ({"embargo": timedelta(0)}, "greater than"),
        (
            {
                "embargo": timedelta(days=1),
                "longest_event_window": timedelta(days=2),
            },
            "at least as long",
        ),
        ({"embargo": "P2D"}, "valid timedelta"),
    ],
)
def test_split_spec_rejects_ambiguous_or_unsafe_values(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        standard_spec(**updates)


def test_rejects_duplicate_cluster_ids() -> None:
    clusters = [make_cluster(f"source-{index}", index) for index in range(11)]

    with pytest.raises(ValueError, match="cluster_id values must be unique"):
        nested_chronological_splits([*clusters, clusters[0]], standard_spec())


def test_rejects_a_source_item_present_in_distinct_story_clusters() -> None:
    shared = make_item("shared", 0)
    common_text = "Exact common project-authored story"
    common_hash = hashlib.sha256(common_text.encode("utf-8")).hexdigest()
    shared = shared.model_copy(
        update={"text": common_text, "content_hash": common_hash}
    )
    first_only = make_item("first-only", 0).model_copy(
        update={"text": common_text, "content_hash": common_hash}
    )
    second_only = make_item("second-only", 0).model_copy(
        update={"text": common_text, "content_hash": common_hash}
    )
    first = cluster_stories([shared, first_only], CLUSTERING_CONFIG)[0]
    second = cluster_stories([shared, second_only], CLUSTERING_CONFIG)[0]

    with pytest.raises(ValueError, match="Source Item IDs must be unique"):
        nested_chronological_splits([first, second], standard_spec())


def test_final_holdout_never_straddles_development_partitions() -> None:
    clusters = [make_cluster(f"source-{index}", index) for index in range(11)]

    folds = nested_chronological_splits(clusters, standard_spec())

    for fold in folds:
        development_ids = {
            source_item_id
            for group in fold.train + fold.embargo + fold.test
            for source_item_id in group.source_item_ids
        }
        inner_ids = {
            source_item_id
            for inner in fold.inner_folds
            for group in inner.train + inner.embargo + inner.validation
            for source_item_id in group.source_item_ids
        }
        assert development_ids.isdisjoint({"source-9", "source-10"})
        assert inner_ids.isdisjoint({"source-9", "source-10"})
        assert set(memberships(fold.train)).isdisjoint(memberships(fold.test))


def test_partial_tail_before_final_holdout_is_not_evaluated() -> None:
    clusters = [make_cluster(f"source-{index}", index) for index in range(12)]

    folds = nested_chronological_splits(clusters, standard_spec())

    assert [memberships(fold.test) for fold in folds] == [
        (("source-5",), ("source-6",)),
        (("source-7",), ("source-8",)),
    ]
    assert memberships(folds[0].final_holdout) == (
        ("source-10",),
        ("source-11",),
    )
    assert all(
        ("source-9",) not in memberships(fold.test)
        for fold in folds
    )


def test_fold_records_reject_identity_straddling_and_nonchronological_order() -> None:
    early = FoldGroup(
        cluster_id="cluster-a",
        event_time=BASE_TIME,
        source_item_ids=("shared-source",),
    )
    late_with_same_source = FoldGroup(
        cluster_id="cluster-b",
        event_time=BASE_TIME + timedelta(days=1),
        source_item_ids=("shared-source",),
    )
    late = FoldGroup(
        cluster_id="cluster-c",
        event_time=BASE_TIME + timedelta(days=2),
        source_item_ids=("late-source",),
    )

    with pytest.raises(ValidationError, match="Source Item identities must not straddle"):
        InnerFold(
            fold_id="outer-0001-inner-0001",
            train=(early,),
            embargo=(),
            validation=(late_with_same_source,),
        )
    with pytest.raises(ValidationError, match="chronological canonical ordering"):
        InnerFold(
            fold_id="outer-0001-inner-0001",
            train=(late,),
            embargo=(),
            validation=(early,),
        )


def test_outer_fold_rejects_an_inner_group_outside_outer_training() -> None:
    train = FoldGroup(
        cluster_id="cluster-a",
        event_time=BASE_TIME,
        source_item_ids=("source-a",),
    )
    test = FoldGroup(
        cluster_id="cluster-b",
        event_time=BASE_TIME + timedelta(days=2),
        source_item_ids=("source-b",),
    )
    holdout = FoldGroup(
        cluster_id="cluster-c",
        event_time=BASE_TIME + timedelta(days=3),
        source_item_ids=("source-c",),
    )
    inner = InnerFold(
        fold_id="outer-0001-inner-0001",
        train=(train,),
        embargo=(),
        validation=(test,),
    )

    with pytest.raises(ValidationError, match="inner fold groups must come from"):
        OuterFold(
            fold_id="outer-0001",
            train=(train,),
            embargo=(),
            test=(test,),
            final_holdout=(holdout,),
            inner_folds=(inner,),
        )


def test_outer_fold_rejects_inner_folds_out_of_chronological_order() -> None:
    groups = tuple(
        FoldGroup(
            cluster_id=f"cluster-{index}",
            event_time=BASE_TIME + timedelta(days=index),
            source_item_ids=(f"source-{index}",),
        )
        for index in range(5)
    )
    first = InnerFold(
        fold_id="outer-0001-inner-0001",
        train=(groups[0],),
        embargo=(),
        validation=(groups[1],),
    )
    second = InnerFold(
        fold_id="outer-0001-inner-0002",
        train=(groups[1],),
        embargo=(),
        validation=(groups[2],),
    )

    with pytest.raises(
        ValidationError,
        match="inner validation blocks must be strictly ordered and disjoint",
    ):
        OuterFold(
            fold_id="outer-0001",
            train=groups[:3],
            embargo=(),
            test=(groups[3],),
            final_holdout=(groups[4],),
            inner_folds=(second, first),
        )


def serialized_outer_fold() -> dict[str, Any]:
    groups = tuple(
        FoldGroup(
            cluster_id=f"cluster-{index}",
            event_time=BASE_TIME + timedelta(days=index),
            source_item_ids=(f"source-{index}",),
        )
        for index in range(8)
    )
    first = InnerFold(
        fold_id="outer-0001-inner-0001",
        train=groups[:2],
        embargo=(),
        validation=groups[2:4],
    )
    second = InnerFold(
        fold_id="outer-0001-inner-0002",
        train=groups[2:4],
        embargo=(),
        validation=groups[4:6],
    )
    fold = OuterFold(
        fold_id="outer-0001",
        train=groups[:6],
        embargo=(),
        test=(groups[6],),
        final_holdout=(groups[7],),
        inner_folds=(first, second),
    )
    return fold.model_dump(mode="json")


def test_serialized_restore_rejects_duplicate_inner_fold_ids() -> None:
    payload = serialized_outer_fold()
    payload["inner_folds"][1]["fold_id"] = "outer-0001-inner-0001"

    with pytest.raises(ValidationError, match="inner fold IDs must be unique"):
        OuterFold.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "second_validation_indices",
    [(2, 3), (3, 4)],
    ids=["equal", "partially-overlapping"],
)
def test_serialized_restore_rejects_overlapping_inner_validation_blocks(
    second_validation_indices: tuple[int, int],
) -> None:
    payload = serialized_outer_fold()
    outer_train = payload["train"]
    payload["inner_folds"][1]["train"] = outer_train[:2]
    payload["inner_folds"][1]["validation"] = [
        outer_train[index] for index in second_validation_indices
    ]

    with pytest.raises(
        ValidationError,
        match="inner validation blocks must be strictly ordered and disjoint",
    ):
        OuterFold.model_validate_json(json.dumps(payload))


def test_rejects_data_without_a_complete_outer_fold_and_inner_fold() -> None:
    clusters = [make_cluster(f"source-{index}", index) for index in range(7)]

    with pytest.raises(ValueError, match="complete outer fold with an inner fold"):
        nested_chronological_splits(clusters, standard_spec())


def test_repeated_hour_folds_use_utc_chronology_for_every_partition() -> None:
    folds = nested_chronological_splits(repeated_hour_clusters(), dst_spec())

    assert [memberships(fold.test) for fold in folds] == [
        (("source-3",),),
        (("source-2",),),
        (("source-4",),),
    ]
    for fold in folds:
        assert all(
            group.event_time.astimezone(timezone.utc)
            < fold.test[0].event_time.astimezone(timezone.utc)
            for group in fold.train
        )
        for inner in fold.inner_folds:
            assert all(
                group.event_time.astimezone(timezone.utc)
                < inner.validation[0].event_time.astimezone(timezone.utc)
                for group in inner.train
            )


def test_repeated_hour_input_permutation_preserves_canonical_utc_order() -> None:
    clusters = repeated_hour_clusters()
    canonical = nested_chronological_splits(clusters, dst_spec())
    replay = nested_chronological_splits(list(reversed(clusters)), dst_spec())

    assert replay == canonical
    assert memberships(replay[-1].train) == (
        ("source-0",),
        ("source-1",),
        ("source-3",),
        ("source-2",),
    )
    assert replay[-1].train[-1].event_time.isoformat() == "2020-11-01T01:30:00-05:00"
    assert replay[-1].train[-1].event_time.fold == 1


def test_repeated_hour_generated_folds_restore_from_json() -> None:
    folds = nested_chronological_splits(repeated_hour_clusters(), dst_spec())

    for fold in folds:
        restored = OuterFold.model_validate_json(fold.model_dump_json())
        assert restored.model_dump_json() == fold.model_dump_json()


def repeated_hour_groups() -> tuple[FoldGroup, FoldGroup]:
    new_york = ZoneInfo("America/New_York")
    early = FoldGroup(
        cluster_id="early",
        event_time=datetime(2020, 11, 1, 1, 45, fold=0, tzinfo=new_york),
        source_item_ids=("early-source",),
    )
    late = FoldGroup(
        cluster_id="late",
        event_time=datetime(2020, 11, 1, 1, 30, fold=1, tzinfo=new_york),
        source_item_ids=("late-source",),
    )
    return early, late


def test_inner_fold_rejects_reversed_utc_order_in_repeated_hour() -> None:
    early, late = repeated_hour_groups()

    with pytest.raises(ValidationError, match="chronological canonical ordering"):
        InnerFold(fold_id="inner", train=(late,), embargo=(), validation=(early,))


def test_inner_fold_accepts_utc_order_when_local_clock_order_reverses() -> None:
    early, late = repeated_hour_groups()

    fold = InnerFold(fold_id="inner", train=(early,), embargo=(), validation=(late,))

    assert fold.train == (early,)
    assert fold.validation == (late,)
    assert InnerFold.model_validate_json(fold.model_dump_json()).model_dump_json() == (
        fold.model_dump_json()
    )


def test_outer_fold_rejects_inner_validation_blocks_reversed_in_utc() -> None:
    early, late = repeated_hour_groups()
    train = FoldGroup(
        cluster_id="train",
        event_time=BASE_TIME,
        source_item_ids=("train-source",),
    )
    test = FoldGroup(
        cluster_id="test",
        event_time=datetime(2020, 11, 2, tzinfo=timezone.utc),
        source_item_ids=("test-source",),
    )
    holdout = FoldGroup(
        cluster_id="holdout",
        event_time=datetime(2020, 11, 3, tzinfo=timezone.utc),
        source_item_ids=("holdout-source",),
    )
    first = InnerFold(fold_id="first", train=(train,), embargo=(), validation=(early,))
    second = InnerFold(fold_id="second", train=(train,), embargo=(), validation=(late,))

    with pytest.raises(
        ValidationError,
        match="inner validation blocks must be strictly ordered and disjoint",
    ):
        OuterFold(
            fold_id="outer",
            train=(train, early, late),
            embargo=(),
            test=(test,),
            final_holdout=(holdout,),
            inner_folds=(second, first),
        )


@pytest.mark.parametrize(
    ("times", "embargo", "expected_train", "expected_embargo"),
    [
        (
            (
                (3, 6, 12, 0, 0),
                (3, 7, 12, 0, 0),
                (3, 8, 1, 30, 0),
                (3, 8, 3, 15, 0),
                (3, 8, 5, 15, 0),
                (3, 9, 12, 0, 0),
            ),
            timedelta(hours=1),
            (("source-0",), ("source-1",)),
            (("source-2",),),
        ),
        (
            (
                (10, 30, 12, 0, 0),
                (10, 31, 12, 0, 0),
                (11, 1, 0, 30, 0),
                (11, 1, 1, 45, 1),
                (11, 1, 4, 0, 0),
                (11, 2, 12, 0, 0),
            ),
            timedelta(hours=2),
            (("source-0",), ("source-1",), ("source-2",)),
            (),
        ),
    ],
    ids=["spring-forward-45m-fails-1h", "fall-back-135m-satisfies-2h"],
)
def test_embargo_uses_elapsed_duration_across_dst(
    times: tuple[tuple[int, int, int, int, int], ...],
    embargo: timedelta,
    expected_train: tuple[tuple[str, ...], ...],
    expected_embargo: tuple[tuple[str, ...], ...],
) -> None:
    new_york = ZoneInfo("America/New_York")
    clusters = [
        make_cluster_at(
            f"source-{index}",
            datetime(2020, month, day, hour, minute, fold=fold, tzinfo=new_york),
        )
        for index, (month, day, hour, minute, fold) in enumerate(times)
    ]

    folds = nested_chronological_splits(clusters, dst_spec(embargo))

    assert memberships(folds[1].test) == (("source-3",),)
    assert memberships(folds[1].train) == expected_train
    assert memberships(folds[1].embargo) == expected_embargo
