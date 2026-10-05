"""Deterministic leakage-safe chronological development splits."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from risk_engine.data.clustering import StoryCluster

_NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_PositiveCount = Annotated[int, Field(strict=True, gt=0)]
_PositiveDuration = Annotated[timedelta, Field(strict=True, gt=timedelta(0))]


class _SplitRecord(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)


class ChronologicalSplitSpec(_SplitRecord):
    """Versioned parameters for nested grouped chronological splits."""

    version: Literal["nested-grouped-chronological-v1"]
    initial_outer_train_groups: _PositiveCount
    outer_test_groups: _PositiveCount
    inner_train_groups: _PositiveCount
    inner_validation_groups: _PositiveCount
    final_holdout_groups: _PositiveCount
    embargo: _PositiveDuration
    longest_event_window: _PositiveDuration

    @model_validator(mode="after")
    def embargo_covers_event_window(self) -> ChronologicalSplitSpec:
        if self.embargo < self.longest_event_window:
            raise ValueError("embargo must be at least as long as longest_event_window")
        return self


class FoldGroup(_SplitRecord):
    """The complete identity and chronology needed from a Story Cluster."""

    cluster_id: _NonEmptyString
    event_time: AwareDatetime
    source_item_ids: Annotated[tuple[_NonEmptyString, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def source_item_ids_are_canonical(self) -> FoldGroup:
        if self.source_item_ids != tuple(sorted(set(self.source_item_ids))):
            raise ValueError("source_item_ids must be unique, complete, and sorted")
        return self


def _group_key(group: FoldGroup) -> tuple[object, str]:
    return group.event_time, group.cluster_id


def _validate_partitions(
    partitions: tuple[tuple[FoldGroup, ...], ...],
) -> None:
    populated = tuple(group for partition in partitions for group in partition)
    if populated != tuple(sorted(populated, key=_group_key)):
        raise ValueError("fold partitions must use chronological canonical ordering")
    cluster_ids = [group.cluster_id for group in populated]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("cluster identities must not straddle fold partitions")
    source_item_ids = [
        source_item_id
        for group in populated
        for source_item_id in group.source_item_ids
    ]
    if len(source_item_ids) != len(set(source_item_ids)):
        raise ValueError("Source Item identities must not straddle fold partitions")


class InnerFold(_SplitRecord):
    """One fixed-width rolling training and validation fold."""

    fold_id: _NonEmptyString
    train: Annotated[tuple[FoldGroup, ...], Field(min_length=1)]
    embargo: tuple[FoldGroup, ...]
    validation: Annotated[tuple[FoldGroup, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def partitions_are_valid(self) -> InnerFold:
        _validate_partitions((self.train, self.embargo, self.validation))
        return self


class OuterFold(_SplitRecord):
    """One expanding training and out-of-sample test fold."""

    fold_id: _NonEmptyString
    train: Annotated[tuple[FoldGroup, ...], Field(min_length=1)]
    embargo: tuple[FoldGroup, ...]
    test: Annotated[tuple[FoldGroup, ...], Field(min_length=1)]
    final_holdout: Annotated[tuple[FoldGroup, ...], Field(min_length=1)]
    inner_folds: Annotated[tuple[InnerFold, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def partitions_are_valid(self) -> OuterFold:
        _validate_partitions(
            (self.train, self.embargo, self.test, self.final_holdout)
        )
        inner_fold_ids = tuple(inner_fold.fold_id for inner_fold in self.inner_folds)
        if len(inner_fold_ids) != len(set(inner_fold_ids)):
            raise ValueError("inner fold IDs must be unique")
        for previous, current in zip(
            self.inner_folds, self.inner_folds[1:], strict=False
        ):
            if _group_key(current.validation[0]) <= _group_key(previous.validation[-1]):
                raise ValueError(
                    "inner validation blocks must be strictly ordered and disjoint"
                )
        outer_train_groups = set(self.train)
        for inner_fold in self.inner_folds:
            inner_groups = (
                inner_fold.train + inner_fold.embargo + inner_fold.validation
            )
            if any(group not in outer_train_groups for group in inner_groups):
                raise ValueError("inner fold groups must come from outer training")
        return self


def _eligible_count(
    groups: tuple[FoldGroup, ...],
    evaluation_start: int,
    embargo: timedelta,
) -> int:
    first_evaluation_time = groups[evaluation_start].event_time
    return sum(
        group.event_time + embargo <= first_evaluation_time
        for group in groups[:evaluation_start]
    )


def _inner_folds(
    groups: tuple[FoldGroup, ...],
    outer_fold_id: str,
    spec: ChronologicalSplitSpec,
) -> tuple[InnerFold, ...]:
    latest_start = len(groups) - spec.inner_validation_groups
    validation_start = spec.inner_train_groups
    while validation_start <= latest_start:
        eligible_count = _eligible_count(groups, validation_start, spec.embargo)
        if eligible_count >= spec.inner_train_groups:
            break
        validation_start += 1
    else:
        return ()

    folds: list[InnerFold] = []
    while validation_start <= latest_start:
        eligible_count = _eligible_count(groups, validation_start, spec.embargo)
        train_start = eligible_count - spec.inner_train_groups
        fold_number = len(folds) + 1
        folds.append(
            InnerFold(
                fold_id=f"{outer_fold_id}-inner-{fold_number:04d}",
                train=groups[train_start:eligible_count],
                embargo=groups[eligible_count:validation_start],
                validation=groups[
                    validation_start : validation_start
                    + spec.inner_validation_groups
                ],
            )
        )
        validation_start += spec.inner_validation_groups
    return tuple(folds)


def _canonical_groups(clusters: Sequence[StoryCluster]) -> tuple[FoldGroup, ...]:
    validated_clusters = tuple(
        StoryCluster.model_validate(cluster.model_dump()) for cluster in clusters
    )
    cluster_ids = [cluster.cluster_id for cluster in validated_clusters]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("cluster_id values must be unique")
    source_item_ids = [
        source_item_id
        for cluster in validated_clusters
        for source_item_id in cluster.source_item_ids
    ]
    if len(source_item_ids) != len(set(source_item_ids)):
        raise ValueError("Source Item IDs must be unique across Story Clusters")
    return tuple(
        FoldGroup(
            cluster_id=cluster.cluster_id,
            event_time=cluster.event_time,
            source_item_ids=tuple(sorted(cluster.source_item_ids)),
        )
        for cluster in sorted(
            validated_clusters,
            key=lambda cluster: (cluster.event_time, cluster.cluster_id),
        )
    )


def nested_chronological_splits(
    clusters: Sequence[StoryCluster],
    spec: ChronologicalSplitSpec,
) -> list[OuterFold]:
    """Create deterministic nested folds without splitting a Story Cluster."""

    groups = _canonical_groups(clusters)
    if len(groups) <= spec.final_holdout_groups:
        raise ValueError("insufficient groups for development and final holdout")
    development = groups[: -spec.final_holdout_groups]
    final_holdout = groups[-spec.final_holdout_groups :]

    latest_start = len(development) - spec.outer_test_groups
    test_start = spec.initial_outer_train_groups
    first_inner_folds: tuple[InnerFold, ...] = ()
    while test_start <= latest_start:
        eligible_count = _eligible_count(development, test_start, spec.embargo)
        if eligible_count >= spec.initial_outer_train_groups:
            first_inner_folds = _inner_folds(
                development[:eligible_count],
                "outer-0001",
                spec,
            )
            if first_inner_folds:
                break
        test_start += 1
    else:
        raise ValueError(
            "insufficient groups for a complete outer fold with an inner fold"
        )

    folds: list[OuterFold] = []
    while test_start <= latest_start:
        eligible_count = _eligible_count(development, test_start, spec.embargo)
        fold_number = len(folds) + 1
        fold_id = f"outer-{fold_number:04d}"
        inner_folds = (
            first_inner_folds
            if fold_number == 1
            else _inner_folds(development[:eligible_count], fold_id, spec)
        )
        if not inner_folds:
            raise ValueError("outer fold does not contain a complete inner fold")
        folds.append(
            OuterFold(
                fold_id=fold_id,
                train=development[:eligible_count],
                embargo=development[eligible_count:test_start],
                test=development[test_start : test_start + spec.outer_test_groups],
                final_holdout=final_holdout,
                inner_folds=inner_folds,
            )
        )
        test_start += spec.outer_test_groups
    return folds


__all__ = [
    "ChronologicalSplitSpec",
    "FoldGroup",
    "InnerFold",
    "OuterFold",
    "nested_chronological_splits",
]
