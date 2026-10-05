"""Typed application configuration loaded from TOML."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - local Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from risk_engine.domain import EventClass

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VersionsConfig(ConfigModel):
    schema_version: NonEmptyString = Field(alias="schema")
    model: NonEmptyString
    calibration: NonEmptyString
    valuation_rules: NonEmptyString


class TaxonomyConfig(ConfigModel):
    event_classes: tuple[EventClass, ...]

    @model_validator(mode="after")
    def contains_the_frozen_taxonomy_once(self) -> TaxonomyConfig:
        expected = tuple(EventClass)
        if len(self.event_classes) != len(expected) or set(self.event_classes) != set(expected):
            raise ValueError("event_classes must contain each frozen Event Class exactly once")
        return self


class RuntimeConfig(ConfigModel):
    offline_replay: bool = True


class ClusteringConfig(ConfigModel):
    similarity_threshold: float = Field(gt=0.0, le=1.0)
    max_time_delta_hours: int = Field(gt=0)


class PolicyConfig(ConfigModel):
    impact_threshold: int = Field(default=8, ge=1, le=10)


MatchingField = Literal["event_class", "subtype", "region", "sector", "exposure_type"]


class ScaleDistanceConfig(ConfigModel):
    name: NonEmptyString
    unit: NonEmptyString
    normalizer: float = Field(gt=0, allow_inf_nan=False)
    weight: float = Field(gt=0, allow_inf_nan=False)


class AnalogueLevelConfig(ConfigModel):
    name: NonEmptyString
    fields: tuple[MatchingField, ...]
    scales: tuple[NonEmptyString, ...]


class AnalogueConfig(ConfigModel):
    """Frozen development candidates, distinct from chronological validation."""

    version: NonEmptyString
    validation_status: Literal["bootstrap_unvalidated", "chronologically_validated"]
    frozen_at: AwareDatetime
    # None means no fit/validation exists, never a fabricated timestamp/evidence.
    fit_as_of: AwareDatetime | None = None
    validation_evidence: NonEmptyString | None = None
    distance: Literal["weighted_normalized_l1"]
    tie_break: Literal["event_id_ascending"]
    nearest_neighbors: int = Field(gt=0)
    minimum_support: int = Field(gt=0)
    scale_distances: tuple[ScaleDistanceConfig, ...]
    levels: tuple[AnalogueLevelConfig, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def frozen_matching_rules_are_consistent(self) -> AnalogueConfig:
        if self.nearest_neighbors < self.minimum_support:
            raise ValueError("nearest_neighbors must cover minimum_support")
        if self.fit_as_of is not None and self.fit_as_of > self.frozen_at:
            raise ValueError("fit_as_of exceeds frozen_at")
        if self.validation_status == "chronologically_validated" and (
            self.fit_as_of is None or self.validation_evidence is None
        ):
            raise ValueError("validated matching requires fit time and validation evidence")
        names = [scale.name for scale in self.scale_distances]
        if len(names) != len(set(names)):
            raise ValueError("duplicate scale distance")
        if len({level.name for level in self.levels}) != len(self.levels):
            raise ValueError("duplicate back-off level")
        previous_fields = set(self.levels[0].fields)
        previous_scales = set(self.levels[0].scales)
        if "event_class" not in previous_fields:
            raise ValueError("first matching level requires event_class")
        for index, level in enumerate(self.levels):
            fields, scales = set(level.fields), set(level.scales)
            if len(fields) != len(level.fields) or len(scales) != len(level.scales):
                raise ValueError("duplicate field or scale in back-off level")
            if not fields <= previous_fields or not scales <= previous_scales:
                raise ValueError("back-off must only relax matching constraints")
            if index and fields == previous_fields and scales == previous_scales:
                raise ValueError("each back-off level must relax at least one constraint")
            if not scales <= set(names):
                raise ValueError("unknown configured scale")
            previous_fields, previous_scales = fields, scales
        if self.levels[-1].name != "global" or previous_fields or previous_scales:
            raise ValueError("final back-off must be unconstrained global")
        return self


class AppConfig(ConfigModel):
    versions: VersionsConfig
    taxonomy: TaxonomyConfig
    runtime: RuntimeConfig
    clustering: ClusteringConfig
    policy: PolicyConfig
    analogues: AnalogueConfig | None = None

    @classmethod
    def load(cls, path: Path) -> AppConfig:
        with path.open("rb") as config_file:
            return cls.model_validate(tomllib.load(config_file))


__all__ = ["AppConfig", "ClusteringConfig"]
