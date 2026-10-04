"""Typed application configuration loaded from TOML."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - local Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

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


class AppConfig(ConfigModel):
    versions: VersionsConfig
    taxonomy: TaxonomyConfig
    runtime: RuntimeConfig
    clustering: ClusteringConfig
    policy: PolicyConfig

    @classmethod
    def load(cls, path: Path) -> AppConfig:
        with path.open("rb") as config_file:
            return cls.model_validate(tomllib.load(config_file))


__all__ = ["AppConfig", "ClusteringConfig"]
