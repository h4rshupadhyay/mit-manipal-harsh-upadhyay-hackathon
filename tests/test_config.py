from pathlib import Path

import pytest
from pydantic import ValidationError

from risk_engine.config import AppConfig, ClusteringConfig
from risk_engine.domain import EventClass

EVENT_CLASSES = [event_class.value for event_class in EventClass]


def write_config(
    path: Path,
    *,
    event_classes: list[str] | None = None,
    include_model_version: bool = True,
    include_calibration_version: bool = True,
) -> Path:
    version_lines = ['schema = "1.0.0"']
    if include_model_version:
        version_lines.append('model = "models-v1"')
    if include_calibration_version:
        version_lines.append('calibration = "calibration-v1"')
    version_lines.append('valuation_rules = "valuation-v1"')
    classes = event_classes if event_classes is not None else EVENT_CLASSES
    rendered_classes = ",\n  ".join(f'"{value}"' for value in classes)
    path.write_text(
        "[versions]\n"
        + "\n".join(version_lines)
        + "\n\n[taxonomy]\nevent_classes = [\n  "
        + rendered_classes
        + "\n]\n\n[runtime]\noffline_replay = true\n"
        + "\n[clustering]\nsimilarity_threshold = 0.8\nmax_time_delta_hours = 24\n"
        + "\n[policy]\nimpact_threshold = 8\n",
        encoding="utf-8",
    )
    return path


def test_load_reads_typed_configuration_with_standard_library_toml(
    tmp_path: Path,
) -> None:
    config = AppConfig.load(write_config(tmp_path / "valid.toml"))

    assert config.taxonomy.event_classes == tuple(EventClass)
    assert config.versions.model == "models-v1"
    assert config.versions.calibration == "calibration-v1"
    assert config.runtime.offline_replay is True
    assert config.clustering.similarity_threshold == 0.8
    assert config.clustering.max_time_delta_hours == 24
    assert config.policy.impact_threshold == 8


def test_load_rejects_unknown_event_class(tmp_path: Path) -> None:
    path = write_config(tmp_path / "unknown.toml", event_classes=[*EVENT_CLASSES, "Rumour"])

    with pytest.raises(ValidationError):
        AppConfig.load(path)


@pytest.mark.parametrize(
    ("include_model_version", "include_calibration_version"),
    [(False, True), (True, False)],
)
def test_load_rejects_missing_required_versions(
    tmp_path: Path,
    include_model_version: bool,
    include_calibration_version: bool,
) -> None:
    path = write_config(
        tmp_path / "missing-version.toml",
        include_model_version=include_model_version,
        include_calibration_version=include_calibration_version,
    )

    with pytest.raises(ValidationError):
        AppConfig.load(path)


def test_committed_default_configuration_is_valid() -> None:
    config = AppConfig.load(Path("config/default.toml"))

    assert config.taxonomy.event_classes == tuple(EventClass)
    assert config.versions.model
    assert config.versions.calibration
    assert config.clustering.similarity_threshold == 0.8


@pytest.mark.parametrize(
    "values",
    [
        {"similarity_threshold": 0.0, "max_time_delta_hours": 24},
        {"similarity_threshold": 1.01, "max_time_delta_hours": 24},
        {"similarity_threshold": 0.8, "max_time_delta_hours": 0},
    ],
)
def test_clustering_configuration_rejects_invalid_bounds(
    values: dict[str, float | int],
) -> None:
    with pytest.raises(ValidationError):
        ClusteringConfig.model_validate(values)
