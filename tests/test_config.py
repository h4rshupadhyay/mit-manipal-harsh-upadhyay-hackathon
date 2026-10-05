from pathlib import Path

import pytest
from pydantic import ValidationError

from risk_engine.config import AnalogueConfig, AppConfig, ClusteringConfig
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


def analogue_payload() -> dict:
    config = AppConfig.load(Path("config/default.toml")).analogues
    assert config is not None
    return config.model_dump()


def test_analogue_default_is_explicitly_unvalidated_and_older_config_remains_loadable(
    tmp_path: Path,
) -> None:
    matching = AppConfig.load(Path("config/default.toml")).analogues
    assert matching is not None
    assert matching.validation_status == "bootstrap_unvalidated"
    assert matching.fit_as_of is None
    assert matching.validation_evidence is None
    assert AppConfig.load(write_config(tmp_path / "older.toml")).analogues is None


@pytest.mark.parametrize(
    "damage",
    [
        "unknown_top_level",
        "unknown_matching_field",
        "k_below_support",
        "zero_normalizer",
        "nan_weight",
        "missing_version",
        "validated_without_fit",
        "validated_without_evidence",
        "non_global_end",
        "tightening_backoff",
        "duplicate_constraints",
    ],
)
def test_analogue_config_rejects_inconsistent_frozen_rules(damage: str) -> None:
    payload = analogue_payload()
    if damage == "unknown_top_level":
        payload["unexpected"] = True
    elif damage == "unknown_matching_field":
        payload["levels"][0]["fields"] = ("sentiment",)
    elif damage == "k_below_support":
        payload["nearest_neighbors"] = 1
    elif damage == "zero_normalizer":
        payload["scale_distances"][0]["normalizer"] = 0
    elif damage == "nan_weight":
        payload["scale_distances"][0]["weight"] = float("nan")
    elif damage == "missing_version":
        del payload["version"]
    elif damage == "validated_without_fit":
        payload.update(validation_status="chronologically_validated", validation_evidence="proof")
    elif damage == "validated_without_evidence":
        payload.update(
            validation_status="chronologically_validated", fit_as_of=payload["frozen_at"]
        )
    elif damage == "non_global_end":
        payload["levels"][-1]["fields"] = ("event_class",)
    elif damage == "tightening_backoff":
        payload["levels"][2]["fields"] = (*payload["levels"][2]["fields"], "subtype")
    elif damage == "duplicate_constraints":
        payload["levels"][1]["fields"] = payload["levels"][0]["fields"]
    with pytest.raises(ValidationError):
        AnalogueConfig.model_validate(payload)
