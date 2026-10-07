"""Verified local input contracts using project-authored case simulations."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from risk_engine.backtest.module import (
    CandidateSpec,
    ObservedOutcome,
    ReferenceLoss,
    ReplayInput,
    TrainingPartition,
    canonical_bytes,
    content_hash,
)
from risk_engine.backtest.runtime_inputs import (
    TrainingCaseEvidence,
    load_runtime_definition,
    load_runtime_evidence_index,
    load_training_evidence,
    resolve_candidate,
)
from tests.backtest.runtime_fixtures import (
    TERMS,
    definition_payload,
    evidence_payload,
    model_and_catalogue_artifacts,
    seal,
    training_pair,
    write_document,
)
from tests.impact.test_analogues import PAST


def prepared(tmp_path: Path):
    training, analogues = training_pair()
    descriptors = []
    for case, outcome, analogue in zip(training.cases, training.outcomes, analogues, strict=True):
        descriptor = write_document(
            tmp_path / f"{case.case_id}.json", evidence_payload(case, outcome, analogue)
        )
        descriptors.append(
            dict(case_id=case.case_id, cluster_id=case.cluster.cluster_id, artifact=descriptor)
        )
    payload = dict(
        schema_version="production-runtime-evidence-index-v1",
        dataset_hash=training.dataset_hash,
        source_terms=TERMS,
        cases=tuple(descriptors),
    )
    payload["content_hash"] = content_hash(payload)
    index_path = tmp_path / "index.json"
    write_document(index_path, payload)
    return training, payload, index_path


def valid_definition(tmp_path: Path):
    model_lock, catalogue = model_and_catalogue_artifacts(tmp_path)
    payload = definition_payload(model_lock, catalogue)
    payload["windows"] = (
        {"key": "leading", "window": {"start": -1, "end": 0}},
        {"key": "session-zero", "window": {"start": 0, "end": 0}},
        {"key": "trailing", "window": {"start": 0, "end": 1}},
    )
    payload["content_hash"] = content_hash(payload)
    path = tmp_path / "definition.json"
    write_document(path, payload)
    return load_runtime_definition(path)


def candidate(definition_hash: str, *, window: str = "leading", **parameters):
    values = dict(
        runtime_definition_sha256=definition_hash,
        event_window=window,
        confidence_fit="binary-temperature-logit-clip1e-12-v1",
        cutpoint_rule="nearest-rank-lower-ties",
        scenario_choice="nearest-median-reference-loss-event-id-v1",
        policy_mode="unselected",
    )
    values.update(parameters)
    return CandidateSpec(
        candidate_id="project-authored-runtime-candidate",
        version="v1",
        parameters=values,
        complexity_dimensions=("rules",),
        complexity=(1,),
        event_window_days=2,
        basket_convention="basket-one",
        matching_convention="matching-one",
    )


def test_training_loader_opens_only_partition_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training, index_payload, index_path = prepared(tmp_path)
    unrelated = tmp_path / "case-99.json"
    index_payload["cases"] = (
        *index_payload["cases"],
        dict(
            case_id="case-99",
            cluster_id="cluster-99",
            artifact={
                "identity": "case-99",
                "path": str(unrelated),
                "sha256": "9" * 64,
                "available_at": training.cutoff,
            },
        ),
    )
    index_payload["content_hash"] = content_hash(
        {k: v for k, v in index_payload.items() if k != "content_hash"}
    )
    write_document(index_path, index_payload)
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == unrelated:
            raise AssertionError("unrelated case opened")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    index = load_runtime_evidence_index(index_path)
    evidence = load_training_evidence(index, training)

    assert tuple(row.case_id for row in evidence) == tuple(case.case_id for case in training.cases)


def test_training_loader_rejects_synthetic_partition(tmp_path: Path) -> None:
    training, _, index_path = prepared(tmp_path)
    synthetic = TrainingPartition.model_validate(
        {
            **training.model_dump(mode="python"),
            "evidence_kind": "synthetic",
        }
    )
    with pytest.raises(ValueError, match="empirical"):
        load_training_evidence(load_runtime_evidence_index(index_path), synthetic)


def test_training_loader_rejects_repeated_partition_membership(tmp_path: Path) -> None:
    training, _, index_path = prepared(tmp_path)
    duplicate_groups = (training.groups[0], training.groups[0])
    duplicate = TrainingPartition.model_validate(
        {
            **training.model_dump(mode="python"),
            "groups": duplicate_groups,
            "cases": (training.cases[0], training.cases[0]),
            "outcomes": (training.outcomes[0], training.outcomes[0]),
            "membership_hash": content_hash(duplicate_groups),
        }
    )
    with pytest.raises(ValueError, match="duplicate training"):
        load_training_evidence(load_runtime_evidence_index(index_path), duplicate)


@pytest.mark.parametrize("resource", ["source", "market", "reference_loss"])
def test_training_loader_rejects_resources_later_than_outcome(
    tmp_path: Path, resource: str
) -> None:
    training, _, index_path = prepared(tmp_path)
    cases = list(training.cases)
    outcomes = list(training.outcomes)
    if resource == "source":
        case_data = cases[0].model_dump(mode="python")
        case_data["as_of"] = PAST + timedelta(hours=8)
        case_data["cluster"]["items"][0]["retrieved_at"] = PAST + timedelta(hours=7)
        cases[0] = ReplayInput.model_validate(case_data)
    elif resource == "market":
        case_data = cases[0].model_dump(mode="python")
        case_data["as_of"] = PAST + timedelta(hours=8)
        case_data["market_available_at"] = PAST + timedelta(hours=7)
        cases[0] = ReplayInput.model_validate(case_data)
    else:
        outcome_data = outcomes[0].model_dump(mode="python", exclude={"evidence_hash"})
        outcome_data["reference_losses"] = (
            ReferenceLoss(
                basket_hash="b" * 64,
                loss=Decimal("1"),
                currency="USD",
                derivation_reference="project-authored contract simulation",
                derivation_hash="c" * 64,
                available_at=PAST + timedelta(hours=7),
            ),
        )
        outcome_data["reference_loss_absence_reason"] = None
        outcomes[0] = seal(ObservedOutcome, outcome_data, "evidence_hash")
        sidecar = tmp_path / "case-01.json"
        evidence = json.loads(sidecar.read_text(encoding="utf-8"))
        evidence["outcome_hash"] = content_hash(outcomes[0])
        sidecar.write_bytes(canonical_bytes(evidence))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["cases"][0]["artifact"]["sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        index["content_hash"] = content_hash(
            {key: value for key, value in index.items() if key != "content_hash"}
        )
        write_document(index_path, index)
    changed = TrainingPartition.model_validate(
        {**training.model_dump(mode="python"), "cases": tuple(cases), "outcomes": tuple(outcomes)}
    )
    with pytest.raises(ValueError, match="outcome availability"):
        load_training_evidence(load_runtime_evidence_index(index_path), changed)


def test_training_loader_rejects_changed_sidecar_bytes(tmp_path: Path) -> None:
    training, _, index_path = prepared(tmp_path)
    sidecar = tmp_path / "case-01.json"
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(ValueError, match="byte hash"):
        load_training_evidence(load_runtime_evidence_index(index_path), training)


def test_training_loader_uses_case_order_when_outcomes_are_reordered(tmp_path: Path) -> None:
    training, _, index_path = prepared(tmp_path)
    reordered = TrainingPartition.model_validate(
        {
            **training.model_dump(mode="python"),
            "outcomes": tuple(reversed(training.outcomes)),
        }
    )
    evidence = load_training_evidence(load_runtime_evidence_index(index_path), reordered)
    assert tuple(row.case_id for row in evidence) == tuple(case.case_id for case in training.cases)


def test_training_loader_parses_the_bytes_it_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training, _, index_path = prepared(tmp_path)
    sidecar = tmp_path / "case-01.json"
    verified = sidecar.read_bytes()
    altered = json.loads(verified)
    altered["targets"][0]["actual_event_class"] = "Credit/Default"
    target = altered["targets"][0]
    target["label_hash"] = content_hash(
        {key: value for key, value in target.items() if key != "label_hash"}
    )
    altered_bytes = canonical_bytes(altered)
    original_read_bytes = Path.read_bytes
    reads = 0

    def switched_read_bytes(path: Path) -> bytes:
        nonlocal reads
        if path == sidecar:
            reads += 1
            return verified if reads == 1 else altered_bytes
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", switched_read_bytes)
    evidence = load_training_evidence(load_runtime_evidence_index(index_path), training)
    assert evidence[0].targets[0].actual_event_class.value == "Geopolitical"
    assert reads == 1


@pytest.mark.parametrize(
    "mutation",
    ["outcome", "source", "duplicate_source", "future_label", "late_analogue", "incomplete_roles"],
)
def test_training_loader_rejects_inconsistent_case_evidence(tmp_path: Path, mutation: str) -> None:
    training, _, index_path = prepared(tmp_path)
    sidecar = tmp_path / "case-01.json"
    evidence = json.loads(sidecar.read_text(encoding="utf-8"))
    if mutation == "outcome":
        evidence["outcome_hash"] = "0" * 64
    elif mutation == "source":
        evidence["source_items"][0]["content_hash"] = "0" * 64
    elif mutation == "duplicate_source":
        evidence["source_items"].append(evidence["source_items"][0])
    elif mutation == "future_label":
        evidence["targets"][0]["label_available_at"] = (PAST + timedelta(hours=4)).isoformat()
        target = evidence["targets"][0]
        target["label_hash"] = content_hash({k: v for k, v in target.items() if k != "label_hash"})
    elif mutation == "late_analogue":
        evidence["analogue"]["available_at"] = (PAST + timedelta(hours=7)).isoformat()
    else:
        evidence["analogue"]["factor_roles"].pop()
    write_document(sidecar, evidence)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["cases"][0]["artifact"]["sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    index["content_hash"] = content_hash({k: v for k, v in index.items() if k != "content_hash"})
    write_document(index_path, index)
    with pytest.raises(ValueError):
        load_training_evidence(load_runtime_evidence_index(index_path), training)


def test_evidence_index_rejects_duplicate_clusters_and_relative_descriptors(tmp_path: Path) -> None:
    _, payload, index_path = prepared(tmp_path)
    records = [dict(row) for row in payload["cases"]]
    records[1]["cluster_id"] = records[0]["cluster_id"]
    payload["cases"] = tuple(records)
    payload["content_hash"] = content_hash(
        {k: v for k, v in payload.items() if k != "content_hash"}
    )
    write_document(index_path, payload)
    with pytest.raises(ValueError):
        load_runtime_evidence_index(index_path)
    records[1]["cluster_id"] = "another-cluster"
    records[1]["artifact"] = dict(records[1]["artifact"].model_dump(), path="relative.json")
    payload["cases"] = tuple(records)
    payload["content_hash"] = content_hash(
        {k: v for k, v in payload.items() if k != "content_hash"}
    )
    write_document(index_path, payload)
    with pytest.raises(ValueError, match="absolute"):
        load_runtime_evidence_index(index_path)


def test_definition_rejects_resource_newer_than_freeze(tmp_path: Path) -> None:
    model_lock, catalogue = model_and_catalogue_artifacts(tmp_path)
    payload = definition_payload(model_lock, catalogue)
    payload["catalogue"] = catalogue.model_copy(update={"available_at": PAST})
    payload["content_hash"] = content_hash(payload)
    path = tmp_path / "definition.json"
    write_document(path, payload)
    with pytest.raises(ValueError, match="frozen|available"):
        load_runtime_definition(path)


def test_runtime_input_json_normalizes_nested_aware_timestamps_to_utc(tmp_path: Path) -> None:
    definition = valid_definition(tmp_path)
    payload = definition.model_dump(mode="json")
    payload["frozen_at"] = "2025-06-01T17:30:00+05:30"
    payload["catalogue"]["available_at"] = "2025-06-01T17:30:00+05:30"
    restored = type(definition).model_validate_json(json.dumps(payload))
    assert restored.frozen_at.utcoffset() == timedelta(0)
    assert restored.catalogue.available_at.utcoffset() == timedelta(0)
    training, _, _ = prepared(tmp_path)
    sidecar = json.loads((tmp_path / "case-01.json").read_text(encoding="utf-8"))
    sidecar["targets"][0]["label_available_at"] = "2025-06-02T20:30:00+05:30"
    sidecar["analogue"]["available_at"] = "2025-06-02T22:30:00+05:30"
    parsed = TrainingCaseEvidence.model_validate_json(json.dumps(sidecar))
    assert parsed.case_id == training.cases[0].case_id
    assert parsed.targets[0].label_available_at.utcoffset() == timedelta(0)
    assert parsed.analogue.available_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize("name", ["model-lock.json", "catalogue.csv"])
def test_definition_rejects_changed_resource_bytes(tmp_path: Path, name: str) -> None:
    valid_definition(tmp_path)
    resource = tmp_path / name
    resource.write_bytes(resource.read_bytes() + b" ")
    with pytest.raises(ValueError, match="byte hash"):
        load_runtime_definition(tmp_path / "definition.json")


def test_candidate_parameters_have_exact_executable_meanings(tmp_path: Path) -> None:
    definition = valid_definition(tmp_path)
    leading = resolve_candidate(candidate(definition.content_hash), definition)
    trailing = resolve_candidate(candidate(definition.content_hash, window="trailing"), definition)
    assert leading.window.window.start == -1
    assert trailing.window.window.start == 0
    assert (
        leading.window.window.end - leading.window.window.start
        == trailing.window.window.end - trailing.window.window.start
    )
    assert leading.basket == definition.baskets[0]
    assert leading.matching == definition.matching[0]
    assert set(leading.candidate.parameters) == {
        "runtime_definition_sha256",
        "event_window",
        "confidence_fit",
        "cutpoint_rule",
        "scenario_choice",
        "policy_mode",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"event_window": "missing"},
        {"runtime_definition_sha256": "0" * 64},
        {"confidence_fit": "unknown"},
        {"cutpoint_rule": "unknown"},
        {"scenario_choice": "unknown"},
        {"policy_mode": "held-out-material-event-f1-v1"},
        {"extra_scalar": 7},
    ],
)
def test_candidate_rejects_undeclared_or_unresolved_parameters(
    tmp_path: Path, change: dict
) -> None:
    definition = valid_definition(tmp_path)
    with pytest.raises(ValueError):
        resolve_candidate(candidate(definition.content_hash, **change), definition)


def test_candidate_rejects_missing_key_and_horizon_mismatch(tmp_path: Path) -> None:
    definition = valid_definition(tmp_path)
    valid = candidate(definition.content_hash)
    parameters = dict(valid.parameters)
    parameters.pop("scenario_choice")
    with pytest.raises(ValueError):
        resolve_candidate(
            CandidateSpec.model_validate({**valid.model_dump(), "parameters": parameters}),
            definition,
        )
    with pytest.raises(ValueError, match="horizon"):
        resolve_candidate(
            CandidateSpec.model_validate({**valid.model_dump(), "event_window_days": 1}), definition
        )
