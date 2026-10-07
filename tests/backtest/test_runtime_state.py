"""Immutable state contracts with project-authored contract simulations."""

import hashlib
from pathlib import Path

import pytest

from risk_engine.backtest.module import canonical_bytes, content_hash
from tests.backtest.runtime_fixtures import frozen_state_payload, seal


def test_frozen_state_exclusive_write_and_exact_reuse(tmp_path):
    from risk_engine.backtest.runtime_state import (
        FrozenRuntimeState,
        load_runtime_state,
        write_runtime_state,
    )

    state = seal(FrozenRuntimeState, frozen_state_payload(tmp_path), "content_hash")
    assert state.content_hash == content_hash(
        state.model_dump(mode="json", exclude={"content_hash"})
    )
    descriptor = write_runtime_state(state, tmp_path / "artifacts")
    data = Path(descriptor.path).read_bytes()
    assert data == canonical_bytes(state.model_dump(mode="json"))
    assert Path(descriptor.path).name == state.content_hash + ".json"
    assert descriptor.sha256 == hashlib.sha256(data).hexdigest()
    assert descriptor.available_at == state.impact_calibration.calibrated_at
    assert b"manifest_hash" not in data and b'"text"' not in data
    assert write_runtime_state(state, tmp_path / "artifacts") == descriptor
    # Training labels are retained provenance; restore never reopens them.
    for artifact in state.training_evidence:
        Path(artifact.path).unlink()
    assert load_runtime_state(descriptor, cutoff=state.fit_cutoff) == state
    for changed in (b"{", data + b" "):
        Path(descriptor.path).write_bytes(changed)
        with pytest.raises(ValueError, match="existing|immutable"):
            write_runtime_state(state, tmp_path / "artifacts")


@pytest.mark.parametrize(
    "change", ["dataset", "definition", "model", "membership", "policy", "scenario"]
)
def test_frozen_state_rejects_rehashed_cross_identity_mismatch(tmp_path, change):
    from risk_engine.backtest.runtime_state import FrozenRuntimeState

    payload = frozen_state_payload(tmp_path)
    if change == "dataset":
        payload["dataset_hash"] = "f" * 64
    elif change == "definition":
        payload["runtime_definition_hash"] = "f" * 64
    elif change == "model":
        payload["model_identity"] = payload["model_identity"].model_copy(
            update={"event_model_version": "wrong"}
        )
    elif change == "membership":
        payload["training_groups"] = tuple(reversed(payload["training_groups"]))
    elif change == "policy":
        payload["policy_absence_reason"] = None
    else:
        payload["scenario_choice"] = "invented"
    with pytest.raises(ValueError):
        seal(FrozenRuntimeState, payload, "content_hash")


@pytest.mark.parametrize("resource", ["catalogue", "model_lock", "model_identity"])
def test_state_load_verifies_runtime_resource_bytes_and_pinned_identity(tmp_path, resource):
    from risk_engine.backtest.module import LocalArtifact
    from risk_engine.backtest.runtime_state import (
        FrozenRuntimeState,
        load_runtime_state,
        write_runtime_state,
    )

    state = seal(FrozenRuntimeState, frozen_state_payload(tmp_path), "content_hash")
    descriptor = write_runtime_state(state, tmp_path / "states")
    if resource == "model_identity":
        # Rehashed state alone cannot replace verified pinned model identity.
        import json

        payload = json.loads(Path(descriptor.path).read_bytes())
        payload["model_identity"]["sentiment_model_version"] = "wrong@revision"
        payload.pop("content_hash")
        payload["content_hash"] = content_hash(payload)
        data = canonical_bytes(payload)
        path = tmp_path / "states" / f"{payload['content_hash']}.json"
        path.write_bytes(data)
        descriptor = LocalArtifact(
            identity=descriptor.identity,
            path=str(path),
            sha256=hashlib.sha256(data).hexdigest(),
            available_at=descriptor.available_at,
        )
    else:
        artifact = getattr(state, resource)
        Path(artifact.path).write_bytes(b"changed runtime prerequisite")
    with pytest.raises(ValueError, match="hash|identity"):
        load_runtime_state(descriptor, cutoff=state.fit_cutoff)


@pytest.mark.parametrize("selected", [True, False])
def test_policy_audit_survives_restore_without_selection_or_training_reads(
    tmp_path, monkeypatch, selected
):
    import risk_engine.backtest.policy_selection as policy_module
    import risk_engine.backtest.runtime as module
    from risk_engine.backtest.runtime import ProductionCandidateFitter
    from risk_engine.impact.reference_basket import ReferenceBasketBuilder
    from risk_engine.risk.confidence import ConfidenceCalibrator
    from tests.backtest.runtime_fixtures import SimulationBackend
    from tests.backtest.test_runtime import fitter_inputs, policy_spec

    spec = policy_spec(train_groups=10, **({} if selected else {"alert_budget": 0}))
    fitter, candidate, training, _ = fitter_inputs(tmp_path, count=12, policy_spec=spec)
    original = fitter.fit(candidate, training, as_of=training.cutoff)
    assert (original.manifest.policy is not None) == selected
    audit = original._state.policy_audit
    assert audit is not None and (audit.selected is not None) == selected
    for descriptor in original._state.training_evidence:
        Path(descriptor.path).unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("restore attempted fit/selection/training read/write")

    monkeypatch.setattr(ConfidenceCalibrator, "fit", forbidden)
    monkeypatch.setattr(ReferenceBasketBuilder, "build", forbidden)
    monkeypatch.setattr(module, "load_training_evidence", forbidden)
    monkeypatch.setattr(module, "write_runtime_state", forbidden)
    monkeypatch.setattr(module, "select_policy", forbidden)
    monkeypatch.setattr(policy_module, "select_policy", forbidden)
    monkeypatch.setattr(ProductionCandidateFitter, "fit", forbidden)
    monkeypatch.setattr(ProductionCandidateFitter, "_fit_core", forbidden)
    original_read = Path.read_bytes

    def no_labels(path):
        if path.name.startswith("case-"):
            forbidden()
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", no_labels)
    restore_fitter = ProductionCandidateFitter(
        configuration=fitter.configuration,
        definition=fitter.definition,
        evidence_index=None,
        model_locations=fitter.model_locations,
        artifact_root=tmp_path / "not-created",
        backend=SimulationBackend(),
    )
    restore_fitter.preflight_restore(original.manifest)
    restored = restore_fitter.restore(original.manifest)
    assert restored.manifest.model_dump_json() == original.manifest.model_dump_json()
    assert restored._state.policy_audit == audit
    assert not (tmp_path / "not-created").exists()
    if not selected:
        assert (
            restored.manifest.policy_absence_reason == "all candidates exceed declared alert budget"
        )


@pytest.mark.parametrize("change", ["omitted_fold", "renamed_fold", "missing_audit"])
def test_restore_rejects_rehashed_infeasible_audit_with_changed_fold_plan(
    tmp_path, monkeypatch, change
):
    from risk_engine.backtest.module import FittedManifest
    from risk_engine.backtest.policy_selection import select_policy
    from risk_engine.backtest.runtime_state import FrozenRuntimeState, write_runtime_state
    from risk_engine.risk.module import RiskEngine
    from tests.backtest.test_runtime import fitter_inputs, policy_spec

    fitter, candidate, training, _ = fitter_inputs(tmp_path, count=4, policy_spec=policy_spec())
    monkeypatch.setattr(RiskEngine, "analyze", lambda *args, **kwargs: [])
    original = fitter.fit(candidate, training, as_of=training.cutoff)
    audit = original._state.policy_audit
    assert original.manifest.policy is None and len(audit.folds) == 2
    if change == "missing_audit":
        changed_audit = None
    else:
        if change == "omitted_fold":
            folds = audit.folds[:1]
            exclusions = audit.exclusions[:1]
        else:
            folds = tuple(
                f.model_copy(update={"fold_id": f"replacement-{n}"})
                for n, f in enumerate(audit.folds)
            )
            exclusions = tuple(
                e.model_copy(update={"fold_id": f.fold_id})
                for e, f in zip(audit.exclusions, folds, strict=True)
            )
        changed_audit = select_policy(
            (),
            exclusions,
            audit.spec,
            folds=folds,
            parent_training_hash=audit.parent_training_hash,
            runtime_definition_hash=audit.runtime_definition_hash,
            configuration_hash=audit.configuration_hash,
        )
        assert changed_audit.absence_reason == original.manifest.policy_absence_reason
    state = seal(
        FrozenRuntimeState,
        original._state.model_dump(mode="json", exclude={"content_hash"})
        | {"policy_audit": changed_audit},
        "content_hash",
    )
    descriptor = write_runtime_state(state, tmp_path / "altered-state")
    locked = seal(
        FittedManifest,
        original.manifest.model_dump(exclude={"manifest_hash"})
        | {"artifacts": (descriptor, fitter.definition.catalogue)},
        "manifest_hash",
    )
    with pytest.raises(ValueError, match="policy.*audit|policy.*fold"):
        fitter.preflight_restore(locked)
