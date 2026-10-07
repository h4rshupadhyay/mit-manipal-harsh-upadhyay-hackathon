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
