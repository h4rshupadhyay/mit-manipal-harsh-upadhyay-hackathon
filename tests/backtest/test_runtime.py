"""Real production composition with explicitly simulated offline local inference."""

import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from risk_engine.backtest.module import content_hash
from risk_engine.backtest.runtime_inputs import load_training_evidence
from risk_engine.impact.reference_basket import BasketConstruction
from tests.backtest.runtime_fixtures import SimulationBackend, production_inputs


def fitter_inputs(tmp_path, **kwargs):
    from risk_engine.backtest.runtime import ProductionCandidateFitter

    configuration, definition, index, locations, candidate, training = production_inputs(
        tmp_path, **kwargs
    )
    backend = SimulationBackend()
    fitter = ProductionCandidateFitter(
        configuration=configuration,
        definition=definition,
        evidence_index=index,
        model_locations=locations,
        artifact_root=tmp_path / "states",
        backend=backend,
    )
    return fitter, candidate, training, backend


def test_production_fit_composes_real_ports_from_exact_training_membership(tmp_path, monkeypatch):
    from risk_engine.backtest.module import LocalArtifact
    from risk_engine.backtest.runtime_inputs import CaseEvidenceReference, RuntimeEvidenceIndex
    from tests.backtest.runtime_fixtures import seal

    fitter, candidate, training, backend = fitter_inputs(tmp_path)
    index = fitter.evidence_index.model_dump(exclude={"content_hash"})
    index["cases"] = (
        *fitter.evidence_index.cases,
        CaseEvidenceReference(
            case_id="future-case",
            cluster_id="future-cluster",
            artifact=LocalArtifact(
                identity="future-case",
                path=str(tmp_path / "future-case.json"),
                sha256="f" * 64,
                available_at=training.cutoff + timedelta(days=1),
            ),
        ),
    )
    fitter.evidence_index = seal(RuntimeEvidenceIndex, index, "content_hash")
    original = Path.read_bytes

    def guard(path):
        assert path.name != "future-case.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    runtime = fitter.fit(candidate, training, as_of=training.cutoff)
    manifest = runtime.manifest
    assert manifest.configuration_hash == content_hash(fitter.configuration)
    assert manifest.dataset_hash == training.dataset_hash
    assert manifest.impact.training_event_ids == tuple(g.cluster_id for g in training.groups)
    assert manifest.impact.cutpoints == (Decimal(10),) * 3 + (Decimal(20),) * 3 + (Decimal(30),) * 3
    assert manifest.confidence_training_source_ids == tuple(
        sorted(i.source_item_id for c in training.cases for i in c.cluster.items)
    )
    assert manifest.model_identity.event_model_version.endswith("@" + "b" * 40)
    assert manifest.model_identity.sentiment_model_version == "ProsusAI/finbert@" + "a" * 40
    assert (
        manifest.model_identity.entity_linker_version
        == "entity-matcher-v1:" + fitter.definition.catalogue.sha256
    )
    assert len(manifest.artifacts) == 2 and manifest.policy is None
    assert backend.loads > 0
    before = backend.releases
    fitter.close()
    assert backend.releases == before + 1


def test_fit_freeze_uses_actual_availability_microsecond_successors(tmp_path):
    from risk_engine.backtest.runtime_state import load_runtime_state

    fitter, candidate, training, _ = fitter_inputs(tmp_path)
    runtime = fitter.fit(candidate, training, as_of=training.cutoff)
    state = load_runtime_state(runtime.manifest.artifacts[0], cutoff=training.cutoff)
    latest = max(training.availability)
    assert state.impact_calibration.training_end == latest + timedelta(microseconds=1)
    assert state.impact_calibration.calibrated_at == latest + timedelta(microseconds=2)
    assert state.confidence_calibrator.evidence.frozen_at == latest + timedelta(microseconds=2)


@pytest.mark.parametrize(
    "mutation", ["synthetic", "tight", "as_of", "candidate", "config", "definition", "membership"]
)
def test_fit_rejects_invalid_inputs_before_inference(tmp_path, mutation):
    fitter, candidate, training, backend = fitter_inputs(tmp_path)
    as_of = training.cutoff
    if mutation == "synthetic":
        training = training.model_copy(update={"evidence_kind": "synthetic"})
    elif mutation == "tight":
        as_of = max(training.availability) + timedelta(microseconds=2)
        training = training.model_copy(update={"cutoff": as_of})
    elif mutation == "as_of":
        as_of += timedelta(seconds=1)
    elif mutation == "candidate":
        candidate = candidate.model_copy(update={"version": "changed"})
    elif mutation == "config":
        fitter.configuration = fitter.configuration.model_copy(
            update={"frozen_at": training.cutoff}
        )
    elif mutation == "definition":
        fitter.definition = fitter.definition.model_copy(update={"version": "changed"})
    else:
        training = training.model_copy(update={"membership_hash": "0" * 64})
    with pytest.raises(ValueError):
        fitter.fit(candidate, training, as_of=as_of)
    assert backend.loads == 0


def test_confidence_rows_cover_independent_clauses_and_entity_truth(tmp_path):
    from risk_engine.backtest.runtime import _confidence_rows
    from risk_engine.nlp.entity_matcher import EntityMatcher
    from risk_engine.nlp.interfaces import ModelLock
    from risk_engine.nlp.interpret import interpret
    from risk_engine.nlp.local_models import LocalModels
    from risk_engine.risk.confidence import ConfidenceScoreDefinition

    inputs = production_inputs(
        tmp_path,
        texts=(
            "Project-authored entity faces conflict; Second entity faces conflict.",
            "An unlinked clause.",
            "Project-authored entity and Project-authored entity face conflict.",
        ),
    )
    _, definition, index, locations, _, training = inputs
    models = LocalModels(
        ModelLock.model_validate_json(Path(definition.model_lock.path).read_bytes()),
        locations.sentiment_snapshot,
        locations.event_snapshot,
        backend=SimulationBackend(),
    )
    events = {
        item.source_item_id: tuple(
            interpret(
                item, EntityMatcher(Path(definition.catalogue.path)), models.sentiment, models.event
            )
        )
        for case in training.cases
        for item in case.cluster.items
    }
    definition_score = ConfidenceScoreDefinition(
        event_model_version="simulation", entity_linker_version="simulation"
    )
    evidence = load_training_evidence(index, training)
    rows = _confidence_rows(training, evidence, events, definition_score)
    assert len(rows) == 4
    assert sum(row.joint_correct for row in rows) == 2
    unknown = next(row for row in rows if row.source_item_id == "source-02")
    assert unknown.entity_id.startswith("unknown:")
    assert unknown.raw_score.value == 0 and not unknown.entity_correct
    repeated = [row for row in rows if row.source_item_id == "source-03"]
    assert len(repeated) == 1
    # Exact clause coverage, bound source hashes and consistent repeated scores are mandatory.
    bad = dict(events)
    bad["source-01"] = bad["source-01"][:-1]
    with pytest.raises(ValueError, match="target|clause"):
        _confidence_rows(training, evidence, bad, definition_score)
    bad = dict(events)
    event = bad["source-03"][0]
    links = event.entity_links
    bad["source-03"] = (
        event.model_copy(
            update={"entity_links": (links[0], links[1].model_copy(update={"confidence": 0.8}))}
        ),
    )
    with pytest.raises(ValueError, match="conflicting"):
        _confidence_rows(training, evidence, bad, definition_score)


def test_insufficient_classes_are_narrow_and_optimizer_errors_remain_hard(tmp_path, monkeypatch):
    import risk_engine.risk.confidence as confidence
    from risk_engine.backtest.runtime import InsufficientCalibrationEvidence

    fitter, candidate, training, _ = fitter_inputs(tmp_path)

    # Both labels exist in normal inputs; optimizer failure must retain its hard identity.
    class Failed:
        success = False
        x = 1.0

    monkeypatch.setattr(confidence, "minimize_scalar", lambda *args, **kwargs: Failed())
    with pytest.raises(ValueError, match="optimizer") as raised:
        fitter.fit(candidate, training, as_of=training.cutoff)
    assert not isinstance(raised.value, InsufficientCalibrationEvidence)


def scoring_case(training):
    from risk_engine.backtest.module import ReplayInput

    case = training.cases[0]
    return ReplayInput.model_validate({**case.model_dump(), "as_of": training.cutoff})


@pytest.mark.parametrize("count", [3, 4])
def test_scenarios_choose_nearest_median_complete_observed_vector(tmp_path, count):
    fitter, candidate, training, _ = fitter_inputs(tmp_path, count=count)
    runtime = fitter.fit(candidate, training, as_of=training.cutoff)
    case = scoring_case(training)
    signal = runtime.risk_engine.analyze(case.cluster.items, as_of=case.as_of)[0]
    result = runtime.scenarios(signal, case, as_of=case.as_of)
    assert len(result.joint_samples) == min(
        count, fitter.definition.matching[0].config.nearest_neighbors
    )
    assert all(
        tuple(s.factor_id for s in sample) == tuple(sorted(s.factor_id for s in sample))
        for sample in result.joint_samples
    )
    assert result.scenario.shocks in result.joint_samples
    assert len(result.scenario.shocks) == 7
    assert {
        next(shock.value for shock in sample if shock.factor_id == "EQUITY-US")
        for sample in result.joint_samples
    } == {-number / 100 for number in range(1, count + 1)}
    assert all(
        next(shock.value for shock in sample if shock.factor_id == "EQUITY-US")
        == next(shock.value for shock in sample if shock.factor_id == "EQUITY-INDIA")
        for sample in result.joint_samples
    )
    support = json.loads(result.support_reference)
    assert result.support_hash == content_hash(support)
    losses = dict((row["event_id"], Decimal(row["loss"])) for row in support["reference_losses"])
    chosen = min(
        losses,
        key=lambda event_id: (
            abs(losses[event_id] - signal.impact.expected_reference_loss),
            event_id,
        ),
    )
    assert result.scenario.reference_event_ids == (chosen,)
    assert result.fit_hash == runtime.manifest.manifest_hash
    assert result.calibration_hash == runtime.manifest.impact.calibration_hash
    assert result.scenario.calibration_version == signal.impact.calibration_version
    assert result.available_at == runtime.manifest.impact.frozen_at
    assert support["source_hashes"] == sorted(i.content_hash for i in case.cluster.items)
    changed_portfolio = case.portfolio.model_copy(update={"portfolio_id": "another-portfolio"})
    assert (
        runtime.scenarios(
            signal, case.model_copy(update={"portfolio": changed_portfolio}), as_of=case.as_of
        )
        == result
    )


@pytest.mark.parametrize(
    "mutation", ["source", "as_of", "signal_id", "model", "calibration", "cohort", "loss", "median"]
)
def test_scenarios_reject_tampered_audit_or_incomplete_observed_support(tmp_path, mutation):
    import hashlib

    from risk_engine.backtest.module import canonical_bytes
    from risk_engine.domain import ImpactEstimate, RiskSignal

    fitter, candidate, training, _ = fitter_inputs(tmp_path)
    runtime = fitter.fit(candidate, training, as_of=training.cutoff)
    case = scoring_case(training)
    signal = runtime.risk_engine.analyze(case.cluster.items, as_of=case.as_of)[0]
    as_of = case.as_of
    if mutation == "source":
        case = case.model_copy(update={"cluster": training.cases[1].cluster})
    elif mutation == "as_of":
        as_of += timedelta(seconds=1)
    elif mutation == "signal_id":
        signal = signal.model_copy(update={"signal_id": "wrong"})
    elif mutation == "model":
        model = json.loads(signal.versions.model_version)
        model["models"]["event_model_version"] = "wrong"
        signal = signal.model_copy(
            update={
                "versions": signal.versions.model_copy(
                    update={"model_version": canonical_bytes(model).decode()}
                )
            }
        )
    elif mutation == "calibration":
        calibration = json.loads(signal.versions.calibration_version)
        calibration["confidence_calibration"]["temperature"] = 1.0
        signal = signal.model_copy(
            update={
                "versions": signal.versions.model_copy(
                    update={"calibration_version": canonical_bytes(calibration).decode()}
                )
            }
        )
    elif mutation == "median":
        signal = signal.model_copy(
            update={
                "impact": signal.impact.model_copy(
                    update={"expected_reference_loss": Decimal("99")}
                )
            }
        )
    else:
        audit = json.loads(signal.impact.calibration_version)
        if mutation == "cohort":
            audit["cohort_sha256"] = "0" * 64
        else:
            audit["selected_losses"][0]["loss"] = "99"
        audit.pop("audit_sha256")
        audit["audit_sha256"] = hashlib.sha256(
            json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        impact = ImpactEstimate.model_validate(
            {
                **signal.impact.model_dump(),
                "calibration_version": json.dumps(audit, sort_keys=True, separators=(",", ":")),
            }
        )
        calibration = json.loads(signal.versions.calibration_version)
        calibration["impact"]["calibration_version"] = impact.calibration_version
        signal = RiskSignal.model_validate(
            {
                **signal.model_dump(),
                "impact": impact,
                "versions": {
                    **signal.versions.model_dump(),
                    "calibration_version": canonical_bytes(calibration).decode(),
                },
            }
        )
    with pytest.raises(ValueError):
        runtime.scenarios(signal, case, as_of=as_of)


@pytest.mark.parametrize("construction", list(BasketConstruction))
def test_restore_reproduces_fitted_outputs_without_refitting_or_label_reads(
    tmp_path, monkeypatch, construction
):
    from risk_engine.backtest.runtime import ProductionCandidateFitter
    from risk_engine.backtest.runtime_state import load_runtime_state
    from risk_engine.impact.reference_basket import ReferenceBasketBuilder
    from risk_engine.risk.confidence import ConfidenceCalibrator

    fitter, candidate, training, _ = fitter_inputs(tmp_path, construction=construction)
    runtime = fitter.fit(candidate, training, as_of=training.cutoff)
    case = scoring_case(training)
    original_signals = runtime.risk_engine.analyze(case.cluster.items, as_of=case.as_of)
    original_scenario = runtime.scenarios(original_signals[0], case, as_of=case.as_of)
    state = load_runtime_state(runtime.manifest.artifacts[0], cutoff=training.cutoff)
    for artifact in state.training_evidence:
        Path(artifact.path).unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("restore attempted training/refitting/writing")

    monkeypatch.setattr(ConfidenceCalibrator, "fit", forbidden)
    monkeypatch.setattr(ReferenceBasketBuilder, "build", forbidden)
    import risk_engine.backtest.runtime as module

    monkeypatch.setattr(module, "load_training_evidence", forbidden)
    monkeypatch.setattr(module, "write_runtime_state", forbidden)
    import risk_engine.backtest.policy_selection as policy_module
    import risk_engine.impact.reference_basket as basket_module

    monkeypatch.setattr(basket_module, "_equal_risk_weights", forbidden)
    monkeypatch.setattr(basket_module, "minimize", forbidden)
    monkeypatch.setattr(policy_module, "select_policy", forbidden)
    original_read = Path.read_bytes

    def no_training_reads(path):
        if path.name.startswith("case-") or path.name == "index.json":
            raise AssertionError("restore attempted training-label/index read")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", no_training_reads)
    restored_fitter = ProductionCandidateFitter(
        configuration=fitter.configuration,
        definition=fitter.definition,
        evidence_index=None,
        model_locations=fitter.model_locations,
        artifact_root=tmp_path / "never-written",
        backend=SimulationBackend(),
    )
    restored_fitter.preflight_restore(runtime.manifest)
    restored = restored_fitter.restore(runtime.manifest)
    signals = restored.risk_engine.analyze(case.cluster.items, as_of=case.as_of)
    assert restored.manifest.model_dump_json() == runtime.manifest.model_dump_json()
    assert [s.model_dump_json() for s in signals] == [s.model_dump_json() for s in original_signals]
    assert (
        restored.scenarios(signals[0], case, as_of=case.as_of).model_dump_json()
        == original_scenario.model_dump_json()
    )
    assert not (tmp_path / "never-written").exists()


@pytest.mark.parametrize(
    "mutation",
    ["none", "missing", "duplicate", "state", "catalogue", "model", "cutpoints", "configuration"],
)
def test_preflight_restore_verifies_local_prerequisites_without_inference(
    tmp_path, monkeypatch, mutation
):
    from risk_engine.backtest.module import FittedManifest
    from risk_engine.backtest.runtime import ProductionCandidateFitter
    from tests.backtest.runtime_fixtures import seal

    fitter, candidate, training, _ = fitter_inputs(tmp_path)
    runtime = fitter.fit(candidate, training, as_of=training.cutoff)
    locked = runtime.manifest
    backend = SimulationBackend()

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight invoked backend load")

    monkeypatch.setattr(backend, "load", forbidden)
    restorer = ProductionCandidateFitter(
        configuration=fitter.configuration,
        definition=fitter.definition,
        evidence_index=None,
        model_locations=fitter.model_locations,
        artifact_root=tmp_path / "unused",
        backend=backend,
    )
    payload = locked.model_dump(mode="python", exclude={"manifest_hash"})
    if mutation == "missing":
        payload["artifacts"] = (locked.artifacts[1],)
    elif mutation == "duplicate":
        payload["artifacts"] = (*locked.artifacts, locked.artifacts[0])
    elif mutation == "state":
        Path(locked.artifacts[0].path).write_bytes(b"tampered")
    elif mutation == "catalogue":
        Path(locked.artifacts[1].path).write_bytes(b"tampered")
    elif mutation == "model":
        (fitter.model_locations.event_snapshot / "model.safetensors").write_bytes(b"tampered")
    elif mutation == "cutpoints":
        payload["impact"]["cutpoints"] = (Decimal(99),) * 9
    elif mutation == "configuration":
        payload["configuration_hash"] = "0" * 64
    locked = seal(FittedManifest, payload, "manifest_hash")
    if mutation == "none":
        restorer.preflight_restore(locked)
        assert not (tmp_path / "unused").exists()
    else:
        with pytest.raises(ValueError):
            restorer.preflight_restore(locked)


def test_restore_accepts_relocated_identical_snapshots_without_rewriting_artifact_paths(tmp_path):
    import shutil

    from risk_engine.backtest.runtime import ProductionCandidateFitter

    fitter, candidate, training, _ = fitter_inputs(tmp_path)
    runtime = fitter.fit(candidate, training, as_of=training.cutoff)
    sentiment = tmp_path / "relocated-sentiment" / fitter.model_locations.sentiment_snapshot.name
    event = tmp_path / "relocated-event" / fitter.model_locations.event_snapshot.name
    shutil.copytree(fitter.model_locations.sentiment_snapshot, sentiment)
    shutil.copytree(fitter.model_locations.event_snapshot, event)
    locations = fitter.model_locations.model_copy(
        update={"sentiment_snapshot": sentiment, "event_snapshot": event}
    )
    restorer = ProductionCandidateFitter(
        configuration=fitter.configuration,
        definition=fitter.definition,
        evidence_index=None,
        model_locations=locations,
        artifact_root=tmp_path / "unused",
        backend=SimulationBackend(),
    )
    assert restorer.restore(runtime.manifest).manifest == runtime.manifest


@pytest.mark.parametrize("condition", ["classes", "neutral", "corrupt_label"])
def test_only_specific_confidence_insufficiency_conditions_are_abstentions(
    tmp_path, monkeypatch, condition
):
    import risk_engine.backtest.runtime as runtime_module
    from risk_engine.backtest.module import canonical_bytes
    from risk_engine.backtest.runtime import InsufficientCalibrationEvidence
    from risk_engine.backtest.runtime_inputs import RuntimeEvidenceIndex
    from tests.backtest.runtime_fixtures import artifact, seal

    fitter, candidate, training, _ = fitter_inputs(tmp_path)
    if condition in {"classes", "corrupt_label"}:
        descriptors = []
        for reference in fitter.evidence_index.cases:
            payload = json.loads(Path(reference.artifact.path).read_bytes())
            if condition == "classes":
                for target in payload["targets"]:
                    target["actual_entity_ids"] = ["entity:one"]
                    target.pop("label_hash")
                    target["label_hash"] = content_hash(target)
            else:
                payload["targets"][0]["label_hash"] = "0" * 64
            Path(reference.artifact.path).write_bytes(canonical_bytes(payload))
            descriptors.append(
                reference.model_copy(
                    update={
                        "artifact": artifact(
                            Path(reference.artifact.path),
                            reference.artifact.identity,
                            reference.artifact.available_at,
                        )
                    }
                )
            )
        index_payload = fitter.evidence_index.model_dump(exclude={"content_hash"})
        index_payload["cases"] = tuple(descriptors)
        fitter.evidence_index = seal(RuntimeEvidenceIndex, index_payload, "content_hash")
    else:
        # An explicit raw output neutralizes the scoring port; target truth remains independent.
        original = runtime_module.interpret

        def neutral(*args, **kwargs):
            return [
                e.model_copy(update={"classification_confidence": 0.5})
                for e in original(*args, **kwargs)
            ]

        monkeypatch.setattr(runtime_module, "interpret", neutral)
    if condition == "corrupt_label":
        with pytest.raises(ValueError, match="label hash") as raised:
            fitter.fit(candidate, training, as_of=training.cutoff)
        assert not isinstance(raised.value, InsufficientCalibrationEvidence)
    else:
        with pytest.raises(InsufficientCalibrationEvidence):
            fitter.fit(candidate, training, as_of=training.cutoff)


def test_public_held_out_mode_is_refused_and_core_does_not_select_policy(tmp_path):
    from risk_engine.backtest.runtime import ProductionCandidateFitter
    from risk_engine.backtest.runtime_inputs import RuntimeDefinition
    from tests.backtest.runtime_fixtures import seal
    from tests.backtest.test_policy_selection import _spec

    configuration, definition, index, locations, candidate, training = production_inputs(tmp_path)
    payload = definition.model_dump(exclude={"content_hash"})
    payload["policy_selection"] = _spec()
    definition = seal(RuntimeDefinition, payload, "content_hash")
    parameters = {
        **candidate.parameters,
        "runtime_definition_sha256": definition.content_hash,
        "policy_mode": "held-out-material-event-f1-v1",
    }
    candidate = candidate.model_copy(update={"parameters": parameters})
    configuration = configuration.model_copy(update={"candidates": (candidate,)})
    backend = SimulationBackend()
    fitter = ProductionCandidateFitter(
        configuration=configuration,
        definition=definition,
        evidence_index=index,
        model_locations=locations,
        artifact_root=tmp_path / "states",
        backend=backend,
    )
    with pytest.raises(ValueError, match="Task 4"):
        fitter.fit(candidate, training, as_of=training.cutoff)
    assert backend.loads == 0
    runtime = fitter._fit_core(candidate, training, as_of=training.cutoff)
    assert runtime.manifest.candidate == candidate and runtime.manifest.policy is None


def test_fit_rejects_analogue_group_time_mismatch_before_inference(tmp_path):
    fitter, candidate, training, backend = fitter_inputs(tmp_path)
    # Source and group record an earlier publication while the observed analogue
    # still binds the original event time. Both records are individually valid.
    case = training.cases[0]
    event_time = case.cluster.event_time - timedelta(seconds=1)
    item = case.cluster.items[0].model_copy(update={"published_at": event_time})
    cluster = case.cluster.model_copy(update={"event_time": event_time, "items": (item,)})
    case = case.model_copy(update={"cluster": cluster})
    group = training.groups[0].model_copy(update={"event_time": event_time})
    groups = (group, *training.groups[1:])
    training = training.model_copy(
        update={
            "groups": groups,
            "cases": (case, *training.cases[1:]),
            "membership_hash": content_hash(groups),
        }
    )
    with pytest.raises(ValueError, match="event time"):
        fitter.fit(candidate, training, as_of=training.cutoff)
    assert backend.loads == 0


def test_scenarios_reject_wrong_signal_schema_and_missing_complete_support(tmp_path, monkeypatch):
    fitter, candidate, training, _ = fitter_inputs(tmp_path)
    runtime = fitter.fit(candidate, training, as_of=training.cutoff)
    case = scoring_case(training)
    signal = runtime.risk_engine.analyze(case.cluster.items, as_of=case.as_of)[0]
    changed = signal.model_copy(
        update={"versions": signal.versions.model_copy(update={"schema_version": "wrong-schema"})}
    )
    with pytest.raises(ValueError, match="schema"):
        runtime.scenarios(changed, case, as_of=case.as_of)
    original_match = runtime._repository.match

    def no_support(event, as_of):
        cohort = original_match(event, as_of)
        return cohort.model_copy(update={"analogues": (), "method": "hypothetical"})

    monkeypatch.setattr(runtime._repository, "match", no_support)
    with pytest.raises(ValueError, match="observed support"):
        runtime.scenarios(signal, case, as_of=case.as_of)


@pytest.mark.parametrize("late", [False, True])
def test_preregistered_definition_may_follow_training_events_but_not_consumed_availability(
    tmp_path, late
):
    from risk_engine.backtest.runtime import ProductionCandidateFitter
    from risk_engine.backtest.runtime_inputs import RuntimeDefinition
    from tests.backtest.runtime_fixtures import seal

    configuration, definition, index, locations, candidate, training = production_inputs(tmp_path)
    payload = definition.model_dump(exclude={"content_hash"})
    payload["frozen_at"] = (
        max(training.availability) + timedelta(hours=1)
        if late
        else training.groups[0].event_time + timedelta(hours=1)
    )
    definition = seal(RuntimeDefinition, payload, "content_hash")
    candidate = candidate.model_copy(
        update={
            "parameters": {
                **candidate.parameters,
                "runtime_definition_sha256": definition.content_hash,
            }
        }
    )
    configuration = configuration.model_copy(
        update={"frozen_at": definition.frozen_at, "candidates": (candidate,)}
    )
    backend = SimulationBackend()
    fitter = ProductionCandidateFitter(
        configuration=configuration,
        definition=definition,
        evidence_index=index,
        model_locations=locations,
        artifact_root=tmp_path / "states",
        backend=backend,
    )
    if late:
        with pytest.raises(ValueError, match="availability|available"):
            fitter.fit(candidate, training, as_of=training.cutoff)
        assert backend.loads == 0
    else:
        assert (
            fitter.fit(candidate, training, as_of=training.cutoff).manifest.candidate == candidate
        )


@pytest.mark.parametrize("mutation", ["future_publication", "future_retrieval", "corrupt_text"])
def test_scenarios_validate_all_support_consumed_cluster_members(tmp_path, mutation):
    from risk_engine.backtest.module import ReplayInput
    from risk_engine.data.clustering import StoryCluster, _cluster_id
    from risk_engine.domain import SourceItem

    fitter, candidate, training, _ = fitter_inputs(tmp_path)
    runtime = fitter.fit(candidate, training, as_of=training.cutoff)
    case = scoring_case(training)
    representative = case.cluster.items[0]
    signal = runtime.risk_engine.analyze((representative,), as_of=case.as_of)[0]
    payload = representative.model_dump(mode="python")
    payload["source_item_id"] = "source-99"
    payload["published_at"] += timedelta(seconds=1)
    payload["retrieved_at"] += timedelta(seconds=1)
    member = SourceItem.model_validate(payload)
    ids = (representative.source_item_id, member.source_item_id)

    def with_member(source):
        cluster = StoryCluster(
            cluster_id=_cluster_id(ids),
            event_time=representative.published_at,
            representative_source_item_id=representative.source_item_id,
            source_item_ids=ids,
            items=(representative, source),
        )
        return ReplayInput.model_validate({**case.model_dump(mode="python"), "cluster": cluster})

    valid = runtime.scenarios(signal, with_member(member), as_of=case.as_of)
    assert member.content_hash in json.loads(valid.support_reference)["source_hashes"]
    if mutation == "future_publication":
        member = member.model_copy(
            update={
                "published_at": case.as_of + timedelta(seconds=1),
                "retrieved_at": case.as_of + timedelta(seconds=2),
            }
        )
    elif mutation == "future_retrieval":
        member = member.model_copy(update={"retrieved_at": case.as_of + timedelta(seconds=1)})
    else:
        member = member.model_copy(update={"text": member.text + " changed bytes"})
    with pytest.raises(ValueError, match="Source Item.*unavailable|Source Item.*hash"):
        runtime.scenarios(signal, with_member(member), as_of=case.as_of)
