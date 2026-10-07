"""CLI mechanics with project-authored local contracts, not empirical fit quality."""

import json
import socket
from pathlib import Path

import pytest

from risk_engine.backtest.module import canonical_bytes
from scripts import run_backtest as runner
from tests.backtest.runtime_fixtures import production_inputs
from tests.backtest.test_backtest_module import write_cli_inputs


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline CLI attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


def location_inputs(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    configuration, definition, index, locations, _, _ = production_inputs(root)
    for name, record in (("definition.json", definition), ("index.json", index)):
        (root / name).write_bytes(canonical_bytes(record))
    location = dict(
        schema_version="production-runtime-locations-v1",
        definition_path="definition.json",
        evidence_index_path="index.json",
        artifact_root="states",
        models=dict(
            sentiment_snapshot=str(locations.sentiment_snapshot.relative_to(root)),
            event_snapshot=str(locations.event_snapshot.relative_to(root)),
            device="cpu",
        ),
    )
    path = root / "runtime.json"
    path.write_text(json.dumps(location))
    return path, configuration, definition, index, locations


def test_runtime_cli_resolves_locations_against_config_file(tmp_path, monkeypatch):
    path, configuration, definition, index, locations = location_inputs(tmp_path)
    _, _, _, args = write_cli_inputs(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_bytes(canonical_bytes(configuration))
    monkeypatch.chdir(tmp_path)
    received = []

    # Constructor is the explicit application boundary; selection itself is covered below.
    def construct(**kwargs):
        received.append(kwargs)
        raise ValueError("construction boundary reached")

    monkeypatch.setattr(runner, "ProductionCandidateFitter", construct, raising=False)
    assert runner.main([*args, "--runtime-config", str(path)]) == 1
    assert received == [
        dict(
            configuration=configuration,
            definition=definition,
            evidence_index=index,
            model_locations=locations,
            artifact_root=path.parent / "states",
        )
    ]
    assert not (tmp_path / "selection").exists()


def test_runtime_selection_requires_evidence_index_before_construction(
    tmp_path, monkeypatch, capsys
):
    path, *_ = location_inputs(tmp_path)
    _, _, _, args = write_cli_inputs(tmp_path)
    payload = json.loads(path.read_text())
    payload["evidence_index_path"] = None
    path.write_text(json.dumps(payload))

    def forbidden(**kwargs):
        pytest.fail("selection constructed without its evidence index")

    monkeypatch.setattr(runner, "ProductionCandidateFitter", forbidden, raising=False)
    assert runner.main([*args, "--runtime-config", str(path)]) == 1
    assert "evidence_index_path" in capsys.readouterr().err


def test_runtime_config_and_injected_fitter_are_mutually_exclusive(tmp_path, monkeypatch, capsys):
    _, _, fitter, args = write_cli_inputs(tmp_path)

    def forbidden(path):
        pytest.fail("mutually exclusive composition read local inputs")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert (
        runner.main([*args, "--runtime-config", str(tmp_path / "missing.json")], fitter=fitter) == 2
    )
    assert "mutually exclusive" in capsys.readouterr().err
    assert fitter.fits == []


def concrete_cli_inputs(tmp_path, monkeypatch):
    from risk_engine.backtest.runtime import ProductionCandidateFitter
    from tests.backtest.runtime_fixtures import SimulationBackend, runtime_cli_contracts

    root = tmp_path / "contracts"
    root.mkdir()
    dataset, config, period, definition, index, locations = runtime_cli_contracts(root)
    for name, record in (
        ("dataset.json", dataset),
        ("config.json", config),
        ("period.json", period),
        ("definition.json", definition),
        ("index.json", index),
    ):
        (root / name).write_bytes(canonical_bytes(record))
    path = root / "runtime.json"
    path.write_bytes(
        canonical_bytes(
            dict(
                schema_version="production-runtime-locations-v1",
                definition_path="definition.json",
                evidence_index_path="index.json",
                artifact_root="states",
                models=locations.model_dump(mode="json"),
            )
        )
    )
    owned, backends = [], []

    def construct(**kwargs):
        backend = SimulationBackend()
        fitter = ProductionCandidateFitter(**kwargs, backend=backend)
        owned.append(fitter)
        backends.append(backend)
        return fitter

    monkeypatch.setattr(runner, "ProductionCandidateFitter", construct)
    select = [
        "--select",
        "--dataset",
        str(root / "dataset.json"),
        "--config",
        str(root / "config.json"),
        "--period",
        str(root / "period.json"),
        "--output-root",
        str(root / "selection"),
        "--runtime-config",
        str(path),
    ]
    final = [
        "--final",
        "--dataset",
        str(root / "dataset.json"),
        "--lock-root",
        str(root / "selection"),
        "--output-file",
        str(root / "final.json"),
        "--runtime-config",
        str(path),
    ]
    return root, select, final, owned, backends


@pytest.mark.parametrize(
    "mutation",
    [
        "absent-state",
        "tampered-state",
        "absent-catalogue",
        "tampered-catalogue",
        "absent-sentiment",
        "tampered-event",
        "definition",
        "configuration",
    ],
)
def test_production_final_preflights_before_output_and_attempt_receipt(
    tmp_path, monkeypatch, mutation
):
    from risk_engine.backtest.runtime_inputs import RuntimeDefinition
    from tests.backtest.runtime_fixtures import seal

    root, select, final, owned, backends = concrete_cli_inputs(tmp_path, monkeypatch)
    assert runner.main(select) == 0
    lock = runner.load_artifact_set(root / "selection")[0].payload.locked
    state = Path(lock.fitted.artifacts[0].path)
    if mutation in {"absent-state", "tampered-state"}:
        if mutation == "absent-state":
            state.unlink()
        else:
            state.write_bytes(state.read_bytes() + b"tamper")
    elif "catalogue" in mutation:
        catalogue = root / "catalogue.csv"
        if mutation == "absent-catalogue":
            catalogue.unlink()
        else:
            catalogue.write_bytes(catalogue.read_bytes() + b"tamper")
    elif mutation == "absent-sentiment":
        next((root / "sentiment").rglob("tokenizer.json")).unlink()
    elif mutation == "tampered-event":
        next((root / "event").rglob("model.safetensors")).write_bytes(b"tamper")
    elif mutation == "definition":
        definition = RuntimeDefinition.model_validate_json((root / "definition.json").read_bytes())
        definition = seal(
            RuntimeDefinition,
            definition.model_dump(exclude={"content_hash"}) | {"version": "changed"},
            "content_hash",
        )
        (root / "definition.json").write_bytes(canonical_bytes(definition))
    else:
        # A changed configuration at the construction boundary must fail frozen-state identity.
        original = runner.ProductionCandidateFitter

        def mismatched(**kwargs):
            kwargs["configuration"] = kwargs["configuration"].model_copy(
                update={"version": "changed"}
            )
            return original(**kwargs)

        monkeypatch.setattr(runner, "ProductionCandidateFitter", mismatched)
    assert runner.main(final) == 1
    assert not (root / "final.json").exists()
    assert not tuple((root / "selection").glob(".final-attempt*"))
    assert all(backend.loads == 0 for backend in backends[1:])


@pytest.mark.parametrize("index_present", [False, True])
def test_final_runtime_never_opens_evidence_index_or_training_labels(
    tmp_path, monkeypatch, index_present
):
    from risk_engine.backtest.module import FinalArtifact

    root, select, final, owned, backends = concrete_cli_inputs(tmp_path, monkeypatch)
    assert runner.main(select) == 0
    if not index_present:
        path = root / "runtime.json"
        locations = json.loads(path.read_text())
        locations["evidence_index_path"] = None
        path.write_text(json.dumps(locations))
    config_bytes = (root / "config.json").read_bytes()
    (root / "config.json").unlink()
    original = Path.read_bytes

    def guard(path):
        assert path.name != "index.json" and not path.name.startswith("case-")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    assert runner.main(final) == 0
    result = FinalArtifact.model_validate_json((root / "final.json").read_bytes())
    assert result.status == "complete" and result.report is not None and result.audit is not None
    assert result.report.historical_case_ids == ("case-08",)
    assert owned[-1].configuration.model_dump_json() == (
        runner.DevelopmentConfiguration.model_validate_json(config_bytes).model_dump_json()
    )
    assert owned[-1].evidence_index is None
    assert tuple((root / "selection").glob(".final-attempt*"))


@pytest.mark.parametrize(
    "mode, failure", [("select", False), ("select", True), ("final", False), ("final", True)]
)
def test_cli_closes_only_owned_fitter_on_success_and_failure(tmp_path, monkeypatch, mode, failure):
    root, select, final, owned, backends = concrete_cli_inputs(tmp_path, monkeypatch)
    if mode == "final":
        assert runner.main(select) == 0
    if failure:
        # A real local-input refusal after model inference has started exercises finally.
        from risk_engine.backtest.runtime import ProductionCandidateFitter

        method = "fit" if mode == "select" else "restore"
        original = getattr(ProductionCandidateFitter, method)

        def fail_after_load(self, *args, **kwargs):
            original(self, *args, **kwargs)
            # restore preflight does not load models: run real inference before failure.
            if method == "restore":
                self._models.sentiment.score("Project-authored entity faces conflict.")
            raise RuntimeError("project-authored post-load failure")

        monkeypatch.setattr(ProductionCandidateFitter, method, fail_after_load)
    assert runner.main(select if mode == "select" else final) == (1 if failure else 0)
    assert backends[-1].loads > 0
    assert backends[-1].releases > 0
    assert owned[-1]._models is None


def test_failure_after_final_reservation_remains_consumed(tmp_path, monkeypatch):
    from risk_engine.backtest.module import FinalArtifact, FinalAttemptReceipt
    from risk_engine.backtest.runtime import ProductionCandidateFitter

    root, select, final, owned, backends = concrete_cli_inputs(tmp_path, monkeypatch)
    assert runner.main(select) == 0

    def failure(self, locked):
        assert (root / "final.json").exists()
        receipts = tuple((root / "selection").glob(".final-attempt*"))
        assert len(receipts) == 1
        receipt = FinalAttemptReceipt.model_validate_json(receipts[0].read_bytes())
        assert receipt.status == "consumed"
        raise RuntimeError("project-authored inference failure after reservation")

    monkeypatch.setattr(ProductionCandidateFitter, "restore", failure)
    assert runner.main(final) == 1
    result = FinalArtifact.model_validate_json((root / "final.json").read_bytes())
    assert result.status == "failed" and result.report is None
    assert "inference failure" in result.failure_reason
    final[final.index("--output-file") + 1] = str(root / "retry.json")
    assert runner.main(final) == 1
    assert not (root / "retry.json").exists()


@pytest.mark.parametrize("invalid", ["dataset", "artifact-set"])
def test_final_artifact_and_dataset_validation_precede_production_preflight(
    tmp_path, monkeypatch, invalid
):
    from risk_engine.backtest.runtime import ProductionCandidateFitter

    root, select, final, owned, backends = concrete_cli_inputs(tmp_path, monkeypatch)
    assert runner.main(select) == 0
    if invalid == "dataset":
        (root / "dataset.json").write_bytes(b"{}")
    else:
        (root / "selection/data/calibration/selected-config.json").unlink()

    def forbidden(self, locked):
        pytest.fail("preflight ran before artifact/dataset validation")

    monkeypatch.setattr(ProductionCandidateFitter, "preflight_restore", forbidden)
    monkeypatch.setattr(runner, "load_runtime_definition", lambda path: forbidden(None, None))
    assert runner.main(final) == 1
    assert not (root / "final.json").exists()
    assert not tuple((root / "selection").glob(".final-attempt*"))


def test_owned_preflight_requires_identical_supplied_fitter(tmp_path):
    from argparse import Namespace

    from risk_engine.backtest.runtime import ProductionCandidateFitter

    path, configuration, definition, index, locations = location_inputs(tmp_path)
    owner = ProductionCandidateFitter(
        configuration=configuration,
        definition=definition,
        evidence_index=index,
        model_locations=locations,
        artifact_root=path.parent / "states",
    )
    _, _, injected, _ = write_cli_inputs(tmp_path)
    with pytest.raises(ValueError, match="identical|identity"):
        runner._final(Namespace(), injected, production_fitter=owner)
