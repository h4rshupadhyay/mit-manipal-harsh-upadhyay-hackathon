"""Offline stress proposals, immutable manual audit, and validated Backtest reads."""

import json
import runpy
import socket
from dataclasses import replace
from datetime import timedelta
from decimal import ROUND_UP, localcontext
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from risk_engine.api.app import StressTestResponse, create_app
from risk_engine.api.dependencies import ArtifactNotFoundError, LocalBacktestArtifactReader
from risk_engine.config import PolicyConfig
from risk_engine.domain import (
    MarketSnapshot,
    Portfolio,
    PortfolioMateriality,
    StressScenario,
)
from risk_engine.risk.policy import ManualOverride, TriggerPolicy
from risk_engine.stress.module import StressEngine
from tests.api.test_signals import TIME, assert_safe_error, request
from tests.api.test_signals import setup as _signal_setup
from tests.stress.test_linear_valuation import market, position

signal_setup = _signal_setup


class Inputs:
    def __init__(self) -> None:
        self.holdings = Portfolio(
            portfolio_id="portfolio-1",
            version="portfolio-v1",
            as_of=TIME,
            valuation_currency="USD",
            positions=(
                position({"EQUITY-US": 1}),
                position({}, derivative=True, notional="30000").model_copy(
                    update={"position_id": "unsupported"}
                ),
            ),
        )
        snapshot = market("EQUITY-US")
        self.snapshot = MarketSnapshot.model_validate(
            snapshot.model_copy(
                update={
                    "as_of": TIME,
                    "observations": (
                        snapshot.observations[0].model_copy(update={"observed_at": TIME}),
                    ),
                }
            ).model_dump()
        )
        self.original = StressScenario.model_validate(
            {
                "scenario_id": "scenario-1",
                "risk_signal_id": "signal:synthetic-1",
                "shocks": [
                    {
                        "factor_id": "EQUITY-US",
                        "shock_type": "relative",
                        "value": -10,
                        "unit": "percent",
                        "horizon_days": 5,
                    }
                ],
                "method": "hypothetical",
                "calibration_version": "synthetic-impact-v1",
            }
        )

    def portfolio(self, portfolio_id: str, version: str) -> Portfolio:
        if portfolio_id == "missing":
            raise ArtifactNotFoundError("/home/secret portfolio")
        return self.holdings

    def market(self, snapshot_id: str) -> MarketSnapshot:
        if snapshot_id == "missing":
            raise ArtifactNotFoundError("/home/secret market")
        return self.snapshot

    def scenario(self, scenario_id: str, calibration_version: str) -> StressScenario:
        if scenario_id == "missing":
            raise ArtifactNotFoundError("/home/secret scenario")
        return self.original


@pytest.fixture
def stress_setup(signal_setup):
    client, _, repository, snapshots = signal_setup
    assert client.post("/v1/signals/analyze", json=request(snapshots[:1])).status_code == 200
    inputs = Inputs()
    container = replace(
        client.app.state.container,
        stress_engine=StressEngine(),
        stress_inputs=inputs,
        trigger_policy=TriggerPolicy(PolicyConfig(), as_of=TIME),
    )
    with TestClient(create_app(container), raise_server_exceptions=False) as stress_client:
        yield stress_client, inputs, repository, container


def proposal(inputs: Inputs, *, manual: bool = False) -> dict[str, object]:
    value: dict[str, object] = {
        "signal_id": "signal:synthetic-1",
        "portfolio_id": "portfolio-1",
        "portfolio_version": "portfolio-v1",
        "market_snapshot_id": "market-1",
        "scenario_id": "scenario-1",
        "calibration_version": "synthetic-impact-v1",
    }
    if manual:
        value["manual_override"] = {"analyst_id": "analyst-1", "reason": "Explore downside"}
        value["scenario"] = {
            **inputs.original.model_dump(mode="json"),
            "scenario_id": "scenario-manual-1",
            "analyst_override": True,
            "override_reason": "Explore downside",
        }
    return value


def test_unselected_policy_returns_traceable_blocked_preview(stress_setup) -> None:
    client, inputs, repository, _ = stress_setup
    before = repository.get("signal:synthetic-1")
    response = client.post("/v1/stress-tests", json=proposal(inputs))
    assert response.status_code == 200
    body = response.json()
    assert body["execution_status"] == "blocked"
    assert body["as_of"] == "2026-01-10T00:00:00Z"
    assert body["snapshot_ids"] == list(before.snapshot_ids)
    assert body["source_item_id"] == "source-1"
    assert body["market_snapshot_id"] == "market-1"
    assert body["portfolio_version"] == "portfolio-v1"
    assert body["original_scenario"] == body["scenario"] == inputs.original.model_dump(mode="json")
    result = body["stress_result"]
    assert (result["base_value"], result["stressed_value"], result["absolute_loss"]) == (
        "10000",
        "9000.00",
        "1000.00",
    )
    assert result["percentage_loss"] == 0.1
    assert result["valuation_coverage"] == 0.25
    assert result["unsupported_position_ids"] == ["unsupported"]
    assert {row["dimension"] for row in result["attribution"]} == {
        "asset",
        "sector",
        "region",
        "obligor",
        "factor",
    }
    assert all(row["loss"] == "1000.00" for row in result["attribution"])
    decision = body["trigger_decision"]
    assert decision["risk_signal"] == before.signal.model_dump(mode="json")
    assert decision["portfolio_materiality"] == {
        "absolute_loss": "1000.00",
        "percentage_loss": 0.1,
        "currency": "USD",
    }
    assert decision["impact_score"] == 8
    assert decision["confidence"] == 0.8
    assert decision["action_priority"] == 0.128
    assert decision["automatic_trigger"] is decision["triggered"] is False
    assert decision["policy_config"]["selected"] is None
    assert len(decision["gates"]) == 9
    assert decision["gates"][0]["passed"] is False
    assert decision["manual_override"] is None
    assert repository.get("signal:synthetic-1") == before


def test_manual_run_retains_new_scenario_and_production_override_audit(stress_setup) -> None:
    client, inputs, repository, _ = stress_setup
    before = (inputs.original.model_dump(), repository.list())
    payload = proposal(inputs, manual=True)
    first = client.post("/v1/stress-tests", json=payload)
    assert first.status_code == 200
    body = first.json()
    assert body["execution_status"] == "manual"
    assert body["scenario"] == payload["scenario"]
    assert body["original_scenario"] == inputs.original.model_dump(mode="json")
    assert body["stress_result"]["scenario_id"] == "scenario-manual-1"
    assert body["trigger_decision"]["manual_override"] == payload["manual_override"]
    assert body["trigger_decision"]["automatic_trigger"] is False
    assert body["trigger_decision"]["triggered"] is True
    assert body["trigger_decision"]["policy_config"]["selected"] is None
    assert client.post("/v1/stress-tests", json=payload).json() == body
    assert (inputs.original.model_dump(), repository.list()) == before


@pytest.mark.parametrize(
    "field", ["signal_id", "portfolio_id", "market_snapshot_id", "scenario_id"]
)
def test_missing_stress_artifacts_are_safe_404(stress_setup, field: str) -> None:
    client, inputs, _, _ = stress_setup
    payload = proposal(inputs)
    payload[field] = "missing"
    assert_safe_error(client.post("/v1/stress-tests", json=payload), 404, "artifact_not_found")


@pytest.mark.parametrize(
    "field",
    [
        "portfolio_id",
        "portfolio_version",
        "market_snapshot_id",
        "scenario_id",
        "calibration_version",
    ],
)
def test_lookup_cannot_rebind_requested_identities(stress_setup, field: str) -> None:
    client, inputs, _, _ = stress_setup
    payload = proposal(inputs)
    payload[field] = "conflicting-version"
    assert_safe_error(client.post("/v1/stress-tests", json=payload), 409, "version_conflict")


@pytest.mark.parametrize(
    "kind",
    [
        "missing_reason",
        "no_audit",
        "no_scenario",
        "reason_mismatch",
        "not_override",
        "same_id",
        "wrong_signal",
        "wrong_calibration",
    ],
)
def test_manual_audit_requires_bound_new_scenario(stress_setup, kind: str) -> None:
    client, inputs, _, _ = stress_setup
    payload = proposal(inputs, manual=True)
    if kind == "no_audit":
        del payload["manual_override"]
    elif kind == "no_scenario":
        del payload["scenario"]
    elif kind == "missing_reason":
        payload["manual_override"]["reason"] = " "
    elif kind == "reason_mismatch":
        payload["scenario"]["override_reason"] = "Different reason"
    elif kind == "not_override":
        payload["scenario"]["analyst_override"] = False
        payload["scenario"]["override_reason"] = None
    elif kind == "same_id":
        payload["scenario"]["scenario_id"] = "scenario-1"
    elif kind == "wrong_signal":
        payload["scenario"]["risk_signal_id"] = "signal-other"
    else:
        payload["scenario"]["calibration_version"] = "wrong"
    expected = 409 if kind in {"wrong_signal", "wrong_calibration"} else 422
    code = "version_conflict" if expected == 409 else "invalid_request"
    assert_safe_error(client.post("/v1/stress-tests", json=payload), expected, code)


@pytest.mark.parametrize(
    "kind", ["unknown", "unit", "below_floor", "zero_horizon", "duplicate", "different_horizons"]
)
def test_bad_shocks_reject_before_valuation(stress_setup, kind: str) -> None:
    client, inputs, _, container = stress_setup
    payload = proposal(inputs, manual=True)
    shock = payload["scenario"]["shocks"][0]
    if kind == "unknown":
        shock["factor_id"] = "secret-factor"
    elif kind == "unit":
        shock["unit"] = "basis_point"
    elif kind == "below_floor":
        shock["value"] = -101
    elif kind == "zero_horizon":
        shock["horizon_days"] = 0
    elif kind == "duplicate":
        payload["scenario"]["shocks"].append(dict(shock))
    else:
        payload["scenario"]["shocks"].append({**shock, "factor_id": "COMMODITY", "horizon_days": 6})

    class ForbiddenEngine:
        def run(self, *args, **kwargs):
            raise AssertionError("bad shock reached valuation")

    client.app.state.container = replace(container, stress_engine=ForbiddenEngine())
    response = client.post("/v1/stress-tests", json=payload)
    assert response.status_code == 422
    assert response.json()["code"] in {"invalid_request", "invalid_domain"}


def test_policy_cutoff_conflict_is_409_instead_of_rewritten(stress_setup) -> None:
    client, inputs, _, container = stress_setup
    client.app.state.container = replace(
        container, trigger_policy=TriggerPolicy(PolicyConfig(), as_of=TIME + timedelta(days=1))
    )
    assert_safe_error(
        client.post("/v1/stress-tests", json=proposal(inputs)), 409, "version_conflict"
    )


@pytest.mark.parametrize(
    "kind",
    [
        "portfolio",
        "market",
        "observation",
        "publication",
        "retrieval",
        "source_identity",
        "source_missing",
        "scenario_signal",
    ],
)
def test_historical_inputs_cannot_cross_stored_analysis_cutoff_or_lineage(
    stress_setup, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    client, inputs, _, container = stress_setup
    future = TIME + timedelta(days=1)
    if kind in {"portfolio", "market"}:
        inputs.holdings = inputs.holdings.model_copy(update={"as_of": future})
        inputs.snapshot = inputs.snapshot.model_copy(update={"as_of": future})
    elif kind == "observation":
        inputs.snapshot = inputs.snapshot.model_copy(
            update={
                "observations": (
                    inputs.snapshot.observations[0].model_copy(update={"observed_at": future}),
                )
            }
        )
    elif kind == "scenario_signal":
        inputs.original = inputs.original.model_copy(update={"risk_signal_id": "signal-other"})
    else:
        replay = container.data.replay

        def changed_replay(snapshot_id):
            items = replay(snapshot_id)
            if kind == "source_missing":
                return []
            changes = (
                {"source_item_id": "wrong"}
                if kind == "source_identity"
                else {
                    "retrieved_at": future,
                    **({"published_at": future} if kind == "publication" else {}),
                }
            )
            return [item.model_copy(update=changes) for item in items]

        monkeypatch.setattr(container.data, "replay", changed_replay)
    expected = 409 if kind in {"source_identity", "source_missing", "scenario_signal"} else 422
    response = client.post("/v1/stress-tests", json=proposal(inputs))
    assert response.status_code == expected
    assert response.json()["code"] in {"invalid_domain", "version_conflict"}


@pytest.mark.parametrize("kind", ["signal", "materiality", "override"])
def test_returned_policy_cannot_substitute_other_valid_audit_inputs(stress_setup, kind) -> None:
    client, inputs, _, container = stress_setup

    class DifferentPolicy:
        def evaluate(self, signal, materiality, *, manual_override=None):
            if kind == "signal":
                signal = signal.model_copy(update={"signal_id": "wrong"})
            elif kind == "materiality":
                materiality = PortfolioMateriality(
                    absolute_loss="1", percentage_loss=0.001, currency="USD"
                )
            else:
                manual_override = ManualOverride(analyst_id="different", reason="different")
            return TriggerPolicy(PolicyConfig(), as_of=TIME).evaluate(
                signal, materiality, manual_override=manual_override
            )

    client.app.state.container = replace(container, trigger_policy=DifferentPolicy())
    assert_safe_error(
        client.post("/v1/stress-tests", json=proposal(inputs)), 409, "version_conflict"
    )


@pytest.mark.parametrize(
    "kind",
    [
        "scenario_id",
        "portfolio_id",
        "portfolio_version",
        "market_version",
        "scenario_version",
        "currency",
    ],
)
def test_returned_stress_result_cannot_substitute_other_input_bindings(stress_setup, kind) -> None:
    client, inputs, _, container = stress_setup

    class DifferentEngine:
        def run(self, portfolio, base_market, scenario):
            result = StressEngine().run(portfolio, base_market, scenario)
            if kind in {"scenario_id", "portfolio_id"}:
                return result.model_copy(update={kind: "wrong"})
            if kind == "currency":
                return result.model_copy(update={"valuation_currency": "EUR"})
            return result.model_copy(
                update={"versions": result.versions.model_copy(update={kind: "wrong"})}
            )

    client.app.state.container = replace(container, stress_engine=DifferentEngine())
    assert_safe_error(
        client.post("/v1/stress-tests", json=proposal(inputs)), 409, "version_conflict"
    )


@pytest.mark.parametrize("gain", [True, False])
def test_materiality_clamps_gain_and_is_independent_of_decimal_context(stress_setup, gain) -> None:
    client, inputs, _, _ = stress_setup
    inputs.original = inputs.original.model_copy(
        update={
            "shocks": (inputs.original.shocks[0].model_copy(update={"value": 10 if gain else -10}),)
        }
    )
    inputs.holdings = inputs.holdings.model_copy(
        update={"positions": (position({"EQUITY-US": 1}, notional="12345.6789"),)}
    )
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        response = client.post("/v1/stress-tests", json=proposal(inputs, manual=True))
    assert response.status_code == 200
    materiality = response.json()["trigger_decision"]["portfolio_materiality"]
    assert materiality == {
        "absolute_loss": "0" if gain else "1234.567890",
        "percentage_loss": 0 if gain else 0.1,
        "currency": "USD",
    }


def test_serialized_response_validation_uses_fixed_decimal_context(stress_setup) -> None:
    client, inputs, _, _ = stress_setup
    inputs.holdings = inputs.holdings.model_copy(
        update={"positions": (position({"EQUITY-US": 1}, notional="12345.6789"),)}
    )
    response = client.post("/v1/stress-tests", json=proposal(inputs, manual=True))
    assert response.status_code == 200
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        restored = StressTestResponse.model_validate_json(response.text)
        assert context.prec == 2
        assert context.rounding == ROUND_UP
    assert (
        restored.stress_result.absolute_loss
        == restored.trigger_decision.portfolio_materiality.absolute_loss
    )


@pytest.mark.parametrize("port", ["stress_engine", "trigger_policy", "stress_inputs"])
def test_unconfigured_stress_port_is_missing_without_loading_defaults(stress_setup, port) -> None:
    client, inputs, _, container = stress_setup
    client.app.state.container = replace(container, **{port: None})
    assert_safe_error(
        client.post("/v1/stress-tests", json=proposal(inputs)), 404, "artifact_not_found"
    )


def test_backtest_missing_and_unresolved_artifacts_are_404(stress_setup, tmp_path: Path) -> None:
    client, _, _, container = stress_setup
    assert_safe_error(client.get("/v1/backtests/latest"), 404, "artifact_not_found")
    for root in (tmp_path, Path.cwd()):
        client.app.state.container = replace(
            container, backtests=LocalBacktestArtifactReader(root=root, final_path=None)
        )
        assert_safe_error(client.get("/v1/backtests/latest"), 404, "artifact_not_found")


def test_backtest_route_reads_canonical_ready_final_and_tampered_artifacts(
    stress_setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _, container = stress_setup
    fixture = runpy.run_path(str(Path(__file__).parents[1] / "backtest/test_backtest_module.py"))
    _, _, fitter, args = fixture["write_cli_inputs"](tmp_path)
    runner = fixture["cli"]()
    assert runner.main(args, fitter=fitter) == 0
    root = tmp_path / "selection"
    final_path = tmp_path / "final.json"
    assert (
        runner.main(
            [
                "--final",
                "--dataset",
                str(tmp_path / "dataset.json"),
                "--lock-root",
                str(root),
                "--output-file",
                str(final_path),
            ],
            fitter=fitter,
        )
        == 0
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("API attempted evaluation or acquisition")

    monkeypatch.setattr(type(fitter), "fit", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for final in (None, final_path):
        reader = LocalBacktestArtifactReader(root=root, final_path=final)
        expected = reader.latest()
        files_before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
        client.app.state.container = replace(container, backtests=reader)
        response = client.get("/v1/backtests/latest")
        assert response.status_code == 200
        assert response.json() == expected.model_dump(mode="json")
        assert response.json()["historical_case_ids"]
        assert {path: path.read_bytes() for path in files_before} == files_before
    final_path.unlink()
    assert_safe_error(client.get("/v1/backtests/latest"), 404, "artifact_not_found")
    client.app.state.container = replace(
        container, backtests=LocalBacktestArtifactReader(root=root, final_path=None)
    )
    selected = root / "data/calibration/selected-config.json"
    damaged = json.loads(selected.read_text())
    damaged["artifact_set_hash"] = "a" * 64
    selected.write_text(json.dumps(damaged))
    assert_safe_error(client.get("/v1/backtests/latest"), 409, "version_conflict")
