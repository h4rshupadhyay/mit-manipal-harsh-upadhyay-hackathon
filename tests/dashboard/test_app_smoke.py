"""Actual offline Streamlit rendering and immutable manual submission boundaries."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from risk_engine.api.app import StressTestResponse
from risk_engine.api.dependencies import AnalysisRecord
from risk_engine.dashboard.app import DashboardEvidence, HttpStressTestClient
from tests.api.test_signals import TIME
from tests.api.test_stress_and_backtests import proposal
from tests.api.test_stress_and_backtests import signal_setup as _signal_setup
from tests.api.test_stress_and_backtests import stress_setup as _stress_setup
from tests.dashboard.test_view_models import (
    analysis,
    display_evidence,
    report,
    source,
    stress_response,
)

signal_setup = _signal_setup
stress_setup = _stress_setup

SCRIPT = """
import streamlit as st
from risk_engine.dashboard.app import render_app
render_app(st.session_state.get('evidence'), st.session_state.get('client'))
"""


def evidence(**updates):
    return replace(
        DashboardEvidence(
            as_of=TIME,
            snapshot_ids=("snapshot-1",),
            mode="snapshot",
            sources=(source(),),
            analysis=analysis(),
            stress=stress_response(),
            backtest=report(),
            backtest_evidence=display_evidence(),
        ),
        **updates,
    )


@pytest.fixture
def app(tmp_path):
    path = tmp_path / "analyst_test.py"
    path.write_text(SCRIPT)

    def create(value=None, client=None):
        test = AppTest.from_file(str(path))
        test.session_state["evidence"] = value
        test.session_state["client"] = client
        return test.run()

    return create


def text(test):
    return "\n".join(
        str(element.value)
        for kind in (
            "markdown",
            "caption",
            "json",
            "info",
            "warning",
            "success",
            "error",
            "metric",
            "dataframe",
        )
        for element in test.get(kind)
    )


def navigate(test, page):
    test.sidebar.radio[0].set_value(page).run()
    assert not test.exception
    assert test.header[0].value == page
    return test


def test_default_entrypoint_registers_all_pages_without_invented_results():
    test = AppTest.from_file(str(Path("src/risk_engine/dashboard/app.py"))).run()
    assert not test.exception
    assert test.sidebar.radio[0].options == [
        "Signal Monitor",
        "Portfolio Stress",
        "Backtest Evidence",
    ]
    for page in test.sidebar.radio[0].options:
        navigate(test, page)
        assert test.info
        assert not test.metric
        assert not test.button
        assert "unavailable" in text(test).lower()


def test_signal_fixture_shows_distinct_measures_and_full_source_audit(app):
    test = app(evidence())
    assert not test.exception
    assert {metric.label: metric.value for metric in test.metric} == {
        "Impact Score": "8 / 10",
        "Confidence": "80%",
        "Portfolio Materiality": "USD 20000",
        "Action Priority": "0.128",
    }
    content = text(test)
    assert "snapshot-1" in content
    assert "2026-01-10" in content
    assert "synthetic:story" in content
    assert "MIT" in content
    assert "Alpha Bank failed" in content


def test_empty_analysis_and_invalid_evidence_show_honest_safe_states(app):
    empty = AnalysisRecord(snapshot_ids=("snapshot-1",), as_of=TIME, signals=())
    test = app(evidence(analysis=empty))
    assert not test.exception
    assert "No Risk Signals" in test.info[0].value
    test = app(evidence(as_of=TIME + timedelta(days=1)))
    assert not test.exception
    assert test.error
    assert not test.metric
    assert "lineage" not in text(test)


def test_blocked_preview_retains_exact_signed_values_gates_shock_units_and_coverage(app):
    test = navigate(app(evidence()), "Portfolio Stress")
    assert "Blocked preview" in test.warning[0].value
    values = {metric.label: metric.value for metric in test.metric}
    assert values["Portfolio value before"] == "USD 10000"
    assert values["Portfolio value after"] == "USD 8999.876543211"
    assert values["Absolute loss"] == "USD 1000.123456789"
    assert values["Percentage loss"] == "10.00123456789%"
    assert values["Valuation coverage"] == "25%"
    content = text(test)
    for value in ["unsupported", "country", "facility", "percent", "5", "hypothetical"]:
        assert value in content
    gates = test.dataframe[0].value
    assert len(gates) == 9
    assert test.button[0].disabled is not False  # no configured submission client
    test = navigate(app(evidence(stress=stress_response(gain=True))), "Portfolio Stress")
    assert {m.label: m.value for m in test.metric}["Absolute loss"] == "USD -1000.123456789"
    assert "gain" in text(test)


def test_backtest_shows_retained_samples_metadata_and_india_cases_without_inference(app):
    test = navigate(app(evidence()), "Backtest Evidence")
    assert "synthetic" in text(test)
    assert {m.label: m.value for m in test.metric} == {
        "Expected loss": "USD 2.9",
        "Tail loss": "USD 8.125",
    }
    assert len(test.get("plotly_chart")) == 1
    content = text(test)
    for value in [
        "RBI monetary policy episode",
        "Indian credit/default episode",
        "one_day",
        "primary",
    ]:
        assert value in content
    assert "looks-like-RBI-india" in content
    test = navigate(app(evidence(backtest_evidence=None)), "Backtest Evidence")
    assert not test.metric
    assert not test.get("plotly_chart")
    assert "Retained loss distribution not supplied" in text(test)
    assert "India case metadata unavailable" in text(test)


def manual_client(stress_setup, *, failure=False, tamper=None):
    backend, inputs, repository, _ = stress_setup
    response = StressTestResponse.model_validate(
        backend.post("/v1/stress-tests", json=proposal(inputs)).json()
    )
    requests = []

    def handle(request):
        requests.append(request)
        if failure:
            return httpx.Response(500, json={"secret": "/home/private token"})
        body = backend.post(
            request.url.path, content=request.content, headers={"content-type": "application/json"}
        ).json()
        if tamper is not None:
            body = tamper(body)
        return httpx.Response(200, json=body)

    client = HttpStressTestClient(
        "http://explicit.local", timeout=2, transport=httpx.MockTransport(handle)
    )
    return client, response, requests, repository, inputs


def submit(test, *, reason="Explore downside"):
    test.text_input(key="analyst_id").set_value("analyst-1")
    test.text_area(key="override_reason").set_value(reason)
    test.button[0].click().run()
    assert not test.exception
    return test


def test_manual_submit_creates_new_complete_audit_without_mutation_or_rerun(stress_setup, app):
    client, response, requests, repository, inputs = manual_client(stress_setup)
    history = repository.list(), inputs.original.model_dump(), response.model_dump()
    test = navigate(app(evidence(stress=response), client), "Portfolio Stress")
    assert not requests
    test.number_input[0].set_value(-20.0)
    submit(test)
    assert len(requests) == 1
    import json

    payload = json.loads(requests[0].content)
    assert payload["scenario_id"] == "scenario-1"
    assert payload["scenario"]["scenario_id"] != "scenario-1"
    assert payload["scenario"]["shocks"] == [
        {
            "factor_id": "EQUITY-US",
            "shock_type": "relative",
            "value": -20.0,
            "unit": "percent",
            "horizon_days": 5,
        }
    ]
    assert payload["scenario"]["reference_event_ids"] == []
    assert payload["scenario"]["calibration_version"] == "synthetic-impact-v1"
    assert payload["scenario"]["analyst_override"] is True
    assert payload["manual_override"] == {"analyst_id": "analyst-1", "reason": "Explore downside"}
    assert "Manual scenario accepted" in text(test)
    assert "scenario-1" in text(test)
    test.run()
    navigate(test, "Signal Monitor")
    navigate(test, "Portfolio Stress")
    assert len(requests) == 1
    assert (repository.list(), inputs.original.model_dump(), response.model_dump()) == history
    test.session_state["evidence"] = evidence(stress=stress_response(gain=True))
    test.run()
    assert "Manual scenario accepted" not in text(test)
    assert len(requests) == 1


def test_manual_audit_history_retains_each_pair_across_context_changes(stress_setup, app):
    import hashlib
    import json

    client, response, requests, repository, inputs = manual_client(stress_setup)
    accepted = []
    returned_records = []

    class RecordingClient:
        reject_next = False

        def submit(self, request):
            if self.reject_next:
                raise RuntimeError("/home/private failed service")
            returned = client.submit(request)
            accepted.append((request.model_dump(mode="json"), returned.model_dump(mode="json")))
            returned_records.append(returned)
            return returned

    original = repository.list(), inputs.original.model_dump(), response.model_dump()
    recording = RecordingClient()
    test = navigate(app(evidence(stress=response), recording), "Portfolio Stress")
    test.number_input[0].set_value(-20.0)
    submit(test, reason="First downside")
    test.number_input[0].set_value(-30.0)
    submit(test, reason="Second downside")
    assert len(requests) == len(accepted) == 2
    identity = hashlib.sha256(response.model_dump_json().encode()).hexdigest()
    history = test.session_state["manual_submission_history"]
    audits = history[identity]
    assert len(audits) == 2
    for audit, (request, returned) in zip(audits, accepted, strict=True):
        assert json.loads(audit.request_json) == request
        assert json.loads(audit.response_json) == returned
        assert audit.originating_identity == identity
        assert audit.scenario_id == request["scenario"]["scenario_id"]
    assert audits[0].scenario_id != audits[1].scenario_id
    # Even a caller bypassing its frozen record cannot rewrite retained bytes.
    object.__setattr__(returned_records[0], "market_snapshot_id", "modified-after-acceptance")
    assert json.loads(audits[0].response_json) == accepted[0][1]
    recording.reject_next = True
    submit(test, reason="Later failed submission")
    assert test.error
    assert "private" not in text(test)
    assert test.session_state["manual_submission_history"][identity] == audits
    assert len(requests) == 2
    assert (repository.list(), inputs.original.model_dump(), response.model_dump()) == original
    for replacement in (evidence(stress=stress_response(gain=True)), evidence(stress=None)):
        test.session_state["evidence"] = replacement
        test.run()
        assert not test.exception
        assert not test.error
        assert "Manual scenario accepted" not in text(test)
        assert test.session_state["manual_submission_history"][identity] == audits
        labels = [element.proto.label for element in test.get("download_button")]
        for audit in audits:
            assert f"Download manual audit {audit.scenario_id}" in labels
        assert len(requests) == 2


@pytest.mark.parametrize("failure", [False, True])
def test_invalid_form_and_failed_submission_do_not_expose_or_fabricate_success(
    stress_setup, failure, app
):
    client, response, requests, _, _ = manual_client(stress_setup, failure=failure)
    test = navigate(app(evidence(stress=response), client), "Portfolio Stress")
    submit(test, reason="   ")
    assert not requests
    assert test.error
    if failure:
        submit(test)
        assert len(requests) == 1
        assert test.error
        assert "private" not in text(test)
        assert "token" not in text(test)
        assert "Manual scenario accepted" not in text(test)
        test.run()
        assert len(requests) == 1


def test_internally_valid_response_for_another_original_is_rejected(stress_setup, app):
    def tamper(body):
        body["original_scenario"]["scenario_id"] = "unrelated-original"
        return body

    client, response, requests, _, _ = manual_client(stress_setup, tamper=tamper)
    test = navigate(app(evidence(stress=response), client), "Portfolio Stress")
    submit(test)
    assert len(requests) == 1
    assert test.error
    assert "Manual scenario accepted" not in text(test)


def test_stress_action_measures_remain_separate_from_valuation_loss(app):
    test = navigate(app(evidence()), "Portfolio Stress")
    values = {metric.label: metric.value for metric in test.metric}
    assert values["Impact Score"] == "8 / 10"
    assert values["Confidence"] == "0.8"
    assert values["Portfolio Materiality"] == "USD 1000.123456789"
    assert values["Action Priority"] == "0.128"


@pytest.mark.parametrize(
    "conflict", ["scenario", "audit", "snapshots", "cutoff", "portfolio", "market"]
)
def test_manual_response_must_match_exact_submission_and_original_context(
    stress_setup, app, conflict
):
    import json

    def tamper(body):
        if conflict == "scenario":
            body["scenario"]["shocks"][0]["value"] = -30
        elif conflict == "audit":
            body["trigger_decision"]["manual_override"]["analyst_id"] = "different-analyst"
        elif conflict == "snapshots":
            body["snapshot_ids"] = ["another-snapshot"]
        elif conflict == "cutoff":
            body["as_of"] = body["trigger_decision"]["as_of"] = "2026-01-11T00:00:00Z"
        elif conflict == "portfolio":
            body["portfolio_id"] = body["stress_result"]["portfolio_id"] = "other-portfolio"
        else:
            body["market_snapshot_id"] = "other-market"
            versions = body["stress_result"]["versions"]
            market = json.loads(versions["market_version"])
            market["snapshot_id"] = "other-market"
            versions["market_version"] = json.dumps(market)
        # Prove internal canonical validation is insufficient for this conflict.
        StressTestResponse.model_validate(body)
        return body

    client, response, requests, _, _ = manual_client(stress_setup, tamper=tamper)
    test = navigate(app(evidence(stress=response), client), "Portfolio Stress")
    submit(test)
    assert len(requests) == 1
    assert test.error
    assert not test.success


@pytest.mark.parametrize(
    "url,timeout",
    [
        ("", 2),
        ("file:///private", 2),
        ("http://user:secret@localhost", 2),
        ("https://localhost?key=secret", 2),
        ("https://localhost", 0),
        ("https://localhost", float("nan")),
        ("https://localhost", float("inf")),
    ],
)
def test_http_adapter_rejects_incomplete_or_secret_bearing_configuration(url, timeout):
    with pytest.raises(ValueError):
        HttpStressTestClient(url, timeout=timeout)


def test_read_only_rendering_never_refreshes_infers_revalues_or_evaluates(app, monkeypatch):
    import socket

    from risk_engine.backtest.module import BacktestModule
    from risk_engine.risk.module import RiskEngine
    from risk_engine.stress.module import StressEngine

    supplied = evidence()

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only rendering invoked an engine or network")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(httpx.Client, "request", forbidden)
    monkeypatch.setattr(RiskEngine, "analyze", forbidden)
    monkeypatch.setattr(StressEngine, "run", forbidden)
    monkeypatch.setattr(BacktestModule, "evaluate", forbidden)
    test = app(supplied)
    for page in test.sidebar.radio[0].options:
        navigate(test, page)
        assert not test.error
        assert not test.exception
