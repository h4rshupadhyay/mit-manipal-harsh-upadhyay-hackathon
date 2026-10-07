"""Retained Stress Test previews and explicit new manual-scenario submissions."""

import hashlib
import json
from typing import Protocol
from uuid import uuid4

import streamlit as st

from risk_engine.api.app import StressTestRequest, StressTestResponse
from risk_engine.dashboard.view_models import PortfolioStressView, build_portfolio_stress
from risk_engine.domain import DomainModel, NonEmptyString, Sha256Hex, StressScenario
from risk_engine.risk.policy import ManualOverride


class StressTestClient(Protocol):
    def submit(self, request: StressTestRequest) -> StressTestResponse: ...


class ManualSubmissionAudit(DomainModel):
    """Immutable serialized copies of an accepted request and its response."""

    originating_identity: Sha256Hex
    scenario_id: NonEmptyString
    request_json: NonEmptyString
    response_json: NonEmptyString


def _history() -> None:
    history: dict[str, tuple[ManualSubmissionAudit, ...]] = st.session_state.get(
        "manual_submission_history", {}
    )
    if not history:
        return
    st.subheader("Manual submission audit history")
    st.caption(
        "Accepted requests and responses are retained in this browser session. "
        "Download each audit for durable storage; current permission is displayed above."
    )
    for audits in history.values():
        for audit in audits:
            payload = {
                "originating_identity": audit.originating_identity,
                "scenario_id": audit.scenario_id,
                "request": json.loads(audit.request_json),
                "response": json.loads(audit.response_json),
            }
            with st.expander(f"Historical manual audit {audit.scenario_id}"):
                st.json(payload)
                st.download_button(
                    f"Download manual audit {audit.scenario_id}",
                    json.dumps(payload),
                    file_name=f"manual-scenario-audit-{audit.scenario_id}.json",
                    mime="application/json",
                    key=f"manual-audit:{audit.scenario_id}",
                )


def _check_submission(
    returned: StressTestResponse, original: StressTestResponse, request: StressTestRequest
) -> PortfolioStressView:
    view = build_portfolio_stress(returned)
    assert view.response is not None
    result = view.response
    if (
        result.original_scenario != original.original_scenario
        or result.scenario != request.scenario
        or result.trigger_decision.manual_override != request.manual_override
        or result.trigger_decision.risk_signal != original.trigger_decision.risk_signal
        or (
            result.signal_id,
            result.source_item_id,
            result.snapshot_ids,
            result.as_of,
            result.portfolio_id,
            result.portfolio_version,
            result.market_snapshot_id,
        )
        != (
            original.signal_id,
            original.source_item_id,
            original.snapshot_ids,
            original.as_of,
            original.portfolio_id,
            original.portfolio_version,
            original.market_snapshot_id,
        )
    ):
        raise ValueError("returned audit differs from explicit submission")
    return view


def _result(view: PortfolioStressView) -> None:
    response = view.response
    assert response is not None
    if view.execution_status == "blocked":
        st.warning(
            "Blocked preview: calculated Stress Result has no automatic or manual "
            "action permission."
        )
    elif view.execution_status == "manual":
        st.success("Manual scenario accepted with the returned analyst audit.")
    else:
        st.success("Automatic permission recorded by the supplied production policy.")
    st.caption(
        f"As of: {response.as_of.isoformat()} | "
        f"Source snapshots: {', '.join(response.snapshot_ids)} | "
        f"Market snapshot: {response.market_snapshot_id}"
    )
    st.write(
        f"Synthetic Portfolio: {response.portfolio_id} | Version: {response.portfolio_version}"
    )
    for label, value in (
        ("Portfolio value before", view.base_value),
        ("Portfolio value after", view.stressed_value),
        ("Absolute loss", view.absolute_loss),
        ("Percentage loss", view.percentage_loss),
        ("Valuation coverage", view.valuation_coverage),
    ):
        if value is not None:
            st.metric(label, value.text)
    st.caption(f"Loss direction: {view.loss_direction}. Values cover supported positions only.")
    st.write(
        f"Unsupported positions: {', '.join(view.unsupported_position_ids) or 'None reported'}"
    )
    st.write(
        "Unavailable attribution dimensions: "
        f"{', '.join(d.value for d in view.unavailable_dimensions) or 'None'}"
    )
    decision = response.trigger_decision
    st.subheader("Action permission")
    columns = st.columns(4)
    columns[0].metric("Impact Score", f"{decision.impact_score} / 10")
    columns[1].metric("Confidence", str(decision.confidence))
    columns[2].metric(
        "Portfolio Materiality",
        f"{decision.portfolio_materiality.currency} "
        f"{decision.portfolio_materiality.absolute_loss:f}",
    )
    columns[3].metric(
        "Action Priority",
        "Unavailable" if decision.action_priority is None else str(decision.action_priority),
    )
    st.caption("Confidence is a calibrated probability on the 0-1 scale.")
    st.write(
        f"Automatic trigger: {decision.automatic_trigger} | "
        f"Manual permission: {decision.manual_override is not None} | "
        f"Combined permission: {decision.triggered}"
    )
    st.dataframe(
        [
            {"Gate": gate.name, "Passed": gate.passed, "Reason": gate.reason}
            for gate in decision.gates
        ],
        hide_index=True,
    )
    st.subheader("Canonical attribution")
    st.caption("Each dimension is a separate attribution of the same loss.")
    st.dataframe(
        [
            {"Dimension": row.dimension.value, "Label": row.label, "Loss": row.loss.text}
            for row in view.attribution
        ],
        hide_index=True,
    )
    for label, scenario in (
        ("Original Stress Scenario", response.original_scenario),
        ("Effective Stress Scenario", response.scenario),
    ):
        with st.expander(label):
            st.write(
                f"Scenario: {scenario.scenario_id} | Method: {scenario.method.value} | "
                f"Calibration: {scenario.calibration_version}"
            )
            st.dataframe(
                [
                    {
                        "Factor": shock.factor_id,
                        "Shock type": shock.shock_type.value,
                        "Value": shock.value,
                        "Unit": shock.unit.value,
                        "Horizon days": shock.horizon_days,
                    }
                    for shock in scenario.shocks
                ],
                hide_index=True,
            )
            st.json(scenario.model_dump(mode="json"))
    with st.expander("Complete Stress Result and production policy audit"):
        st.json(response.model_dump(mode="json"))
    st.download_button(
        "Download retained scenario audit",
        response.model_dump_json(),
        file_name="stress-scenario-audit.json",
        mime="application/json",
    )


def render(view: PortfolioStressView, client: StressTestClient | None = None) -> None:
    st.header("Portfolio Stress")
    response = view.response
    if response is None:
        st.session_state.pop("manual_submission", None)
        st.info(view.unavailable_reason)
        _history()
        return
    identity = hashlib.sha256(response.model_dump_json().encode()).hexdigest()
    stored = st.session_state.get("manual_submission")
    if stored is not None and stored[0] != identity:
        st.session_state.pop("manual_submission", None)
        stored = None
    _result(view)
    st.subheader("New manual Stress Scenario")
    st.caption(
        "Retains the original proposal. Submission records a new scenario and analyst "
        "reason; no historical record is edited."
    )
    if client is None:
        st.info("Manual submission unavailable: no Stress Test service is configured.")
    with st.form("manual_scenario"):
        analyst = st.text_input("Analyst identity", key="analyst_id")
        reason = st.text_area("Override reason", key="override_reason")
        shocks = []
        for shock in response.original_scenario.shocks:
            value = st.number_input(
                f"{shock.factor_id} shock ({shock.unit.value}, {shock.horizon_days} days)",
                value=shock.value,
                key=f"shock:{identity}:{shock.factor_id}",
            )
            shocks.append({**shock.model_dump(mode="python"), "value": value})
        submitted = st.form_submit_button("Submit manual scenario", disabled=client is None)
    if submitted and client is not None:
        if not analyst.strip() or not reason.strip():
            st.error("Analyst identity and an override reason are required.")
        else:
            try:
                original = response.original_scenario
                scenario = StressScenario.model_validate(
                    {
                        **original.model_dump(mode="python"),
                        "scenario_id": f"manual:{uuid4()}",
                        "shocks": tuple(shocks),
                        "analyst_override": True,
                        "override_reason": reason.strip(),
                    }
                )
                request = StressTestRequest(
                    signal_id=response.signal_id,
                    portfolio_id=response.portfolio_id,
                    portfolio_version=response.portfolio_version,
                    market_snapshot_id=response.market_snapshot_id,
                    scenario_id=original.scenario_id,
                    calibration_version=original.calibration_version,
                    scenario=scenario,
                    manual_override=ManualOverride(
                        analyst_id=analyst.strip(), reason=reason.strip()
                    ),
                )
                request_json = request.model_dump_json()
                new_view = _check_submission(
                    client.submit(request),
                    response,
                    StressTestRequest.model_validate_json(request_json),
                )
                assert new_view.response is not None
                audit = ManualSubmissionAudit(
                    originating_identity=identity,
                    scenario_id=scenario.scenario_id,
                    request_json=request_json,
                    response_json=new_view.response.model_dump_json(),
                )
                history = dict(st.session_state.get("manual_submission_history", {}))
                history[identity] = (*history.get(identity, ()), audit)
                st.session_state["manual_submission_history"] = history
                st.session_state["manual_submission"] = (identity, new_view)
                stored = (identity, new_view)
            except Exception:
                st.error(
                    "Manual scenario could not be accepted. Check the scenario and service "
                    "with your administrator before submitting again."
                )
    if stored is not None:
        st.subheader("Retained manual submission")
        _result(stored[1])
    _history()
