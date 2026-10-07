"""Read supplied Backtest metrics and retained empirical/hypothetical/synthetic evidence."""

import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st

from risk_engine.dashboard.view_models import BacktestEvidenceView


def render(view: BacktestEvidenceView | None) -> None:
    st.header("Backtest Evidence")
    if view is None or view.report is None:
        st.info(
            "Backtest evidence unavailable. No retained report is configured."
            if view is None
            else view.unavailable_reason
        )
        return
    report = view.report
    st.caption(
        f"As of: {view.as_of.isoformat()} | Report: {report.report_id} | "
        f"Configuration: {report.configuration_version} | Schema: {report.schema_version}"
    )
    st.write(
        f"Evaluation period: {report.evaluation_start.isoformat()} to "
        f"{report.evaluation_end.isoformat()} (end exclusive)"
    )
    st.write(
        "Evidence kind: "
        f"{view.evidence_kind or 'Unavailable; report does not declare evidence metadata'}"
    )
    st.subheader("Backtest metrics")
    st.json(report.metrics)
    if view.metrics_unavailable_reason is not None:
        st.info(view.metrics_unavailable_reason)
    elif view.evidence is not None and view.evidence.metrics is not None:
        with st.expander("Detailed metric evidence and units"):
            st.json(view.evidence.metrics.model_dump(mode="json"))
    with st.expander("Candidate configurations and sensitivity"):
        st.json(
            {
                "candidates": [c.model_dump(mode="json") for c in report.candidate_configurations],
                "sensitivity": [s.model_dump(mode="json") for s in report.sensitivity_results],
            }
        )
    st.subheader("Retained loss distribution")
    if view.distribution_unavailable_reason is not None:
        st.info(view.distribution_unavailable_reason)
    elif view.loss_samples:
        st.dataframe(
            [
                {"Event": row.event_id, "Evidence kind": row.evidence_kind, "Loss": row.loss.text}
                for row in view.loss_samples
            ],
            hide_index=True,
        )
        figure = go.Figure(go.Histogram(x=[str(row.loss.value) for row in view.loss_samples]))
        figure.update_layout(
            xaxis_title=f"Retained loss ({view.loss_samples[0].loss.currency})",
            yaxis_title="Supplied sample count",
        )
        st.plotly_chart(figure, key="retained_loss_distribution")
        st.caption(
            "Only supplied retained samples are plotted. "
            "Analogue association does not establish causation."
        )
        if view.expected_loss is not None:
            st.metric("Expected loss", view.expected_loss.text)
        if view.tail_loss is not None:
            st.metric("Tail loss", view.tail_loss.text)
    st.subheader("India historical cases")
    if not view.india_cases:
        st.info("India case metadata unavailable.")
    else:
        for case in view.india_cases:
            st.write(
                f"{case.title} | {case.evidence_kind} | {case.india_case_kind or 'India case'}"
            )
    st.subheader("Historical cases")
    for case in view.cases:
        with st.expander(case.title):
            st.write(
                f"Event Class: {case.event_class.value} | "
                f"Countries: {', '.join(case.country_codes)} | Evidence: {case.evidence_kind} | "
                f"Event time: {case.event_at.isoformat()}"
            )
            for reaction in case.reactions:
                observed = reaction.reaction
                st.write(
                    f"{reaction.role}: {observed.car} {observed.unit.value} | "
                    f"{observed.measurement_dimension.value} | "
                    f"Window [{observed.window.start}, {observed.window.end}] | "
                    f"Factor: {reaction.factor_id} | Benchmark: {reaction.benchmark_id}"
                )
            if not case.reactions:
                st.info("One-day and primary-horizon reactions unavailable.")
            st.json(case.model_dump(mode="json"))
    if view.unavailable_case_ids:
        st.info(f"Case metadata unavailable: {', '.join(view.unavailable_case_ids)}")
    if view.evidence is not None:
        with st.expander("Complete retained evidence, provenance and source terms"):
            st.json(view.evidence.model_dump(mode="json"))
    with st.expander("Canonical Backtest report"):
        st.json(report.model_dump(mode="json"))
