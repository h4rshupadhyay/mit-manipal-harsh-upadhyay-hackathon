"""Read-only Risk Signals with independent severity and reliability fields."""

import streamlit as st

from risk_engine.dashboard.view_models import SignalMonitorView


def render(view: SignalMonitorView | None) -> None:
    st.header("Signal Monitor")
    if view is None:
        st.info("Risk Signal evidence unavailable. No retained analysis is configured.")
        return
    st.caption(f"Mode: {view.badge.mode} | As of: {view.badge.as_of.isoformat()}")
    st.caption(f"Source snapshots: {', '.join(view.badge.snapshot_ids)}")
    if view.status == "unavailable":
        st.info(view.unavailable_reason)
        return
    if view.status == "empty":
        st.info("No Risk Signals in this retained analysis.")
        return
    st.caption(
        "Impact Score measures market severity. "
        "Confidence measures entity and Event Class reliability."
    )
    for row in view.rows:
        signal, source = row.signal, row.source
        st.subheader(f"Risk Signal {signal.signal_id}")
        columns = st.columns(4)
        columns[0].metric("Impact Score", row.impact_score.text)
        columns[1].metric("Confidence", row.confidence.text)
        columns[2].metric(
            "Portfolio Materiality",
            "Unavailable" if row.materiality is None else row.materiality.absolute.text,
        )
        columns[3].metric(
            "Action Priority",
            "Unavailable" if row.action_priority is None else str(row.action_priority),
        )
        if row.materiality is not None:
            st.caption(f"Portfolio Materiality percentage: {row.materiality.percentage.text}")
        st.write(f"Event Class: {signal.event_class.value} | Sentiment: {signal.sentiment}")
        st.write(signal.rationale)
        st.write(
            f"Impact method: {signal.impact.method.value} | "
            f"Flags: {', '.join(f.value for f in signal.flags) or 'None'}"
        )
        with st.expander("Source Item and interpretation evidence"):
            st.write(source.text)
            st.write(f"Source reference: {source.source_reference}")
            st.write(
                f"Provider: {source.provider} | Published: {source.published_at.isoformat()} | "
                f"Retrieved: {source.retrieved_at.isoformat()}"
            )
            st.write(f"Provenance: {source.provenance} | Source terms: {source.license}")
            st.write(f"Content hash: {source.content_hash} | Snapshot: {source.snapshot_id}")
            st.json(
                {
                    "source_item": source.model_dump(mode="json"),
                    "risk_signal": signal.model_dump(mode="json"),
                }
            )
