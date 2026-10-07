"""Explicit dashboard composition; default launch loads no engine or evidence."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

import httpx
import streamlit as st

from risk_engine.api.app import StressTestRequest, StressTestResponse
from risk_engine.api.dependencies import AnalysisRecord
from risk_engine.dashboard.pages import backtest_evidence, portfolio_stress, signal_monitor
from risk_engine.dashboard.pages.portfolio_stress import StressTestClient
from risk_engine.dashboard.view_models import (
    BacktestDisplayEvidence,
    build_backtest_evidence,
    build_portfolio_stress,
    build_signal_monitor,
)
from risk_engine.domain import BacktestReport, SourceItem


@dataclass(frozen=True)
class DashboardEvidence:
    """Trusted caller composes retained records with an explicit aware cutoff."""

    as_of: datetime
    snapshot_ids: tuple[str, ...]
    mode: Literal["snapshot", "live"]
    sources: tuple[SourceItem, ...] = ()
    analysis: AnalysisRecord | None = None
    stress: StressTestResponse | None = None
    backtest: BacktestReport | None = None
    backtest_evidence: BacktestDisplayEvidence | None = None


@dataclass(frozen=True)
class HttpStressTestClient:
    """Only an explicitly submitted request contacts the configured Task 29 API."""

    api_url: str
    timeout: float
    transport: httpx.BaseTransport | None = None

    def __post_init__(self) -> None:
        url = urlsplit(self.api_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("an explicit HTTP API URL without embedded credentials is required")
        if url.query or url.fragment or not 0 < self.timeout < float("inf"):
            raise ValueError("an explicit finite positive timeout and clean API URL are required")

    def submit(self, request: StressTestRequest) -> StressTestResponse:
        with httpx.Client(
            timeout=self.timeout, transport=self.transport, follow_redirects=False
        ) as client:
            response = client.post(
                f"{self.api_url.rstrip('/')}/v1/stress-tests", json=request.model_dump(mode="json")
            )
            response.raise_for_status()
            return StressTestResponse.model_validate(response.json())


PAGE_RENDERERS = {
    "Signal Monitor": signal_monitor.render,
    "Portfolio Stress": portfolio_stress.render,
    "Backtest Evidence": backtest_evidence.render,
}


def render_app(
    evidence: DashboardEvidence | None = None, client: StressTestClient | None = None
) -> None:
    st.set_page_config(page_title="Financial Risk Intelligence", layout="wide")
    # These modules are renderers, not standalone automatically discovered scripts.
    st.set_option("client.showSidebarNavigation", False)
    st.title("Financial Risk Intelligence")
    page = st.sidebar.radio("Analyst pages", tuple(PAGE_RENDERERS))
    st.sidebar.caption("Reads retained evidence. Manual scenarios require explicit submission.")
    try:
        if page == "Signal Monitor":
            monitor = (
                None
                if evidence is None
                else build_signal_monitor(
                    evidence.analysis,
                    sources=evidence.sources,
                    snapshot_ids=evidence.snapshot_ids,
                    as_of=evidence.as_of,
                    mode=evidence.mode,
                    unavailable_reason="Risk Signal analysis unavailable"
                    if evidence.analysis is None
                    else None,
                )
            )
            signal_monitor.render(monitor)
        elif page == "Portfolio Stress":
            stress = build_portfolio_stress(
                None if evidence is None else evidence.stress,
                unavailable_reason="Stress Result unavailable"
                if evidence is None or evidence.stress is None
                else None,
            )
            portfolio_stress.render(stress, client)
        else:
            backtest = (
                None
                if evidence is None
                else build_backtest_evidence(
                    evidence.backtest,
                    as_of=evidence.as_of,
                    evidence=evidence.backtest_evidence,
                    unavailable_reason="Backtest report unavailable"
                    if evidence.backtest is None
                    else None,
                )
            )
            backtest_evidence.render(backtest)
    except Exception:
        # Retained/provider records can be invalid. Never echo their exception text.
        st.error(
            "Evidence could not be displayed. Check its identities, versions and cutoff "
            "with your data administrator."
        )


if __name__ == "__main__":
    render_app()
