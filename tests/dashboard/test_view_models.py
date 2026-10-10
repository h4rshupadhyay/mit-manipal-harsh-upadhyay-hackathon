"""Pure dashboard shaping retains supplied results and honest absent evidence."""

import hashlib
import json
from datetime import datetime, timedelta
from decimal import ROUND_UP, Decimal, Inexact, Rounded, localcontext
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from risk_engine.api.app import StressTestResponse
from risk_engine.api.dependencies import AnalysisRecord
from risk_engine.backtest import metrics as metric_functions
from risk_engine.backtest.module import (
    BacktestModule,
    MetricEvidence,
    UndefinedMetric,
    content_hash,
)
from risk_engine.calibration.event_study import EventWindow, WindowReaction
from risk_engine.config import PolicyConfig
from risk_engine.dashboard.view_models import (
    AnalogueLossEvidence,
    BacktestDisplayEvidence,
    BacktestEvidenceView,
    EvidenceProvenance,
    HistoricalCaseEvidence,
    LossDistributionEvidence,
    LossDistributionSummary,
    PortfolioStressView,
    ReactionEvidence,
    SignalMonitorView,
    build_backtest_evidence,
    build_portfolio_stress,
    build_signal_monitor,
)
from risk_engine.domain import (
    BacktestReport,
    CandidateConfiguration,
    EventClass,
    PortfolioMateriality,
    RiskSignal,
    SensitivityResult,
    SignalFlag,
    SourceItem,
    StressAttribution,
    StressResult,
    StressResultVersions,
)
from risk_engine.impact.module import GovernedHypothetical, ImpactEstimator, LossEvidence
from risk_engine.risk.module import RiskEngine
from risk_engine.risk.policy import ManualOverride, TriggerPolicy
from risk_engine.stress.module import StressEngine
from tests.api.test_signals import (
    EXPECTED_SIGNAL,
    TIME,
    direct_analysis,
    grouped_container,
)
from tests.api.test_stress_and_backtests import Inputs
from tests.impact.test_analogues import historical
from tests.risk.test_risk_engine import source as risk_source


def signal():
    return RiskSignal.model_validate(EXPECTED_SIGNAL)


def source():
    return SourceItem(
        source_item_id="source-1",
        source_type="news",
        provider="synthetic",
        text="Alpha Bank failed",
        published_at=TIME,
        retrieved_at=TIME,
        source_reference="synthetic:story",
        content_hash="a" * 64,
        snapshot_id="snapshot-1",
        provenance="project-authored synthetic evidence",
        license="MIT",
    )


def analysis(value=None):
    return AnalysisRecord(
        snapshot_ids=("snapshot-1",), as_of=TIME, signals=(signal() if value is None else value,)
    )


def monitor(record=None, **changes):
    arguments = dict(sources=(source(),), snapshot_ids=("snapshot-1",), as_of=TIME, mode="snapshot")
    arguments.update(changes)
    return build_signal_monitor(analysis() if record is None else record, **arguments)


def test_signal_severity_confidence_and_portfolio_fields_are_independent():
    view = monitor()
    row = view.rows[0]
    assert view.status == "ready"
    assert view.badge.mode == "snapshot"
    assert view.badge.snapshot_ids == ("snapshot-1",)
    assert view.badge.as_of == TIME
    assert row.impact_score.encoding == "severity_decile"
    assert row.impact_score.value == 8
    assert row.impact_score.text == "8 / 10"
    assert row.confidence.encoding == "calibrated_probability"
    assert row.confidence.value == 0.8
    assert row.confidence.text == "80%"
    assert row.materiality.absolute.text == "USD 20000"
    assert row.materiality.percentage.text == "2%"
    assert row.action_priority == 0.128
    assert row.signal.impact.expected_reference_loss == Decimal("100")
    assert row.source.source_reference == "synthetic:story"
    assert row.source.content_hash == "a" * 64
    assert row.source.license == "MIT"
    assert row.signal.evidence == ("Alpha Bank failed",)
    assert row.signal.versions.model_version == "synthetic-model-v1"
    assert row.signal.impact.method.value == "empirical"
    assert row.signal.impact.backoff_level == "event-class-region"


def test_signal_missing_materiality_priority_and_empty_analysis_are_explicit():
    value = signal().model_copy(
        update={
            "portfolio_materiality": None,
            "action_priority": None,
            "flags": (SignalFlag.THIN_HISTORY,),
        }
    )
    view = monitor(analysis(value))
    assert view.rows[0].materiality is None
    assert view.rows[0].action_priority is None
    assert view.rows[0].signal.flags[0].value == "thin_history"
    empty = AnalysisRecord(snapshot_ids=("snapshot-1",), as_of=TIME, signals=())
    assert monitor(empty).status == "empty"
    assert monitor(empty).rows == ()
    unavailable = build_signal_monitor(
        None,
        sources=(),
        snapshot_ids=("snapshot-1",),
        as_of=TIME,
        mode="snapshot",
        unavailable_reason="No analysis",
    )
    assert unavailable.status == "unavailable"
    assert unavailable.unavailable_reason == "No analysis"


@pytest.mark.parametrize(
    "change",
    [
        "cutoff",
        "snapshot",
        "missing_source",
        "late_source",
        "source_snapshot",
        "duplicate_source",
        "duplicate_snapshot",
    ],
)
def test_signal_rejects_conflicting_or_unavailable_source_lineage(change):
    kwargs = {}
    if change == "cutoff":
        kwargs["as_of"] = TIME + timedelta(seconds=1)
    elif change == "snapshot":
        kwargs["snapshot_ids"] = ("different",)
    elif change == "missing_source":
        kwargs["sources"] = ()
    elif change == "late_source":
        kwargs["sources"] = (
            source().model_copy(update={"retrieved_at": TIME + timedelta(seconds=1)}),
        )
    elif change == "source_snapshot":
        kwargs["sources"] = (source().model_copy(update={"snapshot_id": "different"}),)
    elif change == "duplicate_source":
        kwargs["sources"] = (source(), source())
    else:
        kwargs["snapshot_ids"] = ("snapshot-1", "snapshot-1")
    with pytest.raises(ValueError):
        monitor(**kwargs)


def grouped_analysis(*, first=None, second=None, as_of=TIME):
    first = first or risk_source("Beta Bank failed.", "source-first").model_copy(
        update={"snapshot_id": "snapshot"}
    )
    second = second or risk_source("Beta Bank failed.", "source-second").model_copy(
        update={"snapshot_id": "snapshot"}
    )
    signals = direct_analysis(grouped_container((first, second)), as_of=as_of)
    assert len(signals) == 1
    return AnalysisRecord(snapshot_ids=("snapshot",), as_of=as_of, signals=tuple(signals)), (
        first,
        second,
    )


def test_grouped_api_signal_reaches_monitor_with_complete_source_lineage():
    record, sources = grouped_analysis()

    view = build_signal_monitor(
        record, sources=sources, snapshot_ids=("snapshot",), as_of=TIME, mode="snapshot"
    )

    assert view.status == "ready"
    assert view.rows[0].source.source_item_id == "source-first"
    assert json.loads(view.rows[0].signal.versions.snapshot_version)["source_items"] == [
        item.model_dump(mode="json", exclude={"text"}) for item in sources
    ]


@pytest.mark.parametrize(
    "change", ["missing_member", "mismatched_member", "duplicate_member", "order"]
)
def test_grouped_monitor_rejects_incomplete_or_noncanonical_member_lineage(change):
    record, sources = grouped_analysis()
    kwargs = {"sources": sources}
    if change == "missing_member":
        kwargs["sources"] = sources[:1]
    elif change == "mismatched_member":
        kwargs["sources"] = (sources[0], sources[1].model_copy(update={"license": "Other"}))
    else:
        signal_value = record.signals[0]
        versions = signal_value.versions.model_dump()
        manifest = json.loads(versions["snapshot_version"])
        if change == "duplicate_member":
            manifest["source_items"].append(manifest["source_items"][0])
        else:
            manifest["source_items"].reverse()
        versions["snapshot_version"] = json.dumps(manifest)
        tampered = signal_value.model_copy(
            update={"versions": signal_value.versions.model_copy(update=versions)}
        )
        kwargs["analysis"] = AnalysisRecord(
            snapshot_ids=record.snapshot_ids, as_of=record.as_of, signals=(tampered,)
        )
    with pytest.raises(ValueError):
        build_signal_monitor(
            kwargs.pop("analysis", record),
            sources=kwargs["sources"],
            snapshot_ids=("snapshot",),
            as_of=TIME,
            mode="snapshot",
        )


def test_grouped_monitor_checks_dst_fold_availability_by_utc_instant():
    zone = ZoneInfo("America/New_York")
    early = datetime(2026, 11, 1, 1, 15, tzinfo=zone, fold=0)
    late = datetime(2026, 11, 1, 1, 0, tzinfo=zone, fold=1)
    cutoff = datetime(2026, 11, 1, 1, 45, tzinfo=zone, fold=0)
    route_cutoff = datetime(2026, 11, 1, 1, 45, tzinfo=zone, fold=1)
    first = risk_source(
        "Beta Bank failed.", "source-first", published_at=early, retrieved_at=early
    ).model_copy(update={"snapshot_id": "snapshot"})
    second = risk_source(
        "Beta Bank failed.", "source-second", published_at=late, retrieved_at=late
    ).model_copy(update={"snapshot_id": "snapshot"})
    record, sources = grouped_analysis(first=first, second=second, as_of=route_cutoff)
    earlier_record = AnalysisRecord(
        snapshot_ids=record.snapshot_ids, as_of=cutoff, signals=record.signals
    )

    with pytest.raises(ValueError, match="availability"):
        build_signal_monitor(
            earlier_record,
            sources=sources,
            snapshot_ids=("snapshot",),
            as_of=cutoff,
            mode="snapshot",
        )


def stress_response(*, gain=False, manual=False, extra_attribution=False):
    original = Inputs().original
    effective = original
    audit = None
    if manual:
        effective = original.model_copy(
            update={
                "scenario_id": "new-scenario",
                "analyst_override": True,
                "override_reason": "Explore downside",
            }
        )
        audit = ManualOverride(analyst_id="analyst-1", reason="Explore downside")
    loss = Decimal("-1000.123456789") if gain else Decimal("1000.123456789")
    after = Decimal("11000.123456789") if gain else Decimal("8999.876543211")
    rows = tuple(
        StressAttribution(dimension=dimension, label=f"{dimension}-label", loss=loss)
        for dimension in ("asset", "sector", "region", "obligor", "factor")
    )
    if extra_attribution:
        rows += tuple(
            StressAttribution(dimension=d, label=f"{d}-label", loss=loss)
            for d in ("country", "facility")
        )
    result = StressResult(
        stress_result_id="result-1",
        scenario_id=effective.scenario_id,
        portfolio_id="portfolio-1",
        base_value=Decimal("10000"),
        stressed_value=after,
        absolute_loss=loss,
        percentage_loss=-0.1000123456789 if gain else 0.1000123456789,
        valuation_currency="USD",
        attribution=rows,
        valuation_coverage=0.25,
        unsupported_position_ids=("unsupported",),
        versions=StressResultVersions(
            scenario_version="synthetic-impact-v1",
            market_version='{"snapshot_id":"market-1","schema_version":"v1","normalization_version":"market-v1"}',
            portfolio_version="portfolio-v1",
            valuation_rule_version="valuation-v1",
        ),
    )
    materiality = PortfolioMateriality(
        absolute_loss=Decimal(0) if gain else loss,
        percentage_loss=0 if gain else 0.1000123456789,
        currency="USD",
    )
    decision = TriggerPolicy(PolicyConfig(), as_of=TIME).evaluate(
        signal(), materiality, manual_override=audit
    )
    return StressTestResponse(
        execution_status="manual" if manual else "blocked",
        signal_id=signal().signal_id,
        source_item_id="source-1",
        snapshot_ids=("snapshot-1",),
        as_of=TIME,
        portfolio_id="portfolio-1",
        portfolio_version="portfolio-v1",
        market_snapshot_id="market-1",
        original_scenario=original,
        scenario=effective,
        stress_result=result,
        trigger_decision=decision,
    )


def test_stress_preserves_exact_supplied_values_all_attribution_and_unsupported_coverage():
    view = build_portfolio_stress(stress_response(extra_attribution=True))
    assert view.status == "ready"
    assert view.execution_status == "blocked"
    assert view.base_value.text == "USD 10000"
    assert view.stressed_value.text == "USD 8999.876543211"
    assert view.absolute_loss.text == "USD 1000.123456789"
    assert view.percentage_loss.text == "10.00123456789%"
    assert view.valuation_coverage.text == "25%"
    assert view.unsupported_position_ids == ("unsupported",)
    assert {row.dimension.value for row in view.attribution} == {
        "asset",
        "sector",
        "region",
        "country",
        "obligor",
        "facility",
        "factor",
    }
    assert all(row.loss.text == "USD 1000.123456789" for row in view.attribution)
    assert view.unavailable_dimensions == ()
    assert view.response.stress_result.versions.valuation_rule_version == "valuation-v1"
    assert len(view.response.trigger_decision.gates) == 9
    assert view.response.trigger_decision.automatic_trigger is False
    assert view.response.scenario.shocks[0].value == -10
    assert view.response.scenario.shocks[0].unit.value == "percent"
    assert view.response.scenario.shocks[0].horizon_days == 5
    assert view.response.scenario.method.value == "hypothetical"


def test_stress_gains_and_missing_country_facility_are_never_fabricated():
    view = build_portfolio_stress(stress_response(gain=True))
    assert view.absolute_loss.value == Decimal("-1000.123456789")
    assert view.absolute_loss.text == "USD -1000.123456789"
    assert view.percentage_loss.text == "-10.00123456789%"
    assert view.loss_direction == "gain"
    assert tuple(d.value for d in view.unavailable_dimensions) == ("country", "facility")
    assert view.response.trigger_decision.portfolio_materiality.absolute_loss == 0


def test_manual_stress_retains_original_new_scenario_and_full_analyst_audit():
    response = stress_response(manual=True)
    view = build_portfolio_stress(response)
    assert view.execution_status == "manual"
    assert view.response.original_scenario.scenario_id == "scenario-1"
    assert view.response.scenario.scenario_id == "new-scenario"
    assert view.response.scenario.override_reason == "Explore downside"
    assert view.response.trigger_decision.manual_override.analyst_id == "analyst-1"
    assert view.response.trigger_decision.automatic_trigger is False
    assert response.original_scenario.analyst_override is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("signal_id", "other"),
        ("market_snapshot_id", "other"),
        ("portfolio_version", "other"),
        ("as_of", TIME + timedelta(seconds=1)),
        ("execution_status", "automatic"),
    ],
)
def test_stress_rejects_tampered_response_cross_links(field, value):
    response = stress_response().model_copy(update={field: value})
    with pytest.raises(ValueError):
        build_portfolio_stress(response)


def test_stress_unavailable_requires_reason_and_never_creates_zero_result():
    view = build_portfolio_stress(None, unavailable_reason="No Stress Result supplied")
    assert view.status == "unavailable"
    assert view.response is None
    assert view.base_value is None
    assert view.absolute_loss is None
    assert view.valuation_coverage is None
    assert view.execution_status is None
    with pytest.raises(ValueError):
        build_portfolio_stress(None)


def report():
    return BacktestReport(
        report_id="report-1",
        configuration_version="config-v1",
        evaluation_start=TIME - timedelta(days=100),
        evaluation_end=TIME,
        metrics={"macro_f1": 0.72, "severity_rank": 0.41},
        sensitivity_results=(
            SensitivityResult(parameter="horizon_days", value=5, metrics={"macro_f1": 0.7}),
        ),
        candidate_configurations=(
            CandidateConfiguration(
                candidate_id="a",
                parameters={"horizon_days": 1},
                metrics={"macro_f1": 0.65},
                selected=False,
            ),
            CandidateConfiguration(
                candidate_id="b",
                parameters={"horizon_days": 5},
                metrics={"macro_f1": 0.72},
                selected=True,
            ),
        ),
        historical_case_ids=("opaque-1", "opaque-2", "looks-like-RBI-india"),
        schema_version="report-v1",
    )


def provenance(*, at=TIME, sha="b" * 64):
    return EvidenceProvenance(
        reference="synthetic:retained-evidence",
        sha256=sha,
        snapshot_id="evidence-snapshot",
        version="evidence-v1",
        source_terms=("MIT",),
        available_at=at,
    )


def analogue_row(identifier, loss, *, kind="synthetic"):
    analogue = historical(
        identifier, evidence_kind="synthetic" if kind == "synthetic" else "observed"
    )
    digest = hashlib.sha256(
        json.dumps(analogue.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return AnalogueLossEvidence(
        analogue=analogue,
        fallback=None,
        evidence_kind=kind,
        loss=LossEvidence(
            event_id=identifier,
            loss=Decimal(loss),
            evidence_sha256=digest,
            available_at=analogue.available_at,
            stress_result_id=f"stress-{identifier}",
            valuation_rule_version="valuation-v1",
        ),
        currency="USD",
        reference_basket_version="basket-v1",
        calibration_version="impact-v1",
        window=EventWindow(start=0, end=0),
        provenance=provenance(sha=digest),
    )


def display_evidence(*, include_distribution=True):
    one_day = ReactionEvidence(
        role="one_day",
        factor_id="equity",
        benchmark_id="benchmark-equity",
        spec_version="study-v1",
        available_at=TIME,
        reaction=WindowReaction(
            window=EventWindow(start=0, end=0),
            car=-0.04,
            unit="decimal",
            measurement_dimension="return",
        ),
    )
    primary = ReactionEvidence(
        role="primary",
        factor_id="equity",
        benchmark_id="benchmark-equity",
        spec_version="study-v1",
        available_at=TIME,
        reaction=WindowReaction(
            window=EventWindow(start=0, end=4),
            car=-0.1,
            unit="decimal",
            measurement_dimension="return",
        ),
    )
    cases = (
        HistoricalCaseEvidence(
            case_id="opaque-1",
            title="RBI monetary policy episode",
            event_class="Macroeconomic/Monetary",
            country_codes=("IN",),
            india_case_kind="rbi_monetary_policy",
            evidence_kind="synthetic",
            event_at=TIME - timedelta(days=60),
            provenance=provenance(),
            reactions=(one_day, primary),
            analogue_event_ids=("analogue-a",) if include_distribution else (),
        ),
        HistoricalCaseEvidence(
            case_id="opaque-2",
            title="Indian credit/default episode",
            event_class="Credit/Default",
            country_codes=("IN",),
            india_case_kind="credit_default",
            evidence_kind="synthetic",
            event_at=TIME - timedelta(days=30),
            provenance=provenance(),
            reactions=(),
            analogue_event_ids=(),
        ),
    )
    distribution = LossDistributionEvidence(
        currency="USD",
        reference_basket_version="basket-v1",
        calibration_version="impact-v1",
        samples=(analogue_row("analogue-a", "-2.25"), analogue_row("analogue-b", "8.125")),
        summary=LossDistributionSummary(
            expected_loss=Decimal("2.9"),
            lower_bound=Decimal("-2.25"),
            upper_bound=Decimal("8.125"),
            tail_loss=Decimal("8.125"),
            tail_probability=0.95,
            convention="supplied retained nearest-rank result",
            version="summary-v1",
        ),
        provenance=provenance(),
    )
    return BacktestDisplayEvidence(
        report_id="report-1",
        configuration_version="config-v1",
        schema_version="report-v1",
        report_sha256=content_hash(report()),
        evidence_kind="synthetic",
        available_at=TIME,
        provenance=(provenance(),),
        metrics=None,
        metrics_absence_reason="Detailed calibration audit not supplied",
        distribution=distribution if include_distribution else None,
        distribution_absence_reason=None if include_distribution else "No retained loss samples",
        cases=cases,
    )


def test_backtest_keeps_all_candidates_metrics_and_supplied_real_loss_samples():
    view = build_backtest_evidence(report(), as_of=TIME, evidence=display_evidence())
    assert view.status == "ready"
    assert view.evidence_kind == "synthetic"
    assert view.report.metrics == {"macro_f1": 0.72, "severity_rank": 0.41}
    assert tuple(c.candidate_id for c in view.report.candidate_configurations) == ("a", "b")
    assert view.report.sensitivity_results[0].metrics["macro_f1"] == 0.7
    assert tuple(row.loss.value for row in view.loss_samples) == (
        Decimal("-2.25"),
        Decimal("8.125"),
    )
    assert tuple(row.loss.text for row in view.loss_samples) == ("USD -2.25", "USD 8.125")
    assert all(row.evidence_kind == "synthetic" for row in view.loss_samples)
    assert len(view.evidence.distribution.samples[0].analogue.scenario.windows[0].shocks) == 6
    assert view.expected_loss.text == "USD 2.9"
    assert view.tail_loss.text == "USD 8.125"
    assert view.evidence.distribution.summary.convention == "supplied retained nearest-rank result"


def test_india_cases_and_horizons_are_only_explicit_supplied_metadata():
    view = build_backtest_evidence(report(), as_of=TIME, evidence=display_evidence())
    assert tuple(c.case_id for c in view.india_cases) == ("opaque-1", "opaque-2")
    assert view.india_cases[0].india_case_kind == "rbi_monetary_policy"
    assert view.india_cases[1].india_case_kind == "credit_default"
    assert tuple(r.role for r in view.cases[0].reactions) == ("one_day", "primary")
    assert view.cases[0].reactions[1].reaction.window.end == 4
    assert view.cases[1].reactions == ()
    assert view.unavailable_case_ids == ("looks-like-RBI-india",)
    assert view.evidence.metrics is None
    assert view.metrics_unavailable_reason == "Detailed calibration audit not supplied"


def test_backtest_missing_distribution_or_all_display_evidence_stays_unavailable():
    view = build_backtest_evidence(
        report(), as_of=TIME, evidence=display_evidence(include_distribution=False)
    )
    assert view.loss_samples == ()
    assert view.expected_loss is None
    assert view.tail_loss is None
    assert view.distribution_unavailable_reason == "No retained loss samples"
    plain = build_backtest_evidence(report(), as_of=TIME)
    assert plain.report is not None
    assert plain.evidence_kind is None
    assert plain.india_cases == ()
    assert plain.loss_samples == ()
    assert plain.unavailable_case_ids == ("opaque-1", "opaque-2", "looks-like-RBI-india")
    missing = build_backtest_evidence(
        None, as_of=TIME, unavailable_reason="Unresolved Backtest artifact"
    )
    assert missing.status == "unavailable"
    assert missing.report is None
    assert missing.loss_samples == ()
    with pytest.raises(ValueError):
        build_backtest_evidence(None, as_of=TIME)
    with pytest.raises(ValueError):
        build_backtest_evidence(
            None, as_of=TIME, evidence=display_evidence(), unavailable_reason="missing"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("report_id", "other"),
        ("configuration_version", "other"),
        ("schema_version", "other"),
        ("report_sha256", "c" * 64),
        ("available_at", TIME + timedelta(days=1)),
    ],
)
def test_backtest_rejects_wrong_report_versions_hash_and_future_evidence(field, value):
    evidence = display_evidence().model_copy(update={field: value})
    with pytest.raises(ValueError):
        build_backtest_evidence(report(), as_of=TIME, evidence=evidence)


@pytest.mark.parametrize("change", ["case_id", "late_case", "analogue_link", "future_report"])
def test_backtest_rejects_case_identity_chronology_and_analogue_cross_links(change):
    evidence = display_evidence()
    result = report()
    if change == "future_report":
        result = result.model_copy(update={"evaluation_end": TIME + timedelta(seconds=1)})
    else:
        case = evidence.cases[0]
        if change == "case_id":
            case = case.model_copy(update={"case_id": "not-in-report"})
        elif change == "late_case":
            case = case.model_copy(update={"event_at": TIME + timedelta(days=1)})
        else:
            case = case.model_copy(update={"analogue_event_ids": ("absent",)})
        evidence = evidence.model_copy(update={"cases": (case, evidence.cases[1])})
    with pytest.raises(ValueError):
        build_backtest_evidence(result, as_of=TIME, evidence=evidence)


@pytest.mark.parametrize(
    "change", ["event_id", "currency", "basket", "calibration", "kind", "window"]
)
def test_analogue_distribution_rejects_inconsistent_supplied_evidence(change):
    evidence = display_evidence()
    row = evidence.distribution.samples[0]
    if change == "event_id":
        row = row.model_copy(update={"loss": row.loss.model_copy(update={"event_id": "other"})})
    else:
        field, value = {
            "currency": ("currency", "INR"),
            "basket": ("reference_basket_version", "other"),
            "calibration": ("calibration_version", "other"),
            "kind": ("evidence_kind", "empirical"),
            "window": ("window", EventWindow(start=0, end=4)),
        }[change]
        row = row.model_copy(update={field: value})
    distribution = evidence.distribution.model_copy(update={"samples": (row,)})
    evidence = evidence.model_copy(update={"distribution": distribution})
    with pytest.raises(ValueError):
        build_backtest_evidence(report(), as_of=TIME, evidence=evidence)


def manifest_signal():
    value = signal().model_dump(mode="json")
    value["portfolio_materiality"] = None
    value["action_priority"] = None
    definition = {
        "version": "min-entity-class-raw-v1",
        "event_model_version": "event-v1",
        "entity_linker_version": "linker-v1",
    }
    confidence = {
        "probability": 0.8,
        "evidence_kind": "synthetic",
        "target": "entity_and_event_class_joint_correctness",
        "calibration_version": "confidence-v1",
        "evidence_hash": "b" * 64,
        "fitted_artifact_hash": "c" * 64,
        "score_definition": definition,
        "transform_version": "binary-temperature-logit-clip1e-12-v1",
    }
    calibration = {
        "manifest_version": "risk-engine-calibration-manifest-v1",
        "confidence": confidence,
        "confidence_calibration": {
            "target": confidence["target"],
            "calibration_version": "confidence-v1",
            "evidence_kind": "synthetic",
            "calibration_evidence_snapshot_id": "confidence-snapshot",
            "calibration_evidence_snapshot_hash": "d" * 64,
            "development_start": "2025-01-01T00:00:00Z",
            "development_end": "2025-12-01T00:00:00Z",
            "frozen_at": "2025-12-02T00:00:00Z",
            "score_definition": definition,
            "fit_version": "fit-v1",
            "transform_version": confidence["transform_version"],
            "temperature": 1,
            "evidence_hash": "b" * 64,
            "fitted_artifact_hash": "c" * 64,
            "source_terms": ["MIT"],
        },
        "impact": {
            "calibration_version": "synthetic-impact-v1",
            "reference_basket_version": "reference-basket-v1",
        },
    }
    item = source().model_dump(mode="json")
    item.pop("text")
    value["versions"] = {
        "schema_version": "risk-signal-v1",
        "snapshot_version": json.dumps(
            {"manifest_version": "risk-engine-snapshot-manifest-v1", "source_items": [item]}
        ),
        "calibration_version": json.dumps(calibration),
        "model_version": json.dumps(
            {
                "manifest_version": "risk-engine-model-manifest-v1",
                "models": {
                    "event_model_version": "event-v1",
                    "sentiment_model_version": "sentiment-v1",
                    "entity_linker_version": "linker-v1",
                },
                "action_priority": {
                    "method": "impact-confidence-percentage-materiality-product-v1",
                    "validation_status": "unvalidated",
                    "selection_basis": "requirement-driven",
                },
                "portfolio_materiality": None,
            }
        ),
    }
    return RiskSignal.model_validate(value)


@pytest.mark.parametrize(
    "change",
    [
        "source_hash",
        "calibration_impact",
        "calibration_confidence",
        "late_freeze",
        "fit_hash",
        "model_linker",
    ],
)
def test_signal_rejects_known_production_manifests_with_conflicting_provenance(change):
    value = manifest_signal()
    versions = value.versions.model_dump()
    if change == "source_hash":
        snapshot = json.loads(versions["snapshot_version"])
        snapshot["source_items"][0]["content_hash"] = "f" * 64
        versions["snapshot_version"] = json.dumps(snapshot)
    elif change == "model_linker":
        model = json.loads(versions["model_version"])
        model["models"]["entity_linker_version"] = "other"
        versions["model_version"] = json.dumps(model)
    else:
        calibration = json.loads(versions["calibration_version"])
        if change == "calibration_impact":
            calibration["impact"]["reference_basket_version"] = "other"
        elif change == "calibration_confidence":
            calibration["confidence"]["probability"] = 0.6
        elif change == "late_freeze":
            calibration["confidence_calibration"]["frozen_at"] = "2027-01-01T00:00:00Z"
        else:
            calibration["confidence_calibration"]["fitted_artifact_hash"] = "f" * 64
        versions["calibration_version"] = json.dumps(calibration)
    value = RiskSignal.model_validate({**value.model_dump(), "versions": versions})
    with pytest.raises(ValueError):
        monitor(analysis(value))


def test_retained_analogue_loss_cannot_bind_different_content_with_same_event_id():
    evidence = display_evidence()
    row = evidence.distribution.samples[0]
    forged = row.analogue.model_copy(update={"event_class": EventClass.CREDIT_DEFAULT})
    row = row.model_copy(update={"analogue": forged})
    distribution = evidence.distribution.model_copy(
        update={"samples": (row, evidence.distribution.samples[1])}
    )
    evidence = evidence.model_copy(update={"distribution": distribution})
    with pytest.raises(ValueError):
        build_backtest_evidence(report(), as_of=TIME, evidence=evidence)


def test_supplied_hypothetical_and_empirical_rows_retain_their_distinct_labels():
    inputs = Inputs()
    fallback = GovernedHypothetical(
        scenario=inputs.original,
        base_market=inputs.snapshot,
        window=EventWindow(start=0, end=4),
        available_at=TIME,
        governance_version="governance-v1",
        source_terms="MIT",
        source_sha256="e" * 64,
    )
    digest = hashlib.sha256(
        json.dumps(fallback.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    hypothetical = AnalogueLossEvidence(
        analogue=None,
        fallback=fallback,
        evidence_kind="hypothetical",
        loss=LossEvidence(
            event_id="scenario-1",
            loss=Decimal("13.75"),
            evidence_sha256=digest,
            available_at=TIME,
            stress_result_id="fallback-stress",
            valuation_rule_version="valuation-v1",
        ),
        currency="USD",
        reference_basket_version="basket-v1",
        calibration_version="synthetic-impact-v1",
        window=fallback.window,
        provenance=provenance(sha=digest),
    )
    evidence = display_evidence()
    distribution = LossDistributionEvidence(
        currency="USD",
        reference_basket_version="basket-v1",
        calibration_version="synthetic-impact-v1",
        samples=(hypothetical,),
        summary=None,
        provenance=provenance(),
    )
    evidence = evidence.model_copy(
        update={"distribution": distribution, "cases": (), "evidence_kind": "hypothetical"}
    )
    view = build_backtest_evidence(report(), as_of=TIME, evidence=evidence)
    assert view.evidence_kind == "hypothetical"
    assert view.loss_samples[0].evidence_kind == "hypothetical"
    assert view.loss_samples[0].loss.text == "USD 13.75"
    assert view.expected_loss is None
    assert view.evidence.distribution.samples[0].fallback.governance_version == "governance-v1"
    empirical = analogue_row("observed-a", "-1.75", kind="empirical")
    distribution = LossDistributionEvidence(
        currency="USD",
        reference_basket_version="basket-v1",
        calibration_version="impact-v1",
        samples=(empirical,),
        summary=None,
        provenance=provenance(),
    )
    evidence = evidence.model_copy(
        update={"distribution": distribution, "evidence_kind": "empirical"}
    )
    view = build_backtest_evidence(report(), as_of=TIME, evidence=evidence)
    assert view.loss_samples[0].evidence_kind == "empirical"
    assert view.loss_samples[0].loss.text == "USD -1.75"


def test_historical_case_metadata_cannot_be_outside_report_evaluation_period():
    evidence = display_evidence()
    case = evidence.cases[0].model_copy(update={"event_at": TIME - timedelta(days=101)})
    evidence = evidence.model_copy(update={"cases": (case, evidence.cases[1])})
    with pytest.raises(ValueError):
        build_backtest_evidence(report(), as_of=TIME, evidence=evidence)


def detailed_evidence():
    result = report().model_copy(
        update={"metrics": {"macro_f1": 1.0, "brier_score": 0.04, "interval_coverage": 1.0}}
    )
    detailed = MetricEvidence(
        classification=metric_functions.classification_metrics(
            [EventClass.CREDIT_DEFAULT], [EventClass.CREDIT_DEFAULT]
        ),
        entity_link_accuracy=1.0,
        confidence=metric_functions.confidence_metrics([True], [0.8], [0, 0.5, 1]),
        severity=None,
        interval=metric_functions.interval_metrics([-2.25], [8.125], [2.9], "USD", 0.95),
        scenario=None,
        valuation=None,
        alerts=None,
        undefined=(
            UndefinedMetric(
                metric="severity", reason="Insufficient retained ranking pairs", applicable_count=1
            ),
        ),
        scalar_projection={"macro_f1": 1.0, "brier_score": 0.04, "interval_coverage": 1.0},
        monetary_projection_units=(("interval_mean_width", "USD"),),
    )
    evidence = display_evidence().model_copy(
        update={
            "report_sha256": content_hash(result),
            "metrics": detailed,
            "metrics_absence_reason": None,
        }
    )
    return result, evidence


def test_backtest_retains_confusion_reliability_interval_units_and_undefined_reasons():
    result, evidence = detailed_evidence()
    view = build_backtest_evidence(result, as_of=TIME, evidence=evidence)
    detail = view.evidence.metrics
    assert detail.classification.confusion[2][2] == 1
    assert detail.confidence.reliability[0].count == 0
    assert detail.confidence.reliability[0].accuracy is None
    assert detail.confidence.reliability[1].count == 1
    assert detail.interval.coverage == 1
    assert detail.interval.mean_width == 10.375
    assert detail.interval.unit == "USD"
    assert detail.undefined[0].reason == "Insufficient retained ranking pairs"
    assert detail.undefined[0].applicable_count == 1
    assert detail.monetary_projection_units == (("interval_mean_width", "USD"),)
    assert view.metrics_unavailable_reason is None
    changed = detail.model_copy(update={"scalar_projection": {"macro_f1": 0.2}})
    evidence = evidence.model_copy(update={"metrics": changed})
    with pytest.raises(ValueError):
        build_backtest_evidence(result, as_of=TIME, evidence=evidence)


def test_all_builders_and_serialized_views_use_fixed_decimal_context_without_changing_ambient():
    response = stress_response()
    result, evidence = detailed_evidence()
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        monitor_view = monitor(analysis(manifest_signal()))
        stress_view = build_portfolio_stress(response)
        backtest_view = build_backtest_evidence(result, as_of=TIME, evidence=evidence)
        assert context.prec == 2
        assert context.rounding == ROUND_UP
        assert context.traps[Inexact] is True
        assert stress_view.absolute_loss.text == "USD 1000.123456789"
        assert stress_view.percentage_loss.text == "10.00123456789%"
        assert monitor_view.rows[0].confidence.text == "80%"
        assert backtest_view.expected_loss.text == "USD 2.9"
        restored = PortfolioStressView.model_validate_json(stress_view.model_dump_json())
        assert restored.absolute_loss.value == Decimal("1000.123456789")
        assert (
            SignalMonitorView.model_validate_json(monitor_view.model_dump_json())
            .rows[0]
            .impact_score.value
            == 8
        )
        assert BacktestEvidenceView.model_validate_json(
            backtest_view.model_dump_json()
        ).loss_samples[1].loss.value == Decimal("8.125")


def test_view_copies_isolate_canonical_input_dicts_and_repeated_builds():
    result, evidence = detailed_evidence()
    first = build_backtest_evidence(result, as_of=TIME, evidence=evidence)
    second = build_backtest_evidence(result, as_of=TIME, evidence=evidence)
    first.report.metrics["macro_f1"] = -5
    first.report.candidate_configurations[0].parameters["horizon_days"] = 99
    first.report.sensitivity_results[0].metrics["macro_f1"] = -5
    first.evidence.metrics.scalar_projection["macro_f1"] = -5
    assert result.metrics["macro_f1"] == 1
    assert result.candidate_configurations[0].parameters["horizon_days"] == 1
    assert result.sensitivity_results[0].metrics["macro_f1"] == 0.7
    assert evidence.metrics.scalar_projection["macro_f1"] == 1
    assert second.report.metrics["macro_f1"] == 1
    assert second.evidence.metrics.scalar_projection["macro_f1"] == 1
    with pytest.raises(ValidationError):
        second.status = "unavailable"


def test_formatting_never_invokes_model_selection_valuation_calibration_or_network(monkeypatch):
    import socket

    result, evidence = detailed_evidence()
    response = stress_response()

    def forbidden(*args, **kwargs):
        raise AssertionError("formatting attempted financial recomputation or network")

    monkeypatch.setattr(StressEngine, "run", forbidden)
    monkeypatch.setattr(RiskEngine, "analyze", forbidden)
    monkeypatch.setattr(ImpactEstimator, "estimate", forbidden)
    monkeypatch.setattr(BacktestModule, "evaluate", forbidden)
    monkeypatch.setattr(BacktestModule, "_development_projections", forbidden)
    monkeypatch.setattr("risk_engine.backtest.module._metrics", forbidden)
    monkeypatch.setattr(metric_functions, "classification_metrics", forbidden)
    monkeypatch.setattr(metric_functions, "confidence_metrics", forbidden)
    monkeypatch.setattr(metric_functions, "interval_metrics", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    assert monitor(analysis(manifest_signal())).rows[0].signal.signal_id == "signal:synthetic-1"
    assert build_portfolio_stress(response).absolute_loss.text == "USD 1000.123456789"
    assert (
        build_backtest_evidence(result, as_of=TIME, evidence=evidence).tail_loss.text == "USD 8.125"
    )
