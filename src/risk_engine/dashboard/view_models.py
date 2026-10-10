"""Pure analyst presentation over explicit, already calculated offline evidence.

Builders validate and defensively copy records. They never acquire, interpret,
select, calibrate, value portfolios, or derive empirical distributions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from typing import Annotated, Literal, TypeVar

from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
    ModelWrapValidatorHandler,
    StringConstraints,
    model_validator,
)

from risk_engine.api.app import ExecutionStatus, StressTestResponse
from risk_engine.api.dependencies import AnalysisRecord
from risk_engine.backtest.module import MetricEvidence, content_hash
from risk_engine.calibration.event_study import EventWindow, WindowReaction
from risk_engine.domain import (
    AttributionDimension,
    BacktestReport,
    CurrencyCode,
    DomainModel,
    EventClass,
    ImpactScore,
    NonEmptyString,
    PortfolioMateriality,
    Probability,
    RiskSignal,
    Sha256Hex,
    SourceItem,
)
from risk_engine.impact.analogues import HistoricalAnalogue
from risk_engine.impact.module import GovernedHypothetical, LossEvidence
from risk_engine.risk.module import (
    CalibrationManifest,
    ModelManifest,
    SnapshotManifest,
    SourceItemIdentity,
)


def _decimal_context() -> Context:
    return Context(
        prec=28,
        rounding=ROUND_HALF_EVEN,
        Emin=-999999,
        Emax=999999,
        capitals=1,
        clamp=0,
        flags=[],
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )


class ViewRecord(DomainModel):
    @model_validator(mode="wrap")
    @classmethod
    def fixed_validation(
        cls, value: object, handler: ModelWrapValidatorHandler[ViewRecord]
    ) -> ViewRecord:
        with localcontext(_decimal_context()):
            return handler(value)


R = TypeVar("R", bound=BaseModel)


def _copy(value: R) -> R:
    with localcontext(_decimal_context()):
        return type(value).model_validate(value.model_dump(mode="python"))


class SnapshotBadge(ViewRecord):
    mode: Literal["snapshot", "live"]
    snapshot_ids: tuple[NonEmptyString, ...]
    as_of: AwareDatetime

    @model_validator(mode="after")
    def unique_snapshots(self) -> SnapshotBadge:
        if not self.snapshot_ids or len(set(self.snapshot_ids)) != len(self.snapshot_ids):
            raise ValueError("explicit unique snapshot IDs are required")
        return self


class MoneyDisplay(ViewRecord):
    value: Decimal
    currency: CurrencyCode
    text: NonEmptyString


class PercentageDisplay(ViewRecord):
    fraction: float
    text: NonEmptyString


def _money(value: Decimal, currency: str) -> MoneyDisplay:
    return MoneyDisplay(value=value, currency=currency, text=f"{currency} {value:f}")


def _percentage(value: float) -> PercentageDisplay:
    with localcontext(_decimal_context()):
        text = format(Decimal(str(value)) * Decimal(100), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return PercentageDisplay(fraction=value, text=f"{text}%")


class SeverityDisplay(ViewRecord):
    encoding: Literal["severity_decile"] = "severity_decile"
    value: ImpactScore
    text: NonEmptyString


class ConfidenceDisplay(ViewRecord):
    encoding: Literal["calibrated_probability"] = "calibrated_probability"
    value: Probability
    text: NonEmptyString


class MaterialityDisplay(ViewRecord):
    absolute: MoneyDisplay
    percentage: PercentageDisplay


def _materiality(value: PortfolioMateriality | None) -> MaterialityDisplay | None:
    if value is None:
        return None
    return MaterialityDisplay(
        absolute=_money(value.absolute_loss, value.currency),
        percentage=_percentage(value.percentage_loss),
    )


class SignalRow(ViewRecord):
    signal: RiskSignal
    source: SourceItem
    impact_score: SeverityDisplay
    confidence: ConfidenceDisplay
    materiality: MaterialityDisplay | None
    action_priority: float | None


class SignalMonitorView(ViewRecord):
    status: Literal["ready", "empty", "unavailable"]
    unavailable_reason: NonEmptyString | None
    badge: SnapshotBadge
    rows: tuple[SignalRow, ...]


def build_signal_monitor(
    analysis: AnalysisRecord | None,
    *,
    sources: tuple[SourceItem, ...],
    snapshot_ids: tuple[str, ...],
    as_of: datetime,
    mode: Literal["snapshot", "live"],
    unavailable_reason: str | None = None,
) -> SignalMonitorView:
    """The caller declares mode; no clock or provider state is consulted."""
    with localcontext(_decimal_context()):
        badge = SnapshotBadge(mode=mode, snapshot_ids=snapshot_ids, as_of=as_of)
        if analysis is None:
            if unavailable_reason is None or sources:
                raise ValueError("unavailable analysis requires a reason and no Source Items")
            return SignalMonitorView(
                status="unavailable", unavailable_reason=unavailable_reason, badge=badge, rows=()
            )
        if unavailable_reason is not None:
            raise ValueError("available analysis cannot have an unavailable reason")
        analysis = _copy(analysis)
        sources = tuple(_copy(source) for source in sources)
        if analysis.snapshot_ids != badge.snapshot_ids or analysis.as_of != badge.as_of:
            raise ValueError("analysis snapshot/cutoff lineage disagrees")
        indexed = {source.source_item_id: source for source in sources}
        if len(indexed) != len(sources):
            raise ValueError("duplicate Source Item identities")
        if any(
            source.snapshot_id not in snapshot_ids
            or source.retrieved_at.astimezone(UTC) > as_of.astimezone(UTC)
            or source.published_at.astimezone(UTC) > as_of.astimezone(UTC)
            for source in sources
        ):
            raise ValueError("Source Item snapshot or availability disagrees")
        rows = []
        for signal in analysis.signals:
            source = indexed.get(signal.source_item_id)
            if source is None:
                raise ValueError("Risk Signal Source Item is unavailable")
            _validate_signal_manifests(signal, source, indexed, as_of)
            rows.append(
                SignalRow(
                    signal=signal,
                    source=source,
                    impact_score=SeverityDisplay(
                        value=signal.impact.impact_score, text=f"{signal.impact.impact_score} / 10"
                    ),
                    confidence=ConfidenceDisplay(
                        value=signal.confidence, text=_percentage(signal.confidence).text
                    ),
                    materiality=_materiality(signal.portfolio_materiality),
                    action_priority=signal.action_priority,
                )
            )
        return SignalMonitorView(
            status="ready" if rows else "empty",
            unavailable_reason=None,
            badge=badge,
            rows=tuple(rows),
        )


class AttributionDisplay(ViewRecord):
    dimension: AttributionDimension
    label: NonEmptyString
    loss: MoneyDisplay


class PortfolioStressView(ViewRecord):
    status: Literal["ready", "unavailable"]
    unavailable_reason: NonEmptyString | None
    response: StressTestResponse | None
    execution_status: ExecutionStatus | None = None
    base_value: MoneyDisplay | None = None
    stressed_value: MoneyDisplay | None = None
    absolute_loss: MoneyDisplay | None = None
    percentage_loss: PercentageDisplay | None = None
    loss_direction: Literal["loss", "gain", "flat"] | None = None
    attribution: tuple[AttributionDisplay, ...] = ()
    unavailable_dimensions: tuple[AttributionDimension, ...] = ()
    valuation_coverage: PercentageDisplay | None = None
    unsupported_position_ids: tuple[NonEmptyString, ...] = ()


def build_portfolio_stress(
    response: StressTestResponse | None,
    *,
    unavailable_reason: str | None = None,
) -> PortfolioStressView:
    """Present an immutable Task 29 preview and its independent permission audit."""
    with localcontext(_decimal_context()):
        if response is None:
            if unavailable_reason is None:
                raise ValueError("unavailable Stress Result requires a reason")
            return PortfolioStressView(
                status="unavailable", unavailable_reason=unavailable_reason, response=None
            )
        if unavailable_reason is not None:
            raise ValueError("available Stress Result cannot have an unavailable reason")
        response = _copy(response)
        if len(set(response.snapshot_ids)) != len(response.snapshot_ids):
            raise ValueError("duplicate source snapshot identities")
        for scenario in (response.original_scenario, response.scenario):
            if len({shock.horizon_days for shock in scenario.shocks}) != 1:
                raise ValueError("Joint Shock Vector horizons disagree")
        result = response.stress_result
        dimensions = {row.dimension for row in result.attribution}
        return PortfolioStressView(
            status="ready",
            unavailable_reason=None,
            response=response,
            execution_status=response.execution_status,
            base_value=_money(result.base_value, result.valuation_currency),
            stressed_value=_money(result.stressed_value, result.valuation_currency),
            absolute_loss=_money(result.absolute_loss, result.valuation_currency),
            percentage_loss=_percentage(result.percentage_loss),
            loss_direction="loss"
            if result.absolute_loss > 0
            else "gain"
            if result.absolute_loss < 0
            else "flat",
            attribution=tuple(
                AttributionDisplay(
                    dimension=row.dimension,
                    label=row.label,
                    loss=_money(row.loss, result.valuation_currency),
                )
                for row in result.attribution
            ),
            unavailable_dimensions=tuple(d for d in AttributionDimension if d not in dimensions),
            valuation_coverage=_percentage(result.valuation_coverage),
            unsupported_position_ids=result.unsupported_position_ids,
        )


EvidenceKind = Literal["empirical", "hypothetical", "synthetic"]
CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]


class EvidenceProvenance(ViewRecord):
    reference: NonEmptyString
    sha256: Sha256Hex
    snapshot_id: NonEmptyString
    version: NonEmptyString
    source_terms: Annotated[tuple[NonEmptyString, ...], Field(min_length=1)]
    available_at: AwareDatetime


class AnalogueLossEvidence(ViewRecord):
    """One retained Reference Basket loss, never reconstructed from its vector."""

    analogue: HistoricalAnalogue | None
    fallback: GovernedHypothetical | None
    evidence_kind: EvidenceKind
    loss: LossEvidence
    currency: CurrencyCode
    reference_basket_version: NonEmptyString
    calibration_version: NonEmptyString
    window: EventWindow
    provenance: EvidenceProvenance

    @model_validator(mode="after")
    def retained_loss_lineage(self) -> AnalogueLossEvidence:
        if self.loss.evidence_sha256 != self.provenance.sha256:
            raise ValueError("loss evidence hash disagrees with provenance")
        if self.loss.available_at > self.provenance.available_at:
            raise ValueError("loss evidence unavailable by provenance timestamp")
        if self.evidence_kind == "hypothetical":
            fallback = self.fallback
            if self.analogue is not None or fallback is None:
                raise ValueError(
                    "hypothetical loss requires a complete governed fallback, no analogue"
                )
            if (
                fallback.scenario.scenario_id != self.loss.event_id
                or fallback.scenario.calibration_version != self.calibration_version
                or fallback.window != self.window
                or fallback.available_at != self.loss.available_at
                or _impact_evidence_hash(fallback) != self.loss.evidence_sha256
            ):
                raise ValueError("governed fallback identity/window/calibration/hash disagrees")
        else:
            if self.analogue is None or self.fallback is not None:
                raise ValueError("historical loss requires a canonical Historical Analogue")
            analogue = self.analogue
            required_kind = "observed" if self.evidence_kind == "empirical" else "synthetic"
            if (
                analogue.event_id != self.loss.event_id
                or analogue.evidence_kind != required_kind
                or analogue.available_at != self.loss.available_at
                or analogue.scenario is None
                or self.window not in tuple(w.window for w in analogue.scenario.windows)
                or _impact_evidence_hash(analogue) != self.loss.evidence_sha256
            ):
                raise ValueError(
                    "Historical Analogue identity/kind/window/hash/availability disagrees"
                )
        return self


class LossDistributionSummary(ViewRecord):
    """Caller-supplied statistics and their disclosed convention; no fitting here."""

    expected_loss: Decimal
    lower_bound: Decimal
    upper_bound: Decimal
    tail_loss: Decimal
    tail_probability: Probability
    convention: NonEmptyString
    version: NonEmptyString

    @model_validator(mode="after")
    def supplied_bounds(self) -> LossDistributionSummary:
        if not self.lower_bound <= self.expected_loss <= self.upper_bound:
            raise ValueError("supplied expected loss is outside supplied bounds")
        return self


class LossDistributionEvidence(ViewRecord):
    currency: CurrencyCode
    reference_basket_version: NonEmptyString
    calibration_version: NonEmptyString
    samples: Annotated[tuple[AnalogueLossEvidence, ...], Field(min_length=1)]
    summary: LossDistributionSummary | None
    provenance: EvidenceProvenance

    @model_validator(mode="after")
    def one_comparable_retained_cohort(self) -> LossDistributionEvidence:
        ids = tuple(row.loss.event_id for row in self.samples)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate retained loss identity")
        if len({row.window for row in self.samples}) != 1:
            raise ValueError("distribution cannot mix event windows")
        for row in self.samples:
            if (row.currency, row.reference_basket_version, row.calibration_version) != (
                self.currency,
                self.reference_basket_version,
                self.calibration_version,
            ) or row.provenance.available_at > self.provenance.available_at:
                raise ValueError("loss cohort currency/version/availability disagrees")
        return self


class ReactionEvidence(ViewRecord):
    role: Literal["one_day", "primary"]
    factor_id: NonEmptyString
    benchmark_id: NonEmptyString
    spec_version: NonEmptyString
    available_at: AwareDatetime
    reaction: WindowReaction

    @model_validator(mode="after")
    def declared_one_day(self) -> ReactionEvidence:
        if self.role == "one_day" and (
            self.reaction.window.start != 0 or self.reaction.window.end != 0
        ):
            raise ValueError("one-day reaction must refer to event session zero only")
        return self


class HistoricalCaseEvidence(ViewRecord):
    case_id: NonEmptyString
    title: NonEmptyString
    event_class: EventClass
    country_codes: tuple[CountryCode, ...]
    india_case_kind: Literal["rbi_monetary_policy", "credit_default"] | None
    evidence_kind: EvidenceKind
    event_at: AwareDatetime
    provenance: EvidenceProvenance
    reactions: tuple[ReactionEvidence, ...]
    analogue_event_ids: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def case_metadata_is_explicit(self) -> HistoricalCaseEvidence:
        if len(set(self.country_codes)) != len(self.country_codes) or len(
            set(self.analogue_event_ids)
        ) != len(self.analogue_event_ids):
            raise ValueError("duplicate country or analogue identity")
        if self.india_case_kind is not None:
            expected = (
                EventClass.MACROECONOMIC_MONETARY
                if (self.india_case_kind == "rbi_monetary_policy")
                else EventClass.CREDIT_DEFAULT
            )
            if "IN" not in self.country_codes or self.event_class is not expected:
                raise ValueError("India case label disagrees with explicit geography/Event Class")
        if self.event_at > self.provenance.available_at or any(
            reaction.available_at > self.provenance.available_at
            or reaction.available_at < self.event_at
            for reaction in self.reactions
        ):
            raise ValueError("case reactions or metadata unavailable by provenance timestamp")
        keys = tuple((r.role, r.factor_id) for r in self.reactions)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate case reaction role/factor")
        return self


class BacktestDisplayEvidence(ViewRecord):
    """Explicit manual extraction from retained artifacts, bound to exact report bytes.

    Full BacktestAudit validation recomputes metrics. Consume its retained
    MetricEvidence here instead, with caller-declared extraction provenance.
    """

    report_id: NonEmptyString
    configuration_version: NonEmptyString
    schema_version: NonEmptyString
    report_sha256: Sha256Hex
    evidence_kind: EvidenceKind
    available_at: AwareDatetime
    provenance: Annotated[tuple[EvidenceProvenance, ...], Field(min_length=1)]
    metrics: MetricEvidence | None
    metrics_absence_reason: NonEmptyString | None
    distribution: LossDistributionEvidence | None
    distribution_absence_reason: NonEmptyString | None
    cases: tuple[HistoricalCaseEvidence, ...]

    @model_validator(mode="after")
    def explicit_evidence_availability(self) -> BacktestDisplayEvidence:
        if (self.metrics is None) != (self.metrics_absence_reason is not None) or (
            self.distribution is None
        ) != (self.distribution_absence_reason is not None):
            raise ValueError("missing evidence requires exactly one absence reason")
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("duplicate historical case identity")
        times = (
            tuple(p.available_at for p in self.provenance)
            + tuple(c.provenance.available_at for c in self.cases)
            + (() if self.distribution is None else (self.distribution.provenance.available_at,))
        )
        if any(t > self.available_at for t in times):
            raise ValueError("display metadata unavailable by evidence timestamp")
        if self.evidence_kind == "empirical" and (
            any(c.evidence_kind != "empirical" for c in self.cases)
            or (
                self.distribution is not None
                and any(s.evidence_kind != "empirical" for s in self.distribution.samples)
            )
        ):
            raise ValueError(
                "empirical display evidence cannot contain synthetic/hypothetical evidence"
            )
        return self


class LossSampleDisplay(ViewRecord):
    event_id: NonEmptyString
    evidence_kind: EvidenceKind
    loss: MoneyDisplay


class BacktestEvidenceView(ViewRecord):
    status: Literal["ready", "unavailable"]
    unavailable_reason: NonEmptyString | None
    as_of: AwareDatetime
    report: BacktestReport | None
    evidence: BacktestDisplayEvidence | None
    evidence_kind: EvidenceKind | None = None
    metrics_unavailable_reason: NonEmptyString | None = None
    distribution_unavailable_reason: NonEmptyString | None = None
    loss_samples: tuple[LossSampleDisplay, ...] = ()
    expected_loss: MoneyDisplay | None = None
    tail_loss: MoneyDisplay | None = None
    cases: tuple[HistoricalCaseEvidence, ...] = ()
    india_cases: tuple[HistoricalCaseEvidence, ...] = ()
    unavailable_case_ids: tuple[NonEmptyString, ...] = ()


def build_backtest_evidence(
    report: BacktestReport | None,
    *,
    as_of: datetime,
    evidence: BacktestDisplayEvidence | None = None,
    unavailable_reason: str | None = None,
) -> BacktestEvidenceView:
    """Retain complete report comparisons; display only explicitly supplied evidence."""
    with localcontext(_decimal_context()):
        if report is None:
            if unavailable_reason is None or evidence is not None:
                raise ValueError("unavailable Backtest requires a reason and no display evidence")
            return BacktestEvidenceView(
                status="unavailable",
                unavailable_reason=unavailable_reason,
                as_of=as_of,
                report=None,
                evidence=None,
            )
        if unavailable_reason is not None:
            raise ValueError("available Backtest cannot have an unavailable reason")
        report = _copy(report)
        # Validate aware cutoff before comparing timestamps or returning an empty view.
        view = BacktestEvidenceView(
            status="ready", unavailable_reason=None, as_of=as_of, report=report, evidence=None
        )
        if report.evaluation_end > view.as_of:
            raise ValueError("Backtest evaluation extends beyond display cutoff")
        if evidence is None:
            return view.model_copy(
                update={
                    "metrics_unavailable_reason": "Detailed metric evidence not supplied",
                    "distribution_unavailable_reason": "Retained loss distribution not supplied",
                    "unavailable_case_ids": report.historical_case_ids,
                }
            )
        evidence = _copy(evidence)
        if (
            (evidence.report_id, evidence.configuration_version, evidence.schema_version)
            != (report.report_id, report.configuration_version, report.schema_version)
            or evidence.report_sha256 != content_hash(report)
            or evidence.available_at > view.as_of
        ):
            raise ValueError("Backtest display evidence report/version/hash/cutoff disagrees")
        if evidence.metrics is not None and evidence.metrics.scalar_projection != report.metrics:
            raise ValueError("retained metric projection disagrees with canonical report")
        if any(
            not report.evaluation_start <= c.event_at < report.evaluation_end
            for c in evidence.cases
        ):
            raise ValueError("historical case metadata is outside report evaluation period")
        if any(c.case_id not in report.historical_case_ids for c in evidence.cases):
            raise ValueError("historical case metadata is outside report membership")
        distribution = evidence.distribution
        sample_ids = (
            set() if distribution is None else {s.loss.event_id for s in distribution.samples}
        )
        if any(not set(c.analogue_event_ids) <= sample_ids for c in evidence.cases):
            raise ValueError("historical case analogue cross-link is unavailable")
        summary = None if distribution is None else distribution.summary
        return BacktestEvidenceView(
            status="ready",
            unavailable_reason=None,
            as_of=view.as_of,
            report=report,
            evidence=evidence,
            evidence_kind=evidence.evidence_kind,
            metrics_unavailable_reason=evidence.metrics_absence_reason,
            distribution_unavailable_reason=evidence.distribution_absence_reason,
            loss_samples=()
            if distribution is None
            else tuple(
                LossSampleDisplay(
                    event_id=s.loss.event_id,
                    evidence_kind=s.evidence_kind,
                    loss=_money(s.loss.loss, s.currency),
                )
                for s in distribution.samples
            ),
            expected_loss=None
            if summary is None or distribution is None
            else _money(summary.expected_loss, distribution.currency),
            tail_loss=None
            if summary is None or distribution is None
            else _money(summary.tail_loss, distribution.currency),
            cases=evidence.cases,
            india_cases=tuple(c for c in evidence.cases if "IN" in c.country_codes),
            unavailable_case_ids=tuple(
                i
                for i in report.historical_case_ids
                if i not in {c.case_id for c in evidence.cases}
            ),
        )


def _manifest(value: str, record_type: type[R]) -> R | None:
    # Opaque legacy/test version labels stay opaque. Structured production
    # manifests must carry their explicit schema identity and complete records.
    if not value.lstrip().startswith("{"):
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict) or "manifest_version" not in payload:
        raise ValueError("structured version manifest requires an explicit identity")
    return record_type.model_validate(payload)


def _source_identity_matches(identity: SourceItemIdentity, source: SourceItem) -> bool:
    identity_metadata = identity.model_dump(exclude={"published_at", "retrieved_at"})
    source_metadata = source.model_dump(
        exclude={"text", "published_at", "retrieved_at"}
    )
    return (
        identity_metadata == source_metadata
        and identity.published_at.astimezone(UTC) == source.published_at.astimezone(UTC)
        and identity.retrieved_at.astimezone(UTC) == source.retrieved_at.astimezone(UTC)
    )


def _validate_signal_manifests(
    signal: RiskSignal,
    source: SourceItem,
    sources_by_id: dict[str, SourceItem],
    as_of: datetime,
) -> None:
    snapshot = _manifest(signal.versions.snapshot_version, SnapshotManifest)
    if snapshot is not None:
        identities = snapshot.source_items
        member_ids = tuple(identity.source_item_id for identity in identities)
        ordered_identities = tuple(
            sorted(
                identities,
                key=lambda identity: (
                    identity.published_at.astimezone(UTC),
                    identity.source_item_id,
                ),
            )
        )
        if (
            len(set(member_ids)) != len(member_ids)
            or identities != ordered_identities
            or identities[0].source_item_id != signal.source_item_id
            or signal.source_item_id != source.source_item_id
        ):
            raise ValueError("Risk Signal source manifest disagrees with supplied Source Item")
        cutoff = as_of.astimezone(UTC)
        for identity in identities:
            member = sources_by_id.get(identity.source_item_id)
            if (
                member is None
                or not _source_identity_matches(identity, member)
                or identity.published_at.astimezone(UTC) > cutoff
                or identity.retrieved_at.astimezone(UTC) > cutoff
            ):
                raise ValueError("Risk Signal source manifest disagrees with supplied Source Item")
    calibration = _manifest(signal.versions.calibration_version, CalibrationManifest)
    model = _manifest(signal.versions.model_version, ModelManifest)
    if calibration is not None:
        confidence, fit = calibration.confidence, calibration.confidence_calibration
        if (
            confidence.probability != signal.confidence
            or confidence.target != signal.confidence_target
            or (calibration.impact.calibration_version, calibration.impact.reference_basket_version)
            != (signal.impact.calibration_version, signal.impact.reference_basket_version)
            or (
                confidence.target,
                confidence.calibration_version,
                confidence.evidence_kind,
                confidence.evidence_hash,
                confidence.fitted_artifact_hash,
                confidence.score_definition,
                confidence.transform_version,
            )
            != (
                fit.target,
                fit.calibration_version,
                fit.evidence_kind,
                fit.evidence_hash,
                fit.fitted_artifact_hash,
                fit.score_definition,
                fit.transform_version,
            )
            or not fit.development_start <= fit.development_end <= fit.frozen_at <= as_of
        ):
            raise ValueError(
                "Risk Signal Confidence/Impact calibration identity or chronology disagrees"
            )
        if model is not None and (
            fit.score_definition.event_model_version != model.models.event_model_version
            or fit.score_definition.entity_linker_version != model.models.entity_linker_version
        ):
            raise ValueError("Risk Signal model and fitted Confidence definitions disagree")
    if model is not None:
        materiality = model.portfolio_materiality
        if (
            None if materiality is None else materiality.materiality
        ) != signal.portfolio_materiality or (
            materiality is not None and materiality.audit.as_of != as_of
        ):
            raise ValueError("Risk Signal Portfolio Materiality provenance disagrees")


def _impact_evidence_hash(value: DomainModel) -> str:
    # Exact hash domain used by retained ImpactAudit LossEvidence; intentionally
    # different from Backtest's UTC-normalized canonical report hash domain.
    payload = json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
