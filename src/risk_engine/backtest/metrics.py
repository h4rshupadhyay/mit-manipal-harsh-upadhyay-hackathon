"""Pure offline metrics under disclosed ``backtest-metrics-v1`` conventions.

Undefined ratios/ranks are None, never fabricated zeros. Classification evaluates
the union of actual and non-abstained predicted classes. Confidence measures
joint entity/Event Class correctness; binary probabilities are never clipped.
These project conventions are not claims of empirical optimality.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import timezone
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from risk_engine.backtest.splits import FoldGroup, InnerFold
from risk_engine.domain import (
    ConfidenceTarget,
    CurrencyCode,
    EventClass,
    FactorShock,
    ShockType,
    ShockUnit,
)

METRIC_VERSION = "backtest-metrics-v1"
_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_PositiveDays = Annotated[int, Field(strict=True, gt=0)]
_NonnegativeDecimal = Annotated[Decimal, Field(ge=0)]


class _Record(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)


class _MetricResult(_Record):
    convention_version: Literal["backtest-metrics-v1"]


def _aligned(*values: Sequence[object]) -> int:
    if not values or not values[0] or any(len(v) != len(values[0]) for v in values):
        raise ValueError("inputs must be nonempty and aligned")
    return len(values[0])


def _numbers(values: Sequence[float]) -> tuple[float, ...]:
    if any(isinstance(v, bool) or not isinstance(v, int | float) for v in values):
        raise ValueError("values must be finite numbers")
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise ValueError("values must be finite numbers")
    return result


def _unit(unit: str) -> str:
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("an explicit unit is required")
    return unit.strip()


def _mean(values: Sequence[float]) -> float:
    return math.fsum(v / len(values) for v in values)


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


class ClassMetrics(_Record):
    event_class: EventClass
    support: int
    predicted_count: int
    precision: float | None
    recall: float | None
    f1: float | None


class ClassificationMetrics(_MetricResult):
    evaluated_classes: tuple[EventClass, ...]
    macro_f1: float
    non_abstention_coverage: float
    per_class: tuple[ClassMetrics, ...]
    # Rows and first eight columns follow EventClass iteration; last column is abstention.
    confusion: tuple[tuple[int, ...], ...]


def classification_metrics(
    actual: Sequence[EventClass],
    predicted: Sequence[EventClass | None],
) -> ClassificationMetrics:
    count = _aligned(actual, predicted)
    if any(not isinstance(v, EventClass) for v in actual) or any(
        v is not None and not isinstance(v, EventClass) for v in predicted
    ):
        raise ValueError("labels must use the Event Class taxonomy")
    classes = tuple(EventClass)
    confusion = [[0] * (len(classes) + 1) for _ in classes]
    for truth, guess in zip(actual, predicted, strict=True):
        confusion[classes.index(truth)][classes.index(guess) if guess else len(classes)] += 1
    rows = []
    for index, event_class in enumerate(classes):
        support = sum(confusion[index])
        predicted_count = sum(row[index] for row in confusion)
        tp = confusion[index][index]
        rows.append(
            ClassMetrics(
                event_class=event_class,
                support=support,
                predicted_count=predicted_count,
                precision=_ratio(tp, predicted_count),
                recall=_ratio(tp, support),
                f1=_ratio(2 * tp, support + predicted_count),
            )
        )
    defined = [row.f1 for row in rows if row.f1 is not None]
    return ClassificationMetrics(
        convention_version=METRIC_VERSION,
        evaluated_classes=tuple(row.event_class for row in rows if row.f1 is not None),
        macro_f1=_mean(defined),
        non_abstention_coverage=sum(p is not None for p in predicted) / count,
        per_class=tuple(rows),
        confusion=tuple(tuple(row) for row in confusion),
    )


def entity_link_accuracy(actual: Sequence[str], predicted: Sequence[str | None]) -> float:
    count = _aligned(actual, predicted)
    if any(not isinstance(v, str) or not v.strip() for v in actual) or any(
        v is not None and (not isinstance(v, str) or not v.strip()) for v in predicted
    ):
        raise ValueError("entity identities must be nonempty strings or explicit abstentions")
    return sum(a == p for a, p in zip(actual, predicted, strict=True)) / count


class ReliabilityBin(_Record):
    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    accuracy: float | None


class ConfidenceMetrics(_MetricResult):
    target: ConfidenceTarget
    brier_score: float
    log_loss: float
    reliability: tuple[ReliabilityBin, ...]


def confidence_metrics(
    correct: Sequence[bool],
    probabilities: Sequence[float],
    bin_edges: Sequence[float],
) -> ConfidenceMetrics:
    """Binary Brier/NLL; bins [lower, upper), with final upper endpoint included."""
    _aligned(correct, probabilities)
    if any(type(v) is not bool for v in correct):
        raise ValueError("joint-correctness outcomes must be booleans")
    probabilities = _numbers(probabilities)
    edges = _numbers(bin_edges)
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("probabilities must lie in [0, 1]")
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != 1 or any(a >= b for a, b in pairwise(edges)):
        raise ValueError("bin edges must strictly increase from 0 to 1")
    losses = []
    for outcome, p in zip(correct, probabilities, strict=True):
        if (outcome and p == 0) or (not outcome and p == 1):
            raise ValueError("wrong certain probabilities have undefined finite log loss")
        losses.append(-math.log(p) if outcome else -math.log1p(-p))
    bins = []
    for index, (lower, upper) in enumerate(pairwise(edges)):
        members = [
            (p, y)
            for p, y in zip(probabilities, correct, strict=True)
            if lower <= p < upper or (index == len(edges) - 2 and p == upper)
        ]
        bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=len(members),
                mean_confidence=_mean([p for p, _ in members]) if members else None,
                accuracy=_mean([float(y) for _, y in members]) if members else None,
            )
        )
    return ConfidenceMetrics(
        convention_version=METRIC_VERSION,
        target=ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS,
        brier_score=_mean([(p - y) ** 2 for p, y in zip(probabilities, correct, strict=True)]),
        log_loss=_mean(losses),
        reliability=tuple(bins),
    )


def _ranks(values: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return tuple(ranks)


def _spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = _mean(ra), _mean(rb)
    da, db = [v - ma for v in ra], [v - mb for v in rb]
    denominator = math.sqrt(math.fsum(v * v for v in da) * math.fsum(v * v for v in db))
    return _ratio(math.fsum(x * y for x, y in zip(da, db, strict=True)), denominator)


class SeverityBucket(_Record):
    impact_score: int
    count: int
    mean_realized_loss: float


class SeverityMetrics(_MetricResult):
    loss_unit: _Text
    rank_correlation: float | None
    ordinal_mae: float
    bucket_monotonicity: float | None
    buckets: tuple[SeverityBucket, ...]


def severity_metrics(
    impact_scores: Sequence[int],
    realized_deciles: Sequence[int],
    realized_losses: Sequence[float],
    loss_unit: str,
) -> SeverityMetrics:
    """Spearman(loss, Impact Score); fraction of adjacent occupied means nondecreasing.

    Deciles are supplied frozen realized-loss labels, not fitted on these inputs.
    One occupied bucket or constant ranks produce an explicit undefined metric.
    """
    _aligned(impact_scores, realized_deciles, realized_losses)
    if any(type(v) is not int or not 1 <= v <= 10 for v in (*impact_scores, *realized_deciles)):
        raise ValueError("Impact Scores and realized deciles must be integers in 1..10")
    losses = _numbers(realized_losses)
    buckets = []
    for score in sorted(set(impact_scores)):
        members = [loss for s, loss in zip(impact_scores, losses, strict=True) if s == score]
        buckets.append(
            SeverityBucket(
                impact_score=score,
                count=len(members),
                mean_realized_loss=_mean(members),
            )
        )
    monotone = sum(a.mean_realized_loss <= b.mean_realized_loss for a, b in pairwise(buckets))
    return SeverityMetrics(
        convention_version=METRIC_VERSION,
        loss_unit=_unit(loss_unit),
        rank_correlation=_spearman(impact_scores, losses),
        ordinal_mae=_mean(
            [float(abs(a - b)) for a, b in zip(impact_scores, realized_deciles, strict=True)]
        ),
        bucket_monotonicity=_ratio(monotone, len(buckets) - 1),
        buckets=tuple(buckets),
    )


class IntervalMetrics(_MetricResult):
    unit: _Text
    nominal_coverage: Annotated[float, Field(gt=0, le=1)]
    coverage: float
    mean_width: float


def interval_metrics(
    lower: Sequence[float],
    upper: Sequence[float],
    realized: Sequence[float],
    unit: str,
    nominal_coverage: float,
) -> IntervalMetrics:
    """Inclusive empirical interval coverage and arithmetic mean width in supplied units."""
    count = _aligned(lower, upper, realized)
    lows, highs, outcomes = _numbers(lower), _numbers(upper), _numbers(realized)
    if any(a > b for a, b in zip(lows, highs, strict=True)):
        raise ValueError("interval lower bound must not exceed upper bound")
    level = _numbers([nominal_coverage])[0]
    return IntervalMetrics(
        convention_version=METRIC_VERSION,
        unit=_unit(unit),
        nominal_coverage=level,
        coverage=sum(a <= y <= b for a, b, y in zip(lows, highs, outcomes, strict=True)) / count,
        mean_width=_mean([b - a for a, b in zip(lows, highs, strict=True)]),
    )


class ScenarioObservation(_Record):
    case_id: _Text
    occurred_at: AwareDatetime
    snapshot_id: _Text
    configuration_version: _Text
    realized: Annotated[tuple[FactorShock, ...], Field(min_length=1)]
    joint_samples: Annotated[tuple[tuple[FactorShock, ...], ...], Field(min_length=1)]

    @model_validator(mode="after")
    def complete_joint_vectors(self) -> ScenarioObservation:
        ids = tuple(s.factor_id for s in self.realized)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("joint vectors must have unique, sorted factor identities")
        signature = tuple(
            (s.factor_id, s.shock_type, s.unit, s.horizon_days) for s in self.realized
        )
        if any(
            tuple((s.factor_id, s.shock_type, s.unit, s.horizon_days) for s in sample) != signature
            for sample in self.joint_samples
        ):
            raise ValueError(
                "joint vectors must match factor identities, units, types and horizons"
            )
        if len({s.horizon_days for s in self.realized}) != 1:
            raise ValueError("one Joint Shock Vector must have one horizon")
        return self


class FactorScenarioMetrics(_Record):
    factor_id: _Text
    shock_type: ShockType
    unit: ShockUnit
    horizon_days: _PositiveDays
    direction_accuracy: float
    crps: float


class ScenarioMetrics(_MetricResult):
    factors: tuple[FactorScenarioMetrics, ...]


def scenario_metrics(observations: Sequence[ScenarioObservation]) -> ScenarioMetrics:
    """Mean-sign accuracy and per-factor empirical CRPS in original units.

    CRPS = mean |X-y| - 0.5 mean |X-X'| over all ordered sample pairs
    (including diagonal). Complete joint samples remain intact; marginal metrics
    neither recombine factors nor claim to measure cross-factor dependence.
    """
    rows = tuple(ScenarioObservation.model_validate(o.model_dump()) for o in observations)
    _consistent_cases(rows)
    reference = rows[0].realized
    signature = tuple((s.factor_id, s.shock_type, s.unit, s.horizon_days) for s in reference)
    if any(
        tuple((s.factor_id, s.shock_type, s.unit, s.horizon_days) for s in r.realized) != signature
        for r in rows
    ):
        raise ValueError("all cases must have matched factor units and horizons")
    factors = []
    for index, shock in enumerate(reference):
        directions, errors = [], []
        for row in rows:
            samples = [sample[index].value for sample in row.joint_samples]
            actual = row.realized[index].value
            expected = _mean(samples)
            directions.append(float((expected > 0) - (expected < 0) == (actual > 0) - (actual < 0)))
            errors.append(
                _mean([abs(x - actual) for x in samples])
                - 0.5 * _mean([abs(x - y) for x in samples for y in samples])
            )
        factors.append(
            FactorScenarioMetrics(
                factor_id=shock.factor_id,
                shock_type=shock.shock_type,
                unit=shock.unit,
                horizon_days=shock.horizon_days,
                direction_accuracy=_mean(directions),
                crps=_mean(errors),
            )
        )
    return ScenarioMetrics(convention_version=METRIC_VERSION, factors=tuple(factors))


class ValuationObservation(_Record):
    """P&L pairs must refer to the same declared instrument scope and baseline.

    Gross values express valuation support, not a claim that unsupported value
    carries zero risk. Positive total gross value avoids signed-net cancellation.
    """

    case_id: _Text
    occurred_at: AwareDatetime
    comparison_scope: _Text
    snapshot_id: _Text
    configuration_version: _Text
    predicted_pnl: Decimal
    realized_pnl: Decimal
    currency: CurrencyCode
    horizon_days: _PositiveDays
    supported_gross_value: _NonnegativeDecimal
    total_gross_value: Annotated[Decimal, Field(gt=0)]

    @model_validator(mode="after")
    def support_is_bounded(self) -> ValuationObservation:
        if self.supported_gross_value > self.total_gross_value:
            raise ValueError("supported gross value cannot exceed total gross value")
        return self


class ValuationMetrics(_MetricResult):
    currency: CurrencyCode
    comparison_scope: _Text
    horizon_days: _PositiveDays
    pnl_mae: Decimal
    supported_value_share: Decimal


def _consistent_cases(rows: Sequence[ScenarioObservation | ValuationObservation]) -> None:
    if not rows:
        raise ValueError("observations must be nonempty")
    if len({r.case_id for r in rows}) != len(rows):
        raise ValueError("case identities must be unique")
    if len({(r.snapshot_id, r.configuration_version) for r in rows}) != 1:
        raise ValueError("observations must match snapshot and configuration versions")


def valuation_metrics(observations: Sequence[ValuationObservation]) -> ValuationMetrics:
    """Decimal P&L MAE; pooled supported gross value divided by pooled total value."""
    rows = tuple(ValuationObservation.model_validate(o.model_dump()) for o in observations)
    _consistent_cases(rows)
    if len({(r.currency, r.horizon_days) for r in rows}) != 1:
        raise ValueError("P&L currencies and horizons must match")
    if len({r.comparison_scope for r in rows}) != 1:
        raise ValueError("P&L comparison scopes must match")
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
        return ValuationMetrics(
            convention_version=METRIC_VERSION,
            currency=rows[0].currency,
            comparison_scope=rows[0].comparison_scope,
            horizon_days=rows[0].horizon_days,
            pnl_mae=sum((abs(r.predicted_pnl - r.realized_pnl) for r in rows), Decimal(0))
            / len(rows),
            supported_value_share=sum((r.supported_gross_value for r in rows), Decimal(0))
            / sum((r.total_gross_value for r in rows), Decimal(0)),
        )


class AlertObservation(_Record):
    case_id: _Text
    occurred_at: AwareDatetime
    is_alert: Annotated[bool, Field(strict=True)]
    material_event: Annotated[bool, Field(strict=True)]
    action_priority: Annotated[float, Field(strict=True, ge=0)]
    realized_loss: _NonnegativeDecimal


class AlertWindow(_Record):
    start: AwareDatetime
    end: AwareDatetime
    currency: CurrencyCode
    budget: Annotated[int, Field(strict=True, ge=0)]
    snapshot_id: _Text
    configuration_version: _Text

    @model_validator(mode="after")
    def positive_window(self) -> AlertWindow:
        utc = timezone.utc  # noqa: UP017 -- inherited verification runtime is Python 3.10
        if self.end.astimezone(utc) <= self.start.astimezone(utc):
            raise ValueError("alert window must have positive elapsed duration")
        return self


class AlertMetrics(_MetricResult):
    currency: CurrencyCode
    precision: float | None
    recall: float | None
    alerts_per_day: float
    loss_captured_at_budget: Decimal | None
    budget_case_ids: tuple[_Text, ...]


def alert_metrics(observations: Sequence[AlertObservation], window: AlertWindow) -> AlertMetrics:
    """Half-open UTC window; elapsed seconds/86400 includes silent days.

    Fixed budget takes the highest Action Priority proposed alerts, ties by ID.
    Captured loss is their nonnegative realized loss / all observed realized loss.
    Material-event truth remains a separate label, never inferred from priority.
    """
    window = AlertWindow.model_validate(window.model_dump())
    rows = tuple(AlertObservation.model_validate(o.model_dump()) for o in observations)
    if not rows or len({r.case_id for r in rows}) != len(rows):
        raise ValueError("alert cases must be nonempty and unique")
    utc = timezone.utc  # noqa: UP017 -- inherited verification runtime is Python 3.10
    start, end = window.start.astimezone(utc), window.end.astimezone(utc)
    if any(not start <= r.occurred_at.astimezone(utc) < end for r in rows):
        raise ValueError("alert observations must lie within the supplied half-open window")
    alerts = [r for r in rows if r.is_alert]
    true_positive = sum(r.material_event for r in alerts)
    budget_rows = sorted(alerts, key=lambda r: (-r.action_priority, r.case_id))[: window.budget]
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
        total_loss = sum((r.realized_loss for r in rows), Decimal(0))
        captured = (
            sum((r.realized_loss for r in budget_rows), Decimal(0)) / total_loss
            if total_loss
            else None
        )
    return AlertMetrics(
        convention_version=METRIC_VERSION,
        currency=window.currency,
        precision=_ratio(true_positive, len(alerts)),
        recall=_ratio(true_positive, sum(r.material_event for r in rows)),
        alerts_per_day=len(alerts) / ((end - start).total_seconds() / 86400),
        loss_captured_at_budget=captured,
        budget_case_ids=tuple(r.case_id for r in budget_rows),
    )


class InnerDevelopmentResult(_Record):
    fold: InnerFold
    score: Annotated[float, Field(strict=True)]

    @field_validator("fold", mode="before")
    @classmethod
    def normalize_partition_instants(cls, value: object) -> object:
        """Validate grouped chronology using UTC instants, including repeated DST hours."""
        payload = value.model_dump() if isinstance(value, InnerFold) else value
        if not isinstance(payload, dict):
            return value
        normalized = dict(payload)
        utc = timezone.utc  # noqa: UP017 -- inherited verification runtime is Python 3.10
        for partition in ("train", "embargo", "validation"):
            groups = payload.get(partition)
            if not isinstance(groups, tuple | list):
                raise ValueError("fold partitions must be explicit grouped sequences")
            normalized_groups = []
            for group in groups:
                validated = FoldGroup.model_validate(
                    group.model_dump() if isinstance(group, FoldGroup) else group
                )
                normalized_groups.append(
                    validated.model_copy(
                        update={
                            "event_time": validated.event_time.astimezone(utc),
                        }
                    )
                )
            normalized[partition] = tuple(normalized_groups)
        return normalized


class CandidateDevelopmentResult(_Record):
    """One declared configuration's matched inner DEVELOPMENT evidence.

    Chronology and grouped membership are retained in complete InnerFold records.
    The caller binds these to the indicated outer training partition/snapshot;
    this metric module cannot authenticate the caller's historical acquisition.
    Complexity dimensions are predeclared, ordered, nonnegative integer counts.
    """

    candidate_id: _Text
    configuration_version: _Text
    convention_version: Literal["one-standard-error-v1"]
    evaluation_scope: Literal["inner-development"]
    snapshot_id: _Text
    split_version: Literal["nested-grouped-chronological-v1"]
    outer_fold_id: _Text
    objective_name: _Text
    objective_unit: _Text
    direction: Literal["maximize", "minimize"]
    complexity_dimensions: Annotated[tuple[_Text, ...], Field(min_length=1)]
    complexity: Annotated[tuple[Annotated[int, Field(strict=True, ge=0)], ...], Field(min_length=1)]
    fold_results: Annotated[tuple[InnerDevelopmentResult, ...], Field(min_length=2)]

    @model_validator(mode="after")
    def complete_canonical_development_evidence(self) -> CandidateDevelopmentResult:
        if len(self.complexity) != len(self.complexity_dimensions) or len(
            set(self.complexity_dimensions)
        ) != len(self.complexity_dimensions):
            raise ValueError("complexity must match unique declared dimensions")
        folds = tuple(r.fold for r in self.fold_results)
        if len({f.fold_id for f in folds}) != len(folds):
            raise ValueError("development fold IDs must be unique")
        utc = timezone.utc  # noqa: UP017 -- inherited verification runtime is Python 3.10
        for previous, current in pairwise(folds):
            last = previous.validation[-1]
            first = current.validation[0]
            if (first.event_time.astimezone(utc), first.cluster_id) <= (
                last.event_time.astimezone(utc),
                last.cluster_id,
            ):
                raise ValueError(
                    "inner development validation blocks must be chronological and disjoint"
                )
        validation_groups = [g for f in folds for g in f.validation]
        ids = [g.cluster_id for g in validation_groups]
        sources = [s for g in validation_groups for s in g.source_item_ids]
        if len(set(ids)) != len(ids) or len(set(sources)) != len(sources):
            raise ValueError("inner validation group and Source Item identities must be disjoint")
        return self


class CandidateSummary(_Record):
    candidate_id: _Text
    mean_score: float
    standard_error: float


class SelectionResult(_Record):
    convention_version: Literal["one-standard-error-v1"]
    selected_candidate_id: _Text
    best_candidate_id: _Text
    best_mean: float
    best_standard_error: float
    eligibility_threshold: float
    eligible_candidate_ids: tuple[_Text, ...]
    summaries: tuple[CandidateSummary, ...]
    candidates: tuple[CandidateDevelopmentResult, ...]

    @model_validator(mode="after")
    def claims_match_evidence(self) -> SelectionResult:
        rows, summaries, best, threshold, eligible, selected = _selection_values(self.candidates)
        if self.candidates != rows:
            raise ValueError("selection evidence must use canonical candidate identity order")
        if (
            self.summaries != summaries
            or self.best_candidate_id != best.candidate_id
            or self.selected_candidate_id != selected
            or self.best_mean != best.mean_score
            or self.best_standard_error != best.standard_error
            or self.eligibility_threshold != threshold
            or self.eligible_candidate_ids != eligible
        ):
            raise ValueError("selection claims contradict inner development evidence")
        return self


def select_within_one_standard_error(
    candidates: Sequence[CandidateDevelopmentResult],
) -> SelectionResult:
    """Choose lexicographically simplest candidate within the best mean's SE.

    v1 uses unweighted matched-fold means and sample SD(ddof=1)/sqrt(n), n>=2.
    Eligibility is inclusive best_mean +/- best_SE (according to direction).
    Best-mean ties and equal-complexity selection ties use ascending candidate ID.
    The chosen candidate's uncertainty does not enlarge the eligibility band.
    All input evidence and candidate summaries are retained in canonical ID order.
    """
    rows, summaries, best, threshold, eligible_ids, selected = _selection_values(candidates)
    return SelectionResult(
        convention_version="one-standard-error-v1",
        selected_candidate_id=selected,
        best_candidate_id=best.candidate_id,
        best_mean=best.mean_score,
        best_standard_error=best.standard_error,
        eligibility_threshold=threshold,
        eligible_candidate_ids=eligible_ids,
        summaries=summaries,
        candidates=rows,
    )


def _selection_values(
    candidates: Sequence[CandidateDevelopmentResult],
) -> tuple[
    tuple[CandidateDevelopmentResult, ...],
    tuple[CandidateSummary, ...],
    CandidateSummary,
    float,
    tuple[str, ...],
    str,
]:
    rows = tuple(
        sorted(
            (CandidateDevelopmentResult.model_validate(c.model_dump()) for c in candidates),
            key=lambda c: c.candidate_id,
        )
    )
    if not rows or len({r.candidate_id for r in rows}) != len(rows):
        raise ValueError("candidates must be nonempty with unique identities")
    reference = rows[0]

    def signature(row: CandidateDevelopmentResult) -> tuple[object, ...]:
        return (
            row.snapshot_id,
            row.split_version,
            row.outer_fold_id,
            row.objective_name,
            row.objective_unit,
            row.direction,
            row.complexity_dimensions,
            tuple(r.fold for r in row.fold_results),
        )

    if any(signature(r) != signature(reference) for r in rows):
        raise ValueError(
            "candidates must match complete folds, objective, versions and complexity dimensions"
        )
    summaries = []
    for row in rows:
        scores = [r.score for r in row.fold_results]
        mean = _mean(scores)
        n = len(scores)
        deviations = _numbers([v - mean for v in scores])
        scale = max(abs(v) for v in deviations)
        standard_error = (
            math.hypot(*(v / scale for v in deviations)) / math.sqrt(n * (n - 1)) * scale
            if scale
            else 0
        )
        summaries.append(
            CandidateSummary(
                candidate_id=row.candidate_id,
                mean_score=mean,
                standard_error=standard_error,
            )
        )
    maximize = reference.direction == "maximize"
    best = min(
        summaries, key=lambda s: (-s.mean_score if maximize else s.mean_score, s.candidate_id)
    )
    threshold = best.mean_score + (-best.standard_error if maximize else best.standard_error)
    eligible_ids = tuple(
        s.candidate_id
        for s in summaries
        if (s.mean_score >= threshold if maximize else s.mean_score <= threshold)
    )
    selected = min(
        (r for r in rows if r.candidate_id in eligible_ids),
        key=lambda r: (r.complexity, r.candidate_id),
    )
    return rows, tuple(summaries), best, threshold, eligible_ids, selected.candidate_id
