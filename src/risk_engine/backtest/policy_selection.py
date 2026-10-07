"""Pure held-out scoring; retained evidence makes selection independently replayable."""

from __future__ import annotations

from bisect import bisect_left
from datetime import UTC, timedelta
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from statistics import median
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from risk_engine.backtest.module import (
    FittedManifest,
    ObservedOutcome,
    Record,
    TrainingPartition,
    _verify_calibration,
    content_hash,
    decode_impact_audit,
)
from risk_engine.backtest.runtime_inputs import PolicySelectionSpec, RuntimeRecord
from risk_engine.backtest.splits import FoldGroup, InnerFold
from risk_engine.domain import (
    AttributionDimension,
    ConfidenceTarget,
    NonEmptyString,
    PortfolioMateriality,
    ProvenanceMethod,
    RiskSignal,
    Sha256Hex,
    StressResult,
)
from risk_engine.impact.reference_basket import (
    ALLOCATION_ARITHMETIC_VERSION,
    reference_allocation_context,
)
from risk_engine.risk.policy import TriggerCriteria, evaluate_candidate_trigger

Count = Annotated[int, Field(strict=True, ge=0)]


def policy_folds(
    training: TrainingPartition, spec: PolicySelectionSpec, *, embargo: timedelta
) -> tuple[InnerFold, ...]:
    """Plan disjoint validation blocks entirely within supplied training membership."""
    training = TrainingPartition.model_validate(training.model_dump(mode="python"))
    return _policy_folds_from_groups(training.groups, spec, embargo=embargo)


def _policy_folds_from_groups(
    supplied_groups: tuple[FoldGroup, ...], spec: PolicySelectionSpec, *, embargo: timedelta
) -> tuple[InnerFold, ...]:
    """Revalidate the same plan at restore without reopening cases or labels."""
    spec = PolicySelectionSpec.model_validate(spec.model_dump(mode="python"))
    if not isinstance(embargo, timedelta) or embargo <= timedelta(0):
        raise ValueError("policy folds require a positive validated embargo")
    supplied_groups = tuple(
        FoldGroup.model_validate(g.model_dump(mode="python")) for g in supplied_groups
    )
    groups = tuple(
        sorted(
            (
                FoldGroup.model_validate(
                    g.model_dump() | {"event_time": g.event_time.astimezone(UTC)}
                )
                for g in supplied_groups
            ),
            key=lambda g: (g.event_time, g.cluster_id),
        )
    )
    # InnerFold validates duplicate cluster/source identities as well as ordering.
    if len({g.cluster_id for g in groups}) != len(groups) or len(
        {source for g in groups for source in g.source_item_ids}
    ) != sum(len(g.source_item_ids) for g in groups):
        raise ValueError("duplicate policy training cluster/source membership")
    folds: list[InnerFold] = []
    start = spec.train_groups
    last = len(groups) - spec.validation_groups
    while start <= last:
        eligible = sum(g.event_time + embargo <= groups[start].event_time for g in groups[:start])
        if eligible < spec.train_groups:
            start += 1
            continue
        folds.append(
            InnerFold(
                fold_id=f"policy-inner-{len(folds) + 1:04d}",
                train=groups[eligible - spec.train_groups : eligible],
                embargo=groups[eligible:start],
                validation=groups[start : start + spec.validation_groups],
            )
        )
        start += spec.validation_groups
    return tuple(folds)


class PolicyObservation(RuntimeRecord):
    fold_id: NonEmptyString
    case_id: NonEmptyString
    cluster_id: NonEmptyString
    as_of: AwareDatetime
    subfit_manifest: FittedManifest
    signal: RiskSignal
    stress: StressResult
    materiality: PortfolioMateriality
    material_event: bool = Field(strict=True)
    outcome_hash: Sha256Hex
    label_available_at: AwareDatetime
    outcome_available_at: AwareDatetime
    source_terms: tuple[NonEmptyString, ...] = Field(min_length=1)
    observed: ObservedOutcome
    market_hash: Sha256Hex
    portfolio_id: NonEmptyString
    portfolio_gross_values: tuple[tuple[NonEmptyString, Annotated[Decimal, Field(ge=0)]], ...]

    @model_validator(mode="after")
    def verified_evidence(self) -> PolicyObservation:
        observed, manifest, stress = self.observed, self.subfit_manifest, self.stress
        if observed.case_id != self.case_id or self.outcome_hash != content_hash(observed):
            raise ValueError("independent outcome case/hash mismatch")
        if observed.material_event is None or observed.material_event != self.material_event:
            raise ValueError("independent material-event label mismatch or absent")
        if (self.label_available_at, self.outcome_available_at) != (
            observed.label_available_at,
            observed.outcome_available_at,
        ):
            raise ValueError("independent outcome availability mismatch")
        if (
            manifest.fit_cutoff > self.as_of
            or min(
                self.label_available_at,
                self.outcome_available_at,
            )
            <= self.as_of
        ):
            raise ValueError("policy observation chronology requires replay before outcomes")
        if manifest.evidence_kind != "empirical":
            raise ValueError("production policy observation requires empirical-schema subfit")
        _verify_calibration(self.signal, manifest, self.as_of)
        impact_audit = decode_impact_audit(self.signal.impact.calibration_version)
        calibration = impact_audit.calibration
        if (
            manifest.impact.calibration_version != calibration.version
            or manifest.impact.frozen_at != calibration.calibrated_at
            or manifest.impact.currency != self.signal.impact.loss_currency
            or manifest.impact.horizon_days != calibration.window.end - calibration.window.start + 1
            or manifest.impact.quantile_convention != calibration.quantile_convention
        ):
            raise ValueError("Risk Signal complete frozen Impact identity mismatch")
        _verify_impact_output(self.signal)
        expected = PortfolioMateriality(
            absolute_loss=max(Decimal(0), stress.absolute_loss),
            percentage_loss=max(0, stress.percentage_loss),
            currency=stress.valuation_currency,
        )
        if self.materiality != expected or stress.portfolio_id != self.portfolio_id:
            raise ValueError("Portfolio Materiality/portfolio differs from actual StressResult")
        valuation = observed.valuation
        if valuation is None:
            raise ValueError("independent valuation evidence absent; retain an explicit exclusion")
        if valuation.baseline_market_hash != self.market_hash or (
            valuation.currency != stress.valuation_currency
            or valuation.horizon_days != manifest.impact.horizon_days
        ):
            raise ValueError("observed valuation baseline/currency/horizon mismatch")
        ids = tuple(k for k, _ in self.portfolio_gross_values)
        if (
            not ids
            or ids != tuple(sorted(set(ids)))
            or not any(v > 0 for _, v in self.portfolio_gross_values)
        ):
            raise ValueError("complete canonical positive gross-value support required")
        if valuation.gross_values != self.portfolio_gross_values or valuation.position_ids != ids:
            raise ValueError("observed valuation scope/gross values do not cover portfolio")
        asset_ids = tuple(
            sorted(a.label for a in stress.attribution if a.dimension == AttributionDimension.ASSET)
        )
        if stress.valuation_coverage != 1 or stress.unsupported_position_ids or asset_ids != ids:
            raise ValueError("policy selection requires complete supported valuation coverage")
        for dimensions in (
            {AttributionDimension.ASSET},
            {AttributionDimension.SECTOR},
            {AttributionDimension.REGION},
            {AttributionDimension.FACTOR},
            {AttributionDimension.OBLIGOR, AttributionDimension.FACILITY},
        ):
            if (
                _exact_sum(tuple(a.loss for a in stress.attribution if a.dimension in dimensions))
                != stress.absolute_loss
            ):
                raise ValueError("StressResult attribution differs from actual loss")
        _terms(self.source_terms)
        if not set(
            (*observed.source_terms, *manifest.source_terms, *manifest.confidence.source_terms)
        ) <= set(self.source_terms):
            raise ValueError("policy observation omits retained evidence source terms")
        return self


class PolicyExclusion(RuntimeRecord):
    fold_id: NonEmptyString
    case_id: NonEmptyString | None
    cluster_id: NonEmptyString | None
    reason: NonEmptyString
    source_terms: tuple[NonEmptyString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def explicit_scope(self) -> PolicyExclusion:
        if (self.case_id is None) != (self.cluster_id is None):
            raise ValueError("exclusion requires case and cluster or a whole fold")
        _terms(self.source_terms)
        return self


class PolicyCandidateResult(Record):
    criteria: TriggerCriteria
    true_positives: Count
    false_positives: Count
    false_negatives: Count
    true_negatives: Count
    alert_count: Count
    f1: Annotated[float, Field(strict=True, ge=0, le=1)] | None
    feasible: bool = Field(strict=True)
    reason: NonEmptyString | None

    @model_validator(mode="after")
    def complete_counts(self) -> PolicyCandidateResult:
        if self.alert_count != self.true_positives + self.false_positives:
            raise ValueError("alert count differs from confusion counts")
        if self.feasible == (self.reason is not None):
            raise ValueError("infeasible candidate needs exactly one reason")
        return self


class PolicySelectionAudit(RuntimeRecord):
    schema_version: Literal["production-policy-selection-audit-v1"]
    content_hash: Sha256Hex
    observations: tuple[PolicyObservation, ...]
    exclusions: tuple[PolicyExclusion, ...]
    spec: PolicySelectionSpec
    folds: tuple[InnerFold, ...]
    parent_training_hash: Sha256Hex
    runtime_definition_hash: Sha256Hex
    configuration_hash: Sha256Hex
    results: tuple[PolicyCandidateResult, ...]
    selected: TriggerCriteria | None
    absence_reason: NonEmptyString | None
    source_terms: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def replay_selection(self) -> PolicySelectionAudit:
        _validate_population(self)
        expected = _calculate(self.observations, self.spec)
        if (self.results, self.selected, self.absence_reason) != expected:
            raise ValueError("policy results/selection differ from recomputed held-out evidence")
        if self.source_terms != _source_terms(self.observations, self.exclusions):
            raise ValueError("policy audit source terms differ from retained evidence")
        if self.content_hash != content_hash(self.model_dump(exclude={"content_hash"})):
            raise ValueError("policy audit content hash mismatch")
        return self


def _exact_sum(values: tuple[Decimal, ...]) -> Decimal:
    """Sum attribution values without inheriting caller rounding or exponent bounds."""
    if not values:
        return Decimal(0)
    precision = (
        max(v.adjusted() for v in values)
        - min(int(str(v.as_tuple().exponent)) for v in values)
        + len(str(len(values)))
        + 2
    )
    with localcontext(
        Context(
            prec=precision,
            rounding=ROUND_HALF_EVEN,
            Emin=MIN_EMIN,
            Emax=MAX_EMAX,
            capitals=1,
            clamp=0,
            flags=[],
            traps=[InvalidOperation, DivisionByZero, Overflow],
        )
    ):
        return sum(values, Decimal(0))


def _verify_impact_output(signal: RiskSignal) -> None:
    """Replay only retained audited loss reduction and cutpoint ranking, never valuation."""
    impact = signal.impact
    audit = decode_impact_audit(impact.calibration_version)
    losses: tuple[Decimal, ...]
    if impact.method == ProvenanceMethod.HYPOTHETICAL:
        if audit.fallback is None or audit.hypothetical_loss is None:
            raise ValueError("Impact hypothetical output lacks audited fallback loss")
        losses = (audit.hypothetical_loss.loss,)
    else:
        if audit.fallback is not None or audit.hypothetical_loss is not None:
            raise ValueError("Impact historical output conflicts with audited fallback")
        losses = tuple(row.loss for row in audit.selected_losses)
    if not losses:
        raise ValueError("Impact output lacks audited selected losses")
    # Match ImpactEstimator's declared financial arithmetic for an even-sized median.
    with localcontext(reference_allocation_context(ALLOCATION_ARITHMETIC_VERSION)):
        expected_loss = Decimal(median(losses))
    if (
        impact.expected_reference_loss != expected_loss
        or impact.loss_lower_bound != min(losses)
        or impact.loss_upper_bound != max(losses)
        or impact.impact_score != 1 + bisect_left(audit.cutpoints, expected_loss)
        or impact.analogue_ids != tuple(row.event_id for row in audit.selected_losses)
        or impact.analogue_count != len(audit.selected_losses)
    ):
        raise ValueError("Risk Signal Impact output differs from retained audited losses/cutpoints")


def _terms(values: tuple[str, ...]) -> None:
    if values != tuple(sorted(set(values))):
        raise ValueError("source terms must be canonical and unique")


def _source_terms(
    observations: tuple[PolicyObservation, ...],
    exclusions: tuple[PolicyExclusion, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {t for o in observations for t in o.source_terms}
            | {t for e in exclusions for t in e.source_terms}
        )
    )


def _validate_population(audit: PolicySelectionAudit) -> None:
    folds = {fold.fold_id: fold for fold in audit.folds}
    if len(folds) != len(audit.folds):
        raise ValueError("duplicate fold identity")
    clusters: set[str] = set()
    sources: set[str] = set()
    for planned_fold in audit.folds:
        if len(planned_fold.train) != audit.spec.train_groups or (
            len(planned_fold.validation) != audit.spec.validation_groups
        ):
            raise ValueError("planned fold widths differ from policy specification")
        for group in planned_fold.validation:
            if group.cluster_id in clusters or sources.intersection(group.source_item_ids):
                raise ValueError("duplicate held-out cluster/source membership")
            clusters.add(group.cluster_id)
            sources.update(group.source_item_ids)
    covered: set[tuple[str, str]] = set()
    cases: set[str] = set()
    candidate_hashes: set[str] = set()
    dataset_hashes: set[str] = set()
    subfits: dict[str, str] = {}
    for row in audit.observations:
        fold = folds.get(row.fold_id)
        if fold is None:
            raise ValueError("observation outside planned fold")
        validation_group = next(
            (g for g in fold.validation if g.cluster_id == row.cluster_id), None
        )
        if (
            validation_group is None
            or row.signal.source_item_id not in validation_group.source_item_ids
        ):
            raise ValueError("chosen signal outside retained validation cluster")
        if row.as_of < validation_group.event_time:
            raise ValueError("observation precedes validation cluster chronology")
        key = row.fold_id, row.cluster_id
        if key in covered or row.case_id in cases:
            raise ValueError("duplicate held-out case/cluster membership")
        covered.add(key)
        cases.add(row.case_id)
        manifest = row.subfit_manifest
        if manifest.training_groups != fold.train or manifest.training_hash != content_hash(
            fold.train
        ):
            raise ValueError("subfit training identity differs from exact fold.train")
        if manifest.configuration_hash != audit.configuration_hash:
            raise ValueError("subfit configuration identity mismatch")
        if (
            manifest.candidate.parameters.get("runtime_definition_sha256")
            != audit.runtime_definition_hash
        ):
            raise ValueError("subfit runtime definition identity mismatch")
        candidate_hashes.add(content_hash(manifest.candidate))
        dataset_hashes.add(manifest.dataset_hash)
        previous = subfits.setdefault(row.fold_id, manifest.manifest_hash)
        if previous != manifest.manifest_hash:
            raise ValueError("fold observations retain different subfit identities")
    if len(candidate_hashes) > 1 or len(dataset_hashes) > 1:
        raise ValueError("subfit candidate/dataset identities differ across audit")
    for exclusion in audit.exclusions:
        fold = folds.get(exclusion.fold_id)
        if fold is None:
            raise ValueError("exclusion outside planned fold")
        ids = tuple(g.cluster_id for g in fold.validation)
        if exclusion.cluster_id is not None:
            if exclusion.cluster_id not in ids:
                raise ValueError("exclusion outside retained validation cluster")
            ids = (exclusion.cluster_id,)
            assert exclusion.case_id is not None
            if exclusion.case_id in cases:
                raise ValueError("exclusion contradicts retained case membership")
            cases.add(exclusion.case_id)
        for identity in ids:
            key = fold.fold_id, identity
            if key in covered:
                raise ValueError("duplicate or contradictory exclusion/observation scope")
            covered.add(key)
    planned = {(f.fold_id, g.cluster_id) for f in audit.folds for g in f.validation}
    if covered != planned:
        raise ValueError("policy audit omits planned validation cases")


def _calculate(
    observations: tuple[PolicyObservation, ...],
    spec: PolicySelectionSpec,
) -> tuple[tuple[PolicyCandidateResult, ...], TriggerCriteria | None, str | None]:
    """Score the declared grid on one unchanged eligible held-out population."""
    positive_count = sum(row.material_event for row in observations)
    negative_count = len(observations) - positive_count
    contributing_folds = len({row.fold_id for row in observations})
    population_reason: str | None = None
    if len(observations) < spec.minimum_cases:
        population_reason = "insufficient eligible cases"
    elif contributing_folds < spec.minimum_folds:
        population_reason = "insufficient eligible folds"
    elif positive_count < spec.minimum_positive_cases:
        population_reason = "insufficient positive cases"
    elif negative_count < spec.minimum_negative_cases:
        population_reason = "insufficient negative cases"

    scores: list[PolicyCandidateResult] = []
    for confidence in spec.confidence_thresholds:
        for economic_floor in spec.economic_floors:
            criteria = TriggerCriteria(
                confidence_threshold=confidence,
                economic_floor=economic_floor,
                materiality_tolerance=spec.materiality_tolerance,
                currency=spec.currency,
                confidence_target=ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS,
            )
            predictions = tuple(
                evaluate_candidate_trigger(row.signal, row.materiality, criteria).would_trigger
                for row in observations
            )
            true_positives = sum(
                alert and row.material_event
                for alert, row in zip(predictions, observations, strict=True)
            )
            false_positives = sum(
                alert and not row.material_event
                for alert, row in zip(predictions, observations, strict=True)
            )
            false_negatives = positive_count - true_positives
            true_negatives = negative_count - false_positives
            alert_count = true_positives + false_positives
            denominator = 2 * true_positives + false_positives + false_negatives
            f1 = None if denominator == 0 else 2 * true_positives / denominator
            reason = population_reason
            if reason is None:
                if denominator == 0:
                    reason = "undefined F1 denominator"
                elif true_positives == 0:
                    reason = "no true positives"
                elif alert_count > spec.alert_budget:
                    reason = "candidate exceeds declared alert budget"
            scores.append(
                PolicyCandidateResult(
                    criteria=criteria,
                    true_positives=true_positives,
                    false_positives=false_positives,
                    false_negatives=false_negatives,
                    true_negatives=true_negatives,
                    alert_count=alert_count,
                    f1=f1,
                    feasible=reason is None,
                    reason=reason,
                )
            )

    eligible = tuple(score for score in scores if score.feasible)
    if eligible:
        winner = min(
            eligible,
            key=lambda score: (
                -(score.f1 if score.f1 is not None else 0),
                score.alert_count,
                score.criteria.confidence_threshold.copy_negate(),
                score.criteria.economic_floor.copy_negate(),
            ),
        )
        return tuple(scores), winner.criteria, None
    if population_reason is not None:
        absence = population_reason
    elif all(score.alert_count > spec.alert_budget for score in scores):
        absence = "all candidates exceed declared alert budget"
    else:
        absence = (
            "no candidate has a defined F1 and at least one true positive "
            "within declared alert budget"
        )
    return tuple(scores), None, absence


def select_policy(
    observations: tuple[PolicyObservation, ...],
    exclusions: tuple[PolicyExclusion, ...],
    spec: PolicySelectionSpec,
    *,
    folds: tuple[InnerFold, ...],
    parent_training_hash: str,
    runtime_definition_hash: str,
    configuration_hash: str,
) -> PolicySelectionAudit:
    """Retain and verify all inputs, deterministic grid scores and selection evidence."""
    verified_observations = tuple(
        PolicyObservation.model_validate(row.model_dump(mode="python")) for row in observations
    )
    verified_exclusions = tuple(
        PolicyExclusion.model_validate(row.model_dump(mode="python")) for row in exclusions
    )
    verified_spec = PolicySelectionSpec.model_validate(spec.model_dump(mode="python"))
    verified_folds = tuple(
        InnerFold.model_validate(fold.model_dump(mode="python")) for fold in folds
    )
    results, selected, absence_reason = _calculate(verified_observations, verified_spec)
    evidence = {
        "schema_version": "production-policy-selection-audit-v1",
        "observations": verified_observations,
        "exclusions": verified_exclusions,
        "spec": verified_spec,
        "folds": verified_folds,
        "parent_training_hash": parent_training_hash,
        "runtime_definition_hash": runtime_definition_hash,
        "configuration_hash": configuration_hash,
        "results": results,
        "selected": selected,
        "absence_reason": absence_reason,
        "source_terms": _source_terms(verified_observations, verified_exclusions),
    }
    return PolicySelectionAudit.model_validate(
        {
            **evidence,
            "content_hash": content_hash(evidence),
        }
    )
