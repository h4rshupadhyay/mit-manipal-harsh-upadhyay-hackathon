"""Offline production fitting and reconstruction of verified frozen runtimes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from statistics import median

from risk_engine.backtest.module import (
    CandidateSpec,
    DevelopmentConfiguration,
    FittedManifest,
    ImpactFitIdentity,
    ReplayInput,
    ScenarioEvidence,
    TrainingPartition,
    _verify_calibration,
    canonical_bytes,
    content_hash,
    decode_impact_audit,
)
from risk_engine.backtest.policy_selection import PolicySelectionAudit
from risk_engine.backtest.runtime_inputs import (
    LocalModelLocations,
    RuntimeDefinition,
    RuntimeEvidenceIndex,
    TrainingCaseEvidence,
    _verified_bytes,
    load_training_evidence,
    resolve_candidate,
)
from risk_engine.backtest.runtime_state import (
    FrozenRuntimeState,
    load_runtime_state,
    write_runtime_state,
)
from risk_engine.domain import (
    FactorShock,
    InterpretedEvent,
    ProvenanceMethod,
    RiskSignal,
    StressScenario,
)
from risk_engine.impact.analogues import AnalogueRepository
from risk_engine.impact.module import (
    BasketManifest,
    FrozenImpactCalibration,
    ImpactEstimator,
    TrainingEvidence,
)
from risk_engine.impact.reference_basket import (
    ALLOCATION_ARITHMETIC_VERSION,
    ReferenceBasketBuilder,
    reference_allocation_context,
)
from risk_engine.nlp.entity_matcher import EntityMatcher
from risk_engine.nlp.interfaces import ModelLock
from risk_engine.nlp.interpret import _clauses, interpret
from risk_engine.nlp.local_models import LocalBackend, LocalModels, _verify_snapshot
from risk_engine.risk.confidence import (
    CalibrationEvidence,
    CalibrationRow,
    ConfidenceCalibrator,
    ConfidenceScoreDefinition,
    RawConfidenceScore,
)
from risk_engine.risk.module import (
    ActionPriorityCandidate,
    ModelIdentity,
    ModelManifest,
    RiskEngine,
    SnapshotManifest,
    _signal_id,
    _source_identity,
    confidence_calibration_identity,
)
from risk_engine.stress.module import StressEngine


class InsufficientCalibrationEvidence(ValueError):
    """Only missing joint correctness classes or unidentified neutral raw scores."""


def _confidence_rows(
    training: TrainingPartition,
    evidence: tuple[TrainingCaseEvidence, ...],
    events: dict[str, tuple[InterpretedEvent, ...]],
    definition: ConfidenceScoreDefinition,
) -> tuple[CalibrationRow, ...]:
    rows: list[CalibrationRow] = []
    expected_sources = {item.source_item_id for c in training.cases for item in c.cluster.items}
    if set(events) != expected_sources:
        raise ValueError("interpreted Source Item membership mismatch")
    for case, sidecar in zip(training.cases, evidence, strict=True):
        targets = {(t.source_item_id, t.event_id): t for t in sidecar.targets}
        actual_keys = {
            (item.source_item_id, e.event_id)
            for item in case.cluster.items
            for e in events[item.source_item_id]
        }
        if actual_keys != set(targets):
            raise ValueError("independent target coverage differs from interpreted clauses")
        for item in case.cluster.items:
            seen_events: set[str] = set()
            for event in events[item.source_item_id]:
                event = InterpretedEvent.model_validate(event.model_dump(mode="python"))
                if (
                    event.event_id in seen_events
                    or event.source_item_id != item.source_item_id
                    or event.rationale not in item.text
                    or any(e not in item.text for e in event.evidence)
                ):
                    raise ValueError("interpreted clause identity/source/evidence mismatch")
                seen_events.add(event.event_id)
                target = targets[(item.source_item_id, event.event_id)]
                if (
                    target.content_hash != item.content_hash
                    or target.interpretation_version != "interpret-v1"
                ):
                    raise ValueError("independent target source hash/version mismatch")
                entities: dict[str, float] = {}
                for entity in event.entity_links:
                    if (
                        entity.entity_id in entities
                        and entities[entity.entity_id] != entity.confidence
                    ):
                        raise ValueError("conflicting repeated entity raw scores")
                    entities[entity.entity_id] = entity.confidence
                if not entities:
                    entities[f"unknown:{event.event_id}"] = 0.0
                for entity_id, entity_score in entities.items():
                    rows.append(
                        CalibrationRow(
                            row_id=content_hash((item.source_item_id, event.event_id, entity_id)),
                            source_item_id=item.source_item_id,
                            event_id=event.event_id,
                            entity_id=entity_id,
                            event_class=event.event_class,
                            content_hash=item.content_hash,
                            source_terms="; ".join(
                                sorted(
                                    set((item.license, *target.source_terms, *sidecar.source_terms))
                                )
                            ),
                            label_provenance=(
                                f"{target.label_provenance}; label_sha256={target.label_hash}"
                            ),
                            source_available_at=max(item.published_at, item.retrieved_at),
                            label_available_at=target.label_available_at,
                            split="development",
                            raw_score=RawConfidenceScore(
                                definition=definition,
                                entity_link_confidence=entity_score,
                                classification_confidence=event.classification_confidence,
                            ),
                            entity_correct=(
                                entity_id in target.actual_entity_ids
                                and not entity_id.casefold().startswith(("unknown:", "ambiguous:"))
                            ),
                            event_class_correct=event.event_class == target.actual_event_class,
                        )
                    )
    return tuple(sorted(rows, key=lambda row: row.row_id))


def _impact_identity(estimator: ImpactEstimator) -> ImpactFitIdentity:
    return ImpactFitIdentity(
        evidence_kind="empirical", **estimator.calibration_summary.model_dump()
    )


def _model_identity(lock: ModelLock, catalogue_hash: str) -> ModelIdentity:
    return ModelIdentity(
        event_model_version=f"{lock.event.model_id}@{lock.event.revision}",
        sentiment_model_version=f"{lock.sentiment.model_id}@{lock.sentiment.revision}",
        entity_linker_version=f"entity-matcher-v1:{catalogue_hash}",
    )


class ProductionFittedRuntime:
    """Borrow the fitter's sequential inference owner; retain immutable fit identity."""

    def __init__(
        self,
        *,
        manifest: FittedManifest,
        state: FrozenRuntimeState,
        models: LocalModels,
        matcher: EntityMatcher,
        schema_version: str,
    ) -> None:
        self.manifest = manifest
        self._schema_version = schema_version
        self._state = state
        self._repository = AnalogueRepository(
            tuple(row.analogue for row in state.impact_calibration.training),
            state.matching_config,
            (),
        )
        self._impact = ImpactEstimator(
            self._repository, state.impact_calibration, state.base_market
        )
        self.risk_engine = RiskEngine(
            matcher=matcher,
            sentiment_model=models.sentiment,
            event_model=models.event,
            impact_estimator=self._impact,
            confidence_calibrator=state.confidence_calibrator,
            model_identity=state.model_identity,
            schema_version=schema_version,
            materiality_provider=None,
            action_priority=ActionPriorityCandidate(),
        )

    def scenarios(
        self, signal: RiskSignal, case: ReplayInput, *, as_of: datetime
    ) -> ScenarioEvidence:
        signal = RiskSignal.model_validate(signal.model_dump(mode="python"))
        case = ReplayInput.model_validate(case.model_dump(mode="python"))
        if signal.versions.schema_version != self._schema_version:
            raise ValueError("scenario Risk Signal schema version mismatch")
        if (
            as_of.utcoffset() is None
            or as_of != case.as_of
            or as_of <= self._state.impact_calibration.calibrated_at
        ):
            raise ValueError("scenario as_of/case/frozen chronology mismatch")
        for source in case.cluster.items:
            if source.published_at > as_of or source.retrieved_at > as_of:
                raise ValueError(
                    f"scenario Source Item {source.source_item_id} unavailable by as_of"
                )
            if hashlib.sha256(source.text.encode("utf-8")).hexdigest() != source.content_hash:
                raise ValueError(
                    f"scenario Source Item {source.source_item_id} content hash mismatch"
                )
        items = [
            item for item in case.cluster.items if item.source_item_id == signal.source_item_id
        ]
        if len(items) != 1:
            raise ValueError("scenario Source Item/case mismatch")
        item = items[0]
        if SnapshotManifest.model_validate_json(
            signal.versions.snapshot_version
        ) != SnapshotManifest(source_items=(_source_identity(item),)):
            raise ValueError("scenario Source Item snapshot identity mismatch")
        if ModelManifest.model_validate_json(signal.versions.model_version) != ModelManifest(
            models=self._state.model_identity,
            action_priority=ActionPriorityCandidate(),
            portfolio_materiality=None,
        ):
            raise ValueError("scenario ModelManifest identity mismatch")
        _verify_calibration(signal, self.manifest, as_of)
        candidates = []
        for start, end, clause in _clauses(item.text):
            if clause != signal.rationale or signal.evidence != (clause,):
                continue
            identity = json.dumps(
                ["interpret-v1", item.source_item_id, item.content_hash, start, end],
                separators=(",", ":"),
            )
            event = InterpretedEvent(
                event_id="event:" + hashlib.sha256(identity.encode()).hexdigest(),
                source_item_id=item.source_item_id,
                entity_links=(signal.entity,),
                event_class=signal.event_class,
                sentiment=signal.sentiment,
                classification_confidence=0.0,
                rationale=clause,
                evidence=(clause,),
                eligible_for_automatic_stress=False,
            )
            if _signal_id(item, event, signal.entity) == signal.signal_id:
                candidates.append(event)
        if len(candidates) != 1:
            raise ValueError("scenario Risk Signal clause identity mismatch")
        event = candidates[0]
        cohort = self._repository.match(event, as_of)
        if cohort.method is not ProvenanceMethod.EMPIRICAL or not cohort.analogues:
            raise ValueError("scenario lacks complete observed support")
        # Recompute through the public estimator, authenticating the entire audited output.
        if self._impact.estimate(event, as_of) != signal.impact:
            raise ValueError("scenario audited Impact/cohort/loss evidence mismatch")
        audit = decode_impact_audit(signal.impact.calibration_version)
        calibration = self._state.impact_calibration
        bindings = {
            binding.historical_factor_id: binding for binding in calibration.factor_bindings
        }
        selected = {loss.event_id: loss for loss in audit.selected_losses}
        samples: dict[str, tuple[FactorShock, ...]] = {}
        with localcontext(reference_allocation_context(ALLOCATION_ARITHMETIC_VERSION)):
            for analogue in cohort.analogues:
                if analogue.scenario is None:
                    raise ValueError("scenario lacks complete observed vector")
                windows = [w for w in analogue.scenario.windows if w.window == calibration.window]
                if len(windows) != 1 or {s.factor_id for s in windows[0].shocks} != set(bindings):
                    raise ValueError("scenario lacks registered complete window support")
                shocks = []
                for historical in windows[0].shocks:
                    binding = bindings[historical.factor_id]
                    if (
                        historical.unit != binding.source_unit
                        or historical.measurement_dimension != binding.measurement_dimension
                    ):
                        raise ValueError("scenario conversion unit/dimension mismatch")
                    shocks.append(
                        FactorShock(
                            factor_id=binding.valuation_factor_id,
                            shock_type=binding.shock_type,
                            unit=binding.valuation_unit,
                            value=float(Decimal(str(historical.value)) * binding.value_multiplier),
                            horizon_days=calibration.window.end - calibration.window.start + 1,
                        )
                    )
                reference = StressScenario(
                    scenario_id=f"reference-{analogue.event_id}",
                    risk_signal_id=event.event_id,
                    shocks=tuple(shocks),
                    method=ProvenanceMethod.EMPIRICAL,
                    reference_event_ids=(analogue.event_id,),
                    calibration_version=calibration.version,
                )
                valuation = StressEngine().run(
                    calibration.reference_basket, self._state.base_market, reference
                )
                loss = selected.get(analogue.event_id)
                if (
                    loss is None
                    or valuation.valuation_coverage != 1
                    or valuation.unsupported_position_ids
                    or valuation.absolute_loss != loss.loss
                    or valuation.stress_result_id != loss.stress_result_id
                    or valuation.versions.valuation_rule_version != loss.valuation_rule_version
                ):
                    raise ValueError(
                        "scenario Reference Basket revaluation disagrees with loss evidence"
                    )
                samples[analogue.event_id] = tuple(
                    sorted(shocks, key=lambda shock: shock.factor_id)
                )
            cohort_median = Decimal(median(tuple(row.loss for row in audit.selected_losses)))
            chosen = min(
                selected,
                key=lambda event_id: (abs(selected[event_id].loss - cohort_median), event_id),
            )
        support = dict(
            schema_version="production-scenario-support-v1",
            fit_hash=self.manifest.manifest_hash,
            calibration_hash=self.manifest.impact.calibration_hash,
            cohort=cohort.model_dump(mode="json"),
            source_hashes=sorted(i.content_hash for i in case.cluster.items),
            source_items=tuple(
                _source_identity(i).model_dump(mode="json") for i in case.cluster.items
            ),
            window=calibration.window.model_dump(mode="json"),
            factor_bindings=tuple(b.model_dump(mode="json") for b in calibration.factor_bindings),
            reference_losses=tuple(row.model_dump(mode="json") for row in audit.selected_losses),
            chosen_event_id=chosen,
            scenario_choice=self._state.scenario_choice,
        )
        scenario = StressScenario(
            scenario_id="observed:" + content_hash(support),
            risk_signal_id=signal.signal_id,
            shocks=samples[chosen],
            method=ProvenanceMethod.EMPIRICAL,
            reference_event_ids=(chosen,),
            calibration_version=signal.impact.calibration_version,
        )
        return ScenarioEvidence(
            scenario=scenario,
            joint_samples=tuple(samples[row.event_id] for row in cohort.analogues),
            fit_hash=self.manifest.manifest_hash,
            calibration_hash=self.manifest.impact.calibration_hash,
            available_at=max(
                calibration.calibrated_at,
                self._state.maximum_evidence_available_at,
                *(row.available_at for row in cohort.analogues),
            ),
            source_terms=self._state.source_terms,
            support_reference=canonical_bytes(support).decode(),
            support_hash=content_hash(support),
        )


class ProductionCandidateFitter:
    def __init__(
        self,
        *,
        configuration: DevelopmentConfiguration,
        definition: RuntimeDefinition,
        evidence_index: RuntimeEvidenceIndex | None,
        model_locations: LocalModelLocations,
        artifact_root: Path,
        backend: LocalBackend | None = None,
    ) -> None:
        self.configuration = DevelopmentConfiguration.model_validate(
            configuration.model_dump(mode="python")
        )
        self.definition = RuntimeDefinition.model_validate(definition.model_dump(mode="python"))
        self.evidence_index = evidence_index
        self.model_locations = LocalModelLocations.model_validate(
            model_locations.model_dump(mode="python")
        )
        self.artifact_root = artifact_root
        self._backend = backend
        self._models: LocalModels | None = None
        self._configuration_hash = content_hash(self.configuration)
        self._definition_hash = self.definition.content_hash

    def close(self) -> None:
        if self._models is not None:
            self._models.close()
            self._models = None

    def _prerequisites(self) -> tuple[ModelIdentity, EntityMatcher, LocalModels]:
        configuration = DevelopmentConfiguration.model_validate(
            self.configuration.model_dump(mode="python")
        )
        definition = RuntimeDefinition.model_validate(self.definition.model_dump(mode="python"))
        if (
            content_hash(configuration) != self._configuration_hash
            or definition.content_hash != self._definition_hash
        ):
            raise ValueError("fitter configuration/definition changed")
        lock = ModelLock.model_validate_json(
            _verified_bytes(definition.model_lock, definition.frozen_at)
        )
        _verified_bytes(definition.catalogue, definition.frozen_at)
        # Constructor verifies snapshots even on restore/preflight; backend load is inference-only.
        if self._models is None:
            self._models = LocalModels(
                lock,
                self.model_locations.sentiment_snapshot,
                self.model_locations.event_snapshot,
                backend=self._backend,
                device=self.model_locations.device,
            )
        else:
            _verify_snapshot(lock.sentiment, self.model_locations.sentiment_snapshot)
            _verify_snapshot(lock.event, self.model_locations.event_snapshot)
        return (
            _model_identity(lock, definition.catalogue.sha256),
            EntityMatcher(Path(definition.catalogue.path)),
            self._models,
        )

    def _restore_components(
        self, locked: FittedManifest
    ) -> tuple[FittedManifest, FrozenRuntimeState, EntityMatcher, LocalModels]:
        locked = FittedManifest.model_validate(locked.model_dump(mode="python"))
        identity, matcher, models = self._prerequisites()
        if (
            locked.configuration_hash != self._configuration_hash
            or locked.candidate not in self.configuration.candidates
            or locked.evidence_kind != "empirical"
            or locked.model_identity != identity
            or locked.model_lock != self.definition.model_lock
        ):
            raise ValueError("locked manifest configuration/candidate/model identity mismatch")
        state_artifacts = [
            a for a in locked.artifacts if a.identity == "production-runtime-state-v1"
        ]
        if (
            len(state_artifacts) != 1
            or len(locked.artifacts) != 2
            or self.definition.catalogue not in locked.artifacts
        ):
            raise ValueError("locked manifest requires exactly one runtime state and catalogue")
        locked.verify_artifacts()
        state = load_runtime_state(state_artifacts[0], cutoff=locked.fit_cutoff)
        if (
            state.candidate != locked.candidate
            or state.configuration_hash != locked.configuration_hash
            or state.runtime_definition_hash != self.definition.content_hash
            or state.dataset_hash != locked.dataset_hash
            or state.training_groups != locked.training_groups
            or state.maximum_evidence_available_at != locked.maximum_evidence_available_at
            or state.fit_cutoff != locked.fit_cutoff
            or state.model_identity != identity
            or state.model_lock != locked.model_lock
            or state.catalogue != self.definition.catalogue
            or state.source_terms != locked.source_terms
            or confidence_calibration_identity(state.confidence_calibrator) != locked.confidence
            or state.policy_absence_reason != locked.policy_absence_reason
        ):
            raise ValueError("locked manifest/frozen state cross-identity mismatch")
        resolved = resolve_candidate(locked.candidate, self.definition)
        calibration = state.impact_calibration
        basket = resolved.basket
        basket_manifest = BasketManifest.model_validate_json(calibration.reference_basket.version)
        spec_hash = hashlib.sha256(
            json.dumps(
                basket.spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        returns_hash = hashlib.sha256(
            json.dumps(
                [r.model_dump(mode="json") for r in basket.returns],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if (
            state.matching_config != resolved.matching.config
            or state.base_market != basket.base_market
            or calibration.window != resolved.window.window
            or calibration.factor_bindings != basket.factor_bindings
            or state.scenario_choice != resolved.scenario_choice
            or basket_manifest.construction_spec_sha256 != spec_hash
            or basket_manifest.return_rows_sha256 != returns_hash
            or calibration.reference_basket.portfolio_id != basket.spec.basket_id
        ):
            raise ValueError("locked runtime registered conventions/basket identity mismatch")
        repository = AnalogueRepository(
            tuple(r.analogue for r in calibration.training), state.matching_config, ()
        )
        if (
            _impact_identity(ImpactEstimator(repository, calibration, state.base_market))
            != locked.impact
        ):
            raise ValueError("locked Impact calibration/cutpoints differ from recomputed valuation")
        if (
            locked.policy is None
            and state.policy_audit is not None
            and state.policy_audit.selected is not None
        ):
            raise ValueError("locked policy absent despite selected frozen audit")
        return locked, state, matcher, models

    def preflight_restore(self, locked: FittedManifest) -> None:
        self._restore_components(locked)

    def restore(self, locked: FittedManifest) -> ProductionFittedRuntime:
        manifest, state, matcher, models = self._restore_components(locked)
        return ProductionFittedRuntime(
            manifest=manifest,
            state=state,
            models=models,
            matcher=matcher,
            schema_version=self.definition.signal_schema_version,
        )

    def fit(
        self, candidate: CandidateSpec, training: TrainingPartition, *, as_of: datetime
    ) -> ProductionFittedRuntime:
        resolved = resolve_candidate(candidate, self.definition)
        if resolved.policy_mode != "unselected":
            raise ValueError("held-out policy fitting requires Task 4 selection integration")
        return self._fit_core(candidate, training, as_of=as_of)

    def _fit_core(
        self,
        candidate: CandidateSpec,
        training: TrainingPartition,
        *,
        as_of: datetime,
        policy_audit: PolicySelectionAudit | None = None,
    ) -> ProductionFittedRuntime:
        training = TrainingPartition.model_validate(training.model_dump(mode="python"))
        candidate = CandidateSpec.model_validate(candidate.model_dump(mode="python"))
        resolved = resolve_candidate(candidate, self.definition)
        if candidate not in self.configuration.candidates:
            raise ValueError("candidate not declared in configuration")
        if training.evidence_kind != "empirical" or self.evidence_index is None:
            raise ValueError("production fitting requires empirical training evidence/index")
        if as_of.utcoffset() is None or as_of != training.cutoff:
            raise ValueError("fit as_of must equal partition cutoff")
        latest = max(training.availability)
        training_end = latest + timedelta(microseconds=1)
        frozen_at = latest + timedelta(microseconds=2)
        if frozen_at >= training.cutoff:
            raise ValueError("fit cutoff too tight for calibration chronology")
        if max(self.definition.frozen_at, self.configuration.frozen_at) > latest:
            raise ValueError("definition/configuration exceed consumed training availability")
        evidence = load_training_evidence(self.evidence_index, training)
        if any(
            group.event_time != row.analogue.event_at
            for group, row in zip(training.groups, evidence, strict=True)
        ):
            raise ValueError("observed analogue event time differs from training group")
        references = {row.case_id: row for row in self.evidence_index.cases}
        descriptors = tuple(references[case.case_id].artifact for case in training.cases)
        if any(descriptor.available_at > latest for descriptor in descriptors):
            raise ValueError("training descriptor evidence exceeds actual partition availability")
        basket = resolved.basket
        first_event = min(row.analogue.event_at for row in evidence)
        if basket.available_at > first_event:
            raise ValueError("Reference Basket resources unavailable before first training event")
        identity, matcher, models = self._prerequisites()
        # Build/validate deterministic valuation inputs before invoking any inference port.
        impact = FrozenImpactCalibration(
            version="production-impact-v1:" + content_hash((candidate, training.membership_hash)),
            calibrated_at=frozen_at,
            training_start=first_event,
            training_end=training_end,
            reference_basket=ReferenceBasketBuilder().build(basket.returns, basket.spec),
            window=resolved.window.window,
            factor_bindings=basket.factor_bindings,
            quantile_convention=resolved.cutpoint_rule,
            uncertainty_convention="observed-min-max",
            source_terms="; ".join(sorted(set((*training.source_terms, *basket.source_terms)))),
            training=tuple(
                TrainingEvidence(
                    analogue=row.analogue, base_market=basket.base_market, partition="training"
                )
                for row in evidence
            ),
        )
        estimator = ImpactEstimator(
            AnalogueRepository(
                tuple(row.analogue for row in evidence), resolved.matching.config, ()
            ),
            impact,
            basket.base_market,
        )
        events = {
            item.source_item_id: tuple(interpret(item, matcher, models.sentiment, models.event))
            for case in training.cases
            for item in case.cluster.items
        }
        score_definition = ConfidenceScoreDefinition(
            event_model_version=identity.event_model_version,
            entity_linker_version=identity.entity_linker_version,
        )
        rows = _confidence_rows(training, evidence, events, score_definition)
        # Validate row chronology before the two specific insufficiency checks.
        source_start = min(
            max(item.published_at, item.retrieved_at)
            for case in training.cases
            for item in case.cluster.items
        )
        if any(
            not source_start <= row.source_available_at <= row.label_available_at <= frozen_at
            for row in rows
        ):
            raise ValueError("calibration row availability inconsistent")
        if len({row.joint_correct for row in rows}) != 2:
            raise InsufficientCalibrationEvidence("Confidence needs both joint correctness classes")
        if all(row.raw_score.value == 0.5 for row in rows):
            raise InsufficientCalibrationEvidence("Confidence has all-neutral raw scores")
        confidence = ConfidenceCalibrator.fit(
            CalibrationEvidence(
                calibration_version="production-confidence-v1:"
                + content_hash((candidate, training.membership_hash)),
                evidence_kind="empirical",
                snapshot_id=training.snapshot_id,
                snapshot_hash=training.dataset_hash,
                score_definition=score_definition,
                development_start=source_start,
                development_end=training_end,
                frozen_at=frozen_at,
                rows=rows,
            )
        )
        reason = (
            policy_audit.absence_reason
            if policy_audit
            else "candidate policy unselected; no held-out selection performed"
        )
        payload = dict(
            schema_version="production-runtime-state-v1",
            candidate=candidate,
            configuration_hash=self._configuration_hash,
            runtime_definition_hash=self.definition.content_hash,
            dataset_hash=training.dataset_hash,
            training_groups=training.groups,
            maximum_evidence_available_at=latest,
            fit_cutoff=training.cutoff,
            model_identity=identity,
            model_lock=self.definition.model_lock,
            catalogue=self.definition.catalogue,
            source_terms=training.source_terms,
            training_evidence=descriptors,
            confidence_calibrator=confidence,
            impact_calibration=impact.model_dump(mode="json"),
            matching_config=resolved.matching.config,
            base_market=basket.base_market,
            scenario_choice=resolved.scenario_choice,
            policy_audit=policy_audit,
            policy_absence_reason=reason,
        )
        state = FrozenRuntimeState.model_validate(
            {**payload, "content_hash": content_hash(payload)}
        )
        artifact = write_runtime_state(state, self.artifact_root)
        manifest_payload = dict(
            schema_version="backtest-fit-v1",
            candidate=candidate,
            configuration_hash=self._configuration_hash,
            dataset_hash=training.dataset_hash,
            evidence_kind="empirical",
            training_groups=training.groups,
            training_hash=training.membership_hash,
            maximum_evidence_available_at=latest,
            fit_cutoff=training.cutoff,
            model_identity=identity,
            model_lock=self.definition.model_lock,
            artifacts=(artifact, self.definition.catalogue),
            confidence=confidence_calibration_identity(confidence),
            confidence_training_source_ids=tuple(sorted({row.source_item_id for row in rows})),
            impact=_impact_identity(estimator),
            source_terms=training.source_terms,
            policy=None,
            policy_absence_reason=reason,
        )
        manifest = FittedManifest.model_validate(
            {**manifest_payload, "manifest_hash": content_hash(manifest_payload)}
        )
        return ProductionFittedRuntime(
            manifest=manifest,
            state=state,
            models=models,
            matcher=matcher,
            schema_version=self.definition.signal_schema_version,
        )
