"""Locked backtest contracts with explicitly synthetic offline evidence."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from risk_engine.backtest import module as backtest
from risk_engine.backtest.module import (
    BacktestAudit,
    BacktestDataset,
    BacktestModule,
    CandidateSpec,
    DevelopmentConfiguration,
    EvaluationPeriod,
    FinalConfiguration,
    FittedManifest,
    ImpactFitIdentity,
    LocalArtifact,
    ObservedOutcome,
    RealizedValuation,
    ReferenceLoss,
    ReplayInput,
    ScenarioEvidence,
    SnapshotReference,
    SyntheticImpactAudit,
    TrainingPartition,
    canonical_bytes,
    content_hash,
)
from risk_engine.backtest.splits import ChronologicalSplitSpec
from risk_engine.config import ClusteringConfig
from risk_engine.data.clustering import cluster_stories
from risk_engine.domain import (
    AssetType,
    EntityLink,
    EventClass,
    FactorShock,
    ImpactEstimate,
    InterpretedEvent,
    MarketObservation,
    MarketSnapshot,
    Portfolio,
    PortfolioMateriality,
    Position,
    ProvenanceMethod,
    RiskSignal,
    ShockType,
    ShockUnit,
    SourceItem,
    SourceType,
    StressScenario,
)
from risk_engine.nlp.interfaces import EventModel, SentimentModel
from risk_engine.risk.confidence import (
    CalibrationEvidence,
    CalibrationRow,
    ConfidenceCalibrator,
    ConfidenceScoreDefinition,
    RawConfidenceScore,
)
from risk_engine.risk.module import (
    ActionPriorityCandidate,
    AuditedPortfolioMateriality,
    ConfidenceCalibrationIdentity,
    ModelIdentity,
    PortfolioMaterialityAudit,
    RiskEngine,
)
from risk_engine.risk.policy import ManualOverride, TriggerPolicy
from risk_engine.stress.module import StressEngine

TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)
TERMS = ("project-authored synthetic test evidence; no empirical claim",)
BASKET_HASH = content_hash({"synthetic_reference_basket": "v1"})


def seal(cls, payload, digest_field):
    payload[digest_field] = content_hash(payload)
    return cls.model_validate(payload)


def bundle(count=12):
    cases, outcomes, snapshots = [], [], []
    clustering = ClusteringConfig(similarity_threshold=1, max_time_delta_hours=1)
    for i in range(count):
        published = TIME + timedelta(days=i * 2)
        text = f"Alpha Bank credit failed case{i}."
        item = SourceItem(
            source_item_id=f"source-{i:02}",
            source_type=SourceType.NEWS,
            provider="synthetic",
            text=text,
            published_at=published,
            retrieved_at=published + timedelta(hours=1),
            source_reference=f"synthetic:case{i}",
            content_hash=sha256(text.encode()).hexdigest(),
            snapshot_id=f"source-snapshot-{i}",
            provenance=TERMS[0],
            license="MIT",
        )
        cluster = cluster_stories((item,), clustering)[0]
        market = MarketSnapshot(
            snapshot_id=f"market-{i}",
            as_of=published,
            observations=(
                MarketObservation(
                    factor_id="EQUITY-US",
                    observed_at=published,
                    value=100,
                    unit=ShockUnit.INDEX_POINT,
                    provider="synthetic",
                    vintage=f"vintage-{i}",
                ),
            ),
            provider="synthetic",
            schema_version="v1",
            normalization_version="v1",
        )
        portfolio = Portfolio(
            portfolio_id="synthetic-portfolio",
            as_of=published,
            version="v1",
            valuation_currency="USD",
            positions=(
                Position(
                    position_id="equity",
                    asset_type=AssetType.EQUITY,
                    notional=Decimal("1000"),
                    currency="USD",
                    sector="finance",
                    region="US",
                    obligor="Alpha Bank",
                    factor_exposures={"EQUITY-US": 1},
                ),
            ),
        )
        case = ReplayInput(
            case_id=f"case-{i:02}",
            cluster=cluster,
            as_of=published + timedelta(hours=2),
            portfolio=portfolio,
            market=market,
            market_available_at=published,
            market_source=SnapshotReference(
                snapshot_id=market.snapshot_id,
                content_hash=content_hash(market),
                source_terms=TERMS,
            ),
        )
        shock = FactorShock(
            factor_id="EQUITY-US",
            shock_type=ShockType.RELATIVE,
            value=-0.1 - i / 100,
            unit=ShockUnit.DECIMAL,
            horizon_days=1,
        )
        outcome = seal(
            ObservedOutcome,
            dict(
                case_id=case.case_id,
                actual_event_class=EventClass.CREDIT_DEFAULT
                if i % 2 == 0
                else EventClass.MACROECONOMIC_MONETARY,
                actual_entity_id="bank:alpha",
                label_available_at=published + timedelta(hours=3),
                outcome_available_at=published + timedelta(days=1),
                source_terms=TERMS,
                evidence_reference=f"synthetic:outcome-{i}",
                realized_shocks=(shock,),
                scenario_absence_reason=None,
                valuation=RealizedValuation(
                    pnl=Decimal(str(-100 - i * 10)),
                    currency="USD",
                    horizon_days=1,
                    comparison_scope="supported-equity-v1",
                    baseline_market_hash=content_hash(market),
                    position_ids=("equity",),
                    gross_values=(("equity", Decimal("1000")),),
                ),
                valuation_absence_reason=None,
                reference_losses=(
                    ReferenceLoss(
                        basket_hash=BASKET_HASH,
                        loss=Decimal(str(100 + i * 10)),
                        currency="USD",
                        derivation_reference="synthetic hand-calculated equity shock",
                        derivation_hash=content_hash(shock),
                        available_at=published + timedelta(days=1),
                    ),
                ),
                reference_loss_absence_reason=None,
                material_event=True,
                material_event_absence_reason=None,
            ),
            "evidence_hash",
        )
        cases.append(case)
        outcomes.append(outcome)
        snapshots.append(
            SnapshotReference(
                snapshot_id=item.snapshot_id, content_hash=content_hash((item,)), source_terms=TERMS
            )
        )
    return seal(
        BacktestDataset,
        dict(
            schema_version="backtest-dataset-v1",
            snapshot_id="synthetic-offline-v1",
            evidence_kind="synthetic",
            provenance=TERMS[0],
            source_terms=TERMS,
            source_snapshots=tuple(snapshots),
            clustering=clustering,
            cases=tuple(cases),
            outcomes=tuple(outcomes),
        ),
        "content_hash",
    )


def configuration(dataset, *, initial=4, embargo_days=1):
    candidates = tuple(
        CandidateSpec(
            candidate_id=name,
            version="v1",
            parameters={"class-index": index},
            complexity_dimensions=("rules",),
            complexity=(n,),
            event_window_days=1,
            basket_convention="synthetic-fixed-v1",
            matching_convention="synthetic-training-only-v1",
        )
        for name, index, n in (("simple", 2, 1), ("complex", 7, 2))
    )
    period = EvaluationPeriod(
        start=dataset.cases[initial].cluster.event_time,
        end=dataset.cases[-3].as_of + timedelta(hours=1),
        reporting_cutoff=dataset.outcomes[-1].outcome_available_at,
    )
    config = DevelopmentConfiguration(
        schema_version="backtest-development-v1",
        version="v1",
        frozen_at=TIME - timedelta(days=1),
        candidates=candidates,
        split=ChronologicalSplitSpec(
            version="nested-grouped-chronological-v1",
            initial_outer_train_groups=initial,
            outer_test_groups=2,
            inner_train_groups=2,
            inner_validation_groups=1,
            final_holdout_groups=2,
            embargo=timedelta(days=embargo_days),
            longest_event_window=timedelta(days=1),
        ),
        objective="event-class-macro-f1",
        direction="maximize",
        sensitivity_parameters=("class-index",),
        reliability_bin_edges=(0, 0.5, 1),
        nominal_interval_coverage=1,
        alert_budget=2,
        alert_window=EvaluationPeriod(
            start=TIME,
            end=dataset.cases[-1].as_of + timedelta(days=1),
            reporting_cutoff=dataset.outcomes[-1].outcome_available_at + timedelta(days=1),
        ),
        final_fit_cutoff=dataset.cases[-2].cluster.event_time - timedelta(seconds=1),
        reduction_rule="greatest-confidence-then-signal-id-v1",
        projection_rule="defined-finite-scalars-v1",
        final_choice_rule="last-outer-inner-winner-v1",
    )
    return config, period


def with_zero_latency(dataset, case_ids):
    payload = dataset.model_dump(exclude={"content_hash"})
    for case in payload["cases"]:
        if case["case_id"] in case_ids:
            case["as_of"] = case["cluster"]["event_time"]
            for item in case["cluster"]["items"]:
                item["retrieved_at"] = item["published_at"]
    for snapshot in payload["source_snapshots"]:
        members = tuple(
            SourceItem.model_validate(item)
            for case in payload["cases"]
            for item in case["cluster"]["items"]
            if item["snapshot_id"] == snapshot["snapshot_id"]
        )
        snapshot["content_hash"] = content_hash(members)
    return seal(BacktestDataset, payload, "content_hash")


class Matcher:
    def match(self, text):
        return [
            EntityLink(
                entity_id="bank:alpha",
                canonical_name="Alpha Bank",
                confidence=0.8,
                evidence="Alpha Bank",
                ambiguous=False,
            )
        ]


class SyntheticImpact:
    def __init__(self, identity):
        self.identity = identity

    def estimate(self, event: InterpretedEvent, as_of: datetime):
        audit = SyntheticImpactAudit(schema_version="synthetic-impact-fit-v1", fit=self.identity)
        return ImpactEstimate(
            impact_score=8,
            expected_reference_loss=Decimal("150"),
            loss_lower_bound=Decimal("100"),
            loss_upper_bound=Decimal("300"),
            loss_currency="USD",
            analogue_count=0,
            analogue_ids=(),
            backoff_level="synthetic",
            method=ProvenanceMethod.HYPOTHETICAL,
            calibration_version=canonical_bytes(audit).decode(),
            reference_basket_version=self.identity.reference_basket_version,
        )


class Runtime:
    def __init__(self, manifest, engine):
        self.manifest = manifest
        self.risk_engine = engine
        self.scenario_calls = []

    def scenarios(self, signal: RiskSignal, case: ReplayInput, *, as_of: datetime):
        self.scenario_calls.append(case.case_id)
        shock = FactorShock(
            factor_id="EQUITY-US",
            shock_type=ShockType.RELATIVE,
            value=-0.15,
            unit=ShockUnit.DECIMAL,
            horizon_days=1,
        )
        return ScenarioEvidence(
            scenario=StressScenario(
                scenario_id=f"scenario:{signal.signal_id}",
                risk_signal_id=signal.signal_id,
                shocks=(shock,),
                method=ProvenanceMethod.HYPOTHETICAL,
                calibration_version=signal.impact.calibration_version,
            ),
            joint_samples=((shock,),),
            fit_hash=self.manifest.manifest_hash,
            calibration_hash=self.manifest.impact.calibration_hash,
            available_at=self.manifest.fit_cutoff,
            source_terms=TERMS,
            support_reference="project-authored synthetic training scenario",
            support_hash=content_hash(shock),
        )


class Fitter:
    def __init__(self, tmp_path, config, *, materiality_provider=None):
        self.root = tmp_path
        self.config = config
        self.materiality_provider = materiality_provider
        self.fits = []
        self.runtimes = {}
        self.restores = []
        self.fail = False

    def fit(self, candidate: CandidateSpec, training: TrainingPartition, *, as_of: datetime):
        self.fits.append((candidate.candidate_id, training, as_of))
        definition = ConfidenceScoreDefinition(
            event_model_version="synthetic-event-v1", entity_linker_version="synthetic-link-v1"
        )
        rows = tuple(
            CalibrationRow(
                row_id=c.case_id,
                source_item_id=c.cluster.items[0].source_item_id,
                event_id=c.cluster.cluster_id,
                entity_id="bank:alpha",
                event_class=o.actual_event_class,
                content_hash=c.cluster.items[0].content_hash,
                source_terms=TERMS[0],
                label_provenance=o.evidence_reference,
                source_available_at=c.cluster.items[0].retrieved_at,
                label_available_at=o.label_available_at,
                split="development",
                raw_score=RawConfidenceScore(
                    definition=definition, entity_link_confidence=0.8, classification_confidence=0.8
                ),
                entity_correct=True,
                event_class_correct=i % 2 == 0,
            )
            for i, (c, o) in enumerate(zip(training.cases, training.outcomes, strict=True))
        )
        calibrator = ConfidenceCalibrator.fit(
            CalibrationEvidence(
                calibration_version="synthetic-confidence-v1",
                evidence_kind="synthetic",
                snapshot_id=training.snapshot_id,
                snapshot_hash=training.dataset_hash,
                score_definition=definition,
                development_start=min(r.source_available_at for r in rows),
                development_end=max(r.source_available_at for r in rows),
                frozen_at=as_of,
                rows=rows,
            )
        )
        confidence = ConfidenceCalibrationIdentity(
            target=calibrator.target,
            calibration_version=calibrator.evidence.calibration_version,
            evidence_kind="synthetic",
            calibration_evidence_snapshot_id=training.snapshot_id,
            calibration_evidence_snapshot_hash=training.dataset_hash,
            development_start=calibrator.evidence.development_start,
            development_end=calibrator.evidence.development_end,
            frozen_at=as_of,
            score_definition=definition,
            fit_version=calibrator.fit_version,
            transform_version=calibrator.transform_version,
            temperature=calibrator.temperature,
            evidence_hash=calibrator.evidence_hash,
            fitted_artifact_hash=calibrator.fitted_artifact_hash,
            source_terms=TERMS,
        )
        basket_hash = candidate.parameters.get("reference-basket-hash", BASKET_HASH)
        losses = sorted(
            next(ref.loss for ref in outcome.reference_losses if ref.basket_hash == basket_hash)
            for outcome in training.outcomes
        )
        identity = ImpactFitIdentity(
            evidence_kind="synthetic",
            calibration_version="synthetic-impact-v1",
            calibration_hash=content_hash({"training": training.membership_hash, "losses": losses}),
            reference_basket_version="synthetic-reference-v1",
            reference_basket_hash=basket_hash,
            matching_version="synthetic-training-only-v1",
            training_event_ids=tuple(g.cluster_id for g in training.groups),
            cutpoints=tuple(
                losses[min(len(losses) - 1, max(0, (len(losses) * k + 9) // 10 - 1))]
                for k in range(1, 10)
            ),
            currency="USD",
            horizon_days=1,
            quantile_convention="nearest-rank-lower-ties",
            frozen_at=as_of,
        )
        fit_path = self.root / f"fit-{len(self.fits)}.json"
        fit_path.write_bytes(canonical_bytes(calibrator))
        model_path = self.root / "synthetic-model.json"
        if not model_path.exists():
            model_path.write_bytes(canonical_bytes({"synthetic_nlp": "project-authored v1"}))
        manifest = seal(
            FittedManifest,
            dict(
                schema_version="backtest-fit-v1",
                candidate=candidate,
                configuration_hash=content_hash(self.config),
                dataset_hash=training.dataset_hash,
                evidence_kind="synthetic",
                training_groups=training.groups,
                training_hash=training.membership_hash,
                maximum_evidence_available_at=max(training.availability),
                fit_cutoff=as_of,
                model_identity=ModelIdentity(
                    event_model_version="synthetic-event-v1",
                    sentiment_model_version="synthetic-sentiment-v1",
                    entity_linker_version="synthetic-link-v1",
                ),
                model_lock=LocalArtifact(
                    identity="synthetic-model-declaration-v1",
                    path=str(model_path),
                    sha256=sha256(model_path.read_bytes()).hexdigest(),
                    available_at=TIME - timedelta(days=1),
                ),
                artifacts=(
                    LocalArtifact(
                        identity="confidence-calibrator",
                        path=str(fit_path),
                        sha256=sha256(fit_path.read_bytes()).hexdigest(),
                        available_at=as_of,
                    ),
                ),
                confidence=confidence,
                confidence_training_source_ids=tuple(
                    sorted(s for g in training.groups for s in g.source_item_ids)
                ),
                impact=identity,
                source_terms=TERMS,
                policy=None,
                policy_absence_reason="synthetic; no empirical policy selection",
            ),
            "manifest_hash",
        )
        index = candidate.parameters["class-index"]

        def infer(text, hypotheses):
            return [3 if i == index else 0 for i in range(8)]

        runtime = Runtime(
            manifest,
            RiskEngine(
                matcher=Matcher(),
                sentiment_model=SentimentModel(lambda text: (0, 3, 0)),
                event_model=EventModel(infer),
                impact_estimator=SyntheticImpact(identity),
                confidence_calibrator=calibrator,
                model_identity=manifest.model_identity,
                schema_version="risk-signal-schema-v1",
                materiality_provider=self.materiality_provider,
                action_priority=ActionPriorityCandidate(),
            ),
        )
        self.runtimes[manifest.manifest_hash] = runtime
        return runtime

    def restore(self, locked):
        self.restores.append(locked)
        if self.fail:
            raise RuntimeError("restore failure")
        return self.runtimes[locked.manifest_hash]


def test_nested_real_engine_replay_retains_all_candidates_and_train_only_fits(tmp_path):
    dataset = bundle()
    config, period = configuration(dataset)
    fitter = Fitter(tmp_path, config)
    module = BacktestModule(fitter, StressEngine())
    report = module.evaluate(dataset, config, period)
    audit = module.audit
    assert report.schema_version == "backtest-report-v1"
    assert report.report_id.startswith("synthetic:")
    assert {c.candidate_id for c in report.candidate_configurations} == {"simple", "complex"}
    assert [c.candidate_id for c in report.candidate_configurations if c.selected] == ["simple"]
    assert len(report.sensitivity_results) == 2
    assert len(audit.outer_results) == 3
    assert audit.report_hash == content_hash(report)
    assert len(audit.outer_results[0].selection.candidates) == 2
    assert len(audit.outer_results[0].selection.candidates[0].fold_results) == 2
    assert report.historical_case_ids == tuple(c.case_id for c in dataset.cases[4:10])
    assert report.metrics["valuation.pnl_mae"] == pytest.approx(110 / 6)
    assert report.metrics["entity_link_accuracy"] == 1
    assert "severity.rank_correlation" not in report.metrics
    assert any(r.metric == "severity.rank_correlation" for r in audit.metrics.undefined)
    assert audit.metrics.valuation.pnl_mae == Decimal("18.33333333333333333333333333")
    for _, training, as_of in fitter.fits:
        assert training.cutoff == as_of
        assert not {c.case_id for c in training.cases} & {"case-10", "case-11"}
        assert max(training.availability) <= as_of
        assert tuple(g.cluster_id for g in training.groups) == tuple(
            c.cluster.cluster_id for c in training.cases
        )
    assert all(
        c.decision is not None and not c.decision.automatic_trigger
        for o in audit.outer_results
        for c in o.test.cases
    )
    assert all(
        c.chosen_signal_id == c.returned_signal_ids[0]
        for o in audit.outer_results
        for c in o.test.cases
    )


def test_failed_evaluation_clears_previous_successful_audit(tmp_path):
    dataset = bundle()
    config, period = configuration(dataset)
    module = BacktestModule(Fitter(tmp_path, config), StressEngine())
    with pytest.raises(ValueError, match="successful"):
        _ = module.audit
    module.evaluate(dataset, config, period)
    with pytest.raises(ValueError, match="dataset content hash"):
        module.evaluate(dataset.model_copy(update={"content_hash": "0" * 64}), config, period)
    with pytest.raises(ValueError, match="successful"):
        _ = module.audit


def test_refuses_outer_with_only_one_inner_fold(tmp_path):
    dataset = bundle()
    config, period = configuration(dataset, initial=3)
    fitter = Fitter(tmp_path, config)
    with pytest.raises(ValueError, match="at least two"):
        BacktestModule(fitter, StressEngine()).evaluate(dataset, config, period)
    assert fitter.fits == []


def test_final_restores_exact_locked_fit_without_refitting_or_reselection(tmp_path):
    dataset = bundle()
    config, period = configuration(dataset)
    fitter = Fitter(tmp_path, config)
    module = BacktestModule(fitter, StressEngine())
    development = module.evaluate(dataset, config, period)
    lock = module.audit.locked
    fits_before = len(fitter.fits)
    final = module.evaluate(
        dataset,
        FinalConfiguration(schema_version="backtest-final-v1", locked=lock),
        lock.final_period,
    )
    assert len(fitter.fits) == fits_before
    assert fitter.restores == [lock.fitted]
    assert final.historical_case_ids == ("case-10", "case-11")
    assert final.candidate_configurations == development.candidate_configurations
    assert final.sensitivity_results == development.sensitivity_results
    assert module.audit.outer_results == ()
    assert module.audit.final_result.fit == lock.fitted
    assert module.audit.report_hash == content_hash(final)


def test_zero_latency_development_replays_single_group_inner_validations(tmp_path):
    original = bundle()
    dataset = with_zero_latency(original, {case.case_id for case in original.cases})
    config, period = configuration(dataset)
    module = BacktestModule(Fitter(tmp_path, config), StressEngine())
    report = module.evaluate(dataset, config, period)
    assert report.historical_case_ids == tuple(f"case-{i:02}" for i in range(4, 10))
    validations = module.audit.outer_results[0].inner_candidates[0].validations
    assert tuple(len(v.cases) for v in validations) == (1, 1)
    assert validations[0].cases[0].as_of == TIME + timedelta(days=4)


def test_zero_latency_final_tail_includes_event_at_last_replay_cutoff(tmp_path):
    dataset = with_zero_latency(bundle(), {"case-10", "case-11"})
    config, period = configuration(dataset)
    fitter = Fitter(tmp_path, config)
    module = BacktestModule(fitter, StressEngine())
    module.evaluate(dataset, config, period)
    lock = module.audit.locked
    fits_before = len(fitter.fits)
    report = module.evaluate(
        dataset,
        FinalConfiguration(schema_version="backtest-final-v1", locked=lock),
        lock.final_period,
    )
    assert report.historical_case_ids == ("case-10", "case-11")
    assert module.audit.final_result.cases[-1].as_of == TIME + timedelta(days=22)
    assert report.evaluation_end > TIME + timedelta(days=22)
    assert len(fitter.fits) == fits_before


def test_final_independent_embargo_omits_last_development_group(tmp_path):
    dataset = bundle(14)
    config, period = configuration(dataset, initial=5, embargo_days=3)
    module = BacktestModule(Fitter(tmp_path, config), StressEngine())
    module.evaluate(dataset, config, period)
    lock = module.audit.locked
    assert lock.omitted_final_boundary_groups == (
        module.audit.folds[-1]
        .final_holdout[0]
        .model_copy(
            update={
                "cluster_id": dataset.cases[-3].cluster.cluster_id,
                "event_time": dataset.cases[-3].cluster.event_time,
                "source_item_ids": dataset.cases[-3].cluster.source_item_ids,
            }
        ),
    )
    assert dataset.cases[-3].cluster.cluster_id not in {
        g.cluster_id for g in lock.fitted.training_groups
    }


def test_serialized_audit_roundtrip_and_defensive_copy(tmp_path):
    dataset = bundle()
    config, period = configuration(dataset)
    module = BacktestModule(Fitter(tmp_path, config), StressEngine())
    report = module.evaluate(dataset, config, period)
    restored = BacktestAudit.model_validate_json(canonical_bytes(module.audit))
    assert restored == module.audit
    first = module.audit
    first.metrics.scalar_projection["classification.macro_f1"] = 999
    first.locked.fitted.candidate.parameters["class-index"] = 7
    assert (
        module.audit.metrics.scalar_projection["classification.macro_f1"]
        == report.metrics["classification.macro_f1"]
    )
    assert module.audit.locked.fitted.candidate.parameters["class-index"] == 2
    report.metrics["classification.macro_f1"] = 999
    assert module.audit.report_hash != content_hash(report)


@pytest.mark.parametrize("mutation", ("materiality", "manual-override"))
def test_serialized_audit_rejects_decision_contradicting_replay_inputs(tmp_path, mutation):
    dataset = bundle()
    config, period = configuration(dataset)
    module = BacktestModule(Fitter(tmp_path, config), StressEngine())
    module.evaluate(dataset, config, period)
    audit = module.audit
    row = audit.outer_results[0].test.cases[0]
    assert row.stress.absolute_loss == Decimal("150.000")
    replacement = TriggerPolicy(row.policy_binding.concrete_policy, as_of=row.as_of).evaluate(
        row.chosen_signal,
        PortfolioMateriality(absolute_loss=Decimal("999"), percentage_loss=99.9, currency="USD")
        if mutation == "materiality"
        else row.decision.portfolio_materiality,
        manual_override=ManualOverride(analyst_id="synthetic-analyst", reason="test override")
        if mutation == "manual-override"
        else None,
    )
    payload = audit.model_dump()
    payload["outer_results"][0]["test"]["cases"][0]["decision"] = replacement.model_dump()
    with pytest.raises(ValueError, match="materiality|manual override"):
        BacktestAudit.model_validate(payload)


def test_utc_normalization_distinguishes_repeated_dst_hour():
    zone = ZoneInfo("America/New_York")
    start = datetime(2024, 11, 3, 1, 30, tzinfo=zone, fold=0)
    end = datetime(2024, 11, 3, 1, 30, tzinfo=zone, fold=1)
    period = EvaluationPeriod(start=start, end=end, reporting_cutoff=end)
    assert (period.end - period.start).total_seconds() == 3600
    assert period.start.utcoffset() == timedelta(0)
    dataset = bundle()
    payload = dataset.model_dump()
    for case in payload["cases"]:
        for item in case["cluster"]["items"]:
            item["published_at"] = item["published_at"].astimezone(zone)
            item["retrieved_at"] = item["retrieved_at"].astimezone(zone)
        case["cluster"]["event_time"] = case["cluster"]["event_time"].astimezone(zone)
    assert BacktestDataset.model_validate(payload) == dataset


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("source", "Source Item content hash"),
        ("market", "market unavailable"),
        ("outcome", "outcome evidence content hash"),
        ("config", "configuration must be frozen"),
        ("label", "training evidence unavailable"),
    ],
)
def test_rejects_tampered_or_future_inputs_before_fitting(tmp_path, mutation, message):
    dataset = bundle()
    config, period = configuration(dataset)
    payload = dataset.model_dump()
    if mutation == "source":
        payload["cases"][0]["cluster"]["items"][0]["text"] += " revised"
    elif mutation == "market":
        payload["cases"][0]["market_available_at"] = TIME + timedelta(days=100)
    elif mutation == "outcome":
        payload["outcomes"][0]["actual_entity_id"] = "tampered"
    elif mutation == "label":
        payload["outcomes"][0]["label_available_at"] = TIME + timedelta(days=100)
        raw = payload["outcomes"][0]
        raw["evidence_hash"] = content_hash({k: v for k, v in raw.items() if k != "evidence_hash"})
        payload["content_hash"] = content_hash(
            {k: v for k, v in payload.items() if k != "content_hash"}
        )
    else:
        config = config.model_copy(update={"frozen_at": TIME + timedelta(days=100)})
    fitter = Fitter(tmp_path, config)
    with pytest.raises(ValueError, match=message):
        # unchecked copies still revalidate at evaluation entry
        changed = dataset.model_copy(update=payload)
        BacktestModule(fitter, StressEngine()).evaluate(changed, config, period)
    assert fitter.fits == []


def test_final_rejects_changed_dataset_and_period_before_restore(tmp_path):
    dataset = bundle()
    config, period = configuration(dataset)
    fitter = Fitter(tmp_path, config)
    module = BacktestModule(fitter, StressEngine())
    module.evaluate(dataset, config, period)
    lock = module.audit.locked
    payload = dataset.model_dump(exclude={"content_hash"})
    payload["snapshot_id"] = "another-synthetic-snapshot"
    other = seal(BacktestDataset, payload, "content_hash")
    with pytest.raises(ValueError, match="locked dataset"):
        module.evaluate(
            other,
            FinalConfiguration(schema_version="backtest-final-v1", locked=lock),
            lock.final_period,
        )
    with pytest.raises(ValueError, match="locked final period"):
        module.evaluate(
            dataset, FinalConfiguration(schema_version="backtest-final-v1", locked=lock), period
        )
    assert fitter.restores == []


def cli():
    from scripts import run_backtest

    return run_backtest


def write_cli_inputs(tmp_path):
    dataset = bundle()
    config, period = configuration(dataset)
    for name, value in (
        ("dataset.json", dataset),
        ("config.json", config),
        ("period.json", period),
    ):
        (tmp_path / name).write_bytes(canonical_bytes(value))
    fitter = Fitter(tmp_path, config)
    select_args = [
        "--select",
        "--dataset",
        str(tmp_path / "dataset.json"),
        "--config",
        str(tmp_path / "config.json"),
        "--period",
        str(tmp_path / "period.json"),
        "--output-root",
        str(tmp_path / "selection"),
    ]
    return dataset, config, fitter, select_args


def test_select_publishes_complete_cross_hashed_synthetic_set_and_refuses_overwrite(tmp_path):
    dataset, config, fitter, args = write_cli_inputs(tmp_path)
    runner = cli()
    assert runner.main(args, fitter=fitter) == 0
    selected, cutpoints, development = runner.load_artifact_set(tmp_path / "selection")
    assert selected.evidence_kind == "synthetic"
    assert cutpoints.payload.status == "synthetic-only"
    assert selected.payload.locked.dataset_hash == dataset.content_hash
    assert selected.payload.locked.configuration_hash == content_hash(config)
    assert development.payload.report.report_id.startswith("synthetic:")
    assert development.payload.audit.report_hash == content_hash(development.payload.report)
    bytes_before = {p: p.read_bytes() for p in (tmp_path / "selection").rglob("*.json")}
    fits_before = len(fitter.fits)
    assert runner.main(args, fitter=fitter) == 1
    assert len(fitter.fits) == fits_before
    assert all(p.read_bytes() == data for p, data in bytes_before.items())


@pytest.mark.parametrize("repeat_path", ["same", "different"])
def test_final_cli_reserves_and_refuses_second_attempt_before_inference(tmp_path, repeat_path):
    _, _, fitter, args = write_cli_inputs(tmp_path)
    runner = cli()
    assert runner.main(args, fitter=fitter) == 0
    final_args = [
        "--final",
        "--dataset",
        str(tmp_path / "dataset.json"),
        "--lock-root",
        str(tmp_path / "selection"),
        "--output-file",
        str(tmp_path / "final.json"),
    ]
    assert runner.main(final_args, fitter=fitter) == 0
    final = json.loads((tmp_path / "final.json").read_text())
    assert final["status"] == "complete"
    assert final["evidence_kind"] == "synthetic"
    assert final["report"]["historical_case_ids"] == ["case-10", "case-11"]
    restores_before = len(fitter.restores)
    if repeat_path == "different":
        final_args[-1] = str(tmp_path / "another-final.json")
    assert runner.main(final_args, fitter=fitter) == 1
    assert len(fitter.restores) == restores_before


def test_failed_final_claim_remains_consumed_across_output_names(tmp_path):
    _, _, fitter, args = write_cli_inputs(tmp_path)
    runner = cli()
    assert runner.main(args, fitter=fitter) == 0
    final_args = [
        "--final",
        "--dataset",
        str(tmp_path / "dataset.json"),
        "--lock-root",
        str(tmp_path / "selection"),
        "--output-file",
        str(tmp_path / "failed.json"),
    ]
    fitter.fail = True
    assert runner.main(final_args, fitter=fitter) == 1
    assert json.loads((tmp_path / "failed.json").read_text())["status"] == "failed"
    restores_before = len(fitter.restores)
    fitter.fail = False
    final_args[-1] = str(tmp_path / "retry-final.json")
    assert runner.main(final_args, fitter=fitter) == 1
    assert len(fitter.restores) == restores_before


def test_final_tamper_or_incomplete_selection_refuses_before_reservation(tmp_path):
    _, _, fitter, args = write_cli_inputs(tmp_path)
    runner = cli()
    assert runner.main(args, fitter=fitter) == 0
    selected_path = tmp_path / "selection/data/calibration/selected-config.json"
    selected_path.unlink()
    final_args = [
        "--final",
        "--dataset",
        str(tmp_path / "dataset.json"),
        "--lock-root",
        str(tmp_path / "selection"),
        "--output-file",
        str(tmp_path / "final.json"),
    ]
    assert runner.main(final_args, fitter=fitter) == 1
    assert fitter.restores == []
    assert not (tmp_path / "final.json").exists()
    assert not tuple((tmp_path / "selection").glob(".final-attempt*"))


def test_artifact_tamper_rejected(tmp_path):
    _, _, fitter, args = write_cli_inputs(tmp_path)
    runner = cli()
    assert runner.main(args, fitter=fitter) == 0
    development_path = tmp_path / "selection/data/results/development-backtest.json"
    payload = json.loads(development_path.read_text())
    payload["payload"]["report"]["metrics"]["entity_link_accuracy"] = 0.5
    development_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        runner.load_artifact_set(tmp_path / "selection")


def test_direct_cli_has_actionable_composition_refusal_and_help(tmp_path, capsys):
    _, _, _, args = write_cli_inputs(tmp_path)
    runner = cli()
    assert runner.main(args) == 2
    assert "explicit application composition" in capsys.readouterr().err
    with pytest.raises(SystemExit) as help_exit:
        runner.main(["--help"])
    assert help_exit.value.code == 0


def test_checked_in_artifacts_are_valid_unresolved_envelopes():
    runner = cli()
    selected, cutpoints, development = runner.load_artifact_set(Path.cwd(), require_ready=False)
    for artifact in (selected, cutpoints, development):
        assert artifact.status == "unresolved"
        assert artifact.payload.dataset_hash is None
        assert artifact.payload.model_hash is None
        assert artifact.payload.selected_candidate is None
        assert artifact.payload.cutpoints is None
        assert artifact.payload.report is None
        assert len(artifact.payload.reasons) == 4
    with pytest.raises(ValueError, match="unresolved"):
        runner.load_artifact_set(Path.cwd())


def test_backtest_public_interface() -> None:
    assert callable(BacktestModule.evaluate)


def test_serialized_audit_rejects_fabricated_scalar_metrics(tmp_path):
    dataset = bundle()
    config, period = configuration(dataset)
    module = BacktestModule(Fitter(tmp_path, config), StressEngine())
    module.evaluate(dataset, config, period)
    payload = module.audit.model_dump()
    payload["metrics"]["scalar_projection"]["entity_link_accuracy"] = 0.25
    with pytest.raises(ValueError, match="metric evidence"):
        BacktestAudit.model_validate(payload)


def test_serialized_audit_rejects_inner_scores_unbound_from_metrics(tmp_path):
    dataset = bundle()
    config, period = configuration(dataset)
    module = BacktestModule(Fitter(tmp_path, config), StressEngine())
    module.evaluate(dataset, config, period)
    payload = module.audit.model_dump()
    payload["outer_results"][0]["inner_candidates"][0]["validations"][0]["metrics"][
        "scalar_projection"
    ]["classification.macro_f1"] = 0.123
    with pytest.raises(ValueError, match="metric evidence"):
        BacktestAudit.model_validate(payload)


def test_incomplete_gross_support_evidence_makes_valuation_unavailable(tmp_path):
    dataset = bundle()
    payload = dataset.model_dump(exclude={"content_hash"})
    for outcome in payload["outcomes"]:
        outcome["valuation"]["gross_values"] = (("invented-position", Decimal("1000")),)
        outcome["valuation"]["position_ids"] = ("invented-position",)
        outcome["evidence_hash"] = content_hash(
            {k: v for k, v in outcome.items() if k != "evidence_hash"}
        )
    changed = seal(BacktestDataset, payload, "content_hash")
    config, period = configuration(changed)
    module = BacktestModule(Fitter(tmp_path, config), StressEngine())
    report = module.evaluate(changed, config, period)
    assert "valuation.pnl_mae" not in report.metrics
    assert any("gross support" in m.reason for m in module.audit.metrics.undefined)


class SyntheticMaterialityProvider:
    def measure(self, event, entity, estimate, *, as_of):
        return AuditedPortfolioMateriality(
            materiality=PortfolioMateriality(
                absolute_loss=Decimal("150"), percentage_loss=0.15, currency="USD"
            ),
            audit=PortfolioMaterialityAudit(
                provider_version="synthetic-v1",
                calculation_version="synthetic-v1",
                portfolio_id="synthetic-portfolio",
                portfolio_version="v1",
                market_snapshot_id="synthetic-market",
                scenario_id="synthetic-scenario",
                scenario_version="v1",
                valuation_rule_version="v1",
                as_of=as_of,
            ),
        )


@pytest.mark.parametrize("invalid_support", ("gross-values", "position-scope", None))
def test_alert_metrics_require_applicable_portfolio_loss_evidence(tmp_path, invalid_support):
    dataset = bundle()
    if invalid_support is not None:
        payload = dataset.model_dump(exclude={"content_hash"})
        for outcome in payload["outcomes"]:
            if invalid_support == "gross-values":
                outcome["valuation"]["gross_values"] = (("invented-position", Decimal("1000")),)
            outcome["valuation"]["position_ids"] = ("invented-position",)
            outcome["evidence_hash"] = content_hash(
                {k: v for k, v in outcome.items() if k != "evidence_hash"}
            )
        dataset = seal(BacktestDataset, payload, "content_hash")
    config, period = configuration(dataset)
    fitter = Fitter(tmp_path, config, materiality_provider=SyntheticMaterialityProvider())
    module = BacktestModule(fitter, StressEngine())
    report = module.evaluate(dataset, config, period)
    audit = module.audit
    assert all(
        row.chosen_signal.action_priority is not None
        for outer in audit.outer_results
        for row in outer.test.cases
    )
    if invalid_support is not None:
        assert audit.metrics.valuation is None
        assert audit.metrics.alerts is None
        assert not any(name.startswith("alerts.") for name in report.metrics)
        assert any(
            item.metric == "case.case-04.alerts"
            and ("gross support" in item.reason or "supported subset" in item.reason)
            for item in audit.metrics.undefined
        )
    else:
        assert audit.metrics.valuation.pnl_mae == Decimal("18.33333333333333333333333333")
        assert audit.metrics.alerts.recall == 0
        assert audit.metrics.alerts.loss_captured_at_budget == Decimal("0")
        assert report.metrics["alerts.recall"] == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("baseline_market_hash", "0" * 64, "baseline market hash mismatch"),
        ("currency", "INR", "currency/horizon mismatch"),
        ("horizon_days", 2, "currency/horizon mismatch"),
    ),
)
def test_invalid_support_cannot_hide_inconsistent_portfolio_loss_units(
    tmp_path, field, value, message
):
    dataset = bundle()
    payload = dataset.model_dump(exclude={"content_hash"})
    for outcome in payload["outcomes"]:
        outcome["valuation"]["gross_values"] = (("invented-position", Decimal("1000")),)
        outcome["valuation"]["position_ids"] = ("invented-position",)
        outcome["valuation"][field] = value
        outcome["evidence_hash"] = content_hash(
            {k: v for k, v in outcome.items() if k != "evidence_hash"}
        )
    changed = seal(BacktestDataset, payload, "content_hash")
    config, period = configuration(changed)
    fitter = Fitter(tmp_path, config, materiality_provider=SyntheticMaterialityProvider())
    with pytest.raises(ValueError, match=message):
        BacktestModule(fitter, StressEngine()).evaluate(changed, config, period)


def test_production_impact_audit_decodes_and_verifies_its_actual_digest():
    from risk_engine.backtest.module import decode_impact_audit
    from tests.impact.test_analogues import AS_OF, event
    from tests.impact.test_impact_score import estimator

    estimate = estimator().estimate(event(), AS_OF)
    parsed = decode_impact_audit(estimate.calibration_version)
    assert parsed.support_count == 3
    assert parsed.cutpoints == tuple(map(Decimal, range(10, 91, 10)))
    payload = json.loads(estimate.calibration_version)
    payload["matching_version"] = "tampered"
    with pytest.raises(ValueError, match="Impact audit content hash"):
        decode_impact_audit(json.dumps(payload))


def test_fitted_model_or_scenario_mismatch_refuses_and_clears_audit(tmp_path):
    dataset = bundle()
    config, period = configuration(dataset)
    fitter = Fitter(tmp_path, config)
    original_fit = fitter.fit

    class BrokenScenario:
        def __init__(self, runtime):
            self.manifest = runtime.manifest
            self.risk_engine = runtime.risk_engine
            self.runtime = runtime

        def scenarios(self, signal, case, *, as_of):
            return self.runtime.scenarios(signal, case, as_of=as_of).model_copy(
                update={"fit_hash": "0" * 64}
            )

    def broken_fit(candidate, training, *, as_of):
        return BrokenScenario(original_fit(candidate, training, as_of=as_of))

    fitter.fit = broken_fit
    module = BacktestModule(fitter, StressEngine())
    with pytest.raises(ValueError, match="ScenarioEvidence"):
        module.evaluate(dataset, config, period)
    with pytest.raises(ValueError, match="successful"):
        _ = module.audit


def test_replay_is_independent_of_ambient_decimal_context(tmp_path):
    from decimal import ROUND_UP, localcontext

    original = bundle()
    payload = original.model_dump(exclude={"content_hash"})
    for case in payload["cases"]:
        case["portfolio"]["positions"][0]["notional"] = Decimal("1000.125")
    for outcome in payload["outcomes"]:
        outcome["valuation"]["gross_values"] = (("equity", Decimal("1000.125")),)
        outcome["valuation"]["pnl"] -= Decimal("0.125")
        outcome["evidence_hash"] = content_hash(
            {k: v for k, v in outcome.items() if k != "evidence_hash"}
        )
    dataset = seal(BacktestDataset, payload, "content_hash")
    config, period = configuration(dataset)
    fitter = Fitter(tmp_path, config)
    module = BacktestModule(fitter, StressEngine())
    report = module.evaluate(dataset, config, period)
    expected_report_bytes = canonical_bytes(report)
    expected_metrics_bytes = canonical_bytes(module.audit.metrics)
    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_UP
        repeated = module.evaluate(dataset, config, period)
        assert canonical_bytes(repeated) == expected_report_bytes
        assert canonical_bytes(module.audit.metrics) == expected_metrics_bytes


def test_policy_evidence_hash_binds_thresholds_and_exact_composite(tmp_path):
    """Constructed binding contract fixture; never an empirical selection result."""
    from risk_engine.backtest.module import (
        PolicyEvidence,
        PolicyTemplate,
        bind_trigger_policy,
        decode_impact_audit,
    )
    from risk_engine.backtest.splits import FoldGroup
    from risk_engine.domain import ConfidenceTarget
    from risk_engine.risk.module import CalibrationManifest
    from tests.impact.test_analogues import AS_OF, event
    from tests.impact.test_impact_score import estimator

    dataset = bundle()
    config, period = configuration(dataset)
    fitter = Fitter(tmp_path, config)
    module = BacktestModule(fitter, StressEngine())
    module.evaluate(dataset, config, period)
    base = module.audit.selected_final_manifest
    signal = fitter.runtimes[base.manifest_hash].risk_engine.analyze(
        (dataset.cases[-2].cluster.items[0],), dataset.cases[-2].as_of
    )[0]
    impact = estimator().estimate(event(), AS_OF)
    impact_audit = decode_impact_audit(impact.calibration_version)
    groups = tuple(
        FoldGroup(
            cluster_id=loss.event_id, event_time=TIME, source_item_ids=(f"contract-source-{i}",)
        )
        for i, loss in enumerate(impact_audit.training_losses)
    )
    payload = base.model_dump(exclude={"manifest_hash"})
    payload.update(
        evidence_kind="empirical",
        training_groups=groups,
        training_hash=content_hash(groups),
        confidence_training_source_ids=tuple(sorted(s for g in groups for s in g.source_item_ids)),
        fit_cutoff=AS_OF,
        maximum_evidence_available_at=AS_OF - timedelta(days=1),
    )
    payload["confidence"]["evidence_kind"] = "empirical"
    payload["impact"].update(
        evidence_kind="empirical",
        calibration_hash=impact_audit.calibration_sha256,
        reference_basket_version=impact.reference_basket_version,
        reference_basket_hash=impact_audit.calibration.reference_basket_sha256,
        matching_version=impact_audit.matching_version,
        training_event_ids=tuple(g.cluster_id for g in groups),
        cutpoints=impact_audit.cutpoints,
        frozen_at=impact_audit.calibration.calibrated_at,
    )
    unselected = seal(FittedManifest, payload, "manifest_hash")
    template = PolicyTemplate(
        version="constructed-binding-contract-v1",
        validation_status="chronologically_validated",
        evidence_kind="empirical",
        selection_protocol="nested_chronological_development",
        development_start=TIME,
        development_end=TIME + timedelta(days=1),
        frozen_at=AS_OF - timedelta(days=1),
        validation_evidence="Constructed contract fixture only; no empirical optimum claim",
        evidence_hash="0" * 64,
        snapshot_id="constructed-policy-contract-v1",
        source_terms=TERMS[0],
        confidence_target=ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS,
        confidence_threshold=Decimal("0.5"),
        economic_floor=Decimal("1"),
        materiality_tolerance=Decimal("0"),
        currency="USD",
    )
    binding_payload = dict(
        schema_version="backtest-policy-binding-v1",
        candidate_hash=content_hash(unselected.candidate),
        dataset_hash=unselected.dataset_hash,
        training_hash=unselected.training_hash,
        stable_fit_hash=unselected.stable_fit_hash,
        template=template,
    )
    template_without_hash = template.model_dump(exclude={"evidence_hash"})
    digest_payload = {**binding_payload, "template": template_without_hash}
    binding_payload["template"] = template.model_copy(
        update={"evidence_hash": content_hash(digest_payload)}
    )
    binding = PolicyEvidence.model_validate(binding_payload)
    selected_payload = unselected.model_dump(exclude={"manifest_hash"})
    selected_payload.update(policy=binding, policy_absence_reason=None)
    manifest = seal(FittedManifest, selected_payload, "manifest_hash")
    calibration = CalibrationManifest.model_validate_json(signal.versions.calibration_version)
    actual_calibration = calibration.model_dump()
    actual_calibration["confidence"]["evidence_kind"] = "empirical"
    actual_calibration["confidence_calibration"] = manifest.confidence.model_dump()
    actual_calibration["impact"] = dict(
        calibration_version=impact.calibration_version,
        reference_basket_version=impact.reference_basket_version,
    )
    actual_signal = signal.model_copy(
        update={
            "impact": impact,
            "versions": signal.versions.model_copy(
                update={"calibration_version": canonical_bytes(actual_calibration).decode()}
            ),
        }
    )
    config_bound, audit = bind_trigger_policy(actual_signal, manifest)
    assert config_bound.selected.calibration_version == actual_signal.versions.calibration_version
    assert config_bound.selected.confidence_threshold == Decimal("0.5")
    assert config_bound.selected.economic_floor == Decimal("1")
    assert audit.composite == actual_signal.versions.calibration_version
    assert audit.stable_fit_hash == manifest.stable_fit_hash
    tampered = binding.model_dump()
    tampered["template"]["confidence_threshold"] = Decimal("0.6")
    with pytest.raises(ValueError, match="policy selection evidence content hash"):
        PolicyEvidence.model_validate(tampered)
    wrong_fit = actual_signal.model_copy(
        update={
            "versions": actual_signal.versions.model_copy(
                update={"calibration_version": signal.versions.calibration_version}
            )
        }
    )
    with pytest.raises(ValueError, match="Confidence fit identity"):
        bind_trigger_policy(wrong_fit, manifest)


def test_final_fit_cutoff_may_be_after_publication_but_before_replay_as_of(tmp_path):
    dataset = bundle()
    config, period = configuration(dataset)
    config = config.model_copy(
        update={"final_fit_cutoff": dataset.cases[-2].as_of - timedelta(hours=1)}
    )
    module = BacktestModule(Fitter(tmp_path, config), StressEngine())
    module.evaluate(dataset, config, period)
    assert module.audit.locked.fitted.fit_cutoff == config.final_fit_cutoff
    assert module.audit.locked.fitted.fit_cutoff <= dataset.cases[-2].as_of


# Captured from untouched fe084fe, before the objective implementation. These
# digests cover complete records, including every nested case and artifact field.
LEGACY_CANONICAL_HASHES = (
    "81b57bd045422e52c705d3e9d247035a9d80aef95420f24d64d242c45c25d54d",
    "714a54ecd4f2866c47f811020eab3daed73803b1d68e8009c94f7dd36b54a74a",
    "eee5f496fa2ceb19d203762b83ffb318cd992e10b2a049da4038d1092ccfca51",
    "3f626f0e8adc9397eae82e3d1f9dff76dc3b8f44b4b993a1868e6c757843bca7",
    "a800cbcc5ea937a2b38bdd7010681d0eb9d37f8066668ee4a190035aebe636d4",
)
LEGACY_MODEL_JSON_HASHES = (
    "36f81fe2a90bddd02ae2643d898ad79bd7a2c01bbcc372c32acf09e25d5fd49f",
    "d426361b1ea7f230bf91c8d9a9d971b624fd8470576109bc5315c763e383c833",
    "0c7abb622da8d00525fefac3b091396e5031432fc42d40cb2c50ba04fe5020bc",
    "740e3d89931689f4842fe1842abf6ff898587c9cfea6e49cc74837d307eea3e7",
    "4b00aa82d7ff7d478f303c0cef17ce1fdfb6389afe21349ac307cdf0ed367a63",
)
LEGACY_COMPLETE_RECORDS_HASH = "0253a068cef4665e65f64745c32d31b6e4cfe97d838f58a5b6388918fe42f405"


def test_legacy_configuration_lock_and_artifact_bytes_are_unchanged(tmp_path, monkeypatch):
    from risk_engine.backtest.module import CutpointsPayload, DevelopmentPayload, SelectedPayload

    monkeypatch.chdir(tmp_path)
    root = Path("legacy-fit")
    root.mkdir()
    dataset = bundle()
    config, period = configuration(dataset)
    module = BacktestModule(Fitter(root, config), StressEngine())
    report = module.evaluate(dataset, config, period)
    audit = module.audit
    envelopes = cli().make_artifact_set(
        (
            SelectedPayload(artifact_type="selected-config", locked=audit.locked),
            CutpointsPayload(
                artifact_type="impact-cutpoints",
                status="synthetic-only",
                fit_hash=audit.selected_final_manifest.manifest_hash,
                training_hash=audit.selected_final_manifest.training_hash,
                identity=audit.selected_final_manifest.impact,
            ),
            DevelopmentPayload(artifact_type="development-backtest", report=report, audit=audit),
        ),
        artifact_set_id=f"development:{audit.locked.lock_hash}",
        authored_at=config.final_fit_cutoff,
        source_terms=dataset.source_terms,
        evidence_kind=dataset.evidence_kind,
        status="ready",
    )
    records = (config, audit.locked, *envelopes)
    assert tuple(content_hash(record) for record in records) == LEGACY_CANONICAL_HASHES
    assert content_hash(records) == LEGACY_COMPLETE_RECORDS_HASH
    assert tuple(sha256(record.model_dump_json().encode()).hexdigest() for record in records) == (
        LEGACY_MODEL_JSON_HASHES
    )
    for record in records:
        restored = type(record).model_validate_json(canonical_bytes(record))
        assert canonical_bytes(restored) == canonical_bytes(record)
        ordinary = type(record).model_validate_json(record.model_dump_json())
        assert ordinary.model_dump_json() == record.model_dump_json()
    assert config.objective == "event-class-macro-f1"
    for outer in audit.outer_results:
        for candidate in outer.inner_candidates:
            for scored, result in zip(
                candidate.validations, candidate.development_result.fold_results, strict=True
            ):
                assert result.score == scored.metrics.classification.macro_f1


@pytest.fixture(scope="module")
def ordinal_evidence(tmp_path_factory):
    dataset = bundle()
    config, period = configuration(dataset)
    module = BacktestModule(
        Fitter(tmp_path_factory.mktemp("ordinal-evidence"), config), StressEngine()
    )
    module.evaluate(dataset, config, period)
    return module.audit.outer_results[0].inner_candidates[0].validations[0], config, dataset


def ordinal_partition(evidence, loss):
    scored, config, dataset = evidence
    row = scored.cases[0]
    signal = row.chosen_signal.model_copy(
        update={"impact": row.chosen_signal.impact.model_copy(update={"impact_score": 1})}
    )
    outcome_payload = row.observed.model_dump(exclude={"evidence_hash"})
    ref = row.observed.reference_losses[0].model_copy(update={"loss": Decimal(loss)})
    outcome_payload["reference_losses"] = (ref,)
    outcome = seal(ObservedOutcome, outcome_payload, "evidence_hash")
    decision = TriggerPolicy(row.policy_binding.concrete_policy, as_of=row.as_of).evaluate(
        signal, row.decision.portfolio_materiality
    )
    row = backtest.CaseEvaluation.model_validate(
        row.model_copy(
            update={"chosen_signal": signal, "observed": outcome, "decision": decision}
        ).model_dump()
    )
    rows = (row,)
    return scored.model_copy(
        update={"cases": rows, "metrics": backtest._metrics(rows, config, dataset.snapshot_id)}
    )


@pytest.mark.parametrize(("loss", "expected"), (("100", 1.0), ("120", 0.0), ("105", 1 - 5 / 9)))
def test_normalized_ordinal_objective_supports_one_case_folds(ordinal_evidence, loss, expected):
    scored = ordinal_partition(ordinal_evidence, loss)
    assert scored.metrics.severity.rank_correlation is None
    assert scored.cases[0].observed.reference_losses[0].loss != (
        scored.cases[0].chosen_signal.impact.expected_reference_loss
    )
    assert (
        backtest.development_objective(scored, "normalized-ordinal-impact-accuracy-v1") == expected
    )
    assert backtest.development_objective(scored, "event-class-macro-f1") == (
        scored.metrics.classification.macro_f1
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "empty",
        "abstention",
        "absent-loss",
        "duplicate-loss",
        "basket",
        "currency",
        "predicted-currency",
        "missing-severity",
        "nan",
        "infinity",
        "negative",
        "above-nine",
        "population",
        "contradictory-mae",
        "wrong-case-fit",
        "unknown",
    ),
)
def test_ordinal_objective_rejects_partial_or_unmatched_population(ordinal_evidence, mutation):
    scored = ordinal_partition(ordinal_evidence, "105")
    row = scored.cases[0]
    if mutation == "empty":
        scored = scored.model_copy(update={"cases": ()})
    elif mutation == "abstention":
        row = row.model_copy(
            update={
                "chosen_signal": None,
                "chosen_signal_id": None,
                "returned_signal_ids": (),
                "abstention_reason": "project-authored abstention",
                "scenario": None,
                "stress": None,
                "decision": None,
                "policy_binding": None,
            }
        )
    elif mutation in {"absent-loss", "duplicate-loss", "basket", "currency"}:
        ref = row.observed.reference_losses[0]
        losses = (
            ()
            if mutation == "absent-loss"
            else (ref, ref)
            if mutation == "duplicate-loss"
            else (
                ref.model_copy(update={"basket_hash": "0" * 64})
                if mutation == "basket"
                else ref.model_copy(update={"currency": "INR"}),
            )
        )
        outcome = row.observed.model_dump(exclude={"evidence_hash"})
        outcome["reference_losses"] = losses
        outcome["reference_loss_absence_reason"] = (
            "no independent observation" if not losses else None
        )
        outcome["evidence_hash"] = content_hash(outcome)
        row = row.model_copy(update={"observed": ObservedOutcome.model_construct(**outcome)})
    elif mutation == "predicted-currency":
        signal = row.chosen_signal.model_copy(
            update={"impact": row.chosen_signal.impact.model_copy(update={"loss_currency": "INR"})}
        )
        row = row.model_copy(update={"chosen_signal": signal})
    elif mutation == "wrong-case-fit":
        row = row.model_copy(update={"fit_hash": "0" * 64})
    else:
        metrics = scored.metrics
        severity = metrics.severity
        if mutation == "missing-severity":
            severity = None
        elif mutation == "population":
            severity = severity.model_copy(
                update={"buckets": (severity.buckets[0].model_copy(update={"count": 2}),)}
            )
        elif mutation in {"nan", "infinity", "negative", "above-nine", "contradictory-mae"}:
            mae = {
                "nan": float("nan"),
                "infinity": float("inf"),
                "negative": -1.0,
                "above-nine": 10.0,
                "contradictory-mae": 4.0,
            }[mutation]
            severity = severity.model_copy(update={"ordinal_mae": mae})
        scored = scored.model_copy(
            update={"metrics": metrics.model_copy(update={"severity": severity})}
        )
    if mutation != "empty":
        scored = scored.model_copy(update={"cases": (row,)})
    with pytest.raises(ValueError):
        backtest.development_objective(
            scored,
            "unknown-objective"
            if mutation == "unknown"
            else "normalized-ordinal-impact-accuracy-v1",
        )


def test_development_configuration_accepts_ordinal_objective_without_new_fields():
    config, _ = configuration(bundle())
    payload = config.model_dump()
    payload["objective"] = "normalized-ordinal-impact-accuracy-v1"
    ordinal = DevelopmentConfiguration.model_validate(payload)
    assert ordinal.model_dump() == payload
    assert ordinal.direction == "maximize"
    assert ordinal.schema_version == "backtest-development-v1"


ORDINAL_BASKET_HASH = content_hash({"synthetic_reference_basket": "independent-ordinal-v1"})


def ordinal_bundle(*, tied_losses=False):
    payload = bundle().model_dump(exclude={"content_hash"})
    for index, outcome in enumerate(payload["outcomes"]):
        # Two independently declared Reference Basket outcomes. They are never
        # copied from a signal or inferred from its predicted reference loss.
        original = ReferenceLoss.model_validate(outcome["reference_losses"][0])
        simple_loss = original.model_copy(
            update={
                "loss": Decimal("100"),
                "derivation_reference": "project-authored constant basket outcome",
                "derivation_hash": content_hash({"basket": "simple", "case": index, "loss": "100"}),
            }
        )
        complex_amount = Decimal("100") if tied_losses else Decimal(100 + index * 10)
        complex_loss = original.model_copy(
            update={
                "basket_hash": ORDINAL_BASKET_HASH,
                "loss": complex_amount,
                "derivation_reference": "project-authored chronological basket outcome",
                "derivation_hash": content_hash(
                    {"basket": "complex", "case": index, "loss": complex_amount}
                ),
            }
        )
        outcome["reference_losses"] = (simple_loss, complex_loss)
        outcome["evidence_hash"] = content_hash(
            {k: v for k, v in outcome.items() if k != "evidence_hash"}
        )
    return seal(BacktestDataset, payload, "content_hash")


def ordinal_configuration(dataset, objective):
    config, period = configuration(dataset)
    payload = config.model_dump()
    payload["objective"] = objective
    for candidate in payload["candidates"]:
        candidate["parameters"]["class-index"] = 2
        candidate["parameters"]["reference-basket-hash"] = (
            BASKET_HASH if candidate["candidate_id"] == "simple" else ORDINAL_BASKET_HASH
        )
    payload["sensitivity_parameters"] = ("reference-basket-hash",)
    return DevelopmentConfiguration.model_validate(payload), period


def test_ordinal_objective_changes_selection_when_classification_is_tied(tmp_path):
    dataset = ordinal_bundle()
    for objective, expected_winner in (
        ("event-class-macro-f1", "simple"),
        ("normalized-ordinal-impact-accuracy-v1", "complex"),
    ):
        root = tmp_path / objective
        root.mkdir()
        config, period = ordinal_configuration(dataset, objective)
        module = BacktestModule(Fitter(root, config), StressEngine())
        report = module.evaluate(dataset, config, period)
        audit = module.audit
        assert len(audit.outer_results) == 3
        assert audit.locked.selection.selected_candidate_id == expected_winner
        assert audit.locked.fitted.candidate.candidate_id == expected_winner
        assert [c.candidate_id for c in report.candidate_configurations if c.selected] == [
            expected_winner
        ]
        assert len(report.candidate_configurations) == len(report.sensitivity_results) == 2
        assert len(audit.sensitivity_observations) == 2
        for outer in audit.outer_results:
            assert outer.selection.selected_candidate_id == expected_winner
            assert outer.selection.convention_version == "one-standard-error-v1"
            assert {c.candidate.candidate_id for c in outer.inner_candidates} == {
                "simple",
                "complex",
            }
            assert all(len(c.validations) >= 2 for c in outer.inner_candidates)
            classification = [
                tuple(v.metrics.classification.macro_f1 for v in candidate.validations)
                for candidate in outer.inner_candidates
            ]
            assert classification[0] == classification[1]
            for candidate in outer.inner_candidates:
                for validation, result in zip(
                    candidate.validations, candidate.development_result.fold_results, strict=True
                ):
                    assert candidate.development_result.objective_name == objective
                    assert candidate.development_result.objective_unit == "fraction"
                    assert result.score == backtest.development_objective(validation, objective)
                    expected_mae = 7 if candidate.candidate.candidate_id == "simple" else 2
                    assert validation.metrics.severity.ordinal_mae == expected_mae
                    assert {
                        ref.basket_hash
                        for row in validation.cases
                        for ref in row.observed.reference_losses
                    } == {
                        BASKET_HASH,
                        ORDINAL_BASKET_HASH,
                    }
        for candidate in report.candidate_configurations:
            expected_mae = 7 if candidate.candidate_id == "simple" else 2
            assert all(
                value == expected_mae
                for key, value in candidate.metrics.items()
                if key.endswith("severity.ordinal_mae")
            )
        assert report.sensitivity_results[0].metrics == report.sensitivity_results[1].metrics
        assert tuple(s.metrics for s in audit.sensitivity_observations) == tuple(
            s.metrics for s in report.sensitivity_results
        )
    # Replay again with only the independent complex-basket outcomes changed.
    # Constant outcomes tie the ordinal errors and restore the complexity winner.
    tied = ordinal_bundle(tied_losses=True)
    config, period = ordinal_configuration(tied, "normalized-ordinal-impact-accuracy-v1")
    root = tmp_path / "tied-outcomes"
    root.mkdir()
    module = BacktestModule(Fitter(root, config), StressEngine())
    module.evaluate(tied, config, period)
    assert module.audit.locked.selection.selected_candidate_id == "simple"
    assert all(
        result.score == 1 - 7 / 9
        for outer in module.audit.outer_results
        for candidate in outer.inner_candidates
        for result in candidate.development_result.fold_results
    )


def test_rehashed_audit_rejects_ordinal_score_contradicting_retained_evidence(tmp_path):
    from risk_engine.backtest import metrics

    dataset = ordinal_bundle()
    config, period = ordinal_configuration(dataset, "normalized-ordinal-impact-accuracy-v1")
    module = BacktestModule(Fitter(tmp_path, config), StressEngine())
    report = module.evaluate(dataset, config, period)
    audit = module.audit
    payload = audit.model_dump()
    outer = payload["outer_results"][0]
    candidate = outer["inner_candidates"][0]
    candidate["development_result"]["fold_results"][0]["score"] += 0.01
    rebuilt_selection = metrics.select_within_one_standard_error(
        tuple(
            metrics.CandidateDevelopmentResult.model_validate(c["development_result"])
            for c in outer["inner_candidates"]
        )
    )
    assert (
        rebuilt_selection.selected_candidate_id
        == audit.outer_results[0].selection.selected_candidate_id
    )
    outer["selection"] = rebuilt_selection.model_dump()
    # Rehash all envelope receipts to demonstrate rejection of the evidence
    # contradiction itself rather than a stale outer digest.
    development = {"artifact_type": "development-backtest", "report": report, "audit": payload}
    with pytest.raises(ValueError, match="objective contradicts"):
        envelope = cli().make_artifact_set(
            (
                backtest.SelectedPayload(artifact_type="selected-config", locked=audit.locked),
                backtest.CutpointsPayload(
                    artifact_type="impact-cutpoints",
                    status="synthetic-only",
                    fit_hash=audit.selected_final_manifest.manifest_hash,
                    training_hash=audit.selected_final_manifest.training_hash,
                    identity=audit.selected_final_manifest.impact,
                ),
                backtest.DevelopmentPayload.model_construct(**development),
            ),
            artifact_set_id=f"development:{audit.locked.lock_hash}",
            authored_at=config.final_fit_cutoff,
            source_terms=dataset.source_terms,
            evidence_kind=dataset.evidence_kind,
            status="ready",
        )
        backtest.ArtifactEnvelope.model_validate_json(canonical_bytes(envelope[2]))
