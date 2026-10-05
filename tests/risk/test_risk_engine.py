"""End-to-end orchestration fixtures; no empirical performance claims."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256

import pytest

from risk_engine.config import PolicyConfig
from risk_engine.domain import (
    ConfidenceTarget,
    EntityLink,
    EventClass,
    ImpactEstimate,
    InterpretedEvent,
    PortfolioMateriality,
    ProvenanceMethod,
    SignalFlag,
    SourceItem,
    SourceType,
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
    ACTION_PRIORITY_METHOD,
    ActionPriorityCandidate,
    AuditedPortfolioMateriality,
    ModelIdentity,
    PortfolioMaterialityAudit,
    RiskEngine,
)
from risk_engine.risk.policy import TriggerPolicy

TIME = datetime(2026, 1, 10, tzinfo=timezone.utc)
DEFINITION = ConfidenceScoreDefinition(
    event_model_version="event-model-v1", entity_linker_version="entity-linker-v1"
)


def source(
    text: str,
    source_item_id: str = "source-1",
    *,
    published_at: datetime = TIME - timedelta(hours=2),
    retrieved_at: datetime = TIME - timedelta(hours=1),
) -> SourceItem:
    return SourceItem(
        source_item_id=source_item_id,
        source_type=SourceType.NEWS,
        provider="synthetic-provider",
        text=text,
        published_at=published_at,
        retrieved_at=retrieved_at,
        source_reference=f"synthetic:{source_item_id}",
        content_hash=sha256(text.encode()).hexdigest(),
        snapshot_id=f"snapshot:{source_item_id}",
        provenance="project-authored synthetic fixture",
        license="MIT",
    )


def link(
    evidence: str,
    entity_id: str,
    *,
    confidence: float,
    ambiguous: bool = False,
) -> EntityLink:
    return EntityLink(
        entity_id=entity_id,
        canonical_name=evidence,
        confidence=confidence,
        evidence=evidence,
        ambiguous=ambiguous,
        candidate_entity_ids=(entity_id,),
    )


class CatalogueMatcher:
    def __init__(self, links: tuple[EntityLink, ...]) -> None:
        self._links = links
        self.calls: list[str] = []

    def match(self, text: str) -> list[EntityLink]:
        self.calls.append(text)
        return [candidate for candidate in self._links if candidate.evidence in text]


def sentiment_model() -> SentimentModel:
    def infer(text: str) -> tuple[float, float, float]:
        if "rose" in text:
            return (3.0, 0.0, 0.0)
        return (0.0, 3.0, 0.0)

    return SentimentModel(infer)


def event_model() -> EventModel:
    return EventModel(
        lambda text, hypotheses: tuple(
            3.0 if event_class is EventClass.CREDIT_DEFAULT else 0.0
            for event_class in EventClass
        )
    )


def calibration_evidence() -> CalibrationEvidence:
    rows = tuple(
        CalibrationRow(
            row_id=f"row-{index}",
            source_item_id=f"calibration-source-{index}",
            event_id=f"calibration-event-{index}",
            entity_id="calibration-entity",
            event_class=EventClass.CREDIT_DEFAULT,
            content_hash=f"{index:064x}",
            source_terms="project-authored synthetic fixture",
            label_provenance="synthetic joint-correctness fixture",
            source_available_at=TIME - timedelta(days=4),
            label_available_at=TIME - timedelta(days=3),
            split="development",
            raw_score=RawConfidenceScore(
                definition=DEFINITION,
                entity_link_confidence=0.9,
                classification_confidence=0.8,
            ),
            entity_correct=index < 6,
            event_class_correct=index != 6,
        )
        for index in range(10)
    )
    return CalibrationEvidence(
        calibration_version="confidence-calibration-v1",
        evidence_kind="synthetic",
        snapshot_id="confidence-fixture-v1",
        snapshot_hash="a" * 64,
        score_definition=DEFINITION,
        development_start=TIME - timedelta(days=5),
        development_end=TIME - timedelta(days=3),
        frozen_at=TIME - timedelta(days=2),
        rows=rows,
    )


@pytest.fixture(scope="module")
def calibrator() -> ConfidenceCalibrator:
    return ConfidenceCalibrator.fit(calibration_evidence())


def impact(event: InterpretedEvent) -> ImpactEstimate:
    return ImpactEstimate(
        impact_score=8 if "failed" in event.rationale else 4,
        expected_reference_loss=Decimal("120"),
        loss_lower_bound=Decimal("100"),
        loss_upper_bound=Decimal("140"),
        loss_currency="USD",
        analogue_count=2,
        analogue_ids=("analogue-1", "analogue-2"),
        backoff_level="event-class-region",
        method=ProvenanceMethod.EMPIRICAL,
        calibration_version='{"audit":"impact","nested":{"hash":"abc"}}',
        reference_basket_version="reference-basket-v1",
    )


class RecordingImpactEstimator:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[tuple[InterpretedEvent, datetime]] = []
        self.fail_on_call = fail_on_call

    def estimate(self, event: InterpretedEvent, as_of: datetime) -> ImpactEstimate:
        self.calls.append((event, as_of))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("impact evidence unavailable")
        return impact(event)


class HypotheticalImpactEstimator(RecordingImpactEstimator):
    def estimate(self, event: InterpretedEvent, as_of: datetime) -> ImpactEstimate:
        self.calls.append((event, as_of))
        data = impact(event).model_dump(mode="python")
        data.update(
            method=ProvenanceMethod.HYPOTHETICAL,
            analogue_count=0,
            analogue_ids=(),
            backoff_level="global-hypothetical",
        )
        return ImpactEstimate.model_validate(data)


class MaterialityProvider:
    def __init__(
        self,
        percentages: dict[str, float | None],
        *,
        audit_as_of: datetime = TIME,
        portfolio_version: str = "synthetic-portfolio-v1",
    ) -> None:
        self._percentages = percentages
        self._audit_as_of = audit_as_of
        self._portfolio_version = portfolio_version
        self.calls: list[tuple[str, str, int, datetime]] = []

    def measure(
        self,
        event: InterpretedEvent,
        entity: EntityLink,
        estimate: ImpactEstimate,
        *,
        as_of: datetime,
    ) -> AuditedPortfolioMateriality | None:
        self.calls.append((event.event_id, entity.entity_id, estimate.impact_score, as_of))
        percentage = self._percentages.get(entity.entity_id)
        if percentage is None:
            return None
        return AuditedPortfolioMateriality(
            materiality=PortfolioMateriality(
                absolute_loss=Decimal(str(percentage * 1_000_000)),
                percentage_loss=percentage,
                currency="USD",
            ),
            audit=PortfolioMaterialityAudit(
                provider_version="synthetic-materiality-provider-v1",
                calculation_version="stress-result-materiality-v1",
                portfolio_id="synthetic-wholesale-portfolio",
                portfolio_version=self._portfolio_version,
                market_snapshot_id="market-snapshot-v1",
                scenario_id="scenario-credit-v1",
                scenario_version="joint-shock-v1",
                valuation_rule_version="valuation-rules-v1",
                as_of=self._audit_as_of,
            ),
        )


def build_engine(
    calibrator: ConfidenceCalibrator,
    *,
    matcher: CatalogueMatcher | None = None,
    estimator: RecordingImpactEstimator | None = None,
    materiality: MaterialityProvider | None = None,
) -> tuple[RiskEngine, CatalogueMatcher, RecordingImpactEstimator]:
    matcher = matcher or CatalogueMatcher(
        (
            link("Beta Bank", "bank:beta", confidence=0.9),
            link("Alpha Bank", "bank:alpha", confidence=0.7),
        )
    )
    estimator = estimator or RecordingImpactEstimator()
    return (
        RiskEngine(
            matcher=matcher,
            sentiment_model=sentiment_model(),
            event_model=event_model(),
            impact_estimator=estimator,
            confidence_calibrator=calibrator,
            model_identity=ModelIdentity(
                event_model_version="event-model-v1",
                sentiment_model_version="sentiment-model-v1",
                entity_linker_version="entity-linker-v1",
            ),
            schema_version="risk-signal-schema-v1",
            materiality_provider=materiality,
            action_priority=ActionPriorityCandidate(),
        ),
        matcher,
        estimator,
    )


def test_analyze_emits_ordered_evidence_backed_signal_per_pair_and_unlinked_clause(
    calibrator: ConfidenceCalibrator,
) -> None:
    materiality = MaterialityProvider({"bank:beta": 0.03, "bank:alpha": 0.01})
    engine, _, estimator = build_engine(calibrator, materiality=materiality)
    items = (
        source("Beta Bank and Alpha Bank failed."),
        source("No named institution failed.", "source-2"),
    )

    signals = engine.analyze(items, TIME)

    assert [signal.entity.entity_id for signal in signals] == [
        "bank:beta",
        "bank:alpha",
        f"unknown:{estimator.calls[1][0].event_id}",
    ]
    assert [signal.source_item_id for signal in signals] == ["source-1", "source-1", "source-2"]
    assert len(estimator.calls) == 2  # Impact is event-level, never entity/holdings dependent.
    assert all(signal.rationale in items[index // 2].text for index, signal in enumerate(signals))
    assert all(evidence in signal.rationale for signal in signals for evidence in signal.evidence)
    assert signals[0].impact == signals[1].impact
    assert signals[0].confidence != signals[1].confidence
    assert signals[0].confidence_target is ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS
    assert signals[0].portfolio_materiality is not None
    assert signals[0].action_priority == pytest.approx(8 * signals[0].confidence * 0.03)
    assert signals[1].action_priority == pytest.approx(8 * signals[1].confidence * 0.01)
    assert signals[2].portfolio_materiality is None
    assert signals[2].action_priority is None
    assert SignalFlag.AMBIGUOUS_ENTITY in signals[2].flags
    assert signals[2].entity.ambiguous


def test_versions_are_canonical_complete_audit_manifests(
    calibrator: ConfidenceCalibrator,
) -> None:
    engine, _, _ = build_engine(
        calibrator, materiality=MaterialityProvider({"bank:beta": 0.02})
    )
    item = source("Beta Bank failed.")
    signal = engine.analyze((item,), TIME)[0]

    model_manifest = json.loads(signal.versions.model_version)
    assert model_manifest == {
        "action_priority": {
            "method": ACTION_PRIORITY_METHOD,
            "selection_basis": "requirement-driven",
            "validation_status": "unvalidated",
        },
        "manifest_version": "risk-engine-model-manifest-v1",
        "models": {
            "entity_linker_version": "entity-linker-v1",
            "event_model_version": "event-model-v1",
            "sentiment_model_version": "sentiment-model-v1",
        },
        "portfolio_materiality": {
            "audit": {
                "as_of": "2026-01-10T00:00:00Z",
                "calculation_version": "stress-result-materiality-v1",
                "market_snapshot_id": "market-snapshot-v1",
                "portfolio_id": "synthetic-wholesale-portfolio",
                "portfolio_version": "synthetic-portfolio-v1",
                "provider_version": "synthetic-materiality-provider-v1",
                "scenario_id": "scenario-credit-v1",
                "scenario_version": "joint-shock-v1",
                "valuation_rule_version": "valuation-rules-v1",
            },
            "materiality": {
                "absolute_loss": "20000.0",
                "currency": "USD",
                "percentage_loss": 0.02,
            },
        },
    }
    calibration_manifest = json.loads(signal.versions.calibration_version)
    expected_confidence = calibrator.transform(
        RawConfidenceScore(
            definition=DEFINITION,
            entity_link_confidence=0.9,
            classification_confidence=event_model()
            .classify("Beta Bank failed.")
            .probability,
        ),
        as_of=TIME,
    )
    assert calibration_manifest["manifest_version"] == "risk-engine-calibration-manifest-v1"
    assert calibration_manifest["confidence"]["probability"] == signal.confidence
    assert {
        key: value
        for key, value in calibration_manifest["confidence"].items()
        if key != "probability"
    } == {
        key: value
        for key, value in expected_confidence.model_dump(mode="json").items()
        if key != "probability"
    }
    assert calibration_manifest["impact"] == {
        "calibration_version": '{"audit":"impact","nested":{"hash":"abc"}}',
        "reference_basket_version": "reference-basket-v1",
    }
    assert calibration_manifest["confidence"]["evidence_kind"] == "synthetic"
    assert calibration_manifest["confidence"]["fitted_artifact_hash"] == (
        calibrator.fitted_artifact_hash
    )
    assert calibration_manifest["confidence_calibration"] == {
        "calibration_evidence_snapshot_hash": "a" * 64,
        "calibration_evidence_snapshot_id": "confidence-fixture-v1",
        "calibration_version": "confidence-calibration-v1",
        "development_end": "2026-01-07T00:00:00Z",
        "development_start": "2026-01-05T00:00:00Z",
        "evidence_hash": calibrator.evidence_hash,
        "evidence_kind": "synthetic",
        "fit_version": "bounded-logloss-T0.05-20-xatol1e-10-maxiter1000-v1",
        "fitted_artifact_hash": calibrator.fitted_artifact_hash,
        "frozen_at": "2026-01-08T00:00:00Z",
        "score_definition": {
            "entity_linker_version": "entity-linker-v1",
            "event_model_version": "event-model-v1",
            "version": "min-entity-class-raw-v1",
        },
        "source_terms": ["project-authored synthetic fixture"],
        "target": "entity_and_event_class_joint_correctness",
        "temperature": calibrator.temperature,
        "transform_version": "binary-temperature-logit-clip1e-12-v1",
    }
    assert json.loads(signal.versions.snapshot_version) == {
        "manifest_version": "risk-engine-snapshot-manifest-v1",
        "source_items": [
            {
                "content_hash": item.content_hash,
                "license": "MIT",
                "provider": "synthetic-provider",
                "provenance": "project-authored synthetic fixture",
                "published_at": "2026-01-09T22:00:00Z",
                "retrieved_at": "2026-01-09T23:00:00Z",
                "snapshot_id": "snapshot:source-1",
                "source_item_id": "source-1",
                "source_reference": "synthetic:source-1",
                "source_type": "news",
            }
        ],
    }
    assert signal.versions.schema_version == "risk-signal-schema-v1"
    for manifest in (
        signal.versions.model_version,
        signal.versions.calibration_version,
        signal.versions.snapshot_version,
    ):
        assert manifest == json.dumps(json.loads(manifest), sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize(
    "item",
    [
        source(
            "Beta Bank failed.",
            published_at=TIME + timedelta(seconds=1),
            retrieved_at=TIME + timedelta(seconds=2),
        ),
        source("Beta Bank failed.", retrieved_at=TIME + timedelta(seconds=1)),
    ],
)
def test_source_publication_and_retrieval_must_be_available_by_as_of(
    calibrator: ConfidenceCalibrator, item: SourceItem
) -> None:
    engine, matcher, estimator = build_engine(calibrator)

    with pytest.raises(ValueError, match="Source Item.*as_of"):
        engine.analyze((item,), TIME)

    assert matcher.calls == []
    assert estimator.calls == []


def test_as_of_must_be_aware(calibrator: ConfidenceCalibrator) -> None:
    engine, matcher, _ = build_engine(calibrator)
    with pytest.raises(ValueError, match="timezone-aware"):
        engine.analyze((source("Beta Bank failed."),), datetime(2026, 1, 10))
    assert matcher.calls == []


def test_constructor_rejects_model_identity_that_differs_from_calibrator(
    calibrator: ConfidenceCalibrator,
) -> None:
    with pytest.raises(ValueError, match="score definition"):
        RiskEngine(
            matcher=CatalogueMatcher(()),
            sentiment_model=sentiment_model(),
            event_model=event_model(),
            impact_estimator=RecordingImpactEstimator(),
            confidence_calibrator=calibrator,
            model_identity=ModelIdentity(
                event_model_version="different-event-model",
                sentiment_model_version="sentiment-model-v1",
                entity_linker_version="entity-linker-v1",
            ),
            schema_version="risk-signal-schema-v1",
            materiality_provider=None,
            action_priority=ActionPriorityCandidate(),
        )


def test_replay_preserves_input_clause_entity_order_and_stable_ids(
    calibrator: ConfidenceCalibrator,
) -> None:
    engine, _, _ = build_engine(calibrator)
    first_item = source("Beta Bank and Alpha Bank failed; Beta Bank rose.")
    second_item = source("Alpha Bank failed.", "source-2")

    first = engine.analyze((first_item, second_item), TIME)
    replay = engine.analyze((first_item, second_item), TIME)
    reversed_inputs = engine.analyze((second_item, first_item), TIME)

    assert [signal.entity.entity_id for signal in first] == [
        "bank:beta",
        "bank:alpha",
        "bank:beta",
        "bank:alpha",
    ]
    assert [signal.signal_id for signal in replay] == [signal.signal_id for signal in first]
    assert [signal.signal_id for signal in reversed_inputs] == [
        first[3].signal_id,
        first[0].signal_id,
        first[1].signal_id,
        first[2].signal_id,
    ]
    assert [signal.versions for signal in replay] == [signal.versions for signal in first]


def test_ambiguous_clause_remains_visible_and_every_pair_abstains(
    calibrator: ConfidenceCalibrator,
) -> None:
    matcher = CatalogueMatcher(
        (
            link("Alpha", "bank:alpha", confidence=0.9),
            link("Bank", "ambiguous:bank", confidence=0.6, ambiguous=True),
        )
    )
    engine, _, _ = build_engine(calibrator, matcher=matcher)

    signals = engine.analyze((source("Alpha Bank failed."),), TIME)

    assert len(signals) == 2
    assert [signal.entity.entity_id for signal in signals] == ["bank:alpha", "ambiguous:bank"]
    assert all(SignalFlag.AMBIGUOUS_ENTITY in signal.flags for signal in signals)


def test_materiality_changes_only_materiality_and_action_priority(
    calibrator: ConfidenceCalibrator,
) -> None:
    low, _, _ = build_engine(calibrator, materiality=MaterialityProvider({"bank:beta": 0.01}))
    high, _, _ = build_engine(calibrator, materiality=MaterialityProvider({"bank:beta": 0.05}))
    item = source("Beta Bank failed.")

    low_signal = low.analyze((item,), TIME)[0]
    high_signal = high.analyze((item,), TIME)[0]

    assert low_signal.impact == high_signal.impact
    assert low_signal.confidence == high_signal.confidence
    assert low_signal.signal_id == high_signal.signal_id
    assert low_signal.versions.calibration_version == high_signal.versions.calibration_version
    assert low_signal.versions.snapshot_version == high_signal.versions.snapshot_version
    assert low_signal.versions.model_version != high_signal.versions.model_version
    assert low_signal.portfolio_materiality != high_signal.portfolio_materiality
    assert low_signal.action_priority != high_signal.action_priority


def test_materiality_audit_must_match_analysis_as_of(
    calibrator: ConfidenceCalibrator,
) -> None:
    materiality = MaterialityProvider(
        {"bank:beta": 0.01}, audit_as_of=TIME - timedelta(seconds=1)
    )
    engine, _, _ = build_engine(calibrator, materiality=materiality)

    with pytest.raises(ValueError, match="materiality audit as_of"):
        engine.analyze((source("Beta Bank failed."),), TIME)


def test_absent_materiality_always_means_absent_action_priority(
    calibrator: ConfidenceCalibrator,
) -> None:
    engine, _, _ = build_engine(calibrator, materiality=MaterialityProvider({}))
    signal = engine.analyze((source("Beta Bank failed."),), TIME)[0]
    assert signal.portfolio_materiality is None
    assert signal.action_priority is None


def test_hypothetical_impact_discloses_thin_history_and_provenance_flags(
    calibrator: ConfidenceCalibrator,
) -> None:
    engine, _, _ = build_engine(calibrator, estimator=HypotheticalImpactEstimator())
    signal = engine.analyze((source("Beta Bank failed."),), TIME)[0]
    assert signal.impact.method is ProvenanceMethod.HYPOTHETICAL
    assert signal.flags == (SignalFlag.THIN_HISTORY, SignalFlag.HYPOTHETICAL_SCENARIO)


def test_failure_raises_without_caching_or_returning_partial_signals(
    calibrator: ConfidenceCalibrator,
) -> None:
    estimator = RecordingImpactEstimator(fail_on_call=2)
    engine, _, _ = build_engine(calibrator, estimator=estimator)
    item = source("Beta Bank failed; Alpha Bank failed.")

    with pytest.raises(RuntimeError, match="impact evidence unavailable"):
        engine.analyze((item,), TIME)

    estimator.calls.clear()
    estimator.fail_on_call = None
    complete = engine.analyze((item,), TIME)
    assert [signal.entity.entity_id for signal in complete] == ["bank:beta", "bank:alpha"]
    assert not hasattr(engine, "signals")


def test_selected_trigger_policy_binds_exact_composite_calibration_manifest(
    calibrator: ConfidenceCalibrator,
) -> None:
    engine, _, _ = build_engine(
        calibrator, materiality=MaterialityProvider({"bank:beta": 0.02})
    )
    signal = engine.analyze((source("Beta Bank failed."),), TIME)[0]
    assert signal.portfolio_materiality is not None
    policy = TriggerPolicy(
        PolicyConfig.model_validate(
            {
                "selected": {
                    "version": "task24-compatibility-fixture-v1",
                    "validation_status": "chronologically_validated",
                    "evidence_kind": "empirical",
                    "selection_protocol": "nested_chronological_development",
                    "development_start": TIME - timedelta(days=2),
                    "development_end": TIME - timedelta(days=1),
                    "frozen_at": TIME - timedelta(hours=1),
                    "validation_evidence": "synthetic compatibility fixture only",
                    "evidence_hash": "b" * 64,
                    "snapshot_id": "policy-fixture-snapshot-v1",
                    "source_terms": "project-authored synthetic fixture",
                    "confidence_target": ConfidenceTarget.JOINT_ENTITY_AND_EVENT_CLASS,
                    "calibration_version": signal.versions.calibration_version,
                    "confidence_threshold": "0.5",
                    "economic_floor": "10000",
                    "materiality_tolerance": "0.01",
                    "currency": "USD",
                }
            }
        ),
        as_of=TIME,
    )

    decision = policy.evaluate(signal, signal.portfolio_materiality)

    confidence_binding = next(
        gate for gate in decision.gates if gate.name == "confidence_binding"
    )
    assert confidence_binding.passed
    assert decision.automatic_trigger
