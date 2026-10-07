"""Project-authored contract simulations; these are not empirical observations."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from risk_engine.backtest.module import (
    LocalArtifact,
    ObservedOutcome,
    RealizedValuation,
    ReplayInput,
    SnapshotReference,
    TrainingPartition,
    canonical_bytes,
    content_hash,
)
from risk_engine.backtest.splits import FoldGroup
from risk_engine.config import ClusteringConfig
from risk_engine.data.clustering import cluster_stories
from risk_engine.domain import (
    AssetType,
    EventClass,
    MarketObservation,
    MarketSnapshot,
    Portfolio,
    Position,
    ShockUnit,
    SourceItem,
    SourceType,
)
from risk_engine.impact.reference_basket import BasketConstruction
from tests.impact.test_analogues import PAST, config, historical
from tests.impact.test_impact_score import bindings, market
from tests.impact.test_reference_basket import rows, spec

TERMS = ("project-authored contract simulation; no empirical quality claim",)


def artifact(path: Path, identity: str, available_at=PAST - timedelta(days=1)) -> LocalArtifact:
    return LocalArtifact(
        identity=identity,
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        available_at=available_at,
    )


def seal(model, payload: dict, hash_field: str):
    payload[hash_field] = content_hash(payload)
    return model.model_validate(payload)


def simulated_case(number: int):
    name = f"case-{number:02d}"
    text = f"Project-authored credit event {name}."
    item = SourceItem(
        source_item_id=f"source-{number:02d}",
        source_type=SourceType.NEWS,
        provider="project-authored simulation",
        text=text,
        published_at=PAST,
        retrieved_at=PAST + timedelta(hours=1),
        source_reference=f"simulation:{name}",
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        snapshot_id="snapshot",
        provenance=TERMS[0],
        license="MIT",
    )
    cluster = cluster_stories(
        (item,), ClusteringConfig(similarity_threshold=1, max_time_delta_hours=1)
    )[0]
    base_market = MarketSnapshot(
        snapshot_id=f"market-{name}",
        as_of=PAST,
        observations=(
            MarketObservation(
                factor_id="EQUITY-US",
                observed_at=PAST,
                value=100,
                unit=ShockUnit.INDEX_POINT,
                provider="project-authored simulation",
                vintage="v1",
            ),
        ),
        provider="project-authored simulation",
        schema_version="v1",
        normalization_version="v1",
    )
    portfolio = Portfolio(
        portfolio_id="simulation-portfolio",
        as_of=PAST,
        version="v1",
        valuation_currency="USD",
        positions=(
            Position(
                position_id="equity",
                asset_type=AssetType.EQUITY,
                notional=1000,
                currency="USD",
                sector="finance",
                region="US",
                obligor="Project-authored entity",
                factor_exposures={"EQUITY-US": 1},
            ),
        ),
    )
    case = ReplayInput(
        case_id=name,
        cluster=cluster,
        as_of=PAST + timedelta(hours=2),
        portfolio=portfolio,
        market=base_market,
        market_available_at=PAST,
        market_source=SnapshotReference(
            snapshot_id=base_market.snapshot_id,
            content_hash=content_hash(base_market),
            source_terms=TERMS,
        ),
    )
    outcome = seal(
        ObservedOutcome,
        dict(
            case_id=name,
            actual_event_class=EventClass.GEOPOLITICAL,
            actual_entity_id="entity:one",
            label_available_at=PAST + timedelta(hours=3),
            outcome_available_at=PAST + timedelta(hours=6),
            source_terms=TERMS,
            evidence_reference=f"simulation:{name}:outcome",
            realized_shocks=None,
            scenario_absence_reason="contract simulation",
            valuation=None,
            valuation_absence_reason="contract simulation",
            reference_losses=(),
            reference_loss_absence_reason="contract simulation",
            material_event=None,
            material_event_absence_reason="contract simulation",
        ),
        "evidence_hash",
    )
    return case, outcome, historical(cluster.cluster_id)


def training_pair():
    entries = (simulated_case(1), simulated_case(2))
    cases = tuple(row[0] for row in entries)
    outcomes = tuple(row[1] for row in entries)
    groups = tuple(
        FoldGroup(
            cluster_id=case.cluster.cluster_id,
            event_time=case.cluster.event_time,
            source_item_ids=tuple(item.source_item_id for item in case.cluster.items),
        )
        for case in cases
    )
    training = TrainingPartition(
        groups=groups,
        cases=cases,
        outcomes=outcomes,
        cutoff=PAST + timedelta(days=1),
        snapshot_id="project-authored-training-simulation",
        dataset_hash="a" * 64,
        evidence_kind="empirical",
        source_terms=TERMS,
        membership_hash=content_hash(groups),
    )
    return training, tuple(row[2] for row in entries)


def definition_payload(model_lock: LocalArtifact, catalogue: LocalArtifact) -> dict:
    basket = spec(BasketConstruction.EQUAL_NOTIONAL)
    return dict(
        schema_version="production-runtime-definition-v1",
        version="simulation-v1",
        frozen_at=PAST - timedelta(days=1),
        source_terms=TERMS,
        signal_schema_version="risk-signal-schema-v1",
        model_lock=model_lock,
        catalogue=catalogue,
        windows=({"key": "session-zero", "window": {"start": 0, "end": 0}},),
        baskets=(
            {
                "key": "basket-one",
                "spec": basket,
                "returns": rows(),
                "base_market": market(),
                "factor_bindings": bindings(),
                "available_at": basket.calibration_date,
                "source_terms": TERMS,
                "source_hashes": ("b" * 64,),
            },
        ),
        matching=({"key": "matching-one", "config": config()},),
        policy_selection=None,
    )


def write_document(path: Path, payload: dict) -> LocalArtifact:
    path.write_bytes(canonical_bytes(payload))
    return artifact(path, path.stem, available_at=PAST + timedelta(hours=6))


def evidence_payload(case: ReplayInput, outcome: ObservedOutcome, analogue) -> dict:
    item = case.cluster.items[0]
    target = dict(
        source_item_id=item.source_item_id,
        content_hash=item.content_hash,
        event_id=f"independent-clause:{case.case_id}",
        interpretation_version="interpret-v1",
        actual_entity_ids=("entity:one",),
        actual_event_class=EventClass.GEOPOLITICAL,
        label_available_at=outcome.label_available_at,
        label_provenance=TERMS[0],
        source_terms=TERMS,
    )
    target["label_hash"] = content_hash(target)
    return dict(
        schema_version="production-training-case-v1",
        case_id=case.case_id,
        cluster_id=case.cluster.cluster_id,
        outcome_hash=content_hash(outcome),
        source_items=({"source_item_id": item.source_item_id, "content_hash": item.content_hash},),
        targets=(target,),
        analogue=analogue.model_dump(mode="json"),
        source_terms=TERMS,
    )


def model_and_catalogue_artifacts(directory: Path) -> tuple[LocalArtifact, LocalArtifact]:
    lock_path = directory / "model-lock.json"
    pin = dict(
        model_id="project-authored-simulation-model",
        revision="1" * 40,
        tokenizer_sha256="2" * 64,
        config_sha256="3" * 64,
        weights_sha256="4" * 64,
    )
    lock_path.write_bytes(canonical_bytes(dict(schema_version="1.0.0", sentiment=pin, event=pin)))
    catalogue_path = directory / "catalogue.csv"
    catalogue_path.write_text(
        "entity_id,canonical_name,aliases,identifiers,tickers\n"
        "entity:one,Project-authored entity,,,\n",
        encoding="utf-8",
    )
    return artifact(lock_path, "model-lock"), artifact(catalogue_path, "catalogue")


def production_inputs(
    directory: Path,
    count: int = 3,
    texts=None,
    construction=BasketConstruction.EQUAL_NOTIONAL,
    policy_spec=None,
):
    """Independently assemble contract cases; never convert a synthetic dataset."""
    from risk_engine.backtest.module import CandidateSpec, DevelopmentConfiguration
    from risk_engine.backtest.runtime_inputs import (
        LocalModelLocations,
        RuntimeDefinition,
        RuntimeEvidenceIndex,
    )
    from risk_engine.nlp.interfaces import ModelLock, ModelPin, snapshot_weights_sha256
    from tests.impact.test_impact_score import analogue as observed_vector

    pins, locations = [], []
    for kind, model_id, revision in (
        ("sentiment", "ProsusAI/finbert", "a" * 40),
        ("event", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "b" * 40),
    ):
        location = directory / kind / revision
        location.mkdir(parents=True)
        for filename in ("config.json", "tokenizer.json"):
            (location / filename).write_bytes(b"{}")
        (location / "model.safetensors").write_bytes(b"project-authored model contract bytes")
        pins.append(
            ModelPin(
                model_id=model_id,
                revision=revision,
                tokenizer_sha256=hashlib.sha256(b"{}").hexdigest(),
                config_sha256=hashlib.sha256(b"{}").hexdigest(),
                weights_sha256=snapshot_weights_sha256(location),
            )
        )
        locations.append(location)
    lock = ModelLock(schema_version="1.0.0", sentiment=pins[0], event=pins[1])
    lock_path = directory / "lock.json"
    lock_path.write_bytes(canonical_bytes(lock))
    catalogue_path = directory / "catalogue.csv"
    catalogue_path.write_text(
        "entity_id,canonical_name,aliases,identifiers,tickers\n"
        "entity:one,Project-authored entity,,,\nentity:two,Second entity,,,\n"
    )
    payload = definition_payload(
        artifact(lock_path, "model-lock"), artifact(catalogue_path, "catalogue")
    )
    payload["baskets"][0]["spec"] = spec(construction)
    payload["policy_selection"] = policy_spec
    if policy_spec:
        payload["matching"][0]["config"] = config().model_copy(
            update={"minimum_support": 1, "nearest_neighbors": 1}
        )
    definition = seal(RuntimeDefinition, payload, "content_hash")
    cases, outcomes, descriptors = [], [], []
    for number in range(1, count + 1):
        case, outcome, _ = simulated_case(number)
        item = case.cluster.items[0]
        text = texts[number - 1] if texts else f"Project-authored entity faces conflict {number}."
        item = SourceItem.model_validate(
            {
                **item.model_dump(),
                "text": text,
                "content_hash": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
        shift = timedelta(days=7 * (number - 1)) if policy_spec else timedelta(0)
        if policy_spec:
            item = item.model_copy(
                update={
                    "published_at": item.published_at + shift,
                    "retrieved_at": item.retrieved_at + shift,
                }
            )
        cluster = cluster_stories(
            (item,), ClusteringConfig(similarity_threshold=1, max_time_delta_hours=1)
        )[0]
        case = ReplayInput.model_validate(
            {**case.model_dump(), "cluster": cluster, "as_of": case.as_of + shift}
        )
        analogue = observed_vector(
            cluster.cluster_id,
            100 - int(cluster.cluster_id.removeprefix("story-")[:8], 16) / 2**32 * 90
            if policy_spec
            else number * 10,
        )
        if policy_spec:
            # Shift the whole observed-vector contract together, including session dates.
            # These independent fixtures remain explicitly project-authored simulations.
            def shifted(value, delta=shift):
                if isinstance(value, datetime | date):
                    return value + delta
                if isinstance(value, dict):
                    return {key: shifted(entry, delta) for key, entry in value.items()}
                if isinstance(value, tuple | list):
                    return tuple(shifted(entry, delta) for entry in value)
                return value

            analogue = type(analogue).model_validate(shifted(analogue.model_dump()))
            outcome_payload = outcome.model_dump(exclude={"evidence_hash"}) | {
                "label_available_at": outcome.label_available_at + shift,
                "outcome_available_at": outcome.outcome_available_at + shift,
                "material_event": number % 2 == 1,
                "material_event_absence_reason": None,
                "valuation_absence_reason": None,
                "valuation": RealizedValuation(
                    pnl=Decimal(-number * 10),
                    currency="USD",
                    horizon_days=1,
                    comparison_scope="complete project-authored simulation portfolio",
                    baseline_market_hash=content_hash(case.market),
                    position_ids=("equity",),
                    gross_values=(("equity", Decimal("1000")),),
                ),
            }
            outcome = seal(ObservedOutcome, outcome_payload, "evidence_hash")
        payload = evidence_payload(case, outcome, analogue)
        # Independent clause identities use published segmentation offsets, not inference labels.
        import json

        from risk_engine.nlp.interpret import _clauses

        targets = []
        for start, end, _ in _clauses(text):
            target = dict(payload["targets"][0])
            target.pop("label_hash")
            identity = json.dumps(
                ["interpret-v1", item.source_item_id, item.content_hash, start, end],
                separators=(",", ":"),
            )
            target["event_id"] = "event:" + hashlib.sha256(identity.encode()).hexdigest()
            correct = number % 2 == 1 if policy_spec else number != 2
            target["actual_entity_ids"] = ("entity:one",) if correct else ()
            target["label_hash"] = content_hash(target)
            targets.append(target)
        payload["targets"] = tuple(targets)
        descriptor = write_document(directory / f"{case.case_id}.json", payload)
        descriptor = descriptor.model_copy(update={"available_at": descriptor.available_at + shift})
        descriptors.append(
            dict(case_id=case.case_id, cluster_id=cluster.cluster_id, artifact=descriptor)
        )
        cases.append(case)
        outcomes.append(outcome)
    groups = tuple(
        FoldGroup(
            cluster_id=c.cluster.cluster_id,
            event_time=c.cluster.event_time,
            source_item_ids=tuple(i.source_item_id for i in c.cluster.items),
        )
        for c in cases
    )
    training = TrainingPartition(
        groups=groups,
        cases=tuple(cases),
        outcomes=tuple(outcomes),
        cutoff=PAST + timedelta(days=7 * count + 1 if policy_spec else 1),
        snapshot_id="independent-contract-simulation",
        dataset_hash=content_hash(dict(cases=cases, outcomes=outcomes)),
        evidence_kind="empirical",
        source_terms=TERMS,
        membership_hash=content_hash(groups),
    )
    index = seal(
        RuntimeEvidenceIndex,
        dict(
            schema_version="production-runtime-evidence-index-v1",
            dataset_hash=training.dataset_hash,
            source_terms=TERMS,
            cases=tuple(descriptors),
        ),
        "content_hash",
    )
    candidate = CandidateSpec(
        candidate_id="simulation-runtime",
        version="v1",
        parameters=dict(
            runtime_definition_sha256=definition.content_hash,
            event_window="session-zero",
            confidence_fit="binary-temperature-logit-clip1e-12-v1",
            cutpoint_rule="nearest-rank-lower-ties",
            scenario_choice="nearest-median-reference-loss-event-id-v1",
            policy_mode="held-out-material-event-f1-v1" if policy_spec else "unselected",
        ),
        complexity_dimensions=("rules",),
        complexity=(1,),
        event_window_days=1,
        basket_convention="basket-one",
        matching_convention="matching-one",
    )
    configuration = DevelopmentConfiguration(
        schema_version="backtest-development-v1",
        version="simulation-v1",
        frozen_at=definition.frozen_at,
        candidates=(candidate,),
        split=dict(
            version="nested-grouped-chronological-v1",
            initial_outer_train_groups=3,
            outer_test_groups=1,
            inner_train_groups=2,
            inner_validation_groups=1,
            final_holdout_groups=1,
            embargo=timedelta(days=1),
            longest_event_window=timedelta(days=1),
        ),
        objective="event-class-macro-f1",
        direction="maximize",
        sensitivity_parameters=(),
        reliability_bin_edges=(0.0, 0.5, 1.0),
        nominal_interval_coverage=0.9,
        alert_budget=3,
        alert_window=dict(start=PAST, end=training.cutoff, reporting_cutoff=training.cutoff),
        final_fit_cutoff=training.cutoff,
        reduction_rule="greatest-confidence-then-signal-id-v1",
        projection_rule="defined-finite-scalars-v1",
        final_choice_rule="last-outer-inner-winner-v1",
    )
    return (
        configuration,
        definition,
        index,
        LocalModelLocations(
            sentiment_snapshot=locations[0], event_snapshot=locations[1], device="cpu"
        ),
        candidate,
        training,
    )


class SimulationBackend:
    """Explicit offline backend; real adapters still verify every model file."""

    def __init__(self):
        self.loads = 0
        self.releases = 0

    def load(self, snapshot, *, device, local_files_only, trust_remote_code):
        assert local_files_only and not trust_remote_code
        self.loads += 1
        event = snapshot.parent.name == "event"

        class Tokenizer:
            def encode(self, text, hypotheses):
                return text, hypotheses

        class Classifier:
            labels = (
                {0: "neutral", 1: "entailment", 2: "contradiction"}
                if event
                else {0: "neutral", 1: "negative", 2: "positive"}
            )

            def predict(self, tokens):
                if tokens[1] is None:
                    return [[0.0, 2.0, 0.0]]
                return [[0.0, 3.0 if i == 0 else 0.0, 0.0] for i in range(8)]

        return Tokenizer(), Classifier()

    def release(self):
        self.releases += 1


def frozen_state_payload(directory):
    from risk_engine.backtest.runtime_inputs import load_training_evidence
    from risk_engine.impact.module import FrozenImpactCalibration, TrainingEvidence
    from risk_engine.impact.reference_basket import ReferenceBasketBuilder
    from risk_engine.risk.confidence import (
        CalibrationEvidence,
        CalibrationRow,
        ConfidenceCalibrator,
        ConfidenceScoreDefinition,
        RawConfidenceScore,
    )
    from risk_engine.risk.module import ModelIdentity

    configuration, definition, index, _, candidate, training = production_inputs(directory)
    identity = ModelIdentity(
        event_model_version="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli@" + "b" * 40,
        sentiment_model_version="ProsusAI/finbert@" + "a" * 40,
        entity_linker_version="entity-matcher-v1:" + definition.catalogue.sha256,
    )
    score_definition = ConfidenceScoreDefinition(
        event_model_version=identity.event_model_version,
        entity_linker_version=identity.entity_linker_version,
    )
    latest = max(training.availability)
    frozen_at = latest + timedelta(microseconds=2)
    evidence = load_training_evidence(index, training)
    calibration_rows = tuple(
        CalibrationRow(
            row_id=f"simulation-row-{n}",
            source_item_id=c.cluster.items[0].source_item_id,
            event_id=e.targets[0].event_id,
            entity_id="entity:one",
            event_class=EventClass.GEOPOLITICAL,
            content_hash=c.cluster.items[0].content_hash,
            source_terms=TERMS[0],
            label_provenance=TERMS[0],
            source_available_at=c.cluster.items[0].retrieved_at,
            label_available_at=e.targets[0].label_available_at,
            split="development",
            raw_score=RawConfidenceScore(
                definition=score_definition,
                entity_link_confidence=1.0,
                classification_confidence=0.75,
            ),
            entity_correct=n != 1,
            event_class_correct=True,
        )
        for n, (c, e) in enumerate(zip(training.cases, evidence, strict=True))
    )
    confidence = ConfidenceCalibrator.fit(
        CalibrationEvidence(
            calibration_version="simulation-confidence-v1",
            evidence_kind="empirical",
            snapshot_id=training.snapshot_id,
            snapshot_hash=training.dataset_hash,
            score_definition=score_definition,
            development_start=PAST,
            development_end=latest + timedelta(microseconds=1),
            frozen_at=frozen_at,
            rows=calibration_rows,
        )
    )
    basket = definition.baskets[0]
    impact = FrozenImpactCalibration(
        version="simulation-impact-v1",
        calibrated_at=frozen_at,
        training_start=PAST,
        training_end=latest + timedelta(microseconds=1),
        reference_basket=ReferenceBasketBuilder().build(basket.returns, basket.spec),
        window=definition.windows[0].window,
        factor_bindings=basket.factor_bindings,
        quantile_convention="nearest-rank-lower-ties",
        uncertainty_convention="observed-min-max",
        source_terms=TERMS[0],
        training=tuple(
            TrainingEvidence(
                analogue=e.analogue, base_market=basket.base_market, partition="training"
            )
            for e in evidence
        ),
    )
    return dict(
        schema_version="production-runtime-state-v1",
        candidate=candidate,
        configuration_hash=content_hash(configuration),
        runtime_definition_hash=definition.content_hash,
        dataset_hash=training.dataset_hash,
        training_groups=training.groups,
        maximum_evidence_available_at=latest,
        fit_cutoff=training.cutoff,
        model_identity=identity,
        model_lock=definition.model_lock,
        catalogue=definition.catalogue,
        source_terms=TERMS,
        training_evidence=tuple(c.artifact for c in index.cases),
        confidence_calibrator=confidence,
        impact_calibration=impact.model_dump(mode="json"),
        matching_config=definition.matching[0].config,
        base_market=basket.base_market,
        scenario_choice=candidate.parameters["scenario_choice"],
        policy_audit=None,
        policy_absence_reason="candidate explicitly unselected",
    )


def runtime_cli_contracts(directory: Path):
    """Chronological complete CLI contracts, explicitly project-authored simulations."""
    from risk_engine.backtest.module import (
        BacktestDataset,
        DevelopmentConfiguration,
        EvaluationPeriod,
    )
    from risk_engine.backtest.runtime_inputs import RuntimeEvidenceIndex
    from tests.backtest.test_policy_selection import _spec

    configuration, definition, index, locations, candidate, training = production_inputs(
        directory, count=8, policy_spec=_spec(train_groups=2, validation_groups=1)
    )
    items = tuple(
        sorted(
            (item for case in training.cases for item in case.cluster.items),
            key=lambda item: item.source_item_id,
        )
    )
    dataset = seal(
        BacktestDataset,
        dict(
            schema_version="backtest-dataset-v1",
            snapshot_id="project-authored-cli-contracts",
            evidence_kind="empirical",
            provenance=TERMS[0],
            source_terms=TERMS,
            source_snapshots=(
                SnapshotReference(
                    snapshot_id="snapshot",
                    content_hash=content_hash(items),
                    source_terms=TERMS,
                ),
            ),
            clustering=ClusteringConfig(similarity_threshold=1, max_time_delta_hours=1),
            cases=training.cases,
            outcomes=training.outcomes,
        ),
        "content_hash",
    )
    candidate = candidate.model_copy(
        update={
            "parameters": candidate.parameters | {"policy_mode": "unselected"},
        }
    )
    cutoff = dataset.cases[-1].as_of - timedelta(hours=1)
    configuration = DevelopmentConfiguration.model_validate(
        configuration.model_dump()
        | {
            "candidates": (candidate,),
            "split": configuration.split.model_dump() | {"initial_outer_train_groups": 4},
            "final_fit_cutoff": cutoff,
        }
    )
    index = seal(
        RuntimeEvidenceIndex,
        index.model_dump(exclude={"content_hash"}) | {"dataset_hash": dataset.content_hash},
        "content_hash",
    )
    period = EvaluationPeriod(
        start=dataset.cases[4].cluster.event_time,
        end=dataset.cases[-2].as_of + timedelta(hours=1),
        reporting_cutoff=dataset.outcomes[-1].outcome_available_at,
    )
    return dataset, configuration, period, definition, index, locations
