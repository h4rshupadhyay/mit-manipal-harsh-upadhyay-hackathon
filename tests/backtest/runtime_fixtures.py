"""Project-authored contract simulations; these are not empirical observations."""

from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

from risk_engine.backtest.module import (
    LocalArtifact,
    ObservedOutcome,
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
