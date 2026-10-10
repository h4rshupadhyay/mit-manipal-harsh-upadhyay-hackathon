"""Freeze project-authored illustrative inputs; never claim historical calibration.

The calibration-directory CSVs contain hypothetical assumptions, not observed
TrainingEvidence or a BacktestDataset. No models, Risk Signals, Confidence,
Impact Scores, selected policies, market observations, or final results are
manufactured here. Canonical MarketObservation timestamps are exercise time;
the surrounding manifest records October 7 authorship explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, TypeVar

# Direct file launch places scripts/, rather than the repository, on sys.path.
# Use this script's known checkout root so the Task 32 script API is importable.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import AwareDatetime, Field, model_validator

from risk_engine.domain import (
    DomainModel,
    EventClass,
    FactorShock,
    MarketObservation,
    MarketSnapshot,
    NonEmptyString,
    Sha256Hex,
    ShockType,
    ShockUnit,
    SourceItem,
    SourceType,
    StressScenario,
)
from risk_engine.stress.interfaces import FACTOR_REGISTRY
from risk_engine.stress.shocks import normalize_scenario
from scripts.generate_synthetic_portfolio import load_artifact, load_portfolio

SNAPSHOT_ID = "project-authored-demo-20261007-v1"
GENERATOR_VERSION = "demo-snapshot-generator-v1"
ASSUMPTION_VERSION = "hypothetical-exercise-assumptions-v1"
TERMS = "MIT; project-authored fictional exercise data"
PROVIDER = "project-authored-demo"
AUTHORED_AT = datetime(2026, 10, 7, tzinfo=timezone.utc)  # noqa: UP017
EXERCISE_AS_OF = datetime(2026, 10, 4, tzinfo=timezone.utc)  # noqa: UP017
PORTFOLIO_PATH = "data/portfolio/synthetic_portfolio.csv"
PORTFOLIO_HASH = "50c7f826257539935dd606542c8d858384af404b996e41eb96a637d5a5e5aab1"
NEWS_PATH = "data/demo/news.json"
SOCIAL_PATH = "data/demo/social.json"
EVENTS_PATH = "data/calibration/events.csv"
SHOCKS_PATH = "data/calibration/factor_shocks.csv"
MANIFEST_PATH = "data/manifests/demo-snapshot.json"
OUTPUT_PATHS = (NEWS_PATH, SOCIAL_PATH, EVENTS_PATH, SHOCKS_PATH, MANIFEST_PATH)
FACTOR_ROLES = {
    "EQUITY-INDIA": "equity",
    "RATE-INR": "rate",
    "CREDIT-SPREAD": "spread",
    "USD-INR": "fx",
    "COMMODITY": "commodity",
    "VOLATILITY": "volatility",
}
# Declared together before valuation; these are illustrative judgments, not fits.
FROZEN_VECTORS = {
    "demo-rbi-policy-v1": (-2, 25, 10, 1, -1, 2),
    "demo-rbi-policy-override-v1": (-3, 50, 20, 2, -2, 3),
    "demo-indian-credit-v1": (-4, -10, 100, 2, -3, 4),
}
SCENARIO_CASES = {
    "demo-rbi-policy-v1": ("demo-event-rbi-v1", "demo-news-rbi-v1"),
    "demo-rbi-policy-override-v1": ("demo-event-rbi-v1", "demo-news-rbi-v1"),
    "demo-indian-credit-v1": ("demo-event-credit-v1", "demo-news-credit-v1"),
}
NATIVE_LEVELS = {
    "EQUITY-US": 100,
    "EQUITY-EUROPE": 100,
    "EQUITY-INDIA": 100,
    "EQUITY-ASIA": 100,
    "USD-INR": 80,
    "EUR-USD": 1.1,
    "COMMODITY": 100,
    "RATE-USD": 400,
    "RATE-EUR": 300,
    "RATE-INR": 600,
    "CREDIT-SPREAD": 150,
    "VOLATILITY": 20,
    "DEFAULT": 0,
}
MISSING_PREREQUISITES = (
    "Verified immutable local FinBERT and event-classifier model snapshots and resolved lock.",
    "Authentic historical source/label/benchmark and simultaneous six-role observation bundle.",
    "Actual historical RBI-policy and Indian credit/default case evidence.",
    "Production CandidateFitter composition and observed Impact/Confidence calibration.",
    "Chronologically selected and frozen automatic trigger policy.",
    "Untouched final evaluation interval and locked empirical BacktestReport.",
)


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _unique(values: tuple[str, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label}")


class SourceBundle(DomainModel):
    snapshot_id: Literal["project-authored-demo-20261007-v1"]
    schema_version: Literal["demo-source-bundle-v1"]
    generator_version: Literal["demo-snapshot-generator-v1"]
    authored_at: AwareDatetime
    source_terms: Literal["MIT; project-authored fictional exercise data"]
    evidence_kind: Literal["fictional"]
    source_type: SourceType
    items: Annotated[tuple[SourceItem, ...], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def validate_sources(self) -> SourceBundle:
        if self.authored_at != AUTHORED_AT:
            raise ValueError("authored_at must identify actual fixed exercise authorship")
        _unique(tuple(item.source_item_id for item in self.items), "Source Item IDs")
        for item in self.items:
            if (
                item.source_type != self.source_type
                or item.snapshot_id != self.snapshot_id
                or item.provider != PROVIDER
                or item.license != self.source_terms
                or item.published_at != self.authored_at
                or item.retrieved_at != self.authored_at
            ):
                raise ValueError("Source Item metadata disagrees with its bundle")
            if item.content_hash != _digest(item.text.encode("utf-8")):
                raise ValueError("Source Item text hash mismatch")
            if item.source_reference != "project-authored:" + item.source_item_id:
                raise ValueError("source reference must identify its Source Item")
            if (
                "fictional" not in item.text.lower()
                or "hypothetical" not in item.text.lower()
                or "project-authored" not in item.provenance
                or "not historical" not in item.provenance
            ):
                raise ValueError("Source Item requires visible fictional provenance")
        return self


class HypotheticalMetadata(DomainModel):
    snapshot_id: Literal["project-authored-demo-20261007-v1"]
    generator_version: Literal["demo-snapshot-generator-v1"]
    evidence_kind: Literal["hypothetical"]
    source_terms: Literal["MIT; project-authored fictional exercise data"]
    authored_at: AwareDatetime
    assumptions_frozen_at: AwareDatetime
    exercise_as_of: AwareDatetime
    horizon_days: Literal[1]
    derivation_reference: NonEmptyString
    assumption_version: Literal["hypothetical-exercise-assumptions-v1"]

    @model_validator(mode="after")
    def validate_exercise_metadata(self) -> HypotheticalMetadata:
        if (
            self.authored_at != AUTHORED_AT
            or self.assumptions_frozen_at != AUTHORED_AT
            or self.exercise_as_of != EXERCISE_AS_OF
        ):
            raise ValueError("exercise time must remain distinct from authored/frozen time")
        if not self.derivation_reference.startswith("scripts/prepare_demo_snapshot.py#"):
            raise ValueError("project-authored assumption derivation is required")
        return self


class HypotheticalEvent(HypotheticalMetadata):
    schema_version: Literal["demo-hypothetical-events-v1"]
    event_id: NonEmptyString
    source_item_ids: Annotated[tuple[NonEmptyString, ...], Field(min_length=2, max_length=2)]
    case_kind: Literal["hypothetical-rbi-policy", "fictional-indian-credit-stress"]
    event_class: EventClass
    region: Literal["India"]
    description: NonEmptyString

    @model_validator(mode="after")
    def validate_event(self) -> HypotheticalEvent:
        _unique(self.source_item_ids, "event source references")
        expected = (
            EventClass.MACROECONOMIC_MONETARY
            if self.case_kind == "hypothetical-rbi-policy"
            else EventClass.CREDIT_DEFAULT
        )
        if self.event_class != expected or "hypothetical" not in self.description.lower():
            raise ValueError("case description and Event Class must be explicit hypothetical data")
        kind = "rbi" if self.case_kind == "hypothetical-rbi-policy" else "credit"
        if (
            self.event_id != f"demo-event-{kind}-v1"
            or self.source_item_ids != (f"demo-news-{kind}-v1", f"demo-social-{kind}-v1")
            or self.derivation_reference != "scripts/prepare_demo_snapshot.py#" + kind
        ):
            raise ValueError("case identity, source lineage and derivation must agree")
        return self


class HypotheticalShock(HypotheticalMetadata):
    schema_version: Literal["demo-hypothetical-shocks-v1"]
    scenario_id: NonEmptyString
    event_id: NonEmptyString
    source_item_id: NonEmptyString
    factor_id: NonEmptyString
    factor_role: Literal["equity", "rate", "spread", "fx", "commodity", "volatility"]
    shock_type: ShockType
    value: float
    unit: ShockUnit
    sign: Literal["positive", "negative", "zero"]
    rationale: NonEmptyString

    def to_factor_shock(self) -> FactorShock:
        return FactorShock(
            factor_id=self.factor_id,
            shock_type=self.shock_type,
            value=self.value,
            unit=self.unit,
            horizon_days=self.horizon_days,
        )

    @model_validator(mode="after")
    def validate_role_and_sign(self) -> HypotheticalShock:
        if FACTOR_ROLES.get(self.factor_id) != self.factor_role:
            raise ValueError("factor role must match the fixed six-role definition")
        if self.sign != (
            "positive" if self.value > 0 else "negative" if self.value < 0 else "zero"
        ):
            raise ValueError("explicit sign disagrees with shock value")
        self.to_factor_shock()
        return self


class GovernedHypotheticalScenario(DomainModel):
    scenario: StressScenario
    event_id: NonEmptyString
    source_item_id: NonEmptyString
    origin_kind: Literal["manual-source-reference"]
    origin_explanation: Literal["Manual origin reference; not an emitted RiskSignal ID."]
    parent_scenario_id: NonEmptyString | None
    assumptions_frozen_at: AwareDatetime
    assumption_version: Literal["hypothetical-exercise-assumptions-v1"]
    derivation_reference: NonEmptyString
    source_terms: Literal["MIT; project-authored fictional exercise data"]

    @model_validator(mode="after")
    def validate_governance(self) -> GovernedHypotheticalScenario:
        scenario = self.scenario
        if (self.event_id, self.source_item_id) != SCENARIO_CASES.get(scenario.scenario_id):
            raise ValueError("scenario case lineage mismatch")
        if (
            scenario.method.value != "hypothetical"
            or scenario.reference_event_ids
            or scenario.calibration_version != self.assumption_version
            or scenario.risk_signal_id != "manual-exercise:" + self.source_item_id
            or self.assumptions_frozen_at != AUTHORED_AT
            or self.derivation_reference
            != "scripts/prepare_demo_snapshot.py#" + scenario.scenario_id
            or scenario.analyst_override != (self.parent_scenario_id is not None)
        ):
            raise ValueError("manual hypothetical scenario governance mismatch")
        normalized = normalize_scenario(scenario)
        if normalized.horizon_days != 1 or set(s.factor_id for s in scenario.shocks) != set(
            FACTOR_ROLES
        ):
            raise ValueError("hypothetical scenario requires all six simultaneous one-day roles")
        expected = FROZEN_VECTORS.get(scenario.scenario_id)
        shocks = {s.factor_id: s for s in scenario.shocks}
        if expected is None or tuple(shocks[f].value for f in FACTOR_ROLES) != expected:
            raise ValueError("scenario differs from this version's frozen joint assumptions")
        for factor, shock in shocks.items():
            kind, unit = _shock_convention(factor)
            if shock.shock_type != kind or shock.unit != unit:
                raise ValueError("frozen shock convention mismatch")
        return self


class ArtifactReference(DomainModel):
    path: Literal[
        "data/demo/news.json",
        "data/demo/social.json",
        "data/calibration/events.csv",
        "data/calibration/factor_shocks.csv",
        "data/portfolio/synthetic_portfolio.csv",
    ]
    sha256: Sha256Hex
    schema_version: NonEmptyString
    generator_version: NonEmptyString
    snapshot_id: NonEmptyString
    source_terms: NonEmptyString
    evidence_kind: Literal["fictional", "hypothetical", "synthetic"]
    record_count: Annotated[int, Field(ge=1)]


class PortfolioBinding(DomainModel):
    path: Literal["data/portfolio/synthetic_portfolio.csv"]
    sha256: Literal["50c7f826257539935dd606542c8d858384af404b996e41eb96a637d5a5e5aab1"]
    portfolio_id: Literal["synthetic-wholesale-global-v1"]
    portfolio_version: Literal["synthetic-portfolio-v1"]
    snapshot_id: Literal["synthetic-portfolio-20261004-v1"]
    schema_version: Literal["synthetic-portfolio-csv-v1"]
    valuation_currency: Literal["USD"]
    as_of: AwareDatetime


class DemoReadiness(DomainModel):
    illustrative_inputs_complete: Literal[True]
    local_models_available: Literal[False]
    empirical_calibration_available: Literal[False]
    selected_policy_available: Literal[False]
    final_evaluation_available: Literal[False]
    actual_historical_cases_available: Literal[False]
    full_replay_available: Literal[False]
    missing_prerequisites: Annotated[tuple[NonEmptyString, ...], Field(min_length=6)]

    @model_validator(mode="after")
    def retain_all_unavailable_prerequisites(self) -> DemoReadiness:
        if self.missing_prerequisites != MISSING_PREREQUISITES:
            raise ValueError("readiness must retain all frozen unavailable prerequisites")
        return self


class DemoManifest(DomainModel):
    snapshot_id: Literal["project-authored-demo-20261007-v1"]
    schema_version: Literal["demo-snapshot-manifest-v1"]
    generator_version: Literal["demo-snapshot-generator-v1"]
    authored_at: AwareDatetime
    source_terms: Literal["MIT; project-authored fictional exercise data"]
    evidence_kind: Literal["fictional-and-hypothetical"]
    calibration_directory_status: Literal[
        "Hypothetical assumptions only; not observed TrainingEvidence."
    ]
    signal_status: Literal[
        "Unavailable; no RiskSignals, Confidence, Impact Score, or TriggerDecision emitted."
    ]
    time_semantics: Literal[
        "October 4 is fictional valuation/exercise time; authored and frozen October 7. "
        "No historical availability claim."
    ]
    readiness: DemoReadiness
    portfolio: PortfolioBinding
    artifacts: Annotated[tuple[ArtifactReference, ...], Field(min_length=5, max_length=5)]
    market_snapshot: MarketSnapshot
    market_evidence_kind: Literal["project-authored-hypothetical-levels"]
    market_source_terms: Literal["MIT; project-authored fictional exercise data"]
    scenarios: Annotated[
        tuple[GovernedHypotheticalScenario, ...], Field(min_length=3, max_length=3)
    ]
    content_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_manifest(self) -> DemoManifest:
        if self.content_hash != _digest(
            _json_bytes(self.model_dump(mode="json", exclude={"content_hash"}))
        ):
            raise ValueError("manifest content hash mismatch")
        if self.authored_at != AUTHORED_AT or self.portfolio.as_of != EXERCISE_AS_OF:
            raise ValueError("manifest authorship/portfolio exercise time mismatch")
        if tuple(a.path for a in self.artifacts) != (*OUTPUT_PATHS[:-1], PORTFOLIO_PATH):
            raise ValueError("manifest must bind the complete, ordered input set")
        market = self.market_snapshot
        if (
            market.as_of != self.portfolio.as_of
            or market.provider != PROVIDER
            or market.snapshot_id != "project-authored-demo-market-v1"
            or market.schema_version != "market-v1"
            or market.normalization_version != "native-units-v1"
            or {o.factor_id for o in market.observations} != set(NATIVE_LEVELS)
        ):
            raise ValueError("manifest requires a complete explicit hypothetical market snapshot")
        for observation in market.observations:
            if (
                observation.unit != FACTOR_REGISTRY[observation.factor_id].native_unit
                or observation.observed_at != self.portfolio.as_of
                or observation.value != NATIVE_LEVELS[observation.factor_id]
                or observation.provider != PROVIDER
                or observation.vintage != ASSUMPTION_VERSION
            ):
                raise ValueError("hypothetical native market level metadata mismatch")
        scenarios = {g.scenario.scenario_id: g for g in self.scenarios}
        if tuple(scenarios) != tuple(FROZEN_VECTORS):
            raise ValueError("manifest scenario set/order must match the frozen version")
        for governed in self.scenarios:
            if governed.parent_scenario_id is not None:
                parent = scenarios.get(governed.parent_scenario_id)
                if (
                    parent is None
                    or parent.scenario.analyst_override
                    or parent.event_id != governed.event_id
                    or parent.source_item_id != governed.source_item_id
                ):
                    raise ValueError("manual override must retain its original case and scenario")
        return self


class DemoSnapshot(DomainModel):
    manifest: DemoManifest
    news: SourceBundle
    social: SourceBundle
    events: Annotated[tuple[HypotheticalEvent, ...], Field(min_length=2, max_length=2)]
    factor_shocks: Annotated[tuple[HypotheticalShock, ...], Field(min_length=18, max_length=18)]

    @property
    def source_items(self) -> tuple[SourceItem, ...]:
        return self.news.items + self.social.items

    def artifact_bytes(self) -> dict[str, bytes]:
        return {
            NEWS_PATH: _json_bytes(self.news.model_dump(mode="json")),
            SOCIAL_PATH: _json_bytes(self.social.model_dump(mode="json")),
            EVENTS_PATH: _csv_bytes(self.events),
            SHOCKS_PATH: _csv_bytes(self.factor_shocks),
            MANIFEST_PATH: _json_bytes(self.manifest.model_dump(mode="json")),
        }

    @model_validator(mode="after")
    def validate_cross_file_metadata(self) -> DemoSnapshot:
        if (
            self.news.source_type is not SourceType.NEWS
            or self.social.source_type is not SourceType.SOCIAL
        ):
            raise ValueError("news/social source types disagree with their named files")
        sources = {item.source_item_id: item for item in self.source_items}
        if len(sources) != 4:
            raise ValueError("all Source Item IDs must be unique")
        events = {event.event_id: event for event in self.events}
        if len(events) != 2 or {e.case_kind for e in self.events} != {
            "hypothetical-rbi-policy",
            "fictional-indian-credit-stress",
        }:
            raise ValueError("both distinct hypothetical India illustrations are required")
        referenced: set[str] = set()
        for event in self.events:
            if not set(event.source_item_ids) <= sources.keys():
                raise ValueError("event references absent Source Items")
            if {sources[s].source_type for s in event.source_item_ids} != {
                SourceType.NEWS,
                SourceType.SOCIAL,
            }:
                raise ValueError("each exercise case requires both source types")
            referenced.update(event.source_item_ids)
        if referenced != sources.keys():
            raise ValueError("unreferenced Source Items")
        rows_by_scenario: dict[str, list[HypotheticalShock]] = {}
        for row in self.factor_shocks:
            rows_by_scenario.setdefault(row.scenario_id, []).append(row)
        if set(rows_by_scenario) != {g.scenario.scenario_id for g in self.manifest.scenarios}:
            raise ValueError("shock CSV and manifest scenario identifiers disagree")
        for governed in self.manifest.scenarios:
            scenario_event = events.get(governed.event_id)
            if (
                scenario_event is None
                or governed.source_item_id not in scenario_event.source_item_ids
            ):
                raise ValueError("scenario references absent/mismatched event or source")
            rows = rows_by_scenario[governed.scenario.scenario_id]
            if len(rows) != 6 or len({r.factor_id for r in rows}) != 6:
                raise ValueError("shock CSV must contain a complete unique six-role vector")
            for row in rows:
                if (
                    row.event_id != governed.event_id
                    or row.source_item_id != governed.source_item_id
                    or row.derivation_reference != governed.derivation_reference
                ):
                    raise ValueError("shock row lineage disagrees with manifest scenario")
            if tuple(r.to_factor_shock() for r in rows) != governed.scenario.shocks:
                raise ValueError("shock CSV and scenario values/units/horizons disagree")
        bodies = self.artifact_bytes()
        bundles = (self.news, self.social)
        schemas = (
            bundles[0].schema_version,
            bundles[1].schema_version,
            self.events[0].schema_version,
            self.factor_shocks[0].schema_version,
        )
        counts = (2, 2, 2, 18)
        kinds = ("fictional", "fictional", "hypothetical", "hypothetical")
        for index, ref in enumerate(self.manifest.artifacts[:-1]):
            if (
                ref.sha256 != _digest(bodies[ref.path])
                or ref.schema_version != schemas[index]
                or ref.record_count != counts[index]
                or ref.evidence_kind != kinds[index]
                or ref.generator_version != GENERATOR_VERSION
                or ref.source_terms != TERMS
                or ref.snapshot_id != SNAPSHOT_ID
            ):
                raise ValueError("artifact hash or cross-file metadata mismatch")
        portfolio_ref = self.manifest.artifacts[-1]
        if (
            portfolio_ref.sha256 != self.manifest.portfolio.sha256
            or portfolio_ref.schema_version != self.manifest.portfolio.schema_version
            or portfolio_ref.snapshot_id != self.manifest.portfolio.snapshot_id
            or portfolio_ref.generator_version != "synthetic-portfolio-generator-v1"
            or portfolio_ref.source_terms != "MIT; project-authored synthetic data"
            or portfolio_ref.evidence_kind != "synthetic"
            or portfolio_ref.record_count != 60
        ):
            raise ValueError("portfolio dependency metadata mismatch")
        return self


def _shock_convention(factor: str) -> tuple[ShockType, ShockUnit]:
    if factor in {"RATE-INR", "CREDIT-SPREAD"}:
        return ShockType.BASIS_POINT, ShockUnit.BASIS_POINT
    if factor == "VOLATILITY":
        return ShockType.VOLATILITY, ShockUnit.VOLATILITY_POINT
    return ShockType.RELATIVE, ShockUnit.PERCENT


def _csv_bytes(rows: tuple[DomainModel, ...]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=list(type(rows[0]).model_fields), lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        body = row.model_dump(mode="json")
        if "source_item_ids" in body:
            body["source_item_ids"] = json.dumps(body["source_item_ids"], separators=(",", ":"))
        writer.writerow(body)
    return output.getvalue().encode("utf-8")


def _json_load(body: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(body, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON artifact") from error


_Row = TypeVar("_Row", bound=DomainModel)


def _csv_load(body: bytes, record_type: type[_Row]) -> tuple[_Row, ...]:
    try:
        reader = csv.DictReader(io.StringIO(body.decode("utf-8"), newline=""), strict=True)
        if reader.fieldnames != list(record_type.model_fields):
            raise ValueError("CSV columns must exactly match the versioned schema")
        rows: list[_Row] = []
        for cells in reader:
            if None in cells or any(value is None or value == "" for value in cells.values()):
                raise ValueError("CSV row is incomplete or has extra cells")
            parsed: dict[str, object] = dict(cells)
            parsed["horizon_days"] = _json_load(cells["horizon_days"].encode("utf-8"))
            if "source_item_ids" in parsed:
                parsed["source_item_ids"] = _json_load(cells["source_item_ids"].encode("utf-8"))
            rows.append(record_type.model_validate(parsed))
        if not rows:
            raise ValueError("CSV requires hypothetical rows")
        return tuple(rows)
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("invalid CSV artifact") from error


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"missing/unreadable snapshot input: {path}") from error


def _verify_portfolio(root: Path) -> None:
    if _digest(_read(root / PORTFOLIO_PATH)) != PORTFOLIO_HASH:
        raise ValueError("Task 32 portfolio byte hash mismatch")
    portfolio = load_portfolio(root / PORTFOLIO_PATH)
    if (
        portfolio.portfolio_id != "synthetic-wholesale-global-v1"
        or portfolio.version != "synthetic-portfolio-v1"
        or portfolio.as_of != EXERCISE_AS_OF
        or portfolio.valuation_currency != "USD"
    ):
        raise ValueError("Task 32 portfolio identity/version mismatch")


def build_demo_snapshot(root: Path) -> DemoSnapshot:
    """Build exact authored records without writing or valuing anything."""
    _verify_portfolio(root)
    common = dict(
        snapshot_id=SNAPSHOT_ID,
        generator_version=GENERATOR_VERSION,
        authored_at=AUTHORED_AT,
        source_terms=TERMS,
    )
    source_groups: dict[SourceType, list[SourceItem]] = {SourceType.NEWS: [], SourceType.SOCIAL: []}
    case_texts = (
        (
            "rbi",
            "Fictional hypothetical RBI-policy exercise: assume tighter Indian rates and joint "
            "cross-asset stress. No actual RBI decision or observed market response is reported.",
        ),
        (
            "credit",
            "Fictional hypothetical Indian credit-stress exercise: an invented Indian "
            "infrastructure obligor faces financing strain. No real institution default "
            "or observed market response is reported.",
        ),
    )
    for kind, text in case_texts:
        for source_type in SourceType:
            item_id = f"demo-{source_type.value}-{kind}-v1"
            text_for_source = (
                "Project-authored news-style illustration. "
                if source_type is SourceType.NEWS
                else "Project-authored fictional social post. "
            ) + text
            source_groups[source_type].append(
                SourceItem(
                    source_item_id=item_id,
                    source_type=source_type,
                    provider=PROVIDER,
                    text=text_for_source,
                    published_at=AUTHORED_AT,
                    retrieved_at=AUTHORED_AT,
                    source_reference="project-authored:" + item_id,
                    content_hash=_digest(text_for_source.encode("utf-8")),
                    snapshot_id=SNAPSHOT_ID,
                    provenance=(
                        "project-authored fictional illustration; not historical availability; "
                        "authored October 7 for an October 4 valuation exercise"
                    ),
                    license=TERMS,
                )
            )
    news = SourceBundle(
        **common,
        schema_version="demo-source-bundle-v1",
        evidence_kind="fictional",
        source_type=SourceType.NEWS,
        items=tuple(source_groups[SourceType.NEWS]),
    )
    social = SourceBundle(
        **common,
        schema_version="demo-source-bundle-v1",
        evidence_kind="fictional",
        source_type=SourceType.SOCIAL,
        items=tuple(source_groups[SourceType.SOCIAL]),
    )
    metadata = dict(
        **common,
        evidence_kind="hypothetical",
        assumptions_frozen_at=AUTHORED_AT,
        exercise_as_of=EXERCISE_AS_OF,
        horizon_days=1,
        assumption_version=ASSUMPTION_VERSION,
    )
    events = tuple(
        HypotheticalEvent(
            **metadata,
            schema_version="demo-hypothetical-events-v1",
            event_id=f"demo-event-{kind}-v1",
            source_item_ids=(f"demo-news-{kind}-v1", f"demo-social-{kind}-v1"),
            case_kind="hypothetical-rbi-policy"
            if kind == "rbi"
            else "fictional-indian-credit-stress",
            event_class=EventClass.MACROECONOMIC_MONETARY
            if kind == "rbi"
            else EventClass.CREDIT_DEFAULT,
            region="India",
            description=text,
            derivation_reference="scripts/prepare_demo_snapshot.py#" + kind,
        )
        for kind, text in case_texts
    )
    governed_scenarios: list[GovernedHypotheticalScenario] = []
    rows: list[HypotheticalShock] = []
    for scenario_id, values in FROZEN_VECTORS.items():
        kind = "credit" if "credit" in scenario_id else "rbi"
        event_id, source_id = f"demo-event-{kind}-v1", f"demo-news-{kind}-v1"
        derivation = "scripts/prepare_demo_snapshot.py#" + scenario_id
        shock_rows = tuple(
            HypotheticalShock(
                **metadata,
                schema_version="demo-hypothetical-shocks-v1",
                scenario_id=scenario_id,
                event_id=event_id,
                source_item_id=source_id,
                factor_id=factor,
                factor_role=FACTOR_ROLES[factor],
                shock_type=_shock_convention(factor)[0],
                value=value,
                unit=_shock_convention(factor)[1],
                sign="positive" if value > 0 else "negative" if value < 0 else "zero",
                rationale=(
                    "Simultaneous project-authored assumption: "
                    + {
                        "equity": "Indian equity decline",
                        "rate": "Indian rate rise"
                        if value > 0
                        else "Indian rate decline under assumed flight to quality",
                        "spread": "credit spread widening",
                        "fx": "USD-INR rise means INR depreciation",
                        "commodity": "commodity decline",
                        "volatility": "volatility increase",
                    }[FACTOR_ROLES[factor]]
                    + "; illustrative direction and magnitude, not observed or calibrated."
                ),
                derivation_reference=derivation,
            )
            for factor, value in zip(FACTOR_ROLES, values, strict=True)
        )
        rows.extend(shock_rows)
        override = "override" in scenario_id
        scenario = StressScenario(
            scenario_id=scenario_id,
            risk_signal_id="manual-exercise:" + source_id,
            shocks=tuple(row.to_factor_shock() for row in shock_rows),
            method="hypothetical",
            calibration_version=ASSUMPTION_VERSION,
            analyst_override=override,
            override_reason=(
                "Analyst-authored illustration assumes a larger simultaneous policy shock; "
                "original scenario retained."
            )
            if override
            else None,
        )
        governed_scenarios.append(
            GovernedHypotheticalScenario(
                scenario=scenario,
                event_id=event_id,
                source_item_id=source_id,
                origin_kind="manual-source-reference",
                origin_explanation="Manual origin reference; not an emitted RiskSignal ID.",
                parent_scenario_id="demo-rbi-policy-v1" if override else None,
                assumptions_frozen_at=AUTHORED_AT,
                assumption_version=ASSUMPTION_VERSION,
                derivation_reference=derivation,
                source_terms=TERMS,
            )
        )
    market = MarketSnapshot(
        snapshot_id="project-authored-demo-market-v1",
        as_of=EXERCISE_AS_OF,
        observations=tuple(
            MarketObservation(
                factor_id=factor,
                observed_at=EXERCISE_AS_OF,
                value=NATIVE_LEVELS[factor],
                unit=FACTOR_REGISTRY[factor].native_unit,
                provider=PROVIDER,
                vintage=ASSUMPTION_VERSION,
            )
            for factor in sorted(NATIVE_LEVELS)
        ),
        provider=PROVIDER,
        schema_version="market-v1",
        normalization_version="native-units-v1",
    )
    artifacts: list[ArtifactReference] = []
    for path, body, schema, count, evidence in (
        (NEWS_PATH, _json_bytes(news.model_dump(mode="json")), news.schema_version, 2, "fictional"),
        (
            SOCIAL_PATH,
            _json_bytes(social.model_dump(mode="json")),
            social.schema_version,
            2,
            "fictional",
        ),
        (EVENTS_PATH, _csv_bytes(events), events[0].schema_version, 2, "hypothetical"),
        (SHOCKS_PATH, _csv_bytes(tuple(rows)), rows[0].schema_version, 18, "hypothetical"),
    ):
        artifacts.append(
            ArtifactReference(
                path=path,
                sha256=_digest(body),
                schema_version=schema,
                generator_version=GENERATOR_VERSION,
                snapshot_id=SNAPSHOT_ID,
                source_terms=TERMS,
                evidence_kind=evidence,
                record_count=count,
            )
        )
    portfolio_artifact = load_artifact(root / PORTFOLIO_PATH)
    portfolio_row = portfolio_artifact.rows[0]
    artifacts.append(
        ArtifactReference(
            path=PORTFOLIO_PATH,
            sha256=PORTFOLIO_HASH,
            schema_version=portfolio_row.schema_version,
            generator_version=portfolio_row.generator_version,
            snapshot_id=portfolio_row.snapshot_id,
            source_terms=portfolio_row.source_terms,
            evidence_kind="synthetic",
            record_count=len(portfolio_artifact.rows),
        )
    )
    binding = PortfolioBinding(
        path=PORTFOLIO_PATH,
        sha256=PORTFOLIO_HASH,
        portfolio_id=portfolio_row.portfolio_id,
        portfolio_version=portfolio_row.portfolio_version,
        snapshot_id=portfolio_row.snapshot_id,
        schema_version=portfolio_row.schema_version,
        valuation_currency=portfolio_row.valuation_currency,
        as_of=portfolio_row.as_of,
    )
    manifest_body = dict(
        **common,
        schema_version="demo-snapshot-manifest-v1",
        evidence_kind="fictional-and-hypothetical",
        calibration_directory_status=(
            "Hypothetical assumptions only; not observed TrainingEvidence."
        ),
        signal_status=(
            "Unavailable; no RiskSignals, Confidence, Impact Score, or TriggerDecision emitted."
        ),
        time_semantics=(
            "October 4 is fictional valuation/exercise time; authored and frozen October 7. "
            "No historical availability claim."
        ),
        readiness=DemoReadiness(
            illustrative_inputs_complete=True,
            local_models_available=False,
            empirical_calibration_available=False,
            selected_policy_available=False,
            final_evaluation_available=False,
            actual_historical_cases_available=False,
            full_replay_available=False,
            missing_prerequisites=MISSING_PREREQUISITES,
        ).model_dump(mode="json"),
        portfolio=binding.model_dump(mode="json"),
        artifacts=[a.model_dump(mode="json") for a in artifacts],
        market_snapshot=market.model_dump(mode="json"),
        market_evidence_kind="project-authored-hypothetical-levels",
        market_source_terms=TERMS,
        scenarios=[g.model_dump(mode="json") for g in governed_scenarios],
    )
    # Normalize datetimes before hashing; exactly the same encoding is verified on load.
    manifest_body["authored_at"] = "2026-10-07T00:00:00Z"
    manifest = DemoManifest.model_validate(
        dict(manifest_body, content_hash=_digest(_json_bytes(manifest_body)))
    )
    return DemoSnapshot(
        manifest=manifest, news=news, social=social, events=events, factor_shocks=tuple(rows)
    )


def load_demo_snapshot(root: Path) -> DemoSnapshot:
    """Strictly verify every byte, identity, unit, version and cross-file lineage."""
    manifest = DemoManifest.model_validate(_json_load(_read(root / MANIFEST_PATH)))
    _verify_portfolio(root)
    raw: dict[str, bytes] = {ref.path: _read(root / ref.path) for ref in manifest.artifacts}
    for ref in manifest.artifacts:
        if _digest(raw[ref.path]) != ref.sha256:
            raise ValueError(f"artifact byte hash mismatch: {ref.path}")
    snapshot = DemoSnapshot(
        manifest=manifest,
        news=SourceBundle.model_validate(_json_load(raw[NEWS_PATH])),
        social=SourceBundle.model_validate(_json_load(raw[SOCIAL_PATH])),
        events=_csv_load(raw[EVENTS_PATH], HypotheticalEvent),
        factor_shocks=_csv_load(raw[SHOCKS_PATH], HypotheticalShock),
    )
    for path, expected in snapshot.artifact_bytes().items():
        actual = _read(root / path) if path == MANIFEST_PATH else raw[path]
        if actual != expected:
            raise ValueError(f"noncanonical artifact bytes: {path}")
    return snapshot


def prepare_demo_snapshot(root: Path) -> DemoSnapshot:
    """Exclusively publish a new complete set or verify identical existing bytes.

    All existing paths are checked before any creation. Partial existing sets
    require a fresh version/root. The manifest is created last as completion
    marker; loaders refuse interrupted sets. Concurrent writers cannot overwrite.
    """
    snapshot = build_demo_snapshot(root)
    bodies = snapshot.artifact_bytes()
    present = tuple(path for path in OUTPUT_PATHS if (root / path).exists())
    if present:
        if len(present) != len(OUTPUT_PATHS):
            raise FileExistsError("partial immutable snapshot exists; use a new version/root")
        if any(_read(root / path) != bodies[path] for path in OUTPUT_PATHS):
            raise FileExistsError("different immutable snapshot exists; use a new version/root")
        return load_demo_snapshot(root)
    for path in OUTPUT_PATHS:
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(bodies[path])
    return load_demo_snapshot(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args(argv)
    snapshot = prepare_demo_snapshot(arguments.root)
    print(_json_bytes(snapshot.manifest.readiness.model_dump(mode="json")).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
