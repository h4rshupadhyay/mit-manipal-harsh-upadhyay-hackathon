from datetime import datetime, timezone
from hashlib import sha256

import pytest

from risk_engine.domain import EntityLink, EventClass, SourceItem, SourceType
from risk_engine.nlp.interfaces import EventModel, SentimentModel
from risk_engine.nlp.interpret import interpret


def source(text: str, source_item_id: str = "source-1") -> SourceItem:
    now = datetime(2026, 10, 4, tzinfo=timezone.utc)
    return SourceItem(
        source_item_id=source_item_id,
        source_type=SourceType.NEWS,
        provider="synthetic",
        text=text,
        published_at=now,
        retrieved_at=now,
        source_reference="synthetic:story-1",
        content_hash=sha256(text.encode()).hexdigest(),
        snapshot_id="synthetic-v1",
        provenance="project-authored fixture",
        license="MIT",
    )


def link(name: str, entity_id: str | None = None, ambiguous: bool = False) -> EntityLink:
    return EntityLink(
        entity_id=entity_id or name,
        canonical_name=name,
        confidence=0.0 if ambiguous else 1.0,
        evidence=name,
        ambiguous=ambiguous,
        candidate_entity_ids=() if ambiguous else (entity_id or name,),
    )


class FakeMatcher:
    def __init__(self, links: tuple[EntityLink, ...] = ()) -> None:
        self.links = links

    def match(self, text: str) -> list[EntityLink]:
        return [candidate for candidate in self.links if candidate.evidence in text]


def sentiment() -> SentimentModel:
    def infer(text: str) -> tuple[float, float, float]:
        if "rose" in text and "failed" not in text:
            return (3.0, 0.0, 0.0)
        if "failed" in text and "rose" not in text:
            return (0.0, 3.0, 0.0)
        return (0.0, 0.0, 3.0)

    return SentimentModel(infer)


def event_model(event_class: EventClass = EventClass.CREDIT_DEFAULT) -> EventModel:
    return EventModel(
        lambda text, hypotheses: [3.0 if label == event_class else 0.0 for label in EventClass]
    )


@pytest.mark.parametrize("separator", ["; ", ". ", " but ", ", while ", " whereas ", "\n"])
def test_opposing_clauses_have_local_sentiment_links_and_extractive_rationale(
    separator: str,
) -> None:
    item = source(f"Alpha Bank profits rose{separator}Beta Bank failed.")
    events = interpret(
        item, FakeMatcher((link("Beta Bank"), link("Alpha Bank"))), sentiment(), event_model()
    )
    assert len(events) == 2
    assert [event.entity_links[0].entity_id for event in events] == ["Alpha Bank", "Beta Bank"]
    assert events[0].sentiment > 0.8
    assert events[1].sentiment < -0.8
    for event in events:
        assert event.rationale in item.text
        assert event.evidence == (event.rationale,)
        assert all(candidate.evidence in event.rationale for candidate in event.entity_links)
    assert "failed" not in events[0].rationale
    assert "rose" not in events[1].rationale


@pytest.mark.parametrize("event_class", list(EventClass))
def test_fixed_event_taxonomy_is_independent_of_positive_sentiment(event_class: EventClass) -> None:
    result = interpret(
        source("Alpha Bank profits rose."),
        FakeMatcher((link("Alpha Bank"),)),
        sentiment(),
        event_model(event_class),
    )[0]
    assert result.event_class == event_class
    assert result.sentiment > 0.8
    assert result.classification_confidence == pytest.approx(0.7415594891158548)


@pytest.mark.parametrize(
    "links",
    [
        (),
        (link("Unknown Bank", "unknown:bank", True),),
        (link("Unknown Bank", "unknown:bank"),),
        (link("Unknown Bank", "ambiguous:bank", True),),
        (link("Alpha Bank"), link("Unknown Bank", "unknown:bank", True)),
    ],
)
def test_missing_unknown_and_ambiguous_links_cannot_automatically_stress(
    links: tuple[EntityLink, ...],
) -> None:
    events = interpret(
        source("Alpha Bank and Unknown Bank failed."),
        FakeMatcher(links),
        sentiment(),
        event_model(),
    )
    assert len(events) == 1
    assert not events[0].eligible_for_automatic_stress
    assert len(events[0].entity_links) == len(links)


def test_known_links_are_eligible_and_have_stable_source_order_and_unique_event_ids() -> None:
    item = source("Beta Bank and Alpha Bank failed; Beta Bank profits rose.")
    matcher = FakeMatcher((link("Alpha Bank"), link("Beta Bank")))
    first = interpret(item, matcher, sentiment(), event_model())
    second = interpret(item, matcher, sentiment(), event_model())
    assert first == second
    assert [candidate.entity_id for candidate in first[0].entity_links] == [
        "Beta Bank", "Alpha Bank"
    ]
    assert all(event.eligible_for_automatic_stress for event in first)
    assert len({event.event_id for event in first}) == 2
    assert all(event.source_item_id == "source-1" for event in first)
    other = interpret(source(item.text, "source-2"), matcher, sentiment(), event_model())
    assert {event.event_id for event in first}.isdisjoint(event.event_id for event in other)


def test_repeated_identical_clauses_have_distinct_replayable_ids() -> None:
    events = interpret(
        source("Alpha Bank failed; Alpha Bank failed"),
        FakeMatcher((link("Alpha Bank"),)), sentiment(), event_model(),
    )
    assert len(events) == 2
    assert events[0].event_id != events[1].event_id
    assert events[0].rationale == events[1].rationale == "Alpha Bank failed"


def test_whitespace_and_sentence_punctuation_preserve_exact_evidence() -> None:
    item = source("Alpha Bank profits rose 3.5%.\n\n  Beta Bank failed!")
    events = interpret(
        item, FakeMatcher((link("Alpha Bank"), link("Beta Bank"))), sentiment(), event_model()
    )
    assert [event.evidence for event in events] == [
        ("Alpha Bank profits rose 3.5%.",), ("Beta Bank failed!",)
    ]


def test_non_extractive_matcher_evidence_is_rejected() -> None:
    class InventingMatcher:
        def match(self, text: str) -> list[EntityLink]:
            return [link("Invented Bank")]

    with pytest.raises(ValueError, match="extractive"):
        interpret(source("Alpha Bank failed."), InventingMatcher(), sentiment(), event_model())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence", "Alpha"),
        ("canonical_name", "Another canonical name"),
        ("confidence", 0.5),
        ("ambiguous", True),
        ("candidate_entity_ids", ("bank-alpha", "bank-beta")),
    ],
)
def test_tied_entity_links_have_stable_order_when_matcher_results_are_reversed(
    field: str, value: object,
) -> None:
    baseline = link("Alpha Bank", "bank-alpha")
    metadata = baseline.model_dump()
    metadata[field] = value
    variant = EntityLink.model_validate(metadata)
    item = source("Alpha Bank failed.")

    first = interpret(
        item, FakeMatcher((baseline, variant)), sentiment(), event_model()
    )
    reversed_results = interpret(
        item, FakeMatcher((variant, baseline)), sentiment(), event_model()
    )

    assert len(first[0].entity_links) == 2
    assert set(first[0].entity_links) == {baseline, variant}
    assert first == reversed_results
