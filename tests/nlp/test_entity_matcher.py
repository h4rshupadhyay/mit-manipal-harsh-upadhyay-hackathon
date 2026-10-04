from pathlib import Path

import pytest
from pydantic import ValidationError

from risk_engine.domain import EventClass, InterpretedEvent
from risk_engine.nlp.entity_matcher import EntityMatcher

CATALOGUE = Path(__file__).parents[1] / "fixtures" / "entity-catalog.csv"


@pytest.mark.parametrize(
    "text", ["ALPHA BANK", "alpha   banking", "\uff21\uff4c\uff50\uff48\uff41 Bank"]
)
def test_normalized_aliases_preserve_source_evidence(text: str) -> None:
    links = EntityMatcher(CATALOGUE).match(text)
    assert len(links) == 1
    assert links[0].entity_id == "bank-alpha"
    assert links[0].canonical_name == "Alpha Bank"
    assert links[0].evidence == text
    assert not links[0].ambiguous
    assert links[0].candidate_entity_ids == ("bank-alpha",)


@pytest.mark.parametrize("text", ["IN000ALPHA01", "$alpha", "bank-alpha"])
def test_exact_identifiers_and_cashtags(text: str) -> None:
    assert [link.entity_id for link in EntityMatcher(CATALOGUE).match(text)] == ["bank-alpha"]


@pytest.mark.parametrize(
    "text", ["Alpha Banker", "xIN000ALPHA01", "IN000ALPHA01x", "$ALPHABET", "ALPHA"]
)
def test_partial_tokens_and_bare_tickers_do_not_resolve(text: str) -> None:
    links = EntityMatcher(CATALOGUE).match(text)
    assert all(link.ambiguous for link in links)
    assert not any(link.entity_id == "bank-alpha" for link in links)


def test_source_order_and_repeated_mentions_are_deterministic() -> None:
    matcher = EntityMatcher(CATALOGUE)
    text = "$BETA followed Alpha Bank; then $ALPHA and Ora Energy."
    links = matcher.match(text)
    assert [link.entity_id for link in links] == [
        "bank-beta",
        "bank-alpha",
        "bank-alpha",
        "energy-ora",
    ]
    assert [link.evidence for link in links] == ["$BETA", "Alpha Bank", "$ALPHA", "Ora Energy"]
    assert links == matcher.match(text)


def test_ambiguous_alias_does_not_choose_a_catalogue_entity() -> None:
    link = EntityMatcher(CATALOGUE).match("Common Holdings")[0]
    assert link.ambiguous
    assert link.candidate_entity_ids == ("bank-alpha", "bank-beta")
    assert link.entity_id not in link.candidate_entity_ids
    assert link.confidence == 0
    assert link.evidence == "Common Holdings"


def test_unknown_cashtag_stays_visible_and_blocks_automatic_stress() -> None:
    links = EntityMatcher(CATALOGUE).match("$UNKNOWN halted trading.")
    assert len(links) == 1
    assert links[0].evidence == "$UNKNOWN"
    assert links[0].ambiguous
    assert links[0].candidate_entity_ids == ()
    assert links[0].confidence == 0
    with pytest.raises(ValidationError, match="ambiguous or missing entities"):
        InterpretedEvent(
            event_id="event-1",
            source_item_id="source-1",
            entity_links=tuple(links),
            event_class=EventClass.OTHER_UNCERTAIN,
            sentiment=0,
            classification_confidence=0,
            rationale="Unknown issuer",
            evidence=("$UNKNOWN",),
            eligible_for_automatic_stress=True,
        )


def test_unknown_prose_and_empty_text_do_not_invent_entities() -> None:
    matcher = EntityMatcher(CATALOGUE)
    assert matcher.match("Unlisted Fictional Corporation") == []
    assert matcher.match("") == []


def test_cashtag_sentence_punctuation_is_not_part_of_the_symbol() -> None:
    links = EntityMatcher(CATALOGUE).match("$ALPHA. $BETA, $UNKNOWN.")
    assert [link.entity_id for link in links[:2]] == ["bank-alpha", "bank-beta"]
    assert [link.evidence for link in links] == ["$ALPHA", "$BETA", "$UNKNOWN"]


@pytest.mark.parametrize(
    "rows",
    [
        "entity_id,canonical_name,aliases,identifiers,tickers\n,Missing ID,,,\n",
        "entity_id,canonical_name,aliases,identifiers,tickers\ne1,,,,\n",
        "entity_id,canonical_name,aliases,identifiers,tickers\ne1,One,,,\ne1,Two,,,\n",
        "entity_id,canonical_name\ne1,One\n",
    ],
)
def test_incomplete_or_duplicate_catalogue_entities_are_rejected(
    tmp_path: Path,
    rows: str,
) -> None:
    catalogue = tmp_path / "invalid.csv"
    catalogue.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError):
        EntityMatcher(catalogue)
