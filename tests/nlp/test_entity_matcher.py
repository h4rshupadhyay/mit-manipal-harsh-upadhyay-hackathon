import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError

from risk_engine.domain import EventClass, InterpretedEvent
from risk_engine.nlp.entity_matcher import EntityMatcher, _normalize

CATALOGUE = Path(__file__).parents[1] / "fixtures" / "entity-catalog.csv"


@pytest.mark.parametrize("name", ["Café", "Cafe\u0301"])
@pytest.mark.parametrize("text", ["Café", "Cafe\u0301"])
def test_canonical_unicode_names_match_with_verbatim_evidence(
    tmp_path: Path, name: str, text: str
) -> None:
    catalogue = tmp_path / "unicode.csv"
    catalogue.write_text(
        f"entity_id,canonical_name,aliases,identifiers,tickers\ne1,{name},,,\n",
        encoding="utf-8",
    )
    links = EntityMatcher(catalogue).match(text)
    assert len(links) == 1
    assert links[0].entity_id == "e1"
    assert links[0].canonical_name == name
    assert links[0].evidence == text
    assert links[0].candidate_entity_ids == ("e1",)
    assert not links[0].ambiguous


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("À\u0315", "a\u0315\u0300"),
        ("\u0300\u0315", "\u0315\u0300"),
        ("각", "\u1100\u1161\u11a8"),
        ("\u1100\u1161\u11a8", "각"),
        ("ffi", "\ufb03"),
        ("strasse", "Straße"),
    ],
)
def test_unicode_composition_and_expansion_keep_all_source_contributors(
    tmp_path: Path, name: str, text: str
) -> None:
    catalogue = tmp_path / "unicode.csv"
    catalogue.write_text(
        f"entity_id,canonical_name,aliases,identifiers,tickers\ne1,{name},,,\n",
        encoding="utf-8",
    )
    links = EntityMatcher(catalogue).match(text)
    assert [(link.entity_id, link.evidence) for link in links] == [("e1", text)]


def test_reordered_nonstarter_mentions_follow_original_source_order(tmp_path: Path) -> None:
    catalogue = tmp_path / "marks.csv"
    catalogue.write_text(
        "entity_id,canonical_name,aliases,identifiers,tickers\n"
        "e1,\u0315,,,\ne2,\u0300,,,\n",
        encoding="utf-8",
    )
    matcher = EntityMatcher(catalogue)
    links = matcher.match("\u0315\u0300")
    assert [(link.entity_id, link.evidence) for link in links] == [
        ("e1", "\u0315"),
        ("e2", "\u0300"),
    ]
    assert links == matcher.match("\u0315\u0300")


def test_unicode_repeated_overlapping_and_ambiguous_mentions_stay_deterministic(
    tmp_path: Path,
) -> None:
    catalogue = tmp_path / "overlap.csv"
    catalogue.write_text(
        "entity_id,canonical_name,aliases,identifiers,tickers\n"
        "e1,Café Group,Café,,\ne2,Cafe\u0301,,,\n",
        encoding="utf-8",
    )
    matcher = EntityMatcher(catalogue)
    text = "Cafe\u0301 Group; Café; Cafe\u0301 Group"
    links = matcher.match(text)
    assert [link.evidence for link in links] == ["Cafe\u0301 Group", "Café", "Cafe\u0301 Group"]
    assert [link.entity_id for link in (links[0], links[2])] == ["e1", "e1"]
    assert [link.ambiguous for link in links] == [False, True, False]
    assert links[1].candidate_entity_ids == ("e1", "e2")
    assert links[1].confidence == 0
    assert links == matcher.match(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Café",
        "Cafe\u0301",
        "a\u0315\u0300",
        "\u0315\u0300",
        "\u1100\u1161\u11a8",
        "\uac00\u11a8",
        "\ufb03 Straße",
        "\uff21\uff4c\uff50\uff48\uff41 Bank",
        " \t Alpha\n\u00a0Bank \r ",
        "\u00a8\u0301",
        "\u212b\u0327",
        "A\u0305\u0301",
        "A\u0301\u0305",
        "\u0130\u01f0",
    ],
)
def test_normalized_text_matches_full_string_unicode_reference(text: str) -> None:
    normalized, offsets = _normalize(text)
    reference = unicodedata.normalize("NFKC", text).casefold()
    # Split/join collapses whitespace; preserve the matcher's outer-space convention.
    expected = " ".join(reference.split())
    if reference and reference[0].isspace():
        expected = " " + expected
    if reference and reference[-1].isspace() and expected != " ":
        expected += " "
    assert normalized == expected
    assert len(offsets) == len(normalized)
    assert all(0 <= start < end <= len(text) for start, end in offsets)


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


@pytest.mark.parametrize(
    "text", ["Unlisted Fictional Corporation", "Unknown Regional Bank", "FICTIONAL INDUSTRIES LTD"]
)
def test_unknown_organization_names_remain_visible_and_block_automatic_stress(text: str) -> None:
    links = EntityMatcher(CATALOGUE).match(text)
    assert len(links) == 1
    assert links[0].evidence == text
    assert links[0].canonical_name == text
    assert links[0].entity_id.startswith("unknown:")
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
            evidence=(text,),
            eligible_for_automatic_stress=True,
        )


@pytest.mark.parametrize(
    ("text", "entity_id"),
    [
        ("Unlisted Inc\u0327", "unknown:unlisted inç"),
        ("Unlisted Bank\u0301", "unknown:unlisted banḱ"),
        ("Unlisted Ltd\u0327", "unknown:unlisted ltḑ"),
    ],
)
def test_unknown_organization_evidence_includes_composed_suffix_contributors(
    text: str, entity_id: str
) -> None:
    links = EntityMatcher(CATALOGUE).match(text)
    assert len(links) == 1
    assert links[0].evidence == text
    assert links[0].entity_id == entity_id
    assert links[0].canonical_name == text
    assert links[0].ambiguous
    assert links[0].candidate_entity_ids == ()
    assert links[0].confidence == 0


def test_unknown_prose_mentions_follow_source_order_alongside_catalogue_links() -> None:
    matcher = EntityMatcher(CATALOGUE)
    text = (
        "Unlisted Fictional Corporation failed; Alpha Bank responded; Unknown Regional Bank fell."
    )
    links = matcher.match(text)
    assert [link.evidence for link in links] == [
        "Unlisted Fictional Corporation",
        "Alpha Bank",
        "Unknown Regional Bank",
    ]
    assert [link.ambiguous for link in links] == [True, False, True]
    assert links == matcher.match(text)


def test_ordinary_capitalized_words_and_empty_text_do_not_invent_entities() -> None:
    matcher = EntityMatcher(CATALOGUE)
    assert (
        matcher.match("Markets Fell Monday. Investors Await News. Corporation profits rose.") == []
    )
    assert matcher.match("") == []


def test_organization_shape_does_not_override_a_known_name_after_capitalized_prose() -> None:
    links = EntityMatcher(CATALOGUE).match("Today Alpha Bank responded.")
    assert len(links) == 1
    assert links[0].entity_id == "bank-alpha"
    assert links[0].evidence == "Alpha Bank"
    assert not links[0].ambiguous


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
