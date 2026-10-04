"""Offline clause-local interpretation with exact, extractive Source Item evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from typing import Protocol

from risk_engine.domain import EntityLink, InterpretedEvent, SourceItem
from risk_engine.nlp.interfaces import EventModel, SentimentModel


class Matcher(Protocol):
    def match(self, text: str) -> list[EntityLink]: ...


_BOUNDARY = re.compile(
    r";|\r?\n|(?<=[.!?])(?:\s+|$)|(?:,\s*)?\b(?:but|while|whereas|however|yet)\b\s*",
    re.IGNORECASE,
)


def _clauses(text: str) -> Iterator[tuple[int, int, str]]:
    """Conservative lexical boundaries, retaining original characters and offsets.

    Contrast markers and semicolons separate clauses; sentence punctuation remains
    evidence. Decimal points and punctuation inside identifiers are not boundaries.
    This is a deterministic heuristic, not a syntactic or coreference parser.
    """
    start = 0
    for boundary in _BOUNDARY.finditer(text):
        end = boundary.start()
        clause = text[start:end].strip()
        if clause:
            offset = start + len(text[start:end]) - len(text[start:end].lstrip())
            yield offset, offset + len(clause), clause
        start = boundary.end()
    clause = text[start:].strip()
    if clause:
        offset = start + len(text[start:]) - len(text[start:].lstrip())
        yield offset, offset + len(clause), clause


def interpret(
    item: SourceItem,
    matcher: Matcher,
    sentiment_model: SentimentModel,
    event_model: EventModel,
) -> list[InterpretedEvent]:
    """Return one event per clause in source order using only injected local ports.

    Rationale and evidence are the exact clause scored by both models. Unlinked
    clauses remain visible but cannot automatically stress. The classification
    probability is a raw Event Class output, not calibrated joint Confidence.
    Event IDs bind the versioned segmentation, Source Item identity/content hash,
    and original offsets so repeated clauses remain distinct across offline replay.
    """
    events = []
    for start, end, clause in _clauses(item.text):
        links = matcher.match(clause)
        if any(candidate.evidence not in clause for candidate in links):
            raise ValueError("entity match evidence must be extractive from the scored clause")
        links.sort(key=lambda candidate: (clause.index(candidate.evidence), candidate.entity_id))
        sentiment = sentiment_model.score(clause)
        classification = event_model.classify(clause)
        identity = json.dumps(
            ["interpret-v1", item.source_item_id, item.content_hash, start, end],
            separators=(",", ":"),
        )
        eligible = bool(links) and all(
            not candidate.ambiguous
            and not candidate.entity_id.casefold().startswith(("unknown:", "ambiguous:"))
            for candidate in links
        )
        events.append(
            InterpretedEvent(
                event_id="event:" + hashlib.sha256(identity.encode()).hexdigest(),
                source_item_id=item.source_item_id,
                entity_links=tuple(links),
                event_class=classification.event_class,
                sentiment=sentiment.score,
                classification_confidence=classification.probability,
                rationale=clause,
                evidence=(clause,),
                eligible_for_automatic_stress=eligible,
            )
        )
    return events
