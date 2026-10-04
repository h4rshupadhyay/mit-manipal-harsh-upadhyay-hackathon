"""Deterministic entity mentions resolved against an explicit local catalogue."""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

from risk_engine.domain import EntityLink


def _normalize(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Normalize matching text while retaining offsets into the original evidence."""
    characters: list[str] = []
    offsets: list[tuple[int, int]] = []
    for index, character in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", character).casefold():
            if normalized.isspace():
                if characters and characters[-1] == " ":
                    offsets[-1] = (offsets[-1][0], index + 1)
                    continue
                normalized = " "
            characters.append(normalized)
            offsets.append((index, index + 1))
    return "".join(characters), offsets


class EntityMatcher:
    """Match names, aliases, identifiers, and explicit cashtags without network I/O.

    CSV columns are entity_id, canonical_name, aliases, identifiers, and tickers;
    optional alternatives within the last three columns are pipe-separated.
    Results represent mentions in source order. Overlaps prefer the longest span.
    Confidence is a deterministic match indicator, not calibrated model Confidence.
    Unknown cashtags and colliding aliases are ambiguous and cannot auto-stress.
    Unknown organization-shaped prose (capitalized name plus organization suffix)
    is retained as unresolved evidence; it is never resolved beyond the catalogue.
    """

    def __init__(self, catalogue_path: Path) -> None:
        self._names: dict[str, str] = {}
        self._terms: dict[str, set[str]] = {}
        with catalogue_path.open(encoding="utf-8", newline="") as catalogue:
            reader = csv.DictReader(catalogue)
            columns = {"entity_id", "canonical_name", "aliases", "identifiers", "tickers"}
            if set(reader.fieldnames or ()) != columns:
                raise ValueError(
                    "catalogue requires entity_id, canonical_name, aliases, identifiers, tickers"
                )
            for row in reader:
                if any(row[column] is None for column in columns) or None in row:
                    raise ValueError("catalogue row has an inconsistent column count")
                entity_id = row["entity_id"].strip()
                canonical_name = row["canonical_name"].strip()
                if not entity_id or not canonical_name or entity_id in self._names:
                    raise ValueError(
                        "catalogue entity IDs and names must be nonempty; IDs must be unique"
                    )
                self._names[entity_id] = canonical_name
                terms = [entity_id, canonical_name]
                for column in ("aliases", "identifiers"):
                    terms.extend(term.strip() for term in row[column].split("|") if term.strip())
                terms.extend(
                    "$" + ticker.strip().removeprefix("$")
                    for ticker in row["tickers"].split("|")
                    if ticker.strip()
                )
                for term in terms:
                    normalized, _ = _normalize(term)
                    self._terms.setdefault(normalized, set()).add(entity_id)

    def match(self, text: str) -> list[EntityLink]:
        """Return catalogue candidates with verbatim source evidence for each mention."""
        normalized, offsets = _normalize(text)
        mentions: dict[tuple[int, int], set[str]] = {}
        for term, entity_ids in self._terms.items():
            pattern = re.compile(r"(?<![\w$])" + re.escape(term) + r"(?!\w)")
            for match in pattern.finditer(normalized):
                mentions.setdefault(match.span(), set()).update(entity_ids)
        for match in re.finditer(r"(?<![\w$])\$[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*(?!\w)", normalized):
            mentions.setdefault(match.span(), set())
        organization_pattern = (
            r"(?<![\w$])(?:[A-Z][A-Za-z0-9&'-]*\s+){1,5}"
            r"(?i:Corporation|Corp|Company|Inc|Limited|Ltd|Bank|Holdings|Group|LLC|PLC)(?!\w)"
        )
        for match in re.finditer(organization_pattern, text):
            indices = [
                index
                for index, (start, end) in enumerate(offsets)
                if match.start() <= start and end <= match.end()
            ]
            span = (indices[0], indices[-1] + 1)
            if any(
                candidates and start < span[1] and span[0] < end
                for (start, end), candidates in mentions.items()
            ):
                continue
            mentions.setdefault(span, set())

        links: list[EntityLink] = []
        previous_end = -1
        for (start, end), entity_ids in sorted(
            mentions.items(), key=lambda mention: (mention[0][0], -mention[0][1])
        ):
            if start < previous_end:
                continue
            previous_end = end
            evidence = text[offsets[start][0] : offsets[end - 1][1]]
            candidates = tuple(sorted(entity_ids))
            if len(candidates) == 1:
                entity_id = candidates[0]
                name = self._names[entity_id]
            else:
                term = normalized[start:end]
                entity_id = ("ambiguous:" if candidates else "unknown:") + term
                name = evidence
            links.append(
                EntityLink(
                    entity_id=entity_id,
                    canonical_name=name,
                    confidence=1.0 if len(candidates) == 1 else 0.0,
                    evidence=evidence,
                    ambiguous=len(candidates) != 1,
                    candidate_entity_ids=candidates,
                )
            )
        return links
