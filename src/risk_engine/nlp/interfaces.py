"""Offline NLP ports, deterministic logit interpretation, and immutable model locks."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from risk_engine.domain import (
    DomainModel,
    EventClass,
    NonEmptyString,
    Probability,
    SentimentScore,
    Sha256Hex,
)

EVENT_HYPOTHESES = (
    "This text describes a geopolitical event.",
    "This text describes a macroeconomic or monetary event.",
    "This text describes a credit or default event.",
    "This text describes a regulatory or legal event.",
    "This text describes an operational or cyber event.",
    "This text describes a corporate action.",
    "This text describes a climate event or natural disaster.",
    "This text describes another or uncertain event.",
)


class SentimentResult(DomainModel):
    positive: Probability
    negative: Probability
    neutral: Probability
    score: SentimentScore

    @model_validator(mode="after")
    def probabilities_and_score_are_consistent(self) -> SentimentResult:
        if not math.isclose(self.positive + self.negative + self.neutral, 1.0):
            raise ValueError("sentiment probabilities must sum to one")
        if not math.isclose(self.score, self.positive - self.negative, abs_tol=1e-12):
            raise ValueError("score must equal positive minus negative probability")
        return self


class EventResult(DomainModel):
    event_class: EventClass
    probability: Probability
    probabilities: Annotated[tuple[Probability, ...], Field(min_length=8, max_length=8)]

    @model_validator(mode="after")
    def probabilities_and_selection_are_consistent(self) -> EventResult:
        if not math.isclose(sum(self.probabilities), 1.0):
            raise ValueError("event probabilities must sum to one")
        winner = max(range(len(self.probabilities)), key=self.probabilities.__getitem__)
        if self.event_class != tuple(EventClass)[winner]:
            raise ValueError("event_class must select the highest probability in taxonomy order")
        if self.probability != self.probabilities[winner]:
            raise ValueError("probability must match the selected Event Class")
        return self


def _softmax(logits: Sequence[float], count: int) -> tuple[float, ...]:
    values = tuple(logits)
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise ValueError(f"expected {count} finite logits")
    largest = max(values)
    weights = tuple(math.exp(value - largest) for value in values)
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _require_text(text: str) -> None:
    if not text.strip():
        raise ValueError("Source Item text must not be empty")


class SentimentModel:
    """Inject local inference returning positive, negative, neutral logits in that order."""

    def __init__(self, infer: Callable[[str], Sequence[float]]) -> None:
        self._infer = infer

    def score(self, text: str) -> SentimentResult:
        _require_text(text)
        positive, negative, neutral = _softmax(self._infer(text), 3)
        return SentimentResult(
            positive=positive, negative=negative, neutral=neutral, score=positive - negative
        )


class EventModel:
    """Inject local inference returning one comparable logit per fixed hypothesis.

    Probabilities are raw classification outputs, not calibrated joint Confidence.
    Local adapters must extract entailment logits before calling this port.
    """

    def __init__(self, infer: Callable[[str, tuple[str, ...]], Sequence[float]]) -> None:
        self._infer = infer

    def classify(self, text: str) -> EventResult:
        _require_text(text)
        probabilities = _softmax(self._infer(text, EVENT_HYPOTHESES), len(EventClass))
        winner = max(range(len(probabilities)), key=probabilities.__getitem__)
        return EventResult(
            event_class=tuple(EventClass)[winner],
            probability=probabilities[winner],
            probabilities=probabilities,
        )


class ModelPin(DomainModel):
    model_id: NonEmptyString
    revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    tokenizer_sha256: Sha256Hex
    config_sha256: Sha256Hex


class ModelLock(DomainModel):
    schema_version: Literal["1.0.0"]
    sentiment: ModelPin
    event: ModelPin


def load_model_lock(path: Path) -> ModelLock:
    """Validate a lock without resolving revisions, downloading, or refreshing anything."""
    return ModelLock.model_validate_json(path.read_bytes())
