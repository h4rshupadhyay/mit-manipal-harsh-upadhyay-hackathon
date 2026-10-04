"""Pinned local NLP inference with an injectable, strictly offline runtime boundary."""

from __future__ import annotations

import gc
import hashlib
import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from risk_engine.nlp.interfaces import (
    EventModel,
    ModelLock,
    ModelPin,
    SentimentModel,
    snapshot_weights_sha256,
)


class Tokenizer(Protocol):
    def encode(self, text: str, hypotheses: tuple[str, ...] | None) -> object: ...


class Classifier(Protocol):
    @property
    def labels(self) -> Mapping[int, str]: ...

    def predict(self, tokens: object) -> Sequence[Sequence[float]]: ...


class LocalBackend(Protocol):
    def load(
        self, snapshot: Path, *, device: str, local_files_only: bool, trust_remote_code: bool
    ) -> tuple[Tokenizer, Classifier]: ...

    def release(self) -> None: ...


def _verify_snapshot(pin: ModelPin, snapshot: Path) -> None:
    if snapshot.name != pin.revision:
        raise ValueError("local snapshot directory must match the pinned immutable revision")
    for filename, expected in (
        ("config.json", pin.config_sha256),
        ("tokenizer.json", pin.tokenizer_sha256),
    ):
        actual = hashlib.sha256((snapshot / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"local snapshot {filename} hash does not match the model lock")
    if snapshot_weights_sha256(snapshot) != pin.weights_sha256:
        raise ValueError("local snapshot model weights hash does not match the model lock")


class LocalModels:
    """Own one active model at a time; share this owner between both Task 8 ports.

    Snapshots are explicitly supplied, pre-acquired local directories. Instances
    are intended for sequential inference, not concurrent calls. GPU RuntimeError
    triggers one CPU retry of the same verified snapshot and keeps later calls on CPU.
    """

    def __init__(
        self,
        lock: ModelLock,
        sentiment_snapshot: Path,
        event_snapshot: Path,
        *,
        backend: LocalBackend | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if lock.sentiment.model_id != "ProsusAI/finbert":
            raise ValueError("unsupported sentiment model_id")
        if lock.event.model_id != "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli":
            raise ValueError("unsupported event model_id")
        self._snapshots = {
            "sentiment": (lock.sentiment, sentiment_snapshot.absolute()),
            "event": (lock.event, event_snapshot.absolute()),
        }
        for pin, path in self._snapshots.values():
            _verify_snapshot(pin, path)
        self._backend = backend if backend is not None else _HuggingFaceBackend()
        self._device: str = device
        self._active: str | None = None
        self._tokenizer: Tokenizer | None = None
        self._classifier: Classifier | None = None
        self.sentiment = SentimentModel(self._sentiment_logits)
        self.event = EventModel(self._event_logits)

    def close(self) -> None:
        """Release model resources, including after partial load failures."""
        self._active = None
        self._tokenizer = None
        self._classifier = None
        self._backend.release()

    def _predict(
        self, kind: str, text: str, hypotheses: tuple[str, ...] | None
    ) -> tuple[Sequence[Sequence[float]], Mapping[int, str]]:
        pin, snapshot = self._snapshots[kind]
        _verify_snapshot(pin, snapshot)
        try:
            return self._run(kind, snapshot, text, hypotheses)
        except RuntimeError:
            self.close()
            if self._device == "cpu":
                raise
            self._device = "cpu"
            _verify_snapshot(pin, snapshot)
            try:
                return self._run(kind, snapshot, text, hypotheses)
            except Exception:
                self.close()
                raise
        except Exception:
            self.close()
            raise

    def _run(
        self, kind: str, snapshot: Path, text: str, hypotheses: tuple[str, ...] | None
    ) -> tuple[Sequence[Sequence[float]], Mapping[int, str]]:
        if self._active != kind:
            self.close()
            self._tokenizer, self._classifier = self._backend.load(
                snapshot, device=self._device, local_files_only=True, trust_remote_code=False
            )
            self._active = kind
        assert self._tokenizer is not None and self._classifier is not None
        labels = {index: label.lower() for index, label in self._classifier.labels.items()}
        expected = (
            {"positive", "negative", "neutral"}
            if kind == "sentiment"
            else {"entailment", "neutral", "contradiction"}
        )
        if set(labels) != {0, 1, 2} or set(labels.values()) != expected:
            self.close()
            raise ValueError("model configuration must declare the expected semantic labels")
        return self._classifier.predict(self._tokenizer.encode(text, hypotheses)), labels

    def _sentiment_logits(self, text: str) -> Sequence[float]:
        rows, labels = self._predict("sentiment", text, None)
        if len(rows) != 1 or len(rows[0]) != 3:
            raise ValueError("sentiment model must return one row of three logits")
        indices = {label: index for index, label in labels.items()}
        return tuple(rows[0][indices[label]] for label in ("positive", "negative", "neutral"))

    def _event_logits(self, text: str, hypotheses: tuple[str, ...]) -> Sequence[float]:
        rows, labels = self._predict("event", text, hypotheses)
        if len(rows) != len(hypotheses) or any(len(row) != 3 for row in rows):
            raise ValueError("event model must return three NLI logits per fixed hypothesis")
        entailment = next(index for index, label in labels.items() if label == "entailment")
        return tuple(row[entailment] for row in rows)


class _HuggingFaceTokenizer:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str, hypotheses: tuple[str, ...] | None) -> object:
        if hypotheses is None:
            return self._tokenizer(text, return_tensors="pt", truncation=True)
        return self._tokenizer(
            [text] * len(hypotheses),
            list(hypotheses),
            return_tensors="pt",
            padding=True,
            truncation="only_first",
        )


class _HuggingFaceClassifier:
    def __init__(self, model: Any, torch: Any, device: str) -> None:
        self._model = model
        self._torch = torch
        self._device = device

    @property
    def labels(self) -> Mapping[int, str]:
        return {int(index): str(label) for index, label in self._model.config.id2label.items()}

    def predict(self, tokens: object) -> Sequence[Sequence[float]]:
        if not isinstance(tokens, Mapping):
            raise TypeError("tokenizer must return a tensor mapping")
        inputs = {key: value.to(self._device) for key, value in tokens.items()}
        with self._torch.inference_mode():
            rows: list[list[float]] = self._model(**inputs).logits.detach().cpu().tolist()
        return rows


class _HuggingFaceBackend:
    def __init__(self) -> None:
        self._model: Any = None
        self._torch: Any = None

    def load(
        self, snapshot: Path, *, device: str, local_files_only: bool, trust_remote_code: bool
    ) -> tuple[Tokenizer, Classifier]:
        # Import only on explicit inference; unit tests need neither dependency.
        transformers = importlib.import_module("transformers")
        self._torch = importlib.import_module("torch")
        self._torch.use_deterministic_algorithms(True)
        options = {"local_files_only": local_files_only, "trust_remote_code": trust_remote_code}
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(snapshot), **options)
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(
            str(snapshot), **options
        )
        self._model.to(device)
        self._model.eval()
        return _HuggingFaceTokenizer(tokenizer), _HuggingFaceClassifier(
            self._model, self._torch, device
        )

    def release(self) -> None:
        self._model = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
