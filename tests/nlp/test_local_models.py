"""Offline adapter contracts; synthetic snapshots never represent acquired models."""

import hashlib
import os
from pathlib import Path
from typing import cast

import pytest

from risk_engine.domain import EventClass
from risk_engine.nlp.interfaces import ModelLock, ModelPin, load_model_lock
from risk_engine.nlp.local_models import LocalModels


class FakeTokenizer:
    def encode(self, text: str, hypotheses: tuple[str, ...] | None) -> object:
        return (text, hypotheses)


class FakeModel:
    def __init__(self, event: bool, device: str, fail: bool) -> None:
        self.labels = {0: "neutral", 1: "entailment", 2: "contradiction"} if event else {
            0: "neutral", 1: "negative", 2: "positive"
        }
        self.device = device
        self.fail = fail

    def predict(self, tokens: object) -> list[list[float]]:
        if self.fail and self.device == "cuda":
            raise RuntimeError("GPU exhausted")
        _, hypotheses = cast(tuple[str, tuple[str, ...] | None], tokens)
        if hypotheses is None:
            return [[0.0, 0.0, 2.0]]
        assert len(hypotheses) == 8
        return [[100.0, 3.0 if index == 2 else 0.0, 200.0] for index in range(8)]


class FakeBackend:
    def __init__(self, fail: bool = False) -> None:
        self.active = False
        self.fail = fail
        self.loads: list[tuple[str, str]] = []

    def load(self, snapshot: Path, *, device: str, local_files_only: bool,
             trust_remote_code: bool) -> tuple[FakeTokenizer, FakeModel]:
        assert local_files_only and not trust_remote_code
        assert not self.active, "previous model must be released before loading another"
        self.active = True
        self.loads.append((snapshot.name, device))
        return FakeTokenizer(), FakeModel(snapshot.parent.name == "event", device, self.fail)

    def release(self) -> None:
        self.active = False


@pytest.fixture
def snapshots(tmp_path: Path) -> tuple[ModelLock, Path, Path]:
    pins = []
    paths = []
    for kind, model_id, revision in (
        ("sentiment", "ProsusAI/finbert", "a" * 40),
        ("event", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "b" * 40),
    ):
        path = tmp_path / kind / revision
        path.mkdir(parents=True)
        for filename in ("config.json", "tokenizer.json"):
            (path / filename).write_text("{}")
        pins.append(ModelPin(model_id=model_id, revision=revision,
                             tokenizer_sha256=hashlib.sha256(b"{}").hexdigest(),
                             config_sha256=hashlib.sha256(b"{}").hexdigest()))
        paths.append(path)
    return ModelLock(schema_version="1.0.0", sentiment=pins[0], event=pins[1]), paths[0], paths[1]


def test_reorders_sentiment_labels_and_extracts_event_entailment(
    snapshots: tuple[ModelLock, Path, Path],
) -> None:
    lock, sentiment, event = snapshots
    backend = FakeBackend()
    models = LocalModels(lock, sentiment, event, backend=backend)
    assert models.sentiment.score("Profit increased").positive == pytest.approx(0.7869860422)
    result = models.event.classify("A borrower defaults")
    assert result.event_class == EventClass.CREDIT_DEFAULT
    assert result.probability == pytest.approx(0.7415594891)
    assert models.sentiment.score("Profit increased").score > 0
    assert len(backend.loads) == 3


def test_gpu_failure_reloads_exact_snapshot_on_cpu(snapshots: tuple[ModelLock, Path, Path]) -> None:
    lock, sentiment, event = snapshots
    backend = FakeBackend(fail=True)
    models = LocalModels(lock, sentiment, event, backend=backend, device="cuda")
    assert models.sentiment.score("Profit increased").score > 0
    assert backend.loads == [("a" * 40, "cuda"), ("a" * 40, "cpu")]
    assert models.event.classify("Borrower defaults").event_class == EventClass.CREDIT_DEFAULT
    assert backend.loads[-1] == ("b" * 40, "cpu")


@pytest.mark.parametrize("filename", ["config.json", "tokenizer.json"])
def test_rejects_changed_snapshot_bytes(
    snapshots: tuple[ModelLock, Path, Path], filename: str,
) -> None:
    lock, sentiment, event = snapshots
    (sentiment / filename).write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        LocalModels(lock, sentiment, event, backend=FakeBackend())


def test_rejects_wrong_snapshot_revision(snapshots: tuple[ModelLock, Path, Path]) -> None:
    lock, sentiment, event = snapshots
    wrong = sentiment.with_name("c" * 40)
    sentiment.rename(wrong)
    with pytest.raises(ValueError, match="revision"):
        LocalModels(lock, wrong, event, backend=FakeBackend())


def test_rejects_changed_snapshot_between_calls(snapshots: tuple[ModelLock, Path, Path]) -> None:
    lock, sentiment, event = snapshots
    models = LocalModels(lock, sentiment, event, backend=FakeBackend())
    models.sentiment.score("Profit increased")
    (sentiment / "config.json").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        models.sentiment.score("Profit increased")


def test_partial_load_failure_releases_resources_without_cpu_retry(
    snapshots: tuple[ModelLock, Path, Path],
) -> None:
    class BrokenBackend(FakeBackend):
        def load(self, snapshot: Path, *, device: str, local_files_only: bool,
                 trust_remote_code: bool) -> tuple[FakeTokenizer, FakeModel]:
            result = super().load(snapshot, device=device, local_files_only=local_files_only,
                                  trust_remote_code=trust_remote_code)
            if snapshot.parent.name == "sentiment":
                raise ValueError("invalid local model configuration")
            return result

    lock, sentiment, event = snapshots
    backend = BrokenBackend()
    models = LocalModels(lock, sentiment, event, backend=backend, device="cuda")
    with pytest.raises(ValueError, match="configuration"):
        models.sentiment.score("Profit increased")
    assert not backend.active
    assert models.event.classify("Borrower defaults").event_class == EventClass.CREDIT_DEFAULT
    assert backend.loads == [("a" * 40, "cuda"), ("b" * 40, "cuda")]


@pytest.mark.model
def test_supplied_pinned_models_on_cpu() -> None:
    names = ("RISK_MODEL_LOCK", "RISK_SENTIMENT_SNAPSHOT", "RISK_EVENT_SNAPSHOT")
    if not all(os.environ.get(name) for name in names):
        pytest.skip("Explicitly acquired immutable snapshots and resolved lock are unavailable")
    lock_path, sentiment_path, event_path = (Path(os.environ[name]) for name in names)
    lock = load_model_lock(lock_path)
    models = LocalModels(lock, sentiment_path, event_path, device="cpu")
    assert models.sentiment.score("The company reported increased profits.").score > 0
    result = models.event.classify("A company defaulted on its debt payments.")
    assert result.event_class == EventClass.CREDIT_DEFAULT
    print(f"CPU revisions: {lock.sentiment.revision}, {lock.event.revision}")
