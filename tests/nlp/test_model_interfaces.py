import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from risk_engine.domain import EventClass
from risk_engine.nlp.interfaces import EventModel, SentimentModel, load_model_lock


def test_sentiment_is_positive_minus_negative_probability() -> None:
    model = SentimentModel(lambda text: (math.log(6), math.log(3), 0.0))
    result = model.score("profits rise")
    assert result.score == pytest.approx(0.3)
    assert result.positive == pytest.approx(0.6)
    assert result.negative == pytest.approx(0.3)
    assert result.neutral == pytest.approx(0.1)


def test_large_logits_remain_bounded() -> None:
    result = SentimentModel(lambda text: (10000.0, -10000.0, 0.0)).score("news")
    assert result.score == 1.0
    assert result.positive == 1.0
    assert result.negative == 0.0


@pytest.mark.parametrize("logits", [(0.0,), (float("nan"), 0.0, 0.0), (float("inf"), 0.0, 0.0)])
def test_invalid_sentiment_logits_rejected(logits: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        SentimentModel(lambda text: logits).score("news")


def test_fixed_hypotheses_route_event_logits() -> None:
    expected = (
        "This text describes a geopolitical event.",
        "This text describes a macroeconomic or monetary event.",
        "This text describes a credit or default event.",
        "This text describes a regulatory or legal event.",
        "This text describes an operational or cyber event.",
        "This text describes a corporate action.",
        "This text describes a climate event or natural disaster.",
        "This text describes another or uncertain event.",
    )

    def infer(text: str, hypotheses: tuple[str, ...]) -> Sequence[float]:
        assert text == "default"
        assert hypotheses == expected
        return (0, 0, math.log(9), 0, 0, 0, 0, 0)

    result = EventModel(infer).classify("default")
    assert result.event_class == EventClass.CREDIT_DEFAULT
    assert result.probability == pytest.approx(9 / 16)
    assert sum(result.probabilities) == pytest.approx(1)
    assert all(0 <= value <= 1 for value in result.probabilities)


def test_event_ties_use_taxonomy_order() -> None:
    result = EventModel(lambda text, hypotheses: (0,) * 8).classify("news")
    assert result.event_class == EventClass.GEOPOLITICAL


@pytest.mark.parametrize("logits", [(), (0,) * 7, (float("nan"),) * 8])
def test_invalid_event_logits_rejected(logits: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        EventModel(lambda text, hypotheses: logits).classify("news")


def lock_payload() -> dict[str, object]:
    model = {
        "model_id": "fixture/local",
        "revision": "a" * 40,
        "tokenizer_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "weights_sha256": "d" * 64,
    }
    return {"schema_version": "1.0.0", "sentiment": dict(model), "event": dict(model)}


def test_load_immutable_lock(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock_payload()))
    lock = load_model_lock(path)
    assert lock.sentiment.revision == "a" * 40
    with pytest.raises(ValidationError):
        lock.sentiment.revision = "main"  # type: ignore[misc]  # Exercise runtime immutability.


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "main"),
        ("revision", "a" * 7),
        ("revision", ""),
        ("tokenizer_sha256", ""),
        ("config_sha256", "bad"),
        ("weights_sha256", ""),
    ],
)
def test_reject_mutable_or_incomplete_lock(tmp_path: Path, field: str, value: str) -> None:
    payload = lock_payload()
    sentiment = payload["sentiment"]
    assert isinstance(sentiment, dict)
    sentiment[field] = value
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValidationError):
        load_model_lock(path)


@pytest.mark.parametrize("field", ["tokenizer_sha256", "weights_sha256"])
def test_missing_hash_rejected(tmp_path: Path, field: str) -> None:
    payload = lock_payload()
    event = payload["event"]
    assert isinstance(event, dict)
    del event[field]
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValidationError):
        load_model_lock(path)


def test_lock_script_hashes_local_snapshots_and_refuses_overwrite(tmp_path: Path) -> None:
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    (snapshot / "config.json").write_bytes(b"{}")
    (snapshot / "tokenizer.json").write_bytes(b"{}")
    files = {
        "pytorch_model-00001-of-00002.bin": b"first bin shard",
        "pytorch_model-00002-of-00002.bin": b"second bin shard",
        "model.safetensors": b"single safetensors",
        "model-00001-of-00002.safetensors": b"first safe shard",
        "model-00002-of-00002.safetensors": b"second safe shard",
        "model.safetensors.index.json": b'{"weight_map":{}}',
        "pytorch_model.bin.index.json": b'{"weight_map":{}}',
        "nested/pytorch_model.bin": b"nested weights",
    }
    for name, content in reversed(list(files.items())):
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_bytes(content)
    output = tmp_path / "models.lock.json"
    command = [
        sys.executable,
        "scripts/lock_models.py",
        "--sentiment-id",
        "fixture/local",
        "--sentiment-snapshot",
        str(snapshot),
        "--event-id",
        "fixture/local",
        "--event-snapshot",
        str(snapshot),
        "--output",
        str(output),
    ]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    lock = load_model_lock(output)
    assert lock.sentiment.config_sha256 == (
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    )
    assert lock.sentiment.tokenizer_sha256 == lock.sentiment.config_sha256
    manifest = [
        [name, hashlib.sha256(content).hexdigest()] for name, content in sorted(files.items())
    ]
    expected = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
    assert lock.sentiment.weights_sha256 == expected
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_bootstrap_lock_fails_closed() -> None:
    with pytest.raises(ValidationError):
        load_model_lock(Path("config/models.lock.json"))
