"""Resolve supplied local snapshots into a new lock; never contact a model registry.

Snapshot directories must be named by their immutable 40-character revision and
contain config.json and tokenizer.json. Hashes cover the exact bytes of those files.
Local adapters are responsible for verifying these hashes before loading a model.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from risk_engine.nlp.interfaces import ModelLock, ModelPin


def pin_snapshot(model_id: str, snapshot: Path) -> ModelPin:
    # The caller supplies an already acquired snapshot, not a mutable registry ref.
    revision = snapshot.name
    return ModelPin(
        model_id=model_id,
        revision=revision,
        tokenizer_sha256=hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentiment-id", required=True)
    parser.add_argument("--sentiment-snapshot", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--event-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = ModelLock(
        schema_version="1.0.0",
        sentiment=pin_snapshot(args.sentiment_id, args.sentiment_snapshot),
        event=pin_snapshot(args.event_id, args.event_snapshot),
    )
    # Exclusive creation preserves immutable locks, including concurrent writers.
    with args.output.open("x", encoding="utf-8") as output:
        output.write(lock.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
