"""Offline acquisition gate tests using only project-authored bytes."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from scripts import acquire_evaluation_data as acquisition
from scripts.acquire_evaluation_data import AcquisitionRequest

PAYLOAD = b"project-authored evaluation fixture; no third-party rows\n"
STOCKNET_REVISION = "a" * 40
STOCKNET_URL = f"https://codeload.github.com/yumoxu/stocknet-dataset/zip/{STOCKNET_REVISION}"
STOCKNET_TERMS = "https://github.com/yumoxu/stocknet-dataset/blob/master/LICENSE"
STAMP = datetime(2026, 10, 7, 8, tzinfo=timezone.utc)


def _metadata() -> dict[str, object]:
    return {
        "dataset_id": "stocknet",
        "source_url": STOCKNET_URL,
        "terms_url": STOCKNET_TERMS,
        "terms_version": "reviewed-2026-10-07",
        "source_version": STOCKNET_REVISION,
        "snapshot_id": "authored-fixture-v1",
        "available_at": STAMP,
        "retrieved_at": STAMP,
        "expected_sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        "terms_acknowledged": False,
    }


def test_terms_acknowledgement_is_required() -> None:
    """Removing the explicit opt-in must reject acquisition before network access."""
    with pytest.raises(ValueError, match="terms_acknowledged"):
        AcquisitionRequest.model_validate(_metadata())


@pytest.mark.parametrize("acknowledgement", [None, "true", 1, "yes"])
def test_acknowledgement_cannot_be_coerced(acknowledgement: object) -> None:
    with pytest.raises(ValueError):
        AcquisitionRequest.model_validate({**_metadata(), "terms_acknowledged": acknowledgement})


@pytest.mark.parametrize(
    "field",
    ["dataset_id", "source_url", "terms_url", "terms_version", "source_version", "snapshot_id",
     "available_at", "retrieved_at", "expected_sha256", "terms_acknowledged"],
)
def test_complete_explicit_metadata_is_required(field: str) -> None:
    metadata = {**_metadata(), "terms_acknowledged": True}
    del metadata[field]
    with pytest.raises(ValueError):
        AcquisitionRequest.model_validate(metadata)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", "other"), ("expected_sha256", ""), ("expected_sha256", "xyz"),
        ("expected_sha256", "A" * 64), ("expected_sha256", "a" * 63),
        ("terms_version", " "), ("snapshot_id", ""), ("source_version", "master"),
        ("source_url", STOCKNET_URL.replace("https:", "http:")),
        ("source_url", STOCKNET_URL.replace("codeload.github.com", "example.com")),
        ("source_url", STOCKNET_URL + "?token=secret"),
        ("source_url", STOCKNET_URL.replace(STOCKNET_REVISION, "b" * 40)),
        ("source_url", STOCKNET_URL.replace("yumoxu", "unverified")),
        ("terms_url", "https://example.com/license"),
        ("available_at", STAMP.replace(tzinfo=None)),
        ("retrieved_at", datetime(2026, 10, 6, tzinfo=timezone.utc)),
    ],
)
def test_invalid_metadata_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        AcquisitionRequest.model_validate(
            {**_metadata(), "terms_acknowledged": True, field: value}
        )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Only a disposable repository is used; no restricted acquisition occurs."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True, capture_output=True)
    (root / ".gitignore").write_text("data/local/\n", encoding="utf-8")
    return root


def _request(**overrides: object) -> AcquisitionRequest:
    return AcquisitionRequest.model_validate(
        {**_metadata(), "terms_acknowledged": True, **overrides}
    )


def _transport(payload: bytes = PAYLOAD, *, status: int = 200) -> httpx.MockTransport:
    def response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=payload, request=request)
    return httpx.MockTransport(response)


def _acquire(
    repository: Path,
    destination: Path,
    *,
    request: AcquisitionRequest | None = None,
    transport: httpx.BaseTransport | None = None,
    timeout_seconds: float = 5,
) -> Path:
    from scripts.acquire_evaluation_data import acquire_evaluation_data

    return acquire_evaluation_data(
        _request() if request is None else request,
        destination,
        repository_root=repository,
        transport=_transport() if transport is None else transport,
        timeout_seconds=timeout_seconds,
    )


def test_hash_verified_bytes_and_complete_receipt_are_local_only(repository: Path) -> None:
    destination = repository / "data/local/stocknet/fixture-v1"
    assert _acquire(repository, destination) == destination
    assert (destination / "payload.bin").read_bytes() == PAYLOAD
    receipt = json.loads((destination / "receipt.json").read_text())
    assert receipt == {
        **_request().model_dump(mode="json"),
        "schema_version": "restricted-evaluation-acquisition-v1",
        "source_reference": "https://github.com/yumoxu/stocknet-dataset",
        "permission": "local_only_restricted",
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        "byte_count": len(PAYLOAD),
        "http_status": 200,
        "final_url": STOCKNET_URL,
    }
    assert sorted(path.name for path in destination.iterdir()) == ["payload.bin", "receipt.json"]
    assert not list(repository.glob("*.bin"))


def test_mismatch_never_publishes_or_leaves_partial_bytes(repository: Path) -> None:
    destination = repository / "data/local/mismatch"
    with pytest.raises(ValueError, match="SHA256"):
        _acquire(repository, destination, transport=_transport(b"different authored bytes"))
    assert not destination.exists()
    assert not list(repository.rglob("*.bin"))


@pytest.mark.parametrize("relative", ["data/committed", "scripts/raw", "data/local", "data/raw/x"])
def test_destination_must_be_strictly_inside_ignored_local_storage(
    repository: Path, relative: str,
) -> None:
    destination = repository / relative
    with pytest.raises(ValueError, match="data/local"):
        _acquire(repository, destination)
    assert not list(repository.rglob("*.bin"))


def test_unignored_destination_is_rejected(repository: Path) -> None:
    (repository / ".gitignore").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="ignored"):
        _acquire(repository, repository / "data/local/fixture")
    assert not list(repository.rglob("*.bin"))


def test_tracked_destination_is_rejected_even_when_ignored(repository: Path) -> None:
    destination = repository / "data/local/fixture"
    destination.mkdir(parents=True)
    tracked = destination / "existing.txt"
    tracked.write_text("authored tracked placeholder", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "-f", str(tracked)], check=True)
    with pytest.raises(ValueError, match="tracked"):
        _acquire(repository, destination)
    assert tracked.read_text() == "authored tracked placeholder"
    assert not list(repository.rglob("*.bin"))


@pytest.mark.parametrize("component", ["data", "data/local", "data/local/escape"])
def test_symlink_ancestors_cannot_escape_local_storage(
    repository: Path, tmp_path: Path, component: str,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = repository / component
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    destination = repository / "data/local/escape/snapshot"
    with pytest.raises(ValueError, match="symlink|unsafe"):
        _acquire(repository, destination)
    assert list(outside.iterdir()) == []


def test_existing_snapshot_is_immutable(repository: Path) -> None:
    destination = repository / "data/local/immutable"
    _acquire(repository, destination)
    original = {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(FileExistsError):
        _acquire(repository, destination)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == original


@pytest.mark.parametrize("status", [301, 302, 307, 308, 404, 500])
def test_redirects_and_http_errors_never_publish(repository: Path, status: int) -> None:
    def response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, content=PAYLOAD, headers={"location": "http://example.com/unsafe"},
            request=request,
        )
    destination = repository / "data/local/response"
    with pytest.raises(ValueError, match="HTTP|redirect"):
        _acquire(repository, destination, transport=httpx.MockTransport(response))
    assert not destination.exists()
    assert not list(repository.rglob("*.bin"))


def test_transport_uses_explicit_timeout_and_network_failure_leaves_no_output(
    repository: Path,
) -> None:
    def failure(request: httpx.Request) -> httpx.Response:
        assert set(request.extensions["timeout"].values()) == {7.0}
        raise httpx.ReadTimeout("authored timeout", request=request)
    destination = repository / "data/local/timeout"
    with pytest.raises(httpx.ReadTimeout):
        _acquire(repository, destination, transport=httpx.MockTransport(failure), timeout_seconds=7)
    assert not destination.exists()
    assert not list(repository.rglob("*.bin"))


def test_encoded_response_cannot_change_the_meaning_of_the_expected_byte_hash(
    repository: Path,
) -> None:
    def response(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(
            200, content=gzip.compress(PAYLOAD), headers={"content-encoding": "gzip"},
            request=request,
        )
    destination = repository / "data/local/encoded"
    with pytest.raises(ValueError, match="encoding"):
        _acquire(repository, destination, transport=httpx.MockTransport(response))
    assert not destination.exists()
    assert not list(repository.rglob("*.bin"))


def test_empty_response_is_not_an_evaluation_input(repository: Path) -> None:
    destination = repository / "data/local/empty"
    with pytest.raises(ValueError, match="empty"):
        _acquire(
            repository, destination,
            request=_request(expected_sha256=hashlib.sha256(b"").hexdigest()),
            transport=_transport(b""),
        )
    assert not destination.exists()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_timeout_must_be_finite_and_positive(repository: Path, timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        _acquire(repository, repository / "data/local/timeout", timeout_seconds=timeout)
    assert not list(repository.rglob("*.bin"))


def test_complete_fiqa_input_preserves_organizer_source_and_noncommercial_terms(
    repository: Path,
) -> None:
    source_url = "https://drive.google.com/uc?export=download&id=1icRTdnu8UcWyDIXtzpsYc2Hm6ACHT-Ch"
    terms_url = "https://sites.google.com/view/fiqa/"
    request = _request(
        dataset_id="fiqa", source_url=source_url, terms_url=terms_url,
        source_version="organizer-task1-train-final",
    )
    destination = repository / "data/local/fiqa/authored"
    _acquire(repository, destination, request=request)
    receipt = json.loads((destination / "receipt.json").read_text())
    assert receipt["source_reference"] == terms_url
    assert receipt["terms_url"] == terms_url
    assert receipt["source_url"] == source_url
    assert receipt["permission"] == "local_only_restricted"
    assert (destination / "payload.bin").read_bytes() == PAYLOAD


def test_unverified_fiqa_drive_file_is_rejected() -> None:
    with pytest.raises(ValueError, match="organizer-linked"):
        _request(
            dataset_id="fiqa", terms_url="https://sites.google.com/view/fiqa/",
            source_version="explicit-fiqa-version",
            source_url="https://drive.google.com/uc?export=download&id=unverified-file",
        )


def test_receipt_write_failure_removes_all_partial_snapshot_files(
    repository: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write = acquisition._write_bytes

    def fail_receipt(directory: int, name: str, payload: bytes) -> None:
        if name == "receipt.json":
            raise OSError("authored write failure")
        original_write(directory, name, payload)

    monkeypatch.setattr(acquisition, "_write_bytes", fail_receipt)
    destination = repository / "data/local/write-failed"
    with pytest.raises(OSError, match="authored write failure"):
        _acquire(repository, destination)
    assert not destination.exists()
    assert list((repository / "data/local").iterdir()) == []


def test_concurrently_created_empty_snapshot_is_never_overwritten(
    repository: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = repository / "data/local/race"
    original_publish = acquisition._publish_snapshot

    def competing_snapshot(parent: int, stage: str, name: str) -> None:
        destination.mkdir()
        original_publish(parent, stage, name)

    monkeypatch.setattr(acquisition, "_publish_snapshot", competing_snapshot)
    with pytest.raises(FileExistsError):
        _acquire(repository, destination)
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert list(destination.parent.iterdir()) == [destination]


def test_symlink_swap_during_download_is_rechecked_before_writes(
    repository: Path, tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = repository / "data/local/snapshot"

    def swap(request: httpx.Request) -> httpx.Response:
        (repository / "data").mkdir()
        (repository / "data/local").symlink_to(outside, target_is_directory=True)
        return httpx.Response(200, content=PAYLOAD, request=request)

    with pytest.raises(ValueError, match="symlink"):
        _acquire(repository, destination, transport=httpx.MockTransport(swap))
    assert list(outside.iterdir()) == []


def test_cli_without_required_metadata_cannot_start_acquisition() -> None:
    result = subprocess.run(
        ["python3", "scripts/acquire_evaluation_data.py", "--dataset-id", "fiqa"],
        capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert b"--acknowledge-terms" in result.stderr
    assert b"--expected-sha256" in result.stderr


def test_stage_open_failure_removes_the_unpublished_stage_directory(
    repository: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = os.open

    def fail_stage(
        path: str | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None,
    ) -> int:
        if str(path).startswith(".acquire-"):
            raise OSError("authored stage open failure")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", fail_stage)
    destination = repository / "data/local/open-failed"
    with pytest.raises(OSError, match="authored stage open failure"):
        _acquire(repository, destination)
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []
