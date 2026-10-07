"""Explicit acquisition of restricted FiQA/StockNet bytes into ignored local storage.

This script does not grant redistribution rights or accept terms for the caller.
It is separate from analysis and the committed demo; no acquisition occurs on import.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import math
import os
import re
import subprocess
import uuid
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

FIQA_REFERENCE = "https://sites.google.com/view/fiqa/"
STOCKNET_REFERENCE = "https://github.com/yumoxu/stocknet-dataset"
STOCKNET_TERMS = STOCKNET_REFERENCE + "/blob/master/LICENSE"
# These identifiers are linked by the organizers' FiQA Train/Test Data sections.
# Availability is not guaranteed; redirects and error responses fail closed.
FIQA_FILE_IDS = frozenset({
    "1icRTdnu8UcWyDIXtzpsYc2Hm6ACHT-Ch",
    "1t3E2D7sBD3rG-OHNkvu2Fk7tbMzrMws0",
    "1BlWaV-qVPfpGyJoWQJU9bXQgWCATgxEP",
    "1X-zAZqZoZOhRF1cSJwe-NrMmNRNLtKDG",
})


class AcquisitionRequest(BaseModel):
    """Caller-supplied acquisition metadata; acknowledgement is never inferred."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["fiqa", "stocknet"]
    source_url: str
    terms_url: str
    terms_version: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    available_at: datetime
    retrieved_at: datetime
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    terms_acknowledged: StrictBool

    @field_validator("terms_acknowledged")
    @classmethod
    def explicit_acknowledgement(cls, value: bool) -> bool:
        if not value:
            raise ValueError("terms_acknowledged must be explicitly true")
        return value

    @field_validator("terms_version", "source_version", "snapshot_id")
    @classmethod
    def nonblank_metadata(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("version and snapshot metadata must be nonblank and unpadded")
        return value

    @model_validator(mode="after")
    def source_and_time_consistency(self) -> AcquisitionRequest:
        if self.available_at.utcoffset() is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("available_at and retrieved_at require timezones")
        if self.available_at > self.retrieved_at:
            raise ValueError("retrieved_at cannot precede available_at")
        if self.dataset_id == "fiqa":
            sources = {
                f"https://drive.google.com/uc?export=download&id={file_id}"
                for file_id in FIQA_FILE_IDS
            }
            if self.source_url not in sources or self.terms_url != FIQA_REFERENCE:
                raise ValueError("FiQA requires its organizer-linked source and terms URLs")
        else:
            if not re.fullmatch(r"[0-9a-f]{40}", self.source_version):
                raise ValueError("StockNet source_version must be an immutable Git revision")
            source = (
                "https://codeload.github.com/yumoxu/stocknet-dataset/zip/"
                + self.source_version
            )
            if self.source_url != source or self.terms_url != STOCKNET_TERMS:
                raise ValueError("StockNet requires its official pinned source and terms URLs")
        return self


class AcquisitionReceipt(AcquisitionRequest):
    """Provenance of the exact local bytes, without a redistribution grant."""

    schema_version: Literal["restricted-evaluation-acquisition-v1"]
    source_reference: str
    permission: Literal["local_only_restricted"]
    sha256: str
    byte_count: int
    http_status: Literal[200]
    final_url: str


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, check=False, timeout=10,
    )


def _validate_destination(root: Path, destination: Path) -> None:
    """Fail closed unless both outputs are untracked and ignored inside data/local."""
    top = _git(root, "rev-parse", "--show-toplevel")
    if top.returncode != 0 or Path(os.fsdecode(top.stdout.strip())).resolve() != root:
        raise ValueError("repository_root must be the canonical Git repository root")
    try:
        relative = destination.relative_to(root / "data" / "local")
    except ValueError as exc:
        raise ValueError("destination must be strictly inside repository data/local") from exc
    if not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise ValueError("destination must be strictly inside repository data/local")
    current = root
    for component in destination.relative_to(root).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("symlink destination or ancestor is unsafe")
        if current.exists() and not current.is_dir():
            raise ValueError("destination ancestor is unsafe")
    git_path = destination.relative_to(root).as_posix()
    tracked = _git(root, "ls-files", "-z", "--", git_path)
    if tracked.returncode != 0 or tracked.stdout:
        raise ValueError("destination must not contain tracked paths")
    for path in (git_path, git_path + "/payload.bin", git_path + "/receipt.json"):
        ignored = _git(root, "check-ignore", "--quiet", "--no-index", "--", path)
        if ignored.returncode != 0:
            raise ValueError("destination and outputs must be ignored by Git")
    if destination.exists():
        raise FileExistsError("snapshot destination already exists; use a new version")


def _open_parent(root: Path, destination: Path) -> int:
    """Pin directory handles and refuse symlinks when creating local parents."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(root, flags)
    try:
        for component in destination.parent.relative_to(root).parts:
            with suppress(FileExistsError):
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError as exc:
        os.close(descriptor)
        raise ValueError("destination ancestor is unsafe") from exc
    return descriptor


def _write_bytes(directory: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_snapshot(parent: int, stage: str, name: str) -> None:
    """Linux atomic no-replace directory publication; unsupported hosts fail closed."""
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError as exc:
        raise OSError("atomic no-replace snapshot publication is unavailable") from exc
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(parent, os.fsencode(stage), parent, os.fsencode(name), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError("snapshot destination already exists; use a new version")
        raise OSError(error, "atomic snapshot publication failed")


def acquire_evaluation_data(
    request: AcquisitionRequest,
    output_dir: Path,
    *,
    repository_root: Path,
    transport: httpx.BaseTransport,
    timeout_seconds: float,
) -> Path:
    """Acquire one explicit snapshot; never refresh analysis inputs implicitly.

    All provenance, timestamps, acknowledgement, expected SHA256, transport and
    timeout are supplied by the caller. Bytes and receipt publish together only
    after verification. Existing snapshots are immutable. Linux renameat2 is
    required for atomic no-replace publication; no weaker fallback is used.
    """
    request = AcquisitionRequest.model_validate(request.model_dump())
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    root = repository_root.resolve(strict=True)
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    _validate_destination(root, destination)
    with httpx.Client(
        transport=transport, timeout=timeout_seconds, follow_redirects=False, trust_env=False,
        headers={"Accept-Encoding": "identity"},
    ) as client:
        response = client.get(request.source_url)
        if response.status_code != 200 or str(response.url) != request.source_url:
            raise ValueError("HTTP response must be 200 at the exact source URL; redirects refused")
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            raise ValueError("HTTP content encoding must preserve exact source bytes")
        payload = response.content
        if not payload:
            raise ValueError("empty HTTP response is not an evaluation input")
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != request.expected_sha256:
        raise ValueError("downloaded bytes do not match expected SHA256")
    receipt = AcquisitionReceipt(
        **request.model_dump(),
        schema_version="restricted-evaluation-acquisition-v1",
        source_reference=FIQA_REFERENCE if request.dataset_id == "fiqa" else STOCKNET_REFERENCE,
        permission="local_only_restricted", sha256=actual_hash, byte_count=len(payload),
        http_status=200, final_url=str(response.url),
    )
    receipt_bytes = (receipt.model_dump_json(indent=2) + "\n").encode("utf-8")
    _validate_destination(root, destination)
    parent = _open_parent(root, destination)
    stage_name = ".acquire-" + uuid.uuid4().hex
    stage_descriptor: int | None = None
    stage_created = False
    published = False
    try:
        _validate_destination(root, destination.parent / stage_name)
        os.mkdir(stage_name, mode=0o700, dir_fd=parent)
        stage_created = True
        stage_descriptor = os.open(
            stage_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent,
        )
        _write_bytes(stage_descriptor, "payload.bin", payload)
        _write_bytes(stage_descriptor, "receipt.json", receipt_bytes)
        os.fsync(stage_descriptor)
        _publish_snapshot(parent, stage_name, destination.name)
        published = True
        return destination
    finally:
        if stage_descriptor is not None:
            if not published:
                for name in ("payload.bin", "receipt.json"):
                    with suppress(FileNotFoundError):
                        os.unlink(name, dir_fd=stage_descriptor)
            os.close(stage_descriptor)
        if stage_created and not published:
            os.rmdir(stage_name, dir_fd=parent)
        os.close(parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", choices=("fiqa", "stocknet"), required=True)
    for name in (
        "source-url", "terms-url", "terms-version", "source-version", "snapshot-id",
        "available-at", "retrieved-at", "expected-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--acknowledge-terms", action="store_true", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    arguments = vars(parser.parse_args())
    destination = arguments.pop("output_dir")
    timeout = arguments.pop("timeout_seconds")
    arguments["terms_acknowledged"] = arguments.pop("acknowledge_terms")
    request = AcquisitionRequest.model_validate(arguments)
    acquire_evaluation_data(
        request, destination, repository_root=Path(__file__).resolve().parents[1],
        transport=httpx.HTTPTransport(retries=0, verify=True, trust_env=False),
        timeout_seconds=timeout,
    )


if __name__ == "__main__":
    main()
