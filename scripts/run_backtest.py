"""Publish development artefacts and consume a selected lock's final attempt once.

Supply a verified local runtime configuration or inject a CandidateFitter through
application code. No acquisition, implicit refresh, or overwrite mode is provided.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import BinaryIO

from risk_engine.backtest.module import (
    ARTIFACT_PATHS,
    HASH_DOMAIN,
    ArtifactEnvelope,
    BacktestDataset,
    BacktestModule,
    CandidateFitter,
    CutpointsPayload,
    DevelopmentConfiguration,
    DevelopmentPayload,
    EvaluationPeriod,
    FinalArtifact,
    FinalAttemptReceipt,
    FinalConfiguration,
    PayloadDigest,
    SelectedPayload,
    UnresolvedPayload,
    canonical_bytes,
    content_hash,
)
from risk_engine.backtest.runtime import ProductionCandidateFitter
from risk_engine.backtest.runtime_inputs import (
    LocalModelLocations,
    RuntimeLocationConfig,
    load_runtime_definition,
    load_runtime_evidence_index,
)
from risk_engine.stress.module import StressEngine


def make_artifact_set(
    payloads: tuple[
        SelectedPayload | CutpointsPayload | DevelopmentPayload | UnresolvedPayload, ...
    ],
    *,
    artifact_set_id: str,
    authored_at: object,
    source_terms: tuple[str, ...],
    evidence_kind: str | None,
    status: str,
) -> tuple[ArtifactEnvelope, ...]:
    """Noncircular payload digests followed by a shared ordered receipt."""
    digests = tuple(
        PayloadDigest(path=path, digest=content_hash(payload))
        for path, payload in zip(ARTIFACT_PATHS, payloads, strict=True)
    )
    receipt = dict(
        schema_version="backtest-artifact-envelope-v1",
        hash_domain=HASH_DOMAIN,
        status=status,
        evidence_kind=evidence_kind,
        artifact_set_id=artifact_set_id,
        authored_at=authored_at,
        source_terms=source_terms,
        payload_hashes=digests,
    )
    return tuple(
        ArtifactEnvelope.model_validate(
            {
                **receipt,
                "payload": payload,
                "payload_hash": digest.digest,
                "artifact_set_hash": content_hash(receipt),
            }
        )
        for payload, digest in zip(payloads, digests, strict=True)
    )


def load_artifact_set(root: Path, *, require_ready: bool = True) -> tuple[ArtifactEnvelope, ...]:
    """Selected-config is the last-written validity marker; verify the entire set."""
    envelopes = tuple(
        ArtifactEnvelope.model_validate_json((root / path).read_bytes()) for path in ARTIFACT_PATHS
    )
    first = envelopes[0]
    for envelope in envelopes:
        if envelope.artifact_set_hash != first.artifact_set_hash or (
            envelope.payload_hashes != first.payload_hashes
        ):
            raise ValueError("incomplete or mismatched artifact-set receipt hashes")
    if require_ready and first.status != "ready":
        raise ValueError(
            "unresolved backtest evidence is unavailable; it is not a development lock"
        )
    if first.status == "ready":
        selected, cutpoints, development = (e.payload for e in envelopes)
        if (
            not isinstance(selected, SelectedPayload)
            or not isinstance(cutpoints, CutpointsPayload)
            or (not isinstance(development, DevelopmentPayload))
        ):
            raise ValueError("ready artifact set has incorrect payload roles")
        lock = selected.locked
        if (
            development.audit.locked != lock
            or (cutpoints.fit_hash, cutpoints.training_hash, cutpoints.identity)
            != (lock.fitted.manifest_hash, lock.fitted.training_hash, lock.fitted.impact)
            or (any(e.evidence_kind != lock.evidence_kind for e in envelopes))
        ):
            raise ValueError("artifact set contradicts selected lock/fit identities")
    elif any(not isinstance(e.payload, UnresolvedPayload) for e in envelopes):
        raise ValueError("unresolved artifact set has inconsistent payload roles")
    for envelope, artifact_type in zip(
        envelopes, ("selected-config", "impact-cutpoints", "development-backtest"), strict=True
    ):
        if envelope.payload.artifact_type != artifact_type:
            raise ValueError("artifact payload path/role mismatch")
    return envelopes


def _exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _select(args: argparse.Namespace, fitter: CandidateFitter) -> None:
    root: Path = args.output_root
    if root.exists():
        raise FileExistsError("selection output root already exists; choose a new version/root")
    dataset = BacktestDataset.model_validate_json(args.dataset.read_bytes())
    config = DevelopmentConfiguration.model_validate_json(args.config.read_bytes())
    period = EvaluationPeriod.model_validate_json(args.period.read_bytes())
    module = BacktestModule(fitter, StressEngine())
    report = module.evaluate(dataset, config, period)
    audit = module.audit
    payloads = (
        SelectedPayload(artifact_type="selected-config", locked=audit.locked),
        CutpointsPayload(
            artifact_type="impact-cutpoints",
            status="synthetic-only"
            if dataset.evidence_kind == "synthetic"
            else "empirical-trained",
            fit_hash=audit.selected_final_manifest.manifest_hash,
            training_hash=audit.selected_final_manifest.training_hash,
            identity=audit.selected_final_manifest.impact,
        ),
        DevelopmentPayload(artifact_type="development-backtest", report=report, audit=audit),
    )
    envelopes = make_artifact_set(
        payloads,
        artifact_set_id=f"development:{audit.locked.lock_hash}",
        authored_at=config.final_fit_cutoff,
        source_terms=dataset.source_terms,
        evidence_kind=dataset.evidence_kind,
        status="ready",
    )
    prepared = tuple(canonical_bytes(envelope) + b"\n" for envelope in envelopes)
    # Generate and revalidate all bytes before publication; select marker is last.
    for data in prepared:
        ArtifactEnvelope.model_validate_json(data)
    root.mkdir(parents=True, exist_ok=False)
    for index in (1, 2, 0):
        path = root / ARTIFACT_PATHS[index]
        path.parent.mkdir(parents=True, exist_ok=True)
        _exclusive(path, prepared[index])


def _final_artifact(payload: dict[str, object]) -> FinalArtifact:
    return FinalArtifact.model_validate({**payload, "content_hash": content_hash(payload)})


def _complete_reserved(handle: BinaryIO, artifact: FinalArtifact) -> None:
    # Only the exclusively acquired descriptor can transition this reservation.
    data = canonical_bytes(artifact) + b"\n"
    handle.seek(0)
    handle.write(data)
    handle.truncate()
    handle.flush()
    os.fsync(handle.fileno())


def _final_inputs(
    args: argparse.Namespace,
) -> tuple[tuple[ArtifactEnvelope, ...], BacktestDataset]:
    envelopes = load_artifact_set(args.lock_root)
    selected = envelopes[0]
    assert isinstance(selected.payload, SelectedPayload)
    lock = selected.payload.locked
    dataset = BacktestDataset.model_validate_json(args.dataset.read_bytes())
    if dataset.content_hash != lock.dataset_hash or dataset.evidence_kind != lock.evidence_kind:
        raise ValueError("locked dataset identity mismatch")
    # Validate local bytes and frozen identities before claiming a holdout attempt.
    lock.fitted.verify_artifacts()
    return envelopes, dataset


def _final(
    args: argparse.Namespace,
    fitter: CandidateFitter,
    *,
    production_fitter: ProductionCandidateFitter | None = None,
    verified: tuple[tuple[ArtifactEnvelope, ...], BacktestDataset] | None = None,
) -> None:
    if production_fitter is not None and production_fitter is not fitter:
        raise ValueError("owned production fitter must have identical identity to supplied fitter")
    envelopes, dataset = _final_inputs(args) if verified is None else verified
    selected = envelopes[0]
    assert isinstance(selected.payload, SelectedPayload)
    lock = selected.payload.locked
    if production_fitter is not None:
        production_fitter.preflight_restore(lock.fitted)
    output: Path = args.output_file
    receipt_path = args.lock_root / f".final-attempt-{lock.lock_hash}.json"
    if output.exists() or receipt_path.exists():
        raise FileExistsError(
            "final output or selected lock attempt already consumed; repeat refused"
        )
    payload: dict[str, object] = dict(
        schema_version="backtest-final-artifact-v1",
        status="reserved",
        hash_domain=HASH_DOMAIN,
        evidence_kind=lock.evidence_kind,
        dataset_hash=lock.dataset_hash,
        model_hash=lock.model_hash,
        configuration_hash=lock.configuration_hash,
        selection_hash=content_hash(lock.selection),
        fit_hash=lock.fitted.manifest_hash,
        lock_hash=lock.lock_hash,
        artifact_set_hash=selected.artifact_set_hash,
        holdout_groups=lock.final_groups,
        evaluation_period=lock.final_period,
        report=None,
        audit=None,
        failure_reason=None,
    )
    reserved = _final_artifact(payload)
    claim = FinalAttemptReceipt(
        schema_version="backtest-final-attempt-v1",
        status="consumed",
        lock_hash=lock.lock_hash,
        artifact_set_hash=selected.artifact_set_hash,
        output_path=str(output.resolve()),
        evaluation_period=lock.final_period,
    )
    with output.open("x+b") as handle:
        _complete_reserved(handle, reserved)
        # The lock-specific receipt survives failure and alternate output filenames.
        _exclusive(receipt_path, canonical_bytes(claim) + b"\n")
        try:
            module = BacktestModule(fitter, StressEngine())
            report = module.evaluate(
                dataset,
                FinalConfiguration(schema_version="backtest-final-v1", locked=lock),
                lock.final_period,
            )
            payload.update(status="complete", report=report, audit=module.audit)
            complete = _final_artifact(payload)
            FinalArtifact.model_validate_json(canonical_bytes(complete))
            _complete_reserved(handle, complete)
        except Exception as error:
            payload.update(
                status="failed",
                report=None,
                audit=None,
                failure_reason=f"{type(error).__name__}: {error}",
            )
            _complete_reserved(handle, _final_artifact(payload))
            raise


def _runtime_fitter(
    path: Path, configuration: DevelopmentConfiguration, *, selection: bool = True
) -> ProductionCandidateFitter:
    locations = RuntimeLocationConfig.model_validate_json(path.read_bytes())
    root = path.resolve().parent

    def resolve(location: Path) -> Path:
        return (root / location).resolve()

    if selection and locations.evidence_index_path is None:
        raise ValueError("--select runtime configuration requires evidence_index_path")
    definition = load_runtime_definition(resolve(locations.definition_path))
    evidence_index = (
        load_runtime_evidence_index(resolve(locations.evidence_index_path))
        if selection and locations.evidence_index_path is not None
        else None
    )
    return ProductionCandidateFitter(
        configuration=configuration,
        definition=definition,
        evidence_index=evidence_index,
        model_locations=LocalModelLocations(
            sentiment_snapshot=resolve(locations.models.sentiment_snapshot),
            event_snapshot=resolve(locations.models.event_snapshot),
            device=locations.models.device,
        ),
        artifact_root=resolve(locations.artifact_root),
    )


def main(argv: list[str] | None = None, *, fitter: CandidateFitter | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--select", action="store_true")
    modes.add_argument("--final", action="store_true")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--period", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--lock-root", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--runtime-config", type=Path, help="verified local runtime location JSON")
    args = parser.parse_args(argv)
    if fitter is not None and args.runtime_config is not None:
        print("--runtime-config and injected fitter are mutually exclusive", file=sys.stderr)
        return 2
    if args.select and not all((args.config, args.period, args.output_root)):
        parser.error("--select requires --config, --period, and a new --output-root")
    if args.final and not all((args.lock_root, args.output_file)):
        parser.error("--final requires --lock-root and a new --output-file")
    if fitter is None and args.runtime_config is None:
        print(
            "No default inference backend is configured. Provide a verified local historical "
            "bundle and model/fit artifacts through an explicit application composition "
            "invoking main(argv, fitter=...), or use --runtime-config PATH.",
            file=sys.stderr,
        )
        return 2
    production_fitter: ProductionCandidateFitter | None = None
    try:
        verified = None
        if args.runtime_config is not None:
            if args.select:
                configuration = DevelopmentConfiguration.model_validate_json(
                    args.config.read_bytes()
                )
            else:
                verified = _final_inputs(args)
                selected = verified[0][0]
                assert isinstance(selected.payload, SelectedPayload)
                configuration = selected.payload.locked.configuration
            production_fitter = _runtime_fitter(
                args.runtime_config, configuration, selection=args.select
            )
            fitter = production_fitter
        assert fitter is not None
        if args.select:
            _select(args, fitter)
        else:
            _final(args, fitter, production_fitter=production_fitter, verified=verified)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Backtest refused: {error}", file=sys.stderr)
        return 1
    finally:
        if production_fitter is not None:
            production_fitter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
