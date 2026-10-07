# Production fitting and locked local runtime

Status: approved integration scope, 2026-10-07. The user approved typed inputs, production fit/restore, held-out policy selection, normalized ordinal Impact selection, and CLI wiring. This specification replaces the earlier ignored proposal where details conflict. It authorizes code integration and necessary tests only; no demo, presentation, submission, invented empirical result, or implicit acquisition.

Read with `CONTEXT.md`, `docs/development-workflow.md`, ADR 0008, ADR 0010, and the original financial-risk-engine design. Existing reviewed work and unrelated concurrent corrections must be preserved.

## Outcome and compatibility

Implement a real adapter behind the existing `CandidateFitter.fit(candidate, training, *, as_of)` and `CandidateFitter.restore(locked)` interface. Its runtime exposes the existing `manifest`, `risk_engine`, and `scenarios(signal, case, *, as_of)` interface. Compose verified `LocalModels`, `EntityMatcher`, `interpret`, `ConfidenceCalibrator`, `ReferenceBasketBuilder`, `AnalogueRepository`, `ImpactEstimator`, `RiskEngine`, and `StressEngine`.

Preserve the existing public BacktestDataset, TrainingPartition, FittedManifest, DevelopmentLock, RiskSignal, TriggerDecision, and artifact-envelope field sets and schema literals. Do not add defaulted fields to existing serialized records: even harmless defaults can change historical hashes. Expand only DevelopmentConfiguration.objective's accepted literals; old inputs must serialize identically and retain the old objective behavior. Do not change existing trigger reasons, gate order, immutable-write behavior, or final-attempt semantics.

Use existing dependencies and supported Python `>=3.12,<3.13`. Production empirical fitting rejects a synthetic TrainingPartition. Explicit LocalBackend injection remains allowed for offline behavioral tests; it never relaxes snapshot verification or makes test fixtures empirical evidence. Test records exercising empirical-schema branches are explicitly project-authored contract simulations: retain that origin in provenance/source terms, keep outputs in pytest temporary directories, and make no measured empirical claims. Do not relabel an existing synthetic BacktestDataset to evade the production guard; independently construct contract simulations and separately prove synthetic input rejection. No test fixtures are imported by production modules.

## Inputs: definitions are separate from future evidence

A preregistered configuration cannot contain a hash of future event labels. Therefore split local inputs into two records:

1. `RuntimeDefinition`, frozen before the evaluation schedule, binds executable conventions and previously available model/catalogue/basket resources. Its canonical content hash is a candidate parameter.
2. `RuntimeEvidenceIndex`, assembled from frozen dataset evidence, binds `dataset_hash` and content-addressed case-evidence references. It can describe the entire dataset, but the fitter opens only entries in its TrainingPartition. It is not a candidate parameter and is not falsely assigned a preregistration timestamp.

Both live in `risk_engine.backtest.runtime_inputs`, use frozen strict Pydantic records, timezone-aware UTC-normalized datetimes, finite numbers, explicit units/terms, and canonical sorted identities. New records may use the backtest `Record` base; no import from runtime_inputs into backtest.module is needed. Schema values and fields:

| Record | Required fields / invariants |
| --- | --- |
| `RuntimeDefinition` | `schema_version='production-runtime-definition-v1'`, `version`, `content_hash`, `frozen_at`, `source_terms`, `signal_schema_version`, `model_lock: LocalArtifact`, `catalogue: LocalArtifact`, `windows: tuple[WindowConvention,...]`, `baskets: tuple[BasketConvention,...]`, `matching: tuple[MatchingConvention,...]`, `policy_selection: PolicySelectionSpec | None`. Hash excludes only content_hash; registry keys unique. |
| `WindowConvention` | `key`, `window: EventWindow`; offsets include session zero, never inferred solely from horizon. |
| `BasketConvention` | `key`, `spec: ReferenceBasketSpec`, `returns: tuple[ReferenceReturnRow,...]`, `base_market: MarketSnapshot`, `factor_bindings: tuple[FactorBinding,...]`, `available_at`, `source_terms`, `source_hashes: tuple[Sha256Hex,...]`. Availability covers observations and basket calibration; all are no later than definition.frozen_at. |
| `MatchingConvention` | `key`, `config: AnalogueConfig`; version and all matching/back-off settings explicit. Definition requires bootstrap_unvalidated config unless an actual separately bound validation artifact exists; this release accepts bootstrap_unvalidated definitions. |
| `RuntimeEvidenceIndex` | `schema_version='production-runtime-evidence-index-v1'`, `content_hash`, `dataset_hash`, `source_terms`, `cases: tuple[CaseEvidenceReference,...]`; canonical case IDs, unique cases/clusters, own hash excludes content_hash. |
| `CaseEvidenceReference` | `case_id`, `cluster_id`, `artifact: LocalArtifact`; exact file bytes and actual underlying evidence availability. |
| `TrainingCaseEvidence` | `schema_version='production-training-case-v1'`, `case_id`, `cluster_id`, `outcome_hash`, `source_items: tuple[SourceEvidenceBinding,...]`, `targets: tuple[InterpretationTarget,...]`, `analogue: HistoricalAnalogue`, `source_terms`. Analogue event_id equals cluster_id; its original source event references remain in source evidence. No silent relabeling. |
| `SourceEvidenceBinding` | `source_item_id`, `content_hash`; exact complete training-cluster membership. |
| `InterpretationTarget` | `source_item_id`, `content_hash`, `event_id`, `interpretation_version: Literal['interpret-v1']`, `actual_entity_ids: tuple[NonEmptyString,...]`, `actual_event_class`, `label_available_at`, `label_provenance`, `label_hash`, `source_terms`. Actual entity IDs canonical/unique; empty means no identified entity truth. Event ID identifies the independent clause target, not a model-selected class. |
| `LocalModelLocations` | `sentiment_snapshot: Path`, `event_snapshot: Path`, `device: Literal['cpu','cuda']`; locations are supplied separately from frozen identities. |
| `RuntimeLocationConfig` | `schema_version='production-runtime-locations-v1'`, `definition_path: Path`, `evidence_index_path: Path | None`, `artifact_root: Path`, `models: LocalModelLocations`. Resolve relative paths against this configuration file, never against a guessed project root. |

Reuse LocalArtifact file SHA256 and existing model-lock verification. `load_runtime_definition(path: Path) -> RuntimeDefinition` verifies its canonical identity and the model lock/catalogue bytes. `load_runtime_evidence_index(path: Path) -> RuntimeEvidenceIndex` validates index bytes/shape; it does not open all referenced case files. `load_training_evidence(index, training) -> tuple[TrainingCaseEvidence,...]` opens only exactly matching training cases in training order and verifies their bytes, outcome_hash, all source IDs/hashes, independent labels and observed analogue provenance.

Every sidecar label timestamp must be covered by its ObservedOutcome.label_available_at; analogue/source/window/reference availability must be covered by outcome_available_at (and the partition cutoff). Shared definition resources must be historically available before fitting. Missing or later information rejects the dataset; never backdate evidence to satisfy this contract. The manifest's existing `maximum_evidence_available_at == max(training.availability)` remains authoritative. A dataset requiring corrected evidence availability must be regenerated under a new immutable hash; no TrainingPartition schema migration is needed.

Interpretation truth must cover every interpreted training clause, and only clauses belonging to the bound source hash. A case-level label cannot stand in for unrelated clauses or all entity mentions. Repeated mentions of the same entity within the same clause produce one CalibrationRow when raw scores agree; reject conflicting duplicate scores instead of weighting that truth multiple times. Unlinked clauses use the same zero-confidence unknown-entity rule as RiskEngine and are not labeled correct by inventing an entity.

## Candidate semantics

Add `CandidateRuntimeSpec` and `resolve_candidate(candidate, definition) -> CandidateRuntimeSpec`. The required exact `CandidateSpec.parameters` keys are:

- `runtime_definition_sha256`: definition.content_hash.
- `event_window`: a WindowConvention key.
- `confidence_fit`: `binary-temperature-logit-clip1e-12-v1`.
- `cutpoint_rule`: `nearest-rank-lower-ties`.
- `scenario_choice`: `nearest-median-reference-loss-event-id-v1`.
- `policy_mode`: `held-out-material-event-f1-v1` or `unselected`.

Reject unknown/missing keys; all six values are strings. Existing `candidate.basket_convention` resolves a BasketConvention key, `matching_convention` resolves a MatchingConvention key, and `event_window_days` equals `end - start + 1`. Configuration sensitivity_parameters may reference these declared keys; no inert parameter is accepted. Candidate complexity and one-standard-error tie behavior remain the existing protocol.

`policy_mode=held-out-material-event-f1-v1` requires a PolicySelectionSpec; unselected is explicit and needs no invented thresholds. Real candidate registries/grids and numeric configuration are supplied by the caller, never generated from desired output.

Reference Basket lookback/calibration and the shared base market precede the first training event; the existing FrozenImpactCalibration rules are retained. No future returns are used to construct an earlier basket. Query matching uses only Event Class and query attributes genuinely present in frozen fit-time evidence. This release injects no future per-query attributes; unknown query fields take the existing declared back-off. Historical analogue attributes remain fully available for actual matching and audits. No entity-to-geography guesses or future query-label lookup is added.

## Shared identities

Expose `confidence_calibration_identity(calibrator: ConfidenceCalibrator) -> ConfidenceCalibrationIdentity` from risk.module, replacing RiskEngine's inline construction without changing emitted bytes.

Expose `ImpactEstimator.calibration_summary -> ImpactCalibrationSummary` as a defensive validated record in impact.module. Fields: `calibration_version`, `calibration_hash`, `reference_basket_version`, `reference_basket_hash`, `matching_version`, `training_event_ids`, `cutpoints`, `currency`, `horizon_days`, `quantile_convention`, `frozen_at`. Values come from existing actual calibration/repository state, computed losses and cutpoints; do not reimplement severity. The backtest adapter adds evidence_kind='empirical' to construct ImpactFitIdentity. Preserve the existing distinct ASCII-JSON hash domain used by ImpactAudit.

## Fitting and fitted state

Public concrete constructor:

```python
ProductionCandidateFitter(
    *, configuration: DevelopmentConfiguration,
    definition: RuntimeDefinition,
    evidence_index: RuntimeEvidenceIndex | None,
    model_locations: LocalModelLocations,
    artifact_root: Path,
    backend: LocalBackend | None = None,
)
```

It also exposes `close() -> None` and `preflight_restore(locked: FittedManifest) -> None`. The fitter owns one shared sequential LocalModels owner, releases it in close, and creates no network clients. Preflight verifies restore prerequisites without inference or fitting. It does not consume a final attempt. No generic plugin loader is introduced.

`fit` revalidates all records; checks exact declared candidate/configuration/definition/dataset/membership, empirical evidence_kind and `as_of == training.cutoff`; loads only training evidence; then calls a private no-policy core fit. Core fit uses real interpretation outputs and independent targets to fit ConfidenceCalibrator, real ReferenceBasketBuilder output and complete observed HistoricalAnalogues to build FrozenImpactCalibration/AnalogueRepository/ImpactEstimator, and shared identity helpers. Manifest confidence source membership covers every source in every group; Impact training IDs equal the supplied group order.

Choose chronology deterministically at UTC microsecond resolution: let `latest` be maximum recorded training availability (also covering all consumed sidecar resources). `training_end = latest + 1 microsecond`; `frozen_at = latest + 2 microseconds`; require `frozen_at < training.cutoff`. Training start is the minimum analogue event time; require all existing strict calibration rules and basket availability. These are logical replay fit instants, not claimed wall-clock artifact creation dates. No timestamp is moved earlier than actual input availability. Inputs too close to the replay cutoff fail before inference. Matching definition.frozen_at remains its real preregistered timestamp; fitted Confidence and Impact use frozen_at.

CalibrationEvidence uses training.snapshot_id and training.dataset_hash, the exact event/linker score definition, independent target provenance and terms, and development-only rows. ModelIdentity strings are `model_id@revision` for event/sentiment and `entity-matcher-v1:<catalogue.sha256>` for entity linking. Confidence requires both correctness classes and non-neutral identification as today; no synthetic negative rows or fallback constants.

New `FrozenRuntimeState` lives in runtime_state.py with `schema_version='production-runtime-state-v1'`, `content_hash`, `candidate`, `configuration_hash`, `runtime_definition_hash`, `dataset_hash`, `training_groups`, `maximum_evidence_available_at`, `fit_cutoff`, `model_identity`, `model_lock`, `catalogue`, `source_terms`, `training_evidence: tuple[LocalArtifact,...]`, `confidence_calibrator`, `impact_calibration`, `matching_config`, `base_market`, `scenario_choice`, `policy_audit: PolicySelectionAudit | None`, and `policy_absence_reason: str | None`. No complete future evidence index, Source Item text, callbacks, weights, or outer/final labels enter this state. Model/catalogue/training-evidence descriptors bind identity; restore verifies runtime-required bytes and does not reopen training-label files. Retained descriptors are provenance, not permission to refit.

Write canonical state bytes under `artifact_root/<content_hash>.json` exclusively. Existing identical bytes may be verified and reused; differing bytes or partial files fail. FittedManifest.artifacts includes exactly one identity `production-runtime-state-v1` plus the catalogue descriptor; state embeds the full policy audit, avoiding another artifact cycle. State content_hash excludes itself; manifest is computed only after state bytes/hash exist. State never contains manifest_hash. Its policy audit never contains the parent manifest hash. PolicyEvidence stable_fit_hash uses the unchanged existing domain and is built after Confidence/Impact identity construction.

## Runtime and scenarios

`ProductionFittedRuntime` implements the existing protocol and composes the real RiskEngine. Use materiality_provider=None: this interface has no current portfolio argument. BacktestModule already computes actual Portfolio Materiality and TriggerDecision downstream. The adapter does not fabricate Action Priority; the existing Action-Priority-dependent alert metric family can consequently remain explicitly unavailable. Internal policy selection does not need Action Priority and uses actual StressEngine materiality.

For `scenarios`, validate as_of/source/case/model/calibration/fit identity and the actual Impact audit. Recreate the selected observed cohort from frozen state, require IDs/hash/loss evidence agreement, and convert entire vectors using the exact registered window and FactorBindings. Revalue through StressEngine to check reference-loss agreement; do not reach into ImpactEstimator's private methods or use scalar outcomes as shocks.

Choose the full observed vector minimizing `(abs(reference_loss - cohort_median), event_id)`; even-cohort median can lie between members. Never take factorwise tails or medians. Choice is independent of the current portfolio. Retain all converted complete selected vectors in ScenarioEvidence.joint_samples, canonical factor order. ScenarioEvidence.fit_hash is manifest.manifest_hash; calibration_hash is manifest.impact.calibration_hash; scenario.calibration_version is the signal's exact Impact audit; risk_signal_id is the actual signal ID; support_reference/hash binds a canonical record of cohort, source hashes, conversion/window and chosen event. The support record can be embedded in support_reference as canonical JSON for complete replay verification. Availability equals maximum availability of used fit resources and analogue bundles, never now().

Return only full empirical scenarios. Insufficient observed support raises the existing explicit ValueError; do not catch broad errors and fabricate no-risk or hypothetical output. Governed reusable hypothetical templates and an explicit typed abstention interface are outside this integration; missing support is an actionable refusal.

Restore verifies locked manifest/artifact bytes, exact single state artifact, frozen-state cross-identities, catalogue/model snapshot bytes, and configuration/definition identity. Compose only from state. It must not open RuntimeEvidenceIndex or training-label files, call ConfidenceCalibrator.fit, optimize a basket, select policy, write new artifacts, or access the network. ImpactEstimator may recompute deterministic valuation/cutpoints from the frozen state, then must match the locked summary exactly. Existing artifact paths are not rewritten; supplied local model locations may change only when the same pinned bytes verify.

## Policy candidates and chronological selection

Add typed `TriggerCriteria` and `CandidateTriggerEvaluation` in risk.policy and public `evaluate_candidate_trigger(signal, materiality, criteria)`. Criteria has confidence_threshold, economic_floor, materiality_tolerance, currency and the fixed joint Confidence target; validate positive thresholds/floors and tolerance below floor. Evaluation returns the existing substantive gates (entity_identity, confidence_binding, confidence_threshold, impact_threshold, supported_risk, materiality_currency, economic_floor) and `would_trigger`. Candidate confidence_binding checks the fixed target; the policy selector separately verifies the signal's exact fitted calibration identity. There is no policy_selection/policy_chronology success claim and no automatic TriggerDecision. Factor shared predicates so existing TriggerPolicy gate values, ordering, reasons and serialized outputs remain unchanged. Do not add new hypothetical-provenance gates that current TriggerPolicy lacks; empirical production composition already excludes hypothetical scenarios.

`PolicySelectionSpec`, defined in runtime_inputs.py, contains `schema_version='production-policy-selection-spec-v1'`, `version`, `confidence_thresholds: tuple[Decimal,...]`, `economic_floors: tuple[Decimal,...]`, `materiality_tolerance`, `currency`, `train_groups`, `validation_groups`, `minimum_folds`, `minimum_cases`, `minimum_positive_cases`, `minimum_negative_cases`, `alert_budget`, and literals `objective='material-event-f1-v1'`, `tie_rule='fewest-alerts-highest-confidence-highest-floor-v1'`. Grids are positive, sorted, unique, nonempty; all minimum counts/train/validation sizes positive; alert_budget nonnegative. Use configuration.split.embargo, which already covers its longest event window. Do not invent numeric defaults.

`policy_folds(training, spec, *, embargo) -> tuple[InnerFold,...]` scans validation starts from the earliest index with enough eligible preceding groups. Each fold uses the last train_groups with `event_time + embargo <= first_validation.event_time`, all intervening groups as embargo, and the next validation_groups as validation. Advance by validation_groups, so validation blocks never overlap; omit incomplete tails. Fold IDs are deterministic `policy-inner-0001`, etc. No fake final group or outer fold is needed. Naive/DST times are rejected/UTC-normalized under the existing chronology contract.

`PolicyObservation` retains fold ID, case ID, historical as_of, subfit manifest hash, full chosen RiskSignal, actual StressResult, materiality, independent material_event bool, actual outcome hash, and label/outcome availability. Signal reduction is existing greatest-confidence-then-signal-id-v1. Require actual observed valuation evidence to match baseline market, currency, horizon and supported scope/gross values; require complete valuation coverage for policy selection. Missing support excludes the case with an explicit exclusion record; no missing label becomes false. Insufficient remaining evidence means no selected policy, not a successful empty optimum.

`PolicyCandidateResult` retains criteria, TP/FP/FN/TN, alert count, F1, feasibility and reason. All candidates see the identical eligible held-out population. Compute F1 as `2*TP/(2*TP+FP+FN)`, zero only for a mathematically zero numerator with positive denominator; undefined denominator is infeasible. Require at least minimum positive and negative cases, and at least one true positive for a deployable winner; no all-suppressed candidate is selected merely because an alert budget is zero. Feasible candidates satisfy total alert_count <= alert_budget. Maximize F1; ties use fewer alerts, higher threshold, then higher floor. Budget covers this explicit union of disjoint internal validation cases, not a fabricated daily rate.

`select_policy(observations, exclusions, spec, *, folds, parent_training_hash, runtime_definition_hash, configuration_hash) -> PolicySelectionAudit` is pure and deterministic. Audit schema `production-policy-selection-audit-v1` retains all inputs, fold/subfit hashes, all grid results, selected criteria or absence reason, source terms and content hash. Revalidation recomputes candidate counts/results/winner from retained observations; hashes alone cannot validate fabricated scores. The audit can retain subfit manifests (and their immutable state descriptors) to bind actual calibration provenance.

ProductionCandidateFitter obtains observations by fitting the no-policy core only on internal training groups and replaying internal validation cases at original as_of. Prefix calibration never sees held-out labels. Internal subfit configuration_hash stays the actual whole DevelopmentConfiguration hash, while candidate identity is unchanged; no recursive policy selection runs. The core raises a narrow `InsufficientCalibrationEvidence(ValueError)` only for missing both joint correctness classes or all-neutral scores. An internal confidence fit raising that exception is an explicit fold exclusion; artifact/hash/model/data corruption remains a hard error, never an exclusion. Record all exclusions. After selection, core-fit all supplied training groups and bind its selected criteria via PolicyTemplate/PolicyEvidence to the full refit's stable_fit_hash. All selected-policy development timestamps/availability must fit before full frozen_at. If policy selection is unselected/infeasible, use policy=None and a precise reason.

The parent fit's outer validation/test and final period never contribute to internal policy selection. Changing those labels cannot alter fitted numerical state or selected thresholds. Dataset identity hashes may change legitimately.

## Severity objective and backward compatibility

Expand objective to `Literal['event-class-macro-f1', 'normalized-ordinal-impact-accuracy-v1']`; direction remains maximize and objective unit fraction. Add `development_objective(scored: ScoredPartition, objective: str) -> float` in backtest.module and call it from both `_development` and BacktestAudit's objective validation.

Old objective returns existing classification.macro_f1 exactly. New objective requires every scored case to have a chosen signal and exactly matching independent ReferenceLoss for its basket hash/currency; reject missing cases, abstentions, undefined severity, partial populations or nonfinite/out-of-range metrics. Return `1 - ordinal_mae / 9`. Existing severity metrics derive realized deciles using only frozen training cutpoints. This supports one-case folds, unlike rank correlation. Retain all candidates, existing one-standard-error rule, complexity ordering, final-choice rule, sensitivity projections and lock semantics. Do not optimize policy on this metric: threshold/floor choices use their internal policy criterion.

Candidate Reference Basket identities must be stable/predeclared so independent outcomes can include matching losses for every candidate. Missing reference losses are a data-readiness failure, not permission to compute labels from predictions. Old complete JSON fixtures/artifact hashes must round-trip unchanged.

## CLI and operational behavior

Add `--runtime-config PATH` to scripts/run_backtest.py. Preserve `main(argv=None, *, fitter=None)`: supplied fitter remains the explicit test/application seam and cannot be combined with --runtime-config. Without either, preserve actionable refusal/help. For selection, read --config and RuntimeLocationConfig, require evidence_index_path, and build ProductionCandidateFitter. For final, obtain configuration from the verified DevelopmentLock; evidence_index_path may be None and is not opened.

Production path preflight_restore runs after artifact/dataset checks and before opening final output or creating its attempt receipt. It validates state/snapshot/catalogue/definition prerequisites but performs no final inference. Do not impose new optional methods on CandidateFitter or inject a duck-typed callback. CLI knows when it owns a ProductionCandidateFitter and explicitly preflights it; injected fitters retain current behavior. Once final evaluation begins, retain existing consumed-on-failure semantics. Close the owned fitter in finally, including unsuccessful runs. Errors name missing/mismatched local inputs; no downloads, fabricated results or placeholder complete artifacts.

## Acceptance and evidence limits

Behavioral tests cover exact training membership, late/tampered evidence, independent clause targets, true model-port composition, deterministic real valuation/cutpoints, scenario integrity, byte-stable restore without refitting, policy nested leakage, objective sensitivity, legacy hash compatibility, and CLI preflight/final receipts. Run focused TDD and the repository gate (`pytest`, `ruff check .`, `mypy src`) per reviewed checkpoint using supported dependencies; report any unavailable gate honestly.

Authentic local model snapshots and historical data are required to run empirical fits and final evaluation. Missing authentic inputs do not block writing or testing this production code, and mechanics tests do not establish empirical calibration, policy quality, market usefulness, or supported-model smoke success. No production configuration containing guessed numeric empirical values or claimed selected thresholds is committed by this work.
