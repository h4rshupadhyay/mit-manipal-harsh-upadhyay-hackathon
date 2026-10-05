# Financial Risk Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, reproducible NLP Risk Engine that converts news and social text into evidence-backed Risk Signals and runs transparent stress tests on a synthetic wholesale-banking portfolio.

**Architecture:** One local Python deployment contains four deep modules: Data, Risk Engine, Stress Engine, and Backtest. External providers write immutable snapshots; judged analysis replays explicit snapshots through pinned local models, historically calibrated joint scenarios, and deterministic valuation rules. FastAPI exposes machine-readable signals and Streamlit presents Signal Monitor, Portfolio Stress, and Backtest Evidence views.

**Tech Stack:** Python 3.11, Pydantic 2, FastAPI, Streamlit, DuckDB, Polars, NumPy, SciPy, statsmodels OLS, scikit-learn, PyTorch/Transformers, Plotly, HTTPX, pytest, Hypothesis, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-10-04-financial-risk-engine-design.md`

## Global Constraints

- Support Python `>=3.11,<3.13`; GPU acceleration is optional and the same pinned model revision must run on CPU.
- No paid API, remote inference, cloud database, message broker, or generative LLM on the scoring or valuation path.
- The judged workflow defaults to offline replay and must pass with networking disabled.
- Process GDELT news and historical social replay through one canonical Source Item contract. Commit only project-authored synthetic/paraphrased social demo rows; acquire FiQA/StockNet evaluation data separately under their source terms and never redistribute their raw text.
- Emit Sentiment Score `[-1,1]`, one of eight Event Classes, Impact Score `1..10`, calibrated Confidence, evidence, provenance, and version metadata.
- Derive Impact Score from event studies, complete historical Joint Shock Vectors, a versioned Reference Basket, and a frozen empirical loss distribution.
- Keep Impact Score, Confidence, Portfolio Materiality, and Action Priority separate.
- Retain `impact_score >= 8`; select the Confidence threshold and economic materiality floor only on chronological development data.
- Use deterministic valuation rules and explicit factor units; unsupported exposure is disclosed, never treated as zero risk.
- Commit redistributable demo CSV/JSON under `data/`; keep model weights, secrets, large raw downloads, and restricted third-party text out of Git.
- Use an MIT project license and document non-commercial/evaluation-only dataset restrictions separately.

## Commit Discipline

Follow the [development workflow](../../development-workflow.md) for branch
handling, review, authorized per-task pushes, and resumable handoffs. The task
boundaries below govern commit scope; they do not require a branch per task.

- One task below equals one reviewable commit: one coherent behavior or interface plus its tests.
- Complete the red-green cycle locally before committing. Every commit must leave all existing tests green.
- Keep formatting and documentation changes with the behavior that requires them; otherwise give them their own explicitly requested task.
- Use the listed outcome-oriented commit message. Add only the files named by that task.
- When a task changes dependencies, pin direct dependencies in `pyproject.toml` and regenerate `requirements.txt` from the same exact pins before verification.
- Run the focused command before every commit. Run `pytest -q && ruff check . && mypy src` at each phase gate and before final handoff.
- Task 1 is the grandfathered foundation checkpoint already implemented before this incremental policy. Preserve its history; apply these boundaries from Task 2 onward.

## File Structure

```text
pyproject.toml                         package metadata, dependencies, tools
requirements.txt                      evaluator-friendly pinned install input
config/default.toml                   non-secret frozen defaults and taxonomy
config/models.lock.json               model repositories, revisions, hashes
src/risk_engine/domain.py              canonical cross-module Pydantic records
src/risk_engine/config.py              typed configuration loading and validation
src/risk_engine/data/                  provider adapters, snapshots, DuckDB store
src/risk_engine/nlp/                   local model ports and implementations
src/risk_engine/calibration/           event study and historical scenario table
src/risk_engine/impact/                Reference Basket and severity deciles
src/risk_engine/stress/                scenario validation and valuation adapters
src/risk_engine/risk/                  Risk Engine orchestration and trigger policy
src/risk_engine/backtest/              temporal splits, metrics, locked evaluation
src/risk_engine/api/                   FastAPI application
src/risk_engine/dashboard/             Streamlit application and pure view models
scripts/                               acquisition, calibration, evaluation, demo
data/                                  demo, portfolio, calibration, results, manifests
tests/                                 package-mirrored unit and integration tests
docs/                                  methodology, architecture, results, submission
```

## Review Focus

- Duplicate or revised stories must cluster once while retaining every Source Item reference; Task 5 pins this.
- After-close, weekend, and holiday timestamps must map to the next declared market session without leakage; Task 15 pins this.
- Unknown or ambiguous entities must remain visible but abstain from automatic stress; Tasks 7, 10, and 24 pin this.
- Invalid shock units and unsupported instruments must reject or reduce valuation coverage rather than produce zero loss; Tasks 11 and 14 pin this.
- Thin historical support must expose its back-off level and visibly hypothetical provenance; Tasks 20 and 21 pin this.

---

## Phase 1: Foundation

### Task 1: Project Foundation and Canonical Contracts

**Status:** Grandfathered checkpoint. Preserve the completed red-green-review history rather than rewriting it into artificial micro-commits.

**Files:** `AGENTS.md`, `.gitignore`, `pyproject.toml`, `requirements.txt`, `config/default.toml`, `src/risk_engine/__init__.py`, `src/risk_engine/domain.py`, `src/risk_engine/config.py`, `tests/test_domain.py`, `tests/test_config.py`.

**Produces:** Canonical Pydantic records and `AppConfig.load(path: Path) -> AppConfig` with bounded scores, aware timestamps, explicit units, unique identifiers, complete versions, and cross-field invariants.

- [ ] Verify: `pytest tests/test_domain.py tests/test_config.py -v && ruff check src tests && mypy src`
- [ ] Completion: foundation commits exist and the verification command passes; do not create a duplicate commit.

---

## Phase 2: Immutable Data

### Task 2: Snapshot Interfaces and Append-Only Store

**Files:** Create `src/risk_engine/data/interfaces.py`, `src/risk_engine/data/duckdb_store.py`, `tests/data/test_snapshot_store.py`; modify `pyproject.toml`, `requirements.txt` for the pinned DuckDB dependency.

**Interfaces:** Produces `ProviderRequest`, `RawEnvelope`, `SnapshotRef`, `SourceProvider.fetch(request) -> RawEnvelope`, and `DuckDbSnapshotStore.write/load`.

- [ ] Write failing tests for canonical request serialization, SHA-256 preservation, append-only writes, byte-stable load, and incomplete-manifest rejection.
- [ ] Run: `pytest tests/data/test_snapshot_store.py -v`; expect missing-module failures.
- [ ] Implement the interfaces and DuckDB store; every envelope records provider, request, retrieval time, metadata, terms URL, and hash.
- [ ] Verify: `pytest tests/data/test_snapshot_store.py tests/test_domain.py -v`.
- [ ] Commit: `git add pyproject.toml requirements.txt src/risk_engine/data/interfaces.py src/risk_engine/data/duckdb_store.py tests/data/test_snapshot_store.py && git commit -m "feat: persist immutable source snapshots"`.

### Task 3: GDELT News Adapter

**Files:** Create `src/risk_engine/data/gdelt.py`, `tests/data/test_gdelt.py`, `tests/fixtures/gdelt-response.json`; modify `pyproject.toml`, `requirements.txt` for pinned HTTPX.

**Interfaces:** Produces `GdeltProvider.fetch(request: ProviderRequest) -> RawEnvelope` and `normalize_gdelt(envelope) -> list[SourceItem]`.

- [ ] Write failing tests for request parameters, normalized fields, aware timestamps, provenance, HTTP 429, timeout, and schema errors using an injected `httpx.Client`.
- [ ] Run: `pytest tests/data/test_gdelt.py -v`; expect import failure.
- [ ] Implement explicit refresh only; normalization must not perform network access.
- [ ] Verify: `pytest tests/data/test_gdelt.py tests/data/test_snapshot_store.py -v`.
- [ ] Commit: `git add pyproject.toml requirements.txt src/risk_engine/data/gdelt.py tests/data/test_gdelt.py tests/fixtures/gdelt-response.json && git commit -m "feat: normalize GDELT news snapshots"`.

### Task 4: Historical Social CSV Adapter

**Files:** Create `src/risk_engine/data/social_csv.py`, `tests/data/test_social_csv.py`, `tests/fixtures/social.csv`.

**Interfaces:** Produces `SocialCsvProvider.fetch(request) -> RawEnvelope` and `normalize_social_csv(envelope) -> list[SourceItem]`.

- [ ] Write failing tests for UTF-8 parsing, required source-row IDs, project-authored provenance, stable hashes, malformed rows, and news/social contract equivalence.
- [ ] Run: `pytest tests/data/test_social_csv.py -v`; expect import failure.
- [ ] Implement local-file replay with no implicit download or third-party raw-text export.
- [ ] Verify: `pytest tests/data/test_social_csv.py tests/test_domain.py -v`.
- [ ] Commit: `git add src/risk_engine/data/social_csv.py tests/data/test_social_csv.py tests/fixtures/social.csv && git commit -m "feat: replay social text through source contracts"`.

### Task 5: Deterministic Story Clustering

**Files:** Create `src/risk_engine/data/clustering.py`, `tests/data/test_clustering.py`; modify `config/default.toml`.

**Interfaces:** Produces `StoryCluster` and `cluster_stories(items: Sequence[SourceItem], config) -> list[StoryCluster]`.

- [ ] Write failing tests for URL normalization, exact-content matches, the frozen similarity/time rule, earliest event time, deterministic ordering, and duplicate/revised stories retaining all source IDs.
- [ ] Run: `pytest tests/data/test_clustering.py -v`; expect import failure.
- [ ] Implement clustering with configuration-defined thresholds and tie breaks.
- [ ] Verify: `pytest tests/data/test_clustering.py tests/test_config.py -v`.
- [ ] Commit: `git add src/risk_engine/data/clustering.py tests/data/test_clustering.py config/default.toml && git commit -m "feat: cluster duplicate stories deterministically"`.

### Task 6: Data Module Orchestration

**Files:** Create `src/risk_engine/data/module.py`, `tests/data/test_data_module.py`.

**Interfaces:** Produces `DataModule.refresh(provider, request) -> SnapshotRef` and `DataModule.replay(snapshot_id) -> list[SourceItem]`.

- [ ] Write failing tests for explicit refresh, offline replay, provider failure preserving old snapshots, and replay never contacting the network.
- [ ] Run: `pytest tests/data/test_data_module.py -v`; expect import failure.
- [ ] Implement orchestration only through Task 2 interfaces.
- [ ] Verify phase gate: `pytest tests/data tests/test_domain.py tests/test_config.py -v && ruff check . && mypy src`.
- [ ] Commit: `git add src/risk_engine/data/module.py tests/data/test_data_module.py && git commit -m "feat: orchestrate explicit refresh and offline replay"`.

---

## Phase 3: Local NLP Interpretation

### Task 7: Portfolio Entity Matcher

**Files:** Create `src/risk_engine/nlp/entity_matcher.py`, `tests/nlp/test_entity_matcher.py`, `tests/fixtures/entity-catalog.csv`.

**Interfaces:** Produces `EntityMatcher.match(text: str) -> list[EntityLink]`.

- [ ] Write failing tests for normalized aliases, exact identifiers, cashtags, boundaries, deterministic ordering, ambiguity, and unknown entities remaining visible but ineligible for automatic stress.
- [ ] Run: `pytest tests/nlp/test_entity_matcher.py -v`; expect import failure.
- [ ] Implement catalogue-only matching with no network resolution.
- [ ] Verify: `pytest tests/nlp/test_entity_matcher.py tests/test_domain.py -v`.
- [ ] Commit: `git add src/risk_engine/nlp/entity_matcher.py tests/nlp/test_entity_matcher.py tests/fixtures/entity-catalog.csv && git commit -m "feat: link portfolio entities deterministically"`.

### Task 8: NLP Model Ports and Model Lock

**Files:** Create `src/risk_engine/nlp/interfaces.py`, `scripts/lock_models.py`, `config/models.lock.json`, `tests/nlp/test_model_interfaces.py`.

**Interfaces:** Produces `SentimentModel.score(text) -> SentimentResult`, `EventModel.classify(text) -> EventResult`, and a lock-file loader that rejects mutable or incomplete revisions.

- [ ] Write failing tests using fake logits for sentiment arithmetic, fixed hypotheses, bounded probabilities, immutable revisions, and tokenizer/config hashes.
- [ ] Run: `pytest tests/nlp/test_model_interfaces.py -v`; expect import failure.
- [ ] Implement dependency-injected ports and lock validation; the lock script is the only resolver.
- [ ] Verify: `pytest tests/nlp/test_model_interfaces.py -v && ruff check scripts/lock_models.py src/risk_engine/nlp`.
- [ ] Commit: `git add src/risk_engine/nlp/interfaces.py scripts/lock_models.py config/models.lock.json tests/nlp/test_model_interfaces.py && git commit -m "feat: pin local NLP model contracts"`.

### Task 9: Pinned Local Model Adapters

**Files:** Create `src/risk_engine/nlp/local_models.py`, `tests/nlp/test_local_models.py`; modify `pyproject.toml`, `requirements.txt` for pinned PyTorch and Transformers.

**Interfaces:** Produces `ProsusAI/finbert` sentiment and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` event adapters implementing Task 8 ports.

- [ ] Write failing offline unit tests with injected tokenizer/model fakes plus a separately marked CPU model smoke test.
- [ ] Run: `pytest tests/nlp/test_local_models.py -v -m "not model"`; expect import failure.
- [ ] Implement one-active-model-at-a-time loading, pinned-revision enforcement, CPU fallback using the same revision, and no runtime downloads during replay.
- [ ] Verify: `pytest tests/nlp/test_local_models.py -v -m "not model"`.
- [ ] After explicit model acquisition, run `pytest tests/nlp/test_local_models.py -v -m model` on CPU and record the immutable revisions exercised.
- [ ] Commit: `git add pyproject.toml requirements.txt src/risk_engine/nlp/local_models.py tests/nlp/test_local_models.py && git commit -m "feat: run pinned financial NLP models locally"`.

### Task 10: Evidence-Backed Event Interpretation

**Files:** Create `src/risk_engine/nlp/interpret.py`, `tests/nlp/test_interpret.py`.

**Interfaces:** Produces `interpret(item: SourceItem, matcher, sentiment_model, event_model) -> list[InterpretedEvent]`.

- [ ] Write failing tests for clause-local sentiment, extractive evidence, fixed taxonomy, deterministic ordering, and ambiguous/unknown links setting `eligible_for_automatic_stress = false`.
- [ ] Run: `pytest tests/nlp/test_interpret.py -v`; expect import failure.
- [ ] Implement orchestration through Tasks 7-9 without exposing raw model internals.
- [ ] Verify phase gate: `pytest tests/nlp tests/test_domain.py -v -m "not model" && ruff check . && mypy src`.
- [ ] Commit: `git add src/risk_engine/nlp/interpret.py tests/nlp/test_interpret.py && git commit -m "feat: interpret source items with extractive evidence"`.

---

## Phase 4: Transparent Stress Valuation

### Task 11: Shock Normalization and Scenario Validation

**Files:** Create `src/risk_engine/stress/interfaces.py`, `src/risk_engine/stress/shocks.py`, `tests/stress/test_shocks.py`.

**Interfaces:** Produces `normalize_scenario(scenario: StressScenario) -> NormalizedScenario`.

- [ ] Write failing tests for percent, decimal, absolute, basis-point, and volatility-point conversion; reject unknown factors, mixed horizons, invalid signs, and incompatible units atomically.
- [ ] Run: `pytest tests/stress/test_shocks.py -v`; expect import failure.
- [ ] Implement one validated conversion boundary used by every valuation adapter.
- [ ] Verify: `pytest tests/stress/test_shocks.py tests/test_domain.py -v`.
- [ ] Commit: `git add src/risk_engine/stress/interfaces.py src/risk_engine/stress/shocks.py tests/stress/test_shocks.py && git commit -m "feat: normalize stress shocks with explicit units"`.

### Task 12: Spot and Linear Valuation

**Files:** Create `src/risk_engine/stress/valuation.py`, `tests/stress/test_linear_valuation.py`; modify `pyproject.toml`, `requirements.txt` for pinned Hypothesis test support.

**Interfaces:** Produces `SpotValuationAdapter` and `LinearDerivativeValuationAdapter` implementing `value(position, market) -> Decimal` and stressed P&L.

- [ ] Write failing hand-calculated tests for equity, commodity, FX, delta, and DV01 exposures plus zero-shock and monotonicity properties.
- [ ] Run: `pytest tests/stress/test_linear_valuation.py -v`; expect import failure.
- [ ] Implement deterministic signed exposure calculations from normalized shocks.
- [ ] Verify: `pytest tests/stress/test_linear_valuation.py -v`.
- [ ] Commit: `git add pyproject.toml requirements.txt src/risk_engine/stress/valuation.py tests/stress/test_linear_valuation.py && git commit -m "feat: value spot and linear stress exposures"`.

### Task 13: Fixed-Income and Credit Valuation

**Files:** Modify `src/risk_engine/stress/valuation.py`; create `tests/stress/test_credit_valuation.py`.

**Interfaces:** Adds duration/convexity, CS01, and EAD/LGD adapters behind the Task 12 valuation interface.

- [ ] Write failing hand-calculated tests separating risk-free yield, credit spread, and accounting default loss.
- [ ] Run: `pytest tests/stress/test_credit_valuation.py -v`; expect missing adapters.
- [ ] Implement the three explicit rules without a uniform haircut fallback.
- [ ] Verify: `pytest tests/stress/test_credit_valuation.py tests/stress/test_linear_valuation.py -v`.
- [ ] Commit: `git add src/risk_engine/stress/valuation.py tests/stress/test_credit_valuation.py && git commit -m "feat: value bond loan and default stress"`.

### Task 14: Stress Engine, Nonlinear Coverage, and Attribution

**Files:** Create `src/risk_engine/stress/module.py`, `tests/stress/test_stress_module.py`, `tests/stress/test_properties.py`.

**Interfaces:** Produces `StressEngine.run(portfolio, base_market, scenario) -> StressResult`.

- [ ] Write failing tests for delta-gamma-vega only with complete inputs, unsupported nonlinear positions reducing coverage, before/after arithmetic, required attribution dimensions, and per-factor rule versions.
- [ ] Run: `pytest tests/stress/test_stress_module.py tests/stress/test_properties.py -v`; expect import failure.
- [ ] Implement adapter dispatch and hierarchical attribution; unsupported positions receive no invented zero loss.
- [ ] Verify phase gate: `pytest tests/stress tests/test_domain.py -v && ruff check . && mypy src`.
- [ ] Commit: `git add src/risk_engine/stress/module.py tests/stress/test_stress_module.py tests/stress/test_properties.py && git commit -m "feat: run auditable portfolio stress tests"`.

---

## Phase 5: Historical Calibration and Impact

### Task 15: Market-Session Mapping

**Files:** Create `src/risk_engine/calibration/market_calendar.py`, `tests/calibration/test_market_calendar.py`.

**Interfaces:** Produces `map_event_to_session(timestamp, calendar, close_time) -> EventClockDecision`.

- [ ] Write failing tests for intraday, after-close Friday, weekend, holiday, timezone conversion, and no observation beyond the declared evaluation window.
- [ ] Run: `pytest tests/calibration/test_market_calendar.py -v`; expect import failure.
- [ ] Implement explicit calendar/session decisions retained in calibration evidence.
- [ ] Verify: `pytest tests/calibration/test_market_calendar.py -v`.
- [ ] Commit: `git add src/risk_engine/calibration/market_calendar.py tests/calibration/test_market_calendar.py && git commit -m "feat: map events to leakage-safe market sessions"`.

### Task 16: Event-Study Reactions

**Files:** Create `src/risk_engine/calibration/event_study.py`, `tests/calibration/test_event_study.py`, `tests/fixtures/market-series.csv`; modify `pyproject.toml`, `requirements.txt` for pinned NumPy, Polars, and statsmodels.

**Interfaces:** Produces `compute_event_reaction(event, factor_series, benchmark_series, spec) -> EventReaction`.

- [ ] Write failing known-alpha/beta tests for AR/CAR, market-adjusted fallback, missing-data rules, estimation gap, and `[0]`, `[0,+1]`, `[-1,+1]`, `[-2,+2]` windows.
- [ ] Run: `pytest tests/calibration/test_event_study.py -v`; expect import failure.
- [ ] Implement statsmodels OLS while retaining raw observations and event-clock decisions.
- [ ] Verify: `pytest tests/calibration/test_event_study.py tests/calibration/test_market_calendar.py -v`.
- [ ] Commit: `git add pyproject.toml requirements.txt src/risk_engine/calibration/event_study.py tests/calibration/test_event_study.py tests/fixtures/market-series.csv && git commit -m "feat: calculate historical abnormal reactions"`.

### Task 17: Complete Historical Joint Scenarios

**Files:** Create `src/risk_engine/calibration/scenarios.py`, `tests/calibration/test_scenarios.py`.

**Interfaces:** Produces `build_joint_scenario(reactions: Sequence[EventReaction]) -> JointScenario`.

- [ ] Write failing tests for contemporaneous equity, rate, spread, FX, commodity, and volatility vectors; native units; identical event/window identity; and rejection of mixed horizons or independently selected tails.
- [ ] Run: `pytest tests/calibration/test_scenarios.py -v`; expect import failure.
- [ ] Implement complete-vector construction without cross-event factor splicing.
- [ ] Verify: `pytest tests/calibration/test_scenarios.py tests/calibration/test_event_study.py -v`.
- [ ] Commit: `git add src/risk_engine/calibration/scenarios.py tests/calibration/test_scenarios.py && git commit -m "feat: preserve historical joint shock vectors"`.

### Task 18: Reproducible Calibration Dataset Builder

**Files:** Create `scripts/build_calibration_dataset.py`, `tests/calibration/test_build_dataset.py`, `tests/fixtures/historical-events.csv`.

**Interfaces:** Produces derived `events.csv`, `factor_shocks.csv`, coverage report, and manifest with source hashes.

- [ ] Write failing tests for deterministic output, license gating, stratified coverage, RBI policy evidence, Indian credit/default evidence, and missing support recorded rather than fabricated.
- [ ] Run: `pytest tests/calibration/test_build_dataset.py -v`; expect missing script behavior.
- [ ] Implement the builder over committed or locally acquired inputs without exporting restricted raw text.
- [ ] Verify: `pytest tests/calibration -v && ruff check scripts/build_calibration_dataset.py`.
- [ ] Commit: `git add scripts/build_calibration_dataset.py tests/calibration/test_build_dataset.py tests/fixtures/historical-events.csv && git commit -m "feat: build reproducible calibration evidence"`.

### Task 19: Versioned Reference Basket

**Files:** Create `src/risk_engine/impact/reference_basket.py`, `tests/impact/test_reference_basket.py`; modify `pyproject.toml`, `requirements.txt` for pinned SciPy.

**Interfaces:** Produces `ReferenceBasketBuilder.build(returns, spec) -> Portfolio`.

- [ ] Write failing tests for equal-risk contribution, equal-notional and inverse-volatility challengers, frozen constituents/covariance/lookback, and independence from the Synthetic Portfolio.
- [ ] Run: `pytest tests/impact/test_reference_basket.py -v`; expect import failure.
- [ ] Implement deterministic builders behind one interface.
- [ ] Verify: `pytest tests/impact/test_reference_basket.py tests/stress -v`.
- [ ] Commit: `git add pyproject.toml requirements.txt src/risk_engine/impact/reference_basket.py tests/impact/test_reference_basket.py && git commit -m "feat: build versioned reference baskets"`.

### Task 20: Historical Analogue Retrieval and Back-Off

**Files:** Create `src/risk_engine/impact/analogues.py`, `tests/impact/test_analogues.py`; modify `config/default.toml`.

**Interfaces:** Produces `AnalogueRepository.match(event, as_of) -> AnalogueCohort`.

- [ ] Write failing tests for no-future filtering, configured fields/distances/tie breaks, exact back-off hierarchy, support counts, and inadequate global support returning hypothetical provenance.
- [ ] Run: `pytest tests/impact/test_analogues.py -v`; expect import failure.
- [ ] Implement frozen nearest-neighbour and back-off behavior.
- [ ] Verify: `pytest tests/impact/test_analogues.py tests/test_config.py -v`.
- [ ] Commit: `git add src/risk_engine/impact/analogues.py tests/impact/test_analogues.py config/default.toml && git commit -m "feat: retrieve leakage-safe historical analogues"`.

### Task 21: Frozen Empirical Impact Estimation

**Files:** Create `src/risk_engine/impact/module.py`, `tests/impact/test_impact_score.py`.

**Interfaces:** Produces `ImpactEstimator.estimate(event, as_of) -> ImpactEstimate`.

- [ ] Write failing tests for reference-basket revaluation, training-only cutpoints, decile bounds, stable scores when holdings change, uncertainty/support metadata, and hypothetical labeling under thin history.
- [ ] Run: `pytest tests/impact/test_impact_score.py -v`; expect import failure.
- [ ] Implement median expected reference loss and frozen empirical decile mapping.
- [ ] Verify phase gate: `pytest tests/impact tests/stress tests/calibration -v && ruff check . && mypy src`.
- [ ] Commit: `git add src/risk_engine/impact/module.py tests/impact/test_impact_score.py && git commit -m "feat: calibrate portfolio-independent impact scores"`.

---

## Phase 6: Risk Orchestration and Backtesting

### Task 22: Confidence Calibration

**Files:** Create `src/risk_engine/risk/confidence.py`, `tests/risk/test_confidence.py`; modify `pyproject.toml`, `requirements.txt` for pinned scikit-learn.

**Interfaces:** Produces `ConfidenceCalibrator.fit/transform` for the joint probability that entity link and Event Class are correct.

- [ ] Write failing tests for declared target, temperature fitting on development data only, bounded output, deterministic serialization, Brier score, and reliability bins.
- [ ] Run: `pytest tests/risk/test_confidence.py -v`; expect import failure.
- [ ] Implement calibration independently from impact severity and analogue support.
- [ ] Verify: `pytest tests/risk/test_confidence.py -v`.
- [ ] Commit: `git add pyproject.toml requirements.txt src/risk_engine/risk/confidence.py tests/risk/test_confidence.py && git commit -m "feat: calibrate interpretation confidence"`.

### Task 23: Transparent Trigger Policy

**Files:** Create `src/risk_engine/risk/policy.py`, `tests/risk/test_policy.py`; modify `config/default.toml`.

**Interfaces:** Produces `TriggerPolicy.evaluate(signal, portfolio_materiality) -> TriggerDecision`.

- [ ] Write failing tests for `impact_score >= 8`, injected validated Confidence threshold, economic materiality floor/tolerance, every gate/reason, manual override, and ambiguous/unknown entities never auto-triggering.
- [ ] Run: `pytest tests/risk/test_policy.py -v`; expect import failure.
- [ ] Implement a recorded multi-gate decision rather than a bare boolean.
- [ ] Verify: `pytest tests/risk/test_policy.py tests/test_config.py -v`.
- [ ] Commit: `git add src/risk_engine/risk/policy.py tests/risk/test_policy.py config/default.toml && git commit -m "feat: evaluate transparent stress triggers"`.

### Task 24: Risk Engine Orchestration

**Files:** Create `src/risk_engine/risk/module.py`, `tests/risk/test_risk_engine.py`.

**Interfaces:** Produces `RiskEngine.analyze(items, as_of) -> list[RiskSignal]`.

- [ ] Write failing end-to-end tests with fakes for required fields, extractive evidence, versions, deterministic order, separate impact/confidence/materiality, disclosed Action Priority, ambiguity abstention, and no partial-signal publication.
- [ ] Run: `pytest tests/risk/test_risk_engine.py -v`; expect import failure.
- [ ] Implement orchestration over Data, NLP, Impact, Confidence, and policy interfaces only; derive Action Priority from separately stored Impact, Confidence, and Portfolio Materiality inputs.
- [ ] Verify: `pytest tests/risk tests/data tests/nlp tests/impact -v -m "not model"`.
- [ ] Commit: `git add src/risk_engine/risk/module.py tests/risk/test_risk_engine.py && git commit -m "feat: assemble evidence-backed risk signals"`.

### Task 25: Leakage-Safe Chronological Splits

**Files:** Create `src/risk_engine/backtest/splits.py`, `tests/backtest/test_splits.py`.

**Interfaces:** Produces `nested_chronological_splits(...) -> list[OuterFold]`.

- [ ] Write failing tests for expanding outer folds, inner rolling folds, grouped-event isolation, longest-window embargo, untouched final period, and duplicate stories never straddling train/test.
- [ ] Run: `pytest tests/backtest/test_splits.py -v`; expect import failure.
- [ ] Implement deterministic grouped chronological splits.
- [ ] Verify: `pytest tests/backtest/test_splits.py -v`.
- [ ] Commit: `git add src/risk_engine/backtest/splits.py tests/backtest/test_splits.py && git commit -m "feat: create leakage-safe chronological folds"`.

### Task 26: Backtest Metrics and Configuration Selection

**Files:** Create `src/risk_engine/backtest/metrics.py`, `tests/backtest/test_metrics.py`.

**Interfaces:** Produces classification, calibration, severity, interval, scenario, valuation, and alert metrics plus `select_within_one_standard_error(candidates)`.

- [ ] Write failing literal-fixture tests for macro-F1, Brier/log loss, rank correlation, ordinal MAE, monotonicity, coverage, P&L error, alert precision/recall, alerts/day, and simplest-within-one-SE selection.
- [ ] Run: `pytest tests/backtest/test_metrics.py -v`; expect import failure.
- [ ] Implement pure metric functions and deterministic tie breaks.
- [ ] Verify: `pytest tests/backtest/test_metrics.py -v`.
- [ ] Commit: `git add src/risk_engine/backtest/metrics.py tests/backtest/test_metrics.py && git commit -m "feat: measure and select risk configurations"`.

### Task 27: Locked Backtest Orchestration

**Files:** Create `src/risk_engine/backtest/module.py`, `scripts/run_backtest.py`, `tests/backtest/test_backtest_module.py`, `data/calibration/selected-config.json`, `data/calibration/impact-cutpoints.json`, `data/results/development-backtest.json`.

**Interfaces:** Produces `BacktestModule.evaluate(...) -> BacktestReport`; `--select` writes development locks, and `--final` writes once with dataset/model/config hashes.

- [ ] Write failing tests for production-interface reuse, historical `as_of`, all candidate results retained, sensitivity curves, immutable final output, and second-write refusal.
- [ ] Run: `pytest tests/backtest/test_backtest_module.py -v`; expect import failure.
- [ ] Implement nested evaluation and lock-once CLI behavior.
- [ ] Verify phase gate: `pytest tests/backtest tests/calibration tests/impact tests/risk -v && ruff check . && mypy src`.
- [ ] Commit: `git add src/risk_engine/backtest/module.py scripts/run_backtest.py tests/backtest/test_backtest_module.py data/calibration/selected-config.json data/calibration/impact-cutpoints.json data/results/development-backtest.json && git commit -m "feat: run locked chronological backtests"`.

---

## Phase 7: API and Analyst Dashboard

### Task 28: Risk Signal API

**Files:** Create `src/risk_engine/api/dependencies.py`, `src/risk_engine/api/app.py`, `tests/api/test_signals.py`; modify `pyproject.toml`, `requirements.txt` for pinned FastAPI and Uvicorn.

**Interfaces:** Produces `create_app(container: AppContainer) -> FastAPI`, `POST /v1/signals/analyze`, `GET /v1/signals`, and `GET /health`.

- [ ] Write failing OpenAPI/endpoint tests for exact Risk Signal JSON, a stable schema snapshot, explicit snapshot IDs, no implicit refresh, validation errors, and correlation IDs without local paths or secrets.
- [ ] Run: `pytest tests/api/test_signals.py -v`; expect import failure.
- [ ] Implement dependency-injected routes calling module interfaces only.
- [ ] Verify: `pytest tests/api/test_signals.py tests/risk -v`.
- [ ] Commit: `git add pyproject.toml requirements.txt src/risk_engine/api/dependencies.py src/risk_engine/api/app.py tests/api/test_signals.py && git commit -m "feat: expose versioned risk signal endpoints"`.

### Task 29: Stress and Backtest API

**Files:** Modify `src/risk_engine/api/app.py`; create `tests/api/test_stress_and_backtests.py`.

**Interfaces:** Adds `POST /v1/stress-tests` and `GET /v1/backtests/latest`.

- [ ] Write failing tests for bad shocks/versions, traceable TriggerDecision, manual override audit, missing artifacts, and serialized attribution/coverage.
- [ ] Run: `pytest tests/api/test_stress_and_backtests.py -v`; expect missing routes.
- [ ] Implement HTTP 422 for domain validation and 404/409 for missing or conflicting versions.
- [ ] Verify: `pytest tests/api -v`.
- [ ] Commit: `git add src/risk_engine/api/app.py tests/api/test_stress_and_backtests.py && git commit -m "feat: expose stress and backtest evidence"`.

### Task 30: Pure Dashboard View Models

**Files:** Create `src/risk_engine/dashboard/view_models.py`, `tests/dashboard/test_view_models.py`.

**Interfaces:** Produces pure `build_signal_monitor`, `build_portfolio_stress`, and `build_backtest_evidence` view models.

- [ ] Write failing tests for distinct severity/confidence encodings, snapshot badge, provenance, before/after loss, complete attribution, unsupported coverage, analogue distribution, India cases, and empirical/hypothetical labels.
- [ ] Run: `pytest tests/dashboard/test_view_models.py -v`; expect import failure.
- [ ] Implement formatting only; view models never recalculate financial results.
- [ ] Verify: `pytest tests/dashboard/test_view_models.py -v`.
- [ ] Commit: `git add src/risk_engine/dashboard/view_models.py tests/dashboard/test_view_models.py && git commit -m "feat: shape analyst dashboard evidence"`.

### Task 31: Streamlit Analyst Pages

**Files:** Create `src/risk_engine/dashboard/app.py`, `src/risk_engine/dashboard/pages/signal_monitor.py`, `src/risk_engine/dashboard/pages/portfolio_stress.py`, `src/risk_engine/dashboard/pages/backtest_evidence.py`, `tests/dashboard/test_app_smoke.py`; modify `pyproject.toml`, `requirements.txt` for pinned Streamlit and Plotly.

**Interfaces:** Produces three read-focused pages; manual override submits a new `StressScenario` audit record.

- [ ] Write failing smoke tests for fixture rendering, page registration, empty/error states, and override submission without historical mutation.
- [ ] Run: `pytest tests/dashboard/test_app_smoke.py -v`; expect import failure.
- [ ] Implement pages over Task 30 view models.
- [ ] Verify phase gate: `pytest tests/api tests/dashboard -v && ruff check . && mypy src`; smoke-launch Streamlit headlessly.
- [ ] Commit: `git add pyproject.toml requirements.txt src/risk_engine/dashboard/app.py src/risk_engine/dashboard/pages/signal_monitor.py src/risk_engine/dashboard/pages/portfolio_stress.py src/risk_engine/dashboard/pages/backtest_evidence.py tests/dashboard/test_app_smoke.py && git commit -m "feat: add analyst risk dashboard pages"`.

---

## Phase 8: Offline Demonstration

### Task 32: Auditable Synthetic Portfolio

**Files:** Create `scripts/generate_synthetic_portfolio.py`, `data/portfolio/synthetic_portfolio.csv`, `tests/integration/test_synthetic_portfolio.py`.

**Interfaces:** Produces a fixed-seed portfolio of approximately 60 loans, bonds, and derivatives across approved regions, sectors, ratings, maturities, and factors.

- [ ] Write failing tests for deterministic generation, unique IDs, required asset/region coverage, India exposure, factor units, and load through `Portfolio`.
- [ ] Run: `pytest tests/integration/test_synthetic_portfolio.py -v`; expect missing generator/artifact.
- [ ] Implement the generator and commit its reproducible CSV output.
- [ ] Verify: `pytest tests/integration/test_synthetic_portfolio.py -v`.
- [ ] Commit: `git add scripts/generate_synthetic_portfolio.py data/portfolio/synthetic_portfolio.csv tests/integration/test_synthetic_portfolio.py && git commit -m "feat: generate an auditable synthetic portfolio"`.

### Task 33: Restricted Evaluation-Data Acquisition Gate

**Files:** Create `scripts/acquire_evaluation_data.py`, `tests/integration/test_acquire_evaluation_data.py`.

**Interfaces:** Produces an explicit terms-acknowledged, hash-verified acquisition path that writes FiQA/StockNet inputs only under ignored `data/local/`.

- [ ] Write failing tests for required terms acknowledgement, expected hashes, refusal on mismatch, restricted destination, and no raw rows entering tracked paths.
- [ ] Run: `pytest tests/integration/test_acquire_evaluation_data.py -v`; expect missing script behavior.
- [ ] Implement acquisition without making restricted data a prerequisite for the committed demo.
- [ ] Verify: `pytest tests/integration/test_acquire_evaluation_data.py -v && git status --short` with no restricted rows staged.
- [ ] Commit: `git add scripts/acquire_evaluation_data.py tests/integration/test_acquire_evaluation_data.py && git commit -m "feat: gate restricted evaluation data acquisition"`.

### Task 34: Licensed Demo Snapshot

**Files:** Create `scripts/prepare_demo_snapshot.py`, `data/demo/news.json`, `data/demo/social.json`, `data/calibration/events.csv`, `data/calibration/factor_shocks.csv`, `data/manifests/demo-snapshot.json`, `tests/integration/test_demo_snapshot.py`.

**Interfaces:** Produces a complete frozen snapshot containing only redistributable or project-authored text and derived calibration rows.

- [ ] Write failing tests for deterministic manifest hashes, project-authored social text, permitted news text, RBI and Indian credit/default cases, required versions, and incomplete-snapshot refusal.
- [ ] Run: `pytest tests/integration/test_demo_snapshot.py -v`; expect missing artifacts.
- [ ] Implement deterministic preparation without exporting restricted third-party text.
- [ ] Verify: `pytest tests/integration/test_demo_snapshot.py -v && git status --short` with only named artifacts changed.
- [ ] Commit: `git add scripts/prepare_demo_snapshot.py data/demo/news.json data/demo/social.json data/calibration/events.csv data/calibration/factor_shocks.csv data/manifests/demo-snapshot.json tests/integration/test_demo_snapshot.py && git commit -m "feat: freeze licensed offline demo inputs"`.

### Task 35: One-Command Offline Golden Replay

**Files:** Create `scripts/run_demo.py`, `tests/integration/test_offline_demo.py`, `tests/golden/demo-output.json`.

**Interfaces:** Produces `python scripts/run_demo.py` and byte-stable Risk Signal/Stress Result output.

- [ ] Write the failing integration test with networking disabled: load both source types, produce a high-impact eligible signal, run and override a Stress Test, verify before/after values and attribution, and compare golden output.
- [ ] Run: `pytest tests/integration/test_offline_demo.py -v`; expect missing launcher/golden output.
- [ ] Implement artifact/model/version verification, offline replay, and deterministic structured serialization before launching FastAPI/Streamlit.
- [ ] Verify phase gate: `pytest tests/integration/test_offline_demo.py -v && pytest -q && ruff check . && mypy src` with outbound networking disabled.
- [ ] Commit: `git add scripts/run_demo.py tests/integration/test_offline_demo.py tests/golden/demo-output.json && git commit -m "feat: ship deterministic offline replay"`.

---

## Phase 9: Evaluation and Submission

### Task 36: Final Locked Evaluation and Results

**Files:** Create `data/results/final-backtest.json`, `docs/results.md`; modify `scripts/run_backtest.py` only if the locked command fails its existing contract.

**Interfaces:** Produces the immutable final BacktestReport and an honest narrative of measured results, negative findings, limitations, and supported coverage.

- [ ] Run `python scripts/run_backtest.py --final`; verify it writes once and refuses a second overwrite.
- [ ] Validate recorded dataset/model/config hashes against the committed manifests and locks.
- [ ] Write `docs/results.md` from actual output without invented targets.
- [ ] Verify: `pytest tests/backtest tests/integration -v && git diff --check`.
- [ ] Commit: `git add data/results/final-backtest.json docs/results.md scripts/run_backtest.py && git commit -m "docs: report locked evaluation results"`.

### Task 37: Evaluator README, Methodology, and Attribution

**Files:** Modify `README.md`, `LICENSE`, `.gitignore`; create `THIRD_PARTY_NOTICES.md`, `docs/methodology.md`.

**Interfaces:** Produces the evaluator-facing setup, dataset/license assumptions, domain impact, limitations, AI disclosure, and canonical methodology.

- [ ] Write the required README headings, exact Python 3.11 quickstart, `python scripts/run_demo.py`, candidate details, measured results, and artifact links.
- [ ] Add MIT licensing and third-party notices distinguishing code reuse from inspiration and restricted dataset terms.
- [ ] Verify links, commands, ignored secrets/models/databases/restricted data, and consistency with `docs/results.md`.
- [ ] Run: `pytest -q && ruff check . && mypy src && git diff --check`.
- [ ] Commit: `git add README.md LICENSE .gitignore THIRD_PARTY_NOTICES.md docs/methodology.md && git commit -m "docs: explain reproducible risk methodology"`.

### Task 38: Reproducible Architecture Diagram

**Files:** Create `docs/architecture.mmd`, `docs/architecture.png`; modify `README.md` only to embed the diagram.

**Interfaces:** Produces a high-resolution diagram whose source matches the implemented Data, Risk, Stress, Backtest, API, and dashboard boundaries.

- [ ] Write the diagram source from actual module dependencies and offline/online boundaries.
- [ ] Render `docs/architecture.png` and visually verify labels, arrows, legibility, and consistency with the approved design.
- [ ] Verify the README embed resolves and the diagram contains no unimplemented service or data flow.
- [ ] Commit: `git add README.md docs/architecture.mmd docs/architecture.png && git commit -m "docs: diagram the implemented risk architecture"`.

### Task 39: Submission Presentation

**Files:** Create `docs/presentation.pptx`, `docs/presentation.pdf`.

**Interfaces:** Produces a five-to-seven-slide deck covering title, problem/approach, architecture, implementation, measured results, domain impact, limitations, and next steps.

- [ ] Build the deck only from implemented behavior and locked results.
- [ ] Export to PDF and visually verify every slide at presentation dimensions for overflow, contrast, alignment, and readable charts.
- [ ] Verify all figures, claims, and metrics resolve to repository evidence.
- [ ] Commit: `git add docs/presentation.pptx docs/presentation.pdf && git commit -m "docs: present measured risk engine results"`.

### Task 40: Demo Script and Publication Checklist

**Files:** Create `docs/demo-script.md`; modify `README.md` only for final deck/video links.

**Interfaces:** Produces a five-minute jury path, ten-minute recording path, cold-start rehearsal, offline-recovery steps, and manual publication checklist.

- [ ] Write and rehearse source-to-signal, stress result, backtest evidence, manual override, and disabled-network recovery paths.
- [ ] Record the manual gate: the user supplies the final unlisted video URL and approves public naming; no automated publication.
- [ ] Verify final links in an incognito browser after the user supplies them.
- [ ] Run final gate: `pytest -q && ruff check . && mypy src && git status --short`; inspect staged files for secrets, weights, caches, databases, or restricted text.
- [ ] Commit: `git add README.md docs/demo-script.md && git commit -m "docs: script and verify the final demonstration"`.

---

## Final Handoff

- Run a whole-branch review against the merge base with the plan's Review Focus supplied verbatim.
- Fix Critical and Important findings through red-green cycles; record Minor findings explicitly.
- Use `superpowers:finishing-a-development-branch` only after the full verification gate passes.
