# Financial Risk Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, reproducible NLP Risk Engine that converts news and social text into evidence-backed Risk Signals and runs transparent stress tests on a synthetic wholesale-banking portfolio.

**Architecture:** One local Python deployment contains four deep modules: Data, Risk Engine, Stress Engine, and Backtest. External providers write immutable snapshots; all judged analysis replays explicit snapshots through pinned local models, historically calibrated joint scenarios, and deterministic valuation rules. FastAPI exposes machine-readable signals and Streamlit presents Signal Monitor, Portfolio Stress, and Backtest Evidence views.

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
- Retain the problem-statement severity trigger `impact_score >= 8`; select the Confidence threshold and economic materiality floor only on chronological development data.
- Use deterministic valuation rules and explicit factor units; unsupported exposure is disclosed, never treated as zero risk.
- Commit redistributable demo CSV/JSON under `data/`; do not commit model weights, secrets, large raw downloads, or third-party text without redistribution rights. Use an MIT project license and document non-commercial/evaluation-only dataset restrictions separately.
- Every task ends with its focused tests and a commit; run the full quality suite before milestones and final handoff.

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
scripts/                               refresh, calibration, evaluation, demo launch
data/demo/                             redistributable news/social replay inputs
data/portfolio/                        synthetic portfolio CSV/JSON
data/calibration/                      derived events, shocks, and frozen cutpoints
data/manifests/                        provenance, licenses, hashes, versions
tests/                                 package-mirrored unit and integration tests
docs/                                  methodology, architecture, results, submission
```

## Review Focus

- Duplicate or revised stories about one real-world event must cluster without double-counting while retaining every source reference; Task 2 pins this behavior.
- After-close, weekend, and holiday publication timestamps must map to the declared next market session without future leakage; Task 5 pins this behavior.
- Unknown or ambiguous entities must remain visible but abstain from automatic stress; Tasks 3 and 7 pin this behavior.
- Invalid shock units and unsupported instruments must fail or reduce reported valuation coverage rather than silently produce zero loss; Task 4 pins this behavior.
- Thin historical support must expose the back-off level and switch to visibly hypothetical provenance instead of claiming empirical calibration; Task 6 pins this behavior.

---

### Task 1: Project Foundation and Canonical Contracts

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `config/default.toml`
- Create: `src/risk_engine/__init__.py`
- Create: `src/risk_engine/domain.py`
- Create: `src/risk_engine/config.py`
- Create: `tests/test_domain.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: approved glossary in `CONTEXT.md`.
- Produces: `SourceItem`, `MarketObservation`, `MarketSnapshot`, `EntityLink`, `InterpretedEvent`, `ImpactEstimate`, `RiskSignal`, `FactorShock`, `StressScenario`, `Position`, `Portfolio`, `StressResult`, `BacktestReport`, and `AppConfig`.

- [ ] **Step 1: Write failing contract tests**

Assert exact score bounds, timezone-aware timestamps, eight Event Classes plus `Other/Uncertain`, explicit shock units/types, unique IDs, and rejection of incomplete version metadata.

- [ ] **Step 2: Run the contract tests and confirm they fail**

Run: `pytest tests/test_domain.py tests/test_config.py -v`
Expected: collection failure because `risk_engine.domain` and `risk_engine.config` do not exist.

- [ ] **Step 3: Add the package configuration and minimal Pydantic contracts**

Use string enums for event class, source type, shock type, provenance method, and asset type. Use `Decimal` for currency/notional values and floats for normalized scores and market factors. `AppConfig.load(path: Path) -> AppConfig` reads TOML with the standard library and rejects unknown Event Class names or missing model/calibration versions. Generate `requirements.txt` from the locked direct dependencies in `pyproject.toml` so evaluators receive repeatable versions without committing a virtual environment.

- [ ] **Step 4: Run focused quality checks**

Run: `pytest tests/test_domain.py tests/test_config.py -v && ruff check src tests && mypy src`
Expected: all checks pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt config src/risk_engine tests/test_domain.py tests/test_config.py
git commit -m "build: establish risk engine contracts"
```

### Task 2: Immutable Data Module and Two Source Adapters

**Files:**
- Create: `src/risk_engine/data/interfaces.py`
- Create: `src/risk_engine/data/module.py`
- Create: `src/risk_engine/data/duckdb_store.py`
- Create: `src/risk_engine/data/gdelt.py`
- Create: `src/risk_engine/data/social_csv.py`
- Create: `src/risk_engine/data/clustering.py`
- Create: `tests/data/test_data_module.py`
- Create: `tests/data/test_gdelt.py`
- Create: `tests/data/test_social_csv.py`
- Create: `tests/data/test_clustering.py`
- Create: `tests/fixtures/gdelt-response.json`
- Create: `tests/fixtures/social.csv`

**Interfaces:**
- Consumes: `SourceItem`, provider and snapshot metadata from Task 1.
- Produces: `SourceProvider.fetch(request: ProviderRequest) -> RawEnvelope`; `DataModule.refresh(provider, request) -> SnapshotRef`; `DataModule.replay(snapshot_id: str) -> list[SourceItem]`; `cluster_stories(items: Sequence[SourceItem]) -> list[StoryCluster]`.

- [ ] **Step 1: Write failing adapter, snapshot, and clustering tests**

Cover byte-stable replay, hash/provenance preservation, GDELT normalization, social CSV normalization, HTTP 429/schema errors, and two revised/duplicated articles clustering as one event while retaining both Source Item IDs.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/data -v`
Expected: imports fail because the Data Module does not exist.

- [ ] **Step 3: Implement the provider seam and DuckDB snapshot store**

`RawEnvelope` records provider, canonical request, retrieval time, response metadata, source-terms URL, and SHA-256. `DuckDbSnapshotStore.write(...) -> SnapshotRef` is append-only; `load` refuses incomplete manifests. GDELT accepts injected `httpx.Client`; social replay reads UTF-8 CSV with required source-row IDs.

- [ ] **Step 4: Implement deterministic story clustering**

Normalize URLs and text, then cluster on content hash and a frozen similarity/time rule from configuration. Preserve all variants and choose the earliest valid publication time as the cluster event time.

- [ ] **Step 5: Run focused and full foundation tests**

Run: `pytest tests/data tests/test_domain.py tests/test_config.py -v`
Expected: all tests pass without network access.

- [ ] **Step 6: Commit**

```bash
git add src/risk_engine/data tests/data tests/fixtures
git commit -m "feat: add immutable news and social ingestion"
```

### Task 3: Local Entity, Sentiment, and Event Interpretation

**Files:**
- Create: `src/risk_engine/nlp/interfaces.py`
- Create: `src/risk_engine/nlp/entity_matcher.py`
- Create: `src/risk_engine/nlp/local_models.py`
- Create: `src/risk_engine/nlp/interpret.py`
- Create: `scripts/lock_models.py`
- Create: `config/models.lock.json`
- Create: `tests/nlp/test_entity_matcher.py`
- Create: `tests/nlp/test_interpret.py`
- Create: `tests/nlp/test_local_models.py`
- Create: `tests/fixtures/entity-catalog.csv`

**Interfaces:**
- Consumes: `SourceItem`, portfolio entity catalogue, eight-class taxonomy.
- Produces: `EntityMatcher.match(text: str) -> list[EntityLink]`; `SentimentModel.score(text: str) -> SentimentResult`; `EventModel.classify(text: str) -> EventResult`; `interpret(item: SourceItem, ...) -> list[InterpretedEvent]`.

- [ ] **Step 1: Write failing interpretation tests**

Pin alias/cashtag matching, clause-local sentiment, `sentiment = P(positive) - P(negative)`, fixed event hypotheses, evidence spans, deterministic ordering, and ambiguous/unknown entity results that retain candidates but set `eligible_for_automatic_stress = false`.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/nlp -v`
Expected: imports fail because NLP adapters do not exist.

- [ ] **Step 3: Implement the portfolio entity matcher and model ports**

Use normalized aliases plus exact portfolio identifiers; do not add network entity resolution. Model ports accept injected tokenizers/models so unit tests use deterministic fake logits.

- [ ] **Step 4: Implement pinned local model adapters**

Use `ProsusAI/finbert` for initial financial sentiment and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` for the eight descriptive event hypotheses. `scripts/lock_models.py` resolves immutable repository revisions and tokenizer/config hashes into `config/models.lock.json`; runtime refuses an unpinned revision. Load one active model at a time on the RTX 3050 and support the same revision on CPU.

- [ ] **Step 5: Run unit tests and a separately marked local-model smoke test**

Run: `pytest tests/nlp -v -m "not model"`
Expected: unit tests pass with no download.
Run after model acquisition: `pytest tests/nlp -v -m model`
Expected: one known news sentence produces bounded sentiment and one taxonomy label on CPU.

- [ ] **Step 6: Commit**

```bash
git add src/risk_engine/nlp scripts/lock_models.py config/models.lock.json tests/nlp tests/fixtures/entity-catalog.csv
git commit -m "feat: add pinned local NLP interpretation"
```

### Task 4: Stress Engine and Auditable Valuation

**Files:**
- Create: `src/risk_engine/stress/interfaces.py`
- Create: `src/risk_engine/stress/valuation.py`
- Create: `src/risk_engine/stress/module.py`
- Create: `tests/stress/test_valuation.py`
- Create: `tests/stress/test_stress_module.py`
- Create: `tests/stress/test_properties.py`

**Interfaces:**
- Consumes: `Portfolio`, `MarketSnapshot`, and `StressScenario` from Task 1.
- Produces: `ValuationAdapter.value(position, market) -> Decimal`; `StressEngine.run(portfolio, base_market, scenario) -> StressResult`.

- [ ] **Step 1: Write failing hand-calculated and property tests**

Pin spot, duration/convexity, CS01, EAD/LGD, delta, and supported delta-gamma-vega results. Add Hypothesis properties for zero-shock P&L and monotonic declared exposures. Assert unknown shock units reject the whole scenario and unsupported nonlinear positions reduce `valuation_coverage` without receiving zero loss.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/stress -v`
Expected: imports fail because valuation adapters do not exist.

- [ ] **Step 3: Implement scenario validation and valuation adapters**

Convert percent, decimal, absolute, and basis-point shocks at one validated entry point. Keep accounting default loss separate from mark-to-market P&L. Return per-position/factor attribution rows keyed by explicit units and rule version.

- [ ] **Step 4: Run stress tests**

Run: `pytest tests/stress -v`
Expected: fixtures and properties pass.

- [ ] **Step 5: Commit**

```bash
git add src/risk_engine/stress tests/stress
git commit -m "feat: add transparent portfolio stress valuation"
```

### Task 5: Event Study and Historical Joint Scenarios

**Files:**
- Create: `src/risk_engine/calibration/event_study.py`
- Create: `src/risk_engine/calibration/market_calendar.py`
- Create: `src/risk_engine/calibration/scenarios.py`
- Create: `scripts/build_calibration_dataset.py`
- Create: `tests/calibration/test_event_study.py`
- Create: `tests/calibration/test_market_calendar.py`
- Create: `tests/calibration/test_scenarios.py`
- Create: `tests/fixtures/market-series.csv`
- Create: `tests/fixtures/historical-events.csv`

**Interfaces:**
- Consumes: timestamped historical events and factor/benchmark observations.
- Produces: `compute_event_reaction(event, factor_series, benchmark_series, spec) -> EventReaction`; `build_joint_scenario(reactions) -> JointScenario`; derived calibration tables with provenance.

- [ ] **Step 1: Write failing event-study and timestamp tests**

Use synthetic known-alpha/beta data to pin AR/CAR and the market-adjusted fallback. Test `[0]`, `[0,+1]`, `[-1,+1]`, and `[-2,+2]`. Assert an after-close Friday article maps to the next declared market session and never reads observations after its evaluation window.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/calibration -v`
Expected: imports fail because calibration modules do not exist.

- [ ] **Step 3: Implement OLS event reactions and explicit market-session mapping**

Use statsmodels OLS for the market model and retain estimation window, gap, benchmark, missing-data rule, raw observations, and event-clock decision in every `EventReaction`.

- [ ] **Step 4: Preserve complete joint factor vectors**

`build_joint_scenario` may include only contemporaneous observations from the same event/window. Reject mixed horizons or independently selected factor tails.

- [ ] **Step 5: Implement the calibration build script**

The script reads source manifests and committed/locally acquired market files, produces `events.csv`, `factor_shocks.csv`, and a manifest with hashes, and refuses unlicensed raw-text export.

Its coverage report is stratified by Event Class and region. It must include explicit historical support for Geopolitical, Macroeconomic/Monetary, Credit/Default, and Corporate Action, plus an RBI monetary-policy case and an Indian credit/default case; missing minimum support is recorded rather than filled with invented observations.

- [ ] **Step 6: Run calibration tests and static checks**

Run: `pytest tests/calibration -v && ruff check src/risk_engine/calibration scripts/build_calibration_dataset.py`
Expected: all checks pass.

- [ ] **Step 7: Commit**

```bash
git add src/risk_engine/calibration scripts/build_calibration_dataset.py tests/calibration tests/fixtures
git commit -m "feat: derive historical joint stress scenarios"
```

### Task 6: Reference Basket and Impact Score

**Files:**
- Create: `src/risk_engine/impact/reference_basket.py`
- Create: `src/risk_engine/impact/analogues.py`
- Create: `src/risk_engine/impact/module.py`
- Create: `tests/impact/test_reference_basket.py`
- Create: `tests/impact/test_analogues.py`
- Create: `tests/impact/test_impact_score.py`

**Interfaces:**
- Consumes: `InterpretedEvent`, historical `JointScenario` rows from Task 5, and `StressEngine` from Task 4.
- Produces: `ReferenceBasketBuilder.build(returns, spec) -> Portfolio`; `AnalogueRepository.match(event, as_of) -> AnalogueCohort`; `ImpactEstimator.estimate(event, as_of) -> ImpactEstimate`.

- [ ] **Step 1: Write failing basket, analogue, and decile tests**

Assert equal-risk candidate contributions at calibration time, frozen version metadata, training-only empirical cutpoints, ordinal bounds, event-time filtering, stable score when the Synthetic Portfolio changes, and exposure of analogue IDs/support/back-off level.

- [ ] **Step 2: Add thin-history and hypothetical-provenance tests**

Assert the frozen hierarchy is applied exactly; inadequate global support returns `method = hypothetical`, never `empirical`, and includes a capped evidence status rather than an invented analogue count.

- [ ] **Step 3: Run tests and verify failure**

Run: `pytest tests/impact -v`
Expected: imports fail because impact modules do not exist.

- [ ] **Step 4: Implement Reference Basket candidates and analogue retrieval**

Support equal-risk contribution, equal-notional, and inverse-volatility builders behind one interface. Matching uses configuration-defined fields, distances, tie breaks, nearest-neighbour count, and no-future filtering.

- [ ] **Step 5: Implement frozen empirical severity mapping**

Revalue the Reference Basket under every analogue Joint Shock Vector, use the configured expected-loss statistic, and map it through versioned training cutpoints to `1..10`. Return an uncertainty range, support, method, and calibration version.

- [ ] **Step 6: Run impact and upstream financial tests**

Run: `pytest tests/impact tests/stress tests/calibration -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/risk_engine/impact tests/impact
git commit -m "feat: calibrate empirical market impact scores"
```

### Task 7: Risk Engine Orchestration and Trigger Policy

**Files:**
- Create: `src/risk_engine/risk/module.py`
- Create: `src/risk_engine/risk/confidence.py`
- Create: `src/risk_engine/risk/policy.py`
- Create: `tests/risk/test_risk_engine.py`
- Create: `tests/risk/test_confidence.py`
- Create: `tests/risk/test_policy.py`

**Interfaces:**
- Consumes: Data replay, NLP interpretation, ImpactEstimator, portfolio, and frozen policy configuration.
- Produces: `RiskEngine.analyze(items, as_of) -> list[RiskSignal]`; `TriggerPolicy.evaluate(signal, portfolio_materiality) -> TriggerDecision`.

- [ ] **Step 1: Write failing end-to-end Risk Signal tests with fakes**

Assert all required fields, extractive evidence, versions, deterministic ordering, separate Impact/Confidence/Materiality, and no publication of partial signals.

- [ ] **Step 2: Write confidence and trigger-policy tests**

Pin the declared Confidence target, temperature-calibrated output, Impact threshold `>=8`, injected validated Confidence threshold, injected economic materiality floor/tolerance, and manual override. Unknown/ambiguous entities must never auto-trigger.

- [ ] **Step 3: Run tests and verify failure**

Run: `pytest tests/risk -v`
Expected: imports fail because the Risk Engine does not exist.

- [ ] **Step 4: Implement orchestration, calibration, and trigger decisions**

Keep Confidence as the calibrated probability that entity and Event Class are jointly correct. Store analogue support separately. `TriggerDecision` records each gate and reason rather than returning only a boolean.

- [ ] **Step 5: Run Risk Engine and all core-module tests**

Run: `pytest tests/risk tests/data tests/nlp tests/impact tests/stress -v`
Expected: all tests pass without network access.

- [ ] **Step 6: Commit**

```bash
git add src/risk_engine/risk tests/risk
git commit -m "feat: assemble evidence-backed risk signals"
```

### Task 8: Chronological Backtest and Configuration Selection

**Files:**
- Create: `src/risk_engine/backtest/splits.py`
- Create: `src/risk_engine/backtest/metrics.py`
- Create: `src/risk_engine/backtest/module.py`
- Create: `scripts/run_backtest.py`
- Create: `data/calibration/selected-config.json`
- Create: `data/calibration/impact-cutpoints.json`
- Create: `data/results/development-backtest.json`
- Create: `tests/backtest/test_splits.py`
- Create: `tests/backtest/test_metrics.py`
- Create: `tests/backtest/test_backtest_module.py`

**Interfaces:**
- Consumes: historical snapshot IDs, grouped event IDs, candidate configuration grid, RiskEngine, StressEngine.
- Produces: `nested_chronological_splits(...) -> list[OuterFold]`; `BacktestModule.evaluate(...) -> BacktestReport`; locked configuration and results JSON.

- [ ] **Step 1: Write failing temporal-leakage and split tests**

Assert expanding outer folds, inner rolling folds, group isolation, embargo at least the longest event window, and one untouched final period. Duplicate stories from one event may not straddle train and test.

- [ ] **Step 2: Write failing metric and selection tests**

Pin macro-F1, Brier/log loss, severity rank correlation, ordinal MAE, bucket monotonicity, interval coverage, stressed-P&L error, alert precision/recall, alerts/day, and the simplest-within-one-standard-error selector.

- [ ] **Step 3: Run tests and verify failure**

Run: `pytest tests/backtest -v`
Expected: imports fail because backtest modules do not exist.

- [ ] **Step 4: Implement nested chronological evaluation**

Evaluate only the predeclared window, basket, matching/back-off, cutpoint, Confidence threshold, and materiality-floor candidates. Persist all candidate results and sensitivity curves, not only the winner.

- [ ] **Step 5: Implement lock-once evaluation output**

`scripts/run_backtest.py --select` writes `selected-config.json`, training-only `impact-cutpoints.json`, and the development report. `--final` refuses to overwrite an existing final report and records dataset/model/config hashes.

- [ ] **Step 6: Run the full analytical test suite**

Run: `pytest tests/backtest tests/calibration tests/impact tests/risk -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/risk_engine/backtest scripts/run_backtest.py tests/backtest data/calibration/selected-config.json data/calibration/impact-cutpoints.json data/results/development-backtest.json
git commit -m "feat: add leakage-safe historical validation"
```

### Task 9: FastAPI Risk Signal Interface

**Files:**
- Create: `src/risk_engine/api/app.py`
- Create: `src/risk_engine/api/dependencies.py`
- Create: `tests/api/test_app.py`

**Interfaces:**
- Consumes: DataModule, RiskEngine, StressEngine, BacktestReport store.
- Produces: `create_app(container: AppContainer) -> FastAPI`; `POST /v1/signals/analyze`; `GET /v1/signals`; `POST /v1/stress-tests`; `GET /v1/backtests/latest`; `GET /health`.

- [ ] **Step 1: Write failing OpenAPI and endpoint tests**

Assert exact Risk Signal JSON, explicit snapshot IDs, no implicit network refresh, validation errors for bad shocks/versions, traceable TriggerDecision, and stable OpenAPI schema.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/api -v`
Expected: import failure because the API app does not exist.

- [ ] **Step 3: Implement dependency-injected FastAPI routes**

Routes call module interfaces only. Map domain validation to HTTP 422, missing snapshots/versions to 404 or 409, and internal failures to a correlation ID without leaking local paths or secrets.

- [ ] **Step 4: Run API and core contract tests**

Run: `pytest tests/api tests/test_domain.py tests/risk -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/risk_engine/api tests/api
git commit -m "feat: expose versioned risk signal API"
```

### Task 10: Analyst Dashboard

**Files:**
- Create: `src/risk_engine/dashboard/app.py`
- Create: `src/risk_engine/dashboard/view_models.py`
- Create: `src/risk_engine/dashboard/pages/signal_monitor.py`
- Create: `src/risk_engine/dashboard/pages/portfolio_stress.py`
- Create: `src/risk_engine/dashboard/pages/backtest_evidence.py`
- Create: `tests/dashboard/test_view_models.py`
- Create: `tests/dashboard/test_app_smoke.py`

**Interfaces:**
- Consumes: stored Risk Signals, Stress Results, Backtest Reports, and immutable snapshot metadata.
- Produces: three read-focused Streamlit views; pure `build_*_view_model(...)` functions for testable formatting and hierarchy.

- [ ] **Step 1: Write failing view-model tests**

Assert distinct severity and Confidence encodings, visible live/snapshot badge, evidence/provenance, before/after currency and percentage loss, hierarchical attribution, unsupported coverage, analogue distribution, India cases, and hypothetical-vs-empirical labels.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/dashboard -v`
Expected: imports fail because dashboard modules do not exist.

- [ ] **Step 3: Implement pure view models and the three Streamlit pages**

The dashboard reads stored values and never recalculates financial results. Manual scenario overrides submit a new StressScenario with an audit record; they do not mutate historical results.

- [ ] **Step 4: Run dashboard tests and a local smoke launch**

Run: `pytest tests/dashboard -v`
Expected: all tests pass.
Run: `streamlit run src/risk_engine/dashboard/app.py --server.headless true`
Expected: process starts without import or schema errors against fixture data.

- [ ] **Step 5: Commit**

```bash
git add src/risk_engine/dashboard tests/dashboard
git commit -m "feat: add portfolio risk analyst dashboard"
```

### Task 11: Committed Demo Data and Offline Golden Replay

**Files:**
- Create: `scripts/generate_synthetic_portfolio.py`
- Create: `scripts/acquire_evaluation_data.py`
- Create: `scripts/prepare_demo_snapshot.py`
- Create: `scripts/run_demo.py`
- Create: `data/demo/news.json`
- Create: `data/demo/social.json`
- Create: `data/portfolio/synthetic_portfolio.csv`
- Create: `data/calibration/events.csv`
- Create: `data/calibration/factor_shocks.csv`
- Create: `data/manifests/demo-snapshot.json`
- Create: `tests/integration/test_offline_demo.py`
- Create: `tests/golden/demo-output.json`

**Interfaces:**
- Consumes: all four modules, pinned models/configuration, redistributable source material.
- Produces: `python scripts/run_demo.py`; byte-stable golden Risk Signals and Stress Results.

- [ ] **Step 1: Write the failing offline integration test**

Disable network calls, load both source types, generate at least one high-impact eligible signal, run and override one Stress Test, verify before/after values and attribution, and compare structured output with the golden file.

- [ ] **Step 2: Run the integration test and verify failure**

Run: `pytest tests/integration/test_offline_demo.py -v`
Expected: FAIL because committed demo artifacts and launcher do not exist.

- [ ] **Step 3: Generate and commit the auditable synthetic portfolio**

Use a fixed seed and approximately 60 loans, bonds, and derivatives across the approved regions, sectors, ratings, and risk factors. Commit both the generator and its CSV result.

- [ ] **Step 4: Prepare licensed demo inputs and manifests**

Store permitted text or paraphrased/synthetic demo material with explicit provenance; include source/dataset terms, transformations, hashes, dates, and assumptions. The calibration tables contain derived event/factor rows rather than unlicensed article bodies.

The committed social demo rows are project-authored and clearly marked synthetic/paraphrased. `scripts/acquire_evaluation_data.py` fetches FiQA/StockNet evaluation inputs only after explicit terms acknowledgement, verifies expected hashes, and writes them to an ignored local-data directory; raw rows remain uncommitted and are used only under their source terms. The frozen demo and calibration manifests include the RBI policy and Indian credit/default cases required by the approved design.

- [ ] **Step 5: Implement the one-command demo launcher and freeze the golden output**

The launcher verifies every required artifact and model revision before starting FastAPI and Streamlit. It refuses an incomplete snapshot and never refreshes implicitly.

- [ ] **Step 6: Run offline and full quality suites**

Run: `pytest tests/integration/test_offline_demo.py -v`
Expected: PASS with outbound network disabled.
Run: `pytest -q && ruff check . && mypy src`
Expected: all checks pass.

- [ ] **Step 7: Commit**

```bash
git add scripts data tests/integration tests/golden
git commit -m "feat: ship deterministic offline demonstration"
```

### Task 12: Evaluation Results and Submission Package

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `docs/methodology.md`
- Create: `docs/results.md`
- Create: `docs/architecture.png`
- Create: `docs/demo-script.md`
- Create: `docs/presentation.pptx`
- Create: `docs/presentation.pdf`
- Create: `data/results/final-backtest.json`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: locked final BacktestReport, dashboard screenshots, architecture/spec, source/data manifests.
- Produces: complete evaluator-facing public repository and manual video-upload checklist.

- [ ] **Step 1: Run the final locked evaluation once**

Run: `python scripts/run_backtest.py --final`
Expected: writes a versioned immutable final report and refuses any second overwrite. Record actual metrics, negative results, limitations, and supported coverage without inventing targets.

- [ ] **Step 2: Write the evaluator-facing README and methodology**

Follow the supplied README headings exactly. Include candidate details supplied by the user, problem/approach, embedded architecture, tech stack, dataset sources/assumptions/licenses, Python 3.11 quickstart, `python scripts/run_demo.py`, key results/domain impact, demo link field, AI-assistance disclosure, limitations, and links to deeper research.

- [ ] **Step 3: Add license, third-party notices, architecture image, and results**

Use the MIT project license. Attribute every reused dependency/idea and distinguish code reuse from inspiration. `docs/methodology.md` is the concise canonical explanation; the longer research notes remain optional evidence.

- [ ] **Step 4: Build and verify the five-to-seven-slide deck**

Create title, problem/approach, architecture, implementation, measured results, domain impact, and limitations/next-steps slides. Export to `docs/presentation.pdf` and visually verify every slide at presentation dimensions.

- [ ] **Step 5: Prepare and rehearse both demo lengths**

`docs/demo-script.md` contains a five-minute jury path and a ten-minute recording path. Verify a cold local start, offline replay, source-to-signal flow, stress result, backtest evidence, and recovery from a disabled network.

- [ ] **Step 6: Complete the manual publication gate**

The user records and uploads the walkthrough to YouTube as Unlisted, supplies candidate/college details, and approves public repository naming. Insert the final video URL, then verify repository, video, and PDF links in an incognito window; do not publish or transmit on the user's behalf without explicit authorization.

- [ ] **Step 7: Run final repository checks**

Run: `pytest -q && ruff check . && mypy src && git status --short`
Expected: tests/static checks pass and no secret, model weight, cache, local database, or unintended raw dataset is staged.

- [ ] **Step 8: Commit**

```bash
git add README.md LICENSE THIRD_PARTY_NOTICES.md docs data/results/final-backtest.json .gitignore
git commit -m "docs: package hackathon submission"
```
