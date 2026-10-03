# Financial Risk Engine and Portfolio Stress Testing Design

**Date:** 2026-10-04  
**Status:** Approved conversational design, awaiting written-spec review

## 1. Intent

Build a local, reproducible platform that converts real-time or replayed unstructured financial text into structured Risk Signals and uses high-impact signals to propose transparent stress tests on a synthetic wholesale-banking portfolio.

The primary user is a wholesale-bank portfolio risk analyst. The application helps the analyst identify potentially material events, inspect the evidence behind each interpretation, review historically grounded joint market shocks, and quantify portfolio losses by asset, sector, region, and risk factor.

Success means the application can demonstrate one auditable chain:

```text
news/social text
  -> structured Risk Signal
  -> historically grounded Impact Score
  -> joint factor-shock scenario
  -> portfolio revaluation
  -> before/after value and loss attribution
  -> historical validation evidence
```

The system ranks expected severity and supports conditional stress analysis. It does not claim to predict an exact market return, prove that a news event caused an observed market movement, or provide trading advice.

## 2. Scope

### Included

- GDELT news ingestion in live-refresh and frozen-snapshot modes.
- A redistributable historical social-media dataset in replay mode.
- Local entity linking against the portfolio entity catalogue and aliases.
- Local financial sentiment and fixed-taxonomy event classification.
- A machine-readable Risk Signal containing the required Sentiment Score, Event Classification, and Impact Score, plus Confidence, evidence, provenance, and version metadata.
- Event-study calibration using associated abnormal market reactions.
- Matched historical analogues with complete joint factor-shock vectors.
- Deep historical calibration for Geopolitical, Macroeconomic/Monetary, Credit/Default, and Corporate Action events; the remaining classes use declared back-off or hypothetical handling when evidence is thin.
- A versioned cross-asset Reference Basket for portfolio-independent Impact Scores.
- A synthetic global wholesale-banking portfolio with India as a first-class region and approximately 60 loans, bonds, and derivatives.
- Transparent, asset-specific stress approximations.
- An analyst dashboard with Signal Monitor, Portfolio Stress, and Backtest Evidence views.
- FastAPI JSON output for downstream consumption.
- Deterministic offline replay and nested chronological evaluation.
- Complete hackathon packaging and attribution.

### Excluded

- Tactical index rebalancing (Module A).
- A generative LLM on the scoring or valuation path.
- Live social-media firehose ingestion.
- Universal entity resolution beyond the portfolio catalogue.
- Microservices, message brokers, distributed tracing, cloud databases, authentication, or multi-user permissions.
- Full institutional pricing for every derivative.
- Portfolio optimization or automated trading.
- Mandatory remote inference or paid data dependencies.

Additional APIs or models enter the build only when the core path already passes, they measurably improve a required result, and they do not reduce offline-demo reliability.

## 3. Compliance Baseline

The platform processes at least two text-source types, exposes the three required signal fields, and implements Module B. The Stress dashboard explicitly shows portfolio value before and after the scenario, absolute and percentage loss, and hierarchical attribution.

The final repository will include:

- Public source code and incremental commit history.
- A mandatory license and third-party notices.
- A README following the supplied template, with the exact setup and run command.
- Redistributable CSV/JSON demo inputs and the synthetic portfolio under `data/`.
- Dataset sources, terms, transformations, and assumptions.
- `docs/architecture.png`.
- A five-to-seven-slide `docs/presentation.pdf`.
- A ten-minute unlisted video link and a five-minute core live-demo path.
- No confidential S&P Global, Crisil, or client data.
- An honest disclosure of AI-assisted development and attribution for reused code or ideas.

No sample transaction file accompanied the problem statement. The portfolio is therefore synthetic, as explicitly permitted by the submission guidelines, and this assumption will be documented.

## 4. Architecture

The product is one local Python deployment organized around four deep modules. Each module hides its implementation behind a small interface shared by callers and tests.

```text
GDELT / social replay / market files
              |
              v
        1. Data Module
              |
     immutable Source Items
       and Market Snapshots
              |
              v
        2. Risk Engine ------> FastAPI Risk Signal JSON
              |
              v
       proposed Joint Shock Vector
              |
              v
        3. Stress Engine
              |
              v
   stored before/after Stress Results
              |
              v
        Analyst Dashboard

        4. Backtest Module
   reuses the same modules with historical as-of time
```

The deployment contains no hidden network dependency. Refresh operations may contact external providers and write immutable snapshots; analysis and demonstration load explicit snapshot IDs.

### 4.1 Data Module

Interface responsibilities:

- Fetch from a named provider only during an explicit refresh.
- Preserve a raw response envelope, request, publication/effective time, retrieval time, provider metadata, source terms, and content hash.
- Normalize provider data into canonical Source Items or market-factor rows.
- Deduplicate repeated stories without discarding provenance.
- Write and load versioned snapshots.
- Replay snapshots without contacting the network.

Initial adapters are GDELT news, historical social CSV, committed market-factor files, and synthetic portfolio CSV/JSON. Other providers remain deferred.

### 4.2 Risk Engine

Primary interface:

```text
analyze(source_items, as_of, model_versions, calibration_version)
    -> list[RiskSignal]
```

The implementation hides portfolio-aware entity matching, text normalization, sentiment, Event Classification, evidence extraction, analogue retrieval, Impact Score calculation, and Confidence calibration. Internal model probabilities do not leak into callers except through the declared fields and evidence.

The first accuracy path uses a pinned local financial sentiment encoder and one pinned event classifier. Challenger models are evaluated offline rather than run as an uncontrolled ensemble.

### 4.3 Stress Engine

Primary interface:

```text
run(portfolio, market_snapshot, stress_scenario)
    -> StressResult
```

The module validates factor IDs, units, shock types, signs, scenario horizon, and version metadata before valuation. It calculates every financial number deterministically; a language model may not create or alter shocks, scores, valuations, or P&L.

### 4.4 Backtest Module

Primary interface:

```text
evaluate(historical_snapshots, frozen_configuration, evaluation_period)
    -> BacktestReport
```

The module enforces historical `as_of` time, clusters duplicate stories about the same real-world event, reuses the production Risk and Stress Engine interfaces, and produces classification, calibration, severity-ranking, scenario, and valuation metrics.

## 5. Canonical Contracts

Only records crossing module seams are public contracts.

### SourceItem

- Source type and provider.
- Normalized text plus permitted source reference.
- Event/publication and retrieval timestamps.
- URL or dataset row identifier.
- Content hash and snapshot ID.
- Provenance and license metadata.

### MarketSnapshot

- Factor identifier and observation time.
- Value and explicit unit.
- Provider and vintage.
- Snapshot, schema, and normalization versions.

### RiskSignal

- Linked entity and match evidence.
- Sentiment Score in `[-1, 1]`.
- One of eight top-level Event Classes: Geopolitical, Macroeconomic/Monetary, Credit/Default, Regulatory/Legal, Operational/Cyber, Corporate Action, Climate/Natural Disaster, or Other/Uncertain.
- Impact Score in `1..10`.
- Confidence in `[0, 1]` with a declared probability target.
- Expected severity range, analogue count, matching/back-off level, method provenance, and flags.
- Extractive rationale and source evidence.
- Model, calibration, schema, and snapshot versions.

### StressScenario

- Originating Risk Signal.
- Joint factor shocks with shock type, value, unit, and horizon.
- Historical or hypothetical provenance.
- Reference event and calibration versions.
- Explicit analyst overrides.

### StressResult

- Base and stressed portfolio value.
- Absolute and percentage loss.
- Attribution by asset, sector, region, obligor/facility, and factor.
- Supported and unsupported valuation coverage.
- Scenario, market, portfolio, and valuation-rule versions.

### BacktestReport

- Frozen configuration and evaluated time range.
- Classification, confidence, severity, scenario, and valuation metrics.
- Sensitivity results and candidate configurations.
- Historical case-study references.

## 6. Impact Score Methodology

The score is a historically calibrated severity ranking, not an exact return forecast.

### 6.1 Event study

For every historical event, estimate the normal return from a frozen pre-event window and calculate abnormal returns over pre-registered short event windows:

```text
normal_return[t] = alpha + beta * benchmark_return[t]
abnormal_return[t] = observed_return[t] - normal_return[t]
CAR[h] = sum(abnormal_return[t] for t in event_window[h])
```

Market-adjusted returns are the declared fallback when data cannot support a market-model estimate. Rates, spreads, FX, commodities, and volatility preserve their adverse movements in explicit native units.

AR/CAR is described as the abnormal reaction associated with the event. It is not presented as causal when confounding events or information leakage cannot be excluded.

### 6.2 Historical analogues and joint scenarios

The Risk Engine matches only events that were historically available before the scoring timestamp. Match candidates use Event Class, subtype, region, sector, exposure type, and measurable scale indicators. Each analogue retains its complete contemporaneous equity, rate, spread, FX, commodity, and volatility vector. Factors are never independently combined while claiming historical dependence.

Sparse cohorts use a pre-registered back-off or shrinkage rule. The exact hierarchy, support threshold, similarity definition, nearest-neighbour count, and weights are selected on chronological development data and then frozen. Every signal exposes the effective cohort and support count.

### 6.3 Reference Basket and severity deciles

A fixed, versioned cross-asset Reference Basket converts every historical Joint Shock Vector into a comparable stressed loss. Equal-risk contribution is the initial construction candidate; equal-notional and inverse-volatility baskets are validation challengers. Constituents, factor proxies, covariance estimator, lookback, constraints, calibration date, and recalibration rule are explicit.

For matched scenarios:

```text
expected_reference_loss = median(reference_basket_losses)
impact_score = empirical loss decile in a frozen training distribution
```

The required 1–10 score is ordinal. Scores 8–10 mean the upper three deciles of the frozen reference distribution, not automatically a catastrophic event. The display also includes the loss distribution, uncertainty range, sample count, version, and provenance.

### 6.4 Historical and hypothetical provenance

When historical support remains inadequate after back-off, the engine uses a separately governed hypothetical scenario or abstains. A hypothetical reference loss may be located on the same decile scale, but it is always labeled `method = hypothetical` and is never described as directly empirically estimated.

### 6.5 Confidence, materiality, and action

These quantities remain separate:

- **Impact Score:** conditional market severity on the Reference Basket.
- **Confidence:** calibrated probability that entity linking and Event Classification are correct.
- **Portfolio Materiality:** stressed loss on the current Synthetic Portfolio, in currency and percent of value.
- **Action Priority:** analyst-attention ordering based on the preceding fields and a disclosed policy.

The problem-statement trigger `impact_score >= 8` is retained. Automatic proposal additionally requires a confidence threshold selected on held-out calibration data and a configured economic materiality floor. Neither threshold is fixed in advance as a universal constant. Analysts may always run or override a scenario manually.

## 7. Stress Valuation

The Stress Engine applies the same Joint Shock Vector to the Reference Basket and Synthetic Portfolio.

- Equity, commodity, and FX spot: signed market value times relative shock.
- Fixed-rate bonds and loans: duration/convexity response to yield shock, with risk-free and spread sensitivities separated where available.
- Credit spread exposure: CS01 times spread shock in basis points.
- Default/restructuring: EAD times LGD, with recoveries and collateral explicit.
- Linear derivatives: supplied delta or DV01 times factor move.
- Nonlinear derivatives: delta-gamma-vega only when complete inputs exist; otherwise the position is unsupported and disclosed.

Full pricing engines remain a future extension. The prototype reports supported-value share and never interprets unsupported positions as zero risk.

## 8. Synthetic Portfolio

The committed portfolio contains approximately 60 auditable positions across loans, bonds, and derivatives. It spans the United States, Europe, India, and Asia-Pacific, multiple sectors and ratings, and explicit equity, rate, spread, FX, commodity, and volatility sensitivities.

India is one first-class region rather than the whole product scope. Historical evidence includes an RBI monetary-policy episode and an Indian credit/default episode under the same inclusion rules as global cases.

## 9. Dashboard

### Signal Monitor

- Live or snapshot state and `as_of` time.
- Source, entity, Event Class, Sentiment, Impact Score, Confidence, and materiality.
- Clear separation between severity and reliability.
- Evidence drawer with source span, entity-match rationale, versions, flags, and provenance.

### Portfolio Stress

- Proposed or analyst-overridden Joint Shock Vector.
- Base value, stressed value, absolute loss, and percentage loss.
- Attribution from global to region, country, sector, obligor/facility, and factor.
- Unsupported exposure and valuation coverage.

### Backtest Evidence

- Historical analogue cohort and complete shock vectors.
- Expected and tail loss distribution.
- One-day reaction and the frozen primary short horizon.
- Classification, confidence, severity-ranking, and interval metrics.
- Drill-down historical cases, including India evidence.

## 10. Failure Behaviour

- Provider refresh failure never invalidates an existing offline snapshot.
- Replay never silently contacts the network.
- GPU failure may fall back to CPU only with the same pinned model revision.
- Ambiguous entities remain visible but cannot automatically trigger portfolio stress.
- Low classification Confidence produces `Other/Uncertain` or abstention rather than a forced class.
- Thin history exposes back-off and support; hypothetical fallback is explicit.
- Invalid shock units or signs reject the entire scenario before valuation.
- Unsupported positions reduce coverage and are reported rather than assigned zero loss.
- Missing model, calibration, schema, or snapshot versions reject replay.
- Completed immutable records survive a partial pipeline failure, but incomplete Risk Signals are never published as complete.

## 11. Validation and Testing

### Pre-registered methodology

Project conventions are selected under one nested chronological protocol:

1. Freeze data snapshots, event clustering, timestamps, factor definitions, units, and missing-data rules.
2. Use expanding-time outer folds with an embargo at least as long as the longest event window.
3. Use inner rolling folds to compare declared event windows, basket constructions, matching/back-off candidates, score cutpoints, and trigger policies.
4. Prefer the simplest candidate within one standard error of the best development result.
5. Lock the configuration and evaluate once on an untouched final period.
6. Any later change creates a new version and requires a new future holdout.

### Calculation tests

- Published or hand-calculated AR/CAR fixtures.
- Duration/convexity, CS01, EAD/LGD, delta, and unit/sign fixtures.
- Half/base/double-bump convergence for supported approximation domains.

### Invariants and contracts

- Zero shock produces zero P&L.
- Increasing an adverse shock cannot improve loss for a declared monotonic exposure.
- Impact Score remains within `1..10`.
- No future observation enters analogue selection, calibration, or replay.
- Both ingestion adapters emit valid Source Items.
- Risk Signal JSON conforms to the frozen schema.
- Scenario units match valuation-adapter expectations.

### Golden replay

A pinned input snapshot, portfolio, models, and configuration produce byte-identical structured output. The complete judged workflow passes with networking disabled.

### Reported evaluation

- Event-Class macro-F1, per-class confusion, and abstention coverage.
- Entity-link accuracy.
- Confidence Brier/log loss and reliability plots.
- Severity rank correlation, ordinal error, bucket monotonicity, and interval coverage.
- Historical-scenario factor-direction, distribution, and stressed-P&L error.
- Valuation approximation error and supported-value share.
- Trigger precision/recall, alerts per day, and realized loss captured under a fixed alert budget.

## 12. Existing Work and External Data

Existing projects provide patterns rather than an application shell:

- OpenBB: provider adapters and normalized contracts.
- FinRobot: deterministic financial calculations separated from narrative.
- Trade the Event: event-time/entity joins to later market outcomes; code is not reused because the repository lacks a license.
- OpenGamma Strata and Open Source Risk Engine: historical-scenario and revaluation semantics.
- QuantLib: a future full-pricing option, not a core dependency.

The core demo uses frozen, redistributable data. GDELT and free official sources may refresh snapshots, while market data remains behind a replaceable adapter. Raw third-party text is committed only when redistribution is allowed; otherwise the repository contains permitted metadata, derived calibration tables, attribution, and reproducible acquisition instructions.

## 13. Acceptance Criteria

The design is complete when the implementation can demonstrate all of the following:

1. Process news and social-media text through the same canonical pipeline.
2. Emit machine-readable Sentiment Score, Event Classification, and Impact Score.
3. Resolve each displayed score to its input snapshot, evidence, models, and calibration version.
4. Produce historical or explicitly hypothetical Joint Shock Vectors with units and provenance.
5. Trigger a proposed Stress Test under the frozen transparent policy and allow manual override.
6. Revalue the mixed-asset Synthetic Portfolio and show before/after value and attribution.
7. Replay the judged workflow offline with deterministic structured results.
8. Demonstrate temporal validation without future leakage and report honest baseline comparisons.
9. Distinguish established methodology, empirical project results, hypothetical assumptions, and demo policy in UI and documentation.
10. Package all required public-repository, data, diagram, deck, README, license, attribution, and video artifacts.

## 14. Supporting Decisions and Research

- Domain language: [`CONTEXT.md`](../../../CONTEXT.md)
- Architecture decisions: [`docs/adr/`](../../adr/)
- Model and data selection: [`docs/research/model-and-data-selection.md`](../../research/model-and-data-selection.md)
- Existing solutions and APIs: [`docs/research/existing-solutions-and-free-apis.md`](../../research/existing-solutions-and-free-apis.md)
- Time-tested methods: [`docs/research/time-tested-methods.md`](../../research/time-tested-methods.md)
- Methodology audit: [`docs/research/methodology-audit.md`](../../research/methodology-audit.md)
