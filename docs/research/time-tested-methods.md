# Time-tested methods for event-to-portfolio stress

**Decision date:** 2026-10-04  
**Scope:** local, reproducible hackathon pipeline from financial text to event class, expected severity 1–10, confidence and portfolio impact.

## Decision

Among the reviewed primary-source projects, **none is a mature, reusable end-to-end solution for the full chain**. The closest NLP system, Trade the Event, ends in an equity-trading backtest; the mature risk engines, ORE and Strata, begin with structured trades, market data and scenarios. Neither supplies an empirically calibrated text-to-severity bridge.

The strongest design is therefore a composition of established methods:

```text
calibrated event classifier
  -> matched historical events
  -> abnormal market moves and joint factor shocks
  -> empirical expected loss distribution
  -> severity decile 1..10 + uncertainty
  -> bump-and-revalue / sensitivity-based portfolio P&L
```

The 1–10 score is a display layer, not the scenario itself. The portfolio engine must receive the underlying factor-shock vector, horizon and units.

## End-to-end candidates

| Candidate | What it actually covers | License / maturity | Decision |
|---|---|---|---|
| [Trade the Event](https://github.com/Zhihan1996/TradeTheEvent) ([paper](https://aclanthology.org/2021.findings-acl.186/)) | Reuters news scraping, corporate-event detection, sentiment and a stock-trading backtest. This is the nearest published text-to-market pipeline. It does not produce calibrated ordinal severity, cross-asset scenarios or portfolio stress P&L. | The 2021 research repository has only 28 commits and no LICENSE file; the news is Reuters-derived. | **Method inspiration only.** Reimplement the event-time/entity-to-return join; do not reuse code or redistribute data without permission. |
| [Open Source Risk Engine (ORE)](https://github.com/OpenSourceRisk/Engine) | Production-oriented trade/market interfaces, pricing, sensitivity, VaR, stress and XVA. The current [user guide](https://opensourcerisk.org/content/uploads/2026/04/userguide.pdf) documents bump-and-revalue NPV changes under user-defined scenarios and scenario simulation markets. It has no text ingestion or event calibration. | Modified BSD; active, very mature C++/QuantLib codebase with more than 40,000 repository commits, tests and Python/Jupyter launchers. | **Direct reuse only if real derivative revaluation is essential.** Otherwise copy its scenario/baseline/result semantics; integration is too large for ten days. |
| [OpenGamma Strata](https://github.com/OpenGamma/Strata) | Mature Java pricing and market-risk library. Its official [historical-scenario example](https://github.com/OpenGamma/Strata/blob/main/examples/src/main/java/com/opengamma/strata/examples/finance/HistoricalScenarioExample.java) creates dated curve perturbations and scenario P&L. It has no NLP or severity learning. | Apache-2.0; the project states that it is maintained, tested and used in production, with nearly 4,900 commits. | **Architecture reference.** Reuse directly only for a Java stack or broad instrument pricing. |

No candidate should be adopted as the application shell. The missing piece—mapping a newly classified event to a market-impact distribution—remains project-owned.

## Mature composable methods

### 1. Market-model event study: the empirical bridge

The classic method is the event study described by [MacKinlay (1997)](https://www.bu.edu/econ/files/2011/01/MacKinlay-1996-Event-Studies-in-Economics-and-Finance.pdf): estimate the normal return before the event, calculate abnormal returns around the event, and aggregate them into cumulative abnormal return (CAR). This directly answers “how unusually did the market move after comparable events?”

For each historical event `i`, risk factor `f` and horizon `h`:

```text
normal_return[i,f,t] = alpha[i,f] + beta[i,f] * benchmark_return[f,t]
AR[i,f,t]           = observed_return[i,f,t] - normal_return[i,f,t]
CAR[i,f,h]          = sum(AR[i,f,t], t in event window h)
```

Use market-adjusted returns as the transparent fallback when an estimation window is too sparse; otherwise use OLS over a frozen pre-event window. Store windows, benchmark, event-time rule, missing-data rule and raw observations. For rates, spreads and volatility, preserve adverse changes in native units rather than pretending they are returns.

The R package [`estudy2`](https://irudnyts.github.io/estudy2/) has the strongest reviewed statistical coverage: three classical market models, six parametric tests, six nonparametric tests and CAR tests. However, it is GPL-3.0, its [repository is archived](https://github.com/irudnyts/estudy2), and [CRAN removed it in 2022](https://cran.r-project.org/package=estudy2). Use it only as a statistical oracle/reference. The Python [`eventstudy`](https://github.com/LemaireJean-Baptiste/eventstudy) package implements single and aggregate event studies, but is also GPL-3.0 and has seen little substantive activity since 2021 (the latest 2023 commit is documentation only). The required AR/CAR computation is small. **Use MacKinlay's method and project-owned tests, not either runtime dependency.** Validate synthetic zero-impact, known-alpha/beta and sign-direction fixtures.

Important limits:

- Event studies measure association around an event, not structural causality. Overlapping news, leakage and uncertain timestamps must lower confidence.
- Do not select analogues using post-event outcomes.
- Evaluate event windows such as `[0,1]`, `[0,2]` and `[0,5]` separately; do not maximize over them during model selection and then report the maximum as unbiased.
- Split calibration and evaluation chronologically to prevent future events or revised data from leaking backward.

### 2. Severity: empirical ranks first, ordinal models second

Do not train a text model on subjective “severity 1–10” labels. First define a continuous adverse impact measure from observed outcomes. For a matched analogue set `A`:

```text
impact_i = portfolio_loss(shock_vector_i) / reference_exposure
expected_impact = median({impact_i : i in A})
severity = 1 + floor(10 * empirical_CDF_reference(expected_impact))
severity = clamp(severity, 1, 10)
```

For an event-independent UI scale, freeze `empirical_CDF_reference` on a historical training period. For a portfolio-relative score, label it explicitly and keep a second market-impact score; otherwise the same news will change severity merely because holdings changed.

Use hierarchical back-off when cells are thin: `(class, region, sector)` → `(class, region)` → `(class)` → global. Report analogue count, median, 10th/90th percentiles and the back-off level. Preserve each analogue's **joint** FX/rate/equity/spread/commodity vector; separately taking the worst or median of every factor manufactures a scenario that never occurred.

Two ordinal methods are technically available but are secondary:

| Method | Evidence / implementation | Fit |
|---|---|---|
| Proportional-odds ordered logit/probit | [`statsmodels.OrderedModel`](https://www.statsmodels.org/stable/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html), [BSD-3-Clause](https://github.com/statsmodels/statsmodels/blob/main/LICENSE.txt), models an ordered response through latent thresholds. Statsmodels is active and mature, but its own documentation marks this class experimental while noting that core results are verified. | Useful only after the decile target is derived from returns and there are enough examples to estimate stable covariate effects. Pin the version; test the proportional-odds assumption and temporal generalization. |
| Neural ordinal regression | [`dlordinal`](https://github.com/ayrna/dlordinal), BSD-3-Clause, is an actively maintained implementation of CLM, CORN and ordinal losses; the older [`coral-pytorch`](https://github.com/Raschka-research-group/coral-pytorch) reference is MIT and implements rank-consistent thresholds. | **Do not add for the hackathon.** These add training and calibration burden and are newer/less time-tested than ordered logit. They cannot make a subjective target empirically valid. Revisit only with a large, rights-cleared severity dataset. |

The direct empirical-decile mapping is simpler, auditable and naturally monotonic. An ordinal learner is not a substitute for the event-return calibration dataset.

### 3. Historical and hypothetical scenario stress

Use two complementary scenario families:

1. **Historical analogue scenarios:** replay the complete joint factor move from each matched event window. Summarize the resulting portfolio-loss distribution; use the median as expected stress and a high quantile as tail stress.
2. **Hypothetical templates:** deterministic class-specific shocks for events with no credible analogues, with every assumption named. Keep these separate from “empirically backed” severity.

This is the same durable mechanical pattern demonstrated by Strata's historical-scenario example and ORE's user-defined stress analytics: freeze a base market, apply a scenario, revalue, and report `stressed_value - base_value`. For this prototype, implement the small contract rather than integrate either engine:

```text
MarketSnapshot + ShockVector -> StressedSnapshot
Position + MarketSnapshot    -> BaseValue
Position + StressedSnapshot  -> StressedValue
PortfolioPnL                 = sum(StressedValue - BaseValue)
```

Required scenario fields are `factor_id`, shock type (`absolute`, `relative`, `bp`), value, unit, horizon, source event, observation vintage and calibration version. Never mix basis-point, percent and decimal shocks without explicit conversion.

### 4. Confidence calibration

Keep three uncertainties separate:

- **Classification confidence:** calibrate event-class logits on a temporally held-out set. [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html) found temperature scaling to be an effective simple post-processing method. [`CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) directly supports multiclass temperature scaling plus sigmoid and isotonic methods under scikit-learn's [BSD-3-Clause license](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING); its docs caution that isotonic tends to overfit with far fewer than 1,000 calibration samples.
- **Analogue support:** expose `n`, back-off level, similarity range and timestamp/entity quality. Bootstrap the analogue set to show an impact interval, but do not relabel that interval as classifier confidence.
- **Valuation coverage:** disclose the share of portfolio value covered by supported pricing rules and cap overall confidence when material exposures fall back to proxies.

The displayed confidence should answer a declared question. Recommended: “estimated probability that the event class is correct **and** realized severity falls within ±1 bucket.” Fit or validate this score end to end on held-out events. Until enough data exist, show the three components rather than multiply arbitrary quality weights into a pseudo-probability.

### 5. Simple, auditable asset-class valuation

For the hackathon, sensitivity-based revaluation is more defensible than incomplete full pricing:

| Exposure | Stress P&L approximation | Guardrail |
|---|---|---|
| Equity, commodity, FX spot | signed market value × relative shock | Define quote direction and base currency. |
| Fixed-rate bond / loan | `-modified_duration × value × yield_shock + 0.5 × convexity × value × yield_shock²` | Yield shock is decimal; separate risk-free and spread sensitivities where available. |
| Credit spread exposure | `-CS01 × spread_shock_bp` | Add default loss separately; do not infer CS01 from notional. |
| Default / restructuring | `-EAD × LGD` plus recoveries/collateral assumptions | Scenario assumption, not market CAR. Keep accounting-credit loss separate from mark-to-market P&L. |
| Linear derivative | supplied delta/sensitivity × factor move | Refuse unsupported nonlinear instruments rather than guess. |
| Option or nonlinear derivative | delta-gamma-vega approximation or full repricing | Use full repricing only with complete contract and market inputs. |

[`QuantLib`](https://github.com/lballabio/QuantLib) is the time-tested direct-reuse choice when full instrument pricing is truly required: it is a large, actively maintained, non-copyleft library for modelling, trading and risk management, with dedicated duration and pricing code. ORE builds on it. For the judged path, however, project-owned linear and duration/convexity adapters are easier to verify and run on the target laptop. Pin every sign convention and test each adapter against hand-calculated fixtures.

## Recommended hackathon algorithm

1. Classify event text and entity; temperature-scale class probabilities. Abstain below a frozen threshold.
2. Retrieve only historical analogues available before the scoring timestamp, matched by event class, region, sector and exposure type.
3. Compute or load CAR/native-unit risk-factor moves over fixed horizons. Preserve joint shock vectors.
4. Revalue the reference portfolio under every analogue vector using the simple adapters above.
5. Set expected loss to the analogue median; report 10th/90th percentiles and sample size. Map expected loss through a training-period empirical CDF to severity 1–10.
6. Run the selected median and tail scenarios on the actual portfolio; aggregate by global → region → sector → obligor/facility.
7. Render an extractive rationale: class/entity evidence, analogue cohort, `n`, horizon, expected/tail move, portfolio P&L, calibration version and any fallback.

Acceptance checks:

- identical snapshots and model versions produce byte-identical results;
- a zero shock produces zero P&L and increasing adverse shocks never reduce the severity bucket;
- no post-event observation enters analogue selection or expected severity;
- every score links to analogue IDs and every P&L links to a factor, unit and valuation rule;
- benchmark results include class macro-F1, Brier score/calibration curve, severity mean absolute error, ±1-bucket accuracy, interval coverage and portfolio-P&L error;
- live ingestion failure cannot alter the frozen offline demo.

## Reuse summary

| Component | Recommendation | License |
|---|---|---|
| Text-to-event example | Trade the Event concepts only; no code reuse | No repository license |
| Event-impact estimation | Implement classic market-model event study locally; test against published equations | Project license |
| Ordinal interpolation | `statsmodels.OrderedModel` only if data volume supports it | BSD-3-Clause |
| Class-probability calibration | Temperature scaling or scikit-learn calibration | BSD-3-Clause for scikit-learn |
| Scenario/revaluation semantics | Copy Strata/ORE concepts; avoid full integration in ten days | Apache-2.0 / Modified BSD |
| Full pricing, later | QuantLib or ORE | Modified BSD-style / Modified BSD |
| Severity score | Frozen empirical loss deciles with hierarchical back-off | Project-owned method |

**Bottom line:** the time-tested core is not an AI model that emits “8/10.” It is a classic event study feeding historical joint scenarios into transparent revaluation, with the resulting loss distribution mapped to frozen deciles. That is the shortest route to a severity score that is measurable, reproducible and defensible.
