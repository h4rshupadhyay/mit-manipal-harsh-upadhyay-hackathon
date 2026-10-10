# Results and evaluation readiness

Recorded 2026-10-10. **Final historical evaluation has not run.** No final BacktestReport, empirical performance claim, or `data/results/final-backtest.json` is published. The implemented production runtime and lock-once protocol do not substitute for the missing data and model artifacts.

## Available illustrative calculations

The committed project-authored snapshot and fictional 60-position Synthetic Portfolio exercise the production `StressEngine`. These simultaneous one-day shocks are hypothetical assumptions, frozen before inspecting their valuation results. They are not observed historical market movements or forecasts.

Run from the repository root with Python 3.12:

```bash
python scripts/run_demo.py --manual-exercise
```

The portfolio base value is USD 390,418,231.60. Values below round actual Decimal outputs to cents for presentation; the [golden JSON](../tests/golden/demo-output.json) retains the complete calculations, versions and attribution.

| Hypothetical scenario | Stressed value (USD) | Loss (USD) | Loss / base value | Supported gross-value share |
| --- | ---: | ---: | ---: | ---: |
| RBI-policy illustration | 388,785,130.72 | 1,633,100.88 | 0.4183% | 100% |
| Analyst override of that illustration | 387,164,626.97 | 3,253,604.63 | 0.8334% | 100% |
| Fictional Indian credit stress | 385,473,384.43 | 4,944,847.17 | 1.2666% | 100% |

All 60 positions are supported for these explicitly supplied sensitivities. Asset, sector, geography, obligor and factor attribution each reconciles to the total loss. This coverage describes the authored exercise inputs, not coverage of arbitrary real portfolios or institutional pricing accuracy. No Risk Signal, Impact Score, Confidence, Action Priority or automatic TriggerDecision is manufactured. The override creates a new scenario and retains the original.

Provenance: [demo manifest](../data/manifests/demo-snapshot.json), snapshot `project-authored-demo-20261007-v1`, manifest content hash `745d26374447b5f9878ebd29a7c82676812254a9114e00862d95dd40b5f157fa`; portfolio SHA-256 `50c7f826257539935dd606542c8d858384af404b996e41eb96a637d5a5e5aab1`. Source text is project-authored on October 7; October 4 is fictional valuation time, not a historical availability assertion. Scenario version `hypothetical-exercise-assumptions-v1`; valuation engine `stress-engine-v2`. Exact input file hashes, units, assumptions and MIT terms are in the manifest.

## Empirical metrics

| Metric family | Status |
| --- | --- |
| Event Class macro-F1, confusion and abstention | Unavailable: no historical evaluation run |
| Entity-link accuracy | Unavailable |
| Confidence Brier score, log loss and reliability | Unavailable |
| Impact Score rank correlation, ordinal error and bucket/interval quality | Unavailable |
| Historical scenario factor-direction and stressed-P&L error | Unavailable |
| Trigger precision/recall, alerts/day and realized loss captured | Unavailable |
| Valuation approximation error against independent full pricing | Unavailable |

Passing software tests establish calculation and protocol behavior on declared fixtures; they do not estimate these metrics. Neither hypothetical CSVs under `data/calibration/` nor acquired sentiment-only datasets are an observed cross-asset calibration bundle.

## Final-evaluation refusal

`python scripts/run_backtest.py --final` exits 2 because explicit dataset and further final arguments are required. A fully shaped invocation against the committed lock root exits 1 with `Backtest refused: unresolved backtest evidence is unavailable; it is not a development lock`. No final output or final-attempt receipt is created; the untouched holdout has not been consumed.

The [selected configuration](../data/calibration/selected-config.json), [Impact cutpoints](../data/calibration/impact-cutpoints.json) and [development evidence](../data/results/development-backtest.json) are immutable **unresolved status envelopes**, not fitted results. The [model declaration](../config/models.lock.json) is also unresolved. Keep these originals; write any future resolved artifacts to a new versioned root.

## Inputs required to run the genuine evaluation

1. Acquire the supported FinBERT and DeBERTa snapshots under ignored local storage, preserve their license notices and immutable revisions, and create a new verified lock with `scripts/lock_models.py`. Validate actual CPU inference.
2. Supply a historical `BacktestDataset` and `RuntimeEvidenceIndex` with source permissions, immutable snapshots, actual availability, event grouping, independently labeled entity/Event Class targets, observed complete joint factor vectors and independent Reference Basket losses. Include real RBI-policy and Indian credit/default cases. Do not backdate new authored data or fill missing factors with zeros.
3. Supply a preregistered `RuntimeDefinition`, `DevelopmentConfiguration` and `EvaluationPeriod` with sufficient chronological groups and embargo. Definition freezing must precede configuration freezing; final outcomes must remain inaccessible during selection. Freeze source terms, unit conventions and candidate grid before evaluation.
4. Bind local locations through `RuntimeLocationConfig`, run chronological development selection to a fresh root, then restore the selected frozen runtime and evaluate once. Any changed definition/data/policy needs a new version and valid untouched holdout.

The following commands are **templates**, not a claim that the named local inputs exist:

```bash
python scripts/run_backtest.py --select \
  --dataset data/local/evaluation/backtest-dataset.json \
  --config data/local/evaluation/development-config.json \
  --period data/local/evaluation/period.json \
  --runtime-config data/local/runtime/locations.json \
  --output-root data/local/evaluation/run-v1
python scripts/run_backtest.py --final \
  --dataset data/local/evaluation/backtest-dataset.json \
  --runtime-config data/local/runtime/locations.json \
  --lock-root data/local/evaluation/run-v1 \
  --output-file data/local/evaluation/run-v1/data/results/final-backtest.json
```

CLI location paths resolve relative to the locations JSON; hashed `LocalArtifact` descriptor paths are absolute and cannot be rewritten. Before final reservation, the CLI verifies the complete artifact set, dataset and frozen runtime prerequisites. Once evaluation is reserved, a failure consumes the attempt and retries are refused.

## Limitations

The exercise has fictional text, market levels and sensitivities; it cannot establish real historical performance, causation, model reliability, optimal trigger thresholds or predictive market usefulness. Loan/bond/derivative valuation uses declared approximations rather than a full pricing library. Thin historical evidence and unsupported exposures must remain visible. See the [approved design](superpowers/specs/2026-10-04-financial-risk-engine-design.md), [MIT project license](../LICENSE), and demo manifest for conventions and input rights.
