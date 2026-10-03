# Existing solutions and free APIs for the risk-engine hackathon

**Decision date:** 2026-10-03  
**Scope:** the five most useful open/public project references and five free/keyless integration bundles for the local risk engine in `model-and-data-selection.md`.  
**Constraint:** the judged demo must run deterministically without network access, paid APIs or remote inference.

## Executive decision

Do not adopt an existing platform end to end. None combines local financial NLP, event-to-market calibration, global wholesale stress propagation, India coverage and a clean offline license story.

Borrow these seams instead:

1. **OpenBB V5:** typed, swappable provider contracts.
2. **FinRobot:** deterministic financial computation separated from model-generated narrative, with provenance.
3. **Trade the Event:** event extraction joined to time-aligned market outcomes.
4. **OpenGamma Strata and Open Source Risk Engine (ORE):** immutable market snapshots, explicit scenarios and hierarchical stress results.
5. **Riskfolio-Lib and QuantStats:** permissively licensed risk calculations and reporting against local return arrays.

The strongest new data integrations are **GLEIF + OpenFIGI**, **GDELT**, **official structured-event feeds**, **BIS/OFR/ECB/World Bank macro factors**, and **NSE/RBI India data**. Each online adapter should produce an immutable raw envelope and normalized snapshot. The application should default to offline replay; refreshing data is a separate operation.

## Five project references worth using

| Project | What is proven or useful | License and risk | Exact decision |
|---|---|---|---|
| [OpenBB V5 / Open Data Platform](https://github.com/openbq-org/OpenBB) | Its “connect once, consume everywhere” design exposes normalized provider data through Python, REST/FastAPI, MCP and dashboards. | The [official license FAQ](https://docs.openbb.co/odp/python/faqs/license) says all V5 files/packages are Apache-2.0; V4 and earlier remain AGPL-3.0. Provider-returned data retain their own terms. The full platform is more machinery than this prototype needs. | **Reuse the design, not the dependency.** Create a small project-owned `fetch → normalize → snapshot → load` interface and canonical schemas. |
| [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) | Its documented principle is that code computes every financial number while the LLM narrates; outputs pass provenance/audit layers. Its research cockpit places source data, deterministic valuation and narrative together. | Apache-2.0, but the current system is a large multi-agent/full-stack application and several workflows require OpenAI, Financial Modeling Prep or other credentials. | **Architecture/UI inspiration only.** Label numeric outputs “code-calculated”; let a local model explain but never alter scores, shocks or P&L. Add an output-contract gate before rendering. |
| [Trade the Event / EDT](https://github.com/Zhihan1996/TradeTheEvent) and [paper](https://aclanthology.org/2021.findings-acl.186/) | Demonstrates corporate-event extraction, argument structure, time alignment and event-driven market prediction. | Repository has no LICENSE and its news originates from Reuters. Code/content cannot be assumed reusable. | **Method inspiration only.** Reimplement the join from normalized event time/entity/class to later abnormal returns; do not copy code or redistribute EDT content without permission. |
| [OpenGamma Strata](https://github.com/OpenGamma/Strata) and [ORE](https://github.com/OpenSourceRisk/Engine) | Strata's [historical-scenario example](https://github.com/OpenGamma/Strata/blob/main/examples/src/main/java/com/opengamma/strata/examples/finance/HistoricalScenarioExample.java) builds dated curve perturbations and calculates a scenario P&L vector. ORE provides trade/market interfaces, stress/sensitivity/VaR outputs and hierarchical reporting. | Strata is Apache-2.0; ORE is Modified BSD and explicitly permits commercial incorporation. Both are mature but far too broad for a ten-day Python NLP demo; direct integration adds Java or C++/QuantLib complexity. | **Copy semantics, not engines.** Use immutable `MarketSnapshot`, `ScenarioSet`, `ShockVector` and result rows keyed by `(scenario, exposure, factor, measure, unit)`. Revisit these engines only when pricing actual derivatives. |
| [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) and [QuantStats](https://github.com/ranaroussi/quantstats/blob/main/README.md) | Riskfolio supplies VaR/CVaR, drawdown and asset/factor risk contributions; QuantStats separates statistics, plots and HTML reports. | BSD-3-Clause and Apache-2.0 respectively. Advanced Riskfolio optimizations may need commercial solvers; QuantStats convenience downloads inherit external data terms. | **Selective code reuse.** Use only pure risk/contribution and report functions tested against fixed local arrays. Never call their download helpers. Do not add optimization to the demo. |

### Explicit non-selections

- FinGPT/FinNLP and Qlib are valuable research references but add large model/data/workflow surfaces without improving the selected encoder-first path in ten days.
- The Python `eventstudy` package is GPL-3.0 and leaves some statistical tests unimplemented; the required abnormal-return calculation is small enough to implement and validate locally.
- VectorBT uses Apache-2.0 plus the Commons Clause, and `backtesting.py` is AGPL-3.0. Avoid their licensing complexity. If a trading-strategy backtest becomes mandatory, MIT-licensed [`bt`](https://github.com/pmorissette/bt) can consume local prices, but it is not part of the severity engine.

## UI and architecture to borrow

Build one auditable event-to-portfolio flow rather than a generic trading terminal:

1. **Risk event queue:** timestamp, source, linked entity, region, event class, sentiment, expected severity and confidence. Mark India and portfolio-linked rows visibly. Severity and confidence must use different encodings.
2. **Evidence drawer:** source span, canonical identifier, alternative entity candidates, model revision, abstention/duplicate flags and structured-source metadata. Explain “why this entity” separately from “why this class.”
3. **Analogue panel:** historical abnormal-move distribution, sample count, median/tail, chosen window and expected versus realized severity. This is the proof behind 1–10.
4. **Scenario matrix:** rows by region/sector/obligor or facility; columns by equity, rates, FX, spread, commodity and volatility shock. Every value carries a unit and provenance.
5. **Portfolio waterfall and hierarchy:** global → region → country → sector → obligor/facility, plus an “as of” replay control and a prominent `LIVE` or `SNAPSHOT <timestamp>` badge.

OpenBB's linked parameters are the right interaction pattern: changing entity, date or scenario once should synchronize every panel. FinRobot supplies the trust boundary, and Strata/ORE supply the scenario/baseline vocabulary.

## Five API integration bundles

### 1. Legal entity and instrument resolution: GLEIF → OpenFIGI

The [GLEIF API](https://www.gleif.org/en/lei-data/gleif-api/) provides fuzzy legal-name/address search, LEIs, BIC/ISIN mappings and direct/ultimate parent relationships. GLEIF also publishes full and delta Golden Copy files; its [terms release LEI data under CC0](https://www.gleif.org/en/meta/lei-data-terms-of-use). No numeric public quota or SLA should be assumed.

The [OpenFIGI v3 API](https://www.openfigi.com/api/documentation) maps ticker/ISIN/CUSIP plus exchange context to instrument, composite and share-class FIGIs. No key is required; unauthenticated mapping is limited to 25 requests/minute. The official page is inconsistent on five versus ten mapping jobs per unauthenticated request, so implement the stricter five. Its [official FAQ](https://www.openfigi.com/docs/faqs) describes FIGI symbology as free of reuse fees.

**Integration:** GLiNER2 finds spans; a local portfolio alias table resolves known names; GLEIF validates the legal entity/corporate family; OpenFIGI resolves the traded instrument. Cache the ordered candidates, chosen match and reason. A replay never remaps. A FIGI match must not silently override contradictory LEI/legal-name evidence.

### 2. Global news/event discovery: GDELT

[GDELT 2.0](https://gdeltproject.org/data.html) is keyless, global and multilingual, with 15-minute updates, CAMEO events, GKG themes/entities/tone and article URLs. It materially improves geopolitical, protest, conflict and location coverage.

Its event coding is machine-generated and noisy; copied stories are common, and access to metadata does not grant rights to republish underlying publisher articles.

**Integration:** archive exact 15-minute files or exact JSON responses during ingest; store source URL, short permitted snippet, publication/retrieval time and content hash; cluster duplicates before local classification. CAMEO, Goldstein and GKG tone are priors/features, never labels of truth or final severity.

### 3. Official structured-event feeds: SEC, Federal Register, OFAC, GDACS and CISA

These feeds reduce classifier ambiguity more than another remote NLP API:

- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) are keyless and provide issuer submissions/XBRL. SEC guidance imposes a [10 requests/second fair-access ceiling](https://www.sec.gov/about/webmaster-frequently-asked-questions) and requires a declared user agent. Form type and accession ID are deterministic metadata, not substitutes for classifying the relevant text.
- The [Federal Register API](https://www.federalregister.gov/developers/documentation/api/v1) is keyless JSON/CSV for rules, proposed rules, notices and agencies. Only the first 2,000 matching results are paginable, so backfills must be partitioned by date; important events should retain the linked official document because the API rendition is not legally controlling.
- [OFAC's Sanctions List Service](https://ofac.treasury.gov/sanctions-list-service) publishes keyless advanced XML/CSV with aliases, addresses and identifiers. Automated clients need a user agent. A fuzzy name hit is not proof of designation.
- [GDACS](https://www.gdacs.org/Documents/2025/GDACS_API_quickstart_v1.pdf) provides free GeoJSON/API and RSS disaster alerts; feeds update about every six minutes and require source acknowledgement. Disaster alert level is evidence, not market-loss severity.
- [CISA's Known Exploited Vulnerabilities Catalog](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) offers keyless JSON/CSV/schema for vulnerabilities observed in active exploitation. It identifies vendor/product risk, not the affected borrower directly.

**Integration:** preserve official type, identifier, effective/date-added fields and file hashes as high-provenance features. Map GDACS geometry and CISA vendor/product to exposures through reviewed local tables. Conflicts with the local event classifier reduce confidence; they are never overwritten silently.

### 4. Global wholesale and macro factors: BIS + OFR + ECB + World Bank

- The [BIS SDMX API](https://stats.bis.org/api-doc/v2/) is keyless and covers cross-border banking claims, credit, debt securities, property prices and exchange rates. No numeric quota is published. Its quarterly/lagged series describe vulnerability and analogue context, not live market reaction.
- The [OFR Short-term Funding Monitor API](https://www.financialresearch.gov/short-term-funding-monitor/api-specs/api-full-dataset/) is keyless and exposes Treasury constant maturities, repo, primary-dealer, money-market-fund and NY Fed reference-rate series.
- The [ECB Data Portal API](https://data.ecb.europa.eu/help/api/data-examples) is a keyless official SDMX source for euro-area rates, FX, monetary and banking series.
- The [World Bank Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392) needs no key and covers nearly 16,000 indicators. World Bank-produced datasets use licenses listed in its [official public-license catalog](https://datacatalog.worldbank.org/public-licenses), but dataset-specific terms control.

**Integration:** predeclare a small series manifest. Snapshot observations, data-structure definitions/codelists, units, provider vintage and revision time. Use BIS/World Bank for country/wholesale context and OFR/ECB for observable rate/FX/funding factors. Do not perform exploratory series searches during the demo.

### 5. India market and banking context: NSE + RBI

The official [NSE MCP](https://www.nseindia.com/nse-mcp) is keyless and supplies bhavcopy history plus delayed market data. Its terms limit use to informational/non-commercial purposes and prohibit commercial exploitation and model training/fine-tuning.

Official [RBI data releases](https://statistics.rbi.org.in/) cover reference rates, money-market operations, bank credit, external debt, international banking and payments. The DBIE [FAQ permits research use with attribution](https://dbieold.rbi.org.in/DBIE/doc/Frequently%20Asked%20Questions%20-%20DBIE.pdf), but no stable documented public API should be assumed.

**Integration:** use NSE for the hackathon's approved equity/index basket and RBI CSV/XLSX exports for INR, rates, liquidity and banking context. Freeze both before judging and display provider/vintage. Keep the adapter boundary so a commercial continuation can replace NSE data with a licensed feed.

## Market-price gap

No reviewed free/keyless service provides authoritative, production-grade, globally broad equity and bond prices with uncomplicated reuse rights. Do not disguise this gap with an unofficial scraper. Use rights-cleared/user-supplied EOD snapshots for the hackathon, official NSE data under its stated restrictions for India, and official reference-rate/FX series from OFR/ECB/BIS. A later product needs a licensed market-data provider behind the same interface.

## Deterministic offline integration

Use a small OpenBB-like boundary:

```text
Provider.fetch(request) -> RawEnvelope
Normalizer.normalize(RawEnvelope) -> CanonicalRows
SnapshotStore.write(raw, canonical, manifest)
Repository.load(snapshot_id) -> CanonicalRows
```

Every raw envelope records provider, canonical request, retrieval/effective time, response metadata, source terms URL and SHA-256. Every snapshot manifest records schema and normalizer versions, observation vintage, model revisions and hashes of all files. Include raw article content in a distributable bundle only when rights permit; otherwise keep metadata, permitted snippets, hashes and source URLs.

`APP_DATA_MODE=offline` should be the default and must hard-fail if a referenced artifact is missing. A separate `refresh` operation may access the network and fail without affecting the demo. Never silently fall back from a missing snapshot to live data.

Acceptance gates:

- the same raw response normalizes twice to byte-identical canonical output;
- every displayed value resolves to source, observation/publication time, retrieval time and snapshot ID;
- entity-resolution winners and alternatives are frozen with reasons;
- rate-limit/429 and schema-change fixtures are tested;
- the full judged route passes with networking disabled;
- hackathon-only/non-commercial sources are tagged so they cannot enter a future commercial export unnoticed.

## Final recommendation

Implement the provider/snapshot boundary, scenario-result schema and auditable UI; selectively reuse Riskfolio-Lib and QuantStats only after fixed-fixture checks. Keep large agent, quant and derivatives platforms outside the runtime.

Prioritize GLEIF/OpenFIGI, GDELT, structured official feeds, BIS/OFR/ECB/World Bank and NSE/RBI. The demo should appear live-capable but be snapshot-replayable. A visible snapshot badge and complete provenance are strengths: judges can reproduce every entity link, score, shock and portfolio impact even when every external service is unavailable.
