# Model and data selection for a local portfolio-risk engine

**Decision date:** 2026-10-03  
**Scope:** 10-day solo hackathon; Ryzen 7 5800H, 16 GB RAM, RTX 3050 45 W (assume 4 GB VRAM), CPU fallback; no paid APIs.  
**Downstream use:** stress testing a global wholesale portfolio, including an India region.

## Executive decision

Build an encoder-first, evidence-backed pipeline rather than asking one small generative model to invent all five outputs.

| Function | Hackathon choice | Why |
|---|---|---|
| Financial-news sentiment | `ProsusAI/finbert` | Mature 3-way finance classifier, roughly BERT-base size, local CPU/GPU inference, Apache-2.0 repository. Its paper reports better results than prior methods on Financial PhraseBank and FiQA. |
| Social sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | A source-domain-specific RoBERTa-base model trained on about 124M tweets; CC-BY-4.0. |
| Entity and relation extraction | `fastino/gliner2-base-v1` | One 205M-parameter, Apache-2.0 encoder can extract configurable entities, relations and JSON locally, including on CPU. |
| Event class | A two-stage router: `ritessshhh/FinDeBERTa` for its 18 corporate-event labels; `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` for zero-shot macro/sovereign labels | The corporate model is directly relevant but incomplete; the NLI model fills policy, geopolitical, climate and India-specific classes without a 10-day labeling project. Both are local and permissively licensed. |
| Severity 1–10 | Deterministic event-study calibration, not an LLM | Severity should be a percentile of historical abnormal moves in the affected risk-factor basket, with sample size and uncertainty retained. |
| Confidence | Held-out calibration plus evidence-quality penalties | Raw softmax/sigmoid is not a trustworthy confidence score. Temperature-scale it and reduce it for weak entity links, conflicting models, poor source provenance or sparse analogues. |
| Rationale | Extractive template using source spans, linked entity, event label and historical analog statistics | Reproducible and auditable. A 1.5B LLM may rewrite this text only after the structured result is fixed. |

This fits the laptop if models are loaded sequentially or placed selectively on GPU. All chosen encoders are hundreds of millions of parameters or less. Keep only the active model on the 4 GB GPU; CPU inference remains a workable fallback. An optional 4-bit `Qwen2.5-1.5B-Instruct` also fits, but it is not in the accuracy-critical path.

The most important qualification is licensing. The Prosus repository is Apache-2.0, but its sentiment checkpoint was fine-tuned on Financial PhraseBank, whose distributed dataset is marked CC-BY-NC-SA-3.0. That is acceptable for a non-commercial hackathon, not a clean commercial provenance story. For a product follow-on, replace it with an Apache/MIT base encoder fine-tuned on CC-BY SENTiVENT plus newly labeled organization-owned examples, or obtain permission for the original corpus.

## Recommended output contract

Do not collapse different kinds of uncertainty into one unexplained number. Persist enough evidence to reproduce the displayed answer:

```json
{
  "sentiment": -0.72,
  "event_class": "sovereign_or_credit_deterioration",
  "severity": 8,
  "confidence": 0.81,
  "rationale": "Rating downgrade of linked issuer X; comparable events had median 2-day adverse abnormal move of 2.4% and 90th-percentile move of 5.1% (n=47).",
  "entities": [{"text": "...", "type": "issuer", "canonical_id": "LEI..."}],
  "event_time": "...",
  "source_url": "...",
  "model_versions": {"sentiment": "revision SHA", "event": "revision SHA"},
  "calibration_version": "...",
  "flags": ["social_source", "thin_history"]
}
```

The public score can remain compact, while the evidence fields make the engine testable and defensible.

## Model choices

### Financial sentiment

| Candidate | Evidence and fit | Runtime / license | Verdict |
|---|---|---|---|
| [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) | Finance-adapted BERT fine-tuned on Financial PhraseBank; the accompanying paper evaluates both PhraseBank and FiQA | BERT-base scale; repository Apache-2.0, with a non-commercial training-data caveat | Primary news model because the evaluation trail is clearest; re-test on live-like data. |
| [`yiyanghkust/finbert-tone`](https://huggingface.co/yiyanghkust/finbert-tone) | Finance BERT pretrained on 4.9B tokens of 10-K/10-Q filings, earnings calls and analyst reports, then fine-tuned on 10,000 manually labeled analyst-report sentences | BERT-base scale; [official repository Apache-2.0](https://github.com/yya518/FinBERT/blob/master/LICENSE), but the model card does not document redistribution rights for the underlying analyst corpus | Strong challenger for filings/analyst-style prose; benchmark rather than ensemble by default. |
| [`mrm8488/distilroberta-financial`](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis) | DistilRoBERTa fine-tuned on the easiest all-agreement PhraseBank subset; card-reported accuracy is not a live-domain estimate | 82.1M parameters; Apache-2.0 | Latency/CPU baseline. |
| [`cardiffnlp/twitter-roberta-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | Social-domain model trained on about 124M tweets and TweetEval | RoBERTa-base; CC-BY-4.0 | Route social text here; do not replace finance-news model globally. |

#### Primary: ProsusAI FinBERT

[`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) outputs positive, negative and neutral probabilities and was further pretrained on financial text before Financial PhraseBank fine-tuning. The [FinBERT paper](https://arxiv.org/abs/1908.10063) reports improvements on both Financial PhraseBank and FiQA versus the compared prior methods. The [official repository license is Apache-2.0](https://github.com/ProsusAI/finBERT/blob/master/LICENSE).

Use the continuous score

```text
sentiment = P(positive) - P(negative)
```

which is naturally bounded in `[-1, 1]`. Temperature-scale the logits on a held-out, source-stratified set before taking the difference. FiQA is especially useful as an evaluation target because the official task defined aspect sentiment directly on a continuous `[-1, 1]` scale; however, [FiQA states that its train and test data are non-commercial](https://sites.google.com/view/fiqa/home).

Risks:

- Financial PhraseBank contains only 4,846 sentences and its [distributed copy is CC-BY-NC-SA-3.0](https://huggingface.co/datasets/takala/financial_phrasebank/tree/main/data). Do not infer commercial model rights solely from the code license.
- It was built for financial prose, not short, ironic or ticker-heavy social posts.
- A document can mention several entities with different implications. Score the sentence/clause around each linked entity rather than assigning one article-level polarity to all names.

#### Social route: Twitter-RoBERTa

[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) is a RoBERTa-base model trained on about 124M tweets from 2018–2021 and fine-tuned on TweetEval. Its model card declares CC-BY-4.0. Route Bluesky/Mastodon/tweet-like text here, then calculate the same `P(pos)-P(neg)` score. It understands social style better than FinBERT, but it is not finance-specific, so evaluate finance slang, cashtags, sarcasm and India-specific company aliases separately.

#### Lightweight alternative

[`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis) is an 82.1M-parameter Apache-2.0 alternative. Its card reports 98.23% accuracy, but that result is on the small `sentences_allagree` Financial PhraseBank configuration and the card leaves intended-use and data details incomplete. Treat it as a latency baseline, not evidence that it will achieve 98% on live news or social text.

### Event classification

The target taxonomy should reflect stress factors rather than a news publisher's section names. Recommended top-level classes are:

1. earnings / profit warning;
2. rating change / default / restructuring;
3. financing / liquidity / capital action;
4. merger, acquisition or ownership change;
5. legal, fraud, sanctions or regulatory action;
6. cyber, operational outage or facility disruption;
7. supply-chain / trade disruption;
8. monetary policy / rates / liquidity;
9. inflation / growth / labour macro shock;
10. sovereign / fiscal / political instability;
11. geopolitical conflict / security shock;
12. foreign-exchange / capital-control event;
13. commodity / energy shock;
14. climate, natural-disaster or health shock;
15. other / insufficient evidence.

Preserve more specific child labels where available. These top-level classes can map consistently to wholesale exposures, regions, sectors and risk-factor shock templates.

#### Corporate branch: FinDeBERTa

[`ritessshhh/FinDeBERTa`](https://huggingface.co/ritessshhh/FinDeBERTa) is a DeBERTa-v3-large multi-label classifier with 18 financial headline labels, including rating, financing, legal, macroeconomics, M&A, profit/loss and revenue. Its MIT model card self-reports macro-F1 0.692, micro-F1 0.691 and exact match 0.532 using per-class thresholds. It is directly useful for a demo, but has only a small adoption footprint, is optimized for headlines, omits several required macro-risk classes, and its 0.4B FP32 checkpoint is heavier than the other encoders. Benchmark it; do not treat its self-reported score as independently reproduced.

#### Macro and gap-filling branch: NLI zero-shot

[`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) is MIT-licensed and trained on 763,913 NLI pairs from MultiNLI, FEVER-NLI and ANLI. Use descriptive hypotheses such as “This text reports a central-bank tightening event” rather than one-word labels. It is about 370 MB in the referenced checkpoint and substantially easier to run than a generative LLM.

The zero-shot probabilities are not calibrated event probabilities. During the hackathon, manually label a temporally separated, source-balanced gold set (minimum useful target: about 250–400 items, including India) and select label wording and thresholds only on a development portion. Report macro-F1, per-class precision/recall, coverage at the confidence threshold and an “other/abstain” rate.

#### Candidate benchmark, not primary truth

The [SENTiVENT dataset](https://huggingface.co/datasets/GillesJacobs/sentivent) provides expert-annotated company events, arguments, sentiment and cross-event links under CC-BY-4.0. It has only 288 documents in its unified document configuration (228/30/30 train/dev/test), so it is valuable for evaluation and small-domain adaptation rather than training a broad global risk taxonomy from scratch.

### Does a small generative LLM add accuracy?

Not by default. The evidence does not justify putting one in the critical classification path:

- [FinBen](https://arxiv.org/abs/2402.12659) found that instruction tuning can help foundational financial tasks, but all tested LLMs lagged traditional methods on forecasting; it also reports instruction-following problems for models below 70B in its trading setup.
- A 1.5B model is flexible but has no finance-event calibration and can produce plausible unsupported rationales.
- Encoder classifiers are faster, deterministic at inference, easier to calibrate and easier to evaluate per class on this hardware.

If a generative model is desired, use [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct): it is Apache-2.0, has 1.54B parameters, supports structured output, and can run 4-bit locally. Restrict it to one of two roles:

1. rewrite an already-fixed structured result into a short rationale, with no permission to change scores; or
2. act as a second opinion only for abstained/low-confidence items.

Add it to classification only if a blinded, temporally held-out comparison shows a meaningful macro-F1 or selective-risk improvement after accounting for latency. “Looks better on examples” is not sufficient.

### Entity extraction and linking

[`fastino/gliner2-base-v1`](https://huggingface.co/fastino/gliner2-base-v1) is the best hackathon fit: Apache-2.0, 205M parameters, designed for CPU inference, and able to combine configurable NER, relation extraction, classification and schema-based JSON. Ask it for narrow operational types:

- issuer, borrower, bank, sovereign, regulator and parent company;
- country, region, facility and affected market;
- currency, commodity, security, amount and percentage;
- event trigger, effective date, counterparty and direction of impact.

[`urchade/gliner_small-v2.1`](https://huggingface.co/urchade/gliner_small-v2.1) is a simpler Apache-2.0 166M alternative with a more established NER-only interface. A conventional spaCy model is a useful deterministic baseline, but its fixed ORG/GPE categories require more rules for this use case.

NER alone is not enough. Resolve aliases and identifiers with deterministic dictionaries and authoritative sources:

- [GLEIF's API](https://www.gleif.org/en/lei-data/gleif-api/) supports fuzzy entity-name/address search, LEI/BIC/ISIN mappings and parent/child corporate relationships.
- [SEC EDGAR's submissions API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) supplies company name, former names, exchange and ticker metadata without an API key.
- [NSE's security master downloads](https://www.nseindia.com/static/market-data/securities-available-for-trading) provide Indian symbols, ISINs, debt instruments and defaulted debt-security lists.

Keep the original span, normalized name, canonical identifier and match score. Entity-link confidence should directly limit final confidence; an accurate event attached to the wrong obligor is a bad risk signal.

## Empirically backed severity

### Why language-model severity is rejected

An LLM score of “8/10” is not empirically meaningful. The classic event-study formulation measures an event's effect through security prices around the event; [MacKinlay's event-study survey](https://www.bu.edu/econ/files/2011/01/MacKinlay-1996-Event-Studies-in-Economics-and-Finance.pdf) describes the method and its complications. Use that structure to turn observed historical stress into a reproducible score.

### Proposed calibration

For every historical event:

1. Link the event to an entity, sector, country and event class.
2. Select the relevant observable risk-factor basket: issuer/sector equity, regional bank index, sovereign yield, FX rate, credit proxy and commodity proxy as applicable.
3. Estimate normal return from a pre-event window using the simplest supported benchmark (market-adjusted return first; a factor model only when data is sufficient).
4. Calculate adverse cumulative abnormal return (CAR) over `[0,1]`, `[0,2]` and `[0,5]` trading-day windows. For yields/spreads, use adverse basis-point changes; for FX, define the adverse direction from the portfolio exposure.
5. Winsorize only for robustness reporting, not silently. Store both raw and robust measures.
6. Within `(event class, region, exposure type, horizon)`, rank the adverse move against historical analogues using an expanding window.

Recommended score:

```text
impact_stat = max(
  percentile(adverse_2d_move),
  percentile(adverse_5d_move),
  percentile(cross_factor_stress_index)
)

severity = clamp(ceil(10 * impact_stat), 1, 10)
```

For a new event before returns are observed, use the median historical `impact_stat` for matched analogues, adjusted only by measurable modifiers (issuer rating/size, country, event novelty, amount relative to assets, number of affected facilities). Return the analogue count and an interval. After markets move, keep separate `expected_severity` and `realized_severity` fields rather than overwriting history.

For wholesale stress testing, also output the underlying shock vector—for example INR depreciation, India sovereign-yield widening, bank-equity shock and oil move. A single 1–10 number is useful for display, but the portfolio engine needs factor shocks to propagate to obligors and exposures.

GDELT's Goldstein scale can be a geopolitical prior, not the final market severity. GDELT uses more than 300 CAMEO event types and supplies the Goldstein ranking in its event files, but it measures the event taxonomy's cooperative/conflict intensity rather than portfolio loss; see the [GDELT data and codebook index](https://gdeltproject.org/data.html).

### Confidence

Neural classifiers are commonly miscalibrated; [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) found temperature scaling to be an effective simple post-processing method. Fit calibrators on held-out data and report expected calibration error and Brier score, not only accuracy/F1.

A transparent composite is preferable to another learned black box:

```text
confidence = calibrated_event_probability
             × entity_link_quality
             × source_reliability
             × analogue_support
             × model_agreement
```

Cap confidence when the event is out-of-taxonomy, the article lacks a recoverable timestamp, source copies disagree, the entity is ambiguous, or fewer than a chosen number of analogues exist. The rationale should name the cap (“thin history: n=6”), not hide it.

## Historical datasets

| Dataset | What it contributes | Rights and limitations | Decision |
|---|---|---|---|
| [SENTiVENT](https://huggingface.co/datasets/GillesJacobs/sentivent) | Expert company-event triggers, arguments, sentiment and links; permissive CC-BY-4.0 | Small: 288 unified documents; company-news focus | Use for event/extraction evaluation and a small fine-tune. |
| [EDT / Trade the Event](https://github.com/Zhihan1996/TradeTheEvent) and its [paper](https://aclanthology.org/2021.findings-acl.186) | 9,721 token-level event-annotated articles plus 303,893 time-stamped news items with price labels; directly demonstrates event-driven backtesting | Repository exposes no LICENSE; news originates from Reuters | Research benchmark only unless rights are cleared. Do not redistribute. |
| [FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) and its [paper](https://arxiv.org/abs/2402.06698) | 29.7M price rows and 15.7M time-aligned news records for 4,775 US companies, 1999–2023 | Repository LICENSE remains CC-BY-NC; articles aggregate third-party publishers. README statements and the formal license are not fully aligned | Strong offline research/backtest asset; do not make it a product dependency. |
| [StockNet](https://github.com/yumoxu/stocknet-dataset) and its [paper](https://aclanthology.org/P18-1183/) | Tweets plus daily price movements for 88 US stocks; useful for social-domain drift tests | Repository has an MIT license, but source-platform and Yahoo data terms still apply; 2014–2016 is stale | Evaluation only; do not use as live-ground-truth evidence. |
| [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) | Standard 3-way financial sentiment benchmark | 4,846 sentences; CC-BY-NC-SA-3.0; high-agreement subsets are easier than live text | Reproduce published baselines, but add a live-like holdout. |
| [FiQA 2018](https://sites.google.com/view/fiqa/home) | Aspect-level finance sentiment on `[-1,1]`, including news and microblogs | Official site says training/testing data are non-commercial | Excellent score-shape evaluation for the hackathon. |
| [MultiFin](https://github.com/RasmusKaer/MultiFin) | 10,048 headlines in 15 languages with 23 low-level and 6 high-level topics | CC-BY-NC-4.0; topic labels are not a full event-risk taxonomy | Optional multilingual/category benchmark, not primary training. |
| [GDELT](https://gdeltproject.org/data.html) | Global machine-coded CAMEO events from 1979, GKG themes/entities/tone, 15-minute updates in v2 | Machine-coded labels are noisy; article copyright remains with publishers; attribution required | Main global historical/live event spine; validate against source URLs and deduplicate. |

No single dataset cleanly joins global wholesale-risk events, obligor links, India, and licensed multi-asset returns. The defensible solution is to retain dataset provenance and construct a project-specific calibration table, not merge everything into an undocumented training CSV.

## Live and keyless ingestion

| Source | Access | Best use | Caveat |
|---|---|---|---|
| [GDELT 2.0 Events/GKG and DOC API](https://gdeltproject.org/data.html) | Keyless; raw files/API; v2 updates every 15 minutes | Global multilingual news discovery, event candidates, volume and source corroboration | Store metadata, hashes, short snippets and URLs; do not assume rights to republish full articles. Machine-coded events require validation. |
| [SEC EDGAR data APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | No authentication/API key; real-time submissions/XBRL plus nightly bulk | 8-K, 10-Q, 10-K, 6-K filings and entity metadata | Declare a user agent and respect the [10 requests/second fair-access ceiling](https://www.sec.gov/about/webmaster-frequently-asked-questions). |
| [Federal Register API](https://www.federalregister.gov/developers/documentation/api/v1) | Keyless JSON/CSV | US rules, notices and regulatory actions since 1994 | US-only; classify proposed versus final/effective action. |
| [Fed RSS](https://www.federalreserve.gov/feeds/feeds.htm), [ECB RSS](https://www.ecb.europa.eu/home/html/rss.en.html), [SEC RSS](https://www.sec.gov/about/rss-feeds) | Keyless official feeds | High-provenance policy, enforcement and filing triggers | Narrow scope; RSS timestamps and revisions need normalization. |
| [OFAC Sanctions List Service](https://ofac.treasury.gov/sanctions-list-service) | Keyless machine-readable lists and changes | Sanctions/party-screening events | A list match is not automatically an issuer match; entity resolution is mandatory. |
| [Bluesky Jetstream](https://github.com/bluesky-social/jetstream-legacy) | Public WebSocket JSON firehose; filter `app.bsky.feed.post`; official public instances | Keyless live social signal | High noise, manipulation risk and changing infrastructure. Never let post count alone create high severity. |
| RBI/SEBI/NSE official releases and feeds where available | Keyless web/RSS/downloads | India policy, enforcement, corporate announcements | Interfaces are less uniform; archive raw metadata and respect each site's terms. |

For a 10-day build, prioritize GDELT + official regulatory/central-bank feeds and add Bluesky only after the evaluation harness works. Social volume is visually attractive but consumes disproportionate normalization and abuse-detection time.

## Market and macro data

There is no robust, broad, keyless, first-party global listed-equity feed here with uncomplicated production rights. Keep market data behind a provider interface and distinguish hackathon access from deployable access.

### Recommended for the demo

- **India:** NSE now publishes an [official MCP endpoint](https://www.nseindia.com/nse-mcp) with five years of bhavcopy history and market data delayed roughly 1–3 minutes. It is keyless and highly convenient, but NSE explicitly limits the data to informational/non-commercial use and prohibits commercial exploitation or model training/fine-tuning. Use it for the hackathon demo and document the restriction. NSE also exposes [historical reports and index archives](https://www.nseindia.com/static/resources/historical-reports-capital-market-daily-monthly-archives).
- **US historical event calibration:** use the static FNSPID/EDT research datasets where permitted, with their restrictions preserved.
- **Global/macro factors:** use [ECB's keyless SDMX API](https://data.ecb.europa.eu/help/api/data-examples) for euro-area/FX series, official central-bank/RBI series for local factors, and a free-key FRED adapter for macro/rates. [FRED requires a registered API key](https://fred.stlouisfed.org/docs/api/api_key.html), but it is not a paid API.
- **Small ad-hoc price pulls:** Alpha Vantage's free key covers many endpoints but is limited to [25 requests per day](https://www.alphavantage.co/support/api-key/); cache all responses. Nasdaq Data Link permits [up to 50 unauthenticated calls per day](https://docs.data.nasdaq.com/docs/r-installation), though the useful free datasets must be checked individually.

### Demo-only fallback

[`yfinance`](https://github.com/ranaroussi/yfinance/blob/main/README.md) is keyless and convenient, but its own README says it is unaffiliated with Yahoo, intended for research/education, and that Yahoo data is for personal use. Put it behind `DEMO_MARKET_DATA=true`, cache downloaded observations with timestamps, and never describe it as a production market-data source.

### India calibration basket

At minimum, preserve separate shocks for:

- NIFTY 50 and NIFTY Bank / relevant sector index;
- USD/INR;
- Indian sovereign yields and the RBI policy rate/liquidity variables;
- oil and any commodity materially linked to the obligor/sector;
- an equity-volatility proxy;
- global USD/rates/risk-off factors.

Choose the basket from portfolio exposure, not from whichever series is easiest to download. For example, an RBI tightening event may be mildly positive for one lender's margin but adverse for leveraged borrowers and rate-sensitive sectors; one article-level sentiment number cannot replace exposure propagation.

## Reproducibility and evaluation gates

1. Pin every Hugging Face model by commit revision and record tokenizer/config hashes.
2. Save raw input hash, source URL, publication time, retrieval time and normalized text. Do not silently replace revised articles.
3. Split validation/test data chronologically and by story cluster to prevent near-duplicate leakage.
4. Report sentiment macro-F1 plus MSE/R² for continuous score; event macro/micro-F1 plus per-class precision/recall; NER span-F1 and entity-link accuracy; confidence ECE/Brier; severity rank correlation and interval coverage.
5. Evaluate news, filings and social separately. Aggregate scores can hide a failed source route.
6. Establish abstention gates. For a risk engine, correct selective coverage is more useful than forcing a class for every post.
7. Store the analogue events used for every severity prediction so judges can inspect why an 8 differs from a 4.
8. Compare CPU latency and peak RAM as well as GPU latency. The stated GPU is not guaranteed to have spare VRAM in a demo environment.

## Ten-day scope

The accuracy-maximizing scope is deliberately narrow:

- **Days 1–2:** freeze taxonomy/output schema; collect and hand-label the chronological gold set; wire GDELT plus official feeds.
- **Days 3–4:** integrate entity extraction/linking and the two sentiment routes; build repeatable evaluation.
- **Days 5–6:** benchmark FinDeBERTa and NLI event routing; tune thresholds on development data only.
- **Days 7–8:** build the event-study calibration table and India/global shock baskets; expose analogue evidence.
- **Day 9:** calibrate confidence, add abstention and run CPU/GPU profiling.
- **Day 10:** freeze model/data revisions, run the untouched test set, prepare the demo. Add Qwen rationale rewriting only if every core gate already passes.

## Final recommendation

For this hackathon, use **FinBERT + Twitter-RoBERTa + GLiNER2 + FinDeBERTa/NLI routing**, backed by **SENTiVENT, GDELT and time-aligned research datasets**, and compute severity through an **event-study percentile model over explicit exposure-factor baskets**. Use **GDELT, EDGAR, Federal Register, central-bank/regulator feeds, OFAC and optionally Bluesky Jetstream** for live data. Use the **official NSE MCP for the India demo under its non-commercial terms**, free-key/cached providers for missing factors, and isolate `yfinance` as a clearly marked demo fallback.

Do not let the optional 1.5B generative model set sentiment, class, severity or confidence. The largest accuracy gain available in ten days will come from a clean taxonomy, entity linking, a small high-quality chronological gold set, calibrated abstention and empirical return mapping—not from adding free-form generation.
