# Financial Risk Engine - S&P Global & Crisil Campus Hackathon

**Candidate Name:** Harsh Upadhyay<br>
**College Email ID:** harsh10.mitmpl2024@learner.manipal.edu<br>
**College / Campus:** Manipal Institute of Technology, Manipal<br>
**Demo Video Link:** To be added before final submission<br>
**Slide Deck Link (if hosted externally):** Not hosted externally; the final deck will be committed at `docs/presentation.pdf`

> Development status: architecture and implementation plan approved; application implementation is in progress.

## 1. Project Overview / Problem Statement & Approach

Financial institutions need to turn fast-moving news and social-media text into risk information that a portfolio analyst can inspect and act on. This project builds a local AI/NLP Risk Engine that ingests two text-source types, links events to portfolio entities, and emits machine-readable Sentiment Score, Event Classification, Impact Score, Confidence, evidence, provenance, and version metadata.

The selected downstream application is Module B: strategic portfolio stress testing. Impact is not assigned by an LLM or a subjective weighting formula. The system uses historical event studies, matched joint market-shock scenarios, a fixed cross-asset reference basket, and empirical severity deciles. High-impact eligible signals can propose an auditable scenario for a synthetic wholesale-banking portfolio, with before/after value and attribution by asset, sector, region, obligor, and factor.

The judged workflow is designed to run locally and offline from immutable snapshots. Online refreshes are optional, no paid API is required, and deterministic financial calculations remain outside the language-model path.

## 2. Architecture & Tech Stack

The application is a single local Python deployment with four core modules:

1. **Data Module:** GDELT news and historical social replay normalized into immutable source snapshots.
2. **Risk Engine:** local entity linking, financial sentiment, fixed-taxonomy event classification, historical analogues, Impact Score, and calibrated Confidence.
3. **Stress Engine:** validated joint factor shocks and transparent asset-specific portfolio valuation.
4. **Backtest Module:** leakage-safe chronological evaluation using the same production interfaces.

The analyst interface will use FastAPI for machine-readable output and Streamlit for Signal Monitor, Portfolio Stress, and Backtest Evidence views.

**Planned stack:** Python 3.12, Pydantic, FastAPI, Streamlit, DuckDB, Polars, NumPy, SciPy, statsmodels, scikit-learn, PyTorch/Transformers, Plotly, pytest, Hypothesis, Ruff, and mypy.

- [Approved system design](docs/superpowers/specs/2026-10-04-financial-risk-engine-design.md)
- [Implementation plan](docs/superpowers/plans/2026-10-04-financial-risk-engine.md)
- [Development, commit, push, and resume workflow](docs/development-workflow.md)
- The final high-resolution diagram will be committed at `docs/architecture.png`.

## 3. Dataset Used

- **News:** GDELT live refreshes and frozen snapshots for deterministic replay.
- **Social text:** project-authored redistributable demonstration posts plus separately acquired FiQA/StockNet evaluation inputs used only under their source terms; restricted raw text will not be committed.
- **Market factors:** committed or reproducibly acquired public/rights-cleared end-of-day factor observations used for event studies and joint shock vectors.
- **Portfolio:** a fixed-seed synthetic portfolio of approximately 60 loans, bonds, and derivatives spanning the United States, Europe, India, and Asia-Pacific.

The project uses no confidential S&P Global, Crisil, client, or proprietary institutional data. Dataset licenses, hashes, transformations, vintages, and assumptions will be recorded under `data/manifests/` and `THIRD_PARTY_NOTICES.md`.

## 4. Quickstart & Installation

Runtime: Python 3.12 on Linux; CPU execution is supported, with optional NVIDIA GPU acceleration.

The executable application is currently under construction. The final evaluator workflow will be:

```bash
git clone git@github.com:h4rshupadhyay/mit-manipal-harsh-upadhyay-hackathon.git
cd mit-manipal-harsh-upadhyay-hackathon
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py
```

The final judged demo will replay a committed snapshot and will not require network access or paid credentials.

## 5. Key Results & Domain Impact

The finished prototype will demonstrate one traceable chain from news/social text to a structured Risk Signal, an evidence-backed joint shock scenario, portfolio revaluation, loss attribution, and historical validation. Impact Score, model Confidence, current-portfolio Materiality, and Action Priority remain separate so analysts can distinguish market severity from reliability and exposure.

Final measured classification, calibration, severity-ranking, stressed-P&L, coverage, and alert-policy results will be reported here and in `docs/results.md` only after the locked chronological evaluation has run. No target metric is presented as an achieved result before that evaluation.

## Submission Artifacts

- Source code and reproducible demo data: this repository
- Architecture diagram: `docs/architecture.png` before final submission
- Presentation deck: `docs/presentation.pdf` before final submission
- Demo video: ten-minute unlisted YouTube link before final submission
- License: [MIT](LICENSE)

## AI Assistance Disclosure

AI-assisted tools are being used for research synthesis, design review, implementation support, testing, and documentation. The candidate remains responsible for the design decisions, source verification, code, evaluation, and final submission. Reused code, datasets, models, and methodological sources will be attributed explicitly.
