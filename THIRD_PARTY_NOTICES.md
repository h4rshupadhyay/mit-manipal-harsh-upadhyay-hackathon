# Third-party notices and data rights

The repository's [MIT license](LICENSE) covers project-authored code, documentation, and the committed fictional demonstration data described in [its manifest](data/manifests/demo-snapshot.json). It does not relicense upstream packages, model weights, publisher articles, social posts, or market observations. Version pins are in [requirements.txt](requirements.txt) and [pyproject.toml](pyproject.toml); installing them also installs transitive packages with their own notices. The summary below records installed direct-package metadata inspected on 2026-10-10 and is not a substitute for the license files distributed with each package.

| Direct packages | Reported license family or expression | Use here |
| --- | --- | --- |
| DuckDB, FastAPI, mypy, Polars, Plotly, Pydantic, pytest, Ruff | MIT | Storage, API, type/test tooling, analysis and display |
| HTTPX, scikit-learn, statsmodels, Uvicorn | BSD 3-Clause | HTTP, calibration/statistics, API server dependency |
| SciPy | BSD family; distribution contains additional notices | Numerical methods |
| Streamlit, Transformers | Apache 2.0 | Dashboard and local NLP library |
| Hypothesis | MPL 2.0 | Development tests |
| NumPy | Composite metadata: BSD 3-Clause, 0BSD, MIT, Zlib, CC0 1.0 | Numerical arrays and bundled components |
| PyTorch | Composite metadata: Apache 2.0, Apache 2.0 with LLVM exception, BSD 2/3-Clause, Boost 1.0, MIT | Local inference dependency and bundled components |

The package distribution and its bundled license files govern actual reuse. In particular, a top-level package label does not replace bundled or transitive notices. Consult the exact installed distributions and their upstream project records for complete texts.

## Model candidates, not shipped weights

- [ProsusAI FinBERT model card](https://huggingface.co/ProsusAI/finbert): a prospective financial-sentiment component. Its [upstream software repository license](https://raw.githubusercontent.com/ProsusAI/finBERT/master/LICENSE) is Apache 2.0. That software license must not be assumed to grant redistribution of the model weights or training text. No weights are shipped here; a future local acquisition requires checking the exact snapshot and applicable terms.
- [Moritz Laurer DeBERTa-v3-base-mnli-fever-anli model card](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli): a prospective zero-shot Event Class component. The model card reports MIT. No weights are shipped or loaded by the available manual exercise. A future lock must identify the exact revision, hash, terms and tested local CPU behavior.

The present unresolved [model declaration](config/models.lock.json) is not a verified downloaded model lock or a fitted inference artifact.

## External source candidates and restrictions

- [GDELT](https://gdeltproject.org/about.html#termsofuse) is a possible news acquisition source. Its terms require citation and a link to GDELT for use or redistribution of its data. A GDELT record does not by itself convey rights to an external publisher's article, headline or full text. This repository commits no fetched GDELT or publisher text.
- [FiQA organizer](https://sites.google.com/view/fiqa/) material is a possible research/evaluation source. The inspected organizer page did not grant a blanket redistribution license for all associated text. It is not part of the committed demo, and acquisition/use requires checking the particular release and source terms.
- The [StockNet dataset repository](https://github.com/yumoxu/stocknet-dataset) publishes an [MIT software license](https://raw.githubusercontent.com/yumoxu/stocknet-dataset/master/LICENSE). That is not a blanket grant to republish underlying tweets, Yahoo-sourced prices, or other third-party content. No StockNet content is committed here.
- Historical factor observations and benchmarks must carry their own provider, vintage, usage terms, availability time, units and hashes before calibration or evaluation. The committed `data/calibration/` CSVs are explicitly project-authored hypothetical exercise inputs, not observed GDELT, social or market data.

Any future restricted acquisition belongs in ignored `data/local/` or `data/raw/`, with explicit terms acknowledgement and a hashed local descriptor. The repository policy excludes fetched third-party text, model weights, caches and local databases from Git. These are conservative project controls; they do not purport to decide all legal rights for a data provider.

## Method and product inspiration

[OpenBB](https://github.com/openbq-org/OpenBB), [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot), [OpenGamma Strata](https://github.com/OpenGamma/Strata), [Open Source Risk Engine](https://github.com/OpenSourceRisk/Engine), [QuantLib](https://github.com/lballabio/QuantLib), and [Trade the Event](https://github.com/Zhihan1996/TradeTheEvent) informed the research and design comparison. They are not installed dependencies, embedded engines, or sources of copied application code or content. [Research notes](docs/research/existing-solutions-and-free-apis.md) distinguish each reference's role. Methodological sources and project-specific conventions are cited in [methodology](docs/methodology.md).
