# Kelvin Afful · Portfolio

[![CI](https://github.com/Speedkelvyn/Speedy.github.io/actions/workflows/ci.yml/badge.svg)](https://github.com/Speedkelvyn/Speedy.github.io/actions/workflows/ci.yml)

Equity valuation, credit risk and quantitative finance work by Kelvin Afful (MS Quantitative Finance & Economics, Texas State University).

**Live site:** https://speedkelvyn.github.io/Speedy.github.io/

## Valuations

| Company | Call (base case) | Open |
|---|---|---|
| **Dell Technologies (DELL)** | Overvalued: fair value $311.65 vs $473.39 | [Interactive model](https://speedkelvyn.github.io/Speedy.github.io/valuations/dell/) · [Excel](valuations/dell/DELL_valuation.xlsx) · [Inputs](valuations/dell/dell_inputs.json) |
| **Sandisk (SNDK)** | Fairly valued, low confidence: $1,315.59 vs $1,365.38 | [Interactive model](https://speedkelvyn.github.io/Speedy.github.io/valuations/sndk/) · [Excel](valuations/sndk/SNDK_valuation.xlsx) · [Notebook](valuations/sndk/SNDK_valuation.ipynb) |

Each interactive model is a self-contained page built on a single valuation engine (FCFF DCF, Monte Carlo, comparables, peer regression and earnings-quality screens). Switch Bear, Base and Bull cases, edit any input, and audit where every number came from.

## Python engine

The interactive pages and the Python engine in [`valuations/engine`](valuations/engine) give the same DCF, WACC and comparables results (checked by the tests to 1e-9):

```bash
cd valuations/engine
pip install -r requirements.txt
python run_valuation.py dell                    # full valuation report for Dell
python run_valuation.py sndk --scenario Bull    # Sandisk, Bull case
python run_valuation.py dell --excel dell.xlsx  # also write the live-formula Excel workbook
python fetch_yahoo.py AAPL                      # build an inputs file for any ticker from Yahoo Finance
python -m pytest tests
python web/build_page.py ../dell/dell_inputs.json ../dell/index.html   # rebuild a page after editing inputs
```

## Research

**[Forecasting US credit spreads with machine learning](https://speedkelvyn.github.io/Speedy.github.io/projects/credit-spreads/)** Thirteen models (lasso to neural networks) and 145 real-time predictors, 1995–2026. A 27% R² on Moody's monthly Baa spread turns out to be an artifact of time-averaging; on tradable spreads the random walk wins, but a pre-specified momentum rule raises the Sharpe ratio of credit from 0.39 to 0.66 (p = 0.02). Code and data in [`projects/credit-spreads`](projects/credit-spreads).

**[Forecasting stock-market volatility with machine learning](https://speedkelvyn.github.io/Speedy.github.io/projects/volatility/)** Nineteen forecasters, 2000–2026. The month-end VIX is the best single predictor (R²_OS 52% against 43% for HAR) but only since 2013; GJR-GARCH wins on QLIKE; and no volatility-managed portfolio raises the Sharpe ratio significantly (0.49 to 0.52–0.59). Code and data in [`projects/volatility`](projects/volatility).

**[Predicting US recessions and stock-market stress with machine learning](https://speedkelvyn.github.io/Speedy.github.io/projects/recession-risk/)** Nine probability models and about 150 real-time indicators, 1993–2025, with recession labels used only once the NBER would have announced them. A yield-curve probit (AUC 0.85) is not beaten by any richer model; the 2022–24 inversion gave a 90% signal for 21 months and no recession followed. Code and data in [`projects/recession-risk`](projects/recession-risk).

**[Ranking US industries with machine learning](https://speedkelvyn.github.io/Speedy.github.io/projects/industry-ranking/)** Thirteen models rank Kenneth French's 49 industries each month, 1980–2026, using only data available at the time. Lasso and random forests reach a rank correlation of 0.05 with the realised ranking, but 12-month momentum alone gets 0.044; the skill roughly halves after 1999, and after 20 basis points of trading costs the long-short Sharpe ratio is 0.2 to 0.4. Code and data in [`projects/industry-ranking`](projects/industry-ranking).

**[Can models beat the random walk in exchange rates?](https://speedkelvyn.github.io/Exchange-Rate/)** Out-of-sample forecasts for daily USD/JPY (AR-GARCH) and a monthly panel of six currencies (Elastic Net, neural network, ensemble) with macro fundamentals and global risk. No model beats the random walk overall, but the neural network does in high-VIX months (Clark-West p = 0.04). Code and data in [Exchange-Rate](https://github.com/Speedkelvyn/Exchange-Rate).

## Repository layout

```
├── index.html               Portfolio homepage
├── valuations/
│   ├── dell/                DELL model, Excel workbook, inputs, football field
│   ├── sndk/                SNDK model, Excel workbook, notebook, inputs
│   └── engine/              Python engine, CLI, Excel builder, data fetch, tests
│       └── web/             Browser engine (model.js), page template and build_page.py
├── projects/
│   ├── credit-spreads/      Credit-spread forecasting study: 13 models, FRED-MD, timing rule
│   ├── volatility/          Volatility forecasting study: 19 forecasters, VIX, volatility-managed portfolio
│   ├── recession-risk/      Recession and bear-market probabilities: 9 models, real-time labels
│   ├── industry-ranking/    Industry ranking study: 13 models on 49 industries, long-short portfolios
│   ├── fx-forecasting/      Redirect to the exchange-rate study (repo: Exchange-Rate)
│   └── inboxzero/           Redirect to the Gmail triage tool (repo: Inbox-Zero)
└── valuation-ledger/        Redirect to the Valuation Ledger course (repo: Valuation-gem)
```

## Other repositories
- [Valuation-gem](https://github.com/Speedkelvyn/Valuation-gem): The Valuation Ledger, an interactive course in valuing companies
- [Exchange-Rate](https://github.com/Speedkelvyn/Exchange-Rate): Can machine learning beat the random walk in exchange rates?
- [Inbox-Zero](https://github.com/Speedkelvyn/Inbox-Zero): Gmail triage script that scores, labels and clears unread email
- [Growthhub](https://github.com/Speedkelvyn/Growthhub): Focus & Growth routine and study-sprint dashboard
- [Bible365](https://github.com/Speedkelvyn/Bible365): Daily Bible-in-a-year reading tracker

Valuations are for illustration only, not investment advice.

## Development

```bash
pip install ruff pytest
ruff check . && ruff format --check .   # lint and formatting
pytest -q                               # tests
npx html-validate@8.24.1 $(git ls-files '*.html')   # HTML validation
```

Style is set by `ruff.toml`, `.prettierrc.json`, `.htmlvalidate.json` and `.editorconfig`; GitHub Actions runs the same checks on every push.
