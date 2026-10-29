# Kelvin Afful · Portfolio

[![CI](https://github.com/Speedkelvyn/Speedy.github.io/actions/workflows/ci.yml/badge.svg)](https://github.com/Speedkelvyn/Speedy.github.io/actions/workflows/ci.yml)

Equity valuation, credit risk and quantitative finance work by Kelvin Afful (MS Quantitative Finance & Economics, Texas State University).

**Live site:** https://speedkelvyn.github.io/Speedy.github.io/

## Valuations

| Company | Call (base case) | Open |
|---|---|---|
| **Dell Technologies (DELL)** | Overvalued: fair value $311.65 vs $473.39 | [Interactive model](https://speedkelvyn.github.io/Dell-and-Sandisk-plus-valuation-engine/dell/) · [Excel](https://speedkelvyn.github.io/Dell-and-Sandisk-plus-valuation-engine/dell/DELL_valuation.xlsx) · [Inputs](https://speedkelvyn.github.io/Dell-and-Sandisk-plus-valuation-engine/dell/dell_inputs.json) |
| **Sandisk (SNDK)** | Fairly valued, low confidence: $1,315.59 vs $1,365.38 | [Interactive model](https://speedkelvyn.github.io/Dell-and-Sandisk-plus-valuation-engine/sndk/) · [Excel](https://speedkelvyn.github.io/Dell-and-Sandisk-plus-valuation-engine/sndk/SNDK_valuation.xlsx) · [Notebook](https://github.com/Speedkelvyn/Dell-and-Sandisk-plus-valuation-engine/blob/main/sndk/SNDK_valuation.ipynb) |

Each interactive model is a self-contained page built on a single valuation engine (FCFF DCF, Monte Carlo, comparables, peer regression and earnings-quality screens). Switch Bear, Base and Bull cases, edit any input, and audit where every number came from.

The engine, the Python and Excel versions of it, and an app that values any listed stock from a watchlist or a portfolio are in [Dell-and-Sandisk-plus-valuation-engine](https://github.com/Speedkelvyn/Dell-and-Sandisk-plus-valuation-engine).

## Research

**[Forecasting US credit spreads with machine learning](https://speedkelvyn.github.io/Credit-spread-forecasting-/)** Thirteen models (lasso to neural networks) and 145 real-time predictors, 1995–2026. A 27% R² on Moody's monthly Baa spread turns out to be an artifact of time-averaging; on tradable spreads the random walk wins, but a pre-specified momentum rule raises the Sharpe ratio of credit from 0.39 to 0.66 (p = 0.02). Code and data in [Credit-spread-forecasting-](https://github.com/Speedkelvyn/Credit-spread-forecasting-).

**[Forecasting stock-market volatility with machine learning](https://speedkelvyn.github.io/Volatility-forecasting-Study/)** Nineteen forecasters, 2000–2026. The month-end VIX is the best single predictor (R²_OS 52% against 43% for HAR) but only since 2013; GJR-GARCH wins on QLIKE; and no volatility-managed portfolio raises the Sharpe ratio significantly (0.49 to 0.52–0.59). Code and data in [Volatility-forecasting-Study](https://github.com/Speedkelvyn/Volatility-forecasting-Study).

**[Predicting US recessions and stock-market stress with machine learning](https://speedkelvyn.github.io/Recession-Risk-Study/)** Nine probability models and about 150 real-time indicators, 1993–2025, with recession labels used only once the NBER would have announced them. A yield-curve probit (AUC 0.85) is not beaten by any richer model; the 2022–24 inversion gave a 90% signal for 21 months and no recession followed. Code and data in [Recession-Risk-Study](https://github.com/Speedkelvyn/Recession-Risk-Study).

**[Ranking US industries with machine learning](https://speedkelvyn.github.io/Industry-ranking-study/)** Thirteen models rank Kenneth French's 49 industries each month, 1980–2026, using only data available at the time. Lasso and random forests reach a rank correlation of 0.05 with the realised ranking, but 12-month momentum alone gets 0.044; the skill roughly halves after 1999, and after 20 basis points of trading costs the long-short Sharpe ratio is 0.2 to 0.4. Follow-up tests: volatility-managing lifts momentum to the lasso's Sharpe ratio, the learners' edge over momentum sits in its ten worst months, and other industries' lagged returns add little; a random-feature timing model does not beat volatility scaling, and a break test cannot confirm that the post-1999 decline is permanent. Code and data in [Industry-ranking-study](https://github.com/Speedkelvyn/Industry-ranking-study).

**[Speedy Regression Tool](https://speedkelvyn.github.io/Browser-regression-tool/)** OLS, logit and probit on any CSV file, run in the browser: robust standard errors, odds ratios, diagnostics and plots. The numerical core is tested against statsmodels. Code in [Browser-regression-tool](https://github.com/Speedkelvyn/Browser-regression-tool).

**[Can models beat the random walk in exchange rates?](https://speedkelvyn.github.io/Exchange-Rate/)** Out-of-sample forecasts for daily USD/JPY (AR-GARCH) and a monthly panel of six currencies (Elastic Net, neural network, ensemble) with macro fundamentals and global risk. No model beats the random walk overall, but the neural network does in high-VIX months (Clark-West p = 0.04). Code and data in [Exchange-Rate](https://github.com/Speedkelvyn/Exchange-Rate).

## Repository layout

```
├── index.html               Portfolio homepage
├── valuations/              Redirects to the valuation engine (repo: Dell-and-Sandisk-plus-valuation-engine)
├── projects/
│   ├── credit-spreads/      Redirect to the credit-spread study (repo: Credit-spread-forecasting-)
│   ├── volatility/          Redirect to the volatility study (repo: Volatility-forecasting-Study)
│   ├── recession-risk/      Redirect to the recession study (repo: Recession-Risk-Study)
│   ├── industry-ranking/    Redirect to the industry ranking study (repo: Industry-ranking-study)
│   ├── fx-forecasting/      Redirect to the exchange-rate study (repo: Exchange-Rate)
│   └── inboxzero/           Redirect to the Gmail triage tool (repo: Inbox-Zero)
├── tools/
│   └── regression/          Redirect to the browser regression tool (repo: Browser-regression-tool)
└── valuation-ledger/        Redirect to the Valuation Ledger course (repo: Valuation-gem)
```

## Other repositories
- [Dell-and-Sandisk-plus-valuation-engine](https://github.com/Speedkelvyn/Dell-and-Sandisk-plus-valuation-engine): Valuation engine and app, with the Dell and Sandisk models
- [Valuation-gem](https://github.com/Speedkelvyn/Valuation-gem): The Valuation Ledger, an interactive course in valuing companies
- [Exchange-Rate](https://github.com/Speedkelvyn/Exchange-Rate): Can machine learning beat the random walk in exchange rates?
- [Credit-spread-forecasting-](https://github.com/Speedkelvyn/Credit-spread-forecasting-): Thirteen machine-learning models for US credit spreads, 1995–2026
- [Volatility-forecasting-Study](https://github.com/Speedkelvyn/Volatility-forecasting-Study): Nineteen volatility forecasters and the volatility-managed portfolio
- [Recession-Risk-Study](https://github.com/Speedkelvyn/Recession-Risk-Study): Recession and bear-market probabilities from nine models
- [Industry-ranking-study](https://github.com/Speedkelvyn/Industry-ranking-study): Ranking 49 US industries with machine learning
- [Browser-regression-tool](https://github.com/Speedkelvyn/Browser-regression-tool): OLS, logit and probit in the browser, tested against statsmodels
- [Inbox-Zero](https://github.com/Speedkelvyn/Inbox-Zero): Gmail triage script that scores, labels and clears unread email
- [Growthhub](https://github.com/Speedkelvyn/Growthhub): Focus & Growth routine and study-sprint dashboard
- [Bible365](https://github.com/Speedkelvyn/Bible365): Daily Bible-in-a-year reading tracker

Valuations are for illustration only, not investment advice.

## Development

```bash
npx html-validate@8.24.1 $(git ls-files '*.html')   # HTML validation
npx prettier@3.3.3 --check $(git ls-files '*.html' '*.js')   # formatting
```

Style is set by `.prettierrc.json`, `.htmlvalidate.json` and `.editorconfig`; GitHub Actions runs the same checks on every push. The Python work lives in the repositories linked above.
