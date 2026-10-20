# Kelvin Afful · Portfolio

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
```

## Repository layout

```
├── index.html               Portfolio homepage
├── valuations/
│   ├── dell/                DELL model, Excel workbook, inputs, football field
│   ├── sndk/                SNDK model, Excel workbook, notebook, inputs
│   └── engine/              Python valuation engine, CLI, Excel builder, data fetch, tests
├── projects/
│   └── inboxzero/           Gmail triage script, setup guide and project page
└── valuation-ledger/        Redirect to the Valuation Ledger course (repo: Valuation-gem)
```

## Other repositories
- [Valuation-gem](https://github.com/Speedkelvyn/Valuation-gem): The Valuation Ledger, an interactive course in valuing companies
- [Growthhub](https://github.com/Speedkelvyn/Growthhub): Focus & Growth routine and study-sprint dashboard
- [Bible365](https://github.com/Speedkelvyn/Bible365): Daily Bible-in-a-year reading tracker

Valuations are for illustration only, not investment advice.
