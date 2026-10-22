# Predicting US Recessions and Stock-Market Stress with Machine Learning

Nine probability models, about 150 real-time indicators and 32 years of out-of-sample forecasts (1993–2025) of two events within the next twelve months: a US recession, and a fall of at least 20% in the stock market.

**Read the report:** https://speedkelvyn.github.io/Speedy.github.io/projects/recession-risk/

## Findings

- **The yield curve is hard to beat.** A one-variable probit on the 10-year minus 3-month slope separates pre-recession months from calm ones with an AUC of 0.85. Adding the excess bond premium, credit spreads, labour-market data, a lasso, a random forest or boosting gives 0.82–0.86, never significantly better; the 90% block-bootstrap interval for the curve alone is 0.74–0.95.
- **Most models are overconfident.** Every probit, the lasso and boosting score worse than quoting the historical frequency (negative Brier skill). The 2022–24 inversion held the curve probit above 90% for 21 months, and no recession followed. Only the random forest has positive skill.
- **The curve says nothing reliable about bear markets.** For 20% stock-market falls the curve-only probit has an AUC of 0.21: its link to equities changed sign across eras.
- **Three recessions cannot settle a model contest.** The common test sample contains the pre-recession months of 2001, 2008 and 2020; the report treats rankings accordingly.
- **Latest reading (August 2026 data):** recession probabilities of 0.4% to 13% across models against a historical frequency of 15%; bear-market probabilities of 1% to 8% against 9%.

## Design

Real-time labels: the NBER dates a recession months after it starts, so a month joins the training set 24 months later (12 months for bear-market labels). Macro data are lagged one month, quarterly bank data one quarter. Confidence intervals use a 24-month moving-block bootstrap, not independent months.

## Layout

| Path | Contents |
|---|---|
| `rec_data.py` | Predictors, FRED-MD transformations, real-time lags, the two targets |
| `rec_models.py` | The probit, lasso, forest and boosting models and the expanding-window loop |
| `rec_eval.py` | AUC with block bootstrap, log score, Brier skill, lead-time and alarm tables |
| `run_rec.py` | Runs both targets, saves `results/probabilities.csv` |
| `analyze_rec.py` | Builds every table and number in the report |
| `report.qmd` | The report |
| `data/` | Raw inputs and their sources |
| `tests/` | Label-timing, no-look-ahead and scoring checks |

## Reproduce

```bash
pip install -r requirements.txt
python run_rec.py          # about 10 minutes on a 2-core laptop
python analyze_rec.py
quarto render report.qmd
pytest -q
```
