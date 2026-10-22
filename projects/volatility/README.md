# Forecasting Stock-Market Volatility with Machine Learning

Nineteen forecasters, about 140 real-time predictors and 26 years of out-of-sample forecasts (2000–2026) of next month's realised volatility of the US stock market, and a test of whether better forecasts make a better volatility-managed portfolio.

**Read the report:** https://speedkelvyn.github.io/Speedy.github.io/projects/volatility/

## Findings

- **The month-end VIX is the best single predictor.** It explains 51.8% of next month's variation in log realised variance, against 43.0% for the standard HAR regression (Diebold–Mariano p = 0.015). Ridge regression on all predictors matches it (52.1%); the nonlinear models (forest, boosting, neural network) do worse. The advantage dates from 2013: in 2000–2012 the VIX was no better than HAR.
- **On QLIKE, the loss suited to variance, GJR-GARCH-t wins**, and 18 of the 19 forecasters remain in the 10% Model Confidence Set, so the data cannot separate most models.
- **Accuracy does not carry over to money.** Scaling exposure to forecast volatility (Moreira and Muir 2017; leverage capped at 2, 10 bp costs) shrinks the worst month from −17.2% to between −12.5% and −15.4% and the maximum drawdown from −50% to between −37% and −47%. The Sharpe ratio rises from 0.49 to 0.52–0.59, never significantly (p ≥ 0.17). The VIX-only forecast, among the most accurate, gives the weakest strategy; last month's realised variance, the least accurate, the strongest.
- **Forecast for September 2026:** 10–13% annualised volatility against a long-run average of 17.5%; the outcome will be scored when Kenneth French's return file is updated.

## Layout

| Path | Contents |
|---|---|
| `vol_data.py` | Target, volatility features, FRED-MD transformations, real-time lags |
| `vol_baselines.py` | RiskMetrics EWMA and GJR-GARCH-t built from daily returns |
| `vol_models.py` | The regressions, penalised and nonlinear models, and the expanding-window loop |
| `vol_eval.py` | R²_OS, QLIKE, Diebold–Mariano, Model Confidence Set, the managed portfolio and its tests |
| `run_vol.py` | Runs every forecaster, saves `results/forecasts.csv` |
| `analyze_vol.py` | Builds every table and number in the report |
| `report.qmd` | The report |
| `data/` | Raw inputs and their sources |
| `tests/` | Data-timing, no-look-ahead and statistics checks |

## Reproduce

```bash
pip install -r requirements.txt
python run_vol.py          # about 15 minutes on a 2-core laptop
python analyze_vol.py
quarto render report.qmd
pytest -q
```
