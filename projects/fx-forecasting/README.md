# Exchange Rate Forecasting

Can models beat the random walk in exchange rates? Two out-of-sample studies, built so that every forecast uses only information available at the time:

1. **Daily USD/JPY:** AR(1)-GARCH(1,1) against a random walk, for returns and for volatility.
2. **Monthly panel of six currencies (EUR, GBP, JPY, CHF, CAD, AUD vs USD):** Elastic Net, a deep neural network and their ensemble, using interest-rate, inflation and unemployment differentials interacted with global risk (VIX, geopolitical risk, policy uncertainty), following Filippou et al. (2020).

**Read the report:** https://speedkelvyn.github.io/Speedy.github.io/projects/fx-forecasting/

## Findings

- GARCH does not beat the random walk for daily returns (R²_OS 0.10%, Clark-West p = 0.08), but it does forecast volatility.
- In the monthly panel (January 2012 to mid-2025, 872 forecasts), no model beats the random walk overall: R²_OS is −0.11% for the Elastic Net, −4.60% for the DNN and −1.53% for the ensemble.
- In high-risk months (VIX above its historical median) the DNN does beat the random walk: R²_OS +1.68%, Clark-West p = 0.04. In calm months it does much worse.

## Layout

| Path | Contents |
|---|---|
| `report.qmd` / `index.html` | The report (source and rendered) |
| `fx_forecasting.py` | Data loading, panel construction and evaluation tests (R²_OS, Clark-West, Diebold-Mariano, Pesaran-Timmermann) |
| `models.py` | AR-GARCH experiment, Elastic Net, the DNN and the recursive forecasting loop |
| `run_experiments.py` | Runs both experiments and saves forecasts to `results/` |
| `download_data.py` | Refreshes all FRED inputs into `data/raw/` |
| `data/` | Raw inputs and a list of every source series |
| `results/` | Saved forecasts used by the report |
| `tests/` | Checks on data timing and the statistics |

## Reproduce

```bash
pip install -r requirements.txt
python download_data.py      # optional: refresh the FRED data
python run_experiments.py    # about 1 minute for Part I and 20-30 minutes for Part II
quarto render report.qmd
pytest -q
```
