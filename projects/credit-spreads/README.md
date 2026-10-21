# Forecasting US Credit Spreads with Machine Learning

Thirteen models, 145 real-time predictors and 32 years of out-of-sample forecasts (1995–2026) for one-, three- and six-month changes in US corporate credit spreads.

**Read the report:** https://speedkelvyn.github.io/Speedy.github.io/projects/credit-spreads/

## Findings

- **The headline is an artifact.** A lasso explains 27% of next month's change in Moody's Baa spread, and beats the spread's own history by 18% (Diebold–Mariano p < 0.001). But Moody's series is a monthly average, and changes in averages are partly predictable by construction (Working 1960). On the end-of-month spread the same models explain nothing.
- **Tradable spreads are close to a random walk.** For the end-of-month Baa spread and the Gilchrist–Zakrajšek spread no model beats the spread's own history; the Model Confidence Set cannot exclude even the random walk.
- **Momentum pays.** A pre-specified rule that holds Baa credit only when carry outweighs the expected widening, using the spread's own recent changes, raises the Sharpe ratio from 0.39 to 0.66 (p = 0.02) and halves the worst month (−5.4% vs −12.2%), across subperiods and duration assumptions. After 25 bp per switch the Sharpe ratio is 0.47.

## Models

Random walk and AR benchmarks; OLS, ridge, lasso, elastic net, principal components (diffusion index), partial least squares and univariate-forecast combination; random forest, gradient boosting and a neural network; mean and median combinations. Evaluated with Clark–West, Diebold–Mariano (HLN-corrected) and the Model Confidence Set.

## Layout

| Path | Contents |
|---|---|
| `credit_data.py` | Targets, FRED-MD transformations, credit signals and real-time publication lags |
| `credit_models.py` | The model arsenal and the expanding-window forecasting loop |
| `credit_eval.py` | R²_OS, Clark–West, Diebold–Mariano, Model Confidence Set, the credit-timing rule |
| `run_credit.py` | Runs all models for all spreads and horizons, saves `results/` |
| `report.qmd` | The report |
| `data/` | Raw inputs and their sources |
| `tests/` | Data-timing and statistics checks |

## Reproduce

```bash
pip install -r requirements.txt
python run_credit.py          # about an hour on a 2-core laptop
quarto render report.qmd
pytest -q
```
