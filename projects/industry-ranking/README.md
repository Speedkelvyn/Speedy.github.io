# Ranking US Industries with Machine Learning

Thirteen methods rank Kenneth French's 49 US industry portfolios every month from January 1980 to August 2026, using only information available at the time. The target is the return over the next month minus the average industry return, so only the ranking is scored.

**Read the report:** https://speedkelvyn.github.io/Speedy.github.io/projects/industry-ranking/

## Findings

- **The ranking skill is real but small.** Lasso, elastic net and random forest reach an average monthly rank correlation (IC) of 0.049–0.050 with the realised ranking (Newey–West t of 4.2–4.6). Models that forecast in return units have an out-of-sample R² between −0.6% and 0.2%.
- **It is mostly momentum.** A one-line rule, "buy the industries that did best over the last 12 months, skipping the latest", reaches an IC of 0.044 (t = 3.7). No machine-learning model beats it by a significant margin (differences in IC of at most 0.006, t ≤ 1.1).
- **Costs take most of the profit.** Long the top ten, short the bottom ten, rebalanced monthly, with 20 basis points per unit traded: the lasso earns 5.2% a year (Sharpe 0.36), random forest 4.7%, momentum 3.8%. The machine-learning portfolios trade 13–20 times their exposure a year, momentum 11.
- **The skill faded.** IC was 0.066–0.076 in 1980–1999 and 0.024–0.038 since 2000.
- **Two textbook signals fail.** Value (book-to-market) has no ranking power (IC 0.000). Short-term reversal has the wrong sign at the industry level: last month's winners keep winning, and betting on reversal loses money.
- **Latest ranking (end of August 2026):** top five are computers, electronic equipment and chips, machinery, precious metals and metal mining; bottom five are consumer goods, entertainment, medical equipment, paper and apparel. It will be scored when Kenneth French publishes September.

## Design

Expanding window, refitted every 12 months (linear models) or 36 months (trees and networks), with penalties chosen by time-series cross-validation. A model at month *t* learns only from rows whose next-month return was already known. Twelve industry characteristics are ranked across industries each month (Gu, Kelly & Xiu 2020) and interacted with five market-state variables; a state variable stays switched off until it has ten years of history. Book-to-market for fiscal year *Y* enters in July of year *Y*+1.

## Layout

| Path | Contents |
|---|---|
| `ind_data.py` | Industry panel, characteristics, state variables and the target |
| `ind_models.py` | Single signals, ridge, lasso, elastic net, forest, boosting, network and the expanding-window loop |
| `ind_eval.py` | Information coefficients, long–short portfolios with costs, Sharpe and alpha tests |
| `run_ind.py` | Runs the forecasts, saves `results/forecasts.csv` |
| `analyze_ind.py` | Builds every table and number in the report |
| `report.qmd` | The report |
| `data/` | Raw inputs and their sources |
| `tests/` | Timing, no-look-ahead and portfolio checks |

## Reproduce

```bash
pip install -r requirements.txt
python run_ind.py          # about 7 minutes on a 2-core laptop
python analyze_ind.py
quarto render report.qmd
pytest -q
```

## Limits

The portfolios are value-weighted industry indexes, not tradable funds; returns ignore taxes and shorting costs. Alphas control for market, size and value but not for market-wide momentum. The 20 basis-point cost is an assumption; the report shows 0, 10, 20 and 40.
