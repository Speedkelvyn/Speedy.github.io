"""
beta_daily.py -- daily CAPM beta vs the S&P 500, saved into an inputs file.

    python beta_daily.py DELL                      # 2 years of daily returns
    python beta_daily.py DELL --years 5 --index ^GSPC --inputs dell_inputs.json

Regression: R_stock = alpha + beta * R_index on daily simple returns of
adjusted closes, dates matched across both series (same as Excel's
Data Analysis > Regression). Writes beta_daily, its statistics and sets
beta_source = "daily" so the WACC uses it.
"""
import argparse, json
import valuation_v2 as v

ap = argparse.ArgumentParser()
ap.add_argument("ticker")
ap.add_argument("--index", default="^GSPC")
ap.add_argument("--years", type=int, default=2)
ap.add_argument("--inputs", default=None)
ap.add_argument("--adjusted", action="store_true", help="use 0.67*beta + 0.33 in the WACC")
a = ap.parse_args()

import yfinance as yf
px = yf.download([a.ticker, a.index], period=f"{a.years}y", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
stock = [[str(d.date()), float(x)] for d, x in px[a.ticker].items()]
index = [[str(d.date()), float(x)] for d, x in px[a.index].items()]
rs, rm, dates = v.returns_from_prices(stock, index)
r = v.capm_beta(rs, rm)
r = {k: (list(map(float, val)) if isinstance(val, list) else float(val)) for k, val in r.items()}
r["window"] = f"{dates[0]} to {dates[-1]}"
step = max(1, len(rs) // 600)
r["points"] = [[float(rm[i]), float(rs[i])] for i in range(0, len(rs), step)]
r["xmin"], r["xmax"] = float(rm.min()), float(rm.max())

print(f"{a.ticker} vs {a.index}: {r['window']}, n={int(r['n'])}")
print(f"beta {r['beta']:.4f}  (se {r['seBeta']:.4f}, t {r['tBeta']:.1f}, 95% {r['ci95'][0]:.3f}-{r['ci95'][1]:.3f})")
print(f"R^2 {r['r2']:.4f}  adjusted beta {r['blume']:.4f}  Jensen alpha (ann.) {r['alphaAnnual']:.2%}")

path = a.inputs or f"{a.ticker.lower()}_inputs.json"
try:
    inp = json.load(open(path))
except FileNotFoundError:
    inp = json.load(open(f"{a.ticker.upper()}_inputs.json")); path = f"{a.ticker.upper()}_inputs.json"
inp["capital"].update(beta_daily=r["beta"], beta_daily_stats=r, beta_source="daily_adj" if a.adjusted else "daily")
json.dump(inp, open(path, "w"), indent=1)
W = v.wacc(inp)
print(f"saved to {path}:  Ke {W['ke']:.2%}  WACC {W['wacc']:.2%}")
