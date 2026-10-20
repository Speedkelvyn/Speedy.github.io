"""beta_daily.py: daily CAPM beta against the S&P 500, saved into an inputs file.

    python beta_daily.py DELL                      # 2 years of daily returns
    python beta_daily.py DELL --years 5 --index ^GSPC --inputs dell_inputs.json

Regression: R_stock = alpha + beta * R_index on daily simple returns of adjusted
closes, with dates matched across both series (the same as Excel's Data Analysis >
Regression). Writes beta_daily and its statistics, and sets beta_source = "daily" so
the WACC uses it.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import valuation_v2 as v


def daily_beta(ticker: str, index: str = "^GSPC", years: int = 2) -> dict:
    """Download daily closes and regress the stock's returns on the index's."""
    import yfinance as yf  # optional dependency, only needed for live data

    px = yf.download([ticker, index], period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
    px = px["Close"].dropna()
    stock = [[str(d.date()), float(x)] for d, x in px[ticker].items()]
    idx = [[str(d.date()), float(x)] for d, x in px[index].items()]
    rs, rm, dates = v.returns_from_prices(stock, idx)
    r = v.capm_beta(rs, rm)
    r = {k: (list(map(float, val)) if isinstance(val, list) else float(val)) for k, val in r.items()}
    r["window"] = f"{dates[0]} to {dates[-1]}"
    step = max(1, len(rs) // 600)
    r["points"] = [[float(rm[i]), float(rs[i])] for i in range(0, len(rs), step)]
    r["xmin"], r["xmax"] = float(rm.min()), float(rm.max())
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker")
    ap.add_argument("--index", default="^GSPC")
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--inputs", default=None, help="inputs file to update (default TICKER_inputs.json)")
    ap.add_argument("--adjusted", action="store_true", help="use 0.67*beta + 0.33 in the WACC")
    a = ap.parse_args(argv)

    r = daily_beta(a.ticker, a.index, a.years)
    print(f"{a.ticker} vs {a.index}: {r['window']}, n={int(r['n'])}")
    print(
        f"beta {r['beta']:.4f}  (se {r['seBeta']:.4f}, t {r['tBeta']:.1f}, "
        f"95% {r['ci95'][0]:.3f}-{r['ci95'][1]:.3f})"
    )
    print(f"R^2 {r['r2']:.4f}  adjusted beta {r['blume']:.4f}  Jensen alpha (ann.) {r['alphaAnnual']:.2%}")

    candidates = (
        [a.inputs] if a.inputs else [f"{a.ticker.lower()}_inputs.json", f"{a.ticker.upper()}_inputs.json"]
    )
    path = next((pathlib.Path(p) for p in candidates if pathlib.Path(p).exists()), None)
    if path is None:
        raise SystemExit(f"No inputs file found ({', '.join(candidates)}); run fetch_yahoo.py first.")
    inp = json.loads(path.read_text(encoding="utf-8"))
    inp["capital"].update(
        beta_daily=r["beta"], beta_daily_stats=r, beta_source="daily_adj" if a.adjusted else "daily"
    )
    path.write_text(json.dumps(inp, indent=1), encoding="utf-8")
    W = v.wacc(inp)
    print(f"saved to {path}:  Ke {W['ke']:.2%}  WACC {W['wacc']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
