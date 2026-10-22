"""vol_baselines.py: volatility forecasts built directly from daily returns (EWMA and GJR-GARCH).

Both are real-time: at the end of month t they use returns up to t only, and the GARCH parameters
come from the last estimation date before t (every January, on all earlier data).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model


def ewma_forecast(
    daily: pd.DataFrame, month_ends: pd.DatetimeIndex, n_next: pd.Series, lam: float = 0.94
) -> pd.Series:
    """RiskMetrics: next month's variance = daily EWMA variance at month end x number of days next month."""
    r2 = (daily["mkt"] ** 2).to_numpy()
    s = np.empty_like(r2)
    s[0] = r2[:250].mean()
    for i in range(1, len(r2)):
        s[i] = lam * s[i - 1] + (1 - lam) * r2[i - 1]  # variance for day i uses returns up to day i-1
    sig = pd.Series(lam * s + (1 - lam) * r2, index=daily.index)  # variance for the day after each close
    last = sig.groupby(sig.index.to_period("M")).last()
    last.index = last.index.to_timestamp()
    return np.log(last.reindex(month_ends) * n_next.reindex(month_ends))


def _gjr_filter(r: np.ndarray, omega: float, alpha: float, gamma: float, beta: float) -> np.ndarray:
    """Conditional variance for the day after each observation (GJR-GARCH(1,1), zero mean)."""
    v = np.empty(len(r))
    prev = r[:250].var()
    for i, x in enumerate(r):
        prev = omega + (alpha + gamma * (x < 0)) * x * x + beta * prev
        v[i] = prev
    return v


def garch_forecast(daily: pd.DataFrame, month_ends: pd.DatetimeIndex, n_next: pd.Series) -> pd.Series:
    """GJR-GARCH(1,1) with Student-t errors, re-estimated every January on all earlier data.

    The forecast for next month is the sum of the model's expected daily variances over its trading days.
    """
    r = daily["mkt"].to_numpy() * 100  # percent, for a well-scaled optimiser
    idx = daily.index
    out = {}
    params = {}
    for year in sorted({d.year for d in month_ends}):
        cutoff = idx < pd.Timestamp(year, 1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(r[cutoff], mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t").fit(
                disp="off", options={"maxiter": 500}
            )
        params[year] = res.params
    for year, p in params.items():
        om, al, ga, be = p["omega"], p["alpha[1]"], p["gamma[1]"], p["beta[1]"]
        phi = al + ga / 2 + be
        long_run = om / (1 - phi) if phi < 1 else np.nan
        v = _gjr_filter(r, om, al, ga, be)  # variance for the next day, in percent^2
        vs = pd.Series(v, index=idx)
        last = vs.groupby(vs.index.to_period("M")).last()
        last.index = last.index.to_timestamp()
        for t in [d for d in month_ends if d.year == year]:
            n = int(n_next.loc[t])
            v1 = last.loc[t]
            k = np.arange(n)
            total = (long_run * n + (v1 - long_run) * (phi**k).sum()) if np.isfinite(long_run) else v1 * n
            out[t] = np.log(total / 1e4)  # back to decimal returns squared
    return pd.Series(out).reindex(month_ends)
