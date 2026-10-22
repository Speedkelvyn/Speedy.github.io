"""ind_eval.py: evaluate industry rankings by information coefficients and long-short portfolios.

Information coefficient (IC): each month, the rank correlation between the forecast and the realised relative return
across industries; its mean is the average ranking skill and its t-statistic uses Newey-West errors.
Long-short portfolio: equal-weight the top 10 industries by forecast, short the bottom 10 (about the top and
bottom fifth of 49), rebalanced monthly; costs are charged on turnover (default 20 basis points per unit traded).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

N_LEG = 10
COST = 0.0020


def monthly_ic(fc: pd.DataFrame, model: str) -> pd.Series:
    def one(g: pd.DataFrame) -> float:
        g = g.dropna(subset=["y", model])
        return (
            float(stats.spearmanr(g[model], g["y"])[0]) if len(g) > 10 and g[model].nunique() > 1 else np.nan
        )

    return fc.groupby("date")[[model, "y"]].apply(one)


def nw_t(x: pd.Series, lags: int = 3) -> float:
    x = x.dropna().to_numpy()
    n = len(x)
    u = x - x.mean()
    var = u @ u / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    return float(x.mean() / math.sqrt(var / n))


def long_short(fc: pd.DataFrame, model: str, n_leg: int = N_LEG, cost: float = COST) -> pd.DataFrame:
    """Monthly return of long top / short bottom `n_leg` industries (equal weights), gross and net of costs."""
    prev_w = None
    rows = []
    for date, g in fc.dropna(subset=[model, "ret_next"]).groupby("date"):
        g = g.sort_values(model)
        w = pd.Series(0.0, index=g["industry"].to_numpy())
        w[g["industry"].iloc[-n_leg:]] = 1 / n_leg
        w[g["industry"].iloc[:n_leg]] = -1 / n_leg
        r = float((w.to_numpy() * g["ret_next"].to_numpy()).sum())
        turn = float(w.abs().sum()) if prev_w is None else float(w.sub(prev_w, fill_value=0).abs().sum())
        prev_w = w
        rows.append({"date": date, "gross": r, "net": r - cost * turn, "turnover": turn})
    return pd.DataFrame(rows).set_index("date")


def sharpe_diff_p(r1: pd.Series, r0: pd.Series) -> float:
    t = len(r1)
    s1, s0 = r1.mean() / r1.std(), r0.mean() / r0.std()
    rho = float(np.corrcoef(r1, r0)[0, 1])
    v = 2 * (1 - rho) + 0.5 * (s1**2 + s0**2 - 2 * s1 * s0 * rho**2)
    return float(1 - stats.norm.cdf((s1 - s0) / math.sqrt(v / t)))


def factor_alpha(r: pd.Series, fac: pd.DataFrame, lags: int = 3) -> tuple[float, float]:
    """Annual alpha (%) and its t-statistic from a regression on the market, size and value factors."""
    import statsmodels.api as sm

    f = fac.reindex(r.index)[["mktrf", "smb", "hml"]]
    ok = f.notna().all(axis=1)
    res = sm.OLS(r[ok].to_numpy(), sm.add_constant(f[ok].to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return 1200 * float(res.params[0]), float(res.tvalues[0])


def max_drawdown(r: pd.Series) -> float:
    idx = (1 + r).cumprod()
    return float(100 * (idx / idx.cummax() - 1).min())


def r2_os(fc: pd.DataFrame, model: str) -> float:
    """Out-of-sample R² (%) against a forecast of zero relative return (Gu, Kelly & Xiu 2020).

    Only defined for models that forecast returns in return units; the single-signal rankings have arbitrary units.
    """
    if model in ("mom", "mom6", "rev", "value", "value_rel", "mean_ml"):
        return float("nan")
    d = fc.dropna(subset=["y", model])
    return 100 * (1 - ((d["y"] - d[model]) ** 2).sum() / (d["y"] ** 2).sum())
