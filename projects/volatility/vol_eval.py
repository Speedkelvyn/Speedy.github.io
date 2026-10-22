"""vol_eval.py: forecast evaluation and the volatility-managed portfolio for the volatility study.

Forecast accuracy
    * R²_OS on log realised variance against the expanding historical mean (Campbell & Thompson 2008)
      and against HAR, the standard benchmark;
    * QLIKE loss on the variance itself (Patton 2011), which is robust to noise in realised variance;
    * Diebold-Mariano tests against HAR (Harvey-Leybourne-Newbold correction) and the Model Confidence Set.

Strategy (fixed in advance, as in Moreira & Muir 2017)
    Each month hold  w = min(cap, target volatility / forecast volatility)  units of the stock market, the
    rest in Treasury bills. Target volatility is the market's average volatility up to that date, so no
    look-ahead. Defaults: leverage capped at 2, trading cost 10 basis points per unit traded and a
    0.5% a year borrowing spread on leverage.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

CAP = 2.0
COST = 0.0010  # per unit of turnover, one way
BORROW = 0.005  # annual spread paid on the leveraged part


# ----------------------------------------------------------------------------- forecast accuracy
def level_forecast(fc: pd.DataFrame, model: str, min_obs: int = 12) -> pd.Series:
    """Convert a log-variance forecast into a variance forecast, E[RV] = exp(mu + s²/2).

    s² is the variance of the model's own past forecast errors that were already known at the forecast date.
    """
    err = fc["actual"] - fc[model]
    s2 = err.shift(1).expanding(min_periods=min_obs).var().fillna(err.iloc[:min_obs].var())
    return np.exp(fc[model] + 0.5 * s2)


def qlike(rv: np.ndarray, h: np.ndarray) -> np.ndarray:
    """QLIKE loss of a variance forecast h against realised variance rv (zero when h = rv)."""
    return rv / h - np.log(rv / h) - 1


def _hac_t(d: np.ndarray, lags: int) -> float:
    n = len(d)
    u = d - d.mean()
    var = u @ u / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    return d.mean() / math.sqrt(var / n)


def dm_p(loss_bench: np.ndarray, loss_model: np.ndarray, lags: int = 3) -> float:
    """One-sided Diebold-Mariano p-value that the model has lower expected loss than the benchmark."""
    d = np.asarray(loss_bench, dtype=float) - np.asarray(loss_model, dtype=float)
    n, h = len(d), lags + 1
    t = _hac_t(d, lags) * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(1 - stats.t.cdf(t, df=n - 1))


def accuracy_table(fc: pd.DataFrame, models: list[str], names: dict, bench: str = "har") -> pd.DataFrame:
    ok = fc[models].notna().all(axis=1)
    fc = fc[ok].reset_index(drop=True)
    a = fc["actual"].to_numpy()
    rv = fc["rv_next"].to_numpy()
    hist_mean = fc["actual"].shift(1).expanding(min_periods=12).mean().bfill().to_numpy()
    mse_b = np.mean((a - fc[bench]) ** 2)
    ql_b = qlike(rv, level_forecast(fc, bench).to_numpy())
    rows = []
    for m in models:
        p = fc[m].to_numpy()
        ql = qlike(rv, level_forecast(fc, m).to_numpy())
        rows.append(
            {
                "Model": names[m],
                "key": m,
                "R²_OS (%)": 100 * (1 - np.sum((a - p) ** 2) / np.sum((a - hist_mean) ** 2)),
                "R²_OS vs HAR (%)": 100 * (1 - np.mean((a - p) ** 2) / mse_b),
                "QLIKE": ql.mean(),
                "QLIKE vs HAR (%)": 100 * (ql.mean() / ql_b.mean() - 1),
                "p (DM vs HAR, QLIKE)": dm_p(ql_b, ql) if m != bench else np.nan,
                "p (DM vs HAR, MSE)": dm_p((a - fc[bench]) ** 2, (a - p) ** 2) if m != bench else np.nan,
            }
        )
    return pd.DataFrame(rows)


def model_confidence_set(
    fc: pd.DataFrame, models: list[str], loss: str = "qlike", size: float = 0.10, seed: int = 0
) -> pd.Series:
    """Hansen-Lunde-Nason Model Confidence Set p-values; models with p > size stay in the set."""
    from arch.bootstrap import MCS

    ok = fc[models].notna().all(axis=1)
    fc = fc[ok].reset_index(drop=True)
    if loss == "qlike":
        losses = pd.DataFrame(
            {m: qlike(fc["rv_next"].to_numpy(), level_forecast(fc, m).to_numpy()) for m in models}
        )
    else:
        losses = pd.DataFrame({m: (fc["actual"] - fc[m]) ** 2 for m in models})
    mcs = MCS(losses, size=size, reps=2000, block_size=6, method="R", seed=seed)
    mcs.compute()
    return mcs.pvalues["Pvalue"].reindex(models)


# ----------------------------------------------------------------------------- the managed portfolio
def managed(
    fc: pd.DataFrame, model: str, rv_hist: pd.Series, cap: float = CAP, cost: float = COST, borrow: float = BORROW
) -> pd.DataFrame:  # fmt: skip
    """Monthly returns (excess of T-bills) of the volatility-managed market, plus the weights used."""
    h = level_forecast(fc, model)
    target = rv_hist.expanding().mean().reindex(pd.DatetimeIndex(fc["date"])).to_numpy()
    w = np.clip(np.sqrt(target / h.to_numpy()), 0, cap)
    w = pd.Series(w, index=pd.DatetimeIndex(fc["date"]))
    turnover = w.diff().abs().fillna(0.0)
    gross = w * fc["ex_next"].to_numpy()
    net = gross - cost * turnover - borrow / 12 * (w - 1).clip(lower=0)
    return pd.DataFrame({"w": w, "ret": net, "gross": gross, "turnover": turnover})


def buy_and_hold(fc: pd.DataFrame) -> pd.Series:
    return pd.Series(fc["ex_next"].to_numpy(), index=pd.DatetimeIndex(fc["date"]))


def max_drawdown(ex: pd.Series, rf: pd.Series) -> float:
    """Largest peak-to-trough fall of the total-return index (%), counting T-bill interest."""
    idx = (1 + ex + rf.reindex(ex.index).fillna(0)).cumprod()
    return float(100 * (idx / idx.cummax() - 1).min())


def sharpe_diff_p(r1: pd.Series, r0: pd.Series) -> float:
    """One-sided p-value that Sharpe(r1) > Sharpe(r0): Jobson-Korkie with Memmel's (2003) correction."""
    t = len(r1)
    s1, s0 = r1.mean() / r1.std(), r0.mean() / r0.std()
    rho = float(np.corrcoef(r1, r0)[0, 1])
    v = 2 * (1 - rho) + 0.5 * (s1**2 + s0**2 - 2 * s1 * s0 * rho**2)
    return float(1 - stats.norm.cdf((s1 - s0) / math.sqrt(v / t)))


def alpha_test(r1: pd.Series, r0: pd.Series, lags: int = 3) -> tuple[float, float, float]:
    """Regress the strategy on the market: annual alpha (%), beta and the alpha t-statistic (Newey-West)."""
    import statsmodels.api as sm

    res = sm.OLS(r1.to_numpy(), sm.add_constant(r0.to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return 1200 * float(res.params[0]), float(res.params[1]), float(res.tvalues[0])


def perf(
    r: pd.Series, rf: pd.Series | None = None, w: pd.Series | None = None, ref: pd.Series | None = None, cap: float = CAP
) -> dict:  # fmt: skip
    out = {
        "Mean (% p.a.)": 1200 * r.mean(),
        "Volatility (% p.a.)": math.sqrt(12) * 100 * r.std(),
        "Sharpe ratio": math.sqrt(12) * r.mean() / r.std(),
        "Worst month (%)": 100 * r.min(),
    }
    if rf is not None:
        out["Max drawdown (%)"] = max_drawdown(r, rf)
    if w is not None:
        out["Average exposure"] = float(w.mean())
        out["Months at the cap (%)"] = 100 * float((w >= cap - 1e-9).mean())
    if ref is not None:
        a, b, t = alpha_test(r, ref)
        out["Alpha (% p.a.)"], out["Beta"], out["Alpha t-stat"] = a, b, t
        out["p (Sharpe > market)"] = sharpe_diff_p(r, ref)
    return out


def certainty_equivalent_gain(r1: pd.Series, r0: pd.Series, gamma: float = 5.0) -> float:
    """Annual certainty-equivalent return gain (%) of r1 over r0 for an investor with risk aversion gamma."""
    ce = lambda r: 12 * r.mean() - 0.5 * gamma * 12 * r.var()
    return 100 * (ce(r1) - ce(r0))
