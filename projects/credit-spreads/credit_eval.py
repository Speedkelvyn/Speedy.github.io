"""credit_eval.py: forecast evaluation for the credit-spread study."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

DURATION = 8.0  # approximate spread duration of a Baa corporate bond index, in years


def r2_os(actual: np.ndarray, pred: np.ndarray, bench: np.ndarray | None = None) -> float:
    """Out-of-sample R² in percent against a benchmark forecast (default: no change)."""
    bench = np.zeros_like(actual) if bench is None else bench
    return 100 * (1 - np.sum((actual - pred) ** 2) / np.sum((actual - bench) ** 2))


def _hac_t(d: np.ndarray, lags: int) -> float:
    n = len(d)
    u = d - d.mean()
    var = u @ u / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    return d.mean() / math.sqrt(var / n)


def clark_west(actual: np.ndarray, pred: np.ndarray, bench: np.ndarray, lags: int = 0) -> float:
    """One-sided Clark-West p-value: does the model improve on the (nested) benchmark?"""
    f = (actual - bench) ** 2 - ((actual - pred) ** 2 - (bench - pred) ** 2)
    return float(1 - stats.norm.cdf(_hac_t(f, lags)))


def diebold_mariano(actual: np.ndarray, pred: np.ndarray, bench: np.ndarray, lags: int = 0) -> float:
    """One-sided Diebold-Mariano p-value with the Harvey-Leybourne-Newbold small-sample correction."""
    d = (actual - bench) ** 2 - (actual - pred) ** 2
    n, h = len(d), lags + 1
    t = _hac_t(d, lags) * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(1 - stats.t.cdf(t, df=n - 1))


def hit_rate(actual: np.ndarray, pred: np.ndarray) -> float:
    return 100 * float(np.mean(np.sign(actual) == np.sign(pred)))


def summary(fc: pd.DataFrame, models: list[str], names: dict) -> pd.DataFrame:
    """R²_OS against the random walk and against the AR benchmark, with Clark-West / DM p-values."""
    h = int(fc["horizon"].iloc[0])
    a = fc["actual"].to_numpy()
    rw, ar = np.zeros_like(a), fc["ar"].to_numpy()
    rows = []
    for m in models:
        p = fc[m].to_numpy()
        rows.append(
            {
                "Model": names[m],
                "key": m,
                "R²_OS vs RW (%)": r2_os(a, p),
                "p (CW vs RW)": clark_west(a, p, rw, lags=h - 1) if m != "rw" else np.nan,
                "R²_OS vs AR (%)": r2_os(a, p, ar),
                "p (DM vs AR)": diebold_mariano(a, p, ar, lags=h - 1) if m not in ("rw", "ar") else np.nan,
                "Hit rate (%)": hit_rate(a, p) if m != "rw" else np.nan,
            }
        )
    return pd.DataFrame(rows)


def model_confidence_set(fc: pd.DataFrame, models: list[str], size: float = 0.10, seed: int = 0) -> pd.Series:
    """Hansen-Lunde-Nason MCS p-values on squared errors (block bootstrap). Models with p > size are in the set."""
    from arch.bootstrap import MCS

    losses = pd.DataFrame({m: (fc["actual"] - fc[m]) ** 2 for m in models})
    h = int(fc["horizon"].iloc[0])
    mcs = MCS(losses, size=size, reps=2000, block_size=max(h, 3), method="R", seed=seed)
    mcs.compute()
    return mcs.pvalues["Pvalue"].reindex(models)


def credit_timing(fc: pd.DataFrame, model: str, duration: float = DURATION) -> pd.Series:
    """Monthly excess return of a strategy that holds Baa credit when its expected excess return is positive.

    Approximate excess return of credit over Treasuries next month: carry (spread / 12) minus
    duration × the change in the spread. The strategy holds credit when carry minus duration × the
    forecast change is positive, otherwise Treasuries (zero excess return). With the random walk
    forecast it always holds credit, which is the buy-and-hold benchmark.
    """
    carry = fc["level"] / 12
    realised = carry - duration * fc["actual"]
    hold = (carry - duration * fc[model]) > 0
    return pd.Series(np.where(hold, realised, 0.0), index=fc["date"])


def perf(r: pd.Series) -> dict:
    return {
        "Mean (% p.a.)": 12 * r.mean(),
        "Volatility (% p.a.)": math.sqrt(12) * r.std(),
        "Sharpe ratio": math.sqrt(12) * r.mean() / r.std() if r.std() > 0 else np.nan,
        "Worst month (%)": r.min(),
        "Months in credit (%)": 100 * float((r != 0).mean()),
    }
