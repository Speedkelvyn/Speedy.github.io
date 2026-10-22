"""ind_extensions.py: building blocks for the follow-up experiments on the industry-ranking forecasts.

Everything here is a pure function of a forecast table (date, industry, ret_next and one column per model), so the
timing rules can be tested without the 7-minute model run.

Timing: a ranking made at the end of month t is held during month t+1. `book` stores that month-(t+1) return in row t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def book(fc: pd.DataFrame, model: str, n_leg: int = 10) -> tuple[pd.Series, pd.DataFrame]:
    """Gross return of the long-short book and its weights: long the top `n_leg`, short the bottom `n_leg`, equal-weighted."""
    rets, ws = {}, {}
    for date, g in fc.dropna(subset=[model, "ret_next"]).groupby("date"):
        g = g.sort_values(model)
        w = pd.Series(0.0, index=g["industry"].to_numpy())
        w[g["industry"].iloc[-n_leg:]] = 1 / n_leg
        w[g["industry"].iloc[:n_leg]] = -1 / n_leg
        ws[date] = w
        rets[date] = float((w.to_numpy() * g["ret_next"].to_numpy()).sum())
    return pd.Series(rets), pd.DataFrame(ws).T.fillna(0.0)


def turnover(W: pd.DataFrame) -> pd.Series:
    """Sum of absolute weight changes at each rebalance; the first row builds the book from cash."""
    t = W.diff().abs().sum(axis=1)
    t.iloc[0] = W.iloc[0].abs().sum()
    return t


def net_unmanaged(gross: pd.Series, W: pd.DataFrame, cost: float = 0.002) -> pd.Series:
    return gross - cost * turnover(W)


def trailing_vol(gross: pd.Series, win: int = 6) -> pd.Series:
    """Annualised volatility of the strategy's own returns known when the book is set at the end of month t.

    Row t of `gross` is earned in month t+1, so `shift(1)` leaves rows t-win .. t-1, i.e. returns earned in months
    t-win+1 .. t, all realised by the end of month t.
    """
    return gross.shift(1).rolling(win, min_periods=win).std() * np.sqrt(12)


def vol_managed(
    gross: pd.Series,
    W: pd.DataFrame,
    win: int = 6,
    target: float = 0.10,
    cap: float = 2.0,
    cost: float = 0.002,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Scale the book by target / trailing volatility (capped at `cap`), and charge `cost` on the traded exposure.

    Returns (net return, scale, turnover). Until `win` months exist the scale is 1.
    """
    scale = (target / trailing_vol(gross, win)).clip(upper=cap).fillna(1.0)
    pos = W.mul(scale, axis=0)
    turn = turnover(pos)
    return scale * gross - cost * turn, scale, turn


def zscore_sum(fc: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Add the cross-sectional z-scores of several forecasts, month by month (equal weights, no fitting)."""

    def z(s: pd.Series) -> pd.Series:
        sd = s.std()
        return (s - s.mean()) / (sd if sd and sd > 0 else 1.0)

    return sum(fc.groupby("date")[c].transform(z) for c in cols)


def bear_flag(mkt: pd.Series, months: int = 24) -> pd.Series:
    """True when the market's cumulative return over the trailing `months` months, up to and including month t, is negative.

    Daniel & Moskowitz (2016) condition on this state because momentum crashes follow market rebounds from it.
    """
    cum = (1 + mkt).rolling(months, min_periods=months).apply(np.prod, raw=True) - 1
    return cum < 0
