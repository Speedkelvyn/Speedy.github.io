"""ind_timing.py: does a complex model that times the momentum book beat simply scaling it by volatility?

Nagel (2025) shows that return forecasts from random-feature ridge regressions with far more features than months
behave like a weighted average of past returns, i.e. like volatility-timed momentum. This module builds that model for
the industry momentum long-short book and the benchmarks it should be compared with. Everything is real-time:

* The book earns `g[t]` in month t+1 (see ind_extensions.book).
* The features of date t use only returns earned up to the end of month t: `g.shift(1)` and the market state in the panel.
* A forecast made at date t is trained on pairs (features[s], g[s]) with s < t only, since g[t-1] is the latest return
  that is complete at the end of month t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import ind_extensions as X

BOOK_WINDOWS = (1, 3, 6, 12)


def timing_features(g: pd.Series, states: pd.DataFrame) -> pd.DataFrame:
    """Features known at the end of month t: the book's recent returns and volatility, and two market state variables."""
    r = g.shift(1)  # return earned in month t, complete at its end
    f = pd.DataFrame(index=g.index)
    for k in BOOK_WINDOWS:
        f[f"book_ret_{k}"] = r.rolling(k, min_periods=k).sum()
    for k in (6, 12):
        f[f"book_logvol_{k}"] = np.log(r.rolling(k, min_periods=k).std())
    f["mkt_vol"] = states["mkt_vol"].reindex(g.index)
    f["mkt_ret_12"] = states["mkt_ret_12"].reindex(g.index)
    return f.dropna()


def _features(xs: np.ndarray, w: np.ndarray, gamma: float) -> np.ndarray:
    """Random Fourier features of standardised inputs; the inner product of two rows approximates a Gaussian kernel."""
    a = gamma * (xs @ w)
    return np.hstack([np.cos(a), np.sin(a)]) / np.sqrt(w.shape[1])


def rff_forecasts(
    f: pd.DataFrame,
    y: pd.Series,
    first: str | pd.Timestamp,
    gamma: float = 0.5,
    ridges: tuple[float, ...] = (1.0,),
    n_features: int = 3000,
    refit: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Expanding-window ridge on random Fourier features (dual form, intercept = training mean), one column per ridge.

    `n_features` counts cosine and sine features together; at the first forecast the model has several times more
    features than months. Coefficients are refitted every `refit` months, predictions are made every month.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(f.shape[1], n_features // 2))
    dates = f.index[f.index >= pd.Timestamp(first)]
    out = {z: [] for z in ridges}
    fit = None
    for i, t in enumerate(dates):
        if i % refit == 0 or fit is None:
            tr = f.index < t  # rows s < t: g[s] is complete at the end of month t only for s <= t-1
            ytr = y.reindex(f.index)[tr]
            ok = ytr.notna().to_numpy()
            xtr = f.loc[tr].to_numpy()[ok]
            ytr = ytr.to_numpy()[ok]
            mu, sd = xtr.mean(axis=0), xtr.std(axis=0)
            sd[sd == 0] = 1.0
            ztr = _features((xtr - mu) / sd, w, gamma)
            kern = ztr @ ztr.T
            ybar = ytr.mean()
            coefs = {}
            for z in ridges:
                a = np.linalg.solve(kern + z * np.eye(len(ytr)), ytr - ybar)
                coefs[z] = ztr.T @ a
            fit = (mu, sd, ybar, coefs)
        mu, sd, ybar, coefs = fit
        zt = _features((f.loc[[t]].to_numpy() - mu) / sd, w, gamma)
        for z in ridges:
            out[z].append(float(ybar + (zt @ coefs[z]).item()))
    return pd.DataFrame(out, index=dates)


def linear_ridge_forecasts(
    f: pd.DataFrame, y: pd.Series, first: str | pd.Timestamp, refit: int = 12
) -> pd.Series:
    """Benchmark: ridge on the same features, penalty chosen by time-series cross-validation at each refit."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit

    dates = f.index[f.index >= pd.Timestamp(first)]
    preds, model = [], None
    for i, t in enumerate(dates):
        if i % refit == 0 or model is None:
            tr = f.index < t
            ytr = y.reindex(f.index)[tr]
            ok = ytr.notna().to_numpy()
            xtr = f.loc[tr].to_numpy()[ok]
            mu, sd = xtr.mean(axis=0), xtr.std(axis=0)
            sd[sd == 0] = 1.0
            model = RidgeCV(alphas=np.logspace(0, 5, 11), cv=TimeSeriesSplit(5)).fit(
                (xtr - mu) / sd, ytr.to_numpy()[ok]
            )
        preds.append(float(model.predict((f.loc[[t]].to_numpy() - mu) / sd)[0]))
    return pd.Series(preds, index=dates)


def trailing_return_signal(g: pd.Series, win: int = 12) -> pd.Series:
    """Time-series momentum of the book: trailing mean return over trailing variance (a Kelly-style position)."""
    r = g.shift(1)
    return r.rolling(win, min_periods=win).mean() / r.rolling(win, min_periods=win).var()


def normalise(raw: pd.Series) -> pd.Series:
    """Position scaled so that its average absolute size up to the date is one. Uses only the past and present."""
    return raw / raw.abs().expanding().mean()


def timed_net(
    g: pd.Series, W: pd.DataFrame, pi: pd.Series, cost: float = 0.002
) -> tuple[pd.Series, pd.Series]:
    """Net return and turnover of the book scaled by the position `pi` (which may be negative)."""
    idx = pi.index
    pos = W.loc[idx].mul(pi, axis=0)
    turn = X.turnover(pos)
    return pi * g.loc[idx] - cost * turn, turn


def r2_oos(forecast: pd.Series, y: pd.Series) -> float:
    """Campbell–Thompson out-of-sample R² (%) against the expanding mean of the realised returns known at the date."""
    yy = y.reindex(forecast.index)
    hist = y.expanding().mean().shift(1).reindex(forecast.index)  # mean of g[s], s < t
    ok = yy.notna() & hist.notna()
    return 100 * (1 - ((yy[ok] - forecast[ok]) ** 2).sum() / ((yy[ok] - hist[ok]) ** 2).sum())
