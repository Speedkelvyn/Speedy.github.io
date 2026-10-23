"""ind_breaks.py: Andrews (1993) sup-Wald test for one unknown break in a mean, with a simulated null distribution.

The monthly rank correlation of a signal (or the monthly return of a portfolio) has mean m1 before an unknown date and
m2 after. For every candidate date in the middle 70% of the sample the Wald statistic of m1 = m2 is computed with a
Newey–West long-run variance of the residuals; the test statistic is the largest of them. Its null distribution is that
of sup over lambda in [trim, 1 - trim] of B(lambda)^2 / (lambda (1 - lambda)) with B a Brownian bridge, which is simulated.
"""

from __future__ import annotations

import numpy as np


def long_run_variance(e: np.ndarray, lags: int = 3) -> float:
    n = len(e)
    e = e - e.mean()
    v = e @ e / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(v)


def wald_path(x: np.ndarray, trim: float = 0.15, lags: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Wald statistics for a break after observation k, for every k in the trimmed range. Returns (k, W)."""
    n = len(x)
    ks = np.arange(int(np.ceil(trim * n)), int(np.floor((1 - trim) * n)) + 1)
    cs = np.cumsum(x)
    out = np.empty(len(ks))
    for j, k in enumerate(ks):
        m1, m2 = cs[k - 1] / k, (cs[-1] - cs[k - 1]) / (n - k)
        e = np.where(np.arange(n) < k, x - m1, x - m2)
        omega = long_run_variance(e, lags)
        out[j] = (m1 - m2) ** 2 / (omega * (1 / k + 1 / (n - k)))
    return ks, out


_NULL_CACHE: dict[tuple[float, int, int], np.ndarray] = {}


def null_draws(trim: float = 0.15, reps: int = 40000, grid: int = 2000, seed: int = 12345) -> np.ndarray:
    """Simulated draws of sup B(l)^2 / (l (1 - l)) over [trim, 1 - trim], from a discretised Brownian bridge."""
    key = (trim, reps, grid)
    if key not in _NULL_CACHE:
        rng = np.random.default_rng(seed)
        lam = np.arange(1, grid) / grid
        keep = (lam >= trim) & (lam <= 1 - trim)
        draws = np.empty(reps)
        for lo in range(0, reps, 2000):
            hi = min(lo + 2000, reps)
            w = np.cumsum(rng.normal(size=(hi - lo, grid)), axis=1) / np.sqrt(grid)
            b = w[:, :-1] - lam * w[:, [-1]]
            draws[lo:hi] = (b[:, keep] ** 2 / (lam[keep] * (1 - lam[keep]))).max(axis=1)
        _NULL_CACHE[key] = np.sort(draws)
    return _NULL_CACHE[key]


def sup_wald(x, trim: float = 0.15, lags: int = 3) -> dict:
    """Sup-Wald statistic, its simulated p-value, the estimated break position and the means either side."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    ks, w = wald_path(x, trim, lags)
    j = int(np.argmax(w))
    k = int(ks[j])
    draws = null_draws(trim)
    return {
        "sup_wald": float(w[j]),
        "p_value": float((np.searchsorted(draws, w[j], side="right") - len(draws)) / -len(draws)),
        "break_after": k,
        "mean_before": float(x[:k].mean()),
        "mean_after": float(x[k:].mean()),
        "n": len(x),
        "critical_5pct": float(draws[int(0.95 * len(draws))]),
    }
