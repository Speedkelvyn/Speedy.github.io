"""rec_eval.py: evaluation of out-of-sample event probabilities.

Metrics
    AUC            the chance a randomly chosen pre-event month gets a higher probability than a calm month
    Log score      mean log-likelihood of the outcomes; higher is better (the proper score that punishes confident misses)
    Brier skill    1 - Brier score / Brier score of the base rate
Because the event windows overlap and there are only a handful of independent episodes, confidence intervals
come from a moving-block bootstrap with 24-month blocks, never from treating months as independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-4


def log_score(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def table(
    fc: pd.DataFrame, models: list[str], names: dict, ref: str = "probit_ts", reps: int = 1000, seed: int = 0
) -> pd.DataFrame:
    d = fc.dropna(subset=["actual", *models]).reset_index(drop=True)
    y = d["actual"].to_numpy()
    rng = np.random.default_rng(seed)
    boots = [block_bootstrap_indices(len(d), 24, rng) for _ in range(reps)]
    boots = [b for b in boots if 0 < y[b].sum() < len(b)]
    base_brier = brier(y, d["base"].to_numpy())
    rows = []
    for m in models:
        p = d[m].to_numpy()
        auc = roc_auc_score(y, p) if m != "base" else 0.5
        auc_b = np.array([roc_auc_score(y[b], p[b]) for b in boots]) if m != "base" else np.array([0.5])
        diff = (
            np.array([roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], d[ref].to_numpy()[b]) for b in boots])
            if m not in ("base", ref)
            else None
        )
        rows.append(
            {
                "Model": names[m],
                "key": m,
                "AUC": auc,
                "AUC low": np.percentile(auc_b, 5),
                "AUC high": np.percentile(auc_b, 95),
                "Log score": log_score(y, p),
                "Brier skill (%)": 100 * (1 - brier(y, p) / base_brier),
                "p (AUC > curve-only probit)": float(np.mean(diff <= 0)) if diff is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def lead_table(fc: pd.DataFrame, recession_flag: pd.Series, model: str, leads=(12, 9, 6, 3)) -> pd.DataFrame:
    """Probability the model gave k months before each recession began (the start month = first month flagged)."""
    start = recession_flag[(recession_flag == 1) & (recession_flag.shift(1) == 0)].index
    f = fc.set_index("date")[model]
    rows = []
    for s in start:
        if s < f.index.min():
            continue
        row = {"Recession starts": s.strftime("%b %Y")}
        for k in leads:
            t = s - pd.DateOffset(months=k)
            row[f"{k} months before"] = f.get(t, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def alarm_stats(fc: pd.DataFrame, model: str, threshold: float) -> dict:
    d = fc.dropna(subset=["actual", model])
    flagged = d[model] >= threshold
    ev = d["actual"] == 1
    return {
        "Threshold": threshold,
        "Months flagged (%)": 100 * float(flagged.mean()),
        "Pre-recession months caught (%)": 100 * float((flagged & ev).sum() / max(ev.sum(), 1)),
        "False alarms (% of flags)": 100 * float((flagged & ~ev).sum() / max(flagged.sum(), 1)),
    }
