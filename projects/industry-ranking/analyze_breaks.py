"""analyze_breaks.py: is the fade in ranking skill a structural break, and could the test have seen one?

    python analyze_breaks.py     # needs results/forecasts.csv and results/crossind_forecasts.csv; about a minute

For each monthly series (rank correlation of a signal with the realised ranking, or net return of its long-short
portfolio) the Andrews sup-Wald test looks for one break in the mean at an unknown date in the middle 70% of the sample.
Power is simulated at the break the data suggest: means before and after as estimated, noise resampled in blocks from
the residuals. The estimated break is the best of about 400 candidate dates, so its size is overstated and the
simulated power is an upper guide, not a promise.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

import ind_breaks as B
import ind_eval as E

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
LABELS = {
    "mom": "Momentum",
    "lasso": "Lasso",
    "rf": "Random forest",
    "mean_ml": "Average of ML models",
    "xlead": "Cross-industry lead–lag",
}


def block_resample(e: np.ndarray, rng: np.random.Generator, block: int = 6) -> np.ndarray:
    n = len(e)
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([e[s : s + block] for s in starts])[:n]


def power_at(x: np.ndarray, k: int, reps: int = 300, seed: int = 7) -> float:
    """Share of simulated samples with the estimated break (means m1, m2 at k) in which the 5% test rejects."""
    rng = np.random.default_rng(seed)
    m1, m2 = x[:k].mean(), x[k:].mean()
    e = np.where(np.arange(len(x)) < k, x - m1, x - m2)
    hits = 0
    for _ in range(reps):
        y = np.where(np.arange(len(x)) < k, m1, m2) + block_resample(e, rng)
        hits += B.sup_wald(y)["p_value"] < 0.05
    return hits / reps


def main() -> None:
    fc = pd.read_csv(RESULTS / "forecasts.csv", parse_dates=["date"])
    fc = fc[fc["y"].notna()].reset_index(drop=True)
    cx = pd.read_csv(RESULTS / "crossind_forecasts.csv", parse_dates=["date"])
    fc = fc.merge(cx, on=["date", "industry"], how="left")

    series: dict[str, pd.Series] = {}
    for m in ("mom", "lasso", "rf", "mean_ml", "xlead"):
        series[f"IC: {LABELS[m]}"] = E.monthly_ic(fc, m)
    series["IC: Lasso minus momentum"] = series["IC: Lasso"] - series["IC: Momentum"]
    for m in ("mom", "lasso", "rf"):
        series[f"Net return: {LABELS[m]}"] = E.long_short(fc, m)["net"]

    rows = []
    for name, s in series.items():
        s = s.dropna()
        x = s.to_numpy()
        r = B.sup_wald(x)
        k = r["break_after"]
        e = np.where(np.arange(len(x)) < k, x - r["mean_before"], x - r["mean_after"])
        se = np.sqrt(B.long_run_variance(e) * (1 / k + 1 / (len(x) - k)))
        scale = 1200 if name.startswith("Net") else 1
        known = np.where(s.index <= pd.Timestamp("1999-12-01"))[0].size  # the study's own split
        w_known = (
            (x[:known].mean() - x[known:].mean()) ** 2
            / (B.long_run_variance(np.where(np.arange(len(x)) < known, x - x[:known].mean(), x - x[known:].mean())) * (1 / known + 1 / (len(x) - known)))
        )  # fmt: skip
        rows.append({
            "series": name, "sup-Wald": r["sup_wald"], "p (simulated)": r["p_value"], "5% critical value": r["critical_5pct"],
            "last month before break": s.index[k - 1].strftime("%Y-%m"), "mean before": scale * r["mean_before"], "mean after": scale * r["mean_after"],
            "t of difference at that date": (r["mean_before"] - r["mean_after"]) / se,
            "fixed split at 1999-12: Wald": w_known, "fixed split p (chi2, 1 df)": float(stats.chi2.sf(w_known, 1)),
            "power at this break": power_at(x, k), "months": len(x),
        })  # fmt: skip
    out = pd.DataFrame(rows)
    out.round(4).to_csv(RESULTS / "breaks.csv", index=False)
    (RESULTS / "breaks.json").write_text(
        json.dumps(
            {"critical_5pct": B.null_draws()[int(0.95 * len(B.null_draws()))].item(), "trim": 0.15}, indent=1
        ),
        encoding="utf-8",
    )
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
