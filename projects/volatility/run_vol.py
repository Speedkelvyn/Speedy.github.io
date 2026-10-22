"""run_vol.py: run every volatility model out of sample and save the forecasts to results/.

python run_vol.py        # about 10 minutes on a 2-core laptop; writes results/forecasts.csv and importance.csv
"""

from __future__ import annotations

import pathlib
import time

import pandas as pd

import vol_baselines as B
import vol_data as V
import vol_models as M

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
FIRST_TEST = "2000-01-01"


def main() -> None:
    d = V.build_dataset()
    X, y, m = d["X"], d["y"], d["monthly"]
    t0 = time.time()
    fc, imp = M.recursive_forecasts(X, y, first_test=FIRST_TEST)
    print(f"{len(fc)} forecasts in {(time.time() - t0) / 60:.1f} min", flush=True)
    dates = pd.DatetimeIndex(fc["date"])
    n_next = m["n_days"].shift(-1).fillna(21)  # September 2026 has 21 trading days
    fc["ewma"] = B.ewma_forecast(d["daily"], dates, n_next).to_numpy()
    fc["garch"] = B.garch_forecast(d["daily"], dates, n_next).to_numpy()
    fc["rv_next"] = d["rv_next"].reindex(dates).to_numpy()
    fc["ex_next"] = d["ex_next"].reindex(dates).to_numpy()
    fc["rf_next"] = d["rf_next"].reindex(dates).to_numpy()
    fc["rv_now"] = d["rv_now"].reindex(dates).to_numpy()
    fc = fc[["date", "actual", *M.MODELS, "rv_next", "ex_next", "rf_next", "rv_now"]]
    RESULTS.mkdir(exist_ok=True)
    fc.to_csv(RESULTS / "forecasts.csv", index=False)
    imp.to_csv(RESULTS / "importance.csv", index=False)
    print("saved results/forecasts.csv and results/importance.csv")


if __name__ == "__main__":
    main()
