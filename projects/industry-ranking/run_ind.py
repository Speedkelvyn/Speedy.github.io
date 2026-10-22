"""run_ind.py: run every industry-ranking model out of sample and save the forecasts to results/.

python run_ind.py     # about an hour on a 2-core laptop; writes results/forecasts.csv and importance.csv
"""

from __future__ import annotations

import pathlib
import time

import ind_data as D
import ind_models as M

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
FIRST_TEST = "1980-01-01"


def main() -> None:
    d = D.build_panel()
    t0 = time.time()
    fc, imp = M.recursive_forecasts(d["panel"], first_test=FIRST_TEST)
    print(
        f"{fc['date'].nunique()} months, {len(fc)} industry-month forecasts in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    RESULTS.mkdir(exist_ok=True)
    fc.to_csv(RESULTS / "forecasts.csv", index=False)
    imp.to_csv(RESULTS / "importance.csv", index=False)
    print("saved results/forecasts.csv and results/importance.csv")


if __name__ == "__main__":
    main()
