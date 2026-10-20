"""run_experiments.py: run both out-of-sample experiments and save the forecasts to results/.

    python run_experiments.py            # both parts (the panel takes roughly 20-30 minutes on a laptop)
    python run_experiments.py --part 1   # daily USD/JPY AR(1)-GARCH(1,1) vs random walk only
    python run_experiments.py --part 2   # monthly panel only

The report (report.qmd) reads the saved results, so it renders in seconds.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import fx_forecasting as F
import models as M

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"


def part1() -> None:
    prices = F.read_daily("DEXJPUS_daily")
    t = time.time()
    out = M.garch_vs_random_walk(prices, train_share=0.8)
    out.to_csv(RESULTS / "part1_usdjpy_daily_forecasts.csv")
    print(f"Part I: {len(out)} daily forecasts in {time.time() - t:.0f}s")


def part2() -> None:
    panel = F.build_panel()
    panel.data.to_csv(RESULTS / "part2_panel.csv", index=False)
    t = time.time()
    out = M.recursive_panel_forecasts(panel, first_test="2012-01-01")
    out.to_csv(RESULTS / "part2_panel_forecasts.csv", index=False)
    print(f"Part II: {len(out)} currency-month forecasts in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", type=int, choices=[1, 2])
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if a.part in (None, 1):
        part1()
    if a.part in (None, 2):
        part2()
