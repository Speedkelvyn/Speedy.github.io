"""run_credit.py: run every model for both spreads and all horizons; save forecasts to results/.

python run_credit.py              # 3 spreads x 3 horizons, about an hour on a 2-core laptop
python run_credit.py --jobs 4     # use more cores if you have them
"""

from __future__ import annotations

import argparse
import pathlib
import time

import pandas as pd
from joblib import Parallel, delayed

import credit_data as C
import credit_models as M

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
HORIZONS = (1, 3, 6)
FIRST_TEST = "1995-01-01"


def one(target: str, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = C.build_dataset(target, h)
    t = time.time()
    fc, imp = M.recursive_forecasts(d["X"], d["y"], d["level"], h, first_test=FIRST_TEST)
    fc.insert(0, "target", target)
    imp.insert(0, "target", target)
    imp.insert(1, "horizon", h)
    print(f"{target} h={h}: {len(fc)} forecasts in {(time.time() - t) / 60:.1f} min", flush=True)
    return fc, imp


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--targets", nargs="*", default=list(C.TARGETS), choices=list(C.TARGETS))
    ap.add_argument("--out", default="forecasts", help="file name stem for the saved results")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    out = Parallel(n_jobs=a.jobs)(delayed(one)(t, h) for t in a.targets for h in HORIZONS)
    pd.concat([o[0] for o in out]).to_csv(RESULTS / f"{a.out}.csv", index=False)
    pd.concat([o[1] for o in out]).to_csv(
        RESULTS / ("importance.csv" if a.out == "forecasts" else f"{a.out}_importance.csv"), index=False
    )
    print(f"saved results/{a.out}.csv")
