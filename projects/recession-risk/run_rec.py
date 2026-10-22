"""run_rec.py: run every model for both targets and save the probabilities to results/.

python run_rec.py     # about 10 minutes on a 2-core laptop
"""

from __future__ import annotations

import pathlib
import time

import pandas as pd
from joblib import Parallel, delayed

import rec_data as D
import rec_models as M

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
FIRST_TEST = "1980-01-01"  # the 1990-and-later results in the report are subsets of this run


def one(target: str, lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = D.build_dataset()
    t = time.time()
    fc, imp = M.recursive_probabilities(d["X"], d["targets"][target], lag, first_test=FIRST_TEST)
    fc.insert(0, "target", target)
    imp.insert(0, "target", target)
    print(f"{target}: {len(fc)} probabilities in {(time.time() - t) / 60:.1f} min", flush=True)
    return fc, imp


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    out = Parallel(n_jobs=2)(
        delayed(one)(t, lag) for t, lag in (("RECESSION", D.RECESSION_LAG), ("BEAR", D.BEAR_LAG))
    )
    pd.concat([o[0] for o in out]).to_csv(RESULTS / "probabilities.csv", index=False)
    pd.concat([o[1] for o in out]).to_csv(RESULTS / "importance.csv", index=False)
    print("saved results/probabilities.csv")
