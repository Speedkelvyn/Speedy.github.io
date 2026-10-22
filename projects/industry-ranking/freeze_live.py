"""freeze_live.py: write the combined momentum + cross-industry ranking for the month that has not happened yet.

    python freeze_live.py     # reads results/forecasts.csv and results/crossind_forecasts.csv

The file is committed before the month's returns exist, so the git history dates the forecast. It is scored once
Kenneth French publishes the month: `score_mom_plus_x` (the specified signal) and `score_mom_plus_x_rank` (the robustness variant).
"""

from __future__ import annotations

import pathlib

import pandas as pd

import ind_extensions as X

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def main() -> pathlib.Path:
    fc = pd.read_csv(RESULTS / "forecasts.csv", parse_dates=["date"])
    cx = pd.read_csv(RESULTS / "crossind_forecasts.csv", parse_dates=["date"])
    last = fc["date"].max()
    live = fc[fc["date"] == last].merge(cx[cx["date"] == last], on=["date", "industry"], how="left")
    if len(live) != 49 or live["xlead"].isna().any() or live["y"].notna().any():
        raise SystemExit(
            "The last forecast month must have all 49 industries, a cross-industry forecast and no outcome yet."
        )
    live["score_mom_plus_x"] = X.zscore_sum(live, ["mom", "xlead"]).to_numpy()
    live["score_mom_plus_x_rank"] = X.rank_sum(live, ["mom", "xlead"]).to_numpy()
    live["rank"] = live["score_mom_plus_x"].rank(ascending=False).astype(int)
    live["rank_variant"] = live["score_mom_plus_x_rank"].rank(ascending=False).astype(int)
    target = (last + pd.DateOffset(months=1)).strftime("%Y-%m")
    out = live.sort_values("rank")[
        [
            "industry",
            "rank",
            "rank_variant",
            "score_mom_plus_x",
            "score_mom_plus_x_rank",
            "mom",
            "xlead",
            "lasso",
        ]
    ].round(4)
    out.insert(0, "forecast_for", target)
    out.insert(1, "made_with_data_through", last.strftime("%Y-%m"))
    path = RESULTS / f"live_forecast_{target}.csv"
    out.to_csv(path, index=False)
    print(out.head(10).to_string(index=False))
    print("...")
    print(out.tail(5).to_string(index=False))
    return path


if __name__ == "__main__":
    main()
