"""analyze_ind.py: turn results/forecasts.csv into every table and number quoted in the report.

python analyze_ind.py     # about two minutes; writes results/*.csv and results/headline.json
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import ind_data as D
import ind_eval as E
import ind_models as M

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
PERIODS = {
    "1980–1999": ("1980-01-01", "1999-12-31"),
    "2000–2012": ("2000-01-01", "2012-12-31"),
    "2013–2026": ("2013-01-01", "2026-12-31"),
}
COSTS = (0.0, 0.0010, 0.0020, 0.0040)
NAMES_49 = {
    "Agric": "Agriculture", "Food": "Food products", "Soda": "Candy and soda", "Beer": "Beer and liquor", "Smoke": "Tobacco",
    "Toys": "Recreation", "Fun": "Entertainment", "Books": "Printing and publishing", "Hshld": "Consumer goods",
    "Clths": "Apparel", "Hlth": "Healthcare", "MedEq": "Medical equipment", "Drugs": "Pharmaceuticals", "Chems": "Chemicals",
    "Rubbr": "Rubber and plastic", "Txtls": "Textiles", "BldMt": "Construction materials", "Cnstr": "Construction",
    "Steel": "Steel", "FabPr": "Fabricated products", "Mach": "Machinery", "ElcEq": "Electrical equipment",
    "Autos": "Automobiles", "Aero": "Aircraft", "Ships": "Shipbuilding", "Guns": "Defense", "Gold": "Precious metals",
    "Mines": "Metal mining", "Coal": "Coal", "Oil": "Petroleum and gas", "Util": "Utilities", "Telcm": "Communication",
    "PerSv": "Personal services", "BusSv": "Business services", "Hardw": "Computers", "Softw": "Software",
    "Chips": "Electronic equipment and chips", "LabEq": "Measuring and control equipment", "Paper": "Paper", "Boxes": "Containers",
    "Trans": "Transportation", "Whlsl": "Wholesale", "Rtail": "Retail", "Meals": "Restaurants and hotels", "Banks": "Banking",
    "Insur": "Insurance", "RlEst": "Real estate", "Fin": "Trading", "Other": "Other",
}  # fmt: skip


def ann_sharpe(r: pd.Series) -> float:
    return float(np.sqrt(12) * r.mean() / r.std())


def main() -> None:
    d = D.build_panel()
    fac = d["factors"]
    fc_all = pd.read_csv(RESULTS / "forecasts.csv", parse_dates=["date"])
    fc = fc_all[fc_all["y"].notna()].reset_index(drop=True)
    live = fc_all[fc_all["y"].isna()]
    out: dict = {
        "months": int(fc["date"].nunique()),
        "first": str(fc["date"].min().date()),
        "last": str(fc["date"].max().date()),
    }

    # ---- skill and portfolios for every model
    ic = {m: E.monthly_ic(fc, m) for m in M.MODELS}
    ls = {m: E.long_short(fc, m) for m in M.MODELS}
    rows = []
    for m in M.MODELS:
        a, t = E.factor_alpha(ls[m]["net"], fac)
        rows.append(
            {
                "key": m, "Model": M.NAMES[m], "Mean IC": ic[m].mean(), "IC t-stat": E.nw_t(ic[m]), "Months IC > 0 (%)": 100 * (ic[m] > 0).mean(),
                "Gross return (% p.a.)": 1200 * ls[m]["gross"].mean(), "Net return (% p.a.)": 1200 * ls[m]["net"].mean(),
                "Volatility (% p.a.)": np.sqrt(12) * 100 * ls[m]["net"].std(), "Sharpe, gross": ann_sharpe(ls[m]["gross"]),
                "Sharpe, net": ann_sharpe(ls[m]["net"]), "Annual turnover": 12 * ls[m]["turnover"].mean(),
                "Worst month (%)": 100 * ls[m]["net"].min(), "Max drawdown (%)": E.max_drawdown(ls[m]["net"]), "Alpha (% p.a.)": a, "Alpha t-stat": t,
                "R²_OS (%)": E.r2_os(fc, m),
            }
        )  # fmt: skip
    board = pd.DataFrame(rows)
    # ---- against the best single signal (12-1 momentum)
    mom_ic = ic["mom"]
    vs = []
    for m in [*M.ML, "mean_ml"]:
        diff = (ic[m] - mom_ic).dropna()
        vs.append({"key": m, "Model": M.NAMES[m], "IC minus momentum IC": diff.mean(), "t-stat": E.nw_t(diff),
                   "p (Sharpe, net > momentum)": E.sharpe_diff_p(ls[m]["net"], ls["mom"]["net"])})  # fmt: skip
    vs = pd.DataFrame(vs)
    board = board.merge(
        vs[["key", "IC minus momentum IC", "t-stat", "p (Sharpe, net > momentum)"]], on="key", how="left"
    )
    board.to_csv(RESULTS / "scoreboard.csv", index=False)

    # ---- sub-periods and costs for the headline models
    keys = ["mom", "lasso", "rf", "mean_ml"]
    sub = []
    for lab, (a, b) in PERIODS.items():
        for m in keys:
            r = ls[m]["net"][(ls[m].index >= a) & (ls[m].index <= b)]
            i = ic[m][(ic[m].index >= a) & (ic[m].index <= b)]
            sub.append({"Period": lab, "key": m, "Model": M.NAMES[m], "Mean IC": i.mean(), "IC t-stat": E.nw_t(i), "Net return (% p.a.)": 1200 * r.mean(), "Sharpe, net": ann_sharpe(r)})  # fmt: skip
    pd.DataFrame(sub).to_csv(RESULTS / "by_period.csv", index=False)
    cst = []
    for m in keys:
        for c in COSTS:
            r = E.long_short(fc, m, cost=c)["net"]
            cst.append({"key": m, "Model": M.NAMES[m], "Cost (bp per unit traded)": int(c * 1e4), "Net return (% p.a.)": 1200 * r.mean(), "Sharpe, net": ann_sharpe(r)})  # fmt: skip
    pd.DataFrame(cst).to_csv(RESULTS / "costs.csv", index=False)

    # ---- series for the figures
    pd.DataFrame({m: ls[m]["net"] for m in ["mom", "lasso", "rf", "mean_ml"]}).to_csv(
        RESULTS / "long_short_net.csv"
    )
    pd.DataFrame({m: ic[m] for m in ["mom", "lasso", "rf", "mean_ml"]}).to_csv(RESULTS / "ic_series.csv")
    # ---- what the models used
    imp = pd.read_csv(RESULTS / "importance.csv", parse_dates=["date"])
    imp.groupby("predictor")["rf_importance"].mean().sort_values(ascending=False).head(12).to_csv(
        RESULTS / "importance_top.csv"
    )

    # ---- the current ranking (last month, no outcome yet)
    cur = live[live["date"] == live["date"].max()].copy()
    cur["Industry"] = cur["industry"].map(NAMES_49)
    cur["Score"] = cur["mean_ml"]
    cur = cur.sort_values("Score", ascending=False)
    cur[["industry", "Industry", "Score", "lasso", "rf", "mom"]].to_csv(
        RESULTS / "current_ranking.csv", index=False
    )
    out["ranking_date"] = str(cur["date"].max().date())
    out["top5"] = cur["Industry"].head(5).tolist()
    out["bottom5"] = cur["Industry"].tail(5).tolist()
    (RESULTS / "headline.json").write_text(json.dumps(out, indent=2, default=float))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(board.drop(columns=["key"]).round(3).to_string(index=False))
    print(pd.DataFrame(sub).round(3).to_string(index=False))
    print(pd.DataFrame(cst).round(2).to_string(index=False))
    print(
        cur[["Industry", "Score"]].head(6).to_string(index=False),
        "\n",
        cur[["Industry", "Score"]].tail(6).to_string(index=False),
    )
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
