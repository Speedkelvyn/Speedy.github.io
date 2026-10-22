"""analyze_vol.py: turn results/forecasts.csv into every table and number quoted in the report.

python analyze_vol.py     # about a minute; writes results/*.csv and results/headline.json
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import vol_data as V
import vol_eval as E
import vol_models as M

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
STRATEGY_MODELS = ["naive", "ewma", "garch", "har", "har_x", "vix", "har_vix", "har_cred", "ridge", "mean_ml"]
CRISES = [("2008-09-01", "2009-06-30"), ("2020-02-01", "2020-06-30")]  # return months excluded in "ex-crisis"


def ann_vol(h: pd.Series) -> pd.Series:
    """Monthly variance forecast -> annualised volatility in %."""
    return 100 * np.sqrt(12 * h)


def main() -> None:
    d = V.build_dataset()
    rv_hist = d["monthly"]["rv"]
    fc_all = pd.read_csv(RESULTS / "forecasts.csv", parse_dates=["date"])
    live = fc_all[fc_all["actual"].isna()].reset_index(drop=True)
    fc = fc_all[fc_all["actual"].notna()].reset_index(drop=True)
    idx = pd.DatetimeIndex(fc["date"])
    ret_month = idx + pd.offsets.MonthBegin(1)  # the month in which the return is earned
    out: dict = {"n_months": len(fc), "first": str(idx[0].date()), "last": str(idx[-1].date())}

    # ---- accuracy
    acc = E.accuracy_table(fc, M.MODELS, M.NAMES)
    acc["MCS p (QLIKE)"] = E.model_confidence_set(fc, M.MODELS, "qlike").to_numpy()
    acc["MCS p (squared error)"] = E.model_confidence_set(fc, M.MODELS, "mse").to_numpy()
    acc.to_csv(RESULTS / "accuracy.csv", index=False)
    out["mcs_qlike_in"] = acc.loc[acc["MCS p (QLIKE)"] > 0.10, "key"].tolist()
    out["mcs_mse_in"] = acc.loc[acc["MCS p (squared error)"] > 0.10, "key"].tolist()

    # ---- accuracy by period, so that one episode cannot carry the table
    per = []
    for lab, (a, b) in {
        "2000-2012": ("2000-01-01", "2012-12-31"),
        "2013-2026": ("2013-01-01", "2026-12-31"),
    }.items():
        s = fc[(fc["date"] >= a) & (fc["date"] <= b)].reset_index(drop=True)
        t = E.accuracy_table(
            s, ["ewma", "garch", "har", "har_x", "vix", "har_cred", "ridge", "mean_ml"], M.NAMES
        )
        per.append(
            t.assign(Period=lab)[
                ["Period", "key", "Model", "R²_OS (%)", "R²_OS vs HAR (%)", "QLIKE", "QLIKE vs HAR (%)"]
            ]
        )
    pd.concat(per).to_csv(RESULTS / "accuracy_by_period.csv", index=False)

    # ---- strategy
    bh = E.buy_and_hold(fc)
    rf = pd.Series(fc["rf_next"].to_numpy(), index=idx)
    rows, series = {}, {"Buy and hold": bh}
    rows["Buy and hold"] = E.perf(bh, rf)
    for m in STRATEGY_MODELS:
        res = E.managed(fc, m, rv_hist)
        series[M.NAMES[m]] = res["ret"]
        r = E.perf(res["ret"], rf, res["w"], bh)
        r["Annual turnover"] = 12 * float(res["turnover"].mean())
        r["CE gain vs market (% p.a., gamma 5)"] = E.certainty_equivalent_gain(res["ret"], bh)
        rows[M.NAMES[m]] = r
    strat = pd.DataFrame(rows).T
    strat.to_csv(RESULTS / "strategy.csv")
    pd.DataFrame(series).to_csv(RESULTS / "strategy_returns.csv")

    # ---- robustness for the headline forecaster (set by accuracy, but we print every case)
    robust = {}
    for m in ["garch", "har_cred", "ridge"]:
        for lab, kw in {
            "baseline (cap 2, 10bp)": {},
            "no leverage (cap 1)": {"cap": 1.0},
            "no costs": {"cost": 0.0, "borrow": 0.0},
            "25bp trading cost": {"cost": 0.0025},
            "50bp trading cost": {"cost": 0.0050},
        }.items():
            res = E.managed(fc, m, rv_hist, **kw)
            cap = kw.get("cap", E.CAP)
            r = E.perf(res["ret"], rf, res["w"], bh, cap=cap)
            r["Sharpe, buy and hold"] = E.perf(bh)["Sharpe ratio"]
            robust[(M.NAMES[m], lab)] = r
        crisis = pd.Series(False, index=idx)
        for a, b in CRISES:
            crisis |= (ret_month >= pd.Timestamp(a)) & (ret_month <= pd.Timestamp(b))
        res = E.managed(fc, m, rv_hist)
        keep = ~crisis.to_numpy()
        r = E.perf(res["ret"][keep], rf[keep], res["w"][keep], bh[keep])
        r["Sharpe, buy and hold"] = E.perf(bh[keep])["Sharpe ratio"]
        robust[(M.NAMES[m], "excluding 2008-09 and 2020")] = r
        for lab, (a, b) in {
            "2000-2012": ("2000-01-01", "2012-12-31"),
            "2013-2026": ("2013-01-01", "2026-12-31"),
        }.items():
            k = (idx >= a) & (idx <= b)
            r = E.perf(res["ret"][k], rf[k], res["w"][k], bh[k])
            r["Sharpe, buy and hold"] = E.perf(bh[k])["Sharpe ratio"]
            robust[(M.NAMES[m], lab)] = r
    pd.DataFrame(robust).T.to_csv(RESULTS / "strategy_robustness.csv")

    # ---- the live forecast for the coming month
    L = pd.concat([fc, live], ignore_index=True)
    lv = {m: float(ann_vol(E.level_forecast(L, m)).iloc[-1]) for m in M.MODELS}
    out["live_month_forecast"] = str((pd.DatetimeIndex(live["date"])[0] + pd.offsets.MonthBegin(1)).date())
    out["live_vol_pct"] = lv
    hist = {m: ann_vol(E.level_forecast(L, m)).iloc[:-1] for m in ["garch", "har_cred", "ridge"]}
    out["live_percentile"] = {m: float((hist[m] <= lv[m]).mean()) for m in hist}
    out["market_vol_last_month_pct"] = float(ann_vol(live["rv_now"]).iloc[0])
    out["avg_vol_pct"] = float(ann_vol(rv_hist.expanding().mean()).iloc[-1])
    for m in ["garch", "har_cred", "ridge"]:
        target = float(rv_hist.expanding().mean().iloc[-1])
        out.setdefault("live_weight", {})[m] = float(
            np.clip(np.sqrt(target / (lv[m] ** 2 / 1e4 / 12)), 0, E.CAP)
        )

    # ---- what the models relied on
    imp = pd.read_csv(RESULTS / "importance.csv", parse_dates=["date"])
    g = imp.groupby("predictor").agg(
        rf_importance=("rf_importance", "mean"),
        lasso_selected=("lasso_selected", "mean"),
        years=("date", "nunique"),
    )
    g.sort_values("rf_importance", ascending=False).head(15).to_csv(RESULTS / "importance_top.csv")

    # ---- simple descriptives
    out["rv_rank_corr"] = float(fc[["actual", "garch"]].corr(method="spearman").iloc[0, 1])
    out["bh_sharpe"] = float(E.perf(bh)["Sharpe ratio"])
    (RESULTS / "headline.json").write_text(json.dumps(out, indent=2, default=float))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(acc.drop(columns=["key"]).round(3).to_string(index=False))
    print(strat.round(2).to_string())
    print(pd.DataFrame(robust).T.round(2).to_string())
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
