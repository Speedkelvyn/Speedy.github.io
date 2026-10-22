"""analyze_rec.py: turn results/probabilities.csv into every table and number quoted in the report.

python analyze_rec.py     # about a minute; writes results/*.csv and results/headline.json
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

import rec_data as D
import rec_eval as E
import rec_models as M

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
LEAD_MODELS = ["probit_ts", "probit_ebp", "rf", "mean_prob"]
THRESHOLDS = (0.2, 0.3, 0.5)


def main() -> None:
    d = D.build_dataset()
    p = pd.read_csv(RESULTS / "probabilities.csv", parse_dates=["date"])
    out: dict = {}
    for tgt in ("RECESSION", "BEAR"):
        fc = p[p["target"] == tgt].reset_index(drop=True)
        common = fc.dropna(subset=["actual", *M.MODELS]).reset_index(drop=True)
        out[tgt] = {
            "sample_start": str(common["date"].min().date()),
            "sample_end": str(common["date"].max().date()),
            "months": len(common),
            "event_months": int(common["actual"].sum()),
            "base_rate_in_sample": float(common["actual"].mean()),
        }
        t = E.table(common, M.MODELS, M.NAMES, reps=2000)
        t.to_csv(RESULTS / f"scoreboard_{tgt.lower()}.csv", index=False)
        # the same table on each model's own, longer sample (lasso, forest and boosting start in 1980)
        longer = fc.dropna(subset=["actual"]).copy()
        rows = []
        for m in ("lasso", "rf", "gbm"):
            s = longer.dropna(subset=[m])
            s = s[s["date"] >= "1990-01-01"]
            rows.append({"Model": M.NAMES[m], "From": str(s["date"].min().date()), "Months": len(s),
                         "AUC": E.roc_auc_score(s["actual"], s[m]), "Log score": E.log_score(s["actual"].to_numpy(), s[m].to_numpy())})  # fmt: skip
        pd.DataFrame(rows).to_csv(RESULTS / f"longer_sample_{tgt.lower()}.csv", index=False)
        # months in which a probit signalled and what followed
        alarms = []
        for m in ("probit_ts", "probit_ebp", "rf", "mean_prob"):
            for th in THRESHOLDS:
                alarms.append({"key": m, "Model": M.NAMES[m], **E.alarm_stats(common, m, th)})
        pd.DataFrame(alarms).to_csv(RESULTS / f"alarms_{tgt.lower()}.csv", index=False)
        # latest probabilities from the last month for which every input exists
        live = fc[fc["actual"].isna()].dropna(subset=M.MODELS)
        last = live.iloc[-1]
        out[tgt]["latest_date"] = str(last["date"].date())
        out[tgt]["latest"] = {m: float(last[m]) for m in M.MODELS}
        out[tgt]["peak_since_2024"] = {m: float(fc[fc["date"] >= "2024-01-01"][m].max()) for m in ("probit_ts", "probit_ebp", "mean_prob")}  # fmt: skip
    leads = {m: E.lead_table(p[p["target"] == "RECESSION"], d["recession"], m) for m in LEAD_MODELS}
    pd.concat(leads, names=["key"]).reset_index(level=0).to_csv(RESULTS / "leads.csv", index=False)
    imp = pd.read_csv(RESULTS / "importance.csv", parse_dates=["date"])
    for tgt in ("RECESSION", "BEAR"):
        g = imp[imp["target"] == tgt].groupby("predictor").agg(
            rf_importance=("rf_importance", "mean"), lasso_coef=("lasso_coef", "mean"),
            lasso_nonzero=("lasso_coef", lambda s: float((s != 0).mean())),
        )  # fmt: skip
        g.sort_values("rf_importance", ascending=False).head(12).to_csv(
            RESULTS / f"importance_top_{tgt.lower()}.csv"
        )
    X = d["X"]
    out["inputs_latest"] = {
        k: float(X[k].dropna().iloc[-1]) for k in ("TERM_10Y3M", "EBP", "BAA_SPR", "NFCI")
    }
    out["inputs_latest_date"] = str(X["TERM_10Y3M"].dropna().index[-1].date())
    (RESULTS / "headline.json").write_text(json.dumps(out, indent=2, default=float))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for tgt in ("RECESSION", "BEAR"):
        print(tgt)
        print(
            pd.read_csv(RESULTS / f"scoreboard_{tgt.lower()}.csv")
            .drop(columns=["key"])
            .round(3)
            .to_string(index=False)
        )
        print(pd.read_csv(RESULTS / f"longer_sample_{tgt.lower()}.csv").round(3).to_string(index=False))
        print(pd.read_csv(RESULTS / f"alarms_{tgt.lower()}.csv").round(1).to_string(index=False))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
