"""analyze_extensions.py: three follow-up experiments on the industry forecasts, written to results/extensions_*.csv.

    python analyze_extensions.py     # needs results/forecasts.csv and results/crossind_forecasts.csv (run_crossind.py)

A. Volatility-managing the long-short book (window 6 months, target 10% a year, cap 2, 20bp), fixed before the run;
   the 3x3 window/cap grid is reported as robustness only.
B. Where the machine-learning edge over momentum comes from: slope on momentum, crash months, ex-ante bear market.
C. A real-time cross-industry lead-lag model (run_crossind.py) alone and added to momentum.
"""

from __future__ import annotations

import itertools
import json
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

import ind_data as D
import ind_eval as E
import ind_extensions as X

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
PERIODS = {
    "1980–1999": ("1980-01-01", "1999-12-31"),
    "2000–2012": ("2000-01-01", "2012-12-31"),
    "2013–2026": ("2013-01-01", "2026-12-31"),
}


def sharpe(r: pd.Series) -> float:
    return float(np.sqrt(12) * r.mean() / r.std())


def load() -> pd.DataFrame:
    fc = pd.read_csv(RESULTS / "forecasts.csv", parse_dates=["date"])
    fc = fc[fc["y"].notna()].reset_index(drop=True)
    cx = pd.read_csv(RESULTS / "crossind_forecasts.csv", parse_dates=["date"])
    fc = fc.merge(cx, on=["date", "industry"], how="left")
    fc["mom_plus_x"] = X.zscore_sum(fc, ["mom", "xlead"])
    fc["lasso_plus_x"] = X.zscore_sum(fc, ["lasso", "xlead"])
    fc["mom_plus_x_rank"] = X.rank_sum(fc, ["mom", "xlead"])  # robustness: ranks instead of z-scores
    return fc


def hac_mean(x: pd.Series) -> tuple[float, float]:
    """Mean and Newey–West t of a series, from a regression on a constant."""
    r = sm.OLS(x.dropna().to_numpy(), np.ones(len(x.dropna()))).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return float(r.params[0]), float(r.tvalues[0])


def experiment_a(fc: pd.DataFrame, fac: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, grid = [], []
    for m in ("mom", "lasso", "rf", "mean_ml", "mom_plus_x"):
        gross, W = X.book(fc, m)
        base = X.net_unmanaged(gross, W)
        net, scale, turn = X.vol_managed(gross, W)
        for kind, r, tn, sc in (
            ("unmanaged", base, X.turnover(W), 1.0),
            ("vol-managed", net, turn, scale.mean()),
        ):
            a, t = E.factor_alpha(r, fac)
            rows.append({
                "model": m, "kind": kind, "net (% p.a.)": 1200 * r.mean(), "vol (% p.a.)": 100 * np.sqrt(12) * r.std(),
                "Sharpe": sharpe(r), "max drawdown (%)": E.max_drawdown(r), "worst month (%)": 100 * r.min(),
                "alpha (% p.a.)": a, "alpha t": t, "turnover / yr": 12 * tn.mean(), "mean scale": sc,
                "p (Sharpe > unmanaged)": E.sharpe_diff_p(r, base) if kind == "vol-managed" else np.nan,
            })  # fmt: skip
        for win, cap in itertools.product((3, 6, 12), (1.5, 2.0, 3.0)):
            g = X.vol_managed(gross, W, win=win, cap=cap)[0]
            grid.append(
                {
                    "model": m,
                    "window": win,
                    "cap": cap,
                    "Sharpe managed": sharpe(g),
                    "Sharpe unmanaged": sharpe(base),
                }
            )
        sig = X.trailing_vol(gross).dropna()
        rows[-1]["corr(trailing vol, next |return|)"] = float(
            np.corrcoef(sig, gross.reindex(sig.index).abs())[0, 1]
        )
    return pd.DataFrame(rows), pd.DataFrame(grid)


def experiment_b(fc: pd.DataFrame, fac: pd.DataFrame) -> pd.DataFrame:
    g_mom, W_mom = X.book(fc, "mom")
    n_mom = X.net_unmanaged(g_mom, W_mom)
    worst = g_mom.nsmallest(10).index
    bear = X.bear_flag(fac["mkt"]).reindex(g_mom.index)
    rows = []
    for m in ("lasso", "rf", "mean_ml", "mom_plus_x"):
        g, W = X.book(fc, m)
        n = X.net_unmanaged(g, W)
        ols = sm.OLS(g.to_numpy(), sm.add_constant(g_mom.reindex(g.index).to_numpy())).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3}
        )
        keep = ~g.index.isin(worst)  # the same regression without momentum's ten worst months
        ols_x = sm.OLS(g[keep].to_numpy(), sm.add_constant(g_mom.reindex(g.index)[keep].to_numpy())).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3}
        )
        edge = n - n_mom
        mean_edge, t_edge = hac_mean(edge)
        ok = bear.notna()
        reg = sm.OLS(edge[ok].to_numpy(), sm.add_constant(bear[ok].astype(float).to_numpy())).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3}
        )
        rows.append({
            "model": m, "slope on momentum": float(ols.params[1]), "alpha vs momentum (% p.a., gross)": 1200 * float(ols.params[0]),
            "alpha t": float(ols.tvalues[0]), "alpha without the 10 worst months (% p.a.)": 1200 * float(ols_x.params[0]),
            "alpha t without": float(ols_x.tvalues[0]), "net edge over momentum (% p.a.)": 1200 * mean_edge, "edge t": t_edge,
            "momentum's 10 worst months (%)": 100 * g_mom[worst].mean(), "this model, same months (%)": 100 * g[worst].mean(),
            "share of edge earned in those 10 months": float(edge[worst].sum() / edge.sum()),
            "edge in bear market (% p.a.)": 1200 * float(edge[bear == True].mean()),  # noqa: E712
            "edge otherwise (% p.a.)": 1200 * float(edge[bear == False].mean()),  # noqa: E712
            "bear minus other, p (two-sided)": float(reg.pvalues[1]), "bear months": int((bear == True).sum()),  # noqa: E712
        })  # fmt: skip
    return pd.DataFrame(rows)


def experiment_c(fc: pd.DataFrame, fac: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    net_mom = E.long_short(fc, "mom")["net"]
    rows = []
    for m in ("mom", "lasso", "xind", "xlead", "mom_plus_x", "lasso_plus_x", "mom_plus_x_rank"):
        ic, ls = E.monthly_ic(fc, m), E.long_short(fc, m)
        a, t = E.factor_alpha(ls["net"], fac)
        rows.append({
            "model": m, "IC": ic.mean(), "IC t": E.nw_t(ic), "net (% p.a.)": 1200 * ls["net"].mean(), "Sharpe": sharpe(ls["net"]),
            "turnover / yr": 12 * ls["turnover"].mean(), "alpha (% p.a.)": a, "alpha t": t,
            "p (Sharpe > momentum)": E.sharpe_diff_p(ls["net"], net_mom) if m != "mom" else np.nan,
        })  # fmt: skip
    per = []
    for lab, (a, b) in PERIODS.items():
        sub = fc[(fc["date"] >= a) & (fc["date"] <= b)]
        for m in ("mom", "lasso", "xlead", "mom_plus_x"):
            ls = E.long_short(sub, m)
            per.append(
                {
                    "period": lab,
                    "model": m,
                    "IC": E.monthly_ic(sub, m).mean(),
                    "net Sharpe": sharpe(ls["net"]),
                }
            )
    gain = (E.monthly_ic(fc, "mom_plus_x") - E.monthly_ic(fc, "mom")).dropna()
    extra = {
        "ic_gain": float(gain.mean()),
        "ic_gain_t": E.nw_t(gain),
        "mean_active_predictors": float(fc["n_active"].mean()),
    }
    return pd.DataFrame(rows), pd.DataFrame(per), extra


def main() -> None:
    fc, fac = load(), D.load_market()
    a, grid = experiment_a(fc, fac)
    b = experiment_b(fc, fac)
    c, per, extra = experiment_c(fc, fac)
    a.round(4).to_csv(RESULTS / "extensions_volmanaged.csv", index=False)
    grid.round(4).to_csv(RESULTS / "extensions_volgrid.csv", index=False)
    b.round(4).to_csv(RESULTS / "extensions_decomposition.csv", index=False)
    c.round(4).to_csv(RESULTS / "extensions_crossind.csv", index=False)
    per.round(4).to_csv(RESULTS / "extensions_crossind_periods.csv", index=False)
    (RESULTS / "extensions.json").write_text(json.dumps(extra, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for name, t in (
        ("A. volatility-managed", a),
        ("B. decomposition", b),
        ("C. cross-industry", c),
        ("C. by period", per),
    ):
        print(f"\n{name}\n{t.round(3).to_string(index=False)}")
    print("\nrobustness grid (Sharpe managed; unmanaged in last column)")
    print(grid.pivot_table(index=["model", "window"], columns="cap", values="Sharpe managed").round(3))
    print(grid.groupby("model")["Sharpe unmanaged"].first().round(3).to_dict(), extra)


if __name__ == "__main__":
    main()
