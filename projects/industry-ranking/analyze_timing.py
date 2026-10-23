"""analyze_timing.py: does a random-feature ridge model that times the momentum book add anything to volatility-managing?

    python analyze_timing.py     # about 2 minutes; writes results/timing_*.csv and results/timing.json

Pre-specified before the run: features as in ind_timing.timing_features, 3000 random Fourier features (more than five
times the months at the first forecast), kernel width 0.5, ridge penalty 1, refit every 3 months, forecasts from January
1980, position = forecast scaled to an average absolute size of one, 20bp per unit traded. The 3 x 4 grid of widths and
penalties is robustness only.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

import ind_data as D
import ind_eval as E
import ind_extensions as X
import ind_timing as T

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
FIRST = "1980-01-01"
GAMMAS = (0.25, 0.5, 1.0)
RIDGES = (0.01, 1.0, 100.0, 10000.0)
MAIN = (0.5, 1.0)  # kernel width, ridge penalty fixed in advance


def sharpe(r: pd.Series) -> float:
    return float(np.sqrt(12) * r.mean() / r.std())


def summarise(
    name: str,
    net: pd.Series,
    turn: pd.Series,
    ref: pd.Series,
    gross: pd.Series,
    fc: pd.Series | None,
    y: pd.Series,
) -> dict:
    ols = sm.OLS(net.to_numpy(), sm.add_constant(ref.reindex(net.index).to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    return {
        "strategy": name, "net (% p.a.)": 1200 * net.mean(), "vol (% p.a.)": 100 * np.sqrt(12) * net.std(), "Sharpe": sharpe(net),
        "Sharpe, gross": sharpe(gross), "max drawdown (%)": E.max_drawdown(net), "turnover / yr": 12 * turn.mean(),
        "R2_OOS (%)": T.r2_oos(fc, y) if fc is not None else np.nan,
        "corr with vol-managed": float(net.corr(ref.reindex(net.index))),
        "alpha vs vol-managed (% p.a.)": 1200 * float(ols.params[0]), "alpha t": float(ols.tvalues[0]),
        "p (Sharpe > vol-managed)": E.sharpe_diff_p(net, ref.reindex(net.index)) if name != "Volatility-managed" else np.nan,
    }  # fmt: skip


def main() -> None:
    p = D.build_panel()["panel"]
    g, W = X.book(p.dropna(subset=["ret_next"]), "mom_12_1")  # momentum book from 1940
    states = p.groupby("date")[["mkt_vol", "mkt_ret_12"]].first()
    f = T.timing_features(g, states)
    idx = g.index[g.index >= FIRST]
    g1, W1 = g.loc[idx], W.loc[idx]

    # ---- benchmarks
    hold_net = X.net_unmanaged(g1, W1)
    vm_net, scale, vm_turn = X.vol_managed(g1, W1)  # 6-month window, 10% target, cap 2
    pis = {"Buy and hold": pd.Series(1.0, index=idx), "Volatility-managed": scale}
    pis["Trailing-return timing"] = T.normalise(T.trailing_return_signal(g).loc[idx])
    lin = T.linear_ridge_forecasts(f, g, FIRST)
    pis["Linear ridge, same features"] = T.normalise(lin)
    forecasts = {"Linear ridge, same features": lin}

    # ---- random-feature models on the grid (kernel shared by the ridge penalties of one width)
    grid_rows = []
    for gamma in GAMMAS:
        fr = T.rff_forecasts(f, g, FIRST, gamma=gamma, ridges=RIDGES)
        for z in RIDGES:
            pi = T.normalise(fr[z])
            net, _ = T.timed_net(g, W, pi)
            grid_rows.append(
                {
                    "kernel width": gamma,
                    "ridge penalty": z,
                    "Sharpe": sharpe(net),
                    "R2_OOS (%)": T.r2_oos(fr[z], g),
                }
            )
            if (gamma, z) == MAIN:
                pis["Random-feature ridge"] = pi
                forecasts["Random-feature ridge"] = fr[z]
    grid = pd.DataFrame(grid_rows)

    # ---- the same forecasts sized by variance (mean-variance position): is there information beyond the variance?
    var6 = g.shift(1).rolling(6, min_periods=6).var()
    pis["Constant mean / variance"] = T.normalise((g.expanding().mean().shift(1) / var6).loc[idx])
    pis["Random-feature ridge / variance"] = T.normalise(forecasts["Random-feature ridge"] / var6.loc[idx])

    rows, nets = [], {}
    for name, pi in pis.items():
        if name == "Volatility-managed":
            net, turn, gross = vm_net, vm_turn, scale * g1
        elif name == "Buy and hold":
            net, turn, gross = hold_net, X.turnover(W1), g1
        else:
            net, turn = T.timed_net(g, W, pi)
            gross = pi * g.loc[pi.index]
        nets[name] = net
    ref = nets["Volatility-managed"]
    for name, pi in pis.items():
        if name == "Volatility-managed":
            net, turn, gross = vm_net, vm_turn, scale * g1
        elif name == "Buy and hold":
            net, turn, gross = hold_net, X.turnover(W1), g1
        else:
            net, turn = T.timed_net(g, W, pi)
            gross = pi * g.loc[pi.index]
        rows.append(summarise(name, net, turn, ref, gross, forecasts.get(name), g))
    table = pd.DataFrame(rows)

    # ---- what the random-feature position is made of
    rff_pi = pis["Random-feature ridge"]
    r12 = g.shift(1).rolling(12).mean().loc[idx]
    inv_vol = scale
    reg = sm.OLS(
        rff_pi.to_numpy(), sm.add_constant(np.column_stack([inv_vol.to_numpy(), r12.to_numpy()]))
    ).fit()
    span = sm.OLS(
        nets["Random-feature ridge"].to_numpy(),
        sm.add_constant(
            np.column_stack(
                [
                    nets["Volatility-managed"].to_numpy(),
                    nets["Trailing-return timing"].to_numpy(),
                    nets["Buy and hold"].to_numpy(),
                ]
            )
        ),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    info = {
        "months": len(idx), "features": int(f.shape[1]), "random_features": 3000,
        "training_months_first": int((f.index < FIRST).sum()), "training_months_last": int(len(f) - 1),
        "corr_position_vol_scale": float(rff_pi.corr(inv_vol)), "corr_position_trailing_return": float(rff_pi.corr(r12)),
        "r2_position_on_scale_and_trailing": float(reg.rsquared),
        "spanning_alpha_pct": 1200 * float(span.params[0]), "spanning_alpha_t": float(span.tvalues[0]),
        "spanning_betas": [float(b) for b in span.params[1:]],
        "frac_months_rff_negative": float((rff_pi < 0).mean()),
    }  # fmt: skip

    table.round(4).to_csv(RESULTS / "timing_strategies.csv", index=False)
    grid.round(4).to_csv(RESULTS / "timing_grid.csv", index=False)
    (RESULTS / "timing.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.round(3).to_string(index=False))
    print(grid.pivot(index="kernel width", columns="ridge penalty", values="Sharpe").round(3))
    print(grid.pivot(index="kernel width", columns="ridge penalty", values="R2_OOS (%)").round(2))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
