"""Timing, loss-function and strategy checks for the volatility study."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import vol_baselines as B
import vol_data as V
import vol_eval as E
import vol_models as M


@pytest.fixture(scope="module")
def data():
    return V.build_dataset()


# ------------------------------------------------------------------ data timing
def test_target_is_next_months_realised_variance(data):
    d, m = data["daily"], data["monthly"]
    t = pd.Timestamp("2008-09-01")
    nxt = d.loc["2008-10-01":"2008-10-31", "mkt"]
    assert data["y"].loc[t] == pytest.approx(np.log((nxt**2).sum()))
    assert data["rv_next"].loc[t] == pytest.approx((nxt**2).sum())
    assert data["rv_now"].loc[t] == pytest.approx(m.loc[t, "rv"])


def test_features_describe_the_month_just_ended(data):
    t = pd.Timestamp("2015-08-01")
    d = data["daily"].loc["2015-08-01":"2015-08-31", "mkt"]
    X = data["X"]
    assert X.loc[t, "lrv"] == pytest.approx(np.log((d**2).sum()))
    assert X.loc[t, "lrv_5d"] == pytest.approx(np.log((d.iloc[-5:] ** 2).sum() * 21 / 5))
    assert X.loc[t, "ret_1m"] == pytest.approx(float(np.prod(1 + d) - 1))


def test_vix_is_the_last_trading_day_not_the_monthly_average(data):
    # VIX closed at 53.54 on 31 March 2020; its March average was 57.7
    v = np.sqrt(12 * np.exp(data["X"].loc["2020-03-01", "lvix2"])) * 100
    assert v == pytest.approx(53.54, abs=0.01)


def test_last_row_is_the_live_month_without_an_outcome(data):
    assert np.isnan(data["y"].iloc[-1]) and data["y"].iloc[:-1].notna().sum() > 1000


def test_macro_data_lagged_market_data_not():
    raw = pd.read_csv(V.RAW / "fred_md_2026-09.csv").iloc[1:]
    raw.index = pd.to_datetime(raw.pop("sasdate"), format="%m/%d/%Y")
    md = V.load_fred_md()
    t = pd.Timestamp("2015-06-01")
    may = float(raw.loc["2015-05-01", "UNRATE"]) - float(raw.loc["2015-04-01", "UNRATE"])
    assert md.loc[t, "md_UNRATE"] == pytest.approx(may)  # published with a delay
    june = float(raw.loc["2015-06-01", "GS10"]) - float(raw.loc["2015-05-01", "GS10"])
    assert md.loc[t, "md_GS10"] == pytest.approx(june)  # a market rate, known on the day


# ------------------------------------------------------------------ daily-return baselines
def test_ewma_and_garch_ignore_the_future(data):
    daily = data["daily"]
    month_ends = pd.DatetimeIndex(["2005-03-01", "2008-10-01"])
    n_next = pd.Series(21, index=month_ends)
    cut = daily.loc[:"2008-10-31"]
    a = B.ewma_forecast(daily, month_ends, n_next)
    b = B.ewma_forecast(cut, month_ends, n_next)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    g = B.garch_forecast(daily.loc[:"2009-12-31"], month_ends, n_next)
    g2 = B.garch_forecast(daily.loc[:"2010-12-31"], month_ends, n_next)
    assert np.allclose(g.to_numpy(), g2.to_numpy())


def test_garch_filter_matches_the_recursion_by_hand():
    r = np.array([0.0] * 250 + [-2.0, 1.0])
    v = B._gjr_filter(r, omega=0.1, alpha=0.05, gamma=0.1, beta=0.9)
    prev = r[:250].var()
    for x in r[:251]:
        prev = 0.1 + (0.05 + 0.1 * (x < 0)) * x * x + 0.9 * prev
    assert v[250] == pytest.approx(prev)


# ------------------------------------------------------------------ preparation and the recursive loop
def test_prep_uses_training_rows_only():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=200, freq="MS")
    X = pd.DataFrame(rng.normal(size=(200, 3)), index=idx, columns=list("abc"))
    X.iloc[150:, 0] = 1e6  # an extreme future regime
    p = M.Prep().fit(X.iloc[:100])
    Z = p.transform(X.iloc[:100])
    assert np.allclose(Z.mean(axis=0), 0, atol=0.2) and np.allclose(Z.std(axis=0), 1, atol=0.2)
    assert p.hi["a"] < 10  # the future outlier did not move the winsorising bounds


def test_small_regression_never_sees_the_test_month_outcome():
    rng = np.random.default_rng(1)
    idx = pd.date_range("1990-01-01", periods=150, freq="MS")
    X = pd.DataFrame(
        {"lrv": rng.normal(size=150), "lrv_3": rng.normal(size=150), "lrv_12": rng.normal(size=150)},
        index=idx,
    )
    y = pd.Series(0.5 * X["lrv"] + rng.normal(scale=0.1, size=150), index=idx)
    t = idx[120]
    known = (idx <= t - pd.DateOffset(months=1)) & y.notna().to_numpy()
    f1 = M.small_forecast(X, y, known, M.HAR, X.loc[[t]])
    y2 = y.copy()
    y2.loc[idx >= t] = 1e3
    f2 = M.small_forecast(X, y2, known, M.HAR, X.loc[[t]])
    assert f1 == pytest.approx(f2)


def test_recursive_forecasts_do_not_change_when_the_future_changes():
    rng = np.random.default_rng(2)
    idx = pd.date_range("1985-01-01", periods=210, freq="MS")
    cols = ["lrv", "lrv_3", "lrv_12", "lrv_5d", "semi_neg", "abs_last", "ret_1m", "ret_12m", "lvix2", "BAA_SPR", "EBP",
            "NFCI", "TERM_10Y3M", "x1", "x2", "x3"]  # fmt: skip
    X = pd.DataFrame(rng.normal(size=(210, len(cols))), index=idx, columns=cols)
    y = pd.Series(0.6 * X["lrv"] + 0.2 * X["lvix2"] + rng.normal(scale=0.3, size=210), index=idx)
    first = idx[196]
    a, _ = M.recursive_forecasts(X, y, first_test=str(first.date()), refit_every=12)
    y2 = y.copy()
    y2.loc[idx > first] += 50.0
    X2 = X.copy()
    X2.loc[idx > first] *= 7.0
    b, _ = M.recursive_forecasts(X2, y2, first_test=str(first.date()), refit_every=12)
    cols_cmp = [c for c in M.MODELS if c in a and c not in ("garch", "ewma")]
    assert np.allclose(
        a.loc[0, cols_cmp].to_numpy(dtype=float), b.loc[0, cols_cmp].to_numpy(dtype=float), equal_nan=True
    )


# ------------------------------------------------------------------ losses and tests
def test_qlike_is_zero_at_the_truth_and_minimised_there():
    rv = np.array([0.001, 0.004, 0.02])
    assert np.allclose(E.qlike(rv, rv), 0)
    for k in (0.5, 0.9, 1.1, 2.0):
        assert (E.qlike(rv, rv * k) > 0).all()
    # it punishes under-prediction of variance harder than over-prediction of the same size
    assert E.qlike(rv, rv / 2).mean() > E.qlike(rv, rv * 2).mean()


def test_level_forecast_uses_only_past_errors():
    rng = np.random.default_rng(3)
    fc = pd.DataFrame({"actual": rng.normal(size=80), "m": rng.normal(size=80)})
    a = E.level_forecast(fc, "m")
    fc2 = fc.copy()
    fc2.loc[60:, "actual"] += 5
    b = E.level_forecast(fc2, "m")
    assert np.allclose(
        a.iloc[:61], b.iloc[:61]
    )  # errors from month 60 on cannot touch forecasts up to month 60


def test_diebold_mariano_detects_a_better_forecaster():
    rng = np.random.default_rng(4)
    bench = rng.chisquare(1, size=400)
    assert E.dm_p(bench, bench * 0.5) < 0.01
    assert E.dm_p(bench * 0.5, bench) > 0.99
    same = E.dm_p(bench, bench + rng.normal(scale=1e-6, size=400))
    assert 0.2 < same < 0.8


def test_sharpe_test_and_alpha_on_synthetic_returns():
    rng = np.random.default_rng(5)
    mkt = pd.Series(rng.normal(0.005, 0.04, size=600))
    better = mkt * 0.5 + 0.01 + rng.normal(0, 0.002, size=600)
    assert E.sharpe_diff_p(better, mkt) < 0.01
    a, b, t = E.alpha_test(better, mkt)
    assert a == pytest.approx(12.0, abs=1.0) and b == pytest.approx(0.5, abs=0.05) and t > 5


# ------------------------------------------------------------------ the strategy
def _toy_forecasts(n=120):
    rng = np.random.default_rng(6)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    rv = pd.Series(rng.uniform(0.001, 0.01, size=n), index=dates)
    fc = pd.DataFrame(
        {"date": dates, "actual": np.log(rv.shift(-1).fillna(rv.mean())).to_numpy(), "m": np.log(rv.to_numpy()),
         "ex_next": rng.normal(0.006, 0.04, size=n), "rf_next": 0.002, "rv_next": rv.shift(-1).fillna(rv.mean()).to_numpy()}
    )  # fmt: skip
    return fc, rv


def test_weights_are_capped_and_use_only_information_up_to_t():
    fc, rv = _toy_forecasts()
    res = E.managed(fc, "m", rv)
    assert res["w"].between(0, E.CAP).all()
    # change the realised variance after the 60th month: the first 60 weights must not move
    rv2 = rv.copy()
    rv2.iloc[61:] *= 50
    res2 = E.managed(fc, "m", rv2)
    assert np.allclose(res["w"].iloc[:61], res2["w"].iloc[:61])


def test_weight_is_target_over_forecast_volatility_and_costs_reduce_returns():
    fc, rv = _toy_forecasts()
    base = E.managed(fc, "m", rv, cost=0.0, borrow=0.0)
    h = E.level_forecast(fc, "m").to_numpy()
    target = rv.expanding().mean().to_numpy()
    assert np.allclose(base["w"].to_numpy(), np.clip(np.sqrt(target / h), 0, E.CAP))
    assert np.allclose(base["ret"], base["gross"])
    costly = E.managed(fc, "m", rv, cost=0.005)
    assert costly["ret"].sum() < base["ret"].sum()
    unlevered = E.managed(fc, "m", rv, cap=1.0)
    assert unlevered["w"].max() <= 1.0 + 1e-12


def test_max_drawdown_of_a_known_path():
    idx = pd.date_range("2000-01-01", periods=4, freq="MS")
    ex = pd.Series([0.10, -0.20, 0.05, 0.10], index=idx)
    rf = pd.Series(0.0, index=idx)
    assert E.max_drawdown(ex, rf) == pytest.approx(-20.0)
