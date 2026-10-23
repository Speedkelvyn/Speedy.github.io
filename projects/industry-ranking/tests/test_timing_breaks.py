"""Timing and calibration checks for the random-feature timing test and the sup-Wald break test."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import ind_breaks as B
import ind_extensions as X
import ind_timing as T


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(3)
    idx = pd.date_range("1990-01-01", periods=220, freq="MS")
    f = pd.DataFrame(rng.normal(size=(220, 4)), index=idx, columns=list("abcd"))
    y = pd.Series(0.01 + 0.02 * rng.normal(size=220), index=idx)
    return f, y


# ------------------------------------------------------------------ features and forecasts
def test_features_of_month_t_use_returns_earned_up_to_month_t():
    idx = pd.date_range("2000-01-01", periods=40, freq="MS")
    g = pd.Series(np.random.default_rng(1).normal(0, 0.03, 40), index=idx)
    states = pd.DataFrame({"mkt_vol": -7.0, "mkt_ret_12": 0.1}, index=idx)
    base = T.timing_features(g, states)
    altered = g.copy()
    altered.iloc[25:] = altered.iloc[25:] * 5 + 0.1  # g[25] is earned in month 26, after the date-25 features
    other = T.timing_features(altered, states)
    cut = idx[25]
    assert base.loc[:cut].to_numpy() == pytest.approx(other.loc[:cut].to_numpy())
    assert not np.allclose(base.loc[idx[26] :].to_numpy(), other.loc[idx[26] :].to_numpy())


def test_rff_forecast_ignores_everything_after_its_date(toy):
    f, y = toy
    k = 150
    base = T.rff_forecasts(f, y, f.index[120], gamma=0.5, ridges=(1.0,), n_features=200, refit=1)
    f2, y2 = f.copy(), y.copy()
    y2.iloc[k:] = 99.0  # returns earned from month k+1 on
    f2.iloc[k + 1 :] = 99.0  # features of later dates
    other = T.rff_forecasts(f2, y2, f.index[120], gamma=0.5, ridges=(1.0,), n_features=200, refit=1)
    assert other.loc[: f.index[k], 1.0].to_numpy() == pytest.approx(base.loc[: f.index[k], 1.0].to_numpy())
    assert not np.allclose(other.loc[f.index[k + 2] :, 1.0], base.loc[f.index[k + 2] :, 1.0])


def test_a_huge_ridge_penalty_returns_the_mean_of_past_returns(toy):
    f, y = toy
    fr = T.rff_forecasts(f, y, f.index[120], ridges=(1e12,), n_features=200, refit=1)
    expected = y.expanding().mean().shift(1).loc[fr.index]  # the mean of g[s], s < t
    assert fr[1e12].to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-6)


def test_random_features_approximate_the_gaussian_kernel():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 3))
    w = rng.normal(size=(3, 20000))
    z = T._features(x, w, 0.5)
    exact = np.exp(-0.25 * ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2) / 2)
    assert np.abs(z @ z.T - exact).max() < 0.03
    assert np.diag(z @ z.T) == pytest.approx(1.0)


def test_normalised_position_has_average_size_one_and_uses_only_the_past():
    raw = pd.Series([2.0, -4.0, 6.0, 1.0, -3.0])
    pi = T.normalise(raw)
    assert pi.iloc[0] == pytest.approx(1.0)
    assert pi.iloc[2] == pytest.approx(6.0 / np.mean([2.0, 4.0, 6.0]))
    changed = raw.copy()
    changed.iloc[4] = 100.0
    assert T.normalise(changed).iloc[:4].to_numpy() == pytest.approx(pi.iloc[:4].to_numpy())


def test_a_constant_position_of_one_is_the_unmanaged_book():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    fc = pd.DataFrame({"date": np.repeat(idx, 12), "industry": list("abcdefghijkl") * 30})
    fc["ret_next"] = rng.normal(0.01, 0.05, len(fc))
    fc["s"] = rng.normal(size=len(fc))
    g, W = X.book(fc, "s", n_leg=3)
    net, _ = T.timed_net(g, W, pd.Series(1.0, index=g.index))
    assert net.to_numpy() == pytest.approx(X.net_unmanaged(g, W).to_numpy())


def test_oos_r2_is_zero_for_the_expanding_mean_and_negative_for_noise(toy):
    _, y = toy
    hist = y.expanding().mean().shift(1).iloc[10:]
    assert T.r2_oos(hist, y) == pytest.approx(0.0, abs=1e-9)
    noise = hist + np.random.default_rng(2).normal(0, 0.05, len(hist))
    assert T.r2_oos(noise, y) < 0


# ------------------------------------------------------------------ break test
def test_simulated_critical_values_match_andrews_1993():
    d = B.null_draws()
    q = [d[int(p * len(d))] for p in (0.90, 0.95, 0.99)]
    assert q == pytest.approx([7.12, 8.68, 12.16], abs=0.4)  # Andrews (1993), 15% trimming, one parameter


def test_long_run_variance_of_white_noise_is_its_variance():
    x = np.random.default_rng(4).normal(0, 2.0, 5000)
    assert B.long_run_variance(x) == pytest.approx(4.0, rel=0.1)


def test_a_stable_mean_is_rarely_rejected():
    rng = np.random.default_rng(11)
    rej = np.mean([B.sup_wald(rng.normal(size=400))["p_value"] < 0.05 for _ in range(150)])
    assert rej < 0.12  # nominal 5%, a little over in finite samples


def test_a_clear_break_is_found_near_its_date():
    rng = np.random.default_rng(12)
    x = np.r_[rng.normal(0.0, 1.0, 250), rng.normal(1.0, 1.0, 250)]
    r = B.sup_wald(x)
    assert r["p_value"] < 0.01
    assert abs(r["break_after"] - 250) < 30
    assert r["mean_after"] - r["mean_before"] == pytest.approx(1.0, abs=0.25)


def test_missing_values_are_dropped_before_testing():
    rng = np.random.default_rng(13)
    x = rng.normal(size=300)
    y = np.insert(x, 100, np.nan)
    assert B.sup_wald(y)["sup_wald"] == pytest.approx(B.sup_wald(x)["sup_wald"])
