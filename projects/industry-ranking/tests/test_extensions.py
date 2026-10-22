"""Timing and accounting checks for the follow-up experiments (volatility-managing, momentum comparison, combinations)."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import ind_eval as E
import ind_extensions as X


@pytest.fixture(scope="module")
def fc():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2000-01-01", periods=40, freq="MS")
    inds = [f"i{k:02d}" for k in range(24)]
    idx = pd.MultiIndex.from_product([dates, inds], names=["date", "industry"]).to_frame(index=False)
    idx["ret_next"] = rng.normal(0.01, 0.05, len(idx))
    idx["y"] = idx["ret_next"] - idx.groupby("date")["ret_next"].transform("mean")
    idx["sig_a"] = idx["y"] * 0.3 + rng.normal(size=len(idx))
    idx["sig_b"] = rng.normal(size=len(idx))
    return idx


def test_book_is_dollar_neutral_and_earns_the_next_month_return(fc):
    gross, W = X.book(fc, "sig_a", n_leg=5)
    assert np.allclose(W.sum(axis=1), 0) and np.allclose(W.abs().sum(axis=1), 2)
    d = fc["date"].iloc[0]
    g = fc[fc["date"] == d].sort_values("sig_a")
    expected = g["ret_next"].iloc[-5:].mean() - g["ret_next"].iloc[:5].mean()
    assert gross[d] == pytest.approx(expected)


def test_unmanaged_net_matches_the_evaluation_module(fc):
    gross, W = X.book(fc, "sig_a", n_leg=5)
    ref = E.long_short(fc, "sig_a", n_leg=5, cost=0.002)
    assert X.net_unmanaged(gross, W, 0.002).to_numpy() == pytest.approx(ref["net"].to_numpy())
    assert X.turnover(W).to_numpy() == pytest.approx(ref["turnover"].to_numpy())


def test_scale_for_month_t_ignores_returns_from_month_t_plus_1_on(fc):
    gross, W = X.book(fc, "sig_a", n_leg=5)
    _, base, _ = X.vol_managed(gross, W)
    altered = gross.copy()
    altered.iloc[20:] = altered.iloc[20:] * 10 + 0.5  # change what is earned from month 21 on
    _, other, _ = X.vol_managed(altered, W)
    assert other.iloc[:21].to_numpy() == pytest.approx(
        base.iloc[:21].to_numpy()
    )  # row 20 still sees rows <= 19
    assert not np.allclose(other.iloc[21:].to_numpy(), base.iloc[21:].to_numpy())


def test_scale_hits_the_volatility_target_and_respects_the_cap():
    r = pd.Series([0.02, -0.02] * 20, index=pd.date_range("2000-01-01", periods=40, freq="MS"))
    sig = X.trailing_vol(r, 6)
    assert sig.iloc[:6].isna().all() and sig.iloc[6:].notna().all()  # needs six realised months
    W = pd.DataFrame({"a": 0.5, "b": -0.5}, index=r.index)
    _, s, _ = X.vol_managed(r, W, win=6, target=0.10, cap=2.0)
    assert s.iloc[:6].eq(1.0).all()  # no history yet: unscaled
    assert s.iloc[10] == pytest.approx(min(2.0, 0.10 / sig.iloc[10]))
    assert (s <= 2.0 + 1e-12).all()
    _, s_hi, _ = X.vol_managed(r, W, win=6, target=0.50, cap=1.5)
    assert s_hi.iloc[10] == pytest.approx(1.5)


def test_with_the_cap_binding_at_one_managing_changes_nothing(fc):
    gross, W = X.book(fc, "sig_a", n_leg=5)
    net, s, turn = X.vol_managed(gross, W, target=10.0, cap=1.0)
    assert s.eq(1.0).all()
    assert net.to_numpy() == pytest.approx(X.net_unmanaged(gross, W).to_numpy())
    assert turn.to_numpy() == pytest.approx(X.turnover(W).to_numpy())


def test_cost_is_charged_on_the_scaled_exposure(fc):
    gross, W = X.book(fc, "sig_a", n_leg=5)
    net, s, turn = X.vol_managed(gross, W, cost=0.01)
    pos = W.mul(s, axis=0)
    manual = pos.diff().abs().sum(axis=1)
    manual.iloc[0] = pos.iloc[0].abs().sum()
    assert turn.to_numpy() == pytest.approx(manual.to_numpy())
    assert net.to_numpy() == pytest.approx((s * gross - 0.01 * manual).to_numpy())


def test_zscore_sum_is_centred_each_month_and_keeps_a_single_ranking(fc):
    z = X.zscore_sum(fc, ["sig_a", "sig_b"])
    assert z.groupby(fc["date"]).mean().abs().max() < 1e-12
    one = X.zscore_sum(fc, ["sig_a"])
    for _, g in fc.assign(z=one).groupby("date"):
        assert g["sig_a"].rank().to_numpy() == pytest.approx(g["z"].rank().to_numpy())


def test_bear_flag_uses_only_the_past_and_needs_a_full_window():
    idx = pd.date_range("2000-01-01", periods=40, freq="MS")
    down = pd.Series(-0.02, index=idx)
    flag = X.bear_flag(down, 24)
    assert not flag.iloc[:23].any() and flag.iloc[23:].all()  # False until 24 months exist
    up = pd.Series(0.01, index=idx)
    assert not X.bear_flag(up, 24).any()
    mixed = pd.Series(np.r_[np.full(30, 0.01), np.full(10, -0.20)], index=idx)
    changed = mixed.copy()
    changed.iloc[35:] = 0.5  # rewrite the future of month 34
    assert X.bear_flag(mixed, 24).iloc[:35].tolist() == X.bear_flag(changed, 24).iloc[:35].tolist()
