"""Timing, no-look-ahead and portfolio checks for the industry-ranking study."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import ind_data as D
import ind_eval as E
import ind_models as M


@pytest.fixture(scope="module")
def raw():
    ret, bm = D.load_industries()
    return ret, bm, D.load_market()


@pytest.fixture(scope="module")
def panel():
    return D.build_panel()["panel"]


# ------------------------------------------------------------------ data and characteristics
def test_cross_rank_is_centred_and_keeps_the_order():
    x = pd.DataFrame([[1.0, 5.0, 3.0, np.nan], [2.0, 2.0, 9.0, 1.0]], columns=list("abcd"))
    r = D.cross_rank(x)
    assert r.loc[0, ["a", "c", "b"]].tolist() == pytest.approx([-0.5, 0.0, 0.5])
    assert np.isnan(r.loc[0, "d"])
    assert r.loc[1, "a"] == pytest.approx(r.loc[1, "b"]) and r.loc[1, "d"] == pytest.approx(
        -0.5
    )  # ties share a rank


def test_characteristics_describe_the_past_only(raw):
    ret, bm, fac = raw
    t = pd.Timestamp("2005-06-01")
    full = D.characteristics(ret, bm, fac["mkt"])
    cut = D.characteristics(ret.loc[:t], bm, fac["mkt"].loc[:t])
    for k in ("mom_1", "mom_3", "mom_12", "mom_12_1", "rev_36", "vol_12", "dvol_12", "beta_36", "mkt_rel_12"):
        assert np.allclose(full[k].loc[t].to_numpy(), cut[k].loc[t].to_numpy(), equal_nan=True), k


def test_momentum_definitions(raw):
    ret, bm, fac = raw
    t = pd.Timestamp("2005-06-01")
    c = D.characteristics(ret, bm, fac["mkt"])
    r = ret["Oil"]
    assert c["mom_1"].loc[t, "Oil"] == pytest.approx(r.loc[t])
    assert c["mom_12"].loc[t, "Oil"] == pytest.approx((1 + r.loc["2004-07-01":t]).prod() - 1)
    skip = (1 + r.loc["2004-07-01":"2005-05-01"]).prod() - 1  # the latest month is skipped
    assert c["mom_12_1"].loc[t, "Oil"] == pytest.approx(skip)


def test_book_to_market_is_used_with_a_lag(raw):
    ret, bm, fac = raw
    c = D.characteristics(ret, bm, fac["mkt"])["bm"]
    assert c.loc["2010-06-01", "Oil"] == pytest.approx(
        np.log(bm.loc[2008, "Oil"])
    )  # year 2008 enters in July 2009
    assert c.loc["2010-07-01", "Oil"] == pytest.approx(np.log(bm.loc[2009, "Oil"]))


def test_target_is_next_months_return_relative_to_the_average(raw, panel):
    ret = raw[0]
    g = panel[panel["date"] == "2010-03-01"]
    assert g["ret_next"].to_numpy() == pytest.approx(ret.loc["2010-04-01", g["industry"]].to_numpy())
    assert g["y"].sum() == pytest.approx(0, abs=1e-10)
    assert g["y"].iloc[0] == pytest.approx(g["ret_next"].iloc[0] - g["ret_next"].mean())


def test_ranked_characteristics_stay_in_range(panel):
    assert panel[D.CHARS].min().min() >= -0.5 - 1e-12 and panel[D.CHARS].max().max() <= 0.5 + 1e-12


# ------------------------------------------------------------------ models
def test_state_variables_without_history_are_switched_off():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(6000, len(D.STATES))), columns=D.STATES)
    df["term"] = np.nan
    design = M.Design().fit(df)
    assert not design.on[D.STATES.index("term")] and design.on[D.STATES.index("mkt_vol")]
    later = df.copy()
    later["term"] = 3.0  # a variable that appears only after the training window must not enter
    assert (design.states(later)[:, D.STATES.index("term")] == 0).all()


def _toy_panel(n_ind=12, n_months=130, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1990-01-01", periods=n_months, freq="MS")
    names = [f"i{k}" for k in range(n_ind)]
    df = pd.MultiIndex.from_product([dates, names], names=["date", "industry"]).to_frame(index=False)
    for c in D.CHARS:
        df[c] = rng.uniform(-0.5, 0.5, size=len(df))
    for s in D.STATES:
        df[s] = np.repeat(rng.normal(size=n_months), n_ind)
    df["ret_next"] = 0.01 + 0.02 * df["mom_12_1"] + rng.normal(scale=0.05, size=len(df))
    df["y"] = df["ret_next"] - df.groupby("date")["ret_next"].transform("mean")
    return df


def test_forecasts_ignore_outcomes_and_inputs_from_the_future(monkeypatch):
    monkeypatch.setattr(M.Design, "MIN_STATE_OBS", 5)

    def fast_nonlinear(Xb, y, seed=0):  # stands in for the slow tree and network fits
        return {
            "rf": lambda Z: np.full(len(Z), y.mean()),
            "gbm": lambda Z: np.full(len(Z), np.median(y)),
            "nn": lambda Z: np.full(len(Z), y.std()),
            "_rf_importance": np.ones(Xb.shape[1]),
        }

    monkeypatch.setattr(M, "fit_nonlinear", fast_nonlinear)
    panel = _toy_panel()
    t = pd.Timestamp(sorted(panel["date"].unique())[110])
    a, _ = M.recursive_forecasts(panel, first_test=str(t.date()))
    a = a[a["date"] == t].reset_index(drop=True)

    p2 = panel.copy()
    p2.loc[p2["date"] >= t, ["y", "ret_next"]] += 5.0  # outcomes that are not yet known at t
    p2.loc[p2["date"] > t, D.CHARS] = 0.49  # characteristics of later months
    b, _ = M.recursive_forecasts(p2, first_test=str(t.date()))
    b = b[b["date"] == t].reset_index(drop=True)

    for m in ("ols", "ridge", "lasso", "enet", "rf", "gbm", "nn", "mean_ml"):
        assert np.allclose(a[m].to_numpy(), b[m].to_numpy()), m
    # single signals: the stated sign times today's characteristic (the function sorts industries as text)
    now = panel[panel["date"] == t].sort_values("industry").reset_index(drop=True)
    assert a["mom"].to_numpy() == pytest.approx(now["mom_12_1"].to_numpy())
    assert a["rev"].to_numpy() == pytest.approx(-now["mom_1"].to_numpy())


# ------------------------------------------------------------------ evaluation
def _scores(n_ind=49, n_months=60, skill=0.5, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    names = [f"i{k}" for k in range(n_ind)]
    df = pd.MultiIndex.from_product([dates, names], names=["date", "industry"]).to_frame(index=False)
    df["good"] = rng.normal(size=len(df))
    df["noise"] = rng.normal(size=len(df))
    df["y"] = skill * df["good"] + rng.normal(size=len(df))
    df["ret_next"] = 0.01 * df["y"]
    df["static"] = df["industry"].str[1:].astype(float)  # the same ranking every month
    return df


def test_information_coefficient_separates_skill_from_noise():
    df = _scores()
    good, noise = E.monthly_ic(df, "good"), E.monthly_ic(df, "noise")
    assert good.mean() > 0.3 and abs(noise.mean()) < 0.1
    assert E.nw_t(good) > 10 and abs(E.nw_t(noise)) < 3


def test_long_short_buys_the_top_and_shorts_the_bottom():
    df = _scores(n_months=6)
    ls = E.long_short(df, "good", cost=0.0)
    first = df[df["date"] == df["date"].min()].sort_values("good")
    expect = first["ret_next"].iloc[-10:].mean() - first["ret_next"].iloc[:10].mean()
    assert ls["gross"].iloc[0] == pytest.approx(expect)
    assert ls["turnover"].iloc[0] == pytest.approx(2.0)  # building a book that is 100% long and 100% short
    assert (ls["gross"] > 0).all()


def test_costs_are_charged_on_turnover():
    df = _scores(n_months=24)
    free, paid = E.long_short(df, "good", cost=0.0), E.long_short(df, "good", cost=0.002)
    assert np.allclose(free["gross"], paid["gross"])
    assert np.allclose(free["gross"] - paid["net"], 0.002 * paid["turnover"])
    static = E.long_short(df, "static", cost=0.002)  # nothing to trade after the first month
    assert static["turnover"].iloc[1:].abs().max() == 0 and static["turnover"].iloc[0] == pytest.approx(2.0)


def test_factor_alpha_recovers_a_known_alpha():
    rng = np.random.default_rng(2)
    idx = pd.date_range("1990-01-01", periods=500, freq="MS")
    fac = pd.DataFrame(rng.normal(0, 0.04, size=(500, 3)), index=idx, columns=["mktrf", "smb", "hml"])
    r = 0.004 + 0.5 * fac["mktrf"] + rng.normal(0, 0.005, size=500)
    alpha, t = E.factor_alpha(r, fac)
    assert alpha == pytest.approx(4.8, abs=0.8) and t > 5


def test_sharpe_difference_test():
    rng = np.random.default_rng(3)
    r0 = pd.Series(rng.normal(0.0, 0.03, 600))
    r1 = pd.Series(rng.normal(0.02, 0.03, 600))
    assert E.sharpe_diff_p(r1, r0) < 0.01 and E.sharpe_diff_p(r0, r1) > 0.99


def test_r2_os_is_undefined_for_rank_only_signals():
    df = _scores(n_months=6)
    df["ridge"] = 0.5 * df["y"]
    assert np.isnan(E.r2_os(df, "mom")) and np.isnan(E.r2_os(df, "mean_ml"))
    assert E.r2_os(df, "ridge") == pytest.approx(75.0)  # 1 - 0.5² of the squared outcome is left unexplained


def test_max_drawdown_of_a_known_path():
    assert E.max_drawdown(pd.Series([0.10, -0.20, 0.05, 0.10])) == pytest.approx(-20.0)
