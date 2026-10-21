"""Data timing and evaluation checks for the credit-spread study."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import credit_data as C
import credit_eval as E


@pytest.fixture(scope="module")
def baa3():
    return C.build_dataset("BAA_SPREAD", 3)


def test_target_is_future_change(baa3):
    s = C.load_spreads()["BAA_SPREAD"]
    t = pd.Timestamp("2008-09-01")
    assert baa3["y"].loc[t] == pytest.approx(s.loc["2008-12-01"] - s.loc[t])
    assert baa3["y"].iloc[-3:].isna().all()  # outcomes not yet known at the end of the sample


def test_macro_data_lagged_market_data_not():
    raw = pd.read_csv(C.RAW / "fred_md_2026-09.csv").iloc[1:]
    raw.index = pd.to_datetime(raw.pop("sasdate"), format="%m/%d/%Y")
    md = C.load_fred_md()
    t = pd.Timestamp("2015-06-01")
    # UNRATE (code 2): the June row holds May's change
    may = float(raw.loc["2015-05-01", "UNRATE"]) - float(raw.loc["2015-04-01", "UNRATE"])
    assert md.loc[t, "md_UNRATE"] == pytest.approx(may)
    # GS10 is a market rate (code 2): the June row holds June's change
    june = float(raw.loc["2015-06-01", "GS10"]) - float(raw.loc["2015-05-01", "GS10"])
    assert md.loc[t, "md_GS10"] == pytest.approx(june)


def test_quarterly_bank_data_lag_a_quarter():
    sig = C.load_credit_signals()
    raw = C._monthly(C.RAW / "DRTSCILM.csv")
    assert sig.loc["2020-07-01", "LOAN_STANDARDS"] == pytest.approx(raw.loc["2020-04-01"])


def test_end_of_month_spread_matches_fred():
    s = C.load_spreads()
    assert s.loc["2020-03-01", "BAA_EOM"] == pytest.approx(3.93)
    assert s["BAA_EOM"].diff().autocorr() < s["BAA_SPREAD"].diff().autocorr()  # averaging adds persistence


def test_r2_and_tests():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    y = 0.4 * x + rng.normal(size=4000)
    zero = np.zeros_like(y)
    assert E.r2_os(y, 0.4 * x) > 10
    assert E.clark_west(y, 0.4 * x, zero) < 0.01
    assert E.diebold_mariano(y, 0.4 * x, zero) < 0.01
    assert E.diebold_mariano(y, zero, 0.4 * x) > 0.99


def test_timing_rule_with_random_walk_is_buy_and_hold():
    fc = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=4, freq="MS"),
            "level": [2.0] * 4,
            "actual": [0.1, -0.2, 0.0, 0.3],
            "rw": 0.0,
        }
    )
    r = E.credit_timing(fc, "rw")
    assert np.allclose(r.to_numpy(), 2 / 12 - 8 * fc["actual"].to_numpy())
