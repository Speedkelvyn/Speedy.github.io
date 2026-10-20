"""Checks on the data timing and the evaluation statistics."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import fx_forecasting as F


@pytest.fixture(scope="module")
def panel():
    return F.build_panel()


def test_target_is_next_months_return(panel):
    fx = F.load_fx_monthly()
    d = panel.data
    row = d[(d.currency == "GBP") & (d.date == "2015-06-01")].iloc[0]
    expected = 100 * np.log(fx.loc["2015-07-01", "GBP"] / fx.loc["2015-06-01", "GBP"])
    assert row["target"] == pytest.approx(expected)
    assert row["ret_1m"] == pytest.approx(
        100 * np.log(fx.loc["2015-06-01", "GBP"] / fx.loc["2015-05-01", "GBP"])
    )


def test_all_rates_quoted_as_usd_per_foreign_unit():
    fx = F.load_fx_monthly()
    assert fx.loc["2020-01-01", "JPY"] < 0.02  # about 0.009 USD per yen, not 109 yen per USD
    assert 0.5 < fx.loc["2020-01-01", "EUR"] < 2


def test_fundamentals_carry_a_publication_lag(panel):
    raw = F.load_fundamentals()
    raw = raw[(raw.currency == "CAD")].set_index("date")["rate_diff"]
    row = panel.data[(panel.data.currency == "CAD") & (panel.data.date == "2018-05-01")].iloc[0]
    assert row["rate_diff"] == pytest.approx(raw.loc["2018-04-01"])


def test_no_missing_values_and_no_zero_fill(panel):
    assert not panel.data[panel.features].isna().any().any()
    assert (panel.data["infl_diff"].abs() < 15).all()  # recomputed inflation is in a sane range


def test_r2_os_and_clark_west():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=5000)
    y = 0.3 * signal + rng.normal(size=5000)
    assert F.r2_os(y, np.zeros_like(y)) == 0
    assert F.r2_os(y, 0.3 * signal) > 0
    assert F.clark_west(y, 0.3 * signal)["p"] < 0.01
    assert F.clark_west(y, rng.normal(scale=0.3, size=5000))["p"] > 0.05


def test_direction_test_detects_skill():
    rng = np.random.default_rng(1)
    s = rng.normal(size=2000)
    y = s + rng.normal(size=2000)
    assert F.direction_test(y, s)["p"] < 0.01
