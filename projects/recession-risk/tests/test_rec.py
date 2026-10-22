"""Label timing, real-time training and scoring checks for the recession and stress-probability study."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import rec_data as D
import rec_eval as E
import rec_models as M


@pytest.fixture(scope="module")
def data():
    return D.build_dataset()


# ------------------------------------------------------------------ labels
def test_recession_label_looks_twelve_months_ahead(data):
    tg = data["targets"]
    assert tg.loc["2007-01-01", "RECESSION"] == 1  # the 2008 recession starts within the next 12 months
    assert tg.loc["2006-12-01", "RECESSION"] == 0  # one month too early
    assert tg.loc["2019-03-01", "RECESSION"] == 1  # February 2020 peak within 12 months
    assert tg.loc["2015-06-01", "RECESSION"] == 0


def test_months_inside_a_recession_have_no_recession_label(data):
    tg = data["targets"]
    assert np.isnan(tg.loc["2008-06-01", "RECESSION"]) and np.isnan(tg.loc["2020-04-01", "RECESSION"])


def test_months_without_a_full_window_are_unlabelled(data):
    tg = data["targets"]
    assert tg["RECESSION"].iloc[-11:].isna().all()
    assert tg["BEAR"].iloc[-11:].isna().all()


def test_bear_label_is_a_twenty_percent_fall_within_twelve_months(data):
    tg, m = data["targets"], data["market"]
    assert tg.loc["2008-05-01", "BEAR"] == 1
    assert tg.loc["2013-06-01", "BEAR"] == 0
    w = (1 + m).cumprod()
    t = pd.Timestamp("2000-12-01")
    assert tg.loc[t, "BEAR"] == float((w.loc[t:].iloc[1:13].min() / w.loc[t] - 1) <= -0.20)


def test_macro_data_lagged_market_data_not():
    raw = pd.read_csv(D.RAW / "fred_md_2026-09.csv").iloc[1:]
    raw.index = pd.to_datetime(raw.pop("sasdate"), format="%m/%d/%Y")
    md = D.load_fred_md()
    t = pd.Timestamp("2015-06-01")
    may = float(raw.loc["2015-05-01", "UNRATE"]) - float(raw.loc["2015-04-01", "UNRATE"])
    assert md.loc[t, "md_UNRATE"] == pytest.approx(may)
    june = float(raw.loc["2015-06-01", "GS10"]) - float(raw.loc["2015-05-01", "GS10"])
    assert md.loc[t, "md_GS10"] == pytest.approx(june)


def test_quarterly_bank_data_lag_a_quarter():
    sig = D.load_signals()
    raw = D._monthly(D.RAW / "DRTSCILM.csv")
    t = pd.Timestamp("2012-05-01")
    assert sig.loc[t, "LOAN_STANDARDS"] == pytest.approx(raw.resample("MS").ffill().loc["2012-02-01"])


# ------------------------------------------------------------------ models
def test_probit_returns_a_probability_and_handles_missing_inputs():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200)})
    y = (X["a"] + rng.normal(scale=0.8, size=200) > 0.5).astype(int).to_numpy()
    hi = pd.DataFrame({"a": [2.0]})
    lo = pd.DataFrame({"a": [-2.0]})
    assert 0 < M.probit_prob(X, y, lo) < M.probit_prob(X, y, hi) < 1
    assert np.isnan(M.probit_prob(X, y, pd.DataFrame({"a": [np.nan]})))


def test_probit_falls_back_under_perfect_separation():
    X = pd.DataFrame({"a": np.r_[np.linspace(-3, -1, 50), np.linspace(1, 3, 50)]})
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    p = M.probit_prob(X, y, pd.DataFrame({"a": [2.0]}))
    assert np.isfinite(p) and 0.5 < p <= 1


def test_wide_preparation_uses_training_rows_only():
    rng = np.random.default_rng(1)
    idx = pd.date_range("1980-01-01", periods=300, freq="MS")
    X = pd.DataFrame(rng.normal(size=(300, 3)), index=idx, columns=list("abc"))
    X.iloc[200:, 0] = 1e6
    w = M.Wide().fit(X.iloc[:150])
    assert w.hi["a"] < 10


def test_forecasts_ignore_labels_that_were_not_yet_known():
    rng = np.random.default_rng(2)
    n, lag = 260, 24
    idx = pd.date_range("1985-01-01", periods=n, freq="MS")
    cols = sorted({c for v in M.SETS.values() for c in v} | {"x1", "x2", "x3"})
    X = pd.DataFrame(rng.normal(size=(n, len(cols))), index=idx, columns=cols)
    y = pd.Series(
        ((X["TERM_10Y3M"].rolling(3, min_periods=1).mean() + rng.normal(scale=1.0, size=n)) < -0.6).astype(
            float
        ),
        index=idx,
    )
    t = idx[240]
    a, _ = M.recursive_probabilities(X, y, lag, first_test=str(t.date()), refit_every=12)
    y2 = y.copy()
    y2.loc[idx > t - pd.DateOffset(months=lag)] = (
        1.0 - y2.loc[idx > t - pd.DateOffset(months=lag)]
    )  # flip every label not yet known
    b, _ = M.recursive_probabilities(X, y2, lag, first_test=str(t.date()), refit_every=12)
    m = list(M.MODELS)
    assert np.allclose(a.loc[0, m].to_numpy(dtype=float), b.loc[0, m].to_numpy(dtype=float), equal_nan=True)


# ------------------------------------------------------------------ scoring
def test_log_score_and_brier_reward_calibrated_confidence():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0.0])
    good = np.array([0.7, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1])
    base = np.full(8, y.mean())
    bad = 1 - good
    assert E.log_score(y, good) > E.log_score(y, base) > E.log_score(y, bad)
    assert E.brier(y, good) < E.brier(y, base) < E.brier(y, bad)
    assert np.isfinite(E.log_score(y, np.array([1.0, 0, 0, 0, 1, 0, 0, 0])))  # clipped, never -inf


def test_block_bootstrap_keeps_runs_together():
    rng = np.random.default_rng(3)
    idx = E.block_bootstrap_indices(100, 24, rng)
    assert len(idx) == 100
    assert (np.diff(idx[:24]) == 1).all() and idx.max() < 100


def test_table_ranks_an_informative_model_above_noise():
    rng = np.random.default_rng(4)
    n = 300
    dates = pd.date_range("1995-01-01", periods=n, freq="MS")
    y = (np.sin(np.arange(n) / 20) > 0.7).astype(float)
    fc = pd.DataFrame({"date": dates, "actual": y, "base": y.mean(), "good": 0.1 + 0.6 * y + rng.normal(0, 0.05, n),
                       "noise": rng.uniform(size=n)})  # fmt: skip
    names = {"base": "Base", "good": "Good", "noise": "Noise"}
    t = E.table(fc, ["base", "good", "noise"], names, ref="noise", reps=200).set_index("key")
    assert t.loc["good", "AUC"] > 0.95 and 0.35 < t.loc["noise", "AUC"] < 0.65
    assert t.loc["good", "Log score"] > t.loc["base", "Log score"]


def test_lead_table_reads_the_probability_k_months_before_the_start():
    idx = pd.date_range("1999-01-01", periods=40, freq="MS")
    rec = pd.Series(0, index=idx)
    rec.loc["2001-03-01":"2001-12-01"] = 1
    fc = pd.DataFrame({"date": idx, "m": np.arange(40) / 40})
    lt = E.lead_table(fc, rec, "m", leads=(12, 3))
    assert lt.loc[0, "12 months before"] == pytest.approx(fc.set_index("date").loc["2000-03-01", "m"])
    assert lt.loc[0, "3 months before"] == pytest.approx(fc.set_index("date").loc["2000-12-01", "m"])


def test_alarm_stats_count_flags_catches_and_false_alarms():
    fc = pd.DataFrame({"actual": [1, 1, 0, 0, 0, 0, 1, 0], "m": [0.6, 0.2, 0.7, 0.1, 0.1, 0.1, 0.9, 0.1]})
    s = E.alarm_stats(fc, "m", 0.5)
    assert s["Months flagged (%)"] == pytest.approx(37.5)
    assert s["Pre-recession months caught (%)"] == pytest.approx(200 / 3)
    assert s["False alarms (% of flags)"] == pytest.approx(100 / 3)
