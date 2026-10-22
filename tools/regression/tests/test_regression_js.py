"""Checks the regression tool's core (regression.js) against statsmodels, running it in Node.

Needs Node and the `jstat` package (npm install jstat@1.9.6; set NODE_PATH if it is installed elsewhere).
"""

import json
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent


def _node_has_jstat() -> bool:
    if shutil.which("node") is None:
        return False
    r = subprocess.run(["node", "-e", "require('jstat')"], capture_output=True, text=True)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(
    not _node_has_jstat(), reason="Node.js with the jstat package is not available"
)


def run(job: dict) -> dict:
    r = subprocess.run(
        ["node", str(HERE / "run_core.js")], input=json.dumps(job), capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def fit(model, y, X, names, **options):
    res = run({"task": "fit", "model": model, "y": list(map(float, y)), "X": np.asarray(X).tolist(),
               "names": names, "options": options})  # fmt: skip
    assert res["ok"], res.get("error")
    return res["out"]


def coef_table(out, key):
    return np.array([[c[key] for c in out["coefficients"]]])[0]


def data(n=400, seed=0):
    rng = np.random.RandomState(seed)
    x1, x2 = rng.normal(size=n), rng.normal(2, 3, size=n)
    x3 = 0.5 * x1 + rng.normal(size=n)
    eps = rng.normal(size=n) * (1 + 0.5 * np.abs(x1))  # heteroskedastic
    y = 1.0 + 2.0 * x1 - 0.7 * x2 + 0.3 * x3 + eps
    return x1, x2, x3, y


def sm_cols(res, k):
    """statsmodels puts the constant first; move it last, as the tool does."""
    order = [*range(1, k + 1), 0]
    return {
        "coef": np.asarray(res.params)[order],
        "se": np.asarray(res.bse)[order],
        "stat": np.asarray(res.tvalues)[order],
        "p": np.asarray(res.pvalues)[order],
        "lo": np.asarray(res.conf_int())[order, 0],
        "hi": np.asarray(res.conf_int())[order, 1],
    }


# ---------------------------------------------------------------------------------------------- OLS
@pytest.mark.parametrize("robust", [False, True])
def test_ols_matches_statsmodels(robust):
    x1, x2, x3, y = data()
    X = np.column_stack([x1, x2, x3])
    out = fit("ols", y, X, ["x1", "x2", "x3"], robust=robust)
    ref = (
        sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC1", use_t=True)
        if robust
        else sm.OLS(y, sm.add_constant(X)).fit()
    )
    for key, values in sm_cols(ref, 3).items():
        np.testing.assert_allclose(coef_table(out, key), values, rtol=1e-7, atol=1e-9, err_msg=key)
    assert [c["name"] for c in out["coefficients"]] == ["x1", "x2", "x3", "_cons"]
    np.testing.assert_allclose(out["fStat"], ref.fvalue, rtol=1e-7)
    np.testing.assert_allclose(out["fP"], ref.f_pvalue, rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(out["r2"], ref.rsquared, rtol=1e-9)
    np.testing.assert_allclose(out["adjR2"], ref.rsquared_adj, rtol=1e-9)
    np.testing.assert_allclose(out["rootMSE"], np.sqrt(ref.mse_resid), rtol=1e-9)
    assert out["n"] == len(y) and out["dfResid"] == len(y) - 4
    np.testing.assert_allclose(out["residuals"], ref.resid, atol=1e-8)


def test_the_intercept_is_labelled_cons_and_the_first_variable_is_not_the_intercept():
    """The old tool printed the intercept next to the first variable's name."""
    rng = np.random.RandomState(1)
    x = rng.normal(size=300)
    y = 10 + 3 * x + rng.normal(scale=0.1, size=300)
    out = fit("ols", y, x.reshape(-1, 1), ["x"])
    assert out["coefficients"][0]["name"] == "x" and abs(out["coefficients"][0]["coef"] - 3) < 0.05
    assert out["coefficients"][1]["name"] == "_cons" and abs(out["coefficients"][1]["coef"] - 10) < 0.05


def test_badly_scaled_regressors_are_handled():
    """statsmodels itself reports a rank-deficient design here, so the reference works on standardised columns."""
    rng = np.random.RandomState(2)
    n = 300
    year = rng.randint(1990, 2025, size=n).astype(float)  # large mean, small spread
    gdp = rng.normal(2e13, 3e12, size=n)  # huge scale
    y = 5 + 0.2 * (year - 2000) + 3e-13 * gdp + rng.normal(size=n)
    X = np.column_stack([year, gdp])
    out = fit("ols", y, X, ["year", "gdp"])
    m, s = X.mean(axis=0), X.std(axis=0)
    ref = sm.OLS(y, sm.add_constant((X - m) / s)).fit()
    slopes = ref.params[1:] / s
    np.testing.assert_allclose(coef_table(out, "coef")[:2], slopes, rtol=1e-6)
    np.testing.assert_allclose(coef_table(out, "coef")[2], ref.params[0] - (slopes * m).sum(), rtol=1e-6)
    np.testing.assert_allclose(coef_table(out, "se")[:2], ref.bse[1:] / s, rtol=1e-6)
    assert abs(coef_table(out, "coef")[0] - 0.2) < 0.02 and abs(coef_table(out, "coef")[1] - 3e-13) < 3e-14


def test_collinear_and_constant_regressors_are_named():
    x1, x2, _, y = data()
    res = run({"task": "fit", "model": "ols", "y": y.tolist(), "X": np.column_stack([x1, x2, x1 + 2 * x2]).tolist(),
               "names": ["a", "b", "c"], "options": {}})  # fmt: skip
    assert not res["ok"] and '"c"' in res["error"] and "multicollinearity" in res["error"]
    res = run({"task": "fit", "model": "ols", "y": y.tolist(), "X": np.column_stack([x1, np.full(len(y), 7.0)]).tolist(),
               "names": ["a", "k"], "options": {}})  # fmt: skip
    assert not res["ok"] and '"k" is constant' in res["error"]


def test_constant_dependent_variable_is_rejected():
    x1, *_ = data()
    res = run(
        {
            "task": "fit",
            "model": "ols",
            "y": [1.0] * len(x1),
            "X": x1.reshape(-1, 1).tolist(),
            "names": ["x"],
            "options": {},
        }
    )
    assert not res["ok"] and "constant" in res["error"]


# ---------------------------------------------------------------------------------------------- binary models
def binary_data(n=500, seed=3):
    rng = np.random.RandomState(seed)
    x1, x2 = rng.normal(size=n), rng.normal(5, 2, size=n)
    p = 1 / (1 + np.exp(-(-1.0 + 1.2 * x1 - 0.3 * x2 + 1.0)))
    return np.column_stack([x1, x2]), (rng.uniform(size=n) < p).astype(float)


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_binary_models_match_statsmodels(link):
    X, y = binary_data()
    out = fit(link, y, X, ["x1", "x2"])
    model = sm.Logit if link == "logit" else sm.Probit
    ref = model(y, sm.add_constant(X)).fit(disp=0, tol=1e-12, maxiter=200)
    for key, values in sm_cols(ref, 2).items():
        np.testing.assert_allclose(coef_table(out, key), values, rtol=1e-5, atol=1e-8, err_msg=key)
    assert out["converged"] and not out["warnings"]
    np.testing.assert_allclose(out["logLik"], ref.llf, rtol=1e-10)
    np.testing.assert_allclose(out["logLikNull"], ref.llnull, rtol=1e-10)
    np.testing.assert_allclose(out["lrChi2"], ref.llr, rtol=1e-8)
    np.testing.assert_allclose(out["lrP"], ref.llr_pvalue, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(out["pseudoR2"], ref.prsquared, rtol=1e-8)
    np.testing.assert_allclose(out["aic"], ref.aic, rtol=1e-10)
    np.testing.assert_allclose(out["bic"], ref.bic, rtol=1e-10)
    np.testing.assert_allclose(out["yHat"], ref.predict(), atol=1e-8)
    assert abs(out["accuracy"] - np.mean((ref.predict() >= 0.5) == y)) < 1e-12


def test_logit_odds_ratios():
    X, y = binary_data()
    out = fit("logit", y, X, ["x1", "x2"])
    ref = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    np.testing.assert_allclose(
        [c["oddsRatio"] for c in out["coefficients"]][:2], np.exp(ref.params[1:]), rtol=1e-6
    )
    assert "oddsRatio" not in fit("probit", y, X, ["x1", "x2"])["coefficients"][0]


def test_probit_in_the_tails_does_not_underflow():
    """An observation far in the tail used to give log(0)."""
    rng = np.random.RandomState(4)
    x = rng.normal(size=400)
    y = (rng.uniform(size=400) < stats.norm.cdf(0.5 + 1.5 * x)).astype(float)
    x[0], y[0] = 9.0, 0.0  # a very surprising observation
    out = fit("probit", y, x.reshape(-1, 1), ["x"])
    ref = sm.Probit(y, sm.add_constant(x)).fit(disp=0, tol=1e-12, maxiter=200)
    np.testing.assert_allclose(coef_table(out, "coef"), [ref.params[1], ref.params[0]], rtol=1e-5)
    np.testing.assert_allclose(out["logLik"], ref.llf, rtol=1e-9)


def test_normal_tail_helpers():
    z = [-40.0, -12.0, -6.0, -3.5, -3.0, -2.9, 0.0, 2.0, 8.0]
    got = np.array(run({"task": "tails", "z": z})["out"])
    np.testing.assert_allclose(got[:, 0], stats.norm.logcdf(z), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got[:, 1], np.exp(stats.norm.logpdf(z) - stats.norm.logcdf(z)), rtol=1e-9)


def test_perfect_separation_is_flagged():
    rng = np.random.RandomState(5)
    x = rng.normal(size=100)
    out = fit("logit", (x > 0).astype(float), x.reshape(-1, 1), ["x"])
    assert not out["converged"]
    assert any("Perfect prediction" in w for w in out["warnings"])


def test_logit_needs_a_zero_one_outcome():
    X, y = binary_data()
    res = run(
        {
            "task": "fit",
            "model": "logit",
            "y": (y * 3).tolist(),
            "X": X.tolist(),
            "names": ["a", "b"],
            "options": {},
        }
    )
    assert not res["ok"] and "coded 0/1" in res["error"]
    res = run(
        {
            "task": "fit",
            "model": "probit",
            "y": [1.0] * len(y),
            "X": X.tolist(),
            "names": ["a", "b"],
            "options": {},
        }
    )
    assert not res["ok"] and "only one value" in res["error"]


def test_many_observations_are_fast():
    """The old tool built an n by n matrix and froze on a few thousand rows."""
    rng = np.random.RandomState(6)
    n = 50_000
    X = rng.normal(size=(n, 4))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + X @ [0.5, -0.4, 0.2, 0.0])))).astype(float)
    out = fit("logit", y, X, ["a", "b", "c", "d"])
    ref = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    np.testing.assert_allclose(
        coef_table(out, "coef"), np.r_[ref.params[1:], ref.params[0]], rtol=1e-5, atol=1e-8
    )
    assert out["seconds"] < 15


# ---------------------------------------------------------------------------------------------- reading data
def test_missing_values_are_dropped_not_read_as_zero():
    rows = [
        {"y": "1", "x": "2"},
        {"y": "", "x": "3"},
        {"y": "4", "x": ""},
        {"y": "5", "x": "NA"},
        {"y": "6", "x": "7"},
    ]
    res = run({"task": "design", "rows": rows, "yName": "y", "xNames": ["x"]})["out"]
    assert res["y"] == [1, 6] and res["X"] == [[2], [7]]
    assert res["nTotal"] == 5 and res["nDropped"] == 3 and res["missingBy"] == {"y": 1, "x": 2}


def test_text_columns_are_not_numeric():
    rows = [
        {"a": "1", "b": "AAPL", "c": "", "d": "TRUE", "e": "1e3", "f": ".5"},
        {"a": "2.5", "b": "MSFT", "c": "", "d": "false", "e": "-2", "f": "NA"},
    ]
    out = run({"task": "classify", "rows": rows, "headers": list("abcdef")})["out"]
    assert out["numeric"] == ["a", "d", "e", "f"] and out["text"] == ["b", "c"]  # c is empty, so unusable
    vals = run(
        {"task": "toNumber", "values": [None, "", " 3 ", "1,5", "abc", "TRUE", "#N/A", "-.5e1", "+2"]}
    )["out"]
    assert vals == [None, None, 3, "text", "text", 1, None, -5, 2]


def test_text_in_a_selected_column_is_an_error():
    rows = [{"y": "1", "x": "a"}, {"y": "2", "x": "3"}]
    res = run({"task": "design", "rows": rows, "yName": "y", "xNames": ["x"]})
    assert not res["ok"] and "contains text" in res["error"]


def test_complete_case_regression_equals_statsmodels_on_the_kept_rows():
    x1, x2, _, y = data(n=200)
    rows = [
        {"y": str(a), "p": str(b), "q": "" if i % 7 == 0 else str(c)}
        for i, (a, b, c) in enumerate(zip(y, x1, x2, strict=True))
    ]
    d = run({"task": "design", "rows": rows, "yName": "y", "xNames": ["p", "q"]})["out"]
    keep = np.array([i % 7 != 0 for i in range(200)])
    assert d["nUsed"] == keep.sum()
    out = fit("ols", d["y"], d["X"], ["p", "q"])
    ref = sm.OLS(y[keep], sm.add_constant(np.column_stack([x1, x2])[keep])).fit()
    np.testing.assert_allclose(coef_table(out, "coef"), np.r_[ref.params[1:], ref.params[0]], rtol=1e-8)
