"""vol_models.py: the model line-up and the recursive out-of-sample loop for volatility forecasting.

The models, settings and the strategy rules in vol_eval.py were fixed before any out-of-sample result
was seen. Every model is reported, and the Model Confidence Set (Hansen, Lunde & Nason 2011) deals
with the fact that many models are compared at once.

Benchmarks (volatility-specific)
  naive       next month will look like last month
  ewma        RiskMetrics: exponentially weighted daily squared returns, decay 0.94
  garch       GJR-GARCH(1,1) with Student-t errors, re-estimated every January on all data to date
  har         HAR: last month, last 3 months and last 12 months of realised volatility (Corsi 2009)
  har_long    the same HAR regression estimated on every month back to 1927
Regressions on small, hand-picked sets
  har_x       HAR plus the last five days, downside share, last-day move and returns
  vix         the VIX alone (implied volatility)
  har_vix     HAR plus the VIX
  har_cred    HAR, the VIX and credit/funding conditions (Baa spread, excess bond premium, NFCI, term spread)
High-dimensional, all ~140 predictors
  ridge, lasso, enet, pcr (own history + 5 principal components), pls, rf, gbm, nn
Combinations
  mean_ml, median_ml   average and median of the eight high-dimensional models
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

HAR = ["lrv", "lrv_3", "lrv_12"]
SMALL = {
    "har": HAR,
    "har_x": [*HAR, "lrv_5d", "semi_neg", "abs_last", "ret_1m", "ret_12m"],
    "vix": ["lvix2"],
    "har_vix": [*HAR, "lvix2"],
    "har_cred": [*HAR, "lvix2", "BAA_SPR", "EBP", "NFCI", "TERM_10Y3M"],
}
HIGHDIM = ["ridge", "lasso", "enet", "pcr", "pls", "rf", "gbm", "nn"]
MODELS = ["naive", "ewma", "garch", "har", "har_long", *list(SMALL)[1:], *HIGHDIM, "mean_ml", "median_ml"]
NAMES = {
    "naive": "Last month",
    "ewma": "EWMA (RiskMetrics)",
    "garch": "GJR-GARCH-t",
    "har": "HAR",
    "har_long": "HAR, data from 1927",
    "har_x": "HAR + extra realised measures",
    "vix": "VIX alone",
    "har_vix": "HAR + VIX",
    "har_cred": "HAR + VIX + credit conditions",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "pcr": "Principal components",
    "pls": "Partial least squares",
    "rf": "Random forest",
    "gbm": "Gradient boosting",
    "nn": "Neural network",
    "mean_ml": "Mean of high-dimensional models",
    "median_ml": "Median of high-dimensional models",
}
MIN_MONTHS = 60  # a predictor joins once it has five years of history in the training window
MACRO_START = "1960-01-01"  # FRED-MD begins in 1959; models that use macro data train from 1960
MIN_ROWS = 36


class Prep:
    """Column selection, winsorising, median imputation and standardisation, fitted on training rows only."""

    def fit(self, X: pd.DataFrame) -> Prep:
        self.cols = [c for c in X.columns if X[c].notna().sum() >= MIN_MONTHS]
        Z = X[self.cols]
        self.lo, self.hi = Z.quantile(0.005), Z.quantile(0.995)
        Z = Z.clip(self.lo, self.hi, axis=1)
        self.med = Z.median()
        Z = Z.fillna(self.med)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1).fillna(1)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[self.cols].clip(self.lo, self.hi, axis=1).fillna(self.med)
        return ((Z - self.mu) / self.sd).to_numpy(dtype=float)


def tune(X: np.ndarray, y: np.ndarray) -> dict:
    """Penalties for ridge, lasso and elastic net by 5-fold time-series cross-validation."""
    cv = TimeSeriesSplit(n_splits=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        r = RidgeCV(alphas=np.logspace(-1, 5, 40), cv=cv).fit(X, y)
        la = LassoCV(cv=cv, alphas=40, max_iter=20000).fit(X, y)
        en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=cv, alphas=40, max_iter=20000).fit(X, y)
    return {"ridge": r.alpha_, "lasso": la.alpha_, "enet": (en.alpha_, en.l1_ratio_)}


def fit_linear(X: np.ndarray, y: np.ndarray, own: list[int], hp: dict) -> dict:
    other = [j for j in range(X.shape[1]) if j not in own]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ridge = Ridge(alpha=hp["ridge"]).fit(X, y)
        lasso = Lasso(alpha=hp["lasso"], max_iter=20000).fit(X, y)
        enet = ElasticNet(alpha=hp["enet"][0], l1_ratio=hp["enet"][1], max_iter=20000).fit(X, y)
        pca = PCA(n_components=5).fit(X[:, other])
        pcr_x = lambda Z: np.column_stack([Z[:, own], pca.transform(Z[:, other])])
        pcr = LinearRegression().fit(pcr_x(X), y)
        pls = PLSRegression(n_components=3, scale=False).fit(X, y)
    return {
        "ridge": ridge.predict,
        "lasso": lasso.predict,
        "enet": enet.predict,
        "pcr": lambda Z: pcr.predict(pcr_x(Z)),
        "pls": lambda Z: pls.predict(Z).ravel(),
        "_lasso_coef": lasso.coef_,
    }


def fit_nonlinear(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    rf = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=5, max_features=0.33, random_state=seed, n_jobs=1
    ).fit(X, y)
    gbm = HistGradientBoostingRegressor(
        learning_rate=0.03, max_iter=400, max_depth=3, min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=seed,
    ).fit(X, y)  # fmt: skip
    ybar = y.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        nets = [
            MLPRegressor(
                hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3, max_iter=2000,
                early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=seed + s,
            ).fit(X, y - ybar)
            for s in range(5)
        ]  # fmt: skip
    return {
        "rf": rf.predict,
        "gbm": gbm.predict,
        "nn": lambda Z: ybar + np.mean([n.predict(Z) for n in nets], axis=0),
        "_rf_importance": rf.feature_importances_,
    }


def small_forecast(
    X: pd.DataFrame, y: pd.Series, train: np.ndarray, cols: list[str], xt: pd.DataFrame
) -> float:
    """OLS on a few hand-picked predictors, using every training month where all of them exist."""
    ok = train & X[cols].notna().all(axis=1).to_numpy() & y.notna().to_numpy()
    if ok.sum() < MIN_ROWS or xt[cols].isna().any(axis=None):
        return float("nan")
    lr = LinearRegression().fit(X.loc[ok, cols], y[ok])
    return float(lr.predict(xt[cols])[0])


def recursive_forecasts(
    X: pd.DataFrame, y: pd.Series, first_test: str = "2000-01-01", refit_every: int = 12
) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """One-month-ahead forecasts of log realised variance, made at the end of every month.

    At forecast date t a model may only learn from rows whose outcome is known by t, i.e. rows dated
    t - 1 month or earlier. Data preparation is fitted on those rows only. Small regressions and the
    linear high-dimensional models are re-estimated every month; their penalties, and the slower
    nonlinear models, every `refit_every` months (each January with the default).
    Returns the forecasts and a record of which predictors the lasso kept and how much the random forest used each.
    """
    dates = X.index
    macro_ok = dates >= pd.Timestamp(MACRO_START)
    # the last month has no outcome yet: it is forecast live, with the same procedure
    test_dates = [
        t for t in dates if t >= pd.Timestamp(first_test) and (pd.notna(y.loc[t]) or t == dates[-1])
    ]
    rows, imp_rows = [], []
    hp = nl = nl_prep = None
    for k, t in enumerate(test_dates):
        known = (dates <= t - pd.DateOffset(months=1)) & y.notna().to_numpy()
        xt = X.loc[[t]]
        f = {"naive": float(xt["lrv"].iloc[0])}
        for name, cols in SMALL.items():
            f[name] = small_forecast(X, y, known & macro_ok, cols, xt)
        f["har_long"] = small_forecast(X, y, known, HAR, xt)
        train = known & macro_ok
        prep = Prep().fit(X.loc[train])
        Xtr, ytr, xtt = prep.transform(X.loc[train]), y.loc[train].to_numpy(), prep.transform(xt)
        own = [prep.cols.index(c) for c in HAR]
        refit = k % refit_every == 0
        if refit:
            hp = tune(Xtr, ytr)
            nl_prep = prep
            nl = fit_nonlinear(Xtr, ytr)
        lin = fit_linear(Xtr, ytr, own, hp)
        for name in ("ridge", "lasso", "enet", "pcr", "pls"):
            f[name] = float(lin[name](xtt)[0])
        xt_nl = nl_prep.transform(xt)
        for name in ("rf", "gbm", "nn"):
            f[name] = float(nl[name](xt_nl)[0])
        members = [f[n] for n in HIGHDIM]
        f["mean_ml"], f["median_ml"] = float(np.mean(members)), float(np.median(members))
        rows.append({"date": t, "actual": float(y.loc[t]), **f})
        if refit:
            kept = (
                np.isin(np.arange(len(prep.cols)), np.flatnonzero(lin["_lasso_coef"]))
                if prep.cols == nl_prep.cols
                else np.nan
            )
            imp_rows.append(
                pd.DataFrame(
                    {
                        "date": t,
                        "predictor": nl_prep.cols,
                        "rf_importance": nl["_rf_importance"],
                        "lasso_selected": kept,
                    }
                )
            )
    return pd.DataFrame(rows), pd.concat(imp_rows, ignore_index=True)
