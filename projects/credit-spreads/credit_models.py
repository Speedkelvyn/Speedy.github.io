"""credit_models.py: the model arsenal and the recursive out-of-sample loop.

All models, settings and the test design below were fixed before any out-of-sample result was
seen. Every model is reported, and the Model Confidence Set (Hansen, Lunde & Nason 2011) deals
with the fact that many models are compared at once.

Benchmarks
  rw          random walk: the spread does not change
  ar          the spread's own history (level and recent changes), OLS
Linear, high-dimensional
  ols         kitchen sink: every predictor, OLS (shows what overfitting costs)
  ridge       ridge regression
  lasso       lasso
  enet        elastic net
  pcr         own history + 5 principal components of all other predictors (Stock-Watson diffusion index)
  pls         partial least squares, 3 components
  combo_uni   average of one-predictor regressions (Rapach, Strauss & Zhou 2010)
Nonlinear
  rf          random forest
  gbm         gradient-boosted trees
  nn          neural network (two hidden layers), average of five seeds
Combinations
  mean_ml     average of every model except the two benchmarks
  median_ml   median of the same
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

OWN = ["own_level", "own_chg_1", "own_chg_2", "own_chg_3", "own_chg_12"]
LINEAR = ["ar", "ols", "ridge", "lasso", "enet", "pcr", "pls", "combo_uni"]
NONLINEAR = ["rf", "gbm", "nn"]
COMBOS = ["mean_ml", "median_ml"]
MODELS = ["rw", *LINEAR, *NONLINEAR, *COMBOS]
NAMES = {
    "rw": "Random walk",
    "ar": "Own history (AR)",
    "ols": "OLS, all predictors",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "pcr": "Principal components",
    "pls": "Partial least squares",
    "combo_uni": "Univariate combination",
    "rf": "Random forest",
    "gbm": "Gradient boosting",
    "nn": "Neural network",
    "mean_ml": "Mean of models",
    "median_ml": "Median of models",
}
MIN_MONTHS = 60  # a predictor joins once it has five years of history in the training window


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
    """Fit every linear model; returns name -> predict function."""
    other = [j for j in range(X.shape[1]) if j not in own]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ar = LinearRegression().fit(X[:, own], y)
        ols = LinearRegression().fit(X, y)
        ridge = Ridge(alpha=hp["ridge"]).fit(X, y)
        lasso = Lasso(alpha=hp["lasso"], max_iter=20000).fit(X, y)
        enet = ElasticNet(alpha=hp["enet"][0], l1_ratio=hp["enet"][1], max_iter=20000).fit(X, y)
        pca = PCA(n_components=5).fit(X[:, other])
        pcr_x = lambda Z: np.column_stack([Z[:, own], pca.transform(Z[:, other])])
        pcr = LinearRegression().fit(pcr_x(X), y)
        pls = PLSRegression(n_components=3, scale=False).fit(X, y)
        # univariate regressions in closed form: slope_j = cov(x_j, y) / var(x_j) on standardised x
        yc = y - y.mean()
        slopes = (X - X.mean(0)).T @ yc / np.maximum(((X - X.mean(0)) ** 2).sum(0), 1e-12)
        icpt = y.mean() - slopes * X.mean(0)
    return {
        "ar": lambda Z: ar.predict(Z[:, own]),
        "ols": ols.predict,
        "ridge": ridge.predict,
        "lasso": lasso.predict,
        "enet": enet.predict,
        "pcr": lambda Z: pcr.predict(pcr_x(Z)),
        "pls": lambda Z: pls.predict(Z).ravel(),
        "combo_uni": lambda Z: (icpt + Z * slopes).mean(axis=1),
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        nets = [
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                alpha=1e-2,
                learning_rate_init=1e-3,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=seed + s,
            ).fit(X, y)
            for s in range(5)
        ]
    return {
        "rf": rf.predict,
        "gbm": gbm.predict,
        "nn": lambda Z: np.mean([n.predict(Z) for n in nets], axis=0),
        "_rf_importance": rf.feature_importances_,
    }


def recursive_forecasts(
    X: pd.DataFrame, y: pd.Series, level: pd.Series, horizon: int, first_test: str = "1995-01-01",
    refit_every: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """Direct `horizon`-month forecasts of the spread change, made at the end of every month.

    At forecast date t a model may only learn from rows whose outcome is known by t, i.e. rows
    dated t - horizon or earlier. Data preparation is fitted on those rows only. Linear models are
    re-estimated every month; their penalties, and the slower nonlinear models, every
    `refit_every` months (each January with the default).

    Returns the forecasts and a record of which predictors the lasso selected and how important
    each predictor was to the random forest at every refit.
    """
    dates = X.index
    test_dates = [t for t in dates if t >= pd.Timestamp(first_test) and pd.notna(y.loc[t])]
    rows, imp_rows = [], []
    hp = nl = nl_prep = None
    for k, t in enumerate(test_dates):
        train = (dates <= t - pd.DateOffset(months=horizon)) & y.notna().to_numpy()
        prep = Prep().fit(X.loc[train])
        Xtr, ytr, xt = prep.transform(X.loc[train]), y.loc[train].to_numpy(), prep.transform(X.loc[[t]])
        own = [prep.cols.index(c) for c in OWN]
        refit = k % refit_every == 0
        if refit:
            hp = tune(Xtr, ytr)
            nl_prep = prep
            nl = fit_nonlinear(Xtr, ytr)
        lin = fit_linear(Xtr, ytr, own, hp)
        f = {"rw": 0.0}
        for name in LINEAR:
            f[name] = float(lin[name](xt)[0])
        xt_nl = nl_prep.transform(X.loc[[t]])
        for name in NONLINEAR:
            f[name] = float(nl[name](xt_nl)[0])
        members = [f[n] for n in MODELS if n not in ("rw", "ar") and n not in COMBOS]
        f["mean_ml"], f["median_ml"] = float(np.mean(members)), float(np.median(members))
        rows.append(
            {"date": t, "horizon": horizon, "actual": float(y.loc[t]), "level": float(level.loc[t]), **f}
        )
        if refit:
            imp_rows.append(
                pd.DataFrame(
                    {
                        "date": t,
                        "predictor": nl_prep.cols,
                        "rf_importance": nl["_rf_importance"],
                        "lasso_selected": np.isin(
                            np.arange(len(prep.cols)), np.flatnonzero(lin["_lasso_coef"])
                        )
                        if prep.cols == nl_prep.cols
                        else np.nan,
                    }
                )
            )
    return pd.DataFrame(rows), pd.concat(imp_rows, ignore_index=True)
