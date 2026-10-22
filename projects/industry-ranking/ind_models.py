"""ind_models.py: models that rank industries, and the recursive out-of-sample loop.

Fixed before any out-of-sample result was seen. Every model predicts an industry's return next month relative to the
average industry; the forecasts are then used only through their cross-sectional ranking.

Single-signal benchmarks (no estimation)
  mom       industry momentum: 12-month return skipping the latest month
  mom6      6-month momentum (Moskowitz & Grinblatt 1999)
  rev       short-term reversal: minus last month's return
  value     book-to-market, and bm_rel: book-to-market relative to the industry's own 5-year history
Pooled regressions on the 12 characteristics, interacted with 5 market-state variables
  ols, ridge, lasso, enet
Nonlinear models on characteristics and state variables
  rf, gbm, nn
Combination
  mean_ml   average of the standardised forecasts of ridge, lasso, enet, rf, gbm and nn
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

import ind_data as D

SIGNALS = {
    "mom": ("mom_12_1", 1),
    "mom6": ("mom_6", 1),
    "rev": ("mom_1", -1),
    "value": ("bm", 1),
    "value_rel": ("bm_rel", 1),
}
ML = ["ols", "ridge", "lasso", "enet", "rf", "gbm", "nn"]
MODELS = [*SIGNALS, *ML, "mean_ml"]
NAMES = {
    "mom": "Momentum (12-1)",
    "mom6": "Momentum (6 months)",
    "rev": "Short-term reversal",
    "value": "Value (book-to-market)",
    "value_rel": "Value relative to own history",
    "ols": "OLS",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "rf": "Random forest",
    "gbm": "Gradient boosting",
    "nn": "Neural network",
    "mean_ml": "Average of the machine-learning models",
}


class Design:
    """Turn a panel slice into model inputs: characteristics, states and their interactions."""

    MIN_STATE_OBS = 120  # a state variable enters once it has 120 months of history (49 industries per month)

    def fit(self, df: pd.DataFrame) -> Design:
        # A state variable that is (almost) absent from the training window, e.g. the term spread before 1982, is switched off
        # (set to zero at every date) instead of being filled with a meaningless median.
        self.on = (df[D.STATES].notna().sum() >= self.MIN_STATE_OBS * 49).to_numpy()
        self.med = df[D.STATES].median().fillna(0.0)
        s = df[D.STATES].fillna(self.med)
        self.mu, self.sd = s.mean().fillna(0.0), s.std().replace(0, 1).fillna(1.0)
        return self

    def states(self, df: pd.DataFrame) -> np.ndarray:
        z = ((df[D.STATES].fillna(self.med) - self.mu) / self.sd).clip(-4, 4).to_numpy(dtype=float)
        return z * self.on

    def base(self, df: pd.DataFrame) -> np.ndarray:
        c = df[D.CHARS].fillna(0.0).to_numpy(dtype=float)
        return np.column_stack([c, self.states(df)])

    def interacted(self, df: pd.DataFrame) -> np.ndarray:
        c = df[D.CHARS].fillna(0.0).to_numpy(dtype=float)
        s = self.states(df)
        inter = (c[:, :, None] * s[:, None, :]).reshape(len(df), -1)
        return np.column_stack([c, inter])


def tune_linear(Xi: np.ndarray, y: np.ndarray) -> dict:
    """Penalties by 4-fold time-series cross-validation on every third month of data (rows are sorted by date)."""
    cv = TimeSeriesSplit(n_splits=4)
    idx = np.arange(len(y))[::3]
    X, yy = Xi[idx], y[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ridge = RidgeCV(alphas=np.logspace(0, 6, 25), cv=cv).fit(X, yy)
        lasso = LassoCV(cv=cv, alphas=25, max_iter=5000, tol=1e-4).fit(X, yy)
        enet = ElasticNetCV(l1_ratio=[0.2, 0.6], cv=cv, alphas=25, max_iter=5000, tol=1e-4).fit(X, yy)
    return {"ridge": ridge.alpha_, "lasso": lasso.alpha_, "enet": (enet.alpha_, enet.l1_ratio_)}


def fit_linear(Xi: np.ndarray, y: np.ndarray, hp: dict) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ols = LinearRegression().fit(Xi[:, :12], y)  # characteristics only, no interactions
        ridge = Ridge(alpha=hp["ridge"]).fit(Xi, y)
        lasso = Lasso(alpha=hp["lasso"], max_iter=5000, tol=1e-4).fit(Xi, y)
        enet = ElasticNet(alpha=hp["enet"][0], l1_ratio=hp["enet"][1], max_iter=5000, tol=1e-4).fit(Xi, y)
    return {
        "ols": lambda Z: ols.predict(Z[:, :12]),
        "ridge": ridge.predict,
        "lasso": lasso.predict,
        "enet": enet.predict,
        "_lasso_coef": lasso.coef_,
    }


def fit_nonlinear(Xb: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    rf = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=300, max_features=0.4, max_samples=0.3, random_state=seed, n_jobs=2
    ).fit(Xb, y)
    gbm = HistGradientBoostingRegressor(
        learning_rate=0.03, max_iter=300, max_depth=3, min_samples_leaf=400, l2_regularization=5.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=seed,
    ).fit(Xb, y)  # fmt: skip
    ysd = y.std()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        nets = [
            MLPRegressor(
                hidden_layer_sizes=(32, 16), alpha=1e-1, learning_rate_init=1e-3, max_iter=300,
                early_stopping=True, validation_fraction=0.15, n_iter_no_change=10, random_state=seed + s,
            ).fit(Xb, y / ysd)
            for s in range(3)
        ]  # fmt: skip
    return {
        "rf": rf.predict,
        "gbm": gbm.predict,
        "nn": lambda Z: ysd * np.mean([n.predict(Z) for n in nets], axis=0),
        "_rf_importance": rf.feature_importances_,
    }


def recursive_forecasts(
    panel: pd.DataFrame, first_test: str = "1980-01-01", lin_every: int = 12, nl_every: int = 36
) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """Forecast every industry's relative return one month ahead, every month, with an expanding window.

    At forecast date t a model learns only from rows whose next-month outcome is known, i.e. dates up to t - 1 month.
    """
    panel = panel.sort_values(["date", "industry"]).reset_index(drop=True)
    months = sorted(panel.loc[panel["date"] >= pd.Timestamp(first_test), "date"].unique())
    out, imp = [], []
    lin = nl = design = nl_design = hp = None
    for k, t in enumerate(months):
        t = pd.Timestamp(t)
        test = panel[panel["date"] == t]
        if k % lin_every == 0 or lin is None:
            tr = panel[(panel["date"] <= t - pd.DateOffset(months=1)) & panel["y"].notna()]
            design = Design().fit(tr)
            Xi_tr, y_tr = design.interacted(tr), tr["y"].to_numpy()
            if k % nl_every == 0 or hp is None:
                hp = tune_linear(
                    Xi_tr, y_tr
                )  # penalties and the nonlinear models are re-tuned every 36 months
                nl = fit_nonlinear(design.base(tr), y_tr)
                nl_design = design
                imp.append(
                    pd.DataFrame(
                        {"date": t, "predictor": [*D.CHARS, *D.STATES], "rf_importance": nl["_rf_importance"]}
                    )
                )
            lin = fit_linear(Xi_tr, y_tr, hp)
        Xi, Xb = design.interacted(test), nl_design.base(test)
        res = test[["date", "industry", "y", "ret_next"]].copy()
        for name, (col, sign) in SIGNALS.items():
            res[name] = sign * test[col].fillna(0.0).to_numpy()
        for name in ("ols", "ridge", "lasso", "enet"):
            res[name] = lin[name](Xi)
        for name in ("rf", "gbm", "nn"):
            res[name] = nl[name](Xb)
        z = lambda v: (v - v.mean()) / (v.std() if v.std() > 0 else 1)
        res["mean_ml"] = np.mean([z(res[m]) for m in ("ridge", "lasso", "enet", "rf", "gbm", "nn")], axis=0)
        out.append(res)
    return pd.concat(out, ignore_index=True), pd.concat(imp, ignore_index=True)
