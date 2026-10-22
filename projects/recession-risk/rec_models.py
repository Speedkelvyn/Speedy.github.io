"""rec_models.py: the model line-up and the recursive out-of-sample loop for recession and bear-market probabilities.

Fixed before any out-of-sample result was seen. Each model gives the probability that the event happens
within the next 12 months.

  base        the share of past months in which the event followed: no information about today
  probit_ts   probit on the yield-curve slope (Estrella & Mishkin 1998)
  probit_ebp  slope + excess bond premium (Favara, Gilchrist, Lewis & Zakrajsek 2016)
  probit_fin  slope + excess bond premium + Baa spread + financial-conditions index
  probit_real the financial set + unemployment above its 12-month low, employment and industrial production growth
  lasso       L1-penalised logit on all ~150 predictors, penalty chosen by time-series cross-validation
  rf          random forest on the same predictors
  gbm         gradient-boosted trees
  mean_prob   average of the probit_real, lasso, rf and gbm probabilities
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit

SETS = {
    "probit_ts": ["TERM_10Y3M"],
    "probit_ebp": ["TERM_10Y3M", "EBP"],
    "probit_fin": ["TERM_10Y3M", "EBP", "BAA_SPR", "NFCI"],
    "probit_real": ["TERM_10Y3M", "EBP", "BAA_SPR", "NFCI", "UNRATE_vs_min", "PAYEMS_3m", "INDPRO_12m"],
}
MODELS = ["base", *SETS, "lasso", "rf", "gbm", "mean_prob"]
NAMES = {
    "base": "Base rate",
    "probit_ts": "Probit: yield curve",
    "probit_ebp": "Probit: curve + excess bond premium",
    "probit_fin": "Probit: + Baa spread and financial conditions",
    "probit_real": "Probit: + labour market and output",
    "lasso": "Lasso logit, all predictors",
    "rf": "Random forest",
    "gbm": "Gradient boosting",
    "mean_prob": "Average of the four richest models",
}
MIN_ROWS = 60
MIN_EVENTS = 3


def probit_prob(Xtr: pd.DataFrame, ytr: np.ndarray, xt: pd.DataFrame) -> float:
    """Probit probability (logit if the probit cannot be estimated, e.g. under perfect separation)."""
    if xt.isna().any(axis=None):
        return float("nan")
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    Z, z = (Xtr - mu) / sd, (xt - mu) / sd
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Probit(ytr, sm.add_constant(Z, has_constant="add")).fit(disp=0, maxiter=200)
            p = float(res.predict(sm.add_constant(z, has_constant="add"))[0])
            if not np.isfinite(p):
                raise ValueError
            return p
    except Exception:
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z, ytr)
        return float(lr.predict_proba(z)[0, 1])


class Wide:
    """Column selection, winsorising, median imputation and standardisation, fitted on training rows only."""

    def fit(self, X: pd.DataFrame) -> Wide:
        self.cols = [c for c in X.columns if X[c].notna().sum() >= MIN_ROWS]
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


def fit_wide(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lasso = LogisticRegressionCV(
            penalty="l1", solver="saga", Cs=12, cv=TimeSeriesSplit(n_splits=5), scoring="neg_log_loss",
            max_iter=3000, tol=1e-3, random_state=seed,
        ).fit(X, y)  # fmt: skip
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=10, max_features="sqrt", random_state=seed, n_jobs=1
    ).fit(X, y)
    gbm = HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=300, max_depth=3, min_samples_leaf=20, l2_regularization=2.0,
        early_stopping=True, validation_fraction=0.2, n_iter_no_change=30, random_state=seed,
    ).fit(X, y)  # fmt: skip
    return {
        "lasso": lambda Z: lasso.predict_proba(Z)[:, 1],
        "rf": lambda Z: rf.predict_proba(Z)[:, 1],
        "gbm": lambda Z: gbm.predict_proba(Z)[:, 1],
        "_coef": lasso.coef_.ravel(),
        "_imp": rf.feature_importances_,
    }


def recursive_probabilities(
    X: pd.DataFrame, y: pd.Series, lag: int, first_test: str = "1990-01-01", refit_every: int = 12
) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """Probabilities at the end of every month, using only labels that were known at that date.

    A month s is in the training set at forecast date t only if s <= t - `lag` months (the label window plus
    announcement delay). Months after the last labelled one are forecast live with the same procedure.
    """
    dates = X.index
    last_label = y.last_valid_index()  # months after this have no outcome yet and are forecast live
    test_dates = [
        t for t in dates if t >= pd.Timestamp(first_test) and (pd.notna(y.loc[t]) or t > last_label)
    ]
    rows, imp = [], []
    wide = fitted = None
    for k, t in enumerate(test_dates):
        train = (dates <= t - pd.DateOffset(months=lag)) & y.notna().to_numpy()
        ytr = y[train].to_numpy().astype(int)
        xt = X.loc[[t]]
        f = {"base": float(ytr.mean())}
        for name, cols in SETS.items():
            ok = train & X[cols].notna().all(axis=1).to_numpy()
            yy = y[ok].to_numpy().astype(int)
            if ok.sum() < MIN_ROWS or yy.sum() < MIN_EVENTS:
                f[name] = float("nan")
            else:
                f[name] = probit_prob(X.loc[ok, cols], yy, xt[cols])
        if k % refit_every == 0 or wide is None:
            wide = Wide().fit(X.loc[train])
            fitted = fit_wide(wide.transform(X.loc[train]), ytr)
            imp.append(
                pd.DataFrame(
                    {
                        "date": t,
                        "predictor": wide.cols,
                        "rf_importance": fitted["_imp"],
                        "lasso_coef": fitted["_coef"],
                    }
                )
            )
        z = wide.transform(xt)
        for name in ("lasso", "rf", "gbm"):
            f[name] = float(fitted[name](z)[0])
        f["mean_prob"] = float(np.mean([f["probit_real"], f["lasso"], f["rf"], f["gbm"]]))
        rows.append({"date": t, "actual": float(y.loc[t]) if pd.notna(y.loc[t]) else np.nan, **f})
    return pd.DataFrame(rows), pd.concat(imp, ignore_index=True)
