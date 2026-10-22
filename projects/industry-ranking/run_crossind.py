"""exp_crossind.py: real-time cross-industry lead-lag (Rapach, Strauss, Tu & Zhou 2019 style) with our costs and ranking target.

For each of the 49 industries a separate elastic net predicts its next-month return relative to the average industry,
from the 49 industries' last-month returns (relative to that month's average). Expanding window from 1940, coefficients
refit every 12 months, penalty re-chosen every 36 months by time-series cross-validation, no look-ahead.
Output: results/crossind_forecasts.csv with the full forecast (`xind`) and its lead-lag component (`xlead`, forecast
minus the industry's own training-sample mean, so a long-run industry average cannot masquerade as lead-lag).
"""

import warnings

import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

import ind_data as D

FIRST = pd.Timestamp("1980-01-01")
ret, _ = D.load_industries()
R = ret.loc["1940-01-01":]
nxt = R.shift(-1)
Y = nxt.sub(nxt.mean(axis=1), axis=0)  # relative next-month return, the ranking target
X = R.sub(R.mean(axis=1), axis=0).fillna(0.0)  # last month's returns relative to the average industry
dates = list(R.index[R.index >= FIRST])


def one(ind: str) -> pd.DataFrame:
    warnings.simplefilter("ignore", ConvergenceWarning)
    y = Y[ind]
    out = []
    model = alpha = None
    for k, t in enumerate(dates):
        if k % 12 == 0 or model is None:
            tr = (R.index <= t - pd.DateOffset(months=1)) & y.notna().to_numpy()
            Xtr, ytr = X.loc[tr].to_numpy(), y[tr].to_numpy()
            mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
            sd[sd == 0] = 1
            Z = (Xtr - mu) / sd
            if k % 36 == 0 or alpha is None:
                cv = ElasticNetCV(
                    l1_ratio=0.5, alphas=30, cv=TimeSeriesSplit(4), max_iter=3000, tol=1e-4
                ).fit(Z, ytr)
                alpha = cv.alpha_
            model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=3000, tol=1e-4).fit(Z, ytr)
            ybar = ytr.mean()
        z = (X.loc[[t]].to_numpy() - mu) / sd
        pred = float(model.predict(z)[0])
        out.append((t, ind, pred, pred - ybar, int((model.coef_ != 0).sum())))
    return pd.DataFrame(out, columns=["date", "industry", "xind", "xlead", "n_active"])


if __name__ == "__main__":
    res = Parallel(n_jobs=2)(delayed(one)(c) for c in R.columns)
    out = pd.concat(res, ignore_index=True)
    out.to_csv("results/crossind_forecasts.csv", index=False)
    print("saved", out.shape, "mean active predictors:", out["n_active"].mean().round(1))
