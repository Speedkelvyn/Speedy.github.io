"""models.py: forecasting models and the two out-of-sample experiments."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from fx_forecasting import Panel

# =============================================================================== Part I: daily GARCH


def garch_vs_random_walk(prices: pd.Series, train_share: float = 0.8, refit_every: int = 1) -> pd.DataFrame:
    """One-step-ahead USD/JPY forecasts from an AR(1)-GARCH(1,1) and a random walk.

    Expanding window: for each test day the model is estimated on all returns up to the day before.
    With `refit_every` > 1 the parameters are re-estimated every n days and the forecast in between
    uses the latest parameters with the most recent data (the standard "fixed-parameter update").
    """
    from arch import arch_model

    r = (100 * np.log(prices / prices.shift(1))).dropna()
    p = prices.loc[r.index]
    split = int(train_share * len(r))
    rows, res = [], None
    for i in range(split, len(r)):
        window = r.iloc[:i].to_numpy()
        model = arch_model(window, mean="ARX", lags=1, vol="GARCH", p=1, q=1)
        if res is None or (i - split) % refit_every == 0:
            res = model.fit(disp="off", show_warning=False)
            params = res.params
        else:
            res = model.fix(params)
        fc = res.forecast(horizon=1, reindex=False)
        mu, var = float(fc.mean.iloc[-1, 0]), float(fc.variance.iloc[-1, 0])
        rows.append(
            {
                "date": r.index[i],
                "actual_ret": r.iloc[i],
                "garch_ret": mu,
                "garch_vol": np.sqrt(var),
                "price_prev": p.iloc[i - 1],
                "actual_price": p.iloc[i],
            }
        )
    out = pd.DataFrame(rows).set_index("date")
    out["rw_ret"] = 0.0
    out["garch_price"] = out["price_prev"] * np.exp(out["garch_ret"] / 100)
    out["rw_price"] = out["price_prev"]
    return out


# =============================================================================== Part II: panel models


def fit_elastic_net(X: np.ndarray, y: np.ndarray, seed: int = 0) -> ElasticNetCV:
    """Elastic Net with no intercept (as in Filippou et al.), penalty chosen by time-series CV."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0],
            cv=TimeSeriesSplit(n_splits=5),
            fit_intercept=False,
            max_iter=20000,
            random_state=seed,
        ).fit(X, y)


def _torch():
    import torch
    from torch import nn

    return torch, nn


def make_dnn(input_dim: int, dropout: float = 0.1):
    """64-32-16 ReLU network with LayerNorm and dropout; linear output (no squashing)."""
    _, nn = _torch()

    class FXDNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers, width = [], input_dim
            for h, dp in ((64, dropout), (32, dropout), (16, dropout / 2)):
                layers += [nn.Linear(width, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dp)]
                width = h
            layers.append(nn.Linear(width, 1))
            self.net = nn.Sequential(*layers)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return FXDNN()


def fit_dnn(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    max_epochs: int = 500,
    patience: int = 50,
    val_share: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch: int = 32,
):
    """Train with Adam on the earlier 90% of the window, early-stop on the most recent 10%.

    Returns a predict function. The validation block is the latest part of the training window, so
    the stopping rule never sees the month being forecast.
    """
    torch, nn = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_val = max(int(len(X) * val_share), 20)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    Xtr, ytr, Xva, yva = Xt[:-n_val], yt[:-n_val], Xt[-n_val:], yt[-n_val:]
    net = make_dnn(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=patience // 2)
    loss_fn = nn.MSELoss()
    best, best_state, wait = float("inf"), None, 0
    gen = torch.Generator().manual_seed(seed)
    for _ in range(max_epochs):
        net.train()
        for idx in torch.randperm(len(Xtr), generator=gen).split(batch):
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(net(Xtr[idx]), ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(Xva), yva).item()
        sched.step(v)
        if v < best - 1e-7:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    def predict(Xn: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return net(torch.tensor(Xn, dtype=torch.float32)).numpy()

    return predict


def _dnn_forecast(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int) -> np.ndarray:
    """Train one network and forecast (a top-level function so it can run in a worker process)."""
    torch, _ = _torch()
    torch.set_num_threads(1)
    return fit_dnn(Xtr, ytr, seed=seed)(Xte)


def recursive_panel_forecasts(
    panel: Panel,
    first_test: str = "2012-01-01",
    dnn_seeds: tuple[int, ...] = (0, 1, 2),
    n_jobs: int = -1,
    progress: bool = True,
) -> pd.DataFrame:
    """Expanding-window forecasts for every month from `first_test` on, re-estimating every month.

    At forecast date t the models are trained on all rows dated before t. The newest of those rows
    (dated t-1) has as its target the return during month t, which is known at the end of month t,
    so nothing from the future enters. Features are standardised on the training window only. The
    DNN forecast averages several random seeds, which removes most of the noise that comes from a
    single network's random initialisation.
    """
    from joblib import Parallel, delayed

    d = panel.data
    X_all = d[panel.features].to_numpy(dtype=np.float64)
    y_all = d[panel.target].to_numpy(dtype=np.float64)
    dates = np.sort(d["date"].unique())
    test_dates = [t for t in dates if t >= np.datetime64(first_test)]
    out = []
    iterator = test_dates
    if progress:
        from tqdm import tqdm

        iterator = tqdm(test_dates, desc="Recursive forecasts")
    with Parallel(n_jobs=n_jobs) as pool:
        for t in iterator:
            test = (d["date"] == t).to_numpy()
            train = (d["date"] < t).to_numpy()
            scaler = StandardScaler().fit(X_all[train])
            Xtr, Xte = scaler.transform(X_all[train]), scaler.transform(X_all[test])
            p_enet = fit_elastic_net(Xtr, y_all[train]).predict(Xte)
            p_dnn = np.mean(
                pool(delayed(_dnn_forecast)(Xtr, y_all[train], Xte, s) for s in dnn_seeds), axis=0
            )
            block = d.loc[test, ["date", "currency"]].copy()
            block["actual"] = y_all[test]
            block["rw"] = 0.0
            block["enet"] = p_enet
            block["dnn"] = p_dnn
            block["ensemble"] = (p_enet + p_dnn) / 2
            block["VIX"] = d.loc[test, "VIX"].to_numpy()
            out.append(block)
    return pd.concat(out, ignore_index=True)
