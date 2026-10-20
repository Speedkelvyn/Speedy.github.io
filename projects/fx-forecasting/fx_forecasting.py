"""fx_forecasting.py: data, models and evaluation for the exchange-rate forecasting study.

Two experiments share this module:

* Part I  – daily USD/JPY: AR(1)-GARCH(1,1) against a random walk, one-step-ahead, expanding window.
* Part II – monthly panel of six currencies against the USD: Elastic Net, a deep neural network and
  their ensemble against a random walk, using macro fundamentals interacted with global risk states
  (VIX, geopolitical risk, policy uncertainty), in the spirit of Filippou et al. (2020).

Design rules that keep the out-of-sample test honest:

1. Every exchange rate is expressed as USD per unit of foreign currency, so a positive return always
   means the foreign currency appreciated against the dollar.
2. Macro fundamentals enter with a one-month publication lag: the forecast made at the end of month t
   only uses CPI, unemployment and rates for month t-1 or earlier.
3. Nothing is fitted on test data: scaling, model selection and early stopping all use the training
   window that ends at the forecast date. There is no outlier filtering on the target.
4. Missing values are never replaced with zero. A currency-month without its fundamentals is dropped.
5. The forecast is made every month (not every fifth month) and evaluated with the Clark-West test,
   which is the right test for a model nested against the random walk.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

RAW = pathlib.Path(__file__).resolve().parent / "data" / "raw"

CURRENCIES = ["EUR", "GBP", "JPY", "CHF", "CAD", "AUD"]

# FRED series used for each country. Exchange rates are end-of-month values of the daily H.10 series.
FX_SERIES = {  # series id, True if quoted as USD per foreign unit
    "EUR": ("DEXUSEU", True),
    "GBP": ("DEXUSUK", True),
    "JPY": ("DEXJPUS", False),
    "CHF": ("DEXSZUS", False),
    "CAD": ("DEXCAUS", False),
    "AUD": ("DEXUSAL", True),
}
RATE_SERIES = {  # 3-month interbank rates (OECD MEI); Japan uses the call rate for a full history
    "US": "IR3TIB01USM156N",
    "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N",
    "JPY": "IRSTCI01JPM156N",
    "CHF": "IR3TIB01CHM156N",
    "CAD": "IR3TIB01CAM156N",
    "AUD": "IR3TIB01AUM156N",
}
CPI_SERIES = {  # price indices; inflation is the 12-month (4-quarter for Australia) log change
    "US": "CPIAUCSL",
    "EUR": "CP0000EZ19M086NEST",
    "GBP": "GBRCPIALLMINMEI",
    "JPY": "JPNCPIALLMINMEI",
    "CHF": "CHECPIALLMINMEI",
    "CAD": "CANCPIALLMINMEI",
    "AUD": "AUSCPIALLQINMEI",
}
UNEMP_SERIES = {
    "US": "LRHUTTTTUSM156S",
    "EUR": "LRHUTTTTEZM156S",
    "GBP": "LRHUTTTTGBM156S",
    "JPY": "LRHUTTTTJPM156S",
    "CHF": "LRHUTTTTCHQ156S",
    "CAD": "LRHUTTTTCAM156S",
    "AUD": "LRHUTTTTAUM156S",
}
FUNDAMENTALS = ["rate_diff", "infl_diff", "unemp_diff"]
STATES = ["VIX", "GPR", "EPU"]


# =============================================================================== data


def read_series(series_id: str, raw: pathlib.Path = RAW) -> pd.Series:
    """One FRED-style CSV (date,value) as a monthly series indexed by month start."""
    df = pd.read_csv(raw / f"{series_id}.csv", parse_dates=["date"])
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = df["date"].dt.to_period("M").dt.to_timestamp()
    return s.dropna().rename(series_id)


def to_monthly(s: pd.Series, max_fill: int | None = None) -> pd.Series:
    """Monthly frequency. Lower-frequency data are carried forward until the next observation.

    By default the fill limit is the longest gap between observations (so a quarterly or annual
    series covers the months in between) capped at 11 months; data are never filled backwards.
    """
    if max_fill is None:
        gaps = s.index.to_series().diff().dt.days.dropna() // 28
        max_fill = int(min(max(gaps.max() - 1, 0), 11)) if len(gaps) else 0
    idx = pd.date_range(s.index.min(), s.index.max() + pd.offsets.MonthBegin(max_fill), freq="MS")
    s = s.reindex(idx)
    return s.ffill(limit=max_fill) if max_fill else s


def load_fx_monthly(raw: pathlib.Path = RAW) -> pd.DataFrame:
    """End-of-month exchange rates, all as USD per unit of foreign currency."""
    out = {}
    for ccy, (sid, usd_per_foreign) in FX_SERIES.items():
        s = read_series(sid, raw)
        out[ccy] = s if usd_per_foreign else 1.0 / s
    return pd.DataFrame(out)


def yoy_inflation(index: pd.Series) -> pd.Series:
    """Annual inflation in percent from a price index (monthly or quarterly)."""
    quarterly = (index.index.to_series().diff().dt.days.median() or 30) > 45
    lag = 4 if quarterly else 12
    return 100 * np.log(index / index.shift(lag))


def load_fundamentals(raw: pathlib.Path = RAW) -> pd.DataFrame:
    """Long panel (date, currency) of foreign-minus-US rate, inflation and unemployment differentials."""
    rates = {k: to_monthly(read_series(v, raw), 0) for k, v in RATE_SERIES.items()}
    infl = {k: to_monthly(yoy_inflation(read_series(v, raw)).dropna()) for k, v in CPI_SERIES.items()}
    unemp = {k: to_monthly(read_series(v, raw)) for k, v in UNEMP_SERIES.items()}
    frames = []
    for ccy in CURRENCIES:
        f = pd.DataFrame(
            {
                "rate_diff": rates[ccy] - rates["US"],
                "infl_diff": infl[ccy] - infl["US"],
                "unemp_diff": unemp[ccy] - unemp["US"],
            }
        )
        f["currency"] = ccy
        frames.append(f)
    out = pd.concat(frames).rename_axis("date").reset_index()
    return out


def load_states(raw: pathlib.Path = RAW) -> pd.DataFrame:
    """Global risk states: VIX (monthly average), geopolitical risk (GPR) and global policy uncertainty (EPU)."""
    vix = read_series("VIXCLS", raw).rename("VIX")
    gpr = read_series("GPR", raw).rename("GPR")
    epu_raw = pd.read_csv(raw / "EPU_global.csv", parse_dates=["Date"])
    epu = pd.Series(
        epu_raw["GEPU_current"].values, index=epu_raw["Date"].dt.to_period("M").dt.to_timestamp(), name="EPU"
    )
    return pd.concat([vix, gpr, epu], axis=1)


@dataclass
class Panel:
    data: pd.DataFrame  # one row per (date, currency): features, target and the realised return
    features: list[str]
    target: str = "target"


def build_panel(raw: pathlib.Path = RAW, start: str = "1999-01-01", end: str | None = None) -> Panel:
    """Assemble the forecasting panel.

    Row (t, c) holds information available at the end of month t and the target
    r_{c,t+1} = 100 * log(S_{c,t+1} / S_{c,t}), the next month's percentage return of currency c
    against the USD.
    """
    fx = load_fx_monthly(raw)
    ret = 100 * np.log(fx / fx.shift(1))  # return realised during month t
    target = ret.shift(-1)  # next month's return: what we forecast at the end of month t

    fund = load_fundamentals(raw)
    # publication lag: macro data for month t-1 are the latest known at the end of month t
    fund = fund.sort_values(["currency", "date"])
    fund[FUNDAMENTALS] = fund.groupby("currency")[FUNDAMENTALS].shift(1)

    states = load_states(raw)  # VIX, GPR, EPU are observed in real time (GPR/EPU published early next month)
    states = states.shift(1)[STATES].join(states["VIX"].rename("VIX_now"))

    rows = []
    for ccy in CURRENCIES:
        d = pd.DataFrame(
            {
                "ret_1m": ret[ccy],  # momentum features use only past returns
                "ret_3m": ret[ccy].rolling(3).sum(),
                "target": target[ccy],
            }
        )
        d["currency"] = ccy
        rows.append(d.rename_axis("date").reset_index())
    panel = pd.concat(rows).merge(fund, on=["date", "currency"], how="left")
    panel = panel.merge(states.rename_axis("date").reset_index(), on="date", how="left")
    panel["VIX"] = panel["VIX_now"]  # VIX is a market price: the end-of-month average is known at t
    panel = panel.drop(columns="VIX_now")

    # interactions: each fundamental scaled by each risk state, states measured relative to their
    # long-run level so the interaction is "fundamental, amplified in high-risk months"
    for st in STATES:
        panel[f"{st}_rel"] = panel[st] / panel.groupby("currency")[st].transform(
            lambda s: s.expanding().mean()
        )
    for f in FUNDAMENTALS:
        for st in STATES:
            panel[f"{f}_x_{st}"] = panel[f] * panel[f"{st}_rel"]

    # lags of the fundamentals (1 and 3 months beyond the publication lag)
    panel = panel.sort_values(["currency", "date"])
    for f in FUNDAMENTALS:
        for lag in (1, 3):
            panel[f"{f}_lag{lag}"] = panel.groupby("currency")[f].shift(lag)

    features = (
        FUNDAMENTALS
        + STATES
        + ["ret_1m", "ret_3m"]
        + [f"{f}_x_{st}" for f in FUNDAMENTALS for st in STATES]
        + [f"{f}_lag{lag}" for f in FUNDAMENTALS for lag in (1, 3)]
    )
    panel = panel[(panel["date"] >= pd.Timestamp(start))]
    if end:
        panel = panel[panel["date"] <= pd.Timestamp(end)]
    panel = (
        panel.dropna(subset=[*features, "target"]).sort_values(["date", "currency"]).reset_index(drop=True)
    )
    return Panel(panel[["date", "currency", "target", *features]], features)


# =============================================================================== evaluation


def r2_os(actual: np.ndarray, pred: np.ndarray, bench: np.ndarray | None = None) -> float:
    """Campbell-Thompson out-of-sample R² in percent; the default benchmark is the random walk (zero)."""
    bench = np.zeros_like(actual) if bench is None else bench
    return 100 * (1 - np.sum((actual - pred) ** 2) / np.sum((actual - bench) ** 2))


def clark_west(actual: np.ndarray, pred: np.ndarray, bench: np.ndarray | None = None, lags: int = 0) -> dict:
    """Clark & West (2007) test of equal predictive accuracy for a model nested in the benchmark.

    H0: the model does not improve on the benchmark (random walk by default). One-sided; a small
    p-value means the model's forecasts add information. Newey-West standard errors with `lags`.
    """
    bench = np.zeros_like(actual) if bench is None else bench
    f = (actual - bench) ** 2 - ((actual - pred) ** 2 - (bench - pred) ** 2)
    n = len(f)
    fbar = f.mean()
    u = f - fbar
    var = u @ u / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (u[k:] @ u[:-k]) / n
    t = fbar / math.sqrt(var / n)
    return {"stat": t, "p": 1 - stats.norm.cdf(t)}


def diebold_mariano(actual: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, lags: int = 0) -> dict:
    """Diebold-Mariano test on squared errors. Negative statistic: model A is more accurate. Two-sided p."""
    d = (actual - pred_a) ** 2 - (actual - pred_b) ** 2
    n = len(d)
    u = d - d.mean()
    var = u @ u / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    t = d.mean() / math.sqrt(var / n)
    return {"stat": t, "p": 2 * (1 - stats.norm.cdf(abs(t)))}


def direction_test(actual: np.ndarray, pred: np.ndarray) -> dict:
    """Share of correctly predicted signs and a Pesaran-Timmermann (1992) test of independence."""
    a, p = actual > 0, pred > 0
    n = len(a)
    hit = np.mean(a == p)
    pa, pp = a.mean(), p.mean()
    p_star = pa * pp + (1 - pa) * (1 - pp)
    v_hit = p_star * (1 - p_star) / n
    v_star = (
        (2 * pp - 1) ** 2 * pa * (1 - pa) / n
        + (2 * pa - 1) ** 2 * pp * (1 - pp) / n
        + 4 * pa * pp * (1 - pa) * (1 - pp) / n**2
    )
    denom = v_hit - v_star
    stat = (hit - p_star) / math.sqrt(denom) if denom > 0 else float("nan")
    return {
        "hit_rate": 100 * hit,
        "pt_stat": stat,
        "p": 1 - stats.norm.cdf(stat) if denom > 0 else float("nan"),
    }


def summarise(results: pd.DataFrame, models: dict[str, str], lags: int = 0) -> pd.DataFrame:
    """One row per model: R²_OS, Clark-West, RMSE, direction accuracy and the Pesaran-Timmermann test."""
    actual = results["actual"].to_numpy()
    rows = []
    for name, col in models.items():
        pred = results[col].to_numpy()
        cw = clark_west(actual, pred, lags=lags)
        dt = direction_test(actual, pred)
        rows.append(
            {
                "Model": name,
                "R²_OS (%)": r2_os(actual, pred),
                "CW stat": cw["stat"],
                "CW p-value": cw["p"],
                "RMSE": float(np.sqrt(np.mean((actual - pred) ** 2))),
                "Hit rate (%)": dt["hit_rate"],
                "PT p-value": dt["p"],
            }
        )
    return pd.DataFrame(rows)


def read_daily(series_id: str, raw: pathlib.Path = RAW) -> pd.Series:
    """A daily FRED series (no resampling), missing days dropped."""
    df = pd.read_csv(raw / f"{series_id}.csv", parse_dates=["date"])
    return pd.Series(
        pd.to_numeric(df["value"], errors="coerce").values, index=df["date"], name=series_id
    ).dropna()
