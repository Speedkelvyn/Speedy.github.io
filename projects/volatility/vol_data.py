"""vol_data.py: build the real-time dataset for forecasting US stock-market volatility.

Target
    log RV(t+1): the natural log of next month's realised variance of the US stock market, the sum of
    squared daily returns of the CRSP value-weighted market (Kenneth French's data library, July 1926
    to August 2026).

Predictors, all known at the end of month t
    * realised-volatility history: last month, the last 3 and 12 months, the last 5 trading days,
      downside share, the last day's move, 1- and 12-month returns (the HAR family of Corsi 2009);
    * implied volatility: the VIX at month end (from 1990) and the variance risk premium proxy VIX - RV;
    * credit and funding conditions: Baa - 10-year Treasury spread, excess bond premium, Chicago Fed
      financial-conditions indexes, St. Louis Fed stress index, term spreads, bank lending standards
      and delinquencies;
    * the 126 FRED-MD macro series (McCracken and Ng), made stationary with their published codes.

Real-time rules: market prices at month t; macro and index data lagged one month; quarterly bank data
lagged one quarter. Nothing is filled backwards.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd

RAW = pathlib.Path(__file__).resolve().parent / "data" / "raw"

FINANCIAL = {
    "FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA", "BAA", "COMPAPFFx", "TB3SMFFM",
    "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM", "AAAFFM", "BAAFFM", "TWEXAFEGSMTHx", "EXSZUSx", "EXJPUSx",
    "EXUSUKx", "EXCAUSx", "S&P 500", "S&P div yield", "S&P PE ratio", "VIXCLSx",
}  # fmt: skip


# ----------------------------------------------------------------------------- daily market data
def load_daily(raw: pathlib.Path = RAW) -> pd.DataFrame:
    """Daily Fama-French factors: Mkt-RF, SMB, HML, RF in percent, plus the market total return."""
    rows = []
    for line in (raw / "F-F_Research_Data_Factors_daily.csv").read_text(encoding="latin-1").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and re.fullmatch(r"\d{8}", parts[0]):
            rows.append([parts[0], *map(float, parts[1:5])])
    df = pd.DataFrame(rows, columns=["date", "mktrf", "smb", "hml", "rf"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").sort_index()
    df["mkt"] = (df["mktrf"] + df["rf"]) / 100  # total return of the market, decimal
    df["rf_d"] = df["rf"] / 100
    return df


def monthly_from_daily(d: pd.DataFrame) -> pd.DataFrame:
    """Month-by-month realised variance, returns and the other volatility ingredients."""
    g = d.groupby(d.index.to_period("M"))
    r2 = d["mkt"] ** 2
    out = pd.DataFrame(
        {
            "rv": r2.groupby(d.index.to_period("M")).sum(),
            "rv_neg": r2.where(d["mkt"] < 0, 0).groupby(d.index.to_period("M")).sum(),
            "n_days": g["mkt"].size(),
            "rv_5d": g["mkt"].apply(lambda x: float((x.iloc[-5:] ** 2).sum())),
            "abs_last": g["mkt"].apply(lambda x: float(abs(x.iloc[-1]))),
            "ret": g["mkt"].apply(lambda x: float(np.prod(1 + x) - 1)),
            "rf": g["rf_d"].apply(lambda x: float(np.prod(1 + x) - 1)),
        }
    )
    out.index = out.index.to_timestamp()
    out["ex"] = out["ret"] - out["rf"]
    return out


# ----------------------------------------------------------------------------- macro and credit data
def _monthly(path: pathlib.Path, col: str = "value", date_col: str = "date") -> pd.Series:
    df = pd.read_csv(path)
    s = pd.to_numeric(df[col], errors="coerce")
    s.index = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()
    return s[~s.index.duplicated(keep="last")].sort_index()


def transform(x: pd.Series, code: int) -> pd.Series:
    """FRED-MD transformation codes (McCracken & Ng 2016, Appendix)."""
    if code == 1:
        return x
    if code == 2:
        return x.diff()
    if code == 3:
        return x.diff().diff()
    if code == 4:
        return np.log(x)
    if code == 5:
        return np.log(x).diff()
    if code == 6:
        return np.log(x).diff().diff()
    if code == 7:
        return (x / x.shift(1) - 1).diff()
    raise ValueError(f"unknown transformation code {code}")


def load_fred_md(raw: pathlib.Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(raw / "fred_md_2026-09.csv")
    codes = df.iloc[0, 1:].astype(int)
    df = df.iloc[1:].copy()
    df.index = pd.to_datetime(df.pop("sasdate"), format="%m/%d/%Y")
    df = df.apply(pd.to_numeric, errors="coerce")
    out = {}
    for c in df.columns:
        x = transform(df[c], int(codes[c]))
        out[f"md_{c}"] = x if c in FINANCIAL else x.shift(1)
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)


def load_signals(raw: pathlib.Path = RAW) -> pd.DataFrame:
    """Credit, funding and implied-volatility signals with their real-time lags."""
    ebp = pd.read_csv(raw / "ebp.csv", parse_dates=["date"]).set_index("date")
    ebp.index = ebp.index.to_period("M").to_timestamp()
    q = lambda s: s.resample("MS").ffill().shift(3)  # quarterly data: one quarter late
    vix = _monthly(
        raw / "VIXCLS_eom.csv"
    )  # last trading day of the month; a monthly average would be stale by half a month
    return pd.DataFrame(
        {
            "lvix2": np.log((vix / 100) ** 2 / 12),  # VIX as a log monthly variance, same units as log RV
            "VIX_chg": vix.diff(),
            "BAA_SPR": _monthly(raw / "BAA10Y.csv"),
            "EBP": ebp["ebp"].shift(1),
            "GZ": ebp["gz_spread"].shift(1),
            "NFCI": _monthly(raw / "NFCI.csv").shift(1),
            "ANFCI": _monthly(raw / "ANFCI.csv").shift(1),
            "STLFSI": _monthly(raw / "STLFSI4.csv").shift(1),
            "TERM_10Y3M": _monthly(raw / "T10Y3M.csv"),
            "TERM_10Y2Y": _monthly(raw / "T10Y2Y.csv"),
            "LOAN_STANDARDS": q(_monthly(raw / "DRTSCILM.csv")),
            "DELINQ_CI": q(_monthly(raw / "DRBLACBS.csv")),
            "DELINQ_ALL": q(_monthly(raw / "DRCLACBS.csv")),
        }
    )


def recession(raw: pathlib.Path = RAW) -> pd.Series:
    return _monthly(raw / "USREC.csv").astype(bool)


# ----------------------------------------------------------------------------- the dataset
def volatility_features(m: pd.DataFrame) -> pd.DataFrame:
    """Realised-volatility predictors known at the end of each month."""
    lrv = np.log(m["rv"])
    f = pd.DataFrame(
        {
            "lrv": lrv,  # last month
            "lrv_3": np.log(m["rv"].rolling(3).mean()),  # last quarter
            "lrv_12": np.log(m["rv"].rolling(12).mean()),  # last year
            "lrv_5d": np.log(m["rv_5d"] * 21 / 5),  # last five trading days, scaled to a month
            "semi_neg": m["rv_neg"] / m["rv"],  # downside share of variance
            "abs_last": m["abs_last"],
            "ret_1m": m["ret"],
            "ret_12m": (1 + m["ret"]).rolling(12).apply(np.prod, raw=True) - 1,
        }
    )
    return f


def build_dataset(raw: pathlib.Path = RAW, start: str = "1927-07-01") -> dict:
    """X (features at the end of month t), y = log RV(t+1) and the next-month market return."""
    daily = load_daily(raw)
    m = monthly_from_daily(daily)
    X = pd.concat([volatility_features(m), load_signals(raw), load_fred_md(raw)], axis=1)
    X["vrp"] = X["lvix2"] - X["lrv"]  # implied minus realised: the variance risk premium proxy
    X = X[(X.index >= pd.Timestamp(start)) & (X.index <= m.index.max())]
    nxt = m.shift(-1)
    out = {
        "X": X,
        "y": np.log(nxt["rv"]).reindex(X.index).rename("target"),
        "rv_next": nxt["rv"].reindex(X.index),
        "ex_next": nxt["ex"].reindex(X.index),
        "rf_next": nxt["rf"].reindex(X.index),
        "rv_now": m["rv"].reindex(X.index),
        "monthly": m,
        "daily": daily,
    }
    return out
