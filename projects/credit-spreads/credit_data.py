"""credit_data.py: build the real-time forecasting dataset for US credit spreads.

Targets
    BAA_SPREAD  Moody's seasoned Baa corporate yield minus the 10-year Treasury yield (BAA - GS10)
    GZ_SPREAD   Gilchrist-Zakrajsek credit spread (Federal Reserve)
For horizon h the target at month t is the change s(t+h) - s(t), in percentage points.

Predictors (about 150)
    * the 126 FRED-MD macro and financial series (McCracken and Ng), made stationary with
      their published transformation codes;
    * credit-specific signals: the excess bond premium, the Chicago Fed financial-conditions
      indexes, the St. Louis Fed stress index, the VIX, term spreads, bank lending standards
      (senior loan officer survey) and business-loan delinquency rates;
    * the spreads' own recent history (level and last three monthly changes).

Real-time rules (information available at the end of month t)
    * market prices (rates, spreads, equity, VIX, exchange rates) are used as of month t;
    * macro data and survey/index data are lagged one month for publication;
    * quarterly bank data (lending standards, delinquencies) are lagged one full quarter
      after the quarter they describe and carried forward between releases;
    * nothing is ever filled backwards, and missing values are left for the model pipeline,
      which imputes them using the training window only.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

RAW = pathlib.Path(__file__).resolve().parent / "data" / "raw"
TARGETS = {
    "BAA_SPREAD": "Moody's Baa – 10-year Treasury (monthly average)",
    "BAA_EOM": "Moody's Baa – 10-year Treasury (end of month)",
    "GZ_SPREAD": "Gilchrist-Zakrajšek spread",
}
OTHER = {"BAA_SPREAD": "GZ_SPREAD", "BAA_EOM": "GZ_SPREAD", "GZ_SPREAD": "BAA_SPREAD"}

# FRED-MD series that are market prices or market-based rates, observable at the end of month t
FINANCIAL = {
    "FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA", "BAA", "COMPAPFFx", "TB3SMFFM",
    "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM", "AAAFFM", "BAAFFM", "TWEXAFEGSMTHx", "EXSZUSx", "EXJPUSx",
    "EXUSUKx", "EXCAUSx", "S&P 500", "S&P div yield", "S&P PE ratio", "VIXCLSx",
}  # fmt: skip


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
    """FRED-MD, transformed to stationarity, with publication lags applied."""
    df = pd.read_csv(raw / "fred_md_2026-09.csv")
    codes = df.iloc[0, 1:].astype(int)
    df = df.iloc[1:].copy()
    df.index = pd.to_datetime(df.pop("sasdate"), format="%m/%d/%Y")
    df = df.apply(pd.to_numeric, errors="coerce")
    out = {}
    for c in df.columns:
        x = transform(df[c], int(codes[c]))
        out[f"md_{c}"] = x if c in FINANCIAL else x.shift(1)  # macro data: one-month publication lag
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)


def load_spreads(raw: pathlib.Path = RAW) -> pd.DataFrame:
    baa = _monthly(raw / "BAA.csv") - _monthly(raw / "GS10.csv")
    ebp = pd.read_csv(raw / "ebp.csv", parse_dates=["date"]).set_index("date")
    ebp.index = ebp.index.to_period("M").to_timestamp()
    eom = _monthly(raw / "BAA10Y.csv")  # FRED: last daily observation of each month
    return pd.DataFrame({"BAA_SPREAD": baa, "BAA_EOM": eom, "GZ_SPREAD": ebp["gz_spread"]})


def load_credit_signals(raw: pathlib.Path = RAW) -> pd.DataFrame:
    """Credit-specific predictors with their real-time lags."""
    ebp = pd.read_csv(raw / "ebp.csv", parse_dates=["date"]).set_index("date")
    ebp.index = ebp.index.to_period("M").to_timestamp()
    q = lambda s: s.resample("MS").ffill().shift(3)  # quarterly survey/bank data: one quarter late
    sig = pd.DataFrame(
        {
            "EBP": ebp["ebp"].shift(1),  # Fed updates the EBP about a month after the fact
            "EBP_chg": ebp["ebp"].diff().shift(1),
            "NFCI": _monthly(raw / "NFCI.csv").shift(1),
            "ANFCI": _monthly(raw / "ANFCI.csv").shift(1),
            "STLFSI": _monthly(raw / "STLFSI4.csv").shift(1),
            "VIX": _monthly(raw / "VIXCLS.csv"),
            "VIX_chg": _monthly(raw / "VIXCLS.csv").diff(),
            "TERM_10Y3M": _monthly(raw / "T10Y3M.csv"),
            "TERM_10Y2Y": _monthly(raw / "T10Y2Y.csv"),
            "LOAN_STANDARDS": q(_monthly(raw / "DRTSCILM.csv")),
            "DELINQ_CI": q(_monthly(raw / "DRBLACBS.csv")),
            "DELINQ_ALL": q(_monthly(raw / "DRCLACBS.csv")),
        }
    )
    return sig


def build_dataset(target: str, horizon: int, raw: pathlib.Path = RAW, start: str = "1974-01-01") -> dict:
    """Return X (features at the end of month t), y (s[t+h] - s[t]) and the spread level.

    Rows run to the last month with all market data; y is NaN for the final `horizon` months.
    """
    spreads = load_spreads(raw)
    s = spreads[target]
    own = pd.DataFrame(
        {
            "own_level": s,
            "own_chg_1": s.diff(),
            "own_chg_2": s.diff().shift(1),
            "own_chg_3": s.diff().shift(2),
            "own_chg_12": s - s.shift(12),
        }
    )
    other = OTHER[target]
    cross = pd.DataFrame({"other_spread": spreads[other], "other_spread_chg": spreads[other].diff()})
    X = pd.concat([own, cross, load_credit_signals(raw), load_fred_md(raw)], axis=1)
    X = X[(X.index >= pd.Timestamp(start)) & (X.index <= s.dropna().index.max())]
    y = (s.shift(-horizon) - s).reindex(X.index)
    return {
        "X": X,
        "y": y.rename("target"),
        "level": s.reindex(X.index),
        "target": target,
        "horizon": horizon,
    }


def recession(raw: pathlib.Path = RAW) -> pd.Series:
    return _monthly(raw / "USREC.csv").astype(bool)
